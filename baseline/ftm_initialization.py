"""Known first-state freezing and conditional Conv coordinate-sign ablation."""
import hashlib
import json
import math
import time

import torch
from torch import nn

POLICIES=('off','learned-first-frozen','sign-first-frozen')
PURPOSE='ftm-first-conv-coordinate-sign-v1'


def validate_initialization(args):
    policy=getattr(args,'ftm_initialization_control','off')
    auxiliary=getattr(args,'ftm_initialization_aux_seed',None)
    if policy not in POLICIES:raise ValueError('Unknown FTM initialization control')
    if policy=='off':
        if auxiliary is not None:raise ValueError('Initialization aux seed is only used by sign-first-frozen')
        return policy,auxiliary
    if args.source!='rn50' or args.batch_size!=2 or getattr(args,'ftm_input_profile','canonical224-v2')!='canonical224-v2' or args.ftm_layer_profile!='mapped299' or args.ftm_update_mode!='full' or getattr(args,'ftm_state_control','raw')!='raw' or getattr(args,'ftm_efficacy_probe','off')!='off' or getattr(args,'ftm_beta',.01)!=.01:
        raise ValueError('First-initialization controls require RN50V2/canonical224/mapped299/batch2/raw state/full/beta.01/probe off')
    if policy=='sign-first-frozen' and auxiliary not in (20261028,20261029):raise ValueError('Sign-first-frozen requires explicit aux seed20261028 or20261029')
    if policy=='learned-first-frozen' and auxiliary is not None:raise ValueError('Learned-first-frozen does not use an auxiliary stream')
    return policy,auxiliary


def initialization_profile(profile,policy,mode):
    if policy=='off':return profile
    suffix='-update-'+mode
    if mode!='full' or not profile.endswith(suffix):raise ValueError('First-initialization control requires full profile')
    return profile[:-len(suffix)]+'-initialization-'+policy+suffix


def sign_key(auxiliary,batch_id,layer,image_id):
    key=[int(auxiliary),str(batch_id),str(layer),str(image_id),PURPOSE]
    encoded=json.dumps(key,ensure_ascii=False,separators=(',',':')).encode('utf-8')
    return key,int.from_bytes(hashlib.sha256(encoded).digest()[:8],'big')&((1<<63)-1)


def content_sha(tensor):return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def audit_norm(scalar):
    scalar=scalar.detach().reshape(1).to(device='cpu',dtype=torch.float32)
    value=float(scalar.item());bits=int(scalar.view(torch.int32).item())&0xffffffff
    return {'value':value if math.isfinite(value) else None,'fp32_bits':bits,
        'status':'finite' if math.isfinite(value) else 'nan' if math.isnan(value) else 'positive-infinity' if value>0 else 'negative-infinity'}


class FirstInitializationControl:
    """All accepted u remains in context; only masks and small audit rows persist."""
    def __init__(self,policy,auxiliary,batch_id,ids,layers):
        if policy not in POLICIES[1:]:raise ValueError('Off uses no initialization helper')
        if policy=='sign-first-frozen' and auxiliary is None:raise ValueError('Sign control requires aux base seed')
        self.policy=policy;self.auxiliary=auxiliary;self.batch_id=batch_id;self.ids=list(ids)
        self.conv={name for name,module in layers if type(module)==nn.Conv2d}
        self.kind={name:type(module).__name__ for name,module in layers}
        self.initialized={};self.events=[];self.audit_seconds=0.
    def _sign_map(self,shape,seed):
        generator=torch.Generator(device='cpu').manual_seed(seed)
        return (torch.randint(0,2,tuple(shape),generator=generator,device='cpu',dtype=torch.int64)*2-1).to(torch.float32)
    def __call__(self,step,name,old,proposal):
        with torch.no_grad():
            initialized=self.initialized.get(name)
            if initialized is None:initialized=torch.zeros(old.shape[0],dtype=torch.bool,device=old.device)
            nonzero=(proposal.detach()!=0).reshape(old.shape[0],-1).any(1)
            accept=(~initialized)&nonzero
            # Match the old first_nonzero_update primitive even for signed
            # zero: uninitialized samples install their raw proposal without
            # marking initialization; frozen samples preserve old bytes.
            result=torch.where(initialized.view(old.shape[0],*([1]*(old.ndim-1))),old.detach(),proposal.detach())
            indices=torch.nonzero(accept,as_tuple=False).flatten().tolist()
            if indices:
                if old.device.type=='cuda':torch.cuda.synchronize(old.device)
                start=time.perf_counter();events=[]
                for index in indices:
                    before=proposal.detach()[index];key=None;seed=None
                    if self.policy=='sign-first-frozen' and name in self.conv:
                        key,seed=sign_key(self.auxiliary,self.batch_id,name,self.ids[index])
                        signs=self._sign_map(before.shape,seed).to(device=before.device,dtype=before.dtype)
                    else:signs=torch.ones_like(before)
                    # Shared L/R multiplication and assignment path. FC is
                    # always identity-mapped, then frozen in both policies.
                    after=before*signs;result[index]=after
                    events.append((index,key,seed,signs))
                # Author-shaped FP32 whole-layer per-sample norm, observation
                # only. Inf/NaN norm is logged, never a new acceptance gate.
                before_norm=proposal.detach().reshape(old.shape[0],-1).norm(dim=1)
                after_norm=result.detach().reshape(old.shape[0],-1).norm(dim=1)
                for index,key,seed,signs in events:
                    before=proposal.detach()[index].cpu();after=result.detach()[index].cpu();signs=signs.detach().cpu()
                    support=before!=0;x=before.to(torch.float64);s=signs.to(torch.float64);squares=x.square()
                    total=float(squares.sum());fourth=float(squares.square().sum())
                    rho=float((s*squares).sum())/total if total else None
                    neff=total*total/fourth if fourth else None
                    event={'step':step,'layer':name,'layer_type':self.kind[name],'image_id':self.ids[index],
                        'batch_id':self.batch_id,'initialization_control':self.policy,'aux_base_seed':self.auxiliary,
                        'aux_key':key,'aux_seed':seed,'sign_map_sha256':content_sha(signs),'shape':list(before.shape),
                        'dtype':str(before.dtype),'nnz':int(torch.count_nonzero(before)),
                        'before_abs_sha256':content_sha(before.abs()),'after_abs_sha256':content_sha(after.abs()),
                        'before_support_sha256':content_sha(support),'after_support_sha256':content_sha(after!=0),
                        'before_fp32_norm':audit_norm(before_norm[index]),'after_fp32_norm':audit_norm(after_norm[index]),
                        'rho':rho if rho is not None and math.isfinite(rho) else None,
                        'neff':neff if neff is not None and math.isfinite(neff) else None,
                        'support_global_plus':bool((signs[support]==1).all()),'support_global_minus':bool((signs[support]==-1).all()),
                        'scope':'First actual nonzero candidate only. FP64 CPU scalar audit is read-only; no normalization, rejection threshold or optimizer feedback.'}
                    self.events.append(event)
                if old.device.type=='cuda':torch.cuda.synchronize(old.device)
                self.audit_seconds+=time.perf_counter()-start
            self.initialized[name]=initialized|accept
            return result.detach()
    def metadata(self):
        return {'policy':self.policy,'aux_base_seed':self.auxiliary,'first_events':list(self.events),
            'first_event_count':len(self.events),'synchronized_first_accept_audit_seconds':self.audit_seconds,
            'seed_definition':'JSON UTF8 [auxbase,pairID,qualifiedLayer,originalImageID,purpose], SHA256 first8 bytes big-endian masked63 bits; main seed/event step/support excluded',
            'scope':'All-layer first actual nonzero freeze; Conv-only full-shape coordinate sign replacement in R; learned FC in L/R, no extra F/B/RNG from audit.'}
