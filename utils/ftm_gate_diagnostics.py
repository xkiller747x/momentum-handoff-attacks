"""Bounded read-only gate/numeric-chain diagnostics for the fixed FTM source."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .io import sha256,write_json


def validate_mode(args):
    mode=getattr(args,'ftm_gate_diagnostics','off')
    if mode not in ('off','reference','observe'):raise ValueError('Unknown FTM gate diagnostic mode')
    if mode!='off' and (args.source!='rn50' or args.batch_size!=2 or getattr(args,'ftm_input_profile','canonical224-v2')!='canonical224-v2' or args.ftm_layer_profile!='mapped299' or args.ftm_update_mode not in ('full','first-nonzero-only') or getattr(args,'ftm_state_control','raw')!='raw' or getattr(args,'ftm_efficacy_probe','off')!='off' or getattr(args,'ftm_initialization_control','off')!='off' or getattr(args,'ftm_beta',.01)!=.01 or not getattr(args,'deterministic',False)):
        raise ValueError('Gate diagnostics require strict RN50V2/canonical224/mapped299/batch2/beta.01/raw/full-or-first/probe-off/initialization-off')
    return mode


def layout(value):
    return {'shape':list(value.shape),'dtype':str(value.dtype),'device':str(value.device),
        'stride':list(value.stride()),'storage_offset':value.storage_offset(),
        'contiguous':value.is_contiguous(),'channels_last':value.is_contiguous(memory_format=torch.channels_last) if value.ndim==4 else None,
        'requires_grad':value.requires_grad}


def content(value):
    # copy=True is mandatory even on CPU: numpy aliases ONLY this private copy.
    private=value.detach().to(device='cpu',copy=True).contiguous()
    array=private.numpy()
    array=array.astype(array.dtype.newbyteorder('<'),copy=False)
    header=json.dumps({'shape':list(array.shape),'dtype':array.dtype.str},sort_keys=True,separators=(',',':')).encode('utf-8')+b'\n'
    exact=hashlib.sha256(header+array.tobytes(order='C')).hexdigest()
    finite=np.isfinite(array)
    nonfinite=int((~finite).sum());zeros=int((array==0).sum())
    # No NaN canonicalization. Alter zero signs ONLY through the private copy.
    if np.issubdtype(array.dtype,np.floating):array[array==0]=0
    canonical=hashlib.sha256(header+array.tobytes(order='C')).hexdigest()
    return {'bit_sha256':exact,'zero_canonical_sha256':canonical,'nonfinite_elements':nonfinite,'zero_elements':zeros}


def tensor_record(value,ids):
    if value.shape[0]!=len(ids):raise ValueError('Diagnostic tensor is not per-image on the fixed batch axis')
    return {'layout':layout(value),'images':[{'id':identity,**content(value[index])} for index,identity in enumerate(ids)]}


def rng_sha(generator):return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
def cuda_sha(device):return hashlib.sha256(torch.cuda.get_rng_state(device).numpy().tobytes()).hexdigest() if device.type=='cuda' else None


class GateDiagnostics:
    def __init__(self,directory,mode,context,ids,identity,expected_features=42,expected_relu_occurrences=49):
        if mode not in ('reference','observe'):raise ValueError('Off creates no gate helper')
        base=Path(directory);self.directory=base;attempt=1
        while self.directory.exists():
            attempt+=1;self.directory=base.with_name(base.name+f'.attempt{attempt}')
        self.directory.mkdir(parents=True)
        self.mode=mode;self.context=context;self.ids=list(ids);self.identity=identity
        self.expected_features=expected_features;self.expected_relu_occurrences=expected_relu_occurrences
        self.features=[(name,module) for name,module in context.layers if name in context.outputs_tuning]
        if len(self.features)!=expected_features:raise ValueError('Qualified post-FTM diagnostic coverage differs')
        self.relu=[(name,module) for name,module in context.model.named_modules() if isinstance(module,nn.ReLU)]
        self.handles=[];self.active=False;self.current=None;self.closed=False;self.status='partial';self.rows=0
        self.logits=[];self.logit_steps=[];self.last_step=0;self.summary=None
        self.path=self.directory/'steps.jsonl';self.stream=self.path.open('x',encoding='utf-8')
        once={'mode':mode,'image_ids':self.ids,'identity':identity,
            'source_eval':not context.model.training,'parameters_frozen':all(not p.requires_grad for p in context.model.parameters()),
            'batchnorm_eval':{name:not module.training for name,module in context.model.named_modules() if isinstance(module,nn.BatchNorm2d)},
            'qualified_modules':[name for name,_ in self.features],'relu_modules':[name for name,_ in self.relu],
            'expected_post_ftm_outputs':expected_features,'expected_relu_occurrences':expected_relu_occurrences,
            'coverage':'Post-injection qualified Conv/FC layouts and actual model input in both modes; repeated ReLU output>0 masks only in observe. RN50 stem MaxPool precedes qualified injection, not independently masked. No arbitrary graph coverage claim.',
            'semantics':'Target-logit SUM; feature proposal raw subtraction; detached feature/output norms; main CPU generator drives RDI/mix. Existing actual TI/L1/momentum/image tensors read, not recomputed. where/clamp/abs outside forward are covered only by stated signal chain.',
            'hashes':'dtype/shape header + C-order little-endian bytes from PRIVATE CPU copy; exact bits and zero-sign canonical hashes separate; NaN payload not canonicalized; layout separate.'}
        write_json(self.directory/'metadata.json',once)
        try:
            self.handles.append(context.model.register_forward_pre_hook(self._input))
            # Existing FTM hooks are already registered: these hooks see its
            # returned injected output, not the pre-injection module output.
            for name,module in self.features:self.handles.append(module.register_forward_hook(self._feature(name)))
            if mode=='observe':
                for name,module in self.relu:self.handles.append(module.register_forward_hook(self._relu(name)))
        except BaseException:
            self.close('failed');raise
    def _input(self,_module,inputs):
        if self.active:self.current['signals']['post_rdi_input']=tensor_record(inputs[0],self.ids)
    def _feature(self,name):
        def record(_module,_inputs,output):
            if self.active:
                occurrence=self.feature_occurrences.get(name,0)+1;self.feature_occurrences[name]=occurrence
                self.current['post_ftm_layouts'].append({'module':name,'occurrence':occurrence,'layout':layout(output)})
        return record
    def _relu(self,name):
        def record(_module,_inputs,output):
            if self.active:
                occurrence=self.relu_occurrences.get(name,0)+1;self.relu_occurrences[name]=occurrence
                mask=output.detach()>0
                self.current['relu_masks'].append({'module':name,'occurrence':occurrence,
                    'images':[{'id':identity,'positive_count':int(torch.count_nonzero(mask[index])),**content(mask[index])} for index,identity in enumerate(self.ids)]})
        return record
    def before_step(self,step,image,momentum,context):
        if self.active or context is not self.context or context.record or not context.enabled:raise ValueError('Unexpected active FTM diagnostic context')
        self.current={'step':step,'image_ids':self.ids,'signals':{},'post_ftm_layouts':[],
            'relu_masks':[] if self.mode=='observe' else None,'main_rng_before_rdi_sha256':rng_sha(context.generator),
            'cuda_global_rng_before_sha256':cuda_sha(image.device)}
        self.device=image.device;self.feature_occurrences={};self.relu_occurrences={}
        self.current['signals']['before_image']=tensor_record(image,self.ids)
        self.current['signals']['previous_momentum']=tensor_record(momentum,self.ids)
        self.current['pre_update_u']={name:tensor_record(context.outputs_tuning[name],self.ids) for name,_ in self.features}
        self.active=True
    def after_forward(self,step,logits,context):
        self.active=False
        if step!=self.current['step']:raise ValueError('Diagnostic step differs')
        if logits.dtype!=torch.float32:raise ValueError('Gate diagnostic logits must be actual FP32')
        self.current['signals']['logits']=tensor_record(logits,self.ids)
        # The only numeric archive is the bounded small actual logits array.
        self.logits.append(logits.detach().to(device='cpu',copy=True).contiguous().numpy())
        self.logit_steps.append(step)
        self.current['main_rng_after_forward_sha256']=rng_sha(context.generator)
        self.current['selected_layers']=dict(context.mixing_triggered)
        if len(self.current['post_ftm_layouts'])!=self.expected_features or any(self.feature_occurrences.get(name)!=1 for name,_ in self.features):raise ValueError('Actual post-injection output coverage differs')
        if self.mode=='observe' and len(self.current['relu_masks'])!=self.expected_relu_occurrences:raise ValueError('Actual repeated ReLU occurrence coverage differs')
    def signal(self,name,value):
        self.current['signals'][name]=tensor_record(value,self.ids)
        if name=='momentum':self.current['signals']['momentum_sign']=tensor_record(value.detach().sign(),self.ids)
    def after_step(self,step,image,zero_count,context):
        self.current['signals']['after_image']=tensor_record(image,self.ids)
        self.current['momentum_zero_gradient_samples_cumulative']=zero_count
        self.current['cuda_global_rng_after_sha256']=cuda_sha(image.device)
        before=self.current['cuda_global_rng_before_sha256'];after=self.current['cuda_global_rng_after_sha256']
        self.current['cuda_global_rng_unchanged']=before==after if before is not None else None
        self.current['cuda_global_rng_scope']='active initialized image device only, read-only/no restoration' if before is not None else 'not applicable: CPU'
        self.stream.write(json.dumps(self.current,allow_nan=False)+'\n');self.stream.flush();self.rows+=1;self.last_step=step
        if before!=after:raise ValueError('Gate diagnostic step changed active-device CUDA global RNG')
    def close(self,status='partial'):
        if self.closed:return self.summary
        self.active=False
        for handle in self.handles:handle.remove()
        self.handles=[];self.stream.flush();self.stream.close();self.closed=True;self.status=status
        archive=self.directory/'logits.npz'
        np.savez_compressed(archive,steps=np.asarray(self.logit_steps,dtype=np.int64),image_ids=np.asarray(self.ids),
            logits=np.stack(self.logits) if self.logits else np.empty((0,len(self.ids),0),dtype=np.float32))
        self.summary={'mode':self.mode,'status':status,'rows':self.rows,'last_complete_step':self.last_step,
            'steps_path':str(self.path),'steps_sha256':sha256(self.path),'logits_path':str(archive),'logits_sha256':sha256(archive),
            'logit_steps':self.logit_steps,'metadata_path':str(self.directory/'metadata.json'),'metadata_sha256':sha256(self.directory/'metadata.json'),
            'extra_forward_images':0,'extra_input_gradient_images':0}
        write_json(self.directory/'status.json',self.summary)
        self.current=None;self.logits=[]
        return self.summary
