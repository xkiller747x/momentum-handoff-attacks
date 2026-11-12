"""Bounded next-image Conv-state observation; never changes update acceptance.

Pinned FeatureTuning.forward replaces mixing_triggered, its eligible hook writes
that dictionary, and trace-only hooks write output_norms. Record mode alone
writes clean outputs/u; this probe requires record=False and trace_updates=False.
There is no lazy feature/partner cache in this path. Finally restores generator
object/state, u dictionary/tensor references and both forward-written dictionaries.
"""
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn

from baseline.targeted import rdi,target_metrics
from .io import sha256
from .metrics import image_seed

MODES=('off','reference','observe')
BRANCHES=('current','previous','zeroConv','randomConv')


def validate_mode(args):
    mode=getattr(args,'ftm_efficacy_probe','off')
    if mode not in MODES:raise ValueError('Unknown FTM efficacy probe mode')
    if mode!='off':
        if args.trace_updates:raise ValueError('Efficacy reference/observe require trace-updates off')
        if not getattr(args,'deterministic',False):raise ValueError('Efficacy reference/observe require explicit strict deterministic mode')
        if args.source!='rn50' or args.batch_size!=2 or getattr(args,'ftm_input_profile','canonical224-v2')!='canonical224-v2' or args.ftm_layer_profile!='mapped299' or args.ftm_update_mode!='full' or getattr(args,'ftm_beta',.01)!=.01:
            raise ValueError('Efficacy probe requires RN50V2/canonical224/mapped299/batch2/full/beta.01')
    return mode


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def state_sha(generator):return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
def tensor_sha(value):return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def terminal_u_digest(context):
    return {name:{'dtype':str(value.dtype),'shape':list(value.shape),'sha256':tensor_sha(value)} for name,value in context.outputs_tuning.items()}


def matched_random_conv(current,generator):
    """Whole-layer, per-sample FP32 matching; no fallback or acceptance call."""
    if current.dtype!=torch.float32:raise ValueError('Random Conv matching requires FP32 u')
    flat=current.detach().reshape(current.shape[0],-1);norm=flat.norm(dim=1)
    nonzero=(flat!=0).any(1)
    if not torch.isfinite(norm).all():raise ValueError('Nonfinite current Conv FP32 norm in diagnostic')
    if ((norm==0)&nonzero).any():raise ValueError('Current Conv FP32 norm underflow with nonzero coordinates in diagnostic')
    draw=torch.randn(tuple(current.shape),generator=generator,device='cpu',dtype=torch.float32).to(current.device)
    draw_norm=draw.reshape(draw.shape[0],-1).norm(dim=1);active=norm>0
    if ((~torch.isfinite(draw_norm)|(draw_norm==0))&active).any():raise ValueError('Invalid isotropic random Conv FP32 normalization; no resampling')
    result=torch.zeros_like(current)
    if active.any():
        dimensions=(-1,*([1]*(current.ndim-1)))
        result[active]=(draw[active]/draw_norm[active].view(*dimensions))*norm[active].view(*dimensions)
    if not torch.isfinite(result).all():raise ValueError('Nonfinite matched random Conv state')
    return result.detach(),{'dtype':'torch.float32','axis':'whole-layer reshape[B,-1], per sample',
        'current_norm':norm.tolist(),'random_draw_norm':draw_norm.tolist(),
        'matched_random_actual_norm':result.reshape(result.shape[0],-1).norm(dim=1).tolist(),
        'rule':'FP32 draw/draw_norm*current_norm; true zero current maps zero; rounding retained, not exact effective-v matching'}


class ConvEfficacyProbe:
    def __init__(self,path,context,ids,targets,labels,seed,batch_id,steps,counter,identity):
        if context.record or context.trace_updates:raise ValueError('Conv efficacy requires inference feature mode and traceFalse')
        base=Path(path);base.parent.mkdir(parents=True,exist_ok=True);self.path=base;attempt=1
        while self.path.exists():
            attempt+=1;self.path=base.with_name(base.stem+f'.attempt{attempt}'+base.suffix)
        self.stream=self.path.open('x',encoding='utf-8');self.closed=False
        self.context=context;self.ids=list(ids);self.targets=[int(v) for v in targets];self.labels=list(labels)
        self.seed=seed;self.batch_id=batch_id;self.steps=steps;self.counter=counter;self.identity=identity
        self.conv_names=[name for name,module in context.layers if type(module)==nn.Conv2d and name in context.outputs_tuning]
        self.first={name:torch.full((len(ids),),-1,dtype=torch.int64,device=context.outputs_tuning[name].device) for name in self.conv_names}
        self.old={};self.new_init={};self.pending_step=None;self.rows=0;self.seconds=0.;self.capture_seconds=0.;self.last_step=0
    def capture(self,step,name,old,_gradient,after,_context):
        if name not in self.first:return
        sync(old.device);start=time.perf_counter()
        if self.pending_step!=step:self.old={};self.new_init={};self.pending_step=step
        self.old[name]=old.detach().clone()
        changed=(after.detach()!=old.detach()).reshape(len(self.ids),-1).any(1)
        nonzero=(after.detach()!=0).reshape(len(self.ids),-1).any(1)
        initialized=(self.first[name]<0)&changed&nonzero
        self.first[name]=torch.where(initialized,torch.full_like(self.first[name],step),self.first[name])
        self.new_init[name]=initialized.tolist()
        sync(old.device);self.capture_seconds+=time.perf_counter()-start
    def observe(self,step,image,context):
        if context is not self.context or context.record or context.trace_updates:raise ValueError('Unexpected FTM observation context mode')
        sync(image.device);start=time.perf_counter()
        if self.pending_step!=step:self.old={};self.new_init={};self.pending_step=step
        main_generator=context.generator;main_state=main_generator.get_state().clone();main_before=state_sha(main_generator)
        cuda_before=hashlib.sha256(torch.cuda.get_rng_state(image.device).numpy().tobytes()).hexdigest() if image.device.type=='cuda' else None
        mapping=context.outputs_tuning;mixing=context.mixing_triggered;norms=context.output_norms;phase=self.counter.phase
        updated=sorted(self.old);main_selected=sorted(name for name,flag in mixing.items() if flag)
        aux_seed=image_seed(self.seed,f'{self.batch_id}:ftm-efficacy-view:{step}')
        random_seed=image_seed(self.seed,f'{self.batch_id}:ftm-efficacy-randomConv:{step}')
        aux=torch.Generator(device='cpu').manual_seed(aux_seed);random=torch.Generator(device='cpu').manual_seed(random_seed)
        logits={};end_hashes={};fresh_mixed={};random_norms={};main_after=None;cuda_after=None
        try:
            with torch.no_grad():
                # Detached clone prevents any diagnostic alias to x_(t+1).
                view=rdi(image.detach().clone(),aux).detach();input_before=tensor_sha(view)
                replay=aux.get_state().clone();replay_sha=state_sha(aux)
                previous=dict(mapping);previous.update(self.old)
                zero=dict(mapping);random_state=dict(mapping)
                for name in self.conv_names:
                    zero[name]=torch.zeros_like(mapping[name])
                    random_state[name],random_norms[name]=matched_random_conv(mapping[name],random)
                alternatives={'current':dict(mapping),'previous':previous,'zeroConv':zero,'randomConv':random_state}
                context.generator=aux
                for branch in BRANCHES:
                    aux.set_state(replay);context.outputs_tuning=alternatives[branch]
                    self.counter.phase='efficacy_'+branch
                    output=context.forward(view)
                    if output.dtype!=torch.float32 or not torch.isfinite(output).all():raise ValueError('Diagnostic source logits must be finite FP32')
                    logits[branch]=output.detach().clone()
                    fresh_mixed[branch]=sorted(name for name,flag in context.mixing_triggered.items() if flag)
                    end_hashes[branch]=state_sha(aux)
                input_after=tensor_sha(view);main_after=state_sha(main_generator)
                if image.device.type=='cuda':cuda_after=hashlib.sha256(torch.cuda.get_rng_state(image.device).numpy().tobytes()).hexdigest()
                if input_before!=input_after:raise ValueError('Diagnostic formed RDI input changed across branches')
                if len(set(end_hashes.values()))!=1 or any(v!=fresh_mixed['current'] for v in fresh_mixed.values()):raise ValueError('Diagnostic four-branch auxiliary replay differs')
                if main_after!=main_before:raise ValueError('Diagnostic consumed the main attack generator')
                if cuda_before!=cuda_after:raise ValueError('Diagnostic changed current-device CUDA global RNG')
        finally:
            context.generator=main_generator;main_generator.set_state(main_state)
            context.outputs_tuning=mapping;context.mixing_triggered=mixing;context.output_norms=norms
            self.counter.phase=phase
            sync(image.device);elapsed=time.perf_counter()-start;self.seconds+=elapsed
        for index,identity in enumerate(self.ids):
            scores={}
            for branch,output in logits.items():
                value=output[index,self.targets[index]].detach().cpu().contiguous()
                metric=target_metrics(output[index:index+1],self.targets[index],self.labels[index])
                scores[branch]={key:metric[key] for key in ('target_logit','target_margin','target_rank')}
                scores[branch]['target_logit_fp32_bits']=int(value.reshape(1).view(torch.int32).item())&0xffffffff
                scores[branch]['source_logits_sha256']=tensor_sha(output[index])
                scores[branch]['source_logits_dtype']=str(output.dtype)
                scores[branch]['source_logits_shape']=list(output[index].shape)
            initialized={name:self.new_init.get(name,[False]*len(self.ids))[index] for name in self.conv_names}
            current=scores['current']['target_logit']
            row={'id':identity,'batch_id':self.batch_id,'cluster_id':self.batch_id,'full_seed':self.seed,'step':step,
                'terminal':step==self.steps,'image_state':'post-image-update projected float x_(t+1), before PNG',
                'identity':self.identity,'aux_seed':aux_seed,'random_seed':random_seed,'aux_replay_start_sha256':replay_sha,
                'aux_end_sha256':end_hashes,'random_end_sha256':state_sha(random),'main_rng_before_sha256':main_before,
                'main_rng_after_sha256':main_after,'main_rng_unchanged':main_before==main_after,
                'cuda_global_rng_before_sha256':cuda_before,'cuda_global_rng_after_sha256':cuda_after,
                'cuda_global_rng_unchanged':cuda_before==cuda_after if cuda_before is not None else None,
                'cuda_global_rng_scope':'Current initialized image device only; read-only, no restoration' if cuda_before is not None else 'not applicable: CPU input',
                'formed_rdi_input_sha256_before':input_before,'formed_rdi_input_sha256_after':input_after,
                'conv_updated_layers':updated,'main_selected_layers':main_selected,'fresh_mixed_layers':fresh_mixed,
                'conv_updated_layer_set_definition':'Selected Conv update callbacks; actual accepted state can be frozen.',
                'conv_new_initialization':initialized,'any_new_conv_initialization':any(initialized.values()),
                'conv_first_nonzero_step':{name:int(value[index]) if int(value[index])>=0 else None for name,value in self.first.items()},
                'random_norms':{name:{key:(value[index] if isinstance(value,list) else value) for key,value in values.items()} for name,values in random_norms.items()},
                'scores':scores,'previous_minus_current':scores['previous']['target_logit']-current,
                'zeroConv_minus_current':scores['zeroConv']['target_logit']-current,
                'randomConv_minus_current':scores['randomConv']['target_logit']-current,
                'previous_current_logit_bits_equal':scores['previous']['target_logit_fp32_bits']==scores['current']['target_logit_fp32_bits'],
                'previous_current_whole_logits_equal':scores['previous']['source_logits_sha256']==scores['current']['source_logits_sha256'],
                'effect_precision':'Host float64 difference of saved FP32 target logits; positive previous-current means suppression on this fresh next-image view',
                'source_forward_images_this_step':4*len(self.ids),'completed_input_gradient_images_this_step':len(self.ids),
                'synchronized_observe_seconds_batch':elapsed,'capture_seconds_cumulative_batch':self.capture_seconds}
            self.stream.write(json.dumps(row,allow_nan=False)+'\n');self.rows+=1
        self.stream.flush();self.last_step=step;self.old={};self.new_init={}
    def close(self):
        if not self.closed:self.stream.flush();self.stream.close();self.closed=True
    def metadata(self):
        self.close()
        return {'path':str(self.path),'sha256':sha256(self.path),'rows':self.rows,'last_complete_step':self.last_step,
            'synchronized_observe_seconds':self.seconds,'capture_seconds':self.capture_seconds,
            'observe_timing_scope':'Synchronized aux input/random state construction, four forwards, hashes and context restoration; scalar score extraction/JSON additionally included in driver wall.',
            'scope':'Next-image total Conv-state interventions, common actual FC u; no counterfactual acceptance calls or target models. Partial rows retained on error; instrumented wall is not attack efficiency.'}
