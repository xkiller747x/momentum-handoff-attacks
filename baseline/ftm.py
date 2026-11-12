"""FTM author-config adaptation: dedicated RNG, explicit layer map, scoped hooks."""
from contextlib import contextmanager
import hashlib
import json
import torch
from torch import nn
import torch.nn.functional as F
from .targeted import rdi,ti_kernel


def author_layers(model,prefix=''):
    """Author recursive leaf Conv/Linear order, including the last linear."""
    children=list(model.named_children())
    if not children:return [(prefix,model)] if type(model) in (nn.Conv2d,nn.Linear) else []
    return [entry for name,child in children for entry in author_layers(child,f'{prefix}.{name}' if prefix else name)]


def qualification(model,device,size=299):
    layers=author_layers(model);shapes={};handles=[]
    try:
        for name,module in layers:
            def observe(_m,_i,output,key=name):shapes[key]=list(output.shape)
            handles.append(module.register_forward_hook(observe))
        with torch.no_grad():model(torch.zeros(1,3,size,size,device=device,dtype=torch.float32))
    finally:
        for handle in handles:handle.remove()
    entries=[{'name':name,'type':type(module).__name__,'qualification_shape':shapes.get(name),
        'eligible':name in shapes and (type(module)==nn.Linear or shapes[name][-1]<=size//4)} for name,module in layers]
    return entries


def layer_digest(entries):
    return hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class FeatureTuning:
    """Exact C/SelfShuffle/channelwise/M semantics; endpoint disable is state-free."""
    def __init__(self,model,generator,input_size=224,eligible_names=None,prob=.1,beta=.01,trace_updates=False):
        self.model=model;self.generator=generator;self.input_size=input_size
        self.eligible_names=None if eligible_names is None else set(eligible_names)
        self.prob=prob;self.beta=beta;self.outputs={};self.outputs_tuning={};self.mixing_triggered={}
        self.enabled=True;self.record=False;self.handles=[];self.layers=author_layers(model)
        self.trace_updates=trace_updates;self.output_norms={}
    def __enter__(self):
        try:
            for name,module in self.layers:self.handles.append(module.register_forward_hook(self._hook(name)))
        except BaseException:
            self.close();raise
        return self
    def __exit__(self,*_args):self.close()
    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[];self.outputs.clear();self.outputs_tuning.clear();self.mixing_triggered.clear();self.output_norms.clear()
    @contextmanager
    def disabled(self):
        previous=self.enabled;self.enabled=False
        try:yield
        finally:self.enabled=previous
    def _hook(self,name):
        def hook(module,_inputs,output):
            if not self.enabled:return None
            eligible=(name in self.eligible_names if self.eligible_names is not None else type(module)==nn.Linear or output.shape[-1]<=self.input_size//4)
            if not eligible:return None
            if self.record:
                self.outputs[name]=output.clone().detach()
                self.outputs_tuning[name]=torch.zeros_like(output).detach().requires_grad_(True)
                self.mixing_triggered[name]=False
                return None
            if name not in self.outputs:raise ValueError('FTM eligible layer missing clean cache')
            selected=torch.rand(1,generator=self.generator).item()<=self.prob
            self.mixing_triggered[name]=selected
            if selected:
                previous=self.outputs[name].clone().detach()
                permutation=torch.randperm(output.shape[0],generator=self.generator)
                previous=previous[permutation.to(previous.device)].view(previous.size())
                ratio=(torch.rand(previous.shape[0],previous.shape[1],generator=self.generator)*.75).view(
                    previous.shape[0],previous.shape[1],*([1]*(previous.ndim-2))).to(output.device)
            norm=output.detach().reshape(output.shape[0],-1).norm(dim=1)
            if self.trace_updates:self.output_norms[name]=norm.detach().clone()
            perturb=self.outputs_tuning[name]
            perturb_norm=perturb.detach().reshape(output.shape[0],-1).norm(dim=1)
            scale=(self.beta*norm/(perturb_norm+1e-7)).view(output.shape[0],*([1]*(output.ndim-1)))
            if selected:return (1-ratio)*(output+perturb*scale)+ratio*previous
            return output+perturb.detach()*scale
        return hook
    def cache(self,image):
        self.record=True
        try:
            with torch.no_grad():return self.forward(image)
        finally:self.record=False
    def forward(self,image):
        if self.enabled:
            self.mixing_triggered={}
            if self.trace_updates:self.output_norms={}
        return self.model(image)


def first_nonzero_update(old,gradient,initialized):
    """Control only: form every proposal, accept until each sample actually moves."""
    proposal=old-gradient
    mask=initialized.view(old.shape[0],*([1]*(old.ndim-1)))
    updated=torch.where(mask,old,proposal)
    changed=(updated.detach()!=old.detach()).reshape(old.shape[0],-1).any(1)
    return updated,initialized|changed


def optimize_ftm(context,clean,targets,steps,epsilon_bytes,alpha_bytes,checkpoints=(),callback=None,counter=None,observer=None,
                 update_mode='full',update_observer=None,state_transform=None,after_gradient=None,diagnostic_observer=None):
    """Cache is caller-owned; steps counts input gradients, not the cache forward."""
    image=clean.detach().clone();momentum=torch.zeros_like(image);kernel=ti_kernel(clean)
    low=(clean-epsilon_bytes/255).clamp(0,1);high=(clean+epsilon_bytes/255).clamp(0,1)
    zero=0;trigger_counts={name:0 for name,_ in context.layers}
    if update_mode not in ('full','first-nonzero-only'):raise ValueError('Unknown FTM update mode')
    if state_transform is not None and update_mode!='full':raise ValueError('State transform requires full raw-proposal mode')
    initialized={}
    for step in range(1,steps+1):
        if diagnostic_observer:diagnostic_observer.before_step(step,image,momentum,context)
        x=image.detach().requires_grad_(True)
        if counter:counter.phase='optimization'
        logits=context.forward(rdi(x,context.generator))
        if diagnostic_observer:diagnostic_observer.after_forward(step,logits,context)
        loss=logits.gather(1,targets[:,None]).sum()
        if not torch.isfinite(loss):raise ValueError('Nonfinite FTM objective')
        names=[name for name,selected in context.mixing_triggered.items() if selected]
        gradients=torch.autograd.grad(loss,[x,*[context.outputs_tuning[name] for name in names]])
        if after_gradient:after_gradient(step)
        if any(not torch.isfinite(g).all() for g in gradients):raise ValueError('Nonfinite FTM gradient')
        if diagnostic_observer:diagnostic_observer.signal('raw_input_gradient',gradients[0])
        for name,gradient in zip(names,gradients[1:]):
            old=context.outputs_tuning[name]
            if update_mode=='full':
                # Preserve the original expression and operation order exactly.
                context.outputs_tuning[name]=(context.outputs_tuning[name]-gradient).detach().requires_grad_(True)
            else:
                mask=initialized.get(name)
                if mask is None:mask=torch.zeros(old.shape[0],dtype=torch.bool,device=old.device)
                proposal,initialized[name]=first_nonzero_update(old,gradient,mask)
                context.outputs_tuning[name]=proposal.detach().requires_grad_(True)
            if state_transform is not None:
                context.outputs_tuning[name]=state_transform(step,name,old,context.outputs_tuning[name]).detach().requires_grad_(True)
            if update_observer:update_observer(step,name,old,gradient,context.outputs_tuning[name],context)
            trigger_counts[name]+=1
        gradient=F.conv2d(gradients[0],kernel,padding=2,groups=clean.shape[1])
        if diagnostic_observer:diagnostic_observer.signal('ti_gradient',gradient)
        norm=gradient.abs().sum((1,2,3),keepdim=True);zero+=int((norm==0).sum().item())
        if diagnostic_observer:diagnostic_observer.signal('l1_norm',norm)
        momentum=momentum+gradient/torch.where(norm>0,norm,torch.ones_like(norm))
        if diagnostic_observer:diagnostic_observer.signal('momentum',momentum)
        image=torch.clamp(x.detach()+(alpha_bytes/255)*momentum.sign(),low,high).detach()
        if diagnostic_observer:diagnostic_observer.after_step(step,image,zero,context)
        if observer:observer(step,image,context)
        if step in checkpoints and callback:callback(step,image)
    if steps==0 and callback:callback(0,image)
    return image,{'input_gradient_images':steps*clean.shape[0],'zero_gradient_samples':zero,
        'selected_layer_counts':trigger_counts,'gradient_steps':steps,'cache_forward_calls':1}
