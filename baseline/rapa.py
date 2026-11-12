"""RaPA author .data pre-mask/post-restore semantics, without functional repair."""
from contextlib import contextmanager
import hashlib
import json
import torch
from torch import nn
import torch.nn.functional as F
from .targeted import rdi,ti_kernel


def selected_modules(model):
    return [(name,module) for name,module in model.named_modules() if isinstance(module,(nn.Linear,nn.BatchNorm2d))]


def module_table(model):
    return [{'name':name,'type':type(module).__name__,'weight_shape':list(module.weight.shape) if module.weight is not None else None,
             'bias_shape':list(module.bias.shape) if module.bias is not None else None} for name,module in selected_modules(model)]


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def generator_sha(generator):return hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()


class AuthorMask:
    def __init__(self,model,generator,p=.05):
        if not 0<=p<=1:raise ValueError('RaPA p must be in[0,1]')
        self.model=model;self.generator=generator;self.p=p;self.enabled=True;self.handles=[]
        self.modules=selected_modules(model);self.originals={};self.buffers={};self.pending=set()
        self.buffer_pointers={};self.initial_pointers={};self.touched=set()
        self.counts={'module_mask_calls':0,'module_restore_calls':0,'weight_masks':0,'bias_masks':0,'weight_restores':0,'bias_restores':0,'forced_restore_calls':0,'verified_forwards':0}
        for name,module in self.modules:
            if module.weight is None:raise ValueError('Author profile requires affine selected layers')
            self.initial_pointers[name]={'weight':module.weight.data_ptr(),'bias':module.bias.data_ptr() if module.bias is not None else None}
            self.originals[name]={'weight':module.weight.data.clone(),'bias':module.bias.data.clone() if module.bias is not None else None}
            if isinstance(module,nn.BatchNorm2d):
                if module.training:raise ValueError('RaPA BatchNorm must be eval')
                self.buffers[name]={key:value.detach().clone() for key,value in module.named_buffers()}
                self.buffer_pointers[name]={key:value.data_ptr() for key,value in module.named_buffers()}
    def __enter__(self):
        try:
            for name,module in self.modules:
                self.handles.append(module.register_forward_pre_hook(self._pre(name)))
                self.handles.append(module.register_forward_hook(self._post(name)))
        except BaseException:
            self.close();raise
        return self
    def __exit__(self,*_args):self.close()
    def _restore(self,name,module,forced=False):
        module.weight.data=self.originals[name]['weight'];self.counts['weight_restores']+=1
        if module.bias is not None:module.bias.data=self.originals[name]['bias'];self.counts['bias_restores']+=1
        self.counts['module_restore_calls']+=1;self.counts['forced_restore_calls']+=int(forced);self.pending.discard(name)
    def _pre(self,name):
        def pre(module,_inputs):
            if not self.enabled:return None
            self.pending.add(name);self.touched.add(name)
            mask=torch.bernoulli((1-self.p)*torch.ones_like(module.weight),generator=self.generator)
            module.weight.data=self.originals[name]['weight']*mask
            self.counts['weight_masks']+=1
            if module.bias is not None:
                bias_mask=(torch.rand(module.bias.shape,dtype=module.bias.dtype,device=module.bias.device,generator=self.generator)>self.p).float()
                module.bias.data=self.originals[name]['bias']*bias_mask;self.counts['bias_masks']+=1
            self.counts['module_mask_calls']+=1
        return pre
    def _post(self,name):
        def post(module,_inputs,_output):
            if self.enabled:self._restore(name,module)
        return post
    def verify(self,full=True):
        if self.pending:raise ValueError('RaPA forward left masked parameters pending')
        for name,module in self.modules:
            expected_weight=self.originals[name]['weight'].data_ptr() if name in self.touched else self.initial_pointers[name]['weight']
            expected_bias=self.originals[name]['bias'].data_ptr() if name in self.touched and module.bias is not None else self.initial_pointers[name]['bias']
            if module.weight.data_ptr()!=expected_weight or (module.bias is not None and module.bias.data_ptr()!=expected_bias):raise ValueError('RaPA parameter storage restoration failed')
            if full and (not torch.equal(module.weight.data,self.originals[name]['weight']) or (module.bias is not None and not torch.equal(module.bias.data,self.originals[name]['bias']))):raise ValueError('RaPA parameter restoration failed')
            for key,before in self.buffers.get(name,{}).items():
                current=dict(module.named_buffers())[key]
                if module.training or current.data_ptr()!=self.buffer_pointers[name][key] or (full and not torch.equal(current,before)):raise ValueError('RaPA changed BN running statistics')
        return True
    def forward(self,image):
        output=self.model(image);self.verify(full=False);self.counts['verified_forwards']+=1;return output
    @contextmanager
    def disabled(self):
        previous=self.enabled;self.enabled=False
        try:yield
        finally:self.enabled=previous
    def close(self):
        for name,module in self.modules:
            if name in self.pending:self._restore(name,module,forced=True)
        for handle in self.handles:handle.remove()
        self.handles=[];self.verify()
    def bn_record(self):
        result={}
        for name,module in self.modules:
            if name not in self.buffers:continue
            result[name]={key:{'shape':list(value.shape),'dtype':str(value.dtype),
                'before_sha256':hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest(),
                'after_sha256':hashlib.sha256(dict(module.named_buffers())[key].detach().cpu().numpy().tobytes()).hexdigest()} for key,value in self.buffers[name].items()}
        return result


def optimize_rapa(context,clean,target,steps,samples,epsilon_bytes,alpha_bytes,rdi_generator,checkpoints=(),callback=None,counter=None):
    if samples<1 or steps<0:raise ValueError('RaPA S>=1 and steps>=0 required')
    image=clean.detach().clone();momentum=torch.zeros_like(image);kernel=ti_kernel(clean);zero=0
    low=(clean-epsilon_bytes/255).clamp(0,1);high=(clean+epsilon_bytes/255).clamp(0,1)
    for step in range(1,steps+1):
        x=image.detach().requires_grad_(True);gradient=0
        for _sample in range(samples):
            if counter:counter.phase='optimization'
            logits=context.forward(rdi(x,rdi_generator));loss=logits[:,target].sum()
            if not torch.isfinite(loss):raise ValueError('Nonfinite RaPA logit objective')
            current=torch.autograd.grad(loss,x)[0]
            if not torch.isfinite(current).all():raise ValueError('Nonfinite RaPA input gradient')
            gradient=gradient+current  # Author MultiGetGradient sum, not mean.
        gradient=F.conv2d(gradient,kernel,padding=2,groups=clean.shape[1])
        norm=gradient.abs().sum((1,2,3),keepdim=True);zero+=int((norm==0).sum())
        momentum=momentum+gradient/torch.where(norm>0,norm,torch.ones_like(norm))
        image=torch.clamp(x.detach()+(alpha_bytes/255)*momentum.sign(),low,high).detach()
        if step in checkpoints and callback:callback(step,image)
    if steps==0 and callback:callback(0,image)
    return image,{'input_gradient_images':steps*samples*clean.shape[0],'zero_gradient_updates':zero,'gradient_aggregation':'S raw gradients summed; one TI/L1/MI per outer update; zero norm maps to zero'}
