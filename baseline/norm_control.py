"""Known per-view L2 principle applied to the unchanged OPS fresh sampler.

Gradients are wrt the original input, not the transformed view. This changes
the update vector field and is not an unbiased raw-CE gradient estimator.
"""
from dataclasses import asdict
import time

import torch
import torch.nn.functional as F

from .ops import apply_ops
from .ops_refresh import OPSRefreshContext

PROFILE='ops-fresh-perview-l2-wrtx-control'


def unit_l2(gradient):
    norm=torch.linalg.vector_norm(gradient.double())
    if norm.item()==0:return torch.zeros_like(gradient),0.
    # FP64 division also avoids casting a subnormal norm back to zero in FP32.
    return (gradient.double()/norm).to(dtype=gradient.dtype),norm.item()


def norm_description(norms,raw_mean=None,unit_mean=None):
    values=torch.tensor(norms,dtype=torch.float64)
    total=values.sum().item();squares=values.square().sum().item()
    largest=values.max().item() if len(values) else 0.
    result={'min_view_l2':values.min().item() if len(values) else None,'max_view_l2':largest,
        'zero_view_count':sum(n==0 for n in norms),'view_l2_below_1e_minus_12_count':sum(n<1e-12 for n in norms),
        'base_norm_share':norms[0]/total if total else 0.,'max_norm_share':largest/total if total else 0.,
        'base_is_tied_max':bool(norms[0]==largest),'norm_participation_ratio':total*total/squares if squares else 0.,
        'all_view_norms_zero':total==0,'view_count':len(norms),'view_l2':norms,
        'norm_scalar_dtype':'float64','gradient_and_update_dtype':'float32',
        'tiny_norm_rule':'1e-12 is descriptive only; every nonzero view is normalized without a floor'}
    if raw_mean is not None and unit_mean is not None:
        a,b=raw_mean.double().flatten(),unit_mean.double().flatten()
        na,nb=a.norm().item(),b.norm().item()
        result.update({'raw_mean_zero':na==0,'unit_mean_zero':nb==0,
            'raw_unit_cosine':float(torch.dot(a,b).item()/(na*nb)) if na and nb else None})
    return result


def view_gradient_mean(model,x,label,plans,counter=None,*,identity=False):
    if x.shape[0]!=1 or x.dtype!=torch.float32 or len(plans)!=26:
        raise ValueError('Control requires one FP32 image and exactly26 views')
    raw_total=torch.zeros_like(x);unit_total=torch.zeros_like(x);identity_total=torch.zeros_like(x)
    norms=[];views=[]
    for index,plan in enumerate(plans):
        leaf=x.detach().requires_grad_(True)
        transformed=apply_ops(leaf,plan)
        logits=model(transformed);ce=F.cross_entropy(logits,label,reduction='sum')
        # Internal identity reproduces the OLD loss-before-backward /26 path.
        differentiated=ce/len(plans) if identity else ce
        gradient=torch.autograd.grad(differentiated,leaf)[0].detach()
        if not torch.isfinite(gradient).all():raise FloatingPointError('Nonfinite original-input gradient')
        if identity:
            identity_total.add_(gradient)
            diagnostic_gradient=gradient*len(plans)
        else:diagnostic_gradient=gradient
        normalized,norm=unit_l2(diagnostic_gradient)
        raw_total.add_(diagnostic_gradient);unit_total.add_(normalized);norms.append(norm)
        views.append({'view':index,'base':index==0,'pool_index':plan.pool_index,'ce':ce.item(),'l2_wrt_original_x':norm,
                      'operators_outer_to_inner':[asdict(op) for op in plan.operators]})
        if counter is not None:
            counter['forward_images']=counter.get('forward_images',0)+1
            counter['input_gradient_images']=counter.get('input_gradient_images',0)+1
            pixels=transformed.shape[-2]*transformed.shape[-1]
            for key in ('forward_pixels','input_gradient_pixels'):counter[key]=counter.get(key,0)+pixels
            resolutions=counter.setdefault('resolution_counts',{});key=f'{transformed.shape[-2]}x{transformed.shape[-1]}'
            resolutions[key]=resolutions.get(key,0)+1
    raw_mean,unit_mean=raw_total/len(plans),unit_total/len(plans)
    return (identity_total if identity else unit_mean),norm_description(norms,raw_mean,unit_mean),views


def control_attack(model,clean,labels,method='ops-view-l2',steps=10,views=1,epsilon_bytes=8.,alpha_mult=1.,seed=0,
                   view_batch=1,ne=5,np_=5,beta=2.,profile=PROFILE,*,identity=False,**_unused):
    if (steps,views,epsilon_bytes,alpha_mult,view_batch,ne,np_,beta)!=(10,1,8.,1.,1,5,5,2.):
        raise ValueError('Frozen control requires T10,EPS8,alpha1,views1,viewbatch1,Ne=Np5,beta2')
    if method!='ops-view-l2' or profile not in (None,PROFILE):raise ValueError('Control method/profile mismatch')
    model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    clean=clean.detach().float();adv=clean.clone();momentum=torch.zeros_like(clean)
    cuda=clean.device.type=='cuda'
    if cuda:torch.cuda.synchronize(clean.device);torch.cuda.reset_peak_memory_stats(clean.device)
    start=time.perf_counter();context=OPSRefreshContext(clean.shape,8/255,5,5,2.,seed)
    stats={'forward_images':0,'input_gradient_images':0,'forward_pixels':0,'input_gradient_pixels':0,'resolution_counts':{},
           'zero_grad_count':0,'near_zero_grad_count':0,'normalization_floor':1e-12,'steps_norm_summary':[],
           'aggregation':'identity engineering only' if identity else 'each dCE/doriginal_x unit L2, zeros included in denominator26'}
    for step in range(10):
        plans=context.step_plans()
        gradient,description,_views=view_gradient_mean(model,adv,labels,plans,stats,identity=identity)
        stats['steps_norm_summary'].append({'step':step+1,**description})
        denominator=gradient.abs().mean(dim=(1,2,3),keepdim=True)
        stats['zero_grad_count']+=int((denominator==0).sum().item())
        stats['near_zero_grad_count']+=int((denominator<1e-12).sum().item())
        momentum=momentum+gradient/denominator.clamp_min(1e-12)
        adv=adv+(8/(255*10))*momentum.sign()
        adv=torch.maximum(torch.minimum(adv,clean+8/255),clean-8/255).clamp(0,1).detach()
    if cuda:torch.cuda.synchronize(clean.device)
    stats.update({'elapsed_seconds':time.perf_counter()-start,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(clean.device) if cuda else 0,
        'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(clean.device) if cuda else 0,**context.lifecycle_record()})
    # The sampler's lifecycle profile is recorded separately from this treatment.
    stats['sampler_profile']=stats.pop('profile');stats['profile']=PROFILE
    return adv,stats
