"""Local plain-MI AA/GAA/exact-PAA and logit MI/RTM numerical primitives."""
import math
import torch
import torch.nn.functional as F

FEATURE_METHODS=('aa','gaa','paa-poly')
METHODS=('logit-mi','logit-rtm',*FEATURE_METHODS)
PROFILES={'logit-mi':'target-logit-plain-mi', 'logit-rtm':'target-logit-rdi-nearest-ti5-mi',
          'aa':'aa-l2-plain-mi-local224', 'gaa':'gaa-mean-std-plain-mi-local224',
          'paa-poly':'paa-exact-poly-c0-d2-plain-mi-local224'}


def feature_distance(source,reference,method):
    """Return one loss per image. PAA equals exact c=0,d=2 MMD /4C²."""
    if source.shape!=reference.shape or source.ndim!=4:raise ValueError('Matching NCHW features required')
    x,y=source.flatten(2),reference.flatten(2)
    if method=='aa':return torch.linalg.vector_norm((x-y).flatten(1),dim=1)
    if method=='gaa':
        mx,my=x.mean(2),y.mean(2)
        # vector_norm defines the zero-vector derivative as zero, with no epsilon
        # changing the author's population standard deviation forward value.
        sx=torch.linalg.vector_norm(x-mx.unsqueeze(2),dim=2)/math.sqrt(x.shape[2])
        sy=torch.linalg.vector_norm(y-my.unsqueeze(2),dim=2)/math.sqrt(y.shape[2])
        return ((mx-my).square()+(sx-sy).square()).sum(1)*x.shape[2]/x.shape[1]**2
    if method=='paa-poly':
        gram=x.bmm(x.transpose(1,2))-y.bmm(y.transpose(1,2))
        return gram.square().sum((1,2))/(4*x.shape[2]**2*x.shape[1]**2)
    raise ValueError('Unknown feature loss')


def choose_reference(candidates,rule,index=None):
    ordered=sorted(candidates,key=lambda r:r['id'])
    if not ordered or any(not math.isfinite(r['distance']) for r in ordered):raise ValueError('Finite reference candidates required')
    if rule=='first':return ordered[0]
    if rule=='index':
        if index is None or not 0<=index<len(ordered):raise ValueError('Reference index outside sorted candidates')
        return ordered[index]
    if rule in ('nearest','farthest'):
        return min(ordered,key=lambda r:r['distance'] if rule=='nearest' else -r['distance'])
    raise ValueError('Unknown reference rule')


class FeatureTap:
    """Single forward hook scoped by a context manager; never stores old graphs."""
    def __init__(self,model,layer):
        self.module=model.get_submodule('model.'+layer if hasattr(model,'model') else layer)
        self.value=None;self.handle=None
    def __enter__(self):
        def capture(_module,_inputs,output):self.value=output
        self.handle=self.module.register_forward_hook(capture);return self
    def __exit__(self,*_args):
        self.handle.remove();self.value=None


class ForwardCounter:
    def __init__(self,model):
        self.model=model;self.phase='unspecified';self.counts={};self.resolutions={};self.handle=None
    def __enter__(self):
        def count(_module,inputs):
            n=inputs[0].shape[0];self.counts[self.phase]=self.counts.get(self.phase,0)+n
            shape='x'.join(map(str,inputs[0].shape[-2:]));self.resolutions[shape]=self.resolutions.get(shape,0)+n
        self.handle=self.model.register_forward_pre_hook(count);return self
    def __exit__(self,*_args):self.handle.remove()
    @property
    def total(self):return sum(self.counts.values())


def target_metrics(logits,target,label):
    if logits.shape[0]!=1 or not torch.isfinite(logits).all():raise ValueError('One finite logit vector required')
    z=logits[0];other=torch.cat((z[:target],z[target+1:]))
    prediction=int(z.argmax().item())
    return {'prediction':prediction,'true_correct':prediction==label,'targeted_success':prediction==target,
            'target_logit':float(z[target].item()),'target_margin':float((z[target]-other.max()).item()),
            'target_rank':1+int((z>z[target]).sum().item()),'rank_ties':'competition rank; top1 uses torch.argmax first-index tie',
            'target_log_odds':float((z[target]-torch.logsumexp(other,0)).item()),
            'target_confidence':float(z.softmax(0)[target].item())}


def rdi(image,generator):
    """Author draw order: resize, left, top. Dedicated CPU torch RNG."""
    size=image.shape[-1]
    if image.shape[-2]!=size:raise ValueError('RDI profile expects square canonical input')
    amount=int(size*340/299)-size
    if amount<=0:return image
    rnd=int(torch.rand(1,generator=generator).item()*amount)+size
    pad_max=size+amount-rnd
    left=int(torch.rand(1,generator=generator).item()*pad_max)
    top=int(torch.rand(1,generator=generator).item()*pad_max)
    enlarged=F.interpolate(image,size=(rnd,rnd),mode='nearest')
    padded=F.pad(enlarged,(left,pad_max-left,top,pad_max-top))
    return F.interpolate(padded,size=(size,size),mode='nearest') if size>64 else padded


def ti_kernel(image):
    # The author samples norm.pdf at linspace(-3,3,5), not pixel-coordinate σ3.
    points=torch.linspace(-3,3,5,dtype=torch.float64,device=image.device)
    one=torch.exp(-points.square()/2);kernel=one[:,None]*one[None,:];kernel/=kernel.sum()
    return kernel.to(image.dtype)[None,None].expand(image.shape[1],1,5,5).contiguous()


def optimize(model,clean,target,method,steps,epsilon_bytes,alpha_bytes,seed,
             tap=None,reference=None,checkpoints=(),callback=None,counter=None,state_callback=None):
    """One input gradient per step, endpoint forwards belong to caller callbacks."""
    if method not in METHODS or steps<0 or epsilon_bytes<0 or alpha_bytes<0:raise ValueError('Invalid attack settings')
    if method in FEATURE_METHODS and (tap is None or reference is None):raise ValueError('Feature attack needs captured layer and reference')
    generator=torch.Generator(device='cpu').manual_seed(seed)
    image=clean.detach().clone();momentum=torch.zeros_like(image);zero=0
    kernel=ti_kernel(clean) if method=='logit-rtm' else None
    for step in range(1,steps+1):
        x=image.detach().requires_grad_(True)
        if counter:counter.phase='optimization'
        logits=model(rdi(x,generator) if method=='logit-rtm' else x)
        loss=feature_distance(tap.value,reference,method).sum() if method in FEATURE_METHODS else logits[:,target].sum()
        if not torch.isfinite(loss):raise ValueError('Nonfinite attack objective')
        grad=torch.autograd.grad(loss,x)[0]
        if not torch.isfinite(grad).all():raise ValueError('Nonfinite input gradient')
        if kernel is not None:grad=F.conv2d(grad,kernel,padding=2,groups=x.shape[1])
        norm=grad.abs().sum((1,2,3),keepdim=True)
        zero+=int((norm==0).sum().item())
        # No positive-norm floor: preserve author L1 weighting, define zero/zero=0.
        momentum=momentum+grad/torch.where(norm>0,norm,torch.ones_like(norm))
        direction=-1 if method in FEATURE_METHODS else 1
        image=x.detach()+direction*(alpha_bytes/255)*momentum.sign()
        image=torch.minimum(torch.maximum(image,clean-epsilon_bytes/255),clean+epsilon_bytes/255).clamp(0,1).detach()
        if step in checkpoints and state_callback:state_callback(step,momentum)
        if step in checkpoints and callback:callback(step,image)
    if steps==0 and callback:callback(0,image)
    return image,{'input_gradient_images':steps*clean.shape[0],'zero_gradient_steps':zero,
                  'normalization':'per-sample L1; exact zero maps to zero; mu=1',
                  'randomness':'dedicated CPU torch.Generator seeded by SHA256(seed:imageID)'}
