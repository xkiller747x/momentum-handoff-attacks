"""Author pure-hook oracle and leaf Jacobian measurement; no real network load."""
import ast
import copy
from pathlib import Path
import torch
from torch import nn
from baseline.rapa import AuthorMask,generator_sha,selected_modules
from .io import sha256,write_json

SOURCE=Path(__file__).resolve().parents[1]/'test'/'reference'/'rapa_dropconnect.py.txt'


def author_functions():
    if sha256(SOURCE)!='445540161ecf2ec27771e7db19deefa9f3e07d6e970b3f0fafaefbe4e84bd86a':raise ValueError('Pinned RaPA hook source SHA differs')
    names={'dropconnect_forward_pre_hook','dropconnect_forward_hook','is_selected_layer','is_selected_type_layer','is_selected_position_layer'}
    tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
    pure=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    namespace={'torch':torch,'nn':nn};exec(compile(pure,str(SOURCE),'exec'),namespace)
    return namespace


def errors(a,b):
    x,y=a.detach().double(),b.detach().double();delta=x-y
    return {'exact_equal':torch.equal(a.detach(),b.detach()),'max_abs':float(delta.abs().max()),
            'relative_l2':float(delta.norm()/x.norm()) if float(x.norm()) else None,'absolute_l2':float(delta.norm())}


def _evaluate(model,x,p,seed,oracle,namespace):
    modules=selected_modules(model);handles=[];masked={};outputs={}
    device=x.device;generator=torch.Generator(device=device).manual_seed(seed)
    if oracle:
        if device.type=='cpu':torch.random.default_generator.manual_seed(seed)
        else:torch.cuda.default_generators[device.index if device.index is not None else torch.cuda.current_device()].manual_seed(seed)
        for _name,module in modules:
            module.drop_prob=p
            handles.append(module.register_forward_pre_hook(namespace['dropconnect_forward_pre_hook']))
            handles.append(module.register_forward_hook(namespace['dropconnect_forward_hook']))
        context=None
    else:
        context=AuthorMask(model,generator,p);context.__enter__()
    try:
        for name,module in modules:
            def capture(module,_inputs,key=name):
                masked[key]={'weight':module.weight.detach().clone(),'bias':module.bias.detach().clone() if module.bias is not None else None}
            def output(_module,_inputs,value,key=name):outputs[key]=value.detach().clone()
            handles.extend([module.register_forward_pre_hook(capture),module.register_forward_hook(output)])
        image=x.detach().clone().requires_grad_(True)
        result=model(image) if oracle else context.forward(image)
        coefficients=torch.linspace(.25,1,result.numel(),device=device).reshape_as(result)
        gradient=torch.autograd.grad((result*coefficients).sum(),image)[0]
        state=(torch.random.get_rng_state() if device.type=='cpu' else torch.cuda.get_rng_state(device)) if oracle else generator.get_state()
        return result.detach(),gradient.detach(),masked,outputs,state
    finally:
        for handle in handles:handle.remove()
        if context:context.__exit__(None,None,None)
        else:
            for _name,module in modules:
                if hasattr(module,'original_weight'):module.weight.data=module.original_weight
                if module.bias is not None and hasattr(module,'original_bias'):module.bias.data=module.original_bias


def semantic_probe(device='cpu',p=.05,seed=28):
    device=torch.device(device);devices=[] if device.type=='cpu' else [device.index if device.index is not None else torch.cuda.current_device()]
    namespace=author_functions();records=[]
    with torch.random.fork_rng(devices=devices):
        for kind in ('Linear','BatchNorm2d','Combined'):
            if kind=='Linear':model=nn.Linear(128,4);shape=(1,128)
            elif kind=='BatchNorm2d':model=nn.BatchNorm2d(128);shape=(1,128,2,2)
            else:model=nn.Sequential(nn.BatchNorm2d(128),nn.Flatten(),nn.Linear(128*2*2,4));shape=(1,128,2,2)
            model=model.to(device=device,dtype=torch.float32).eval()
            with torch.no_grad():
                for parameter in model.parameters():parameter.copy_(torch.linspace(.1,.9,parameter.numel(),device=device).reshape_as(parameter))
                for module in model.modules():
                    if isinstance(module,nn.BatchNorm2d):
                        module.running_mean.copy_(torch.linspace(-.2,.2,128,device=device));module.running_var.copy_(torch.linspace(.5,2,128,device=device))
            for parameter in model.parameters():parameter.requires_grad_(False)
            image=torch.linspace(-.3,.7,int(torch.tensor(shape).prod()),device=device).reshape(shape)
            reference=_evaluate(copy.deepcopy(model),image,p,seed,True,namespace)
            local=_evaluate(copy.deepcopy(model),image,p,seed,False,namespace)
            # Strict same-device oracle checks. Semantic classification below is
            # descriptive, never forced into a whole-network masked/original bin.
            torch.testing.assert_close(reference[0],local[0],rtol=0,atol=0)
            torch.testing.assert_close(reference[1],local[1],rtol=0,atol=0)
            if not torch.equal(reference[4],local[4]):raise ValueError('Author/local mask RNG state differs')
            row={'kind':kind,'p':p,'seed':seed,'torch_version':torch.__version__,'device':str(device),
                 'forward_author_local':errors(reference[0],local[0]),'input_gradient_author_local':errors(reference[1],local[1]),'rng_state_equal':True,'layers':[]}
            for name,module in selected_modules(model):
                masked=reference[2][name]
                row['layers'].append({'name':name,'type':type(module).__name__,
                    'weight_zero_elements':int((masked['weight']==0).sum()),'weight_elements':masked['weight'].numel(),
                    'bias_zero_elements':int((masked['bias']==0).sum()) if masked['bias'] is not None else None,
                    'output_author_local':errors(reference[3][name],local[3][name])})
            if kind!='Combined':
                masked=reference[2][''];coefficients=torch.linspace(.25,1,reference[0].numel(),device=device).reshape_as(reference[0])
                if kind=='Linear':
                    mathematical=coefficients@masked['weight'];original=coefficients@model.weight.detach()
                else:
                    denominator=(model.running_var+model.eps).sqrt()[None,:,None,None]
                    mathematical=coefficients*masked['weight'][None,:,None,None]/denominator
                    original=coefficients*model.weight.detach()[None,:,None,None]/denominator
                row['input_gradient_vs_explicit_masked_math']=errors(reference[1],mathematical)
                row['input_gradient_vs_original_jacobian']=errors(reference[1],original)
            records.append(row)
    return records


def probe_command(args):
    from .runner import _device,_output
    output=_output(args.output);device=_device(args.device)
    records=[row for p in args.probabilities.split(',') for row in semantic_probe(device,float(p),args.seed)]
    report={'records':records,'model_downloads':0,'target_forward_images':0,
        'interpretation':'Same-device pinned author-hook semantics measured on toy Linear, eval BN, and composition. Explicit masked mathematics and original Jacobian are comparisons, not assumed whole-network answers. No new attack repair.'}
    write_json(output/'semantic_probe.json',report);return report
