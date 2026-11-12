"""Pinned CFM FeatureMixup oracle, zero-tolerance matched beta-zero checks."""
import ast
import argparse
import copy
import hashlib
from pathlib import Path
import random
import tempfile
from typing import Callable
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn,Tensor
import torch.nn.functional as F

from baseline.ftm import FeatureTuning,author_layers,optimize_ftm,qualification
from baseline.targeted import rdi,ti_kernel
from utils.models import NormalizeModel
from utils.targeted_ftm import ftm_beta,beta_profile,final_u_nonzero
from utils import targeted_ftm

REFERENCE=Path(__file__).parent/'reference'


def author_definition(probability=.1):
    path=REFERENCE/'cfm_attacks.py.txt'
    if hashlib.sha256(path.read_bytes()).hexdigest()!='89e856d6292fdace66952f87c11e4952c5ddbdc35b7dadbcee0964b45b16276e':raise ValueError('CFM attacks fixture SHA mismatch')
    config=REFERENCE/'cfm_config.py.txt'
    if hashlib.sha256(config.read_bytes()).hexdigest()!='bf60018b8d6ffbd67c2e5c9d9a9f4c3a18fb937edd401a71c4409c8349fd95cf':raise ValueError('CFM config fixture SHA mismatch')
    settings=ast.literal_eval(next(n.value for n in ast.parse(config.read_text(encoding='utf-8')).body if isinstance(n,ast.Assign)))
    settings=copy.deepcopy(settings);settings[578]['mix_prob']=probability
    pure=ast.Module(body=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='FeatureMixup'],type_ignores=[])
    namespace={'torch':torch,'nn':nn,'Tensor':Tensor,'Callable':Callable,'F':F,'exp_configuration':settings}
    exec(compile(pure,str(path),'exec'),namespace)
    return namespace['FeatureMixup'],settings[578]


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,4,1,stride=4,bias=False)
        self.second=nn.Conv2d(4,3,1,stride=2,bias=False);self.pool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Linear(3,3,bias=False)
        with torch.no_grad():
            for index,value in enumerate(self.parameters()):value.copy_(torch.linspace(.03+index*.01,.3+index*.01,value.numel()).reshape_as(value))
        self.eval()
        for value in self.parameters():value.requires_grad_(False)
    def forward(self,image):return self.fc(self.pool(self.second(self.conv(image))).flatten(1))


def capture_layers(model):
    rows=[];handles=[]
    for name,module in author_layers(model):
        handles.append(module.register_forward_hook(lambda _m,_i,output,key=name:rows.append((key,output.detach().clone()))))
    return rows,handles


# The CFM snapshots carry no redistribution license, so public copies of this repository omit them.
@unittest.skipUnless((REFERENCE/'cfm_attacks.py.txt').exists(),'CFM reference snapshot not distributed')
class CFMMatchedControlTests(unittest.TestCase):
    def test_author_cfm_beta_zero_batch1_batch2_probability_grid_exact(self):
        # Only Tensor.cuda is mapped to CPU in this test. Original AST formula,
        # selection boundary, ratio and randperm calls are unchanged.
        for batch in (1,2):
            for probability in (0.,.1,1.):
                with self.subTest(batch=batch,probability=probability),torch.random.fork_rng(devices=[]):
                    oracle,settings=author_definition(probability)
                    self.assertEqual((settings['mixed_image_type_feature'],settings['shuffle_image_feature'],settings['blending_mode_feature']),('C','SelfShuffle','M'))
                    self.assertEqual((settings['channelwise'],settings['divisor'],settings['mix_upper_bound_feature']),(True,4,.75))
                    author_model=Tiny();local_model=copy.deepcopy(author_model)
                    clean=torch.linspace(.1,.7,batch*3*224*224).reshape(batch,3,224,224);targets=torch.arange(batch)%3
                    python_state=random.getstate();numpy_state=np.random.get_state();torch.manual_seed(1234)
                    author=oracle(author_model,578,224);reference=clean.clone();momentum=torch.zeros_like(clean)
                    reference_grads=[];author_layers_seen=[];handles=[]
                    try:
                        author.start_feature_record()
                        with torch.no_grad():author(clean)
                        author.end_feature_record()
                        self.assertTrue(all(not v.requires_grad and v.grad_fn is None for v in author.outputs.values()))
                        author_layers_seen,handles=capture_layers(author_model)
                        low=(clean-16/255).clamp(0,1);high=(clean+16/255).clamp(0,1)
                        with patch.object(torch.Tensor,'cuda',lambda value,*_a,**_kw:value):
                            for _step in range(3):
                                image=reference.detach().requires_grad_(True)
                                logits=author(rdi(image,None));loss=logits.gather(1,targets[:,None]).sum()
                                gradient=torch.autograd.grad(loss,image)[0];reference_grads.append(gradient.detach().clone())
                                gradient=F.conv2d(gradient,ti_kernel(clean),padding=2,groups=3)
                                norm=gradient.abs().sum((1,2,3),keepdim=True)
                                momentum=momentum+gradient/torch.where(norm>0,norm,torch.ones_like(norm))
                                reference=torch.clamp(image.detach()+(.2/255)*momentum.sign(),low,high).detach()
                        reference_rng=torch.get_rng_state().clone()
                    finally:
                        for handle in handles:handle.remove()
                        author.remove_hooks()
                    generator=torch.Generator().manual_seed(1234);local_grads=[];selected_gradients=[]
                    raw_grad=torch.autograd.grad
                    def observe_gradient(*a,**kw):
                        gradients=raw_grad(*a,**kw);local_grads.append(gradients[0].detach().clone())
                        selected_gradients.extend(g.detach().clone() for g in gradients[1:])
                        for gradient in gradients[1:]:
                            self.assertTrue(torch.isfinite(gradient).all());self.assertEqual(int(torch.count_nonzero(gradient)),0)
                        return gradients
                    with FeatureTuning(local_model,generator,input_size=224,prob=probability,beta=0.) as context:
                        context.cache(clean)
                        self.assertTrue(all(not v.requires_grad and v.grad_fn is None for v in context.outputs.values()))
                        local_layers_seen,handles=capture_layers(local_model)
                        try:
                            with patch('torch.autograd.grad',side_effect=observe_gradient):
                                result,stats=optimize_ftm(context,clean,targets,3,16,.2)
                            for value in context.outputs_tuning.values():
                                self.assertTrue(torch.isfinite(value).all());self.assertEqual(int(torch.count_nonzero(value)),0)
                            before_rng=generator.get_state().clone();before_global=torch.get_rng_state().clone()
                            before_parameters={key:value.detach().clone() for key,value in context.outputs_tuning.items()}
                            before_image=result.clone()
                            self.assertEqual(final_u_nonzero(context),[0]*batch)
                            self.assertTrue(torch.equal(before_rng,generator.get_state()))
                            self.assertTrue(torch.equal(before_global,torch.get_rng_state()))
                            self.assertTrue(torch.equal(before_image,result))
                            for key,value in before_parameters.items():self.assertTrue(torch.equal(value,context.outputs_tuning[key]))
                        finally:
                            for handle in handles:handle.remove()
                        self.assertEqual(stats['input_gradient_images'],3*batch)
                    self.assertEqual(len(reference_grads),len(local_grads))
                    for a,b in zip(reference_grads,local_grads):self.assertTrue(torch.equal(a,b),'CFM input gradient differs; zero tolerance')
                    self.assertEqual(len(author_layers_seen),len(local_layers_seen))
                    for (name,a),(other,b) in zip(author_layers_seen,local_layers_seen):
                        self.assertEqual(name,other);self.assertTrue(torch.equal(a,b),f'CFM forward differs at {name}; zero tolerance')
                    self.assertTrue(torch.equal(reference,result),'CFM final image differs; zero tolerance')
                    self.assertTrue(torch.equal(reference_rng,generator.get_state()))
                    self.assertTrue(torch.equal(reference_rng,torch.get_rng_state()),'Local dedicated stream changed global CPU RNG')
                    self.assertEqual(random.getstate(),python_state)
                    after=np.random.get_state();self.assertEqual(after[0],numpy_state[0]);np.testing.assert_array_equal(after[1],numpy_state[1]);self.assertEqual(after[2:],numpy_state[2:])
                    if probability==1.:self.assertGreater(len(selected_gradients),0)
                    self.assertTrue(all(not m._forward_hooks for model in (author_model,local_model) for m in model.modules()))

    def test_beta_validation_default_and_profile_identity(self):
        base='ftm-author-config-mapped299-local224-batch2-update-full'
        self.assertEqual(ftm_beta(.01),.01);self.assertEqual(beta_profile(base,.01,'full'),base)
        self.assertEqual(ftm_beta(0.),0.);self.assertFalse(np.signbit(ftm_beta(-0.)))
        self.assertEqual(beta_profile(base,0.,'full'),'ftm-author-config-mapped299-local224-batch2-beta0-update-full')
        self.assertTrue(beta_profile(base,.02,'full').endswith('-beta0.02-update-full'))
        for value in (-.1,float('inf'),float('-inf'),float('nan')):
            with self.subTest(value=value),self.assertRaises(ValueError):ftm_beta(value)

    def test_handler_records_actual_beta_and_normalizes_negative_zero(self):
        class StopAfterConfig(Exception):pass
        with tempfile.TemporaryDirectory() as temporary:
            for value in (.01,0.,-0.):
                with self.subTest(beta=value):
                    arguments=argparse.Namespace(steps=0,epsilon_bytes=16.,alpha_bytes=2.,ftm_beta=value,
                        labels_csv='unused',batch_size=2,seed=28,checkpoints=None,output=temporary,
                        device='cpu',source='rn50',ftm_layer_profile='mapped299',ftm_update_mode='full',trace_updates=False)
                    spec={'input_size':224,'qualification_size':299,'profile_component':'local224','adaptation':'unchanged'}
                    with patch.object(targeted_ftm.runner,'_records',return_value=[{'id':'a'}]),patch.object(targeted_ftm,'input_spec',return_value=spec),patch.object(targeted_ftm,'target_labels',return_value={'a':1}),patch.object(targeted_ftm.runner,'_output',return_value=Path(temporary)),patch.object(targeted_ftm.runner,'_device',return_value=torch.device('cpu')),patch.object(targeted_ftm,'load_model',return_value=(nn.Identity(),{'name':'rn50'})),patch.object(targeted_ftm,'qualification',return_value=[{'name':'fc','eligible':True}]),patch.object(targeted_ftm,'sha256',return_value='labels'),patch.object(targeted_ftm.runner,'_config',return_value={}) as configuration,patch.object(targeted_ftm.runner,'_open_run',side_effect=StopAfterConfig):
                        with self.assertRaises(StopAfterConfig):targeted_ftm.attack_command(arguments)
                    saved=configuration.call_args.kwargs
                    self.assertEqual(saved['settings']['ftm_beta'],value)
                    self.assertFalse(np.signbit(saved['settings']['ftm_beta']))
                    self.assertEqual('beta0' in saved['profile'],value==0.)

    def test_unique_rn50_author_recursive_order_and_299_qualification_meta(self):
        from torchvision.models import resnet50
        oracle,_=author_definition()
        # No weights, image dataset, download or CUDA: only RN50 meta shapes.
        with torch.device('meta'):
            model=NormalizeModel(resnet50(weights=None),[.485,.456,.406],[.229,.224,.225]).eval()
        author=oracle(model,578,299)
        try:
            order=[]
            for handle in author.forward_hooks:
                names=[name for name,module in model.named_modules() if handle.id in module._forward_hooks]
                self.assertEqual(len(names),1);order.append(names[0])
            self.assertEqual(order,[name for name,_ in author_layers(model)])
            author.start_feature_record()
            with torch.no_grad():author(torch.zeros(1,3,299,299,device='meta'))
            author.end_feature_record()
            author_eligible={order[index] for index in author.outputs}
            # Keep the author hook in recording mode during read-only shape
            # passes: no mix draws or hard-coded .cuda call on meta tensors.
            author.start_feature_record()
            local=qualification(model,'meta',299)
            self.assertEqual(author_eligible,{r['name'] for r in local if r['eligible']})
            self.assertIn('model.fc',author_eligible)
            native=qualification(model,'meta',224)
            self.assertNotEqual(author_eligible,{r['name'] for r in native if r['eligible']})
        finally:author.remove_hooks()
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))
