"""Pinned author pure-code oracle and local FTM lifecycle tests; no model loaders."""
import ast
import copy
import hashlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from typing import Callable

import numpy as np
import torch
from torch import nn,Tensor
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from baseline.ftm import FeatureTuning,author_layers,layer_digest,optimize_ftm,qualification
from baseline.targeted import ForwardCounter
from utils.targeted_ftm import atomic_progress,pair_table
from utils.io import read_jsonl

REFERENCE=Path(__file__).parent/'reference'


def author_namespace():
    source=REFERENCE/'ftm_attacks.py.txt'
    if hashlib.sha256(source.read_bytes()).hexdigest()!='4673e39590d3ad362563e4b90232b34bc8df3e5bc9df66b50d0819c2ade6e8ad':raise ValueError('FTM oracle fixture SHA mismatch')
    tree=ast.parse(source.read_text(encoding='utf-8'))
    names={'ftm_attack','FeatureTuning','LogitLoss','RDI','gkern'}
    pure=ast.Module(body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names],type_ignores=[])
    # Only normal PDF is stubbed to avoid a new scipy test dependency. This is
    # the exact PDF used by author gkern, not a replacement attack algorithm.
    normal=types.SimpleNamespace(pdf=lambda x:np.exp(-np.asarray(x)**2/2)/np.sqrt(2*np.pi))
    namespace=dict(torch=torch,nn=nn,Tensor=Tensor,F=F,np=np,st=types.SimpleNamespace(norm=normal),
                   transforms=transforms,InterpolationMode=InterpolationMode,Callable=Callable)
    exec(compile(pure,str(source),'exec'),namespace)
    return namespace


def settings(prob):
    path=REFERENCE/'ftm_config.py.txt'
    if hashlib.sha256(path.read_bytes()).hexdigest()!='84863e7c1104e5c7f266be42068dd2361e4820432ba1a82e8b7b1ee222493665':raise ValueError('FTM config fixture SHA mismatch')
    tree=ast.parse(path.read_text(encoding='utf-8'))
    config=ast.literal_eval(next(node.value for node in tree.body if isinstance(node,ast.Assign)))
    result=dict(config[1]);result.update(mix_prob=prob,alpha=.2)
    return result


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,4,1,stride=4,bias=False);self.second=nn.Conv2d(4,3,1,stride=2,bias=False)
        self.pool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Linear(3,3,bias=False)
        with torch.no_grad():
            for i,p in enumerate(self.parameters()):p.copy_(torch.linspace(.03+i*.01,.3+i*.01,p.numel()).reshape_as(p))
        for p in self.parameters():p.requires_grad_(False)
        self.eval()
    def forward(self,x):return self.fc(self.pool(self.second(self.conv(x))).flatten(1))


class FTMTests(unittest.TestCase):
    def assert_tensor_pair(self,a,b):
        a,b=a.detach(),b.detach()
        maxabs=float((a-b).abs().max());relative=float((a-b).norm()/(a.norm()+1e-30))
        torch.testing.assert_close(a,b,rtol=3e-5,atol=1e-5,msg=f'maxabs={maxabs}, relativeL2={relative}')

    def test_author_native_rtmf_batch1_batch2_probabilities_and_inner_states(self):
        for batch in (1,2):
            for probability in (0.,1.,.1):
                with self.subTest(batch=batch,probability=probability):
                    namespace=author_namespace();original=namespace['FeatureTuning'];instances=[]
                    class ObserveAuthor(original):
                        # Observation wrapper only. Original forward/hook/attack
                        # definitions execute unchanged from pinned author AST.
                        def __init__(self,*args,**kwargs):
                            super().__init__(*args,**kwargs);self.trace=[];instances.append(self)
                        def forward(self,x):
                            result=super().forward(x)
                            if not self.record:self.trace.append({k:v.detach().clone() for k,v in self.outputs_tuning.items()})
                            return result
                        def remove_hooks(self):
                            self.final_tuning={k:v.detach().clone() for k,v in self.outputs_tuning.items()};super().remove_hooks()
                    namespace['FeatureTuning']=ObserveAuthor
                    author_model=Tiny();local_model=copy.deepcopy(author_model)
                    x=torch.linspace(.1,.7,batch*3*224*224).reshape(batch,3,224,224);targets=torch.tensor([i%3 for i in range(batch)])
                    torch.manual_seed(1234)
                    with patch('torch.cuda.empty_cache'),ForwardCounter(author_model) as author_count:
                        reference=namespace['ftm_attack']([author_model],x,targets,target_label=targets,exp_settings=settings(probability),device='cpu',num_iter=4,max_epsilon=16)
                    self.assertEqual(author_count.total,4*batch)
                    expected_rng=torch.get_rng_state().clone();generator=torch.Generator().manual_seed(1234)
                    with FeatureTuning(local_model,generator,input_size=224,prob=probability) as context,ForwardCounter(local_model) as outer_count:
                        context.cache(x);traces=[]
                        def observe(_m,_i,_o):
                            if not context.record:traces.append({k:v.detach().clone() for k,v in context.outputs_tuning.items()})
                        handle=local_model.register_forward_hook(observe)
                        try:
                            with ForwardCounter(local_model) as count:
                                result,stats=optimize_ftm(context,x,targets,3,16,.2,counter=count)
                            self.assertEqual(count.total,3*batch);self.assertEqual(stats['input_gradient_images'],3*batch)
                            self.assertEqual(count.resolutions,{'224x224':3*batch})
                        finally:handle.remove()
                        names=[name for name,_module in context.layers]
                        for author_trace,local_trace in zip(instances[0].trace,traces):
                            for index,tensor in author_trace.items():self.assert_tensor_pair(tensor,local_trace[names[index]])
                        for index,tensor in instances[0].final_tuning.items():self.assert_tensor_pair(tensor,context.outputs_tuning[names[index]])
                        self.assertEqual(len(traces),3);self.assertIn('fc',context.outputs)
                        self.assertEqual(outer_count.total,4*batch)
                    self.assert_tensor_pair(reference,result);self.assertTrue(torch.equal(generator.get_state(),expected_rng))
                    self.assertTrue(all(not m._forward_hooks for m in local_model.modules()))

    def test_unselected_layers_keep_detached_perturbation_and_endpoint_is_state_free(self):
        model=Tiny();generator=torch.Generator().manual_seed(9);x=torch.full((2,3,224,224),.3,requires_grad=True)
        with FeatureTuning(model,generator,prob=0.) as context:
            context.cache(x)
            self.assertTrue(all(torch.count_nonzero(p)==0 for p in context.outputs_tuning.values()))
            for name in context.outputs_tuning:context.outputs_tuning[name]=torch.ones_like(context.outputs_tuning[name]).requires_grad_(True)
            before=generator.get_state().clone();snapshot={k:v.detach().clone() for k,v in context.outputs_tuning.items()};cached={k:v.clone() for k,v in context.outputs.items()}
            context.mixing_triggered={'sentinel':True}
            with context.disabled():
                clean=model(x)
                with self.assertRaises(RuntimeError):
                    with context.disabled():raise RuntimeError('endpoint failure')
            self.assertTrue(context.enabled);self.assertTrue(torch.equal(before,generator.get_state()))
            self.assertEqual(context.mixing_triggered,{'sentinel':True})
            for key in snapshot:self.assertTrue(torch.equal(context.outputs_tuning[key],snapshot[key]));self.assertTrue(torch.equal(context.outputs[key],cached[key]))
            perturbed=context.forward(x)
            self.assertFalse(torch.equal(clean,perturbed));self.assertTrue(all(not v for v in context.mixing_triggered.values()))
            gradient=torch.autograd.grad(perturbed.sum(),list(context.outputs_tuning.values()),allow_unused=True)
            self.assertTrue(all(value is None for value in gradient))

    def test_selected_fc_inner_descent_and_local_input_ascent(self):
        class Flat(nn.Module):
            def __init__(self):
                super().__init__();self.fc=nn.Linear(12,2,bias=False)
                with torch.no_grad():self.fc.weight.fill_(.2)
                self.fc.weight.requires_grad_(False)
            def forward(self,x):return self.fc(x.flatten(1))
        model=Flat();x=torch.full((1,3,2,2),.3)
        with FeatureTuning(model,torch.Generator().manual_seed(5),input_size=2,prob=1.) as context:
            context.cache(x);image,_=optimize_ftm(context,x,torch.tensor([1]),1,16,.001)
            self.assertLess(context.outputs_tuning['fc'][0,1].item(),0)
            self.assertEqual(context.outputs_tuning['fc'][0,0].item(),0)
            with context.disabled():self.assertGreater(model(image)[0,1].item(),model(x)[0,1].item())

    def test_qualification_boundary_mapping_includes_fc_and_cleanup(self):
        model=Tiny();original=[len(m._forward_hooks) for m in model.modules()]
        mapped=qualification(model,'cpu',299);native=qualification(model,'cpu',224)
        a={r['name']:r for r in mapped};b={r['name']:r for r in native}
        self.assertEqual(a['conv']['qualification_shape'][-1],75);self.assertFalse(a['conv']['eligible'])
        self.assertEqual(b['conv']['qualification_shape'][-1],56);self.assertTrue(b['conv']['eligible'])
        self.assertTrue(a['fc']['eligible']);self.assertNotEqual(layer_digest(mapped),layer_digest(native))
        with FeatureTuning(model,torch.Generator().manual_seed(1),eligible_names=[r['name'] for r in mapped if r['eligible']]) as context:
            context.cache(torch.ones(1,3,224,224));self.assertNotIn('conv',context.outputs);self.assertIn('second',context.outputs)
        self.assertEqual([len(m._forward_hooks) for m in model.modules()],original)

    def test_observer_and_endpoint_counter_do_not_change_delta_or_rng(self):
        model=Tiny();x=torch.full((2,3,224,224),.2);results=[];states=[]
        for observing in (False,True):
            generator=torch.Generator().manual_seed(2026)
            with FeatureTuning(model,generator) as context:
                context.cache(x)
                def endpoint(_step,image):
                    with context.disabled(),ForwardCounter(model) as count,torch.no_grad():model(image)
                    self.assertEqual(count.total,2)
                result,_=optimize_ftm(context,x,torch.tensor([1,2]),2,16,.1,
                    checkpoints=[1,2] if observing else [],callback=endpoint if observing else None,
                    observer=(lambda *_:None) if observing else None)
                results.append(result);states.append(generator.get_state())
        self.assertTrue(torch.equal(results[0],results[1]));self.assertTrue(torch.equal(states[0],states[1]))

    def test_fixed_pairs_atomic_rows_and_exception_hook_cleanup(self):
        records=[{'id':i} for i in ['z','b','a']]
        pairs=pair_table(records,2,28)
        self.assertEqual(pairs[0]['image_ids'],['a','b']);self.assertEqual(pairs[1]['image_ids'],['z']);self.assertTrue(pairs[1]['odd_singleton'])
        self.assertEqual(pair_table(list(reversed(records)),2,28),pairs)
        self.assertNotEqual(pair_table(records,2,29)[0]['rng_seed'],pairs[0]['rng_seed'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.jsonl';atomic_progress(path,[{'id':'a'},{'id':'b'}])
            self.assertEqual([r['id'] for r in read_jsonl(path)],['a','b'])
        model=Tiny()
        with self.assertRaises(RuntimeError):
            with FeatureTuning(model,torch.Generator().manual_seed(1)):raise RuntimeError('synthetic failure')
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))
