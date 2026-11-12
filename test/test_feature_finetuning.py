"""Pinned author AST oracle; CPU transports only, never upstream entrypoints.

Array/draw comparisons use zero tolerance. Parameter-backward vs frozen local
autograd.grad comparisons explicitly use atol2e-6/rtol2e-5 and report maxabs.
"""
import ast
import copy
import hashlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from baseline.feature_finetuning import (Layer3Source, diversity, finetune, gaussian,
                                        guidance, masked_input, patch_mask)
from baseline.targeted import ForwardCounter, optimize
from utils.io import read_json, read_pixels, save_adversarial, sha256, write_json
from utils.models import NormalizeModel
from utils.targeted_data import target_labels
from utils.targeted_finetuning import (assert_budget, fresh_output, pixel_sha, png299, records299)
from utils.targeted_finetuning import attack_command, guidance_scalars
from utils import runner
from utils.original299 import ORIGINAL_PREPROCESSING

REFERENCE = Path(__file__).parent/'reference'/'feature_finetuning'


def pinned(filename):
    provenance=read_json(REFERENCE/'fft_aaf_source_provenance_v1.json')
    entry=next(x for x in provenance if x['local_file']==filename)
    path=REFERENCE/filename
    if not path.exists():
        # Author code carries no redistribution license, so public copies omit these snapshots.
        raise unittest.SkipTest('author reference snapshot not distributed')
    if sha256(path)!=entry['sha256']:
        raise ValueError('Author fixture SHA mismatch')
    return path


def oracle(aaf=False):
    filename='aaf_attack.source.txt' if aaf else 'fft_attack.source.txt'
    path=pinned(filename)
    data=pinned('aaf_data.source.txt' if aaf else 'fft_data.source.txt')
    normal=types.SimpleNamespace(pdf=lambda x:np.exp(-np.asarray(x)**2/2)/np.sqrt(2*np.pi))
    ns=dict(torch=torch,nn=nn,F=F,np=np,st=types.SimpleNamespace(norm=normal),device='cpu')
    tree=ast.parse(data.read_text(encoding='utf-8'))
    functions=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Normalize','FIAloss','DI_keepresolution','gkern')]
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(data),'exec'),ns)
    ns['norm']=ns['Normalize']([.485,.456,.406],[.229,.224,.225])
    ns['kernel']=ns['gkern'](5,3).astype(np.float32)
    ns['gaussian_kernel']=torch.from_numpy(np.stack([ns['kernel']]*3)[:,None])
    tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
    # Read-only loop/return instrumentation: original mathematical statements
    # and their order are untouched. Never import caller/model loaders.
    for node in ast.walk(ast.Module(body=nodes,type_ignores=[])):
        if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='epoch':
            node.body.append(ast.parse('capture("step", locals())').body[0])
        if isinstance(node,ast.FunctionDef) and node.name=='perturb':
            index=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Return))
            node.body.insert(index,ast.parse('capture("final", locals())').body[0])
    ns['events']=[]
    def capture(kind,values):
        ns['events'].append((kind,{k:v.detach().clone() if isinstance(v,torch.Tensor) else v
                                  for k,v in values.items() if k in ('g','eta_prev','x_adv_ft','x_adv_2',
                                      'grad_sum_l3','grad_sum_mid_l3','grad_sum_new_l3','epoch')}))
    ns['capture']=capture
    pure=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    exec(compile(pure,str(path),'exec'),ns)
    return ns


class Toy(nn.Module):
    """Small299 four-layer author interface; spatial/channel gradients nonconstant."""
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,4,3,stride=16,padding=1)
        self.bn1=nn.Identity();self.relu=nn.Tanh();self.maxpool=nn.Identity()
        self.layer1=nn.Sequential(nn.Conv2d(4,4,1),nn.Tanh())
        self.layer2=nn.Sequential(nn.Conv2d(4,4,1),nn.Tanh())
        self.layer3=nn.Sequential(nn.Conv2d(4,4,1),nn.Tanh())
        self.layer4=nn.Sequential(nn.Conv2d(4,4,3,padding=1),nn.Tanh())
        self.avgpool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Linear(4,3)
        generator=torch.Generator().manual_seed(53)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand(p.shape,generator=generator)*.8-.4)
        self.eval()
    def multi_layer_features(self,x):
        x=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        a=self.layer1(x);b=self.layer2(a);c=self.layer3(b);d=self.layer4(c)
        return a,b,c,d
    def features_grad_multi_layers(self,x):
        features=self.multi_layer_features(x)
        for value in features:value.retain_grad()
        return self.fc(self.avgpool(features[-1]).flatten(1)),*features
    def forward(self,x):
        return self.fc(self.avgpool(self.multi_layer_features(x)[-1]).flatten(1))


def images():
    x=(torch.arange(3*299*299).reshape(1,3,299,299)%193).float()/255+.1
    initial=(x+torch.sin(torch.arange(x.numel()).reshape_as(x).float())*7/255).clamp(0,1)
    return x,initial


def local_model(network=None):
    wrapped=NormalizeModel(network or Toy(),[.485,.456,.406],[.229,.224,.225]).eval()
    for p in wrapped.parameters():p.requires_grad_(False)
    return Layer3Source(wrapped)


class FeatureFinetuningTests(unittest.TestCase):
    def close_tensor(self,a,b):
        error=float((a.detach()-b.detach()).abs().max())
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5,msg=f'parameter backward vs autograd.grad maxabs={error}')

    def test_pinned_patch_and_di_arrays_and_complete_rng_draws(self):
        ns=oracle(True);global_state=np.random.get_state()
        try:
            for size in (3,5,7):
                np.random.seed(144)
                expected=ns['patch_by_strides']((1,299,299,3),(size,size),.7)
                expected_state=np.random.get_state()
                rng=np.random.RandomState(144)
                actual=patch_mask((1,299,299,3),(size,size),.7,rng)
                np.testing.assert_array_equal(actual,expected)
                np.testing.assert_array_equal(rng.get_state()[1],expected_state[1])
                self.assertEqual(rng.get_state()[2:],expected_state[2:])
                self.assertTrue((actual[:,-(299%size):,:,:]==1).all() if 299%size else True)
            x,_=images()
            np.random.seed(11);rng=np.random.RandomState(11)
            for _ in range(20):
                torch.testing.assert_close(diversity(x,rng),ns['DI_keepresolution'](x),atol=0,rtol=0)
            np.testing.assert_array_equal(rng.get_state()[1],np.random.get_state()[1])
            self.assertEqual(rng.get_state()[2:],np.random.get_state()[2:])
        finally:np.random.set_state(global_state)

    def test_author_fft_aaf_guidance_momentum_feedback_final_and_cost(self):
        x,initial=images();global_state=np.random.get_state()
        try:
            for aaf in (False,True):
                ns=oracle(aaf);author_model=Toy();local=local_model(copy.deepcopy(author_model))
                attack=ns['AaFAttack' if aaf else 'FeatureFT'](author_model,'cpu',mask_num=4)
                np.random.seed(812)
                with patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self):
                    expected=attack.perturb(x,initial,torch.tensor([2]),torch.tensor([0]))
                end_state=np.random.get_state();events=[];cost={'feature':0,'input':0}
                def after(kind):cost[kind]+=1
                def observe(kind,values):
                    events.append((kind,{k:v.detach().clone() if isinstance(v,torch.Tensor) else v for k,v in values.items()}))
                rng=np.random.RandomState(812)
                try:
                    with ForwardCounter(local) as counter:
                        actual=finetune(local,x,initial,2,0,'aaf' if aaf else 'fft',16,rng,after,observe,mask_count=4)
                    self.assertEqual(counter.total,8+(15 if aaf else 10))
                    self.assertEqual(cost,{'feature':8,'input':15 if aaf else 10})
                    self.close_tensor(actual,expected)
                    with tempfile.TemporaryDirectory() as temp:
                        clean_pixels=np.rint(x[0].permute(1,2,0).numpy()*255).astype(np.uint8)
                        left,right=Path(temp)/'local.png',Path(temp)/'author.png'
                        save_adversarial(left,actual,clean_pixels,16)
                        save_adversarial(right,expected,clean_pixels,16)
                        np.testing.assert_array_equal(read_pixels(left),read_pixels(right))
                    author_final=ns['events'][-1][1];guide=events[0][1]
                    self.close_tensor(guide['original'],author_final['grad_sum_l3'])
                    self.close_tensor(guide['target'],author_final['grad_sum_mid_l3'])
                    self.close_tensor(guide['combined'],author_final['grad_sum_new_l3'])
                    for (_,local_step),(_,author_step) in zip(events[1:],ns['events'][:-1]):
                        self.close_tensor(local_step['momentum'],author_step['g'])
                        self.close_tensor(local_step['image'],author_step['x_adv_2' if aaf else 'x_adv_ft'])
                        if aaf and local_step['step']>5:self.close_tensor(local_step['eta_previous'],author_step['eta_prev'])
                    np.testing.assert_array_equal(rng.get_state()[1],end_state[1]);self.assertEqual(rng.get_state()[2:],end_state[2:])
                    self.assertTrue(all(p.grad is None for p in local.parameters()))
                finally:local.close()
        finally:np.random.set_state(global_state)

    def test_mask_normalization_std_smoothing_negative_controls(self):
        x,_=images();model=local_model()
        try:
            actual=masked_input(x,model.normalize,False,0,np.random.RandomState(5))
            mask=torch.tensor(np.random.RandomState(5).binomial(1,.7,size=x.shape))
            torch.testing.assert_close(actual,model.normalize(x)*mask,atol=0,rtol=0)
            self.assertFalse(torch.equal(actual,model.normalize(x*mask)))
            a=guidance(model,x,0,False,np.random.RandomState(7),count=3)
            # The author std is not an L2 normalization or correction0.
            self.assertAlmostEqual(float(a.std()),1,places=5)
            self.assertFalse(torch.allclose(a,a/a.norm()))
            self.assertNotEqual(float(a.std(correction=0)),float(a.std(correction=1)))
            channels=torch.arange(4*9*9,dtype=torch.float32).reshape(1,4,9,9).sin()
            smooth=F.conv2d(channels,gaussian(4,channels),padding=2,groups=4)
            self.assertFalse(torch.allclose(channels/channels.std(),smooth/smooth.std()))
            # L1 reweighting changes a multi-step accumulated direction when
            # successive gradient magnitudes differ; no final-sign claim from std.
            g1=torch.tensor([100.,-1.]);g2=torch.tensor([-2.,.5])
            self.assertFalse(torch.equal((g1+g2).sign(),(g1/g1.abs().sum()+g2/g2.abs().sum()).sign()))
            eta=torch.tensor([3.]);previous=torch.tensor([-2.]);weight=.8
            proper=(eta+weight*previous)/(weight+1)
            wrong=(eta.clamp(-1,1)+weight*previous.clamp(-1,1))/(weight+1)
            self.assertFalse(torch.equal(proper,wrong))
            clean=torch.tensor([.5]);initial=clean+.05;proposal=initial+.05
            self.assertFalse(torch.equal(clean+(proposal-clean).clamp(-.06,.06),
                                         initial+(proposal-initial).clamp(-.06,.06)))
        finally:model.close()

    def test_logit_controls_no_guidance_and_global_rng_passivity(self):
        import random
        x,initial=images();model=local_model()
        cpu=torch.get_rng_state().clone();numpy=np.random.get_state();python=random.getstate()
        try:
            for method,steps in (('logit-fft',10),('logit-aaf',15)):
                cost={'feature':0,'input':0}
                with ForwardCounter(model) as count:
                    result=finetune(model,x,initial,2,0,method,16,np.random.RandomState(17),lambda k:cost.__setitem__(k,cost[k]+1),mask_count=4)
                self.assertEqual(cost,{'feature':0,'input':steps});self.assertEqual(count.total,steps)
                self.assertLessEqual(float((result-x).abs().max()),16/255+1e-7)
            self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
            np.testing.assert_array_equal(numpy[1],np.random.get_state()[1]);self.assertEqual(numpy[2:],np.random.get_state()[2:])
            self.assertEqual(python,random.getstate())
        finally:model.close()

    def test_matched_logit_controls_discard_masks_align_di_parameters_and_end_rng(self):
        class DrawLog:
            def __init__(self,seed):self.rng=np.random.RandomState(seed);self.draws=[]
            def __getattr__(self,name):return getattr(self.rng,name)
            def randint(self,*a,**k):
                result=self.rng.randint(*a,**k);self.draws.append(('randint',np.asarray(result).tolist()));return result
            def rand(self,*a,**k):
                result=self.rng.rand(*a,**k);self.draws.append(('rand',np.asarray(result).tolist()));return result
        x,initial=images();model=local_model()
        try:
            for method in ('fft','aaf'):
                feature_rng,control_rng=DrawLog(77),DrawLog(77)
                feature_cost={'feature':0,'input':0};control_cost={'feature':0,'input':0};events=[]
                finetune(model,x,initial,2,0,method,16,feature_rng,
                         lambda k:feature_cost.__setitem__(k,feature_cost[k]+1),mask_count=4)
                finetune(model,x,initial,2,0,'logit-'+method,16,control_rng,
                         lambda k:control_cost.__setitem__(k,control_cost[k]+1),
                         lambda k,v:events.append((k,v)) if k=='matching_rng_advance' else None,mask_count=4)
                self.assertEqual(feature_rng.draws,control_rng.draws)
                np.testing.assert_array_equal(feature_rng.get_state()[1],control_rng.get_state()[1])
                self.assertEqual(feature_rng.get_state()[2:],control_rng.get_state()[2:])
                self.assertEqual(feature_cost['feature'],8);self.assertEqual(control_cost['feature'],0)
                self.assertEqual(feature_cost['input'],control_cost['input'])
                self.assertEqual(events[0][1]['mask_draws'],8)
        finally:model.close()

    def test_degenerate_guidance_explicit_failure_no_epsilon_substitution(self):
        model=local_model();x,_=images()
        try:
            with torch.no_grad():model.wrapped.model.fc.weight.zero_()
            with self.assertRaisesRegex(ValueError,'guidance std'):
                guidance(model,x,0,False,np.random.RandomState(2),count=2)
        finally:model.close()

    def test_base_prefix_checkpoint_callback_does_not_change_rdi_stream(self):
        x,_=images();model=local_model();saved={}
        try:
            short,_=optimize(model,x,2,'logit-rtm',2,16,2,63)
            full,_=optimize(model,x,2,'logit-rtm',3,16,2,63,checkpoints=(2,3),callback=lambda n,y:saved.__setitem__(n,y.clone()))
            plain,_=optimize(model,x,2,'logit-rtm',3,16,2,63)
            torch.testing.assert_close(short,saved[2],atol=0,rtol=0)
            torch.testing.assert_close(full,plain,atol=0,rtol=0)
        finally:model.close()

    def test_original299_manifest_labels_pixels_png_and_holdout(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);original=root/'x.png'
            raw=np.full((299,299,3),127,dtype=np.uint8);Image.fromarray(raw).save(original)
            record={'id':'x','label':0,'original_label':0,'split':'pilot','original_path':str(original),
                    'original_sha256':sha256(original),'canonical_path':str(original),'canonical_sha256':sha256(original)}
            source=root/'source.json';write_json(source,{'records':[record],'counts':{'pilot':1}})
            manifest=root/'manifest.json';write_json(manifest,{'records':[record],'counts':{'pilot':1},
                'input_profile':'original299-v1','preprocessing':ORIGINAL_PREPROCESSING,
                'source_manifest_path':str(source),'source_manifest_sha256':sha256(source)})
            args=types.SimpleNamespace(manifest=str(manifest),split='pilot',allow_holdout=False,limit=None,ids=None)
            self.assertEqual(records299(args),[record])
            csv=root/'labels.csv';csv.write_text('filename,label,targeted_label\nx.png,0,999\n',encoding='utf-8')
            self.assertEqual(target_labels(csv,[record]),{'x':999})
            csv.write_text('filename,label,targeted_label\nx.png,999,0\n',encoding='utf-8')
            self.assertEqual(target_labels(csv,[{**record,'label':999}]),{'x':0})
            with self.assertRaises(ValueError):target_labels(csv,[record])
            adv=root/'adv.png';clean=torch.from_numpy(raw).permute(2,0,1)[None].float()/255
            metrics=save_adversarial(adv,clean+16/255,raw,16)
            self.assertEqual(metrics['png_max_byte_difference'],16);assert_budget(adv,raw,16)
            self.assertEqual(pixel_sha(adv),hashlib.sha256(read_pixels(adv).tobytes()).hexdigest())
            self.assertNotEqual(pixel_sha(adv),sha256(adv))
            with self.assertRaises(ValueError):assert_budget(adv,raw,15)
            args.split='holdout'
            with self.assertRaises(ValueError):records299(args)
            Image.fromarray(raw[:224,:224]).save(adv)
            with self.assertRaises(ValueError):png299(adv)

    def test_untrained_resnet_layer3_bn_parameters_and_hook_ownership(self):
        from torchvision.models import resnet50
        network=resnet50(weights=None).eval()
        wrapped=NormalizeModel(network,[.485,.456,.406],[.229,.224,.225]).eval()
        for p in wrapped.parameters():p.requires_grad_(False)
        before={n:t.clone() for n,t in network.named_buffers()}
        seen=[];external=network.layer3.register_forward_hook(lambda _m,_i,o:seen.append(o.detach().clone()))
        previous=set(network.layer3._forward_hooks)
        model=Layer3Source(wrapped)
        try:
            x=torch.linspace(.05,.9,3*64*64).reshape(1,3,64,64).requires_grad_()
            with ForwardCounter(model) as count:
                logits=model(x);feature=model.value
                gradient=torch.autograd.grad(logits[:,0].sum(),feature,retain_graph=True)[0]
                inputgrad=torch.autograd.grad((feature*gradient.detach()).sum(),x)[0]
            torch.testing.assert_close(feature.detach(),seen[0],atol=0,rtol=0)
            self.assertTrue(torch.isfinite(inputgrad).all());self.assertEqual(count.total,1)
            self.assertTrue(all(p.grad is None for p in network.parameters()))
            for name,tensor in network.named_buffers():torch.testing.assert_close(tensor,before[name],atol=0,rtol=0)
        finally:model.close()
        self.assertEqual(set(network.layer3._forward_hooks),previous)
        external.remove();self.assertFalse(network.layer3._forward_hooks)

    def test_failed_attempt_after_backward_keeps_actual_cost_and_rejects_reuse(self):
        from utils.io import read_jsonl
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'x.png';Image.fromarray(np.full((299,299,3),127,dtype=np.uint8)).save(path)
            manifest=root/'manifest.json';write_json(manifest,{})
            labels=root/'labels.csv';labels.write_text('filename,label,targeted_label\nx.png,0,2\n',encoding='utf-8')
            record={'id':'x','label':0,'split':'pilot','canonical_path':str(path),'canonical_sha256':sha256(path)}
            args=types.SimpleNamespace(command='feature-finetune299',manifest=str(manifest),split='pilot',limit=None,ids=None,
                allow_holdout=False,labels_csv=str(labels),output=str(root/'results'/'failed'),device='cpu',
                seed=33,method='logit-fft',epsilon_bytes=16,save_float=False)
            wrapped=NormalizeModel(Toy(),[.485,.456,.406],[.229,.224,.225]).eval()
            for p in wrapped.parameters():p.requires_grad_(False)
            for method in ('logit-fft','fft'):
                args.method=method;args.output=str(root/'results'/method)
                with patch.object(runner,'ROOT',root),patch.object(runner,'numeric_settings',return_value={}),\
                     patch('utils.targeted_finetuning.records299',return_value=[record]),\
                     patch('utils.targeted_finetuning.load_model',return_value=(wrapped,{})),\
                     patch('utils.targeted_finetuning.initial_rows',return_value=({'x':{'png_path':str(path)}},None)),\
                     patch('utils.targeted_finetuning.guidance_scalars',wraps=guidance_scalars) as observed_summary,\
                     patch('baseline.feature_finetuning.gaussian',side_effect=ValueError('synthetic post-backward TI failure')):
                    with self.assertRaisesRegex(ValueError,'post-backward'):attack_command(args)
                    attempt=read_jsonl(Path(args.output)/'attempts.jsonl')[0]
                    self.assertEqual(attempt['forward_images'],62 if method=='fft' else 2)
                    self.assertEqual(attempt['input_gradient_images'],1)
                    self.assertEqual(attempt['feature_gradient_images'],60 if method=='fft' else 0)
                    self.assertEqual(attempt['guidance_summary_applicable'],method=='fft')
                    if method=='logit-fft':
                        self.assertEqual(attempt['guidance_summary'],{})
                        observed_summary.assert_not_called()
                    else:
                        observed_summary.assert_called_once()
                        tensors,actual_method=observed_summary.call_args.args
                        self.assertEqual(actual_method,'fft')
                        for name in ('original','target','combined'):
                            scalar=attempt['guidance_summary'][name];tensor=tensors[name].detach()
                            self.assertEqual(scalar['method'],'fft')
                            self.assertEqual(scalar['std_correction1'],float(tensor.std(correction=1)))
                            self.assertEqual(scalar['l2_norm'],float(torch.linalg.vector_norm(tensor.double())))
                            self.assertEqual(scalar['nonfinite_count'],int((~torch.isfinite(tensor)).sum()))
                            self.assertEqual(scalar['zero_count'],int((tensor==0).sum()))
                            self.assertEqual(scalar['numel'],tensor.numel())
                    self.assertEqual(read_json(Path(args.output)/'summary.json')['status'],'partial')
                    with self.assertRaises(FileExistsError):fresh_output(args.output)
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in wrapped.modules()))


if __name__=='__main__':
    unittest.main()
