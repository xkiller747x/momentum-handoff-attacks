"""Pinned author-hook and immediate S-gradient tests; toy CPU models only."""
import argparse
import ast
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn

from baseline.rapa import AuthorMask,optimize_rapa
from baseline.targeted import ForwardCounter,optimize
from utils import runner,targeted_rapa
from utils.io import read_jsonl,sha256,write_json
from utils.rapa_probe import SOURCE as RAPA_SOURCE,author_functions,semantic_probe

# The RaPA hook snapshot carries no upstream license, so public copies omit it.
NEEDS_RAPA=unittest.skipUnless(RAPA_SOURCE.exists(),'RaPA reference snapshot not distributed')


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.bn=nn.BatchNorm2d(3);self.pool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Linear(3,2)
        with torch.no_grad():
            self.bn.weight.fill_(1.);self.bn.bias.fill_(.1);self.fc.weight.copy_(torch.tensor([[.2,.1,.3],[.4,.6,.5]]));self.fc.bias.fill_(.2)
        for parameter in self.parameters():parameter.requires_grad_(False)
        self.eval()
    def forward(self,x):return self.fc(self.pool(self.bn(x)).flatten(1))


class RaPATests(unittest.TestCase):
    @NEEDS_RAPA
    def test_author_cpu_oracle_p0_p005_p1_and_leaf_measurements(self):
        namespace=author_functions();model=Toy()
        expected=[name for name,module in model.named_modules() if namespace['is_selected_layer'](name,module,{'drop_name_list':['all'],'type_list':['Linear','Normalization']})]
        self.assertEqual([name for name,_ in AuthorMask(model,torch.Generator(),.05).modules],expected)
        for probability in (0.,.05,1.):
            rows=semantic_probe('cpu',probability,28)
            self.assertEqual({r['kind'] for r in rows},{'Linear','BatchNorm2d','Combined'})
            for row in rows:
                self.assertTrue(row['forward_author_local']['exact_equal']);self.assertTrue(row['input_gradient_author_local']['exact_equal']);self.assertTrue(row['rng_state_equal'])
                if row['kind']!='Combined':
                    self.assertIn('input_gradient_vs_explicit_masked_math',row);self.assertIn('input_gradient_vs_original_jacobian',row)
            if probability in (.05,1.):
                self.assertGreater(sum(layer['weight_zero_elements'] for r in rows for layer in r['layers']),0)

    def test_endpoints_are_noop_and_each_forward_restores_bn_and_parameters(self):
        model=Toy();original=copy.deepcopy(model.state_dict());generator=torch.Generator().manual_seed(15);image=torch.ones(1,3,4,4,requires_grad=True)
        with AuthorMask(model,generator,.5) as context:
            value=context.forward(image);torch.autograd.grad(value.sum(),image)
            self.assertEqual(context.counts['module_mask_calls'],2);self.assertEqual(context.counts['module_restore_calls'],2)
            state=generator.get_state().clone();counts=dict(context.counts)
            with context.disabled(),torch.no_grad():model(image);model(image)
            self.assertTrue(torch.equal(state,generator.get_state()));self.assertEqual(context.counts,counts);self.assertTrue(context.verify())
            for record in context.bn_record().values():
                for item in record.values():self.assertEqual(item['before_sha256'],item['after_sha256'])
        for key,value in original.items():self.assertTrue(torch.equal(value,model.state_dict()[key]))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_exception_restores_pending_mask_and_removes_hooks(self):
        class Failing(nn.Linear):
            def forward(self,x):raise RuntimeError('synthetic masked-forward failure')
        model=Failing(3,2);before=copy.deepcopy(model.state_dict());scope=None
        with self.assertRaises(RuntimeError):
            with AuthorMask(model,torch.Generator().manual_seed(1),1.) as scope:model(torch.ones(1,3))
        self.assertEqual(scope.counts['forced_restore_calls'],1)
        for key,value in before.items():self.assertTrue(torch.equal(value,model.state_dict()[key]))
        self.assertFalse(model._forward_pre_hooks);self.assertFalse(model._forward_hooks)

    @NEEDS_RAPA
    def test_p0_bias_zero_draw_boundary_is_not_short_circuited(self):
        namespace=author_functions();a=nn.Linear(2,2);b=copy.deepcopy(a)
        with torch.no_grad():a.bias.fill_(2.);b.bias.fill_(2.)
        a.drop_prob=0.
        with patch('torch.rand_like',side_effect=lambda value:torch.zeros_like(value)):
            namespace['dropconnect_forward_pre_hook'](a,(torch.ones(1,2),))
        self.assertTrue(torch.equal(a.bias,torch.zeros_like(a.bias)))
        namespace['dropconnect_forward_hook'](a,None,None)
        captured=[]
        with AuthorMask(b,torch.Generator().manual_seed(7),0.) as scope:
            handle=b.register_forward_pre_hook(lambda m,_i:captured.append(m.bias.detach().clone()))
            try:
                with patch('torch.rand',side_effect=lambda shape,**kwargs:torch.zeros(shape,device=kwargs.get('device'),dtype=kwargs.get('dtype'))):b(torch.ones(1,2))
            finally:handle.remove()
        self.assertTrue(torch.equal(captured[0],torch.zeros_like(b.bias)));self.assertTrue(torch.equal(b.bias,torch.full_like(b.bias,2.)))

    def test_original_multigradient_sums_and_backpropagates_immediately(self):
        path=Path(__file__).parent/'reference'/'rapa_multigradient.py.txt'
        if not path.exists():self.skipTest('RaPA reference snapshot not distributed (no upstream license)')
        self.assertEqual(sha256(path),'58d840a50e63d3683bb71cb964b37b5f76bb1103415f03285cb80e26fa5abf5a')
        node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='MultiGetGradient');node.decorator_list=[]
        class Base:
            def __init__(self,loss_fn,input_transform,**_kwargs):self.loss_fn=loss_fn;self.input_transform=input_transform
        namespace={'torch':torch,'BaseGetGradient':Base};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        model=nn.Linear(3,2,bias=False);events=[];original_grad=torch.autograd.grad;calls=[0]
        def transform(x):calls[0]+=1;events.append('F');return x*calls[0]
        def gradient(*args,**kwargs):events.append('B');return original_grad(*args,**kwargs)
        operator=namespace['MultiGetGradient'](lambda logits,label:logits[:,1].sum(),transform,times=3)
        delta=torch.zeros(1,3,requires_grad=True)
        with patch('torch.autograd.grad',side_effect=gradient):result=operator(model,torch.ones(1,3),delta,None)
        torch.testing.assert_close(result,6*model.weight[1:2]);self.assertEqual(events,['F','B']*3)

    def test_local_s_counts_and_p0_fixed_seed_rtm_equivalence(self):
        image=torch.linspace(.1,.7,3*224*224).reshape(1,3,224,224);model=Toy();original=copy.deepcopy(model)
        with AuthorMask(model,torch.Generator().manual_seed(19),0.) as context,ForwardCounter(model) as counter:
            result,stats=optimize_rapa(context,image,1,3,1,16,2,torch.Generator().manual_seed(17),counter=counter)
            self.assertEqual(counter.total,3);self.assertEqual(stats['input_gradient_images'],3)
        reference,_=optimize(original,image,1,'logit-rtm',3,16,2,17)
        self.assertTrue(torch.equal(result,reference))
        with AuthorMask(model,torch.Generator().manual_seed(19),.05) as context,ForwardCounter(model) as counter:
            _result,stats=optimize_rapa(context,image,1,2,5,16,2,torch.Generator().manual_seed(17),counter=counter)
            self.assertEqual(counter.total,10);self.assertEqual(stats['input_gradient_images'],10)
            self.assertEqual(context.counts['module_mask_calls'],20);self.assertEqual(context.counts['weight_masks'],20);self.assertEqual(context.counts['bias_masks'],20)

    def test_synthetic_runner_cost_resume_and_dual_stream_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);image=root/'clean.png';Image.new('RGB',(4,4),(102,102,102)).save(image)
            record={'id':'a','label':0,'split':'pilot','canonical_path':str(image),'canonical_sha256':sha256(image)}
            manifest=root/'manifest.json';write_json(manifest,{'records':[record]})
            labels=root/'labels.csv';labels.write_text('filename,label,targeted_label\na.png,0,1\n',encoding='utf-8')
            args=argparse.Namespace(command='targeted-rapa',manifest=str(manifest),split='pilot',allow_holdout=False,limit=None,ids=None,device='cpu',
                labels_csv=str(labels),source='rn50',method='rapa',steps=2,samples=3,drop_probability=.05,epsilon_bytes=16.,alpha_bytes=2.,seed=28,
                checkpoints='2',save_float=True,output=str(root/'results'/'attack'))
            loader=lambda name,device:(Toy().to(device),{'name':name})
            with patch.object(runner,'ROOT',root),patch.object(targeted_rapa,'load_model',side_effect=loader):
                summary=targeted_rapa.attack_command(args)
                self.assertEqual(summary['forward_images'],9);self.assertEqual(summary['input_gradient_images'],6);self.assertEqual(summary['target_forward_images'],0)
                row=read_jsonl(Path(args.output)/'progress.jsonl')[0]
                self.assertTrue(row['parameters_restored']);self.assertEqual(row['mask_restore_counts']['module_mask_calls'],12)
                self.assertNotEqual(row['dual_stream_seeds']['rdi_cpu'],row['dual_stream_seeds']['mask_device'])
                self.assertEqual(targeted_rapa.attack_command(args)['count'],1)
                args.samples=5
                with self.assertRaises(ValueError):targeted_rapa.attack_command(args)
