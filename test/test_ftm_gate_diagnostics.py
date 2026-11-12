"""Bounded diagnostic hooks/private hashes and unchanged toy optimization."""
import argparse
import ast
import builtins
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from baseline.ftm import FeatureTuning,optimize_ftm
from baseline.ftm import qualification
from baseline.targeted import ForwardCounter
from utils.ftm_gate_diagnostics import GateDiagnostics,content,tensor_record,validate_mode
from utils.io import read_jsonl
from utils import targeted_ftm
from utils.models import NormalizeModel


class Reused(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,3,1,stride=16,bias=False)
        self.second=nn.Conv2d(3,3,1,bias=False);self.relu=nn.ReLU(inplace=True)
        self.pool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Linear(3,2,bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(.1);self.second.weight.copy_(torch.diag(torch.tensor([-1.,1.,2.])).reshape(3,3,1,1));self.fc.weight.fill_(.2)
        for p in self.parameters():p.requires_grad_(False)
        self.eval()
    def forward(self,x):return self.fc(self.pool(self.relu(self.second(self.relu(self.conv(x))))).flatten(1))


def clean():return torch.linspace(.1,.7,2*3*224*224).reshape(2,3,224,224)
def make(path,mode,ctx):return GateDiagnostics(path,mode,ctx,['a','b'],{'toy':True},expected_features=3,expected_relu_occurrences=2)


class GateDiagnosticTests(unittest.TestCase):
    def test_private_hash_negative_zero_nan_payload_header_and_layout(self):
        bits=torch.tensor([[0,-2147483648,2143289345],[1065353216,-1082130432,2143289346]],dtype=torch.int32)
        value=bits.view(torch.float32).transpose(0,1);before=value.contiguous().view(torch.int32).clone()
        pointer=value.data_ptr();version=value._version
        record=content(value)
        self.assertEqual(record['nonfinite_elements'],2)
        self.assertTrue(torch.equal(value.contiguous().view(torch.int32),before))
        self.assertEqual(value.data_ptr(),pointer);self.assertEqual(value._version,version)
        negative_tensor=torch.tensor([-0.],dtype=torch.float32);negative_bits=negative_tensor.view(torch.int32).clone()
        negative_pointer=negative_tensor.data_ptr();negative_version=negative_tensor._version
        positive=content(torch.tensor([0.],dtype=torch.float32));negative=content(negative_tensor)
        self.assertEqual(negative_tensor.data_ptr(),negative_pointer);self.assertEqual(negative_tensor._version,negative_version)
        self.assertTrue(torch.equal(negative_tensor.view(torch.int32),negative_bits))
        self.assertNotEqual(positive['bit_sha256'],negative['bit_sha256'])
        self.assertEqual(positive['zero_canonical_sha256'],negative['zero_canonical_sha256'])
        first=content(torch.tensor([2143289345],dtype=torch.int32).view(torch.float32));second=content(torch.tensor([2143289346],dtype=torch.int32).view(torch.float32))
        self.assertNotEqual(first['zero_canonical_sha256'],second['zero_canonical_sha256'])
        self.assertNotEqual(content(torch.ones(2,2))['bit_sha256'],content(torch.ones(4))['bit_sha256'])
        normal=torch.arange(48,dtype=torch.float32).reshape(2,3,2,4);channel=normal.contiguous(memory_format=torch.channels_last)
        snapshots=[(tensor.data_ptr(),tensor._version,tensor.contiguous().view(torch.int32).clone()) for tensor in (normal,channel)]
        a=tensor_record(normal,['a','b']);b=tensor_record(channel,['a','b'])
        self.assertEqual(a['images'],b['images']);self.assertNotEqual(a['layout']['stride'],b['layout']['stride'])
        self.assertTrue(b['layout']['channels_last']);self.assertTrue(torch.equal(normal,channel))
        for tensor,(pointer,version,before) in zip((normal,channel),snapshots):
            self.assertEqual(tensor.data_ptr(),pointer);self.assertEqual(tensor._version,version)
            self.assertTrue(torch.equal(tensor.contiguous().view(torch.int32),before))

    def test_repeated_relu_post_injection_layout_active_reset_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            model=Reused();image=clean()
            with FeatureTuning(model,torch.Generator().manual_seed(3),prob=0.) as ctx:
                ctx.cache(image)
                # Test-only returned-output layout change, registered after FTM
                # but before diagnostics, proves the observer sees post hooks.
                alter=model.conv.register_forward_hook(lambda _m,_i,o:o.transpose(-1,-2))
                gate=make(Path(temporary)/'observe','observe',ctx)
                try:
                    with ctx.disabled(),torch.no_grad():model(image)
                    self.assertIsNone(gate.current)
                    gate.before_step(1,image,torch.zeros_like(image),ctx)
                    with torch.no_grad():logits=ctx.forward(image)
                    gate.after_forward(1,logits,ctx)
                    gate.after_step(1,image,0,ctx)
                    # Endpoints do not change captured occurrence counts/rows.
                    with ctx.disabled(),torch.no_grad():model(image)
                    gate.before_step(2,image,torch.zeros_like(image),ctx)
                    with torch.no_grad():logits=ctx.forward(image)
                    gate.after_forward(2,logits,ctx);gate.after_step(2,image,0,ctx)
                    metadata=gate.close('complete');rows=read_jsonl(metadata['steps_path'])
                    for row in rows:
                        self.assertEqual([r['occurrence'] for r in row['relu_masks']],[1,2])
                        self.assertGreater(row['relu_masks'][0]['images'][0]['positive_count'],row['relu_masks'][1]['images'][0]['positive_count'])
                        self.assertFalse(row['post_ftm_layouts'][0]['layout']['contiguous'])
                    self.assertEqual(len(rows),2)
                finally:gate.close();alter.remove()
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_reference_observe_common_chain_logits_rng_u_endpoints_cost_match_off(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Reused();image=clean();targets=torch.tensor([0,1])
            for update_mode in ('full','first-nonzero-only'):
                results=[];common=[]
                for mode in ('off','reference','observe'):
                    model=copy.deepcopy(base);generator=torch.Generator().manual_seed(7);completed=[];gate=None
                    with FeatureTuning(model,generator,prob=1.) as ctx,ForwardCounter(model) as counter:
                        ctx.cache(image)
                        if mode!='off':gate=make(Path(temporary)/(update_mode+'-'+mode),mode,ctx)
                        global_cpu_before=torch.get_rng_state().clone()
                        endpoints=[]
                        def endpoint(_step,value):
                            with ctx.disabled(),torch.no_grad():endpoints.append(model(value).clone())
                        result,stats=optimize_ftm(ctx,image,targets,3,16,.2,[1,3],endpoint,counter,
                            update_mode=update_mode,after_gradient=completed.append,diagnostic_observer=gate)
                        results.append((result.clone(),[content(v) for v in endpoints],{k:content(v) for k,v in ctx.outputs_tuning.items()},generator.get_state(),counter.total,stats['input_gradient_images']))
                        self.assertEqual(completed,[1,2,3])
                        if gate:
                            metadata=gate.close('complete');rows=read_jsonl(metadata['steps_path'])
                            with np.load(metadata['logits_path'],allow_pickle=False) as archive:
                                self.assertEqual(archive['logits'].shape,(3,2,2));self.assertEqual(archive['logits'].dtype,np.float32);self.assertEqual(archive['steps'].tolist(),[1,2,3])
                            for row in rows:
                                for name in ('before_image','previous_momentum','post_rdi_input','logits','raw_input_gradient','ti_gradient','l1_norm','momentum','momentum_sign','after_image'):self.assertIn(name,row['signals'])
                                self.assertIsNone(row['cuda_global_rng_unchanged'])
                                if mode=='reference':self.assertIsNone(row['relu_masks'])
                                row.pop('relu_masks')
                            common.append(rows)
                        self.assertTrue(torch.equal(global_cpu_before,torch.get_rng_state()))
                    self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
                for result in results[1:]:
                    self.assertTrue(torch.equal(result[0],results[0][0]));self.assertEqual(result[1:3],results[0][1:3]);self.assertTrue(torch.equal(result[3],results[0][3]));self.assertEqual(result[4:],(12,6))
                self.assertEqual(common[0],common[1])

    def test_post_backward_failure_partial_npz_hooks_and_completed_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            model=Reused();image=clean();completed=[]
            with FeatureTuning(model,torch.Generator().manual_seed(4),prob=1.) as ctx,ForwardCounter(model) as count:
                ctx.cache(image)
                baseline_hooks={name:({(key,id(hook)) for key,hook in module._forward_hooks.items()},
                    {(key,id(hook)) for key,hook in module._forward_pre_hooks.items()}) for name,module in model.named_modules()}
                gate=make(Path(temporary)/'partial','observe',ctx);original=gate.signal
                def fail(name,value):
                    original(name,value)
                    if name=='l1_norm':raise RuntimeError('synthetic post-backward diagnostic failure')
                with patch.object(gate,'signal',side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError,'post-backward'):
                        optimize_ftm(ctx,image,torch.tensor([0,1]),2,16,.2,counter=count,after_gradient=completed.append,diagnostic_observer=gate)
                metadata=gate.close('partial');self.assertEqual(completed,[1]);self.assertEqual(count.total,4)
                self.assertEqual(metadata['rows'],0);self.assertEqual(metadata['logit_steps'],[1])
                with np.load(metadata['logits_path'],allow_pickle=False) as archive:self.assertEqual(archive['logits'].shape,(1,2,2))
                restored={name:({(key,id(hook)) for key,hook in module._forward_hooks.items()},
                    {(key,id(hook)) for key,hook in module._forward_pre_hooks.items()}) for name,module in model.named_modules()}
                self.assertEqual(restored,baseline_hooks)
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_coverage_failure_is_explicit_and_closes_hooks(self):
        with tempfile.TemporaryDirectory() as temporary:
            model=Reused();image=clean()
            with FeatureTuning(model,torch.Generator().manual_seed(1),prob=0.) as ctx:
                ctx.cache(image);gate=GateDiagnostics(Path(temporary)/'bad','observe',ctx,['a','b'],{},expected_features=3,expected_relu_occurrences=49)
                gate.before_step(1,image,torch.zeros_like(image),ctx)
                with torch.no_grad():logits=ctx.forward(image)
                with self.assertRaisesRegex(ValueError,'ReLU occurrence'):gate.after_forward(1,logits,ctx)
                gate.close('partial')
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_bounded_domain_and_default_off_does_not_import_helper(self):
        args=argparse.Namespace(ftm_gate_diagnostics='reference',source='rn50',batch_size=2,ftm_input_profile='canonical224-v2',ftm_layer_profile='mapped299',ftm_update_mode='full',ftm_beta=.01,deterministic=True)
        self.assertEqual(validate_mode(args),'reference')
        for key,value in [('source','rn50_v1'),('batch_size',20),('ftm_input_profile','original299-v1'),('ftm_update_mode','bad'),('ftm_state_control','unit-continuous'),('ftm_efficacy_probe','observe'),('ftm_initialization_control','learned-first-frozen'),('deterministic',False)]:
            changed=copy.copy(args);setattr(changed,key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):validate_mode(changed)
        tree=ast.parse(Path(targeted_ftm.__file__).read_text(encoding='utf-8'))
        self.assertFalse(any(isinstance(node,ast.ImportFrom) and node.module and node.module.endswith('ftm_gate_diagnostics') for node in tree.body))
        original=builtins.__import__
        def reject_import(name,*a,**kw):
            if name.endswith('ftm_gate_diagnostics'):raise AssertionError('Off imported gate helper')
            return original(name,*a,**kw)
        class StopBeforeModels(Exception):pass
        off=argparse.Namespace(steps=0,epsilon_bytes=16.,alpha_bytes=2.,ftm_update_mode='full')
        with patch('builtins.__import__',side_effect=reject_import),patch.object(targeted_ftm.runner,'_records',side_effect=StopBeforeModels):
            with self.assertRaises(StopBeforeModels):targeted_ftm.attack_command(off)

    def test_untrained_cpu_rn50_mapped299_actual_42_features_49_relu_occurrences(self):
        from torchvision.models import resnet50
        with torch.random.fork_rng(devices=[]),tempfile.TemporaryDirectory() as temporary:
            model=NormalizeModel(resnet50(weights=None),[.485,.456,.406],[.229,.224,.225]).eval()
            for parameter in model.parameters():parameter.requires_grad_(False)
            # Eligibility comes from an actual299 qualification, independent of
            # the smaller64 diagnostic image used solely to reduce CPU cost.
            entries=qualification(model,'cpu',299)
            names=[row['name'] for row in entries if row['eligible']]
            self.assertEqual(len(names),42)
            self.assertEqual(sum(isinstance(module,nn.ReLU) for module in model.modules()),17)
            value=torch.full((2,3,64,64),.3);generator=torch.Generator().manual_seed(1)
            with FeatureTuning(model,generator,input_size=299,eligible_names=names,prob=0.) as ctx:
                ctx.cache(value)
                self.assertEqual(set(ctx.outputs_tuning),set(names))
                before=torch.get_rng_state().clone()
                gate=GateDiagnostics(Path(temporary)/'rn50','observe',ctx,['a','b'],{'untrained_cpu_coverage':True})
                try:
                    with ctx.disabled(),torch.no_grad():model(value)
                    self.assertIsNone(gate.current);self.assertEqual(gate.rows,0)
                    gate.before_step(1,value,torch.zeros_like(value),ctx)
                    with torch.no_grad():logits=ctx.forward(value)
                    gate.after_forward(1,logits,ctx);gate.after_step(1,value,0,ctx)
                    metadata=gate.close('complete');row=read_jsonl(metadata['steps_path'])[0]
                    self.assertEqual(len(row['post_ftm_layouts']),42)
                    self.assertEqual(len({entry['module'] for entry in row['post_ftm_layouts']}),42)
                    self.assertEqual(len(row['relu_masks']),49)
                    self.assertEqual(len({entry['module'] for entry in row['relu_masks']}),17)
                    self.assertTrue(any(entry['occurrence']==3 for entry in row['relu_masks']))
                    with ctx.disabled(),torch.no_grad():model(value)
                    self.assertEqual(gate.rows,1)
                    self.assertTrue(torch.equal(before,torch.get_rng_state()))
                finally:gate.close('partial')
            self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()))
