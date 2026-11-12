"""Four-branch replay and passivity on tiny frozen CPU models, no real weights."""
import argparse
import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from baseline.ftm import FeatureTuning,optimize_ftm
from baseline.ftm_sphere import UnitStateControl
from baseline.targeted import ForwardCounter
from utils.ftm_efficacy import ConvEfficacyProbe,matched_random_conv,terminal_u_digest,validate_mode
from utils.io import image_tensor,read_jsonl,sha256


class Small(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,3,1,stride=16,bias=False)
        self.pool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Linear(3,2,bias=False)
        with torch.no_grad():
            for index,p in enumerate(self.parameters()):p.copy_(torch.linspace(.03+index*.01,.3+index*.01,p.numel()).reshape_as(p))
        for p in self.parameters():p.requires_grad_(False)
        self.eval()
    def forward(self,x):return self.fc(self.pool(self.conv(x)).flatten(1))


def image():return torch.linspace(.1,.7,2*3*224*224).reshape(2,3,224,224)
def probe(path,context,counter,steps=2):return ConvEfficacyProbe(path,context,['a','b'],[0,1],[0,1],30,'pair-test',steps,counter,{'fixture':True})


class EfficacyTests(unittest.TestCase):
    def test_four_branches_old_clone_init_no_selected_replay_state_and_global_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            clean=image();generator=torch.Generator().manual_seed(9);model=Small()
            with FeatureTuning(model,generator,prob=1.) as context,ForwardCounter(model) as counter:
                context.cache(clean);observer=probe(Path(temporary)/'rows.jsonl',context,counter)
                old=context.outputs_tuning['conv'];after=torch.ones_like(old).requires_grad_(True)
                context.outputs_tuning['conv']=after
                observer.capture(1,'conv',old,None,after,context)
                with torch.no_grad():old.fill_(99)
                self.assertEqual(int(torch.count_nonzero(observer.old['conv'])),0)
                mapping=context.outputs_tuning;references=dict(mapping);mixing=context.mixing_triggered;norms=context.output_norms
                state=generator.get_state().clone();cpu=torch.get_rng_state().clone();python=random.getstate();numpy=np.random.get_state()
                observer.observe(1,clean,context);observer.observe(2,clean,context)
                self.assertIs(context.generator,generator);self.assertTrue(torch.equal(state,generator.get_state()))
                self.assertIs(context.outputs_tuning,mapping);self.assertIs(context.mixing_triggered,mixing);self.assertIs(context.output_norms,norms)
                for name,value in references.items():self.assertIs(context.outputs_tuning[name],value)
                self.assertTrue(torch.equal(cpu,torch.get_rng_state()));self.assertEqual(python,random.getstate())
                updated=np.random.get_state();self.assertEqual(numpy[0],updated[0]);np.testing.assert_array_equal(numpy[1],updated[1]);self.assertEqual(numpy[2:],updated[2:])
                rows=read_jsonl(observer.metadata()['path']);self.assertEqual(len(rows),4)
                self.assertTrue(all(row['any_new_conv_initialization'] for row in rows[:2]))
                self.assertTrue(all(not row['any_new_conv_initialization'] and not row['conv_updated_layers'] for row in rows[2:]))
                self.assertTrue(all(row['previous_current_whole_logits_equal'] for row in rows[2:]))
                for row in rows:
                    self.assertEqual(len(set(row['aux_end_sha256'].values())),1)
                    self.assertEqual(row['formed_rdi_input_sha256_before'],row['formed_rdi_input_sha256_after'])
                    self.assertTrue(row['main_rng_unchanged'])
                self.assertEqual(counter.total,18)
                for branch in ('current','previous','zeroConv','randomConv'):self.assertEqual(counter.counts['efficacy_'+branch],4)

    def test_error_restores_references_rng_and_retains_partial_rows_actual_forwards(self):
        with tempfile.TemporaryDirectory() as temporary:
            clean=image();model=Small();generator=torch.Generator().manual_seed(4)
            with FeatureTuning(model,generator,prob=1.) as context,ForwardCounter(model) as counter:
                context.cache(clean);observer=probe(Path(temporary)/'rows.jsonl',context,counter)
                observer.observe(1,clean,context)
                mapping=context.outputs_tuning;values=dict(mapping);mixing=context.mixing_triggered;norms=context.output_norms;phase=counter.phase;state=generator.get_state().clone()
                original=context.forward
                def fail(value):
                    if counter.phase=='efficacy_previous':raise RuntimeError('synthetic diagnostic failure')
                    return original(value)
                with patch.object(context,'forward',side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError,'diagnostic failure'):observer.observe(2,clean,context)
                self.assertIs(context.generator,generator);self.assertTrue(torch.equal(state,generator.get_state()))
                self.assertIs(context.outputs_tuning,mapping);self.assertIs(context.mixing_triggered,mixing);self.assertIs(context.output_norms,norms);self.assertEqual(counter.phase,phase)
                for name,value in values.items():self.assertIs(context.outputs_tuning[name],value)
                metadata=observer.metadata();self.assertEqual(metadata['rows'],2);self.assertEqual(metadata['last_complete_step'],1)
                self.assertEqual(len(read_jsonl(metadata['path'])),2);self.assertEqual(counter.total,12)

    def test_whole_layer_random_norm_zero_and_explicit_invalid_domain(self):
        current=torch.arange(16,dtype=torch.float32).reshape(2,2,2,2);current[0]=0
        matched,metadata=matched_random_conv(current,torch.Generator().manual_seed(42))
        self.assertEqual(int(torch.count_nonzero(matched[0])),0)
        torch.testing.assert_close(matched.flatten(1).norm(dim=1),current.flatten(1).norm(dim=1),rtol=1e-6,atol=0)
        self.assertEqual(metadata['axis'],'whole-layer reshape[B,-1], per sample')
        with patch('torch.randn',return_value=torch.zeros(1,2)):
            with self.assertRaisesRegex(ValueError,'no resampling'):matched_random_conv(torch.ones(1,2),torch.Generator())
        with self.assertRaisesRegex(ValueError,'underflow'):matched_random_conv(torch.full((1,2),1e-30),torch.Generator())

    def test_off_reference_observe_exact_endpoints_u_rng_and_cost_for_three_states(self):
        clean=image();targets=torch.tensor([0,1]);base=Small()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for state in ('raw','unit-first-frozen','unit-continuous'):
                results=[]
                for mode in ('off','reference','observe'):
                    model=copy.deepcopy(base);parameters={key:value.clone() for key,value in model.state_dict().items()}
                    generator=torch.Generator().manual_seed(17);observer=None;png_paths=[];completed=[]
                    with FeatureTuning(model,generator,prob=1.) as context,ForwardCounter(model) as counter:
                        counter.phase='clean'
                        with context.disabled(),torch.no_grad():model(clean)
                        counter.phase='cache';context.cache(clean)
                        if mode=='observe':observer=probe(root/(state+'.jsonl'),context,counter)
                        def endpoint(_step,value):
                            with context.disabled(),torch.no_grad():
                                counter.phase='checkpoint_float';model(value)
                                for index in range(2):
                                    path=root/f'{state}-{mode}-{index}.png';pixels=value[index].detach().mul(255).round().byte().permute(1,2,0).numpy()
                                    Image.fromarray(pixels).save(path);png_paths.append(path)
                                counter.phase='checkpoint_png';model(torch.cat([image_tensor(path) for path in png_paths]))
                        actual,stats=optimize_ftm(context,clean,targets,2,16,.2,[2],endpoint,counter,
                            observer.observe if observer else None,update_observer=observer.capture if observer else None,
                            state_transform=UnitStateControl(state) if state!='raw' else None,after_gradient=completed.append)
                        digest=terminal_u_digest(context) if mode!='off' else None
                        # Off digest is test-only evidence, not runtime work.
                        test_digest=terminal_u_digest(context)
                        results.append((actual.clone(),[sha256(path) for path in png_paths],test_digest,generator.get_state(),counter.total,stats['input_gradient_images']))
                        self.assertEqual(completed,[1,2])
                        if digest:self.assertEqual(digest,test_digest)
                    if observer:observer.close()
                    self.assertTrue(all(torch.equal(parameters[key],value) for key,value in model.state_dict().items()))
                for result in results[1:]:
                    self.assertTrue(torch.equal(result[0],results[0][0]));self.assertEqual(result[1:3],results[0][1:3]);self.assertTrue(torch.equal(result[3],results[0][3]));self.assertEqual(result[5],4)
                self.assertEqual([r[4] for r in results],[12,12,28])

    def test_nonoff_requires_trace_false_and_explicit_strict_settings(self):
        args=argparse.Namespace(ftm_efficacy_probe='observe',trace_updates=False,deterministic=True,
            source='rn50',batch_size=2,ftm_input_profile='canonical224-v2',ftm_layer_profile='mapped299',ftm_update_mode='full',ftm_beta=.01)
        self.assertEqual(validate_mode(args),'observe')
        args.deterministic=False
        with self.assertRaisesRegex(ValueError,'strict'):validate_mode(args)
        args.deterministic=True;args.trace_updates=True
        with self.assertRaisesRegex(ValueError,'trace'):validate_mode(args)
        self.assertEqual(validate_mode(argparse.Namespace(ftm_efficacy_probe='off')),'off')
