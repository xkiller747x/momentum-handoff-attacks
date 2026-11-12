"""First-state/sign ablation algebra and source-only toy workflow contracts."""
import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from baseline.ftm import FeatureTuning,first_nonzero_update,optimize_ftm
from baseline.ftm_initialization import (FirstInitializationControl,sign_key,validate_initialization,
    initialization_profile,content_sha)
from baseline.targeted import ForwardCounter
from utils import runner,targeted_ftm
from utils.io import read_json,read_jsonl,sha256,write_json


class Small(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,3,1,stride=16,bias=False)
        self.pool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Linear(3,2,bias=False)
        with torch.no_grad():
            for i,p in enumerate(self.parameters()):p.copy_(torch.linspace(.03+i*.01,.3+i*.01,p.numel()).reshape_as(p))
        for p in self.parameters():p.requires_grad_(False)
        self.eval()
    def forward(self,x):return self.fc(self.pool(self.conv(x)).flatten(1))


def control(policy='sign-first-frozen',aux=20261028,ids=('a','b'),pair='pair-test'):
    return FirstInitializationControl(policy,aux,pair,ids,[('conv',nn.Conv2d(1,1,1)),('fc',nn.Linear(1,1))])


class FirstInitializationTests(unittest.TestCase):
    def test_structured_key_full_shape_repeated_crossed_maps_and_global_rng(self):
        key,seed=sign_key(20261028,'pair-test','conv','a')
        encoded=json.dumps(key,ensure_ascii=False,separators=(',',':')).encode('utf-8')
        self.assertEqual(seed,int.from_bytes(hashlib.sha256(encoded).digest()[:8],'big')&((1<<63)-1))
        seeds=[sign_key(*values)[1] for values in [(20261028,'pair-test','conv','a'),(20261029,'pair-test','conv','a'),(20261028,'other-pair','conv','a'),(20261028,'pair-test','other-layer','a'),(20261028,'pair-test','conv','b')]]
        self.assertEqual(len(set(seeds)),5)
        # Main seed/event step/support are deliberately absent from sign_key.
        first=control();second=control();cpu=torch.get_rng_state().clone();python=random.getstate();numpy=np.random.get_state()
        x=torch.zeros(2,2,3,4);x[0,0,0,0]=1.;x[1,0,0,0]=2.
        y=torch.zeros_like(x);y[0,1,2,3]=3.;y[1,1,2,3]=4.
        first(2,'conv',torch.zeros_like(x),x);second(9,'conv',torch.zeros_like(y),y)
        for a,b in zip(first.events,second.events):self.assertEqual(a['sign_map_sha256'],b['sign_map_sha256'])
        self.assertTrue(torch.equal(first._sign_map(x[0].shape,seed),first._sign_map(x[0].shape,seed)))
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()));self.assertEqual(python,random.getstate())
        actual=np.random.get_state();self.assertEqual(numpy[0],actual[0]);np.testing.assert_array_equal(numpy[1],actual[1]);self.assertEqual(numpy[2:],actual[2:])

    def test_asynchronous_true_zero_no_aux_draw_and_fc_identity_freeze(self):
        current=torch.zeros(2,3);helper=control()
        with patch.object(helper,'_sign_map',wraps=helper._sign_map) as draw:
            current=helper(1,'conv',current,torch.zeros_like(current));draw.assert_not_called()
            current=helper(2,'conv',current,torch.tensor([[0.,0.,0.],[1.,2.,3.]]));self.assertEqual(draw.call_count,1)
            accepted_second=current[1].clone()
            current=helper(3,'conv',current,torch.tensor([[2.,3.,4.],[float('inf'),0.,0.]]));self.assertEqual(draw.call_count,2)
            self.assertTrue(torch.equal(current[1],accepted_second))
            frozen=current.clone();current=helper(4,'conv',current,torch.full_like(current,float('nan')))
            self.assertTrue(torch.equal(current,frozen));self.assertEqual(draw.call_count,2)
            fc=torch.zeros(2,3);candidate=torch.tensor([[1.,0.,0.],[0.,2.,0.]])
            fc=helper(1,'fc',fc,candidate);self.assertTrue(torch.equal(fc,candidate))
            self.assertTrue(torch.equal(helper(2,'fc',fc,torch.zeros_like(fc)),candidate));self.assertEqual(draw.call_count,2)
        self.assertEqual([(r['image_id'],r['step']) for r in helper.events[:2]],[('b',2),('a',3)])
        self.assertTrue(all(r['aux_key'] is None and r['support_global_plus'] for r in helper.events if r['layer']=='fc'))

    def test_learned_direct_exact_old_first_nonzero_rule_and_unused_proposals(self):
        # torch.equal(float) ignores signed zero; use int32 bit comparisons for
        # both the delayed-zero path and forced-identity R acceptances.
        signed_old=torch.full((2,3),-0.);empty=torch.zeros(2,dtype=torch.bool)
        for gradient in (torch.full((2,3),-0.),torch.tensor([[-0.,-1.,0.],[0.,0.,-2.]])):
            expected,_=first_nonzero_update(signed_old,gradient,empty)
            for policy in ('learned-first-frozen','sign-first-frozen'):
                exact=control(policy,None if policy=='learned-first-frozen' else 20261028)
                with patch.object(exact,'_sign_map',side_effect=lambda shape,_seed:torch.ones(shape)):
                    actual=exact(1,'conv',signed_old,signed_old-gradient)
                self.assertTrue(torch.equal(actual.view(torch.int32),expected.view(torch.int32)))
        helper=control('learned-first-frozen',None);old=torch.zeros(2,3);reference=old.clone();mask=torch.zeros(2,dtype=torch.bool)
        for step,gradient in enumerate([torch.tensor([[0.,0.,0.],[1.,2.,3.]]),torch.tensor([[3.,0.,0.],[1.,1.,1.]]),torch.full((2,3),float('inf'))],1):
            proposal=old-gradient
            actual=helper(step,'conv',old,proposal)
            reference,mask=first_nonzero_update(reference,gradient,mask)
            self.assertEqual(content_sha(actual),content_sha(reference));old=actual
        self.assertTrue(all(r['rho']==1. and r['support_global_plus'] for r in helper.events))

    def test_first_audit_abs_support_normbits_rho_neff_and_nullable_norm(self):
        helper=control(ids=('a',));proposal=torch.tensor([[3.,4.,0.]])
        with patch.object(helper,'_sign_map',return_value=torch.tensor([1.,-1.,-1.])):
            actual=helper(1,'conv',torch.zeros_like(proposal),proposal)
        self.assertTrue(torch.equal(actual,torch.tensor([[3.,-4.,-0.]])))
        row=helper.events[0]
        self.assertEqual(row['before_abs_sha256'],row['after_abs_sha256']);self.assertEqual(row['before_support_sha256'],row['after_support_sha256'])
        self.assertEqual(row['before_fp32_norm']['fp32_bits'],row['after_fp32_norm']['fp32_bits'])
        self.assertAlmostEqual(row['rho'],-7/25);self.assertAlmostEqual(row['neff'],625/337)
        self.assertFalse(row['support_global_plus']);self.assertFalse(row['support_global_minus'])
        single=control(ids=('a',))
        with patch.object(single,'_sign_map',return_value=-torch.ones(3)):single(1,'conv',torch.zeros(1,3),torch.tensor([[0.,2.,0.]]))
        self.assertEqual(single.events[0]['rho'],-1.);self.assertEqual(single.events[0]['neff'],1.);self.assertTrue(single.events[0]['support_global_minus'])
        for value in (1e30,1e-30):
            with self.subTest(value=value):
                learned=control('learned-first-frozen',None,ids=('a',));candidate=torch.full((1,2),value)
                self.assertTrue(torch.equal(learned(1,'conv',torch.zeros_like(candidate),candidate),candidate))
                row=learned.events[0];self.assertEqual(row['before_fp32_norm']['fp32_bits'],row['after_fp32_norm']['fp32_bits'])
                if value==1e30:self.assertIsNone(row['before_fp32_norm']['value']);self.assertEqual(row['before_fp32_norm']['status'],'positive-infinity')

    def test_actual_optimizer_learned_equals_old_first_and_forced_plus_signs_cost_rng(self):
        base=Small();clean=torch.linspace(.1,.7,2*3*224*224).reshape(2,3,224,224);targets=torch.tensor([0,1]);results=[]
        for policy in ('old','learned-first-frozen','sign-first-frozen','sign-real'):
            model=copy.deepcopy(base);generator=torch.Generator().manual_seed(11)
            with FeatureTuning(model,generator,prob=1.) as context,ForwardCounter(model) as counter:
                context.cache(clean);actual_policy='sign-first-frozen' if policy=='sign-real' else policy
                helper=None if policy=='old' else FirstInitializationControl(actual_policy,20261028 if actual_policy=='sign-first-frozen' else None,'pair-test',['a','b'],context.layers)
                manager=patch.object(helper,'_sign_map',side_effect=lambda shape,_seed:torch.ones(shape)) if policy=='sign-first-frozen' else nullcontext()
                with manager:
                    actual,stats=optimize_ftm(context,clean,targets,3,16,.2,counter=counter,
                        update_mode='first-nonzero-only' if policy=='old' else 'full',state_transform=helper)
                results.append((actual.clone(),{k:content_sha(v) for k,v in context.outputs_tuning.items()},generator.get_state(),counter.total,stats['input_gradient_images']))
        for actual in results[1:3]:
            self.assertTrue(torch.equal(actual[0],results[0][0]));self.assertEqual(actual[1],results[0][1]);self.assertTrue(torch.equal(actual[2],results[0][2]));self.assertEqual(actual[3:],(8,6))
        self.assertTrue(torch.equal(results[3][2],results[0][2]));self.assertEqual(results[3][3:],(8,6))

    def test_validation_off_is_unrestricted_old_path_and_nonoff_closed_combinations(self):
        self.assertEqual(validate_initialization(argparse.Namespace()),('off',None))
        args=argparse.Namespace(ftm_initialization_control='learned-first-frozen',ftm_initialization_aux_seed=None,
            source='rn50',batch_size=2,ftm_input_profile='canonical224-v2',ftm_layer_profile='mapped299',ftm_update_mode='full',ftm_state_control='raw',ftm_efficacy_probe='off',ftm_beta=.01)
        self.assertEqual(validate_initialization(args),('learned-first-frozen',None))
        for key,value in [('source','rn50_v1'),('batch_size',20),('ftm_input_profile','original299-v1'),('ftm_state_control','unit-continuous'),('ftm_efficacy_probe','observe'),('ftm_update_mode','first-nonzero-only'),('ftm_beta',0.)]:
            changed=copy.copy(args);setattr(changed,key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):validate_initialization(changed)
        args.ftm_initialization_control='sign-first-frozen'
        with self.assertRaises(ValueError):validate_initialization(args)
        args.ftm_initialization_aux_seed=20261029;self.assertEqual(validate_initialization(args),('sign-first-frozen',20261029))
        base='ftm-author-config-mapped299-local224-batch2-update-full';self.assertEqual(initialization_profile(base,'off','full'),base)
        self.assertTrue(initialization_profile(base,'sign-first-frozen','full').endswith('-initialization-sign-first-frozen-update-full'))

    def test_driver_endstep_rng_every_step_without_selected_updates_and_default_off_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);records=[]
            for identity,label in [('a',0),('b',1)]:
                path=root/(identity+'.png');Image.new('RGB',(224,224),(51,102,153)).save(path)
                records.append({'id':identity,'label':label,'split':'pilot','canonical_path':str(path),'canonical_sha256':sha256(path)})
            manifest=root/'manifest.json';write_json(manifest,{'records':records});labels=root/'labels.csv';labels.write_text('filename,label,targeted_label\na.png,0,1\nb.png,1,0\n')
            base=argparse.Namespace(command='targeted-ftm',method='ftm',manifest=str(manifest),split='pilot',allow_holdout=False,ids=None,limit=None,
                labels_csv=str(labels),device='cpu',source='rn50',batch_size=2,ftm_layer_profile='mapped299',ftm_input_profile='canonical224-v2',
                ftm_update_mode='full',ftm_beta=.01,ftm_state_control='raw',ftm_efficacy_probe='off',steps=2,epsilon_bytes=16.,alpha_bytes=.2,seed=28,checkpoints='2',save_float=False,trace_updates=True)
            rows=[]
            # Negative probability is a test-only no-selection wrapper to test
            # end-step recording unconditionally, including the <=0 boundary.
            def no_selection(*a,**kw):kw['prob']=-1.;return FeatureTuning(*a,**kw)
            for policy,aux in [('off',None),('learned-first-frozen',None),('sign-first-frozen',20261028)]:
                args=copy.copy(base);args.output=str(root/'results'/policy);args.ftm_initialization_control=policy;args.ftm_initialization_aux_seed=aux
                with patch.object(runner,'ROOT',root),patch.object(targeted_ftm,'load_model',return_value=(Small(),{'name':'rn50'})),patch.object(targeted_ftm,'FeatureTuning',side_effect=no_selection):
                    result=targeted_ftm.attack_command(args)
                self.assertEqual((result['forward_images'],result['input_gradient_images']),(13,4))
                entry=read_jsonl(Path(args.output)/'progress.jsonl')[0];batch=read_json(entry['shared_batch_total_path']);rows.append(batch)
                if policy=='off':self.assertNotIn('initialization_end_step_main_rng',batch)
                else:
                    self.assertEqual([r['step'] for r in batch['initialization_end_step_main_rng']],[1,2])
                    self.assertEqual(batch['initialization_audit']['first_event_count'],0)
            self.assertEqual(rows[1]['initialization_end_step_main_rng'],rows[2]['initialization_end_step_main_rng'])
            self.assertEqual(rows[0]['final_generator_state_sha256'],rows[1]['final_generator_state_sha256'])
