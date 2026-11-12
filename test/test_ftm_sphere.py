"""Known unit-state controls, post-transform trace and completed-backward cost."""
import argparse
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

from baseline.ftm import FeatureTuning,optimize_ftm
from baseline.ftm_sphere import UnitStateControl,StateTransformError,validate_state_control,state_profile
from baseline.targeted import ForwardCounter
from utils.ftm_update import UpdateTrace,trace_summary
from utils import runner,targeted_ftm
from utils.io import read_jsonl,sha256,write_json


class Small(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,3,1,stride=16,bias=False)
        self.pool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Linear(3,2,bias=False)
        with torch.no_grad():
            for index,p in enumerate(self.parameters()):p.copy_(torch.linspace(.03+index*.01,.3+index*.01,p.numel()).reshape_as(p))
        for p in self.parameters():p.requires_grad_(False)
        self.eval()
    def forward(self,x):return self.fc(self.pool(self.conv(x)).flatten(1))


class SphereTests(unittest.TestCase):
    def test_fp32_projection_direct_division_new_tensor_and_delayed_initialization(self):
        old=torch.zeros(2,2,requires_grad=True);proposal=torch.tensor([[3.,4.],[0.,0.]],requires_grad=True)
        original=proposal.detach().clone();control=UnitStateControl('unit-continuous')
        actual=control(1,'conv',old,proposal)
        expected=original.clone();expected[0]=original[0]/original[0].norm()
        self.assertTrue(torch.equal(actual,expected));self.assertTrue(torch.equal(proposal.detach(),original))
        self.assertTrue(torch.equal(control.initialized['conv'],torch.tensor([True,False])))
        self.assertNotEqual(actual.data_ptr(),old.data_ptr());self.assertNotEqual(actual.data_ptr(),proposal.data_ptr())
        self.assertFalse(actual.requires_grad)
        second=control(2,'conv',actual,torch.tensor([[1.,2.],[0.,2.]]))
        self.assertTrue(torch.equal(second[1],torch.tensor([0.,1.])))
        self.assertTrue(control.initialized['conv'].all())

    def test_first_frozen_ignores_frozen_candidates_but_initializes_each_sample(self):
        control=UnitStateControl('unit-first-frozen');old=torch.zeros(2,2)
        first=control(1,'conv',old,torch.tensor([[3.,4.],[0.,0.]]))
        second=control(2,'conv',first,torch.tensor([[0.,0.],[0.,2.]]))
        self.assertTrue(torch.equal(second[0],first[0]));self.assertTrue(torch.equal(second[1],torch.tensor([0.,1.])))
        ignored=control(3,'conv',second,torch.full_like(second,float('nan')))
        self.assertTrue(torch.equal(ignored,second))

    def test_unit_domain_zero_underflow_nonfinite_and_mode_rejection(self):
        control=UnitStateControl('unit-continuous');unit=control(1,'fc',torch.zeros(1,2),torch.tensor([[3.,4.]]))
        with self.assertRaises(StateTransformError) as error:control(2,'fc',unit,torch.zeros_like(unit))
        self.assertIn('finite degenerate zero',error.exception.details['kind'])
        for value,kind in [(1e-30,'underflow'),(1e30,'nonfinite')]:
            with self.subTest(value=value),self.assertRaises(StateTransformError) as error:
                UnitStateControl('unit-continuous')(1,'fc',torch.zeros(1,2),torch.full((1,2),value))
            self.assertIn(kind,error.exception.details['kind'])
        self.assertEqual(validate_state_control('raw','first-nonzero-only'),'raw')
        for state in ('unit-first-frozen','unit-continuous'):
            with self.assertRaises(ValueError):validate_state_control(state,'first-nonzero-only')
        base='ftm-author-config-mapped299-local224-batch2-update-full'
        self.assertEqual(state_profile(base,'raw','full'),base)
        self.assertTrue(state_profile(base,'unit-continuous','full').endswith('-state-unit-continuous-update-full'))

    def test_none_default_raw_preserves_image_state_rng_forward_backward(self):
        clean=torch.linspace(.1,.7,2*3*224*224).reshape(2,3,224,224);targets=torch.tensor([0,1]);model=Small();results=[]
        for explicit in (False,True):
            local=copy.deepcopy(model);generator=torch.Generator().manual_seed(7)
            with FeatureTuning(local,generator,prob=1.) as context,ForwardCounter(local) as counter:
                context.cache(clean);completed=[]
                kwargs={'state_transform':None} if explicit else {}
                image,stats=optimize_ftm(context,clean,targets,3,16,.2,counter=counter,after_gradient=completed.append,**kwargs)
                results.append((image,{k:v.detach().clone() for k,v in context.outputs_tuning.items()},generator.get_state(),counter.total,stats['input_gradient_images']))
                self.assertEqual(completed,[1,2,3])
        self.assertTrue(torch.equal(results[0][0],results[1][0]));self.assertTrue(torch.equal(results[0][2],results[1][2]))
        for key in results[0][1]:self.assertTrue(torch.equal(results[0][1][key],results[1][1][key]))
        self.assertEqual(results[0][3:],(8,6));self.assertEqual(results[0][3:],results[1][3:])

    def test_real_postprojection_trace_and_trace_toggle_no_numeric_rng_cost_change(self):
        clean=torch.linspace(.1,.7,2*3*224*224).reshape(2,3,224,224);targets=torch.tensor([0,1]);model=Small()
        with tempfile.TemporaryDirectory() as temporary:
            for state in ('unit-first-frozen','unit-continuous'):
                results=[]
                for tracing in (False,True):
                    local=copy.deepcopy(model);generator=torch.Generator().manual_seed(11);writer=None
                    with FeatureTuning(local,generator,prob=1.,trace_updates=tracing) as context,ForwardCounter(local) as counter:
                        context.cache(clean)
                        if tracing:writer=UpdateTrace(Path(temporary)/(state+'.jsonl'),['a','b'],{'conv':'Conv2d','fc':'Linear'},'full',state_control=state)
                        image,stats=optimize_ftm(context,clean,targets,3,16,.2,counter=counter,state_transform=UnitStateControl(state),update_observer=writer)
                        results.append((image,generator.get_state(),counter.total,stats['input_gradient_images']))
                        self.assertTrue(all(torch.isfinite(v).all() for v in context.outputs_tuning.values()))
                    if writer:
                        metadata=writer.commit();rows=read_jsonl(metadata['path'])
                        for row in rows:
                            self.assertAlmostEqual(row['new_u_norm'],1.,places=5)
                            self.assertEqual(row['state_control'],state)
                        self.assertTrue(all(r['old_to_new_unit_direction_chord'] is None for r in rows if r['step']==1))
                        self.assertTrue(all(r['old_to_new_unit_direction_chord'] is not None for r in rows if r['step']>1))
                        if state=='unit-first-frozen':self.assertTrue(all(r['actual_delta_u_norm']==0 for r in rows if r['step']>1))
                self.assertTrue(torch.equal(results[0][0],results[1][0]));self.assertTrue(torch.equal(results[0][1],results[1][1]))
                self.assertEqual(results[0][2:],results[1][2:]);self.assertEqual(results[0][2:],(8,6))

    def test_trace_chord_reversal_and_legacy_missing_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer=UpdateTrace(Path(temporary)/'trace.jsonl',['a'],{'fc':'Linear'},'full',state_control='unit-continuous')
            context=SimpleNamespace(output_norms={'fc':torch.tensor([1.])})
            writer(1,'fc',torch.zeros(1,2),torch.tensor([[-1.,0.]]),torch.tensor([[1.,0.]]),context)
            writer(2,'fc',torch.tensor([[1.,0.]]),torch.tensor([[2.,0.]]),torch.tensor([[-1.,0.]]),context)
            rows=read_jsonl(writer.commit()['path'])
            self.assertEqual(rows[1]['old_to_new_unit_direction_chord'],2.)
            self.assertTrue(rows[1]['old_to_new_unit_dot_negative'])
            self.assertEqual(rows[1]['theoretical_relative_step'],2.)
            for row in rows:
                row.pop('old_to_new_unit_direction_chord');row.pop('old_to_new_unit_dot_negative')
            old=next(r for r in trace_summary(rows) if r['layer_type']=='Linear')
            self.assertEqual(old['old_to_new_unit_direction_chord']['missing_count'],1)
            self.assertIsNone(old['old_to_new_unit_direction_chord']['mean'])

    def test_handler_transform_failure_records_current_completed_backward_and_real_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);records=[]
            for identity,label in [('a',0),('b',1)]:
                path=root/(identity+'.png');Image.new('RGB',(224,224),(51,102,153)).save(path)
                records.append({'id':identity,'label':label,'split':'pilot','canonical_path':str(path),'canonical_sha256':sha256(path)})
            manifest=root/'manifest.json';write_json(manifest,{'records':records})
            labels=root/'labels.csv';labels.write_text('filename,label,targeted_label\na.png,0,1\nb.png,1,0\n')
            arguments=argparse.Namespace(command='targeted-ftm',method='ftm',manifest=str(manifest),split='pilot',allow_holdout=False,ids=None,limit=None,
                labels_csv=str(labels),output=str(root/'results/run'),device='cpu',source='rn50',batch_size=2,ftm_layer_profile='mapped299',
                ftm_input_profile='canonical224-v2',ftm_update_mode='full',ftm_beta=.01,ftm_state_control='unit-continuous',
                steps=2,epsilon_bytes=16.,alpha_bytes=.2,seed=28,checkpoints='2',save_float=False,trace_updates=False)
            def failure(*_a):raise StateTransformError('finite degenerate zero test',1,'conv',[0])
            def context(*a,**kw):kw['prob']=1.;return FeatureTuning(*a,**kw)
            with patch.object(runner,'ROOT',root),patch.object(targeted_ftm,'load_model',return_value=(Small(),{'name':'rn50'})),patch.object(targeted_ftm,'FeatureTuning',side_effect=context),patch.object(targeted_ftm,'UnitStateControl',return_value=failure):
                with self.assertRaises(StateTransformError):targeted_ftm.attack_command(arguments)
            attempt=read_jsonl(root/'results/run/attempts.jsonl')[0]
            self.assertEqual(attempt['status'],'failed');self.assertEqual(attempt['completed_input_gradient_images'],2)
            self.assertEqual(attempt['actual_forward_images'],6)
            self.assertIn('finite degenerate zero',attempt['state_transform_failure']['kind'])
