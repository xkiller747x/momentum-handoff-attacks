"""Optional FTM audit/control tests; CPU tensors and synthetic saved runs only."""
import argparse
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn

from baseline.ftm import FeatureTuning,first_nonzero_update,optimize_ftm
from baseline.targeted import ForwardCounter
from utils import runner
from utils.ftm_update import UpdateTrace,compatible_configs,report_command
from utils.io import read_json,read_jsonl,sha256,write_json
from utils.targeted_ftm import atomic_progress


class AuditToy(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,2,1,stride=4,bias=False);self.pool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Linear(2,2,bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(.2);self.fc.weight.copy_(torch.tensor([[.1,.3],[.2,.4]]))
        for p in self.parameters():p.requires_grad_(False)
    def forward(self,x):return self.fc(self.pool(self.conv(x)).flatten(1))


class UpdateAuditTests(unittest.TestCase):
    def test_linear_first_raw_gradient_scale_and_later_fp32_rounding(self):
        class Flat(nn.Module):
            def __init__(self):
                super().__init__();self.fc=nn.Linear(3,2,bias=False)
                with torch.no_grad():self.fc.weight.fill_(.2)
                self.fc.weight.requires_grad_(False)
            def forward(self,x):return self.fc(x.flatten(1))
        model=Flat();x=torch.full((1,3,1,1),.3);seed=11
        replay=torch.Generator().manual_seed(seed)
        torch.rand(1,generator=replay);torch.randperm(1,generator=replay)
        ratio=torch.rand(1,2,generator=replay)*.75
        z=model(x).detach();expected=(1-ratio[0,1])*.01*z.norm()/1e-7
        with FeatureTuning(model,torch.Generator().manual_seed(seed),input_size=1,prob=1.) as context:
            context.cache(x);logits=context.forward(x)
            gradient=torch.autograd.grad(logits[0,1],context.outputs_tuning['fc'])[0]
            torch.testing.assert_close(gradient[0,1],expected)
            self.assertGreater(float(gradient.detach().norm()),1000)
            context.outputs_tuning['fc']=(-gradient).detach().requires_grad_(True)
            logits=context.forward(x);later=torch.autograd.grad(logits[0,1],context.outputs_tuning['fc'])[0]
            self.assertGreater(float(later.detach().norm()),0)
            self.assertTrue(torch.equal(context.outputs_tuning['fc'].detach()-later,context.outputs_tuning['fc'].detach()))

    def test_full_trace_on_off_bitwise_output_tuning_rng_and_cost(self):
        x=torch.linspace(.1,.6,2*3*224*224).reshape(2,3,224,224);results=[];states=[];tuning=[];counts=[]
        with tempfile.TemporaryDirectory() as directory:
            for trace in (False,True):
                model=AuditToy().eval();generator=torch.Generator().manual_seed(123)
                writer=UpdateTrace(Path(directory)/'trace.jsonl',['a','b'],{'conv':'Conv2d','fc':'Linear'},'full') if trace else None
                with FeatureTuning(model,generator,prob=1.,trace_updates=trace) as context,ForwardCounter(model) as count:
                    count.phase='cache';context.cache(x)
                    result,stats=optimize_ftm(context,x,torch.tensor([0,1]),3,16,.1,counter=count,update_observer=writer)
                    results.append(result);states.append(generator.get_state());tuning.append({k:v.detach().clone() for k,v in context.outputs_tuning.items()});counts.append(count.total)
                    if not trace:self.assertEqual(context.output_norms,{})
                if writer:
                    description=writer.commit();self.assertEqual(description['rows'],12)
                    self.assertEqual({r['layer_type'] for r in read_jsonl(description['path'])},{'Conv2d','Linear'})
            self.assertTrue(torch.equal(results[0],results[1]));self.assertTrue(torch.equal(states[0],states[1]));self.assertEqual(counts,[8,8])
            for key in tuning[0]:self.assertTrue(torch.equal(tuning[0][key],tuning[1][key]))

    def test_first_zero_gradient_does_not_initialize_sample(self):
        old=torch.zeros(2,2);mask=torch.zeros(2,dtype=torch.bool)
        first,mask=first_nonzero_update(old,torch.tensor([[0.,0.],[2.,3.]]),mask)
        self.assertEqual(mask.tolist(),[False,True]);self.assertTrue(torch.equal(first[0],old[0]))
        second,mask=first_nonzero_update(first,torch.tensor([[4.,5.],[9.,9.]]),mask)
        self.assertEqual(mask.tolist(),[True,True]);self.assertTrue(torch.equal(second[1],first[1]));self.assertTrue(torch.equal(second[0],torch.tensor([-4.,-5.])))
        third,_=first_nonzero_update(second,torch.ones_like(second),mask)
        self.assertTrue(torch.equal(third,second))

    def test_trace_separates_proposal_rounding_direction_and_zero_nulls(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trace.jsonl';writer=UpdateTrace(path,['a'],{'conv':'Conv2d'},'full')
            context=SimpleNamespace(output_norms={'conv':torch.tensor([10.])})
            old=torch.zeros(1,1,1,1);gradient=torch.full_like(old,1e8);after=old-gradient
            writer(1,'conv',old,gradient,after,context)
            old=after;gradient=torch.full_like(old,1e-8);after=old-gradient
            writer(2,'conv',old,gradient,after,context)
            gradient=torch.full_like(old,-2e8);after=old-gradient
            writer(3,'conv',old,gradient,after,context);writer.commit()
            rows=read_jsonl(path)
            self.assertIsNone(rows[0]['actual_relative_step']);self.assertIsNone(rows[0]['effective_v_relative_change_fixed_z'])
            self.assertGreater(rows[1]['theoretical_relative_step'],0);self.assertEqual(rows[1]['actual_relative_step'],0)
            self.assertEqual(rows[1]['changed_element_fraction'],0);self.assertEqual(rows[1]['unit_direction_l2_drift_from_first'],0)
            self.assertAlmostEqual(rows[2]['unit_direction_l2_drift_from_first'],2.)
            self.assertGreater(rows[2]['actual_relative_step'],0)

    def test_control_preserves_forward_backward_rng_and_first_update(self):
        x=torch.full((2,3,224,224),.3);runs=[]
        for mode in ('full','first-nonzero-only'):
            model=AuditToy().eval();generator=torch.Generator().manual_seed(28);traces=[]
            with FeatureTuning(model,generator,prob=1.) as context,ForwardCounter(model) as count:
                context.cache(x)
                def observe(step,name,old,gradient,new,_context):traces.append((step,name,old.detach().clone(),new.detach().clone(),gradient.detach().clone()))
                _result,stats=optimize_ftm(context,x,torch.tensor([0,1]),3,16,.1,counter=count,update_mode=mode,update_observer=observe)
                runs.append((count.total,stats['input_gradient_images'],generator.get_state(),traces))
        self.assertEqual(runs[0][:2],runs[1][:2]);self.assertTrue(torch.equal(runs[0][2],runs[1][2]))
        for a,b in zip([r for r in runs[0][3] if r[0]==1],[r for r in runs[1][3] if r[0]==1]):self.assertTrue(torch.equal(a[3],b[3]))
        for step,name,old,new,gradient in runs[1][3]:
            if step>1:self.assertTrue(torch.equal(old,new))

    def test_report_mode_compatibility_partial_and_real_float_png_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pairs=[{'batch_id':'p','image_ids':['a','b'],'rng_seed':7}]
            base={'kind':'targeted-attack','manifest_sha256':'manifest','labels_csv_sha256':'labels','target_labels':{'a':1,'b':1},
                'selected_ids':['a','b'],'source_identity':{'name':'toy'},'pairs':pairs,'pair_table_sha256':'pair','layer_table_sha256':'layer',
                'qualified_layer_names':['conv','fc'],'settings':{'mix_prob':.1},'checkpoint_steps':[1],'code_sha256':{'baseline/ftm.py':'same'},'numeric_settings':{'fp32':True},
                'arguments':{'source':'rn50','steps':1,'epsilon_bytes':16.,'alpha_bytes':2.,'seed':28,'batch_size':2,'ftm_layer_profile':'native224'}}
            roots=[];configs=[]
            for mode in ('full','first-nonzero-only'):
                destination=root/'results'/mode;destination.mkdir(parents=True);roots.append(destination)
                config=copy.deepcopy(base);config['update_mode']=mode;config['profile']='profile-'+mode
                write_json(destination/'config.json',config);configs.append(config)
                batch=destination/'batch.json';write_json(batch,{'final_generator_state_sha256':'same_rng','update_trace':None,
                    'forward_images':10,'input_gradient_images':2,'elapsed_seconds':.1,'peak_cuda_allocated_bytes':0})
                rows=[]
                for identity in ('a','b'):
                    png=destination/(identity+'.png');Image.new('RGB',(2,2),(51,51,51) if mode=='full' else (52,52,52)).save(png)
                    floating=destination/(identity+'.npz');np.savez_compressed(floating,image=np.full((1,3,2,2),.2 if mode=='full' else .21,dtype=np.float32))
                    entry={'png_path':str(png),'png_sha256':sha256(png),'float_path':str(floating),'float_sha256':sha256(floating)}
                    rows.append({'id':identity,'batch_id':'p','batch_rng_seed':7,'target_label':1,'layer_table_sha256':'layer',
                        'shared_batch_total_path':str(batch),'shared_batch_total_sha256':sha256(batch),'checkpoints':{'1':entry}})
                atomic_progress(destination/'progress.jsonl',rows)
            compatible_configs(*configs)
            changed=copy.deepcopy(configs[1]);changed['arguments']['seed']=29
            with self.assertRaises(ValueError):compatible_configs(configs[0],changed)
            args=argparse.Namespace(full_run=str(roots[0]),control_run=str(roots[1]),output=str(root/'results'/'report'))
            with patch.object(runner,'ROOT',root):report_command(args)
            report=read_json(Path(args.output)/'report.json');self.assertEqual(report['status'],'complete')
            self.assertFalse(report['image_comparisons'][0]['float_exact_equal']);self.assertGreater(report['image_comparisons'][0]['float_max_abs'],0)
            self.assertEqual(report['image_comparisons'][0]['png_max_abs_bytes'],1)
            atomic_progress(roots[1]/'progress.jsonl',[])
            args.output=str(root/'results'/'partial')
            with patch.object(runner,'ROOT',root):report_command(args)
            self.assertEqual(read_json(Path(args.output)/'report.json')['status'],'partial')
