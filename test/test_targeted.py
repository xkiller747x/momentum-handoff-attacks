"""Independent mathematical and synthetic workflow checks, no pretrained models."""
import argparse
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from baseline.targeted import (FeatureTap,ForwardCounter,choose_reference,feature_distance,
    optimize,rdi,target_metrics,ti_kernel)
from utils import runner,targeted_runner
from utils.data import select_records
from utils.io import image_tensor,read_json,read_jsonl,save_adversarial,sha256,write_json
from utils.targeted_data import reference_files,target_labels


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.layer3=nn.Identity();self.layer4=nn.Identity()
    def forward(self,x):
        feature=self.layer4(self.layer3(x));value=feature.flatten(1).mean(1)
        return torch.stack((-value,value),1)


class TargetedMathTests(unittest.TestCase):
    def test_logit_ascent_feature_descent_and_projection(self):
        model=Toy();x=torch.full((1,3,3,3),.4)
        adv,stats=optimize(model,x,1,'logit-mi',1,16,.01,17)
        self.assertGreater(model(adv)[0,1],model(x)[0,1]);self.assertEqual(stats['input_gradient_images'],1)
        reference=torch.full_like(x,.1)
        for method in ('aa','gaa','paa-poly'):
            with FeatureTap(model,'layer3') as tap:
                adv,_=optimize(model,x,1,method,1,16,.001,17,tap,reference)
            self.assertLess(feature_distance(adv,reference,method),feature_distance(x,reference,method))
        adv,_=optimize(model,x,1,'logit-mi',100,2,20,17)
        self.assertLessEqual(float((adv-x).abs().max()),2/255+1e-7)
        self.assertTrue(((adv>=0)&(adv<=1)).all())

    def test_spatial_sensitivity_and_invariance(self):
        x=torch.tensor([[[[1.,2.,4.],[3.,6.,8.]],[[2.,7.,3.],[1.,5.,9.]]]])
        y=x.flatten(2).flip(2).reshape_as(x)
        self.assertGreater(feature_distance(x,y,'aa').item(),0)
        self.assertAlmostEqual(feature_distance(x,y,'gaa').item(),0,places=6)
        self.assertAlmostEqual(feature_distance(x,y,'paa-poly').item(),0,places=6)

    def test_gaa_author_std_scaling_zero_std_finite(self):
        x=torch.ones((1,2,2,3),requires_grad=True);y=torch.arange(12.).reshape_as(x)
        loss=feature_distance(x,y,'gaa')
        expected=((x.flatten(2).mean(2)-y.flatten(2).mean(2)).square()+
                  (x.flatten(2).std(2,correction=0)-y.flatten(2).std(2,correction=0)).square()).sum(1)*6/4
        torch.testing.assert_close(loss,expected)
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss.sum(),x)[0]).all())

    def test_exact_poly_matches_independent_spatial_kernel(self):
        x=torch.tensor([[[[1.,2.]],[[3.,5.]],[[2.,4.]]]],dtype=torch.float64)
        y=torch.tensor([[[[2.,1.]],[[1.,4.]],[[5.,2.]]]],dtype=torch.float64)
        a=x.flatten(2).transpose(1,2)[0];b=y.flatten(2).transpose(1,2)[0]
        # Direct pairwise spatial kernel sums, independent of channel Gram implementation.
        within_x=sum(torch.dot(u,v)**2 for u in a for v in a)
        within_y=sum(torch.dot(u,v)**2 for u in b for v in b)
        between=sum(torch.dot(u,v)**2 for u in a for v in b)
        expected=(within_x+within_y-2*between)/(4*2**2*3**2)
        torch.testing.assert_close(feature_distance(x,y,'paa-poly')[0],expected)

    def test_reference_rules_are_distance_specific_and_tie_stable(self):
        rows=[{'id':'z','distance':2.},{'id':'a','distance':2.},{'id':'b','distance':1.}]
        self.assertEqual(choose_reference(rows,'first')['id'],'a')
        self.assertEqual(choose_reference(rows,'farthest')['id'],'a')
        self.assertEqual(choose_reference(rows,'nearest')['id'],'b')
        self.assertEqual(choose_reference(rows,'index',2)['id'],'z')
        with self.assertRaises(ValueError):choose_reference(rows,'index',3)

    def test_rdi_exact_plan_order_and_ti_author_samples(self):
        image=torch.arange(224*224,dtype=torch.float32).reshape(1,1,224,224)
        first=torch.Generator().manual_seed(19);second=torch.Generator().manual_seed(19)
        amount=30;rnd=int(torch.rand(1,generator=second).item()*amount)+224
        pad=254-rnd;left=int(torch.rand(1,generator=second).item()*pad);top=int(torch.rand(1,generator=second).item()*pad)
        import torch.nn.functional as F
        expected=F.interpolate(F.pad(F.interpolate(image,size=(rnd,rnd),mode='nearest'),(left,pad-left,top,pad-top)),size=(224,224),mode='nearest')
        torch.testing.assert_close(rdi(image,first),expected,rtol=0,atol=0)
        self.assertTrue(torch.equal(first.get_state(),second.get_state()))
        kernel=ti_kernel(torch.zeros(1,3,3,3))[0,0].double()
        points=np.linspace(-3,3,5);one=np.exp(-points**2/2);expected=np.outer(one,one);expected/=expected.sum()
        np.testing.assert_allclose(kernel.numpy(),expected,rtol=1e-7,atol=1e-9)

    def test_zero_step_png_and_scoped_hooks_on_exception(self):
        model=Toy();x=torch.full((1,3,2,2),.4);called=[]
        result,stats=optimize(model,x,1,'logit-mi',0,16,2,17,callback=lambda s,z:called.append(s))
        self.assertTrue(torch.equal(result,x));self.assertEqual(stats['input_gradient_images'],0);self.assertEqual(called,[0])
        with tempfile.TemporaryDirectory() as directory:
            pixels=np.full((2,2,3),102,dtype=np.uint8)
            value=save_adversarial(Path(directory)/'image.png',result,pixels,16)
            self.assertEqual(value['png_max_byte_difference'],0);self.assertEqual(value['png_roundtrip_max_byte_error'],0)
        before=len(model.layer3._forward_hooks);outer=len(model._forward_pre_hooks)
        with self.assertRaises(RuntimeError):
            with ForwardCounter(model),FeatureTap(model,'layer3'):
                model(x);raise RuntimeError('synthetic failure')
        self.assertEqual(len(model.layer3._forward_hooks),before);self.assertEqual(len(model._forward_pre_hooks),outer)

    def test_target_success_and_conditional_denominators(self):
        # Wrong true class0, but prediction2 is not the specified target1.
        metrics=target_metrics(torch.tensor([[0.,1.,2.]]),1,0)
        self.assertFalse(metrics['targeted_success']);self.assertEqual(metrics['target_rank'],2)
        self.assertAlmostEqual(metrics['target_log_odds'],1-float(torch.logsumexp(torch.tensor([0.,2.]),0)),places=6)
        rows=[{'id':'a','clean':{'true_correct':True},'source_clean_correct':False,'adversarial':metrics},
              {'id':'b','clean':{'true_correct':True},'source_clean_correct':True,'adversarial':{'targeted_success':True,'true_correct':False}}]
        result=targeted_runner.denominator_summary(rows)
        self.assertEqual(result['all']['targeted_successes'],1);self.assertEqual(result['all']['denominator'],2)
        self.assertEqual(result['source-target-clean']['denominator_ids'],['b'])


class TargetedDataWorkflowTests(unittest.TestCase):
    def test_numeric_class_directory_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for label in range(12):
                path=root/str(label);path.mkdir();Image.new('RGB',(2,2)).save(path/'x.png')
            rows=reference_files(root,expected_per_class=1,classes=12)
            self.assertEqual([r[0] for r in rows],list(range(12)))
            self.assertEqual(rows[2][2].parent.name,'2');self.assertEqual(rows[10][2].parent.name,'10')

    def test_target_csv_stem_join_and_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'labels.csv';path.write_text('filename,label,targeted_label\nb.png,2,4\na.png,0,3\n',encoding='utf-8')
            self.assertEqual(target_labels(path,[{'id':'a','label':0},{'id':'b','label':2}]),{'a':3,'b':4})
            with self.assertRaises(ValueError):target_labels(path,[{'id':'b','label':1}])
            path.write_text('filename,label,targeted_label\na.png,0,3\na.jpg,0,3\n',encoding='utf-8')
            with self.assertRaises(ValueError):target_labels(path,[{'id':'a','label':0}])

    def test_synthetic_feature_workflow_cost_resume_and_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'clean.png';Image.new('RGB',(4,4),(102,102,102)).save(path)
            record={'id':'image','label':0,'split':'pilot','canonical_path':str(path),'canonical_sha256':sha256(path)}
            manifest=root/'manifest.json';write_json(manifest,{'records':[record,{**record,'id':'reserved','split':'holdout'}]})
            labels=root/'labels.csv';labels.write_text('filename,label,targeted_label\nimage.png,0,1\n',encoding='utf-8')
            refs=[]
            for i in range(10):
                ref=root/f'ref{i}.png';Image.new('RGB',(4,4),(i*20,i*20,i*20)).save(ref)
                refs.append({'id':f'r{i}','label':1,'canonical_path':str(ref),'canonical_sha256':sha256(ref)})
            gallery=root/'gallery.json';write_json(gallery,{'fixture':True})
            args=argparse.Namespace(command='targeted-attack',manifest=str(manifest),split='pilot',allow_holdout=False,limit=None,ids=None,device='cpu',
                labels_csv=str(labels),output=str(root/'results'/'attack'),source='rn50',method='aa',steps=2,epsilon_bytes=16.,alpha_bytes=.01,seed=7,
                layer='layer3',reference_manifest=str(gallery),reference_rule='farthest',reference_index=None,checkpoints='1,2',save_float=True)
            loader=lambda name,device:(Toy().to(device).eval(),{'name':name})
            with patch.object(runner,'ROOT',root),patch.object(targeted_runner,'load_model',side_effect=loader),patch.object(targeted_runner,'reference_gallery',return_value={1:refs}):
                summary=targeted_runner.attack_command(args)
                # clean1+refs10+gradient2+two endpoints per checkpoint×2=17.
                self.assertEqual(summary['forward_images'],17);self.assertEqual(summary['input_gradient_images'],2)
                saved=read_jsonl(root/'results'/'attack'/'progress.jsonl')[0]
                self.assertEqual(saved['actual_forward_hook']['by_phase']['reference'],10)
                self.assertEqual(saved['forward_pixels'],17*16);self.assertEqual(saved['input_gradient_pixels'],2*16)
                self.assertEqual(len(saved['checkpoints']),2)
                # Source clone remains102/255, independent of final candidate180/255.
                expected=torch.linalg.vector_norm(torch.full((1,3,4,4),102/255)-image_tensor(refs[-1]['canonical_path'])).item()
                self.assertAlmostEqual(saved['reference_candidates'][-1]['distance'],expected,places=5)
                self.assertEqual(targeted_runner.attack_command(args)['forward_images'],17)
                args.alpha_bytes=.02
                with self.assertRaises(ValueError):targeted_runner.attack_command(args)
            with self.assertRaises(ValueError):select_records(manifest,'holdout')

    def test_empty_partial_report_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'results'/'empty';source.mkdir(parents=True)
            write_json(source/'config.json',{'kind':'targeted-evaluate','selected_ids':['a','b'],'target_identities':{'dn121':{}},'arguments':{'checkpoint':None}})
            args=argparse.Namespace(runs=[str(source)],output=str(root/'results'/'report'))
            with patch.object(runner,'ROOT',root):targeted_runner.report_command(args)
            report=read_json(root/'results'/'report'/'report.json')
            self.assertEqual(report['runs'][0]['status'],'partial');self.assertFalse(report['runs'][0]['complete'])
            self.assertEqual(report['runs'][0]['models']['dn121']['all']['denominator'],0)
