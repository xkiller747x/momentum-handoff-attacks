"""CPU-only checks for the known-principle control; root runs these tests."""
import unittest
from unittest import mock

import torch
from torch import nn
import torch.nn.functional as F

from baseline.attacks import attack
from baseline.ops import OPSViewPlan,BasicPlan,apply_ops
from baseline.ops_refresh import OPSRefreshContext
from baseline.norm_control import unit_l2,norm_description,view_gradient_mean,control_attack


class NormToy(nn.Module):
    def forward(self,x):
        weights=torch.linspace(.1,1.,x[0].numel()).reshape_as(x[0])
        value=(x.square()*weights).mean((1,2,3))+.2*x.mean((1,2,3))
        return torch.stack((value,-value),dim=1)


class ZeroNormToy(nn.Module):
    def forward(self,x):
        value=x.sum((1,2,3))*0
        return torch.stack((value,value),dim=1)


class NormControlTests(unittest.TestCase):
    def test_zero_tiny_nonzero_and_norm_participation(self):
        zero,norm=unit_l2(torch.zeros(1,3,2,2));self.assertEqual(norm,0);self.assertEqual(zero.sum().item(),0)
        tiny=torch.tensor([[[[1e-40,0.]]]],dtype=torch.float32)
        unit,norm=unit_l2(tiny);self.assertGreater(norm,0);torch.testing.assert_close(unit,torch.tensor([[[[1.,0.]]]]))
        summary=norm_description([1.]*26)
        self.assertEqual(summary['norm_participation_ratio'],26)
        self.assertEqual(summary['base_norm_share'],1/26)
        self.assertTrue(summary['base_is_tied_max'])
        self.assertEqual(norm_description([0.]*26)['norm_participation_ratio'],0)

    def test_independent_perview_reference_and_zero_denominator26(self):
        x=torch.linspace(.1,.9,3*11*11).reshape(1,3,11,11);label=torch.tensor([0]);model=NormToy()
        plans=[OPSViewPlan(None,())]+[OPSViewPlan(None,(BasicPlan(14+(i%7)),)) for i in range(25)]
        actual,summary,details=view_gradient_mean(model,x,label,plans)
        reference=torch.zeros_like(x)
        for plan in plans:
            leaf=x.clone().requires_grad_();loss=F.cross_entropy(model(apply_ops(leaf,plan)),label)
            g=torch.autograd.grad(loss,leaf)[0];n=torch.linalg.vector_norm(g.double())
            if n>0:reference+=(g.double()/n).float()
        torch.testing.assert_close(actual,reference/26,atol=0,rtol=0)
        self.assertEqual(len(details),26);self.assertEqual(summary['view_count'],26)
        zero,summary,_=view_gradient_mean(ZeroNormToy(),x,label,plans)
        self.assertEqual(summary['zero_view_count'],26);self.assertEqual(zero.abs().sum().item(),0)

    def test_identity_replays_old_fresh_math_and_control_hook_count(self):
        x=torch.linspace(.1,.9,3*11*11).reshape(1,3,11,11);label=torch.tensor([0])
        old,_=attack(NormToy(),x,label,method='ops-refresh',steps=10,epsilon_bytes=8,seed=123,view_batch=1)
        identity,stats=control_attack(NormToy(),x,label,seed=123,identity=True)
        torch.testing.assert_close(identity,old,atol=0,rtol=0)
        self.assertEqual(stats['forward_images'],260)
        model=NormToy();counts=[];handle=model.register_forward_pre_hook(lambda _,inputs:counts.append(inputs[0].shape[0]))
        controlled,stats=control_attack(model,x,label,seed=123);handle.remove()
        self.assertEqual(sum(counts),260);self.assertEqual(stats['noise_pool_generation_count'],10)
        self.assertEqual(len(stats['steps_norm_summary']),10);self.assertTrue(torch.isfinite(controlled).all())
        with self.assertRaises(ValueError):control_attack(model,x,label,view_batch=2)

    def test_sampler_global_rng_isolation(self):
        from utils.ops_profile import rng_snapshot
        with mock.patch.object(torch.cuda,'get_rng_state',side_effect=AssertionError('CPU-only')):
            before=rng_snapshot();context=OPSRefreshContext((1,3,11,11),8/255,5,5,2.,7)
            for _ in range(10):context.step_plans()
            self.assertEqual(before,rng_snapshot())

    def test_control_all_zero_outer_meanabs_and_order(self):
        x=torch.full((1,3,11,11),.5);label=torch.tensor([0])
        out,stats=control_attack(ZeroNormToy(),x,label,seed=12)
        torch.testing.assert_close(out,x,atol=0,rtol=0)
        self.assertEqual(stats['zero_grad_count'],10)
        self.assertEqual(stats['near_zero_grad_count'],10)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(all(step['zero_view_count']==26 for step in stats['steps_norm_summary']))
        # Unit-vector averaging cannot be replaced by normalizing the raw mean.
        a=torch.tensor([3.,0.]);b=torch.tensor([0.,1.])
        ua,_=unit_l2(a);ub,_=unit_l2(b);wrong,_=unit_l2((a+b)/2)
        self.assertFalse(torch.allclose((ua+ub)/2,wrong))

    def test_partial_population_CI_and_discordants(self):
        from utils.view_l2_report import population_difference
        ids=[str(i) for i in range(276)]
        pairs=[{'id':identity,'raw_success':bool(i%4 in (0,2)),'control_success':bool(i%4 in (1,2)),
                'raw_adversarial_margin':1.,'control_adversarial_margin':.5,'raw_adversarial_ce':2.,'control_adversarial_ce':3.,
                'raw_adversarial_correct':bool(i%4 in (1,3)),'control_adversarial_correct':bool(i%4 in (0,3))} for i,identity in enumerate(ids)]
        full=population_difference(pairs,ids,True)
        self.assertEqual(full['raw_only'],69);self.assertEqual(full['control_only'],69)
        self.assertEqual(full['exact_mcnemar_two_sided'],1)
        self.assertEqual(full['endpoints']['success']['ci']['image_count'],276)
        self.assertEqual(full['endpoints']['success']['ci']['bootstrap_seed'],20261110)
        self.assertTrue(all(e['ci'] is None for e in population_difference(pairs[:69],ids,False)['endpoints'].values()))

    def test_formal_clock_preserves_frozen_gate(self):
        from datetime import datetime,timezone
        from utils.view_l2_control import may_start_block
        config={'freeze':{'last_new_block_utc':'2026-10-27T04:30:11Z','last_estimated_block_end_utc':'2026-10-27T04:45:11Z'}}
        state={'blocks':{'1':{'status':'complete','total_attempt_wall_seconds':900}}}
        self.assertTrue(may_start_block(config,state,datetime(2026,10,27,4,29,tzinfo=timezone.utc),900)[0])
        self.assertFalse(may_start_block(config,state,datetime(2026,10,27,4,31,tzinfo=timezone.utc),900)[0])
        self.assertFalse(may_start_block(config,{'blocks':{'1':{'status':'complete','total_attempt_wall_seconds':1200}}},datetime(2026,10,27,4,29,tzinfo=timezone.utc),900)[0])
        self.assertFalse(may_start_block(config,state,datetime(2026,10,27,4,20,tzinfo=timezone.utc),2400)[0])
        self.assertEqual(1248+104+520+520+71760,74152)
