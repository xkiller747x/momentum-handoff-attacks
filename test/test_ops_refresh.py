"""Lifecycle-specific CPU toy checks; no pretrained models and no CUDA state calls."""
import random
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from baseline.attacks import attack, mean_view_gradient
from baseline.ops import OPSContext, RATIOS
from baseline.ops_refresh import OPSRefreshContext
from utils.ops_profile import gradient_comparison, rng_snapshot
from utils.ops_profile_report import paired_population


class LifecycleToy(nn.Module):
    def forward(self,x):
        weight=torch.linspace(.2,1.2,x[0].numel()).reshape_as(x[0])
        value=(x.square()*weight).mean((1,2,3))+.1*x.mean((1,2,3))
        return torch.stack((value,-value),dim=1)


class OPSRefreshTests(unittest.TestCase):
    def test_exact_first_pool_and_ten_continuous_pools_paired_plans(self):
        shape,seed=(1,3,11,11),773
        fixed=OPSContext(shape,8/255,5,5,2.,seed)
        refresh=OPSRefreshContext(shape,8/255,5,5,2.,seed)
        self.assertEqual(refresh.pool_generation_count,1)
        for a,b in zip(fixed.pool,refresh.pool):torch.testing.assert_close(a,b,atol=0,rtol=0)
        generator=torch.Generator(device='cpu').manual_seed(seed)
        first_pool=[v.clone() for v in fixed.pool]
        for step in range(10):
            old,new=fixed.step_plans(),refresh.step_plans()
            self.assertEqual([(p.pool_index,p.operators) for p in old],[(p.pool_index,p.operators) for p in new])
            self.assertEqual(fixed.rng.getstate(),refresh.rng.getstate())
            expected=[]
            for ratio in RATIOS:
                radius=2*(8/255)*ratio
                tensor=torch.empty((5,*shape[1:]),dtype=torch.float32).uniform_(-radius,radius,generator=generator)
                expected.extend(tensor[i:i+1] for i in range(5))
            for index,(actual,reference) in enumerate(zip(refresh.pool,expected)):
                torch.testing.assert_close(actual,reference,atol=0,rtol=0)
                self.assertEqual(actual.shape,shape);self.assertEqual(actual.dtype,torch.float32)
                self.assertLessEqual(actual.abs().max().item(),2*(8/255)*RATIOS[index//5]+1e-8)
            self.assertEqual(refresh.pool_generation_count,step+1)
            for a,b in zip(fixed.pool,first_pool):torch.testing.assert_close(a,b,atol=0,rtol=0)
        record=refresh.lifecycle_record()
        self.assertEqual(record['noise_pool_generation_count'],10)
        self.assertEqual(record['noise_pool_steps_used'],10)
        self.assertEqual(record['noise_tensors_generated'],300)
        self.assertEqual(len(record['noise_pool_sha256_by_pool']),10)
        self.assertEqual(len(set(record['noise_pool_sha256_by_pool'])),10)
        self.assertEqual(refresh.pool_generation_count,10) # Recording creates no eleventh pool.

    def test_global_rng_isolation_cpu_never_initializes_cuda(self):
        with mock.patch.object(torch.cuda,'get_rng_state',side_effect=AssertionError('CPU test must not inspect CUDA')):
            before=rng_snapshot()
            context=OPSRefreshContext((1,3,11,11),8/255,5,5,2.,91)
            for _ in range(10):context.step_plans()
            self.assertEqual(before,rng_snapshot())

    def test_first_raw_gradient_microbatch_and_fixed_are_paired(self):
        clean=torch.linspace(.1,.9,3*11*11).reshape(1,3,11,11)
        fixed=OPSContext(clean.shape,8/255,5,5,2.,28).step_plans()
        refresh=OPSRefreshContext(clean.shape,8/255,5,5,2.,28).step_plans()
        model=LifecycleToy();label=torch.tensor([0])
        first=mean_view_gradient(model,clean,label,fixed,1)
        one=mean_view_gradient(model,clean,label,refresh,1)
        two=mean_view_gradient(model,clean,label,refresh,2)
        torch.testing.assert_close(first,one,atol=0,rtol=0)
        torch.testing.assert_close(one,two,atol=2e-7,rtol=2e-5)
        self.assertEqual(gradient_comparison(first,one)['max_abs_error'],0)

    def test_attack_hook_260_ten_pools_deterministic_and_sampling_off(self):
        clean=torch.linspace(.1,.9,3*11*11).reshape(1,3,11,11)
        model=LifecycleToy();counts=[]
        handle=model.register_forward_pre_hook(lambda _,inputs:counts.append(inputs[0].shape[0]))
        actual,stats=attack(model,clean,torch.tensor([0]),method='ops-refresh',steps=10,epsilon_bytes=8,seed=42,view_batch=2)
        handle.remove()
        self.assertEqual(sum(counts),260);self.assertEqual(stats['forward_images'],260);self.assertEqual(stats['input_gradient_images'],260)
        self.assertEqual(stats['noise_pool_generation_count'],10)
        self.assertGreaterEqual(stats['elapsed_seconds'],stats['noise_pool_generation_seconds'])
        repeated,again=attack(LifecycleToy(),clean,torch.tensor([0]),method='ops-refresh',steps=10,epsilon_bytes=8,seed=42,view_batch=2)
        torch.testing.assert_close(actual,repeated,atol=0,rtol=0)
        self.assertEqual(stats['noise_pool_sha256_by_pool'],again['noise_pool_sha256_by_pool'])
        mi,_=attack(LifecycleToy(),clean,torch.tensor([0]),steps=2,epsilon_bytes=8)
        for ne,np_ in ((0,5),(5,0)):
            out,record=attack(LifecycleToy(),clean,torch.tensor([0]),method='ops-refresh',steps=2,epsilon_bytes=8,ne=ne,np_=np_)
            torch.testing.assert_close(out,mi,atol=0,rtol=0)
            self.assertEqual(record['forward_images'],2);self.assertEqual(record['noise_pool_generation_count'],0)

    def test_discordants_image_unit_and_partial_CI_withheld(self):
        rows=[]
        for index in range(276):
            fixed=bool(index%4 in (0,2));refresh=bool(index%4 in (1,2))
            rows.append({'id':str(index),'fixed_success':fixed,'refresh_success':refresh,
                         'fixed_adversarial_margin':.2,'refresh_adversarial_margin':-.1,
                         'fixed_adversarial_ce':2.,'refresh_adversarial_ce':3.})
        complete=paired_population(rows,[str(i) for i in range(276)],True)
        self.assertEqual(complete['fixed_success_refresh_failure'],69)
        self.assertEqual(complete['fixed_failure_refresh_success'],69)
        self.assertEqual(complete['refresh_minus_fixed_asr'],0)
        self.assertEqual(complete['ci']['image_count'],276)
        self.assertEqual(complete['ci']['bootstrap_seed'],20261103)
        partial=paired_population(rows[:69],[str(i) for i in range(276)],False)
        self.assertIsNone(partial['ci']);self.assertEqual(partial['observed_images'],69)
        self.assertIsNone(paired_population(rows[:69],[str(i) for i in range(276)],True)['ci'])
        self.assertEqual(276*260,71760);self.assertEqual(276*2*2,1104);self.assertEqual(2*26*3,156)
