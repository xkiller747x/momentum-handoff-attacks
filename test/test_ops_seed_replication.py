"""Only new seed-flow checks; root executes all CPU tests."""
from datetime import datetime,timezone
import unittest
from unittest import mock

import numpy as np
import torch

from utils.ops_seed_replication import seed_units,seed_tasks,seed_identities,cpu_pair_identity,may_start_unit
from utils.ops_seed_report import exact_mcnemar,image_seed_mean,summarize_seed_results
from utils.metrics import image_seed


class OPSSeedTests(unittest.TestCase):
    def test_16_task_order_cost_and_global_seed_identities(self):
        ids=[f'image{i:03}' for i in range(276)]
        units=seed_units(ids)
        self.assertEqual(len(units),8)
        self.assertEqual(sum(len(u['profiles']) for u in units),16)
        self.assertEqual([u['seed'] for u in units],[20261028]*4+[20261029]*4)
        self.assertEqual([u['profiles'] for u in units],[['fixed','refresh'],['refresh','fixed'],['fixed','refresh'],['refresh','fixed'],
                                                     ['refresh','fixed'],['fixed','refresh'],['refresh','fixed'],['fixed','refresh']])
        self.assertEqual(sum(u['source_forward_images'] for u in units),287040)
        self.assertEqual(sum(u['source_gradient_images'] for u in units),287040)
        self.assertEqual(sum(u['target_forward_images'] for u in units),4416)
        tasks=seed_tasks(units)
        self.assertEqual(len(tasks),16)
        self.assertEqual(len({t['id'] for t in tasks}),16)
        self.assertEqual(sum(t['source_forward_images'] for t in tasks),287040)
        self.assertEqual(sum(t['target_forward_images'] for t in tasks),4416)
        mapping=seed_identities(ids)
        self.assertEqual(len({value for values in mapping.values() for value in values.values()}),828)
        for unit in reversed(units):
            for identity in unit['selected_ids']:
                self.assertEqual(mapping[str(unit['seed'])][identity],image_seed(unit['seed'],identity))

    def test_CPU_pair_identity_deterministic_no_CUDA_or_model(self):
        with mock.patch.object(torch.cuda,'get_rng_state',side_effect=AssertionError('CPU only')):
            a=cpu_pair_identity(20261028,'toy',(1,3,11,11));b=cpu_pair_identity(20261028,'toy',(1,3,11,11))
        for key in ('image_seed','first_pool_sha256','step_index_basicplan_sha256','complete_10step_plan_sha256','refresh_pool_sha256_by_pool'):
            self.assertEqual(a[key],b[key])
        self.assertEqual(len(a['step_index_basicplan_sha256']),10)
        self.assertEqual(len(a['refresh_pool_sha256_by_pool']),10)
        self.assertEqual(a['pool_generation_count'],10)
        self.assertTrue(a['global_cpu_rng_unchanged'])
        self.assertFalse(a['cuda_rng_accessed'])
        self.assertEqual(a['extra_forward_images'],0)
        self.assertNotEqual(a['first_pool_sha256'],cpu_pair_identity(20261029,'toy',(1,3,11,11))['first_pool_sha256'])

    def test_exact_binomial_discordant_boundaries(self):
        self.assertEqual(exact_mcnemar(0,0),1)
        self.assertEqual(exact_mcnemar(0,1),1)
        self.assertEqual(exact_mcnemar(1,7),.0703125)
        self.assertEqual(exact_mcnemar(7,1),.0703125)
        self.assertEqual(exact_mcnemar(0,8),.0078125)

    def test_within_image_seed_means_and_no_shrunk_intersection(self):
        differences={27:{'a':1,'b':0},28:{'a':0,'b':1},29:{'a':1,'b':1}}
        np.testing.assert_array_equal(image_seed_mean(differences,[28,29],['a','b']),[.5,1])
        np.testing.assert_allclose(image_seed_mean(differences,[27,28,29],['a','b']),[2/3,2/3])
        with self.assertRaises(ValueError):image_seed_mean({28:{'a':1},29:{'a':1,'b':0}},[28,29],['a','b'])

    def test_new_seed_primary_complete_gate_and_seed_unit_not_IID(self):
        ids=[f'{i:03}' for i in range(276)]
        plan={'targets':['dn121','vit_b_16'],'selected_ids':ids,'populations':{
            'dn121':{'target_clean_ids':ids[:248],'source_target_clean_ids':ids[:241]},
            'vit_b_16':{'target_clean_ids':ids[:262],'source_target_clean_ids':ids[:255]}}}
        data={}
        for seed in (20261027,20261028,20261029):
            data[seed]={}
            for profile in ('fixed','refresh'):
                data[seed][profile]={target:{identity:{'success':profile=='refresh','adversarial_correct':profile=='fixed',
                    'adversarial_margin':-1. if profile=='refresh' else 1.,'adversarial_ce':3. if profile=='refresh' else 1.}
                    for identity in ids} for target in plan['targets']}
        full=summarize_seed_results(plan,data,{20261027:True,20261028:True,20261029:True})
        primary=next(r for r in full['estimands'] if r['role']=='primary replication' and r['target']=='dn121')
        self.assertEqual(primary['ci']['image_count'],248)
        self.assertEqual(primary['estimate'],1)
        self.assertEqual(primary['ci']['bootstrap_seed'],20261104)
        continuous=next(r for r in full['estimands'] if r['estimand']=='three_seeds_secondary' and r['population']=='all276' and r['endpoint']=='adversarial_margin')
        self.assertEqual(continuous['estimate'],-2)
        partial=summarize_seed_results(plan,data,{20261027:True,20261028:True,20261029:False})
        self.assertTrue(all(r['ci'] is None for r in partial['estimands']))
        self.assertTrue(any(r['exact_mcnemar_two_sided'] is not None for r in partial['per_seed'] if r['seed']==20261028))
        self.assertTrue(all(r['exact_mcnemar_two_sided'] is None for r in partial['per_seed'] if r['seed']==20261029))

    def test_clock_rules_no_effect_stop_and_first_unit_projection(self):
        plan={'last_gpu_start_utc':'2026-10-27T02:00:11Z','last_new_unit_start_utc':'2026-10-27T03:45:11Z',
              'last_estimated_unit_end_utc':'2026-10-27T04:00:11Z'}
        now=datetime(2026,10,27,1,59,tzinfo=timezone.utc)
        self.assertTrue(may_start_unit(plan,{},now,0)[0])
        self.assertFalse(may_start_unit(plan,{},datetime(2026,10,27,2,1,tzinfo=timezone.utc),0)[0])
        state={'first_complete_unit_wall_seconds':600}
        self.assertTrue(may_start_unit(plan,state,datetime(2026,10,27,3,44,tzinfo=timezone.utc),6000)[0])
        self.assertFalse(may_start_unit(plan,state,datetime(2026,10,27,3,46,tzinfo=timezone.utc),6000)[0])
        self.assertFalse(may_start_unit(plan,{'first_complete_unit_wall_seconds':1200},datetime(2026,10,27,3,44,tzinfo=timezone.utc),6000)[0])
        self.assertFalse(may_start_unit(plan,{**state,'first_unit_projected_over_limit':True},now,600)[0])
        self.assertFalse(may_start_unit(plan,state,now,7200)[0])
