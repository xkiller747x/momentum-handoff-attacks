"""CPU toy checks only; no pretrained model, dataset, target or GPU use."""
import unittest

import numpy as np
import torch
from torch import nn

from baseline.ops import OPSContext, apply_ops, OPSViewPlan, BasicPlan
from utils.metrics import image_seed
from utils.pool_diagnostic import (FORWARDS_PER_IMAGE, endpoint_values, evaluation_operators,
                                   measure_image, reconstruct_pools)
from utils.pool_report import describe_usage, summarize_endpoints


class ProbeToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.8))
        self.grad_modes = []

    def forward(self, x):
        self.grad_modes.append(torch.is_grad_enabled())
        value = (x.square().mean((1, 2, 3)) + x.mean((1, 2, 3))) * self.weight
        return torch.stack((value, -value, value * .2), dim=1)


class PoolDiagnosticTests(unittest.TestCase):
    def test_fit_pool_exact_and_complete_usage_trajectory(self):
        identity, shape = 'synthetic-image', (1, 3, 11, 11)
        seed = image_seed(20261027, identity)
        original = OPSContext(shape, 8 / 255, 5, 5, 2., seed)
        pools, metadata = reconstruct_pools(identity, seed, shape)
        self.assertEqual([len(pool) for pool in pools], [30] * 4)
        for left, right in zip(original.pool, pools[0]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        usage, last, trajectory = [0] * 30, [None] * 30, []
        for step in range(1, 11):
            selected = original.step_plans()[1:]
            indices = list(dict.fromkeys(view.pool_index for view in selected))
            self.assertEqual(len(indices), 5)
            self.assertTrue(all(sum(view.pool_index == index for view in selected) == 5 for index in indices))
            trajectory.append(indices)
            for index in indices:
                usage[index] += 1
                last[index] = step
        self.assertEqual(metadata['usage_count'], usage)
        self.assertEqual(metadata['last_step'], last)
        self.assertEqual(metadata['training_neighbor_indices_by_step'], trajectory)
        self.assertEqual(sum(usage), 50)
        again, repeated = reconstruct_pools(identity, seed, shape)
        self.assertEqual(metadata, repeated)
        for left, right in zip(pools[3], again[3]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_full_operator_parameters_shared_and_deterministic(self):
        shape = (1, 3, 11, 11)
        plans, metadata = evaluation_operators('shared-plan', shape)
        repeated, repeated_metadata = evaluation_operators('shared-plan', shape)
        self.assertEqual(plans, repeated)
        self.assertEqual(metadata, repeated_metadata)
        self.assertEqual(len(plans), 30)
        self.assertTrue(all(len(selected) == 2 for selected in plans))
        for row in metadata['parameters']:
            self.assertEqual(len({value['composite_pool_index'] for value in row['selected']}), 2)
        x = torch.linspace(-.1, 1.1, 3 * 11 * 11).reshape(shape)
        for operators in plans[0]:
            reference = apply_ops(x, OPSViewPlan(torch.zeros_like(x), operators))
            for _pool in range(4):
                for _state in range(2):
                    torch.testing.assert_close(reference, apply_ops(x, OPSViewPlan(torch.zeros_like(x), operators)), rtol=0, atol=0)
        self.assertNotEqual(metadata['plan_sha256'], evaluation_operators('other-image', shape)[1]['plan_sha256'])

    def test_clean_replacement_zero_D_actual_hook_count_and_no_grad(self):
        shape = (1, 3, 11, 11)
        x = torch.linspace(.1, .9, 3 * 11 * 11).reshape(shape)
        pools, _ = reconstruct_pools('clean-control', image_seed(20261027, 'clean-control'), shape)
        operators, _ = evaluation_operators('clean-control', shape)
        model = ProbeToy()
        independently_counted = []
        hook = model.register_forward_pre_hook(lambda _, inputs: independently_counted.append(inputs[0].shape[0]))
        result = measure_image(model, x, x.clone(), 0, pools, operators)
        hook.remove()
        self.assertEqual(result['ce_endpoints']['D'], 0)
        self.assertEqual(result['ce_endpoints']['A'], 0)
        self.assertEqual(result['margin_descriptive']['D'], 0)
        self.assertEqual(result['actual_counter']['forward_images'], 482)
        self.assertEqual(sum(independently_counted), 482)
        self.assertEqual(len(model.grad_modes), 482)
        self.assertFalse(any(model.grad_modes))
        self.assertTrue(result['actual_counter']['all_no_grad'])
        self.assertIsNone(model.weight.grad)
        self.assertEqual(result['actual_counter']['input_gradient_images'], 0)
        np.testing.assert_array_equal(result['noise_residual_d'], np.zeros(30))

    def test_hand_paired_D_and_placebo_algebra(self):
        values = np.array([[2., 5.], [1., 3.], [3., 6.], [2., 4.]])
        result = endpoint_values(values)
        expected_g = 5 - 13 / 3
        expected_c = 2 - 2
        self.assertAlmostEqual(result['D'], expected_g - expected_c)
        self.assertAlmostEqual(result['G'], expected_g)
        self.assertAlmostEqual(result['C'], expected_c)
        self.assertAlmostEqual(result['A'], 13 / 3 - 2)
        for endpoint in ('clean', 'png', 'gain_difference'):
            self.assertAlmostEqual(sum(r[endpoint] for r in result['fresh_placebo']), 0)

    def test_full_image_inference_partial_and_exact_cost(self):
        ids = [f'id{i:03}' for i in range(276)]
        plan = {'formal_all_ids': ids, 'selected_ids': ids, 'source_clean_ids': ids[9:],
                'mode': 'formal276', 'planned_forward_images': 133032}
        rows = [{'id': identity, 'source_clean_correct': index >= 9, 'forward_images': 482,
                 'D': 0. if index < 138 else 2., 'G': 2. if index < 138 else 4., 'C': 2., 'A': 4.}
                for index, identity in enumerate(ids)]
        summary = summarize_endpoints(plan, rows)
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(FORWARDS_PER_IMAGE, 482)
        self.assertEqual(FORWARDS_PER_IMAGE * 276, 133032)
        self.assertEqual(FORWARDS_PER_IMAGE * 2, 964)
        primary = summary['populations']['source_clean267']['endpoints']['D']['ci']
        self.assertEqual(primary['image_count'], 267)
        self.assertAlmostEqual(primary['estimate'], sum(r['D'] for r in rows[9:]) / 267)
        self.assertEqual(summary['populations']['all276']['endpoints']['D']['ci']['image_count'], 276)
        self.assertAlmostEqual(summary['populations']['all276']['D_over_A_descriptive_point_ratio'], .25)
        partial = summarize_endpoints(plan, rows[:69])
        self.assertEqual(partial['status'], 'partial')
        self.assertTrue(all(v['ci'] is None for p in partial['populations'].values() for v in p['endpoints'].values()))
        engineering_plan = {**plan, 'mode': 'engineering_first2', 'selected_ids': ids[:2], 'planned_forward_images': 964}
        engineering = summarize_endpoints(engineering_plan, rows[:2])
        self.assertEqual(engineering['status'], 'engineering_only')
        self.assertIsNone(engineering['populations']['source_clean267']['endpoints']['D']['ci'])

    def test_usage_radius_missing_groups_stay_missing(self):
        detail = {'id': 'one', 'source_clean_correct': True,
                  'pool_metadata': {'usage_count': [10] * 5 + [0] * 25, 'last_step': [10] * 5 + [None] * 25},
                  'noise_residual_d': [float(i) for i in range(30)]}
        result = describe_usage([detail])
        source = [r for r in result['radius_used_unused_summary'] if r['population'] == 'source_clean']
        self.assertTrue(all(r['available_images'] == 0 and r['missing_images'] == 1 and r['mean_used_minus_unused_d'] is None for r in source))
        usage10 = next(r for r in result['usage_count_summary'] if r['population'] == 'source_clean' and r['usage_count'] == 10)
        self.assertEqual(usage10['available_images'], 1)
        self.assertEqual(usage10['noise_count'], 5)
        self.assertEqual(usage10['mean_within_image_mean_d'], 2)
