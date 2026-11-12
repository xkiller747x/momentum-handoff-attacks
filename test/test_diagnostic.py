"""Synthetic tests for the frozen matrix and its statistical unit."""
import random
import unittest

import numpy as np
import torch

from baseline.attacks import attack
from baseline.transforms import make_plans
from test_pipeline import Toy
from utils.matrix import build_jobs, SEEDS
from utils.matrix_report import analyze
from utils.metrics import image_seed, logit_metrics, paired_bootstrap, perturbation_metrics, seed_average


class DiagnosticTests(unittest.TestCase):
    def test_din_one_view_equals_momentum_di(self):
        clean = torch.linspace(.1, .9, 3 * 12 * 12).reshape(1, 3, 12, 12)
        common = dict(steps=3, views=1, seed=82, di_probability=1, resize_rate=1.5)
        di, _ = attack(Toy(), clean, torch.tensor([0]), method='di', **common)
        din, _ = attack(Toy(), clean, torch.tensor([0]), method='di-n', **common)
        torch.testing.assert_close(di, din, rtol=0, atol=0)

    def test_local_seed_order_independent_and_cross_alpha_shared(self):
        ids = ['c', 'a', 'b']
        def sample(order):
            return {identity: make_plans('sia', 5, random.Random(image_seed(20261027, identity))) for identity in order}
        first = sample(ids)
        random.seed(991)
        for _ in range(100):
            random.random()
        self.assertEqual(first, sample(list(reversed(ids))))
        self.assertNotEqual(image_seed(20261027, 'a'), image_seed(20261028, 'a'))
        # Alpha is deliberately absent from the image/view seed contract.
        self.assertEqual(sample(ids), first)

    def test_perturbation_units_and_channel_boundary_fraction(self):
        clean = np.full((1, 2, 3), 100, np.uint8)
        adversarial = np.array([[[108, 92, 100], [104, 100, 100]]], dtype=np.uint8)
        result = perturbation_metrics(clean, adversarial, 8)
        self.assertEqual(result['png_channel_element_count'], 6)
        self.assertAlmostEqual(result['png_mean_abs_bytes'], 20 / 6)
        self.assertAlmostEqual(result['png_mean_abs_unit'], 20 / (6 * 255))
        self.assertAlmostEqual(result['png_l2_unit'], 12 / 255)
        self.assertAlmostEqual(result['png_boundary_element_fraction'], 2 / 6)

    def test_margin_and_cross_entropy(self):
        logits = torch.tensor([[2., -1., 4.]])
        result = logit_metrics(logits, 0)
        self.assertEqual(result['margin'], -2)
        self.assertAlmostEqual(result['ce'], float(torch.logsumexp(logits[0], 0) - 2), places=6)
        self.assertEqual(logit_metrics(logits, 2)['margin'], 2)

    def test_seed_averaging_and_paired_image_bootstrap(self):
        ids = ['a', 'b', 'c']
        values = {1: {'a': 0, 'b': 1, 'c': 0}, 2: {'a': 1, 'b': 1, 'c': 0}, 3: {'a': 1, 'b': 0, 'c': 0}}
        averaged = seed_average(values, [1, 2, 3], ids)
        np.testing.assert_allclose(averaged, [2 / 3, 2 / 3, 0])
        result = paired_bootstrap(averaged, resamples=10000)
        self.assertEqual(result['image_count'], 3)
        self.assertAlmostEqual(result['estimate'], 4 / 9)
        self.assertEqual(result, paired_bootstrap(averaged, resamples=10000))
        # Perfectly paired cells cancel before bootstrap, regardless of their variation.
        zero = paired_bootstrap(averaged - averaged)
        self.assertEqual(zero['ci95_low'], 0)
        self.assertEqual(zero['ci95_high'], 0)
        with self.assertRaises(ValueError):
            seed_average(values, [1, 4], ids)

    def test_matrix_exact_cost_configurations_and_priority(self):
        jobs = build_jobs()
        self.assertEqual(len(jobs), 44)
        self.assertEqual(len({j['id'] for j in jobs}), 44)
        self.assertEqual(sum(j['planned_gradient_images'] for j in jobs), 430560)
        self.assertEqual(sum(j['budget'] == 'B20' for j in jobs), 36)
        self.assertEqual(sum(j['budget'] == 'B200' for j in jobs), 4)
        self.assertEqual(sum(j['budget'] == 'classic' for j in jobs), 4)
        self.assertEqual([j['order'] for j in jobs], list(range(1, 45)))
        stages = [j['stage'] for j in jobs]
        self.assertEqual(stages[:18], ['b20-a1'] * 18)
        self.assertEqual(stages[18:22], ['classic'] * 4)
        self.assertEqual(stages[22:26], ['b200'] * 4)
        self.assertEqual(stages[26:32], ['b20-a2-s27'] * 6)
        self.assertTrue(all(j['seed'] == SEEDS[0] for j in jobs[22:32]))
        self.assertEqual([(j['steps'], j['views']) for j in jobs[22:26]], [(10, 20), (10, 20), (40, 5), (40, 5)])

    def test_missing_cells_do_not_produce_contrast_ci(self):
        plan = {'selected_ids': ['a', 'b']}
        aggregate, contrasts = analyze(plan, {'dn121': True, 'vit_b_16': True},
                                       {'dn121': ['a', 'b'], 'vit_b_16': ['a', 'b']}, {}, [])
        self.assertEqual(aggregate, [])
        self.assertTrue(all(r['status'] == 'missing' for r in contrasts))
        self.assertTrue(all('ci95_low' not in r for r in contrasts))

    def test_primary_contrast_averages_seeds_before_image_resampling(self):
        plan = {'selected_ids': ['a', 'b']}
        groups = {}
        for method in ('sia', 'di-n'):
            for steps, views in ((5, 4), (20, 1)):
                key = ('B20', method, steps, views, 1)
                groups[key] = {}
                for index, seed in enumerate(SEEDS):
                    value = int(method == 'sia' and steps == 5 and index != 1)
                    groups[key][seed] = {'dn121': {
                        'a': {'success': value, 'adversarial_margin': -float(value)},
                        'b': {'success': 0, 'adversarial_margin': 0.}}}
        _, contrasts = analyze(plan, {'dn121': True, 'vit_b_16': False},
                               {'dn121': ['a', 'b'], 'vit_b_16': []}, groups, [])
        primary = next(r for r in contrasts if r['target'] == 'dn121'
                       and r['contrast'] == 'primary_B20_alpha1' and r['endpoint'] == 'success'
                       and r['comparison'] == 'difference_in_differences')
        self.assertEqual(primary['image_count'], 2)
        self.assertEqual(primary['seed_count'], 3)
        self.assertAlmostEqual(primary['estimate'], 1 / 3)
        self.assertAlmostEqual(primary['ci95_low'], 0)
        self.assertAlmostEqual(primary['ci95_high'], 2 / 3)
