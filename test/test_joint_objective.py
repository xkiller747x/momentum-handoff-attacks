"""Joint (non-switching) fine-tuning objective: logit + weight * guidance."""
import unittest

import numpy as np
import torch

from baseline.feature_finetuning import finetune
from test_feature_finetuning import images, local_model

BASE = {'momentum_norm': 'l1', 'di': 'rdi', 'alpha_bytes': 2.0}


def run(method, protocol, after=None, observer=None):
    x, initial = images()
    model = local_model()
    try:
        return finetune(model, x, initial, 2, 0, method, 16, np.random.RandomState(77), after, observer,
                        mask_count=4, protocol=protocol)
    finally:
        model.close()


def first_momentum(method, protocol):
    kept = []
    run(method, protocol, observer=lambda k, v: kept.append(v['momentum']) if k == 'step' else None)
    return kept[0]


class JointObjectiveTests(unittest.TestCase):
    def test_zero_weight_is_logit_continuation(self):
        joint = run('fft', {**BASE, 'joint_weight': 0.0})
        logit = run('logit-fft', BASE)
        torch.testing.assert_close(joint, logit, atol=1e-6, rtol=0)

    def test_first_step_combines_normalized_parts(self):
        logit = first_momentum('logit-fft', BASE)
        guide = first_momentum('fft', BASE)
        joint = first_momentum('fft', {**BASE, 'joint_weight': 3.0})
        combined = logit + 3 * guide
        torch.testing.assert_close(joint, combined / combined.abs().sum(), atol=1e-9, rtol=1e-4)
        self.assertAlmostEqual(float(joint.abs().sum()), 1.0, places=5)

    def test_two_input_gradients_per_step(self):
        counts = {'feature': 0, 'input': 0}
        def after(kind):
            counts[kind] += 1
        run('fft', {**BASE, 'joint_weight': 1.0}, after)
        self.assertEqual(counts, {'feature': 8, 'input': 20})

    def test_budget_kept(self):
        x, _ = images()
        image = run('aaf', {**BASE, 'joint_weight': 1.0})
        self.assertLessEqual(float((image - x).abs().max()), 16 / 255 + 1e-7)
        self.assertTrue(bool(((image >= 0) & (image <= 1)).all()))

    def test_invalid_combinations_rejected(self):
        for method, protocol in (('logit-fft', {**BASE, 'joint_weight': 1.0}),
                                 ('fft', {'joint_weight': 1.0}),
                                 ('fft', {**BASE, 'joint_weight': -1.0}),
                                 ('fft', {**BASE, 'joint_weight': float('nan')})):
            with self.assertRaises(ValueError):
                run(method, protocol)


if __name__ == '__main__':
    unittest.main()
