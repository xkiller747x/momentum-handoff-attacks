"""Configurable fine-tuning step count (earlier switch at a fixed total budget)."""
import unittest

import numpy as np
import torch

from baseline.feature_finetuning import finetune
from test_feature_finetuning import images, local_model


def steps_of(method, protocol):
    x, initial = images()
    model = local_model()
    kept = []
    try:
        image = finetune(model, x, initial, 2, 0, method, 16, np.random.RandomState(77),
                         observer=lambda k, v: kept.append(v['image']) if k == 'step' else None,
                         mask_count=4, protocol=protocol)
    finally:
        model.close()
    return image, kept


class FineStepsTests(unittest.TestCase):
    def test_author_count_is_bitwise_default(self):
        for method, count in (('fft', 10), ('aaf', 15), ('ila', 10)):
            a, _ = steps_of(method, None)
            b, _ = steps_of(method, {'fine_steps': count})
            self.assertTrue(torch.equal(a, b), method)

    def test_step_count_and_prefix(self):
        for method, protocol in (('fft', {}), ('aaf', {}),
                                 ('ila', {'momentum_norm': 'l1', 'di': 'rdi', 'alpha_bytes': 2.0})):
            short, few = steps_of(method, {**protocol, 'fine_steps': 3})
            _, many = steps_of(method, {**protocol, 'fine_steps': 20})
            self.assertEqual((len(few), len(many)), (3, 20))
            for a, b in zip(few, many[:3]):
                self.assertTrue(torch.equal(a, b), method)
            self.assertTrue(torch.equal(short, few[-1]))
            x, _ = images()
            self.assertLessEqual(float((many[-1] - x).abs().max()), 16 / 255 + 1e-7)

    def test_invalid_counts_rejected(self):
        for value in (0, -1, 2.5):
            with self.assertRaises(ValueError):
                steps_of('fft', {'fine_steps': value})


if __name__ == '__main__':
    unittest.main()
