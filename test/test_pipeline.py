"""Synthetic CPU checks; no downloads, pretrained weights or dataset needed."""
import csv
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from baseline.attacks import attack, mean_view_gradient
from baseline.transforms import SIA_OPERATIONS, ViewPlan, apply_plan, make_plans, sia_operation
from utils.data import prepare, read_labels, select_records, split_ids
from utils.io import image_tensor, read_pixels, save_adversarial
from utils.runner import _open_run


class Toy(nn.Module):
    def forward(self, x):
        weights = torch.linspace(.2, 1.7, x[0].numel(), device=x.device).reshape_as(x[0])
        value = (x.square() * weights).mean((1, 2, 3))
        return torch.stack((value, -value, value * .3), dim=1)


class LinearToy(nn.Module):
    def forward(self, x):
        value = x.mean((1, 2, 3))
        return torch.stack((value, -value), dim=1)


class ZeroToy(nn.Module):
    def forward(self, x):
        zero = x.sum((1, 2, 3)) * 0
        return torch.stack((zero, zero), dim=1)


class PipelineTests(unittest.TestCase):
    def test_png_bounds_and_float_roundtrip(self):
        pixels = np.array([[[0, 255, 127], [255, 0, 128]]], dtype=np.uint8)
        clean = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float() / 255
        adv = (clean + torch.tensor([[[[16 / 255, -16 / 255]], [[-16 / 255, 16 / 255]], [[.5 / 255, -.5 / 255]]]])).clamp(0, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            stats = save_adversarial(path, adv, pixels, 16)
            self.assertEqual(stats['png_roundtrip_max_byte_error'], 0)
            self.assertLessEqual(stats['png_max_byte_difference'], 16)
            self.assertLessEqual(stats['float_to_png_max_abs_error'], .501 / 255)
            torch.testing.assert_close(image_tensor(path), torch.from_numpy(read_pixels(path)).permute(2, 0, 1).unsqueeze(0).float() / 255)
            stats = save_adversarial(path, (clean + 20 / 255).clamp(0, 1), pixels, 15.5)
            self.assertLessEqual(stats['png_max_byte_difference'], 15)

    def test_labels_offsets_split_and_prepare(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            images = base / 'raw'
            images.mkdir()
            csv_path = base / 'labels.csv'
            with open(csv_path, 'w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['filename', 'label', 'targeted_label'])
                for index in range(8):
                    writer.writerow([f'id{index}.png', index + 1, 20])
                    Image.new('RGB', (12, 9), (index, 128, 255)).save(images / f'id{index}.png')
            labels = read_labels(csv_path, 1)
            self.assertEqual(labels[0], ('id0', 1, 0))
            self.assertEqual(read_labels(csv_path, 0)[0], ('id0', 1, 1))
            with self.assertRaises(ValueError):
                read_labels(csv_path, 2)
            result = prepare(images, csv_path, base / 'prepared', 1, expected_count=8, development=4, pilot=2)
            self.assertEqual(result['counts'], {'pilot': 2, 'dev': 2, 'holdout': 4})
            manifest = base / 'prepared' / 'manifest.json'
            self.assertEqual(len(select_records(manifest, 'pilot')), 2)
            with self.assertRaises(ValueError):
                select_records(manifest, 'holdout')
            self.assertEqual(len(select_records(manifest, 'holdout', True)), 4)
            self.assertTrue(all(Path(r['original_path']).exists() for r in result['records']))
            for record in result['records']:
                self.assertEqual(record['original_label'] - 1, record['label'])
                self.assertEqual(read_pixels(record['canonical_path']).shape, (224, 224, 3))
        ids = [f'{i:04}' for i in range(1000)]
        splits = split_ids(ids)
        self.assertEqual(splits, split_ids(list(reversed(ids))))
        self.assertEqual([list(splits.values()).count(s) for s in ('pilot', 'dev', 'holdout')], [24, 276, 700])

    def test_mi_direction_budget_and_counters(self):
        clean = torch.full((1, 3, 8, 8), .5)
        # Label1 loss increases with x, so each MI update must increase pixels.
        adv, stats = attack(LinearToy(), clean, torch.tensor([1]), steps=2, epsilon_bytes=10)
        torch.testing.assert_close(adv, clean + 10 / 255)
        self.assertEqual(stats['forward_images'], 2)
        self.assertEqual(stats['input_gradient_images'], 2)
        self.assertEqual(stats['input_gradient_pixels'], 2 * 8 * 8)
        self.assertEqual(stats['resolution_counts'], {'8x8': 2})
        decrease, _ = attack(LinearToy(), clean, torch.tensor([0]), steps=2, epsilon_bytes=10)
        torch.testing.assert_close(decrease, clean - 10 / 255)

    def test_fixed_plan_microbatch_gradient_against_joint_graph(self):
        clean = torch.linspace(.02, .98, 3 * 12 * 12).reshape(1, 3, 12, 12)
        label = torch.tensor([2])
        model = Toy().eval()
        plans = make_plans('di-n', 5, random.Random(99), di_probability=1, resize_rate=1.5)
        # Independent reference: one joint computation graph of all transformed
        # copies with CE mean, differentiated once wrt a shared original image.
        leaf = clean.clone().requires_grad_()
        transformed = torch.cat([apply_plan(leaf, plan) for plan in plans])
        loss = F.cross_entropy(model(transformed), label.repeat(5), reduction='mean')
        expected = torch.autograd.grad(loss, leaf)[0]
        for batch in (1, 2, 3, 5):
            actual = mean_view_gradient(model, clean, label, plans, batch)
            torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-5)
        self.assertGreater(expected.abs().max().item(), 0)

    def test_di_zero_probability_equals_mi(self):
        clean = torch.linspace(.1, .9, 3 * 8 * 8).reshape(1, 3, 8, 8)
        label = torch.tensor([0])
        mi, _ = attack(Toy(), clean, label, method='mi', steps=3, epsilon_bytes=8)
        di, _ = attack(Toy(), clean, label, method='di', steps=3, epsilon_bytes=8, di_probability=0)
        din, _ = attack(Toy(), clean, label, method='di-n', steps=3, views=5, view_batch=2,
                        epsilon_bytes=8, di_probability=0)
        torch.testing.assert_close(di, mi, rtol=0, atol=0)
        torch.testing.assert_close(din, mi, rtol=0, atol=0)

    def test_seed_repeatability_and_microbatch_attack(self):
        clean = torch.linspace(.1, .9, 3 * 12 * 12).reshape(1, 3, 12, 12)
        label = torch.tensor([0])
        args = dict(method='di-n', steps=2, views=5, seed=37, di_probability=1, resize_rate=1.5)
        first, _ = attack(Toy(), clean, label, view_batch=1, **args)
        second, _ = attack(Toy(), clean, label, view_batch=1, **args)
        grouped, _ = attack(Toy(), clean, label, view_batch=3, **args)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first, grouped, rtol=0, atol=0)
        self.assertEqual(make_plans('sia', 4, random.Random(8)), make_plans('sia', 4, random.Random(8)))

    def test_sia_ten_operations_shape_and_finite_gradients(self):
        self.assertEqual(len(SIA_OPERATIONS), 10)
        for choice in range(10):
            for shape in ((1, 3, 11, 13), (1, 3, 3, 4)):
                x = torch.linspace(.1, .9, int(np.prod(shape))).reshape(shape).requires_grad_()
                plan = ViewPlan('sia', 82, sia_choice=choice)
                transformed = apply_plan(x, plan)
                self.assertEqual(transformed.shape, x.shape)
                self.assertTrue(torch.isfinite(transformed).all())
                gradient = torch.autograd.grad(transformed.square().sum(), x)[0]
                self.assertTrue(torch.isfinite(gradient).all())
                torch.testing.assert_close(transformed, apply_plan(x, plan), rtol=0, atol=0)
        # Author semantics: tiny DCT axes are all zero due to -0: slicing.
        x = torch.ones(1, 3, 1, 2, requires_grad=True)
        out = sia_operation(x, 8, random.Random(0), torch.Generator().manual_seed(0))
        torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-7, rtol=0)
        # Dropout2d masks channels, never individual spatial locations.
        x = torch.ones(1, 30, 4, 5)
        out = sia_operation(x, 9, random.Random(0), torch.Generator().manual_seed(0))
        self.assertTrue(torch.all(out == out[:, :, :1, :1]))
        self.assertTrue(torch.all((out == 0) | torch.isclose(out, torch.tensor(1 / .9))))

    def test_zero_gradient_is_finite_and_recorded(self):
        clean = torch.full((1, 3, 4, 4), .5)
        adv, stats = attack(ZeroToy(), clean, torch.tensor([0]), steps=2)
        torch.testing.assert_close(adv, clean)
        self.assertEqual(stats['zero_grad_count'], 2)
        self.assertTrue(torch.isfinite(adv).all())

    def test_resume_configuration_refuses_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertEqual(_open_run(output, {'seed': 1}), [])
            self.assertEqual(_open_run(output, {'seed': 1}), [])
            with self.assertRaises(ValueError):
                _open_run(output, {'seed': 2})
