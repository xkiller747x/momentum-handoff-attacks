"""Fine-tuning optimizer protocol ablation: default must equal the author profile."""
import unittest

import numpy as np
import torch

from baseline.feature_finetuning import PROTOCOL_DEFAULT, finetune
from test_feature_finetuning import images, local_model


def run(method, protocol=None, seed=77, observer=None):
    x, initial = images()
    model = local_model()
    try:
        image = finetune(model, x, initial, 2, 0, method, 16, np.random.RandomState(seed),
                         observer=observer, mask_count=4, protocol=protocol)
    finally:
        model.close()
    return x, initial, image


class FineProtocolTests(unittest.TestCase):
    def test_default_protocol_is_bitwise_author_profile(self):
        for method in ('fft', 'aaf', 'logit-fft', 'logit-aaf'):
            _, _, a = run(method)
            _, _, b = run(method, dict(PROTOCOL_DEFAULT))
            _, _, c = run(method, {'alpha_bytes': 1.0})  # epsilon16/16 written explicitly
            self.assertTrue(torch.equal(a, b), method)
            self.assertTrue(torch.equal(a, c), method)

    def test_each_knob_changes_trajectory_and_keeps_budget(self):
        _, _, author = run('logit-fft')
        for protocol in ({'alpha_bytes': 2.0}, {'di': 'rdi'}, {'momentum_norm': 'l1'},
                         {'alpha_bytes': 2.0, 'di': 'rdi', 'momentum_norm': 'l1'}):
            x, _, image = run('logit-fft', protocol)
            self.assertFalse(torch.equal(author, image), protocol)
            self.assertLessEqual(float((image - x).abs().max()), 16 / 255 + 1e-7)
            self.assertTrue(bool(((image >= 0) & (image <= 1)).all()))

    def test_alpha_sets_first_step_size(self):
        steps = []
        observer = lambda kind, values: steps.append(values['image']) if kind == 'step' else None
        x, initial, _ = run('logit-fft', {'alpha_bytes': 2.0}, observer=observer)
        delta = (steps[0] - initial).abs()
        inside = ((initial - x).abs() < 14 / 255) & (initial > 3 / 255) & (initial < 1 - 3 / 255)
        moved = delta[inside]
        # Toy stride leaves exact-zero gradients (sign 0); every other pixel moves exactly alpha.
        self.assertTrue(bool(((moved < 1e-6) | ((moved - 2 / 255).abs() < 1e-6)).all()))
        self.assertGreater(int(((moved - 2 / 255).abs() < 1e-6).sum()), 1000)

    def test_l1_momentum_is_sum_of_normalized_gradients(self):
        raw, normalized = [], []
        run('logit-fft', observer=lambda k, v: raw.append(v['momentum']) if k == 'step' else None)
        run('logit-fft', {'momentum_norm': 'l1'},
            observer=lambda k, v: normalized.append(v['momentum']) if k == 'step' else None)
        # Single step: normalized momentum is the raw gradient divided by its L1 norm.
        torch.testing.assert_close(normalized[0], raw[0] / raw[0].abs().sum(), atol=1e-9, rtol=1e-5)
        self.assertAlmostEqual(float(normalized[0].abs().sum()), 1.0, places=5)

    def test_rdi_keeps_feature_and_logit_rng_matched(self):
        for method in ('fft', 'aaf'):
            rngs = []
            for name in (method, 'logit-' + method):
                x, initial = images()
                model = local_model()
                rng = np.random.RandomState(5)
                try:
                    finetune(model, x, initial, 2, 0, name, 16, rng, mask_count=4, protocol={'di': 'rdi'})
                finally:
                    model.close()
                rngs.append(rng.get_state())
            np.testing.assert_array_equal(rngs[0][1], rngs[1][1])
            self.assertEqual(rngs[0][2:], rngs[1][2:])

    def test_zero_inherited_momentum_equals_restart(self):
        x, initial = images()
        for method in ('fft', 'logit-fft'):
            _, _, restart = run(method, {'momentum_norm': 'l1', 'di': 'rdi', 'alpha_bytes': 2.0})
            _, _, zero = run(method, {'momentum_norm': 'l1', 'di': 'rdi', 'alpha_bytes': 2.0,
                                      'init_momentum': torch.zeros_like(x)})
            self.assertTrue(torch.equal(restart, zero), method)

    def test_inherited_momentum_dominates_first_step(self):
        x, initial = images()
        inherited = torch.sign(torch.randn(x.shape, generator=torch.Generator().manual_seed(3))) * 50
        steps = []
        run('logit-fft', {'momentum_norm': 'l1', 'alpha_bytes': 2.0, 'init_momentum': inherited},
            observer=lambda k, v: steps.append(v) if k == 'step' else None)
        # One unit-L1 gradient cannot flip a magnitude-50 momentum coordinate.
        self.assertTrue(torch.equal(torch.sign(steps[0]['momentum']), torch.sign(inherited)))
        expected = (x + (initial - x + 2 / 255 * torch.sign(inherited)).clamp(-16 / 255, 16 / 255)).clamp(0, 1)
        torch.testing.assert_close(steps[0]['image'], expected, atol=1e-6, rtol=0)

    def test_base_state_callback_reports_momentum_without_changing_images(self):
        from baseline.targeted import optimize
        x, _ = images()
        model = local_model()
        try:
            plain, seen, states = [], [], {}
            optimize(model, x, 2, 'logit-rtm', 4, 16, 2, 9, checkpoints=(2, 4),
                     callback=lambda s, im: plain.append(im))
            optimize(model, x, 2, 'logit-rtm', 4, 16, 2, 9, checkpoints=(2, 4),
                     callback=lambda s, im: seen.append(im),
                     state_callback=lambda s, m: states.__setitem__(s, m.clone()))
        finally:
            model.close()
        self.assertEqual(sorted(states), [2, 4])
        for a, b in zip(plain, seen):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(states[4].shape, x.shape)

    def test_conflict_reset_zeroes_only_opposing_inherited_coordinates(self):
        x, _ = images()
        inherited = torch.randn(x.shape, generator=torch.Generator().manual_seed(8)) * 50
        proto = {'momentum_norm': 'l1', 'alpha_bytes': 2.0, 'di': 'rdi'}
        steps, conflicts, restart = [], [], []
        run('fft', {**proto, 'init_momentum': inherited, 'conflict_reset': True},
            observer=lambda k, v: steps.append(v) if k == 'step' else conflicts.append(v) if k == 'conflict_reset' else None)
        run('fft', proto, observer=lambda k, v: restart.append(v) if k == 'step' else None)
        gradient = restart[0]['momentum']  # restart momentum after one step is exactly that step's gradient
        conflict = (torch.sign(inherited) != torch.sign(gradient)) & (gradient != 0)
        expected = torch.where(conflict, gradient, inherited + gradient)
        torch.testing.assert_close(steps[0]['momentum'], expected, atol=1e-6, rtol=1e-6)
        self.assertAlmostEqual(conflicts[0]['fraction'], float(conflict.float().mean()), places=6)
        self.assertEqual(len(conflicts), 1)
        # Where the first gradient is nonzero, the first update equals a full restart's;
        # exact-zero-gradient pixels keep moving along the inherited direction instead.
        moving = gradient != 0
        self.assertTrue(torch.equal(steps[0]['image'][moving], restart[0]['image'][moving]))
        self.assertFalse(torch.equal(steps[0]['image'][~moving], restart[0]['image'][~moving]))
        self.assertFalse(torch.equal(steps[-1]['momentum'], restart[-1]['momentum']))

    def test_averaged_conflict_judgment_counts_extra_gradients(self):
        x, initial = images()
        inherited = torch.randn(x.shape, generator=torch.Generator().manual_seed(9)) * 50
        proto = {'momentum_norm': 'l1', 'alpha_bytes': 2.0, 'di': 'rdi', 'init_momentum': inherited,
                 'conflict_reset': True}
        for k in (1, 4):
            model = local_model()
            cost, info = {'feature': 0, 'input': 0}, []
            try:
                finetune(model, x, initial, 2, 0, 'logit-fft', 16, np.random.RandomState(2),
                         lambda kind: cost.__setitem__(kind, cost[kind] + 1),
                         lambda kind, v: info.append(v) if kind == 'conflict_reset' else None,
                         mask_count=4, protocol={**proto, 'conflict_samples': k})
            finally:
                model.close()
            self.assertEqual(cost['input'], 10 + (k if k > 1 else 0))
            self.assertEqual(info[0]['samples'], k)
            if k == 1:
                self.assertEqual(info[0]['fraction'], info[0]['step_gradient_fraction'])

    def test_conflict_samples_validation(self):
        with self.assertRaises(ValueError):
            run('fft', {'conflict_samples': 4})
        x, _ = images()
        with self.assertRaises(ValueError):
            run('fft', {'init_momentum': torch.zeros_like(x), 'conflict_reset': True, 'conflict_samples': 0})

    def test_conflict_reset_requires_inherited_momentum(self):
        with self.assertRaises(ValueError):
            run('fft', {'conflict_reset': True})

    def test_start_selection_png_float_and_continue(self):
        import tempfile, types
        from pathlib import Path
        from PIL import Image
        from utils.targeted_finetuning import continue_start
        from utils.io import sha256
        x, _ = images()
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / 'a.png'
            Image.fromarray((x[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(png)
            floating = Path(tmp) / 'a.npz'; momentum = Path(tmp) / 'a.momentum.npz'
            np.savez_compressed(floating, image=x.numpy()); np.savez_compressed(momentum, momentum=np.ones_like(x.numpy()))
            entry = {'png_path': str(png), 'float_path': str(floating), 'float_sha256': sha256(floating),
                     'momentum_path': str(momentum), 'momentum_sha256': sha256(momentum)}
            args = lambda **k: types.SimpleNamespace(epsilon_bytes=16, **k)
            device = torch.device('cpu')
            image, proto = continue_start(entry, x, args(), device)
            self.assertIsNone(proto['init_momentum'])
            self.assertFalse(torch.equal(image, x))  # PNG is quantized
            image, proto = continue_start(entry, x, args(fine_float_start=True), device)
            self.assertTrue(torch.equal(image, x)); self.assertIsNone(proto['init_momentum'])
            image, proto = continue_start(entry, x, args(fine_continue=True), device)
            self.assertTrue(torch.equal(image, x)); self.assertTrue(torch.equal(proto['init_momentum'], torch.ones_like(x)))
            with self.assertRaises(ValueError):
                continue_start({'png_path': str(png)}, x, args(fine_float_start=True), device)
            image, proto = continue_start(entry, x, args(fine_continue=True, fine_momentum_decay=0.25), device)
            self.assertTrue(torch.equal(proto['init_momentum'], torch.full_like(x, 0.25)))
            with self.assertRaises(ValueError):
                continue_start(entry, x, args(fine_continue=True, fine_momentum_decay=1.5), device)

    def test_ila_guide_matches_projection_loss_gradient(self):
        x, initial = images()
        model = local_model()
        try:
            seen = []
            finetune(model, x, initial, 2, 0, 'ila', 16, np.random.RandomState(3),
                     observer=lambda k, v: seen.append(v) if k == 'guidance' else None, mask_count=4)
            guide = seen[0]['combined']
            with torch.no_grad():
                model(x); f_clean = model.value.clone(); model(initial); f0 = model.value.clone()
            torch.testing.assert_close(guide, f0 - f_clean)
            probe = initial.clone().requires_grad_(True)
            model(probe)
            g_proj, = torch.autograd.grad(((model.value - f_clean) * (f0 - f_clean)).sum(), probe)
            probe = initial.clone().requires_grad_(True)
            model(probe)
            g_guide, = torch.autograd.grad((guide * model.value).sum(), probe)
            torch.testing.assert_close(g_proj, g_guide, atol=1e-7, rtol=1e-5)
        finally:
            model.close()

    def test_ila_and_logit_ila_share_rng_and_stay_in_budget(self):
        states = []
        for method in ('ila', 'logit-ila'):
            rng = np.random.RandomState(6)
            x, _, image = run(method, seed=6)
            model = local_model(); x, initial = images()
            try:
                finetune(model, x, initial, 2, 0, method, 16, rng, mask_count=4)
            finally:
                model.close()
            states.append(rng.get_state()[1].copy())
            self.assertLessEqual(float((image - x).abs().max()), 16 / 255 + 1e-7)
        np.testing.assert_array_equal(states[0], states[1])

    def test_holdout_gate_requires_registered_protocol(self):
        import tempfile, types
        from pathlib import Path
        from utils.targeted_finetuning import holdout_gate
        ns = lambda **k: types.SimpleNamespace(**k)
        self.assertFalse(holdout_gate(ns(split='dev')))
        with self.assertRaises(ValueError):
            holdout_gate(ns(split='holdout'))
        with self.assertRaises(ValueError):
            holdout_gate(ns(split='holdout', allow_holdout=True))
        with self.assertRaises(ValueError):
            holdout_gate(ns(split='dev,holdout', allow_holdout=True, holdout_protocol='results/holdout_protocol_v1.md'))
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / 'p.md'; fake.write_text('other')
            with self.assertRaises(ValueError):
                holdout_gate(ns(split='holdout', allow_holdout=True, holdout_protocol=str(fake)))
        self.assertTrue(holdout_gate(ns(split='holdout', allow_holdout=True, holdout_protocol='results/holdout_protocol_v1.md')))

    def test_unknown_protocol_rejected(self):
        with self.assertRaises(ValueError):
            run('fft', {'di': 'other'})
        with self.assertRaises(ValueError):
            run('fft', {'momentum_norm': 'l2'})


if __name__ == '__main__':
    unittest.main()
