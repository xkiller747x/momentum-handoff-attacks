"""CPU-only fixed-plan upstream AST oracle tests; root executes these tests."""
import ast
import functools
import hashlib
from pathlib import Path
import random
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import functional as TFF

from baseline.attacks import attack, mean_view_gradient
from baseline.bsr import BSRPlan, BSRStrip, apply_bsr, lengths_from_uniform, sample_plan
from baseline.ops import (ANGLES, BASIC_OPERATORS, BasicPlan, OPSContext, OPSViewPlan,
                          RATES, apply_basic, apply_ops, composite_pool, plan_basic)
from baseline.transforms import apply_plan
from utils.models import NormalizeModel
from utils.metrics import image_seed
from utils.related_matrix import build_related_tasks
from utils.related_report import related_analysis

REFERENCE = Path(__file__).parent / 'reference'
HASHES = {'bsr': '7f054309c25616c66a581c4497a7a1841e41b949699d4b6baaa0e227cbfe0a05',
          'ops': 'b25bafbac3c5e008f351f40612ce50d981f9fea689b5d40a3dc5dc31084ae231'}


class StubAttack:
    def __init__(self, *args, **kwargs):
        self.device = torch.device('cpu')
        self.loss = nn.CrossEntropyLoss()
        self.targeted = False

    def get_logits(self, x):
        return self.model(x)

    def get_loss(self, logits, label):
        return self.loss(logits, label)

    def get_grad(self, loss, delta):
        return torch.autograd.grad(loss, delta)[0]


def oracle(name):
    content = (REFERENCE / f'{name}.py.txt').read_bytes()
    if hashlib.sha256(content).hexdigest() != HASHES[name]:
        raise AssertionError('Upstream fixture hash differs')
    tree = ast.parse(content.decode('utf-8-sig'))
    # Imports are deliberately excluded: inherited model loaders never execute.
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    namespace = {'torch': torch, 'np': np, 'random': random, 'functools': functools,
                 'F': F, 'T': T, 'TFF': TFF, 'Attack': StubAttack, 'MIFGSM': StubAttack}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f'{name}-fixed-upstream', 'exec'), namespace)
    return namespace


class SmallModel(nn.Module):
    def forward(self, x):
        # Supports upstream DI's square output even for non-square test inputs.
        value = x.square().mean((1, 2, 3)) + x.mean((1, 2, 3)) * .2
        return torch.stack((value, -value), dim=1)


def compare(test, local, reference):
    torch.testing.assert_close(local, reference, atol=1e-6, rtol=1e-5)
    difference = (local - reference).double().flatten()
    maximum = difference.abs().max().item()
    relative = difference.norm().item() / max(reference.double().norm().item(), 1e-30)
    print(f'upstream differential shape={tuple(local.shape)} max_abs={maximum:.9g} relative_l2={relative:.9g}', flush=True)
    test.assertLessEqual(maximum, 1e-6 + reference.abs().max().item() * 1e-5)


class RelatedTests(unittest.TestCase):
    def test_bsr_lengths_legacy_semantics_and_10000_plans(self):
        namespace = oracle('bsr')
        reference = namespace['BSR']('stub')
        reference.num_block = 3
        rng = np.random.RandomState(820)
        for index in range(10000):
            length = (11, 13, 224)[index % 3]
            uniforms = rng.random_sample(3)
            with mock.patch.object(namespace['np'].random, 'uniform', return_value=2 - uniforms):
                expected = reference.get_length(length)
            actual = lengths_from_uniform(length, uniforms)
            self.assertEqual(actual, tuple(expected))
            self.assertEqual(sum(actual), length)
            self.assertTrue(all(value > 0 for value in actual))
        # Actual installed legacy API must implement low+(high-low)*U, high=1.
        with mock.patch.object(np.random, 'random_sample', wraps=np.random.random_sample):
            np.random.seed(921)
            observed = np.random.uniform(2, size=3)
            np.random.seed(921)
            expected = 2 - np.random.random_sample(3)
            np.testing.assert_array_equal(observed, expected)
        self.assertEqual(lengths_from_uniform(10, [0, 0, 0]), (4, 3, 3))
        for index in range(100):
            plan = sample_plan((1, 3, 11, 13), index)
            self.assertEqual(sorted(plan.permutation), [0, 1, 2])
            self.assertTrue(all(sorted(s.permutation) == [0, 1, 2] for s in plan.strips))

    def test_bsr_fixed_upstream_output_and_gradient(self):
        for axis in (2, 3):
            namespace = oracle('bsr')
            reference = namespace['BSR']('stub')
            first = (3, 4, 4) if axis == 2 else (4, 4, 5)
            second = (4, 4, 5) if axis == 2 else (3, 4, 4)
            strips = tuple(BSRStrip(angle, second, permutation) for angle, permutation in
                           zip((0., 17., -23.), ((2, 0, 1), (1, 2, 0), (0, 2, 1))))
            plan = BSRPlan(axis, first, (2, 0, 1), strips)
            permutations = iter([(axis - 2, 3 - axis), plan.permutation] + [s.permutation for s in strips])
            def shuffle(values):
                permutation = next(permutations)
                values[:] = [values[i] for i in permutation]
            reference.get_length = mock.Mock(side_effect=[first] + [second] * 3)
            x = torch.linspace(.1, .9, 3 * 11 * 13).reshape(1, 3, 11, 13).requires_grad_()
            with mock.patch.object(namespace['random'], 'shuffle', side_effect=shuffle), \
                 mock.patch.object(T.RandomRotation, 'get_params', side_effect=[s.angle for s in strips]):
                expected = reference.shuffle(x)
            actual = apply_bsr(x, plan)
            compare(self, actual, expected)
            compare(self, torch.autograd.grad(actual.square().sum(), x)[0],
                    torch.autograd.grad(expected.square().sum(), x)[0])
            zero_plan = BSRPlan(axis, first, (2, 0, 1), tuple(BSRStrip(0, second, s.permutation) for s in strips))
            zero = apply_bsr(x, zero_plan)
            torch.testing.assert_close(zero.flatten().sort().values, x.flatten().sort().values)

    def test_bsr_fill_before_normalization(self):
        x = torch.ones(1, 3, 11, 13)
        plan = BSRPlan(2, (3, 4, 4), (0, 1, 2), tuple(BSRStrip(24, (4, 4, 5), (0, 1, 2)) for _ in range(3)))
        transformed = apply_bsr(x, plan)
        self.assertLess(transformed.min().item(), .5)
        wrapped = NormalizeModel(nn.Identity(), [.5] * 3, [.5] * 3)
        torch.testing.assert_close(wrapped(transformed), (transformed - .5) / .5)
        self.assertLess(wrapped(transformed).min().item(), 0)

    def test_ops_31_basic_functions_against_fixed_upstream(self):
        namespace = oracle('ops')
        reference = namespace['OPS']('stub', num_sample_neighbor=5, num_sample_operator=5)
        self.assertEqual(len(reference.basic_ops), len(BASIC_OPERATORS))
        self.assertEqual(len(BASIC_OPERATORS), 31)
        for index in range(31):
            x = torch.linspace(-.1, 1.1, 3 * 21 * 23).reshape(1, 3, 21, 23).requires_grad_()
            operation = plan_basic(index, x.shape, random.Random(221 + index))
            if index in (3, 4):
                with mock.patch.object(namespace['np'].random, 'randint', return_value=operation.parameters[0]):
                    expected = reference.basic_ops[index](x)
            elif index >= 21:
                resized, _, top, left = operation.parameters
                side_effect = [torch.tensor([value], dtype=torch.int32) for value in (resized, top, left)]
                with mock.patch.object(namespace['torch'], 'randint', side_effect=side_effect):
                    expected = reference.basic_ops[index](x)
            else:
                expected = reference.basic_ops[index](x)
            actual = apply_basic(x, operation)
            compare(self, actual, expected)
            grad = torch.autograd.grad(actual.square().sum(), x)[0]
            compare(self, grad, torch.autograd.grad(expected.square().sum(), x)[0])
            self.assertTrue(torch.isfinite(actual).all() and torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
            print(f'OPS basic{index}: output={tuple(actual.shape)} range=[{actual.min().item():.6g},{actual.max().item():.6g}] '
                  f'out_of_range={((actual < 0) | (actual > 1)).float().mean().item():.6g}', flush=True)

    def test_ops_composition_pool_and_fixed_neighborhoods(self):
        namespace = oracle('ops')
        reference = namespace['OPS']('stub', num_sample_neighbor=5, num_sample_operator=5)
        # Actual upstream reduce order with explicitly noncommuting operators.
        selected = [lambda x: x + 2, lambda x: x * 3, lambda x: x.square()]
        with mock.patch.object(namespace['random'], 'choices', return_value=selected):
            composite = reference.get_new_ops(k=3)
        self.assertEqual(composite(torch.tensor(2.)).item(), 14.)
        x = torch.arange(3 * 21 * 23).float().reshape(1, 3, 21, 23) / 1000
        operations = (BasicPlan(3, (2,)), BasicPlan(5), BasicPlan(14))
        with mock.patch.object(namespace['random'], 'choices', return_value=[reference.basic_ops[i] for i in (3, 5, 14)]):
            composite = reference.get_new_ops(k=3)
        with mock.patch.object(namespace['np'].random, 'randint', return_value=2):
            expected = composite(x)
        compare(self, apply_ops(x, OPSViewPlan(None, operations)), expected)
        pool = composite_pool(random.Random(81))
        self.assertEqual([len(p) for p in pool], [2] * 31 + [3] * 31 + [4] * 31)
        reference.init_op_list()
        self.assertEqual(reference.op_num, 93)
        context = OPSContext((1, 3, 21, 23), 8 / 255, 5, 5, 2, 77)
        stored = [noise.clone() for noise in context.pool]
        addresses = [noise.data_ptr() for noise in context.pool]
        self.assertEqual(len(stored), 30)
        for _ in range(3):
            plans = context.step_plans()
            self.assertEqual(len(plans), 26)
            self.assertIsNone(plans[0].noise)
            for plan in plans[1:]:
                self.assertEqual(plan.noise.data_ptr(), addresses[plan.pool_index])
        for left, right in zip(stored, context.pool):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        reference.num_extra_eps = 5
        reference.sample_radius = np.array([.25, .5, .75, 1, 1.25, 1.5]) * 2 * 8 / 255
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(77)
            reference.init_eps_list(torch.zeros(1, 3, 21, 23))
        for local, upstream in zip(context.pool, reference.eps_list):
            torch.testing.assert_close(local.squeeze(0), upstream, rtol=0, atol=0)

    def test_ops_raw_single_step_oracle_and_microbatch(self):
        namespace = oracle('ops')
        reference = namespace['OPS']('stub', num_sample_neighbor=1, num_sample_operator=2)
        reference.model = SmallModel()
        clean = torch.linspace(.1, .9, 3 * 21 * 21).reshape(1, 3, 21, 21)
        noise = torch.full_like(clean, .2)
        plans = [OPSViewPlan(None, ()), OPSViewPlan(noise, (BasicPlan(14),)),
                 OPSViewPlan(noise, (BasicPlan(5),))]
        reference.eps_list = [noise.squeeze(0)]
        reference.init_op_list = lambda: None
        reference.op_list = [reference.basic_ops[14], reference.basic_ops[5]]
        delta = torch.zeros_like(clean, requires_grad=True)
        expected = reference.get_averaged_gradient(clean, delta, torch.tensor([0]))
        for batch_size in (1, 2, 3):
            actual = mean_view_gradient(SmallModel(), clean, torch.tensor([0]), plans, batch_size)
            compare(self, actual, expected)
        # Zero radius alone is not MI; all-identity zero-radius plans are.
        identity = [OPSViewPlan(torch.zeros_like(clean), (BasicPlan(0),))] * 3
        expected_identity = mean_view_gradient(SmallModel(), clean, torch.tensor([0]), [OPSViewPlan(None, ())])
        compare(self, mean_view_gradient(SmallModel(), clean, torch.tensor([0]), identity), expected_identity)
        zero_radius_nonidentity = [OPSViewPlan(torch.zeros_like(clean), (BasicPlan(14),))]
        different = mean_view_gradient(SmallModel(), clean, torch.tensor([0]), zero_radius_nonidentity)
        self.assertGreater((different - expected_identity).abs().max().item(), 1e-8)

    def test_forward_hook_cost_degeneracy_and_no_neighbor_clip(self):
        clean = torch.full((1, 3, 21, 21), .95)
        label = torch.tensor([0])
        for method, views, ne, np_, expected in [('bsr', 3, 5, 5, 6), ('ops', 1, 2, 3, 14)]:
            count = []
            model = SmallModel()
            hook = model.register_forward_pre_hook(lambda module, inputs: count.append(inputs[0].shape[0]))
            _, stats = attack(model, clean, label, method=method, steps=2, views=views, ne=ne, np_=np_, view_batch=2)
            hook.remove()
            self.assertEqual(sum(count), expected)
            self.assertEqual(stats['input_gradient_images'], expected)
        mi, _ = attack(SmallModel(), clean, label, steps=2)
        for ne, np_ in ((0, 5), (5, 0)):
            actual, stats = attack(SmallModel(), clean, label, method='ops', steps=2, ne=ne, np_=np_)
            torch.testing.assert_close(actual, mi, rtol=0, atol=0)
            self.assertEqual(stats['forward_images'], 2)
        unbounded = apply_ops(clean, OPSViewPlan(torch.full_like(clean, .2), (BasicPlan(0),)))
        self.assertGreater(unbounded.max().item(), 1)

    def test_new_local_plans_are_order_independent_and_bsr_microbatch(self):
        clean = torch.linspace(.1, .9, 3 * 21 * 21).reshape(1, 3, 21, 21)
        def values(order, method):
            result = {}
            for identity in order:
                seed = image_seed(20261027, identity)
                if method == 'ops':
                    plans = OPSContext(clean.shape, 8 / 255, 2, 2, 2., seed).step_plans()
                else:
                    plans = [sample_plan(clean.shape, seed + index) for index in range(3)]
                result[identity] = [apply_plan(clean, plan) for plan in plans]
            return result
        for method in ('bsr', 'ops'):
            first, second = values(['a', 'b'], method), values(['b', 'a'], method)
            for identity in ('a', 'b'):
                for left, right in zip(first[identity], second[identity]):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
        plans = [sample_plan(clean.shape, seed) for seed in (10, 11, 12)]
        one = mean_view_gradient(SmallModel(), clean, torch.tensor([0]), plans, 1)
        two = mean_view_gradient(SmallModel(), clean, torch.tensor([0]), plans, 2)
        compare(self, one, two)

    def test_related_schedule_cost_and_complete_block_pairing(self):
        ids = [f'id{i:03}' for i in range(276)]
        blocks, tasks = build_related_tasks(ids)
        self.assertEqual([len(block) for block in blocks], [69] * 4)
        self.assertEqual(len(tasks), 20)
        self.assertEqual(sum(t['planned_gradient_images'] for t in tasks), 342240)
        self.assertEqual([t['configuration'] for t in tasks[:8]], ['ops260', 'bsr200'] * 4)
        self.assertEqual([t['configuration'] for t in tasks[8:]], ['din260', 'sia260', 'bsr260'] * 4)
        plan = {'tasks': tasks, 'blocks': blocks, 'targets': ['dn121', 'vit_b_16'], 'seed': 20261027}
        clean = {(model, identity): {'correct': True} for model in ('rn50', 'dn121', 'vit_b_16') for identity in ids}
        mappings, complete = {}, {}
        for task in tasks:
            is_complete = task['block'] == 1
            complete[task['id']] = is_complete
            chosen = task['selected_ids'] if is_complete else task['selected_ids'][:2]
            mappings[task['id']] = {target: {identity: {'id': identity,
                'success': task['configuration'] == 'ops260', 'adversarial_margin': -float(task['configuration'] == 'ops260'),
                'adversarial_correct': task['configuration'] != 'ops260'} for identity in chosen} for target in plan['targets']}
        summaries, contrasts, matched = related_analysis(plan, clean, mappings, complete)
        self.assertEqual(matched, [1])
        for contrast in contrasts:
            self.assertEqual(contrast['image_count'], 69)
            self.assertFalse(contrast['matrix_complete'])
            self.assertEqual(contrast['estimate'], 1 if contrast['endpoint'] == 'success' else -1)
