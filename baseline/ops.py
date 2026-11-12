"""Author OPS 31 basic operators, 93 composites and fixed neighborhood pool.

All randomness is materialized into reusable plans. No neighbor/view clipping.
Composition [a,b,c] executes a(b(c(x))), matching author's functools.reduce.
"""
from dataclasses import dataclass
import random

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate

PROFILE = 'ops-author-31ops-93pool'
ANGLES = (5, -5, 15, -15, 45, -45, 90, -90, 180)
RATES = (1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9)
RATIOS = (.25, .5, .75, 1., 1.25, 1.5)
BASIC_OPERATORS = ('identity', 'vertical_flip', 'horizontal_flip', 'vertical_shift', 'horizontal_shift') + tuple(
    f'rotate{angle}' for angle in ANGLES) + tuple(f'divide{scale}' for scale in range(2, 9)) + tuple(f'di{rate}' for rate in RATES)


@dataclass(frozen=True)
class BasicPlan:
    index: int
    parameters: tuple = ()


@dataclass(frozen=True)
class OPSViewPlan:
    noise: object
    operators: tuple
    pool_index: int = -1
    method: str = 'ops'


def plan_basic(index, shape, rng):
    if index == 3:
        return BasicPlan(index, (rng.randrange(shape[-2]),))
    if index == 4:
        return BasicPlan(index, (rng.randrange(shape[-1]),))
    if index >= 21:
        size = shape[-1]
        enlarged = int(size * RATES[index - 21])
        if enlarged <= size:
            raise ValueError('OPS DI rate produces no enlargement at this image size')
        resized = rng.randrange(size, enlarged)
        remainder = enlarged - resized
        return BasicPlan(index, (resized, enlarged, rng.randrange(remainder), rng.randrange(remainder)))
    return BasicPlan(index)


def apply_basic(x, plan):
    index = plan.index
    if index == 0:
        return x
    if index == 1:
        return x.flip((2,))
    if index == 2:
        return x.flip((3,))
    if index == 3:
        return x.roll(plan.parameters[0], dims=2)
    if index == 4:
        return x.roll(plan.parameters[0], dims=3)
    if 5 <= index < 14:
        return rotate(x, ANGLES[index - 5])  # upstream default NEAREST, fill0
    if 14 <= index < 21:
        return x / (index - 12)  # integer2..8, not a scale exponent
    if 21 <= index < 31:
        resized, enlarged, top, left = plan.parameters
        small = F.interpolate(x, size=(resized, resized), mode='bilinear', align_corners=False)
        remainder = enlarged - resized
        padded = F.pad(small, (left, remainder - left, top, remainder - top))
        return F.interpolate(padded, size=(x.shape[-1], x.shape[-1]), mode='bilinear', align_corners=False)
    raise ValueError(f'Unknown OPS basic operator {index}')


def apply_ops(x, plan):
    value = x if plan.noise is None else x + plan.noise.to(device=x.device, dtype=x.dtype)
    for operation in reversed(plan.operators):
        value = apply_basic(value, operation)
    return value


def composite_pool(rng):
    return [tuple(rng.choices(range(31), k=level)) for level in (2, 3, 4) for _ in range(31)]


class OPSContext:
    def __init__(self, shape, epsilon, ne, np_, beta, seed):
        if ne < 0 or np_ < 0 or np_ > 93 or beta < 0:
            raise ValueError('OPS requires Ne>=0, 0<=Np<=93 and beta>=0')
        self.shape, self.ne, self.np = tuple(shape), ne, np_
        self.rng = random.Random(seed)
        self.pool = []
        if ne * np_:
            generator = torch.Generator(device='cpu').manual_seed(seed)
            for ratio in RATIOS:
                radius = beta * epsilon * ratio
                noise = torch.empty((ne, *shape[1:]), dtype=torch.float32).uniform_(-radius, radius, generator=generator)
                self.pool.extend(noise[index:index + 1] for index in range(ne))

    def step_plans(self):
        plans = [OPSViewPlan(None, ())]  # the author's untransformed base gradient
        if not self.ne * self.np:
            return plans
        for pool_index in self.rng.sample(range(len(self.pool)), self.ne):
            # Rebuild level2/3/4 pools for EACH selected neighbor, every iteration.
            pool = composite_pool(self.rng)
            for composite in self.rng.sample(pool, self.np):
                # Execute inner operators first when sampling size-dependent choices.
                shape, reverse_operations = self.shape, []
                for index in reversed(composite):
                    operation = plan_basic(index, shape, self.rng)
                    reverse_operations.append(operation)
                    if index >= 21:
                        shape = (*shape[:-2], shape[-1], shape[-1])
                plans.append(OPSViewPlan(self.pool[pool_index], tuple(reversed(reverse_operations)), pool_index))
        return plans
