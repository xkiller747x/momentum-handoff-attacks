"""TransferAttack 3-block BSR: strip shuffle, rotate, orthogonal shuffle."""
from dataclasses import dataclass
import random

import numpy as np
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate
import torch

PROFILE = 'bsr-transferattack-3blocks'


def lengths_from_uniform(length, uniforms):
    # np.random.uniform(2,size=3) in legacy NumPy means 2+(1-2)*U.
    weights = 2 - np.asarray(uniforms, dtype=np.float64)
    lengths = np.round(weights / weights.sum() * length).astype(np.int32)
    lengths[lengths.argmax()] += length - lengths.sum()
    return tuple(int(n) for n in lengths)


@dataclass(frozen=True)
class BSRStrip:
    angle: float
    lengths: tuple
    permutation: tuple


@dataclass(frozen=True)
class BSRPlan:
    axis: int
    lengths: tuple
    permutation: tuple
    strips: tuple
    method: str = 'bsr'


def sample_plan(shape, seed):
    rng = random.Random(seed)
    numpy_rng = np.random.RandomState(seed % (2**32))
    axes = [2, 3]
    rng.shuffle(axes)
    first = lengths_from_uniform(shape[axes[0]], numpy_rng.random_sample(3))
    permutation = list(range(3))
    rng.shuffle(permutation)
    strips = []
    for _ in permutation:
        angle = rng.uniform(-24, 24)
        lengths = lengths_from_uniform(shape[axes[1]], numpy_rng.random_sample(3))
        second_permutation = list(range(3))
        rng.shuffle(second_permutation)
        strips.append(BSRStrip(angle, lengths, tuple(second_permutation)))
    return BSRPlan(axes[0], first, tuple(permutation), tuple(strips))


def apply_bsr(x, plan):
    axis, other = plan.axis, 5 - plan.axis
    source_strips = x.split(plan.lengths, dim=axis)
    result = []
    for source_index, operation in zip(plan.permutation, plan.strips):
        strip = rotate(source_strips[source_index], operation.angle,
                       interpolation=InterpolationMode.BILINEAR, fill=0)
        pieces = strip.split(operation.lengths, dim=other)
        result.append(torch.cat([pieces[index] for index in operation.permutation], dim=other))
    return torch.cat(result, dim=axis)
