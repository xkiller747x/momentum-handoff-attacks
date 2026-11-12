"""Reusable per-view random plans, independent of view microbatch grouping.

SIA follows SIT commit 6d5432d (10 operations). We avoid in-place block
assignment so autograd can differentiate every block, preserving forward math.
"""
from dataclasses import dataclass
import random

import torch
import torch.nn.functional as F

SIA_OPERATIONS = ('resize', 'vertical_shift', 'horizontal_shift', 'vertical_flip',
                  'horizontal_flip', 'rotate180', 'scale', 'add_noise', 'dct', 'dropout2d')


@dataclass(frozen=True)
class ViewPlan:
    method: str
    seed: int
    di_probability: float = .5
    resize_rate: float = 1.1
    sia_choice: int = -1


def make_plans(method, views, rng, di_probability=.5, resize_rate=1.1):
    return [ViewPlan(method, rng.randrange(2**63), di_probability, resize_rate) for _ in range(views)]


def _rand(shape, generator, x):
    return torch.rand(shape, generator=generator, dtype=x.dtype).to(x.device)


def sia_operation(x, choice, rng, generator):
    height, width = x.shape[-2:]
    if choice == 0:
        # Author's intermediate size uses (int(width*.8)+1,int(height*.8)+1).
        small = F.interpolate(x, size=(int(width * .8) + 1, int(height * .8) + 1), mode='bilinear', align_corners=False)
        return F.interpolate(small, size=(height, width), mode='bilinear', align_corners=False).clamp(0, 1)
    if choice == 1:
        return x.roll(rng.randrange(height), dims=2)
    if choice == 2:
        return x.roll(rng.randrange(width), dims=3)
    if choice == 3:
        return x.flip((2,))
    if choice == 4:
        return x.flip((3,))
    if choice == 5:
        return x.rot90(2, (2, 3))
    if choice == 6:
        return x * _rand((1,), generator, x)
    if choice == 7:
        return (x + (_rand(x.shape, generator, x) * 2 - 1) * (16 / 255)).clamp(0, 1)
    if choice == 8:
        import torch_dct
        coefficients = torch_dct.dct_2d(x)
        mask = torch.ones_like(coefficients)
        # Preserve Python -0 slicing in the author's implementation: tiny blocks
        # with int(.4*size)==0 have the corresponding full axis zeroed.
        mask[:, :, -int(height * .4):, :] = 0
        mask[:, :, :, -int(width * .4):] = 0
        return torch_dct.idct_2d(coefficients * mask)
    if choice == 9:
        # dropout2d is one Bernoulli per batch/channel, scaled by 1/(1-p).
        keep = (_rand((x.shape[0], x.shape[1], 1, 1), generator, x) >= .1).to(x.dtype)
        return x * keep / .9
    raise ValueError(f'Unknown SIA operation {choice}')


def apply_plan(x, plan):
    if plan.method == 'bsr':
        from .bsr import BSRPlan, apply_bsr, sample_plan
        return apply_bsr(x, plan if isinstance(plan, BSRPlan) else sample_plan(x.shape, plan.seed))
    if plan.method == 'ops':
        from .ops import apply_ops
        return apply_ops(x, plan)
    rng = random.Random(plan.seed)
    generator = torch.Generator(device='cpu').manual_seed(plan.seed)
    if plan.method == 'mi':
        return x
    if plan.method in {'di', 'di-n'}:
        if rng.random() >= plan.di_probability:
            return x
        size = x.shape[-1]
        enlarged = int(size * plan.resize_rate)
        if enlarged <= size:
            raise ValueError('DI resize_rate must produce an enlarged image')
        resized = rng.randrange(size, enlarged)
        remainder = enlarged - resized
        top, left = rng.randrange(remainder), rng.randrange(remainder)
        small = F.interpolate(x, size=(resized, resized), mode='bilinear', align_corners=False)
        padded = F.pad(small, (left, remainder - left, top, remainder - top))
        return F.interpolate(padded, size=x.shape[-2:], mode='bilinear', align_corners=False)
    if plan.method == 'sia':
        height, width = x.shape[-2:]
        if min(height, width) < 3:
            raise ValueError('SIA 3x3 blocks require both dimensions >=3')
        ys = [0] + sorted(rng.sample(range(1, height), 2)) + [height]
        xs = [0] + sorted(rng.sample(range(1, width), 2)) + [width]
        rows = []
        for row in range(3):
            blocks = []
            for col in range(3):
                choice = plan.sia_choice if plan.sia_choice >= 0 else rng.randrange(10)
                block = x[:, :, ys[row]:ys[row + 1], xs[col]:xs[col + 1]]
                blocks.append(sia_operation(block, choice, rng, generator))
            rows.append(torch.cat(blocks, dim=3))
        return torch.cat(rows, dim=2)
    raise ValueError(f'Unknown method {plan.method}')
