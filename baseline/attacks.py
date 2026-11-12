"""Untargeted FP32 MI, momentum DI, DI-N and author-profile SIA."""
import random
import time

import torch
import torch.nn.functional as F

from .transforms import apply_plan, make_plans


def mean_view_gradient(model, x, labels, plans, view_batch=1, counter=None):
    """Average raw d(loss)/d(original input); normalize only after this mean.

    Each microbatch CE is summed then divided by the TOTAL view count. Thus a
    shorter final microbatch receives exactly its number of views' weight.
    x is one image; grouping views is the only batch dimension used here.
    """
    if x.shape[0] != 1 or not plans or view_batch < 1:
        raise ValueError('Expected one image, nonempty plans and positive view_batch')
    total = torch.zeros_like(x)
    for start in range(0, len(plans), view_batch):
        leaf = x.detach().requires_grad_(True)
        batch = torch.cat([apply_plan(leaf, p) for p in plans[start:start + view_batch]], dim=0)
        logits = model(batch)
        loss = F.cross_entropy(logits, labels.repeat(batch.shape[0]), reduction='sum') / len(plans)
        gradient = torch.autograd.grad(loss, leaf, only_inputs=True)[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite input gradient')
        total.add_(gradient.detach())
        if counter is not None:
            count, height, width = batch.shape[0], batch.shape[-2], batch.shape[-1]
            counter['forward_images'] += count
            counter['input_gradient_images'] += count
            counter['forward_pixels'] += count * height * width
            counter['input_gradient_pixels'] += count * height * width
            key = f'{height}x{width}'
            counter['resolution_counts'][key] = counter['resolution_counts'].get(key, 0) + count
            if any(plan.method in {'bsr', 'ops'} for plan in plans[start:start + view_batch]):
                counter['transformed_channel_elements'] = counter.get('transformed_channel_elements', 0) + batch.numel()
                counter['transformed_out_of_range_elements'] = counter.get('transformed_out_of_range_elements', 0) + int(((batch < 0) | (batch > 1)).sum().item())
                counter['transformed_min'] = min(counter.get('transformed_min', float('inf')), float(batch.min().item()))
                counter['transformed_max'] = max(counter.get('transformed_max', -float('inf')), float(batch.max().item()))
    return total


def attack(model, clean, labels, method='mi', steps=10, views=1, epsilon_bytes=16,
           alpha_mult=1., seed=0, view_batch=1, decay=1., di_probability=.5,
           resize_rate=1.1, fixed_plans=None, ne=5, np_=5, beta=2., profile=None):
    if steps < 1 or views < 1 or view_batch < 1 or not 0 <= epsilon_bytes <= 255 or alpha_mult <= 0:
        raise ValueError('Invalid attack parameters')
    if not 0 <= di_probability <= 1 or resize_rate <= 1:
        raise ValueError('Invalid DI parameters')
    if method in {'mi', 'di'} and views != 1:
        raise ValueError('MI and momentum DI require --views 1; use di-n for multiple DI views')
    if method in {'ops', 'ops-refresh'} and views != 1:
        raise ValueError('OPS requires --views 1 for its base view; use --ne/--np for sampling')
    if method not in {'mi', 'di', 'di-n', 'sia', 'bsr', 'ops', 'ops-refresh'}:
        raise ValueError(f'Unknown attack {method}')
    if method in {'bsr', 'ops', 'ops-refresh'}:
        expected_profile = {'bsr': 'bsr-transferattack-3blocks', 'ops': 'ops-author-31ops-93pool',
                            'ops-refresh': 'ops-author-noiserefresh'}[method]
        if profile is not None and profile != expected_profile:
            raise ValueError(f'{method} requires profile {expected_profile}')
    clean = clean.detach().to(dtype=torch.float32)
    if clean.shape[0] != 1 or clean.min() < 0 or clean.max() > 1:
        raise ValueError('Expected a single clean image in [0,1]')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rng = random.Random(seed)
    ops_context = None
    epsilon, alpha = epsilon_bytes / 255, alpha_mult * epsilon_bytes / (255 * steps)
    momentum, adv = torch.zeros_like(clean), clean.clone()
    stats = {'forward_images': 0, 'input_gradient_images': 0, 'forward_pixels': 0,
             'input_gradient_pixels': 0, 'resolution_counts': {}, 'zero_grad_count': 0,
             'near_zero_grad_count': 0, 'normalization_floor': 1e-12}
    cuda = clean.device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize(clean.device)
        torch.cuda.reset_peak_memory_stats(clean.device)
    start = time.perf_counter()
    if method == 'ops':
        from .ops import OPSContext
        pool_start = time.perf_counter()
        ops_context = OPSContext(clean.shape, epsilon_bytes / 255, ne, np_, beta, seed)
        stats['pool_init_seconds'] = time.perf_counter() - pool_start
    if method == 'ops-refresh':
        from .ops_refresh import OPSRefreshContext
        pool_start = time.perf_counter()
        ops_context = OPSRefreshContext(clean.shape, epsilon_bytes / 255, ne, np_, beta, seed)
        stats['pool_init_seconds'] = time.perf_counter() - pool_start
    for step in range(steps):
        plans = (fixed_plans[step] if fixed_plans is not None else ops_context.step_plans() if ops_context is not None
                 else make_plans(method, views, rng, di_probability, resize_rate))
        expected_views = (1 + ne * np_) if method in {'ops', 'ops-refresh'} else views
        if len(plans) != expected_views:
            raise ValueError('Fixed plan view count mismatch')
        gradient = mean_view_gradient(model, adv, labels, plans, view_batch, stats)
        denominator = gradient.abs().mean(dim=(1, 2, 3), keepdim=True)
        stats['zero_grad_count'] += int((denominator == 0).sum().item())
        stats['near_zero_grad_count'] += int((denominator < 1e-12).sum().item())
        momentum = decay * momentum + gradient / denominator.clamp_min(1e-12)
        adv = adv + alpha * momentum.sign()
        adv = torch.maximum(torch.minimum(adv, clean + epsilon), clean - epsilon).clamp(0, 1).detach()
    if cuda:
        torch.cuda.synchronize(clean.device)
    stats['elapsed_seconds'] = time.perf_counter() - start
    stats['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(clean.device) if cuda else 0
    stats['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(clean.device) if cuda else 0
    if stats.get('transformed_channel_elements'):
        stats['transformed_out_of_range_fraction'] = stats['transformed_out_of_range_elements'] / stats['transformed_channel_elements']
    if method == 'ops-refresh':
        stats.update(ops_context.lifecycle_record())
    return adv, stats
