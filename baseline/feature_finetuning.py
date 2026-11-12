"""Batch-one FFT/AaF author-code profiles; independent legacy NumPy stream.

All forwards are complete source forwards. Guidance backward ends at layer3;
fine-tuning backward ends at the input. No parameter gradients are accumulated.
"""
import numpy as np
import time
import torch
import torch.nn.functional as F

METHODS = ('fft', 'aaf', 'logit-fft', 'logit-aaf', 'ila', 'logit-ila')


def gaussian(channels, image):
    # Author scipy.stats.norm.pdf(linspace(-3,3,5)), float32 before convolution.
    points = np.linspace(-3, 3, 5)
    one = np.exp(-points ** 2 / 2) / np.sqrt(2 * np.pi)
    kernel = np.outer(one, one)
    kernel = (kernel / kernel.sum()).astype(np.float32)
    return torch.from_numpy(np.repeat(kernel[None, None], channels, axis=0)).to(image)


def patch_mask(shape, patch_size, prob, rng):
    """Author RPA storage/concatenation, including untouched bottom/right edges."""
    mask = np.ones(shape)
    n0, h0, w0, c0 = mask.shape
    ph, pw = h0 // patch_size[0], w0 // patch_size[1]
    x = mask[:, :ph * patch_size[0], :pw * patch_size[1]]
    n, h, w, c = x.shape
    view = np.lib.stride_tricks.as_strided(x,
        shape=(n, ph, pw, patch_size[0], patch_size[1], c),
        strides=(x.strides[0], x.strides[1] * patch_size[0],
                 x.strides[2] * patch_size[0], *x.strides[1:]))
    length = view.shape[1] * view.shape[2] * view.shape[-1]
    selected = rng.choice(length, int(length * (1 - prob)), replace=False)
    for i in range(view.shape[1]):
        for j in range(view.shape[2]):
            for k in range(view.shape[-1]):
                if i * view.shape[2] * view.shape[-1] + j * view.shape[-1] + k in selected:
                    view[:, i, j, :, :, k] = rng.uniform(0, 1, (n, view.shape[3], view.shape[4]))
    result = np.concatenate(view, axis=0)
    result = np.concatenate(result, axis=1)
    result = np.concatenate(result, axis=1).reshape((n, h, w, c))
    mask[:, :ph * patch_size[0], :pw * patch_size[1]] = result
    return mask


def sample_mask(shape, aaf, index, rng):
    n, c, h, w = shape
    if not aaf:
        return rng.binomial(1, .7, size=(n, c, h, w))
    if index % 4 == 0:
        keep = rng.binomial(1, .7, size=(n, h, w, c))
        uniform = rng.uniform(0, 1, size=(n, h, w, c))
        mask = np.where(keep == 1, 1, uniform)
    else:
        size = (3, 5, 7)[index % 4 - 1]
        mask = patch_mask((n, h, w, c), (size, size), .7, rng)
    return mask.transpose(0, 3, 1, 2)


def masked_input(image, normalize, aaf, index, rng):
    sampled = sample_mask(image.shape, aaf, index, rng)
    mask = torch.tensor(sampled, dtype=torch.float32 if aaf else None, device=image.device)
    if not aaf:
        return normalize(image).clone() * mask
    return normalize(image.clone() * mask)


def diversity(image, rng):
    if image.shape[-2:] != (299, 299):
        raise ValueError('Author fine DI requires299x299')
    rnd = rng.randint(270, 299, size=1)[0]
    remaining = 299 - rnd
    top = rng.randint(0, remaining, size=1)[0]
    left = rng.randint(0, remaining, size=1)[0]
    probability = rng.rand(1)
    if probability <= .7:
        return F.pad(F.interpolate(image, size=(int(rnd), int(rnd)), mode='nearest'),
                     (int(left), int(remaining-left), int(top), int(remaining-top)))
    return image


SOURCE_LAYERS = {'rn50_v1': 'layer3', 'dn121': 'features.transition3', 'vgg16_bn': 'features.33',
                 'inc_v3': 'Mixed_6b'}


class Layer3Source(torch.nn.Module):
    """Read-only middle-layer tap on a normalized torchvision wrapper.

    Default is ResNet layer3; DenseNet uses transition3 and VGG16-BN uses features[33]
    (block-4 output after Conv4_3 and its pool), the author code's l3 taps.
    """
    def __init__(self, wrapped, layer='layer3'):
        super().__init__()
        self.wrapped = wrapped
        self.value = None
        self.handle = wrapped.model.get_submodule(layer).register_forward_hook(self._capture)
        self.eval()

    def _capture(self, _module, _inputs, output):
        self.value = output

    def normalize(self, image):
        return (image - self.wrapped.mean) / self.wrapped.std

    def forward(self, image, normalized=False):
        return self.wrapped.model(image) if normalized else self.wrapped(image)

    def close(self):
        self.handle.remove()
        self.value = None


def guidance(model, image, label, aaf, rng, count=30, after_gradient=None):
    total = None
    for index in range(count):
        x = image.detach().requires_grad_(True)
        logits = model(masked_input(x, model.normalize, aaf, index, rng), normalized=True)
        gradient = torch.autograd.grad(logits[:, label].sum(), model.value)[0].detach()
        if after_gradient:
            after_gradient('feature')
        total = torch.zeros_like(gradient) if total is None else total
        total += gradient
    if aaf:
        total = F.conv2d(total, gaussian(total.shape[1], total), padding=2, groups=total.shape[1])
    std = total.std(correction=1)
    if not torch.isfinite(total).all() or not torch.isfinite(std) or std == 0:
        raise ValueError('Degenerate/nonfinite author guidance std; no substitute direction')
    return (total / std).detach()


PROTOCOL_DEFAULT = {'alpha_bytes': None, 'di': 'author', 'momentum_norm': 'none', 'init_momentum': None,
                    'conflict_reset': False, 'conflict_samples': 1, 'joint_weight': None,
                    'fine_steps': None}


def finetune(model, clean, initial, target, label, method, epsilon_bytes, rng,
             after_gradient=None, observer=None, mask_count=30, protocol=None):
    """mask_count is exposed for CPU oracle tests; CLI always30.

    protocol is a diagnostic ablation of the fine-tuning optimizer only; the default
    (alpha epsilon/16, author DI, raw momentum) is the author profile unchanged.
    di='rdi' uses the base attack's RDI with a torch generator seeded from one
    extra rng draw after the mask draws, so feature/logit pairs stay matched.
    init_momentum continues from a saved base momentum instead of restarting at zero.
    conflict_reset (needs init_momentum) zeroes, once at the first fine step, only the
    inherited coordinates whose sign opposes that step's gradient; agreeing ones are kept.
    Re-deciding every step would make the update sign(current gradient), i.e. no momentum.
    conflict_samples>1 judges conflict against the mean of that many extra DI-sampled,
    TI-smoothed, L1-normalized objective gradients at the start image (counted as input B).
    joint_weight=w (needs a feature method and l1) keeps the logit objective instead of
    switching: from one forward, the logit and guidance input gradients are each TI-smoothed
    and L1-normalized, then combined as g_logit + w*g_guide (two input B per step).
    fine_steps overrides the author step count (10, or 15 for AaF), e.g. for an earlier switch
    at a fixed total budget; AaF averaging still starts after the fifth step.
    """
    protocol = {**PROTOCOL_DEFAULT, **(protocol or {})}
    if protocol['di'] not in ('author', 'rdi') or protocol['momentum_norm'] not in ('none', 'l1'):
        raise ValueError('Unknown fine-tuning protocol')
    if protocol['conflict_reset'] and protocol['init_momentum'] is None:
        raise ValueError('conflict_reset needs an inherited momentum')
    if int(protocol['conflict_samples']) != protocol['conflict_samples'] or protocol['conflict_samples'] < 1             or (protocol['conflict_samples'] > 1 and not protocol['conflict_reset']):
        raise ValueError('conflict_samples must be a positive integer used with conflict_reset')
    joint = protocol['joint_weight']
    if joint is not None and (not np.isfinite(joint) or joint < 0 or method.startswith('logit-')
                              or protocol['momentum_norm'] != 'l1' or protocol['conflict_reset']):
        raise ValueError('joint_weight needs a finite nonnegative weight, a feature method and l1, no conflict reset')
    steps = protocol['fine_steps']
    if steps is not None and (int(steps) != steps or steps < 1):
        raise ValueError('fine_steps must be a positive integer')
    if method not in METHODS or clean.shape != initial.shape or clean.shape != (1, 3, 299, 299):
        raise ValueError('Fine-tuning requires known method and batch1 RGB299')
    if not np.isfinite(epsilon_bytes) or epsilon_bytes < 0:
        raise ValueError('Finite nonnegative epsilon required')
    aaf = method.endswith('aaf')
    feature = not method.startswith('logit-')
    guide = None
    if method == 'ila':
        # ILA projection loss (F(x)-F(clean))·(F(x0)-F(clean)) has the same input gradient as
        # (guide·F(x)) with guide = F(x0)-F(clean); two no-grad reference forwards, no masks.
        with torch.no_grad():
            model(clean)
            reference = model.value.detach().clone()
            model(initial)
            guide = (model.value - reference).detach()
        if observer:
            observer('guidance', {'original': reference, 'target': guide + reference, 'combined': guide})
    elif method == 'logit-ila':
        pass
    elif feature:
        original = guidance(model, clean, label, aaf, rng, mask_count, after_gradient)
        desired = guidance(model, initial, target, aaf, rng, mask_count, after_gradient)
        guide = desired - .2 * original
        if observer:
            observer('guidance', {'original': original, 'target': desired, 'combined': guide})
    else:
        start = time.perf_counter()
        for _class in range(2):
            for index in range(mask_count):
                sample_mask(clean.shape, aaf, index, rng)
        if observer:
            observer('matching_rng_advance', {'mask_draws':2*mask_count,
                                              'seconds':time.perf_counter()-start})
    image = initial.detach().clone()
    momentum = 0 if protocol['init_momentum'] is None else protocol['init_momentum'].detach().clone()
    eta_previous = 0
    accumulated = 0
    epsilon = epsilon_bytes / 255
    alpha = epsilon / 16 if protocol['alpha_bytes'] is None else protocol['alpha_bytes'] / 255
    generator = None
    if protocol['di'] == 'rdi':
        from baseline.targeted import rdi
        generator = torch.Generator(device='cpu').manual_seed(int(rng.randint(0, 2**31 - 1)))
    judge = None
    if protocol['conflict_reset'] and protocol['conflict_samples'] > 1:
        judge = 0
        for _sample in range(protocol['conflict_samples']):
            x = image.detach().requires_grad_(True)
            logits = model(diversity(x, rng) if generator is None else rdi(x, generator))
            loss = (guide * model.value).sum() if feature else logits[:, target].sum()
            sample = torch.autograd.grad(loss, x)[0]
            if after_gradient:
                after_gradient('input')
            sample = F.conv2d(sample, gaussian(3, sample), padding=2, groups=3)
            norm = sample.abs().sum((1, 2, 3), keepdim=True)
            judge = judge + sample / torch.where(norm > 0, norm, torch.ones_like(norm))
        judge = judge / protocol['conflict_samples']
    for epoch in range(int(steps) if steps is not None else 15 if aaf else 10):
        x = image.detach().requires_grad_(True)
        logits = model(diversity(x, rng) if generator is None else rdi(x, generator))
        loss = (guide * model.value).sum() if feature else logits[:, target].sum()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite fine-tuning objective')
        if joint is not None:
            parts = []
            for objective in (logits[:, target].sum(), loss):
                part = torch.autograd.grad(objective, x, retain_graph=not parts)[0]
                if after_gradient:
                    after_gradient('input')
                part = F.conv2d(part, gaussian(3, part), padding=2, groups=3)
                norm = part.abs().sum((1, 2, 3), keepdim=True)
                parts.append(part / torch.where(norm > 0, norm, torch.ones_like(norm)))
            gradient = parts[0] + joint * parts[1]
            if not torch.isfinite(gradient).all():
                raise ValueError('Nonfinite fine-tuning input gradient')
        else:
            gradient = torch.autograd.grad(loss, x)[0]
            if after_gradient:
                after_gradient('input')
            if not torch.isfinite(gradient).all():
                raise ValueError('Nonfinite fine-tuning input gradient')
            gradient = F.conv2d(gradient, gaussian(3, gradient), padding=2, groups=3)
        if protocol['momentum_norm'] == 'l1':
            norm = gradient.abs().sum((1, 2, 3), keepdim=True)
            gradient = gradient / torch.where(norm > 0, norm, torch.ones_like(norm))
        if protocol['conflict_reset'] and epoch == 0:
            direction = gradient if judge is None else judge
            conflict = (torch.sign(momentum) != torch.sign(direction)) & (direction != 0)
            step_conflict = (torch.sign(momentum) != torch.sign(gradient)) & (gradient != 0)
            momentum = momentum.masked_fill(conflict, 0)
            if observer:
                observer('conflict_reset', {'fraction': float(conflict.float().mean()),
                                            'step_gradient_fraction': float(step_conflict.float().mean()),
                                            'samples': protocol['conflict_samples'],
                                            'nonzero_gradient_fraction': float((gradient != 0).float().mean())})
        momentum = momentum + gradient  # Author fine momentum has NO L1 normalization.
        proposal = x.detach() + alpha * momentum.sign()
        eta = proposal - clean
        if aaf and epoch >= 5:
            eta = (eta + accumulated * eta_previous) / (accumulated + 1)
            accumulated = .8 * (accumulated + 1)
            eta_previous = eta  # Deliberately preclip feedback history.
        image = (clean + eta.clamp(-epsilon, epsilon)).clamp(0, 1).detach()
        if observer:
            observer('step', {'step': epoch + 1, 'image': image, 'momentum': momentum.detach(),
                             'eta_previous': eta_previous, 'loss': loss.detach()})
    return image
