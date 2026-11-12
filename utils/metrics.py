"""Prespecified scalar endpoints and image-level paired resampling."""
import hashlib

import numpy as np
import torch
import torch.nn.functional as F


def image_seed(seed, identity):
    """Existing seed:imageID stream; deliberately excludes alpha and method."""
    return int.from_bytes(hashlib.sha256(f'{seed}:{identity}'.encode()).digest()[:8], 'big') % (2**63)


def perturbation_metrics(clean_pixels, adversarial_pixels, epsilon_bytes):
    delta = adversarial_pixels.astype(np.float64) - clean_pixels.astype(np.float64)
    absolute = np.abs(delta)
    return {'png_mean_abs_bytes': float(absolute.mean()),
            'png_mean_abs_unit': float(absolute.mean() / 255),
            'png_l2_unit': float(np.linalg.norm(delta.ravel() / 255)),
            'png_boundary_element_fraction': float(np.mean(absolute == epsilon_bytes)),
            'png_channel_element_count': int(delta.size)}


def logit_metrics(logits, label):
    """One image's true-class margin and CE, using its existing forward output."""
    if logits.shape[0] != 1:
        raise ValueError('Expected exactly one image')
    other = logits.clone()
    other[0, label] = -torch.inf
    margin = logits[0, label] - other.max(dim=1).values[0]
    ce = F.cross_entropy(logits, torch.tensor([label], device=logits.device))
    return {'margin': float(margin.item()), 'ce': float(ce.item())}


def seed_average(seed_to_id_values, seeds, ids):
    """Averaging is within image, never treating seeds as extra images."""
    missing = [seed for seed in seeds if seed not in seed_to_id_values or
               any(identity not in seed_to_id_values[seed] for identity in ids)]
    if missing:
        raise ValueError(f'Missing seeds/images: {missing}')
    return np.array([[seed_to_id_values[seed][identity] for identity in ids] for seed in seeds], dtype=np.float64).mean(axis=0)


def paired_bootstrap(values, resamples=10000, seed=20261030):
    """CI of a per-image contrast. Same n/seed gives identical sampled ID indices.

    The contrast is formed per ID before resampling, so all contrast cells use
    precisely the same paired bootstrap IDs. Chunking only limits temporary RAM.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite, nonempty per-image contrast')
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, resamples, 250):
        indices = rng.integers(0, len(values), size=(min(250, resamples - start), len(values)))
        estimates.extend(values[indices].mean(axis=1).tolist())
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {'estimate': float(values.mean()), 'ci95_low': float(low), 'ci95_high': float(high),
            'ci95_half_width': float((high - low) / 2), 'image_count': len(values),
            'resamples': resamples, 'bootstrap_seed': seed}
