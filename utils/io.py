"""Lossless image IO and deterministic run records."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .metrics import perturbation_metrics


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def append_jsonl(path, value):
    with open(path, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def read_pixels(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB'), dtype=np.uint8, copy=True)


def image_tensor(path, device='cpu'):
    return torch.from_numpy(read_pixels(path)).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255


def save_adversarial(path, adv, clean_pixels, epsilon_bytes):
    """Round to nearest integer, enforce integer L-inf bounds, and read back PNG."""
    floating = adv.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    if not np.isfinite(floating).all():
        raise ValueError('Nonfinite adversarial pixels')
    # Integer bounds also support a fractional epsilon without exceeding that budget.
    low = np.ceil(clean_pixels.astype(np.float64) - epsilon_bytes).clip(0, 255)
    high = np.floor(clean_pixels.astype(np.float64) + epsilon_bytes).clip(0, 255)
    encoded = np.minimum(np.maximum(np.rint(floating * 255), low), high).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    Image.fromarray(encoded).save(temp, format='PNG')
    temp.replace(path)
    decoded = read_pixels(path)
    byte_difference = int(np.abs(decoded.astype(np.int16) - clean_pixels.astype(np.int16)).max())
    if byte_difference > epsilon_bytes + 1e-8 or not np.array_equal(encoded, decoded):
        raise RuntimeError('PNG roundtrip/budget check failed')
    return {'png_max_byte_difference': byte_difference,
            'png_roundtrip_max_byte_error': int(np.abs(encoded.astype(np.int16) - decoded.astype(np.int16)).max()),
            'float_to_png_max_abs_error': float(np.abs(floating - decoded.astype(np.float32) / 255).max()),
            'float_max_byte_difference': float(np.abs(floating * 255 - clean_pixels.astype(np.float32)).max()),
            'png_sha256': sha256(path),
            **perturbation_metrics(clean_pixels, decoded, epsilon_bytes)}
