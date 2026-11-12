"""Explicit strict opt-in; never changes the default numeric profile."""
import os
import torch


def configure_strict(args):
    if not getattr(args,'deterministic',False):return
    if str(getattr(args,'device','cpu')).startswith('cuda'):
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
            raise ValueError('Strict CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) set before process startup; no environment fallback is applied')
        if torch.cuda.is_initialized():raise ValueError('Strict CUDA must be configured before CUDA initialization')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
