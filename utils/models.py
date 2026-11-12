"""Explicit torchvision weights, normalization only, canonical224 inputs."""
import os
from pathlib import Path

import torch
from torch import nn

from .io import sha256

ROOT = Path(__file__).resolve().parents[1]
TORCH_HOME = ROOT / 'data' / 'models' / 'torch'
os.environ['TORCH_HOME'] = str(TORCH_HOME)


class NormalizeModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


class WindowAdaptiveAvgPool2d(nn.Module):
    """AdaptiveAvgPool2d via explicit windows (PyTorch floor/ceil bounds).

    Same math as nn.AdaptiveAvgPool2d; built from slices and mean so its CUDA backward
    is deterministic (adaptive_avg_pool2d_backward_cuda is not). Forward agrees up to
    float summation order, not bitwise.
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = tuple(output_size)

    def forward(self, x):
        height, width = x.shape[-2:]
        out_h, out_w = self.output_size
        rows = []
        for i in range(out_h):
            top, bottom = (i * height) // out_h, -((-(i + 1) * height) // out_h)
            cols = [x[..., top:bottom, (j * width) // out_w:-((-(j + 1) * width) // out_w)].mean((-2, -1))
                    for j in range(out_w)]
            rows.append(torch.stack(cols, -1))
        return torch.stack(rows, -2)


def load_model(name, device):
    import torchvision
    from torchvision import models
    specifications = {
        'rn50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
        'rn50_v1': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
        'dn121': (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
        'vit_b_16': (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1),
        'vgg16_bn': (models.vgg16_bn, models.VGG16_BN_Weights.IMAGENET1K_V1),
        # Pretrained Inception-v3 uses transform_input=True and ships AuxLogits weights.
        'inc_v3': (lambda weights=None: models.inception_v3(weights=None, aux_logits=True,
                                                            transform_input=True, init_weights=False),
                   models.Inception_V3_Weights.IMAGENET1K_V1),
    }
    if name not in specifications:
        raise ValueError(f'Model must be one of {list(specifications)}')
    constructor, weights = specifications[name]
    checkpoint = TORCH_HOME / 'hub' / 'checkpoints' / weights.url.rsplit('/', 1)[-1]
    if not checkpoint.exists():
        raise FileNotFoundError(f'Pre-download official weights to {checkpoint} from {weights.url}')
    # Loading local state explicitly prevents hidden downloads and records the actual file.
    network = constructor(weights=None)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    # torchvision's DenseNet legacy checkpoint key migration is normally applied by
    # its download helper; use the same migration for this explicit local loading.
    if name == 'dn121':
        import re
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.([12]\.(?:weight|bias|running_mean|running_var))$')
        state = {pattern.sub(r'\1\2', key): value for key, value in state.items()}
    network.load_state_dict(state)
    if name == 'vgg16_bn':
        network.avgpool = WindowAdaptiveAvgPool2d(network.avgpool.output_size)
    transform = weights.transforms()
    wrapped = NormalizeModel(network, transform.mean, transform.std).eval().float().to(device)
    for parameter in wrapped.parameters():
        parameter.requires_grad_(False)
    identity = {'name': name, 'weights': str(weights), 'checkpoint': str(checkpoint),
                'checkpoint_sha256': sha256(checkpoint), 'url': weights.url,
                'torch_version': torch.__version__, 'torchvision_version': torchvision.__version__,
                'input_size': [224, 224], 'mean': transform.mean, 'std': transform.std,
                'preprocessing': 'normalization only on shared canonical224 uint8 PNG',
                'precision': 'FP32', 'eval': True, 'parameters_frozen': True,
                'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                'autocast': False}
    if name == 'vgg16_bn':
        identity['avgpool'] = 'WindowAdaptiveAvgPool2d((7,7)): explicit-window equivalent of AdaptiveAvgPool2d for deterministic CUDA backward'
    return wrapped, identity
