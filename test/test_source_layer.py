"""Second-source support: the middle-layer tap is chosen by source name."""
import unittest

import numpy as np
import torch
from torch import nn

from baseline.feature_finetuning import SOURCE_LAYERS, Layer3Source, finetune
from test_feature_finetuning import Toy, images, local_model
from utils.models import TORCH_HOME, NormalizeModel


class DenseToy(nn.Module):
    """Minimal DenseNet-shaped module: features.{conv0,transition3,denseblock4} then fc."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(3, 4, 3, stride=16, padding=1))
        self.features.add_module('transition3', nn.Sequential(nn.Conv2d(4, 4, 1), nn.Tanh()))
        self.features.add_module('denseblock4', nn.Sequential(nn.Conv2d(4, 4, 1), nn.Tanh()))
        self.classifier = nn.Linear(4, 3)
        generator = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand(p.shape, generator=generator) * .8 - .4)
        self.eval()

    def forward(self, x):
        return self.classifier(self.features(x).mean((2, 3)))


def dense_source():
    wrapped = NormalizeModel(DenseToy(), [.485, .456, .406], [.229, .224, .225]).eval()
    for p in wrapped.parameters():
        p.requires_grad_(False)
    return Layer3Source(wrapped, SOURCE_LAYERS['dn121'])


class SourceLayerTests(unittest.TestCase):
    def test_layer_map_matches_author_taps(self):
        self.assertEqual(SOURCE_LAYERS, {'rn50_v1': 'layer3', 'dn121': 'features.transition3',
                                         'vgg16_bn': 'features.33', 'inc_v3': 'Mixed_6b'})

    @unittest.skipUnless((TORCH_HOME / 'hub' / 'checkpoints' / 'inception_v3_google-0cc3c7bd.pth').exists(),
                         'official Inception-v3 weights not downloaded')
    def test_inception_v3_loads_official_checkpoint_with_transform_input(self):
        from utils.models import load_model
        wrapped, identity = load_model('inc_v3', torch.device('cpu'))
        self.assertTrue(wrapped.model.transform_input)
        self.assertEqual(identity['checkpoint_sha256'][:8], '0cc3c7bd')
        with torch.no_grad():
            out = wrapped(torch.rand(1, 3, 299, 299, generator=torch.Generator().manual_seed(1)))
        self.assertEqual(tuple(out.shape), (1, 1000))
        self.assertIsNotNone(wrapped.model.get_submodule('Mixed_6b'))

    def test_vgg16_bn_tap_is_block4_pool(self):
        from torchvision import models
        network = models.vgg16_bn(weights=None).eval()
        self.assertIsInstance(network.features[33], nn.MaxPool2d)
        self.assertIsInstance(network.features[32], nn.ReLU)
        self.assertEqual(network.features[30].out_channels, 512)  # Conv4_3

    def test_default_tap_is_resnet_layer3(self):
        x, _ = images()
        a, b = local_model(), Layer3Source(local_model().wrapped, 'layer3')
        try:
            a(x)
            b(x)
            self.assertTrue(torch.equal(a.value, b.value))
            self.assertEqual(tuple(a.value.shape[:2]), (1, 4))
        finally:
            a.close()
            b.close()

    def test_dense_tap_captures_transition3_and_finetunes_within_budget(self):
        x, initial = images()
        model = dense_source()
        try:
            captured = {}
            handle = model.wrapped.model.features.transition3.register_forward_hook(
                lambda _m, _i, out: captured.__setitem__('out', out))
            model(x)
            handle.remove()
            self.assertTrue(torch.equal(model.value, captured['out']))
            for method in ('fft', 'aaf', 'logit-fft'):
                image = finetune(model, x, initial, 2, 0, method, 16, np.random.RandomState(4), mask_count=4)
                self.assertLessEqual(float((image - x).abs().max()), 16 / 255 + 1e-7)
                self.assertFalse(torch.equal(image, initial))
        finally:
            model.close()

    def test_window_adaptive_pool_matches_torch_forward_and_backward(self):
        from utils.models import WindowAdaptiveAvgPool2d
        generator = torch.Generator().manual_seed(5)
        for size, out in (((9, 9), (7, 7)), ((10, 13), (7, 7)), ((7, 7), (7, 7)), ((5, 11), (3, 4))):
            x = torch.randn(2, 3, *size, generator=generator, dtype=torch.float64, requires_grad=True)
            weight = torch.randn(2, 3, *out, generator=generator, dtype=torch.float64)
            ref = nn.AdaptiveAvgPool2d(out)(x)
            got = WindowAdaptiveAvgPool2d(out)(x)
            torch.testing.assert_close(got, ref, atol=1e-12, rtol=1e-12)
            g_ref, = torch.autograd.grad((ref * weight).sum(), x)
            g_got, = torch.autograd.grad((got * weight).sum(), x)
            torch.testing.assert_close(g_got, g_ref, atol=1e-12, rtol=1e-12)

    def test_unknown_layer_fails_loudly(self):
        with self.assertRaises(AttributeError):
            Layer3Source(NormalizeModel(Toy(), [.5] * 3, [.5] * 3), 'features.transition3')


if __name__ == '__main__':
    unittest.main()
