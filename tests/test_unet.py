"""Tests for UNet model forward pass and output properties."""

import pytest
import torch

from src.segmentation.unet import LightweightUNet3D


class TestLightweightUNet3D:
    def test_forward_pass_shape(self):
        model = LightweightUNet3D(in_channels=4, out_channels=1, base_channels=4, depth=2)
        model.eval()
        x = torch.randn(1, 4, 16, 16, 16)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 16, 16, 16)

    def test_forward_different_sizes(self):
        model = LightweightUNet3D(in_channels=4, out_channels=1, base_channels=4, depth=2)
        model.eval()
        for size in [8, 16, 32]:
            x = torch.randn(1, 4, size, size, size)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 1, size, size, size), f"Failed for size {size}"

    def test_sigmoid_range(self):
        model = LightweightUNet3D(in_channels=4, out_channels=1, base_channels=4, depth=2)
        model.eval()
        x = torch.randn(1, 4, 16, 16, 16)
        with torch.no_grad():
            out = torch.sigmoid(model(x))
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_batch_dimension(self):
        model = LightweightUNet3D(in_channels=4, out_channels=1, base_channels=4, depth=2)
        model.eval()
        x = torch.randn(2, 4, 16, 16, 16)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 2

    def test_deep_supervision(self):
        model = LightweightUNet3D(
            in_channels=4, out_channels=1, base_channels=4, depth=2, deep_supervision=True
        )
        model.eval()
        x = torch.randn(1, 4, 16, 16, 16)
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, tuple)
        assert out[0].shape == (1, 1, 16, 16, 16)

    def test_attention_and_residual(self):
        model = LightweightUNet3D(
            in_channels=4, out_channels=1, base_channels=4, depth=2,
            use_attention=True, use_residual=True,
        )
        model.eval()
        x = torch.randn(1, 4, 16, 16, 16)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 16, 16, 16)
