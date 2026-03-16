"""Tests for SmartEnsemble loading and inference."""

import pytest
import torch
import yaml
from pathlib import Path

from src.segmentation.ensemble import SmartEnsemble


class TestSmartEnsemble:
    def test_from_config_no_models(self, tmp_path):
        """Test ensemble creation with no actual model files (graceful skip)."""
        config = {
            "ensemble": {"fusion_mode": "union"},
            "models": [
                {
                    "name": "nonexistent",
                    "path": "model/nonexistent.pth",
                    "architecture": "lightweight",
                    "patch_size": 8,
                    "threshold": 0.3,
                    "base_channels": 4,
                    "depth": 2,
                    "use_attention": True,
                    "use_residual": True,
                }
            ],
            "inference": {},
        }
        config_path = tmp_path / "configs" / "models.yaml"
        config_path.parent.mkdir()
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        ensemble = SmartEnsemble.from_config(str(config_path), device="cpu")
        assert len(ensemble.models) == 0

    def test_from_config_with_model(self, dummy_checkpoint, tmp_path):
        """Test ensemble loads a real checkpoint."""
        config = {
            "ensemble": {"fusion_mode": "union"},
            "models": [
                {
                    "name": "test_model",
                    "path": str(dummy_checkpoint),
                    "architecture": "lightweight",
                    "patch_size": 8,
                    "threshold": 0.3,
                    "base_channels": 4,
                    "depth": 2,
                    "use_attention": True,
                    "use_residual": True,
                }
            ],
            "inference": {},
        }

        # Need to make full_path work: model_configs get full_path from from_config
        ensemble = SmartEnsemble(
            [{"full_path": str(dummy_checkpoint), **config["models"][0]}],
            device="cpu",
            fusion_mode="union",
        )
        assert len(ensemble.models) == 1
        assert ensemble.names[0] == "test_model"

    def test_forward_single_model(self, dummy_checkpoint):
        """Test forward pass with a single model."""
        config = {
            "name": "test",
            "full_path": str(dummy_checkpoint),
            "architecture": "lightweight",
            "patch_size": 8,
            "threshold": 0.3,
            "base_channels": 4,
            "depth": 2,
            "use_attention": True,
            "use_residual": True,
        }
        ensemble = SmartEnsemble([config], device="cpu", fusion_mode="union")
        x = torch.randn(1, 4, 8, 8, 8)
        with torch.no_grad():
            out = ensemble(x, target_size=8)
        assert out.shape == (1, 1, 8, 8, 8)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_predict_volume_output_structure(self, dummy_checkpoint, synthetic_volume):
        """Test predict_volume returns expected dict structure."""
        config = {
            "name": "test",
            "full_path": str(dummy_checkpoint),
            "architecture": "lightweight",
            "patch_size": 8,
            "threshold": 0.3,
            "base_channels": 4,
            "depth": 2,
            "use_attention": True,
            "use_residual": True,
        }
        ensemble = SmartEnsemble([config], device="cpu", fusion_mode="union")

        image, _, _, _ = synthetic_volume
        result = ensemble.predict_volume(
            image,
            window_size=(16, 16, 16),
            overlap=0.5,
            threshold=0.5,
            postprocess=True,
        )

        assert "probability_map" in result
        assert "binary_mask" in result
        assert "lesion_count" in result
        assert "lesion_details" in result
        assert isinstance(result["lesion_count"], int)
        assert isinstance(result["lesion_details"], list)

    def test_predict_with_details(self, dummy_checkpoint):
        """Test predict_with_details returns per-model results."""
        config = {
            "name": "test",
            "full_path": str(dummy_checkpoint),
            "architecture": "lightweight",
            "patch_size": 8,
            "threshold": 0.3,
            "base_channels": 4,
            "depth": 2,
            "use_attention": True,
            "use_residual": True,
        }
        ensemble = SmartEnsemble([config], device="cpu", fusion_mode="union")
        x = torch.randn(1, 4, 8, 8, 8)

        details = ensemble.predict_with_details(x, target_size=8)
        assert "individual" in details
        assert "fused" in details
        assert "test" in details["individual"]

    def test_fusion_modes(self, dummy_checkpoint):
        """Test all fusion modes produce valid output."""
        base_config = {
            "name": "test",
            "full_path": str(dummy_checkpoint),
            "architecture": "lightweight",
            "patch_size": 8,
            "threshold": 0.3,
            "base_channels": 4,
            "depth": 2,
            "use_attention": True,
            "use_residual": True,
        }

        for mode in ["union", "hybrid"]:
            ensemble = SmartEnsemble([base_config], device="cpu", fusion_mode=mode)
            x = torch.randn(1, 4, 8, 8, 8)
            with torch.no_grad():
                out = ensemble(x, target_size=8)
            assert out.shape == (1, 1, 8, 8, 8)
