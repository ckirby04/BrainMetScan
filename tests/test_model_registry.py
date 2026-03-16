"""Tests for ModelRegistry."""

import pytest
import torch
import yaml

from src.segmentation.model_registry import ModelRegistry


class TestModelRegistry:
    def test_register_and_list(self, dummy_checkpoint, tmp_path):
        """Test registering a model and listing it."""
        registry = ModelRegistry(tmp_path)

        registry.register_model(
            source_path=str(dummy_checkpoint),
            name="test_model",
            patch_size=8,
            architecture="lightweight",
            threshold=0.3,
        )

        models = registry.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "test_model"
        assert models[0]["patch_size"] == 8
        assert models[0]["exists"] is True

    def test_validate_valid_checkpoint(self, dummy_checkpoint):
        """Test checkpoint validation passes for valid checkpoint."""
        registry = ModelRegistry()
        assert registry.validate_checkpoint(str(dummy_checkpoint)) is True

    def test_validate_invalid_checkpoint(self, tmp_path):
        """Test checkpoint validation fails for invalid file."""
        bad_path = tmp_path / "bad.pth"
        torch.save({"wrong_key": "nope"}, bad_path)

        registry = ModelRegistry()
        with pytest.raises(ValueError, match="missing 'model_state_dict'"):
            registry.validate_checkpoint(str(bad_path))

    def test_get_ensemble_config(self, dummy_checkpoint, tmp_path):
        """Test ensemble config generation."""
        registry = ModelRegistry(tmp_path)
        registry.register_model(
            source_path=str(dummy_checkpoint),
            name="model_a",
            patch_size=12,
            threshold=0.25,
        )

        config = registry.get_ensemble_config()
        assert len(config["models"]) == 1
        assert config["models"][0]["name"] == "model_a"
        assert "full_path" in config["models"][0]

    def test_register_replaces_existing(self, dummy_checkpoint, tmp_path):
        """Test re-registering with same name replaces old entry."""
        registry = ModelRegistry(tmp_path)

        registry.register_model(
            source_path=str(dummy_checkpoint),
            name="same_name",
            patch_size=8,
            threshold=0.3,
        )
        registry.register_model(
            source_path=str(dummy_checkpoint),
            name="same_name",
            patch_size=16,
            threshold=0.5,
        )

        models = registry.list_models()
        assert len(models) == 1
        assert models[0]["patch_size"] == 16
