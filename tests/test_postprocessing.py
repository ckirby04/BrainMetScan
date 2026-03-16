"""Tests for postprocessing operations."""

import numpy as np
import pytest

from src.segmentation.postprocessing import (
    extract_lesion_details,
    full_postprocessing_pipeline,
    morphological_closing,
    morphological_opening,
    remove_small_components,
)


class TestRemoveSmallComponents:
    def test_removes_tiny_components(self):
        mask = np.zeros((32, 32, 32), dtype=np.float32)
        # Large component: sphere radius 5 (~523 voxels)
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    if (i - 16) ** 2 + (j - 16) ** 2 + (k - 16) ** 2 < 25:
                        mask[i, j, k] = 1
        # Tiny component: 3 voxels
        mask[0, 0, 0] = 1
        mask[0, 0, 1] = 1
        mask[0, 1, 0] = 1

        result = remove_small_components(mask, min_size=10)
        # Large component should remain
        assert result[16, 16, 16] == 1
        # Tiny component should be removed
        assert result[0, 0, 0] == 0

    def test_keeps_large_components(self):
        mask = np.zeros((16, 16, 16), dtype=np.float32)
        mask[4:12, 4:12, 4:12] = 1  # 512 voxels
        result = remove_small_components(mask, min_size=100)
        assert result.sum() == 512

    def test_batch_mode(self):
        mask = np.zeros((2, 1, 16, 16, 16), dtype=np.float32)
        mask[0, 0, 4:12, 4:12, 4:12] = 1
        mask[0, 0, 0, 0, 0] = 1  # tiny
        result = remove_small_components(mask, min_size=5)
        assert result[0, 0, 0, 0, 0] == 0
        assert result[0, 0, 8, 8, 8] == 1


class TestMorphologicalOps:
    def test_opening_removes_noise(self):
        mask = np.zeros((16, 16, 16), dtype=np.float32)
        mask[4:12, 4:12, 4:12] = 1  # Main blob
        mask[0, 0, 0] = 1  # Noise point
        result = morphological_opening(mask, structure_size=1)
        assert result[0, 0, 0] == 0
        assert result[8, 8, 8] == 1

    def test_closing_fills_holes(self):
        mask = np.zeros((16, 16, 16), dtype=np.float32)
        mask[4:12, 4:12, 4:12] = 1
        mask[8, 8, 8] = 0  # Small hole
        result = morphological_closing(mask, structure_size=1)
        assert result[8, 8, 8] == 1


class TestFullPipeline:
    def test_pipeline_output_binary(self):
        probs = np.random.rand(16, 16, 16).astype(np.float32)
        result = full_postprocessing_pipeline(probs, threshold=0.5)
        assert set(np.unique(result)).issubset({0.0, 1.0})

    def test_pipeline_reduces_noise(self):
        probs = np.zeros((16, 16, 16), dtype=np.float32)
        probs[4:12, 4:12, 4:12] = 0.8  # Main lesion
        probs[0, 0, 0] = 0.9  # Noise voxel
        result = full_postprocessing_pipeline(probs, threshold=0.5, min_size=10)
        assert result[0, 0, 0] == 0
        assert result[8, 8, 8] == 1


class TestExtractLesionDetails:
    def test_extracts_single_lesion(self):
        mask = np.zeros((32, 32, 32), dtype=np.float32)
        mask[10:20, 10:20, 10:20] = 1
        details = extract_lesion_details(mask)
        assert len(details) == 1
        assert details[0]["volume_voxels"] == 1000
        assert details[0]["id"] == 1

    def test_extracts_multiple_lesions(self):
        mask = np.zeros((32, 32, 32), dtype=np.float32)
        mask[2:6, 2:6, 2:6] = 1   # Lesion 1
        mask[20:28, 20:28, 20:28] = 1  # Lesion 2
        details = extract_lesion_details(mask)
        assert len(details) == 2

    def test_empty_mask(self):
        mask = np.zeros((16, 16, 16), dtype=np.float32)
        details = extract_lesion_details(mask)
        assert len(details) == 0

    def test_volume_mm3_with_spacing(self):
        mask = np.zeros((16, 16, 16), dtype=np.float32)
        mask[4:8, 4:8, 4:8] = 1  # 64 voxels
        details = extract_lesion_details(mask, voxel_spacing=(2.0, 2.0, 2.0))
        assert details[0]["volume_voxels"] == 64
        assert details[0]["volume_mm3"] == 64 * 8.0  # 2*2*2 = 8 mm3/voxel

    def test_confidence_from_probability_map(self):
        mask = np.zeros((16, 16, 16), dtype=np.float32)
        mask[4:8, 4:8, 4:8] = 1
        prob = np.zeros((16, 16, 16), dtype=np.float32)
        prob[4:8, 4:8, 4:8] = 0.9
        details = extract_lesion_details(mask, probability_map=prob)
        assert abs(details[0]["confidence"] - 0.9) < 0.01
