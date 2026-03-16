"""Tests for RECIST 1.1 measurements and classification."""

import numpy as np
import pytest

from src.rag.recist import RECISTMeasurer


class TestRECISTMeasurer:
    def setup_method(self):
        self.measurer = RECISTMeasurer()

    def test_measure_single_lesion(self):
        """Test measurement of a simple cubic lesion."""
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[10:20, 10:20, 10:20] = 1  # 10x10x10 cube

        result = self.measurer.measure_lesion(mask, voxel_spacing=(1.0, 1.0, 1.0))
        assert result["longest_diameter_mm"] > 0
        assert result["volume_mm3"] == 1000.0
        assert result["measurable"] is True  # 10mm side > 10mm threshold

    def test_small_lesion_not_measurable(self):
        """Lesions < 10mm should be non-measurable per RECIST."""
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[14:17, 14:17, 14:17] = 1  # 3mm cube

        result = self.measurer.measure_lesion(mask, voxel_spacing=(1.0, 1.0, 1.0))
        assert result["measurable"] is False

    def test_empty_mask(self):
        mask = np.zeros((16, 16, 16), dtype=np.uint8)
        result = self.measurer.measure_lesion(mask)
        assert result["longest_diameter_mm"] == 0.0
        assert result["measurable"] is False

    def test_voxel_spacing_affects_diameter(self):
        """Larger voxels should produce larger diameter."""
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[10:20, 10:20, 10:20] = 1

        result_1mm = self.measurer.measure_lesion(mask, voxel_spacing=(1.0, 1.0, 1.0))
        result_2mm = self.measurer.measure_lesion(mask, voxel_spacing=(2.0, 2.0, 2.0))

        assert result_2mm["longest_diameter_mm"] > result_1mm["longest_diameter_mm"]

    def test_sum_of_diameters(self):
        """Test SoD computation with multiple lesions."""
        mask = np.zeros((64, 64, 64), dtype=np.uint8)
        mask[5:20, 5:20, 5:20] = 1    # Large lesion
        mask[40:52, 40:52, 40:52] = 1  # Medium lesion

        sod, measurements = self.measurer.compute_sum_of_diameters(mask, voxel_spacing=(1.0, 1.0, 1.0))
        assert sod > 0
        assert len(measurements) == 2


class TestRECISTClassification:
    def setup_method(self):
        self.measurer = RECISTMeasurer()

    def test_cr(self):
        assert self.measurer.classify_response(50.0, 0.0) == "CR"

    def test_pr(self):
        """30% decrease -> PR."""
        assert self.measurer.classify_response(100.0, 65.0) == "PR"

    def test_sd(self):
        """10% decrease -> SD (not enough for PR)."""
        assert self.measurer.classify_response(100.0, 90.0) == "SD"

    def test_pd_from_nadir(self):
        """20% increase from nadir + 5mm -> PD."""
        assert self.measurer.classify_response(100.0, 130.0, nadir_sod=100.0) == "PD"

    def test_pd_new_lesions(self):
        """New lesions always = PD."""
        assert self.measurer.classify_response(100.0, 100.0, new_lesions=True) == "PD"

    def test_sd_small_increase(self):
        """Small increase (< 20% or < 5mm absolute) -> SD."""
        assert self.measurer.classify_response(100.0, 115.0, nadir_sod=100.0) == "SD"

    def test_pd_requires_both_thresholds(self):
        """PD requires BOTH 20% increase AND 5mm absolute increase."""
        # 20% increase but only 2mm absolute -> SD
        assert self.measurer.classify_response(10.0, 12.0, nadir_sod=10.0) == "SD"
