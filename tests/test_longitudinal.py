"""Tests for longitudinal tracking and lesion matching."""

import numpy as np
import pytest

from src.segmentation.longitudinal import LongitudinalTracker


class TestLongitudinalTracker:
    def setup_method(self):
        self.tracker = LongitudinalTracker(max_match_distance_mm=20.0)

    def _make_result(self, mask):
        """Helper to create a result dict from a mask."""
        from src.segmentation.postprocessing import extract_lesion_details
        details = extract_lesion_details(mask.astype(np.float32))
        return {
            "binary_mask": mask.astype(np.float32),
            "lesion_details": details,
        }

    def test_matching_single_lesion(self):
        """Same lesion at same location should match."""
        mask = np.zeros((32, 32, 32), dtype=np.float32)
        mask[10:18, 10:18, 10:18] = 1

        baseline = self._make_result(mask)
        followup = self._make_result(mask)

        result = self.tracker.compare_timepoints(baseline, followup)
        assert len(result["matched_lesions"]) == 1
        assert result["new_lesions"] == 0
        assert result["resolved_lesions"] == 0

    def test_new_lesion_detected(self):
        """Additional lesion in follow-up should be counted as new."""
        bl_mask = np.zeros((32, 32, 32), dtype=np.float32)
        bl_mask[5:10, 5:10, 5:10] = 1

        fu_mask = np.zeros((32, 32, 32), dtype=np.float32)
        fu_mask[5:10, 5:10, 5:10] = 1
        fu_mask[25:30, 25:30, 25:30] = 1  # New lesion

        baseline = self._make_result(bl_mask)
        followup = self._make_result(fu_mask)

        result = self.tracker.compare_timepoints(baseline, followup)
        assert result["new_lesions"] == 1

    def test_resolved_lesion(self):
        """Lesion gone in follow-up should be counted as resolved."""
        bl_mask = np.zeros((32, 32, 32), dtype=np.float32)
        bl_mask[5:10, 5:10, 5:10] = 1
        bl_mask[25:30, 25:30, 25:30] = 1  # Will resolve

        fu_mask = np.zeros((32, 32, 32), dtype=np.float32)
        fu_mask[5:10, 5:10, 5:10] = 1

        baseline = self._make_result(bl_mask)
        followup = self._make_result(fu_mask)

        result = self.tracker.compare_timepoints(baseline, followup)
        assert result["resolved_lesions"] == 1

    def test_response_pd_new_lesions(self):
        """New lesions should trigger PD classification."""
        bl_mask = np.zeros((32, 32, 32), dtype=np.float32)
        bl_mask[5:10, 5:10, 5:10] = 1

        fu_mask = np.zeros((32, 32, 32), dtype=np.float32)
        fu_mask[5:10, 5:10, 5:10] = 1
        fu_mask[25:30, 25:30, 25:30] = 1

        baseline = self._make_result(bl_mask)
        followup = self._make_result(fu_mask)

        result = self.tracker.compare_timepoints(baseline, followup)
        assert result["response_category"] == "PD"

    def test_response_cr(self):
        """Complete resolution should be CR."""
        bl_mask = np.zeros((32, 32, 32), dtype=np.float32)
        bl_mask[5:15, 5:15, 5:15] = 1

        fu_mask = np.zeros((32, 32, 32), dtype=np.float32)

        baseline = self._make_result(bl_mask)
        followup = self._make_result(fu_mask)

        result = self.tracker.compare_timepoints(baseline, followup)
        assert result["response_category"] == "CR"

    def test_empty_baseline_and_followup(self):
        """Both empty should be CR."""
        empty_mask = np.zeros((16, 16, 16), dtype=np.float32)
        baseline = self._make_result(empty_mask)
        followup = self._make_result(empty_mask)

        result = self.tracker.compare_timepoints(baseline, followup)
        assert result["response_category"] == "CR"
