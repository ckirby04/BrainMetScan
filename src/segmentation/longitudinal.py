"""
Longitudinal tracking for brain metastasis lesions across timepoints.
Matches lesions between baseline and follow-up using centroid proximity,
computes volume changes, and classifies response per RECIST 1.1.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import ndimage

from ..rag.recist import RECISTMeasurer


class LongitudinalTracker:
    """
    Tracks and compares brain metastasis lesions across timepoints.
    """

    def __init__(self, max_match_distance_mm: float = 20.0):
        """
        Args:
            max_match_distance_mm: Maximum distance in mm to consider lesions as matched.
        """
        self.max_match_distance_mm = max_match_distance_mm
        self.recist = RECISTMeasurer()

    def compare_timepoints(
        self,
        baseline_result: Dict,
        followup_result: Dict,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Dict:
        """
        Compare lesions between baseline and follow-up segmentation results.

        Args:
            baseline_result: Output from SmartEnsemble.predict_volume()
                             (must have 'binary_mask' and 'lesion_details')
            followup_result: Same format as baseline
            voxel_spacing: Voxel dimensions in mm

        Returns:
            Dict with:
                matched_lesions: list of matched lesion pairs with volume changes
                new_lesions: count of lesions only in follow-up
                resolved_lesions: count of lesions only in baseline
                response_category: CR/PR/SD/PD per RECIST 1.1
                sum_of_diameters_baseline_mm: RECIST SoD at baseline
                sum_of_diameters_followup_mm: RECIST SoD at follow-up
        """
        baseline_lesions = baseline_result["lesion_details"]
        followup_lesions = followup_result["lesion_details"]

        # Match lesions
        matches, unmatched_baseline, unmatched_followup = self._match_lesions(
            baseline_lesions, followup_lesions, voxel_spacing
        )

        # Compute volume changes for matched lesions
        matched_details = []
        for b_idx, f_idx in matches:
            bl = baseline_lesions[b_idx]
            fl = followup_lesions[f_idx]
            baseline_vol = bl["volume_mm3"]
            followup_vol = fl["volume_mm3"]

            if baseline_vol > 0:
                vol_change_pct = ((followup_vol - baseline_vol) / baseline_vol) * 100
            else:
                vol_change_pct = 100.0 if followup_vol > 0 else 0.0

            matched_details.append({
                "baseline_id": bl["id"],
                "followup_id": fl["id"],
                "baseline_volume_mm3": baseline_vol,
                "followup_volume_mm3": followup_vol,
                "volume_change_percent": round(vol_change_pct, 1),
            })

        # RECIST measurements
        baseline_sod, _ = self.recist.compute_sum_of_diameters(
            baseline_result["binary_mask"], voxel_spacing
        )
        followup_sod, _ = self.recist.compute_sum_of_diameters(
            followup_result["binary_mask"], voxel_spacing
        )

        has_new_lesions = len(unmatched_followup) > 0
        response = self.recist.classify_response(
            baseline_sod, followup_sod, new_lesions=has_new_lesions
        )

        return {
            "matched_lesions": matched_details,
            "new_lesions": len(unmatched_followup),
            "resolved_lesions": len(unmatched_baseline),
            "response_category": response,
            "sum_of_diameters_baseline_mm": baseline_sod,
            "sum_of_diameters_followup_mm": followup_sod,
        }

    def _match_lesions(
        self,
        baseline_lesions: List[Dict],
        followup_lesions: List[Dict],
        voxel_spacing: Tuple[float, float, float],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match lesions between timepoints using Hungarian algorithm on centroid distances.

        Returns:
            (matched_pairs, unmatched_baseline_indices, unmatched_followup_indices)
        """
        if not baseline_lesions or not followup_lesions:
            return (
                [],
                list(range(len(baseline_lesions))),
                list(range(len(followup_lesions))),
            )

        n_baseline = len(baseline_lesions)
        n_followup = len(followup_lesions)

        # Build cost matrix of Euclidean distances in mm
        # Use large sentinel value instead of np.inf for scipy compatibility
        cost_matrix = np.full((n_baseline, n_followup), 1e9)
        for i, bl in enumerate(baseline_lesions):
            for j, fl in enumerate(followup_lesions):
                bc = np.array(bl["centroid"]) * np.array(voxel_spacing)
                fc = np.array(fl["centroid"]) * np.array(voxel_spacing)
                dist = np.linalg.norm(bc - fc)
                cost_matrix[i, j] = dist

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        matched_baseline = set()
        matched_followup = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= self.max_match_distance_mm:
                matched.append((r, c))
                matched_baseline.add(r)
                matched_followup.add(c)

        unmatched_baseline = [i for i in range(n_baseline) if i not in matched_baseline]
        unmatched_followup = [j for j in range(n_followup) if j not in matched_followup]

        return matched, unmatched_baseline, unmatched_followup
