"""
RECIST 1.1 measurement and response classification for brain metastases.
Provides longest diameter, perpendicular diameter, and sum-of-diameters calculations.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import pdist


class RECISTMeasurer:
    """
    RECIST 1.1 compliant measurement and response classification.
    """

    # RECIST 1.1 thresholds
    CR_THRESHOLD = 0.0       # Complete disappearance
    PR_THRESHOLD = -30.0     # >= 30% decrease in SoD
    PD_THRESHOLD = 20.0      # >= 20% increase in SoD from nadir
    PD_ABSOLUTE_MM = 5.0     # Plus absolute increase >= 5mm

    def measure_lesion(
        self,
        binary_mask: np.ndarray,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Dict:
        """
        Measure a single lesion per RECIST 1.1 criteria.

        Args:
            binary_mask: Binary mask of a single lesion (H, W, D)
            voxel_spacing: Voxel dimensions in mm (h, w, d)

        Returns:
            Dict with longest_diameter_mm, perpendicular_diameter_mm,
            volume_mm3, measurable (True if >= 10mm longest diameter)
        """
        coords = np.argwhere(binary_mask > 0)
        if len(coords) < 2:
            return {
                "longest_diameter_mm": 0.0,
                "perpendicular_diameter_mm": 0.0,
                "volume_mm3": 0.0,
                "measurable": False,
            }

        # Scale coordinates to mm
        coords_mm = coords.astype(np.float64) * np.array(voxel_spacing)

        # Find longest diameter via pairwise distances
        if len(coords_mm) > 5000:
            # Subsample for large lesions to keep computation manageable
            idx = np.random.default_rng(42).choice(len(coords_mm), 5000, replace=False)
            coords_sub = coords_mm[idx]
        else:
            coords_sub = coords_mm

        distances = pdist(coords_sub)
        longest_diameter_mm = float(distances.max()) if len(distances) > 0 else 0.0

        # Perpendicular diameter: approximate as max extent perpendicular to longest axis
        if longest_diameter_mm > 0 and len(coords_sub) > 1:
            # Find the two points defining the longest diameter
            from scipy.spatial.distance import squareform
            dist_matrix = squareform(distances)
            max_idx = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
            p1, p2 = coords_sub[max_idx[0]], coords_sub[max_idx[1]]

            # Project all points onto the longest axis and find max perpendicular distance
            axis = p2 - p1
            axis_norm = axis / (np.linalg.norm(axis) + 1e-8)

            # Vector from p1 to each point
            vecs = coords_sub - p1

            # Perpendicular component
            projections = np.dot(vecs, axis_norm)[:, None] * axis_norm
            perp_vecs = vecs - projections
            perp_distances = np.linalg.norm(perp_vecs, axis=1)

            perpendicular_diameter_mm = float(perp_distances.max()) * 2
        else:
            perpendicular_diameter_mm = 0.0

        # Volume
        voxel_vol = float(np.prod(voxel_spacing))
        volume_mm3 = int(binary_mask.sum()) * voxel_vol

        # RECIST measurability: >= 10mm longest diameter for brain lesions
        measurable = longest_diameter_mm >= 10.0

        return {
            "longest_diameter_mm": round(longest_diameter_mm, 2),
            "perpendicular_diameter_mm": round(perpendicular_diameter_mm, 2),
            "volume_mm3": round(volume_mm3, 2),
            "measurable": measurable,
        }

    def compute_sum_of_diameters(
        self,
        binary_mask: np.ndarray,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        max_target_lesions: int = 5,
    ) -> Tuple[float, List[Dict]]:
        """
        Compute RECIST 1.1 Sum of Diameters for target lesions.
        Selects up to max_target_lesions largest measurable lesions.

        Args:
            binary_mask: Full segmentation mask (H, W, D)
            voxel_spacing: Voxel dimensions in mm
            max_target_lesions: Max number of target lesions (RECIST 1.1: 5 total, 2 per organ)

        Returns:
            Tuple of (sum_of_diameters_mm, list of lesion measurements)
        """
        labeled, n_lesions = ndimage.label(binary_mask > 0)

        measurements = []
        for i in range(1, n_lesions + 1):
            lesion_mask = (labeled == i).astype(np.uint8)
            measurement = self.measure_lesion(lesion_mask, voxel_spacing)
            measurement["lesion_id"] = i
            measurements.append(measurement)

        # Sort by longest diameter descending, select top measurable lesions
        measurements.sort(key=lambda x: x["longest_diameter_mm"], reverse=True)
        target_lesions = [m for m in measurements if m["measurable"]][:max_target_lesions]

        sum_of_diameters = sum(m["longest_diameter_mm"] for m in target_lesions)

        return round(sum_of_diameters, 2), measurements

    def classify_response(
        self,
        baseline_sod: float,
        followup_sod: float,
        nadir_sod: float = None,
        new_lesions: bool = False,
    ) -> str:
        """
        Classify response per RECIST 1.1 criteria.

        Args:
            baseline_sod: Baseline Sum of Diameters (mm)
            followup_sod: Follow-up Sum of Diameters (mm)
            nadir_sod: Nadir (smallest) Sum of Diameters (mm). If None, uses baseline.
            new_lesions: Whether new lesions appeared

        Returns:
            Response category: 'CR', 'PR', 'SD', or 'PD'
        """
        if nadir_sod is None:
            nadir_sod = baseline_sod

        # New lesions = automatic PD
        if new_lesions:
            return "PD"

        # Complete Response: disappearance of all target lesions
        if followup_sod == 0 and baseline_sod > 0:
            return "CR"

        if baseline_sod == 0:
            return "CR" if followup_sod == 0 else "PD"

        # Percent change from baseline (for PR)
        change_from_baseline = ((followup_sod - baseline_sod) / baseline_sod) * 100

        # Percent change from nadir (for PD)
        change_from_nadir = ((followup_sod - nadir_sod) / (nadir_sod + 1e-8)) * 100
        absolute_increase = followup_sod - nadir_sod

        # Progressive Disease: >= 20% increase from nadir AND >= 5mm absolute increase
        if change_from_nadir >= self.PD_THRESHOLD and absolute_increase >= self.PD_ABSOLUTE_MM:
            return "PD"

        # Partial Response: >= 30% decrease from baseline
        if change_from_baseline <= self.PR_THRESHOLD:
            return "PR"

        # Stable Disease: neither PR nor PD
        return "SD"
