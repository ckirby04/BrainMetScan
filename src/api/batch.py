"""
Batch processing for running segmentation on cohorts of cases.
Produces CSV exports and optional waterfall plots for clinical trial reporting.
"""

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch

from ..segmentation.ensemble import SmartEnsemble
from ..segmentation.longitudinal import LongitudinalTracker
from ..segmentation.postprocessing import extract_lesion_details


class BatchProcessor:
    """
    Process a cohort of cases with ensemble segmentation,
    optional longitudinal comparison, and CSV export.
    """

    def __init__(
        self,
        ensemble: SmartEnsemble,
        sequences: List[str] = None,
    ):
        self.ensemble = ensemble
        self.sequences = sequences or ["t1_pre", "t1_gd", "flair", "t2"]
        self.tracker = LongitudinalTracker()

    def process_cohort(
        self,
        case_directories: List[str],
        output_dir: str,
        threshold: float = 0.5,
        use_tta: bool = False,
        baseline_directories: List[str] = None,
    ) -> List[Dict]:
        """
        Process all cases in a cohort.

        Args:
            case_directories: List of paths to case directories (each with NIfTI sequences)
            output_dir: Output directory for masks and reports
            threshold: Segmentation threshold
            use_tta: Whether to use test-time augmentation
            baseline_directories: Optional parallel list of baseline directories for longitudinal comparison

        Returns:
            List of result dicts per case
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, case_dir in enumerate(case_directories):
            case_dir = Path(case_dir)
            case_id = case_dir.name
            print(f"Processing {case_id} ({i + 1}/{len(case_directories)})...")

            try:
                start = time.time()
                tensor, spacing, affine = self._load_case(case_dir)

                result = self.ensemble.predict_volume(
                    tensor,
                    threshold=threshold,
                    use_tta=use_tta,
                    voxel_spacing=spacing,
                )
                elapsed = time.time() - start

                # Save mask
                mask_path = output_dir / f"{case_id}_pred.nii.gz"
                nii_out = nib.Nifti1Image(result["binary_mask"].astype(np.float32), affine)
                nib.save(nii_out, str(mask_path))

                case_result = {
                    "case_id": case_id,
                    "lesion_count": result["lesion_count"],
                    "total_volume_voxels": sum(l["volume_voxels"] for l in result["lesion_details"]),
                    "total_volume_mm3": sum(l["volume_mm3"] for l in result["lesion_details"]),
                    "processing_time_s": round(elapsed, 2),
                    "status": "success",
                    "mask_path": str(mask_path),
                }

                # Longitudinal comparison
                if baseline_directories and i < len(baseline_directories):
                    baseline_dir = Path(baseline_directories[i])
                    if baseline_dir.exists():
                        bl_tensor, bl_spacing, _ = self._load_case(baseline_dir)
                        bl_result = self.ensemble.predict_volume(
                            bl_tensor, threshold=threshold, voxel_spacing=bl_spacing
                        )
                        comparison = self.tracker.compare_timepoints(
                            bl_result, result, voxel_spacing=spacing
                        )
                        case_result.update({
                            "response_category": comparison["response_category"],
                            "sod_baseline_mm": comparison["sum_of_diameters_baseline_mm"],
                            "sod_followup_mm": comparison["sum_of_diameters_followup_mm"],
                            "new_lesions": comparison["new_lesions"],
                            "resolved_lesions": comparison["resolved_lesions"],
                        })

                results.append(case_result)

            except Exception as e:
                print(f"  Error: {e}")
                results.append({"case_id": case_id, "status": "error", "error": str(e)})

        # Auto-export CSV
        csv_path = output_dir / "cohort_results.csv"
        self.export_to_csv(results, str(csv_path))

        return results

    def export_to_csv(self, results: List[Dict], output_path: str):
        """
        Export results to SDTM/ADaM-compatible CSV.

        Args:
            results: List of result dicts from process_cohort()
            output_path: Path for CSV file
        """
        if not results:
            return

        fieldnames = sorted(set().union(*(r.keys() for r in results)))
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Results exported to {output_path}")

    def generate_waterfall_plot(
        self,
        results: List[Dict],
        output_path: str,
        metric: str = "total_volume_mm3",
    ):
        """
        Generate a waterfall plot showing volume changes across the cohort.

        Args:
            results: List of result dicts (must have baseline comparison data)
            output_path: Path for PNG file
            metric: Metric to plot ('total_volume_mm3' or 'sod_followup_mm')
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Filter to cases with longitudinal data
        longitudinal = [r for r in results if "sod_baseline_mm" in r and r["status"] == "success"]
        if not longitudinal:
            print("No longitudinal data for waterfall plot")
            return

        # Compute percent changes
        changes = []
        for r in longitudinal:
            bl = r.get("sod_baseline_mm", 0)
            fu = r.get("sod_followup_mm", 0)
            if bl > 0:
                pct = ((fu - bl) / bl) * 100
            else:
                pct = 0
            changes.append((r["case_id"], pct, r.get("response_category", "SD")))

        # Sort by change
        changes.sort(key=lambda x: x[1])

        case_ids = [c[0] for c in changes]
        pct_changes = [c[1] for c in changes]
        categories = [c[2] for c in changes]

        colors = {"CR": "green", "PR": "blue", "SD": "gray", "PD": "red"}
        bar_colors = [colors.get(cat, "gray") for cat in categories]

        fig, ax = plt.subplots(figsize=(max(8, len(changes) * 0.5), 6))
        ax.bar(range(len(changes)), pct_changes, color=bar_colors)
        ax.axhline(y=-30, color="blue", linestyle="--", alpha=0.5, label="PR threshold (-30%)")
        ax.axhline(y=20, color="red", linestyle="--", alpha=0.5, label="PD threshold (+20%)")
        ax.set_xlabel("Cases")
        ax.set_ylabel("Change in Sum of Diameters (%)")
        ax.set_title("Waterfall Plot: Longitudinal Response")
        ax.legend()
        ax.set_xticks(range(len(changes)))
        ax.set_xticklabels(case_ids, rotation=90, fontsize=7)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Waterfall plot saved to {output_path}")

    def _load_case(self, case_dir: Path) -> Tuple[torch.Tensor, tuple, np.ndarray]:
        """Load a case's NIfTI sequences into a tensor."""
        images = []
        affine = np.eye(4)
        spacing = (1.0, 1.0, 1.0)

        for seq in self.sequences:
            img_path = case_dir / f"{seq}.nii.gz"
            if not img_path.exists():
                raise FileNotFoundError(f"Missing {seq} at {img_path}")

            nii = nib.load(str(img_path))
            img = nii.get_fdata().astype(np.float32)

            if seq == self.sequences[0]:
                affine = nii.affine
                spacing = tuple(abs(float(x)) for x in nii.header.get_zooms()[:3])

            mean = img.mean()
            std = img.std()
            if std > 0:
                img = (img - mean) / std

            images.append(img)

        tensor = torch.from_numpy(np.stack(images, axis=0)).float()
        return tensor, spacing, affine
