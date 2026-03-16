"""
DICOM-SEG writer for creating DICOM Segmentation objects from segmentation masks.
Uses highdicom to produce DICOM-SEG that PACS viewers can display.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np


class DICOMSEGWriter:
    """
    Creates DICOM-SEG objects from segmentation masks.
    Wraps highdicom to produce standard DICOM Segmentation IODs.
    """

    def __init__(
        self,
        algorithm_name: str = "BrainMetScan",
        algorithm_version: str = "1.23.0",
        manufacturer: str = "BrainMetScan",
    ):
        self.algorithm_name = algorithm_name
        self.algorithm_version = algorithm_version
        self.manufacturer = manufacturer

    def write(
        self,
        mask: np.ndarray,
        source_dicom_files: List[str],
        output_path: str,
        segment_label: str = "Brain Metastasis",
        segment_description: str = "AI-detected brain metastasis segmentation",
    ) -> str:
        """
        Create a DICOM-SEG file from a segmentation mask and source DICOM series.

        Args:
            mask: Binary segmentation mask (D, H, W) or (H, W, D)
            source_dicom_files: List of paths to source DICOM files
            output_path: Path to write DICOM-SEG output
            segment_label: Label for the segmentation
            segment_description: Description of the segment

        Returns:
            Path to the written DICOM-SEG file
        """
        import pydicom
        from pydicom.sr.codedict import codes
        import highdicom as hd

        # Load source DICOM images
        source_images = [pydicom.dcmread(f) for f in sorted(source_dicom_files)]

        # Ensure mask is (n_frames, H, W) - one frame per slice
        if mask.ndim == 3:
            # Assume (D, H, W) ordering
            mask_frames = mask.astype(np.uint8)
        else:
            raise ValueError(f"Expected 3D mask, got shape {mask.shape}")

        # Validate slice count matches source DICOM series
        if mask_frames.shape[0] != len(source_images):
            raise ValueError(
                f"Mask has {mask_frames.shape[0]} slices but "
                f"{len(source_images)} DICOM files were provided"
            )

        # Define segment description
        segment = hd.seg.SegmentDescription(
            segment_number=1,
            segment_label=segment_label,
            segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
            segmented_property_type=codes.SCT.Neoplasm,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=hd.AlgorithmIdentificationSequence(
                name=self.algorithm_name,
                version=self.algorithm_version,
                family=codes.DCM.ArtificialIntelligence,
            ),
            tracking_uid=hd.UID(),
            tracking_id=segment_description,
        )

        # Create DICOM-SEG
        seg = hd.seg.Segmentation(
            source_images=source_images,
            pixel_array=mask_frames,
            segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
            segment_descriptions=[segment],
            series_instance_uid=hd.UID(),
            series_number=100,
            sop_instance_uid=hd.UID(),
            instance_number=1,
            manufacturer=self.manufacturer,
            manufacturer_model_name=self.algorithm_name,
            software_versions=[self.algorithm_version],
            device_serial_number="0",
        )

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        seg.save_as(str(output_path))

        return str(output_path)
