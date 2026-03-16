"""
DICOM ingestion handler for converting DICOM series to tensors.
Supports sorting by SeriesInstanceUID, sequence identification, and NIfTI conversion.
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch


class DICOMIngester:
    """
    Ingests DICOM files, identifies MRI sequences, and converts to tensors.
    """

    # Heuristic mapping of SeriesDescription keywords to sequence names
    SEQUENCE_KEYWORDS = {
        "t1_pre": ["t1 pre", "t1_pre", "t1w pre", "t1 without", "t1w_pre", "pre-contrast", "precontrast"],
        "t1_gd": ["t1 post", "t1_gd", "t1w post", "t1 gad", "t1+c", "t1_post", "post-contrast",
                   "postcontrast", "t1 with", "t1w_gd", "t1ce", "t1 contrast", "gd-t1", "bravo"],
        "flair": ["flair", "t2_flair", "t2flair", "dark-fluid"],
        "t2": ["t2w", "t2 ", "t2_", "t2-weighted"],
    }

    def ingest_dicom_files(self, file_paths: List[str]) -> Dict[str, sitk.Image]:
        """
        Sort DICOM files by SeriesInstanceUID and convert each series to a SimpleITK Image.

        Args:
            file_paths: List of paths to DICOM files

        Returns:
            Dict mapping identified sequence name -> SimpleITK Image
        """
        # Group files by SeriesInstanceUID
        series_files: Dict[str, List[str]] = {}
        series_descriptions: Dict[str, str] = {}

        for fp in file_paths:
            try:
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(fp))
                reader.LoadPrivateTagsOn()
                reader.ReadImageInformation()

                series_uid = reader.GetMetaData("0020|000e").strip()
                series_desc = reader.GetMetaData("0008|103e").strip() if reader.HasMetaDataKey("0008|103e") else ""

                series_files.setdefault(series_uid, []).append(str(fp))
                series_descriptions[series_uid] = series_desc
            except Exception:
                continue

        # Convert each series and identify sequence
        result = {}
        for series_uid, files in series_files.items():
            desc = series_descriptions.get(series_uid, "")
            seq_name = self._identify_sequence(desc)
            if seq_name is None:
                continue

            try:
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(sorted(files))
                image = reader.Execute()
                result[seq_name] = image
            except Exception:
                continue

        return result

    def _identify_sequence(self, series_description: str) -> Optional[str]:
        """
        Heuristically map a DICOM SeriesDescription to a sequence name.

        Args:
            series_description: DICOM SeriesDescription tag value

        Returns:
            Sequence name ('t1_pre', 't1_gd', 'flair', 't2') or None
        """
        desc_lower = series_description.lower()

        for seq_name, keywords in self.SEQUENCE_KEYWORDS.items():
            for kw in keywords:
                if kw in desc_lower:
                    return seq_name

        return None

    def load_nifti_as_tensor(
        self,
        nifti_paths: Dict[str, str],
        expected_sequences: List[str] = None,
    ) -> Tuple[torch.Tensor, tuple]:
        """
        Load NIfTI files into a (4, H, W, D) tensor with z-score normalization.

        Args:
            nifti_paths: Dict mapping sequence name -> NIfTI file path
            expected_sequences: Ordered list of sequences to load
                                (default: ['t1_pre', 't1_gd', 'flair', 't2'])

        Returns:
            Tuple of (tensor [4, H, W, D], voxel_spacing tuple)
        """
        if expected_sequences is None:
            expected_sequences = ["t1_pre", "t1_gd", "flair", "t2"]

        images = []
        spacing = (1.0, 1.0, 1.0)

        for seq in expected_sequences:
            path = nifti_paths.get(seq)
            if path is None:
                raise ValueError(f"Missing required sequence: {seq}")

            sitk_img = sitk.ReadImage(str(path))
            # SimpleITK returns spacing as (x, y, z) but GetArrayFromImage
            # returns data as (z, y, x), so reverse spacing to match array axes
            sitk_spacing = sitk_img.GetSpacing()
            spacing = (sitk_spacing[2], sitk_spacing[1], sitk_spacing[0])
            arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)

            # Z-score normalization per volume
            mean = arr.mean()
            std = arr.std()
            if std > 0:
                arr = (arr - mean) / std

            images.append(arr)

        tensor = torch.from_numpy(np.stack(images, axis=0)).float()
        return tensor, spacing

    def dicom_to_nifti(self, dicom_images: Dict[str, sitk.Image], output_dir: str) -> Dict[str, str]:
        """
        Convert SimpleITK Images from DICOM to NIfTI files.

        Args:
            dicom_images: Dict mapping sequence name -> SimpleITK Image
            output_dir: Directory to write NIfTI files

        Returns:
            Dict mapping sequence name -> NIfTI file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        nifti_paths = {}
        for seq_name, image in dicom_images.items():
            out_path = output_dir / f"{seq_name}.nii.gz"
            sitk.WriteImage(image, str(out_path))
            nifti_paths[seq_name] = str(out_path)

        return nifti_paths
