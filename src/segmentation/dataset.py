"""
Dataset loader for BrainMetShare MRI data
Supports multi-modal input and efficient 3D patch-based loading for consumer GPUs
"""

import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Optional
from scipy.ndimage import zoom


class BrainMetDataset(Dataset):
    """
    Dataset for brain metastasis MRI segmentation

    Args:
        data_dir: Path to train or test directory
        sequences: List of MRI sequences to load (default: all 4)
        patch_size: Size of 3D patches for training (None = full volume)
        target_size: Target size to resize all volumes to (H, W, D). Required for batching.
        transform: Optional transforms
        metadata_path: Path to metadata CSV file
        augment: Whether to apply data augmentation (default: False)
        augmentation_prob: Probability for each augmentation (default: 0.3)
    """

    def __init__(
        self,
        data_dir: str,
        sequences: List[str] = ['t1_pre', 't1_gd', 'flair', 'bravo'],
        patch_size: Optional[Tuple[int, int, int]] = None,
        target_size: Tuple[int, int, int] = (128, 128, 128),
        transform=None,
        metadata_path: Optional[str] = None,
        augment: bool = False,
        augmentation_prob: float = 0.3
    ):
        self.data_dir = Path(data_dir)
        self.sequences = sequences
        self.patch_size = patch_size
        self.target_size = target_size
        self.transform = transform

        # Get all case directories (support multiple naming conventions)
        valid_prefixes = ('Mets_', 'UCSF_', 'BraTS_', 'Yale_', 'BMS_')
        self.cases = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith(valid_prefixes)])

        # Load metadata if provided
        self.metadata = None
        if metadata_path and os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
            self.metadata['Patient ID'] = self.metadata['Patient ID'].astype(str)

        # Check if this is training data (has segmentation masks)
        self.has_masks = (self.data_dir.name == 'train')

        # Create augmentation pipeline if requested
        self.augmentation = None
        if augment:
            from augmentation import AugmentationPipeline
            self.augmentation = AugmentationPipeline(augmentation_probability=augmentation_prob)
            print(f"Augmentation enabled with probability: {augmentation_prob}")

        print(f"Loaded {len(self.cases)} cases from {data_dir}")
        print(f"Sequences: {sequences}")
        print(f"Has segmentation masks: {self.has_masks}")

    def __len__(self):
        return len(self.cases)

    def _load_nifti(self, path: Path) -> np.ndarray:
        """Load NIfTI file and return numpy array (memory-efficient)"""
        nii = nib.load(str(path))
        # Load directly as float32 to avoid float64 intermediate
        return np.asarray(nii.dataobj, dtype=np.float32)

    def _load_numpy(self, path: Path) -> np.ndarray:
        """Load numpy file (much faster than NIfTI)"""
        if path.suffix == '.npz':
            return np.load(str(path))['data'].astype(np.float32)
        return np.load(str(path)).astype(np.float32)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Z-score normalization per volume (memory-efficient, no temp arrays)"""
        # Compute mean and std without creating temporary arrays
        # Using: std = sqrt(E[x²] - E[x]²)
        n = img.size
        mean = img.sum() / n
        # Compute sum of squares in chunks to avoid large temp arrays
        sum_sq = 0.0
        chunk_size = 1024 * 1024  # 1M elements at a time
        flat = img.ravel()
        for i in range(0, n, chunk_size):
            chunk = flat[i:i + chunk_size]
            sum_sq += (chunk * chunk).sum()
        variance = (sum_sq / n) - (mean * mean)
        std = np.sqrt(max(variance, 0))  # max to handle numerical issues

        if std > 0:
            img -= mean
            img /= std
        return img

    def _resize_volume(self, img: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """Resize 3D volume to target size using trilinear interpolation"""
        if self.target_size is None:
            return img

        current_shape = img.shape
        target_shape = self.target_size

        # Calculate zoom factors
        zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]

        # Use order=0 (nearest neighbor) for masks to preserve binary values
        # Use order=1 (bilinear) for images
        order = 0 if is_mask else 1

        return zoom(img, zoom_factors, order=order, mode='nearest')

    def __getitem__(self, idx: int):
        case_dir = self.cases[idx]
        case_id = case_dir.name

        # Load all sequences (try .npz/.npy first, fall back to .nii.gz)
        images = []
        for seq in self.sequences:
            npz_path = case_dir / f"{seq}.npz"
            npy_path = case_dir / f"{seq}.npy"
            nii_path = case_dir / f"{seq}.nii.gz"

            if npz_path.exists():
                img = self._load_numpy(npz_path)
            elif npy_path.exists():
                img = self._load_numpy(npy_path)
            elif nii_path.exists():
                img = self._load_nifti(nii_path)
            else:
                raise FileNotFoundError(f"Missing sequence {seq} for case {case_id}")

            img = self._resize_volume(img, is_mask=False)  # Resize to target size
            img = self._normalize(img)
            images.append(img)

        # Stack sequences along channel dimension -> (C, H, W, D)
        images = np.stack(images, axis=0)

        # Load segmentation mask if available (try .npz/.npy first, fall back to .nii.gz)
        mask = None
        if self.has_masks:
            npz_mask = case_dir / "seg.npz"
            npy_mask = case_dir / "seg.npy"
            nii_mask = case_dir / "seg.nii.gz"

            if npz_mask.exists():
                mask = self._load_numpy(npz_mask)
            elif npy_mask.exists():
                mask = self._load_numpy(npy_mask)
            elif nii_mask.exists():
                mask = self._load_nifti(nii_mask)

            if mask is not None:
                mask = self._resize_volume(mask, is_mask=True)  # Resize mask
                # Convert to binary (0 or 1)
                mask = (mask > 0).astype(np.float32)
                mask = np.expand_dims(mask, axis=0)  # Add channel dim

        # Apply transforms if provided
        if self.transform:
            sample = {'image': images, 'mask': mask, 'case_id': case_id}
            sample = self.transform(sample)
            images = sample['image']
            mask = sample['mask']

            # Convert MetaTensor to numpy array if needed (MONAI transforms return MetaTensor)
            if hasattr(images, 'numpy'):
                images = images.numpy()
            elif hasattr(images, 'array'):
                images = images.array
            elif not isinstance(images, np.ndarray):
                images = np.array(images)

            if mask is not None:
                if hasattr(mask, 'numpy'):
                    mask = mask.numpy()
                elif hasattr(mask, 'array'):
                    mask = mask.array
                elif not isinstance(mask, np.ndarray):
                    mask = np.array(mask)

        # Extract patch if specified (for memory efficiency)
        if self.patch_size is not None and mask is not None:
            images, mask = self._extract_random_patch(images, mask)

        # Apply augmentation after patch extraction (if enabled)
        if self.augmentation is not None and mask is not None:
            sample = {
                'image': images,  # numpy array
                'mask': mask      # numpy array
            }
            sample = self.augmentation(sample)
            images = sample['image']
            mask = sample['mask']

            # Convert MetaTensor to numpy array if needed (MONAI transforms return MetaTensor)
            if hasattr(images, 'array'):
                images = images.array
            elif not isinstance(images, np.ndarray):
                images = np.array(images)

            if hasattr(mask, 'array'):
                mask = mask.array
            elif not isinstance(mask, np.ndarray):
                mask = np.array(mask)

        # Convert to torch tensors
        images = torch.from_numpy(images).float()

        if mask is not None:
            mask = torch.from_numpy(mask).float()
            return images, mask, case_id
        else:
            return images, case_id

    def _extract_random_patch(
        self,
        images: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract random 3D patch from volume
        Biased towards patches containing metastases
        Ensures all patches are exactly patch_size dimensions
        """
        C, H, W, D = images.shape
        ph, pw, pd = self.patch_size

        # Ensure volume is large enough for patch extraction
        if H < ph or W < pw or D < pd:
            # Pad if volume is smaller than patch size
            pad_h = max(0, ph - H)
            pad_w = max(0, pw - W)
            pad_d = max(0, pd - D)

            images = np.pad(images, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            mask = np.pad(mask, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            C, H, W, D = images.shape

        # Find foreground voxels (where mask > 0)
        foreground = np.where(mask[0] > 0)

        # 90% chance to sample from foreground, 10% random (IMPROVED for better anomaly detection)
        if len(foreground[0]) > 0 and np.random.rand() > 0.1:
            # Sample center from foreground
            idx = np.random.randint(len(foreground[0]))
            ch = foreground[0][idx]
            cw = foreground[1][idx]
            cd = foreground[2][idx]
        else:
            # Random center
            ch = np.random.randint(ph//2, H - ph//2)
            cw = np.random.randint(pw//2, W - pw//2)
            cd = np.random.randint(pd//2, D - pd//2)

        # Calculate start coordinates centered on chosen point
        h_start = ch - ph//2
        w_start = cw - pw//2
        d_start = cd - pd//2

        # Ensure patch stays within bounds by adjusting start coordinates
        h_start = max(0, min(h_start, H - ph))
        w_start = max(0, min(w_start, W - pw))
        d_start = max(0, min(d_start, D - pd))

        # Extract patch with exact dimensions
        h_end = h_start + ph
        w_end = w_start + pw
        d_end = d_start + pd

        img_patch = images[:, h_start:h_end, w_start:w_end, d_start:d_end]
        mask_patch = mask[:, h_start:h_end, w_start:w_end, d_start:d_end]

        # Verify patch size (should always be true now)
        assert img_patch.shape == (C, ph, pw, pd), f"Patch shape mismatch: {img_patch.shape} vs expected {(C, ph, pw, pd)}"
        assert mask_patch.shape == (1, ph, pw, pd), f"Mask shape mismatch: {mask_patch.shape} vs expected {(1, ph, pw, pd)}"

        return img_patch, mask_patch

    def get_metadata(self, case_id: str):
        """Get metadata for a specific case"""
        if self.metadata is None:
            return None

        # Extract patient ID from case_id (e.g., "Mets_040" -> "40")
        patient_id = case_id.split('_')[1].lstrip('0') or '0'

        row = self.metadata[self.metadata['Patient ID'] == patient_id]
        if len(row) > 0:
            return row.iloc[0].to_dict()
        return None


def get_train_val_split(data_source, val_ratio: float = 0.15, val_split: float = None, seed: int = 42):
    """
    Create train/validation split from training directory or dataset length

    Args:
        data_source: Either path to training directory (str) or dataset length (int)
        val_ratio: Fraction of data for validation (legacy parameter)
        val_split: Fraction of data for validation (alternative name)
        seed: Random seed for reproducibility

    Returns:
        If data_source is str: train_cases, val_cases (Lists of case directory paths)
        If data_source is int: train_indices, val_indices (Lists of indices)
    """
    import random
    random.seed(seed)

    # Use val_split if provided, otherwise use val_ratio
    split_fraction = val_split if val_split is not None else val_ratio

    # Handle both string (directory) and int (dataset length) inputs
    if isinstance(data_source, str):
        # Original behavior: split directory paths
        data_dir = Path(data_source)
        cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('Mets_')])

        # Shuffle and split
        cases_shuffled = cases.copy()
        random.shuffle(cases_shuffled)

        val_size = int(len(cases) * split_fraction)
        val_cases = cases_shuffled[:val_size]
        train_cases = cases_shuffled[val_size:]

        print(f"Training cases: {len(train_cases)}")
        print(f"Validation cases: {len(val_cases)}")

        return train_cases, val_cases

    elif isinstance(data_source, int):
        # New behavior: split indices
        n_samples = data_source
        indices = list(range(n_samples))
        random.shuffle(indices)

        val_size = int(n_samples * split_fraction)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")

        return train_indices, val_indices

    else:
        raise ValueError(f"data_source must be str or int, got {type(data_source)}")


if __name__ == "__main__":
    # Test dataset loading
    print("Testing BrainMetDataset...")

    dataset = BrainMetDataset(
        data_dir="../train",
        patch_size=(96, 96, 96),
        metadata_path="../metadata.csv"
    )

    print(f"\nDataset size: {len(dataset)}")

    # Load first sample
    images, mask, case_id = dataset[0]
    print(f"\nFirst sample: {case_id}")
    print(f"Image shape: {images.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"Mask unique values: {torch.unique(mask)}")

    # Get metadata
    metadata = dataset.get_metadata(case_id)
    if metadata:
        print(f"Metadata: {metadata}")
