"""
Test fixtures for BrainMetScan test suite.
Provides synthetic volumes, dummy models, and temporary data.
"""

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from src.segmentation.unet import LightweightUNet3D


@pytest.fixture
def synthetic_volume():
    """
    Create a 4-channel synthetic volume with a known spherical lesion.
    Returns (image_tensor, mask_array, lesion_center, lesion_radius)
    """
    size = 32
    center = (16, 16, 16)
    radius = 5

    # Create 4-channel image with some signal
    rng = np.random.default_rng(42)
    image = rng.standard_normal((4, size, size, size)).astype(np.float32)

    # Create spherical lesion mask
    mask = np.zeros((size, size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                dist = np.sqrt(
                    (i - center[0]) ** 2 + (j - center[1]) ** 2 + (k - center[2]) ** 2
                )
                if dist < radius:
                    mask[i, j, k] = 1.0
                    # Make lesion brighter in t1_gd channel (index 1)
                    image[1, i, j, k] += 3.0

    return torch.from_numpy(image), mask, center, radius


@pytest.fixture
def synthetic_volume_two_lesions():
    """Synthetic volume with two lesions of different sizes."""
    size = 64
    rng = np.random.default_rng(42)
    image = rng.standard_normal((4, size, size, size)).astype(np.float32)
    mask = np.zeros((size, size, size), dtype=np.float32)

    # Lesion 1: large (radius 8)
    c1 = (20, 20, 20)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if np.sqrt((i - c1[0]) ** 2 + (j - c1[1]) ** 2 + (k - c1[2]) ** 2) < 8:
                    mask[i, j, k] = 1.0

    # Lesion 2: small (radius 3)
    c2 = (45, 45, 45)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if np.sqrt((i - c2[0]) ** 2 + (j - c2[1]) ** 2 + (k - c2[2]) ** 2) < 3:
                    mask[i, j, k] = 1.0

    return torch.from_numpy(image), mask


@pytest.fixture
def dummy_model():
    """Create a small UNet model for testing (CPU only)."""
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=4,  # Very small for fast tests
        depth=2,
        dropout_p=0.0,
        use_attention=True,
        use_residual=True,
    )
    model.eval()
    return model


@pytest.fixture
def dummy_checkpoint(dummy_model, tmp_path):
    """Save a dummy model checkpoint and return its path."""
    checkpoint_path = tmp_path / "dummy_model.pth"
    torch.save(
        {
            "model_state_dict": dummy_model.state_dict(),
            "epoch": 1,
            "val_dice": 0.5,
            "args": {
                "base_channels": 4,
                "depth": 2,
                "dropout": 0.0,
                "use_attention": True,
                "use_residual": True,
                "model_type": "lightweight",
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


@pytest.fixture
def temp_nifti_case(synthetic_volume, tmp_path):
    """
    Create a temporary directory with NIfTI files for a test case.
    Returns path to the case directory.
    """
    image_tensor, mask, _, _ = synthetic_volume
    case_dir = tmp_path / "Mets_TEST"
    case_dir.mkdir()

    sequences = ["t1_pre", "t1_gd", "flair", "t2"]
    for i, seq in enumerate(sequences):
        nii = nib.Nifti1Image(image_tensor[i].numpy(), np.eye(4))
        nib.save(nii, str(case_dir / f"{seq}.nii.gz"))

    # Save mask
    nii = nib.Nifti1Image(mask, np.eye(4))
    nib.save(nii, str(case_dir / "seg.nii.gz"))

    return case_dir
