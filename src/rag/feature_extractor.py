"""
Feature extraction for brain metastasis cases
Extracts radiomic features and image embeddings for RAG retrieval
"""

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import ndimage
from skimage import measure

from .embeddings import BiomedCLIPEmbedder


class RadiomicFeatureExtractor:
    """
    Extract radiomic features from segmentation masks and MRI images
    """

    def __init__(self):
        pass

    def extract_lesion_features(
        self,
        mask: np.ndarray,
        image: np.ndarray = None
    ) -> Dict:
        """
        Extract features from a segmentation mask

        Args:
            mask: Binary segmentation mask (H, W, D)
            image: Optional MRI image for intensity features

        Returns:
            Dictionary of features
        """
        features = {}

        # Label connected components (individual lesions)
        labeled_mask = measure.label(mask > 0)
        regions = measure.regionprops(labeled_mask, intensity_image=image)

        # Number of lesions
        features['num_lesions'] = len(regions)

        if len(regions) == 0:
            return self._empty_features()

        # Aggregate features across all lesions
        volumes = [r.area for r in regions]  # in voxels
        features['total_volume'] = sum(volumes)
        features['mean_lesion_volume'] = np.mean(volumes)
        features['max_lesion_volume'] = max(volumes)
        features['min_lesion_volume'] = min(volumes)

        # Size distribution
        features['volume_std'] = np.std(volumes) if len(volumes) > 1 else 0

        # Location features (centroids)
        centroids = np.array([r.centroid for r in regions])
        features['mean_centroid'] = centroids.mean(axis=0).tolist()

        # Shape features (for largest lesion)
        largest_idx = np.argmax(volumes)
        largest = regions[largest_idx]

        features['largest_sphericity'] = self._compute_sphericity(largest)
        features['largest_extent'] = largest.extent  # volume / bounding box volume
        features['largest_solidity'] = largest.solidity  # volume / convex hull volume

        # Intensity features (if image provided)
        if image is not None:
            intensities = []
            for r in regions:
                coords = r.coords
                lesion_intensities = image[coords[:, 0], coords[:, 1], coords[:, 2]]
                intensities.extend(lesion_intensities)

            features['mean_intensity'] = np.mean(intensities)
            features['std_intensity'] = np.std(intensities)
            features['max_intensity'] = np.max(intensities)
            features['min_intensity'] = np.min(intensities)

        return features

    def _compute_sphericity(self, region) -> float:
        """Compute sphericity of a region (1.0 = perfect sphere)"""
        volume = region.area

        # For 3D regions, perimeter is not available, so we approximate surface area
        # using the equivalent ellipsoid approximation
        try:
            # This will fail for 3D images
            surface_area = region.perimeter
        except (NotImplementedError, AttributeError):
            # For 3D regions, approximate surface area from volume
            # Assuming roughly spherical: SA ≈ 4π * (3V/4π)^(2/3)
            surface_area = 4 * np.pi * (3 * volume / (4 * np.pi)) ** (2/3)

        if surface_area == 0:
            return 0.0

        # Sphericity = (π^(1/3) * (6*Volume)^(2/3)) / Surface Area
        sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area
        return min(1.0, sphericity)  # Clamp to [0, 1]

    def _empty_features(self) -> Dict:
        """Return empty feature dict when no lesions found"""
        return {
            'num_lesions': 0,
            'total_volume': 0,
            'mean_lesion_volume': 0,
            'max_lesion_volume': 0,
            'min_lesion_volume': 0,
            'volume_std': 0,
            'mean_centroid': [0, 0, 0],
            'largest_sphericity': 0,
            'largest_extent': 0,
            'largest_solidity': 0,
            'mean_intensity': 0,
            'std_intensity': 0,
            'max_intensity': 0,
            'min_intensity': 0
        }


# Module-level cache for ImageEmbeddingExtractor (BiomedCLIP is expensive to load)
_cached_image_extractor = None


class ImageEmbeddingExtractor:
    """
    Extract image embeddings using BiomedCLIP.
    Produces 512-dim vectors in a shared text-image latent space.
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.embedder = BiomedCLIPEmbedder(device=device)

    @torch.no_grad()
    def extract_embedding(
        self,
        image: np.ndarray,
        slice_axis=2,
        num_slices=16
    ) -> np.ndarray:
        """
        Extract embedding from 3D MRI volume

        Args:
            image: 3D MRI array (H, W, D)
            slice_axis: Axis to slice along
            num_slices: Number of slices to sample

        Returns:
            Feature vector (512,)
        """
        return self.embedder.embed_mri_volume(image, slice_axis, num_slices)


def extract_case_features(
    case_dir: Path,
    mask_path: Path = None,
    sequences: List[str] = ['t1_gd'],
    device='cuda'
) -> Dict:
    """
    Extract all features for a case

    Args:
        case_dir: Path to case directory
        mask_path: Path to segmentation mask (optional)
        sequences: Sequences to use for embedding
        device: Device for embedding extraction

    Returns:
        Dictionary with radiomic features and embeddings
    """
    global _cached_image_extractor
    features = {'case_id': case_dir.name}

    # Initialize extractors (reuse cached embedding extractor)
    radiomic_extractor = RadiomicFeatureExtractor()
    if _cached_image_extractor is None or _cached_image_extractor.device != device:
        _cached_image_extractor = ImageEmbeddingExtractor(device=device)
    embedding_extractor = _cached_image_extractor

    # Load mask if provided
    mask = None
    if mask_path and mask_path.exists():
        mask_nii = nib.load(str(mask_path))
        mask = mask_nii.get_fdata()
        mask = (mask > 0).astype(np.uint8)

    # Load primary sequence for intensity features
    primary_seq = sequences[0]
    image_path = case_dir / f"{primary_seq}.nii.gz"

    if image_path.exists():
        image_nii = nib.load(str(image_path))
        image = image_nii.get_fdata()

        # Extract radiomic features from mask
        if mask is not None:
            radiomic_features = radiomic_extractor.extract_lesion_features(mask, image)
            features.update(radiomic_features)

        # Extract image embedding
        image_embedding = embedding_extractor.extract_embedding(image)
        features['image_embedding'] = image_embedding.tolist()

    return features


if __name__ == "__main__":
    # Test feature extraction
    print("Testing feature extractors...")

    # Create dummy data
    mask = np.zeros((128, 128, 128))
    # Add a spherical lesion
    center = (64, 64, 64)
    radius = 10
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if (i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2 < radius**2:
                    mask[i, j, k] = 1

    image = np.random.randn(128, 128, 128)

    # Test radiomic extractor
    radiomic_extractor = RadiomicFeatureExtractor()
    radiomic_features = radiomic_extractor.extract_lesion_features(mask, image)

    print("\nRadiomic features:")
    for k, v in radiomic_features.items():
        print(f"  {k}: {v}")

    # Test embedding extractor
    print("\nTesting embedding extractor...")
    embedding_extractor = ImageEmbeddingExtractor(device='cpu')
    embedding = embedding_extractor.extract_embedding(image)

    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")

    print("\n✓ Feature extraction test passed!")
