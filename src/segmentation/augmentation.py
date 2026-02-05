"""
Data augmentation pipeline for brain metastasis segmentation
Uses MONAI transforms for 3D medical imaging
"""

from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandGaussianSharpend,
)
import numpy as np


class AugmentationPipeline:
    """
    Training augmentation pipeline using MONAI

    Applies realistic, clinically-relevant augmentations that reflect
    MRI acquisition variability for brain metastasis images

    Args:
        augmentation_probability: Probability of applying each transform (default: 0.3)
    """

    def __init__(self, augmentation_probability=0.3):
        """
        Initialize augmentation pipeline

        Args:
            augmentation_probability: Probability for each augmentation (default: 0.3)
        """
        self.prob = augmentation_probability

        self.train_transforms = Compose([
            # Geometric augmentations (applied to both image and mask)
            RandFlipd(
                keys=["image", "mask"],
                spatial_axis=[0, 1, 2],  # All axes
                prob=0.5,  # Higher probability for flipping
            ),
            RandRotate90d(
                keys=["image", "mask"],
                prob=self.prob,
                spatial_axes=(0, 1),  # Axial rotation
            ),
            RandAffined(
                keys=["image", "mask"],
                prob=self.prob,
                rotate_range=[0.1, 0.1, 0.1],  # ~6 degrees in radians
                scale_range=[0.1, 0.1, 0.1],   # ±10% scaling
                mode=["bilinear", "nearest"],  # bilinear for image, nearest for mask
                padding_mode="border",
            ),

            # Intensity augmentations (only on image, not mask)
            RandGaussianNoised(
                keys=["image"],
                prob=0.2,
                mean=0.0,
                std=0.1,
            ),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.2,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            RandScaleIntensityd(
                keys=["image"],
                factors=0.2,  # ±20%
                prob=self.prob,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.1,
                prob=self.prob,
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=self.prob,
                gamma=(0.8, 1.2),
            ),
            RandGaussianSharpend(
                keys=["image"],
                prob=0.2,
                sigma1_x=(0.5, 1.0),
                sigma1_y=(0.5, 1.0),
                sigma1_z=(0.5, 1.0),
                sigma2_x=(0.5, 1.0),
                sigma2_y=(0.5, 1.0),
                sigma2_z=(0.5, 1.0),
            ),
        ])

    def __call__(self, sample):
        """
        Apply transforms to sample dictionary

        Args:
            sample: Dictionary with 'image' and 'mask' keys (numpy arrays)

        Returns:
            Augmented sample dictionary
        """
        # MONAI expects dict with keys
        data = {
            "image": sample["image"],
            "mask": sample["mask"]
        }

        # Apply transforms
        augmented = self.train_transforms(data)

        # Update sample
        sample["image"] = augmented["image"]
        sample["mask"] = augmented["mask"]

        return sample


class ValidationAugmentation:
    """
    No augmentation for validation
    Just format conversion to maintain API compatibility
    """
    def __call__(self, sample):
        """Return sample unchanged"""
        return sample


if __name__ == "__main__":
    # Test augmentation pipeline
    print("Testing Augmentation Pipeline...")

    # Create dummy data
    dummy_image = np.random.randn(4, 96, 96, 96).astype(np.float32)
    dummy_mask = np.random.randint(0, 2, (1, 96, 96, 96)).astype(np.float32)

    sample = {
        "image": dummy_image,
        "mask": dummy_mask
    }

    # Test augmentation
    aug_pipeline = AugmentationPipeline(augmentation_probability=0.3)

    print(f"Input image shape: {sample['image'].shape}")
    print(f"Input mask shape: {sample['mask'].shape}")

    augmented_sample = aug_pipeline(sample)

    print(f"Augmented image shape: {augmented_sample['image'].shape}")
    print(f"Augmented mask shape: {augmented_sample['mask'].shape}")
    print(f"Image value range: [{augmented_sample['image'].min():.3f}, {augmented_sample['image'].max():.3f}]")
    print(f"Mask unique values: {np.unique(augmented_sample['mask'])}")

    print("\n✓ Augmentation pipeline test passed!")
