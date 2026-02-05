"""
Test-Time Augmentation (TTA) for improved inference
Applies multiple transformations and averages predictions
"""

import torch
import torch.nn.functional as F
import numpy as np


class TestTimeAugmentation:
    """
    Test-Time Augmentation wrapper for 3D medical image segmentation

    Applies multiple geometric transformations and averages predictions
    to improve robustness and accuracy

    Args:
        model: Trained segmentation model
        device: Device to run inference on
        num_rotations: Number of 90-degree rotations to try (0-3)
        use_flips: Whether to use flips (left-right, up-down, front-back)
        use_brightness: Whether to apply slight brightness variations
    """
    def __init__(
        self,
        model,
        device,
        num_rotations=4,
        use_flips=True,
        use_brightness=False
    ):
        self.model = model
        self.device = device
        self.num_rotations = num_rotations
        self.use_flips = use_flips
        self.use_brightness = use_brightness

    def _rotate_3d(self, tensor, k, dims):
        """Rotate tensor by k*90 degrees in specified plane"""
        if k == 0:
            return tensor
        return torch.rot90(tensor, k=k, dims=dims)

    def _flip_3d(self, tensor, dim):
        """Flip tensor along specified dimension"""
        return torch.flip(tensor, dims=[dim])

    def _brightness_adjust(self, tensor, factor):
        """Adjust brightness"""
        return tensor * factor

    @torch.no_grad()
    def predict(self, image, threshold=0.5):
        """
        Perform TTA prediction

        Args:
            image: Input image tensor (1, C, H, W, D)
            threshold: Threshold for binary prediction

        Returns:
            mask: Averaged prediction mask
            mask_binary: Thresholded binary mask
        """
        self.model.eval()

        predictions = []

        # Original prediction
        pred = torch.sigmoid(self.model(image.to(self.device)))
        predictions.append(pred.cpu())

        # Rotations in axial plane (dims 2,3 = H,W)
        for k in range(1, self.num_rotations):
            # Rotate input
            rotated = self._rotate_3d(image, k, dims=(2, 3))
            pred = torch.sigmoid(self.model(rotated.to(self.device)))
            # Rotate prediction back
            pred = self._rotate_3d(pred, -k, dims=(2, 3))
            predictions.append(pred.cpu())

        # Flips
        if self.use_flips:
            flip_dims = [2, 3, 4]  # H, W, D

            for dim in flip_dims:
                # Flip input
                flipped = self._flip_3d(image, dim)
                pred = torch.sigmoid(self.model(flipped.to(self.device)))
                # Flip prediction back
                pred = self._flip_3d(pred, dim)
                predictions.append(pred.cpu())

        # Brightness variations (only if enabled)
        if self.use_brightness:
            brightness_factors = [0.9, 1.1]

            for factor in brightness_factors:
                adjusted = self._brightness_adjust(image, factor)
                pred = torch.sigmoid(self.model(adjusted.to(self.device)))
                predictions.append(pred.cpu())

        # Average all predictions
        avg_pred = torch.stack(predictions).mean(dim=0)

        # Apply threshold
        mask_binary = (avg_pred > threshold).float()

        return avg_pred, mask_binary

    def predict_batch(self, images, threshold=0.5):
        """
        Predict on a batch of images

        Args:
            images: Batch of images (B, C, H, W, D)
            threshold: Threshold for binary prediction

        Returns:
            masks: List of averaged prediction masks
            masks_binary: List of thresholded binary masks
        """
        masks = []
        masks_binary = []

        for i in range(images.shape[0]):
            mask, mask_bin = self.predict(images[i:i+1], threshold)
            masks.append(mask)
            masks_binary.append(mask_bin)

        return masks, masks_binary


class MinimalTTA:
    """
    Lightweight TTA with only essential augmentations
    Faster than full TTA, still provides improvement

    Only applies:
    - Original
    - Horizontal flip
    - Vertical flip
    - 180-degree rotation
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(self, image, threshold=0.5):
        """Minimal TTA prediction"""
        self.model.eval()

        predictions = []

        # Original
        pred = torch.sigmoid(self.model(image.to(self.device)))
        predictions.append(pred.cpu())

        # Horizontal flip (dim 3 = W)
        flipped_h = torch.flip(image, dims=[3])
        pred = torch.sigmoid(self.model(flipped_h.to(self.device)))
        pred = torch.flip(pred, dims=[3])
        predictions.append(pred.cpu())

        # Vertical flip (dim 2 = H)
        flipped_v = torch.flip(image, dims=[2])
        pred = torch.sigmoid(self.model(flipped_v.to(self.device)))
        pred = torch.flip(pred, dims=[2])
        predictions.append(pred.cpu())

        # 180-degree rotation
        rotated = torch.rot90(image, k=2, dims=(2, 3))
        pred = torch.sigmoid(self.model(rotated.to(self.device)))
        pred = torch.rot90(pred, k=-2, dims=(2, 3))
        predictions.append(pred.cpu())

        # Average
        avg_pred = torch.stack(predictions).mean(dim=0)
        mask_binary = (avg_pred > threshold).float()

        return avg_pred, mask_binary


class AdaptiveTTA:
    """
    Adaptive TTA that selects augmentations based on uncertainty

    Starts with minimal TTA, adds more augmentations if prediction
    uncertainty is high
    """
    def __init__(self, model, device, uncertainty_threshold=0.3):
        self.model = model
        self.device = device
        self.uncertainty_threshold = uncertainty_threshold
        self.minimal_tta = MinimalTTA(model, device)
        self.full_tta = TestTimeAugmentation(model, device)

    def _compute_uncertainty(self, predictions):
        """
        Compute prediction uncertainty
        High uncertainty = predictions vary a lot

        Args:
            predictions: List of prediction tensors

        Returns:
            uncertainty: Scalar uncertainty measure
        """
        pred_stack = torch.stack(predictions)
        variance = pred_stack.var(dim=0)
        return variance.mean().item()

    @torch.no_grad()
    def predict(self, image, threshold=0.5):
        """
        Adaptive TTA prediction

        Args:
            image: Input image
            threshold: Binary threshold

        Returns:
            mask: Averaged prediction
            mask_binary: Binary mask
        """
        # Start with minimal TTA
        mask, mask_bin = self.minimal_tta.predict(image, threshold)

        # Compute uncertainty from minimal predictions
        self.model.eval()
        predictions = []

        # Get individual predictions for uncertainty estimation
        pred = torch.sigmoid(self.model(image.to(self.device))).cpu()
        predictions.append(pred)

        flipped_h = torch.flip(image, dims=[3])
        pred = torch.sigmoid(self.model(flipped_h.to(self.device))).cpu()
        pred = torch.flip(pred, dims=[3])
        predictions.append(pred)

        uncertainty = self._compute_uncertainty(predictions)

        # If uncertainty is high, use full TTA
        if uncertainty > self.uncertainty_threshold:
            mask, mask_bin = self.full_tta.predict(image, threshold)

        return mask, mask_bin


def ensemble_predictions(predictions, method='mean'):
    """
    Ensemble multiple predictions

    Args:
        predictions: List of prediction tensors
        method: Ensemble method ('mean', 'max', 'median', 'weighted_mean')

    Returns:
        Ensembled prediction
    """
    pred_stack = torch.stack(predictions)

    if method == 'mean':
        return pred_stack.mean(dim=0)

    elif method == 'max':
        return pred_stack.max(dim=0)[0]

    elif method == 'median':
        return pred_stack.median(dim=0)[0]

    elif method == 'weighted_mean':
        # Weight by confidence (distance from 0.5)
        confidence = torch.abs(pred_stack - 0.5)
        weights = confidence / confidence.sum(dim=0, keepdim=True)
        return (pred_stack * weights).sum(dim=0)

    else:
        raise ValueError(f"Unknown ensemble method: {method}")


if __name__ == "__main__":
    print("Test-Time Augmentation module loaded successfully!")

    # Example usage
    print("\nExample usage:")
    print("  from tta import TestTimeAugmentation")
    print("  tta = TestTimeAugmentation(model, device)")
    print("  mask, mask_binary = tta.predict(image)")
