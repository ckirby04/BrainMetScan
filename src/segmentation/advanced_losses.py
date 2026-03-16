"""
Advanced loss functions optimized for small lesion detection
Specifically designed for brain metastasis segmentation with tiny targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TverskyLoss(nn.Module):
    """
    Tversky Loss - better for handling false negatives in small lesion detection

    Args:
        alpha: Weight for false positives (lower = more FP tolerance)
        beta: Weight for false negatives (higher = penalize missing lesions more)
        smooth: Smoothing constant
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()

        tversky_index = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)

        return 1 - tversky_index


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance
    Focuses learning on hard examples (small lesions)

    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # pred is logits, use binary_cross_entropy_with_logits (safe for autocast)
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Focal loss calculation using logits
        bce = F.binary_cross_entropy_with_logits(pred_flat, target_flat, reduction='none')

        # Modulating factor - need sigmoid for probability
        pred_prob = torch.sigmoid(pred_flat)
        pt = torch.where(target_flat == 1, pred_prob, 1 - pred_prob)
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting
        alpha_weight = torch.where(target_flat == 1, self.alpha, 1 - self.alpha)

        loss = alpha_weight * focal_weight * bce

        return loss.mean()


class ComboLoss(nn.Module):
    """
    Combination of Dice and Weighted BCE for small lesion detection
    Optimized based on: https://arxiv.org/abs/1805.02798

    Args:
        alpha: Weight for Dice component
        beta: Weight for BCE component
        ce_ratio: Weight for positive class in BCE
    """
    def __init__(self, alpha=0.5, beta=0.5, ce_ratio=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_ratio = ce_ratio

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)

        # Dice component
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + 1) / (pred_flat.sum() + target_flat.sum() + 1)

        # Weighted BCE - use binary_cross_entropy_with_logits (safe for autocast)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Weight positive examples more
        weights = torch.where(target == 1, self.ce_ratio, 1 - self.ce_ratio)
        weighted_bce = (weights * bce).mean()

        # Combined
        combo = self.alpha * (1 - dice) + self.beta * weighted_bce

        return combo


class SensitivitySpecificityLoss(nn.Module):
    """
    Loss that explicitly balances sensitivity and specificity
    Crucial for small lesion detection where we want high recall

    Args:
        r: Weight for sensitivity (higher = prioritize recall)
        smooth: Smoothing constant
    """
    def __init__(self, r=0.7, smooth=1.0):
        super().__init__()
        self.r = r
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        TN = ((1 - target_flat) * (1 - pred_flat)).sum()
        FN = (target_flat * (1 - pred_flat)).sum()

        # Sensitivity (Recall) - what % of actual lesions we detect
        sensitivity = (TP + self.smooth) / (TP + FN + self.smooth)

        # Specificity - what % of negative pixels we correctly classify
        specificity = (TN + self.smooth) / (TN + FP + self.smooth)

        # Weighted combination
        loss = 1 - (self.r * sensitivity + (1 - self.r) * specificity)

        return loss


class AsymmetricFocalTverskyLoss(nn.Module):
    """
    Asymmetric Focal Tversky Loss optimized for tiny lesions
    Combines Tversky (handles imbalance) with Focal (focuses on hard examples)

    Based on: https://arxiv.org/abs/1810.07842

    Args:
        alpha: FP weight (lower = tolerate more FP for better recall)
        beta: FN weight (higher = severely penalize missed lesions)
        gamma: Focal parameter (higher = focus more on hard examples)
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()

        tversky_index = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)

        # Apply focal weight to focus on hard cases
        focal_tversky = torch.pow(1 - tversky_index, self.gamma)

        return focal_tversky


class SmallLesionOptimizedLoss(nn.Module):
    """
    Combined loss specifically optimized for tiny brain metastasis detection

    Components:
    1. Asymmetric Focal Tversky (60%) - handles severe imbalance + focuses on hard cases
    2. Focal Loss (25%) - additional hard example mining
    3. Sensitivity Loss (15%) - explicitly maximize recall

    This combination prioritizes finding all lesions (high recall) while managing FPs
    """
    def __init__(
        self,
        tversky_weight=0.60,
        focal_weight=0.25,
        sensitivity_weight=0.15,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        tversky_gamma=1.5,
        focal_alpha=0.25,
        focal_gamma=2.5,
        sensitivity_r=0.75
    ):
        super().__init__()

        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        self.sensitivity_weight = sensitivity_weight

        self.tversky_loss = AsymmetricFocalTverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma
        )

        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma
        )

        self.sensitivity_loss = SensitivitySpecificityLoss(
            r=sensitivity_r
        )

    def forward(self, pred, target):
        tversky = self.tversky_loss(pred, target)
        focal = self.focal_loss(pred, target)
        sensitivity = self.sensitivity_loss(pred, target)

        combined = (
            self.tversky_weight * tversky +
            self.focal_weight * focal +
            self.sensitivity_weight * sensitivity
        )

        return combined


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss for deep supervision
    Applies loss at multiple resolution levels in the network

    Args:
        loss_fn: Base loss function to apply at each scale
        weights: List of weights for each scale (finest to coarsest)
    """
    def __init__(self, loss_fn, weights=None):
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights if weights is not None else [1.0, 0.5, 0.25]

    def forward(self, predictions, target):
        """
        Args:
            predictions: List of predictions at different scales
            target: Ground truth mask
        """
        if not isinstance(predictions, (list, tuple)):
            # Single prediction, no multi-scale
            return self.loss_fn(predictions, target)

        total_loss = 0
        for i, pred in enumerate(predictions):
            # Resize target to match prediction size if needed
            if pred.shape != target.shape:
                target_resized = F.interpolate(
                    target,
                    size=pred.shape[2:],
                    mode='nearest'
                )
            else:
                target_resized = target

            weight = self.weights[i] if i < len(self.weights) else 0.1
            total_loss += weight * self.loss_fn(pred, target_resized)

        return total_loss


def get_loss_function(loss_type, **kwargs):
    """
    Factory function to get loss by name

    Args:
        loss_type: Name of loss function
        **kwargs: Parameters for the loss function

    Returns:
        Loss function instance
    """
    losses = {
        'tversky': TverskyLoss,
        'focal': FocalLoss,
        'combo': ComboLoss,
        'sensitivity': SensitivitySpecificityLoss,
        'focal_tversky': AsymmetricFocalTverskyLoss,
        'small_lesion': SmallLesionOptimizedLoss,
    }

    if loss_type not in losses:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(losses.keys())}")

    return losses[loss_type](**kwargs)
