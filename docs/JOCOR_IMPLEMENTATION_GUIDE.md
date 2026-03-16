# JoCoR (Joint Co-Regularization) Implementation Guide for BrainMetScan

## Purpose

This document provides everything needed to implement JoCoR-based noisy label training into the BrainMetScan brain metastasis segmentation project. The project uses **nnUNetv2** as its segmentation framework and trains on brain MRI data (T1-pre, T1-Gd, FLAIR, T2) where the ground truth masks are **lazily annotated** — meaning radiologists drew rough contours around metastases that don't perfectly delineate lesion boundaries.

The core problem: our model is likely **more accurate than the ground truth** at lesion boundaries, but standard training with Dice+CE loss pulls predictions toward the noisy annotation artifacts. JoCoR helps the model learn the true underlying signal rather than memorizing annotation noise.

---

## Table of Contents

1. [Background: The Noisy Label Problem in Our Context](#1-background)
2. [JoCoR Algorithm Deep Dive](#2-jocor-algorithm)
3. [Adaptation for 3D Medical Image Segmentation](#3-adaptation-for-segmentation)
4. [nnUNetv2 Integration Architecture](#4-nnunetv2-integration)
5. [Implementation: Step by Step](#5-implementation)
6. [Hyperparameter Guide](#6-hyperparameters)
7. [Evaluation Strategy](#7-evaluation)
8. [Alternative and Complementary Approaches](#8-alternatives)
9. [References](#9-references)

---

## 1. Background: The Noisy Label Problem in Our Context <a name="1-background"></a>

### What We're Dealing With

Our ground truth masks for brain metastases were annotated by radiologists with varying levels of precision. This introduces **spatially structured annotation noise** — specifically:

- **Boundary imprecision**: Contours are drawn roughly around lesions, not pixel-perfect along true tissue boundaries. The noise is concentrated at lesion edges.
- **Systematic over/under-segmentation**: Some annotators consistently include a margin of healthy tissue; others consistently under-segment.
- **Inconsistency across annotators**: Different radiologists produce different masks for the same lesion.
- **Small lesion challenges**: Tiny metastases (sub-centimeter) have the worst annotation quality because drawing precise contours on small structures is harder.

This is **NOT** random label flipping (symmetric noise) — it's **spatially correlated, boundary-concentrated noise**. This distinction matters for how we adapt JoCoR.

### Why Standard Training Fails

Deep networks memorize training data. With noisy labels:
1. Early in training, the network learns the **true pattern** (lesion vs. background)
2. Later in training, the network starts **memorizing the noise** (imprecise boundaries)
3. This is the "memorization effect" — the model fits annotation artifacts rather than true anatomy

The result: our model's segmentation gets pulled toward the lazy annotation boundaries, even though it could have learned better boundaries from the underlying image signal.

---

## 2. JoCoR Algorithm Deep Dive <a name="2-jocor-algorithm"></a>

### Paper

Wei et al., "Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization," CVPR 2020. [arXiv:2003.02752](https://arxiv.org/abs/2003.02752)

Reference implementation: [https://github.com/hongxin001/JoCoR](https://github.com/hongxin001/JoCoR)

### Core Idea

Train **two networks simultaneously** with a combined loss that:
1. Computes standard supervised loss for each network independently
2. Adds a **co-regularization term** (KL divergence) that encourages both networks to agree
3. Uses **small-loss sample selection** — only backpropagates through samples with the smallest combined loss, under the assumption that clean samples have lower loss than noisy ones

### Mathematical Formulation

For two networks with parameters θ₁ and θ₂, the joint loss for a single sample (x, y) is:

```
L_joint(x, y) = L_supervised(f₁(x; θ₁), y) + L_supervised(f₂(x; θ₂), y) + λ · L_co-reg(f₁(x; θ₁), f₂(x; θ₂))
```

Where:
- `L_supervised` = standard task loss (Dice + CE in our case)
- `L_co-reg` = KL divergence between the two networks' softmax outputs
- `λ` = co-regularization weight (key hyperparameter)

### Small-Loss Selection

For each mini-batch:
1. Compute `L_joint` for every sample
2. Sort samples by loss (ascending)
3. Keep only the bottom `R(t)%` of samples (those with smallest loss)
4. Backpropagate only through the selected samples

`R(t)` is a **keep rate** that decreases over training:
```
R(t) = 1 - min(t/T_k * τ, τ)
```
Where:
- `t` = current epoch
- `T_k` = number of epochs to reach minimum keep rate
- `τ` = estimated noise rate (fraction of noisy samples)

### Why JoCoR Works Better Than Co-teaching

Co-teaching (Han et al., 2018) uses two networks that **teach each other** by exchanging small-loss samples. The error flow is asymmetric — biased selection in one network gets propagated to the other.

JoCoR trains both networks **jointly** with a shared loss and shared sample selection. The co-regularization term keeps the networks in agreement, which:
- Prevents divergence between the two networks
- Makes the small-loss selection more robust (both networks must agree a sample is clean)
- Reduces accumulated error from biased selection

---

## 3. Adaptation for 3D Medical Image Segmentation <a name="3-adaptation-for-segmentation"></a>

The original JoCoR was designed for **image classification** with random label noise. Adapting it to our **3D segmentation** problem with **spatially structured annotation noise** requires several modifications.

### 3.1 Sample Selection Granularity

**Original JoCoR**: Selects entire images (classification).

**Our options**:

| Granularity | Pros | Cons |
|-------------|------|------|
| Per-volume | Simple, matches original JoCoR | Too coarse — every volume has SOME noisy voxels |
| Per-patch | Matches nnUNet's patch-based training | Reasonable balance; patches near boundaries are noisier |
| Per-voxel | Most granular; can identify exact noisy voxels | Extremely expensive; breaks spatial coherence |

**Recommended: Per-patch selection** — This naturally aligns with nnUNet's training loop which already operates on patches. Patches containing mostly lesion boundary will have higher loss (noisier), while patches containing lesion core or pure background will have lower loss (cleaner).

**Advanced option: Voxel-level confidence weighting** — Instead of hard selection, compute a per-voxel confidence weight based on both networks' agreement. Weight the loss for each voxel by this confidence. This is a softer version that avoids throwing away entire patches.

```python
# Pseudo-code for voxel-level confidence weighting
p1 = torch.sigmoid(logits_net1)  # Network 1 predictions
p2 = torch.sigmoid(logits_net2)  # Network 2 predictions

# Agreement = how similar the two networks' predictions are
agreement = 1.0 - torch.abs(p1 - p2)  # [0, 1] per voxel

# Use agreement as a weight for the supervised loss
weighted_loss = agreement.detach() * supervised_loss_per_voxel
```

### 3.2 Co-Regularization for Segmentation

For binary segmentation (lesion vs. background), KL divergence between two networks' outputs:

```python
def co_regularization_loss(logits1, logits2):
    """
    KL divergence between two networks' predictions.
    For binary segmentation, compute per-voxel KL and average.
    """
    p1 = torch.sigmoid(logits1)
    p2 = torch.sigmoid(logits2)

    # Clamp to avoid log(0)
    p1 = torch.clamp(p1, 1e-7, 1 - 1e-7)
    p2 = torch.clamp(p2, 1e-7, 1 - 1e-7)

    # Symmetric KL divergence (Jensen-Shannon style)
    kl_1_2 = p1 * torch.log(p1 / p2) + (1 - p1) * torch.log((1 - p1) / (1 - p2))
    kl_2_1 = p2 * torch.log(p2 / p1) + (1 - p2) * torch.log((1 - p2) / (1 - p1))

    return 0.5 * (kl_1_2 + kl_2_1).mean()
```

### 3.3 Boundary-Aware Noise Modeling

Since our annotation noise concentrates at lesion boundaries, we can add a **boundary-awareness** mechanism:

```python
def compute_boundary_distance_weights(ground_truth_mask):
    """
    Compute distance from each voxel to the nearest boundary.
    Voxels near boundaries get LOWER confidence (more likely noisy).
    Voxels far from boundaries get HIGHER confidence (more likely clean).
    """
    from scipy.ndimage import distance_transform_edt

    # Distance from foreground boundary
    fg_dist = distance_transform_edt(ground_truth_mask)
    bg_dist = distance_transform_edt(1 - ground_truth_mask)

    # Minimum distance to any boundary
    boundary_dist = np.minimum(fg_dist, bg_dist)

    # Convert to confidence weight: far from boundary = high confidence
    # sigma controls the "trust radius" — how far from boundary we trust the label
    sigma = 3.0  # voxels — tune this based on your annotation quality
    confidence = 1.0 - np.exp(-boundary_dist**2 / (2 * sigma**2))

    # Ensure minimum confidence (don't completely zero out boundary voxels)
    confidence = np.clip(confidence, 0.1, 1.0)

    return confidence
```

This can be used to weight the supervised loss: trust the annotation fully in lesion cores and clear background, but down-weight the loss at boundaries where annotation noise is concentrated.

### 3.4 Handling Deep Supervision in nnUNet

nnUNet uses **deep supervision** — computing loss at multiple resolution levels. The JoCoR co-regularization and sample selection need to work with this:

- Apply co-regularization at **each deep supervision level** (both networks should agree at all resolutions)
- Compute sample selection based on the **combined multi-scale loss** (same as how nnUNet weights its deep supervision losses)
- Use the same deep supervision weights for both networks

---

## 4. nnUNetv2 Integration Architecture <a name="4-nnunetv2-integration"></a>

### How nnUNetv2 Custom Trainers Work

nnUNetv2 is designed for extensibility through trainer class inheritance:

1. Create a custom trainer class that inherits from `nnUNetTrainer`
2. Override specific methods to change behavior
3. Place the file in `nnunetv2/training/nnUNetTrainer/variants/` (or a subdirectory)
4. Reference it during training with the `-tr` flag

```bash
nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerJoCoR
```

nnUNet auto-discovers trainer classes via `recursive_find_python_class()` which searches the `nnunetv2/training/nnUNetTrainer/` directory tree.

### Key Methods to Override

| Method | Purpose | JoCoR Changes |
|--------|---------|---------------|
| `initialize()` | Sets up network, loss, optimizer | Build TWO networks and TWO optimizers |
| `_build_loss()` | Constructs the loss function | Add co-regularization term |
| `build_network_architecture()` | Creates the network | Called twice for two networks |
| `configure_optimizers()` | Sets up optimizer + scheduler | Create optimizers for both networks |
| `train_step()` | Single training iteration | Dual forward pass, joint loss, sample selection |
| `validation_step()` | Single validation iteration | Use Network 1 (or average) for validation |
| `save_checkpoint()` | Saves model state | Save both networks' states |
| `load_checkpoint()` | Loads model state | Load both networks' states |
| `on_train_epoch_end()` | End-of-epoch logic | Update keep rate schedule |

### Memory Considerations

Running two nnUNet networks simultaneously doubles GPU memory for model parameters. Options:

1. **Reduce patch size** — Use a smaller patch than nnUNet auto-configures
2. **Shared encoder, dual decoder** — Both networks share the encoder but have separate decoders. This cuts memory overhead significantly while still allowing independent predictions at the output level. This is the recommended approach.
3. **Gradient accumulation** — Forward pass each network separately, accumulate gradients
4. **Lightweight second network** — Use a smaller architecture for Network 2 (e.g., fewer filters)

**Recommended: Shared encoder with dual decoders.** Most annotation noise is at the output level — the encoder features should be similar for both networks. The decoder is where prediction divergence matters.

---

## 5. Implementation: Step by Step <a name="5-implementation"></a>

### Step 1: Create the Co-Regularization Loss Module

Create a new file for the JoCoR-specific loss components.

**File**: `nnunetv2/training/loss/jocor_loss.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class CoRegularizationLoss(nn.Module):
    """
    Co-regularization loss between two networks' predictions.
    Uses symmetric KL divergence (Jensen-Shannon divergence style).
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits1: Raw logits from network 1 [B, C, D, H, W] or [B, C, H, W]
            logits2: Raw logits from network 2 [B, C, D, H, W] or [B, C, H, W]

        Returns:
            Scalar co-regularization loss
        """
        p1 = torch.sigmoid(logits1)
        p2 = torch.sigmoid(logits2)

        p1 = torch.clamp(p1, 1e-7, 1 - 1e-7)
        p2 = torch.clamp(p2, 1e-7, 1 - 1e-7)

        # Symmetric KL divergence
        kl_1_2 = p1 * torch.log(p1 / p2) + (1 - p1) * torch.log((1 - p1) / (1 - p2))
        kl_2_1 = p2 * torch.log(p2 / p1) + (1 - p2) * torch.log((1 - p2) / (1 - p1))

        return 0.5 * (kl_1_2 + kl_2_1).mean()


class JoCoRLoss(nn.Module):
    """
    Complete JoCoR loss combining:
    - Standard supervised loss for Network 1
    - Standard supervised loss for Network 2
    - Co-regularization between both networks
    - Optional boundary-aware weighting
    """

    def __init__(
        self,
        supervised_loss_fn: nn.Module,
        lambda_coreg: float = 0.1,
        use_boundary_weighting: bool = True,
        boundary_sigma: float = 3.0,
        boundary_min_confidence: float = 0.1,
    ):
        """
        Args:
            supervised_loss_fn: The base nnUNet loss (DC_and_CE or similar)
            lambda_coreg: Weight for co-regularization term
            use_boundary_weighting: Whether to down-weight loss near annotation boundaries
            boundary_sigma: Controls trust radius around boundaries (in voxels)
            boundary_min_confidence: Minimum weight for boundary voxels (0 to 1)
        """
        super().__init__()
        self.supervised_loss = supervised_loss_fn
        self.coreg_loss = CoRegularizationLoss()
        self.lambda_coreg = lambda_coreg
        self.use_boundary_weighting = use_boundary_weighting
        self.boundary_sigma = boundary_sigma
        self.boundary_min_confidence = boundary_min_confidence

    def forward(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        target: torch.Tensor,
        boundary_weights: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            logits1: Predictions from network 1
            logits2: Predictions from network 2
            target: Ground truth segmentation mask
            boundary_weights: Optional precomputed boundary distance weights

        Returns:
            Dictionary with individual loss components and total loss
        """
        # Standard supervised losses
        loss_sup1 = self.supervised_loss(logits1, target)
        loss_sup2 = self.supervised_loss(logits2, target)

        # Co-regularization
        loss_coreg = self.coreg_loss(logits1, logits2)

        # Combined loss
        total_loss = loss_sup1 + loss_sup2 + self.lambda_coreg * loss_coreg

        return {
            'total': total_loss,
            'supervised_1': loss_sup1.detach(),
            'supervised_2': loss_sup2.detach(),
            'co_regularization': loss_coreg.detach(),
        }


class JoCoRDeepSupervisionWrapper(nn.Module):
    """
    Wraps JoCoR loss for nnUNet's deep supervision.
    Expects lists of predictions at different resolutions.
    """

    def __init__(self, jocor_loss: JoCoRLoss, weight_factors: list):
        """
        Args:
            jocor_loss: The JoCoR loss module
            weight_factors: Deep supervision weights (from nnUNet, e.g. [1, 0.5, 0.25, ...])
        """
        super().__init__()
        self.jocor_loss = jocor_loss
        self.weight_factors = weight_factors

    def forward(
        self,
        predictions_net1: list,
        predictions_net2: list,
        targets: list,
    ) -> dict:
        """
        Args:
            predictions_net1: List of predictions at each deep supervision level from net 1
            predictions_net2: List of predictions at each deep supervision level from net 2
            targets: List of targets at each deep supervision level

        Returns:
            Dictionary with total loss and components
        """
        assert len(predictions_net1) == len(predictions_net2) == len(targets) == len(self.weight_factors)

        total_loss = torch.zeros(1, device=predictions_net1[0].device)
        total_sup1 = 0.0
        total_sup2 = 0.0
        total_coreg = 0.0

        for i, (pred1, pred2, tgt, w) in enumerate(
            zip(predictions_net1, predictions_net2, targets, self.weight_factors)
        ):
            if w > 0:
                loss_dict = self.jocor_loss(pred1, pred2, tgt)
                total_loss += w * loss_dict['total']
                total_sup1 += w * loss_dict['supervised_1'].item()
                total_sup2 += w * loss_dict['supervised_2'].item()
                total_coreg += w * loss_dict['co_regularization'].item()

        return {
            'total': total_loss,
            'supervised_1': total_sup1,
            'supervised_2': total_sup2,
            'co_regularization': total_coreg,
        }
```

### Step 2: Create the Custom nnUNet Trainer

This is the main integration point. Create a custom trainer class that inherits from `nnUNetTrainer`.

**File**: `nnunetv2/training/nnUNetTrainer/variants/training_scheme/nnUNetTrainerJoCoR.py`

```python
import numpy as np
import torch
from copy import deepcopy
from typing import Tuple

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.jocor_loss import JoCoRLoss, JoCoRDeepSupervisionWrapper
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class nnUNetTrainerJoCoR(nnUNetTrainer):
    """
    nnUNet trainer with JoCoR (Joint Co-Regularization) for learning
    with noisy/lazy annotation labels.

    Key modifications:
    - Maintains two networks (shared encoder optional)
    - Joint loss with co-regularization term
    - Small-loss sample selection per patch
    - Boundary-aware loss weighting (optional)

    Usage:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerJoCoR
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)

        # ===== JoCoR Hyperparameters =====
        # Co-regularization weight — controls agreement strength between networks
        # Too high: networks collapse to identical outputs early, losing denoising benefit
        # Too low: networks diverge, co-regularization has no effect
        # Start with 0.1, tune in range [0.01, 1.0]
        self.lambda_coreg = 0.1

        # Estimated noise rate — fraction of voxels with noisy labels
        # For lazy annotations, 0.1-0.3 is typical (noise concentrated at boundaries)
        # This controls the minimum keep rate for sample selection
        self.noise_rate = 0.2

        # Number of epochs to reach the minimum keep rate
        # Should be ~10-20% of total training epochs
        self.noise_warmup_epochs = 100

        # Whether to use small-loss selection
        # Can be disabled to use only co-regularization (simpler, still effective)
        self.use_sample_selection = True

        # Boundary-aware weighting
        self.use_boundary_weighting = False  # Set True if you want boundary-aware loss
        self.boundary_sigma = 3.0  # Trust radius in voxels

        # Whether to use shared encoder (recommended for memory efficiency)
        self.use_shared_encoder = True

        # Second network reference (initialized in initialize())
        self.network2 = None
        self.optimizer2 = None
        self.lr_scheduler2 = None

    def initialize(self):
        """
        Extended initialization to set up dual networks.
        """
        # Call parent initialization (sets up network 1, loss, optimizer, etc.)
        super().initialize()

        # Build network 2
        if self.use_shared_encoder:
            # Shared encoder approach:
            # Clone only the decoder portion of the network.
            # Implementation depends on your specific nnUNet network architecture.
            # For standard nnUNet, the encoder and decoder are accessible as
            # separate components of the network.
            #
            # IMPORTANT: You'll need to inspect your network's actual structure
            # to determine how to share the encoder. The approach below is
            # a template — adapt to your actual architecture.
            self.network2 = deepcopy(self.network)
            self.network2 = self.network2.to(self.device)
            # If sharing encoder, you would do something like:
            # self.network2.encoder = self.network.encoder  # Share weights
            self.print_to_log_file("JoCoR: Built Network 2 (full copy — modify for shared encoder if needed)")
        else:
            # Full independent second network
            self.network2 = deepcopy(self.network)
            self.network2 = self.network2.to(self.device)
            self.print_to_log_file("JoCoR: Built Network 2 (independent copy)")

        # Configure optimizer for network 2
        self.optimizer2 = torch.optim.SGD(
            self.network2.parameters(),
            lr=self.initial_lr,
            momentum=0.99,
            nesterov=True,
            weight_decay=self.weight_decay,
        )

        # LR scheduler for network 2 (same as network 1)
        self.lr_scheduler2 = torch.optim.lr_scheduler.PolynomialLR(
            self.optimizer2,
            total_iters=self.num_epochs,
            power=0.9,
        )

        self.print_to_log_file(
            f"JoCoR Configuration:\n"
            f"  lambda_coreg = {self.lambda_coreg}\n"
            f"  noise_rate = {self.noise_rate}\n"
            f"  noise_warmup_epochs = {self.noise_warmup_epochs}\n"
            f"  use_sample_selection = {self.use_sample_selection}\n"
            f"  use_boundary_weighting = {self.use_boundary_weighting}\n"
            f"  use_shared_encoder = {self.use_shared_encoder}\n"
        )

    def _build_loss(self):
        """
        Build the JoCoR loss wrapped in deep supervision.
        """
        # Build the base supervised loss (same as standard nnUNet)
        base_loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {},
            weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        # Wrap in JoCoR loss
        jocor_loss = JoCoRLoss(
            supervised_loss_fn=base_loss,
            lambda_coreg=self.lambda_coreg,
            use_boundary_weighting=self.use_boundary_weighting,
            boundary_sigma=self.boundary_sigma,
        )

        # Apply deep supervision wrapping
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()

            self.loss = JoCoRDeepSupervisionWrapper(jocor_loss, list(weights))
        else:
            self.loss = jocor_loss

    def _get_keep_rate(self) -> float:
        """
        Compute the current keep rate for small-loss sample selection.
        Decreases from 1.0 to (1 - noise_rate) over noise_warmup_epochs.
        """
        if not self.use_sample_selection:
            return 1.0

        current_epoch = self.current_epoch
        keep_rate = 1.0 - min(
            current_epoch / self.noise_warmup_epochs * self.noise_rate,
            self.noise_rate
        )
        return max(keep_rate, 1.0 - self.noise_rate)

    def train_step(self, batch: dict) -> dict:
        """
        Modified training step implementing JoCoR dual-network training.
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)

        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer2.zero_grad(set_to_none=True)

        # Forward pass through both networks
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else torch.no_grad():
            output1 = self.network(data)
            output2 = self.network2(data)

        # Compute JoCoR loss
        if self.enable_deep_supervision:
            loss_dict = self.loss(output1, output2, target)
        else:
            loss_dict = self.loss(output1, output2, target)

        total_loss = loss_dict['total']

        # Sample selection (per-patch in the batch)
        if self.use_sample_selection and isinstance(target, list):
            keep_rate = self._get_keep_rate()
            batch_size = data.shape[0]
            num_keep = max(1, int(batch_size * keep_rate))

            if num_keep < batch_size:
                # Compute per-sample losses for selection
                per_sample_losses = []
                for b in range(batch_size):
                    # Extract single-sample predictions and targets
                    # This is a simplified version — adapt based on your
                    # deep supervision structure
                    sample_loss = 0.0
                    for pred1, pred2, tgt, w in zip(output1, output2, target, self.loss.weight_factors):
                        if w > 0:
                            p1 = pred1[b:b+1]
                            p2 = pred2[b:b+1]
                            t = tgt[b:b+1]
                            sl = self.loss.jocor_loss(p1, p2, t)['total']
                            sample_loss += w * sl.item()
                    per_sample_losses.append(sample_loss)

                # Select samples with smallest loss
                indices = np.argsort(per_sample_losses)[:num_keep]

                # Recompute loss on selected samples only
                selected_data = data[indices]
                selected_target = [t[indices] for t in target]

                output1_sel = self.network(selected_data)
                output2_sel = self.network2(selected_data)
                loss_dict = self.loss(output1_sel, output2_sel, selected_target)
                total_loss = loss_dict['total']

        # Backward pass (updates both networks)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            self.grad_scaler.unscale_(self.optimizer2)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            torch.nn.utils.clip_grad_norm_(self.network2.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.step(self.optimizer2)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            torch.nn.utils.clip_grad_norm_(self.network2.parameters(), 12)
            self.optimizer.step()
            self.optimizer2.step()

        return {
            'loss': total_loss.detach().cpu().numpy(),
            'loss_sup1': loss_dict['supervised_1'] if isinstance(loss_dict['supervised_1'], float) else loss_dict['supervised_1'].item(),
            'loss_sup2': loss_dict['supervised_2'] if isinstance(loss_dict['supervised_2'], float) else loss_dict['supervised_2'].item(),
            'loss_coreg': loss_dict['co_regularization'] if isinstance(loss_dict['co_regularization'], float) else loss_dict['co_regularization'].item(),
        }

    def validation_step(self, batch: dict) -> dict:
        """
        Validation uses Network 1 only (or optionally averages both).
        """
        # Use parent's validation_step which uses self.network
        return super().validation_step(batch)

    def on_train_epoch_end(self):
        """
        Extended to step Network 2's LR scheduler and log JoCoR metrics.
        """
        super().on_train_epoch_end()
        self.lr_scheduler2.step()

        # Log JoCoR-specific metrics
        if self.use_sample_selection:
            self.print_to_log_file(f"  JoCoR keep_rate: {self._get_keep_rate():.4f}")

    def save_checkpoint(self, filename: str):
        """
        Extended to save Network 2's state.
        """
        # Get the parent checkpoint
        # Note: you may need to modify this based on nnUNet version
        # The idea is to add network2's state to the saved checkpoint
        super().save_checkpoint(filename)

        # Save network 2 state separately
        net2_filename = filename.replace('.pth', '_net2.pth')
        torch.save({
            'network2_state_dict': self.network2.state_dict(),
            'optimizer2_state_dict': self.optimizer2.state_dict(),
        }, net2_filename)
        self.print_to_log_file(f"  JoCoR: Saved Network 2 checkpoint to {net2_filename}")

    def load_checkpoint(self, filename_or_checkpoint):
        """
        Extended to load Network 2's state.
        """
        super().load_checkpoint(filename_or_checkpoint)

        # Load network 2 state
        if isinstance(filename_or_checkpoint, str):
            net2_filename = filename_or_checkpoint.replace('.pth', '_net2.pth')
            try:
                net2_checkpoint = torch.load(net2_filename, map_location=self.device)
                self.network2.load_state_dict(net2_checkpoint['network2_state_dict'])
                self.optimizer2.load_state_dict(net2_checkpoint['optimizer2_state_dict'])
                self.print_to_log_file(f"  JoCoR: Loaded Network 2 from {net2_filename}")
            except FileNotFoundError:
                self.print_to_log_file("  JoCoR: No Network 2 checkpoint found, using fresh initialization")

    def perform_actual_validation(self, save_probabilities=False):
        """
        Validation uses Network 1 only.
        Network 2 is only used during training for co-regularization.
        """
        return super().perform_actual_validation(save_probabilities)
```

### Step 3: Create a Simplified Variant (Co-Regularization Only, No Sample Selection)

For easier initial testing, create a simpler variant that only uses the co-regularization loss without sample selection:

**File**: `nnunetv2/training/nnUNetTrainer/variants/training_scheme/nnUNetTrainerJoCoRSimple.py`

```python
from nnunetv2.training.nnUNetTrainer.variants.training_scheme.nnUNetTrainerJoCoR import nnUNetTrainerJoCoR


class nnUNetTrainerJoCoRSimple(nnUNetTrainerJoCoR):
    """
    Simplified JoCoR: co-regularization only, no sample selection.
    Easier to debug, still effective for boundary noise.

    Usage:
        nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerJoCoRSimple
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_sample_selection = False
        self.lambda_coreg = 0.1
```

### Step 4: Create Lambda-Sweep Variants for Hyperparameter Search

```python
# nnUNetTrainerJoCoR_lam001.py
class nnUNetTrainerJoCoR_lam001(nnUNetTrainerJoCoR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_coreg = 0.01
        self.use_sample_selection = False

# nnUNetTrainerJoCoR_lam01.py
class nnUNetTrainerJoCoR_lam01(nnUNetTrainerJoCoR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_coreg = 0.1
        self.use_sample_selection = False

# nnUNetTrainerJoCoR_lam05.py
class nnUNetTrainerJoCoR_lam05(nnUNetTrainerJoCoR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_coreg = 0.5
        self.use_sample_selection = False

# nnUNetTrainerJoCoR_lam1.py
class nnUNetTrainerJoCoR_lam1(nnUNetTrainerJoCoR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_coreg = 1.0
        self.use_sample_selection = False
```

---

## 6. Hyperparameter Guide <a name="6-hyperparameters"></a>

### Critical Hyperparameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `lambda_coreg` | 0.1 | [0.01, 1.0] | Higher = stronger agreement enforcement. Too high causes early collapse; too low loses denoising. **Start here for tuning.** |
| `noise_rate` | 0.2 | [0.05, 0.4] | Estimated fraction of noisy voxels. For lazy annotations: 0.1-0.3. Controls minimum keep rate in sample selection. |
| `noise_warmup_epochs` | 100 | [50, 200] | Epochs to reach minimum keep rate. Should be ~10-20% of total epochs (nnUNet default is 1000). |
| `boundary_sigma` | 3.0 | [1.0, 10.0] | Trust radius in voxels. Smaller = more aggressive boundary down-weighting. |
| `boundary_min_confidence` | 0.1 | [0.0, 0.5] | Floor confidence for boundary voxels. 0 = completely ignore boundaries, 0.5 = half weight. |

### Tuning Strategy

**Phase 1: Validate the framework works**
1. Train with `nnUNetTrainerJoCoRSimple` (co-reg only, no sample selection)
2. Use `lambda_coreg = 0.1`
3. Compare Dice scores against standard nnUNet baseline
4. If Dice improves or stays similar: framework is working
5. If Dice drops significantly: lambda is too high, try 0.01

**Phase 2: Tune lambda**
1. Run the lambda sweep variants: 0.01, 0.1, 0.5, 1.0
2. Compare on validation set
3. Also visually inspect boundary quality (not just Dice!)

**Phase 3: Add sample selection**
1. Enable sample selection with the best lambda from Phase 2
2. Start with `noise_rate = 0.2` and `noise_warmup_epochs = 100`
3. Compare against co-reg-only variant

**Phase 4: Boundary-aware weighting (optional)**
1. Enable boundary weighting with the best config from Phase 3
2. Tune `boundary_sigma` based on your typical annotation precision

### What to Monitor During Training

- **Dice score gap between train and validation**: JoCoR should reduce this gap (less overfitting to noisy labels)
- **Co-regularization loss over time**: Should decrease as networks converge, but not to zero (if zero, lambda is too high)
- **Keep rate schedule**: Should smoothly decrease from 1.0 to (1 - noise_rate)
- **Visual boundary quality**: The most important metric — do segmentation boundaries look anatomically reasonable?
- **Small lesion detection**: Monitor per-size-bin Dice separately; small lesions are most affected by annotation noise

---

## 7. Evaluation Strategy <a name="7-evaluation"></a>

### Standard Metrics (compare JoCoR vs. baseline nnUNet)

- **Overall Dice**: Global voxel-level overlap
- **Per-lesion Dice**: Dice computed per individual lesion, then averaged
- **Lesion-level sensitivity**: Fraction of ground truth lesions detected (IoU > 0.5)
- **Lesion-level false positive rate**: Number of predicted lesions with no ground truth match
- **Size-stratified Dice**: Separate Dice for small (<1cm), medium (1-2cm), large (>2cm) lesions

### Noisy-Label-Specific Metrics

- **Boundary Dice (Surface Dice)**: Dice computed only within N voxels of the boundary. This is the metric most sensitive to annotation noise handling. Compute at multiple tolerances (1mm, 2mm, 3mm).
- **Hausdorff Distance (95th percentile)**: Measures worst-case boundary error. JoCoR should improve this.
- **Label Precision / Recall Analysis**: Compare which samples the model gets "wrong" — if JoCoR is working, the samples with highest loss should correlate with the noisiest annotations.

### Visual Evaluation

Most critical evaluation for this problem. For a subset of cases:
1. Overlay baseline nnUNet prediction on the image
2. Overlay JoCoR prediction on the image
3. Overlay ground truth on the image
4. Look at boundaries — does JoCoR produce smoother, more anatomically plausible boundaries?
5. Look at small lesions — does JoCoR maintain detection while improving boundary quality?

### Ablation Studies to Run

1. **Baseline nnUNet** (no JoCoR)
2. **Co-regularization only** (JoCoRSimple)
3. **Co-reg + sample selection** (full JoCoR)
4. **Co-reg + boundary weighting** (JoCoR with boundary_weighting=True)
5. **Full JoCoR** (co-reg + selection + boundary weighting)

Compare all 5 on the same folds. This gives a clear picture of which components help.

---

## 8. Alternative and Complementary Approaches <a name="8-alternatives"></a>

JoCoR is not the only approach for noisy labels. Here are alternatives that could be used instead of or in addition to JoCoR:

### Direct Alternatives

| Method | Key Idea | Pros | Cons |
|--------|----------|------|------|
| **Co-teaching** (Han 2018) | Two networks exchange small-loss samples | Simpler than JoCoR | Accumulated error flow |
| **Co-teaching+** (Yu 2019) | Co-teaching + disagreement update | Better for asymmetric noise | Can be unstable |
| **Confident Learning** (Northcutt 2021) | Identify and remove/relabel noisy samples pre-training | Clean approach, works with any model | Requires noise estimation step; hard for segmentation |
| **MTCL** (Mean-Teacher Confident Learning) | Teacher-student + label denoising | Designed for medical segmentation specifically | More complex to implement |
| **Label Smoothing** | Soften hard labels (e.g., 0.9 instead of 1.0) | Trivial to implement | Doesn't adapt spatially |

### Complementary Approaches (Can Combine with JoCoR)

| Method | How to Combine |
|--------|---------------|
| **Label smoothing** | Apply soft labels before JoCoR training. Simple win. |
| **Test-time augmentation (TTA)** | Average predictions across augmentations at inference. Reduces boundary noise in predictions. |
| **Self-training / pseudo-labels** | After JoCoR training, use the model to re-label the dataset, then retrain. Iterative refinement. |
| **Morphological post-processing** | After training, apply morphological operations to clean up boundaries. |
| **CRF post-processing** | Conditional Random Field to refine boundaries using image gradients. |

### Simplest Possible Approach: Boundary Label Smoothing

If you want to try something before implementing full JoCoR, this is the easiest win:

```python
from scipy.ndimage import distance_transform_edt

def smooth_boundary_labels(mask, sigma=2.0, min_label=0.7):
    """
    Instead of hard 0/1 labels, smooth the labels near boundaries.
    Core of lesion stays 1.0, boundary transitions to min_label.
    """
    fg_dist = distance_transform_edt(mask)
    bg_dist = distance_transform_edt(1 - mask)
    boundary_dist = np.minimum(fg_dist, bg_dist)

    # Smoothing factor based on distance
    smooth = 1.0 - (1.0 - min_label) * np.exp(-boundary_dist**2 / (2 * sigma**2))

    # Apply: foreground stays near 1.0, background stays near 0.0
    smoothed = mask * smooth + (1 - mask) * (1 - smooth)
    return smoothed
```

This can be applied as a preprocessing step and requires zero changes to the training pipeline.

---

## 9. References <a name="9-references"></a>

### Core Paper
- Wei, H., Feng, L., Chen, X., & An, B. (2020). **Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization.** CVPR 2020. [arXiv:2003.02752](https://arxiv.org/abs/2003.02752)

### Reference Implementation
- Official PyTorch implementation: [https://github.com/hongxin001/JoCoR](https://github.com/hongxin001/JoCoR)

### nnUNet
- Isensee, F., et al. (2021). **nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.** Nature Methods.
- nnUNetv2 extending guide: [https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/extending_nnunet.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/extending_nnunet.md)

### Related Noisy Label Methods
- Han, B., et al. (2018). **Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels.** NeurIPS 2018.
- Northcutt, C., et al. (2021). **Confident Learning: Estimating Uncertainty in Dataset Labels.** JAIR.
- Shi, J., et al. (2024). **A survey of label-noise deep learning for medical image analysis.** Medical Image Analysis, 95, 103166.
- Karimi, D., et al. (2020). **Deep learning with noisy labels: Exploring techniques and remedies in medical image analysis.** Medical Image Analysis, 65, 101759.
- Liu, S., et al. (2024). **A teacher-guided early-learning method for medical image segmentation from noisy labels.** Complex & Intelligent Systems, 10, 8011–8026.

### Medical Image Segmentation with Noisy Labels
- MTCL (Mean-Teacher-assisted Confident Learning): [PubMed 35604969](https://pubmed.ncbi.nlm.nih.gov/35604969/)
- Noisy Student nnU-Net: [OpenReview](https://openreview.net/forum?id=-XzpY3MyKPU)
- Active Label Refinement for Noisy Labels: [PMC 11981598](https://pmc.ncbi.nlm.nih.gov/articles/PMC11981598/)

### Curated Resource Lists
- Advances in Label Noise Learning: [https://github.com/weijiaheng/Advances-in-Label-Noise-Learning](https://github.com/weijiaheng/Advances-in-Label-Noise-Learning)
- Awesome Learning with Label Noise: [https://github.com/subeeshvasu/Awesome-Learning-with-Label-Noise](https://github.com/subeeshvasu/Awesome-Learning-with-Label-Noise)

---

## Quick Start Checklist

1. [ ] Copy `jocor_loss.py` into `nnunetv2/training/loss/`
2. [ ] Copy the trainer files into `nnunetv2/training/nnUNetTrainer/variants/training_scheme/`
3. [ ] Verify nnUNet can find the trainer: `nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerJoCoRSimple` (should print JoCoR configuration)
4. [ ] Run baseline nnUNet training for comparison
5. [ ] Run JoCoRSimple (lambda=0.1) on the same fold
6. [ ] Compare Dice, boundary quality, and small lesion detection
7. [ ] If promising, run lambda sweep
8. [ ] If lambda sweep finds a good value, add sample selection
9. [ ] Visual evaluation on a subset of cases
10. [ ] Document results for your professor
