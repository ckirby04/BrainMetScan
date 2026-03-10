"""
Ensemble Segmentation Model for Brain Metastasis Detection
Combines multiple patch-size models (24, 36) with mean-based fusion
Uses matched inference patch sizes for optimal performance (~70% Dice)

Key insight: Each model must use the same patch size for inference as it was trained on.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.segmentation.unet import LightweightUNet3D


# Mapping from model name to training patch size
TRAINING_PATCH_SIZES = {
    '12-patch': 12,
    '24-patch': 24,
    '36-patch': 36,
}


class EnsembleSegmentationModel:
    """
    Ensemble model combining multiple patch-size trained models.

    IMPORTANT: Each model is trained on a specific patch size and must use that
    same patch size during inference for optimal results. Using larger patches
    (e.g., 96) during inference causes severe over-prediction.

    Default configuration uses only 24-patch and 36-patch models as the 12-patch
    model tends to over-predict significantly.

    Args:
        model_paths: Dictionary mapping model name to checkpoint path
        device: torch device to use
    """

    def __init__(
        self,
        model_paths: Dict[str, Path],
        device: torch.device = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models: Dict[str, nn.Module] = {}
        self.model_info: Dict[str, dict] = {}
        self.training_patch_sizes: Dict[str, int] = {}

        for name, path in model_paths.items():
            model, info = self._load_model(path)
            self.models[name] = model
            self.model_info[name] = info
            # Extract training patch size from model name
            self.training_patch_sizes[name] = TRAINING_PATCH_SIZES.get(name, 36)
            print(f"Loaded {name}: Epoch {info.get('epoch', '?')}, Dice {info.get('val_dice', 0):.2%}, patch_size={self.training_patch_sizes[name]}")

        print(f"Ensemble loaded with {len(self.models)} models on {self.device}")

    def _load_model(self, checkpoint_path: Path) -> Tuple[nn.Module, dict]:
        """Load a single model from checkpoint, inferring architecture from state dict"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Get state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Infer model configuration from state dict
        if 'args' in checkpoint:
            model_args = checkpoint['args']
            base_channels = model_args.get('base_channels', 16)
            depth = model_args.get('depth', 3)
            use_residual = model_args.get('use_residual', True)
            use_attention = model_args.get('use_attention', False)
        else:
            # Infer from weights
            if 'inc.conv1.weight' in state_dict:
                base_channels = state_dict['inc.conv1.weight'].shape[0]
            else:
                base_channels = 16

            # Check for attention gates
            use_attention = any('attention' in k for k in state_dict.keys())

            # Check for residual connections
            use_residual = any('residual_conv' in k for k in state_dict.keys())

            # Infer depth from down_blocks
            down_blocks = [k for k in state_dict.keys() if k.startswith('down_blocks.')]
            if down_blocks:
                depth = max(int(k.split('.')[1]) for k in down_blocks) + 1
            else:
                depth = 3

        model = LightweightUNet3D(
            in_channels=4,
            out_channels=1,
            base_channels=base_channels,
            depth=depth,
            dropout_p=0.0,
            use_residual=use_residual,
            use_attention=use_attention
        ).to(self.device)

        # Load weights
        model.load_state_dict(state_dict)

        info = {
            'epoch': checkpoint.get('epoch', 'unknown') if isinstance(checkpoint, dict) else 'unknown',
            'val_dice': checkpoint.get('val_dice', 0) if isinstance(checkpoint, dict) else 0,
            'base_channels': base_channels,
            'depth': depth,
            'use_attention': use_attention,
            'use_residual': use_residual
        }

        model.eval()
        return model, info

    def predict_single(
        self,
        volume: np.ndarray,
        model_name: str,
        patch_size: int = None
    ) -> np.ndarray:
        """
        Run inference with a single model using sliding window.

        Args:
            volume: Input volume (C, H, W, D) normalized
            model_name: Name of model to use
            patch_size: Patch size for sliding window. If None, uses training patch size.

        Returns:
            Probability map (H, W, D)
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models.keys())}")

        # Use training patch size by default for matched inference
        if patch_size is None:
            patch_size = self.training_patch_sizes[model_name]

        model = self.models[model_name]
        H, W, D = volume.shape[1:]

        output = np.zeros((H, W, D), dtype=np.float32)
        count = np.zeros((H, W, D), dtype=np.float32)

        stride = patch_size // 2

        h_starts = list(range(0, max(1, H - patch_size + 1), stride))
        if H > patch_size and (H - patch_size) not in h_starts:
            h_starts.append(H - patch_size)

        w_starts = list(range(0, max(1, W - patch_size + 1), stride))
        if W > patch_size and (W - patch_size) not in w_starts:
            w_starts.append(W - patch_size)

        d_starts = list(range(0, max(1, D - patch_size + 1), stride))
        if D > patch_size and (D - patch_size) not in d_starts:
            d_starts.append(D - patch_size)

        with torch.no_grad():
            for h_start in h_starts:
                for w_start in w_starts:
                    for d_start in d_starts:
                        h_end = min(h_start + patch_size, H)
                        w_end = min(w_start + patch_size, W)
                        d_end = min(d_start + patch_size, D)

                        patch = volume[:, h_start:h_end, w_start:w_end, d_start:d_end]

                        pad_h = patch_size - patch.shape[1]
                        pad_w = patch_size - patch.shape[2]
                        pad_d = patch_size - patch.shape[3]

                        if pad_h > 0 or pad_w > 0 or pad_d > 0:
                            patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')

                        input_tensor = torch.from_numpy(patch).unsqueeze(0).float().to(self.device)
                        pred = torch.sigmoid(model(input_tensor)).cpu().numpy()[0, 0]

                        valid_h = h_end - h_start
                        valid_w = w_end - w_start
                        valid_d = d_end - d_start

                        output[h_start:h_end, w_start:w_end, d_start:d_end] += pred[:valid_h, :valid_w, :valid_d]
                        count[h_start:h_end, w_start:w_end, d_start:d_end] += 1

        return output / np.maximum(count, 1)

    def predict_all(
        self,
        volume: np.ndarray,
        use_matched_patch_sizes: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Run inference with all models.

        Args:
            volume: Input volume (C, H, W, D) normalized
            use_matched_patch_sizes: If True, use training patch size for each model

        Returns:
            Dictionary mapping model name to probability map
        """
        predictions = {}
        for name in self.models:
            patch_size = self.training_patch_sizes[name] if use_matched_patch_sizes else 96
            predictions[name] = self.predict_single(volume, name, patch_size)
        return predictions

    def fuse_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        method: str = 'mean'
    ) -> np.ndarray:
        """
        Fuse predictions from multiple models.

        Args:
            predictions: Dictionary of model predictions
            method: Fusion method - 'max' (union), 'mean' (average), 'min' (intersection)

        Returns:
            Fused probability map
        """
        pred_stack = np.stack(list(predictions.values()), axis=0)

        if method == 'max':
            return np.max(pred_stack, axis=0)
        elif method == 'mean':
            return np.mean(pred_stack, axis=0)
        elif method == 'min':
            return np.min(pred_stack, axis=0)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def compute_agreement_map(
        self,
        predictions: Dict[str, np.ndarray],
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Compute agreement map showing how many models agree on each voxel.

        Args:
            predictions: Dictionary of model predictions
            threshold: Threshold for binary decision

        Returns:
            Agreement map (0 to num_models)
        """
        binary_preds = [(pred > threshold).astype(np.float32) for pred in predictions.values()]
        return np.sum(np.stack(binary_preds, axis=0), axis=0)

    def predict_ensemble(
        self,
        volume: np.ndarray,
        patch_size: int = None,  # Ignored when use_matched_patch_sizes=True
        fusion_method: str = 'mean',
        return_individual: bool = False,
        use_matched_patch_sizes: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Run full ensemble inference.

        Args:
            volume: Input volume (C, H, W, D) normalized
            patch_size: Patch size for sliding window (ignored if use_matched_patch_sizes=True)
            fusion_method: How to fuse predictions ('max', 'mean', 'min')
            return_individual: Whether to return individual model predictions
            use_matched_patch_sizes: Use training patch size for each model (recommended)

        Returns:
            Dictionary with 'fused' prediction and optionally individual predictions
        """
        predictions = self.predict_all(volume, use_matched_patch_sizes)
        fused = self.fuse_predictions(predictions, fusion_method)

        result = {'fused': fused}

        if return_individual:
            result['individual'] = predictions
            result['agreement'] = self.compute_agreement_map(predictions)

        return result

    def get_ensemble_info(self) -> dict:
        """Get information about the ensemble models"""
        return {
            'num_models': len(self.models),
            'model_names': list(self.models.keys()),
            'models': self.model_info,
            'training_patch_sizes': self.training_patch_sizes,
            'device': str(self.device)
        }


def create_ensemble_model(
    model_dir: Path,
    device: torch.device = None,
    include_12patch: bool = False
) -> EnsembleSegmentationModel:
    """
    Create ensemble model with recommended configuration.

    By default, excludes the 12-patch model which tends to over-predict.
    Uses only 24-patch and 36-patch models for better specificity.

    Args:
        model_dir: Directory containing model checkpoints
        device: torch device
        include_12patch: Whether to include the 12-patch model (not recommended)

    Returns:
        EnsembleSegmentationModel instance
    """
    model_paths = {
        '24-patch': model_dir / 'improved_24patch_best.pth',
        '36-patch': model_dir / 'improved_36patch_best.pth',
    }

    if include_12patch:
        model_paths['12-patch'] = model_dir / 'improved_12patch_best.pth'

    # Verify all models exist
    for name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

    return EnsembleSegmentationModel(model_paths, device)


if __name__ == "__main__":
    # Test ensemble
    print("Testing EnsembleSegmentationModel...")

    model_dir = Path(__file__).parent.parent.parent / "model"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        ensemble = create_ensemble_model(model_dir, device, include_12patch=False)
        info = ensemble.get_ensemble_info()
        print(f"\nEnsemble info:")
        print(f"  Models: {info['model_names']}")
        print(f"  Patch sizes: {info['training_patch_sizes']}")
        print(f"  Device: {info['device']}")

        # Test with dummy data
        dummy_volume = np.random.randn(4, 96, 96, 96).astype(np.float32)

        print("\nRunning test inference with matched patch sizes...")
        result = ensemble.predict_ensemble(
            dummy_volume,
            fusion_method='mean',
            return_individual=True,
            use_matched_patch_sizes=True
        )

        print(f"Fused shape: {result['fused'].shape}")
        print(f"Individual models: {list(result['individual'].keys())}")
        print(f"Agreement map range: {result['agreement'].min():.0f} - {result['agreement'].max():.0f}")

        print("\nEnsemble test passed!")

    except FileNotFoundError as e:
        print(f"Models not found: {e}")
        print("Copy models from 1.21/model/ to 1.3/model/ first")
