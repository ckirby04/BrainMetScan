# Brain Metastasis Segmentation Model v1.2

**Smart Ensemble**: Multi-scale 3D U-Net ensemble for high-sensitivity brain metastasis detection

## Performance Summary

### Smart Ensemble (Union Fusion)

| Threshold | Voxel Sensitivity | Specificity | Lesion Detection | Lesion F1 |
|-----------|------------------|-------------|------------------|-----------|
| 0.5 | 97.0% | 86.8% | 92.2% (47/51) | 63.1% |
| **0.6** | **95.6%** | **90.9%** | **92.2%** (47/51) | **70.1%** |
| 0.7 | 92.8% | 93.3% | 84.1% (43/51) | 73.5% |

**Recommended Clinical Threshold**: 0.6 (best balance of sensitivity and specificity)

### Best Individual Model: exp3_12patch_maxfn

| Metric | Score |
|--------|-------|
| Lesion Sensitivity | 90.4% |
| Lesion F1 | 85.4% |
| Tiny Lesion Dice | 85-92% |
| Optimal Threshold | 0.55 |

### Improvement Over Baseline

| Metric | Baseline (96 patch) | Current Best | Improvement |
|--------|-------------------|--------------|-------------|
| Overall Dice | 67.2% | 73.1% | +5.9% |
| Tiny Lesion Dice | 18.5% | 85-92% | +66-74% |
| Lesion Sensitivity | 71.9% | 92.2% | +20.3% |
| Voxel Sensitivity | ~65% | 95.6% | +30.6% |

## Ensemble Architecture

The smart ensemble combines 4 models trained at different patch sizes:

| Model | Patch Size | Strength |
|-------|------------|----------|
| exp3_12patch_maxfn | 12x12x12 | Tiny lesions (<500 voxels) |
| exp1_8patch | 8x8x8 | Ultra-small lesions |
| improved_24patch | 24x24x24 | Balanced detection |
| improved_36patch | 36x36x36 | Large lesions + context |

**Fusion Strategy**: Union (MAX probability across models) - maximizes sensitivity

## Quick Start

### Inference with Smart Ensemble

```python
import torch
from pathlib import Path
from src.segmentation.unet import LightweightUNet3D

# Load models
model_dir = Path('model')
models = []
for name in ['exp3_12patch_maxfn', 'exp1_8patch', 'improved_24patch', 'improved_36patch']:
    model = LightweightUNet3D(in_channels=4, out_channels=1, use_attention=True, use_residual=True)
    checkpoint = torch.load(model_dir / f'{name}_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    models.append(model)

# Union fusion: take max probability across models
# See scripts/train_smart_ensemble.py for full implementation
```

### Evaluation Scripts

```bash
# Smart ensemble evaluation
python scripts/train_smart_ensemble.py

# Confusion matrix at different thresholds
python scripts/confusion_matrix.py

# Threshold optimization for single model
python scripts/optimize_threshold.py --model exp3_12patch_maxfn --patch-size 12

# Post-processing evaluation
python scripts/postprocessing.py
```

## Key Training Innovations

### 1. Aggressive FN Loss (Tversky alpha=0.1, beta=0.9)
- 9x penalty for missing lesions vs false positives
- Critical for achieving 90%+ sensitivity

### 2. Multi-Scale Patch Training
- Smaller patches (8, 12) dramatically improve tiny lesion detection
- 2.7x improvement in tiny lesion Dice (18.5% to 50.4% with 48 patch)

### 3. Union-Based Ensemble
- Take MAX probability across models
- Ensures any lesion detected by ANY model is included
- Maximizes sensitivity at cost of some false positives

## Files

```
1.2/
├── model/
│   ├── exp3_12patch_maxfn_best.pth  # Best sensitivity model
│   ├── exp1_8patch_best.pth          # Ultra-small patch model
│   ├── improved_24patch_best.pth     # Balanced model
│   └── improved_36patch_best.pth     # Large patch model
├── src/
│   └── segmentation/
│       ├── unet.py                   # LightweightUNet3D architecture
│       ├── dataset.py                # BrainMetDataset
│       ├── losses.py                 # Tversky, FocalDice losses
│       └── tta.py                    # Test-time augmentation
├── scripts/
│   ├── train_smart_ensemble.py       # Ensemble evaluation
│   ├── confusion_matrix.py           # Confusion matrix generation
│   ├── optimize_threshold.py         # Threshold optimization
│   ├── postprocessing.py             # Post-processing evaluation
│   └── overnight_experiments.py      # Batch training experiments
├── outputs/
│   ├── smart_ensemble_results.json   # Ensemble metrics
│   ├── postprocessing_results.json   # Post-processing results
│   └── confusion_matrix_*.png        # Confusion matrices
├── configs/
│   └── *.yaml                        # Training configurations
└── README.md                         # This file
```

## Post-Processing Analysis

Post-processing was evaluated but found to reduce sensitivity too much:

| Method | Voxel Sens | Lesion Sens | Notes |
|--------|------------|-------------|-------|
| Baseline (t=0.6) | 96.1% | 85.7% | Recommended |
| Size filter (50) | 97.7% | 84.1% | Minimal impact |
| Full pipeline | 86.4% | 77.8% | Too aggressive |

**Conclusion**: For clinical use, prioritize high sensitivity. Use threshold 0.6 without post-processing.

## Next Steps

1. **External Validation** - Test on datasets from other institutions
2. **FDA 510(k) Preparation** - QMS documentation, clinical study design
3. **Model Improvements** - Self-supervised pretraining on unlabeled data

## Training Details

| Parameter | Value |
|-----------|-------|
| Architecture | LightweightUNet3D with attention + residual |
| Input Channels | 4 (T1-pre, T1-Gd, FLAIR, T2) |
| Base Channels | 20 |
| Loss | Tversky (alpha=0.1, beta=0.9) + Focal Dice |
| Optimizer | AdamW |
| Training Data | 566 labeled cases |

## Citation

```
Brain Metastasis Segmentation Model v1.2
Smart Ensemble with Union Fusion
Performance: 95.6% voxel sensitivity, 92.2% lesion detection at threshold 0.6
```

## License

[Your license here]

## Contact

[Your contact info here]
