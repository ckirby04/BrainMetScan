# Brain Metastasis Segmentation - Progress Update

**Date**: February 24, 2026
**Previous Update**: January 23, 2026

---

## Executive Summary

nnU-Net v2 training completed and is now the best-performing model at **0.7596 mean Dice** (0.8099 median), surpassing the custom lightweight ensemble (0.7475). Full lesionwise evaluation infrastructure was built for apples-to-apples comparison. The ResEncM (deeper architecture) experiment was attempted but abandoned due to VRAM limitations on the RTX 3070 Ti.

---

## nnU-Net v2 Results (fold 0, 1000 epochs)

### Training

- **Architecture**: PlainConvUNet [32, 64, 128, 256, 320, 320], 6 stages
- **Configuration**: 3d_fullres, 128^3 patches, batch 2, SGD + poly LR
- **Data**: 452 train / 114 val (nnU-Net 5-fold split)
- **Training time**: ~27 hours at ~97s/epoch
- **Script**: `scripts/setup_nnunet.py` prepared data preserving original case IDs

### Metrics vs Custom Ensemble

| Metric | nnU-Net | Ensemble | Delta |
|--------|---------|----------|-------|
| **Voxel Dice** | **0.7596** | 0.7475 | +0.012 |
| Sensitivity | 0.7353 | **0.7839** | -0.049 |
| Precision | **0.8099** | 0.7430 | +0.067 |
| Surface Dice @2 | **0.8756** | 0.8674 | +0.008 |
| HD95 (mean) | **18.0** | 20.3 | -2.3 |
| Lesion Recall | 0.654 | **0.725** | -0.071 |
| Lesion Precision | **0.813** | 0.784 | +0.029 |

### Key Finding: Models Are Complementary

- **nnU-Net excels** on medium-to-huge lesions (precise, fewer false positives)
- **Custom ensemble excels** on tiny/small lesions (sensitive, catches more)
- nnU-Net missed 381/1100 lesions; 76% of misses were <50 voxels
- Combining both models could yield significant improvements

### Dice by Lesion Size

| Size | nnU-Net | Ensemble | Winner |
|------|---------|----------|--------|
| Tiny (<100) | 0.741 | **0.833** | Ensemble (+9.2%) |
| Small (100-500) | 0.610 | **0.724** | Ensemble (+11.4%) |
| Medium (500-5k) | **0.721** | 0.682 | nnU-Net (+3.9%) |
| Large (5k-20k) | **0.838** | 0.830 | nnU-Net (+0.8%) |
| Huge (>20k) | **0.853** | 0.844 | nnU-Net (+0.9%) |

---

## nnU-Net Postprocessing

nnU-Net's built-in postprocessing was tested (connected component removal — keeping only the largest foreground region). Result: **no benefit**. Brain metastases are inherently multi-focal with many tiny real lesions, so removing small components hurts performance. Postprocessing functions were set to empty (no-op).

---

## ResEncM Experiment (ABANDONED)

### What We Tried

The nnU-Net ResidualEncoderUNet Medium (ResEncM) architecture uses deeper encoder blocks [1, 3, 4, 6, 6, 6] vs the standard [2, 2, 2, 2, 2, 2]. It has been shown to outperform the standard architecture on many nnU-Net benchmarks.

### Why It Failed

| Session | Epoch Time | Expected |
|---------|-----------|----------|
| Original | ~143s | ~145s (correct) |
| Resume 1 | ~500s | ~145s |
| Resume 2 | ~611s | ~145s |
| Resume 3 | **~1374s** | ~145s |

**Root cause**: The ResEncM model consumed 7.4/8.2 GB VRAM on the RTX 3070 Ti (90% utilization). When VRAM overflows, PyTorch silently spills tensors to system RAM, causing a ~10x slowdown. The progressive degradation across sessions was caused by memory fragmentation.

At 1374s/epoch, completing 1000 epochs would take **~16 days** (vs ~27 hours for standard). The experiment was abandoned at epoch 83 with a pseudo Dice of 0.78.

### Lesson Learned

ResEncM requires more than 8 GB VRAM for this dataset (4-channel, 128^3 patches). The standard PlainConvUNet is the correct choice for the RTX 3070 Ti.

Additionally, nnU-Net defaults to 12 data augmentation workers (`nnUNet_n_proc_DA`), which is tuned for Linux servers. On Windows (which uses `spawn` multiprocessing), this creates significant overhead. Setting `nnUNet_n_proc_DA=4` is recommended for Windows desktops.

---

## Evaluation Infrastructure Built

### `scripts/evaluate_nnunet.py`

Full lesionwise evaluation matching `lesionwise_eval.py` format exactly:

- Voxel Dice, Sensitivity, Precision
- Lesion detection F1, Recall, Precision
- Per-lesion Dice
- Surface Dice at tolerances 1, 2, 3 voxels
- Hausdorff 95
- Relaxed Dice at dilations 1, 2, 3
- Missed lesion analysis by size
- Dice by lesion size bucket
- Auto-comparison with ensemble results

```bash
python scripts/evaluate_nnunet.py --fold 0                    # Evaluate fold 0
python scripts/evaluate_nnunet.py --trainer "custom_trainer"   # Different trainer
python scripts/evaluate_nnunet.py --postprocessed              # Postprocessed only
```

---

## Repository Cleanup

- Deleted ResEncM results (1.6 GB), plans JSON, and training script
- Previous spring cleaning archived 33 deprecated scripts, 27 model JSONs, 6 configs, 15 outputs

---

## Next Steps

1. **Multi-fold nnU-Net**: Train folds 1-4 for 2-5 fold ensemble — expected +2-4% Dice
2. **Cross-model ensemble**: Combine nnU-Net + custom lightweight model predictions
3. **Test-time augmentation**: Mirror augmentation during nnU-Net inference (~0.5-1% gain)
4. **Threshold tuning**: Sweep softmax thresholds on nnU-Net probability outputs

---

## Files Changed Since Last Update

### New
- `scripts/evaluate_nnunet.py` — nnU-Net evaluation (lesionwise format)
- `scripts/setup_nnunet.py` — nnU-Net data preparation
- `model/nnunet_evaluation.json` — nnU-Net fold 0 metrics

### Deleted
- `scripts/run_resenc.py` — ResEncM training script (experiment abandoned)
- `nnUNet/.../nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/` — ResEncM results (1.6 GB)
- `nnUNet/nnUNet_preprocessed/.../nnUNetResEncUNetMPlans.json` — ResEncM plans

### Updated
- `README.md` — Full rewrite reflecting current project state
