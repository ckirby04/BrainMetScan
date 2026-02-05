# Brain Metastasis Segmentation - Progress Update

**Date**: January 22, 2026
**Previous Update**: January 18, 2026
**Project Version**: 1.2

---

## Executive Summary

Strategy A (Curriculum Learning) has completed all 3 phases of training on the consolidated Superset (566 cases). The model achieved **54.77% Dice on the test set**, which is below the target. The primary bottleneck remains **tiny lesion detection**. We are now implementing a **multi-scale approach** using smaller patches (48³) to improve detection of lesions <500 voxels.

---

## Training Results: Strategy A Complete

### Final Status

| Phase | Epochs | Status | Output |
|-------|--------|--------|--------|
| Phase 1: High-Quality Bootstrap | 50/50 | ✅ Complete | `curriculum_phase1_best.pth` |
| Phase 2: Full Dataset Curriculum | 100/100 | ✅ Complete | `curriculum_phase2_best.pth` |
| Phase 3: Hard Sample Mining | 50/50 | ✅ Complete | `curriculum_final.pth` |

### Test Set Performance

| Metric | Result |
|--------|--------|
| **Overall Dice** | **54.77%** |
| Timestamp | 2026-01-22 17:47:20 |
| Test Cases | 51 |

### Comparison to Baseline

| Model | Dice Score | Notes |
|-------|------------|-------|
| `best_model.pth` (baseline) | 67.2% (validation) | Older model, single-phase training |
| `curriculum_final.pth` | 54.77% (test) | Strategy A, all 3 phases |

**Analysis**: The 12.4% gap between validation (67.2%) and test (54.77%) suggests:
1. Possible overfitting during curriculum learning
2. Test set contains harder cases than validation split
3. Domain shift between high-quality bootstrap data and test distribution

---

## Identified Bottleneck: Tiny Lesion Detection

### Performance by Lesion Size (Historical)

| Category | Voxels | Mean Dice | Success Rate | Assessment |
|----------|--------|-----------|--------------|------------|
| **Tiny** | <500 | **18.5%** | 25% | 🔴 Critical failure |
| Small | 500-2,000 | ~45% | ~40% | 🟡 Needs improvement |
| Medium | 2,000-5,000 | 64.2% | 67% | 🟢 Acceptable |
| Large | >5,000 | 67.7% | 60% | 🟢 Good |

### Why Tiny Lesions Fail

1. **Resolution Problem**: A 500-voxel lesion (~8×8×8 voxels) in a 96³ patch occupies only ~0.5% of the patch volume
2. **Class Imbalance**: Tiny lesions are rare but clinically critical (early-stage metastases)
3. **Context Mismatch**: 96³ patches provide too much background for tiny structures

---

## New Approach: Multi-Scale Patch Training

### Rationale

| Patch Size | Lesion Relative Size | Context | Best For |
|------------|---------------------|---------|----------|
| 96³ | 8/96 = 8.3% | More | Medium/Large lesions |
| 48³ | 8/48 = 16.7% | Less | **Tiny/Small lesions** |

**Key Insight**: Smaller patches make tiny lesions occupy a larger relative area, making them easier to detect.

### Implementation Plan

```
Phase 1: Train Small-Patch Model (48³)
├── Config: configs/config_tiny_lesion.yaml
├── Script: scripts/train_tiny_lesion.py
├── Focus: Lesions <500 voxels
├── Loss: Tversky (α=0.2, β=0.8) - high recall
└── Output: model/tiny_lesion_48patch_best.pth

Phase 2: Evaluate Size-Stratified Performance
├── Compare 48³ vs 96³ models on each size category
├── Identify crossover point (where large patches beat small)
└── Document performance gains

Phase 3: Multi-Scale Inference (if Phase 1 successful)
├── Run both models over same image
├── 48³ model: Dense inference with 75% overlap
├── 96³ model: Standard inference with 50% overlap
├── Merge: Combine predictions with confidence weighting
```

### New Files Created

| File | Purpose |
|------|---------|
| `configs/config_tiny_lesion.yaml` | Configuration for 48³ patch training |
| `scripts/train_tiny_lesion.py` | Training script with size-stratified evaluation |

### Key Configuration Differences

| Parameter | Medium Model (96³) | Tiny Model (48³) |
|-----------|-------------------|------------------|
| Patch size | 96×96×96 | **48×48×48** |
| Batch size | 2 | **8** |
| Tversky α | 0.4 | **0.2** (more FP tolerance) |
| Tversky β | 0.6 | **0.8** (penalize FN heavily) |
| Lesion weight | 5.0x | **15.0x** |
| Attention | Off | **On** |
| Eval overlap | 50% | **75%** |

---

## Model Inventory

### Current Checkpoints

| Model | Size | Date | Performance |
|-------|------|------|-------------|
| `best_model.pth` | 26MB | Dec 26 | 67.2% val Dice |
| `curriculum_phase1_best.pth` | 38MB | Jan 21 | Phase 1 complete |
| `curriculum_phase2_best.pth` | 12MB | Jan 22 | 52.97% best Dice |
| `curriculum_final.pth` | 12MB | Jan 22 | **54.77% test Dice** |
| `curriculum_final_resume.pth` | 38MB | Jan 22 | Full state for resume |

### Pending Models

| Model | Status | Expected |
|-------|--------|----------|
| `tiny_lesion_48patch_best.pth` | Training not started | TBD |
| `pretrain_finetune_best.pth` | Strategy B not started | TBD |
| `ensemble_best.pth` | Awaiting component models | TBD |

---

## Strategy B Status: Not Started

Strategy B (Self-Supervised Pretraining + Fine-tuning) has not been executed. It remains available as an alternative approach:

- **Phase 1**: Masked reconstruction on 1,430 Yale unlabeled scans
- **Phase 2**: Supervised fine-tuning on 566 labeled cases
- **Phase 3**: Hard sample refinement

**Recommendation**: Run Strategy B in parallel with multi-scale experiments for comparison.

---

## Next Steps

### Immediate (This Session)

1. ✅ Create tiny lesion config (`config_tiny_lesion.yaml`)
2. ✅ Create training script (`train_tiny_lesion.py`)
3. ⏳ Run tiny lesion model training
4. ⏳ Evaluate with size stratification

### Short-Term (This Week)

5. Compare 48³ vs 96³ models across all size categories
6. If successful, implement multi-scale inference pipeline
7. Consider running Strategy B for ensemble potential

### Medium-Term

8. Implement cascade detection (detect then segment)
9. Test DeepSupervisedUNet3D architecture
10. Create final ensemble model

---

## Technical Notes

### Training Infrastructure

- **GPU**: Single CUDA device
- **Training Time**: ~6-8 hours per strategy
- **TensorBoard**: Logs in `logs/tensorboard/`
- **Checkpoints**: Saved in `model/`

### Code Quality

- Resume functionality implemented for all training phases
- Size-stratified validation built into new training script
- Weighted scoring prioritizes tiny lesion performance (60% tiny + 40% overall)

---

## Lessons Learned

1. **Curriculum learning alone insufficient**: The 3-phase curriculum approach didn't achieve expected gains, suggesting the fundamental issue is architectural/resolution, not training strategy.

2. **Validation-test gap is real**: 12% drop from validation to test indicates need for better generalization or harder validation splits.

3. **Multi-scale may be necessary**: Single patch size cannot optimally detect both tiny (<500 voxel) and large (>5000 voxel) lesions.

4. **Aggressive weighting has limits**: Even 10x weighting on tiny lesions during Phase 3 wasn't enough - the model fundamentally can't see them at 96³ resolution.

---

## Changelog

- **2026-01-22**:
  - Strategy A completed all 3 phases (54.77% test Dice)
  - Created multi-scale training infrastructure (48³ patches)
  - Documented tiny lesion detection bottleneck
  - Added size-stratified evaluation to training pipeline

- **2026-01-18**:
  - Initial Superset implementation
  - Built 566 supervised + 1,430 pretraining cases
  - Created dual training strategy (A + B)
