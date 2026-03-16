# Atlas-Registered Model Experiment — Post-Mortem

## Summary

**Dates**: March 11–12, 2026
**Objective**: Train a dedicated segmentation model on atlas-registered (SRI24) MRI data to address the domain shift that causes the native-geometry model (Dice 0.747) to drop to 0.17–0.20 on atlas-registered external datasets.
**Outcome**: Abandoned. Insufficient data (244 cases) to produce a competitive model. Best full-volume ensemble Dice was 0.571, well below the 0.747 target.
**Decision**: Focus resources on improving the native-geometry model (566 cases, already competitive with published benchmarks) and use it as leverage to acquire more institutional data.

## What We Did

### Dataset Assembly
Combined two publicly available atlas-registered brain metastasis datasets:

| Dataset | Source | Cases | Format |
|---------|--------|:-----:|--------|
| PRETREAT-METSTOBRAIN-MASKS | Yale / TCIA (DOI: 10.7937/3YAT-E768) | 200 | SRI24, 240×240×155 @ 1mm iso |
| PROTEAS | Cyprus / Zenodo (DOI: 10.5281/zenodo.17253793) | 44 | SRI24, 240×240×155 @ 1mm iso |
| **Total** | | **244** | Resized to 256³, then 128³ for training |

### Training Configuration
- **Architecture**: LightweightUNet3D (base_channels=20, attention=True, residual=True) — same as native model
- **Models**: 3 patch sizes (12, 24, 36), 250 epochs each
- **Loss**: Combined (Dice 0.4 + Focal 0.3 + Tversky 0.3)
- **Augmentation**: Basic flips, rotations, intensity scaling
- **GPUs**: RTX 3070 Ti (8GB) + RTX 5060 Ti (17GB), parallel training
- **Total training time**: ~3 hours

### Hardware
- GPU 0 (RTX 3070 Ti): 12-patch model
- GPU 1 (RTX 5060 Ti): 24-patch and 36-patch models (sequential)

## Results

### Training Validation (Patch-Level)

| Model | Best Val Dice | Best Epoch | Sensitivity | Tiny Dice |
|-------|:---:|:---:|:---:|:---:|
| 12-patch | 0.892 | 239 | 0.890 | 0.690 |
| 24-patch | 0.910 | 243 | 0.920 | 0.651 |
| 36-patch | 0.913 | 245 | 0.925 | 0.579 |

### Full-Volume Evaluation (36 held-out val cases, threshold 0.40, overlap 0.25)

| Model | Mean Dice | Std |
|-------|:---------:|:---:|
| 12-patch | 0.396 | 0.310 |
| 24-patch | 0.543 | 0.353 |
| 36-patch | 0.622 | 0.347 |
| Ensemble (avg of 3) | 0.571 | 0.355 |

## Key Findings

### 1. Massive patch-to-volume gap
Training validation Dice was 0.89–0.91 on patches, but full-volume evaluation dropped to 0.40–0.62. The sliding window inference used only 0.25 overlap, likely producing stitching artifacts and boundary issues at patch seams. Higher overlap (0.5) would partially mitigate this but was not tested in the baseline run.

### 2. Simple average ensemble hurt performance
The 3-model ensemble (0.571) was **worse** than the best individual model (36-patch at 0.622). The weak 12-patch model (0.396) dragged down the average. A weighted ensemble or 24+36-only combination would have outperformed.

### 3. High variance across cases
Standard deviations of ~0.35 indicate highly inconsistent performance — some cases near-zero, others above 0.80. Small metastases were likely being missed entirely during whole-volume reassembly.

### 4. Data quantity is the binding constraint
At 244 cases, the atlas model has less than half the data of the native model (566 cases). No training trick can compensate for this gap at the dataset sizes we're working with. The native model at 566 cases achieves 0.747 — scaling laws in medical image segmentation strongly favor more data over architecture/loss improvements.

### 5. Planned improvements would not have been sufficient
We designed a Phase 1 experiment (SmallLesionOptimizedLoss + MONAI augmentation + weighted sampling) and a Phase 2 experiment (DeepSupervisedUNet3D architecture). Even optimistically, these could push the atlas model from 0.57 to perhaps 0.65–0.68 — still well below the 0.747 target and not enough to justify the GPU time.

## What We Learned About the Problem

### Domain shift is real and well-characterized
The native model drops from 0.747 → 0.17–0.20 on atlas-registered data (documented in external_validation_report.md). This is a systematic geometric domain shift from SRI24 atlas warping, not a data quality issue.

### The atlas model needs ~500+ cases to be useful
Extrapolating from the native model's performance curve (566 cases → 0.747 Dice), a competitive atlas model would need at least 500 annotated atlas-registered cases. The BraTS-METS 2025 challenge dataset on Synapse offers 1,778 cases in exactly this format — that would be transformative.

### All public brain met datasets are atlas-registered
Every publicly available brain metastasis segmentation dataset with multi-sequence MRI (T1-pre, T1-post, FLAIR, T2) and masks is distributed in BraTS format (SRI24 atlas). This means the atlas model is the only one that can benefit from public data expansion. The native model can only grow through direct institutional partnerships.

### Native model is already publishable
The native model's 0.7763 stacking Dice (6-model ensemble) is competitive with the UCSF-BMSR official nnU-Net benchmark (0.75 median). This is the strongest asset for partnership outreach.

## Available Public Datasets for Future Atlas Work

When data becomes available, these are the highest-priority additions:

| Dataset | Cases | Status | Impact |
|---------|:-----:|--------|--------|
| BraTS-METS 2025 (Synapse) | 1,778 | Available now (registration required) | Would 7× atlas training data |
| BraTS-METS 2023 (Synapse) | ~402 | Available now | Likely overlaps with 2025 |
| Brain-Mets-Lung-MRI-Path-Segs (TCIA) | 111 | Available now | Native geometry, T1CE+FLAIR only |
| BCBM-RadioGenomics (TCIA) | 268 MRIs | Available now | T1c only, CC BY 4.0 |
| MOTUM (Harvard Dataverse) | 38 met cases | Available now | Small but multi-center |

## Cleanup

All atlas-related files were removed on March 12, 2026:
- Training scripts: `train_atlas_registered.py`, `train_atlas_overnight.py`, `train_atlas_phase1.py`, `train_atlas_phase1_overnight.py`
- Preprocessing: `prepare_atlas_training.py`
- Model checkpoints: `model/atlas_registered/` (~77MB)
- Data symlinks: `data/atlas_registered/` (symlinks only; source data in `data/external_validation/` preserved)
- Results: `results/atlas_registered/`, `results/atlas_phase1/`
- Documentation: `docs/atlas_registered_dataset.md`

The external validation datasets (PRETREAT in `data/external_validation/`, PROTEAS in `data/external_validation_proteas/`) and their evaluation scripts are preserved — they remain useful for testing generalization of the native model.

## Next Steps

1. **Improve native model** — overnight pipeline (`scripts/training/native_overnight.py`) applies SmallLesionOptimizedLoss, MONAI augmentation, weighted sampling, then rebuilds stacking pipeline. Target: push 0.7763 → 0.80+.
2. **Use native model for institutional outreach** — competitive benchmark results as negotiating tool for data partnerships.
3. **Register on Synapse for BraTS-METS 2025** — 1,778 free atlas-registered cases would make the atlas model viable. Revisit atlas training only after acquiring this data.
