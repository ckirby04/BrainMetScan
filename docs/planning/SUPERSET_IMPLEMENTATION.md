# Brain Metastasis Segmentation - Superset Implementation

**Date**: January 18, 2026
**Project Version**: 1.2
**Goal**: Consolidate public datasets and implement advanced training strategies to improve segmentation accuracy

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Dataset Analysis](#dataset-analysis)
3. [Superset Architecture](#superset-architecture)
4. [Quality Filtering System](#quality-filtering-system)
5. [Training Strategies](#training-strategies)
6. [Implementation Files](#implementation-files)
7. [Usage Guide](#usage-guide)
8. [Expected Outcomes](#expected-outcomes)

---

## Executive Summary

### Before
- **Training data**: 105 cases (BrainMetShare only)
- **Model performance**: 67.2% Dice
- **Tiny lesion performance**: 18.5% Dice (critical weakness)

### After (Expected)
- **Training data**: 566 supervised cases + 1,430 pretraining cases
- **Expected improvement**: 10-20% Dice increase
- **Two parallel training strategies** for optimal results

### Key Changes
1. Consolidated **UCSF (461 cases)** + **BrainMetShare (105 cases)** = **566 supervised training cases**
2. Added **Yale (1,430 timepoints)** for self-supervised pretraining
3. Implemented **quality filtering** to create high-quality subset (top 30%)
4. Designed **dual training strategy** for parallel execution

---

## Dataset Analysis

### Raw Data Inventory (112GB total)

| Dataset | Size | Cases | Format | Has Masks | Status |
|---------|------|-------|--------|-----------|--------|
| **UCSF_BrainMetastases** | 7.9GB | 461 | NIfTI 3D | ✓ YES | **Used for supervised** |
| **BrainMetShare** | 2.7GB | 105+51 | NIfTI 3D | ✓ YES | **Used for supervised** |
| **Yale-Brain-Mets-Longitudinal** | 43GB | 1,430 | NIfTI 3D | ✗ NO | **Used for pretraining** |
| Kaggle_Dataset | 872MB | 105+51 | 2D JPG | N/A | Not used (2D format) |
| BraTS | 7.6GB | Many | H5 slices | N/A | Not used (different task) |
| OASIS-1 | 16GB | Many | Archived | ✗ NO | Not used (healthy brains) |
| IXI_Dataset | 28GB | Many | Archived | ✗ NO | Not used (healthy brains) |

### Modality Mapping

Each dataset uses different naming conventions. The consolidation script normalizes everything to:

| Standard Name | BrainMetShare | UCSF | Yale |
|---------------|---------------|------|------|
| `t1_pre.nii.gz` | `t1_pre.nii.gz` | `{id}_T1pre.nii.gz` | `*_PRE.nii.gz` |
| `t1_gd.nii.gz` | `t1_gd.nii.gz` | `{id}_T1post.nii.gz` | `*_POST.nii.gz` |
| `flair.nii.gz` | `flair.nii.gz` | `{id}_FLAIR.nii.gz` | `*_FLAIR.nii.gz` |
| `t2.nii.gz` | `bravo.nii.gz` | `{id}_T2Synth.nii.gz` | `*_T2.nii.gz` |
| `seg.nii.gz` | `seg.nii.gz` | `{id}_seg.nii.gz` | N/A |

---

## Superset Architecture

### Directory Structure

The Superset is located in the **parent directory** (`brainMetShare/Superset/`) to allow sharing across versions.

```
../Superset/
├── full/                      # Complete dataset
│   ├── train/                 # 566 cases (BMS_* + UCSF_*)
│   │   ├── BMS_Mets_005/     # BrainMetShare cases (prefixed BMS_)
│   │   │   ├── t1_pre.nii.gz
│   │   │   ├── t1_gd.nii.gz
│   │   │   ├── flair.nii.gz
│   │   │   ├── t2.nii.gz
│   │   │   └── seg.nii.gz
│   │   └── UCSF_100101A/     # UCSF cases (prefixed UCSF_)
│   │       └── ...
│   └── test/                  # 51 test cases
│
├── high_quality/              # Quality-filtered subset (top 30%)
│   ├── train/                 # ~170 highest quality cases
│   └── test/                  # (if applicable)
│
├── pretraining/               # Unlabeled data for self-supervised learning
│   ├── Yale_YG_01M98E_2016-11-13/
│   │   ├── t1_pre.nii.gz
│   │   ├── t1_gd.nii.gz
│   │   ├── flair.nii.gz
│   │   └── t2.nii.gz
│   └── ...                    # 1,430+ timepoints
│
├── metadata.csv               # Quality scores for all cases
└── quality_summary.json       # Statistics and thresholds
```

### Case Naming Convention

- **BMS_Mets_XXX**: BrainMetShare cases
- **UCSF_XXXXXXX**: UCSF cases (e.g., UCSF_100101A)
- **Yale_YG_XXXX_DATE**: Yale pretraining cases

---

## Quality Filtering System

### Quality Metrics

Each case is scored on 5 metrics (normalized 0-1):

| Metric | Weight | Description |
|--------|--------|-------------|
| **SNR** | 25% | Signal-to-noise ratio (brain vs background) |
| **Contrast** | 25% | Lesion-to-brain contrast |
| **Sharpness** | 20% | Image gradient magnitude |
| **Completeness** | 15% | Modality availability (4 modalities = 1.0) |
| **Mask Quality** | 15% | Segmentation coherence (fewer isolated voxels = better) |

### Overall Score Calculation

```
overall_score = (0.25 × SNR_norm) + (0.25 × contrast_norm) +
                (0.20 × sharpness_norm) + (0.15 × completeness) +
                (0.15 × mask_quality)
```

### High-Quality Threshold

- Default: **70th percentile** (top 30% of cases)
- Configurable via `--quality-percentile` argument
- High-quality cases copied to `../Superset/high_quality/`

### Output Files

1. **`metadata.csv`**: Per-case quality scores
   ```csv
   case_id,source,snr,contrast,sharpness,completeness,mask_volume,mask_quality,overall_score,is_high_quality
   UCSF_100101A,UCSF_BrainMetastases,45.2,0.82,1243.5,1.0,15234,0.95,0.78,True
   ...
   ```

2. **`quality_summary.json`**: Aggregate statistics
   ```json
   {
     "total_cases": 566,
     "high_quality_cases": 170,
     "quality_threshold": 0.65,
     "sources": {"UCSF_BrainMetastases": 461, "BrainMetShare": 105}
   }
   ```

---

## Training Strategies

### Overview: Two Parallel Approaches

```
┌─────────────────────────────────────────────────────────────┐
│                    PARALLEL TRAINING                        │
├────────────────────────┬────────────────────────────────────┤
│    STRATEGY A          │         STRATEGY B                 │
│  Curriculum Learning   │    Pretrain + Fine-tune            │
├────────────────────────┼────────────────────────────────────┤
│ Phase 1: High-quality  │ Phase 1: Self-supervised on Yale   │
│          bootstrap     │          (masked reconstruction)   │
│          (50 epochs)   │          (100 epochs)              │
├────────────────────────┼────────────────────────────────────┤
│ Phase 2: Full data     │ Phase 2: Supervised fine-tuning    │
│          curriculum    │          on full data              │
│          (100 epochs)  │          (100 epochs)              │
├────────────────────────┼────────────────────────────────────┤
│ Phase 3: Hard sample   │ Phase 3: Hard sample refinement    │
│          focus         │                                    │
│          (50 epochs)   │          (50 epochs)               │
├────────────────────────┴────────────────────────────────────┤
│                    ENSEMBLE (Optional)                      │
│             Combine best models from A and B                │
└─────────────────────────────────────────────────────────────┘
```

### Strategy A: Curriculum Learning

**Concept**: Train on easy examples first, gradually introduce harder ones.

| Phase | Data | Epochs | Key Settings |
|-------|------|--------|--------------|
| 1. Bootstrap | High-quality only | 50 | LR=0.001, Light augmentation |
| 2. Curriculum | Full (progressive) | 100 | Start 70th percentile → all |
| 3. Hard Focus | Full (weighted) | 50 | 10x weight on failures |

**Curriculum Schedule**:
- Epoch 1-50: Quality score ≥ 70th percentile only
- Linear interpolation to include all data by epoch 50
- Epochs 50-100: All data with uniform sampling

### Strategy B: Pretraining + Fine-tuning

**Concept**: Learn brain anatomy from unlabeled data first, then specialize.

| Phase | Data | Task | Epochs |
|-------|------|------|--------|
| 1. Pretrain | Yale (1,430) | Masked reconstruction | 100 |
| 2. Fine-tune | Full supervised | Segmentation | 100 |
| 3. Refine | Full (weighted) | Hard samples | 50 |

**Pretraining Task**:
- Mask 40% of image patches randomly
- Train encoder to reconstruct masked regions
- Transfers general brain anatomy features

### Loss Function Progression

| Phase | Loss Type | Rationale |
|-------|-----------|-----------|
| Bootstrap | Combo (Dice+BCE+Tversky) | Balanced initial learning |
| Curriculum | Small-lesion weighted | 5x weight for <5000 voxel lesions |
| Hard Focus | Tversky (α=0.8) | Maximize recall for tiny lesions |

---

## Implementation Files

### Created Files

| File | Purpose |
|------|---------|
| `scripts/build_superset.py` | Data consolidation + quality filtering |
| `scripts/train_superset.py` | Training implementation (both strategies) |
| `scripts/launch_parallel_training.py` | Parallel execution launcher |
| `configs/training_strategy.yaml` | Complete training configuration |

### Key Classes

```python
# Quality filtering
class QualityFilter:
    def compute_snr(img_data) -> float
    def compute_contrast(img_data, mask_data) -> float
    def compute_sharpness(img_data) -> float
    def compute_mask_quality(mask_data) -> float
    def evaluate_case(case_path) -> QualityMetrics

# Superset building
class SupersetBuilder:
    def process_brainmetshare() -> List[QualityMetrics]
    def process_ucsf() -> List[QualityMetrics]
    def process_yale_pretraining() -> int
    def create_high_quality_subset(metrics) -> List[QualityMetrics]
    def build() -> pd.DataFrame

# Training
class CurriculumSampler:
    def get_sample_weights(epoch) -> np.ndarray
    def get_case_ids(epoch) -> List[str]

class DifficultyWeightedSampler:
    def update_difficulties(case_performances: dict)
    def get_sample_weights() -> np.ndarray
```

---

## Usage Guide

### Step 1: Build the Superset

```bash
cd C:\Users\Clark\TalentAccelorator\brainMetShare\1.2

# Build with default 70% quality threshold
python scripts/build_superset.py

# Or customize
python scripts/build_superset.py --quality-percentile 80  # Top 20% high quality
```

**Output** (in parent directory `brainMetShare/`):
- `../Superset/full/train/` - 566 supervised training cases
- `../Superset/high_quality/train/` - ~170 high-quality cases
- `../Superset/pretraining/` - 1,430 pretraining cases
- `../Superset/metadata.csv` - Quality scores

### Step 2: Run Training

**Option A: Run both strategies sequentially** (single GPU)
```bash
python scripts/train_superset.py --strategy both
```

**Option B: Run strategies in parallel** (2+ GPUs)
```bash
python scripts/launch_parallel_training.py
```

**Option C: Run individual strategy**
```bash
python scripts/train_superset.py --strategy A --gpu 0  # Curriculum only
python scripts/train_superset.py --strategy B --gpu 1  # Pretrain only
```

### Step 3: Evaluate Results

```bash
# Evaluate curriculum model
python evaluate_model.py --model model/curriculum_final.pth

# Evaluate pretrained model
python evaluate_model.py --model model/pretrain_final.pth

# Compare both
python evaluate_model.py --model model/curriculum_final.pth model/pretrain_final.pth
```

### Step 4: Create Ensemble (Optional)

```python
# In Python
import torch

model_a = torch.load('model/curriculum_final.pth')
model_b = torch.load('model/pretrain_final.pth')

# Average predictions during inference
output = 0.5 * model_a(input) + 0.5 * model_b(input)
```

---

## Expected Outcomes

### Performance Predictions

| Metric | Before | After (Conservative) | After (Optimistic) |
|--------|--------|---------------------|-------------------|
| Overall Dice | 67.2% | 75-80% | 82-85% |
| Tiny Lesion Dice | 18.5% | 35-45% | 50-60% |
| Medium Lesion Dice | 64.2% | 75-80% | 80-85% |
| Large Lesion Dice | 67.7% | 78-82% | 83-87% |

### Key Improvements

1. **5x more training data** (105 → 566 supervised cases)
2. **Quality-aware training** (high-quality bootstrap + curriculum)
3. **Self-supervised pretraining** (transfer from 1,430 unlabeled cases)
4. **Hard sample mining** (10x focus on failures)
5. **Tversky loss** (α=0.8 for tiny lesion recall)

### Risk Factors

- Domain shift between UCSF and BrainMetShare (mitigated by mixed training)
- Yale pretraining may not transfer well (backup: skip pretraining phase)
- Memory constraints with larger batch sizes (use gradient accumulation)

---

## Appendix: Configuration Reference

### Full Configuration File

See `configs/training_strategy.yaml` for complete settings.

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Patch size | 96×96×96 | Standard for 3D segmentation |
| Base channels | 24 | Increased from 20 for more capacity |
| Batch size | 2-4 | Memory-dependent |
| Learning rate | 0.001 (phase 1), 0.0005 (phase 2), 0.0001 (phase 3) | Decreasing schedule |
| Small lesion weight | 5.0 → 10.0 | Increased emphasis |
| Tversky α | 0.7-0.8 | Recall-focused |
| Curriculum schedule | 50 epochs | Linear from 70% → 0% |
| Difficulty multiplier | 10.0 | Hard sample oversampling |

---

## Changelog

- **2026-01-18**: Initial implementation
  - Created superset builder with quality filtering
  - Implemented dual training strategy (curriculum + pretraining)
  - Added parallel training launcher
  - Built Superset with 566 supervised + 1,430 pretraining cases
