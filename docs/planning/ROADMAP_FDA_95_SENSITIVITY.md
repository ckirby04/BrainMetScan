# Brain Metastasis Segmentation: Path to 95% Sensitivity & FDA 510(k)

## Goals

- **Primary**: 95% sensitivity (lesion detection rate) across all sizes
- **Tiny Lesion Target**: 85% sensitivity for lesions <500 voxels (currently ~25%)
- **Regulatory**: FDA 510(k) clearance for clinical use
- **Constraint**: Solo developer, bootstrapped

---

## Current State vs Target State

| Metric | Baseline (96³) | Best (48³) | Target |
|--------|---------------|------------|--------|
| Tiny Lesion Dice | 18.5% | 50.4% | 70%+ |
| Tiny Lesion Sensitivity | ~25% | ~60% | **85%** |
| Overall Sensitivity | ~65% | ~70% | **95%** |
| Test Dice | 54.77% | 56.7% | 65%+ |

**Key Insight**: Smaller patches dramatically improve tiny lesion detection (2.7x improvement proven).

---

## Phase 1: Technical Performance (Weeks 1-8)

### Step 1.1: Train the 24³ Ultra-Fine Model (Weeks 1-2)

**Why**: Smaller patches (24³ vs 48³ vs 96³) dramatically improve tiny lesion detection. The 48³ model already showed 2.7x improvement over 96³ for tiny lesions.

**How**:
1. Config already exists at `configs/config_tiny_lesion_24patch.yaml`
2. Run: `python scripts/train_tiny_lesion.py --config configs/config_tiny_lesion_24patch.yaml`
3. Training uses:
   - 24³ patches with 85% overlap during inference
   - Tversky loss (α=0.15, β=0.85) to prioritize recall
   - 25x weight boost for tiny lesions (<500 voxels)
   - Difficulty-based sampling to oversample hard cases

**Expected Outcome**: 65-75% Dice on tiny lesions (vs 50.4% with 48³)

### Step 1.2: Build Multi-Scale Ensemble (Weeks 3-4)

**Why**: Different patch sizes excel at different lesion sizes. Combining them captures the best of each.

**Architecture**:
```
Ensemble System
├── 24³ model → Specializes in tiny lesions (<500 voxels)
├── 48³ model → Handles small/medium lesions (500-5000 voxels)
└── 96³ model → Captures context for large lesions (>5000 voxels)
```

**Fusion Strategy**:
1. Run all three models on each case via sliding window inference
2. Use connected component analysis to identify lesion candidates
3. Route predictions based on candidate size:
   - Tiny candidates: weight 24³ model heavily (e.g., 60%)
   - Medium candidates: weight 48³ model (50%)
   - Large candidates: weight 96³ model (50%)
4. Apply confidence-weighted soft voting
5. Use lower threshold for tiny lesion channel (0.3 vs 0.5)

**Implementation**: Create `src/segmentation/ensemble.py` that:
- Loads all three trained models
- Runs parallel inference
- Implements size-aware fusion
- Outputs combined probability map

### Step 1.3: Maximize Sensitivity (Weeks 5-6)

**Threshold Optimization**:
- Current default: 0.5 (balanced precision/recall)
- For 95% sensitivity: sweep thresholds 0.2-0.4
- Lower thresholds catch more lesions at cost of more false positives
- Acceptable trade-off: radiologists will verify, missed lesions are worse than false alarms

**Aggressive Test-Time Augmentation**:
- The codebase has `AdaptiveTTA` in `src/segmentation/tta.py`
- Apply 8-16 augmentation variations (flips, rotations, brightness)
- Use **union** of predictions (maximize recall) instead of average
- Every augmentation that detects a lesion contributes to final prediction

**Sensitivity-Focused Post-Processing**:
1. Morphological dilation (3x3x3 kernel) to catch partial detections
2. Remove minimum size filtering (catch all tiny lesions)
3. Connected component labeling for lesion counting
4. No aggressive hole-filling that might remove small detections

### Step 1.4: Internal Validation (Weeks 7-8)

**Create Sensitivity Evaluation Script** (`scripts/evaluate_sensitivity.py`):

```
For each test case:
1. Run ensemble inference
2. Extract ground truth lesions (connected components)
3. Extract predicted lesions (connected components)
4. Match predictions to ground truth (IoU > 0.1 = detected)
5. Calculate per-lesion sensitivity by size bucket:
   - Tiny (<500 voxels): Target 85%
   - Small (500-2000 voxels): Target 90%
   - Medium (2000-5000 voxels): Target 95%
   - Large (>5000 voxels): Target 98%
   - Overall: Target 95%
6. Track false positives per case
7. Generate detailed metrics report
```

**Iterate Until Targets Met**:
- If tiny lesion sensitivity < 85%: lower threshold further, increase TTA
- If too many false positives: adjust post-processing, tune ensemble weights
- Document all parameter choices for FDA submission

---

## Phase 2: External Validation (Weeks 9-16)

### Step 2.1: Acquire External Datasets

**Why Required**: FDA requires multi-site validation to prove generalizability.

**Target Sources**:
1. BraTS-METS Challenge dataset (public)
2. Stanford AIMI Center datasets
3. Academic collaborators (negotiate data sharing agreements)

**Minimum Requirements**:
- 3+ different institutions
- 200+ patients external to training data
- Diverse scanner manufacturers (GE, Siemens, Philips)
- Mix of 1.5T and 3T field strengths

### Step 2.2: Multi-Site Validation Protocol

**For Each External Site**:
1. Apply identical preprocessing pipeline (normalization, resampling)
2. Run ensemble inference with frozen parameters
3. Calculate per-lesion sensitivity (same methodology as internal)
4. Compare to radiologist ground truth annotations
5. Document any performance degradation
6. Analyze failure cases by scanner type, sequence, lesion characteristics

**Target**: Sensitivity ≥95% maintained across all external sites

---

## Phase 3: FDA 510(k) Preparation (Weeks 17-32)

### Step 3.1: Predicate Device Research

**Find Substantially Equivalent Devices**:
- Search FDA 510(k) database for "brain lesion detection", "brain metastasis"
- Potential predicates: Viz.ai, Aidoc, Subtle Medical, icometrix
- Document how your device compares in intended use, technology, performance

### Step 3.2: Quality Management System (QMS)

**Required Documentation**:
1. **Design History File (DHF)**: Record of all design decisions
2. **Software Development Life Cycle (SDLC)**: Version control, testing, releases
3. **Risk Management File (ISO 14971)**: Hazard analysis, mitigations
4. **Verification & Validation (V&V)**: Test protocols, results
5. **Cybersecurity Documentation**: Threat model, security controls
6. **Software Bill of Materials (SBOM)**: All dependencies listed

**Solo Developer Approach**:
- Use QMS templates (OpenRegulatory is free, Greenlight Guru has templates)
- Budget $20-50K for regulatory consultant to review
- Follow IEC 62304 for software lifecycle
- Follow ISO 14971 for risk management

### Step 3.3: Clinical Evidence Package

**Study Design**: Multi-Reader Multi-Case (MRMC) Study
- Retrospective study using collected cases
- 3+ board-certified radiologists as readers
- Each reader reviews cases with and without AI assistance
- Washout period between reads

**Primary Endpoint**: Per-lesion sensitivity improvement with AI assistance

**Statistical Requirements**:
- Pre-specified sample size calculation
- 95% confidence interval lower bound ≥90% sensitivity
- Pre-registered analysis plan
- Handling of reader variability

### Step 3.4: Submit 510(k)

**Submission Contents**:
1. Device description and intended use
2. Substantial equivalence argument to predicate
3. Performance testing (bench testing + clinical study)
4. Labeling (Instructions for Use, warnings, contraindications)
5. Software documentation per IEC 62304

**Timeline & Cost**:
- FDA user fee: ~$15K (small business rate)
- Regulatory consulting: $20-50K
- Review timeline: 6-12 months after submission

---

## Phase 4: Commercialization (Post-FDA)

### Step 4.1: Pre-FDA Revenue (Research Use Only)

**While Awaiting Clearance**:
- Sell to academic medical centers as "Research Use Only"
- Price: $5-20K/year
- No clinical use claims
- Builds relationships and generates evidence

### Step 4.2: Post-FDA Launch

**Target Market**: Community hospitals (underserved by neuroradiology)

**Pricing**: $50-100K/year per site

**Technical Infrastructure**:
- HIPAA-compliant cloud (AWS GovCloud or Azure Healthcare)
- DICOM integration for seamless workflow
- SOC 2 Type II compliance for enterprise sales

---

## Budget Summary

| Item | Cost |
|------|------|
| GPU compute | $500-2K |
| External datasets | $0-5K |
| Regulatory consultant | $20-50K |
| FDA user fee | ~$15K |
| Clinical study (readers) | $10-30K |
| Legal | $5-10K |
| **Total to FDA Clearance** | **$50-110K** |

---

## Timeline Summary

| Phase | Duration | Milestone |
|-------|----------|-----------|
| Technical (95% sensitivity) | Weeks 1-8 | Multi-scale ensemble achieving target |
| External validation | Weeks 9-16 | 3+ site validation complete |
| FDA preparation | Weeks 17-32 | 510(k) submission ready |
| FDA review | Weeks 33-52 | Clearance received |
| Commercial launch | Week 53+ | First paying customer |

**Total: ~12-18 months to FDA clearance**

---

## Immediate Next Actions

1. **Train 24³ model** - Config ready, execute training command
2. **Create ensemble.py** - Combine 24³/48³/96³ model predictions
3. **Create evaluate_sensitivity.py** - Per-lesion detection metrics by size
4. **Threshold sweep** - Find optimal threshold for 95% sensitivity
5. **Research predicates** - FDA 510(k) database search

---

## Critical Files

| Purpose | Path |
|---------|------|
| 24³ config | `configs/config_tiny_lesion_24patch.yaml` |
| Training script | `scripts/train_tiny_lesion.py` |
| Evaluation | `scripts/evaluate_model.py` |
| TTA implementation | `src/segmentation/tta.py` |
| Inference | `src/segmentation/inference.py` |
| **New: Ensemble** | `src/segmentation/ensemble.py` (to create) |
| **New: Sensitivity eval** | `scripts/evaluate_sensitivity.py` (to create) |

---

## Verification Criteria

Implementation is successful when:

- [ ] 24³ model trained with Dice >65% on tiny lesions
- [ ] Ensemble achieves ≥85% sensitivity on tiny lesions (<500 voxels)
- [ ] Ensemble achieves ≥95% overall sensitivity
- [ ] All metrics documented in `model/metrics.json`
- [ ] External validation maintains performance

---

## Technical Implementation Details

### Ensemble Architecture (ensemble.py)

```python
# Conceptual structure for src/segmentation/ensemble.py

class MultiScaleEnsemble:
    """
    Ensemble of models trained at different patch sizes.
    Routes predictions based on lesion size for optimal detection.
    """

    def __init__(self, model_paths: dict, device='cuda'):
        """
        model_paths = {
            'tiny': 'model/tiny_lesion_24patch_best.pth',    # 24³ patches
            'small': 'model/tiny_lesion_48patch_best.pth',   # 48³ patches
            'large': 'model/medium_lesion_96patch_best.pth'  # 96³ patches
        }
        """
        self.models = {}
        self.patch_sizes = {
            'tiny': (24, 24, 24),
            'small': (48, 48, 48),
            'large': (96, 96, 96)
        }
        # Load each model...

    def predict(self, image, threshold=0.5):
        """
        Run all models and fuse predictions.

        1. Run sliding window inference with each model
        2. Identify candidate regions via connected components
        3. For each candidate, weight models by candidate size
        4. Apply threshold for final binary mask
        """
        pass

    def _size_aware_fusion(self, predictions: dict, candidate_sizes: list):
        """
        Weight model predictions based on lesion candidate size.

        - Tiny (<500 voxels): 60% tiny model, 25% small, 15% large
        - Small (500-2000): 25% tiny, 50% small, 25% large
        - Medium (2000-5000): 15% tiny, 35% small, 50% large
        - Large (>5000): 10% tiny, 30% small, 60% large
        """
        pass
```

### Sensitivity Evaluation (evaluate_sensitivity.py)

```python
# Conceptual structure for scripts/evaluate_sensitivity.py

def evaluate_per_lesion_sensitivity(pred_mask, gt_mask, iou_threshold=0.1):
    """
    Calculate per-lesion detection rate.

    1. Extract connected components from ground truth
    2. Extract connected components from prediction
    3. For each GT lesion, check if any prediction overlaps with IoU > threshold
    4. Return detection rate by size bucket
    """

    size_buckets = {
        'tiny': (0, 500),
        'small': (500, 2000),
        'medium': (2000, 5000),
        'large': (5000, float('inf'))
    }

    # For each GT lesion:
    #   - Compute size (voxel count)
    #   - Check if detected (IoU with any prediction > threshold)
    #   - Categorize into bucket

    # Return sensitivity per bucket + overall

def main():
    # Load test cases
    # Run ensemble inference
    # Calculate per-lesion sensitivity
    # Output detailed report
    # Save metrics to model/metrics.json
```

### Threshold Optimization

```python
# Add to inference.py or create scripts/optimize_threshold.py

def find_optimal_threshold(model, val_loader, target_sensitivity=0.95):
    """
    Sweep thresholds to find one achieving target sensitivity.

    1. Run inference on validation set with threshold=0.0 (get probabilities)
    2. For threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        - Apply threshold
        - Calculate sensitivity
        - If sensitivity >= target: record threshold and specificity
    3. Return lowest threshold achieving target (maximizes specificity)
    """
    pass
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| 24³ model doesn't improve further | Use 32³ intermediate, try self-supervised pretraining |
| Ensemble too slow for clinical use | Implement cascade (coarse→fine), GPU optimization |
| External validation drops significantly | Domain adaptation, scanner-specific fine-tuning |
| FDA rejects predicate | Pivot to De Novo pathway (longer, more expensive) |
| Budget overrun | Prioritize RUO sales to fund regulatory path |

---

*Document created: 2026-01-25*
*Last updated: 2026-01-25*
