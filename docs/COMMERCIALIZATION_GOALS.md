# Commercialization Goals & External Validation Strategy

*Created: March 2026*

---

## Target: Partnership with a Hospital or Research Lab

The near-term goal is to demonstrate a strong enough proof-of-concept to
secure a partnership with a research hospital or lab for access to a larger,
privately curated dataset and clinical collaboration.

This is more realistic than a direct product sale (which requires FDA
clearance, multi-site clinical studies, and $5-20M+ investment). A partnership
lets us build the model that could eventually become that product.

### What We're Pitching

"Here's what 566 public cases gets you. Imagine what 5,000 curated cases from
your institution could do."

We need to show:
1. A working pipeline that produces competitive results
2. Detailed understanding of where and why it fails
3. A credible argument that more data closes the gap
4. External validation proving it generalizes (not just overfitting one dataset)

---

## Performance Targets

### Primary Metrics (Internal Validation)

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Val Dice (mean) | 0.76 | >= 0.80 | In progress |
| Val Dice (median) | 0.78 | >= 0.83 | In progress |
| Lesion recall (all) | 0.725 | >= 0.85 | Critical gap |
| Lesion recall (>= 3mm) | ~0.80 | >= 0.90 | Need to measure |
| Lesion precision | 0.74-0.81 | >= 0.80 | Close |
| Surface Dice @2mm | ~0.65 | >= 0.72 | In progress |
| HD95 | 18-22 | <= 15 | In progress |

### External Validation Targets

| Dataset | Cases | Target Dice | Purpose |
|---------|-------|-------------|---------|
| BraTS-METS 2023 | 238 | >= 0.75 | Same-task external |
| PRETREAT (Yale/TCIA) | 200 | >= 0.73 | Same-task, different institution |
| Competition test set | 51 | >= 0.78 | Official benchmark |
| BraTS Glioma (subset) | ~100 | any detection | Cross-domain experiment |

The key insight: **a model that does 0.78 on two independent datasets is more
compelling than one that does 0.82 on one.**

### Reliability of Current Metrics

With 84 val cases and std ~0.20:
- 95% CI: reported Dice +/- 0.043 (a 9-point range)
- Completing 5-fold CV (all 566 cases) tightens to +/- 0.016
- Threshold/weight tuning on val adds ~0.005-0.015 optimistic bias
- Real-world generalization gap: typically 5-15% Dice drop

What a reported 0.75 really means:
- Same dataset (honest held-out): 0.73-0.75
- Same institution, new patients: 0.70-0.75
- Different institution, similar protocol: 0.63-0.72
- Very different scanner/protocol: 0.55-0.68

---

## External Validation Datasets

### Tier 1: Direct 4-Channel Compatibility (Plug-and-Play)

**BraTS-METS 2023** — HIGHEST PRIORITY
- Source: https://www.synapse.org/Synapse:syn51156910
- Cases: 238 annotated training (multi-institutional)
- Sequences: T1, T1-Gd, T2, FLAIR (exact match to our pipeline)
- Masks: Yes (NETC + SNFH + ET — merge to binary)
- Format: NIfTI, 1mm isotropic, skull-stripped
- Access: Free Synapse registration
- Notes: Different institutions from our training data. The single most
  important external validation dataset for brain mets.

**PRETREAT-METSTOBRAIN-MASKS (Yale/TCIA)**
- Source: https://www.cancerimagingarchive.net/collection/pretreat-metstobrain-masks/
- Paper: Nature Scientific Data 2024
- Cases: 200 patients (pretreatment scans)
- Sequences: T1pre, T1post, T2, FLAIR (exact match)
- Masks: Yes (necrosis + edema + enhancing — merge to binary)
- Format: NIfTI
- Access: Free TCIA agreement
- Notes: Yale data — completely independent from Stanford/UCSF sources.
  Diverse primary cancers (NSCLC 43%, melanoma 20%, breast 13%).

**MOTUM Multi-Center**
- Source: https://doi.gin.g-node.org/10.12751/g-node.tvzqc5/
- Cases: 38 with brain mets (+ 29 gliomas)
- Sequences: T1, T1-CE, T2, FLAIR (all four)
- Masks: Yes
- Format: DICOM + NIfTI
- Limitation: Small (38 mets), 5mm slice thickness

### Tier 2: Minor Adaptation Needed

**UCSF-BMSR** (partially overlaps our data)
- 461 training cases with T1, T1-CE, FLAIR, Subtraction
- Missing T2 — would need subtraction as 4th channel or zero-fill

**MSLesSeg (Multiple Sclerosis)**
- 115 scans, T1/T2/FLAIR (no T1-Gd)
- MS lesions are small white matter lesions — analogous to tiny brain mets
- Could test with zero-filled T1-Gd channel

### Tier 3: Cross-Domain Experiment

**BraTS Adult Glioma 2023/2024**
- ~1,470 cases, T1/T1-CE/T2/FLAIR (perfect channel match)
- Gliomas are very different morphologically (larger, diffuse, usually single)
- Tests whether our model learned "lesion detection" vs "brain met detection"
- High upside: if it detects gliomas at all, argues for transferable architecture

---

## The Transferability Thesis

Our architecture (LightweightUNet3D + nnU-Net ensemble) is trained to detect
brain lesions ranging from <100 voxels to >20,000 voxels on multi-sequence
MRI. The hypothesis:

> If the model learns generalizable features for "abnormal tissue on
> multi-contrast MRI" rather than memorizing brain-met-specific patterns,
> the same architecture should transfer to other lesion types with minimal
> retraining.

### Cross-Domain Experiments (Aspirational)

| Modality | Lesion Type | Feasibility | Notes |
|----------|-------------|-------------|-------|
| Brain MRI | Gliomas (BraTS) | High | Same channels, direct test |
| Brain MRI | MS lesions | Medium | Missing T1-Gd channel |
| Brain MRI | Stroke | Low | Different sequences (DWI/ADC) |
| Mammography | Breast lesions | Future | 2D, completely different modality |
| Chest X-ray | Lung nodules | Future | 2D, different modality |
| Body CT | Liver/kidney lesions | Future | Different modality, 3D applicable |

The brain MRI cross-domain tests are achievable now. Mammography/X-ray would
require architecture changes (2D) and new training data, but the core
detection principles (multi-scale attention, size-adaptive loss) transfer.

### Why This Matters for a Partnership Pitch

A demo showing "here's our brain met model detecting gliomas zero-shot" is
extremely compelling, even if performance is mediocre. It argues:
1. The architecture learns generalizable lesion features
2. With your institution's labeled data, we can fine-tune to your specific task
3. The pipeline is flexible — not a one-trick model

---

## Action Plan

### Phase 1: Maximize Current Performance (Now)
- [ ] Complete nnU-Net 5-fold cross-validation (folds 3-4)
- [ ] Run cross-model ensemble evaluation
- [ ] Generate competition test set predictions
- [ ] Target: 0.80 internal val Dice

### Phase 2: External Validation (Next)
- [ ] Download BraTS-METS 2023 (Synapse registration)
- [ ] Download PRETREAT-METSTOBRAIN-MASKS (TCIA agreement)
- [ ] Build preprocessing pipeline to normalize external data
- [ ] Run model on both datasets, report honest numbers
- [ ] Target: 0.75+ Dice on at least one external dataset

### Phase 3: Cross-Domain Experiments (Then)
- [ ] Run brain met model on BraTS Glioma data (zero-shot)
- [ ] Analyze what it detects vs misses (enhancing vs non-enhancing)
- [ ] If promising, fine-tune on small glioma subset and measure improvement
- [ ] Document transferability results

### Phase 4: Partnership Materials (When Ready)
- [ ] One-page technical summary with key metrics
- [ ] Visualization gallery (good cases, failure cases, cross-domain)
- [ ] GitHub repo cleanup and documentation
- [ ] Identify target labs/hospitals and prepare outreach

---

## Key Numbers for the Pitch

When we have them, these are the headline metrics:

```
Internal validation (5-fold CV, 566 cases):   Dice = ?.??
External validation (BraTS-METS, 238 cases):  Dice = ?.??
External validation (Yale TCIA, 200 cases):   Dice = ?.??
Competition test set (51 cases):               Dice = ?.??
Cross-domain glioma detection:                 [qualitative]
Human radiologist baseline:                    Dice = 0.85
```

The story writes itself once we fill in those numbers.
