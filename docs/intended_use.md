# BrainMetScan — Intended Use Statement

**Product Name:** BrainMetScan
**Version:** 1.23.0
**Classification:** Software as a Medical Device (SaMD) — Class II (anticipated)
**Date:** February 2026

---

## 1. Intended Use

BrainMetScan is an AI-powered software tool intended to assist qualified neuroradiologists and radiation oncologists in the **detection and volumetric segmentation of brain metastases** on multi-sequence MRI scans.

The software analyzes four standard MRI sequences (T1 pre-contrast, T1 post-gadolinium, FLAIR, and T2-weighted) and produces:

1. **Automated segmentation masks** highlighting suspected metastatic lesions
2. **Volumetric measurements** for each detected lesion (volume in mm³, longest diameter in mm)
3. **RECIST 1.1 measurements** including Sum of Diameters for treatment response monitoring
4. **Longitudinal comparison** between baseline and follow-up scans to track disease progression

## 2. Indications for Use

BrainMetScan is indicated for use as a **computer-aided detection/diagnosis (CADe/CADx)** tool to:

- Identify and delineate suspected brain metastases on contrast-enhanced MRI
- Quantify lesion volumes and diameters for treatment planning and monitoring
- Compare sequential MRI scans to assess treatment response per RECIST 1.1 criteria
- Reduce reader variability in lesion measurement and counting

## 3. Target User Population

- **Primary users:** Board-certified neuroradiologists, diagnostic radiologists, and radiation oncologists
- **Secondary users:** Neurosurgeons, medical oncologists, and clinical trial coordinators reviewing AI-assisted findings
- All users must have training in brain MRI interpretation and familiarity with brain metastasis imaging patterns.

## 4. Target Patient Population

- Adult patients (≥18 years) undergoing brain MRI for known or suspected brain metastases
- Patients with primary cancers known to metastasize to the brain (lung, breast, melanoma, renal cell carcinoma, colorectal, etc.)
- Patients undergoing treatment response monitoring for existing brain metastases

## 5. Clinical Environment

- Hospital radiology departments and outpatient imaging centers
- Academic medical centers and clinical trial sites
- Neuro-oncology clinics with PACS-integrated workflows

## 6. Input Requirements

| Sequence | Required | Typical Parameters |
|----------|----------|--------------------|
| T1 pre-contrast | Yes | 3D volumetric, ≤1.5mm slice thickness |
| T1 post-gadolinium | Yes | 3D volumetric, ≤1.5mm slice thickness |
| FLAIR | Yes | 3D or 2D, ≤3mm slice thickness |
| T2-weighted | Yes | 3D or 2D, ≤3mm slice thickness |

**Supported formats:** DICOM, NIfTI
**Supported field strengths:** 1.5T and 3.0T MRI scanners
**Supported vendors:** Scanner-agnostic (validated on Siemens, GE, Philips*)

*\* Multi-vendor validation pending — see Limitations.*

## 7. Contraindications

BrainMetScan should **NOT** be used:

- As a standalone diagnostic tool without physician review
- For primary brain tumors (glioma, meningioma, etc.) — the model is trained specifically on metastatic lesions
- On pediatric patients (<18 years)
- On MRI scans without gadolinium-based contrast enhancement
- On scans with significant motion artifacts, susceptibility artifacts, or incomplete sequence coverage
- For treatment planning decisions without independent radiologist confirmation

## 8. Limitations

1. **Research use only** — BrainMetScan has not been cleared or approved by the FDA, CE-marked, or approved by any regulatory body
2. **Single-institution training** — Current models are trained on data from a single institution; multi-site generalization has not been validated
3. **Lesion size** — Detection sensitivity may be reduced for very small lesions (<5mm diameter)
4. **Leptomeningeal disease** — Not designed to detect or measure leptomeningeal metastatic disease
5. **Post-surgical cavities** — May produce false positives in post-surgical resection cavities
6. **Radiation necrosis** — Cannot distinguish radiation necrosis from viable tumor (a known MRI limitation)
7. **Scanner variability** — Performance may vary across MRI scanner vendors, field strengths, and acquisition protocols not represented in the training data

## 9. Performance Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Voxel-level Sensitivity | 95.6% | Smart ensemble at threshold 0.6 |
| Lesion-level Detection Rate | 92.2% | Per-lesion sensitivity |
| Training Dataset | 566 cases | Single institution, multi-sequence MRI |
| Architecture | 4-model ensemble | Multi-scale patch sizes (8, 12, 24, 36) |

*Detailed validation results available upon request.*

## 10. Regulatory Pathway

BrainMetScan is being developed following the FDA's **predetermined change control plan** framework for AI/ML-based SaMD:

- **Target classification:** De Novo or 510(k) Class II
- **Predicate devices:** FDA-cleared AI radiology tools (e.g., quantitative volumetric analysis software)
- **Quality Management System:** Under development per ISO 13485:2016
- **Risk Management:** Per ISO 14971:2019 (see Risk Analysis document)
- **Software Lifecycle:** Per IEC 62304:2006+AMD1:2015

## 11. Post-Market Surveillance Plan

Upon regulatory clearance, BrainMetScan will implement:

- Continuous performance monitoring on production cases
- Automated drift detection comparing production metrics to validation baselines
- Quarterly performance reports
- Adverse event reporting per 21 CFR Part 803
- User feedback collection and analysis

---

**DISCLAIMER:** This document describes the intended future use of BrainMetScan. The software is currently designated for **Research Use Only (RUO)** and must not be used for clinical decision-making until regulatory clearance is obtained.
