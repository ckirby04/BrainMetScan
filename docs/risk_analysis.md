# BrainMetScan — Risk Analysis

**Document Type:** Risk Management Report (per ISO 14971:2019)
**Product:** BrainMetScan v1.23.0
**Classification:** SaMD Class II (anticipated)
**Date:** February 2026
**Status:** Draft — Research Use Only

---

## 1. Scope

This document identifies, evaluates, and mitigates risks associated with the BrainMetScan brain metastasis segmentation system. It follows the framework of ISO 14971:2019 "Medical devices — Application of risk management to medical devices."

## 2. Risk Acceptability Criteria

| Severity | Probability: Frequent | Probability: Probable | Probability: Occasional | Probability: Remote | Probability: Improbable |
|----------|----------------------|----------------------|------------------------|--------------------|-----------------------|
| **Catastrophic** | Unacceptable | Unacceptable | Unacceptable | ALARP | ALARP |
| **Critical** | Unacceptable | Unacceptable | ALARP | ALARP | Acceptable |
| **Serious** | Unacceptable | ALARP | ALARP | Acceptable | Acceptable |
| **Minor** | ALARP | ALARP | Acceptable | Acceptable | Acceptable |
| **Negligible** | Acceptable | Acceptable | Acceptable | Acceptable | Acceptable |

**ALARP** = As Low As Reasonably Practicable (requires mitigation)

---

## 3. Hazard Identification and Risk Assessment

### H-001: False Negative — Missed Brain Metastasis

| Attribute | Value |
|-----------|-------|
| **Hazard** | AI fails to detect an existing brain metastasis |
| **Hazardous Situation** | Clinician relies on AI output and does not independently identify the lesion |
| **Harm** | Delayed treatment; disease progression; potential neurological decline |
| **Severity** | Critical |
| **Pre-mitigation Probability** | Occasional |
| **Pre-mitigation Risk** | ALARP |

**Mitigations:**
1. Clear labeling as **CADe tool** — not a replacement for radiologist review (M-001)
2. Intended use requires physician review of all scans regardless of AI findings (M-002)
3. Multi-scale ensemble architecture improves sensitivity to 92.2% lesion detection (M-003)
4. System reports confidence scores for each detection — low-confidence regions flagged (M-004)
5. Training on diverse lesion sizes including tiny lesions (<5mm) (M-005)

**Post-mitigation Probability:** Remote
**Post-mitigation Risk:** ALARP — acceptable given physician oversight requirement

---

### H-002: False Positive — Incorrectly Identified Metastasis

| Attribute | Value |
|-----------|-------|
| **Hazard** | AI identifies a non-metastatic structure as a brain metastasis |
| **Hazardous Situation** | Clinician acts on false positive without verification |
| **Harm** | Unnecessary biopsy, surgery, or radiation; patient anxiety; healthcare costs |
| **Severity** | Serious |
| **Pre-mitigation Probability** | Probable |
| **Pre-mitigation Risk** | ALARP |

**Mitigations:**
1. Postprocessing pipeline removes small spurious detections (M-006)
2. Per-lesion confidence scores allow filtering at adjustable thresholds (M-007)
3. DICOM-SEG output enables side-by-side review in PACS viewers (M-008)
4. Intended use requires radiologist confirmation before any intervention (M-002)

**Post-mitigation Probability:** Occasional
**Post-mitigation Risk:** ALARP — acceptable given physician confirmation requirement

---

### H-003: Incorrect Volume Measurement

| Attribute | Value |
|-----------|-------|
| **Hazard** | Segmentation boundary is inaccurate, leading to incorrect volume measurement |
| **Hazardous Situation** | Treatment response assessed incorrectly based on volume change |
| **Harm** | Premature treatment discontinuation or unnecessary treatment escalation |
| **Severity** | Serious |
| **Pre-mitigation Probability** | Probable |
| **Pre-mitigation Risk** | ALARP |

**Mitigations:**
1. RECIST 1.1 response classification uses diameter-based criteria with standard thresholds (M-009)
2. Longitudinal comparison shows matched lesion pairs with volume change percentages (M-010)
3. Probability maps available for manual boundary adjustment (M-011)
4. Measurement precision validated against expert annotations (M-012)

**Post-mitigation Probability:** Occasional
**Post-mitigation Risk:** ALARP — acceptable with physician review of measurements

---

### H-004: Wrong Patient / Data Mismatch

| Attribute | Value |
|-----------|-------|
| **Hazard** | Input data from different patients is processed together, or results are attributed to wrong patient |
| **Hazardous Situation** | Clinician receives segmentation results for wrong patient |
| **Harm** | Incorrect treatment decisions for both patients involved |
| **Severity** | Critical |
| **Pre-mitigation Probability** | Remote |
| **Pre-mitigation Risk** | ALARP |

**Mitigations:**
1. DICOM handler validates PatientID consistency within uploaded series (M-013)
2. Case ID tracked through the entire pipeline (M-014)
3. API audit logging records all input files and timestamps (M-015)
4. Database persistence enables traceability from result back to input (M-016)

**Post-mitigation Probability:** Improbable
**Post-mitigation Risk:** Acceptable

---

### H-005: System Unavailability

| Attribute | Value |
|-----------|-------|
| **Hazard** | System crashes or becomes unavailable during clinical workflow |
| **Hazardous Situation** | Clinician cannot access AI results when needed for time-sensitive decision |
| **Harm** | Treatment delay; clinician must proceed without AI assistance |
| **Severity** | Minor |
| **Pre-mitigation Probability** | Occasional |
| **Pre-mitigation Risk** | Acceptable |

**Mitigations:**
1. Docker containerization with automatic restart policy (M-017)
2. Health check endpoint for monitoring (M-018)
3. System designed as supplementary tool — clinical workflow not dependent on AI availability (M-019)

**Post-mitigation Risk:** Acceptable

---

### H-006: Data Privacy Breach

| Attribute | Value |
|-----------|-------|
| **Hazard** | Patient imaging data or PHI is exposed to unauthorized parties |
| **Hazardous Situation** | Data transmitted without encryption or stored insecurely |
| **Harm** | HIPAA violation; patient privacy compromise; legal liability |
| **Severity** | Critical |
| **Pre-mitigation Probability** | Remote |
| **Pre-mitigation Risk** | ALARP |

**Mitigations:**
1. API key authentication with rate limiting (M-020)
2. No PHI stored in system logs — only anonymized case IDs (M-021)
3. Temporary file cleanup after processing (M-022)
4. HTTPS/TLS encryption for all API communication (deployment requirement) (M-023)
5. Audit logging for all data access events (M-024)

**Post-mitigation Probability:** Improbable
**Post-mitigation Risk:** Acceptable

---

### H-007: Model Degradation Over Time

| Attribute | Value |
|-----------|-------|
| **Hazard** | Model performance degrades due to distribution shift (new scanners, protocols, patient populations) |
| **Hazardous Situation** | System produces increasing false negatives or false positives without detection |
| **Harm** | Systematic diagnostic errors across patient population |
| **Severity** | Critical |
| **Pre-mitigation Probability** | Probable (over long deployment) |
| **Pre-mitigation Risk** | Unacceptable |

**Mitigations:**
1. Performance monitoring via database analytics — detection rate tracked over time (M-025)
2. Benchmark suite for periodic revalidation (M-026)
3. Model registry with version tracking — rollback capability (M-027)
4. Post-market surveillance plan with quarterly performance reviews (M-028)
5. Multi-site validation planned before production deployment (M-029)

**Post-mitigation Probability:** Remote
**Post-mitigation Risk:** ALARP — acceptable with monitoring

---

### H-008: Incorrect Sequence Identification (DICOM)

| Attribute | Value |
|-----------|-------|
| **Hazard** | DICOM handler misidentifies MRI sequence type (e.g., FLAIR as T2) |
| **Hazardous Situation** | Model receives incorrectly ordered input channels |
| **Harm** | Degraded segmentation accuracy; false negatives or positives |
| **Severity** | Serious |
| **Pre-mitigation Probability** | Occasional |
| **Pre-mitigation Risk** | ALARP |

**Mitigations:**
1. Sequence identification uses multiple DICOM tags (SeriesDescription, ProtocolName, etc.) (M-030)
2. Heuristic matching with confidence reporting (M-031)
3. NIfTI upload path allows explicit sequence labeling by user (M-032)
4. Warning when sequence identification confidence is low (M-033)

**Post-mitigation Probability:** Remote
**Post-mitigation Risk:** Acceptable

---

## 4. Risk-Benefit Analysis

### Benefits
- Reduced inter-reader variability in lesion counting and measurement
- Faster lesion detection and volumetric analysis (seconds vs. minutes)
- Standardized RECIST 1.1 measurements for clinical trials
- Automated longitudinal tracking reduces manual tracking burden
- Potential to detect small lesions missed by visual inspection

### Residual Risks
All residual risks are at or below ALARP level, contingent on:
- System use within intended use conditions
- Qualified physician review of all AI findings
- Proper deployment with security controls (HTTPS, authentication)
- Ongoing performance monitoring

### Conclusion
The benefits of BrainMetScan outweigh the residual risks when used within the specified intended use by qualified clinicians with appropriate oversight.

---

## 5. Mitigation Traceability Matrix

| Mitigation ID | Description | Implemented | Verified |
|--------------|-------------|-------------|----------|
| M-001 | CADe labeling in UI and reports | Yes | - |
| M-002 | Physician review requirement in intended use | Yes | - |
| M-003 | Multi-scale ensemble architecture | Yes | Yes |
| M-004 | Per-lesion confidence scores | Yes | Yes |
| M-005 | Tiny lesion training data | Yes | Yes |
| M-006 | Small component removal postprocessing | Yes | Yes |
| M-007 | Adjustable confidence threshold | Yes | Yes |
| M-008 | DICOM-SEG output for PACS | Yes | - |
| M-009 | RECIST 1.1 response classification | Yes | Yes |
| M-010 | Longitudinal lesion matching | Yes | Yes |
| M-011 | Probability map download | Yes | Yes |
| M-012 | Validation against expert annotations | Partial | - |
| M-013 | DICOM PatientID consistency check | Yes | - |
| M-014 | Case ID tracking | Yes | Yes |
| M-015 | API audit logging | Yes | Yes |
| M-016 | Database persistence | Yes | Yes |
| M-017 | Docker auto-restart | Yes | - |
| M-018 | Health check endpoint | Yes | Yes |
| M-019 | Supplementary tool design | Yes | - |
| M-020 | API key authentication | Yes | Yes |
| M-021 | No PHI in logs | Yes | - |
| M-022 | Temp file cleanup | Yes | Yes |
| M-023 | HTTPS/TLS (deployment) | Config | - |
| M-024 | Audit logging | Yes | Yes |
| M-025 | Performance analytics | Yes | - |
| M-026 | Benchmark suite | Yes | Yes |
| M-027 | Model registry versioning | Yes | Yes |
| M-028 | Post-market surveillance plan | Documented | - |
| M-029 | Multi-site validation | Planned | - |
| M-030 | Multi-tag sequence identification | Yes | - |
| M-031 | Sequence ID confidence | Partial | - |
| M-032 | NIfTI explicit labeling | Yes | Yes |
| M-033 | Low confidence warning | Partial | - |

---

**Document Control:**
- Author: BrainMetScan Engineering Team
- Review Status: Draft
- Next Review: Upon initiation of regulatory submission process
