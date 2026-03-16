# BrainMetScan: From Research Prototype to Profitable Product

## Full Codebase Audit + Strategic Product Roadmap

*Generated February 14, 2026 — Based on analysis of github.com/ckirby04/BrainMetScan*

---

## PART 1: WHERE YOU ARE NOW (Honest Codebase Assessment)

### What You Have (Strengths)

**Segmentation Engine — Solid Foundation**
- `LightweightUNet3D` with attention gates, residual connections, deep supervision — these are real, publishable architectural choices, not toy code
- 4-model multi-scale ensemble (8/12/24/36 patch sizes) with union fusion — this is a sophisticated approach that addresses the tiny lesion problem elegantly
- 95.6% voxel sensitivity, 92.2% lesion detection at threshold 0.6 — these are strong numbers
- Tiny lesion Dice improved from 18.5% → 85-92% — this is the headline result that makes the project commercially interesting
- Custom loss functions (Tversky α=0.1/β=0.9, FocalDice, BoundaryLoss) — shows deep understanding of the imbalanced segmentation problem
- Test-time augmentation (`tta.py`) with adaptive strategies
- 566 labeled training cases with multi-modal input (T1-pre, T1-Gd, FLAIR, T2)

**RAG Pipeline — Early but Functional**
- ChromaDB vector database with case retrieval + medical knowledge retrieval
- Radiomic feature extraction (volume, shape, intensity, sphericity)
- Image embedding via ResNet18 backbone (slice-based 3D→2D projection)
- Report generation with template fallback + Claude/GPT API integration
- Medical literature corpus in `data/Context/` (brain anatomy, metastases, research lit)

**Documentation & Engineering Practices**
- FDA roadmap doc already exists (`ROADMAP_FDA_95_SENSITIVITY.md`) — shows regulatory awareness
- YAML-driven configs for reproducible experiments
- Leaderboard tracking, experiment state files, stacking results
- Colab notebooks for cloud training

### What's Missing (Critical Gaps for Product Viability)

**Gap 1: No Inference Pipeline / API**
- No REST API, no DICOM ingestion, no containerized service
- `inference.py` exists but is script-level, not service-level
- No PACS/DICOM integration — clinical systems can't talk to your model
- **Impact**: Cannot deploy to any real clinical environment

**Gap 2: No External Validation**
- All results are on your training/validation split (566 cases, single institution)
- No multi-site testing, no scanner diversity analysis
- FDA requires multi-site validation — this is a hard blocker
- **Impact**: No regulatory submission possible; no credibility with clinical buyers

**Gap 3: RAG is Prototype-Quality**
- ResNet18 2D embeddings for 3D volumes is a weak retrieval signal
- Knowledge base is static markdown files, not a curated medical ontology
- No RECIST/RANO criteria integration for treatment response
- Report generation prompts are generic, not clinically validated
- **Impact**: RAG adds demo value but isn't production-ready for clinical decisions

**Gap 4: No Quality Management System (QMS)**
- No version-controlled design history file
- No risk management documentation (ISO 14971)
- No software development lifecycle (IEC 62304) compliance
- No cybersecurity documentation
- **Impact**: Cannot begin FDA submission process

**Gap 5: No Business Infrastructure**
- No licensing model, no deployment architecture, no customer onboarding
- No HIPAA compliance, no SOC 2, no BAA templates
- No pricing strategy validated against buyer willingness-to-pay

---

## PART 2: THE BEST PRODUCT PATH (Where the Money Is)

Based on extensive market research across medical AI ($1.7B → $17B by 2034), clinical trial imaging ($2.4B → $5.6B by 2034), and the competitive landscape (Aidoc, Rad AI, Lunit, etc.), here is the recommended product strategy in priority order.

### RECOMMENDED: Dual-Track Strategy

**Track A (Revenue NOW, 3-6 months): Clinical Trial Imaging Analytics**
**Track B (Revenue LATER, 12-18 months): FDA-Cleared Clinical Decision Support**

The reason for this dual-track: Track A generates revenue and validates the technology while you build toward Track B. They share 80% of the same codebase.

---

## PART 3: BIG-PICTURE STEPS (The Roadmap)

### PHASE 1: PRODUCTIZE THE CORE (Months 1-3)

**Goal**: Turn research code into a deployable service that can process DICOM images and return structured results.

#### Step 1.1: Build the Inference API
What to build:
- FastAPI/Flask REST service wrapping the ensemble inference pipeline
- DICOM input handling (pydicom for parsing, proper orientation handling)
- NIfTI conversion pipeline (dcm2niix or equivalent)
- Structured JSON output (lesion count, volumes, locations, confidence scores)
- DICOM-SEG output (segmentation overlay that PACS can display)
- Docker container for reproducible deployment

Why it matters:
- Every buyer (CRO, hospital, teleradiology company) needs a service they can call, not a Python script
- DICOM-in/DICOM-out is the lingua franca of medical imaging
- Containerization is required for deployment to any clinical or cloud environment

What to build from your existing code:
```
src/
├── segmentation/
│   ├── unet.py              ← EXISTS (keep as-is)
│   ├── inference.py          ← EXISTS (refactor into service)
│   ├── tta.py                ← EXISTS (keep as-is)
│   └── ensemble.py           ← NEW (formalize the 4-model ensemble)
├── api/
│   ├── server.py             ← NEW (FastAPI service)
│   ├── dicom_handler.py      ← NEW (DICOM ingestion/conversion)
│   ├── schema.py             ← NEW (request/response models)
│   └── dicom_seg_writer.py   ← NEW (output DICOM-SEG)
├── rag/
│   ├── query.py              ← EXISTS (extend)
│   └── ...
└── Dockerfile                ← NEW
```

Effort: ~3-4 weeks for a solo developer

#### Step 1.2: Harden the RAG Pipeline
What to improve:
- Replace ResNet18 slice embedding with a 3D medical imaging encoder (e.g., Med3D, SwinUNETR pretrained features)
- Add RECIST 1.1 criteria to the knowledge base (tumor response categories: CR, PR, SD, PD)
- Add RANO-BM criteria for brain metastasis-specific response assessment
- Structure reports by clinical use case (initial detection vs. treatment monitoring vs. trial endpoint)
- Add source traceability — every claim in the report cites a specific retrieved document

Why it matters:
- CROs need RECIST-compliant reporting for trial endpoints
- Radiation oncologists need RANO-BM for treatment monitoring
- Source traceability is what makes RAG clinically trustworthy vs. a black-box LLM

Effort: ~2-3 weeks

#### Step 1.3: Internal Testing & Benchmarking
What to do:
- Comprehensive per-lesion sensitivity evaluation by size bucket (you have `evaluate_model.py` — extend it)
- Comparison against published benchmarks (BraTS-METS challenge results)
- Speed benchmarking (latency per case on GPU vs. CPU)
- Memory profiling for edge deployment feasibility
- Generate a formal "Technical Validation Report" documenting all results

Why it matters:
- This report IS your sales material for Track A
- CROs and pharma companies need to see quantitative evidence before licensing
- Establishes your technical credibility

Effort: ~2 weeks

---

### PHASE 2: FIRST REVENUE — CLINICAL TRIAL ANALYTICS (Months 3-6)

**Goal**: Sell imaging analytics services to CROs and pharma sponsors for oncology clinical trials.

#### Why This Is the Best First Market

1. **Your tech stack IS the product they need**: Oncology trials require automated tumor segmentation at every timepoint for every patient. Manual contouring is a massive bottleneck. Your U-Net does exactly this.

2. **Revenue per engagement is enormous**: A single Phase III oncology trial = 500 patients × 4 timepoints = 2,000 scans. At $50-200/scan, that's $100K-$400K per trial. There are thousands of active oncology trials globally.

3. **Lower regulatory bar**: Trial imaging analytics tools operate under the CRO's regulatory umbrella, NOT as standalone diagnostic devices. You likely don't need FDA clearance for this use case (though regulatory counsel should confirm).

4. **Validated market need**: ICON plc partnered with a Korean AI company (Nov 2024) specifically for automated tumor segmentation in trials. This proves the market is actively buying.

5. **RAG is a genuine differentiator here**: Nobody else combines segmentation + literature-grounded clinical context for trial reporting. Aidoc does triage. Rad AI does dictation. You do both.

#### Step 2.1: Build Trial-Specific Features
What to add:
- Longitudinal tracking (compare baseline vs. follow-up scans, calculate volume change %)
- RECIST-compliant measurements (longest diameter, perpendicular measurements)
- Automated response categorization (CR/PR/SD/PD based on volume change)
- Multi-timepoint visualization (waterfall plots, spider plots)
- Batch processing (ingest an entire trial cohort at once)
- Structured CSV/JSON export compatible with SDTM/ADaM trial data formats

What your RAG adds:
- "0.8cm cerebellar lesion shows 23% volume reduction at 3mo post-SRS, consistent with expected response for HER2+ breast cancer mets treated with SRS + trastuzumab per [Smith et al., JCO 2024]"
- This context is what turns a measurement into an insight

Effort: ~4-6 weeks

#### Step 2.2: Go-to-Market for CRO/Pharma
Target buyers:
- Mid-size CROs (Medpace, ICON, PPD) — more willing to work with startups than IQVIA
- Pharma biotech companies running their own trials (especially brain met-focused)
- Academic medical centers running investigator-initiated trials

How to reach them:
- Present at RSNA (Radiological Society of North America) — next meeting Nov 2026
- Present at SNMMI or ASCO with a poster showing your validation results
- Direct outreach to CRO imaging core lab directors (LinkedIn, cold email with technical validation report)
- Apply to RSNA Ventures (they just launched 2025, partnered with Rad AI — they're actively looking for imaging AI startups)

Pricing model:
- Per-scan analysis: $50-200 depending on complexity
- Per-trial license: $50K-200K flat fee for a trial cohort
- Annual platform license: $100K-500K for unlimited scans

Revenue target: 2-3 trial contracts in Year 1 = $200K-$800K

#### Step 2.3: "Research Use Only" Sales (Parallel)
While building trial features, sell RUO licenses to academic centers:
- Price: $5-20K/year per site
- No regulatory claims, no clinical use
- Builds relationships, generates external validation data, creates reference customers
- Target: neuro-oncology departments at major academic medical centers

---

### PHASE 3: EXTERNAL VALIDATION & REGULATORY FOUNDATION (Months 6-12)

**Goal**: Build the clinical evidence package needed for FDA submission while continuing to sell on Track A.

#### Step 3.1: Multi-Site Validation
What to do:
- Acquire 3+ external datasets from different institutions and scanner manufacturers
- Sources: BraTS-METS public dataset, Stanford AIMI, collaborating academic centers (offer RUO licenses in exchange for de-identified validation data)
- Run frozen ensemble on all external data — document performance by site, scanner, field strength
- Target: Maintain ≥90% lesion sensitivity across all external sites

Why it matters:
- FDA requires evidence of generalizability across sites and scanners
- External validation results dramatically strengthen your sales pitch for Track A too
- This is where most academic AI projects fail — proving it works beyond your own data

Effort: ~2-3 months (heavily dependent on data access timelines)

#### Step 3.2: Begin QMS Documentation
What to create:
- Design History File (DHF) — retrospective documentation of all design decisions
- Software Development Lifecycle (SDLC) per IEC 62304 — formalize your git workflow
- Risk Management File per ISO 14971 — hazard analysis, failure modes, mitigations
- Software Bill of Materials (SBOM) — your `requirements.txt` is a start, needs more detail
- Cybersecurity documentation — threat model for the API/deployment architecture

Approach for a solo developer:
- Use OpenRegulatory (free templates) or Qualio ($300/mo) for QMS
- Budget $20-50K for a regulatory consultant to review and guide
- This can be done in parallel with everything else — 2-4 hours/week of documentation

#### Step 3.3: Predicate Device Research
What to find:
- Search FDA 510(k) database for cleared brain lesion detection devices
- Potential predicates: Viz.ai (stroke), Aidoc (intracranial hemorrhage), icometrix (brain volumetrics)
- Document how BrainMetScan compares: same intended use population, similar technology, similar performance
- If no suitable predicate exists, prepare for De Novo pathway (more expensive, longer, but doable)

---

### PHASE 4: FDA CLEARANCE & CLINICAL PRODUCT (Months 12-18)

**Goal**: Submit FDA 510(k) and launch the clinical product (Track B).

#### Step 4.1: Clinical Study
Design: Multi-Reader Multi-Case (MRMC) study
- 3+ board-certified neuroradiologists
- 200+ cases (mix of internal + external)
- Each reader reviews cases with and without AI assistance
- Washout period between reads
- Primary endpoint: Per-lesion sensitivity improvement with AI
- Statistical requirement: 95% CI lower bound ≥90% sensitivity

Cost: $10-30K for radiologist readers + statistical analysis

#### Step 4.2: 510(k) Submission
Contents:
- Device description and intended use statement
- Substantial equivalence argument to predicate device
- Performance testing results (bench + clinical study)
- Software documentation per IEC 62304
- Labeling (Instructions for Use, warnings, contraindications)

Cost and timeline:
- FDA user fee: ~$15K (small business rate)
- Regulatory consultant: $20-50K
- Review timeline: 6-12 months after submission

#### Step 4.3: Post-Clearance Launch
Target market: Community hospitals lacking neuroradiology subspecialty coverage
Pricing: $50-100K/year per site (SaaS)
Infrastructure needed:
- HIPAA-compliant cloud deployment (AWS GovCloud or Azure Healthcare)
- DICOM integration with major PACS vendors
- SOC 2 Type II certification

---

## PART 4: SPECIFIC CODEBASE CHANGES NEEDED

### Priority 1 (Do Now)

| What | Where | Why |
|------|-------|-----|
| Formalize ensemble into single class | `src/segmentation/ensemble.py` (new) | Clean API for inference service |
| Build FastAPI inference server | `src/api/server.py` (new) | Deployable service |
| DICOM ingestion pipeline | `src/api/dicom_handler.py` (new) | Clinical data compatibility |
| Docker container | `Dockerfile` (new) | Reproducible deployment |
| Per-lesion sensitivity eval by size | Extend `scripts/evaluate_model.py` | Technical validation report |
| Longitudinal tracking | `src/segmentation/longitudinal.py` (new) | Trial analytics feature |

### Priority 2 (Month 2-3)

| What | Where | Why |
|------|-------|-----|
| RECIST measurement extraction | `src/rag/recist.py` (new) | Trial compliance |
| Upgrade RAG embeddings to 3D | Replace ResNet18 in `feature_extractor.py` | Better case retrieval |
| Structured clinical report templates | Extend `src/rag/query.py` | Different use cases need different reports |
| Batch processing for trial cohorts | `src/api/batch.py` (new) | Trial analytics workflow |
| DICOM-SEG output writer | `src/api/dicom_seg_writer.py` (new) | PACS integration |

### Priority 3 (Month 4-6)

| What | Where | Why |
|------|-------|-----|
| Add tests (unit + integration) | `tests/` (new directory) | QMS requirement, reliability |
| CI/CD pipeline | `.github/workflows/` | Automated testing, IEC 62304 compliance |
| Logging and audit trail | `src/api/logging.py` (new) | QMS requirement, traceability |
| Model versioning system | `src/segmentation/versioning.py` (new) | FDA total product lifecycle |
| HIPAA compliance layer | Encryption, access controls | Required for any clinical deployment |

---

## PART 5: BUDGET & TIMELINE SUMMARY

### Total Estimated Cost to First Revenue (Track A)

| Item | Cost | Timeline |
|------|------|----------|
| GPU compute (cloud training) | $500-2K | Ongoing |
| Regulatory consultant (initial guidance) | $5-10K | Month 3 |
| Conference attendance (RSNA/ASCO) | $3-5K | Month 6-8 |
| Legal (company formation, IP) | $3-5K | Month 1-2 |
| **Total to first CRO contract** | **$12-22K** | **Month 4-6** |

### Total Estimated Cost to FDA Clearance (Track B)

| Item | Cost | Timeline |
|------|------|----------|
| Everything from Track A | $12-22K | Months 1-6 |
| External dataset acquisition | $0-5K | Months 6-9 |
| QMS platform & documentation | $2-5K | Months 6-12 |
| Regulatory consultant (full 510k) | $20-50K | Months 9-15 |
| Clinical study (radiologist readers) | $10-30K | Months 12-15 |
| FDA user fee | ~$15K | Month 15 |
| **Total to FDA clearance** | **$60-130K** | **Month 15-18** |

### Revenue Projections

| Milestone | Timeline | Expected Revenue |
|-----------|----------|-----------------|
| First RUO license | Month 4 | $5-20K |
| First CRO trial contract | Month 6 | $50-200K |
| 3 trial contracts (Year 1) | Month 12 | $200-600K |
| FDA clearance + clinical launch | Month 18 | Opens $50-100K/yr per site market |
| Year 2 (mixed CRO + clinical) | Month 24 | $500K-1.5M |

---

## PART 6: YOUR COMPETITIVE MOAT

What makes BrainMetScan defensible:

1. **Segmentation + RAG is a unique combination** — Aidoc does triage only, Rad AI does report generation only, Microsoft Dragon Copilot does dictation only. Nobody ships "segment the lesion AND explain what it means in clinical context with literature citations." This is your differentiator.

2. **Multi-scale ensemble for small lesions** — Your tiny lesion Dice of 85-92% addresses the hardest unsolved problem in brain met detection. Sub-centimeter metastases are where current commercial tools fail. Published benchmarks drop below 0.85 Dice at this size range. You're competitive or better.

3. **Lightweight architecture = edge deployment** — Your model runs on consumer GPUs. Competitors like Aidoc require cloud infrastructure. Edge deployment is growing at 30.8% CAGR — fastest in the market. This opens portable/point-of-care use cases.

4. **Clinical trial switching costs** — Once your tool is embedded in a CRO's trial pipeline and data has been processed with your methodology, switching mid-trial is nearly impossible. This creates natural lock-in.

5. **Data flywheel** — Every trial you process generates validation data (with proper agreements). More data → better model → better results → more contracts.

---

## PART 7: IMMEDIATE NEXT ACTIONS (This Week)

1. **Create `src/segmentation/ensemble.py`** — Formalize the 4-model ensemble into a single class with a clean `predict()` method
2. **Create `Dockerfile`** — Containerize the inference pipeline
3. **Write a 2-page Technical Validation Summary** — Your README metrics reformatted as a professional document for sharing with potential buyers
4. **Research 3 target CROs** — Identify imaging core lab directors at Medpace, ICON, PPD
5. **Apply to RSNA Ventures** — They're actively investing in imaging AI startups
6. **Form an LLC/Corp** — Needed before any commercial activity (consult a lawyer, ~$500-1K)

---

*This roadmap assumes a solo developer. Timelines compress significantly with a co-founder or small team. The most critical hire would be someone with regulatory/clinical affairs experience — they pay for themselves 10x over in avoided mistakes during the FDA process.*
