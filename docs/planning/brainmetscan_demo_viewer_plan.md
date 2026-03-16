# BrainMetScan Demo Viewer Plan

A proof of concept viewer designed to communicate clinical value to medical advisors and potential stakeholders in a digestible, ergonomic format.

---

## Core Viewing Experience

### Scan Display
- Clean 3D MRI viewer with segmentation overlay
- Side-by-side view: original scan alongside segmented version
- Intuitive slice navigation (scroll through axial/sagittal/coronal planes)
- Optional 3D rendering view of detected lesions in spatial context

### Lesion Visualization
- Clear color-coded overlay distinguishing lesions from healthy tissue
- Size-based color differentiation to highlight detection capability:
  - Standard lesions (typically caught by radiologists)
  - Small/tiny lesions (frequently missed — your key differentiator)
- Adjustable overlay opacity to compare against original scan

---

## Lesion Inventory Panel

### Individual Lesion Cards
- Thumbnail preview of the lesion
- Click-to-navigate: jump to relevant slice
- Location (anatomical region or coordinates)
- Dimensions and volume

### Aggregate Statistics
- Total lesion count
- Total tumor burden volume
- Size distribution breakdown (histogram or simple chart)
- Comparison indicator: "X lesions under 3mm detected"

---

## Clinical Report Section

### Generated Summary
- AI-generated clinical narrative summarizing findings
- Structured format familiar to clinicians (impression, findings, measurements)

### RAG Differentiation Display
- Visual indication of literature sources informing the analysis
- Example citations or references to recent publications
- Timestamp showing knowledge base recency ("Literature updated: [Month Year]")

### Customization Preview
- Brief demonstration of query customization capability
- Example: toggling between report formats (clinical summary vs. research detail)

---

## Performance Context

### Model Metrics
- Sensitivity/specificity summary
- Voxel-level performance stats
- Tiny lesion detection performance highlight (your competitive advantage)

### Validation Context
- Dataset information (size, source type)
- Brief comparison to baseline or published benchmarks
- Note on external validation status/plans

---

## UX Considerations

### For the Demo Video
- Keep interactions smooth and deliberate — no frantic clicking
- Pause on key moments (first lesion detection, report generation)
- Use a real or realistic case that demonstrates range of lesion sizes

### General Ergonomics
- Minimal chrome — let the scan and results be the focus
- Logical information hierarchy: scan first, details on demand
- Responsive feedback for all interactions (loading states, hover effects)

---

## Suggested Demo Flow

1. **Open case** — show patient metadata briefly
2. **Display original scan** — scroll through a few slices
3. **Activate segmentation overlay** — reveal detected lesions
4. **Highlight size differentiation** — point out tiny lesions
5. **Open lesion inventory** — click through a few examples
6. **Show aggregate stats** — total burden, count, distribution
7. **Display clinical report** — generated summary with literature grounding
8. **Close with metrics** — performance context and validation notes

---

## Future Enhancements (Post-POC)

- DICOM import workflow demonstration
- Multi-timepoint comparison view (treatment response tracking)
- Exportable reports in clinical formats
- Integration preview (API, PACS connectivity)
