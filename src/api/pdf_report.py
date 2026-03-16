"""
PDF clinical report generator for BrainMetScan.
Produces branded, professional PDF reports with lesion tables,
RECIST summary, and slice visualizations.

Uses reportlab for PDF generation (lightweight, no external dependencies).
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class PDFReportGenerator:
    """Generates clinical PDF reports from segmentation results."""

    def __init__(
        self,
        product_name: str = "BrainMetScan",
        product_version: str = "1.23.0",
        organization: str = "BrainMetScan AI",
    ):
        self.product_name = product_name
        self.product_version = product_version
        self.organization = organization

    def generate(
        self,
        result: Dict,
        case_id: str,
        output_path: Optional[str] = None,
        comparison: Optional[Dict] = None,
        rag_report: Optional[str] = None,
        slice_images: Optional[List[bytes]] = None,
    ) -> bytes:
        """
        Generate a clinical PDF report.

        Args:
            result: Segmentation result dict (lesion_count, lesion_details, etc.)
            case_id: Patient/case identifier
            output_path: Optional file path to save the PDF
            comparison: Optional longitudinal comparison result
            rag_report: Optional RAG-generated clinical narrative
            slice_images: Optional list of PNG bytes for slice visualizations

        Returns:
            PDF content as bytes
        """
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=1.0 * inch,
            bottomMargin=0.75 * inch,
        )

        styles = getSampleStyleSheet()
        elements = []

        # --- Custom Styles ---
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=22,
            textColor=colors.HexColor("#0C4DA2"),
            spaceAfter=6,
        )
        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontSize=11,
            textColor=colors.HexColor("#878681"),
            spaceAfter=20,
        )
        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=14,
            textColor=colors.HexColor("#0C4DA2"),
            spaceBefore=16,
            spaceAfter=8,
        )
        body_style = ParagraphStyle(
            "BodyText",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            spaceAfter=6,
        )
        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#999999"),
            spaceBefore=20,
            spaceAfter=6,
        )

        # --- Header ---
        elements.append(Paragraph(f"{self.product_name} Segmentation Report", title_style))
        elements.append(
            Paragraph(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                f"Version: {self.product_version} | "
                f"Case: {case_id}",
                subtitle_style,
            )
        )

        # --- Summary Section ---
        elements.append(Paragraph("Summary", heading_style))

        lesion_count = result.get("lesion_count", 0)
        lesions = result.get("lesion_details", [])
        total_vol = sum(l.get("volume_mm3", 0) for l in lesions)

        summary_data = [
            ["Metric", "Value"],
            ["Total Lesions Detected", str(lesion_count)],
            ["Total Tumor Volume", f"{total_vol:.1f} mm\u00b3"],
            ["Processing Time", f"{result.get('processing_time_seconds', 0):.1f}s"],
        ]

        summary_table = Table(summary_data, colWidths=[3 * inch, 3.5 * inch])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0C4DA2")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F8F8")]),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 12))

        # --- Lesion Details Table ---
        if lesions:
            elements.append(Paragraph("Lesion Details", heading_style))

            lesion_header = ["ID", "Volume (mm\u00b3)", "Max Diameter (mm)", "Confidence", "Centroid"]
            lesion_rows = [lesion_header]

            for l in lesions:
                centroid = l.get("centroid", [0, 0, 0])
                centroid_str = f"({centroid[0]:.0f}, {centroid[1]:.0f}, {centroid[2]:.0f})"
                lesion_rows.append([
                    str(l.get("id", "")),
                    f"{l.get('volume_mm3', 0):.1f}",
                    f"{l.get('max_diameter_mm', 0):.1f}",
                    f"{l.get('confidence', 0):.2f}",
                    centroid_str,
                ])

            lesion_table = Table(lesion_rows, colWidths=[0.6 * inch, 1.3 * inch, 1.5 * inch, 1.0 * inch, 2.1 * inch])
            lesion_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0C4DA2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F8F8")]),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            elements.append(lesion_table)
            elements.append(Spacer(1, 12))

        # --- Longitudinal Comparison ---
        if comparison:
            elements.append(Paragraph("Longitudinal Comparison (RECIST 1.1)", heading_style))

            response = comparison.get("response_category", "N/A")
            response_colors = {
                "CR": colors.HexColor("#28a745"),
                "PR": colors.HexColor("#17a2b8"),
                "SD": colors.HexColor("#ffc107"),
                "PD": colors.HexColor("#dc3545"),
            }

            comp_data = [
                ["Metric", "Value"],
                ["Response Category", response],
                ["Baseline SoD", f"{comparison.get('sum_of_diameters_baseline_mm', 0):.1f} mm"],
                ["Follow-up SoD", f"{comparison.get('sum_of_diameters_followup_mm', 0):.1f} mm"],
                ["Matched Lesions", str(len(comparison.get("matched_lesions", [])))],
                ["New Lesions", str(comparison.get("new_lesions", 0))],
                ["Resolved Lesions", str(comparison.get("resolved_lesions", 0))],
            ]

            comp_table = Table(comp_data, colWidths=[3 * inch, 3.5 * inch])
            comp_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0C4DA2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F8F8")]),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]))
            elements.append(comp_table)
            elements.append(Spacer(1, 12))

        # --- Slice Visualizations ---
        if slice_images:
            elements.append(Paragraph("Segmentation Visualizations", heading_style))
            for i, img_bytes in enumerate(slice_images[:6]):  # Max 6 images
                img_io = io.BytesIO(img_bytes)
                img = Image(img_io, width=6 * inch, height=3 * inch)
                elements.append(img)
                elements.append(Spacer(1, 8))

        # --- RAG Clinical Narrative ---
        if rag_report:
            elements.append(Paragraph("AI-Generated Clinical Summary", heading_style))
            for para in rag_report.split("\n\n"):
                if para.strip():
                    elements.append(Paragraph(para.strip(), body_style))
            elements.append(Spacer(1, 12))

        # --- Disclaimer ---
        elements.append(Paragraph(
            f"<b>DISCLAIMER:</b> This report is generated by {self.product_name} v{self.product_version}, "
            "an AI-assisted tool for research use only. This is NOT a medical device and has NOT been "
            "cleared or approved by the FDA or any regulatory body. All findings must be reviewed and "
            "confirmed by a qualified radiologist or physician. Do not use for clinical decision-making "
            "without independent verification.",
            disclaimer_style,
        ))
        elements.append(Paragraph(
            f"\u00a9 {datetime.now().year} {self.organization}. Confidential.",
            disclaimer_style,
        ))

        # Build PDF
        doc.build(elements)
        pdf_bytes = buffer.getvalue()

        if output_path:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(pdf_bytes)

        return pdf_bytes


def generate_slice_images(
    volume: np.ndarray,
    mask: np.ndarray,
    n_slices: int = 3,
    channel: int = 1,
) -> List[bytes]:
    """
    Generate PNG images of axial slices with segmentation overlay.

    Args:
        volume: Input volume (C, H, W, D) or (H, W, D)
        mask: Binary mask (H, W, D)
        n_slices: Number of representative slices
        channel: Channel index to display (1=T1-Gd typically)

    Returns:
        List of PNG bytes
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Get the display volume
    if volume.ndim == 4:
        display_vol = volume[min(channel, volume.shape[0] - 1)]
    else:
        display_vol = volume

    # Find slices with largest lesion area
    slice_areas = np.sum(mask > 0, axis=(0, 1))  # Sum over H, W for each D
    if slice_areas.max() == 0:
        # No lesions — show middle slices
        mid = display_vol.shape[2] // 2
        indices = [max(0, mid - 10), mid, min(display_vol.shape[2] - 1, mid + 10)]
    else:
        top_indices = np.argsort(slice_areas)[-n_slices:]
        indices = sorted(top_indices)

    images = []
    for idx in indices:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=150)
        ax.imshow(display_vol[:, :, idx].T, cmap="gray", origin="lower")

        # Overlay mask
        mask_slice = mask[:, :, idx].T
        if mask_slice.max() > 0:
            overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
            ax.imshow(overlay, cmap="autumn", alpha=0.5, origin="lower")

        ax.set_title(f"Axial Slice {idx}", fontsize=10)
        ax.axis("off")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="black")
        plt.close(fig)
        buf.seek(0)
        images.append(buf.getvalue())

    return images
