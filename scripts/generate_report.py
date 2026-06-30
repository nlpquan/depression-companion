"""Generate clinical PDF reports from analysis results.

Uses ReportLab for PDF generation (free, no watermark).
Install: pip install reportlab
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


def generate_clinical_report(
    analysis_data: dict,
    output_path: Optional[Path] = None,
) -> bytes:
    """Generate a clinical PDF report from analysis results.
    
    Args:
        analysis_data: Dictionary with analysis results.
        output_path: Optional path to save PDF file.
        
    Returns:
        PDF as bytes.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch, mm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
        from reportlab.graphics.shapes import Drawing, Rect, String
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=20*mm,
        )
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1a365d'),
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#2d3748'),
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            leading=16,
        )
        
        # Build story
        story = []
        
        # Title
        story.append(Paragraph("Depression Companion", title_style))
        story.append(Paragraph("Clinical Analysis Report", heading_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            styles['Normal'],
        ))
        story.append(Spacer(1, 20))
        
        # Patient info (anonymized)
        story.append(Paragraph("Patient Information", heading_style))
        info_data = [
            ["Report ID:", analysis_data.get("id", "N/A")],
            ["Date:", analysis_data.get("timestamp", datetime.now().isoformat())],
            ["Analysis Type:", analysis_data.get("analysis_type", "Multimodal")],
        ]
        
        info_table = Table(info_data, colWidths=[100, 300])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#4a5568')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 20))
        
        # Scores section
        story.append(Paragraph("Clinical Scores", heading_style))
        
        scores = analysis_data.get("scores", {})
        
        # Score interpretation
        depression_score = scores.get("depression_score", 0)
        anxiety_score = scores.get("anxiety_score", 0)
        
        # Severity interpretation
        if depression_score < 0.3:
            dep_severity = "Minimal"
            dep_color = colors.HexColor('#38a169')  # Green
        elif depression_score < 0.5:
            dep_severity = "Mild"
            dep_color = colors.HexColor('#d69e2e')  # Yellow
        elif depression_score < 0.7:
            dep_severity = "Moderate"
            dep_color = colors.HexColor('#dd6b20')  # Orange
        else:
            dep_severity = "Severe"
            dep_color = colors.HexColor('#e53e3e')  # Red
        
        score_data = [
            ["Metric", "Score", "Severity", "Reference Range"],
            [
                "Depression",
                f"{depression_score:.2f}",
                dep_severity,
                "< 0.3 (Normal)",
            ],
            [
                "Anxiety",
                f"{anxiety_score:.2f}",
                "Mild" if anxiety_score < 0.5 else "Moderate",
                "< 0.3 (Normal)",
            ],
            [
                "Mood",
                f"{scores.get('mood_score', 0):.2f}",
                "Normal" if scores.get('mood_score', 0) > 0.5 else "Low",
                "> 0.5 (Normal)",
            ],
        ]
        
        score_table = Table(score_data, colWidths=[120, 80, 100, 120])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d3748')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 20))
        
        # Risk Assessment
        story.append(Paragraph("Risk Assessment", heading_style))
        
        risk_level = analysis_data.get("risk_level", "low")
        
        risk_text = {
            "low": "No immediate concerns detected. Continue monitoring.",
            "medium": "Some warning signals present. Recommend follow-up within 1 week.",
            "high": "Multiple warning signals detected. Recommend immediate clinical follow-up.",
        }.get(risk_level, "Unknown")
        
        story.append(Paragraph(f"<b>Risk Level: {risk_level.upper()}</b>", body_style))
        story.append(Paragraph(risk_text, body_style))
        
        # Warning signals
        warnings = analysis_data.get("warnings", [])
        if warnings:
            story.append(Paragraph("Active Warning Signals:", body_style))
            for warning in warnings:
                story.append(Paragraph(f"• {warning}", body_style))
        
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Recommendations", heading_style))
        
        recommendations = analysis_data.get("recommendations", [
            "Continue daily mood tracking",
            "Practice sleep hygiene (consistent schedule)",
            "Engage in behavioral activation (schedule pleasant activities)",
            "Consider reaching out to a mental health professional",
        ])
        
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", body_style))
        
        story.append(Spacer(1, 30))
        
        # Disclaimer
        story.append(Paragraph(
            "<i>Disclaimer: This report is generated by an AI system and is not a clinical diagnosis. "
            "It should not replace professional medical advice. If you are experiencing a crisis, "
            "please contact emergency services or a crisis helpline immediately.</i>",
            ParagraphStyle(
                'Disclaimer',
                parent=styles['Normal'],
                fontSize=8,
                textColor=colors.HexColor('#a0aec0'),
                leading=12,
            ),
        ))
        
        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
            logger.info(f"Report saved to: {output_path}")
        
        return pdf_bytes
        
    except ImportError:
        logger.warning("reportlab not installed. Install: pip install reportlab")
        return _generate_text_report(analysis_data, output_path)


def _generate_text_report(
    analysis_data: dict,
    output_path: Optional[Path] = None,
) -> bytes:
    """Fallback: Generate a text-based report."""
    
    lines = []
    lines.append("=" * 50)
    lines.append("DEPRESSION COMPANION - CLINICAL REPORT")
    lines.append("=" * 50)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Report ID: {analysis_data.get('id', 'N/A')}")
    lines.append("")
    
    scores = analysis_data.get("scores", {})
    lines.append("CLINICAL SCORES:")
    lines.append(f"  Depression: {scores.get('depression_score', 0):.3f}")
    lines.append(f"  Anxiety:    {scores.get('anxiety_score', 0):.3f}")
    lines.append(f"  Mood:       {scores.get('mood_score', 0):.3f}")
    lines.append(f"  Confidence: {scores.get('confidence', 0):.3f}")
    lines.append("")
    
    lines.append(f"RISK LEVEL: {analysis_data.get('risk_level', 'low').upper()}")
    lines.append("")
    
    warnings = analysis_data.get("warnings", [])
    if warnings:
        lines.append("ACTIVE WARNINGS:")
        for w in warnings:
            lines.append(f"  • {w}")
    else:
        lines.append("No active warnings.")
    
    lines.append("")
    lines.append("DISCLAIMER: This is an AI-generated report, not clinical diagnosis.")
    
    report_text = "\n".join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as .txt if reportlab not available
        txt_path = output_path.with_suffix(".txt")
        with open(txt_path, "w") as f:
            f.write(report_text)
        logger.info(f"Text report saved to: {txt_path}")
    
    return report_text.encode("utf-8")


def generate_report_from_api(
    analysis_id: str,
    scores: dict,
    risk_level: str = "low",
    warnings: list[str] = [],
    output_dir: str = "reports",
) -> Path:
    """Generate a report using data from API response.
    
    Args:
        analysis_id: Analysis identifier.
        scores: Depression scores dictionary.
        risk_level: Risk level string.
        warnings: List of warning strings.
        output_dir: Output directory.
        
    Returns:
        Path to generated report.
    """
    analysis_data = {
        "id": analysis_id,
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "Multimodal",
        "scores": scores,
        "risk_level": risk_level,
        "warnings": warnings,
    }
    
    output_path = Path(output_dir) / f"report_{analysis_id}.pdf"
    generate_clinical_report(analysis_data, output_path)
    
    return output_path


if __name__ == "__main__":
    # Test report generation
    test_data = {
        "id": "test_001",
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "Multimodal",
        "scores": {
            "depression_score": 0.45,
            "anxiety_score": 0.38,
            "mood_score": 0.55,
            "confidence": 0.82,
        },
        "risk_level": "medium",
        "warnings": [
            "Sleep disruption detected (2-day lead time)",
            "Reduced speech energy (1.5-day lead time)",
        ],
    }
    
    pdf_bytes = generate_clinical_report(test_data, Path("reports/test_report.pdf"))
    print(f"Generated report: {len(pdf_bytes)} bytes")