"""
PDF Report Generator for X-Ray Analysis
Creates downloadable PDF reports with analysis results and heatmap visualizations
"""

import os
import numpy as np
from datetime import datetime
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from PIL import Image as PILImage
import pydicom


class PDFReportGenerator:
    """
    Generates professional PDF reports for X-ray analysis results
    """

    def __init__(self):
        """Initialize the PDF generator with default settings"""
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
        os.makedirs(self.output_dir, exist_ok=True)

        # Define custom styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1e3a5f')
        ))

        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#64748b')
        ))

        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#1e3a5f'),
            borderPadding=(0, 0, 5, 0)
        ))

        # Normal text style
        self.styles.add(ParagraphStyle(
            name='ReportText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor('#334155')
        ))

        # Status normal style
        self.styles.add(ParagraphStyle(
            name='StatusNormal',
            parent=self.styles['Normal'],
            fontSize=16,
            spaceBefore=10,
            spaceAfter=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#10b981'),
            fontName='Helvetica-Bold'
        ))

        # Status abnormal style
        self.styles.add(ParagraphStyle(
            name='StatusAbnormal',
            parent=self.styles['Normal'],
            fontSize=16,
            spaceBefore=10,
            spaceAfter=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#ef4444'),
            fontName='Helvetica-Bold'
        ))

        # Disclaimer style
        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontSize=8,
            spaceBefore=20,
            spaceAfter=10,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor('#94a3b8'),
            borderPadding=10
        ))

    def generate_report(self, analysis_results, original_image_path, heatmap_image_path=None,
                       heatmap_results=None, filename=None):
        """
        Generate a PDF report for the X-ray analysis

        Args:
            analysis_results: Dictionary containing pathology analysis results
            original_image_path: Path to the original X-ray image
            heatmap_image_path: Optional path to the GradCAM heatmap image
            heatmap_results: Optional dictionary containing heatmap analysis results
            filename: Optional custom filename for the PDF

        Returns:
            Path to the generated PDF file
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"xray_report_{timestamp}.pdf"

        pdf_path = os.path.join(self.output_dir, filename)

        # Create PDF document
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        # Build content
        story = []

        # Add header
        story.extend(self._create_header())

        # Add patient info section (placeholder)
        story.extend(self._create_info_section(analysis_results))

        # Add original X-ray image
        story.extend(self._create_image_section(original_image_path, "Original X-Ray Image"))

        # Add analysis status
        story.extend(self._create_status_section(analysis_results))

        # Add findings section
        story.extend(self._create_findings_section(analysis_results))

        # Add all pathologies table
        story.extend(self._create_pathologies_table(analysis_results))

        # Add heatmap section if available
        if heatmap_image_path and os.path.exists(heatmap_image_path):
            story.append(PageBreak())
            story.extend(self._create_heatmap_section(heatmap_image_path, heatmap_results))

        # Add recommendations
        story.extend(self._create_recommendations_section(analysis_results))

        # Add disclaimer
        story.extend(self._create_disclaimer_section(analysis_results))

        # Add footer
        story.extend(self._create_footer())

        # Build PDF
        doc.build(story)

        return pdf_path

    def _create_header(self):
        """Create report header"""
        elements = []

        # Title
        elements.append(Paragraph("X-Ray Analysis Report", self.styles['ReportTitle']))

        # Subtitle with date
        date_str = datetime.now().strftime('%B %d, %Y at %I:%M %p')
        elements.append(Paragraph(f"Generated on {date_str}", self.styles['ReportSubtitle']))

        # Horizontal line
        elements.append(Spacer(1, 10))

        return elements

    def _create_info_section(self, results):
        """Create analysis information section"""
        elements = []

        elements.append(Paragraph("Analysis Information", self.styles['SectionHeader']))

        # Create info table
        info_data = [
            ['Analysis Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Model Used:', 'DenseNet-121 (torchxrayvision)'],
            ['Pathologies Screened:', str(results.get('total_pathologies', 18))],
            ['Detection Threshold:', f"{results.get('threshold', 0.5) * 100:.0f}%"],
        ]

        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1e3a5f')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#334155')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))

        elements.append(info_table)
        elements.append(Spacer(1, 20))

        return elements

    def _load_image_for_pdf(self, image_path):
        """
        Load image from various formats (DICOM, PNG, JPEG) for PDF embedding

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (image_buffer, width, height) or (None, 0, 0) if failed
        """
        ext = os.path.splitext(image_path)[1].lower()

        try:
            if ext in ['.dcm', '.dicom']:
                # DICOM handling
                dcm = pydicom.dcmread(image_path)
                img_array = dcm.pixel_array.astype(np.float32)

                # Handle PhotometricInterpretation (invert if needed)
                if hasattr(dcm, 'PhotometricInterpretation'):
                    if dcm.PhotometricInterpretation == 'MONOCHROME1':
                        img_array = img_array.max() - img_array

                # Apply DICOM windowing if available
                if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
                    wc = dcm.WindowCenter
                    ww = dcm.WindowWidth
                    if isinstance(wc, pydicom.multival.MultiValue):
                        wc = wc[0]
                    if isinstance(ww, pydicom.multival.MultiValue):
                        ww = ww[0]

                    img_min = wc - ww / 2
                    img_max = wc + ww / 2
                    img_array = np.clip(img_array, img_min, img_max)
                    img_array = ((img_array - img_min) / (img_max - img_min)) * 255.0
                else:
                    img_min, img_max = img_array.min(), img_array.max()
                    if img_max > img_min:
                        img_array = ((img_array - img_min) / (img_max - img_min)) * 255.0

                # Convert to PIL Image
                img = PILImage.fromarray(img_array.astype(np.uint8))

                # Convert to RGB if grayscale (for PDF compatibility)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Save to buffer
                img_buffer = BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)

                return img_buffer, img.width, img.height
            else:
                # PNG/JPEG handling
                img = PILImage.open(image_path)
                return image_path, img.width, img.height

        except Exception as e:
            print(f"Error loading image for PDF: {e}")
            return None, 0, 0

    def _create_image_section(self, image_path, title):
        """Create image section with the X-ray"""
        elements = []

        elements.append(Paragraph(title, self.styles['SectionHeader']))

        if os.path.exists(image_path):
            # Load image (handles DICOM, PNG, JPEG)
            img_source, img_width, img_height = self._load_image_for_pdf(image_path)

            if img_source and img_width > 0 and img_height > 0:
                # Calculate scaling to fit within page
                max_width = 5 * inch
                max_height = 4 * inch

                scale = min(max_width / img_width, max_height / img_height)
                display_width = img_width * scale
                display_height = img_height * scale

                # Add image
                report_img = Image(img_source, width=display_width, height=display_height)
                elements.append(report_img)
            else:
                elements.append(Paragraph("Image could not be loaded", self.styles['ReportText']))
        else:
            elements.append(Paragraph("Image not available", self.styles['ReportText']))

        elements.append(Spacer(1, 20))

        return elements

    def _create_status_section(self, results):
        """Create analysis status section"""
        elements = []

        elements.append(Paragraph("Analysis Status", self.styles['SectionHeader']))

        status = results.get('status', 'UNKNOWN')
        status_message = results.get('status_message', '')

        if status == 'NORMAL':
            status_style = self.styles['StatusNormal']
            status_text = "✓ NORMAL - No Significant Abnormalities Detected"
        else:
            status_style = self.styles['StatusAbnormal']
            abnormal_count = results.get('abnormal_count', 0)
            status_text = f"⚠ ABNORMAL - {abnormal_count} Potential Finding(s) Detected"

        elements.append(Paragraph(status_text, status_style))
        elements.append(Paragraph(status_message, self.styles['ReportText']))
        elements.append(Spacer(1, 15))

        return elements

    def _create_findings_section(self, results):
        """Create key findings section"""
        elements = []

        top_findings = results.get('top_findings', [])

        if top_findings:
            elements.append(Paragraph("Key Findings Requiring Attention", self.styles['SectionHeader']))

            for finding in top_findings[:5]:
                name = finding.get('name', 'Unknown')
                percentage = finding.get('percentage', 0)
                severity = finding.get('severity', 'unknown')
                description = finding.get('description', '')

                # Severity color
                if severity == 'high':
                    severity_color = colors.HexColor('#ef4444')
                elif severity == 'moderate':
                    severity_color = colors.HexColor('#f59e0b')
                else:
                    severity_color = colors.HexColor('#eab308')

                # Create finding entry
                finding_data = [
                    [Paragraph(f"<b>{name}</b>", self.styles['ReportText']),
                     Paragraph(f"<b>{percentage}%</b>", self.styles['ReportText']),
                     Paragraph(f"<font color='#{severity_color.hexval()[2:]}'>{severity.upper()}</font>",
                              self.styles['ReportText'])]
                ]

                finding_table = Table(finding_data, colWidths=[3*inch, 1*inch, 1.5*inch])
                finding_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                    ('ALIGN', (1, 0), (1, 0), 'CENTER'),
                    ('ALIGN', (2, 0), (2, 0), 'RIGHT'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ]))

                elements.append(finding_table)

                # Description
                if description:
                    elements.append(Paragraph(description, self.styles['ReportText']))

                elements.append(Spacer(1, 10))

        return elements

    def _create_pathologies_table(self, results):
        """Create table of all screened pathologies"""
        elements = []

        pathologies = results.get('pathologies', [])

        if pathologies:
            elements.append(Paragraph("Complete Pathology Screening Results", self.styles['SectionHeader']))

            # Table header
            table_data = [['Pathology', 'Score', 'Status']]

            for path in pathologies:
                name = path.get('name', 'Unknown')
                percentage = path.get('percentage', 0)
                is_abnormal = path.get('is_abnormal', False)

                status = "ABNORMAL" if is_abnormal else "Normal"
                table_data.append([name, f"{percentage}%", status])

            # Create table
            path_table = Table(table_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
            path_table.setStyle(TableStyle([
                # Header style
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('TOPPADDING', (0, 0), (-1, 0), 10),

                # Body style
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('ALIGN', (1, 1), (1, -1), 'CENTER'),
                ('ALIGN', (2, 1), (2, -1), 'CENTER'),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                ('TOPPADDING', (0, 1), (-1, -1), 6),

                # Alternating row colors
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),

                # Grid
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ]))

            # Highlight abnormal rows
            for i, path in enumerate(pathologies, start=1):
                if path.get('is_abnormal', False):
                    path_table.setStyle(TableStyle([
                        ('TEXTCOLOR', (2, i), (2, i), colors.HexColor('#ef4444')),
                        ('FONTNAME', (2, i), (2, i), 'Helvetica-Bold'),
                    ]))

            elements.append(path_table)
            elements.append(Spacer(1, 20))

        return elements

    def _create_heatmap_section(self, heatmap_image_path, heatmap_results):
        """Create GradCAM heatmap section"""
        elements = []

        elements.append(Paragraph("Region Heatmap Analysis (GradCAM)", self.styles['SectionHeader']))

        elements.append(Paragraph(
            "The heatmap below highlights regions of the X-ray that most influenced the AI's pathology predictions. "
            "Warmer colors (red, orange, yellow) indicate areas of higher attention.",
            self.styles['ReportText']
        ))

        elements.append(Spacer(1, 10))

        # Add heatmap image
        elements.extend(self._create_image_section(heatmap_image_path, "GradCAM Heatmap Visualization"))

        # Add heatmap findings if available
        if heatmap_results:
            detections = heatmap_results.get('detections', [])
            if detections:
                elements.append(Paragraph("Highlighted Pathology Regions:", self.styles['ReportText']))

                for det in detections[:5]:
                    class_name = det.get('class_name', 'Unknown')
                    confidence = det.get('confidence', 0)
                    elements.append(Paragraph(
                        f"• <b>{class_name}</b>: {confidence}% confidence",
                        self.styles['ReportText']
                    ))

            summary = heatmap_results.get('summary', '')
            if summary:
                elements.append(Spacer(1, 10))
                elements.append(Paragraph(summary, self.styles['ReportText']))

        return elements

    def _create_recommendations_section(self, results):
        """Create recommendations section"""
        elements = []

        recommendations = results.get('recommendations', [])

        if recommendations:
            elements.append(Paragraph("Clinical Recommendations", self.styles['SectionHeader']))

            for rec in recommendations:
                elements.append(Paragraph(f"• {rec}", self.styles['ReportText']))

            elements.append(Spacer(1, 15))

        return elements

    def _create_disclaimer_section(self, results):
        """Create disclaimer section"""
        elements = []

        disclaimer = results.get('disclaimer', '')

        if disclaimer:
            elements.append(Spacer(1, 20))

            # Disclaimer box
            disclaimer_text = f"""
            <b>IMPORTANT MEDICAL DISCLAIMER</b><br/><br/>
            {disclaimer}<br/><br/>
            This report is generated by an AI system and is intended for informational purposes only.
            It should not be used as the sole basis for medical decisions. Always consult with a
            qualified healthcare professional for proper diagnosis and treatment planning.
            """

            elements.append(Paragraph(disclaimer_text, self.styles['Disclaimer']))

        return elements

    def _create_footer(self):
        """Create report footer"""
        elements = []

        elements.append(Spacer(1, 30))

        footer_text = f"""
        <para alignment="center">
        <font size="8" color="#94a3b8">
        Generated by X-Ray Analyzer | Powered by TorchXRayVision DenseNet-121 + GradCAM<br/>
        Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')} | © {datetime.now().year} X-Ray Analyzer
        </font>
        </para>
        """

        elements.append(Paragraph(footer_text, self.styles['Normal']))

        return elements


# Singleton instance
_pdf_generator = None

def get_pdf_generator():
    """Get or create the PDF generator instance"""
    global _pdf_generator
    if _pdf_generator is None:
        _pdf_generator = PDFReportGenerator()
    return _pdf_generator
