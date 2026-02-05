"""
PDF Report Exporter for CODEFINDER
===================================
Exports Digital Type Case and Print Block HTML reports to high-quality PDF.
Uses WeasyPrint for HTML-to-PDF conversion with consistent styling.

Usage:
    from app.services.pdf_report_exporter import PDFReportExporter
    
    exporter = PDFReportExporter()
    pdf_path = exporter.export_type_case("reports/digital_type_case/index.html")
    print(f"PDF saved to: {pdf_path}")
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Check for WeasyPrint availability
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    logger.warning(
        "WeasyPrint not available. Install with: pip install weasyprint\n"
        "Note: WeasyPrint requires additional system dependencies. See:\n"
        "https://doc.courtbouillon.org/weasyprint/stable/first_steps.html"
    )


class PDFReportExporter:
    """
    Exports CODEFINDER HTML reports to high-quality PDF.
    
    Provides print-optimized CSS overrides for:
    - Digital Type Case reports
    - Print Block Analysis reports
    - Anomaly reports
    """
    
    # Print-optimized CSS for consistent character grids
    PRINT_CSS = """
        @page {
            size: A4 landscape;
            margin: 1.5cm;
        }
        
        @media print {
            body {
                background: white !important;
                color: #333 !important;
                font-size: 10pt;
            }
            
            h1 {
                background: none !important;
                -webkit-background-clip: initial !important;
                -webkit-text-fill-color: #e94560 !important;
                color: #e94560 !important;
                font-size: 18pt;
                page-break-after: avoid;
            }
            
            h2 {
                color: #333 !important;
                font-size: 14pt;
                page-break-after: avoid;
            }
            
            .char-block, .character-section {
                page-break-inside: avoid;
                background: #f8f9fa !important;
                border: 1px solid #ddd !important;
                margin-bottom: 1rem;
                padding: 1rem;
            }
            
            .cluster, .sample-images {
                display: flex !important;
                flex-wrap: wrap !important;
                gap: 4px !important;
                background: white !important;
            }
            
            .cluster img, .sample-images img {
                width: 36px !important;
                height: 48px !important;
                object-fit: contain !important;
                border: 1px solid #ccc !important;
            }
            
            /* Hide interactive elements */
            .cluster img:hover, .sample-images img:hover {
                transform: none !important;
                box-shadow: none !important;
            }
            
            .stat-card {
                background: #f8f9fa !important;
                border: 1px solid #ddd !important;
            }
            
            .stat-card .value {
                color: #e94560 !important;
            }
        }
    """
    
    def __init__(self):
        if not WEASYPRINT_AVAILABLE:
            raise ImportError(
                "WeasyPrint is required for PDF export. "
                "Install with: pip install weasyprint"
            )
    
    def export_type_case(
        self, 
        html_path: str, 
        output_path: str = None,
        title: str = "Digital Type Case"
    ) -> str:
        """
        Convert Digital Type Case HTML to PDF.
        
        Args:
            html_path: Path to index.html or print_block_report.html
            output_path: Optional output PDF path (auto-generated if None)
            title: Title for the PDF metadata
            
        Returns:
            Path to the generated PDF file
        """
        html_path = Path(html_path)
        
        if not html_path.exists():
            raise FileNotFoundError(f"HTML report not found: {html_path}")
        
        # Generate output path if not specified
        if output_path is None:
            stem = html_path.stem
            output_path = html_path.parent / f"{stem}.pdf"
        else:
            output_path = Path(output_path)
        
        logger.info(f"Converting {html_path} to PDF...")
        
        # Create print CSS
        print_css = CSS(string=self.PRINT_CSS)
        
        # Convert to PDF
        HTML(filename=str(html_path)).write_pdf(
            str(output_path),
            stylesheets=[print_css]
        )
        
        logger.info(f"PDF saved to: {output_path}")
        return str(output_path)
    
    def export_all_reports(self, reports_dir: str = "reports") -> list:
        """
        Export all HTML reports in the reports directory to PDF.
        
        Args:
            reports_dir: Base reports directory
            
        Returns:
            List of generated PDF paths
        """
        reports_dir = Path(reports_dir)
        generated = []
        
        # Find all HTML reports
        html_files = list(reports_dir.rglob("*.html"))
        
        for html_file in html_files:
            try:
                pdf_path = self.export_type_case(str(html_file))
                generated.append(pdf_path)
            except Exception as e:
                logger.error(f"Failed to export {html_file}: {e}")
        
        return generated


def main():
    """CLI interface for PDF export."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export CODEFINDER HTML reports to PDF"
    )
    parser.add_argument(
        "html_file",
        nargs="?",
        default="reports/digital_type_case/index.html",
        help="Path to HTML report (default: reports/digital_type_case/index.html)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output PDF path (default: same directory as HTML)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all HTML reports in reports/ directory"
    )
    
    args = parser.parse_args()
    
    if not WEASYPRINT_AVAILABLE:
        print("❌ WeasyPrint not installed.")
        print("   Install with: pip install weasyprint")
        print("   See: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html")
        return 1
    
    exporter = PDFReportExporter()
    
    if args.all:
        print("📄 Exporting all reports to PDF...")
        pdfs = exporter.export_all_reports()
        print(f"✅ Generated {len(pdfs)} PDFs")
        for pdf in pdfs:
            print(f"   - {pdf}")
    else:
        print(f"📄 Exporting {args.html_file} to PDF...")
        pdf_path = exporter.export_type_case(args.html_file, args.output)
        print(f"✅ PDF saved to: {pdf_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
