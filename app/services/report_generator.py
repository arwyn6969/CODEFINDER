"""
Comprehensive Reporting System for Ancient Text Analysis
Generates detailed analysis reports with multiple export formats (PDF, CSV, JSON, HTML)
"""
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import csv
import io
import logging
from datetime import datetime
from pathlib import Path
import statistics
from jinja2 import Template, Environment, FileSystemLoader
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from app.core.database import get_db
from app.models.database_models import Document, Pattern, Page
from app.services.pattern_significance_ranker import PatternSignificanceRanker
from app.services.geometric_visualizer import GeometricVisualizer
from app.services.anomaly_detector import AnomalyDetector


class ReportFormat(Enum):
    """Supported report export formats"""
    PDF = "pdf"
    CSV = "csv"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    EXCEL = "xlsx"


class ReportType(Enum):
    """Types of analysis reports"""
    COMPREHENSIVE = "comprehensive"
    PATTERN_ANALYSIS = "pattern_analysis"
    GEOMETRIC_ANALYSIS = "geometric_analysis"
    CIPHER_ANALYSIS = "cipher_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    CROSS_DOCUMENT = "cross_document"
    SUMMARY = "summary"
    STATISTICAL = "statistical"


class ReportSection(Enum):
    """Report sections that can be included"""
    EXECUTIVE_SUMMARY = "executive_summary"
    DOCUMENT_OVERVIEW = "document_overview"
    PATTERN_FINDINGS = "pattern_findings"
    GEOMETRIC_ANALYSIS = "geometric_analysis"
    CIPHER_DETECTION = "cipher_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    VISUALIZATIONS = "visualizations"
    RECOMMENDATIONS = "recommendations"
    APPENDICES = "appendices"
    METHODOLOGY = "methodology"
    REFERENCES = "references"


@dataclass
class ReportConfiguration:
    """Configuration for report generation"""
    report_type: ReportType
    format: ReportFormat
    sections: List[ReportSection]
    include_visualizations: bool = True
    include_raw_data: bool = False
    include_statistical_details: bool = True
    confidence_threshold: float = 0.7
    max_patterns_per_section: int = 50
    template_name: Optional[str] = None
    custom_styling: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportData:
    """Structured data for report generation"""
    document_info: Dict[str, Any]
    analysis_summary: Dict[str, Any]
    pattern_findings: List[Dict[str, Any]]
    geometric_analysis: Dict[str, Any]
    cipher_findings: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    statistical_data: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    cross_references: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any]


@dataclass
class GeneratedReport:
    """Container for generated report"""
    report_id: str
    document_id: int
    report_type: ReportType
    format: ReportFormat
    content: Union[str, bytes]
    file_path: Optional[str]
    generation_time: datetime
    configuration: ReportConfiguration
    summary: Dict[str, Any]
    size_bytes: int


class ReportGenerator:
    """
    Service for generating comprehensive analysis reports
    """
    
    def __init__(self, db_session: Session = None):
        self.db = db_session or next(get_db())
        self.logger = logging.getLogger(__name__)
        
        # Initialize analysis services
        self.significance_ranker = PatternSignificanceRanker(self.db)
        self.geometric_visualizer = GeometricVisualizer(self.db)
        self.anomaly_detector = AnomalyDetector(self.db)
        
        # Report templates directory
        self.templates_dir = Path("app/templates/reports")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True
        )
        
        # Default report configurations
        self.default_configs = {
            ReportType.COMPREHENSIVE: ReportConfiguration(
                report_type=ReportType.COMPREHENSIVE,
                format=ReportFormat.HTML,
                sections=[
                    ReportSection.EXECUTIVE_SUMMARY,
                    ReportSection.DOCUMENT_OVERVIEW,
                    ReportSection.PATTERN_FINDINGS,
                    ReportSection.GEOMETRIC_ANALYSIS,
                    ReportSection.CIPHER_DETECTION,
                    ReportSection.ANOMALY_DETECTION,
                    ReportSection.STATISTICAL_ANALYSIS,
                    ReportSection.VISUALIZATIONS,
                    ReportSection.RECOMMENDATIONS,
                    ReportSection.METHODOLOGY
                ]
            ),
            ReportType.SUMMARY: ReportConfiguration(
                report_type=ReportType.SUMMARY,
                format=ReportFormat.HTML,
                sections=[
                    ReportSection.EXECUTIVE_SUMMARY,
                    ReportSection.DOCUMENT_OVERVIEW,
                    ReportSection.STATISTICAL_ANALYSIS
                ],
                include_visualizations=False,
                max_patterns_per_section=10
            )
        }
    
    def generate_report(self, document_id: int, 
                       config: Optional[ReportConfiguration] = None) -> GeneratedReport:
        """
        Generate a comprehensive analysis report for a document
        """
        try:
            # Use default config if none provided
            if config is None:
                config = self.default_configs[ReportType.COMPREHENSIVE]
            
            # Collect report data
            report_data = self._collect_report_data(document_id, config)
            
            # Generate report content based on format
            if config.format == ReportFormat.HTML:
                content = self._generate_html_report(report_data, config)
            elif config.format == ReportFormat.PDF:
                content = self._generate_pdf_report(report_data, config)
            elif config.format == ReportFormat.JSON:
                content = self._generate_json_report(report_data, config)
            elif config.format == ReportFormat.CSV:
                content = self._generate_csv_report(report_data, config)
            elif config.format == ReportFormat.MARKDOWN:
                content = self._generate_markdown_report(report_data, config)
            elif config.format == ReportFormat.EXCEL:
                content = self._generate_excel_report(report_data, config)
            else:
                raise ValueError(f"Unsupported report format: {config.format}")
            
            # Create report object
            report = GeneratedReport(
                report_id=f"report_{document_id}_{datetime.now().timestamp()}",
                document_id=document_id,
                report_type=config.report_type,
                format=config.format,
                content=content,
                file_path=None,
                generation_time=datetime.now(),
                configuration=config,
                summary=self._generate_report_summary(report_data),
                size_bytes=len(content) if isinstance(content, (str, bytes)) else 0
            )
            
            self.logger.info(f"Generated {config.report_type.value} report for document {document_id}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise
    
    def generate_multi_document_report(self, document_ids: List[int],
                                     config: Optional[ReportConfiguration] = None) -> GeneratedReport:
        """
        Generate a cross-document analysis report
        """
        try:
            if config is None:
                config = self.default_configs[ReportType.COMPREHENSIVE]
                config.report_type = ReportType.CROSS_DOCUMENT
            
            # Collect data from all documents
            all_report_data = []
            for doc_id in document_ids:
                doc_data = self._collect_report_data(doc_id, config)
                all_report_data.append(doc_data)
            
            # Merge and analyze cross-document patterns
            merged_data = self._merge_multi_document_data(all_report_data)
            
            # Generate report content
            if config.format == ReportFormat.HTML:
                content = self._generate_html_report(merged_data, config)
            elif config.format == ReportFormat.JSON:
                content = self._generate_json_report(merged_data, config)
            else:
                content = self._generate_markdown_report(merged_data, config)
            
            report = GeneratedReport(
                report_id=f"multi_report_{datetime.now().timestamp()}",
                document_id=0,  # Multi-document report
                report_type=ReportType.CROSS_DOCUMENT,
                format=config.format,
                content=content,
                file_path=None,
                generation_time=datetime.now(),
                configuration=config,
                summary=self._generate_report_summary(merged_data),
                size_bytes=len(content) if isinstance(content, (str, bytes)) else 0
            )
            
            self.logger.info(f"Generated cross-document report for {len(document_ids)} documents")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating multi-document report: {str(e)}")
            raise
    
    def save_report(self, report: GeneratedReport, output_path: str) -> str:
        """
        Save generated report to file
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(report.content, str):
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report.content)
            else:
                with open(output_file, 'wb') as f:
                    f.write(report.content)
            
            report.file_path = str(output_file)
            self.logger.info(f"Saved report to {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Error saving report: {str(e)}")
            raise
    
    def create_custom_template(self, template_name: str, template_content: str) -> str:
        """
        Create a custom report template
        """
        try:
            template_path = self.templates_dir / f"{template_name}.html"
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
            
            self.logger.info(f"Created custom template: {template_name}")
            return str(template_path)
            
        except Exception as e:
            self.logger.error(f"Error creating custom template: {str(e)}")
            raise
    
    def get_available_templates(self) -> List[str]:
        """
        Get list of available report templates
        """
        try:
            templates = []
            if self.templates_dir.exists():
                for template_file in self.templates_dir.glob("*.html"):
                    templates.append(template_file.stem)
            return templates
            
        except Exception as e:
            self.logger.error(f"Error listing templates: {str(e)}")
            return []
    
    # Private helper methods
    
    def _collect_report_data(self, document_id: int, config: ReportConfiguration) -> ReportData:
        """
        Collect all necessary data for report generation
        """
        try:
            # Get document information
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            document_info = {
                'id': document.id,
                'filename': document.filename,
                'file_size': document.file_size,
                'upload_date': document.upload_date,
                'total_pages': document.total_pages,
                'processing_status': document.processing_status
            }
            
            # Get patterns
            patterns = self.db.query(Pattern).filter(Pattern.document_id == document_id).all()
            
            # Rank patterns by significance
            ranked_patterns = self.significance_ranker.rank_patterns(document_id)
            
            # Get geometric analysis
            geometric_analysis = self._collect_geometric_analysis(document_id)
            
            # Get anomalies
            anomalies = self.anomaly_detector.detect_anomalies(document_id)
            
            # Get statistical data
            statistical_data = self._collect_statistical_data(document_id, patterns)
            
            # Generate visualizations if requested
            visualizations = []
            if config.include_visualizations:
                visualizations = self._generate_visualizations(document_id)
            
            # Create analysis summary
            analysis_summary = {
                'total_patterns': len(patterns),
                'significant_patterns': len([p for p in ranked_patterns if p.significance_score > config.confidence_threshold]),
                'anomalies_found': len(anomalies),
                'processing_date': datetime.now(),
                'confidence_threshold': config.confidence_threshold
            }
            
            # Format pattern findings
            pattern_findings = []
            for pattern_data in ranked_patterns[:config.max_patterns_per_section]:
                pattern_findings.append({
                    'id': pattern_data.pattern.id,
                    'type': pattern_data.pattern.pattern_type,
                    'description': pattern_data.pattern.description,
                    'confidence': pattern_data.pattern.confidence,
                    'significance_score': pattern_data.significance_score,
                    'page_number': pattern_data.pattern.page_number,
                    'coordinates': pattern_data.pattern.coordinates,
                    'supporting_evidence': pattern_data.supporting_evidence
                })
            
            # Format cipher findings
            cipher_findings = []
            cipher_patterns = [p for p in patterns if 'cipher' in p.pattern_type.lower()]
            for pattern in cipher_patterns:
                cipher_findings.append({
                    'type': pattern.pattern_type,
                    'description': pattern.description,
                    'confidence': pattern.confidence,
                    'page_number': pattern.page_number,
                    'method': pattern.metadata.get('cipher_method', 'Unknown') if pattern.metadata else 'Unknown'
                })
            
            # Format anomalies
            anomaly_list = []
            for anomaly in anomalies:
                anomaly_list.append({
                    'type': anomaly.anomaly_type,
                    'description': anomaly.description,
                    'severity': anomaly.severity,
                    'confidence': anomaly.confidence,
                    'page_number': anomaly.page_number,
                    'statistical_significance': anomaly.statistical_significance
                })
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                patterns, anomalies, geometric_analysis, statistical_data
            )
            
            return ReportData(
                document_info=document_info,
                analysis_summary=analysis_summary,
                pattern_findings=pattern_findings,
                geometric_analysis=geometric_analysis,
                cipher_findings=cipher_findings,
                anomalies=anomaly_list,
                statistical_data=statistical_data,
                visualizations=visualizations,
                cross_references=[],  # Will be populated for multi-document reports
                recommendations=recommendations,
                metadata={
                    'generation_time': datetime.now(),
                    'generator_version': '1.0.0',
                    'configuration': config.__dict__
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting report data: {str(e)}")
            raise
    
    def _collect_geometric_analysis(self, document_id: int) -> Dict[str, Any]:
        """
        Collect geometric analysis data
        """
        try:
            # Create coordinate plot visualization
            coord_plot = self.geometric_visualizer.create_interactive_coordinate_plot(document_id)
            
            # Create angle measurements
            angle_viz = self.geometric_visualizer.create_angle_measurement_visualization(document_id)
            
            # Create distance analysis
            distance_viz = self.geometric_visualizer.create_distance_analysis_visualization(document_id)
            
            # Create sacred geometry analysis
            sacred_viz = self.geometric_visualizer.create_sacred_geometry_visualization(document_id)
            
            return {
                'coordinate_analysis': coord_plot.measurements_summary,
                'angle_analysis': angle_viz.measurements_summary,
                'distance_analysis': distance_viz.measurements_summary,
                'sacred_geometry': {
                    'patterns_found': len(sacred_viz.sacred_geometry_patterns),
                    'patterns': sacred_viz.sacred_geometry_patterns,
                    'summary': sacred_viz.measurements_summary
                },
                'visualizations': {
                    'coordinate_plot_config': coord_plot.d3_config,
                    'angle_viz_config': angle_viz.d3_config,
                    'distance_viz_config': distance_viz.d3_config,
                    'sacred_viz_config': sacred_viz.d3_config
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting geometric analysis: {str(e)}")
            return {}
    
    def _collect_statistical_data(self, document_id: int, patterns: List[Pattern]) -> Dict[str, Any]:
        """
        Collect statistical analysis data
        """
        try:
            # Get pages for character analysis
            pages = self.db.query(Page).filter(Page.document_id == document_id).all()
            
            # Calculate pattern statistics
            pattern_types = {}
            confidence_scores = []
            pages_with_patterns = set()
            
            for pattern in patterns:
                pattern_type = pattern.pattern_type
                pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
                confidence_scores.append(pattern.confidence)
                if pattern.page_number:
                    pages_with_patterns.add(pattern.page_number)
            
            # Calculate character statistics
            total_characters = 0
            character_distribution = {}
            
            for page in pages:
                if page.extracted_text:
                    total_characters += len(page.extracted_text)
                    for char in page.extracted_text:
                        character_distribution[char] = character_distribution.get(char, 0) + 1
            
            # Calculate statistical measures
            stats = {
                'pattern_statistics': {
                    'total_patterns': len(patterns),
                    'pattern_types': pattern_types,
                    'average_confidence': statistics.mean(confidence_scores) if confidence_scores else 0,
                    'confidence_std_dev': statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0,
                    'pages_with_patterns': len(pages_with_patterns),
                    'pattern_density': len(patterns) / len(pages) if pages else 0
                },
                'character_statistics': {
                    'total_characters': total_characters,
                    'unique_characters': len(character_distribution),
                    'character_distribution': dict(sorted(character_distribution.items(), 
                                                        key=lambda x: x[1], reverse=True)[:50]),  # Top 50
                    'average_chars_per_page': total_characters / len(pages) if pages else 0
                },
                'document_statistics': {
                    'total_pages': len(pages),
                    'pages_processed': len([p for p in pages if p.extracted_text]),
                    'processing_completion': len([p for p in pages if p.extracted_text]) / len(pages) if pages else 0
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error collecting statistical data: {str(e)}")
            return {}
    
    def _generate_visualizations(self, document_id: int) -> List[Dict[str, Any]]:
        """
        Generate visualization configurations for the report
        """
        try:
            visualizations = []
            
            # Coordinate plot
            coord_viz = self.geometric_visualizer.create_interactive_coordinate_plot(document_id)
            visualizations.append({
                'type': 'coordinate_plot',
                'title': 'Interactive Coordinate Plot',
                'description': 'Complete geometric analysis with all detected patterns',
                'config': coord_viz.d3_config,
                'data': coord_viz.export_data
            })
            
            # Angle measurements
            angle_viz = self.geometric_visualizer.create_angle_measurement_visualization(document_id)
            visualizations.append({
                'type': 'angle_analysis',
                'title': 'Angle Measurements',
                'description': 'Analysis of angles between geometric points',
                'config': angle_viz.d3_config,
                'data': angle_viz.export_data
            })
            
            # Distance analysis
            distance_viz = self.geometric_visualizer.create_distance_analysis_visualization(document_id)
            visualizations.append({
                'type': 'distance_analysis',
                'title': 'Distance Analysis',
                'description': 'Distance measurements with golden ratio detection',
                'config': distance_viz.d3_config,
                'data': distance_viz.export_data
            })
            
            # Sacred geometry
            sacred_viz = self.geometric_visualizer.create_sacred_geometry_visualization(document_id)
            visualizations.append({
                'type': 'sacred_geometry',
                'title': 'Sacred Geometry Patterns',
                'description': 'Detection of sacred geometry patterns and mathematical constants',
                'config': sacred_viz.d3_config,
                'data': sacred_viz.export_data
            })
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            return []
    
    def _generate_recommendations(self, patterns: List[Pattern], anomalies: List[Any],
                                geometric_analysis: Dict[str, Any], 
                                statistical_data: Dict[str, Any]) -> List[str]:
        """
        Generate analysis recommendations based on findings
        """
        recommendations = []
        
        try:
            # Pattern-based recommendations
            if len(patterns) > 100:
                recommendations.append("High pattern density detected. Consider focusing analysis on most significant patterns.")
            
            if len([p for p in patterns if p.confidence > 0.8]) < len(patterns) * 0.3:
                recommendations.append("Many patterns have low confidence scores. Consider adjusting detection parameters.")
            
            # Geometric analysis recommendations
            if geometric_analysis.get('sacred_geometry', {}).get('patterns_found', 0) > 0:
                recommendations.append("Sacred geometry patterns detected. Investigate potential intentional geometric encoding.")
            
            # Anomaly-based recommendations
            if len(anomalies) > 0:
                recommendations.append(f"Found {len(anomalies)} anomalies. These may indicate encoding or transcription errors.")
            
            # Statistical recommendations
            pattern_density = statistical_data.get('pattern_statistics', {}).get('pattern_density', 0)
            if pattern_density > 5:
                recommendations.append("Very high pattern density suggests systematic encoding throughout the document.")
            elif pattern_density < 0.5:
                recommendations.append("Low pattern density. Consider expanding search parameters or analyzing specific sections.")
            
            # Default recommendation
            if not recommendations:
                recommendations.append("Analysis complete. Review pattern findings and consider cross-referencing with other documents.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Error generating recommendations. Please review analysis results manually."]
    
    def _generate_html_report(self, data: ReportData, config: ReportConfiguration) -> str:
        """
        Generate HTML format report
        """
        try:
            # Use custom template if specified, otherwise use default
            template_name = config.template_name or "comprehensive_report.html"
            
            # Create default template if it doesn't exist
            if not (self.templates_dir / template_name).exists():
                self._create_default_html_template(template_name)
            
            template = self.jinja_env.get_template(template_name)
            
            # Render template with data
            html_content = template.render(
                data=data,
                config=config,
                generation_time=datetime.now(),
                sections=config.sections
            )
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {str(e)}")
            raise
    
    def _generate_json_report(self, data: ReportData, config: ReportConfiguration) -> str:
        """
        Generate JSON format report
        """
        try:
            # Convert dataclass to dict for JSON serialization
            report_dict = {
                'document_info': data.document_info,
                'analysis_summary': data.analysis_summary,
                'pattern_findings': data.pattern_findings,
                'geometric_analysis': data.geometric_analysis,
                'cipher_findings': data.cipher_findings,
                'anomalies': data.anomalies,
                'statistical_data': data.statistical_data,
                'visualizations': data.visualizations if config.include_visualizations else [],
                'cross_references': data.cross_references,
                'recommendations': data.recommendations,
                'metadata': data.metadata,
                'configuration': {
                    'report_type': config.report_type.value,
                    'format': config.format.value,
                    'sections': [s.value for s in config.sections],
                    'confidence_threshold': config.confidence_threshold
                }
            }
            
            return json.dumps(report_dict, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Error generating JSON report: {str(e)}")
            raise
    
    def _generate_csv_report(self, data: ReportData, config: ReportConfiguration) -> str:
        """
        Generate CSV format report (patterns summary)
        """
        try:
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                'Pattern ID', 'Type', 'Description', 'Confidence', 
                'Significance Score', 'Page Number', 'Supporting Evidence'
            ])
            
            # Write pattern data
            for pattern in data.pattern_findings:
                writer.writerow([
                    pattern.get('id', ''),
                    pattern.get('type', ''),
                    pattern.get('description', ''),
                    pattern.get('confidence', ''),
                    pattern.get('significance_score', ''),
                    pattern.get('page_number', ''),
                    pattern.get('supporting_evidence', '')
                ])
            
            return output.getvalue()
            
        except Exception as e:
            self.logger.error(f"Error generating CSV report: {str(e)}")
            raise
    
    def _generate_markdown_report(self, data: ReportData, config: ReportConfiguration) -> str:
        """
        Generate Markdown format report
        """
        try:
            md_content = []
            
            # Title
            md_content.append(f"# Analysis Report: {data.document_info['filename']}")
            md_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            md_content.append("")
            
            # Executive Summary
            if ReportSection.EXECUTIVE_SUMMARY in config.sections:
                md_content.append("## Executive Summary")
                md_content.append(f"- Total Patterns Found: {data.analysis_summary['total_patterns']}")
                md_content.append(f"- Significant Patterns: {data.analysis_summary['significant_patterns']}")
                md_content.append(f"- Anomalies Detected: {data.analysis_summary['anomalies_found']}")
                md_content.append("")
            
            # Document Overview
            if ReportSection.DOCUMENT_OVERVIEW in config.sections:
                md_content.append("## Document Overview")
                md_content.append(f"- **Filename:** {data.document_info['filename']}")
                md_content.append(f"- **Total Pages:** {data.document_info['total_pages']}")
                md_content.append(f"- **File Size:** {data.document_info['file_size']} bytes")
                md_content.append(f"- **Processing Status:** {data.document_info['processing_status']}")
                md_content.append("")
            
            # Pattern Findings
            if ReportSection.PATTERN_FINDINGS in config.sections:
                md_content.append("## Pattern Findings")
                for i, pattern in enumerate(data.pattern_findings[:10], 1):
                    md_content.append(f"### Pattern {i}: {pattern['type']}")
                    md_content.append(f"- **Description:** {pattern['description']}")
                    md_content.append(f"- **Confidence:** {pattern['confidence']:.2f}")
                    md_content.append(f"- **Significance:** {pattern['significance_score']:.2f}")
                    md_content.append(f"- **Page:** {pattern['page_number']}")
                    md_content.append("")
            
            # Recommendations
            if ReportSection.RECOMMENDATIONS in config.sections:
                md_content.append("## Recommendations")
                for i, rec in enumerate(data.recommendations, 1):
                    md_content.append(f"{i}. {rec}")
                md_content.append("")
            
            return "\n".join(md_content)
            
        except Exception as e:
            self.logger.error(f"Error generating Markdown report: {str(e)}")
            raise
    
    def _generate_pdf_report(self, data: ReportData, config: ReportConfiguration) -> bytes:
        """
        Generate PDF format report (placeholder - would need additional PDF library)
        """
        try:
            # For now, generate HTML and note that PDF conversion would be needed
            html_content = self._generate_html_report(data, config)
            
            # In a real implementation, you would use a library like:
            # - weasyprint
            # - reportlab
            # - pdfkit
            # to convert HTML to PDF
            
            # For now, return HTML as bytes with a note
            pdf_note = "<!-- PDF generation requires additional PDF library implementation -->\n"
            return (pdf_note + html_content).encode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {str(e)}")
            raise
    
    def _generate_excel_report(self, data: ReportData, config: ReportConfiguration) -> bytes:
        """
        Generate Excel format report (placeholder - would need openpyxl)
        """
        try:
            # For now, return CSV content as bytes
            csv_content = self._generate_csv_report(data, config)
            return csv_content.encode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Error generating Excel report: {str(e)}")
            raise
    
    def _merge_multi_document_data(self, all_data: List[ReportData]) -> ReportData:
        """
        Merge data from multiple documents for cross-document analysis
        """
        try:
            # Merge document info
            merged_doc_info = {
                'documents': [data.document_info for data in all_data],
                'total_documents': len(all_data),
                'combined_pages': sum(data.document_info['total_pages'] for data in all_data)
            }
            
            # Merge patterns
            all_patterns = []
            for data in all_data:
                all_patterns.extend(data.pattern_findings)
            
            # Merge analysis summaries
            merged_summary = {
                'total_patterns': sum(data.analysis_summary['total_patterns'] for data in all_data),
                'total_significant_patterns': sum(data.analysis_summary['significant_patterns'] for data in all_data),
                'total_anomalies': sum(data.analysis_summary['anomalies_found'] for data in all_data),
                'documents_analyzed': len(all_data)
            }
            
            # Create merged report data
            return ReportData(
                document_info=merged_doc_info,
                analysis_summary=merged_summary,
                pattern_findings=all_patterns,
                geometric_analysis={},  # Would need cross-document geometric analysis
                cipher_findings=[],
                anomalies=[],
                statistical_data={},
                visualizations=[],
                cross_references=[],  # Would analyze cross-document correlations
                recommendations=["Multi-document analysis complete. Review cross-document patterns."],
                metadata={'type': 'multi_document', 'generation_time': datetime.now()}
            )
            
        except Exception as e:
            self.logger.error(f"Error merging multi-document data: {str(e)}")
            raise
    
    def _generate_report_summary(self, data: ReportData) -> Dict[str, Any]:
        """
        Generate a summary of the report contents
        """
        # Being defensive as tests may mock ReportData
        pattern_findings = getattr(data, 'pattern_findings', []) or []
        anomalies = getattr(data, 'anomalies', []) or []
        cipher_findings = getattr(data, 'cipher_findings', []) or []
        visualizations = getattr(data, 'visualizations', []) or []
        recommendations = getattr(data, 'recommendations', []) or []
        document_info = getattr(data, 'document_info', {}) or {}

        return {
            'total_patterns': len(pattern_findings),
            'high_confidence_patterns': len([p for p in pattern_findings if p.get('confidence', 0) > 0.8]),
            'anomalies_count': len(anomalies),
            'cipher_findings_count': len(cipher_findings),
            'visualizations_count': len(visualizations),
            'recommendations_count': len(recommendations),
            'document_pages': document_info.get('total_pages', 0)
        }
    
    def _create_default_html_template(self, template_name: str):
        """
        Create a default HTML template for reports
        """
        default_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ancient Text Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
        .section { margin-bottom: 30px; }
        .pattern { background: #f5f5f5; padding: 15px; margin: 10px 0; border-left: 4px solid #007acc; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .stat-box { background: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }
        .recommendations { background: #fff3cd; padding: 20px; border-radius: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Ancient Text Analysis Report</h1>
        <p><strong>Document:</strong> {{ data.document_info.filename }}</p>
        <p><strong>Generated:</strong> {{ generation_time.strftime('%Y-%m-%d %H:%M:%S') }}</p>
    </div>

    {% if 'executive_summary' in [s.value for s in sections] %}
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="stats">
            <div class="stat-box">
                <h3>{{ data.analysis_summary.total_patterns }}</h3>
                <p>Total Patterns</p>
            </div>
            <div class="stat-box">
                <h3>{{ data.analysis_summary.significant_patterns }}</h3>
                <p>Significant Patterns</p>
            </div>
            <div class="stat-box">
                <h3>{{ data.analysis_summary.anomalies_found }}</h3>
                <p>Anomalies Found</p>
            </div>
        </div>
    </div>
    {% endif %}

    {% if 'pattern_findings' in [s.value for s in sections] %}
    <div class="section">
        <h2>Pattern Findings</h2>
        {% for pattern in data.pattern_findings[:10] %}
        <div class="pattern">
            <h4>{{ pattern.type }} (Confidence: {{ "%.2f"|format(pattern.confidence) }})</h4>
            <p>{{ pattern.description }}</p>
            <p><strong>Page:</strong> {{ pattern.page_number }} | <strong>Significance:</strong> {{ "%.2f"|format(pattern.significance_score) }}</p>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    {% if 'statistical_analysis' in [s.value for s in sections] %}
    <div class="section">
        <h2>Statistical Analysis</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Characters</td><td>{{ data.statistical_data.character_statistics.total_characters }}</td></tr>
            <tr><td>Unique Characters</td><td>{{ data.statistical_data.character_statistics.unique_characters }}</td></tr>
            <tr><td>Pattern Density</td><td>{{ "%.2f"|format(data.statistical_data.pattern_statistics.pattern_density) }}</td></tr>
            <tr><td>Average Confidence</td><td>{{ "%.2f"|format(data.statistical_data.pattern_statistics.average_confidence) }}</td></tr>
        </table>
    </div>
    {% endif %}

    {% if 'recommendations' in [s.value for s in sections] %}
    <div class="section">
        <h2>Recommendations</h2>
        <div class="recommendations">
            <ul>
            {% for rec in data.recommendations %}
                <li>{{ rec }}</li>
            {% endfor %}
            </ul>
        </div>
    </div>
    {% endif %}

    <div class="section">
        <h2>Report Metadata</h2>
        <p><strong>Generator Version:</strong> {{ data.metadata.generator_version }}</p>
        <p><strong>Configuration:</strong> {{ config.report_type.value }} report in {{ config.format.value }} format</p>
    </div>
</body>
</html>
        """
        
        template_path = self.templates_dir / template_name
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(default_template.strip())