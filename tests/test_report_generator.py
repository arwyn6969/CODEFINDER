"""
Tests for Comprehensive Reporting System
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from sqlalchemy.orm import Session

from app.services.report_generator import (
    ReportGenerator, ReportConfiguration, ReportData, GeneratedReport,
    ReportFormat, ReportType, ReportSection
)
from app.models.database_models import Document, Pattern, Page


class TestReportGenerator:
    """Test cases for ReportGenerator"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session"""
        return Mock(spec=Session)
    
    @pytest.fixture
    def sample_document(self):
        """Create sample document"""
        doc = Mock(spec=Document)
        doc.id = 1
        doc.filename = "test_document.pdf"
        doc.file_size = 1024000
        doc.upload_date = datetime.now()
        doc.total_pages = 10
        doc.processing_status = "completed"
        return doc
    
    @pytest.fixture
    def sample_patterns(self):
        """Create sample patterns"""
        patterns = []
        for i in range(5):
            pattern = Mock(spec=Pattern)
            pattern.id = i + 1
            pattern.document_id = 1
            pattern.pattern_type = f"pattern_type_{i}"
            pattern.description = f"Test pattern {i}"
            pattern.confidence = 0.8 + (i * 0.05)
            pattern.page_number = i + 1
            pattern.coordinates = [{"x": i * 10, "y": i * 10}]
            pattern.metadata = {"test": f"metadata_{i}"}
            patterns.append(pattern)
        return patterns
    
    @pytest.fixture
    def sample_pages(self):
        """Create sample pages"""
        pages = []
        for i in range(3):
            page = Mock(spec=Page)
            page.id = i + 1
            page.document_id = 1
            page.page_number = i + 1
            page.extracted_text = f"Sample text for page {i + 1} " * 50
            pages.append(page)
        return pages
    
    @pytest.fixture
    def report_generator(self, mock_db_session):
        """Create ReportGenerator instance"""
        with patch('app.services.report_generator.get_db') as mock_get_db:
            mock_get_db.return_value = mock_db_session
            
            # Mock the analysis services
            with patch('app.services.report_generator.PatternSignificanceRanker') as mock_ranker, \
                 patch('app.services.report_generator.GeometricVisualizer') as mock_visualizer, \
                 patch('app.services.report_generator.AnomalyDetector') as mock_detector:
                
                generator = ReportGenerator(mock_db_session)
                generator.significance_ranker = mock_ranker.return_value
                generator.geometric_visualizer = mock_visualizer.return_value
                generator.anomaly_detector = mock_detector.return_value
                
                return generator
    
    def test_init(self, report_generator, mock_db_session):
        """Test ReportGenerator initialization"""
        assert report_generator.db == mock_db_session
        assert report_generator.templates_dir.exists()
        assert ReportType.COMPREHENSIVE in report_generator.default_configs
        assert ReportType.SUMMARY in report_generator.default_configs
    
    def test_default_configurations(self, report_generator):
        """Test default report configurations"""
        comprehensive_config = report_generator.default_configs[ReportType.COMPREHENSIVE]
        assert comprehensive_config.report_type == ReportType.COMPREHENSIVE
        assert comprehensive_config.format == ReportFormat.HTML
        assert ReportSection.EXECUTIVE_SUMMARY in comprehensive_config.sections
        assert comprehensive_config.include_visualizations is True
        
        summary_config = report_generator.default_configs[ReportType.SUMMARY]
        assert summary_config.report_type == ReportType.SUMMARY
        assert summary_config.include_visualizations is False
        assert summary_config.max_patterns_per_section == 10
    
    @patch('app.services.report_generator.ReportGenerator._collect_report_data')
    @patch('app.services.report_generator.ReportGenerator._generate_html_report')
    def test_generate_report_html(self, mock_generate_html, mock_collect_data, 
                                 report_generator, sample_document):
        """Test HTML report generation"""
        # Mock data collection
        mock_report_data = Mock(spec=ReportData)
        mock_collect_data.return_value = mock_report_data
        
        # Mock HTML generation
        mock_generate_html.return_value = "<html>Test Report</html>"
        
        # Generate report
        config = ReportConfiguration(
            report_type=ReportType.COMPREHENSIVE,
            format=ReportFormat.HTML,
            sections=[ReportSection.EXECUTIVE_SUMMARY]
        )
        
        report = report_generator.generate_report(document_id=1, config=config)
        
        # Verify report
        assert isinstance(report, GeneratedReport)
        assert report.document_id == 1
        assert report.report_type == ReportType.COMPREHENSIVE
        assert report.format == ReportFormat.HTML
        assert report.content == "<html>Test Report</html>"
        assert report.size_bytes > 0
        
        # Verify method calls
        mock_collect_data.assert_called_once_with(1, config)
        mock_generate_html.assert_called_once_with(mock_report_data, config)
    
    @patch('app.services.report_generator.ReportGenerator._collect_report_data')
    def test_generate_report_json(self, mock_collect_data, report_generator):
        """Test JSON report generation"""
        # Create mock report data
        mock_data = ReportData(
            document_info={'filename': 'test.pdf'},
            analysis_summary={'total_patterns': 5},
            pattern_findings=[],
            geometric_analysis={},
            cipher_findings=[],
            anomalies=[],
            statistical_data={},
            visualizations=[],
            cross_references=[],
            recommendations=[],
            metadata={}
        )
        mock_collect_data.return_value = mock_data
        
        # Generate JSON report
        config = ReportConfiguration(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.JSON,
            sections=[ReportSection.EXECUTIVE_SUMMARY]
        )
        
        report = report_generator.generate_report(document_id=1, config=config)
        
        # Verify JSON content
        assert report.format == ReportFormat.JSON
        json_data = json.loads(report.content)
        assert 'document_info' in json_data
        assert 'analysis_summary' in json_data
        assert json_data['document_info']['filename'] == 'test.pdf'
    
    @patch('app.services.report_generator.ReportGenerator._collect_report_data')
    def test_generate_report_csv(self, mock_collect_data, report_generator):
        """Test CSV report generation"""
        # Create mock report data with patterns
        mock_data = ReportData(
            document_info={'filename': 'test.pdf'},
            analysis_summary={'total_patterns': 2},
            pattern_findings=[
                {
                    'id': 1,
                    'type': 'test_pattern',
                    'description': 'Test pattern 1',
                    'confidence': 0.8,
                    'significance_score': 0.9,
                    'page_number': 1,
                    'supporting_evidence': 'Evidence 1'
                },
                {
                    'id': 2,
                    'type': 'another_pattern',
                    'description': 'Test pattern 2',
                    'confidence': 0.7,
                    'significance_score': 0.8,
                    'page_number': 2,
                    'supporting_evidence': 'Evidence 2'
                }
            ],
            geometric_analysis={},
            cipher_findings=[],
            anomalies=[],
            statistical_data={},
            visualizations=[],
            cross_references=[],
            recommendations=[],
            metadata={}
        )
        mock_collect_data.return_value = mock_data
        
        # Generate CSV report
        config = ReportConfiguration(
            report_type=ReportType.PATTERN_ANALYSIS,
            format=ReportFormat.CSV,
            sections=[ReportSection.PATTERN_FINDINGS]
        )
        
        report = report_generator.generate_report(document_id=1, config=config)
        
        # Verify CSV content
        assert report.format == ReportFormat.CSV
        lines = report.content.strip().split('\n')
        assert len(lines) == 3  # Header + 2 data rows
        assert 'Pattern ID' in lines[0]
        assert 'test_pattern' in lines[1]
        assert 'another_pattern' in lines[2]
    
    @patch('app.services.report_generator.ReportGenerator._collect_report_data')
    def test_generate_report_markdown(self, mock_collect_data, report_generator):
        """Test Markdown report generation"""
        # Create mock report data
        mock_data = ReportData(
            document_info={'filename': 'test.pdf', 'total_pages': 10, 'file_size': 1024, 'processing_status': 'completed'},
            analysis_summary={'total_patterns': 5, 'significant_patterns': 3, 'anomalies_found': 1},
            pattern_findings=[
                {
                    'type': 'test_pattern',
                    'description': 'Test pattern description',
                    'confidence': 0.85,
                    'significance_score': 0.9,
                    'page_number': 1
                }
            ],
            geometric_analysis={},
            cipher_findings=[],
            anomalies=[],
            statistical_data={},
            visualizations=[],
            cross_references=[],
            recommendations=['Test recommendation'],
            metadata={}
        )
        mock_collect_data.return_value = mock_data
        
        # Generate Markdown report
        config = ReportConfiguration(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.MARKDOWN,
            sections=[ReportSection.EXECUTIVE_SUMMARY, ReportSection.DOCUMENT_OVERVIEW, 
                     ReportSection.PATTERN_FINDINGS, ReportSection.RECOMMENDATIONS]
        )
        
        report = report_generator.generate_report(document_id=1, config=config)
        
        # Verify Markdown content
        assert report.format == ReportFormat.MARKDOWN
        content = report.content
        assert '# Analysis Report: test.pdf' in content
        assert '## Executive Summary' in content
        assert '## Document Overview' in content
        assert '## Pattern Findings' in content
        assert '## Recommendations' in content
        assert 'Total Patterns Found: 5' in content
        assert 'Test recommendation' in content
    
    def test_collect_report_data(self, report_generator, sample_document, sample_patterns, sample_pages):
        """Test report data collection"""
        # Mock database queries
        report_generator.db.query.return_value.filter.return_value.first.return_value = sample_document
        report_generator.db.query.return_value.filter.return_value.all.return_value = sample_patterns
        
        # Mock pages query
        def mock_query_side_effect(model):
            if model == Pattern:
                return Mock(filter=Mock(return_value=Mock(all=Mock(return_value=sample_patterns))))
            elif model == Page:
                return Mock(filter=Mock(return_value=Mock(all=Mock(return_value=sample_pages))))
            else:
                return Mock(filter=Mock(return_value=Mock(first=Mock(return_value=sample_document))))
        
        report_generator.db.query.side_effect = mock_query_side_effect
        
        # Mock analysis services
        mock_ranked_patterns = []
        for pattern in sample_patterns:
            mock_ranked_pattern = Mock()
            mock_ranked_pattern.pattern = pattern
            mock_ranked_pattern.significance_score = 0.8
            mock_ranked_pattern.supporting_evidence = "Test evidence"
            mock_ranked_patterns.append(mock_ranked_pattern)
        
        report_generator.significance_ranker.rank_patterns.return_value = mock_ranked_patterns
        report_generator.anomaly_detector.detect_anomalies.return_value = []
        
        # Mock geometric visualizer
        mock_viz = Mock()
        mock_viz.measurements_summary = {'test': 'data'}
        mock_viz.d3_config = {'config': 'test'}
        mock_viz.export_data = {'export': 'test'}
        mock_viz.sacred_geometry_patterns = []
        
        report_generator.geometric_visualizer.create_interactive_coordinate_plot.return_value = mock_viz
        report_generator.geometric_visualizer.create_angle_measurement_visualization.return_value = mock_viz
        report_generator.geometric_visualizer.create_distance_analysis_visualization.return_value = mock_viz
        report_generator.geometric_visualizer.create_sacred_geometry_visualization.return_value = mock_viz
        
        # Create config
        config = ReportConfiguration(
            report_type=ReportType.COMPREHENSIVE,
            format=ReportFormat.HTML,
            sections=[ReportSection.EXECUTIVE_SUMMARY],
            confidence_threshold=0.7,
            max_patterns_per_section=10
        )
        
        # Collect data
        data = report_generator._collect_report_data(document_id=1, config=config)
        
        # Verify collected data
        assert isinstance(data, ReportData)
        assert data.document_info['filename'] == 'test_document.pdf'
        assert data.document_info['total_pages'] == 10
        assert data.analysis_summary['total_patterns'] == 5
        assert len(data.pattern_findings) == 5
        assert data.statistical_data['pattern_statistics']['total_patterns'] == 5
        assert len(data.recommendations) > 0
    
    def test_collect_statistical_data(self, report_generator, sample_patterns, sample_pages):
        """Test statistical data collection"""
        # Mock database query for pages
        report_generator.db.query.return_value.filter.return_value.all.return_value = sample_pages
        
        # Collect statistical data
        stats = report_generator._collect_statistical_data(document_id=1, patterns=sample_patterns)
        
        # Verify statistics
        assert 'pattern_statistics' in stats
        assert 'character_statistics' in stats
        assert 'document_statistics' in stats
        
        pattern_stats = stats['pattern_statistics']
        assert pattern_stats['total_patterns'] == 5
        assert pattern_stats['average_confidence'] > 0
        assert pattern_stats['pattern_density'] > 0
        
        char_stats = stats['character_statistics']
        assert char_stats['total_characters'] > 0
        assert char_stats['unique_characters'] > 0
        
        doc_stats = stats['document_statistics']
        assert doc_stats['total_pages'] == 3
        assert doc_stats['pages_processed'] == 3
    
    def test_generate_recommendations(self, report_generator, sample_patterns):
        """Test recommendation generation"""
        # Create mock anomalies
        mock_anomalies = [Mock(), Mock()]
        
        # Create mock geometric analysis
        geometric_analysis = {
            'sacred_geometry': {'patterns_found': 2}
        }
        
        # Create mock statistical data
        statistical_data = {
            'pattern_statistics': {'pattern_density': 3.5}
        }
        
        # Generate recommendations
        recommendations = report_generator._generate_recommendations(
            sample_patterns, mock_anomalies, geometric_analysis, statistical_data
        )
        
        # Verify recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('Sacred geometry patterns detected' in rec for rec in recommendations)
        assert any('anomalies' in rec.lower() for rec in recommendations)
    
    def test_save_report(self, report_generator):
        """Test report saving"""
        # Create a test report
        report = GeneratedReport(
            report_id="test_report",
            document_id=1,
            report_type=ReportType.SUMMARY,
            format=ReportFormat.HTML,
            content="<html>Test Report Content</html>",
            file_path=None,
            generation_time=datetime.now(),
            configuration=Mock(),
            summary={},
            size_bytes=100
        )
        
        # Save to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.html"
            
            saved_path = report_generator.save_report(report, str(output_path))
            
            # Verify file was saved
            assert Path(saved_path).exists()
            assert report.file_path == saved_path
            
            # Verify content
            with open(saved_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert content == "<html>Test Report Content</html>"
    
    def test_create_custom_template(self, report_generator):
        """Test custom template creation"""
        template_content = """
        <html>
        <body>
            <h1>Custom Template</h1>
            <p>{{ data.document_info.filename }}</p>
        </body>
        </html>
        """
        
        template_path = report_generator.create_custom_template("custom_test", template_content)
        
        # Verify template was created
        assert Path(template_path).exists()
        
        # Verify content
        with open(template_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()
            assert "Custom Template" in saved_content
            assert "{{ data.document_info.filename }}" in saved_content
    
    def test_get_available_templates(self, report_generator):
        """Test listing available templates"""
        # Create some test templates
        test_templates = ["template1.html", "template2.html", "template3.html"]
        
        for template_name in test_templates:
            template_path = report_generator.templates_dir / template_name
            template_path.write_text("<html>Test</html>")
        
        # Get available templates
        available = report_generator.get_available_templates()
        
        # Verify templates are listed
        for template in test_templates:
            template_stem = template.replace('.html', '')
            assert template_stem in available
    
    @patch('app.services.report_generator.ReportGenerator._collect_report_data')
    def test_generate_multi_document_report(self, mock_collect_data, report_generator):
        """Test multi-document report generation"""
        # Create mock data for multiple documents
        mock_data_1 = ReportData(
            document_info={'filename': 'doc1.pdf', 'total_pages': 5},
            analysis_summary={'total_patterns': 3, 'significant_patterns': 2, 'anomalies_found': 1},
            pattern_findings=[],
            geometric_analysis={},
            cipher_findings=[],
            anomalies=[],
            statistical_data={},
            visualizations=[],
            cross_references=[],
            recommendations=[],
            metadata={}
        )
        
        mock_data_2 = ReportData(
            document_info={'filename': 'doc2.pdf', 'total_pages': 8},
            analysis_summary={'total_patterns': 5, 'significant_patterns': 3, 'anomalies_found': 2},
            pattern_findings=[],
            geometric_analysis={},
            cipher_findings=[],
            anomalies=[],
            statistical_data={},
            visualizations=[],
            cross_references=[],
            recommendations=[],
            metadata={}
        )
        
        mock_collect_data.side_effect = [mock_data_1, mock_data_2]
        
        # Generate multi-document report
        config = ReportConfiguration(
            report_type=ReportType.CROSS_DOCUMENT,
            format=ReportFormat.JSON,
            sections=[ReportSection.EXECUTIVE_SUMMARY]
        )
        
        report = report_generator.generate_multi_document_report([1, 2], config)
        
        # Verify multi-document report
        assert report.report_type == ReportType.CROSS_DOCUMENT
        assert report.document_id == 0  # Multi-document indicator
        
        # Parse JSON content to verify merged data
        json_data = json.loads(report.content)
        assert json_data['document_info']['total_documents'] == 2
        assert json_data['document_info']['combined_pages'] == 13
        assert json_data['analysis_summary']['total_patterns'] == 8
        assert json_data['analysis_summary']['documents_analyzed'] == 2
    
    def test_report_configuration_dataclass(self):
        """Test ReportConfiguration dataclass"""
        config = ReportConfiguration(
            report_type=ReportType.PATTERN_ANALYSIS,
            format=ReportFormat.PDF,
            sections=[ReportSection.PATTERN_FINDINGS, ReportSection.STATISTICAL_ANALYSIS],
            include_visualizations=False,
            confidence_threshold=0.8,
            max_patterns_per_section=25
        )
        
        assert config.report_type == ReportType.PATTERN_ANALYSIS
        assert config.format == ReportFormat.PDF
        assert len(config.sections) == 2
        assert config.include_visualizations is False
        assert config.confidence_threshold == 0.8
        assert config.max_patterns_per_section == 25
    
    def test_generated_report_dataclass(self):
        """Test GeneratedReport dataclass"""
        config = ReportConfiguration(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.HTML,
            sections=[ReportSection.EXECUTIVE_SUMMARY]
        )
        
        report = GeneratedReport(
            report_id="test_123",
            document_id=1,
            report_type=ReportType.SUMMARY,
            format=ReportFormat.HTML,
            content="<html>Test</html>",
            file_path="/path/to/report.html",
            generation_time=datetime.now(),
            configuration=config,
            summary={'patterns': 5},
            size_bytes=1024
        )
        
        assert report.report_id == "test_123"
        assert report.document_id == 1
        assert report.report_type == ReportType.SUMMARY
        assert report.format == ReportFormat.HTML
        assert report.size_bytes == 1024
        assert report.summary['patterns'] == 5
    
    def test_error_handling_missing_document(self, report_generator):
        """Test error handling for missing document"""
        # Mock database to return None for document
        report_generator.db.query.return_value.filter.return_value.first.return_value = None
        
        config = ReportConfiguration(
            report_type=ReportType.SUMMARY,
            format=ReportFormat.HTML,
            sections=[ReportSection.EXECUTIVE_SUMMARY]
        )
        
        # Should raise ValueError for missing document
        with pytest.raises(ValueError, match="Document 999 not found"):
            report_generator._collect_report_data(document_id=999, config=config)
    
    def test_error_handling_invalid_format(self, report_generator):
        """Test error handling for invalid report format"""
        with patch.object(report_generator, '_collect_report_data') as mock_collect:
            mock_collect.return_value = Mock(spec=ReportData)
            
            # Create config with invalid format (simulate enum extension)
            config = ReportConfiguration(
                report_type=ReportType.SUMMARY,
                format="INVALID_FORMAT",  # This would normally be caught by enum
                sections=[ReportSection.EXECUTIVE_SUMMARY]
            )
            
            # Should raise ValueError for unsupported format
            with pytest.raises(ValueError, match="Unsupported report format"):
                report_generator.generate_report(document_id=1, config=config)
    
    def test_html_template_creation(self, report_generator):
        """Test HTML template creation and rendering"""
        # Test that default template is created when it doesn't exist
        template_name = "test_template.html"
        template_path = report_generator.templates_dir / template_name
        
        # Ensure template doesn't exist
        if template_path.exists():
            template_path.unlink()
        
        # Create default template
        report_generator._create_default_html_template(template_name)
        
        # Verify template was created
        assert template_path.exists()
        
        # Verify template content
        content = template_path.read_text()
        assert "Ancient Text Analysis Report" in content
        assert "{{ data.document_info.filename }}" in content
        assert "{{ generation_time.strftime" in content
    
    def test_report_data_structure_completeness(self):
        """Test that ReportData contains all necessary fields"""
        # Create a complete ReportData instance
        report_data = ReportData(
            document_info={'test': 'info'},
            analysis_summary={'test': 'summary'},
            pattern_findings=[{'test': 'pattern'}],
            geometric_analysis={'test': 'geometry'},
            cipher_findings=[{'test': 'cipher'}],
            anomalies=[{'test': 'anomaly'}],
            statistical_data={'test': 'stats'},
            visualizations=[{'test': 'viz'}],
            cross_references=[{'test': 'ref'}],
            recommendations=['test recommendation'],
            metadata={'test': 'meta'}
        )
        
        # Verify all fields are accessible
        assert report_data.document_info['test'] == 'info'
        assert report_data.analysis_summary['test'] == 'summary'
        assert len(report_data.pattern_findings) == 1
        assert report_data.geometric_analysis['test'] == 'geometry'
        assert len(report_data.cipher_findings) == 1
        assert len(report_data.anomalies) == 1
        assert report_data.statistical_data['test'] == 'stats'
        assert len(report_data.visualizations) == 1
        assert len(report_data.cross_references) == 1
        assert len(report_data.recommendations) == 1
        assert report_data.metadata['test'] == 'meta'


class TestReportEnums:
    """Test report-related enums"""
    
    def test_report_format_enum(self):
        """Test ReportFormat enum values"""
        assert ReportFormat.PDF.value == "pdf"
        assert ReportFormat.CSV.value == "csv"
        assert ReportFormat.JSON.value == "json"
        assert ReportFormat.HTML.value == "html"
        assert ReportFormat.MARKDOWN.value == "markdown"
        assert ReportFormat.EXCEL.value == "xlsx"
    
    def test_report_type_enum(self):
        """Test ReportType enum values"""
        assert ReportType.COMPREHENSIVE.value == "comprehensive"
        assert ReportType.PATTERN_ANALYSIS.value == "pattern_analysis"
        assert ReportType.GEOMETRIC_ANALYSIS.value == "geometric_analysis"
        assert ReportType.CIPHER_ANALYSIS.value == "cipher_analysis"
        assert ReportType.ANOMALY_DETECTION.value == "anomaly_detection"
        assert ReportType.CROSS_DOCUMENT.value == "cross_document"
        assert ReportType.SUMMARY.value == "summary"
        assert ReportType.STATISTICAL.value == "statistical"
    
    def test_report_section_enum(self):
        """Test ReportSection enum values"""
        assert ReportSection.EXECUTIVE_SUMMARY.value == "executive_summary"
        assert ReportSection.DOCUMENT_OVERVIEW.value == "document_overview"
        assert ReportSection.PATTERN_FINDINGS.value == "pattern_findings"
        assert ReportSection.GEOMETRIC_ANALYSIS.value == "geometric_analysis"
        assert ReportSection.CIPHER_DETECTION.value == "cipher_detection"
        assert ReportSection.ANOMALY_DETECTION.value == "anomaly_detection"
        assert ReportSection.STATISTICAL_ANALYSIS.value == "statistical_analysis"
        assert ReportSection.VISUALIZATIONS.value == "visualizations"
        assert ReportSection.RECOMMENDATIONS.value == "recommendations"
        assert ReportSection.APPENDICES.value == "appendices"
        assert ReportSection.METHODOLOGY.value == "methodology"
        assert ReportSection.REFERENCES.value == "references"


if __name__ == '__main__':
    pytest.main([__file__])