"""
Tests for End-to-End Processing Pipeline
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session

from app.services.processing_pipeline import (
    ProcessingPipeline, ProcessingStage, ProcessingStatus, ProcessingConfiguration,
    StageResult, ProcessingProgress
)
from app.models.database_models import Document, Page


class TestProcessingPipeline:
    """Test cases for ProcessingPipeline"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session"""
        return Mock(spec=Session)
    
    @pytest.fixture
    def pipeline_config(self):
        """Create test pipeline configuration"""
        return ProcessingConfiguration(
            ocr_confidence_threshold=0.8,
            enable_grid_analysis=True,
            enable_geometric_analysis=True,
            enable_etymology_analysis=False,  # Disable for faster testing
            enable_cipher_detection=True,
            enable_bardcode_analysis=False,  # Disable for faster testing
            enable_cross_document_analysis=False,  # Disable for faster testing
            enable_anomaly_detection=True,
            max_concurrent_operations=2,
            timeout_per_stage_minutes=1,  # Short timeout for testing
            save_intermediate_results=False
        )
    
    @pytest.fixture
    def pipeline(self, mock_db_session, pipeline_config):
        """Create ProcessingPipeline instance with mocks"""
        with patch.multiple(
            'app.services.processing_pipeline',
            PDFProcessor=Mock,
            ImageProcessor=Mock,
            AdvancedOCR=Mock,
            TextAnalyzer=Mock,
            GridGenerator=Mock,
            GeometricAnalyzer=Mock,
            EtymologyEngine=Mock,
            SearchService=Mock,
            CipherDetector=Mock,
            BardCodeAnalyzer=Mock,
            RelationshipAnalyzer=Mock,
            CrossDocumentAnalyzer=Mock,
            CrossDocumentPatternDatabase=Mock,
            CipherExplanationValidator=Mock,
            CrossReferenceVisualizer=Mock,
            AnomalyDetector=Mock,
            PatternSignificanceRanker=Mock
        ):
            return ProcessingPipeline(mock_db_session, pipeline_config)
    
    @pytest.fixture
    def sample_document(self):
        """Create sample document for testing"""
        return Document(
            id=1,
            filename="test_document.pdf",
            file_path="/path/to/test_document.pdf",
            content="Sample text content for testing",
            upload_date=datetime.now(),
            processing_status="pending"
        )
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert pipeline is not None
        assert pipeline.config is not None
        assert pipeline.progress.total_stages == len(ProcessingStage)
        assert pipeline.progress.completed_stages == 0
        assert len(pipeline.progress_callbacks) == 0
    
    def test_add_progress_callback(self, pipeline):
        """Test adding progress callbacks"""
        callback = Mock()
        pipeline.add_progress_callback(callback)
        
        assert len(pipeline.progress_callbacks) == 1
        assert callback in pipeline.progress_callbacks
    
    def test_notify_progress(self, pipeline):
        """Test progress notification"""
        callback1 = Mock()
        callback2 = Mock()
        
        pipeline.add_progress_callback(callback1)
        pipeline.add_progress_callback(callback2)
        
        pipeline._notify_progress()
        
        callback1.assert_called_once_with(pipeline.progress)
        callback2.assert_called_once_with(pipeline.progress)
    
    def test_notify_progress_with_callback_error(self, pipeline):
        """Test progress notification handles callback errors gracefully"""
        good_callback = Mock()
        bad_callback = Mock(side_effect=Exception("Callback error"))
        
        pipeline.add_progress_callback(good_callback)
        pipeline.add_progress_callback(bad_callback)
        
        # Should not raise exception
        pipeline._notify_progress()
        
        good_callback.assert_called_once()
        bad_callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_stage_success(self, pipeline):
        """Test successful stage execution"""
        async def mock_stage_func(*args, **kwargs):
            return {"test": "data"}
        
        result = await pipeline._execute_stage(
            ProcessingStage.TEXT_ANALYSIS,
            mock_stage_func,
            "test_arg"
        )
        
        assert isinstance(result, StageResult)
        assert result.stage == ProcessingStage.TEXT_ANALYSIS
        assert result.status == ProcessingStatus.COMPLETED
        assert result.data == {"test": "data"}
        assert result.duration is not None
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_execute_stage_failure(self, pipeline):
        """Test stage execution with failure"""
        async def mock_stage_func(*args, **kwargs):
            raise ValueError("Test error")
        
        result = await pipeline._execute_stage(
            ProcessingStage.TEXT_ANALYSIS,
            mock_stage_func,
            "test_arg"
        )
        
        assert isinstance(result, StageResult)
        assert result.stage == ProcessingStage.TEXT_ANALYSIS
        assert result.status == ProcessingStatus.FAILED
        assert result.error == "Test error"
        assert result.data == {}
    
    @pytest.mark.asyncio
    async def test_execute_stage_timeout(self, pipeline):
        """Test stage execution with timeout"""
        async def mock_stage_func(*args, **kwargs):
            await asyncio.sleep(2)  # Longer than timeout
            return {"test": "data"}
        
        # Set very short timeout for testing
        pipeline.config.timeout_per_stage_minutes = 0.01  # 0.6 seconds
        
        result = await pipeline._execute_stage(
            ProcessingStage.TEXT_ANALYSIS,
            mock_stage_func,
            "test_arg"
        )
        
        assert result.status == ProcessingStatus.FAILED
        assert "timed out" in result.error.lower()
    
    def test_record_performance_metrics(self, pipeline):
        """Test performance metrics recording"""
        stage = ProcessingStage.TEXT_ANALYSIS
        result = StageResult(
            stage=stage,
            status=ProcessingStatus.COMPLETED,
            start_time=datetime.now(),
            duration=1.5
        )
        
        pipeline._record_performance_metrics(stage, result)
        
        assert stage in pipeline.performance_metrics
        assert len(pipeline.performance_metrics[stage]) == 1
        
        metrics = pipeline.performance_metrics[stage][0]
        assert metrics['duration'] == 1.5
        assert metrics['status'] == 'completed'
    
    @pytest.mark.asyncio
    async def test_process_pdf_stage(self, pipeline, mock_db_session):
        """Test PDF processing stage"""
        # Mock PDF processor
        pipeline.pdf_processor.extract_pages.return_value = {
            'pages': [
                {'page_number': 1, 'image_path': '/path/to/page1.png', 'width': 800, 'height': 600},
                {'page_number': 2, 'image_path': '/path/to/page2.png', 'width': 800, 'height': 600}
            ],
            'metadata': {'title': 'Test Document', 'author': 'Test Author'}
        }
        
        result = await pipeline._process_pdf("/path/to/test.pdf", 1)
        
        assert result['pages_extracted'] == 2
        assert 'metadata' in result
        assert len(result['pages']) == 2
        
        # Verify database calls
        assert mock_db_session.add.call_count == 2  # Two pages added
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_images_stage(self, pipeline):
        """Test image processing stage"""
        # Mock image processor
        pipeline.image_processor.enhance_image.return_value = "enhanced_image_data"
        pipeline.image_processor.preprocess_for_ocr.return_value = "preprocessed_image_data"
        pipeline.image_processor.assess_image_quality.return_value = {"quality": 0.9}
        
        pdf_data = {
            'pages': [
                {'page_number': 1, 'image_path': '/path/to/page1.png'},
                {'page_number': 2, 'image_path': '/path/to/page2.png'}
            ]
        }
        
        result = await pipeline._process_images(pdf_data, 1)
        
        assert result['total_images'] == 2
        assert len(result['processed_images']) == 2
        
        # Verify image processor calls
        assert pipeline.image_processor.enhance_image.call_count == 2
        assert pipeline.image_processor.preprocess_for_ocr.call_count == 2
    
    @pytest.mark.asyncio
    async def test_extract_text_stage(self, pipeline, mock_db_session, sample_document):
        """Test OCR text extraction stage"""
        # Mock database query
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_document
        
        # Mock OCR engine
        pipeline.ocr_engine.extract_text.return_value = {
            'text': 'Extracted text from page',
            'confidence': 0.95,
            'character_positions': []
        }
        
        image_data = {
            'processed_images': [
                {'page_number': 1, 'preprocessed_image': 'image_data'},
                {'page_number': 2, 'preprocessed_image': 'image_data'}
            ]
        }
        
        result = await pipeline._extract_text(image_data, 1)
        
        assert result['total_pages'] == 2
        assert result['average_confidence'] == 0.95
        assert len(result['extracted_pages']) == 2
        
        # Verify OCR calls
        assert pipeline.ocr_engine.extract_text.call_count == 2
    
    @pytest.mark.asyncio
    async def test_analyze_text_stage(self, pipeline, mock_db_session, sample_document):
        """Test text analysis stage"""
        # Mock database query
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_document
        
        # Mock text analyzer
        pipeline.text_analyzer.analyze_character_distribution.return_value = {"char_analysis": "data"}
        pipeline.text_analyzer.analyze_word_patterns.return_value = {"word_analysis": "data"}
        pipeline.text_analyzer.perform_statistical_analysis.return_value = {"stats": "data"}
        pipeline.text_analyzer.detect_language.return_value = "english"
        
        ocr_data = {'extracted_pages': []}
        
        result = await pipeline._analyze_text(ocr_data, 1)
        
        assert result['document_id'] == 1
        assert result['text_length'] == len(sample_document.content)
        assert 'character_analysis' in result
        assert 'word_analysis' in result
        assert 'statistical_analysis' in result
        assert result['language_detection'] == "english"
    
    @pytest.mark.asyncio
    async def test_generate_grids_stage(self, pipeline):
        """Test grid generation stage"""
        # Mock grid generator
        mock_grid = Mock()
        mock_grid.id = 1
        pipeline.grid_generator.generate_grids.return_value = [mock_grid]
        pipeline.grid_generator.find_patterns_in_grid.return_value = [{"pattern": "data"}]
        
        text_data = {}
        
        result = await pipeline._generate_grids(text_data, 1)
        
        assert 'total_grids' in result
        assert 'grids_by_size' in result
        assert 'total_patterns' in result
        assert 'grids' in result
    
    @pytest.mark.asyncio
    async def test_detect_anomalies_stage(self, pipeline, mock_db_session):
        """Test anomaly detection stage"""
        # Mock database query for documents
        mock_db_session.query.return_value.all.return_value = [Mock(id=1), Mock(id=2)]
        
        # Mock anomaly detector - use MagicMock behavior (already set via patch)
        # The pipeline calls detect_text_anomalies, detect_pattern_anomalies, etc.
        pipeline.anomaly_detector = MagicMock()
        pipeline.anomaly_detector.detect_text_anomalies.return_value = [{"anomaly": "text"}]
        pipeline.anomaly_detector.detect_pattern_anomalies.return_value = [{"anomaly": "pattern"}]
        pipeline.anomaly_detector.detect_cipher_anomalies.return_value = [{"anomaly": "cipher"}]
        pipeline.anomaly_detector.detect_geometric_anomalies.return_value = []
        pipeline.anomaly_detector.generate_anomaly_report.return_value = {"report": "data"}
        
        pipeline_results = {}
        
        result = await pipeline._detect_anomalies(pipeline_results, 1)
        
        assert result['text_anomalies'] == 1
        assert result['pattern_anomalies'] == 1
        assert result['cipher_anomalies'] == 1
        assert result['geometric_anomalies'] == 0
        assert result['total_anomalies'] == 3
        assert 'anomaly_report' in result
    
    @pytest.mark.asyncio
    async def test_rank_patterns_stage(self, pipeline):
        """Test pattern ranking stage"""
        # Mock pattern ranker - use MagicMock
        mock_ranking_result = MagicMock()
        mock_ranking_result.ranked_patterns = [MagicMock(), MagicMock(), MagicMock()]
        
        pipeline.pattern_ranker = MagicMock()
        pipeline.pattern_ranker.rank_patterns_by_significance.return_value = mock_ranking_result
        pipeline.pattern_ranker.rank_anomalies_by_significance.return_value = mock_ranking_result
        
        pipeline_results = {}
        
        result = await pipeline._rank_patterns(pipeline_results, 1)
        
        assert result['ranked_patterns'] == 3
        assert result['ranked_anomalies'] == 3
        assert 'pattern_ranking' in result
        assert 'anomaly_ranking' in result
        assert len(result['top_patterns']) <= 10
        assert len(result['top_anomalies']) <= 10
    
    @pytest.mark.asyncio
    async def test_validate_results_stage(self, pipeline):
        """Test result validation stage"""
        # Mock cipher validator
        pipeline.cipher_validator.validate_cipher_explanation.return_value = {
            "valid": True,
            "confidence": 0.8
        }
        
        pipeline_results = {
            'cipher_detection': {
                'cipher_results': [
                    {'explanation': 'Test explanation', 'evidence': {'test': 'data'}}
                ]
            },
            'pattern_ranking': {
                'top_patterns': [
                    Mock(
                        pattern=Mock(id=1),
                        score=Mock(significance_score=0.9, reliability_measure=0.8),
                        evidence_strength=0.7
                    )
                ]
            }
        }
        
        result = await pipeline._validate_results(pipeline_results, 1)
        
        assert 'cipher_validations' in result
        assert 'pattern_validations' in result
        assert len(result['cipher_validations']) == 1
        assert len(result['pattern_validations']) == 1
    
    @pytest.mark.asyncio
    async def test_finalize_processing_stage(self, pipeline):
        """Test processing finalization stage"""
        pipeline_results = {
            'pattern_ranking': {'ranked_patterns': 5},
            'anomaly_detection': {'total_anomalies': 3}
        }
        
        result = await pipeline._finalize_processing(pipeline_results, 1)
        
        assert result['document_id'] == 1
        assert 'processing_time' in result
        assert result['stages_completed'] == pipeline.progress.completed_stages
        assert result['total_patterns_found'] == 5
        assert result['total_anomalies_found'] == 3
        assert 'performance_metrics' in result
    
    def test_generate_final_report(self, pipeline, sample_document):
        """Test final report generation"""
        pipeline_results = {
            'pattern_ranking': {
                'top_patterns': [
                    Mock(
                        score=Mock(pattern_type='cipher', significance_score=0.9),
                        recommendation_priority='critical'
                    )
                ]
            },
            'anomaly_detection': {
                'anomaly_report': {
                    'summary_statistics': {'high_severity_count': 2}
                }
            }
        }
        
        # Add some stage results
        pipeline.progress.stage_results = [
            StageResult(
                stage=ProcessingStage.TEXT_ANALYSIS,
                status=ProcessingStatus.COMPLETED,
                start_time=datetime.now(),
                duration=1.5
            )
        ]
        
        report = pipeline._generate_final_report(pipeline_results, sample_document)
        
        assert 'document_info' in report
        assert 'processing_summary' in report
        assert 'key_findings' in report
        assert 'detailed_results' in report
        assert 'performance_metrics' in report
        assert 'stage_results' in report
        
        # Check document info
        assert report['document_info']['id'] == sample_document.id
        assert report['document_info']['filename'] == sample_document.filename
        
        # Check key findings
        assert len(report['key_findings']) >= 1
        assert any(finding['type'] == 'pattern' for finding in report['key_findings'])
    
    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self, pipeline, mock_db_session, sample_document):
        """Test integration of multiple pipeline stages"""
        # Mock database operations
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_document
        mock_db_session.query.return_value.all.return_value = [sample_document]
        
        # Mock all service responses
        pipeline.pdf_processor.extract_pages.return_value = {
            'pages': [{'page_number': 1, 'image_path': '/test.png', 'width': 800, 'height': 600}],
            'metadata': {}
        }
        
        pipeline.image_processor.enhance_image.return_value = "enhanced"
        pipeline.image_processor.preprocess_for_ocr.return_value = "preprocessed"
        pipeline.image_processor.assess_image_quality.return_value = {"quality": 0.9}
        
        pipeline.ocr_engine.extract_text.return_value = {
            'text': 'Test text',
            'confidence': 0.9,
            'character_positions': []
        }
        
        pipeline.text_analyzer.analyze_character_distribution.return_value = {}
        pipeline.text_analyzer.analyze_word_patterns.return_value = {}
        pipeline.text_analyzer.perform_statistical_analysis.return_value = {}
        pipeline.text_analyzer.detect_language.return_value = "english"
        
        pipeline.grid_generator.generate_grids.return_value = []
        pipeline.geometric_analyzer.analyze_character_positions.return_value = []
        pipeline.cipher_detector.analyze_text.return_value = []
        
        # Use MagicMock for services with dynamic method access
        pipeline.relationship_analyzer = MagicMock()
        pipeline.relationship_analyzer.analyze_pattern_relationships.return_value = []
        pipeline.relationship_analyzer.find_analysis_correlations.return_value = {}
        
        pipeline.anomaly_detector = MagicMock()
        pipeline.anomaly_detector.detect_text_anomalies.return_value = []
        pipeline.anomaly_detector.detect_pattern_anomalies.return_value = []
        pipeline.anomaly_detector.detect_cipher_anomalies.return_value = []
        pipeline.anomaly_detector.detect_geometric_anomalies.return_value = []
        pipeline.anomaly_detector.generate_anomaly_report.return_value = {}
        
        mock_ranking = MagicMock()
        mock_ranking.ranked_patterns = []
        pipeline.pattern_ranker = MagicMock()
        pipeline.pattern_ranker.rank_patterns_by_significance.return_value = mock_ranking
        pipeline.pattern_ranker.rank_anomalies_by_significance.return_value = mock_ranking
        
        pipeline.visualizer = MagicMock()
        pipeline.visualizer.create_comprehensive_visualization.return_value = []
        
        # Process document
        result = await pipeline.process_document("/test/document.pdf", "test_document")
        
        # Verify result structure
        assert 'document_info' in result
        assert 'processing_summary' in result
        assert 'detailed_results' in result
        
        # Verify document was updated
        assert sample_document.processing_status == "completed"
        assert sample_document.completion_date is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, pipeline, mock_db_session, sample_document):
        """Test pipeline error handling and recovery"""
        # Mock database operations
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_document
        
        # Mock PDF processor to raise an error
        pipeline.pdf_processor.extract_pages.side_effect = Exception("PDF processing failed")
        
        # Process should handle the error gracefully
        with pytest.raises(Exception, match="PDF processing failed"):
            await pipeline.process_document("/test/document.pdf", "test_document")
        
        # Verify document status was updated to failed
        assert sample_document.processing_status == "failed"
        assert sample_document.error_message == "PDF processing failed"
    
    def test_processing_configuration_defaults(self):
        """Test default processing configuration values"""
        config = ProcessingConfiguration()
        
        assert config.ocr_confidence_threshold == 0.85
        assert config.ocr_language == "eng"
        assert config.enable_character_analysis is True
        assert config.enable_word_analysis is True
        assert config.grid_sizes == [10, 15, 20, 25, 30]
        assert config.max_grids_per_size == 5
        assert config.enable_geometric_analysis is True
        assert config.etymology_languages == ["hebrew", "greek", "latin"]
        assert config.cipher_confidence_threshold == 0.7
        assert config.anomaly_significance_threshold == 0.05
        assert config.max_concurrent_operations == 4
        assert config.timeout_per_stage_minutes == 30
        assert config.export_formats == ["json", "pdf", "csv"]
    
    def test_processing_stage_enum(self):
        """Test ProcessingStage enumeration"""
        stages = list(ProcessingStage)
        
        # Verify all expected stages are present
        expected_stages = [
            "initialization", "pdf_processing", "image_processing", "ocr_extraction",
            "text_analysis", "grid_generation", "geometric_analysis", "etymology_analysis",
            "cipher_detection", "bardcode_analysis", "relationship_analysis",
            "cross_document_analysis", "anomaly_detection", "pattern_ranking",
            "visualization", "validation", "finalization"
        ]
        
        stage_values = [stage.value for stage in stages]
        for expected_stage in expected_stages:
            assert expected_stage in stage_values
    
    def test_processing_status_enum(self):
        """Test ProcessingStatus enumeration"""
        statuses = list(ProcessingStatus)
        expected_statuses = ["pending", "running", "completed", "failed", "skipped"]
        
        status_values = [status.value for status in statuses]
        for expected_status in expected_statuses:
            assert expected_status in status_values
    
    def test_stage_result_dataclass(self):
        """Test StageResult dataclass"""
        stage_result = StageResult(
            stage=ProcessingStage.TEXT_ANALYSIS,
            status=ProcessingStatus.COMPLETED,
            start_time=datetime.now(),
            data={"test": "data"},
            warnings=["test warning"]
        )
        
        assert stage_result.stage == ProcessingStage.TEXT_ANALYSIS
        assert stage_result.status == ProcessingStatus.COMPLETED
        assert stage_result.data == {"test": "data"}
        assert stage_result.warnings == ["test warning"]
        assert stage_result.error is None
    
    def test_processing_progress_dataclass(self):
        """Test ProcessingProgress dataclass"""
        progress = ProcessingProgress(
            total_stages=10,
            completed_stages=5
        )
        
        assert progress.total_stages == 10
        assert progress.completed_stages == 5
        assert progress.current_stage is None
        assert progress.current_stage_progress == 0.0
        assert progress.overall_progress == 0.0
        assert len(progress.stage_results) == 0


class TestProcessingPipelineIntegration:
    """Integration tests for ProcessingPipeline"""
    
    @pytest.fixture
    def integration_pipeline(self):
        """Create pipeline for integration testing"""
        # This would use real services for integration testing
        # For now, we'll skip these tests in unit testing
        pytest.skip("Integration tests require real services and database")
    
    @pytest.mark.asyncio
    async def test_full_document_processing_workflow(self, integration_pipeline):
        """Test complete document processing workflow with real services"""
        # This would test the full workflow with real data
        pass
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, integration_pipeline):
        """Test performance monitoring and metrics collection"""
        # This would test performance monitoring with real processing
        pass
    
    @pytest.mark.asyncio
    async def test_error_recovery_mechanisms(self, integration_pipeline):
        """Test error recovery and retry mechanisms"""
        # This would test error recovery with real failure scenarios
        pass


if __name__ == "__main__":
    pytest.main([__file__])