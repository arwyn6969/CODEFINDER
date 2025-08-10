"""
Tests for the SearchService
"""
import pytest
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from app.services.search_service import SearchService
from app.services.etymology_engine import Language
from app.models.database_models import Document, Page, Word, Pattern, GeometricMeasurement


class TestSearchService:
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        return Mock(spec=Session)
    
    @pytest.fixture
    def search_service(self, mock_db):
        """Create a SearchService instance with mocked dependencies"""
        with patch('app.services.search_service.DocumentRepository'), \
             patch('app.services.search_service.PageRepository'), \
             patch('app.services.search_service.CharacterRepository'), \
             patch('app.services.search_service.WordRepository'), \
             patch('app.services.search_service.PatternRepository'), \
             patch('app.services.search_service.GridRepository'), \
             patch('app.services.search_service.GeometricRepository'), \
             patch('app.services.search_service.EtymologyRepository'):
            return SearchService(mock_db)
    
    def test_search_service_initialization(self, search_service, mock_db):
        """Test that SearchService initializes correctly"""
        assert search_service.db == mock_db
        assert hasattr(search_service, 'document_repo')
        assert hasattr(search_service, 'word_repo')
        assert hasattr(search_service, 'pattern_repo')
    
    def test_search_text_basic(self, search_service, mock_db):
        """Test basic text search functionality"""
        # Mock word objects
        mock_word = Mock()
        mock_word.id = 1
        mock_word.text = "test"
        mock_word.page_id = 1
        mock_word.x = 100.0
        mock_word.y = 200.0
        mock_word.confidence = 0.95
        
        mock_page = Mock()
        mock_page.page_number = 1
        mock_page.document_id = 1
        
        # Mock database queries
        mock_query = Mock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = [mock_word]
        
        mock_db.query.return_value = mock_query
        mock_db.query.return_value.filter.return_value.first.return_value = mock_page
        
        # Execute search
        results = search_service.search_text("test")
        
        # Verify results
        assert results['query'] == "test"
        assert results['total_matches'] >= 0
        assert 'word_matches' in results
        assert 'pattern_matches' in results
    
    def test_search_text_case_sensitive(self, search_service, mock_db):
        """Test case-sensitive text search"""
        mock_query = Mock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []
        
        mock_db.query.return_value = mock_query
        
        results = search_service.search_text("Test", case_sensitive=True)
        
        assert results['case_sensitive'] == True
        assert results['query'] == "Test"
    
    def test_search_text_whole_word(self, search_service, mock_db):
        """Test whole word text search"""
        mock_query = Mock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []
        
        mock_db.query.return_value = mock_query
        
        results = search_service.search_text("word", whole_word=True)
        
        assert results['whole_word'] == True
        assert results['query'] == "word"
    
    def test_search_patterns(self, search_service, mock_db):
        """Test pattern search functionality"""
        # Mock pattern object
        mock_pattern = Mock()
        mock_pattern.id = 1
        mock_pattern.pattern_type = "geometric"
        mock_pattern.description = "Triangle pattern"
        mock_pattern.document_id = 1
        mock_pattern.confidence = 0.85
        mock_pattern.significance_score = 0.9
        mock_pattern.page_numbers = [1, 2]
        mock_pattern.coordinates = {"x": 100, "y": 200}
        mock_pattern.pattern_data = {"angle": 45.0}
        
        # Mock database query
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = [mock_pattern]
        
        mock_db.query.return_value = mock_query
        
        # Execute search
        criteria = {
            'pattern_type': 'geometric',
            'min_confidence': 0.8,
            'min_significance': 0.7
        }
        results = search_service.search_patterns(criteria)
        
        # Verify results
        assert len(results) == 1
        assert results[0]['pattern_type'] == 'geometric'
        assert results[0]['confidence'] == 0.85
    
    def test_search_geometric_relationships(self, search_service, mock_db):
        """Test geometric relationship search"""
        # Mock geometric measurement
        mock_measurement = Mock()
        mock_measurement.id = 1
        mock_measurement.measurement_type = "angle"
        mock_measurement.measurement_value = 45.0
        mock_measurement.measurement_unit = "degrees"
        mock_measurement.document_id = 1
        mock_measurement.page_id = 1
        mock_measurement.significance_score = 0.8
        mock_measurement.is_significant = True
        mock_measurement.coordinates = {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
        mock_measurement.description = "Right angle measurement"
        
        # Mock database query
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = [mock_measurement]
        
        mock_db.query.return_value = mock_query
        
        # Execute search
        criteria = {
            'measurement_type': 'angle',
            'min_value': 30.0,
            'max_value': 60.0,
            'is_significant': True
        }
        results = search_service.search_geometric_relationships(criteria)
        
        # Verify results
        assert len(results) == 1
        assert results[0]['measurement_type'] == 'angle'
        assert results[0]['measurement_value'] == 45.0
        assert results[0]['is_significant'] == True
    
    @patch('app.services.search_service.EtymologyEngine')
    def test_search_etymology(self, mock_etymology_engine, search_service):
        """Test etymology search functionality"""
        # Mock etymology engine
        mock_engine_instance = Mock()
        mock_analysis = Mock()
        mock_analysis.confidence = 0.8
        mock_analysis.word = "test"
        mock_analysis.root_words = []
        mock_analysis.definitions = ["a trial or examination"]
        
        mock_engine_instance.analyze_word.return_value = mock_analysis
        mock_etymology_engine.return_value = mock_engine_instance
        
        # Mock repository
        search_service.etymology_repo.get_etymology.return_value = None
        
        # Execute search
        results = search_service.search_etymology("test", [Language.ENGLISH])
        
        # Verify results
        assert results['query'] == "test"
        assert 'results' in results
        assert results['total_matches'] >= 0
    
    def test_search_cross_document_patterns(self, search_service):
        """Test cross-document pattern search"""
        # Mock pattern repository
        mock_pattern1 = Mock()
        mock_pattern1.id = 1
        mock_pattern1.pattern_type = "geometric"
        mock_pattern1.description = "Triangle pattern with angle 45 degrees"
        mock_pattern1.significance_score = 0.8
        
        mock_pattern2 = Mock()
        mock_pattern2.id = 2
        mock_pattern2.pattern_type = "geometric"
        mock_pattern2.description = "Triangle pattern with angle 60 degrees"
        mock_pattern2.significance_score = 0.9
        
        search_service.pattern_repo.get_significant_patterns.side_effect = [
            [mock_pattern1],  # Document 1 patterns
            [mock_pattern2]   # Document 2 patterns
        ]
        
        # Execute search
        results = search_service.search_cross_document_patterns([1, 2], 0.7)
        
        # Verify results
        assert results['document_ids'] == [1, 2]
        assert 'common_patterns' in results
        assert 'unique_patterns' in results
        assert 'correlation_score' in results
    
    def test_complex_query(self, search_service, mock_db):
        """Test complex multi-dimensional query"""
        # Mock database queries
        mock_query = Mock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.all.return_value = []
        
        mock_db.query.return_value = mock_query
        
        # Mock repository methods
        search_service.pattern_repo.get_significant_patterns.return_value = []
        search_service.etymology_repo.get_etymology.return_value = None
        
        # Execute complex query
        query_spec = {
            'text': {
                'query': 'test',
                'case_sensitive': False
            },
            'patterns': {
                'pattern_type': 'geometric',
                'min_confidence': 0.8
            }
        }
        
        results = search_service.complex_query(query_spec)
        
        # Verify results
        assert results['query_spec'] == query_spec
        assert 'results' in results
        assert 'total_matches' in results
    
    def test_get_search_suggestions(self, search_service, mock_db):
        """Test search suggestion functionality"""
        # Mock word suggestions
        mock_word1 = Mock()
        mock_word1.text = "test"
        mock_word2 = Mock()
        mock_word2.text = "testing"
        
        mock_pattern1 = Mock()
        mock_pattern1.pattern_type = "geometric"
        
        # Mock database queries
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.distinct.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.side_effect = [[mock_word1, mock_word2], [mock_pattern1]]
        
        mock_db.query.return_value = mock_query
        
        # Execute search
        suggestions = search_service.get_search_suggestions("te", "all")
        
        # Verify results
        assert isinstance(suggestions, list)
        # Should contain unique suggestions
        assert len(set(suggestions)) == len(suggestions)
    
    def test_simplify_description(self, search_service):
        """Test description simplification for pattern matching"""
        description = "Triangle pattern with angle 45.67 degrees at coordinates (123, 456)"
        simplified = search_service._simplify_description(description)
        
        # Should remove numbers and special characters
        assert "45.67" not in simplified
        assert "123" not in simplified
        assert "(" not in simplified
        assert simplified.islower()
        assert "triangle pattern with angle" in simplified


if __name__ == "__main__":
    pytest.main([__file__])