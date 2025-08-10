"""
Tests for Cross-Document Pattern Database Service
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import numpy as np
from datetime import datetime

from app.services.cross_document_pattern_database import (
    CrossDocumentPatternDatabase, PatternSearchResult, ClusteringResult
)
from app.models.cross_document_models import (
    CrossDocumentPattern, CrossPatternInstance, PatternRelationship,
    DocumentCluster, SharedConstruction, PatternSimilarityIndex
)
from app.models.database_models import Document


class TestCrossDocumentPatternDatabase:
    """Test cases for CrossDocumentPatternDatabase"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_db = Mock()
        self.db_service = CrossDocumentPatternDatabase(db_session=self.mock_db)
        
        # Sample pattern data
        self.sample_pattern_data = {
            'pattern_type': 'cipher',
            'pattern_subtype': 'caesar_shift',
            'pattern_name': 'Caesar Shift Pattern',
            'description': 'A Caesar cipher with shift of 3',
            'pattern_signature': {
                'shift_amount': 3,
                'alphabet': 'latin',
                'direction': 'forward'
            },
            'pattern_parameters': {
                'key': 3,
                'method': 'caesar'
            },
            'pattern_complexity': 0.2,
            'discovery_method': 'automated',
            'instances': [
                {
                    'document_id': 1,
                    'page_numbers': [1, 2],
                    'coordinates': {'x': 100, 'y': 200},
                    'instance_data': {'text': 'KHOOR ZRUOG'},
                    'confidence': 0.85,
                    'quality_score': 0.9
                }
            ]
        }
        
        # Sample cross-document pattern (using Mock to avoid SQLAlchemy initialization issues)
        self.sample_cross_pattern = Mock()
        self.sample_cross_pattern.id = 1
        self.sample_cross_pattern.pattern_hash = 'abc123'
        self.sample_cross_pattern.pattern_type = 'cipher'
        self.sample_cross_pattern.pattern_subtype = 'caesar_shift'
        self.sample_cross_pattern.pattern_name = 'Test Pattern'
        self.sample_cross_pattern.description = 'Test description'
        self.sample_cross_pattern.pattern_signature = {'shift': 3}
        self.sample_cross_pattern.document_count = 2
        self.sample_cross_pattern.total_occurrences = 5
        self.sample_cross_pattern.average_confidence = 0.8
        self.sample_cross_pattern.significance_score = 0.7
    
    def test_database_initialization(self):
        """Test database service initialization"""
        assert self.db_service.db == self.mock_db
        assert self.db_service.similarity_threshold == 0.7
        assert self.db_service.clustering_threshold == 0.6
        assert isinstance(self.db_service.similarity_cache, dict)
    
    def test_store_cross_document_pattern_new(self):
        """Test storing a new cross-document pattern"""
        # Mock database queries
        self.mock_db.query.return_value.filter.return_value.first.return_value = None
        self.mock_db.add = Mock()
        self.mock_db.commit = Mock()
        self.mock_db.refresh = Mock()
        
        # Mock the pattern creation
        mock_pattern = Mock()
        mock_pattern.id = 1
        
        with patch.object(self.db_service, '_store_pattern_instances') as mock_store_instances, \
             patch.object(self.db_service, '_update_pattern_statistics') as mock_update_stats:
            
            # Mock the database add operation to set the pattern
            def mock_add_side_effect(pattern):
                pattern.id = 1
            self.mock_db.add.side_effect = mock_add_side_effect
            self.mock_db.refresh.side_effect = lambda p: setattr(p, 'id', 1)
            
            result = self.db_service.store_cross_document_pattern(self.sample_pattern_data)
            
            # Verify database operations
            self.mock_db.add.assert_called_once()
            self.mock_db.commit.assert_called()
            mock_store_instances.assert_called_once()
            mock_update_stats.assert_called_once()
    
    def test_store_cross_document_pattern_existing(self):
        """Test updating an existing cross-document pattern"""
        # Mock existing pattern
        existing_pattern = Mock()
        existing_pattern.id = 1
        self.mock_db.query.return_value.filter.return_value.first.return_value = existing_pattern
        
        with patch.object(self.db_service, '_update_existing_pattern') as mock_update:
            mock_update.return_value = existing_pattern
            
            result = self.db_service.store_cross_document_pattern(self.sample_pattern_data)
            
            mock_update.assert_called_once_with(existing_pattern, self.sample_pattern_data)
            assert result == existing_pattern
    
    def test_search_similar_patterns(self):
        """Test searching for similar patterns"""
        # Mock database query
        mock_candidates = [self.sample_cross_pattern]
        self.mock_db.query.return_value.filter.return_value.all.return_value = mock_candidates
        
        # Mock pattern instances
        mock_instances = [Mock()]
        self.mock_db.query.return_value.filter.return_value.all.side_effect = [mock_candidates, mock_instances]
        
        with patch.object(self.db_service, '_calculate_pattern_similarity') as mock_calc_sim, \
             patch.object(self.db_service, '_calculate_relevance_score') as mock_calc_rel:
            
            mock_calc_sim.return_value = 0.8
            mock_calc_rel.return_value = 0.75
            
            query_pattern = {
                'pattern_type': 'cipher',
                'pattern_signature': {'shift': 3}
            }
            
            results = self.db_service.search_similar_patterns(query_pattern, limit=5)
            
            assert len(results) == 1
            assert isinstance(results[0], PatternSearchResult)
            assert results[0].similarity_score == 0.8
            assert results[0].relevance_score == 0.75
    
    def test_search_similar_patterns_no_candidates(self):
        """Test searching when no similar patterns exist"""
        self.mock_db.query.return_value.filter.return_value.all.return_value = []
        
        query_pattern = {'pattern_type': 'nonexistent'}
        results = self.db_service.search_similar_patterns(query_pattern)
        
        assert results == []
    
    def test_identify_shared_constructions(self):
        """Test identifying shared constructions across documents"""
        document_ids = [1, 2, 3]
        
        # Mock document patterns
        mock_patterns = [
            Mock(pattern_hash='hash1'),
            Mock(pattern_hash='hash2'),
            Mock(pattern_hash='hash1')  # Shared pattern
        ]
        
        with patch.object(self.db_service, '_get_document_patterns') as mock_get_patterns, \
             patch.object(self.db_service, '_create_shared_construction') as mock_create_construction:
            
            mock_get_patterns.side_effect = [
                [mock_patterns[0], mock_patterns[1]],  # Doc 1
                [mock_patterns[2]],  # Doc 2
                [mock_patterns[0]]   # Doc 3 (shares hash1)
            ]
            
            mock_construction = Mock()
            mock_create_construction.return_value = mock_construction
            
            results = self.db_service.identify_shared_constructions(document_ids)
            
            # Should find one shared construction (hash1 appears in docs 1 and 3)
            mock_create_construction.assert_called_once()
            assert len(results) == 1
    
    def test_identify_shared_constructions_insufficient_docs(self):
        """Test shared constructions with insufficient documents"""
        result = self.db_service.identify_shared_constructions([1])
        assert result == []
    
    def test_cluster_documents_by_patterns(self):
        """Test clustering documents by their patterns"""
        document_ids = [1, 2, 3, 4]
        
        # Mock document-pattern matrix
        mock_matrix = np.array([
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1]
        ])
        
        with patch.object(self.db_service, '_build_document_pattern_matrix') as mock_build_matrix, \
             patch.object(self.db_service, '_hierarchical_clustering') as mock_clustering, \
             patch.object(self.db_service, '_store_document_cluster') as mock_store_cluster:
            
            mock_build_matrix.return_value = mock_matrix
            
            # Mock clustering results
            mock_clusters = [
                {'cluster_id': 1, 'document_ids': [1, 2], 'document_count': 2},
                {'cluster_id': 2, 'document_ids': [3, 4], 'document_count': 2}
            ]
            mock_assignments = {1: 1, 2: 1, 3: 2, 4: 2}
            mock_metrics = {'n_clusters': 2, 'silhouette_score': 0.8}
            
            mock_clustering.return_value = (mock_clusters, mock_assignments, mock_metrics)
            
            # Mock stored clusters
            mock_stored_clusters = [Mock(), Mock()]
            mock_store_cluster.side_effect = mock_stored_clusters
            
            result = self.db_service.cluster_documents_by_patterns(document_ids)
            
            assert isinstance(result, ClusteringResult)
            assert len(result.clusters) == 2
            assert result.cluster_assignments == mock_assignments
            assert result.quality_metrics == mock_metrics
    
    def test_cluster_documents_insufficient_docs(self):
        """Test clustering with insufficient documents"""
        result = self.db_service.cluster_documents_by_patterns([1])
        
        assert isinstance(result, ClusteringResult)
        assert result.clusters == []
        assert result.cluster_assignments == {}
        assert result.quality_metrics == {}
    
    def test_build_pattern_similarity_index(self):
        """Test building the pattern similarity index"""
        # Mock patterns
        mock_patterns = [Mock(id=i) for i in range(1, 6)]
        self.mock_db.query.return_value.all.return_value = mock_patterns
        
        with patch.object(self.db_service, '_process_similarity_batch') as mock_process_batch:
            self.db_service.build_pattern_similarity_index(batch_size=2)
            
            # Should process 3 batches (2, 2, 1 patterns each)
            assert mock_process_batch.call_count == 3
            assert self.mock_db.commit.call_count == 3
    
    def test_get_pattern_relationships(self):
        """Test getting pattern relationships"""
        pattern_id = 1
        mock_relationships = [Mock(), Mock()]
        self.mock_db.query.return_value.filter.return_value.all.return_value = mock_relationships
        
        results = self.db_service.get_pattern_relationships(pattern_id)
        
        assert results == mock_relationships
        self.mock_db.query.assert_called_with(PatternRelationship)
    
    def test_analyze_pattern_evolution(self):
        """Test analyzing pattern evolution"""
        pattern_id = 1
        
        # Mock pattern and instances
        mock_pattern = Mock()
        mock_instances = [Mock(), Mock(), Mock()]
        
        self.mock_db.query.return_value.get.return_value = mock_pattern
        self.mock_db.query.return_value.filter.return_value.all.return_value = mock_instances
        
        with patch.object(self.db_service, '_analyze_confidence_trend') as mock_trend, \
             patch.object(self.db_service, '_analyze_parameter_evolution') as mock_params, \
             patch.object(self.db_service, '_analyze_temporal_distribution') as mock_temporal, \
             patch.object(self.db_service, '_calculate_evolution_quality_metrics') as mock_quality:
            
            mock_trend.return_value = {'mean': 0.8}
            mock_params.return_value = {'stability': 0.9}
            mock_temporal.return_value = {'distribution': 'uniform'}
            mock_quality.return_value = {'completeness': 1.0}
            
            result = self.db_service.analyze_pattern_evolution(pattern_id)
            
            assert result['pattern_id'] == pattern_id
            assert result['total_instances'] == 3
            assert 'confidence_trend' in result
            assert 'parameter_evolution' in result
    
    def test_analyze_pattern_evolution_no_pattern(self):
        """Test pattern evolution analysis with non-existent pattern"""
        self.mock_db.query.return_value.get.return_value = None
        
        result = self.db_service.analyze_pattern_evolution(999)
        assert result == {}
    
    def test_get_document_pattern_profile(self):
        """Test getting document pattern profile"""
        document_id = 1
        
        # Mock patterns and instances
        mock_patterns = [Mock(document_count=2), Mock(document_count=1)]
        mock_instances = [Mock(), Mock()]
        
        with patch.object(self.db_service, '_get_document_patterns') as mock_get_patterns, \
             patch.object(self.db_service, '_analyze_pattern_types') as mock_types, \
             patch.object(self.db_service, '_analyze_complexity_distribution') as mock_complexity, \
             patch.object(self.db_service, '_analyze_significance_scores') as mock_significance, \
             patch.object(self.db_service, '_calculate_document_quality_metrics') as mock_quality:
            
            mock_get_patterns.return_value = mock_patterns
            self.mock_db.query.return_value.filter.return_value.all.return_value = mock_instances
            
            mock_types.return_value = {'cipher': 2}
            mock_complexity.return_value = {'mean': 0.5}
            mock_significance.return_value = {'mean': 0.7}
            mock_quality.return_value = {'average_confidence': 0.8}
            
            result = self.db_service.get_document_pattern_profile(document_id)
            
            assert result['document_id'] == document_id
            assert result['total_patterns'] == 2
            assert result['shared_pattern_count'] == 1
            assert result['unique_pattern_count'] == 1
            assert 'pattern_types' in result
    
    def test_generate_pattern_hash(self):
        """Test pattern hash generation"""
        pattern_data = {
            'pattern_type': 'cipher',
            'pattern_subtype': 'caesar',
            'pattern_signature': {'shift': 3},
            'pattern_parameters': {'key': 3}
        }
        
        hash1 = self.db_service._generate_pattern_hash(pattern_data)
        hash2 = self.db_service._generate_pattern_hash(pattern_data)
        
        # Same data should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length
        
        # Different data should produce different hash
        pattern_data['pattern_signature']['shift'] = 5
        hash3 = self.db_service._generate_pattern_hash(pattern_data)
        assert hash1 != hash3
    
    def test_calculate_pattern_similarity(self):
        """Test pattern similarity calculation"""
        signature1 = {
            'shift_amount': 3,
            'alphabet_size': 26,
            'direction': 'forward'
        }
        
        signature2 = {
            'shift_amount': 3,
            'alphabet_size': 26,
            'direction': 'forward'
        }
        
        similarity = self.db_service._calculate_pattern_similarity(signature1, signature2)
        assert similarity > 0.9  # Should be very similar
        
        # Test with different signatures
        signature3 = {
            'shift_amount': 5,
            'alphabet_size': 26,
            'direction': 'backward'
        }
        
        similarity2 = self.db_service._calculate_pattern_similarity(signature1, signature3)
        assert similarity2 < similarity  # Should be less similar
    
    def test_calculate_pattern_similarity_empty(self):
        """Test pattern similarity with empty signatures"""
        similarity = self.db_service._calculate_pattern_similarity({}, {})
        assert similarity == 0.0
        
        similarity2 = self.db_service._calculate_pattern_similarity({'a': 1}, {})
        assert similarity2 == 0.0
    
    def test_signature_to_features(self):
        """Test converting pattern signature to features"""
        signature = {
            'numeric_value': 3.14,
            'string_value': 'test',
            'list_value': [1, 2, 3, 4, 5],
            'nested_dict': {'inner': 'value'}
        }
        
        features = self.db_service._signature_to_features(signature)
        
        assert 'num_numeric_value' in features
        assert features['num_numeric_value'] == 3.14
        assert 'str_string_value' in features
        assert 'list_list_value_len' in features
        assert features['list_list_value_len'] == 5
        assert 'list_list_value_mean' in features
        assert features['list_list_value_mean'] == 3.0
    
    def test_calculate_relevance_score(self):
        """Test relevance score calculation"""
        pattern = Mock()
        pattern.significance_score = 0.8
        pattern.document_count = 5
        pattern.first_discovered = datetime.now()
        
        relevance = self.db_service._calculate_relevance_score(pattern, 0.9)
        
        assert 0.0 <= relevance <= 1.0
        assert isinstance(relevance, float)
    
    def test_calculate_consistency_score(self):
        """Test consistency score calculation"""
        # High consistency (low variation)
        instances1 = [Mock(confidence=0.8), Mock(confidence=0.82), Mock(confidence=0.78)]
        consistency1 = self.db_service._calculate_consistency_score(instances1)
        
        # Low consistency (high variation)
        instances2 = [Mock(confidence=0.2), Mock(confidence=0.9), Mock(confidence=0.5)]
        consistency2 = self.db_service._calculate_consistency_score(instances2)
        
        assert consistency1 > consistency2
        assert 0.0 <= consistency1 <= 1.0
        assert 0.0 <= consistency2 <= 1.0
    
    def test_calculate_consistency_score_edge_cases(self):
        """Test consistency score edge cases"""
        # Empty instances
        assert self.db_service._calculate_consistency_score([]) == 0.0
        
        # Single instance
        assert self.db_service._calculate_consistency_score([Mock(confidence=0.8)]) == 1.0
        
        # Zero confidence
        instances = [Mock(confidence=0.0), Mock(confidence=0.0)]
        assert self.db_service._calculate_consistency_score(instances) == 0.0
    
    def test_calculate_rarity_score(self):
        """Test rarity score calculation"""
        # Rare pattern (few documents, few occurrences)
        rarity1 = self.db_service._calculate_rarity_score(2, 5)
        
        # Common pattern (many documents, many occurrences)
        rarity2 = self.db_service._calculate_rarity_score(50, 500)
        
        assert rarity1 > rarity2
        assert 0.0 <= rarity1 <= 1.0
        assert 0.0 <= rarity2 <= 1.0
    
    def test_calculate_significance_score(self):
        """Test significance score calculation"""
        pattern = Mock()
        pattern.rarity_score = 0.8
        pattern.consistency_score = 0.7
        pattern.average_confidence = 0.9
        pattern.pattern_complexity = 0.6
        pattern.document_count = 3
        
        significance = self.db_service._calculate_significance_score(pattern)
        
        assert 0.0 <= significance <= 1.0
        assert isinstance(significance, float)
    
    def test_build_document_pattern_matrix(self):
        """Test building document-pattern matrix"""
        document_ids = [1, 2, 3]
        
        # Mock patterns for each document
        mock_patterns = {
            1: [Mock(id=1), Mock(id=2)],
            2: [Mock(id=1), Mock(id=3)],
            3: [Mock(id=2), Mock(id=3)]
        }
        
        with patch.object(self.db_service, '_get_document_patterns') as mock_get_patterns:
            mock_get_patterns.side_effect = lambda doc_id: mock_patterns[doc_id]
            
            # Mock pattern significance scores
            mock_pattern_objects = {
                1: Mock(significance_score=0.8),
                2: Mock(significance_score=0.6),
                3: Mock(significance_score=0.9)
            }
            self.mock_db.query.return_value.get.side_effect = lambda pid: mock_pattern_objects[pid]
            
            matrix = self.db_service._build_document_pattern_matrix(document_ids)
            
            assert matrix.shape == (3, 3)  # 3 documents, 3 unique patterns
            assert matrix.dtype == np.float64
    
    def test_hierarchical_clustering(self):
        """Test hierarchical clustering"""
        # Simple test matrix
        matrix = np.array([
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1]
        ])
        document_ids = [1, 2, 3]
        
        clusters, assignments, metrics = self.db_service._hierarchical_clustering(matrix, document_ids)
        
        assert len(clusters) > 0
        assert len(assignments) == len(document_ids)
        assert 'n_clusters' in metrics
        assert all(doc_id in assignments for doc_id in document_ids)
    
    def test_hierarchical_clustering_insufficient_data(self):
        """Test hierarchical clustering with insufficient data"""
        matrix = np.array([[1]])
        document_ids = [1]
        
        clusters, assignments, metrics = self.db_service._hierarchical_clustering(matrix, document_ids)
        
        assert clusters == []
        assert assignments == {}
        assert metrics == {}
    
    def test_error_handling(self):
        """Test error handling in various methods"""
        # Test with database errors
        self.mock_db.query.side_effect = Exception("Database error")
        
        # These should not raise exceptions but return empty/default results
        result1 = self.db_service.search_similar_patterns({'pattern_type': 'test'})
        assert result1 == []
        
        result2 = self.db_service.get_pattern_relationships(1)
        assert result2 == []
        
        result3 = self.db_service.analyze_pattern_evolution(1)
        assert result3 == {}
    
    def test_integration_with_cross_document_analyzer(self):
        """Test integration with existing cross-document analyzer"""
        # This test ensures the pattern database works with the cross-document analyzer
        from app.services.cross_document_analyzer import CrossDocumentAnalyzer
        
        # Mock some integration scenarios
        analyzer = CrossDocumentAnalyzer()
        
        # Test that pattern database can store results from analyzer
        mock_correlation_result = {
            'pattern_type': 'cipher',
            'pattern_subtype': 'correlation',
            'description': 'Cross-document correlation pattern',
            'pattern_signature': {'correlation_score': 0.85},
            'instances': []
        }
        
        # This should work without errors
        with patch.object(self.db_service, '_generate_pattern_hash') as mock_hash, \
             patch.object(self.db_service, '_store_pattern_instances') as mock_store_instances, \
             patch.object(self.db_service, '_update_pattern_statistics') as mock_update_stats:
            
            mock_hash.return_value = 'test_hash'
            self.mock_db.query.return_value.filter.return_value.first.return_value = None
            
            # Mock the database add operation
            def mock_add_side_effect(pattern):
                pattern.id = 1
            self.mock_db.add.side_effect = mock_add_side_effect
            self.mock_db.refresh.side_effect = lambda p: setattr(p, 'id', 1)
            
            # Should not raise exception
            try:
                result = self.db_service.store_cross_document_pattern(mock_correlation_result)
                # Test passes if no exception is raised
                assert True
            except Exception as e:
                pytest.fail(f"Integration test failed: {str(e)}")
    
    def test_performance_considerations(self):
        """Test performance-related aspects"""
        # Test similarity cache
        assert hasattr(self.db_service, 'similarity_cache')
        assert isinstance(self.db_service.similarity_cache, dict)
        
        # Test batch processing parameters
        assert hasattr(self.db_service, 'max_pattern_cache')
        assert self.db_service.max_pattern_cache > 0
        
        # Test threshold values are reasonable
        assert 0 < self.db_service.similarity_threshold < 1
        assert 0 < self.db_service.clustering_threshold < 1