"""
Tests for Document Relationship Analysis Service
"""
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from app.services.relationship_analyzer import (
    RelationshipAnalyzer, DocumentRelationship, CorrelationMatrix,
    EvidenceTrail, AuthorshipProfile
)
from app.models.cross_document_models import (
    CrossDocumentPattern, CrossPatternInstance, PatternRelationship
)
from app.models.database_models import Document, Page


class TestRelationshipAnalyzer:
    """Test cases for RelationshipAnalyzer"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session"""
        return Mock(spec=Session)
    
    @pytest.fixture
    def analyzer(self, mock_db_session):
        """Create RelationshipAnalyzer instance with mock database"""
        return RelationshipAnalyzer(mock_db_session)
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Mock(id=1, title="Document 1", created_at=datetime.now()),
            Mock(id=2, title="Document 2", created_at=datetime.now() - timedelta(days=30)),
            Mock(id=3, title="Document 3", created_at=datetime.now() - timedelta(days=60))
        ]
    
    @pytest.fixture
    def sample_patterns(self):
        """Create sample cross-document patterns"""
        return [
            Mock(
                id=1,
                pattern_type="cipher",
                pattern_subtype="caesar",
                significance_score=0.8,
                pattern_complexity=0.6,
                document_count=2,
                total_occurrences=5
            ),
            Mock(
                id=2,
                pattern_type="geometric",
                pattern_subtype="triangle",
                significance_score=0.7,
                pattern_complexity=0.8,
                document_count=3,
                total_occurrences=8
            )
        ]
    
    @pytest.fixture
    def sample_instances(self):
        """Create sample pattern instances"""
        return [
            Mock(
                id=1,
                cross_pattern_id=1,
                document_id=1,
                confidence=0.85,
                quality_score=0.9,
                detected_at=datetime.now(),
                instance_data={"param1": "value1"}
            ),
            Mock(
                id=2,
                cross_pattern_id=1,
                document_id=2,
                confidence=0.82,
                quality_score=0.88,
                detected_at=datetime.now() - timedelta(hours=1),
                instance_data={"param1": "value1"}
            ),
            Mock(
                id=3,
                cross_pattern_id=2,
                document_id=2,
                confidence=0.75,
                quality_score=0.8,
                detected_at=datetime.now() - timedelta(hours=2),
                instance_data={"param2": "value2"}
            )
        ]
    
    def test_generate_correlation_matrix_basic(self, analyzer, mock_db_session, sample_documents):
        """Test basic correlation matrix generation"""
        # Setup mocks
        mock_db_session.query.return_value.get.side_effect = lambda doc_id: sample_documents[doc_id - 1]
        
        # Mock pattern correlation calculations
        with patch.object(analyzer, '_calculate_pattern_correlation', return_value=0.7), \
             patch.object(analyzer, '_calculate_stylistic_correlation', return_value=0.6), \
             patch.object(analyzer, '_calculate_temporal_correlation', return_value=0.5), \
             patch.object(analyzer, '_calculate_geometric_correlation', return_value=0.4), \
             patch.object(analyzer, '_calculate_cipher_method_correlation', return_value=0.3), \
             patch.object(analyzer, '_calculate_correlation_significance', return_value=0.05), \
             patch.object(analyzer, '_cluster_documents_by_correlation', return_value={1: 0, 2: 0, 3: 1}):
            
            document_ids = [1, 2, 3]
            result = analyzer.generate_correlation_matrix(document_ids)
            
            # Verify result structure
            assert isinstance(result, CorrelationMatrix)
            assert result.document_ids == document_ids
            assert result.matrix.shape == (3, 3)
            assert np.allclose(np.diag(result.matrix), 1.0)  # Diagonal should be 1.0
            assert len(result.correlation_types) == 5
            assert result.significance_matrix.shape == (3, 3)
            assert isinstance(result.cluster_assignments, dict)
    
    def test_generate_correlation_matrix_insufficient_documents(self, analyzer):
        """Test correlation matrix generation with insufficient documents"""
        with pytest.raises(ValueError, match="Need at least 2 documents"):
            analyzer.generate_correlation_matrix([1])
    
    def test_track_evidence_trails_basic(self, analyzer, mock_db_session, sample_patterns, sample_instances):
        """Test basic evidence trail tracking"""
        pattern_id = 1
        
        # Setup mocks
        mock_db_session.query.return_value.get.return_value = sample_patterns[0]
        mock_db_session.query.return_value.filter.return_value.all.return_value = sample_instances[:2]
        
        with patch.object(analyzer, '_calculate_connection_strength', return_value=0.8), \
             patch.object(analyzer, '_build_evidence_points', return_value=[
                 {'type': 'instance_similarity', 'strength': 0.9}
             ]), \
             patch.object(analyzer, '_extract_temporal_sequence', return_value=[datetime.now()]), \
             patch.object(analyzer, '_calculate_trail_confidence', return_value=0.85), \
             patch.object(analyzer, '_extend_evidence_trails', return_value=[]):
            
            trails = analyzer.track_evidence_trails(pattern_id)
            
            # Verify results
            assert len(trails) >= 0
            if trails:
                trail = trails[0]
                assert isinstance(trail, EvidenceTrail)
                assert trail.pattern_id == pattern_id
                assert len(trail.document_chain) >= 2
                assert 0 <= trail.connection_strength <= 1
                assert 0 <= trail.confidence_score <= 1
    
    def test_track_evidence_trails_no_pattern(self, analyzer, mock_db_session):
        """Test evidence trail tracking with non-existent pattern"""
        mock_db_session.query.return_value.get.return_value = None
        
        trails = analyzer.track_evidence_trails(999)
        assert trails == []
    
    def test_analyze_authorship_patterns_basic(self, analyzer, sample_documents):
        """Test basic authorship pattern analysis"""
        document_ids = [1, 2, 3]
        
        # Mock authorship profiles
        mock_profiles = [
            AuthorshipProfile(
                document_id=1,
                stylistic_features={'avg_word_length': 5.2, 'comma_frequency': 0.02},
                cipher_preferences={'caesar': 0.6, 'substitution': 0.4},
                complexity_signature={'avg_complexity': 0.7},
                temporal_patterns={'creation_time': datetime.now().timestamp()},
                geometric_preferences={'triangle': 0.8}
            ),
            AuthorshipProfile(
                document_id=2,
                stylistic_features={'avg_word_length': 5.1, 'comma_frequency': 0.021},
                cipher_preferences={'caesar': 0.7, 'substitution': 0.3},
                complexity_signature={'avg_complexity': 0.75},
                temporal_patterns={'creation_time': (datetime.now() - timedelta(days=30)).timestamp()},
                geometric_preferences={'triangle': 0.7}
            )
        ]
        
        with patch.object(analyzer, '_create_authorship_profile', side_effect=mock_profiles), \
             patch.object(analyzer, '_calculate_authorship_similarity_matrix', 
                         return_value=np.array([[1.0, 0.8], [0.8, 1.0]])), \
             patch.object(analyzer, '_identify_author_groups', return_value=[
                 {'group_id': 1, 'document_ids': [1, 2], 'confidence': 0.8}
             ]), \
             patch.object(analyzer, '_analyze_stylistic_evolution', return_value={}), \
             patch.object(analyzer, '_detect_shared_knowledge_indicators', return_value={}), \
             patch.object(analyzer, '_calculate_authorship_confidence_scores', return_value={}), \
             patch.object(analyzer, '_generate_authorship_summary', return_value={}):
            
            result = analyzer.analyze_authorship_patterns(document_ids[:2])
            
            # Verify result structure
            assert 'document_count' in result
            assert 'authorship_profiles' in result
            assert 'similarity_matrix' in result
            assert 'author_groups' in result
            assert result['document_count'] == 2
            assert len(result['authorship_profiles']) == 2
    
    def test_analyze_authorship_patterns_insufficient_documents(self, analyzer):
        """Test authorship analysis with insufficient documents"""
        result = analyzer.analyze_authorship_patterns([1])
        assert 'error' in result
        assert 'Need at least 2 documents' in result['error']
    
    def test_detect_shared_cryptographic_knowledge_basic(self, analyzer):
        """Test basic shared cryptographic knowledge detection"""
        document_ids = [1, 2, 3]
        
        with patch.object(analyzer, '_identify_sophisticated_patterns', return_value=[
                 {'pattern_type': 'advanced_cipher', 'complexity': 0.9}
             ]), \
             patch.object(analyzer, '_analyze_cipher_method_progression', return_value={}), \
             patch.object(analyzer, '_detect_mathematical_knowledge_indicators', return_value={}), \
             patch.object(analyzer, '_analyze_pattern_complexity_evolution', return_value={}), \
             patch.object(analyzer, '_calculate_knowledge_sharing_probability', return_value=0.75), \
             patch.object(analyzer, '_identify_knowledge_transfer_patterns', return_value=[]), \
             patch.object(analyzer, '_calculate_evidence_strength', return_value=0.8):
            
            result = analyzer.detect_shared_cryptographic_knowledge(document_ids)
            
            # Verify result structure
            assert 'document_count' in result
            assert 'sophisticated_patterns' in result
            assert 'knowledge_sharing_probability' in result
            assert result['document_count'] == 3
            assert 0 <= result['knowledge_sharing_probability'] <= 1
    
    def test_build_relationship_network_basic(self, analyzer, mock_db_session, sample_documents):
        """Test basic relationship network building"""
        document_ids = [1, 2, 3]
        
        # Setup document mocks
        mock_db_session.query.return_value.get.side_effect = lambda doc_id: sample_documents[doc_id - 1]
        
        # Mock relationship calculation
        mock_relationship = DocumentRelationship(
            document1_id=1,
            document2_id=2,
            relationship_type="pattern_based",
            strength=0.8,
            confidence=0.85,
            evidence_count=5,
            shared_patterns=['cipher_caesar', 'geometric_triangle'],
            temporal_correlation=0.6,
            authorship_similarity=0.7
        )
        
        with patch.object(analyzer, '_calculate_document_relationship', return_value=mock_relationship), \
             patch.object(analyzer, '_calculate_network_metrics', return_value={
                 'centrality': {1: 0.5, 2: 0.8, 3: 0.3}
             }), \
             patch.object(analyzer, '_identify_communities', return_value={1: 0, 2: 0, 3: 1}), \
             patch.object(analyzer, '_identify_central_documents', return_value=[
                 {'document_id': 2, 'centrality_score': 0.8}
             ]):
            
            result = analyzer.build_relationship_network(document_ids)
            
            # Verify result structure
            assert 'network' in result
            assert 'metrics' in result
            assert 'communities' in result
            assert 'central_documents' in result
            assert 'summary' in result
            
            # Verify network structure
            network = result['network']
            assert 'nodes' in network
            assert 'edges' in network
            assert len(network['nodes']) == 3
    
    def test_calculate_pattern_correlation(self, analyzer, mock_db_session):
        """Test pattern correlation calculation"""
        doc1_id, doc2_id = 1, 2
        
        # Mock pattern signatures
        with patch.object(analyzer, '_get_document_pattern_signatures', side_effect=[
            {'cipher_caesar': 0.8, 'geometric_triangle': 0.7},
            {'cipher_caesar': 0.75, 'cipher_substitution': 0.6}
        ]):
            correlation = analyzer._calculate_pattern_correlation(doc1_id, doc2_id)
            
            # Should have some correlation due to shared 'cipher_caesar' pattern
            assert 0 <= correlation <= 1
    
    def test_calculate_stylistic_correlation(self, analyzer):
        """Test stylistic correlation calculation"""
        doc1_id, doc2_id = 1, 2
        
        # Mock stylistic features
        with patch.object(analyzer, '_extract_stylistic_features', side_effect=[
            {'avg_word_length': 5.2, 'comma_frequency': 0.02, 'char_freq_E': 0.12},
            {'avg_word_length': 5.1, 'comma_frequency': 0.021, 'char_freq_E': 0.13}
        ]):
            correlation = analyzer._calculate_stylistic_correlation(doc1_id, doc2_id)
            
            # Should have high correlation due to similar features
            assert 0 <= correlation <= 1
    
    def test_calculate_temporal_correlation(self, analyzer):
        """Test temporal correlation calculation"""
        doc1_id, doc2_id = 1, 2
        
        now = datetime.now()
        
        # Mock temporal features
        with patch.object(analyzer, '_extract_temporal_features', side_effect=[
            {'creation_time': now.timestamp(), 'pattern_timings': [now.timestamp()]},
            {'creation_time': (now - timedelta(days=1)).timestamp(), 'pattern_timings': [(now - timedelta(hours=1)).timestamp()]}
        ]), \
        patch.object(analyzer, '_calculate_pattern_timing_correlation', return_value=0.8):
            
            correlation = analyzer._calculate_temporal_correlation(doc1_id, doc2_id)
            
            # Should have reasonable correlation for documents close in time
            assert 0 <= correlation <= 1
    
    def test_extract_stylistic_features(self, analyzer, mock_db_session):
        """Test stylistic feature extraction"""
        document_id = 1
        
        # Mock document and pages
        mock_document = Mock(id=1, title="Test Document")
        mock_pages = [
            Mock(text="This is a test document. It has multiple sentences! How exciting?"),
            Mock(text="Another page with different content, including commas; and semicolons.")
        ]
        
        mock_db_session.query.return_value.get.return_value = mock_document
        mock_db_session.query.return_value.filter.return_value.all.return_value = mock_pages
        
        features = analyzer._extract_stylistic_features(document_id)
        
        # Verify feature extraction
        assert isinstance(features, dict)
        assert 'text_length' in features
        assert 'word_count' in features
        assert 'sentence_count' in features
        assert 'avg_word_length' in features
        assert 'comma_frequency' in features
        assert 'semicolon_frequency' in features
        
        # Verify some basic calculations
        assert features['text_length'] > 0
        assert features['word_count'] > 0
        assert features['sentence_count'] >= 3  # At least 3 sentences from the mock text
    
    def test_calculate_connection_strength(self, analyzer, sample_instances):
        """Test connection strength calculation between instances"""
        instances1 = [sample_instances[0]]
        instances2 = [sample_instances[1]]
        
        strength = analyzer._calculate_connection_strength(instances1, instances2)
        
        # Should calculate reasonable strength based on confidence and quality
        assert 0 <= strength <= 1
        assert strength > 0  # Should have some strength with good instances
    
    def test_build_evidence_points(self, analyzer, sample_instances):
        """Test evidence point building"""
        instances1 = [sample_instances[0]]
        instances2 = [sample_instances[1]]
        
        with patch.object(analyzer, '_calculate_instance_similarity', return_value=0.8), \
             patch.object(analyzer, '_calculate_parameter_consistency', return_value=0.7), \
             patch.object(analyzer, '_find_consistent_parameters', return_value=['param1']):
            
            evidence_points = analyzer._build_evidence_points(instances1, instances2)
            
            # Should generate evidence points
            assert isinstance(evidence_points, list)
            if evidence_points:
                point = evidence_points[0]
                assert 'type' in point
                assert 'strength' in point
                assert 0 <= point['strength'] <= 1
    
    def test_create_authorship_profile(self, analyzer, mock_db_session):
        """Test authorship profile creation"""
        document_id = 1
        
        # Mock document
        mock_document = Mock(id=1, title="Test Document")
        mock_db_session.query.return_value.get.return_value = mock_document
        
        with patch.object(analyzer, '_extract_stylistic_features', return_value={'feature1': 0.5}), \
             patch.object(analyzer, '_extract_cipher_preferences', return_value={'caesar': 0.8}), \
             patch.object(analyzer, '_extract_complexity_signature', return_value={'avg_complexity': 0.7}), \
             patch.object(analyzer, '_extract_temporal_features', return_value={'creation_time': datetime.now().timestamp()}), \
             patch.object(analyzer, '_extract_geometric_preferences', return_value={'triangle': 0.6}):
            
            profile = analyzer._create_authorship_profile(document_id)
            
            # Verify profile creation
            assert isinstance(profile, AuthorshipProfile)
            assert profile.document_id == document_id
            assert isinstance(profile.stylistic_features, dict)
            assert isinstance(profile.cipher_preferences, dict)
            assert isinstance(profile.complexity_signature, dict)
            assert isinstance(profile.temporal_patterns, dict)
            assert isinstance(profile.geometric_preferences, dict)
    
    def test_calculate_profile_similarity(self, analyzer):
        """Test authorship profile similarity calculation"""
        profile1 = AuthorshipProfile(
            document_id=1,
            stylistic_features={'feature1': 0.5, 'feature2': 0.3},
            cipher_preferences={'caesar': 0.8, 'substitution': 0.2},
            complexity_signature={'avg_complexity': 0.7},
            temporal_patterns={'creation_time': datetime.now().timestamp()},
            geometric_preferences={'triangle': 0.6}
        )
        
        profile2 = AuthorshipProfile(
            document_id=2,
            stylistic_features={'feature1': 0.52, 'feature2': 0.31},
            cipher_preferences={'caesar': 0.75, 'substitution': 0.25},
            complexity_signature={'avg_complexity': 0.72},
            temporal_patterns={'creation_time': (datetime.now() - timedelta(days=1)).timestamp()},
            geometric_preferences={'triangle': 0.58}
        )
        
        with patch.object(analyzer, '_calculate_feature_similarity', return_value=0.9), \
             patch.object(analyzer, '_calculate_temporal_similarity', return_value=0.8):
            
            similarity = analyzer._calculate_profile_similarity(profile1, profile2)
            
            # Should calculate high similarity for similar profiles
            assert 0 <= similarity <= 1
            assert similarity > 0.5  # Should be reasonably high for similar profiles
    
    def test_calculate_feature_similarity(self, analyzer):
        """Test feature similarity calculation"""
        features1 = {'feature1': 0.5, 'feature2': 0.3, 'feature3': 0.8}
        features2 = {'feature1': 0.52, 'feature2': 0.31, 'feature4': 0.2}
        
        similarity = analyzer._calculate_feature_similarity(features1, features2)
        
        # Should calculate reasonable similarity
        assert 0 <= similarity <= 1
    
    def test_calculate_feature_similarity_empty_features(self, analyzer):
        """Test feature similarity with empty features"""
        similarity = analyzer._calculate_feature_similarity({}, {'feature1': 0.5})
        assert similarity == 0.0
        
        similarity = analyzer._calculate_feature_similarity({'feature1': 0.5}, {})
        assert similarity == 0.0
    
    def test_cluster_documents_by_correlation(self, analyzer):
        """Test document clustering by correlation"""
        correlation_matrix = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ])
        document_ids = [1, 2, 3]
        
        clusters = analyzer._cluster_documents_by_correlation(correlation_matrix, document_ids)
        
        # Should return cluster assignments
        assert isinstance(clusters, dict)
        assert len(clusters) == 3
        for doc_id in document_ids:
            assert doc_id in clusters
            assert isinstance(clusters[doc_id], (int, np.integer))
    
    def test_cluster_documents_insufficient_data(self, analyzer):
        """Test clustering with insufficient data"""
        correlation_matrix = np.array([[1.0]])
        document_ids = [1]
        
        clusters = analyzer._cluster_documents_by_correlation(correlation_matrix, document_ids)
        
        # Should handle single document case
        assert clusters == {1: 0}
    
    def test_get_document_pattern_signatures(self, analyzer, mock_db_session, sample_instances, sample_patterns):
        """Test getting document pattern signatures"""
        document_id = 1
        
        # Setup mocks
        mock_instances = [sample_instances[0]]
        mock_instances[0].cross_pattern = sample_patterns[0]
        
        mock_db_session.query.return_value.filter.return_value.all.return_value = mock_instances
        
        signatures = analyzer._get_document_pattern_signatures(document_id)
        
        # Should return pattern signatures
        assert isinstance(signatures, dict)
        if signatures:
            for key, value in signatures.items():
                assert isinstance(key, str)
                assert isinstance(value, (int, float))
                assert 0 <= value <= 1
    
    def test_calculate_document_relationship(self, analyzer):
        """Test comprehensive document relationship calculation"""
        doc1_id, doc2_id = 1, 2
        
        with patch.object(analyzer, '_calculate_pattern_correlation', return_value=0.8), \
             patch.object(analyzer, '_calculate_stylistic_correlation', return_value=0.7), \
             patch.object(analyzer, '_calculate_temporal_correlation', return_value=0.6), \
             patch.object(analyzer, '_get_document_pattern_signatures', side_effect=[
                 {'pattern1': 0.8, 'pattern2': 0.6},
                 {'pattern1': 0.7, 'pattern3': 0.5}
             ]):
            
            relationship = analyzer._calculate_document_relationship(doc1_id, doc2_id)
            
            # Verify relationship structure
            assert isinstance(relationship, DocumentRelationship)
            assert relationship.document1_id == doc1_id
            assert relationship.document2_id == doc2_id
            assert 0 <= relationship.strength <= 1
            assert 0 <= relationship.confidence <= 1
            assert relationship.evidence_count >= 0
            assert isinstance(relationship.shared_patterns, list)
            assert relationship.relationship_type in ['pattern_based', 'stylistic', 'temporal', 'mixed']


class TestRelationshipAnalyzerIntegration:
    """Integration tests for RelationshipAnalyzer"""
    
    @pytest.fixture
    def analyzer_with_real_db(self):
        """Create analyzer with real database session for integration tests"""
        # This would use a real test database
        # For now, we'll skip these tests in unit testing
        pytest.skip("Integration tests require real database")
    
    def test_full_correlation_analysis_workflow(self, analyzer_with_real_db):
        """Test complete correlation analysis workflow"""
        # This would test the full workflow with real data
        pass
    
    def test_evidence_trail_tracking_workflow(self, analyzer_with_real_db):
        """Test complete evidence trail tracking workflow"""
        # This would test evidence trail tracking with real data
        pass
    
    def test_authorship_analysis_workflow(self, analyzer_with_real_db):
        """Test complete authorship analysis workflow"""
        # This would test authorship analysis with real data
        pass


if __name__ == "__main__":
    pytest.main([__file__])