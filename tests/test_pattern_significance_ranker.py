"""
Tests for Pattern Significance Ranking System
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from app.services.pattern_significance_ranker import (
    PatternSignificanceRanker, PatternScore, RankedPattern, RankingResult
)
from app.models.cross_document_models import CrossDocumentPattern, CrossPatternInstance
from app.services.anomaly_detector import AnomalyResult


class TestPatternSignificanceRanker:
    """Test cases for PatternSignificanceRanker"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session"""
        return Mock(spec=Session)
    
    @pytest.fixture
    def ranker(self, mock_db_session):
        """Create PatternSignificanceRanker instance with mock database"""
        return PatternSignificanceRanker(mock_db_session)
    
    @pytest.fixture
    def sample_patterns(self):
        """Create sample patterns for testing"""
        return [
            Mock(
                id=1,
                pattern_type="cipher",
                pattern_subtype="caesar",
                significance_score=0.85,
                statistical_p_value=0.01,
                rarity_score=0.7,
                document_count=3,
                pattern_complexity=0.6,
                consistency_score=0.8,
                average_confidence=0.9
            ),
            Mock(
                id=2,
                pattern_type="geometric",
                pattern_subtype="triangle",
                significance_score=0.75,
                statistical_p_value=0.05,
                rarity_score=0.6,
                document_count=2,
                pattern_complexity=0.8,
                consistency_score=0.7,
                average_confidence=0.85
            ),
            Mock(
                id=3,
                pattern_type="linguistic",
                pattern_subtype="etymology",
                significance_score=0.65,
                statistical_p_value=0.1,
                rarity_score=0.5,
                document_count=1,
                pattern_complexity=0.4,
                consistency_score=0.6,
                average_confidence=0.8
            )
        ]
    
    @pytest.fixture
    def sample_anomalies(self):
        """Create sample anomalies for testing"""
        return [
            Mock(
                id=1,
                anomaly_type="character_distribution",
                deviation_score=2.5,
                significance_level=0.01
            ),
            Mock(
                id=2,
                anomaly_type="word_pattern",
                anomaly_score=1.8,
                statistical_significance=0.05
            ),
            Mock(
                id=3,
                anomaly_type="punctuation_size",
                deviation_score=1.2,
                significance_level=0.1
            )
        ]
    
    def test_rank_patterns_by_significance_basic(self, ranker, mock_db_session, sample_patterns):
        """Test basic pattern ranking by significance"""
        document_id = 1
        
        # Mock database queries
        mock_instances = [
            Mock(cross_pattern_id=1, document_id=1),
            Mock(cross_pattern_id=2, document_id=1),
            Mock(cross_pattern_id=3, document_id=1)
        ]
        
        mock_db_session.query.return_value.filter.return_value.all.return_value = mock_instances
        mock_db_session.query.return_value.filter.return_value.in_.return_value.all.return_value = sample_patterns
        
        result = ranker.rank_patterns_by_significance(document_id=document_id)
        
        # Verify result structure
        assert isinstance(result, RankingResult)
        assert len(result.ranked_patterns) <= len(sample_patterns)
        assert 'ranking_metadata' in result.__dict__
        assert 'summary_statistics' in result.__dict__
        
        # Verify ranking order (should be sorted by significance)
        if len(result.ranked_patterns) > 1:
            for i in range(len(result.ranked_patterns) - 1):
                assert result.ranked_patterns[i].score.significance_score >= result.ranked_patterns[i + 1].score.significance_score
    
    def test_rank_anomalies_by_significance_basic(self, ranker, sample_anomalies):
        """Test basic anomaly ranking by significance"""
        document_id = 1
        
        # Mock anomaly detector results
        anomaly_results = {
            'document_id': document_id,
            'character_distribution_anomalies': [sample_anomalies[0]],
            'word_pattern_anomalies': [sample_anomalies[1]],
            'punctuation_anomalies': [sample_anomalies[2]],
            'numerical_pattern_anomalies': [],
            'composite_anomalies': [],
            'summary': {}
        }
        
        with patch.object(ranker.anomaly_detector, 'detect_all_anomalies', return_value=anomaly_results):
            result = ranker.rank_anomalies_by_significance(document_id)
            
            # Verify result structure
            assert isinstance(result, RankingResult)
            assert len(result.ranked_patterns) == 3
            
            # Verify ranking order
            for i in range(len(result.ranked_patterns) - 1):
                assert result.ranked_patterns[i].score.significance_score >= result.ranked_patterns[i + 1].score.significance_score
    
    def test_calculate_confidence_intervals_basic(self, ranker):
        """Test confidence interval calculation"""
        pattern_scores = [0.8, 0.75, 0.9, 0.85, 0.7]
        
        lower, upper = ranker.calculate_confidence_intervals(pattern_scores, confidence_level=0.95)
        
        # Verify confidence interval properties
        assert 0.0 <= lower <= upper <= 1.0
        assert lower < upper
        
        # Mean should be within the interval
        mean_score = np.mean(pattern_scores)
        assert lower <= mean_score <= upper
    
    def test_calculate_confidence_intervals_edge_cases(self, ranker):
        """Test confidence interval calculation with edge cases"""
        # Empty list
        lower, upper = ranker.calculate_confidence_intervals([])
        assert lower == 0.0 and upper == 1.0
        
        # Single value
        lower, upper = ranker.calculate_confidence_intervals([0.5])
        assert lower == 0.0 and upper == 1.0
        
        # Two values
        lower, upper = ranker.calculate_confidence_intervals([0.4, 0.6])
        assert 0.0 <= lower <= upper <= 1.0
    
    def test_assess_pattern_reliability_comprehensive(self, ranker):
        """Test comprehensive pattern reliability assessment"""
        # High reliability pattern
        high_reliability_pattern = Mock(
            consistency_score=0.9,
            document_count=5,
            significance_score=0.05,  # Low p-value = high reliability
            pattern_complexity=0.8,
            average_confidence=0.95
        )
        
        reliability = ranker.assess_pattern_reliability(high_reliability_pattern)
        assert 0.7 <= reliability <= 1.0
        
        # Low reliability pattern
        low_reliability_pattern = Mock(
            consistency_score=0.3,
            document_count=1,
            significance_score=0.5,  # High p-value = low reliability
            pattern_complexity=0.2,
            average_confidence=0.4
        )
        
        reliability = ranker.assess_pattern_reliability(low_reliability_pattern)
        assert 0.0 <= reliability <= 0.6
    
    def test_assess_pattern_reliability_missing_attributes(self, ranker):
        """Test pattern reliability assessment with missing attributes"""
        minimal_pattern = Mock()
        
        reliability = ranker.assess_pattern_reliability(minimal_pattern)
        assert reliability == 0.5  # Default reliability
    
    def test_suggest_pattern_interpretations_cipher(self, ranker):
        """Test interpretation suggestions for cipher patterns"""
        cipher_pattern = Mock(
            pattern_type="cipher",
            pattern_subtype="caesar",
            pattern_complexity=0.8,
            document_count=3
        )
        
        suggestions = ranker.suggest_pattern_interpretations(cipher_pattern, 0.9)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5
        assert any("cipher" in s.lower() for s in suggestions)
        assert any("caesar" in s.lower() for s in suggestions)
    
    def test_suggest_pattern_interpretations_geometric(self, ranker):
        """Test interpretation suggestions for geometric patterns"""
        geometric_pattern = Mock(
            pattern_type="geometric",
            pattern_subtype="triangle",
            pattern_complexity=0.7,
            document_count=2
        )
        
        suggestions = ranker.suggest_pattern_interpretations(geometric_pattern, 0.8)
        
        assert isinstance(suggestions, list)
        assert any("geometric" in s.lower() or "triangle" in s.lower() for s in suggestions)
    
    def test_suggest_pattern_interpretations_high_significance(self, ranker):
        """Test interpretation suggestions for high significance patterns"""
        high_sig_pattern = Mock(
            pattern_type="cipher",
            pattern_subtype="substitution",
            pattern_complexity=0.9,
            document_count=5
        )
        
        suggestions = ranker.suggest_pattern_interpretations(high_sig_pattern, 0.98)
        
        assert any("extremely high significance" in s.lower() or "priority" in s.lower() for s in suggestions)
        assert any("cross-document" in s.lower() for s in suggestions)
        assert any("sophisticated" in s.lower() for s in suggestions)
    
    def test_calculate_pattern_significance_score_comprehensive(self, ranker, sample_patterns):
        """Test comprehensive pattern significance score calculation"""
        pattern = sample_patterns[0]  # High significance pattern
        
        score = ranker._calculate_pattern_significance_score(pattern)
        
        # Verify score structure
        assert isinstance(score, PatternScore)
        assert score.pattern_id == pattern.id
        assert score.pattern_type == pattern.pattern_type
        assert 0.0 <= score.significance_score <= 1.0
        assert len(score.confidence_interval) == 2
        assert 0.0 <= score.reliability_measure <= 1.0
        assert isinstance(score.statistical_measures, dict)
        assert isinstance(score.ranking_factors, dict)
        
        # Verify ranking factors
        expected_factors = ['statistical_significance', 'frequency_rarity', 'cross_document_presence', 
                          'complexity_score', 'consistency_measure']
        for factor in expected_factors:
            assert factor in score.ranking_factors
    
    def test_calculate_anomaly_significance_score_basic(self, ranker, sample_anomalies):
        """Test anomaly significance score calculation"""
        anomaly = sample_anomalies[0]  # Character distribution anomaly
        
        score = ranker._calculate_anomaly_significance_score(anomaly)
        
        # Verify score structure
        assert isinstance(score, PatternScore)
        assert score.pattern_type == anomaly.anomaly_type
        assert 0.0 <= score.significance_score <= 1.0
        assert len(score.confidence_interval) == 2
        assert 0.0 <= score.reliability_measure <= 1.0
    
    def test_generate_interpretation_suggestions_basic(self, ranker, sample_patterns):
        """Test interpretation suggestion generation"""
        pattern = sample_patterns[0]
        score = PatternScore(
            pattern_id=1,
            pattern_type="cipher",
            significance_score=0.85,
            confidence_interval=(0.8, 0.9),
            reliability_measure=0.8,
            statistical_measures={},
            ranking_factors={}
        )
        
        suggestions = ranker._generate_interpretation_suggestions(pattern, score)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5
        assert all(isinstance(s, str) for s in suggestions)
    
    def test_generate_anomaly_interpretations_basic(self, ranker, sample_anomalies):
        """Test anomaly interpretation generation"""
        anomaly = sample_anomalies[0]
        score = PatternScore(
            pattern_id=1,
            pattern_type="character_distribution",
            significance_score=0.9,
            confidence_interval=(0.85, 0.95),
            reliability_measure=0.85,
            statistical_measures={},
            ranking_factors={}
        )
        
        suggestions = ranker._generate_anomaly_interpretations(anomaly, score)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 4
        assert any("character" in s.lower() for s in suggestions)
    
    def test_calculate_evidence_strength_basic(self, ranker, sample_patterns):
        """Test evidence strength calculation"""
        pattern = sample_patterns[0]
        score = PatternScore(
            pattern_id=1,
            pattern_type="cipher",
            significance_score=0.85,
            confidence_interval=(0.8, 0.9),
            reliability_measure=0.8,
            statistical_measures={},
            ranking_factors={}
        )
        
        evidence_strength = ranker._calculate_evidence_strength(pattern, score)
        
        assert 0.0 <= evidence_strength <= 1.0
        assert evidence_strength > 0.5  # Should be high for good pattern
    
    def test_determine_recommendation_priority_levels(self, ranker):
        """Test recommendation priority determination"""
        score_high = PatternScore(1, "test", 0.95, (0.9, 1.0), 0.9, {}, {})
        score_medium = PatternScore(2, "test", 0.6, (0.5, 0.7), 0.6, {}, {})
        score_low = PatternScore(3, "test", 0.3, (0.2, 0.4), 0.3, {}, {})
        
        priority_high = ranker._determine_recommendation_priority(score_high, 0.9)
        priority_medium = ranker._determine_recommendation_priority(score_medium, 0.6)
        priority_low = ranker._determine_recommendation_priority(score_low, 0.3)
        
        assert priority_high in ["critical", "high"]
        assert priority_medium in ["medium", "high"]
        assert priority_low in ["low", "medium"]
    
    def test_generate_ranking_metadata_basic(self, ranker):
        """Test ranking metadata generation"""
        # Create mock ranked patterns
        ranked_patterns = [
            Mock(
                score=Mock(pattern_type="cipher", significance_score=0.9),
                recommendation_priority="critical"
            ),
            Mock(
                score=Mock(pattern_type="geometric", significance_score=0.8),
                recommendation_priority="high"
            ),
            Mock(
                score=Mock(pattern_type="cipher", significance_score=0.7),
                recommendation_priority="medium"
            )
        ]
        
        metadata = ranker._generate_ranking_metadata(ranked_patterns)
        
        assert 'total_patterns' in metadata
        assert metadata['total_patterns'] == 3
        assert 'ranking_method' in metadata
        assert 'priority_distribution' in metadata
        assert 'pattern_type_distribution' in metadata
        assert 'top_pattern' in metadata
    
    def test_calculate_summary_statistics_basic(self, ranker):
        """Test summary statistics calculation"""
        # Create mock ranked patterns
        ranked_patterns = [
            Mock(
                score=Mock(significance_score=0.9, reliability_measure=0.85),
                evidence_strength=0.88,
                recommendation_priority="critical"
            ),
            Mock(
                score=Mock(significance_score=0.7, reliability_measure=0.75),
                evidence_strength=0.72,
                recommendation_priority="high"
            ),
            Mock(
                score=Mock(significance_score=0.5, reliability_measure=0.6),
                evidence_strength=0.55,
                recommendation_priority="medium"
            )
        ]
        
        stats = ranker._calculate_summary_statistics(ranked_patterns)
        
        assert 'mean_significance' in stats
        assert 'median_significance' in stats
        assert 'std_significance' in stats
        assert 'mean_evidence_strength' in stats
        assert 'mean_reliability' in stats
        assert 'high_priority_percentage' in stats
        
        # Verify statistical calculations
        assert 0.0 <= stats['mean_significance'] <= 1.0
        assert 0.0 <= stats['high_priority_percentage'] <= 100.0
    
    def test_cipher_specific_interpretations(self, ranker):
        """Test cipher-specific interpretation generation"""
        caesar_pattern = Mock(pattern_subtype="caesar")
        substitution_pattern = Mock(pattern_subtype="substitution")
        skip_pattern = Mock(pattern_subtype="skip")
        
        caesar_suggestions = ranker._generate_cipher_interpretations(caesar_pattern, 0.8)
        substitution_suggestions = ranker._generate_cipher_interpretations(substitution_pattern, 0.8)
        skip_suggestions = ranker._generate_cipher_interpretations(skip_pattern, 0.8)
        
        assert any("caesar" in s.lower() for s in caesar_suggestions)
        assert any("substitution" in s.lower() for s in substitution_suggestions)
        assert any("skip" in s.lower() for s in skip_suggestions)
    
    def test_geometric_specific_interpretations(self, ranker):
        """Test geometric-specific interpretation generation"""
        triangle_pattern = Mock(pattern_subtype="triangle")
        circle_pattern = Mock(pattern_subtype="circle")
        golden_ratio_pattern = Mock(pattern_subtype="golden_ratio")
        
        triangle_suggestions = ranker._generate_geometric_interpretations(triangle_pattern, 0.8)
        circle_suggestions = ranker._generate_geometric_interpretations(circle_pattern, 0.8)
        golden_suggestions = ranker._generate_geometric_interpretations(golden_ratio_pattern, 0.8)
        
        assert any("triangle" in s.lower() for s in triangle_suggestions)
        assert any("circle" in s.lower() for s in circle_suggestions)
        assert any("golden ratio" in s.lower() for s in golden_suggestions)
    
    def test_error_handling_no_patterns(self, ranker, mock_db_session):
        """Test error handling when no patterns are found"""
        # Mock empty query results
        mock_db_session.query.return_value.filter.return_value.all.return_value = []
        mock_db_session.query.return_value.filter.return_value.in_.return_value.all.return_value = []
        
        result = ranker.rank_patterns_by_significance(document_id=1)
        
        assert isinstance(result, RankingResult)
        assert len(result.ranked_patterns) == 0
        assert 'message' in result.ranking_metadata
    
    def test_error_handling_anomaly_detector_error(self, ranker):
        """Test error handling when anomaly detector fails"""
        with patch.object(ranker.anomaly_detector, 'detect_all_anomalies', 
                         return_value={'error': 'Database error'}):
            
            result = ranker.rank_anomalies_by_significance(1)
            
            assert isinstance(result, RankingResult)
            assert len(result.ranked_patterns) == 0
            assert 'error' in result.ranking_metadata
    
    def test_ranking_consistency(self, ranker, sample_patterns):
        """Test that ranking is consistent across multiple calls"""
        # Mock database to return same patterns
        with patch.object(ranker, '_get_all_patterns', return_value=sample_patterns):
            result1 = ranker.rank_patterns_by_significance()
            result2 = ranker.rank_patterns_by_significance()
            
            # Rankings should be identical
            assert len(result1.ranked_patterns) == len(result2.ranked_patterns)
            
            for i in range(len(result1.ranked_patterns)):
                assert result1.ranked_patterns[i].rank == result2.ranked_patterns[i].rank
                assert abs(result1.ranked_patterns[i].score.significance_score - 
                          result2.ranked_patterns[i].score.significance_score) < 0.001


class TestPatternSignificanceRankerIntegration:
    """Integration tests for PatternSignificanceRanker"""
    
    @pytest.fixture
    def ranker_with_real_db(self):
        """Create ranker with real database session for integration tests"""
        # This would use a real test database
        # For now, we'll skip these tests in unit testing
        pytest.skip("Integration tests require real database")
    
    def test_full_pattern_ranking_workflow(self, ranker_with_real_db):
        """Test complete pattern ranking workflow"""
        # This would test the full workflow with real data
        pass
    
    def test_anomaly_ranking_workflow(self, ranker_with_real_db):
        """Test complete anomaly ranking workflow"""
        # This would test anomaly ranking with real data
        pass


if __name__ == "__main__":
    pytest.main([__file__])