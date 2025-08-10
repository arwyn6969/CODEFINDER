"""
Tests for Cross-Document Correlation Analysis System
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from app.services.cross_document_analyzer import (
    CrossDocumentAnalyzer, DocumentFingerprint, CorrelationResult, 
    SharedEncodingScheme
)
from app.services.cipher_detector import CipherMatch, CipherType


class TestCrossDocumentAnalyzer:
    
    @pytest.fixture
    def analyzer(self):
        """Create a CrossDocumentAnalyzer instance"""
        return CrossDocumentAnalyzer()
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            {
                'id': 'doc1',
                'text': 'THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG',
                'pages': [
                    {'page_number': 1, 'text': 'THE QUICK BROWN FOX'},
                    {'page_number': 2, 'text': 'JUMPS OVER THE LAZY DOG'}
                ]
            },
            {
                'id': 'doc2', 
                'text': 'WKH TXLFN EURZQ IRA MXPSV RYHU WKH ODCB GRJ',  # Caesar shift 3
                'pages': [
                    {'page_number': 1, 'text': 'WKH TXLFN EURZQ IRA'},
                    {'page_number': 2, 'text': 'MXPSV RYHU WKH ODCB GRJ'}
                ]
            },
            {
                'id': 'doc3',
                'text': 'ANOTHER SAMPLE TEXT FOR TESTING PURPOSES',
                'pages': [
                    {'page_number': 1, 'text': 'ANOTHER SAMPLE TEXT'},
                    {'page_number': 3, 'text': 'FOR TESTING PURPOSES'}  # Gap in numbering
                ]
            }
        ]
    
    @pytest.fixture
    def sample_fingerprint(self):
        """Sample document fingerprint"""
        return DocumentFingerprint(
            document_id='test_doc',
            cipher_patterns={'caesar': 2, 'substitution': 1},
            frequency_profile={'E': 0.12, 'T': 0.09, 'A': 0.08},
            statistical_measures={'index_of_coincidence': 0.067, 'entropy': 4.1, 'chi_squared': 25.0},
            anomaly_markers=[{'type': 'page_numbering_gap', 'severity': 1}],
            geometric_patterns=[{'type': 'triangle', 'significance_score': 0.8}]
        )
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.min_correlation_threshold == 0.6
        assert analyzer.min_significance_level == 0.05
        assert hasattr(analyzer, 'cipher_detector')
    
    def test_extract_document_text(self, analyzer, sample_documents):
        """Test document text extraction"""
        doc = sample_documents[0]
        text = analyzer._extract_document_text(doc)
        
        assert 'THE QUICK BROWN FOX' in text
        assert 'JUMPS OVER THE LAZY DOG' in text
        assert len(text) > 0
    
    def test_create_document_fingerprint(self, analyzer, sample_documents):
        """Test document fingerprint creation"""
        doc = sample_documents[0]
        
        with patch.object(analyzer.cipher_detector, 'detect_ciphers') as mock_detect:
            mock_detect.return_value = [
                Mock(cipher_type=CipherType.CAESAR, method_details={'shift_amount': 3})
            ]
            
            fingerprint = analyzer._create_document_fingerprint(doc)
            
            assert fingerprint.document_id == 'doc1'
            assert isinstance(fingerprint.cipher_patterns, dict)
            assert isinstance(fingerprint.frequency_profile, dict)
            assert isinstance(fingerprint.statistical_measures, dict)
            assert isinstance(fingerprint.anomaly_markers, list)
            assert isinstance(fingerprint.geometric_patterns, list)
    
    def test_extract_cipher_patterns(self, analyzer):
        """Test cipher pattern extraction"""
        cipher_matches = [
            Mock(cipher_type=CipherType.CAESAR, method_details={'shift_amount': 3}),
            Mock(cipher_type=CipherType.CAESAR, method_details={'shift_amount': 5}),
            Mock(cipher_type=CipherType.ATBASH, method_details={}),
            Mock(cipher_type=CipherType.SKIP_PATTERN, method_details={'skip_amount': 2})
        ]
        
        patterns = analyzer._extract_cipher_patterns(cipher_matches)
        
        assert 'caesar' in patterns
        assert patterns['caesar'] == 2
        assert 'caesar_shift_3' in patterns
        assert 'caesar_shift_5' in patterns
        assert 'atbash' in patterns
        assert 'skip_pattern' in patterns
        assert 'skip_2' in patterns
    
    def test_calculate_frequency_profile(self, analyzer):
        """Test frequency profile calculation"""
        text = "HELLO WORLD"
        profile = analyzer._calculate_frequency_profile(text)
        
        assert isinstance(profile, dict)
        assert 'L' in profile
        assert profile['L'] == 3/10  # 3 L's out of 10 letters
        assert 'O' in profile
        assert profile['O'] == 2/10  # 2 O's out of 10 letters
        
        # Check that frequencies sum to 1
        total_freq = sum(profile.values())
        assert abs(total_freq - 1.0) < 0.001
    
    def test_calculate_statistical_measures(self, analyzer):
        """Test statistical measures calculation"""
        text = "THE QUICK BROWN FOX"
        measures = analyzer._calculate_statistical_measures(text)
        
        assert 'index_of_coincidence' in measures
        assert 'entropy' in measures
        assert 'chi_squared' in measures
        assert 'text_length' in measures
        assert 'unique_letters' in measures
        
        assert measures['text_length'] == 16  # Letters only (THEQUICKBROWNFOX = 16 letters)
        assert measures['index_of_coincidence'] >= 0
        assert measures['entropy'] >= 0
        assert measures['chi_squared'] >= 0
    
    def test_extract_anomaly_markers(self, analyzer, sample_documents):
        """Test anomaly marker extraction"""
        doc = sample_documents[2]  # Has page numbering gap
        anomalies = analyzer._extract_anomaly_markers(doc)
        
        assert len(anomalies) > 0
        
        # Check for page numbering gap
        gap_anomalies = [a for a in anomalies if a['type'] == 'page_numbering_gap']
        assert len(gap_anomalies) > 0
        
        gap_anomaly = gap_anomalies[0]
        assert gap_anomaly['severity'] == 1  # Gap of 1 (page 2 missing)
    
    def test_cipher_pattern_correlation(self, analyzer):
        """Test cipher pattern correlation calculation"""
        patterns1 = {'caesar': 2, 'atbash': 1, 'substitution': 1}
        patterns2 = {'caesar': 3, 'atbash': 1, 'skip_pattern': 2}
        
        correlation = analyzer._calculate_cipher_pattern_correlation(patterns1, patterns2)
        
        assert 0 <= correlation <= 1
        assert correlation > 0  # Should have some correlation due to shared patterns
        
        # Test with identical patterns
        identical_corr = analyzer._calculate_cipher_pattern_correlation(patterns1, patterns1)
        assert identical_corr == 1.0
        
        # Test with no patterns
        empty_corr = analyzer._calculate_cipher_pattern_correlation({}, {})
        assert empty_corr == 0.0
    
    def test_frequency_correlation(self, analyzer):
        """Test frequency correlation calculation"""
        freq1 = {'A': 0.1, 'B': 0.2, 'C': 0.3, 'D': 0.4}
        freq2 = {'A': 0.15, 'B': 0.25, 'C': 0.25, 'D': 0.35}
        
        correlation = analyzer._calculate_frequency_correlation(freq1, freq2)
        
        assert 0 <= correlation <= 1
        assert correlation > 0.5  # Should be highly correlated
        
        # Test with identical frequencies
        identical_corr = analyzer._calculate_frequency_correlation(freq1, freq1)
        assert identical_corr == 1.0
    
    def test_statistical_correlation(self, analyzer):
        """Test statistical correlation calculation"""
        stats1 = {'index_of_coincidence': 0.067, 'entropy': 4.1, 'chi_squared': 25.0}
        stats2 = {'index_of_coincidence': 0.070, 'entropy': 4.0, 'chi_squared': 30.0}
        
        correlation = analyzer._calculate_statistical_correlation(stats1, stats2)
        
        assert 0 <= correlation <= 1
        
        # Test with identical stats
        identical_corr = analyzer._calculate_statistical_correlation(stats1, stats1)
        assert identical_corr == 1.0
    
    def test_calculate_document_correlation(self, analyzer, sample_fingerprint):
        """Test document correlation calculation"""
        fp1 = sample_fingerprint
        fp2 = DocumentFingerprint(
            document_id='test_doc2',
            cipher_patterns={'caesar': 1, 'atbash': 2},
            frequency_profile={'E': 0.11, 'T': 0.10, 'A': 0.07},
            statistical_measures={'index_of_coincidence': 0.065, 'entropy': 4.2, 'chi_squared': 28.0},
            anomaly_markers=[{'type': 'page_numbering_gap', 'severity': 2}],
            geometric_patterns=[{'type': 'circle', 'significance_score': 0.7}]
        )
        
        correlation = analyzer._calculate_document_correlation(fp1, fp2)
        
        assert isinstance(correlation, CorrelationResult)
        assert correlation.document1_id == 'test_doc'
        assert correlation.document2_id == 'test_doc2'
        assert 0 <= correlation.correlation_score <= 1
        assert isinstance(correlation.shared_patterns, list)
        assert isinstance(correlation.evidence, dict)
        assert 0 <= correlation.significance_level <= 1
    
    def test_find_shared_patterns(self, analyzer, sample_fingerprint):
        """Test shared pattern identification"""
        fp1 = sample_fingerprint
        fp2 = DocumentFingerprint(
            document_id='test_doc2',
            cipher_patterns={'caesar': 1, 'substitution': 2},  # Shares caesar and substitution
            frequency_profile={'E': 0.11, 'T': 0.10, 'A': 0.07},
            statistical_measures={'index_of_coincidence': 0.065, 'entropy': 4.2, 'chi_squared': 28.0},
            anomaly_markers=[{'type': 'page_numbering_gap', 'severity': 2}],  # Shares anomaly type
            geometric_patterns=[{'type': 'triangle', 'significance_score': 0.7}]  # Shares triangle
        )
        
        shared_patterns = analyzer._find_shared_patterns(fp1, fp2)
        
        assert 'cipher_caesar' in shared_patterns
        assert 'cipher_substitution' in shared_patterns
        assert 'anomaly_page_numbering_gap' in shared_patterns
        assert 'geometric_triangle' in shared_patterns
    
    def test_analyze_document_correlations(self, analyzer, sample_documents):
        """Test comprehensive document correlation analysis"""
        with patch.object(analyzer.cipher_detector, 'detect_ciphers') as mock_detect:
            mock_detect.return_value = [
                Mock(cipher_type=CipherType.CAESAR, method_details={'shift_amount': 3})
            ]
            
            analysis = analyzer.analyze_document_correlations(sample_documents)
            
            assert 'document_count' in analysis
            assert analysis['document_count'] == 3
            assert 'fingerprints' in analysis
            assert 'pairwise_correlations' in analysis
            assert 'shared_encoding_schemes' in analysis
            assert 'cross_document_anomalies' in analysis
            assert 'correlation_matrix' in analysis
            assert 'analysis_summary' in analysis
            
            # Check fingerprints
            assert len(analysis['fingerprints']) == 3
            
            # Check pairwise correlations (3 documents = 3 pairs)
            assert len(analysis['pairwise_correlations']) == 3
    
    def test_identify_shared_encoding_schemes(self, analyzer):
        """Test shared encoding scheme identification"""
        fingerprints = [
            DocumentFingerprint(
                document_id='doc1',
                cipher_patterns={'caesar': 2, 'caesar_shift_3': 1},
                frequency_profile={'E': 0.12, 'T': 0.09},
                statistical_measures={'index_of_coincidence': 0.067},
                anomaly_markers=[],
                geometric_patterns=[]
            ),
            DocumentFingerprint(
                document_id='doc2',
                cipher_patterns={'caesar': 1, 'caesar_shift_3': 1},
                frequency_profile={'E': 0.11, 'T': 0.10},
                statistical_measures={'index_of_coincidence': 0.065},
                anomaly_markers=[],
                geometric_patterns=[]
            ),
            DocumentFingerprint(
                document_id='doc3',
                cipher_patterns={'atbash': 2},
                frequency_profile={'E': 0.08, 'T': 0.12},
                statistical_measures={'index_of_coincidence': 0.070},
                anomaly_markers=[],
                geometric_patterns=[]
            )
        ]
        
        schemes = analyzer._identify_shared_encoding_schemes(fingerprints)
        
        assert isinstance(schemes, list)
        
        # Should find shared Caesar scheme between doc1 and doc2
        caesar_schemes = [s for s in schemes if 'caesar' in s.scheme_type]
        if caesar_schemes:
            scheme = caesar_schemes[0]
            assert len(scheme.documents) >= 2
            assert 'doc1' in scheme.documents or 'doc2' in scheme.documents
    
    def test_detect_cross_document_anomalies(self, analyzer):
        """Test cross-document anomaly detection"""
        fingerprints = [
            DocumentFingerprint(
                document_id='doc1',
                cipher_patterns={},
                frequency_profile={},
                statistical_measures={},
                anomaly_markers=[
                    {'type': 'page_numbering_gap', 'severity': 1},
                    {'type': 'unusual_letter_frequency', 'letter': 'Z', 'frequency': 0.25}
                ],
                geometric_patterns=[]
            ),
            DocumentFingerprint(
                document_id='doc2',
                cipher_patterns={},
                frequency_profile={},
                statistical_measures={},
                anomaly_markers=[
                    {'type': 'page_numbering_gap', 'severity': 2},
                    {'type': 'different_anomaly', 'value': 1}
                ],
                geometric_patterns=[]
            ),
            DocumentFingerprint(
                document_id='doc3',
                cipher_patterns={},
                frequency_profile={},
                statistical_measures={},
                anomaly_markers=[
                    {'type': 'page_numbering_gap', 'severity': 1}
                ],
                geometric_patterns=[]
            )
        ]
        
        cross_anomalies = analyzer._detect_cross_document_anomalies(fingerprints)
        
        assert isinstance(cross_anomalies, list)
        
        # Should find page_numbering_gap across all 3 documents
        gap_anomalies = [a for a in cross_anomalies if a['type'] == 'page_numbering_gap']
        if gap_anomalies:
            gap_anomaly = gap_anomalies[0]
            assert gap_anomaly['document_count'] == 3
            assert len(gap_anomaly['documents']) == 3
    
    def test_build_correlation_matrix(self, analyzer):
        """Test correlation matrix building"""
        fingerprints = [
            DocumentFingerprint('doc1', {}, {}, {}, [], []),
            DocumentFingerprint('doc2', {}, {}, {}, [], []),
            DocumentFingerprint('doc3', {}, {}, {}, [], [])
        ]
        
        correlations = [
            CorrelationResult('doc1', 'doc2', 0.8, 'test', [], {}, 0.05),
            CorrelationResult('doc1', 'doc3', 0.6, 'test', [], {}, 0.1),
            CorrelationResult('doc2', 'doc3', 0.7, 'test', [], {}, 0.08)
        ]
        
        matrix_data = analyzer._build_correlation_matrix(fingerprints, correlations)
        
        assert 'matrix' in matrix_data
        assert 'document_ids' in matrix_data
        assert 'average_correlation' in matrix_data
        assert 'max_correlation' in matrix_data
        assert 'min_correlation' in matrix_data
        
        matrix = np.array(matrix_data['matrix'])
        assert matrix.shape == (3, 3)
        
        # Check diagonal is 1.0
        assert all(matrix[i][i] == 1.0 for i in range(3))
        
        # Check symmetry
        assert matrix[0][1] == matrix[1][0]
        assert matrix[0][2] == matrix[2][0]
        assert matrix[1][2] == matrix[2][1]
    
    def test_detect_shared_cryptographic_knowledge(self, analyzer, sample_documents):
        """Test shared cryptographic knowledge detection"""
        with patch.object(analyzer, 'analyze_document_correlations') as mock_analyze:
            mock_analyze.return_value = {
                'fingerprints': [
                    DocumentFingerprint('doc1', {'caesar': 2}, {}, {'index_of_coincidence': 0.067}, [], []),
                    DocumentFingerprint('doc2', {'caesar': 1}, {}, {'index_of_coincidence': 0.065}, [], [])
                ],
                'shared_encoding_schemes': [
                    SharedEncodingScheme('scheme1', 'caesar_based', ['doc1', 'doc2'], {}, 0.8, 0.7)
                ]
            }
            
            knowledge_analysis = analyzer.detect_shared_cryptographic_knowledge(sample_documents)
            
            assert 'sophisticated_patterns' in knowledge_analysis
            assert 'complexity_analysis' in knowledge_analysis
            assert 'mathematical_relationships' in knowledge_analysis
            assert 'knowledge_sharing_probability' in knowledge_analysis
            assert 'evidence_summary' in knowledge_analysis
            
            assert 0 <= knowledge_analysis['knowledge_sharing_probability'] <= 1
    
    def test_generate_relationship_scoring(self, analyzer):
        """Test relationship scoring generation"""
        correlations = [
            CorrelationResult('doc1', 'doc2', 0.8, 'cipher_based', ['pattern1', 'pattern2'], {}, 0.05),
            CorrelationResult('doc1', 'doc3', 0.6, 'frequency_based', ['pattern1'], {}, 0.1),
            CorrelationResult('doc2', 'doc3', 0.9, 'statistical_based', ['pattern1', 'pattern2', 'pattern3'], {}, 0.02)
        ]
        
        scoring = analyzer.generate_relationship_scoring(correlations)
        
        assert 'score_statistics' in scoring
        assert 'high_correlation_count' in scoring
        assert 'significance_results' in scoring
        assert 'ranked_relationships' in scoring
        assert 'relationship_clusters' in scoring
        
        # Check score statistics
        stats = scoring['score_statistics']
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
        
        # Check ranked relationships
        ranked = scoring['ranked_relationships']
        assert len(ranked) == 3
        assert ranked[0]['score'] >= ranked[1]['score']  # Should be sorted by score
    
    def test_cipher_complexity_calculation(self, analyzer):
        """Test cipher complexity calculation"""
        # Simple fingerprint
        simple_fp = DocumentFingerprint(
            'doc1', 
            {'caesar': 1}, 
            {}, {}, [], []
        )
        simple_complexity = analyzer._calculate_cipher_complexity(simple_fp)
        
        # Complex fingerprint
        complex_fp = DocumentFingerprint(
            'doc2',
            {'caesar': 1, 'substitution': 1, 'biliteral': 1, 'revolving': 1},
            {}, {}, [], []
        )
        complex_complexity = analyzer._calculate_cipher_complexity(complex_fp)
        
        assert complex_complexity > simple_complexity
        assert simple_complexity >= 0
        assert complex_complexity <= 5.0
    
    def test_trend_calculation(self, analyzer):
        """Test trend calculation"""
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        decreasing_values = [5.0, 4.0, 3.0, 2.0, 1.0]
        stable_values = [3.0, 3.1, 2.9, 3.0, 3.1]
        
        assert analyzer._calculate_trend(increasing_values) == "increasing"
        assert analyzer._calculate_trend(decreasing_values) == "decreasing"
        assert analyzer._calculate_trend(stable_values) == "stable"
        
        # Test insufficient data
        assert analyzer._calculate_trend([1.0]) == "insufficient_data"
    
    def test_significance_test(self, analyzer):
        """Test statistical significance testing"""
        high_correlation = CorrelationResult(
            'doc1', 'doc2', 0.9, 'cipher_based', 
            ['pattern1', 'pattern2', 'pattern3'], {}, 0.0
        )
        
        low_correlation = CorrelationResult(
            'doc3', 'doc4', 0.3, 'weak_correlation',
            [], {}, 0.0
        )
        
        high_sig = analyzer._perform_significance_test(high_correlation)
        low_sig = analyzer._perform_significance_test(low_correlation)
        
        assert high_sig['is_significant'] == True
        assert low_sig['is_significant'] == False
        
        assert high_sig['p_value'] < low_sig['p_value']
        assert len(high_sig['confidence_interval']) == 2
    
    def test_relationship_clusters(self, analyzer):
        """Test relationship cluster identification"""
        correlations = [
            CorrelationResult('doc1', 'doc2', 0.8, 'test', [], {}, 0.05),
            CorrelationResult('doc2', 'doc3', 0.7, 'test', [], {}, 0.08),
            CorrelationResult('doc1', 'doc3', 0.6, 'test', [], {}, 0.1),
            CorrelationResult('doc4', 'doc5', 0.9, 'test', [], {}, 0.02)
        ]
        
        clusters = analyzer._identify_relationship_clusters(correlations)
        
        assert isinstance(clusters, list)
        assert len(clusters) >= 1
        
        # Should find cluster with doc1, doc2, doc3
        large_clusters = [c for c in clusters if c['size'] >= 3]
        if large_clusters:
            cluster = large_clusters[0]
            assert 'doc1' in cluster['documents']
            assert 'doc2' in cluster['documents']
            assert 'doc3' in cluster['documents']
    
    def test_error_handling(self, analyzer):
        """Test error handling for edge cases"""
        # Test with insufficient documents
        result = analyzer.analyze_document_correlations([])
        assert 'error' in result
        
        result = analyzer.analyze_document_correlations([{'id': 'single_doc'}])
        assert 'error' in result
        
        # Test with empty fingerprints
        empty_fp = DocumentFingerprint('empty', {}, {}, {}, [], [])
        correlation = analyzer._calculate_document_correlation(empty_fp, empty_fp)
        assert correlation.correlation_score >= 0
        
        # Test with empty correlations
        scoring = analyzer.generate_relationship_scoring([])
        assert 'error' in scoring


if __name__ == "__main__":
    pytest.main([__file__])