"""
Cross-Document Correlation Analysis System
Analyzes patterns across multiple documents to identify shared encoding schemes,
cipher relationships, and document connections.
"""
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import math
import numpy as np
from scipy.stats import pearsonr, chi2_contingency
from scipy.spatial.distance import cosine
import itertools

from app.services.cipher_detector import CipherDetector, CipherMatch, CipherType
from app.models.database_models import Document, Page, Pattern


@dataclass
class DocumentFingerprint:
    """Represents a document's cryptographic fingerprint"""
    document_id: str
    cipher_patterns: Dict[str, int]
    frequency_profile: Dict[str, float]
    statistical_measures: Dict[str, float]
    anomaly_markers: List[Dict[str, Any]]
    geometric_patterns: List[Dict[str, Any]]
    
    
@dataclass
class CorrelationResult:
    """Represents correlation between two documents"""
    document1_id: str
    document2_id: str
    correlation_score: float
    correlation_type: str
    shared_patterns: List[str]
    evidence: Dict[str, Any]
    significance_level: float


@dataclass
class SharedEncodingScheme:
    """Represents a shared encoding scheme across documents"""
    scheme_id: str
    scheme_type: str
    documents: List[str]
    pattern_details: Dict[str, Any]
    confidence: float
    evidence_strength: float


class CrossDocumentAnalyzer:
    """
    Analyzes correlations and shared patterns across multiple documents
    """
    
    def __init__(self):
        self.cipher_detector = CipherDetector()
        self.min_correlation_threshold = 0.6
        self.min_significance_level = 0.05
        
    def analyze_document_correlations(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze correlations between multiple documents
        """
        if len(documents) < 2:
            return {'error': 'Need at least 2 documents for correlation analysis'}
        
        # Create fingerprints for all documents
        fingerprints = []
        for doc in documents:
            fingerprint = self._create_document_fingerprint(doc)
            fingerprints.append(fingerprint)
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                correlation = self._calculate_document_correlation(
                    fingerprints[i], fingerprints[j]
                )
                correlations.append(correlation)
        
        # Identify shared encoding schemes
        shared_schemes = self._identify_shared_encoding_schemes(fingerprints)
        
        # Detect anomaly patterns across documents
        cross_anomalies = self._detect_cross_document_anomalies(fingerprints)
        
        # Generate correlation matrix
        correlation_matrix = self._build_correlation_matrix(fingerprints, correlations)
        
        return {
            'document_count': len(documents),
            'fingerprints': fingerprints,
            'pairwise_correlations': correlations,
            'shared_encoding_schemes': shared_schemes,
            'cross_document_anomalies': cross_anomalies,
            'correlation_matrix': correlation_matrix,
            'analysis_summary': self._generate_correlation_summary(correlations, shared_schemes)
        }   
 
    def _create_document_fingerprint(self, document: Dict[str, Any]) -> DocumentFingerprint:
        """
        Create a cryptographic fingerprint for a document
        """
        doc_id = document.get('id', 'unknown')
        text_content = self._extract_document_text(document)
        
        # Detect cipher patterns
        cipher_matches = self.cipher_detector.detect_ciphers(text_content)
        cipher_patterns = self._extract_cipher_patterns(cipher_matches)
        
        # Calculate frequency profile
        frequency_profile = self._calculate_frequency_profile(text_content)
        
        # Calculate statistical measures
        statistical_measures = self._calculate_statistical_measures(text_content)
        
        # Extract anomaly markers
        anomaly_markers = self._extract_anomaly_markers(document)
        
        # Extract geometric patterns if available
        geometric_patterns = self._extract_geometric_patterns(document)
        
        return DocumentFingerprint(
            document_id=doc_id,
            cipher_patterns=cipher_patterns,
            frequency_profile=frequency_profile,
            statistical_measures=statistical_measures,
            anomaly_markers=anomaly_markers,
            geometric_patterns=geometric_patterns
        )
    
    def _calculate_document_correlation(self, fp1: DocumentFingerprint, 
                                     fp2: DocumentFingerprint) -> CorrelationResult:
        """
        Calculate correlation between two document fingerprints
        """
        # Cipher pattern correlation
        cipher_correlation = self._calculate_cipher_pattern_correlation(
            fp1.cipher_patterns, fp2.cipher_patterns
        )
        
        # Frequency profile correlation
        frequency_correlation = self._calculate_frequency_correlation(
            fp1.frequency_profile, fp2.frequency_profile
        )
        
        # Statistical measures correlation
        statistical_correlation = self._calculate_statistical_correlation(
            fp1.statistical_measures, fp2.statistical_measures
        )
        
        # Anomaly pattern correlation
        anomaly_correlation = self._calculate_anomaly_correlation(
            fp1.anomaly_markers, fp2.anomaly_markers
        )
        
        # Geometric pattern correlation
        geometric_correlation = self._calculate_geometric_correlation(
            fp1.geometric_patterns, fp2.geometric_patterns
        )
        
        # Combined correlation score
        weights = {
            'cipher': 0.3,
            'frequency': 0.25,
            'statistical': 0.2,
            'anomaly': 0.15,
            'geometric': 0.1
        }
        
        combined_score = (
            cipher_correlation * weights['cipher'] +
            frequency_correlation * weights['frequency'] +
            statistical_correlation * weights['statistical'] +
            anomaly_correlation * weights['anomaly'] +
            geometric_correlation * weights['geometric']
        )
        
        # Determine correlation type
        correlation_type = self._determine_correlation_type(
            cipher_correlation, frequency_correlation, statistical_correlation
        )
        
        # Find shared patterns
        shared_patterns = self._find_shared_patterns(fp1, fp2)
        
        # Calculate significance level
        significance_level = self._calculate_significance_level(
            combined_score, len(shared_patterns)
        )
        
        return CorrelationResult(
            document1_id=fp1.document_id,
            document2_id=fp2.document_id,
            correlation_score=combined_score,
            correlation_type=correlation_type,
            shared_patterns=shared_patterns,
            evidence={
                'cipher_correlation': cipher_correlation,
                'frequency_correlation': frequency_correlation,
                'statistical_correlation': statistical_correlation,
                'anomaly_correlation': anomaly_correlation,
                'geometric_correlation': geometric_correlation
            },
            significance_level=significance_level
        )
    
    def _identify_shared_encoding_schemes(self, fingerprints: List[DocumentFingerprint]) -> List[SharedEncodingScheme]:
        """
        Identify shared encoding schemes across documents
        """
        shared_schemes = []
        
        # Group documents by similar cipher patterns
        cipher_groups = self._group_by_cipher_patterns(fingerprints)
        
        for pattern_signature, docs in cipher_groups.items():
            if len(docs) >= 2:  # Need at least 2 documents for shared scheme
                scheme = SharedEncodingScheme(
                    scheme_id=f"scheme_{len(shared_schemes) + 1}",
                    scheme_type=self._classify_encoding_scheme(pattern_signature),
                    documents=[fp.document_id for fp in docs],
                    pattern_details=self._analyze_pattern_details(docs),
                    confidence=self._calculate_scheme_confidence(docs),
                    evidence_strength=self._calculate_evidence_strength(docs)
                )
                shared_schemes.append(scheme)
        
        # Look for frequency-based correlations
        frequency_schemes = self._identify_frequency_based_schemes(fingerprints)
        shared_schemes.extend(frequency_schemes)
        
        # Look for anomaly-based correlations
        anomaly_schemes = self._identify_anomaly_based_schemes(fingerprints)
        shared_schemes.extend(anomaly_schemes)
        
        return shared_schemes
    
    def _detect_cross_document_anomalies(self, fingerprints: List[DocumentFingerprint]) -> List[Dict[str, Any]]:
        """
        Detect anomalies that appear across multiple documents
        """
        cross_anomalies = []
        
        # Collect all anomaly types
        anomaly_types = defaultdict(list)
        for fp in fingerprints:
            for anomaly in fp.anomaly_markers:
                anomaly_type = anomaly.get('type', 'unknown')
                anomaly_types[anomaly_type].append({
                    'document_id': fp.document_id,
                    'anomaly': anomaly
                })
        
        # Find anomalies that appear in multiple documents
        for anomaly_type, occurrences in anomaly_types.items():
            if len(occurrences) >= 2:
                # Calculate correlation strength
                correlation_strength = self._calculate_anomaly_correlation_strength(occurrences)
                
                if correlation_strength >= 0.5:
                    cross_anomaly = {
                        'type': anomaly_type,
                        'document_count': len(occurrences),
                        'documents': [occ['document_id'] for occ in occurrences],
                        'correlation_strength': correlation_strength,
                        'pattern_details': self._analyze_anomaly_pattern(occurrences),
                        'significance': self._calculate_anomaly_significance(occurrences)
                    }
                    cross_anomalies.append(cross_anomaly)
        
        return cross_anomalies
    
    def _build_correlation_matrix(self, fingerprints: List[DocumentFingerprint], 
                                correlations: List[CorrelationResult]) -> Dict[str, Any]:
        """
        Build a correlation matrix from pairwise correlations
        """
        doc_ids = [fp.document_id for fp in fingerprints]
        n_docs = len(doc_ids)
        
        # Initialize matrix
        matrix = np.zeros((n_docs, n_docs))
        
        # Fill diagonal with 1.0 (perfect self-correlation)
        np.fill_diagonal(matrix, 1.0)
        
        # Fill matrix with correlation scores
        for corr in correlations:
            i = doc_ids.index(corr.document1_id)
            j = doc_ids.index(corr.document2_id)
            matrix[i][j] = corr.correlation_score
            matrix[j][i] = corr.correlation_score  # Symmetric matrix
        
        return {
            'matrix': matrix.tolist(),
            'document_ids': doc_ids,
            'average_correlation': np.mean(matrix[np.triu_indices(n_docs, k=1)]),
            'max_correlation': np.max(matrix[np.triu_indices(n_docs, k=1)]),
            'min_correlation': np.min(matrix[np.triu_indices(n_docs, k=1)])
        }
    
    def detect_shared_cryptographic_knowledge(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect evidence of shared cryptographic knowledge between documents
        """
        analysis = self.analyze_document_correlations(documents)
        
        # Look for sophisticated cipher patterns
        sophisticated_patterns = []
        for scheme in analysis['shared_encoding_schemes']:
            if self._is_sophisticated_scheme(scheme):
                sophisticated_patterns.append(scheme)
        
        # Analyze cipher complexity progression
        complexity_analysis = self._analyze_cipher_complexity_progression(
            analysis['fingerprints']
        )
        
        # Look for mathematical relationships
        mathematical_relationships = self._find_mathematical_relationships(
            analysis['fingerprints']
        )
        
        # Calculate knowledge sharing probability
        knowledge_sharing_prob = self._calculate_knowledge_sharing_probability(
            sophisticated_patterns, complexity_analysis, mathematical_relationships
        )
        
        return {
            'sophisticated_patterns': sophisticated_patterns,
            'complexity_analysis': complexity_analysis,
            'mathematical_relationships': mathematical_relationships,
            'knowledge_sharing_probability': knowledge_sharing_prob,
            'evidence_summary': self._generate_knowledge_sharing_evidence(
                sophisticated_patterns, complexity_analysis
            )
        }
    
    def generate_relationship_scoring(self, correlations: List[CorrelationResult]) -> Dict[str, Any]:
        """
        Generate relationship scoring and significance testing
        """
        if not correlations:
            return {'error': 'No correlations provided'}
        
        # Calculate score distribution
        scores = [corr.correlation_score for corr in correlations]
        score_stats = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'median': np.median(scores)
        }
        
        # Identify high-correlation pairs
        high_correlations = [
            corr for corr in correlations 
            if corr.correlation_score >= self.min_correlation_threshold
        ]
        
        # Calculate significance levels
        significance_results = []
        for corr in correlations:
            significance = self._perform_significance_test(corr)
            significance_results.append({
                'document_pair': f"{corr.document1_id}-{corr.document2_id}",
                'correlation_score': corr.correlation_score,
                'p_value': significance['p_value'],
                'is_significant': significance['is_significant'],
                'confidence_interval': significance['confidence_interval']
            })
        
        # Rank relationships by strength
        ranked_relationships = sorted(
            correlations, 
            key=lambda x: x.correlation_score, 
            reverse=True
        )
        
        return {
            'score_statistics': score_stats,
            'high_correlation_count': len(high_correlations),
            'significance_results': significance_results,
            'ranked_relationships': [
                {
                    'rank': i + 1,
                    'documents': f"{rel.document1_id}-{rel.document2_id}",
                    'score': rel.correlation_score,
                    'type': rel.correlation_type,
                    'shared_patterns': len(rel.shared_patterns)
                }
                for i, rel in enumerate(ranked_relationships[:10])  # Top 10
            ],
            'relationship_clusters': self._identify_relationship_clusters(correlations)
        }    
 
   # Helper methods for document analysis
    def _extract_document_text(self, document: Dict[str, Any]) -> str:
        """Extract text content from document"""
        text_parts = []
        
        # Extract from pages
        pages = document.get('pages', [])
        for page in pages:
            page_text = page.get('text', '')
            if page_text:
                text_parts.append(page_text)
        
        # Extract from direct text field
        if 'text' in document:
            text_parts.append(document['text'])
        
        return ' '.join(text_parts)
    
    def _extract_cipher_patterns(self, cipher_matches: List[CipherMatch]) -> Dict[str, int]:
        """Extract cipher pattern counts from matches"""
        patterns = defaultdict(int)
        
        for match in cipher_matches:
            cipher_type = match.cipher_type.value
            patterns[cipher_type] += 1
            
            # Add specific pattern details
            if cipher_type == 'caesar':
                shift = match.method_details.get('shift_amount', 0)
                patterns[f'caesar_shift_{shift}'] += 1
            elif cipher_type == 'skip_pattern':
                skip = match.method_details.get('skip_amount', 0)
                patterns[f'skip_{skip}'] += 1
        
        return dict(patterns)
    
    def _calculate_frequency_profile(self, text: str) -> Dict[str, float]:
        """Calculate letter frequency profile"""
        clean_text = ''.join(c.upper() for c in text if c.isalpha())
        
        if not clean_text:
            return {}
        
        letter_counts = Counter(clean_text)
        total_letters = len(clean_text)
        
        return {letter: count / total_letters for letter, count in letter_counts.items()}
    
    def _calculate_statistical_measures(self, text: str) -> Dict[str, float]:
        """Calculate statistical measures for text"""
        clean_text = ''.join(c.upper() for c in text if c.isalpha())
        
        if not clean_text:
            return {}
        
        # Index of Coincidence
        letter_counts = Counter(clean_text)
        n = len(clean_text)
        ic = sum(count * (count - 1) for count in letter_counts.values()) / (n * (n - 1)) if n > 1 else 0
        
        # Entropy
        frequencies = [count / n for count in letter_counts.values()]
        entropy = -sum(f * math.log2(f) for f in frequencies if f > 0)
        
        # Chi-squared against English
        english_freq = {
            'E': 0.127, 'T': 0.091, 'A': 0.082, 'O': 0.075, 'I': 0.070, 'N': 0.067,
            'S': 0.063, 'H': 0.061, 'R': 0.060, 'D': 0.043, 'L': 0.040, 'C': 0.028,
            'U': 0.028, 'M': 0.024, 'W': 0.024, 'F': 0.022, 'G': 0.020, 'Y': 0.020,
            'P': 0.019, 'B': 0.013, 'V': 0.010, 'K': 0.008, 'J': 0.002, 'X': 0.002,
            'Q': 0.001, 'Z': 0.001
        }
        
        chi_squared = 0
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            observed = letter_counts.get(letter, 0) / n
            expected = english_freq.get(letter, 0.001)
            chi_squared += ((observed - expected) ** 2) / expected
        
        return {
            'index_of_coincidence': ic,
            'entropy': entropy,
            'chi_squared': chi_squared,
            'text_length': len(clean_text),
            'unique_letters': len(letter_counts)
        }
    
    def _extract_anomaly_markers(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract anomaly markers from document"""
        anomalies = []
        
        # Check for page numbering anomalies
        pages = document.get('pages', [])
        if len(pages) > 1:
            page_numbers = [page.get('page_number', 0) for page in pages]
            for i in range(1, len(page_numbers)):
                if page_numbers[i] - page_numbers[i-1] != 1:
                    anomalies.append({
                        'type': 'page_numbering_gap',
                        'location': f"pages_{page_numbers[i-1]}_{page_numbers[i]}",
                        'severity': abs(page_numbers[i] - page_numbers[i-1] - 1)
                    })
        
        # Check for unusual character distributions
        text = self._extract_document_text(document)
        if text:
            char_dist = Counter(text.upper())
            total_chars = sum(char_dist.values())
            
            # Flag unusual letter frequencies
            for letter, count in char_dist.items():
                if letter.isalpha():
                    frequency = count / total_chars
                    if frequency > 0.2:  # Unusually high frequency
                        anomalies.append({
                            'type': 'unusual_letter_frequency',
                            'letter': letter,
                            'frequency': frequency,
                            'severity': frequency - 0.2
                        })
        
        return anomalies
    
    def _extract_geometric_patterns(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract geometric patterns from document"""
        patterns = []
        
        # Look for geometric analysis results
        if 'geometric_analysis' in document:
            geo_analysis = document['geometric_analysis']
            
            # Extract triangle patterns
            triangles = geo_analysis.get('triangular_constructions', [])
            for triangle in triangles:
                patterns.append({
                    'type': 'triangle',
                    'significance_score': triangle.get('significance_score', 0),
                    'properties': triangle.get('properties', {})
                })
            
            # Extract mathematical constants
            constants = geo_analysis.get('mathematical_constants', [])
            for constant in constants:
                patterns.append({
                    'type': 'mathematical_constant',
                    'name': constant.get('name', ''),
                    'confidence': constant.get('confidence', 0)
                })
        
        return patterns
    
    def _calculate_cipher_pattern_correlation(self, patterns1: Dict[str, int], 
                                           patterns2: Dict[str, int]) -> float:
        """Calculate correlation between cipher patterns"""
        if not patterns1 or not patterns2:
            return 0.0
        
        # Get all pattern types
        all_patterns = set(patterns1.keys()) | set(patterns2.keys())
        
        if not all_patterns:
            return 0.0
        
        # Create vectors
        vec1 = [patterns1.get(pattern, 0) for pattern in all_patterns]
        vec2 = [patterns2.get(pattern, 0) for pattern in all_patterns]
        
        # Calculate cosine similarity
        try:
            correlation = 1 - cosine(vec1, vec2)
            return max(0, correlation)  # Ensure non-negative
        except:
            return 0.0
    
    def _calculate_frequency_correlation(self, freq1: Dict[str, float], 
                                       freq2: Dict[str, float]) -> float:
        """Calculate correlation between frequency profiles"""
        if not freq1 or not freq2:
            return 0.0
        
        # Get all letters
        all_letters = set(freq1.keys()) | set(freq2.keys())
        
        if not all_letters:
            return 0.0
        
        # Create frequency vectors
        vec1 = [freq1.get(letter, 0) for letter in all_letters]
        vec2 = [freq2.get(letter, 0) for letter in all_letters]
        
        # Calculate Pearson correlation
        try:
            correlation, _ = pearsonr(vec1, vec2)
            return abs(correlation) if not math.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _calculate_statistical_correlation(self, stats1: Dict[str, float], 
                                         stats2: Dict[str, float]) -> float:
        """Calculate correlation between statistical measures"""
        if not stats1 or not stats2:
            return 0.0
        
        # Compare key statistical measures
        measures = ['index_of_coincidence', 'entropy', 'chi_squared']
        correlations = []
        
        for measure in measures:
            if measure in stats1 and measure in stats2:
                # Normalize difference to correlation score
                val1, val2 = stats1[measure], stats2[measure]
                max_val = max(abs(val1), abs(val2), 1)
                similarity = 1 - abs(val1 - val2) / max_val
                correlations.append(max(0, similarity))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_anomaly_correlation(self, anomalies1: List[Dict[str, Any]], 
                                     anomalies2: List[Dict[str, Any]]) -> float:
        """Calculate correlation between anomaly patterns"""
        if not anomalies1 or not anomalies2:
            return 0.0
        
        # Count anomaly types
        types1 = Counter(a.get('type', 'unknown') for a in anomalies1)
        types2 = Counter(a.get('type', 'unknown') for a in anomalies2)
        
        # Calculate overlap
        common_types = set(types1.keys()) & set(types2.keys())
        total_types = set(types1.keys()) | set(types2.keys())
        
        if not total_types:
            return 0.0
        
        # Jaccard similarity
        return len(common_types) / len(total_types)
    
    def _calculate_geometric_correlation(self, patterns1: List[Dict[str, Any]], 
                                       patterns2: List[Dict[str, Any]]) -> float:
        """Calculate correlation between geometric patterns"""
        if not patterns1 or not patterns2:
            return 0.0
        
        # Count pattern types
        types1 = Counter(p.get('type', 'unknown') for p in patterns1)
        types2 = Counter(p.get('type', 'unknown') for p in patterns2)
        
        # Calculate similarity
        common_types = set(types1.keys()) & set(types2.keys())
        total_types = set(types1.keys()) | set(types2.keys())
        
        if not total_types:
            return 0.0
        
        return len(common_types) / len(total_types)
    
    def _determine_correlation_type(self, cipher_corr: float, freq_corr: float, 
                                  stat_corr: float) -> str:
        """Determine the primary type of correlation"""
        correlations = {
            'cipher_based': cipher_corr,
            'frequency_based': freq_corr,
            'statistical_based': stat_corr
        }
        
        max_type = max(correlations.items(), key=lambda x: x[1])
        
        if max_type[1] > 0.7:
            return max_type[0]
        elif max_type[1] > 0.5:
            return f"moderate_{max_type[0]}"
        else:
            return "weak_correlation"
    
    def _find_shared_patterns(self, fp1: DocumentFingerprint, 
                            fp2: DocumentFingerprint) -> List[str]:
        """Find shared patterns between two fingerprints"""
        shared = []
        
        # Shared cipher patterns
        common_ciphers = set(fp1.cipher_patterns.keys()) & set(fp2.cipher_patterns.keys())
        shared.extend([f"cipher_{cipher}" for cipher in common_ciphers])
        
        # Shared anomaly types
        anomaly_types1 = {a.get('type') for a in fp1.anomaly_markers}
        anomaly_types2 = {a.get('type') for a in fp2.anomaly_markers}
        common_anomalies = anomaly_types1 & anomaly_types2
        shared.extend([f"anomaly_{anomaly}" for anomaly in common_anomalies])
        
        # Shared geometric patterns
        geo_types1 = {p.get('type') for p in fp1.geometric_patterns}
        geo_types2 = {p.get('type') for p in fp2.geometric_patterns}
        common_geo = geo_types1 & geo_types2
        shared.extend([f"geometric_{geo}" for geo in common_geo])
        
        return shared
    
    def _calculate_significance_level(self, correlation_score: float, 
                                    shared_pattern_count: int) -> float:
        """Calculate statistical significance level"""
        # Simple significance calculation based on correlation strength and evidence
        base_significance = 1 - correlation_score
        
        # Adjust based on number of shared patterns
        pattern_adjustment = max(0, 0.1 * (5 - shared_pattern_count))
        
        return min(1.0, base_significance + pattern_adjustment)
    
    def _generate_correlation_summary(self, correlations: List[CorrelationResult], 
                                    shared_schemes: List[SharedEncodingScheme]) -> Dict[str, Any]:
        """Generate summary of correlation analysis"""
        if not correlations:
            return {'message': 'No correlations found'}
        
        high_correlations = [c for c in correlations if c.correlation_score >= 0.7]
        moderate_correlations = [c for c in correlations if 0.4 <= c.correlation_score < 0.7]
        
        return {
            'total_correlations': len(correlations),
            'high_correlation_count': len(high_correlations),
            'moderate_correlation_count': len(moderate_correlations),
            'shared_scheme_count': len(shared_schemes),
            'average_correlation': np.mean([c.correlation_score for c in correlations]),
            'strongest_correlation': max(correlations, key=lambda x: x.correlation_score) if correlations else None,
            'correlation_types': Counter(c.correlation_type for c in correlations)
        }   
 
    # Additional helper methods for advanced analysis
    def _group_by_cipher_patterns(self, fingerprints: List[DocumentFingerprint]) -> Dict[str, List[DocumentFingerprint]]:
        """Group documents by similar cipher patterns"""
        groups = defaultdict(list)
        
        for fp in fingerprints:
            # Create a signature from cipher patterns
            pattern_signature = self._create_pattern_signature(fp.cipher_patterns)
            groups[pattern_signature].append(fp)
        
        return dict(groups)
    
    def _create_pattern_signature(self, cipher_patterns: Dict[str, int]) -> str:
        """Create a signature string from cipher patterns"""
        if not cipher_patterns:
            return "no_patterns"
        
        # Sort patterns by frequency and create signature
        sorted_patterns = sorted(cipher_patterns.items(), key=lambda x: x[1], reverse=True)
        signature_parts = []
        
        for pattern, count in sorted_patterns[:5]:  # Top 5 patterns
            signature_parts.append(f"{pattern}:{count}")
        
        return "|".join(signature_parts)
    
    def _classify_encoding_scheme(self, pattern_signature: str) -> str:
        """Classify the type of encoding scheme"""
        if "caesar" in pattern_signature:
            return "caesar_based"
        elif "substitution" in pattern_signature:
            return "substitution_based"
        elif "skip" in pattern_signature:
            return "skip_pattern_based"
        elif "biliteral" in pattern_signature:
            return "biliteral_based"
        elif "revolving" in pattern_signature:
            return "revolving_based"
        else:
            return "unknown_scheme"
    
    def _analyze_pattern_details(self, documents: List[DocumentFingerprint]) -> Dict[str, Any]:
        """Analyze detailed patterns across documents"""
        details = {
            'document_count': len(documents),
            'common_patterns': {},
            'pattern_variations': {},
            'consistency_score': 0.0
        }
        
        # Find common patterns
        all_patterns = set()
        for doc in documents:
            all_patterns.update(doc.cipher_patterns.keys())
        
        for pattern in all_patterns:
            counts = [doc.cipher_patterns.get(pattern, 0) for doc in documents]
            if sum(counts) > 0:
                details['common_patterns'][pattern] = {
                    'total_occurrences': sum(counts),
                    'document_frequency': sum(1 for c in counts if c > 0),
                    'average_per_document': np.mean(counts),
                    'consistency': 1 - (np.std(counts) / (np.mean(counts) + 1))
                }
        
        # Calculate overall consistency
        if details['common_patterns']:
            consistency_scores = [p['consistency'] for p in details['common_patterns'].values()]
            details['consistency_score'] = np.mean(consistency_scores)
        
        return details
    
    def _calculate_scheme_confidence(self, documents: List[DocumentFingerprint]) -> float:
        """Calculate confidence in shared encoding scheme"""
        if len(documents) < 2:
            return 0.0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = self._calculate_cipher_pattern_correlation(
                    documents[i].cipher_patterns,
                    documents[j].cipher_patterns
                )
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_evidence_strength(self, documents: List[DocumentFingerprint]) -> float:
        """Calculate strength of evidence for shared scheme"""
        # Evidence factors
        document_count_factor = min(1.0, len(documents) / 5)  # More documents = stronger
        
        # Pattern diversity factor
        all_patterns = set()
        for doc in documents:
            all_patterns.update(doc.cipher_patterns.keys())
        
        pattern_diversity_factor = min(1.0, len(all_patterns) / 10)
        
        # Statistical consistency factor
        statistical_consistency = self._calculate_statistical_consistency(documents)
        
        return (document_count_factor + pattern_diversity_factor + statistical_consistency) / 3
    
    def _calculate_statistical_consistency(self, documents: List[DocumentFingerprint]) -> float:
        """Calculate statistical consistency across documents"""
        if len(documents) < 2:
            return 0.0
        
        # Compare statistical measures
        measures = ['index_of_coincidence', 'entropy', 'chi_squared']
        consistencies = []
        
        for measure in measures:
            values = []
            for doc in documents:
                if measure in doc.statistical_measures:
                    values.append(doc.statistical_measures[measure])
            
            if len(values) >= 2:
                # Calculate coefficient of variation (lower = more consistent)
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / (abs(mean_val) + 1e-10)  # Avoid division by zero
                consistency = max(0, 1 - cv)
                consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 0.0
    
    def _identify_frequency_based_schemes(self, fingerprints: List[DocumentFingerprint]) -> List[SharedEncodingScheme]:
        """Identify shared schemes based on frequency analysis"""
        schemes = []
        
        # Group documents with similar frequency profiles
        frequency_groups = defaultdict(list)
        
        for fp in fingerprints:
            # Create frequency signature
            freq_signature = self._create_frequency_signature(fp.frequency_profile)
            frequency_groups[freq_signature].append(fp)
        
        for signature, docs in frequency_groups.items():
            if len(docs) >= 2:
                scheme = SharedEncodingScheme(
                    scheme_id=f"freq_scheme_{len(schemes) + 1}",
                    scheme_type="frequency_based",
                    documents=[fp.document_id for fp in docs],
                    pattern_details={
                        'frequency_signature': signature,
                        'document_count': len(docs),
                        'similarity_analysis': self._analyze_frequency_similarity(docs)
                    },
                    confidence=self._calculate_frequency_scheme_confidence(docs),
                    evidence_strength=self._calculate_frequency_evidence_strength(docs)
                )
                schemes.append(scheme)
        
        return schemes
    
    def _create_frequency_signature(self, frequency_profile: Dict[str, float]) -> str:
        """Create signature from frequency profile"""
        if not frequency_profile:
            return "no_frequency_data"
        
        # Get top 5 most frequent letters
        sorted_freq = sorted(frequency_profile.items(), key=lambda x: x[1], reverse=True)
        top_letters = [letter for letter, _ in sorted_freq[:5]]
        
        return "".join(top_letters)
    
    def _analyze_frequency_similarity(self, documents: List[DocumentFingerprint]) -> Dict[str, Any]:
        """Analyze frequency similarity across documents"""
        if len(documents) < 2:
            return {}
        
        # Calculate pairwise frequency correlations
        correlations = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                corr = self._calculate_frequency_correlation(
                    documents[i].frequency_profile,
                    documents[j].frequency_profile
                )
                correlations.append(corr)
        
        return {
            'average_correlation': np.mean(correlations),
            'min_correlation': np.min(correlations),
            'max_correlation': np.max(correlations),
            'correlation_consistency': 1 - np.std(correlations)
        }
    
    def _calculate_frequency_scheme_confidence(self, documents: List[DocumentFingerprint]) -> float:
        """Calculate confidence for frequency-based scheme"""
        similarity_analysis = self._analyze_frequency_similarity(documents)
        return similarity_analysis.get('average_correlation', 0.0)
    
    def _calculate_frequency_evidence_strength(self, documents: List[DocumentFingerprint]) -> float:
        """Calculate evidence strength for frequency-based scheme"""
        similarity_analysis = self._analyze_frequency_similarity(documents)
        
        # Factors: correlation strength, consistency, document count
        correlation_strength = similarity_analysis.get('average_correlation', 0.0)
        consistency = similarity_analysis.get('correlation_consistency', 0.0)
        document_factor = min(1.0, len(documents) / 5)
        
        return (correlation_strength + consistency + document_factor) / 3
    
    def _identify_anomaly_based_schemes(self, fingerprints: List[DocumentFingerprint]) -> List[SharedEncodingScheme]:
        """Identify shared schemes based on anomaly patterns"""
        schemes = []
        
        # Group documents by anomaly patterns
        anomaly_groups = defaultdict(list)
        
        for fp in fingerprints:
            anomaly_signature = self._create_anomaly_signature(fp.anomaly_markers)
            if anomaly_signature != "no_anomalies":
                anomaly_groups[anomaly_signature].append(fp)
        
        for signature, docs in anomaly_groups.items():
            if len(docs) >= 2:
                scheme = SharedEncodingScheme(
                    scheme_id=f"anomaly_scheme_{len(schemes) + 1}",
                    scheme_type="anomaly_based",
                    documents=[fp.document_id for fp in docs],
                    pattern_details={
                        'anomaly_signature': signature,
                        'shared_anomaly_analysis': self._analyze_shared_anomalies(docs)
                    },
                    confidence=self._calculate_anomaly_scheme_confidence(docs),
                    evidence_strength=self._calculate_anomaly_evidence_strength(docs)
                )
                schemes.append(scheme)
        
        return schemes
    
    def _create_anomaly_signature(self, anomaly_markers: List[Dict[str, Any]]) -> str:
        """Create signature from anomaly markers"""
        if not anomaly_markers:
            return "no_anomalies"
        
        # Count anomaly types
        anomaly_types = Counter(a.get('type', 'unknown') for a in anomaly_markers)
        
        # Create signature from most common anomalies
        signature_parts = []
        for anomaly_type, count in anomaly_types.most_common(3):
            signature_parts.append(f"{anomaly_type}:{count}")
        
        return "|".join(signature_parts)
    
    def _analyze_shared_anomalies(self, documents: List[DocumentFingerprint]) -> Dict[str, Any]:
        """Analyze shared anomalies across documents"""
        # Collect all anomaly types
        all_anomaly_types = set()
        for doc in documents:
            for anomaly in doc.anomaly_markers:
                all_anomaly_types.add(anomaly.get('type', 'unknown'))
        
        # Analyze each anomaly type
        shared_analysis = {}
        for anomaly_type in all_anomaly_types:
            docs_with_anomaly = []
            total_occurrences = 0
            
            for doc in documents:
                count = sum(1 for a in doc.anomaly_markers if a.get('type') == anomaly_type)
                if count > 0:
                    docs_with_anomaly.append(doc.document_id)
                    total_occurrences += count
            
            if len(docs_with_anomaly) >= 2:
                shared_analysis[anomaly_type] = {
                    'document_count': len(docs_with_anomaly),
                    'total_occurrences': total_occurrences,
                    'documents': docs_with_anomaly,
                    'prevalence': len(docs_with_anomaly) / len(documents)
                }
        
        return shared_analysis
    
    def _calculate_anomaly_scheme_confidence(self, documents: List[DocumentFingerprint]) -> float:
        """Calculate confidence for anomaly-based scheme"""
        shared_analysis = self._analyze_shared_anomalies(documents)
        
        if not shared_analysis:
            return 0.0
        
        # Calculate average prevalence of shared anomalies
        prevalences = [info['prevalence'] for info in shared_analysis.values()]
        return np.mean(prevalences)
    
    def _calculate_anomaly_evidence_strength(self, documents: List[DocumentFingerprint]) -> float:
        """Calculate evidence strength for anomaly-based scheme"""
        shared_analysis = self._analyze_shared_anomalies(documents)
        
        if not shared_analysis:
            return 0.0
        
        # Factors: number of shared anomaly types, prevalence, document count
        anomaly_type_factor = min(1.0, len(shared_analysis) / 5)
        prevalence_factor = np.mean([info['prevalence'] for info in shared_analysis.values()])
        document_factor = min(1.0, len(documents) / 5)
        
        return (anomaly_type_factor + prevalence_factor + document_factor) / 3
    
    def _calculate_anomaly_correlation_strength(self, occurrences: List[Dict[str, Any]]) -> float:
        """Calculate correlation strength for cross-document anomalies"""
        if len(occurrences) < 2:
            return 0.0
        
        # Simple correlation based on occurrence count and similarity
        document_count = len(occurrences)
        
        # Extract severity/significance values if available
        severities = []
        for occ in occurrences:
            anomaly = occ.get('anomaly', {})
            severity = anomaly.get('severity', 1.0)
            severities.append(severity)
        
        # Calculate consistency of severities
        if severities:
            severity_consistency = 1 - (np.std(severities) / (np.mean(severities) + 1e-10))
        else:
            severity_consistency = 0.5
        
        # Combine factors
        document_factor = min(1.0, document_count / 5)
        
        return (document_factor + severity_consistency) / 2
    
    def _analyze_anomaly_pattern(self, occurrences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze pattern details for cross-document anomaly"""
        pattern_details = {
            'occurrence_count': len(occurrences),
            'documents': [occ['document_id'] for occ in occurrences],
            'severity_analysis': {},
            'location_analysis': {}
        }
        
        # Analyze severities
        severities = []
        for occ in occurrences:
            anomaly = occ.get('anomaly', {})
            severity = anomaly.get('severity', 0)
            severities.append(severity)
        
        if severities:
            pattern_details['severity_analysis'] = {
                'mean': np.mean(severities),
                'std': np.std(severities),
                'min': np.min(severities),
                'max': np.max(severities)
            }
        
        # Analyze locations if available
        locations = []
        for occ in occurrences:
            anomaly = occ.get('anomaly', {})
            location = anomaly.get('location', '')
            if location:
                locations.append(location)
        
        if locations:
            pattern_details['location_analysis'] = {
                'unique_locations': len(set(locations)),
                'location_patterns': Counter(locations)
            }
        
        return pattern_details
    
    def _calculate_anomaly_significance(self, occurrences: List[Dict[str, Any]]) -> float:
        """Calculate statistical significance of cross-document anomaly"""
        # Simple significance calculation based on occurrence frequency
        document_count = len(occurrences)
        
        # More documents with same anomaly = higher significance
        base_significance = min(1.0, document_count / 10)
        
        # Adjust based on anomaly severity consistency
        severities = []
        for occ in occurrences:
            anomaly = occ.get('anomaly', {})
            severity = anomaly.get('severity', 0)
            severities.append(severity)
        
        if severities and len(severities) > 1:
            severity_consistency = 1 - (np.std(severities) / (np.mean(severities) + 1e-10))
            significance = (base_significance + severity_consistency) / 2
        else:
            significance = base_significance
        
        return min(1.0, significance)   
 
    # Advanced analysis methods
    def _is_sophisticated_scheme(self, scheme: SharedEncodingScheme) -> bool:
        """Determine if a scheme shows sophisticated cryptographic knowledge"""
        sophistication_indicators = 0
        
        # Multiple cipher types indicate sophistication
        if scheme.confidence > 0.8:
            sophistication_indicators += 1
        
        # Multiple documents using same scheme
        if len(scheme.documents) >= 3:
            sophistication_indicators += 1
        
        # Complex pattern types
        complex_types = ['revolving_based', 'biliteral_based', 'mathematical_constant']
        if scheme.scheme_type in complex_types:
            sophistication_indicators += 2
        
        # High evidence strength
        if scheme.evidence_strength > 0.7:
            sophistication_indicators += 1
        
        return sophistication_indicators >= 3
    
    def _analyze_cipher_complexity_progression(self, fingerprints: List[DocumentFingerprint]) -> Dict[str, Any]:
        """Analyze progression of cipher complexity across documents"""
        if len(fingerprints) < 2:
            return {'error': 'Need at least 2 documents for progression analysis'}
        
        # Calculate complexity scores for each document
        complexity_scores = []
        for fp in fingerprints:
            complexity = self._calculate_cipher_complexity(fp)
            complexity_scores.append({
                'document_id': fp.document_id,
                'complexity_score': complexity,
                'cipher_types': list(fp.cipher_patterns.keys())
            })
        
        # Sort by document ID (assuming chronological order)
        complexity_scores.sort(key=lambda x: x['document_id'])
        
        # Analyze progression
        scores = [cs['complexity_score'] for cs in complexity_scores]
        
        progression_analysis = {
            'document_complexities': complexity_scores,
            'complexity_trend': self._calculate_trend(scores),
            'average_complexity': np.mean(scores),
            'complexity_variance': np.var(scores),
            'progression_type': self._classify_progression(scores)
        }
        
        return progression_analysis
    
    def _calculate_cipher_complexity(self, fingerprint: DocumentFingerprint) -> float:
        """Calculate complexity score for a document's cipher patterns"""
        complexity_weights = {
            'caesar': 1.0,
            'atbash': 1.5,
            'substitution': 2.0,
            'skip_pattern': 2.5,
            'biliteral': 3.0,
            'revolving': 3.5
        }
        
        total_complexity = 0
        total_patterns = 0
        
        for pattern, count in fingerprint.cipher_patterns.items():
            base_pattern = pattern.split('_')[0]  # Remove specific parameters
            weight = complexity_weights.get(base_pattern, 1.0)
            total_complexity += weight * count
            total_patterns += count
        
        if total_patterns == 0:
            return 0.0
        
        # Normalize by pattern count and add diversity bonus
        base_complexity = total_complexity / total_patterns
        diversity_bonus = len(fingerprint.cipher_patterns) * 0.1
        
        return min(5.0, base_complexity + diversity_bonus)
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear regression slope
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _classify_progression(self, scores: List[float]) -> str:
        """Classify the type of complexity progression"""
        if len(scores) < 3:
            return "insufficient_data"
        
        # Calculate differences between consecutive scores
        diffs = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
        
        # Analyze pattern
        if all(d > 0 for d in diffs):
            return "consistent_increase"
        elif all(d < 0 for d in diffs):
            return "consistent_decrease"
        elif abs(max(diffs) - min(diffs)) < 0.5:
            return "stable_complexity"
        else:
            return "variable_complexity"
    
    def _find_mathematical_relationships(self, fingerprints: List[DocumentFingerprint]) -> List[Dict[str, Any]]:
        """Find mathematical relationships in cipher patterns"""
        relationships = []
        
        # Look for mathematical constants in geometric patterns
        for fp in fingerprints:
            for pattern in fp.geometric_patterns:
                if pattern.get('type') == 'mathematical_constant':
                    relationships.append({
                        'document_id': fp.document_id,
                        'type': 'mathematical_constant',
                        'constant_name': pattern.get('name', ''),
                        'confidence': pattern.get('confidence', 0)
                    })
        
        # Look for numerical patterns in cipher parameters
        for fp in fingerprints:
            for cipher_pattern, count in fp.cipher_patterns.items():
                if '_' in cipher_pattern:
                    parts = cipher_pattern.split('_')
                    if len(parts) >= 2 and parts[-1].isdigit():
                        parameter = int(parts[-1])
                        relationships.append({
                            'document_id': fp.document_id,
                            'type': 'cipher_parameter',
                            'cipher_type': parts[0],
                            'parameter_value': parameter,
                            'occurrences': count
                        })
        
        return relationships
    
    def _calculate_knowledge_sharing_probability(self, sophisticated_patterns: List[SharedEncodingScheme],
                                               complexity_analysis: Dict[str, Any],
                                               mathematical_relationships: List[Dict[str, Any]]) -> float:
        """Calculate probability of shared cryptographic knowledge"""
        probability_factors = []
        
        # Factor 1: Number of sophisticated shared patterns
        if sophisticated_patterns:
            sophistication_factor = min(1.0, len(sophisticated_patterns) / 3)
            probability_factors.append(sophistication_factor)
        
        # Factor 2: Complexity progression consistency
        if 'progression_type' in complexity_analysis:
            progression_type = complexity_analysis['progression_type']
            if progression_type in ['consistent_increase', 'stable_complexity']:
                probability_factors.append(0.8)
            elif progression_type == 'variable_complexity':
                probability_factors.append(0.6)
            else:
                probability_factors.append(0.3)
        
        # Factor 3: Mathematical relationships
        if mathematical_relationships:
            math_factor = min(1.0, len(mathematical_relationships) / 5)
            probability_factors.append(math_factor)
        
        # Factor 4: Cross-document consistency
        if 'complexity_variance' in complexity_analysis:
            variance = complexity_analysis['complexity_variance']
            consistency_factor = max(0, 1 - variance)
            probability_factors.append(consistency_factor)
        
        if not probability_factors:
            return 0.0
        
        return np.mean(probability_factors)
    
    def _generate_knowledge_sharing_evidence(self, sophisticated_patterns: List[SharedEncodingScheme],
                                           complexity_analysis: Dict[str, Any]) -> List[str]:
        """Generate evidence summary for knowledge sharing"""
        evidence = []
        
        if sophisticated_patterns:
            evidence.append(f"Found {len(sophisticated_patterns)} sophisticated shared encoding schemes")
            
            for pattern in sophisticated_patterns:
                evidence.append(f"Scheme '{pattern.scheme_type}' used across {len(pattern.documents)} documents")
        
        if 'progression_type' in complexity_analysis:
            progression = complexity_analysis['progression_type']
            if progression == 'consistent_increase':
                evidence.append("Cipher complexity shows consistent increase, suggesting learning progression")
            elif progression == 'stable_complexity':
                evidence.append("Stable cipher complexity suggests established knowledge base")
        
        if 'average_complexity' in complexity_analysis:
            avg_complexity = complexity_analysis['average_complexity']
            if avg_complexity > 2.5:
                evidence.append("High average cipher complexity indicates advanced cryptographic knowledge")
        
        return evidence
    
    def _perform_significance_test(self, correlation: CorrelationResult) -> Dict[str, Any]:
        """Perform statistical significance test on correlation"""
        # Simple significance test based on correlation strength and shared patterns
        correlation_score = correlation.correlation_score
        shared_pattern_count = len(correlation.shared_patterns)
        
        # Calculate p-value approximation
        # Higher correlation and more shared patterns = lower p-value
        # Make p-value lower (more significant) when correlation is high and evidence is present
        base_p_value = 1 - correlation_score
        evidence_bonus = 0.05 * min(shared_pattern_count, 5)
        p_value = max(0.001, base_p_value - evidence_bonus)
        
        # Determine significance
        is_significant = p_value < self.min_significance_level
        
        # Calculate confidence interval (simplified)
        margin_of_error = 0.1 * (1 - correlation_score)
        confidence_interval = [
            max(0, correlation_score - margin_of_error),
            min(1, correlation_score + margin_of_error)
        ]
        
        return {
            'p_value': p_value,
            'is_significant': is_significant,
            'confidence_interval': confidence_interval,
            'significance_level': self.min_significance_level
        }
    
    def _identify_relationship_clusters(self, correlations: List[CorrelationResult]) -> List[Dict[str, Any]]:
        """Identify clusters of related documents"""
        if not correlations:
            return []
        
        # Create document network
        document_connections = defaultdict(list)
        
        for corr in correlations:
            if corr.correlation_score >= self.min_correlation_threshold:
                document_connections[corr.document1_id].append(corr.document2_id)
                document_connections[corr.document2_id].append(corr.document1_id)
        
        # Find clusters using simple connected components
        clusters = []
        visited = set()
        
        for doc_id in document_connections:
            if doc_id not in visited:
                cluster = self._find_connected_component(doc_id, document_connections, visited)
                if len(cluster) >= 2:
                    clusters.append({
                        'cluster_id': len(clusters) + 1,
                        'documents': list(cluster),
                        'size': len(cluster),
                        'internal_correlations': self._calculate_internal_correlations(cluster, correlations)
                    })
        
        return clusters
    
    def _find_connected_component(self, start_doc: str, connections: Dict[str, List[str]], 
                                visited: Set[str]) -> Set[str]:
        """Find connected component starting from a document"""
        component = set()
        stack = [start_doc]
        
        while stack:
            doc = stack.pop()
            if doc not in visited:
                visited.add(doc)
                component.add(doc)
                
                # Add connected documents
                for connected_doc in connections.get(doc, []):
                    if connected_doc not in visited:
                        stack.append(connected_doc)
        
        return component
    
    def _calculate_internal_correlations(self, cluster: Set[str], 
                                       correlations: List[CorrelationResult]) -> Dict[str, float]:
        """Calculate internal correlation statistics for a cluster"""
        internal_corrs = []
        
        for corr in correlations:
            if corr.document1_id in cluster and corr.document2_id in cluster:
                internal_corrs.append(corr.correlation_score)
        
        if not internal_corrs:
            return {'average': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'average': np.mean(internal_corrs),
            'min': np.min(internal_corrs),
            'max': np.max(internal_corrs),
            'count': len(internal_corrs)
        }