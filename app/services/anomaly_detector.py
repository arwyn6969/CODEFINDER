"""
Statistical Anomaly Detection System
Implements automated detection of unusual patterns and potential encoding markers
in ancient texts through statistical analysis of character distributions, word patterns,
and punctuation variations.
"""
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import math
import numpy as np
from scipy import stats
from scipy.stats import chi2, zscore, iqr
import logging
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

from app.models.database_models import Document, Page, Pattern, Character, Word
from app.models.cross_document_models import CrossDocumentPattern, CrossPatternInstance
from app.services.text_analyzer import TextAnalyzer
from app.core.database import get_db


@dataclass
class AnomalyResult:
    """Represents a detected anomaly"""
    anomaly_id: str
    anomaly_type: str
    location: Dict[str, Any]
    severity: float
    confidence: float
    description: str
    statistical_measures: Dict[str, float]
    suggested_interpretations: List[str]
    evidence: Dict[str, Any]


@dataclass
class CharacterDistributionAnomaly:
    """Represents an anomaly in character distribution"""
    character: str
    expected_frequency: float
    observed_frequency: float
    deviation_score: float
    significance_level: float
    context: Dict[str, Any]


@dataclass
class WordPatternAnomaly:
    """Represents an anomaly in word patterns"""
    word: str
    pattern_type: str
    anomaly_score: float
    occurrences: List[Dict[str, Any]]
    statistical_significance: float
    context_analysis: Dict[str, Any]


@dataclass
class PunctuationAnomaly:
    """Represents an anomaly in punctuation"""
    punctuation_mark: str
    anomaly_type: str  # size_variation, positioning, frequency
    measurements: Dict[str, float]
    deviation_score: float
    locations: List[Dict[str, Any]]


@dataclass
class SignificanceTest:
    """Represents a statistical significance test result"""
    test_name: str
    statistic: float
    p_value: float
    critical_value: float
    is_significant: bool
    confidence_level: float
    interpretation: str


class AnomalyDetector:
    """
    Service for detecting statistical anomalies and potential encoding markers
    """
    
    def __init__(self, db_session: Session = None):
        self.db = db_session or next(get_db())
        self.text_analyzer = TextAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        # Statistical thresholds
        self.significance_threshold = 0.05
        self.anomaly_score_threshold = 2.0  # Z-score threshold
        self.frequency_deviation_threshold = 0.02
        
        # Expected English letter frequencies (for comparison)
        self.english_frequencies = {
            'E': 0.127, 'T': 0.091, 'A': 0.082, 'O': 0.075, 'I': 0.070, 'N': 0.067,
            'S': 0.063, 'H': 0.061, 'R': 0.060, 'D': 0.043, 'L': 0.040, 'C': 0.028,
            'U': 0.028, 'M': 0.024, 'W': 0.024, 'F': 0.022, 'G': 0.020, 'Y': 0.020,
            'P': 0.019, 'B': 0.013, 'V': 0.010, 'K': 0.008, 'J': 0.002, 'X': 0.002,
            'Q': 0.001, 'Z': 0.001
        }
    
    def detect_all_anomalies(self, document_id: int) -> Dict[str, Any]:
        """
        Detect all types of anomalies in a document
        """
        try:
            anomalies = {
                'document_id': document_id,
                'character_distribution_anomalies': [],
                'word_pattern_anomalies': [],
                'punctuation_anomalies': [],
                'numerical_pattern_anomalies': [],
                'composite_anomalies': [],
                'summary': {}
            }
            
            # Detect character distribution anomalies
            char_anomalies = self.detect_character_distribution_anomalies(document_id)
            anomalies['character_distribution_anomalies'] = char_anomalies
            
            # Detect word pattern anomalies
            word_anomalies = self.detect_word_pattern_anomalies(document_id)
            anomalies['word_pattern_anomalies'] = word_anomalies
            
            # Detect punctuation anomalies
            punct_anomalies = self.detect_punctuation_anomalies(document_id)
            anomalies['punctuation_anomalies'] = punct_anomalies
            
            # Detect numerical pattern anomalies
            numerical_anomalies = self.detect_numerical_pattern_anomalies(document_id)
            anomalies['numerical_pattern_anomalies'] = numerical_anomalies
            
            # Detect composite anomalies (combinations of the above)
            composite_anomalies = self.detect_composite_anomalies(
                char_anomalies, word_anomalies, punct_anomalies, numerical_anomalies
            )
            anomalies['composite_anomalies'] = composite_anomalies
            
            # Generate summary
            anomalies['summary'] = self._generate_anomaly_summary(anomalies)
            
            self.logger.info(f"Detected {len(char_anomalies + word_anomalies + punct_anomalies)} anomalies in document {document_id}")
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies in document {document_id}: {str(e)}")
            return {'error': str(e), 'document_id': document_id}
    
    def detect_character_distribution_anomalies(self, document_id: int) -> List[CharacterDistributionAnomaly]:
        """
        Detect anomalies in character frequency distributions
        """
        try:
            # Get document text
            document_text = self._get_document_text(document_id)
            if not document_text:
                return []
            
            # Calculate character frequencies
            char_frequencies = self._calculate_character_frequencies(document_text)
            
            anomalies = []
            
            # Compare against expected English frequencies
            for char, expected_freq in self.english_frequencies.items():
                observed_freq = char_frequencies.get(char, 0)
                
                # Calculate deviation score
                deviation_score = abs(observed_freq - expected_freq) / expected_freq if expected_freq > 0 else 0
                
                # Perform chi-square test
                expected_count = expected_freq * len(document_text)
                observed_count = observed_freq * len(document_text)
                
                if expected_count > 5:  # Chi-square test requirement
                    chi_stat = ((observed_count - expected_count) ** 2) / expected_count
                    p_value = 1 - stats.chi2.cdf(chi_stat, df=1)
                    
                    # Flag as anomaly if significant deviation
                    if p_value < self.significance_threshold and deviation_score > 0.5:
                        anomaly = CharacterDistributionAnomaly(
                            character=char,
                            expected_frequency=expected_freq,
                            observed_frequency=observed_freq,
                            deviation_score=deviation_score,
                            significance_level=p_value,
                            context={
                                'chi_square_statistic': chi_stat,
                                'expected_count': expected_count,
                                'observed_count': observed_count,
                                'deviation_type': 'overrepresented' if observed_freq > expected_freq else 'underrepresented'
                            }
                        )
                        anomalies.append(anomaly)
            
            # Check for unusual characters not in English alphabet
            unusual_chars = self._detect_unusual_characters(char_frequencies)
            for char, freq in unusual_chars.items():
                if freq > 0.001:  # More than 0.1% frequency
                    anomaly = CharacterDistributionAnomaly(
                        character=char,
                        expected_frequency=0.0,
                        observed_frequency=freq,
                        deviation_score=freq / 0.001,  # Relative to threshold
                        significance_level=0.001,
                        context={
                            'anomaly_type': 'unusual_character',
                            'character_code': ord(char),
                            'frequency_threshold': 0.001
                        }
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting character distribution anomalies: {str(e)}")
            return []    

    def detect_word_pattern_anomalies(self, document_id: int) -> List[WordPatternAnomaly]:
        """
        Detect anomalies in word patterns and placements
        """
        try:
            # Get document words
            words = self._get_document_words(document_id)
            if not words:
                return []
            
            anomalies = []
            
            # Analyze word frequency distribution
            word_freq_anomalies = self._detect_word_frequency_anomalies(words)
            anomalies.extend(word_freq_anomalies)
            
            # Analyze word positioning anomalies
            positioning_anomalies = self._detect_word_positioning_anomalies(words)
            anomalies.extend(positioning_anomalies)
            
            # Analyze word repetition patterns
            repetition_anomalies = self._detect_word_repetition_anomalies(words)
            anomalies.extend(repetition_anomalies)
            
            # Analyze unusual word lengths
            length_anomalies = self._detect_word_length_anomalies(words)
            anomalies.extend(length_anomalies)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting word pattern anomalies: {str(e)}")
            return []
    
    def detect_punctuation_anomalies(self, document_id: int) -> List[PunctuationAnomaly]:
        """
        Detect anomalies in punctuation size variations and positioning
        """
        try:
            # Get punctuation characters with position and size data
            punctuation_data = self._get_punctuation_data(document_id)
            if not punctuation_data:
                return []
            
            anomalies = []
            
            # Group by punctuation type
            punct_groups = defaultdict(list)
            for punct in punctuation_data:
                punct_groups[punct['character']].append(punct)
            
            # Analyze each punctuation type
            for punct_char, instances in punct_groups.items():
                if len(instances) < 3:  # Need minimum instances for statistical analysis
                    continue
                
                # Analyze size variations
                size_anomalies = self._detect_punctuation_size_anomalies(punct_char, instances)
                anomalies.extend(size_anomalies)
                
                # Analyze positioning anomalies
                position_anomalies = self._detect_punctuation_position_anomalies(punct_char, instances)
                anomalies.extend(position_anomalies)
                
                # Analyze frequency anomalies
                frequency_anomalies = self._detect_punctuation_frequency_anomalies(punct_char, instances)
                anomalies.extend(frequency_anomalies)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting punctuation anomalies: {str(e)}")
            return []
    
    def detect_numerical_pattern_anomalies(self, document_id: int) -> List[AnomalyResult]:
        """
        Detect anomalies in numerical patterns and mathematical relationships
        """
        try:
            anomalies = []
            
            # Get geometric analysis results
            geometric_patterns = self._get_geometric_patterns(document_id)
            
            # Analyze mathematical constant occurrences
            constant_anomalies = self._detect_mathematical_constant_anomalies(geometric_patterns)
            anomalies.extend(constant_anomalies)
            
            # Analyze ratio patterns
            ratio_anomalies = self._detect_ratio_pattern_anomalies(geometric_patterns)
            anomalies.extend(ratio_anomalies)
            
            # Analyze numerical sequences in text
            sequence_anomalies = self._detect_numerical_sequence_anomalies(document_id)
            anomalies.extend(sequence_anomalies)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting numerical pattern anomalies: {str(e)}")
            return []
    
    def detect_composite_anomalies(self, char_anomalies: List, word_anomalies: List, 
                                 punct_anomalies: List, numerical_anomalies: List) -> List[AnomalyResult]:
        """
        Detect composite anomalies that combine multiple types
        """
        try:
            composite_anomalies = []
            
            # Look for correlations between different anomaly types
            correlations = self._find_anomaly_correlations(
                char_anomalies, word_anomalies, punct_anomalies, numerical_anomalies
            )
            
            for correlation in correlations:
                if correlation['strength'] > 0.7:
                    composite_anomaly = AnomalyResult(
                        anomaly_id=f"composite_{len(composite_anomalies)}",
                        anomaly_type="composite",
                        location=correlation['location'],
                        severity=correlation['strength'],
                        confidence=correlation['confidence'],
                        description=f"Composite anomaly involving {', '.join(correlation['types'])}",
                        statistical_measures=correlation['statistics'],
                        suggested_interpretations=correlation['interpretations'],
                        evidence=correlation['evidence']
                    )
                    composite_anomalies.append(composite_anomaly)
            
            return composite_anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting composite anomalies: {str(e)}")
            return []
    
    def flag_potential_codes(self, anomalies: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Flag potential encoding schemes based on detected anomalies
        """
        try:
            potential_codes = []
            
            # Analyze anomaly patterns for encoding indicators
            encoding_indicators = self._analyze_encoding_indicators(anomalies)
            
            for indicator in encoding_indicators:
                if indicator['confidence'] > 0.6:
                    code_flag = {
                        'flag_id': f"code_{len(potential_codes)}",
                        'encoding_type': indicator['type'],
                        'confidence': indicator['confidence'],
                        'evidence': indicator['evidence'],
                        'location': indicator['location'],
                        'suggested_analysis': indicator['suggested_analysis'],
                        'priority': self._calculate_flag_priority(indicator)
                    }
                    potential_codes.append(code_flag)
            
            # Sort by priority
            potential_codes.sort(key=lambda x: x['priority'], reverse=True)
            
            return potential_codes
            
        except Exception as e:
            self.logger.error(f"Error flagging potential codes: {str(e)}")
            return []
    
    # Private helper methods
    
    def _get_document_text(self, document_id: int) -> str:
        """Get full text content of a document"""
        try:
            pages = self.db.query(Page).filter(Page.document_id == document_id).all()
            return ' '.join([page.text for page in pages if page.text])
        except:
            return ""
    
    def _calculate_character_frequencies(self, text: str) -> Dict[str, float]:
        """Calculate character frequency distribution"""
        if not text:
            return {}
        
        # Clean text and convert to uppercase
        clean_text = ''.join(c.upper() for c in text if c.isalpha())
        
        if not clean_text:
            return {}
        
        # Count characters
        char_counts = Counter(clean_text)
        total_chars = len(clean_text)
        
        # Convert to frequencies
        return {char: count / total_chars for char, count in char_counts.items()}
    
    def _detect_unusual_characters(self, char_frequencies: Dict[str, float]) -> Dict[str, float]:
        """Detect characters not typically found in English text"""
        unusual_chars = {}
        
        for char, freq in char_frequencies.items():
            if char not in self.english_frequencies and char.isalpha():
                unusual_chars[char] = freq
        
        return unusual_chars
    
    def _get_document_words(self, document_id: int) -> List[Dict[str, Any]]:
        """Get word data for a document"""
        try:
            words = self.db.query(Word).filter(Word.document_id == document_id).all()
            return [
                {
                    'text': word.text,
                    'frequency': word.frequency,
                    'positions': word.positions or [],
                    'page_number': word.page_number,
                    'confidence': getattr(word, 'confidence', 1.0)
                }
                for word in words
            ]
        except:
            return []
    
    def _detect_word_frequency_anomalies(self, words: List[Dict[str, Any]]) -> List[WordPatternAnomaly]:
        """Detect anomalies in word frequency distribution"""
        anomalies = []
        
        try:
            # Calculate frequency distribution
            frequencies = [word['frequency'] for word in words if word['frequency'] > 0]
            
            if len(frequencies) < 10:
                return anomalies
            
            # Calculate statistical measures
            mean_freq = np.mean(frequencies)
            std_freq = np.std(frequencies)
            
            # Detect outliers using Z-score
            for word in words:
                if word['frequency'] > 0:
                    z_score = (word['frequency'] - mean_freq) / std_freq if std_freq > 0 else 0
                    
                    if abs(z_score) > self.anomaly_score_threshold:
                        # Calculate significance
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                        
                        anomaly = WordPatternAnomaly(
                            word=word['text'],
                            pattern_type='frequency_outlier',
                            anomaly_score=abs(z_score),
                            occurrences=[word],
                            statistical_significance=p_value,
                            context_analysis={
                                'z_score': z_score,
                                'frequency': word['frequency'],
                                'mean_frequency': mean_freq,
                                'std_frequency': std_freq,
                                'outlier_type': 'high' if z_score > 0 else 'low'
                            }
                        )
                        anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting word frequency anomalies: {str(e)}")
            return []
    
    def _detect_word_positioning_anomalies(self, words: List[Dict[str, Any]]) -> List[WordPatternAnomaly]:
        """Detect anomalies in word positioning patterns"""
        anomalies = []
        
        try:
            # Group words by text
            word_groups = defaultdict(list)
            for word in words:
                word_groups[word['text']].append(word)
            
            # Analyze positioning patterns for repeated words
            for word_text, instances in word_groups.items():
                if len(instances) >= 3:  # Need multiple instances
                    # Analyze position intervals
                    positions = []
                    for instance in instances:
                        if instance['positions']:
                            positions.extend(instance['positions'])
                    
                    if len(positions) >= 3:
                        # Calculate intervals between occurrences
                        intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                        
                        # Check for regular patterns (potential cipher indicators)
                        if len(set(intervals)) == 1:  # All intervals are the same
                            anomaly = WordPatternAnomaly(
                                word=word_text,
                                pattern_type='regular_positioning',
                                anomaly_score=len(instances) / 3.0,  # Score based on repetition
                                occurrences=instances,
                                statistical_significance=0.01,  # High significance for regular patterns
                                context_analysis={
                                    'interval': intervals[0],
                                    'pattern_type': 'fixed_interval',
                                    'occurrence_count': len(instances),
                                    'positions': positions
                                }
                            )
                            anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting word positioning anomalies: {str(e)}")
            return []
    
    def _detect_word_repetition_anomalies(self, words: List[Dict[str, Any]]) -> List[WordPatternAnomaly]:
        """Detect unusual word repetition patterns"""
        anomalies = []
        
        try:
            # Calculate word frequency distribution
            word_counts = Counter(word['text'] for word in words)
            total_words = len(words)
            
            # Expected frequency for common words (Zipf's law approximation)
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            
            for i, (word_text, count) in enumerate(sorted_words[:100]):  # Top 100 words
                # Expected frequency based on rank (Zipf's law)
                expected_freq = 1 / (i + 1) * 0.1  # Rough approximation
                observed_freq = count / total_words
                
                # Check for significant deviation
                if observed_freq > expected_freq * 3:  # Much higher than expected
                    anomaly = WordPatternAnomaly(
                        word=word_text,
                        pattern_type='excessive_repetition',
                        anomaly_score=observed_freq / expected_freq,
                        occurrences=[w for w in words if w['text'] == word_text],
                        statistical_significance=0.05,
                        context_analysis={
                            'observed_frequency': observed_freq,
                            'expected_frequency': expected_freq,
                            'repetition_ratio': observed_freq / expected_freq,
                            'rank': i + 1,
                            'count': count
                        }
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting word repetition anomalies: {str(e)}")
            return []
    
    def _detect_word_length_anomalies(self, words: List[Dict[str, Any]]) -> List[WordPatternAnomaly]:
        """Detect anomalies in word length distribution"""
        anomalies = []
        
        try:
            # Calculate word lengths
            word_lengths = [len(word['text']) for word in words if word['text']]
            
            if len(word_lengths) < 10:
                return anomalies
            
            # Calculate statistical measures
            mean_length = np.mean(word_lengths)
            std_length = np.std(word_lengths)
            
            # Group words by length
            length_groups = defaultdict(list)
            for word in words:
                length_groups[len(word['text'])].append(word)
            
            # Check for unusual length distributions
            for length, word_list in length_groups.items():
                if length > mean_length + 3 * std_length:  # Unusually long words
                    if len(word_list) > 1:  # Multiple instances of unusually long words
                        anomaly = WordPatternAnomaly(
                            word=f"length_{length}_words",
                            pattern_type='unusual_length',
                            anomaly_score=(length - mean_length) / std_length,
                            occurrences=word_list,
                            statistical_significance=0.05,
                            context_analysis={
                                'word_length': length,
                                'mean_length': mean_length,
                                'std_length': std_length,
                                'instance_count': len(word_list),
                                'example_words': [w['text'] for w in word_list[:5]]
                            }
                        )
                        anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting word length anomalies: {str(e)}")
            return []
    
    def _get_punctuation_data(self, document_id: int) -> List[Dict[str, Any]]:
        """Get punctuation data with position and size information"""
        try:
            characters = self.db.query(Character).filter(
                and_(
                    Character.document_id == document_id,
                    Character.character.in_(['.', ',', ';', ':', '!', '?', '"', "'"])
                )
            ).all()
            
            return [
                {
                    'character': char.character,
                    'x': char.x,
                    'y': char.y,
                    'width': char.width,
                    'height': char.height,
                    'page_number': char.page_number,
                    'confidence': char.confidence
                }
                for char in characters
            ]
        except:
            return []
    
    def _detect_punctuation_size_anomalies(self, punct_char: str, 
                                         instances: List[Dict[str, Any]]) -> List[PunctuationAnomaly]:
        """Detect size variation anomalies in punctuation"""
        anomalies = []
        
        try:
            # Extract size measurements
            sizes = []
            for instance in instances:
                if instance['width'] and instance['height']:
                    size = instance['width'] * instance['height']  # Area as size measure
                    sizes.append(size)
            
            if len(sizes) < 3:
                return anomalies
            
            # Calculate statistical measures
            mean_size = np.mean(sizes)
            std_size = np.std(sizes)
            
            if std_size == 0:
                return anomalies
            
            # Calculate coefficient of variation
            cv = std_size / mean_size
            
            # Flag if variation is unusually high
            if cv > 0.3:  # 30% coefficient of variation threshold
                # Find specific outliers
                outlier_instances = []
                for i, instance in enumerate(instances):
                    if i < len(sizes):
                        z_score = (sizes[i] - mean_size) / std_size
                        if abs(z_score) > 2.0:
                            outlier_instances.append({
                                'instance': instance,
                                'size': sizes[i],
                                'z_score': z_score
                            })
                
                if outlier_instances:
                    anomaly = PunctuationAnomaly(
                        punctuation_mark=punct_char,
                        anomaly_type='size_variation',
                        measurements={
                            'mean_size': mean_size,
                            'std_size': std_size,
                            'coefficient_variation': cv,
                            'outlier_count': len(outlier_instances)
                        },
                        deviation_score=cv,
                        locations=[outlier['instance'] for outlier in outlier_instances]
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting punctuation size anomalies: {str(e)}")
            return []
    
    def _detect_punctuation_position_anomalies(self, punct_char: str, 
                                             instances: List[Dict[str, Any]]) -> List[PunctuationAnomaly]:
        """Detect positioning anomalies in punctuation"""
        anomalies = []
        
        try:
            # Extract positions
            positions = [(inst['x'], inst['y']) for inst in instances if inst['x'] and inst['y']]
            
            if len(positions) < 5:
                return anomalies
            
            # Analyze position clustering
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            # Check for unusual clustering in x-coordinates (vertical alignment)
            x_std = np.std(x_coords)
            x_mean = np.mean(x_coords)
            
            # Count instances with similar x-coordinates (potential vertical alignment)
            aligned_count = 0
            for x in x_coords:
                if abs(x - x_mean) < x_std * 0.1:  # Within 10% of std dev
                    aligned_count += 1
            
            alignment_ratio = aligned_count / len(x_coords)
            
            if alignment_ratio > 0.8:  # 80% of punctuation aligned
                anomaly = PunctuationAnomaly(
                    punctuation_mark=punct_char,
                    anomaly_type='vertical_alignment',
                    measurements={
                        'alignment_ratio': alignment_ratio,
                        'aligned_count': aligned_count,
                        'total_count': len(x_coords),
                        'x_mean': x_mean,
                        'x_std': x_std
                    },
                    deviation_score=alignment_ratio,
                    locations=instances
                )
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting punctuation position anomalies: {str(e)}")
            return []
    
    def _detect_punctuation_frequency_anomalies(self, punct_char: str, 
                                              instances: List[Dict[str, Any]]) -> List[PunctuationAnomaly]:
        """Detect frequency anomalies in punctuation usage"""
        anomalies = []
        
        try:
            # Expected frequencies for common punctuation marks
            expected_frequencies = {
                '.': 0.065,  # Period
                ',': 0.061,  # Comma
                ';': 0.003,  # Semicolon
                ':': 0.003,  # Colon
                '!': 0.001,  # Exclamation
                '?': 0.001,  # Question mark
                '"': 0.027,  # Quote
                "'": 0.024   # Apostrophe
            }
            
            expected_freq = expected_frequencies.get(punct_char, 0.001)
            
            # Calculate observed frequency (need total character count)
            # This is a simplified calculation - in practice, you'd get total from document
            total_chars = 1000  # Placeholder - should be actual document character count
            observed_freq = len(instances) / total_chars
            
            # Check for significant deviation
            if expected_freq > 0:
                deviation_ratio = observed_freq / expected_freq
                
                if deviation_ratio > 3.0 or deviation_ratio < 0.3:  # 3x higher or lower than expected
                    anomaly = PunctuationAnomaly(
                        punctuation_mark=punct_char,
                        anomaly_type='frequency_anomaly',
                        measurements={
                            'observed_frequency': observed_freq,
                            'expected_frequency': expected_freq,
                            'deviation_ratio': deviation_ratio,
                            'instance_count': len(instances)
                        },
                        deviation_score=abs(math.log(deviation_ratio)),
                        locations=instances
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting punctuation frequency anomalies: {str(e)}")
            return []   
 
    def _get_geometric_patterns(self, document_id: int) -> List[Dict[str, Any]]:
        """Get geometric patterns for numerical analysis"""
        try:
            patterns = self.db.query(Pattern).filter(
                and_(
                    Pattern.document_id == document_id,
                    Pattern.pattern_type == 'geometric'
                )
            ).all()
            
            return [
                {
                    'pattern_id': pattern.id,
                    'measurements': pattern.measurements or {},
                    'significance_score': pattern.significance_score,
                    'pattern_data': pattern.pattern_data or {}
                }
                for pattern in patterns
            ]
        except:
            return []
    
    def _detect_mathematical_constant_anomalies(self, geometric_patterns: List[Dict[str, Any]]) -> List[AnomalyResult]:
        """Detect anomalies in mathematical constant occurrences"""
        anomalies = []
        
        try:
            # Known mathematical constants
            constants = {
                'pi': math.pi,
                'e': math.e,
                'phi': (1 + math.sqrt(5)) / 2,  # Golden ratio
                'sqrt2': math.sqrt(2),
                'sqrt3': math.sqrt(3)
            }
            
            # Count occurrences of each constant
            constant_counts = defaultdict(int)
            constant_locations = defaultdict(list)
            
            for pattern in geometric_patterns:
                measurements = pattern.get('measurements', {})
                
                for key, value in measurements.items():
                    if isinstance(value, (int, float)):
                        # Check if value is close to any mathematical constant
                        for const_name, const_value in constants.items():
                            if abs(value - const_value) < 0.01:  # Within 1% tolerance
                                constant_counts[const_name] += 1
                                constant_locations[const_name].append({
                                    'pattern_id': pattern['pattern_id'],
                                    'measurement_key': key,
                                    'value': value,
                                    'expected': const_value,
                                    'deviation': abs(value - const_value)
                                })
            
            # Flag constants that appear unusually frequently
            for const_name, count in constant_counts.items():
                if count >= 3:  # 3 or more occurrences might be significant
                    # Calculate probability of random occurrence
                    random_prob = (0.01 ** count) * len(geometric_patterns)  # Very rough estimate
                    
                    if random_prob < 0.05:  # Unlikely to be random
                        anomaly = AnomalyResult(
                            anomaly_id=f"constant_{const_name}",
                            anomaly_type="mathematical_constant",
                            location={'constant_name': const_name, 'occurrences': constant_locations[const_name]},
                            severity=min(1.0, count / 10.0),
                            confidence=1.0 - random_prob,
                            description=f"Mathematical constant {const_name} appears {count} times",
                            statistical_measures={
                                'occurrence_count': count,
                                'random_probability': random_prob,
                                'constant_value': constants[const_name]
                            },
                            suggested_interpretations=[
                                f"Intentional use of {const_name} in geometric construction",
                                "Sacred geometry or mathematical encoding",
                                "Architectural or artistic proportion system"
                            ],
                            evidence={'locations': constant_locations[const_name]}
                        )
                        anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting mathematical constant anomalies: {str(e)}")
            return []
    
    def _detect_ratio_pattern_anomalies(self, geometric_patterns: List[Dict[str, Any]]) -> List[AnomalyResult]:
        """Detect anomalies in ratio patterns"""
        anomalies = []
        
        try:
            # Extract all ratios from geometric patterns
            ratios = []
            ratio_sources = []
            
            for pattern in geometric_patterns:
                measurements = pattern.get('measurements', {})
                
                # Look for ratio measurements
                for key, value in measurements.items():
                    if 'ratio' in key.lower() and isinstance(value, (int, float)):
                        ratios.append(value)
                        ratio_sources.append({
                            'pattern_id': pattern['pattern_id'],
                            'measurement_key': key,
                            'value': value
                        })
            
            if len(ratios) < 3:
                return anomalies
            
            # Analyze ratio distribution
            ratio_counts = Counter([round(r, 2) for r in ratios])  # Round to 2 decimal places
            
            # Find ratios that appear multiple times
            for ratio_value, count in ratio_counts.items():
                if count >= 3:  # Appears 3 or more times
                    # Calculate significance
                    total_ratios = len(ratios)
                    expected_freq = 1 / 100  # Assuming 100 possible ratio values (rough estimate)
                    observed_freq = count / total_ratios
                    
                    if observed_freq > expected_freq * 5:  # 5x more frequent than expected
                        # Find all instances of this ratio
                        instances = [source for source in ratio_sources if round(source['value'], 2) == ratio_value]
                        
                        anomaly = AnomalyResult(
                            anomaly_id=f"ratio_{ratio_value}",
                            anomaly_type="recurring_ratio",
                            location={'ratio_value': ratio_value, 'instances': instances},
                            severity=min(1.0, count / 10.0),
                            confidence=min(1.0, observed_freq / expected_freq / 5),
                            description=f"Ratio {ratio_value} appears {count} times",
                            statistical_measures={
                                'ratio_value': ratio_value,
                                'occurrence_count': count,
                                'observed_frequency': observed_freq,
                                'expected_frequency': expected_freq
                            },
                            suggested_interpretations=[
                                "Intentional use of specific proportional relationship",
                                "Geometric or architectural design principle",
                                "Encoded numerical pattern"
                            ],
                            evidence={'instances': instances}
                        )
                        anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting ratio pattern anomalies: {str(e)}")
            return []
    
    def _detect_numerical_sequence_anomalies(self, document_id: int) -> List[AnomalyResult]:
        """Detect anomalies in numerical sequences within text"""
        anomalies = []
        
        try:
            # Get document text
            text = self._get_document_text(document_id)
            if not text:
                return anomalies
            
            # Extract numbers from text
            import re
            numbers = re.findall(r'\b\d+\b', text)
            
            if len(numbers) < 5:
                return anomalies
            
            # Convert to integers
            number_values = [int(n) for n in numbers if n.isdigit()]
            
            # Look for arithmetic sequences
            sequence_anomalies = self._detect_arithmetic_sequences(number_values)
            anomalies.extend(sequence_anomalies)
            
            # Look for geometric sequences
            geometric_anomalies = self._detect_geometric_sequences(number_values)
            anomalies.extend(geometric_anomalies)
            
            # Look for Fibonacci-like sequences
            fibonacci_anomalies = self._detect_fibonacci_sequences(number_values)
            anomalies.extend(fibonacci_anomalies)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting numerical sequence anomalies: {str(e)}")
            return []
    
    def _detect_arithmetic_sequences(self, numbers: List[int]) -> List[AnomalyResult]:
        """Detect arithmetic sequences in number list"""
        anomalies = []
        
        try:
            # Look for sequences of at least 3 numbers
            for i in range(len(numbers) - 2):
                for length in range(3, min(8, len(numbers) - i + 1)):  # Max length 7
                    subsequence = numbers[i:i+length]
                    
                    # Check if it's an arithmetic sequence
                    if len(subsequence) >= 3:
                        differences = [subsequence[j+1] - subsequence[j] for j in range(len(subsequence)-1)]
                        
                        if len(set(differences)) == 1:  # All differences are the same
                            # Calculate probability of random occurrence
                            random_prob = (1 / 100) ** (length - 2)  # Rough estimate
                            
                            if random_prob < 0.1:  # Unlikely to be random
                                anomaly = AnomalyResult(
                                    anomaly_id=f"arithmetic_seq_{i}",
                                    anomaly_type="arithmetic_sequence",
                                    location={'start_index': i, 'sequence': subsequence},
                                    severity=min(1.0, length / 7.0),
                                    confidence=1.0 - random_prob,
                                    description=f"Arithmetic sequence: {subsequence}",
                                    statistical_measures={
                                        'sequence_length': length,
                                        'common_difference': differences[0],
                                        'random_probability': random_prob
                                    },
                                    suggested_interpretations=[
                                        "Intentional numerical pattern",
                                        "Page numbering or indexing system",
                                        "Encoded sequence"
                                    ],
                                    evidence={'sequence': subsequence, 'differences': differences}
                                )
                                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting arithmetic sequences: {str(e)}")
            return []
    
    def _detect_geometric_sequences(self, numbers: List[int]) -> List[AnomalyResult]:
        """Detect geometric sequences in number list"""
        anomalies = []
        
        try:
            # Look for sequences of at least 3 numbers
            for i in range(len(numbers) - 2):
                for length in range(3, min(6, len(numbers) - i + 1)):  # Max length 5 (geometric grows fast)
                    subsequence = numbers[i:i+length]
                    
                    # Check if it's a geometric sequence
                    if len(subsequence) >= 3 and all(n > 0 for n in subsequence):
                        ratios = [subsequence[j+1] / subsequence[j] for j in range(len(subsequence)-1)]
                        
                        # Check if all ratios are approximately equal
                        if len(set([round(r, 1) for r in ratios])) == 1:  # All ratios are the same (rounded)
                            ratio = ratios[0]
                            
                            # Skip trivial cases
                            if ratio != 1.0 and 0.1 < ratio < 10:
                                random_prob = (1 / 50) ** (length - 2)  # Rough estimate
                                
                                if random_prob < 0.1:
                                    anomaly = AnomalyResult(
                                        anomaly_id=f"geometric_seq_{i}",
                                        anomaly_type="geometric_sequence",
                                        location={'start_index': i, 'sequence': subsequence},
                                        severity=min(1.0, length / 5.0),
                                        confidence=1.0 - random_prob,
                                        description=f"Geometric sequence: {subsequence}",
                                        statistical_measures={
                                            'sequence_length': length,
                                            'common_ratio': ratio,
                                            'random_probability': random_prob
                                        },
                                        suggested_interpretations=[
                                            "Intentional exponential pattern",
                                            "Mathematical progression",
                                            "Encoded geometric relationship"
                                        ],
                                        evidence={'sequence': subsequence, 'ratios': ratios}
                                    )
                                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting geometric sequences: {str(e)}")
            return []
    
    def _detect_fibonacci_sequences(self, numbers: List[int]) -> List[AnomalyResult]:
        """Detect Fibonacci-like sequences in number list"""
        anomalies = []
        
        try:
            # Look for sequences where each number is sum of previous two
            for i in range(len(numbers) - 2):
                for length in range(3, min(8, len(numbers) - i + 1)):
                    subsequence = numbers[i:i+length]
                    
                    # Check if it follows Fibonacci pattern
                    is_fibonacci = True
                    for j in range(2, len(subsequence)):
                        if subsequence[j] != subsequence[j-1] + subsequence[j-2]:
                            is_fibonacci = False
                            break
                    
                    if is_fibonacci and length >= 4:  # At least 4 numbers for significance
                        random_prob = (1 / 1000) ** (length - 3)  # Very unlikely to be random
                        
                        anomaly = AnomalyResult(
                            anomaly_id=f"fibonacci_seq_{i}",
                            anomaly_type="fibonacci_sequence",
                            location={'start_index': i, 'sequence': subsequence},
                            severity=min(1.0, length / 7.0),
                            confidence=1.0 - random_prob,
                            description=f"Fibonacci-like sequence: {subsequence}",
                            statistical_measures={
                                'sequence_length': length,
                                'random_probability': random_prob
                            },
                            suggested_interpretations=[
                                "Intentional Fibonacci pattern",
                                "Mathematical or natural encoding",
                                "Sacred number sequence"
                            ],
                            evidence={'sequence': subsequence}
                        )
                        anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting Fibonacci sequences: {str(e)}")
            return []
    
    def _find_anomaly_correlations(self, char_anomalies: List, word_anomalies: List, 
                                 punct_anomalies: List, numerical_anomalies: List) -> List[Dict[str, Any]]:
        """Find correlations between different types of anomalies"""
        correlations = []
        
        try:
            # Simple correlation: check for co-occurrence in similar locations
            all_anomalies = [
                ('character', char_anomalies),
                ('word', word_anomalies),
                ('punctuation', punct_anomalies),
                ('numerical', numerical_anomalies)
            ]
            
            # Look for anomalies that occur in similar locations or contexts
            for i in range(len(all_anomalies)):
                for j in range(i + 1, len(all_anomalies)):
                    type1, anomalies1 = all_anomalies[i]
                    type2, anomalies2 = all_anomalies[j]
                    
                    # Find correlations between these two types
                    type_correlations = self._find_type_correlations(type1, anomalies1, type2, anomalies2)
                    correlations.extend(type_correlations)
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error finding anomaly correlations: {str(e)}")
            return []
    
    def _find_type_correlations(self, type1: str, anomalies1: List, type2: str, anomalies2: List) -> List[Dict[str, Any]]:
        """Find correlations between two specific anomaly types"""
        correlations = []
        
        try:
            # This is a simplified correlation analysis
            # In practice, you'd want more sophisticated spatial/temporal correlation
            
            if len(anomalies1) > 0 and len(anomalies2) > 0:
                # Simple correlation based on co-occurrence
                correlation_strength = min(len(anomalies1), len(anomalies2)) / max(len(anomalies1), len(anomalies2))
                
                if correlation_strength > 0.5:
                    correlation = {
                        'types': [type1, type2],
                        'strength': correlation_strength,
                        'confidence': 0.7,  # Default confidence
                        'location': {'type1_count': len(anomalies1), 'type2_count': len(anomalies2)},
                        'statistics': {
                            'correlation_coefficient': correlation_strength,
                            'type1_anomalies': len(anomalies1),
                            'type2_anomalies': len(anomalies2)
                        },
                        'interpretations': [
                            f"Correlation between {type1} and {type2} anomalies",
                            "Potential systematic encoding pattern",
                            "Multi-dimensional cipher or pattern"
                        ],
                        'evidence': {
                            'type1_examples': anomalies1[:3],  # First 3 examples
                            'type2_examples': anomalies2[:3]
                        }
                    }
                    correlations.append(correlation)
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error finding type correlations: {str(e)}")
            return []
    
    def _analyze_encoding_indicators(self, anomalies: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze anomalies for potential encoding scheme indicators"""
        indicators = []
        
        try:
            # Character distribution indicators
            char_anomalies = anomalies.get('character_distribution_anomalies', [])
            if len(char_anomalies) >= 3:
                indicators.append({
                    'type': 'substitution_cipher',
                    'confidence': min(1.0, len(char_anomalies) / 10.0),
                    'evidence': char_anomalies,
                    'location': {'anomaly_count': len(char_anomalies)},
                    'suggested_analysis': [
                        'Frequency analysis for substitution cipher',
                        'Character mapping analysis',
                        'Statistical cipher detection'
                    ]
                })
            
            # Word pattern indicators
            word_anomalies = anomalies.get('word_pattern_anomalies', [])
            positioning_anomalies = [a for a in word_anomalies if hasattr(a, 'pattern_type') and a.pattern_type == 'regular_positioning']
            
            if len(positioning_anomalies) >= 2:
                indicators.append({
                    'type': 'skip_cipher',
                    'confidence': min(1.0, len(positioning_anomalies) / 5.0),
                    'evidence': positioning_anomalies,
                    'location': {'positioning_anomalies': len(positioning_anomalies)},
                    'suggested_analysis': [
                        'Skip pattern analysis',
                        'Grid-based cipher detection',
                        'Interval pattern analysis'
                    ]
                })
            
            # Punctuation indicators
            punct_anomalies = anomalies.get('punctuation_anomalies', [])
            size_anomalies = [a for a in punct_anomalies if hasattr(a, 'anomaly_type') and a.anomaly_type == 'size_variation']
            
            if len(size_anomalies) >= 2:
                indicators.append({
                    'type': 'steganographic_encoding',
                    'confidence': min(1.0, len(size_anomalies) / 3.0),
                    'evidence': size_anomalies,
                    'location': {'size_anomalies': len(size_anomalies)},
                    'suggested_analysis': [
                        'Punctuation size analysis',
                        'Steganographic pattern detection',
                        'Visual encoding analysis'
                    ]
                })
            
            # Numerical pattern indicators
            numerical_anomalies = anomalies.get('numerical_pattern_anomalies', [])
            if len(numerical_anomalies) >= 2:
                indicators.append({
                    'type': 'mathematical_encoding',
                    'confidence': min(1.0, len(numerical_anomalies) / 4.0),
                    'evidence': numerical_anomalies,
                    'location': {'numerical_anomalies': len(numerical_anomalies)},
                    'suggested_analysis': [
                        'Mathematical constant analysis',
                        'Numerical sequence detection',
                        'Geometric encoding analysis'
                    ]
                })
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error analyzing encoding indicators: {str(e)}")
            return []
    
    def _calculate_flag_priority(self, indicator: Dict[str, Any]) -> float:
        """Calculate priority score for a potential code flag"""
        try:
            base_priority = indicator['confidence']
            
            # Boost priority based on encoding type
            type_multipliers = {
                'substitution_cipher': 1.2,
                'skip_cipher': 1.1,
                'steganographic_encoding': 1.3,
                'mathematical_encoding': 1.4
            }
            
            multiplier = type_multipliers.get(indicator['type'], 1.0)
            
            # Boost based on evidence count
            evidence_count = len(indicator.get('evidence', []))
            evidence_boost = min(0.3, evidence_count * 0.05)
            
            priority = (base_priority * multiplier) + evidence_boost
            
            return min(1.0, priority)
            
        except Exception as e:
            self.logger.error(f"Error calculating flag priority: {str(e)}")
            return 0.5
    
    def _generate_anomaly_summary(self, anomalies: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for detected anomalies"""
        try:
            summary = {
                'total_anomalies': 0,
                'anomaly_types': {},
                'severity_distribution': {'low': 0, 'medium': 0, 'high': 0},
                'confidence_distribution': {'low': 0, 'medium': 0, 'high': 0},
                'top_anomalies': [],
                'encoding_likelihood': 0.0
            }
            
            all_anomalies = []
            
            # Collect all anomalies
            for anomaly_type, anomaly_list in anomalies.items():
                if anomaly_type != 'summary' and isinstance(anomaly_list, list):
                    summary['anomaly_types'][anomaly_type] = len(anomaly_list)
                    summary['total_anomalies'] += len(anomaly_list)
                    all_anomalies.extend(anomaly_list)
            
            # Analyze severity and confidence distributions
            for anomaly in all_anomalies:
                # Severity distribution
                if hasattr(anomaly, 'severity'):
                    severity = anomaly.severity
                elif hasattr(anomaly, 'deviation_score'):
                    severity = min(1.0, anomaly.deviation_score / 3.0)
                else:
                    severity = 0.5
                
                if severity < 0.3:
                    summary['severity_distribution']['low'] += 1
                elif severity < 0.7:
                    summary['severity_distribution']['medium'] += 1
                else:
                    summary['severity_distribution']['high'] += 1
                
                # Confidence distribution
                if hasattr(anomaly, 'confidence'):
                    confidence = anomaly.confidence
                elif hasattr(anomaly, 'statistical_significance'):
                    confidence = 1.0 - anomaly.statistical_significance
                else:
                    confidence = 0.5
                
                if confidence < 0.3:
                    summary['confidence_distribution']['low'] += 1
                elif confidence < 0.7:
                    summary['confidence_distribution']['medium'] += 1
                else:
                    summary['confidence_distribution']['high'] += 1
            
            # Calculate encoding likelihood
            high_confidence_count = summary['confidence_distribution']['high']
            high_severity_count = summary['severity_distribution']['high']
            
            if summary['total_anomalies'] > 0:
                summary['encoding_likelihood'] = (
                    (high_confidence_count + high_severity_count) / 
                    (2 * summary['total_anomalies'])
                )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating anomaly summary: {str(e)}")
            return {'error': str(e)}
    
    def _detect_entropy_anomalies(self, document_id: int, text: str) -> List[AnomalyResult]:
        """Detect anomalies in text entropy"""
        try:
            anomalies = []
            
            # Calculate entropy
            entropy = self._calculate_entropy(text)
            
            # Get baseline model
            cipher_model = self.statistical_models.get('cipher_complexity')
            if not cipher_model:
                return []
            
            baseline = cipher_model.parameters['entropy']
            z_score = (entropy - baseline['mean']) / baseline['std'] if baseline['std'] > 0 else 0
            
            if abs(z_score) > self.anomaly_threshold:
                anomaly = AnomalyResult(
                    anomaly_type='text_entropy',
                    location={'document_id': document_id, 'metric': 'entropy'},
                    severity=min(1.0, abs(z_score) / 3.0),
                    confidence=self.calculate_anomaly_confidence(
                        {'z_score': z_score, 'sample_size': len(text)},
                        baseline
                    ),
                    statistical_significance=2 * (1 - stats.norm.cdf(abs(z_score))),
                    description=f"Text entropy ({entropy:.4f}) deviates from baseline ({baseline['mean']:.4f})",
                    evidence={
                        'observed_entropy': entropy,
                        'expected_entropy': baseline['mean'],
                        'z_score': z_score,
                        'interpretation': 'High entropy may indicate encryption' if entropy > baseline['mean'] else 'Low entropy may indicate simple cipher'
                    },
                    baseline_stats=baseline,
                    detected_at=datetime.now()
                )
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting entropy anomalies: {str(e)}")
            return []
    
    def _detect_frequency_analysis_anomalies(self, document_id: int, text: str) -> List[AnomalyResult]:
        """Detect anomalies in frequency analysis patterns"""
        try:
            anomalies = []
            
            # Calculate frequency distribution
            char_freq = self._calculate_character_frequencies(text)
            
            # Check for flat distribution (possible cipher)
            frequencies = list(char_freq.values())
            if frequencies:
                freq_std = np.std(frequencies)
                freq_mean = np.mean(frequencies)
                
                # Coefficient of variation
                cv = freq_std / freq_mean if freq_mean > 0 else 0
                
                # Low CV indicates flat distribution
                if cv < 0.5:  # Threshold for flat distribution
                    anomaly = AnomalyResult(
                        anomaly_type='flat_frequency_distribution',
                        location={'document_id': document_id, 'metric': 'frequency_distribution'},
                        severity=1.0 - cv,  # Lower CV = higher severity
                        confidence=0.8,
                        statistical_significance=0.05,
                        description=f"Unusually flat character frequency distribution (CV: {cv:.3f})",
                        evidence={
                            'coefficient_of_variation': cv,
                            'frequency_std': freq_std,
                            'frequency_mean': freq_mean,
                            'interpretation': 'May indicate substitution cipher'
                        },
                        baseline_stats={'expected_cv': 1.0},
                        detected_at=datetime.now()
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting frequency analysis anomalies: {str(e)}")
            return []
    
    def _detect_index_of_coincidence_anomalies(self, document_id: int, text: str) -> List[AnomalyResult]:
        """Detect anomalies in index of coincidence"""
        try:
            anomalies = []
            
            # Calculate index of coincidence
            ic = self._calculate_index_of_coincidence(text)
            
            # Get baseline model
            cipher_model = self.statistical_models.get('cipher_complexity')
            if cipher_model and 'index_of_coincidence' in cipher_model.parameters:
                baseline = cipher_model.parameters['index_of_coincidence']
                z_score = (ic - baseline['mean']) / baseline['std'] if baseline['std'] > 0 else 0
                
                if abs(z_score) > self.anomaly_threshold:
                    interpretation = ""
                    if ic < 0.04:
                        interpretation = "Very low IC suggests polyalphabetic cipher"
                    elif ic > 0.08:
                        interpretation = "High IC suggests monoalphabetic substitution or plaintext"
                    
                    anomaly = AnomalyResult(
                        anomaly_type='index_of_coincidence',
                        location={'document_id': document_id, 'metric': 'index_of_coincidence'},
                        severity=min(1.0, abs(z_score) / 3.0),
                        confidence=0.9,
                        statistical_significance=2 * (1 - stats.norm.cdf(abs(z_score))),
                        description=f"Index of coincidence ({ic:.4f}) deviates from baseline ({baseline['mean']:.4f})",
                        evidence={
                            'observed_ic': ic,
                            'expected_ic': baseline['mean'],
                            'z_score': z_score,
                            'interpretation': interpretation
                        },
                        baseline_stats=baseline,
                        detected_at=datetime.now()
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting index of coincidence anomalies: {str(e)}")
            return []
    
    def _detect_cipher_method_anomalies(self, document_id: int, text: str) -> List[AnomalyResult]:
        """Detect anomalies suggesting specific cipher methods"""
        try:
            anomalies = []
            
            # Use cipher detector to analyze text
            cipher_results = self.cipher_detector.analyze_text(text)
            
            for result in cipher_results:
                if result.get('confidence', 0) > 0.8:  # High confidence cipher detection
                    anomaly = AnomalyResult(
                        anomaly_type='cipher_method_detected',
                        location={'document_id': document_id, 'cipher_type': result.get('type', 'unknown')},
                        severity=result.get('confidence', 0),
                        confidence=result.get('confidence', 0),
                        statistical_significance=0.01,
                        description=f"Detected potential {result.get('type', 'unknown')} cipher (confidence: {result.get('confidence', 0):.3f})",
                        evidence={
                            'cipher_type': result.get('type', 'unknown'),
                            'detection_confidence': result.get('confidence', 0),
                            'evidence': result.get('evidence', {}),
                            'suggested_key': result.get('key', None)
                        },
                        baseline_stats={'cipher_detection_threshold': 0.8},
                        detected_at=datetime.now()
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting cipher method anomalies: {str(e)}")
            return []
    
    def _detect_key_pattern_anomalies(self, document_id: int, text: str) -> List[AnomalyResult]:
        """Detect anomalies in potential key patterns"""
        try:
            anomalies = []
            
            # Look for repeating patterns that might be keys
            pattern_lengths = [3, 4, 5, 6, 7, 8, 9, 10]  # Common key lengths
            
            for length in pattern_lengths:
                patterns = self._find_repeating_patterns(text, length)
                
                for pattern, occurrences in patterns.items():
                    if len(occurrences) >= 3:  # Pattern repeats at least 3 times
                        # Calculate distances between occurrences
                        distances = []
                        for i in range(len(occurrences) - 1):
                            distances.append(occurrences[i+1] - occurrences[i])
                        
                        # Check if distances have common factors (suggesting key length)
                        if distances:
                            gcd_distance = distances[0]
                            for d in distances[1:]:
                                gcd_distance = math.gcd(gcd_distance, d)
                            
                            if gcd_distance > 1 and gcd_distance <= 20:  # Reasonable key length
                                anomaly = AnomalyResult(
                                    anomaly_type='repeating_key_pattern',
                                    location={'document_id': document_id, 'pattern': pattern},
                                    severity=min(1.0, len(occurrences) / 10),
                                    confidence=0.7,
                                    statistical_significance=0.05,
                                    description=f"Repeating pattern '{pattern}' suggests key length {gcd_distance}",
                                    evidence={
                                        'pattern': pattern,
                                        'pattern_length': length,
                                        'occurrences': len(occurrences),
                                        'positions': occurrences,
                                        'distances': distances,
                                        'suggested_key_length': gcd_distance
                                    },
                                    baseline_stats={'min_repetitions': 3},
                                    detected_at=datetime.now()
                                )
                                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting key pattern anomalies: {str(e)}")
            return []
    
    # Geometric anomaly detection methods
    
    def _detect_spatial_distribution_anomalies(self, document_id: int, geometric_patterns: List) -> List[AnomalyResult]:
        """Detect anomalies in spatial distribution of geometric patterns"""
        try:
            anomalies = []
            
            if not geometric_patterns:
                return []
            
            # Extract coordinates
            coordinates = []
            for pattern in geometric_patterns:
                if hasattr(pattern, 'coordinates') and pattern.coordinates:
                    coordinates.extend(pattern.coordinates)
            
            if len(coordinates) < 10:  # Need sufficient points
                return []
            
            # Convert to numpy array
            coords_array = np.array(coordinates)
            
            # Detect clustering using statistical methods
            outlier_indices, outlier_scores = self.detect_outliers_multivariate(coords_array)
            
            for idx, score in zip(outlier_indices, outlier_scores):
                anomaly = AnomalyResult(
                    anomaly_type='spatial_outlier',
                    location={'document_id': document_id, 'coordinates': coordinates[idx]},
                    severity=min(1.0, score),
                    confidence=0.8,
                    statistical_significance=0.05,
                    description=f"Geometric point at {coordinates[idx]} is spatially isolated",
                    evidence={
                        'coordinates': coordinates[idx],
                        'outlier_score': score,
                        'total_points': len(coordinates)
                    },
                    baseline_stats={'expected_clustering': 'normal'},
                    detected_at=datetime.now()
                )
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting spatial distribution anomalies: {str(e)}")
            return []
    
    def _detect_sacred_geometry_anomalies(self, document_id: int, geometric_patterns: List) -> List[AnomalyResult]:
        """Detect anomalies related to sacred geometry patterns"""
        try:
            anomalies = []
            
            # Use geometric analyzer to detect sacred geometry
            sacred_patterns = self.geometric_analyzer.detect_sacred_geometry_patterns(geometric_patterns)
            
            for pattern in sacred_patterns:
                if pattern.get('significance', 0) > 0.8:  # High significance
                    anomaly = AnomalyResult(
                        anomaly_type='sacred_geometry',
                        location={'document_id': document_id, 'pattern_type': pattern.get('type', 'unknown')},
                        severity=pattern.get('significance', 0),
                        confidence=pattern.get('confidence', 0),
                        statistical_significance=0.01,
                        description=f"Detected sacred geometry pattern: {pattern.get('type', 'unknown')}",
                        evidence={
                            'pattern_type': pattern.get('type', 'unknown'),
                            'significance': pattern.get('significance', 0),
                            'measurements': pattern.get('measurements', {}),
                            'mathematical_constants': pattern.get('constants', [])
                        },
                        baseline_stats={'sacred_geometry_threshold': 0.8},
                        detected_at=datetime.now()
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting sacred geometry anomalies: {str(e)}")
            return []
    
    def _detect_symmetry_anomalies(self, document_id: int, geometric_patterns: List) -> List[AnomalyResult]:
        """Detect anomalies in symmetry patterns"""
        try:
            anomalies = []
            
            # Analyze symmetry
            symmetry_analysis = self.geometric_analyzer.analyze_symmetry(geometric_patterns)
            
            if symmetry_analysis.get('perfect_symmetry', False):
                anomaly = AnomalyResult(
                    anomaly_type='perfect_symmetry',
                    location={'document_id': document_id, 'symmetry_type': symmetry_analysis.get('type', 'unknown')},
                    severity=0.9,
                    confidence=symmetry_analysis.get('confidence', 0),
                    statistical_significance=0.001,
                    description=f"Perfect {symmetry_analysis.get('type', 'unknown')} symmetry detected",
                    evidence={
                        'symmetry_type': symmetry_analysis.get('type', 'unknown'),
                        'symmetry_score': symmetry_analysis.get('score', 0),
                        'axis_of_symmetry': symmetry_analysis.get('axis', None),
                        'deviation': symmetry_analysis.get('deviation', 0)
                    },
                    baseline_stats={'perfect_symmetry_threshold': 0.99},
                    detected_at=datetime.now()
                )
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting symmetry anomalies: {str(e)}")
            return []
    
    def _detect_mathematical_constant_anomalies(self, document_id: int, geometric_patterns: List) -> List[AnomalyResult]:
        """Detect anomalies related to mathematical constants"""
        try:
            anomalies = []
            
            # Check for mathematical constants in measurements
            constants = {
                'pi': math.pi,
                'phi': (1 + math.sqrt(5)) / 2,  # Golden ratio
                'e': math.e,
                'sqrt2': math.sqrt(2),
                'sqrt3': math.sqrt(3),
                'sqrt5': math.sqrt(5)
            }
            
            tolerance = 0.01  # 1% tolerance
            
            for pattern in geometric_patterns:
                if hasattr(pattern, 'measurements') and pattern.measurements:
                    measurements = pattern.measurements
                    
                    for measurement_name, value in measurements.items():
                        if isinstance(value, (int, float)):
                            for const_name, const_value in constants.items():
                                # Check if measurement is close to constant
                                if abs(value - const_value) / const_value < tolerance:
                                    anomaly = AnomalyResult(
                                        anomaly_type='mathematical_constant',
                                        location={
                                            'document_id': document_id,
                                            'pattern_id': getattr(pattern, 'id', None),
                                            'measurement': measurement_name
                                        },
                                        severity=0.8,
                                        confidence=1.0 - (abs(value - const_value) / const_value),
                                        statistical_significance=0.01,
                                        description=f"Measurement '{measurement_name}' ({value:.6f}) approximates {const_name} ({const_value:.6f})",
                                        evidence={
                                            'measurement_name': measurement_name,
                                            'observed_value': value,
                                            'constant_name': const_name,
                                            'constant_value': const_value,
                                            'difference': abs(value - const_value),
                                            'relative_error': abs(value - const_value) / const_value
                                        },
                                        baseline_stats={'tolerance': tolerance},
                                        detected_at=datetime.now()
                                    )
                                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting mathematical constant anomalies: {str(e)}")
            return []
    
    # Statistical testing methods
    
    def _perform_normality_test(self, data: np.ndarray) -> SignificanceTest:
        """Perform normality test on data"""
        try:
            # Use Shapiro-Wilk test for small samples, Anderson-Darling for larger
            if len(data) <= 5000:
                statistic, p_value = shapiro(data)
                test_name = "Shapiro-Wilk"
                critical_value = 0.05  # Standard significance level
            else:
                result = anderson(data, dist='norm')
                statistic = result.statistic
                p_value = 0.05 if statistic > result.critical_values[2] else 0.1  # Approximate p-value
                test_name = "Anderson-Darling"
                critical_value = result.critical_values[2]  # 5% significance level
            
            is_significant = p_value < self.significance_level
            
            interpretation = "Data appears normally distributed" if not is_significant else "Data deviates from normal distribution"
            
            return SignificanceTest(
                test_name=test_name,
                statistic=statistic,
                p_value=p_value,
                critical_value=critical_value,
                is_significant=is_significant,
                confidence_level=self.confidence_level,
                interpretation=interpretation
            )
            
        except Exception as e:
            return SignificanceTest(
                test_name="normality",
                statistic=0.0,
                p_value=1.0,
                critical_value=0.0,
                is_significant=False,
                confidence_level=self.confidence_level,
                interpretation=f"Error in normality test: {str(e)}"
            )
    
    def _perform_outlier_test(self, data: np.ndarray) -> SignificanceTest:
        """Perform outlier detection test"""
        try:
            # Use modified Z-score method
            median = np.median(data)
            mad = np.median(np.abs(data - median))  # Median Absolute Deviation
            
            if mad == 0:
                mad = np.std(data)  # Fallback to standard deviation
            
            modified_z_scores = 0.6745 * (data - median) / mad
            outlier_threshold = 3.5
            
            outliers = np.abs(modified_z_scores) > outlier_threshold
            outlier_count = np.sum(outliers)
            
            # Calculate test statistic as proportion of outliers
            statistic = outlier_count / len(data)
            
            # Approximate p-value based on expected outlier rate
            expected_outlier_rate = 0.01  # 1% expected outliers
            p_value = 1.0 - stats.poisson.cdf(outlier_count, expected_outlier_rate * len(data))
            
            is_significant = p_value < self.significance_level
            
            interpretation = f"Found {outlier_count} outliers ({statistic:.3f} of data)"
            if is_significant:
                interpretation += " - significantly more than expected"
            
            return SignificanceTest(
                test_name="Modified Z-Score Outlier Test",
                statistic=statistic,
                p_value=p_value,
                critical_value=outlier_threshold,
                is_significant=is_significant,
                confidence_level=self.confidence_level,
                interpretation=interpretation
            )
            
        except Exception as e:
            return SignificanceTest(
                test_name="outlier",
                statistic=0.0,
                p_value=1.0,
                critical_value=0.0,
                is_significant=False,
                confidence_level=self.confidence_level,
                interpretation=f"Error in outlier test: {str(e)}"
            )
    
    def _perform_randomness_test(self, data: np.ndarray) -> SignificanceTest:
        """Perform randomness test using runs test"""
        try:
            # Convert to binary based on median
            median = np.median(data)
            binary_data = (data > median).astype(int)
            
            # Count runs
            runs = 1
            for i in range(1, len(binary_data)):
                if binary_data[i] != binary_data[i-1]:
                    runs += 1
            
            # Calculate expected runs and variance
            n1 = np.sum(binary_data)  # Number of 1s
            n2 = len(binary_data) - n1  # Number of 0s
            
            if n1 == 0 or n2 == 0:
                return SignificanceTest(
                    test_name="Runs Test",
                    statistic=0.0,
                    p_value=1.0,
                    critical_value=0.0,
                    is_significant=False,
                    confidence_level=self.confidence_level,
                    interpretation="Cannot perform runs test - data is constant"
                )
            
            expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
            variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
            
            # Calculate Z-score
            if variance_runs > 0:
                z_score = (runs - expected_runs) / math.sqrt(variance_runs)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                z_score = 0
                p_value = 1.0
            
            is_significant = p_value < self.significance_level
            
            interpretation = f"Observed {runs} runs, expected {expected_runs:.1f}"
            if is_significant:
                interpretation += " - data shows non-random pattern"
            else:
                interpretation += " - data appears random"
            
            return SignificanceTest(
                test_name="Runs Test",
                statistic=z_score,
                p_value=p_value,
                critical_value=1.96,  # 95% confidence
                is_significant=is_significant,
                confidence_level=self.confidence_level,
                interpretation=interpretation
            )
            
        except Exception as e:
            return SignificanceTest(
                test_name="randomness",
                statistic=0.0,
                p_value=1.0,
                critical_value=0.0,
                is_significant=False,
                confidence_level=self.confidence_level,
                interpretation=f"Error in randomness test: {str(e)}"
            )
    
    def _perform_independence_test(self, data: np.ndarray) -> SignificanceTest:
        """Perform independence test using autocorrelation"""
        try:
            # Calculate lag-1 autocorrelation
            if len(data) < 2:
                return SignificanceTest(
                    test_name="Autocorrelation Test",
                    statistic=0.0,
                    p_value=1.0,
                    critical_value=0.0,
                    is_significant=False,
                    confidence_level=self.confidence_level,
                    interpretation="Insufficient data for independence test"
                )
            
            # Standardize data
            data_std = (data - np.mean(data)) / np.std(data)
            
            # Calculate autocorrelation at lag 1
            autocorr = np.corrcoef(data_std[:-1], data_std[1:])[0, 1]
            
            # Test statistic (approximate)
            n = len(data)
            se = 1 / math.sqrt(n)  # Standard error for autocorrelation
            z_score = autocorr / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            is_significant = p_value < self.significance_level
            
            interpretation = f"Lag-1 autocorrelation: {autocorr:.3f}"
            if is_significant:
                interpretation += " - data shows significant dependence"
            else:
                interpretation += " - data appears independent"
            
            return SignificanceTest(
                test_name="Autocorrelation Test",
                statistic=z_score,
                p_value=p_value,
                critical_value=1.96,  # 95% confidence
                is_significant=is_significant,
                confidence_level=self.confidence_level,
                interpretation=interpretation
            )
            
        except Exception as e:
            return SignificanceTest(
                test_name="independence",
                statistic=0.0,
                p_value=1.0,
                critical_value=0.0,
                is_significant=False,
                confidence_level=self.confidence_level,
                interpretation=f"Error in independence test: {str(e)}"
            )
    
    def _perform_stationarity_test(self, data: np.ndarray) -> SignificanceTest:
        """Perform stationarity test using variance ratio"""
        try:
            if len(data) < 10:
                return SignificanceTest(
                    test_name="Stationarity Test",
                    statistic=0.0,
                    p_value=1.0,
                    critical_value=0.0,
                    is_significant=False,
                    confidence_level=self.confidence_level,
                    interpretation="Insufficient data for stationarity test"
                )
            
            # Split data into two halves
            mid = len(data) // 2
            first_half = data[:mid]
            second_half = data[mid:]
            
            # Calculate variances
            var1 = np.var(first_half, ddof=1)
            var2 = np.var(second_half, ddof=1)
            
            if var2 == 0:
                return SignificanceTest(
                    test_name="Stationarity Test",
                    statistic=0.0,
                    p_value=1.0,
                    critical_value=0.0,
                    is_significant=False,
                    confidence_level=self.confidence_level,
                    interpretation="Cannot test stationarity - zero variance in second half"
                )
            
            # F-test for equal variances
            f_statistic = var1 / var2 if var1 > var2 else var2 / var1
            df1 = len(first_half) - 1
            df2 = len(second_half) - 1
            
            p_value = 2 * (1 - stats.f.cdf(f_statistic, df1, df2))
            
            is_significant = p_value < self.significance_level
            
            interpretation = f"Variance ratio: {f_statistic:.3f}"
            if is_significant:
                interpretation += " - data shows non-stationarity"
            else:
                interpretation += " - data appears stationary"
            
            return SignificanceTest(
                test_name="Variance Ratio Test",
                statistic=f_statistic,
                p_value=p_value,
                critical_value=stats.f.ppf(0.975, df1, df2),  # 95% confidence
                is_significant=is_significant,
                confidence_level=self.confidence_level,
                interpretation=interpretation
            )
            
        except Exception as e:
            return SignificanceTest(
                test_name="stationarity",
                statistic=0.0,
                p_value=1.0,
                critical_value=0.0,
                is_significant=False,
                confidence_level=self.confidence_level,
                interpretation=f"Error in stationarity test: {str(e)}"
            )
    
    # Utility methods
    
    def _get_document_text(self, document_id: int) -> str:
        """Get text content for a document"""
        try:
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if document and document.content:
                return document.content
            return ""
        except Exception as e:
            self.logger.error(f"Error getting document text: {str(e)}")
            return ""
    
    def _get_geometric_patterns(self, document_id: int) -> List:
        """Get geometric patterns for a document"""
        try:
            # This would typically query a geometric patterns table
            # For now, return empty list as placeholder
            return []
        except Exception as e:
            self.logger.error(f"Error getting geometric patterns: {str(e)}")
            return []
    
    def _calculate_character_frequencies(self, text: str) -> Dict[str, float]:
        """Calculate character frequency distribution"""
        if not text:
            return {}
        
        char_counts = Counter(text.lower())
        total_chars = len(text)
        
        return {char: count / total_chars for char, count in char_counts.items()}
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        char_freq = self._calculate_character_frequencies(text)
        entropy = 0.0
        
        for freq in char_freq.values():
            if freq > 0:
                entropy -= freq * math.log2(freq)
        
        return entropy
    
    def _calculate_index_of_coincidence(self, text: str) -> float:
        """Calculate index of coincidence"""
        if len(text) < 2:
            return 0.0
        
        # Count only alphabetic characters
        alpha_text = ''.join(c.upper() for c in text if c.isalpha())
        if len(alpha_text) < 2:
            return 0.0
        
        char_counts = Counter(alpha_text)
        n = len(alpha_text)
        
        ic = sum(count * (count - 1) for count in char_counts.values()) / (n * (n - 1))
        return ic
    
    def _calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity score"""
        if not text:
            return 0.0
        
        # Simple complexity measure based on various factors
        factors = []
        
        # Vocabulary diversity
        words = text.split()
        if words:
            unique_words = len(set(words))
            vocab_diversity = unique_words / len(words)
            factors.append(vocab_diversity)
        
        # Character diversity
        unique_chars = len(set(text.lower()))
        char_diversity = unique_chars / 26  # Normalize to alphabet size
        factors.append(min(1.0, char_diversity))
        
        # Entropy factor
        entropy = self._calculate_entropy(text)
        entropy_factor = entropy / 5.0  # Normalize (max entropy ~5 for English)
        factors.append(min(1.0, entropy_factor))
        
        return np.mean(factors) if factors else 0.0
    
    def _calculate_mahalanobis_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate Mahalanobis distance for each point"""
        try:
            mean = np.mean(data, axis=0)
            cov = np.cov(data.T)
            
            # Handle singular covariance matrix
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse for singular matrices
                inv_cov = np.linalg.pinv(cov)
            
            distances = []
            for point in data:
                diff = point - mean
                distance = np.sqrt(diff.T @ inv_cov @ diff)
                distances.append(distance)
            
            return np.array(distances)
            
        except Exception as e:
            self.logger.error(f"Error calculating Mahalanobis distance: {str(e)}")
            return np.zeros(len(data))
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - could be improved with NLP libraries
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _analyze_sentence_structure(self, sentence: str) -> Dict[str, Any]:
        """Analyze structure of a single sentence"""
        words = sentence.split()
        
        return {
            'length': len(sentence),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'punctuation_density': sum(1 for c in sentence if c in '.,!?;:') / len(sentence) if sentence else 0,
            'complexity_score': self._calculate_sentence_complexity(sentence)
        }
    
    def _calculate_sentence_complexity(self, sentence: str) -> float:
        """Calculate complexity score for a sentence"""
        # Simple complexity measure
        factors = []
        
        # Length factor
        length_factor = min(1.0, len(sentence) / 100)  # Normalize to 100 chars
        factors.append(length_factor)
        
        # Word length variance
        words = sentence.split()
        if len(words) > 1:
            word_lengths = [len(w) for w in words]
            length_variance = np.var(word_lengths) / 10  # Normalize
            factors.append(min(1.0, length_variance))
        
        # Punctuation complexity
        punct_count = sum(1 for c in sentence if c in '.,!?;:()[]{}')
        punct_factor = min(1.0, punct_count / 5)  # Normalize
        factors.append(punct_factor)
        
        return np.mean(factors) if factors else 0.0
    
    def _identify_common_sentence_patterns(self, sentence_stats: List[Dict]) -> Dict[str, Any]:
        """Identify common patterns in sentence structures"""
        if not sentence_stats:
            return {}
        
        lengths = [s['length'] for s in sentence_stats]
        word_counts = [s['word_count'] for s in sentence_stats]
        
        return {
            'common_length_range': [np.percentile(lengths, 25), np.percentile(lengths, 75)],
            'common_word_count_range': [np.percentile(word_counts, 25), np.percentile(word_counts, 75)],
            'typical_complexity': np.median([s['complexity_score'] for s in sentence_stats])
        }
    
    def _find_character_repetitions(self, text: str) -> List[Dict[str, Any]]:
        """Find character repetition patterns"""
        repetitions = []
        i = 0
        
        while i < len(text):
            char = text[i]
            count = 1
            j = i + 1
            
            while j < len(text) and text[j] == char:
                count += 1
                j += 1
            
            if count > 2:  # Only report repetitions of 3 or more
                repetitions.append({
                    'character': char,
                    'length': count,
                    'position': i
                })
            
            i = j if count > 1 else i + 1
        
        return repetitions
    
    def _find_word_repetitions(self, text: str) -> List[Dict[str, Any]]:
        """Find word repetition patterns"""
        words = text.split()
        word_positions = defaultdict(list)
        
        for i, word in enumerate(words):
            word_positions[word.lower()].append(i)
        
        repetitions = []
        for word, positions in word_positions.items():
            if len(positions) > 2:  # Word appears more than twice
                # Check if repetitions are in close proximity
                close_repetitions = 0
                for i in range(len(positions) - 1):
                    if positions[i+1] - positions[i] <= 10:  # Within 10 words
                        close_repetitions += 1
                
                if close_repetitions > 0:
                    repetitions.append({
                        'word': word,
                        'count': len(positions),
                        'positions': positions,
                        'close_repetitions': close_repetitions
                    })
        
        return repetitions
    
    def _find_unusual_character_combinations(self, text: str) -> List[Dict[str, Any]]:
        """Find unusual character combinations"""
        # Look for patterns that are uncommon in natural language
        unusual_patterns = []
        
        # Check for repeated character pairs
        bigrams = [text[i:i+2] for i in range(len(text)-1)]
        bigram_counts = Counter(bigrams)
        
        total_bigrams = len(bigrams)
        
        for bigram, count in bigram_counts.items():
            frequency = count / total_bigrams
            
            # Check if it's an unusual pattern (non-alphabetic or very frequent)
            if not bigram.isalpha() or frequency > 0.05:  # More than 5% frequency
                examples = []
                for i, bg in enumerate(bigrams):
                    if bg == bigram:
                        start = max(0, i-5)
                        end = min(len(text), i+7)
                        examples.append(text[start:end])
                        if len(examples) >= 5:
                            break
                
                unusual_patterns.append({
                    'combination': bigram,
                    'count': count,
                    'frequency': frequency,
                    'examples': examples
                })
        
        return unusual_patterns
    
    def _detect_punctuation_anomalies(self, text: str) -> List[AnomalyResult]:
        """Detect anomalies in punctuation patterns"""
        anomalies = []
        
        # Count different types of punctuation
        punct_counts = {
            'periods': text.count('.'),
            'commas': text.count(','),
            'semicolons': text.count(';'),
            'colons': text.count(':'),
            'exclamations': text.count('!'),
            'questions': text.count('?'),
            'quotes': text.count('"') + text.count("'"),
            'parentheses': text.count('(') + text.count(')'),
            'brackets': text.count('[') + text.count(']'),
            'braces': text.count('{') + text.count('}')
        }
        
        total_chars = len(text)
        
        # Check for unusual punctuation patterns
        for punct_type, count in punct_counts.items():
            if count > 0:
                frequency = count / total_chars
                
                # Define thresholds for different punctuation types
                thresholds = {
                    'periods': 0.02,
                    'commas': 0.03,
                    'semicolons': 0.005,
                    'colons': 0.005,
                    'exclamations': 0.01,
                    'questions': 0.01,
                    'quotes': 0.02,
                    'parentheses': 0.01,
                    'brackets': 0.005,
                    'braces': 0.002
                }
                
                threshold = thresholds.get(punct_type, 0.01)
                
                if frequency > threshold:
                    anomaly = AnomalyResult(
                        anomaly_type='punctuation_anomaly',
                        location={'punctuation_type': punct_type},
                        severity=min(1.0, frequency / threshold),
                        confidence=0.6,
                        statistical_significance=0.1,
                        description=f"Unusual frequency of {punct_type}: {frequency:.4f}",
                        evidence={
                            'punctuation_type': punct_type,
                            'count': count,
                            'frequency': frequency,
                            'threshold': threshold
                        },
                        baseline_stats={'expected_frequency': threshold},
                        detected_at=datetime.now()
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _find_repeating_patterns(self, text: str, pattern_length: int) -> Dict[str, List[int]]:
        """Find repeating patterns of specified length"""
        patterns = defaultdict(list)
        
        for i in range(len(text) - pattern_length + 1):
            pattern = text[i:i + pattern_length]
            patterns[pattern].append(i)
        
        # Only return patterns that repeat
        return {pattern: positions for pattern, positions in patterns.items() if len(positions) > 1}
    
    def _extract_geometric_features(self, patterns: List) -> Dict[str, Any]:
        """Extract features from geometric patterns"""
        # Placeholder implementation
        return {
            'density': 0.5,
            'symmetry': 0.7,
            'complexity': 0.6
        }
    
    def _calculate_geometric_complexity_distribution(self, features: List[Dict]) -> Dict[str, float]:
        """Calculate distribution of geometric complexity"""
        if not features:
            return {}
        
        complexities = [f.get('complexity', 0) for f in features]
        
        return {
            'mean': np.mean(complexities),
            'std': np.std(complexities),
            'min': np.min(complexities),
            'max': np.max(complexities)
        }
    
    def _identify_common_geometric_ratios(self, features: List[Dict]) -> List[float]:
        """Identify common ratios in geometric patterns"""
        # Placeholder implementation
        return [1.618, 1.414, 2.0]  # Golden ratio, sqrt(2), 2:1
    
    def _validate_character_model(self, char_stats: Dict, all_char_freqs: List[Dict]) -> float:
        """Validate character frequency model"""
        # Simple validation - could be improved
        return 0.8
    
    def _validate_word_length_model(self, word_length_stats: Dict, all_word_lengths: List) -> float:
        """Validate word length model"""
        # Simple validation - could be improved
        return 0.85
    
    def _calculate_distribution(self, values: List[float]) -> Dict[str, float]:
        """Calculate distribution statistics"""
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75)
        }
    
    def _generate_anomaly_recommendations(self, anomalies: List[AnomalyResult]) -> List[str]:
        """Generate recommendations based on detected anomalies"""
        recommendations = []
        
        if not anomalies:
            recommendations.append("No significant anomalies detected. Document appears to follow normal patterns.")
            return recommendations
        
        # Categorize anomalies
        high_severity = [a for a in anomalies if a.severity >= 0.8]
        cipher_related = [a for a in anomalies if 'cipher' in a.anomaly_type or 'entropy' in a.anomaly_type]
        pattern_related = [a for a in anomalies if 'pattern' in a.anomaly_type]
        geometric_related = [a for a in anomalies if 'geometric' in a.anomaly_type or 'spatial' in a.anomaly_type]
        
        if high_severity:
            recommendations.append(f"Found {len(high_severity)} high-severity anomalies requiring immediate attention.")
        
        if cipher_related:
            recommendations.append(f"Detected {len(cipher_related)} cipher-related anomalies. Consider cryptographic analysis.")
        
        if pattern_related:
            recommendations.append(f"Found {len(pattern_related)} pattern anomalies. Review pattern detection algorithms.")
        
        if geometric_related:
            recommendations.append(f"Identified {len(geometric_related)} geometric anomalies. Examine spatial relationships.")
        
        # Specific recommendations based on anomaly types
        anomaly_types = [a.anomaly_type for a in anomalies]
        
        if 'flat_frequency_distribution' in anomaly_types:
            recommendations.append("Flat character frequency suggests substitution cipher. Try frequency analysis.")
        
        if 'repeating_key_pattern' in anomaly_types:
            recommendations.append("Repeating patterns detected. Consider Vigenère or similar polyalphabetic cipher.")
        
        if 'sacred_geometry' in anomaly_types:
            recommendations.append("Sacred geometry patterns found. Investigate esoteric or symbolic meanings.")
        
        if 'mathematical_constant' in anomaly_types:
            recommendations.append("Mathematical constants detected. Examine for intentional mathematical encoding.")
        
        return recommendations
    
    def _anomaly_to_dict(self, anomaly: AnomalyResult) -> Dict[str, Any]:
        """Convert AnomalyResult to dictionary for serialization"""
        return {
            'anomaly_type': anomaly.anomaly_type,
            'location': anomaly.location,
            'severity': anomaly.severity,
            'confidence': anomaly.confidence,
            'statistical_significance': anomaly.statistical_significance,
            'description': anomaly.description,
            'evidence': anomaly.evidence,
            'baseline_stats': anomaly.baseline_stats,
            'detected_at': anomaly.detected_at.isoformat()
        }