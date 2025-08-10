"""
Text Analysis Service for Ancient Text Analyzer
Performs detailed character and word-level analysis for pattern detection
"""
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
from enum import Enum
import re
import statistics
import math

try:
    from app.services.ocr_engine import CharacterBox, OCRResult
except ImportError:
    # For testing without OCR dependencies
    from tests.mock_ocr_engine import CharacterBox, OCRResult

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of text anomalies that can be detected"""
    CHARACTER_SIZE_VARIATION = "character_size_variation"
    UNUSUAL_SPACING = "unusual_spacing"
    FONT_INCONSISTENCY = "font_inconsistency"
    PUNCTUATION_ANOMALY = "punctuation_anomaly"
    WORD_FREQUENCY_ANOMALY = "word_frequency_anomaly"
    POSITION_ANOMALY = "position_anomaly"
    CONFIDENCE_ANOMALY = "confidence_anomaly"

@dataclass
class CharacterAnalysis:
    """Container for character-level analysis results"""
    total_characters: int
    unique_characters: int
    character_frequencies: Dict[str, int]
    character_positions: Dict[str, List[Tuple[float, float]]]
    size_statistics: Dict[str, Dict[str, float]]
    confidence_statistics: Dict[str, float]
    spacing_analysis: Dict[str, Any]
    baseline_analysis: Dict[str, Any]

@dataclass
class Anomaly:
    """Container for detected anomalies"""
    type: AnomalyType
    location: Tuple[float, float, float, float]  # x, y, width, height
    description: str
    severity: float  # 0.0 to 1.0
    confidence: float
    suggested_analysis: List[str]
    character_data: Optional[CharacterBox] = None
    context: Optional[str] = None

@dataclass
class PunctuationAnalysis:
    """Container for punctuation-specific analysis"""
    punctuation_marks: Dict[str, List[CharacterBox]]
    size_variations: Dict[str, Dict[str, float]]
    position_patterns: Dict[str, List[Tuple[float, float]]]
    anomalies: List[Anomaly]

class TextAnalyzer:
    """
    Advanced text analyzer for character-level pattern detection
    Specialized for detecting encoding markers and anomalies in historical texts
    """
    
    def __init__(self):
        self.punctuation_marks = set('.,;:!?()[]{}"\'-/')
        self.common_letters = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.digits = set('0123456789')
        
    def analyze_characters(self, text: str, characters: List[CharacterBox]) -> CharacterAnalysis:
        """
        Perform comprehensive character-level analysis
        
        Args:
            text: Full text string
            characters: List of character boxes with position data
            
        Returns:
            CharacterAnalysis with detailed statistics
        """
        logger.info(f"Starting character analysis for {len(characters)} characters")
        
        try:
            # Basic character statistics
            char_frequencies = Counter(char.character for char in characters)
            
            # Position mapping
            char_positions = defaultdict(list)
            for char in characters:
                char_positions[char.character].append((char.x, char.y))
            
            # Size statistics by character
            size_stats = self._calculate_size_statistics(characters)
            
            # Confidence statistics
            confidence_stats = self._calculate_confidence_statistics(characters)
            
            # Spacing analysis
            spacing_analysis = self._analyze_character_spacing(characters)
            
            # Baseline analysis
            baseline_analysis = self._analyze_baseline_alignment(characters)
            
            analysis = CharacterAnalysis(
                total_characters=len(characters),
                unique_characters=len(char_frequencies),
                character_frequencies=dict(char_frequencies),
                character_positions=dict(char_positions),
                size_statistics=size_stats,
                confidence_statistics=confidence_stats,
                spacing_analysis=spacing_analysis,
                baseline_analysis=baseline_analysis
            )
            
            logger.info(f"Character analysis completed: {analysis.unique_characters} unique characters")
            return analysis
            
        except Exception as e:
            logger.error(f"Character analysis failed: {e}")
            # Return empty analysis on failure
            return CharacterAnalysis(
                total_characters=0,
                unique_characters=0,
                character_frequencies={},
                character_positions={},
                size_statistics={},
                confidence_statistics={},
                spacing_analysis={},
                baseline_analysis={}
            )
    
    def _calculate_size_statistics(self, characters: List[CharacterBox]) -> Dict[str, Dict[str, float]]:
        """
        Calculate size statistics for each character type
        """
        char_sizes = defaultdict(lambda: {'heights': [], 'widths': []})
        
        for char in characters:
            char_sizes[char.character]['heights'].append(char.height)
            char_sizes[char.character]['widths'].append(char.width)
        
        size_stats = {}
        for char_type, sizes in char_sizes.items():
            if sizes['heights']:  # Ensure we have data
                size_stats[char_type] = {
                    'height_mean': statistics.mean(sizes['heights']),
                    'height_median': statistics.median(sizes['heights']),
                    'height_std': statistics.stdev(sizes['heights']) if len(sizes['heights']) > 1 else 0.0,
                    'height_min': min(sizes['heights']),
                    'height_max': max(sizes['heights']),
                    'width_mean': statistics.mean(sizes['widths']),
                    'width_median': statistics.median(sizes['widths']),
                    'width_std': statistics.stdev(sizes['widths']) if len(sizes['widths']) > 1 else 0.0,
                    'width_min': min(sizes['widths']),
                    'width_max': max(sizes['widths']),
                    'aspect_ratio': statistics.mean(sizes['widths']) / statistics.mean(sizes['heights']) if statistics.mean(sizes['heights']) > 0 else 0,
                    'size_variation_coefficient': statistics.stdev(sizes['heights']) / statistics.mean(sizes['heights']) if statistics.mean(sizes['heights']) > 0 and len(sizes['heights']) > 1 else 0
                }
        
        return size_stats
    
    def _calculate_confidence_statistics(self, characters: List[CharacterBox]) -> Dict[str, float]:
        """
        Calculate confidence statistics across all characters
        """
        if not characters:
            return {}
        
        confidences = [char.confidence for char in characters]
        
        return {
            'mean_confidence': statistics.mean(confidences),
            'median_confidence': statistics.median(confidences),
            'std_confidence': statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'low_confidence_count': sum(1 for c in confidences if c < 0.8),
            'high_confidence_count': sum(1 for c in confidences if c > 0.95)
        }
    
    def _analyze_character_spacing(self, characters: List[CharacterBox]) -> Dict[str, Any]:
        """
        Analyze spacing between characters
        """
        if len(characters) < 2:
            return {}
        
        # Sort characters by position (left to right, top to bottom)
        sorted_chars = sorted(characters, key=lambda c: (c.y, c.x))
        
        horizontal_gaps = []
        vertical_gaps = []
        
        for i in range(len(sorted_chars) - 1):
            current = sorted_chars[i]
            next_char = sorted_chars[i + 1]
            
            # Calculate horizontal gap (if on same line)
            if abs(current.y - next_char.y) < current.height * 0.5:  # Same line
                h_gap = next_char.x - (current.x + current.width)
                if h_gap >= 0:  # Valid gap
                    horizontal_gaps.append(h_gap)
            
            # Calculate vertical gap (if in same column)
            if abs(current.x - next_char.x) < current.width * 0.5:  # Same column
                v_gap = next_char.y - (current.y + current.height)
                if v_gap >= 0:  # Valid gap
                    vertical_gaps.append(v_gap)
        
        spacing_stats = {}
        
        if horizontal_gaps:
            spacing_stats['horizontal'] = {
                'mean': statistics.mean(horizontal_gaps),
                'median': statistics.median(horizontal_gaps),
                'std': statistics.stdev(horizontal_gaps) if len(horizontal_gaps) > 1 else 0.0,
                'min': min(horizontal_gaps),
                'max': max(horizontal_gaps)
            }
        
        if vertical_gaps:
            spacing_stats['vertical'] = {
                'mean': statistics.mean(vertical_gaps),
                'median': statistics.median(vertical_gaps),
                'std': statistics.stdev(vertical_gaps) if len(vertical_gaps) > 1 else 0.0,
                'min': min(vertical_gaps),
                'max': max(vertical_gaps)
            }
        
        return spacing_stats
    
    def _analyze_baseline_alignment(self, characters: List[CharacterBox]) -> Dict[str, Any]:
        """
        Analyze baseline alignment of characters
        """
        if not characters:
            return {}
        
        # Group characters by approximate line (using y-coordinate clustering)
        lines = self._group_characters_by_line(characters)
        
        baseline_stats = {}
        line_variations = []
        
        for line_id, line_chars in lines.items():
            if len(line_chars) < 2:
                continue
            
            # Calculate baseline positions (bottom of characters)
            baselines = [char.y + char.height for char in line_chars]
            
            line_stats = {
                'character_count': len(line_chars),
                'baseline_mean': statistics.mean(baselines),
                'baseline_std': statistics.stdev(baselines) if len(baselines) > 1 else 0.0,
                'baseline_range': max(baselines) - min(baselines)
            }
            
            baseline_stats[f'line_{line_id}'] = line_stats
            line_variations.append(line_stats['baseline_std'])
        
        # Overall baseline analysis
        if line_variations:
            baseline_stats['overall'] = {
                'line_count': len(lines),
                'average_line_variation': statistics.mean(line_variations),
                'max_line_variation': max(line_variations),
                'consistent_lines': sum(1 for var in line_variations if var < 2.0)
            }
        
        return baseline_stats
    
    def _group_characters_by_line(self, characters: List[CharacterBox]) -> Dict[int, List[CharacterBox]]:
        """
        Group characters into lines based on y-coordinate clustering
        """
        if not characters:
            return {}
        
        # Sort by y-coordinate
        sorted_chars = sorted(characters, key=lambda c: c.y)
        
        lines = {}
        current_line_id = 0
        current_line_y = sorted_chars[0].y
        current_line = [sorted_chars[0]]
        
        for char in sorted_chars[1:]:
            # If character is close to current line, add to current line
            if abs(char.y - current_line_y) < char.height * 0.5:
                current_line.append(char)
            else:
                # Start new line
                lines[current_line_id] = current_line
                current_line_id += 1
                current_line_y = char.y
                current_line = [char]
        
        # Add the last line
        if current_line:
            lines[current_line_id] = current_line
        
        return lines
    
    def detect_anomalies(self, analysis: CharacterAnalysis) -> List[Anomaly]:
        """
        Detect anomalies in character analysis that may indicate encoding
        
        Args:
            analysis: Character analysis results
            
        Returns:
            List of detected anomalies
        """
        logger.info("Starting anomaly detection")
        anomalies = []
        
        try:
            # Detect character size variations
            size_anomalies = self._detect_size_anomalies(analysis)
            anomalies.extend(size_anomalies)
            
            # Detect spacing anomalies
            spacing_anomalies = self._detect_spacing_anomalies(analysis)
            anomalies.extend(spacing_anomalies)
            
            # Detect confidence anomalies
            confidence_anomalies = self._detect_confidence_anomalies(analysis)
            anomalies.extend(confidence_anomalies)
            
            # Detect frequency anomalies
            frequency_anomalies = self._detect_frequency_anomalies(analysis)
            anomalies.extend(frequency_anomalies)
            
            logger.info(f"Detected {len(anomalies)} anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    def _detect_size_anomalies(self, analysis: CharacterAnalysis) -> List[Anomaly]:
        """
        Detect unusual character size variations
        """
        anomalies = []
        
        for char_type, stats in analysis.size_statistics.items():
            # Check for high size variation coefficient
            if stats['size_variation_coefficient'] > 0.2:  # 20% variation threshold
                positions = analysis.character_positions.get(char_type, [])
                if positions:
                    # Use first position as representative
                    x, y = positions[0]
                    
                    anomaly = Anomaly(
                        type=AnomalyType.CHARACTER_SIZE_VARIATION,
                        location=(x, y, stats['width_mean'], stats['height_mean']),
                        description=f"Character '{char_type}' shows high size variation (CV: {stats['size_variation_coefficient']:.3f})",
                        severity=min(stats['size_variation_coefficient'] * 2, 1.0),
                        confidence=0.8,
                        suggested_analysis=[
                            "Check for intentional size modifications",
                            "Analyze pattern in size variations",
                            "Compare with surrounding characters"
                        ],
                        context=f"Height range: {stats['height_min']:.1f}-{stats['height_max']:.1f}"
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_spacing_anomalies(self, analysis: CharacterAnalysis) -> List[Anomaly]:
        """
        Detect unusual spacing patterns
        """
        anomalies = []
        
        spacing = analysis.spacing_analysis
        
        # Check horizontal spacing anomalies
        if 'horizontal' in spacing:
            h_stats = spacing['horizontal']
            if h_stats['std'] > h_stats['mean'] * 0.5:  # High variation in spacing
                anomaly = Anomaly(
                    type=AnomalyType.UNUSUAL_SPACING,
                    location=(0, 0, 0, 0),  # General anomaly
                    description=f"Unusual horizontal spacing variation (std: {h_stats['std']:.2f}, mean: {h_stats['mean']:.2f})",
                    severity=min(h_stats['std'] / h_stats['mean'], 1.0),
                    confidence=0.7,
                    suggested_analysis=[
                        "Check for intentional spacing modifications",
                        "Look for hidden word boundaries",
                        "Analyze spacing patterns across lines"
                    ],
                    context=f"Spacing range: {h_stats['min']:.1f}-{h_stats['max']:.1f}"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_confidence_anomalies(self, analysis: CharacterAnalysis) -> List[Anomaly]:
        """
        Detect confidence-related anomalies
        """
        anomalies = []
        
        conf_stats = analysis.confidence_statistics
        
        # Check for unusually low overall confidence
        if conf_stats.get('mean_confidence', 1.0) < 0.7:
            anomaly = Anomaly(
                type=AnomalyType.CONFIDENCE_ANOMALY,
                location=(0, 0, 0, 0),
                description=f"Low overall OCR confidence: {conf_stats['mean_confidence']:.3f}",
                severity=1.0 - conf_stats['mean_confidence'],
                confidence=0.9,
                suggested_analysis=[
                    "Review OCR preprocessing steps",
                    "Check for unusual fonts or decorations",
                    "Consider manual verification of low-confidence regions"
                ],
                context=f"Low confidence characters: {conf_stats.get('low_confidence_count', 0)}"
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_frequency_anomalies(self, analysis: CharacterAnalysis) -> List[Anomaly]:
        """
        Detect unusual character frequency patterns
        """
        anomalies = []
        
        # Calculate expected frequencies for English text
        expected_frequencies = {
            'e': 0.127, 't': 0.091, 'a': 0.082, 'o': 0.075, 'i': 0.070,
            'n': 0.067, 's': 0.063, 'h': 0.061, 'r': 0.060, 'd': 0.043
        }
        
        total_chars = analysis.total_characters
        if total_chars == 0:
            return anomalies
        
        for char, expected_freq in expected_frequencies.items():
            actual_count = analysis.character_frequencies.get(char, 0) + analysis.character_frequencies.get(char.upper(), 0)
            actual_freq = actual_count / total_chars
            
            # Check for significant deviation
            if abs(actual_freq - expected_freq) > expected_freq * 0.5:  # 50% deviation threshold
                positions = analysis.character_positions.get(char, [])
                if positions:
                    x, y = positions[0]
                    
                    anomaly = Anomaly(
                        type=AnomalyType.WORD_FREQUENCY_ANOMALY,
                        location=(x, y, 0, 0),
                        description=f"Character '{char}' frequency deviation: expected {expected_freq:.3f}, actual {actual_freq:.3f}",
                        severity=abs(actual_freq - expected_freq) / expected_freq,
                        confidence=0.6,
                        suggested_analysis=[
                            "Check for non-English text",
                            "Look for cipher substitutions",
                            "Analyze character distribution patterns"
                        ],
                        context=f"Occurrences: {actual_count}/{total_chars}"
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def measure_punctuation_variations(self, characters: List[CharacterBox]) -> PunctuationAnalysis:
        """
        Analyze punctuation marks for size and position variations
        
        Args:
            characters: List of character boxes
            
        Returns:
            PunctuationAnalysis with detailed punctuation data
        """
        logger.info("Starting punctuation analysis")
        
        try:
            # Filter punctuation characters
            punctuation_chars = [char for char in characters if char.character in self.punctuation_marks]
            
            # Group by punctuation type
            punct_groups = defaultdict(list)
            for char in punctuation_chars:
                punct_groups[char.character].append(char)
            
            # Analyze size variations for each punctuation type
            size_variations = {}
            position_patterns = {}
            
            for punct_type, punct_list in punct_groups.items():
                if len(punct_list) > 1:  # Need multiple instances for variation analysis
                    heights = [p.height for p in punct_list]
                    widths = [p.width for p in punct_list]
                    
                    size_variations[punct_type] = {
                        'count': len(punct_list),
                        'height_mean': statistics.mean(heights),
                        'height_std': statistics.stdev(heights),
                        'height_cv': statistics.stdev(heights) / statistics.mean(heights) if statistics.mean(heights) > 0 else 0,
                        'width_mean': statistics.mean(widths),
                        'width_std': statistics.stdev(widths),
                        'width_cv': statistics.stdev(widths) / statistics.mean(widths) if statistics.mean(widths) > 0 else 0,
                        'size_range': max(heights) - min(heights)
                    }
                    
                    # Position patterns
                    positions = [(p.x, p.y) for p in punct_list]
                    position_patterns[punct_type] = positions
            
            # Detect punctuation anomalies
            punct_anomalies = self._detect_punctuation_anomalies(size_variations, position_patterns)
            
            analysis = PunctuationAnalysis(
                punctuation_marks=dict(punct_groups),
                size_variations=size_variations,
                position_patterns=position_patterns,
                anomalies=punct_anomalies
            )
            
            logger.info(f"Punctuation analysis completed: {len(punct_groups)} types, {len(punct_anomalies)} anomalies")
            return analysis
            
        except Exception as e:
            logger.error(f"Punctuation analysis failed: {e}")
            return PunctuationAnalysis(
                punctuation_marks={},
                size_variations={},
                position_patterns={},
                anomalies=[]
            )
    
    def _detect_punctuation_anomalies(self, size_variations: Dict[str, Dict[str, float]], 
                                    position_patterns: Dict[str, List[Tuple[float, float]]]) -> List[Anomaly]:
        """
        Detect anomalies in punctuation marks
        """
        anomalies = []
        
        for punct_type, stats in size_variations.items():
            # Check for high size variation
            if stats['height_cv'] > 0.15:  # 15% coefficient of variation threshold
                positions = position_patterns.get(punct_type, [(0, 0)])
                x, y = positions[0]
                
                anomaly = Anomaly(
                    type=AnomalyType.PUNCTUATION_ANOMALY,
                    location=(x, y, stats['width_mean'], stats['height_mean']),
                    description=f"Punctuation '{punct_type}' shows significant size variation (CV: {stats['height_cv']:.3f})",
                    severity=min(stats['height_cv'] * 3, 1.0),
                    confidence=0.8,
                    suggested_analysis=[
                        "Check for intentional punctuation modifications",
                        "Look for encoding in punctuation sizes",
                        "Compare with historical printing standards"
                    ],
                    context=f"Size range: {stats['size_range']:.1f}, Count: {stats['count']}"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def generate_word_patterns(self, text: str) -> Dict[str, Any]:
        """
        Generate word pattern analysis for anomaly detection
        
        Args:
            text: Full text string
            
        Returns:
            Dictionary with word pattern analysis
        """
        logger.info("Generating word pattern analysis")
        
        try:
            # Clean and tokenize text
            words = re.findall(r'\b\w+\b', text.lower())
            
            if not words:
                return {"error": "No words found in text"}
            
            # Basic word statistics
            word_freq = Counter(words)
            word_lengths = [len(word) for word in words]
            
            # Pattern analysis
            patterns = {
                'total_words': len(words),
                'unique_words': len(word_freq),
                'vocabulary_richness': len(word_freq) / len(words),
                'average_word_length': statistics.mean(word_lengths),
                'word_length_std': statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0,
                'most_common_words': word_freq.most_common(10),
                'hapax_legomena': sum(1 for count in word_freq.values() if count == 1),
                'long_words': [word for word in words if len(word) > 10],
                'short_words': [word for word in words if len(word) <= 2]
            }
            
            # Advanced pattern detection
            patterns.update(self._detect_word_patterns(words, word_freq))
            
            # Detect unusual patterns
            unusual_patterns = []
            
            # Very long words (potential encoding)
            if patterns['long_words']:
                unusual_patterns.append({
                    'type': 'long_words',
                    'description': f"Found {len(patterns['long_words'])} words longer than 10 characters",
                    'examples': patterns['long_words'][:5]
                })
            
            # High vocabulary richness (potential cipher)
            if patterns['vocabulary_richness'] > 0.8:
                unusual_patterns.append({
                    'type': 'high_vocabulary_richness',
                    'description': f"Very high vocabulary richness: {patterns['vocabulary_richness']:.3f}",
                    'suggestion': "May indicate cipher or unusual text"
                })
            
            # Unusual word repetition patterns
            if patterns.get('repetition_anomalies'):
                unusual_patterns.append({
                    'type': 'repetition_anomalies',
                    'description': f"Found {len(patterns['repetition_anomalies'])} unusual repetition patterns",
                    'examples': patterns['repetition_anomalies'][:3]
                })
            
            patterns['unusual_patterns'] = unusual_patterns
            
            logger.info(f"Word pattern analysis completed: {patterns['total_words']} words, {patterns['unique_words']} unique")
            return patterns
            
        except Exception as e:
            logger.error(f"Word pattern analysis failed: {e}")
            return {"error": str(e)}
    
    def _detect_word_patterns(self, words: List[str], word_freq: Counter) -> Dict[str, Any]:
        """
        Detect advanced word patterns that may indicate encoding
        
        Args:
            words: List of words
            word_freq: Word frequency counter
            
        Returns:
            Dictionary with pattern analysis results
        """
        patterns = {}
        
        # Detect palindromes
        palindromes = [word for word in words if len(word) > 3 and word == word[::-1]]
        patterns['palindromes'] = palindromes
        
        # Detect words with unusual character patterns
        patterns['unusual_character_patterns'] = self._find_unusual_character_patterns(words)
        
        # Detect repetition anomalies
        patterns['repetition_anomalies'] = self._find_repetition_anomalies(word_freq)
        
        # Detect potential anagrams
        patterns['potential_anagrams'] = self._find_potential_anagrams(words)
        
        # Analyze word positioning patterns
        patterns['positioning_patterns'] = self._analyze_word_positioning(words)
        
        # Detect archaic or unusual spellings
        patterns['archaic_spellings'] = self._detect_archaic_spellings(words)
        
        return patterns
    
    def _find_unusual_character_patterns(self, words: List[str]) -> List[Dict[str, Any]]:
        """
        Find words with unusual character patterns that may indicate encoding
        """
        unusual_patterns = []
        
        for word in set(words):  # Use set to avoid duplicates
            if len(word) < 3:
                continue
            
            # Check for alternating consonants/vowels
            vowels = set('aeiou')
            pattern = ''.join('V' if c in vowels else 'C' for c in word)
            
            # Detect highly regular patterns
            if len(set(pattern)) <= 2 and len(pattern) > 4:
                unusual_patterns.append({
                    'word': word,
                    'pattern': pattern,
                    'type': 'alternating_pattern',
                    'description': f"Word '{word}' has regular consonant/vowel pattern: {pattern}"
                })
            
            # Check for repeated character sequences
            for i in range(len(word) - 2):
                for j in range(i + 2, len(word)):
                    if word[i:i+2] == word[j:j+2] and i + 2 < j:
                        unusual_patterns.append({
                            'word': word,
                            'pattern': word[i:i+2],
                            'type': 'repeated_sequence',
                            'description': f"Word '{word}' contains repeated sequence '{word[i:i+2]}'"
                        })
                        break
            
            # Check for unusual letter frequencies within word
            char_freq = Counter(word)
            if any(count > len(word) * 0.4 for count in char_freq.values()):
                dominant_char = max(char_freq, key=char_freq.get)
                unusual_patterns.append({
                    'word': word,
                    'pattern': dominant_char,
                    'type': 'dominant_character',
                    'description': f"Word '{word}' has dominant character '{dominant_char}' ({char_freq[dominant_char]} times)"
                })
        
        return unusual_patterns
    
    def _find_repetition_anomalies(self, word_freq: Counter) -> List[Dict[str, Any]]:
        """
        Find unusual word repetition patterns
        """
        anomalies = []
        total_words = sum(word_freq.values())
        
        for word, count in word_freq.items():
            frequency = count / total_words
            
            # Check for unusually high frequency of uncommon words
            if len(word) > 6 and frequency > 0.01:  # Long word appearing more than 1% of time
                anomalies.append({
                    'word': word,
                    'count': count,
                    'frequency': frequency,
                    'type': 'high_frequency_long_word',
                    'description': f"Long word '{word}' appears unusually frequently ({count} times, {frequency:.3f})"
                })
            
            # Check for exact repetition patterns (same word appearing in clusters)
            if count > 3:  # Word appears more than 3 times
                anomalies.append({
                    'word': word,
                    'count': count,
                    'frequency': frequency,
                    'type': 'high_repetition',
                    'description': f"Word '{word}' shows high repetition ({count} occurrences)"
                })
        
        return anomalies
    
    def _find_potential_anagrams(self, words: List[str]) -> List[Dict[str, Any]]:
        """
        Find potential anagrams that might indicate encoding
        """
        anagrams = []
        word_signatures = defaultdict(list)
        
        # Group words by their sorted character signature
        for word in set(words):
            if len(word) > 3:  # Only consider words longer than 3 characters
                signature = ''.join(sorted(word))
                word_signatures[signature].append(word)
        
        # Find groups with multiple words (potential anagrams)
        for signature, word_group in word_signatures.items():
            if len(word_group) > 1:
                anagrams.append({
                    'signature': signature,
                    'words': word_group,
                    'count': len(word_group),
                    'description': f"Potential anagrams: {', '.join(word_group)}"
                })
        
        return anagrams
    
    def _analyze_word_positioning(self, words: List[str]) -> Dict[str, Any]:
        """
        Analyze word positioning patterns that might indicate structure
        """
        positioning = {
            'word_count': len(words),
            'unique_positions': {},
            'repeated_sequences': []
        }
        
        # Look for repeated word sequences
        for seq_len in range(2, min(6, len(words) // 2)):  # Sequences of 2-5 words
            for i in range(len(words) - seq_len + 1):
                sequence = tuple(words[i:i + seq_len])
                
                # Check if this sequence appears later
                for j in range(i + seq_len, len(words) - seq_len + 1):
                    if tuple(words[j:j + seq_len]) == sequence:
                        positioning['repeated_sequences'].append({
                            'sequence': sequence,
                            'positions': [i, j],
                            'length': seq_len,
                            'description': f"Sequence {sequence} repeats at positions {i} and {j}"
                        })
                        break
        
        # Analyze word positions for specific words
        word_positions = defaultdict(list)
        for i, word in enumerate(words):
            word_positions[word].append(i)
        
        # Find words that appear at regular intervals
        for word, positions in word_positions.items():
            if len(positions) > 2:
                intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                if len(set(intervals)) == 1:  # All intervals are the same
                    positioning['unique_positions'][word] = {
                        'positions': positions,
                        'interval': intervals[0],
                        'description': f"Word '{word}' appears at regular intervals of {intervals[0]}"
                    }
        
        return positioning
    
    def _detect_archaic_spellings(self, words: List[str]) -> List[Dict[str, Any]]:
        """
        Detect potentially archaic or unusual spellings
        """
        archaic_patterns = []
        
        # Common archaic spelling patterns
        archaic_indicators = [
            (r'ye\b', 'the', 'archaic_article'),
            (r'\bvs\b', 'us', 'v_for_u'),
            (r'[a-z]*ſ[a-z]*', 'long_s', 'long_s_usage'),  # Long s character
            (r'\b\w*eth\b', 'eth_ending', 'archaic_verb_ending'),
            (r'\b\w*est\b', 'est_ending', 'archaic_verb_ending'),
            (r'\bwhilſt\b', 'whilst', 'archaic_spelling'),
            (r'\bamongſt\b', 'amongst', 'archaic_spelling')
        ]
        
        for word in set(words):
            for pattern, modern_form, pattern_type in archaic_indicators:
                if re.search(pattern, word, re.IGNORECASE):
                    archaic_patterns.append({
                        'word': word,
                        'pattern': pattern,
                        'modern_form': modern_form,
                        'type': pattern_type,
                        'description': f"Word '{word}' shows archaic spelling pattern"
                    })
        
        return archaic_patterns