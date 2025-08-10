"""
Classical Cipher Detection System
Implements detection algorithms for various classical ciphers including Caesar, Atbash, 
substitution ciphers, biliteral ciphers, and skip patterns.
"""
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import re
import string
import math
import itertools
from enum import Enum

from app.models.database_models import Character, Word, Pattern


class CipherType(Enum):
    """Enumeration of supported cipher types"""
    CAESAR = "caesar"
    ATBASH = "atbash"
    SUBSTITUTION = "substitution"
    BILITERAL = "biliteral"
    SKIP_PATTERN = "skip_pattern"
    REVOLVING = "revolving"
    MONOALPHABETIC = "monoalphabetic"
    POLYALPHABETIC = "polyalphabetic"


@dataclass
class CipherMatch:
    """Represents a detected cipher pattern"""
    cipher_type: CipherType
    confidence: float
    key_or_pattern: str
    decoded_text: str
    original_text: str
    position_start: int
    position_end: int
    method_details: Dict[str, Any]
    statistical_analysis: Dict[str, float]


@dataclass
class FrequencyAnalysis:
    """Results of frequency analysis on text"""
    letter_frequencies: Dict[str, float]
    bigram_frequencies: Dict[str, float]
    trigram_frequencies: Dict[str, float]
    index_of_coincidence: float
    chi_squared: float
    entropy: float


class CipherDetector:
    """
    Advanced cipher detection system for classical cryptographic methods
    """
    
    # English letter frequencies (standard reference)
    ENGLISH_FREQUENCIES = {
        'A': 8.12, 'B': 1.49, 'C': 2.78, 'D': 4.25, 'E': 12.02, 'F': 2.23,
        'G': 2.02, 'H': 6.09, 'I': 6.97, 'J': 0.15, 'K': 0.77, 'L': 4.03,
        'M': 2.41, 'N': 6.75, 'O': 7.51, 'P': 1.93, 'Q': 0.10, 'R': 5.99,
        'S': 6.33, 'T': 9.06, 'U': 2.76, 'V': 0.98, 'W': 2.36, 'X': 0.15,
        'Y': 1.97, 'Z': 0.07
    }
    
    # Common English bigrams
    COMMON_BIGRAMS = ['TH', 'HE', 'IN', 'ER', 'AN', 'RE', 'ED', 'ND', 'ON', 'EN']
    
    # Common English trigrams
    COMMON_TRIGRAMS = ['THE', 'AND', 'ING', 'HER', 'HAT', 'HIS', 'THA', 'ERE', 'FOR', 'ENT']
    
    def __init__(self):
        self.min_confidence = 0.3  # Minimum confidence for cipher detection (lowered for testing)
        self.min_text_length = 20  # Minimum text length for analysis
    
    def detect_ciphers(self, text: str, characters: List[Character] = None) -> List[CipherMatch]:
        """
        Main method to detect various cipher types in text
        """
        if len(text) < self.min_text_length:
            return []
        
        detected_ciphers = []
        
        # Clean text for analysis
        clean_text = self._clean_text(text)
        
        # Perform frequency analysis
        freq_analysis = self._analyze_frequencies(clean_text)
        
        # Test for different cipher types
        caesar_matches = self._detect_caesar_cipher(clean_text, freq_analysis)
        detected_ciphers.extend(caesar_matches)
        
        atbash_matches = self._detect_atbash_cipher(clean_text, freq_analysis)
        detected_ciphers.extend(atbash_matches)
        
        substitution_matches = self._detect_substitution_cipher(clean_text, freq_analysis)
        detected_ciphers.extend(substitution_matches)
        
        # If character data is available, test for biliteral ciphers
        if characters:
            biliteral_matches = self._detect_biliteral_cipher(text, characters)
            detected_ciphers.extend(biliteral_matches)
        
        skip_pattern_matches = self._detect_skip_patterns(clean_text)
        detected_ciphers.extend(skip_pattern_matches)
        
        revolving_matches = self._detect_revolving_cipher(clean_text)
        detected_ciphers.extend(revolving_matches)
        
        # Filter by confidence and remove duplicates
        high_confidence_matches = [match for match in detected_ciphers 
                                 if match.confidence >= self.min_confidence]
        
        return self._remove_duplicate_matches(high_confidence_matches)
    
    def _detect_caesar_cipher(self, text: str, freq_analysis: FrequencyAnalysis) -> List[CipherMatch]:
        """
        Detect Caesar shift ciphers by testing all possible shifts
        """
        matches = []
        
        for shift in range(1, 26):  # Test all possible shifts
            decoded = self._apply_caesar_shift(text, shift)
            
            # Analyze the decoded text
            decoded_freq = self._analyze_frequencies(decoded)
            
            # Calculate confidence based on frequency analysis
            confidence = self._calculate_frequency_confidence(decoded_freq)
            
            # Check for common English patterns
            pattern_confidence = self._check_english_patterns(decoded)
            
            # Combined confidence
            combined_confidence = (confidence + pattern_confidence) / 2
            
            if combined_confidence >= self.min_confidence:
                match = CipherMatch(
                    cipher_type=CipherType.CAESAR,
                    confidence=combined_confidence,
                    key_or_pattern=f"shift_{shift}",
                    decoded_text=decoded,
                    original_text=text,
                    position_start=0,
                    position_end=len(text),
                    method_details={
                        'shift_amount': shift,
                        'frequency_confidence': confidence,
                        'pattern_confidence': pattern_confidence
                    },
                    statistical_analysis={
                        'chi_squared': decoded_freq.chi_squared,
                        'index_of_coincidence': decoded_freq.index_of_coincidence,
                        'entropy': decoded_freq.entropy
                    }
                )
                matches.append(match)
        
        return matches
    
    def _detect_atbash_cipher(self, text: str, freq_analysis: FrequencyAnalysis) -> List[CipherMatch]:
        """
        Detect Atbash cipher (A=Z, B=Y, C=X, etc.)
        """
        matches = []
        
        # Apply Atbash transformation
        decoded = self._apply_atbash(text)
        
        # Analyze the decoded text
        decoded_freq = self._analyze_frequencies(decoded)
        
        # Calculate confidence
        confidence = self._calculate_frequency_confidence(decoded_freq)
        pattern_confidence = self._check_english_patterns(decoded)
        combined_confidence = (confidence + pattern_confidence) / 2
        
        if combined_confidence >= self.min_confidence:
            match = CipherMatch(
                cipher_type=CipherType.ATBASH,
                confidence=combined_confidence,
                key_or_pattern="atbash",
                decoded_text=decoded,
                original_text=text,
                position_start=0,
                position_end=len(text),
                method_details={
                    'transformation': 'A=Z, B=Y, C=X, ...',
                    'frequency_confidence': confidence,
                    'pattern_confidence': pattern_confidence
                },
                statistical_analysis={
                    'chi_squared': decoded_freq.chi_squared,
                    'index_of_coincidence': decoded_freq.index_of_coincidence,
                    'entropy': decoded_freq.entropy
                }
            )
            matches.append(match)
        
        return matches
    
    def _detect_substitution_cipher(self, text: str, freq_analysis: FrequencyAnalysis) -> List[CipherMatch]:
        """
        Detect simple substitution ciphers using frequency analysis
        """
        matches = []
        
        # Get letter frequency order
        text_freq_order = sorted(freq_analysis.letter_frequencies.items(), 
                               key=lambda x: x[1], reverse=True)
        english_freq_order = sorted(self.ENGLISH_FREQUENCIES.items(), 
                                  key=lambda x: x[1], reverse=True)
        
        # Create substitution mapping based on frequency
        substitution_map = {}
        for i, (cipher_letter, _) in enumerate(text_freq_order[:len(english_freq_order)]):
            if i < len(english_freq_order):
                plain_letter, _ = english_freq_order[i]
                substitution_map[cipher_letter] = plain_letter
        
        # Apply substitution
        decoded = self._apply_substitution(text, substitution_map)
        
        # Analyze decoded text
        decoded_freq = self._analyze_frequencies(decoded)
        confidence = self._calculate_frequency_confidence(decoded_freq)
        pattern_confidence = self._check_english_patterns(decoded)
        combined_confidence = (confidence + pattern_confidence) / 2
        
        if combined_confidence >= self.min_confidence:
            match = CipherMatch(
                cipher_type=CipherType.SUBSTITUTION,
                confidence=combined_confidence,
                key_or_pattern=str(substitution_map),
                decoded_text=decoded,
                original_text=text,
                position_start=0,
                position_end=len(text),
                method_details={
                    'substitution_mapping': substitution_map,
                    'frequency_confidence': confidence,
                    'pattern_confidence': pattern_confidence
                },
                statistical_analysis={
                    'chi_squared': decoded_freq.chi_squared,
                    'index_of_coincidence': decoded_freq.index_of_coincidence,
                    'entropy': decoded_freq.entropy
                }
            )
            matches.append(match)
        
        return matches
    
    def _detect_biliteral_cipher(self, text: str, characters: List[Character]) -> List[CipherMatch]:
        """
        Detect biliteral ciphers based on font variations (italic/roman)
        """
        matches = []
        
        # Group characters by font style
        italic_chars = []
        roman_chars = []
        
        for char in characters:
            # Assume we have font style information in character metadata
            font_style = getattr(char, 'font_style', 'roman')
            if font_style == 'italic':
                italic_chars.append(char.character)
            else:
                roman_chars.append(char.character)
        
        # Create binary representation (italic=1, roman=0)
        binary_sequence = []
        for char in characters:
            font_style = getattr(char, 'font_style', 'roman')
            binary_sequence.append('1' if font_style == 'italic' else '0')
        
        # Try to decode as biliteral cipher
        if len(binary_sequence) >= 10:  # Need minimum length
            decoded_attempts = self._decode_biliteral(''.join(binary_sequence))
            
            for decoded in decoded_attempts:
                if decoded and len(decoded) > 3:
                    confidence = self._check_english_patterns(decoded)
                    
                    if confidence >= self.min_confidence:
                        match = CipherMatch(
                            cipher_type=CipherType.BILITERAL,
                            confidence=confidence,
                            key_or_pattern="italic=1, roman=0",
                            decoded_text=decoded,
                            original_text=text,
                            position_start=0,
                            position_end=len(text),
                            method_details={
                                'binary_sequence': ''.join(binary_sequence),
                                'italic_count': len(italic_chars),
                                'roman_count': len(roman_chars),
                                'decoding_method': 'font_style_binary'
                            },
                            statistical_analysis={
                                'pattern_confidence': confidence,
                                'binary_length': len(binary_sequence)
                            }
                        )
                        matches.append(match)
        
        return matches
    
    def _detect_skip_patterns(self, text: str) -> List[CipherMatch]:
        """
        Detect skip patterns (every nth character)
        """
        matches = []
        
        # Test various skip patterns
        for skip in range(2, min(20, len(text) // 4)):  # Test skips from 2 to 19
            for start_pos in range(skip):
                # Extract characters at skip intervals
                skip_text = ""
                positions = []
                
                pos = start_pos
                while pos < len(text):
                    skip_text += text[pos]
                    positions.append(pos)
                    pos += skip
                
                if len(skip_text) >= self.min_text_length:
                    # Analyze the skip pattern text
                    freq_analysis = self._analyze_frequencies(skip_text)
                    confidence = self._calculate_frequency_confidence(freq_analysis)
                    pattern_confidence = self._check_english_patterns(skip_text)
                    combined_confidence = (confidence + pattern_confidence) / 2
                    
                    if combined_confidence >= self.min_confidence:
                        match = CipherMatch(
                            cipher_type=CipherType.SKIP_PATTERN,
                            confidence=combined_confidence,
                            key_or_pattern=f"skip_{skip}_start_{start_pos}",
                            decoded_text=skip_text,
                            original_text=text,
                            position_start=start_pos,
                            position_end=positions[-1] if positions else start_pos,
                            method_details={
                                'skip_amount': skip,
                                'start_position': start_pos,
                                'extracted_positions': positions,
                                'frequency_confidence': confidence,
                                'pattern_confidence': pattern_confidence
                            },
                            statistical_analysis={
                                'chi_squared': freq_analysis.chi_squared,
                                'index_of_coincidence': freq_analysis.index_of_coincidence,
                                'entropy': freq_analysis.entropy
                            }
                        )
                        matches.append(match)
        
        return matches
    
    def _detect_revolving_cipher(self, text: str) -> List[CipherMatch]:
        """
        Detect revolving ciphers (MOAI-style patterns)
        """
        matches = []
        
        # Look for repeating patterns that could indicate revolving cipher
        pattern_lengths = [2, 3, 4, 5, 6, 8, 10, 12]  # Common revolving pattern lengths
        
        for pattern_length in pattern_lengths:
            if len(text) < pattern_length * 3:  # Need at least 3 repetitions
                continue
            
            # Extract text in groups of pattern_length
            groups = []
            for i in range(0, len(text) - pattern_length + 1, pattern_length):
                group = text[i:i + pattern_length]
                if len(group) == pattern_length:
                    groups.append(group)
            
            if len(groups) >= 3:  # Need minimum groups for analysis
                # Try different revolving transformations
                for shift_pattern in self._generate_shift_patterns(pattern_length):
                    decoded_groups = []
                    
                    for i, group in enumerate(groups):
                        # Apply revolving shift based on position
                        shift_index = i % len(shift_pattern)
                        shift_amount = shift_pattern[shift_index]
                        decoded_group = self._apply_caesar_shift(group, shift_amount)
                        decoded_groups.append(decoded_group)
                    
                    decoded_text = ''.join(decoded_groups)
                    
                    # Analyze decoded text
                    if len(decoded_text) >= self.min_text_length:
                        freq_analysis = self._analyze_frequencies(decoded_text)
                        confidence = self._calculate_frequency_confidence(freq_analysis)
                        pattern_confidence = self._check_english_patterns(decoded_text)
                        combined_confidence = (confidence + pattern_confidence) / 2
                        
                        if combined_confidence >= self.min_confidence:
                            match = CipherMatch(
                                cipher_type=CipherType.REVOLVING,
                                confidence=combined_confidence,
                                key_or_pattern=f"revolving_{pattern_length}_{shift_pattern}",
                                decoded_text=decoded_text,
                                original_text=text,
                                position_start=0,
                                position_end=len(text),
                                method_details={
                                    'pattern_length': pattern_length,
                                    'shift_pattern': shift_pattern,
                                    'group_count': len(groups),
                                    'frequency_confidence': confidence,
                                    'pattern_confidence': pattern_confidence
                                },
                                statistical_analysis={
                                    'chi_squared': freq_analysis.chi_squared,
                                    'index_of_coincidence': freq_analysis.index_of_coincidence,
                                    'entropy': freq_analysis.entropy
                                }
                            )
                            matches.append(match)
        
        return matches
    
    def _analyze_frequencies(self, text: str) -> FrequencyAnalysis:
        """
        Perform comprehensive frequency analysis on text
        """
        clean_text = self._clean_text(text).upper()
        
        # Letter frequencies
        letter_counts = Counter(clean_text)
        total_letters = sum(letter_counts.values())
        letter_frequencies = {letter: (count / total_letters) * 100 
                            for letter, count in letter_counts.items() 
                            if letter.isalpha()}
        
        # Bigram frequencies
        bigrams = [clean_text[i:i+2] for i in range(len(clean_text)-1) 
                  if clean_text[i].isalpha() and clean_text[i+1].isalpha()]
        bigram_counts = Counter(bigrams)
        total_bigrams = len(bigrams)
        bigram_frequencies = {bigram: (count / total_bigrams) * 100 
                            for bigram, count in bigram_counts.items()}
        
        # Trigram frequencies
        trigrams = [clean_text[i:i+3] for i in range(len(clean_text)-2) 
                   if all(c.isalpha() for c in clean_text[i:i+3])]
        trigram_counts = Counter(trigrams)
        total_trigrams = len(trigrams)
        trigram_frequencies = {trigram: (count / total_trigrams) * 100 
                             for trigram, count in trigram_counts.items()}
        
        # Index of Coincidence
        ic = self._calculate_index_of_coincidence(clean_text)
        
        # Chi-squared test against English
        chi_squared = self._calculate_chi_squared(letter_frequencies)
        
        # Entropy
        entropy = self._calculate_entropy(letter_frequencies)
        
        return FrequencyAnalysis(
            letter_frequencies=letter_frequencies,
            bigram_frequencies=bigram_frequencies,
            trigram_frequencies=trigram_frequencies,
            index_of_coincidence=ic,
            chi_squared=chi_squared,
            entropy=entropy
        )
    
    def _calculate_frequency_confidence(self, freq_analysis: FrequencyAnalysis) -> float:
        """
        Calculate confidence based on how closely frequencies match English
        """
        # Chi-squared test (lower is better)
        chi_squared_score = max(0, 1 - (freq_analysis.chi_squared / 1000))  # Normalize
        
        # Index of Coincidence (English ≈ 0.067)
        ic_score = max(0, 1 - abs(freq_analysis.index_of_coincidence - 0.067) * 10)
        
        # Entropy (English ≈ 4.1)
        entropy_score = max(0, 1 - abs(freq_analysis.entropy - 4.1) / 4.1)
        
        # Combined score
        return (chi_squared_score + ic_score + entropy_score) / 3
    
    def _check_english_patterns(self, text: str) -> float:
        """
        Check for common English patterns and words
        """
        clean_text = self._clean_text(text).upper()
        
        if not clean_text:
            return 0.0
        
        # Check for common bigrams
        bigram_score = 0
        bigram_count = 0
        for bigram in self.COMMON_BIGRAMS:
            count = clean_text.count(bigram)
            if count > 0:
                bigram_score += count
            bigram_count += len(clean_text) - 1  # Total possible bigrams
        
        bigram_score = bigram_score / max(1, len(self.COMMON_BIGRAMS)) if bigram_count > 0 else 0
        
        # Check for common trigrams
        trigram_score = 0
        for trigram in self.COMMON_TRIGRAMS:
            count = clean_text.count(trigram)
            if count > 0:
                trigram_score += count
        
        trigram_score = trigram_score / max(1, len(self.COMMON_TRIGRAMS))
        
        # Check for common English words
        common_words = ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY']
        word_score = 0
        words = clean_text.split()
        total_words = len(words)
        
        if total_words > 0:
            for word in common_words:
                if word in words:
                    word_score += words.count(word)
            word_score = word_score / total_words
        
        # Check for vowel distribution (English has ~40% vowels)
        vowels = 'AEIOU'
        vowel_count = sum(clean_text.count(v) for v in vowels)
        vowel_ratio = vowel_count / len(clean_text) if clean_text else 0
        vowel_score = 1 - abs(vowel_ratio - 0.4) * 2.5  # Penalize deviation from 40%
        vowel_score = max(0, vowel_score)
        
        # Combined pattern score with weights
        return (bigram_score * 0.3 + trigram_score * 0.3 + word_score * 0.3 + vowel_score * 0.1)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for analysis (remove non-alphabetic characters)
        """
        return re.sub(r'[^A-Za-z]', '', text)
    
    def _apply_caesar_shift(self, text: str, shift: int) -> str:
        """
        Apply Caesar cipher shift to text
        """
        result = ""
        for char in text:
            if char.isalpha():
                # Handle uppercase and lowercase
                base = ord('A') if char.isupper() else ord('a')
                shifted = (ord(char) - base - shift) % 26
                result += chr(shifted + base)
            else:
                result += char
        return result
    
    def _apply_atbash(self, text: str) -> str:
        """
        Apply Atbash cipher transformation
        """
        result = ""
        for char in text:
            if char.isalpha():
                if char.isupper():
                    result += chr(ord('Z') - (ord(char) - ord('A')))
                else:
                    result += chr(ord('z') - (ord(char) - ord('a')))
            else:
                result += char
        return result
    
    def _apply_substitution(self, text: str, substitution_map: Dict[str, str]) -> str:
        """
        Apply substitution cipher mapping
        """
        result = ""
        for char in text:
            if char.upper() in substitution_map:
                mapped_char = substitution_map[char.upper()]
                result += mapped_char.lower() if char.islower() else mapped_char
            else:
                result += char
        return result
    
    def _decode_biliteral(self, binary_sequence: str) -> List[str]:
        """
        Decode biliteral cipher from binary sequence
        """
        decoded_attempts = []
        
        # Standard biliteral alphabet (A=00000, B=00001, etc.)
        biliteral_alphabet = {}
        for i, letter in enumerate(string.ascii_uppercase):
            binary = format(i, '05b')  # 5-bit binary
            biliteral_alphabet[binary] = letter
        
        # Try decoding with 5-bit groups
        if len(binary_sequence) >= 5:
            decoded = ""
            for i in range(0, len(binary_sequence) - 4, 5):
                group = binary_sequence[i:i+5]
                if group in biliteral_alphabet:
                    decoded += biliteral_alphabet[group]
            
            if decoded:
                decoded_attempts.append(decoded)
        
        # Try other group sizes (4-bit, 6-bit)
        for group_size in [4, 6]:
            if len(binary_sequence) >= group_size:
                decoded = ""
                for i in range(0, len(binary_sequence) - group_size + 1, group_size):
                    group = binary_sequence[i:i+group_size]
                    # Simple mapping for demonstration
                    char_index = int(group, 2) % 26
                    decoded += chr(ord('A') + char_index)
                
                if decoded:
                    decoded_attempts.append(decoded)
        
        return decoded_attempts
    
    def _generate_shift_patterns(self, pattern_length: int) -> List[List[int]]:
        """
        Generate possible shift patterns for revolving ciphers
        """
        patterns = []
        
        # Simple patterns
        patterns.append([1] * pattern_length)  # All shift by 1
        patterns.append([i + 1 for i in range(pattern_length)])  # 1, 2, 3, ...
        patterns.append([(-i - 1) for i in range(pattern_length)])  # -1, -2, -3, ...
        
        # Alternating patterns
        if pattern_length >= 2:
            patterns.append([1, -1] * (pattern_length // 2))  # 1, -1, 1, -1, ...
            patterns.append([2, -2] * (pattern_length // 2))  # 2, -2, 2, -2, ...
        
        # Mathematical patterns
        if pattern_length >= 3:
            base_pattern = [1, 2, 3]
            full_pattern = (base_pattern * ((pattern_length // 3) + 1))[:pattern_length]
            patterns.append(full_pattern)
            
            pi_pattern = [3, 1, 4]
            full_pi_pattern = (pi_pattern * ((pattern_length // 3) + 1))[:pattern_length]
            patterns.append(full_pi_pattern)
        
        return patterns[:10]  # Limit to prevent excessive computation
    
    def _calculate_index_of_coincidence(self, text: str) -> float:
        """
        Calculate Index of Coincidence for the text
        """
        clean_text = self._clean_text(text).upper()
        n = len(clean_text)
        
        if n <= 1:
            return 0
        
        letter_counts = Counter(clean_text)
        ic = sum(count * (count - 1) for count in letter_counts.values()) / (n * (n - 1))
        
        return ic
    
    def _calculate_chi_squared(self, observed_frequencies: Dict[str, float]) -> float:
        """
        Calculate chi-squared statistic against English letter frequencies
        """
        chi_squared = 0
        
        for letter in string.ascii_uppercase:
            observed = observed_frequencies.get(letter, 0)
            expected = self.ENGLISH_FREQUENCIES.get(letter, 0)
            
            if expected > 0:
                chi_squared += ((observed - expected) ** 2) / expected
        
        return chi_squared
    
    def _calculate_entropy(self, frequencies: Dict[str, float]) -> float:
        """
        Calculate Shannon entropy of the frequency distribution
        """
        entropy = 0
        total = sum(frequencies.values())
        
        for freq in frequencies.values():
            if freq > 0:
                probability = freq / total
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _remove_duplicate_matches(self, matches: List[CipherMatch]) -> List[CipherMatch]:
        """
        Remove duplicate or overlapping cipher matches
        """
        # Sort by confidence (highest first)
        sorted_matches = sorted(matches, key=lambda x: x.confidence, reverse=True)
        
        unique_matches = []
        used_ranges = []
        
        for match in sorted_matches:
            # Check if this match overlaps significantly with existing matches
            overlap = False
            for start, end in used_ranges:
                overlap_start = max(match.position_start, start)
                overlap_end = min(match.position_end, end)
                overlap_length = max(0, overlap_end - overlap_start)
                
                # If more than 50% overlap, consider it duplicate
                match_length = match.position_end - match.position_start
                if overlap_length > match_length * 0.5:
                    overlap = True
                    break
            
            if not overlap:
                unique_matches.append(match)
                used_ranges.append((match.position_start, match.position_end))
        
        return unique_matches
    
    def analyze_cipher_patterns(self, matches: List[CipherMatch]) -> Dict[str, Any]:
        """
        Analyze patterns in detected ciphers for cross-document correlation
        """
        analysis = {
            'cipher_type_distribution': {},
            'confidence_statistics': {},
            'pattern_complexity': {},
            'potential_relationships': []
        }
        
        if not matches:
            return analysis
        
        # Cipher type distribution
        type_counts = Counter(match.cipher_type.value for match in matches)
        analysis['cipher_type_distribution'] = dict(type_counts)
        
        # Confidence statistics
        confidences = [match.confidence for match in matches]
        analysis['confidence_statistics'] = {
            'mean': sum(confidences) / len(confidences),
            'min': min(confidences),
            'max': max(confidences),
            'high_confidence_count': len([c for c in confidences if c >= 0.8])
        }
        
        # Pattern complexity analysis
        for match in matches:
            cipher_type = match.cipher_type.value
            if cipher_type not in analysis['pattern_complexity']:
                analysis['pattern_complexity'][cipher_type] = {
                    'average_length': 0,
                    'key_complexity': 0,
                    'statistical_strength': 0
                }
            
            # Update complexity metrics
            complexity = analysis['pattern_complexity'][cipher_type]
            text_length = len(match.decoded_text)
            complexity['average_length'] = (complexity['average_length'] + text_length) / 2
            complexity['statistical_strength'] = (complexity['statistical_strength'] + 
                                                match.statistical_analysis.get('chi_squared', 0)) / 2
        
        # Look for potential relationships between matches
        for i, match1 in enumerate(matches):
            for match2 in matches[i+1:]:
                if self._are_related_ciphers(match1, match2):
                    analysis['potential_relationships'].append({
                        'cipher1': match1.cipher_type.value,
                        'cipher2': match2.cipher_type.value,
                        'relationship_type': 'similar_pattern',
                        'confidence': min(match1.confidence, match2.confidence)
                    })
        
        return analysis
    
    def _are_related_ciphers(self, match1: CipherMatch, match2: CipherMatch) -> bool:
        """
        Determine if two cipher matches are potentially related
        """
        # Same cipher type with similar keys
        if match1.cipher_type == match2.cipher_type:
            return True
        
        # Different types but similar statistical properties
        stats1 = match1.statistical_analysis
        stats2 = match2.statistical_analysis
        
        if ('chi_squared' in stats1 and 'chi_squared' in stats2 and
            abs(stats1['chi_squared'] - stats2['chi_squared']) < 50):
            return True
        
        return False
    
    def generate_cipher_report(self, matches: List[CipherMatch]) -> Dict[str, Any]:
        """
        Generate comprehensive report of cipher detection results
        """
        report = {
            'summary': {
                'total_matches': len(matches),
                'cipher_types_found': len(set(match.cipher_type for match in matches)),
                'highest_confidence': max([match.confidence for match in matches]) if matches else 0,
                'average_confidence': sum([match.confidence for match in matches]) / len(matches) if matches else 0
            },
            'matches_by_type': {},
            'statistical_analysis': {},
            'recommendations': []
        }
        
        # Group matches by cipher type
        for match in matches:
            cipher_type = match.cipher_type.value
            if cipher_type not in report['matches_by_type']:
                report['matches_by_type'][cipher_type] = []
            
            match_info = {
                'confidence': match.confidence,
                'key_pattern': match.key_or_pattern,
                'decoded_preview': match.decoded_text[:100] + "..." if len(match.decoded_text) > 100 else match.decoded_text,
                'position': f"{match.position_start}-{match.position_end}",
                'method_details': match.method_details
            }
            report['matches_by_type'][cipher_type].append(match_info)
        
        # Statistical analysis
        report['statistical_analysis'] = self.analyze_cipher_patterns(matches)
        
        # Generate recommendations
        report['recommendations'] = self._generate_cipher_recommendations(matches)
        
        return report
    
    def _generate_cipher_recommendations(self, matches: List[CipherMatch]) -> List[str]:
        """
        Generate recommendations based on cipher detection results
        """
        recommendations = []
        
        if not matches:
            recommendations.append("No cipher patterns detected. Consider adjusting detection parameters or analyzing different text sections.")
            return recommendations
        
        # High confidence matches
        high_confidence = [m for m in matches if m.confidence >= 0.8]
        if high_confidence:
            recommendations.append(f"Found {len(high_confidence)} high-confidence cipher matches. Focus analysis on these patterns.")
        
        # Multiple cipher types
        cipher_types = set(match.cipher_type for match in matches)
        if len(cipher_types) > 2:
            recommendations.append("Multiple cipher types detected. This may indicate layered encryption or different encoding schemes.")
        
        # Caesar cipher specific
        caesar_matches = [m for m in matches if m.cipher_type == CipherType.CAESAR]
        if len(caesar_matches) > 1:
            shifts = [int(m.key_or_pattern.split('_')[1]) for m in caesar_matches]
            if len(set(shifts)) == 1:
                recommendations.append(f"Consistent Caesar shift of {shifts[0]} detected across multiple sections.")
        
        # Skip pattern analysis
        skip_matches = [m for m in matches if m.cipher_type == CipherType.SKIP_PATTERN]
        if skip_matches:
            recommendations.append("Skip patterns detected. Consider analyzing document layout and spacing for additional clues.")
        
        # Biliteral cipher
        biliteral_matches = [m for m in matches if m.cipher_type == CipherType.BILITERAL]
        if biliteral_matches:
            recommendations.append("Font-based biliteral cipher detected. Examine typography and formatting variations.")
        
        return recommendations