"""
Cipher Explanation and Validation System
Provides detailed explanations of detected cipher methods, statistical validation,
confidence scoring, and cipher method recommendations.
"""
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import re
import string
import math
import statistics
from enum import Enum

from app.services.cipher_detector import CipherDetector, CipherMatch, CipherType, FrequencyAnalysis


@dataclass
class CipherExplanation:
    """Detailed explanation of a cipher method"""
    cipher_type: CipherType
    method_name: str
    description: str
    historical_context: str
    how_it_works: str
    key_characteristics: List[str]
    detection_method: str
    example_transformation: str
    difficulty_level: str
    common_uses: List[str]


@dataclass
class ValidationResult:
    """Statistical validation results for a cipher detection"""
    is_valid: bool
    confidence_score: float
    statistical_tests: Dict[str, float]
    validation_details: Dict[str, Any]
    reliability_assessment: str
    potential_issues: List[str]
    recommendation: str


@dataclass
class CipherRecommendation:
    """Recommendation for cipher analysis approach"""
    recommended_methods: List[str]
    analysis_priority: List[CipherType]
    suggested_parameters: Dict[str, Any]
    expected_success_rate: float
    reasoning: str
    alternative_approaches: List[str]


class ValidationLevel(Enum):
    """Levels of validation rigor"""
    BASIC = "basic"
    STANDARD = "standard"
    RIGOROUS = "rigorous"
    COMPREHENSIVE = "comprehensive"


class CipherExplanationValidator:
    """
    System for explaining cipher methods and validating detection results
    """
    
    def __init__(self):
        self.cipher_detector = CipherDetector()
        self.min_validation_confidence = 0.5
        self.statistical_significance_threshold = 0.05
        
        # Initialize cipher explanations database
        self.cipher_explanations = self._initialize_cipher_explanations()
        
        # Statistical validation thresholds
        self.validation_thresholds = {
            'chi_squared_max': 50.0,
            'index_of_coincidence_min': 0.04,
            'index_of_coincidence_max': 0.08,
            'entropy_min': 3.0,
            'entropy_max': 5.0,
            'pattern_confidence_min': 0.3,
            'frequency_match_min': 0.4
        }
    
    def explain_cipher_method(self, cipher_type: CipherType) -> CipherExplanation:
        """
        Provide detailed explanation of a cipher method
        """
        return self.cipher_explanations.get(cipher_type, self._create_generic_explanation(cipher_type))
    
    def validate_cipher_detection(self, cipher_match: CipherMatch, 
                                validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """
        Perform statistical validation of cipher detection results
        """
        validation_tests = {}
        validation_details = {}
        potential_issues = []
        
        # Basic validation tests
        validation_tests.update(self._perform_basic_validation(cipher_match))
        
        if validation_level in [ValidationLevel.STANDARD, ValidationLevel.RIGOROUS, ValidationLevel.COMPREHENSIVE]:
            # Standard validation tests
            validation_tests.update(self._perform_frequency_validation(cipher_match))
            validation_tests.update(self._perform_pattern_validation(cipher_match))
        
        if validation_level in [ValidationLevel.RIGOROUS, ValidationLevel.COMPREHENSIVE]:
            # Rigorous validation tests
            validation_tests.update(self._perform_statistical_significance_tests(cipher_match))
            validation_tests.update(self._perform_cross_validation(cipher_match))
        
        if validation_level == ValidationLevel.COMPREHENSIVE:
            # Comprehensive validation tests
            validation_tests.update(self._perform_linguistic_validation(cipher_match))
            validation_tests.update(self._perform_contextual_validation(cipher_match))
        
        # Calculate overall confidence score
        confidence_score = self._calculate_validation_confidence(validation_tests)
        
        # Determine if validation passes
        is_valid = confidence_score >= self.min_validation_confidence
        
        # Generate reliability assessment
        reliability_assessment = self._assess_reliability(confidence_score, validation_tests)
        
        # Identify potential issues
        potential_issues = self._identify_validation_issues(validation_tests, cipher_match)
        
        # Generate recommendation
        recommendation = self._generate_validation_recommendation(
            is_valid, confidence_score, potential_issues, cipher_match
        )
        
        # Compile validation details
        validation_details = {
            'validation_level': validation_level.value,
            'tests_performed': list(validation_tests.keys()),
            'cipher_characteristics': self._analyze_cipher_characteristics(cipher_match),
            'text_properties': self._analyze_text_properties(cipher_match.decoded_text),
            'method_specific_analysis': self._perform_method_specific_analysis(cipher_match)
        }
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            statistical_tests=validation_tests,
            validation_details=validation_details,
            reliability_assessment=reliability_assessment,
            potential_issues=potential_issues,
            recommendation=recommendation
        )
    
    def generate_cipher_recommendations(self, text: str, 
                                     analysis_context: Dict[str, Any] = None) -> CipherRecommendation:
        """
        Generate recommendations for cipher analysis approach
        """
        # Analyze text characteristics
        text_analysis = self._analyze_text_for_recommendations(text)
        
        # Determine recommended methods based on text characteristics
        recommended_methods = self._determine_recommended_methods(text_analysis, analysis_context)
        
        # Set analysis priority
        analysis_priority = self._determine_analysis_priority(text_analysis, recommended_methods)
        
        # Suggest parameters
        suggested_parameters = self._suggest_analysis_parameters(text_analysis, analysis_priority)
        
        # Estimate success rate
        expected_success_rate = self._estimate_success_rate(text_analysis, recommended_methods)
        
        # Generate reasoning
        reasoning = self._generate_recommendation_reasoning(
            text_analysis, recommended_methods, analysis_priority
        )
        
        # Suggest alternative approaches
        alternative_approaches = self._suggest_alternative_approaches(
            text_analysis, recommended_methods
        )
        
        return CipherRecommendation(
            recommended_methods=recommended_methods,
            analysis_priority=analysis_priority,
            suggested_parameters=suggested_parameters,
            expected_success_rate=expected_success_rate,
            reasoning=reasoning,
            alternative_approaches=alternative_approaches
        )
    
    def explain_detection_results(self, cipher_matches: List[CipherMatch]) -> Dict[str, Any]:
        """
        Provide comprehensive explanation of cipher detection results
        """
        explanations = {}
        
        for match in cipher_matches:
            cipher_explanation = self.explain_cipher_method(match.cipher_type)
            validation_result = self.validate_cipher_detection(match)
            
            explanations[f"{match.cipher_type.value}_{match.confidence:.2f}"] = {
                'cipher_match': match,
                'explanation': cipher_explanation,
                'validation': validation_result,
                'detailed_analysis': self._create_detailed_analysis(match),
                'visual_examples': self._create_visual_examples(match),
                'next_steps': self._suggest_next_steps(match, validation_result)
            }
        
        # Add summary analysis
        explanations['summary'] = self._create_summary_analysis(cipher_matches)
        
        return explanations
    
    def _initialize_cipher_explanations(self) -> Dict[CipherType, CipherExplanation]:
        """
        Initialize database of cipher explanations
        """
        explanations = {}
        
        # Caesar Cipher
        explanations[CipherType.CAESAR] = CipherExplanation(
            cipher_type=CipherType.CAESAR,
            method_name="Caesar Cipher",
            description="A substitution cipher where each letter is shifted by a fixed number of positions in the alphabet.",
            historical_context="Named after Julius Caesar, who used it for military communications around 50 BC. One of the oldest known ciphers.",
            how_it_works="Each letter in the plaintext is shifted by a constant number (the key). For example, with a shift of 3: A→D, B→E, C→F, etc.",
            key_characteristics=[
                "Fixed shift amount for all letters",
                "Preserves letter frequency patterns",
                "Vulnerable to frequency analysis",
                "Only 25 possible keys (excluding shift of 0)"
            ],
            detection_method="Test all 25 possible shifts and analyze resulting text for English-like characteristics using frequency analysis and pattern matching.",
            example_transformation="HELLO → KHOOR (shift of 3)",
            difficulty_level="Beginner",
            common_uses=[
                "Ancient military communications",
                "Simple message obfuscation",
                "Educational cryptography examples",
                "ROT13 text encoding"
            ]
        )
        
        # Atbash Cipher
        explanations[CipherType.ATBASH] = CipherExplanation(
            cipher_type=CipherType.ATBASH,
            method_name="Atbash Cipher",
            description="A monoalphabetic substitution cipher where the alphabet is reversed (A↔Z, B↔Y, C↔X, etc.).",
            historical_context="Originally used for Hebrew text in biblical times. The name comes from the first and last letters of the Hebrew alphabet.",
            how_it_works="Each letter is replaced with its counterpart from the reversed alphabet. A becomes Z, B becomes Y, and so on.",
            key_characteristics=[
                "Fixed substitution pattern",
                "Self-reciprocal (encoding and decoding use same process)",
                "Preserves letter frequency distribution",
                "No key required - the method is the key"
            ],
            detection_method="Apply the Atbash transformation and analyze the result for English-like characteristics.",
            example_transformation="HELLO → SVOOL",
            difficulty_level="Beginner",
            common_uses=[
                "Biblical and religious texts",
                "Simple message concealment",
                "Historical document encoding",
                "Puzzle and game ciphers"
            ]
        )
        
        # Substitution Cipher
        explanations[CipherType.SUBSTITUTION] = CipherExplanation(
            cipher_type=CipherType.SUBSTITUTION,
            method_name="Simple Substitution Cipher",
            description="A cipher where each letter of the alphabet is replaced with another letter according to a fixed substitution table.",
            historical_context="Used throughout history, notably by Mary Queen of Scots in the 16th century. Basis for many cryptographic systems.",
            how_it_works="A substitution alphabet is created (often by rearranging the standard alphabet), and each plaintext letter is replaced with its corresponding ciphertext letter.",
            key_characteristics=[
                "One-to-one letter mapping",
                "Preserves letter frequencies",
                "Vulnerable to frequency analysis",
                "26! possible keys (extremely large key space)"
            ],
            detection_method="Use frequency analysis to match cipher letters with probable plaintext letters based on English letter frequencies.",
            example_transformation="HELLO → XAZZM (using a specific substitution key)",
            difficulty_level="Intermediate",
            common_uses=[
                "Historical diplomatic communications",
                "Newspaper puzzle sections",
                "Literary works and novels",
                "Basic cryptographic education"
            ]
        )
        
        # Biliteral Cipher
        explanations[CipherType.BILITERAL] = CipherExplanation(
            cipher_type=CipherType.BILITERAL,
            method_name="Biliteral Cipher (Bacon's Cipher)",
            description="A steganographic cipher that hides messages using two different forms of the same character (e.g., italic vs. roman type).",
            historical_context="Invented by Francis Bacon in 1605. Used to hide messages in plain sight within seemingly normal text.",
            how_it_works="Each letter is represented by a 5-bit binary code, with the two binary states represented by different typefaces or character forms.",
            key_characteristics=[
                "Steganographic (hidden in plain sight)",
                "Requires two distinct character forms",
                "5-bit encoding allows for 32 combinations",
                "Message length is 5 times the hidden text length"
            ],
            detection_method="Identify patterns in character formatting (italic/roman, bold/normal, etc.) and attempt binary decoding.",
            example_transformation="italic-roman-italic-italic-roman → 10110 → specific letter",
            difficulty_level="Advanced",
            common_uses=[
                "Historical document concealment",
                "Literary steganography",
                "Academic and scholarly hiding",
                "Modern digital steganography"
            ]
        )
        
        # Skip Pattern Cipher
        explanations[CipherType.SKIP_PATTERN] = CipherExplanation(
            cipher_type=CipherType.SKIP_PATTERN,
            method_name="Skip Pattern Cipher",
            description="A cipher where the hidden message is formed by reading every nth character from the text.",
            historical_context="Used in various forms throughout history, including in some interpretations of biblical texts and literary works.",
            how_it_works="Starting at a specific position, every nth character is extracted to form the hidden message. The skip distance and starting position form the key.",
            key_characteristics=[
                "Requires knowledge of skip distance and starting position",
                "Hidden message is much shorter than cover text",
                "Can be combined with other cipher methods",
                "Vulnerable if pattern is discovered"
            ],
            detection_method="Test various skip distances and starting positions, analyzing extracted text for meaningful content.",
            example_transformation="Every 3rd character starting at position 2: 'abCdeFghIjkL' → 'CeIL'",
            difficulty_level="Intermediate",
            common_uses=[
                "Literary steganography",
                "Religious text interpretation",
                "Modern puzzle design",
                "Academic cryptographic exercises"
            ]
        )
        
        # Revolving Cipher
        explanations[CipherType.REVOLVING] = CipherExplanation(
            cipher_type=CipherType.REVOLVING,
            method_name="Revolving Cipher (Polyalphabetic)",
            description="A cipher that uses multiple substitution alphabets, cycling through them according to a pattern.",
            historical_context="Advanced form of polyalphabetic ciphers, used in sophisticated cryptographic systems. Related to Vigenère-type ciphers.",
            how_it_works="The text is divided into groups, and each group is encrypted with a different shift or substitution pattern that revolves according to a predetermined sequence.",
            key_characteristics=[
                "Multiple encryption keys in sequence",
                "More secure than simple substitution",
                "Pattern length determines key complexity",
                "Requires knowledge of the revolving pattern"
            ],
            detection_method="Analyze text for repeating patterns and test various revolving shift sequences.",
            example_transformation="Pattern [1,3,2]: ABC → BDF (shifts of 1,3,2 respectively)",
            difficulty_level="Advanced",
            common_uses=[
                "Military communications",
                "Diplomatic correspondence",
                "Advanced puzzle systems",
                "Historical cryptographic challenges"
            ]
        )
        
        return explanations
    
    def _create_generic_explanation(self, cipher_type: CipherType) -> CipherExplanation:
        """
        Create a generic explanation for unknown cipher types
        """
        return CipherExplanation(
            cipher_type=cipher_type,
            method_name=f"{cipher_type.value.replace('_', ' ').title()} Cipher",
            description=f"A {cipher_type.value} cipher method.",
            historical_context="Historical context not available.",
            how_it_works="Method details not available.",
            key_characteristics=["Characteristics not documented"],
            detection_method="Standard cryptographic analysis methods.",
            example_transformation="Example not available.",
            difficulty_level="Unknown",
            common_uses=["Usage patterns not documented"]
        )
    
    def _perform_basic_validation(self, cipher_match: CipherMatch) -> Dict[str, float]:
        """
        Perform basic validation tests
        """
        tests = {}
        
        # Confidence threshold test
        tests['confidence_threshold'] = 1.0 if cipher_match.confidence >= 0.3 else 0.0
        
        # Text length adequacy
        min_length = 20
        tests['text_length_adequacy'] = min(1.0, len(cipher_match.decoded_text) / min_length)
        
        # Decoded text quality (basic character distribution)
        decoded_clean = re.sub(r'[^A-Za-z]', '', cipher_match.decoded_text)
        if decoded_clean:
            unique_chars = len(set(decoded_clean.upper()))
            tests['character_diversity'] = min(1.0, unique_chars / 10)  # Expect at least 10 different letters
        else:
            tests['character_diversity'] = 0.0
        
        return tests
    
    def _perform_frequency_validation(self, cipher_match: CipherMatch) -> Dict[str, float]:
        """
        Perform frequency-based validation tests
        """
        tests = {}
        
        # Analyze decoded text frequencies
        freq_analysis = self.cipher_detector._analyze_frequencies(cipher_match.decoded_text)
        
        # Chi-squared test
        chi_squared_score = max(0, 1 - (freq_analysis.chi_squared / self.validation_thresholds['chi_squared_max']))
        tests['chi_squared_validation'] = chi_squared_score
        
        # Index of Coincidence validation
        ic = freq_analysis.index_of_coincidence
        ic_min = self.validation_thresholds['index_of_coincidence_min']
        ic_max = self.validation_thresholds['index_of_coincidence_max']
        
        if ic_min <= ic <= ic_max:
            tests['index_of_coincidence_validation'] = 1.0
        else:
            # Penalize based on distance from acceptable range
            if ic < ic_min:
                tests['index_of_coincidence_validation'] = ic / ic_min
            else:
                tests['index_of_coincidence_validation'] = ic_max / ic
        
        # Entropy validation
        entropy = freq_analysis.entropy
        entropy_min = self.validation_thresholds['entropy_min']
        entropy_max = self.validation_thresholds['entropy_max']
        
        if entropy_min <= entropy <= entropy_max:
            tests['entropy_validation'] = 1.0
        else:
            if entropy < entropy_min:
                tests['entropy_validation'] = entropy / entropy_min
            else:
                tests['entropy_validation'] = entropy_max / entropy
        
        return tests
    
    def _perform_pattern_validation(self, cipher_match: CipherMatch) -> Dict[str, float]:
        """
        Perform pattern-based validation tests
        """
        tests = {}
        
        # English pattern validation
        pattern_confidence = self.cipher_detector._check_english_patterns(cipher_match.decoded_text)
        tests['english_patterns'] = pattern_confidence
        
        # Word formation validation
        words = re.findall(r'[A-Za-z]+', cipher_match.decoded_text)
        if words:
            # Check for reasonable word lengths
            avg_word_length = sum(len(word) for word in words) / len(words)
            word_length_score = 1.0 if 3 <= avg_word_length <= 7 else max(0, 1 - abs(avg_word_length - 5) / 5)
            tests['word_length_distribution'] = word_length_score
            
            # Check for very short or very long words (potential gibberish indicators)
            reasonable_words = sum(1 for word in words if 2 <= len(word) <= 15)
            tests['reasonable_word_ratio'] = reasonable_words / len(words)
        else:
            tests['word_length_distribution'] = 0.0
            tests['reasonable_word_ratio'] = 0.0
        
        return tests
    
    def _perform_statistical_significance_tests(self, cipher_match: CipherMatch) -> Dict[str, float]:
        """
        Perform statistical significance tests
        """
        tests = {}
        
        # Compare with random text
        random_confidence = self._test_against_random_text(cipher_match.decoded_text)
        tests['random_text_comparison'] = random_confidence
        
        # Method-specific significance tests
        if cipher_match.cipher_type == CipherType.CAESAR:
            tests['caesar_significance'] = self._test_caesar_significance(cipher_match)
        elif cipher_match.cipher_type == CipherType.SUBSTITUTION:
            tests['substitution_significance'] = self._test_substitution_significance(cipher_match)
        
        return tests
    
    def _perform_cross_validation(self, cipher_match: CipherMatch) -> Dict[str, float]:
        """
        Perform cross-validation tests
        """
        tests = {}
        
        # Split text and test consistency
        if len(cipher_match.original_text) >= 40:
            mid_point = len(cipher_match.original_text) // 2
            first_half = cipher_match.original_text[:mid_point]
            second_half = cipher_match.original_text[mid_point:]
            
            # Test if same method works on both halves
            first_matches = self.cipher_detector.detect_ciphers(first_half)
            second_matches = self.cipher_detector.detect_ciphers(second_half)
            
            # Check for consistent cipher type detection
            first_types = {match.cipher_type for match in first_matches}
            second_types = {match.cipher_type for match in second_matches}
            
            if cipher_match.cipher_type in first_types and cipher_match.cipher_type in second_types:
                tests['cross_validation_consistency'] = 1.0
            elif cipher_match.cipher_type in first_types or cipher_match.cipher_type in second_types:
                tests['cross_validation_consistency'] = 0.5
            else:
                tests['cross_validation_consistency'] = 0.0
        else:
            tests['cross_validation_consistency'] = 0.5  # Neutral for short texts
        
        return tests
    
    def _perform_linguistic_validation(self, cipher_match: CipherMatch) -> Dict[str, float]:
        """
        Perform linguistic validation tests
        """
        tests = {}
        
        # Syllable structure analysis
        tests['syllable_structure'] = self._analyze_syllable_structure(cipher_match.decoded_text)
        
        # Consonant-vowel distribution
        tests['consonant_vowel_distribution'] = self._analyze_consonant_vowel_distribution(cipher_match.decoded_text)
        
        # Common letter combinations
        tests['letter_combinations'] = self._analyze_letter_combinations(cipher_match.decoded_text)
        
        return tests
    
    def _perform_contextual_validation(self, cipher_match: CipherMatch) -> Dict[str, float]:
        """
        Perform contextual validation tests
        """
        tests = {}
        
        # Historical context appropriateness
        tests['historical_context'] = self._assess_historical_context(cipher_match)
        
        # Method complexity vs. text characteristics
        tests['complexity_appropriateness'] = self._assess_complexity_appropriateness(cipher_match)
        
        return tests
    
    def _calculate_validation_confidence(self, validation_tests: Dict[str, float]) -> float:
        """
        Calculate overall validation confidence score
        """
        if not validation_tests:
            return 0.0
        
        # Weight different test categories
        weights = {
            'confidence_threshold': 0.15,
            'text_length_adequacy': 0.05,
            'character_diversity': 0.05,
            'chi_squared_validation': 0.15,
            'index_of_coincidence_validation': 0.15,
            'entropy_validation': 0.10,
            'english_patterns': 0.20,
            'word_length_distribution': 0.05,
            'reasonable_word_ratio': 0.05,
            'random_text_comparison': 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for test_name, score in validation_tests.items():
            weight = weights.get(test_name, 0.02)  # Default small weight for unspecified tests
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _assess_reliability(self, confidence_score: float, validation_tests: Dict[str, float]) -> str:
        """
        Assess the reliability of the cipher detection
        """
        if confidence_score >= 0.8:
            return "High reliability - Strong evidence supports this cipher detection"
        elif confidence_score >= 0.6:
            return "Moderate reliability - Good evidence with some uncertainty"
        elif confidence_score >= 0.4:
            return "Low reliability - Weak evidence, requires further investigation"
        else:
            return "Very low reliability - Insufficient evidence, likely false positive"
    
    def _identify_validation_issues(self, validation_tests: Dict[str, float], 
                                  cipher_match: CipherMatch) -> List[str]:
        """
        Identify potential issues with the cipher detection
        """
        issues = []
        
        # Check for specific test failures
        if validation_tests.get('chi_squared_validation', 1.0) < 0.3:
            issues.append("Letter frequency distribution doesn't match English patterns")
        
        if validation_tests.get('english_patterns', 1.0) < 0.2:
            issues.append("Decoded text lacks common English word patterns")
        
        if validation_tests.get('text_length_adequacy', 1.0) < 0.5:
            issues.append("Text may be too short for reliable cipher detection")
        
        if validation_tests.get('character_diversity', 1.0) < 0.3:
            issues.append("Limited character diversity in decoded text")
        
        if cipher_match.confidence < 0.5:
            issues.append("Original detection confidence is relatively low")
        
        return issues
    
    def _generate_validation_recommendation(self, is_valid: bool, confidence_score: float,
                                         potential_issues: List[str], cipher_match: CipherMatch) -> str:
        """
        Generate validation recommendation
        """
        if is_valid and confidence_score >= 0.7:
            return f"Accept this {cipher_match.cipher_type.value} cipher detection with high confidence"
        elif is_valid:
            return f"Accept this {cipher_match.cipher_type.value} cipher detection with moderate confidence"
        elif confidence_score >= 0.4:
            return f"Consider this {cipher_match.cipher_type.value} cipher detection as possible but investigate further"
        else:
            return f"Reject this {cipher_match.cipher_type.value} cipher detection - insufficient evidence"
    
    def _analyze_text_for_recommendations(self, text: str) -> Dict[str, Any]:
        """
        Analyze text characteristics to inform cipher recommendations
        """
        analysis = {}
        
        # Basic text properties
        analysis['length'] = len(text)
        analysis['clean_length'] = len(re.sub(r'[^A-Za-z]', '', text))
        analysis['character_count'] = len(set(text.upper()))
        
        # Frequency analysis
        freq_analysis = self.cipher_detector._analyze_frequencies(text)
        analysis['frequency_analysis'] = freq_analysis
        
        # Pattern indicators
        analysis['repeated_patterns'] = self._find_repeated_patterns(text)
        analysis['character_spacing'] = self._analyze_character_spacing(text)
        analysis['formatting_variations'] = self._detect_formatting_variations(text)
        
        # Statistical properties
        analysis['entropy'] = freq_analysis.entropy
        analysis['index_of_coincidence'] = freq_analysis.index_of_coincidence
        analysis['chi_squared'] = freq_analysis.chi_squared
        
        return analysis
    
    def _determine_recommended_methods(self, text_analysis: Dict[str, Any], 
                                     context: Dict[str, Any] = None) -> List[str]:
        """
        Determine recommended cipher analysis methods based on text characteristics
        """
        methods = []
        
        # Always recommend basic methods for any text
        methods.extend(["frequency_analysis", "pattern_detection"])
        
        # Length-based recommendations
        if text_analysis['length'] >= 100:
            methods.extend(["caesar_cipher", "substitution_cipher", "skip_patterns"])
        
        if text_analysis['length'] >= 20:  # Lower threshold for atbash
            methods.append("atbash_cipher")
        
        # Entropy-based recommendations
        entropy = text_analysis.get('entropy', 0)
        if entropy < 3.5:
            methods.append("simple_substitution")
        elif entropy > 4.5:
            methods.extend(["polyalphabetic_cipher", "revolving_cipher"])
        
        # Index of Coincidence recommendations
        ic = text_analysis.get('index_of_coincidence', 0)
        if ic > 0.065:
            methods.append("monoalphabetic_substitution")
        elif ic < 0.045:
            methods.append("polyalphabetic_analysis")
        
        # Pattern-based recommendations
        if text_analysis.get('repeated_patterns'):
            methods.extend(["skip_cipher", "revolving_cipher"])
        
        if text_analysis.get('formatting_variations'):
            methods.append("biliteral_cipher")
        
        # Context-based recommendations
        if context:
            historical_period = context.get('historical_period')
            if historical_period == 'ancient':
                methods.extend(["atbash_cipher", "simple_substitution"])
                # Add context to text_analysis for later use
                text_analysis['historical_context'] = historical_period
            elif historical_period == 'renaissance':
                methods.extend(["biliteral_cipher", "polyalphabetic_cipher"])
                text_analysis['historical_context'] = historical_period
            elif historical_period == 'modern':
                methods.extend(["advanced_substitution", "mechanical_cipher"])
                text_analysis['historical_context'] = historical_period
        
        return list(set(methods))  # Remove duplicates
    
    def _determine_analysis_priority(self, text_analysis: Dict[str, Any], 
                                   recommended_methods: List[str]) -> List[CipherType]:
        """
        Determine priority order for cipher analysis
        """
        priority = []
        
        # High priority: Quick and reliable methods
        if "caesar_cipher" in recommended_methods:
            priority.append(CipherType.CAESAR)
        
        if "atbash_cipher" in recommended_methods:
            priority.append(CipherType.ATBASH)
        
        # Medium priority: More complex but common methods
        if any(method in recommended_methods for method in ["substitution_cipher", "simple_substitution", "monoalphabetic_substitution"]):
            priority.append(CipherType.SUBSTITUTION)
        
        if "skip_patterns" in recommended_methods or "skip_cipher" in recommended_methods:
            priority.append(CipherType.SKIP_PATTERN)
        
        # Lower priority: Complex or specialized methods
        if "biliteral_cipher" in recommended_methods:
            priority.append(CipherType.BILITERAL)
        
        if "revolving_cipher" in recommended_methods or "polyalphabetic_cipher" in recommended_methods:
            priority.append(CipherType.REVOLVING)
        
        return priority
    
    def _suggest_analysis_parameters(self, text_analysis: Dict[str, Any], 
                                   priority: List[CipherType]) -> Dict[str, Any]:
        """
        Suggest optimal parameters for cipher analysis
        """
        parameters = {}
        
        # General parameters
        parameters['min_confidence'] = 0.3 if text_analysis['length'] < 100 else 0.4
        parameters['max_iterations'] = min(1000, text_analysis['length'] * 2)
        
        # Method-specific parameters
        if CipherType.SKIP_PATTERN in priority:
            max_skip = min(20, text_analysis['length'] // 10)
            parameters['skip_pattern'] = {
                'max_skip_distance': max_skip,
                'min_extracted_length': 10
            }
        
        if CipherType.SUBSTITUTION in priority:
            parameters['substitution'] = {
                'use_frequency_analysis': True,
                'try_common_keys': True,
                'max_key_attempts': 100
            }
        
        if CipherType.REVOLVING in priority:
            parameters['revolving'] = {
                'max_pattern_length': min(12, text_analysis['length'] // 20),
                'test_mathematical_patterns': True
            }
        
        return parameters
    
    def _estimate_success_rate(self, text_analysis: Dict[str, Any], 
                             recommended_methods: List[str]) -> float:
        """
        Estimate the expected success rate for cipher detection
        """
        base_rate = 0.3  # Base success rate
        
        # Adjust based on text length
        length_factor = min(1.0, text_analysis['length'] / 200)
        
        # Adjust based on character diversity
        diversity_factor = min(1.0, text_analysis['character_count'] / 15)
        
        # Adjust based on entropy (moderate entropy is better)
        entropy = text_analysis.get('entropy', 4.0)
        entropy_factor = 1.0 - abs(entropy - 4.0) / 4.0
        
        # Adjust based on number of methods
        method_factor = min(1.0, len(recommended_methods) / 8)
        
        # Calculate weighted success rate
        success_rate = base_rate + (
            0.3 * length_factor +
            0.2 * diversity_factor +
            0.3 * entropy_factor +
            0.2 * method_factor
        ) * 0.5
        
        return min(0.95, max(0.05, success_rate))
    
    def _generate_recommendation_reasoning(self, text_analysis: Dict[str, Any],
                                         recommended_methods: List[str],
                                         priority: List[CipherType]) -> str:
        """
        Generate reasoning for the recommendations
        """
        reasoning_parts = []
        
        # Text characteristics
        length = text_analysis['length']
        entropy = text_analysis.get('entropy', 0)
        ic = text_analysis.get('index_of_coincidence', 0)
        
        reasoning_parts.append(f"Text length of {length} characters")
        
        if entropy < 3.5:
            reasoning_parts.append("low entropy suggests simple substitution methods")
        elif entropy > 4.5:
            reasoning_parts.append("high entropy indicates complex or polyalphabetic ciphers")
        else:
            reasoning_parts.append("moderate entropy allows for various cipher types")
        
        if ic > 0.065:
            reasoning_parts.append("high index of coincidence suggests monoalphabetic substitution")
        elif ic < 0.045:
            reasoning_parts.append("low index of coincidence indicates polyalphabetic methods")
        
        # Historical context
        if text_analysis.get('historical_context'):
            reasoning_parts.append(f"historical context ({text_analysis['historical_context']}) influences method selection")
        
        # Method-specific reasoning
        if CipherType.CAESAR in priority:
            reasoning_parts.append("Caesar cipher recommended due to its simplicity and high detection rate")
        
        if CipherType.SUBSTITUTION in priority:
            reasoning_parts.append("substitution cipher analysis recommended based on frequency characteristics")
        
        return "Analysis based on: " + ", ".join(reasoning_parts) + "."
    
    def _suggest_alternative_approaches(self, text_analysis: Dict[str, Any],
                                      recommended_methods: List[str]) -> List[str]:
        """
        Suggest alternative analysis approaches
        """
        alternatives = []
        
        # If standard methods are recommended, suggest advanced alternatives
        if "caesar_cipher" in recommended_methods:
            alternatives.append("Try Vigenère cipher analysis with short keys")
        
        if "substitution_cipher" in recommended_methods:
            alternatives.append("Consider homophonic substitution if simple substitution fails")
        
        # General alternatives
        alternatives.extend([
            "Manual pattern analysis for unusual cipher types",
            "Statistical analysis of character positions",
            "Cross-reference with known historical cipher methods",
            "Consider steganographic methods if no cipher is detected"
        ])
        
        # Text-specific alternatives
        if text_analysis['length'] > 1000:
            alternatives.append("Segment text and analyze sections independently")
        
        if text_analysis.get('formatting_variations'):
            alternatives.append("Investigate typography-based steganography")
        
        return alternatives
    
    def _create_detailed_analysis(self, cipher_match: CipherMatch) -> Dict[str, Any]:
        """
        Create detailed analysis of a cipher match
        """
        analysis = {
            'cipher_type': cipher_match.cipher_type.value,
            'confidence_breakdown': {
                'overall_confidence': cipher_match.confidence,
                'method_details': cipher_match.method_details,
                'statistical_analysis': cipher_match.statistical_analysis
            },
            'text_analysis': {
                'original_length': len(cipher_match.original_text),
                'decoded_length': len(cipher_match.decoded_text),
                'position_range': f"{cipher_match.position_start}-{cipher_match.position_end}",
                'key_or_pattern': cipher_match.key_or_pattern
            },
            'quality_indicators': self._assess_quality_indicators(cipher_match),
            'potential_improvements': self._suggest_improvements(cipher_match)
        }
        
        return analysis
    
    def _create_visual_examples(self, cipher_match: CipherMatch) -> Dict[str, str]:
        """
        Create visual examples of the cipher transformation
        """
        examples = {}
        
        # Show first 50 characters of transformation
        original_sample = cipher_match.original_text[:50]
        decoded_sample = cipher_match.decoded_text[:50]
        
        examples['transformation_sample'] = f"'{original_sample}' → '{decoded_sample}'"
        
        # Method-specific examples
        if cipher_match.cipher_type == CipherType.CAESAR:
            shift = cipher_match.method_details.get('shift_amount', 0)
            examples['method_example'] = f"Caesar shift of {shift}: A→{chr((ord('A') - ord('A') - shift) % 26 + ord('A'))}, B→{chr((ord('B') - ord('A') - shift) % 26 + ord('A'))}, etc."
        
        elif cipher_match.cipher_type == CipherType.ATBASH:
            examples['method_example'] = "Atbash: A→Z, B→Y, C→X, D→W, E→V, etc."
        
        elif cipher_match.cipher_type == CipherType.SKIP_PATTERN:
            skip = cipher_match.method_details.get('skip_amount', 1)
            start = cipher_match.method_details.get('start_position', 0)
            examples['method_example'] = f"Skip pattern: every {skip} characters starting at position {start}"
        
        return examples
    
    def _suggest_next_steps(self, cipher_match: CipherMatch, 
                          validation_result: ValidationResult) -> List[str]:
        """
        Suggest next steps based on cipher detection and validation results
        """
        steps = []
        
        if validation_result.is_valid:
            steps.append("Proceed with detailed analysis of the decoded text")
            steps.append("Look for additional cipher layers or steganographic content")
            steps.append("Cross-reference findings with historical context")
        else:
            steps.append("Re-examine the original text for preprocessing errors")
            steps.append("Try alternative cipher detection methods")
            steps.append("Consider that this may be a false positive")
        
        # Method-specific next steps
        if cipher_match.cipher_type == CipherType.CAESAR:
            steps.append("Verify the shift amount by testing adjacent shift values")
        
        elif cipher_match.cipher_type == CipherType.SUBSTITUTION:
            steps.append("Refine the substitution key using additional frequency analysis")
        
        elif cipher_match.cipher_type == CipherType.SKIP_PATTERN:
            steps.append("Test variations in skip distance and starting position")
        
        return steps
    
    def _create_summary_analysis(self, cipher_matches: List[CipherMatch]) -> Dict[str, Any]:
        """
        Create summary analysis of all cipher detection results
        """
        if not cipher_matches:
            return {
                'total_matches': 0,
                'recommendation': "No cipher patterns detected. Consider alternative analysis methods."
            }
        
        summary = {
            'total_matches': len(cipher_matches),
            'cipher_types_found': list(set(match.cipher_type.value for match in cipher_matches)),
            'confidence_range': {
                'min': min(match.confidence for match in cipher_matches),
                'max': max(match.confidence for match in cipher_matches),
                'average': sum(match.confidence for match in cipher_matches) / len(cipher_matches)
            },
            'best_match': max(cipher_matches, key=lambda x: x.confidence),
            'recommendations': []
        }
        
        # Generate recommendations based on results
        if summary['confidence_range']['max'] > 0.7:
            summary['recommendations'].append("High-confidence cipher detected - proceed with detailed analysis")
        elif summary['confidence_range']['max'] > 0.4:
            summary['recommendations'].append("Moderate-confidence detection - validate with additional methods")
        else:
            summary['recommendations'].append("Low-confidence results - consider alternative approaches")
        
        if len(set(match.cipher_type for match in cipher_matches)) > 1:
            summary['recommendations'].append("Multiple cipher types detected - investigate layered encryption")
        
        return summary
    
    # Helper methods for validation
    def _test_against_random_text(self, text: str) -> float:
        """Test decoded text against random text characteristics"""
        # Simple test: random text should have more uniform character distribution
        clean_text = re.sub(r'[^A-Za-z]', '', text.upper())
        if not clean_text:
            return 0.0
        
        char_counts = Counter(clean_text)
        # Calculate coefficient of variation
        counts = list(char_counts.values())
        if len(counts) < 2:
            return 0.0
        
        mean_count = statistics.mean(counts)
        std_count = statistics.stdev(counts)
        cv = std_count / mean_count if mean_count > 0 else 0
        
        # English text typically has CV around 1.5-2.5, random text closer to 0.3
        return min(1.0, max(0.0, (cv - 0.3) / 2.0))
    
    def _test_caesar_significance(self, cipher_match: CipherMatch) -> float:
        """Test statistical significance of Caesar cipher detection"""
        # Check if this shift produces significantly better results than others
        original_confidence = cipher_match.confidence
        
        # Test a few other shifts for comparison
        other_confidences = []
        for test_shift in [1, 5, 10, 15, 20]:  # Sample of other shifts
            if test_shift != cipher_match.method_details.get('shift_amount', 0):
                test_decoded = self.cipher_detector._apply_caesar_shift(
                    cipher_match.original_text, test_shift
                )
                test_confidence = self.cipher_detector._check_english_patterns(test_decoded)
                other_confidences.append(test_confidence)
        
        if not other_confidences:
            return 0.5
        
        avg_other = statistics.mean(other_confidences)
        return min(1.0, max(0.0, (original_confidence - avg_other) / 0.5))
    
    def _test_substitution_significance(self, cipher_match: CipherMatch) -> float:
        """Test statistical significance of substitution cipher detection"""
        # Simple significance test based on frequency matching
        freq_confidence = cipher_match.method_details.get('frequency_confidence', 0)
        pattern_confidence = cipher_match.method_details.get('pattern_confidence', 0)
        
        # Both should be reasonably high for a valid substitution cipher
        return (freq_confidence + pattern_confidence) / 2
    
    def _analyze_syllable_structure(self, text: str) -> float:
        """Analyze syllable structure of decoded text"""
        # Simple vowel-consonant pattern analysis
        clean_text = re.sub(r'[^A-Za-z]', '', text.upper())
        if not clean_text:
            return 0.0
        
        vowels = set('AEIOU')
        consonants = set('BCDFGHJKLMNPQRSTVWXYZ')
        
        vowel_count = sum(1 for c in clean_text if c in vowels)
        consonant_count = sum(1 for c in clean_text if c in consonants)
        
        if vowel_count + consonant_count == 0:
            return 0.0
        
        vowel_ratio = vowel_count / (vowel_count + consonant_count)
        # English typically has 35-45% vowels
        return max(0.0, 1.0 - abs(vowel_ratio - 0.4) * 2.5)
    
    def _analyze_consonant_vowel_distribution(self, text: str) -> float:
        """Analyze consonant-vowel distribution patterns"""
        # Look for reasonable alternation patterns
        clean_text = re.sub(r'[^A-Za-z]', '', text.upper())
        if len(clean_text) < 4:
            return 0.5
        
        vowels = set('AEIOU')
        
        # Count consonant/vowel clusters
        clusters = []
        current_type = None
        current_length = 0
        
        for char in clean_text:
            char_type = 'V' if char in vowels else 'C'
            if char_type == current_type:
                current_length += 1
            else:
                if current_length > 0:
                    clusters.append(current_length)
                current_type = char_type
                current_length = 1
        
        if current_length > 0:
            clusters.append(current_length)
        
        if not clusters:
            return 0.0
        
        # Reasonable cluster lengths (not too many long consonant/vowel runs)
        reasonable_clusters = sum(1 for length in clusters if 1 <= length <= 4)
        return reasonable_clusters / len(clusters)
    
    def _analyze_letter_combinations(self, text: str) -> float:
        """Analyze common letter combinations"""
        clean_text = re.sub(r'[^A-Za-z]', '', text.upper())
        if len(clean_text) < 3:
            return 0.0
        
        # Common English letter combinations
        common_combinations = {
            'TH', 'HE', 'IN', 'ER', 'AN', 'RE', 'ED', 'ND', 'ON', 'EN',
            'AT', 'OU', 'IT', 'IS', 'OR', 'TI', 'HI', 'ST', 'AR', 'NT'
        }
        
        bigrams = [clean_text[i:i+2] for i in range(len(clean_text)-1)]
        common_found = sum(1 for bigram in bigrams if bigram in common_combinations)
        
        return common_found / len(bigrams) if bigrams else 0.0
    
    def _assess_historical_context(self, cipher_match: CipherMatch) -> float:
        """Assess if cipher method fits historical context"""
        # This would ideally use external context, but for now return neutral
        return 0.5
    
    def _assess_complexity_appropriateness(self, cipher_match: CipherMatch) -> float:
        """Assess if cipher complexity matches text characteristics"""
        text_length = len(cipher_match.original_text)
        
        # Simple heuristic: longer texts can support more complex ciphers
        if cipher_match.cipher_type in [CipherType.CAESAR, CipherType.ATBASH]:
            return 1.0 if text_length >= 20 else text_length / 20
        elif cipher_match.cipher_type == CipherType.SUBSTITUTION:
            return 1.0 if text_length >= 100 else text_length / 100
        elif cipher_match.cipher_type in [CipherType.REVOLVING, CipherType.BILITERAL]:
            return 1.0 if text_length >= 200 else text_length / 200
        else:
            return 0.5
    
    def _assess_quality_indicators(self, cipher_match: CipherMatch) -> Dict[str, str]:
        """Assess quality indicators for the cipher match"""
        indicators = {}
        
        if cipher_match.confidence > 0.7:
            indicators['confidence'] = "High"
        elif cipher_match.confidence > 0.4:
            indicators['confidence'] = "Moderate"
        else:
            indicators['confidence'] = "Low"
        
        # Check statistical measures
        chi_squared = cipher_match.statistical_analysis.get('chi_squared', 100)
        if chi_squared < 20:
            indicators['frequency_match'] = "Excellent"
        elif chi_squared < 50:
            indicators['frequency_match'] = "Good"
        else:
            indicators['frequency_match'] = "Poor"
        
        return indicators
    
    def _suggest_improvements(self, cipher_match: CipherMatch) -> List[str]:
        """Suggest improvements for cipher detection"""
        improvements = []
        
        if cipher_match.confidence < 0.5:
            improvements.append("Try adjusting cipher parameters")
            improvements.append("Consider preprocessing the text differently")
        
        if cipher_match.cipher_type == CipherType.SUBSTITUTION:
            improvements.append("Refine substitution key using additional context")
        
        improvements.append("Validate results with independent methods")
        
        return improvements
    
    def _find_repeated_patterns(self, text: str) -> List[str]:
        """Find repeated patterns in text"""
        patterns = []
        clean_text = re.sub(r'[^A-Za-z]', '', text.upper())
        
        # Look for patterns of length 2-6
        for length in range(2, 7):
            pattern_counts = Counter()
            for i in range(len(clean_text) - length + 1):
                pattern = clean_text[i:i+length]
                pattern_counts[pattern] += 1
            
            # Find patterns that repeat at least 3 times
            for pattern, count in pattern_counts.items():
                if count >= 3:
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_character_spacing(self, text: str) -> Dict[str, Any]:
        """Analyze character spacing patterns"""
        # This would analyze spacing in the original document
        # For now, return basic analysis
        return {
            'uniform_spacing': True,
            'unusual_gaps': False,
            'spacing_variations': []
        }
    
    def _detect_formatting_variations(self, text: str) -> bool:
        """Detect formatting variations that might indicate biliteral cipher"""
        # This would detect italic/bold/font variations
        # For now, return False as we don't have formatting info
        return False    

    def _analyze_cipher_characteristics(self, cipher_match: CipherMatch) -> Dict[str, Any]:
        """Analyze characteristics of the cipher match"""
        return {
            'cipher_type': cipher_match.cipher_type.value,
            'key_pattern': cipher_match.key_or_pattern,
            'text_length': len(cipher_match.decoded_text),
            'confidence': cipher_match.confidence,
            'method_details': cipher_match.method_details
        }
    
    def _analyze_text_properties(self, text: str) -> Dict[str, Any]:
        """Analyze properties of the decoded text"""
        clean_text = re.sub(r'[^A-Za-z]', '', text)
        return {
            'length': len(text),
            'clean_length': len(clean_text),
            'word_count': len(text.split()) if text else 0,
            'character_diversity': len(set(clean_text.upper())) if clean_text else 0
        }
    
    def _perform_method_specific_analysis(self, cipher_match: CipherMatch) -> Dict[str, Any]:
        """Perform method-specific analysis"""
        analysis = {}
        
        if cipher_match.cipher_type == CipherType.CAESAR:
            shift = cipher_match.method_details.get('shift_amount', 0)
            analysis['shift_analysis'] = {
                'shift_amount': shift,
                'is_common_shift': shift in [1, 3, 13, 25],  # Common Caesar shifts
                'reverse_shift': 26 - shift if shift > 0 else 0
            }
        
        elif cipher_match.cipher_type == CipherType.SUBSTITUTION:
            mapping = cipher_match.method_details.get('substitution_mapping', {})
            analysis['substitution_analysis'] = {
                'mapping_size': len(mapping),
                'is_complete_alphabet': len(mapping) == 26,
                'common_mappings': sum(1 for k, v in mapping.items() if k == v)
            }
        
        elif cipher_match.cipher_type == CipherType.SKIP_PATTERN:
            skip = cipher_match.method_details.get('skip_amount', 1)
            analysis['skip_analysis'] = {
                'skip_distance': skip,
                'is_prime_skip': self._is_prime(skip),
                'extracted_length': len(cipher_match.decoded_text)
            }
        
        return analysis
    
    def _is_prime(self, n: int) -> bool:
        """Check if a number is prime"""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True