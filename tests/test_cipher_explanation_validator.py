"""
Tests for Cipher Explanation and Validation System
"""
import pytest
from unittest.mock import Mock, patch
import string

from app.services.cipher_explanation_validator import (
    CipherExplanationValidator, CipherExplanation, ValidationResult, 
    CipherRecommendation, ValidationLevel
)
from app.services.cipher_detector import CipherMatch, CipherType, FrequencyAnalysis


class TestCipherExplanationValidator:
    """Test cases for CipherExplanationValidator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = CipherExplanationValidator()
        
        # Create sample cipher match for testing
        self.sample_caesar_match = CipherMatch(
            cipher_type=CipherType.CAESAR,
            confidence=0.75,
            key_or_pattern="shift_3",
            decoded_text="HELLO WORLD THIS IS A TEST MESSAGE",
            original_text="KHOOR ZRUOG WKLV LV D WHVW PHVVDJH",
            position_start=0,
            position_end=35,
            method_details={
                'shift_amount': 3,
                'frequency_confidence': 0.7,
                'pattern_confidence': 0.8
            },
            statistical_analysis={
                'chi_squared': 25.5,
                'index_of_coincidence': 0.067,
                'entropy': 4.1
            }
        )
        
        self.sample_substitution_match = CipherMatch(
            cipher_type=CipherType.SUBSTITUTION,
            confidence=0.65,
            key_or_pattern="{'A': 'X', 'B': 'Y', 'C': 'Z'}",
            decoded_text="THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
            original_text="WKH TXLFN EURZQ IRA MXPSV RYHU WKH ODCB GRJ",
            position_start=0,
            position_end=43,
            method_details={
                'substitution_mapping': {'A': 'X', 'B': 'Y', 'C': 'Z'},
                'frequency_confidence': 0.6,
                'pattern_confidence': 0.7
            },
            statistical_analysis={
                'chi_squared': 35.2,
                'index_of_coincidence': 0.065,
                'entropy': 4.3
            }
        )
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        assert self.validator.min_validation_confidence == 0.5
        assert self.validator.statistical_significance_threshold == 0.05
        assert len(self.validator.cipher_explanations) >= 6  # Should have explanations for main cipher types
        assert 'chi_squared_max' in self.validator.validation_thresholds
    
    def test_explain_cipher_method_caesar(self):
        """Test cipher method explanation for Caesar cipher"""
        explanation = self.validator.explain_cipher_method(CipherType.CAESAR)
        
        assert isinstance(explanation, CipherExplanation)
        assert explanation.cipher_type == CipherType.CAESAR
        assert explanation.method_name == "Caesar Cipher"
        assert "Julius Caesar" in explanation.historical_context
        assert "shifted" in explanation.how_it_works.lower()
        assert len(explanation.key_characteristics) > 0
        assert explanation.difficulty_level == "Beginner"
        assert len(explanation.common_uses) > 0
    
    def test_explain_cipher_method_atbash(self):
        """Test cipher method explanation for Atbash cipher"""
        explanation = self.validator.explain_cipher_method(CipherType.ATBASH)
        
        assert explanation.cipher_type == CipherType.ATBASH
        assert explanation.method_name == "Atbash Cipher"
        assert "hebrew" in explanation.historical_context.lower()
        assert "reversed" in explanation.how_it_works.lower()
        assert "self-reciprocal" in explanation.key_characteristics[1].lower()
    
    def test_explain_cipher_method_substitution(self):
        """Test cipher method explanation for substitution cipher"""
        explanation = self.validator.explain_cipher_method(CipherType.SUBSTITUTION)
        
        assert explanation.cipher_type == CipherType.SUBSTITUTION
        assert explanation.method_name == "Simple Substitution Cipher"
        assert "mary queen of scots" in explanation.historical_context.lower()
        assert "frequency" in explanation.key_characteristics[2].lower()
    
    def test_explain_cipher_method_biliteral(self):
        """Test cipher method explanation for biliteral cipher"""
        explanation = self.validator.explain_cipher_method(CipherType.BILITERAL)
        
        assert explanation.cipher_type == CipherType.BILITERAL
        assert explanation.method_name == "Biliteral Cipher (Bacon's Cipher)"
        assert "francis bacon" in explanation.historical_context.lower()
        assert "steganographic" in explanation.description.lower()
        assert "5-bit" in explanation.how_it_works
    
    def test_explain_cipher_method_skip_pattern(self):
        """Test cipher method explanation for skip pattern cipher"""
        explanation = self.validator.explain_cipher_method(CipherType.SKIP_PATTERN)
        
        assert explanation.cipher_type == CipherType.SKIP_PATTERN
        assert explanation.method_name == "Skip Pattern Cipher"
        assert "nth character" in explanation.description.lower()
        assert "skip distance" in explanation.key_characteristics[0].lower()
    
    def test_explain_cipher_method_revolving(self):
        """Test cipher method explanation for revolving cipher"""
        explanation = self.validator.explain_cipher_method(CipherType.REVOLVING)
        
        assert explanation.cipher_type == CipherType.REVOLVING
        assert explanation.method_name == "Revolving Cipher (Polyalphabetic)"
        assert "multiple substitution" in explanation.description.lower()
        assert "polyalphabetic" in explanation.historical_context.lower()
    
    def test_explain_cipher_method_unknown(self):
        """Test cipher method explanation for unknown cipher type"""
        # Create a mock cipher type that doesn't exist in explanations
        unknown_type = Mock()
        unknown_type.value = "unknown_cipher"
        
        explanation = self.validator.explain_cipher_method(unknown_type)
        
        assert explanation.cipher_type == unknown_type
        assert "Unknown Cipher Cipher" in explanation.method_name
        assert explanation.historical_context == "Historical context not available."
    
    def test_validate_cipher_detection_basic(self):
        """Test basic cipher detection validation"""
        validation = self.validator.validate_cipher_detection(
            self.sample_caesar_match, ValidationLevel.BASIC
        )
        
        assert isinstance(validation, ValidationResult)
        assert validation.confidence_score > 0
        assert 'confidence_threshold' in validation.statistical_tests
        assert 'text_length_adequacy' in validation.statistical_tests
        assert 'character_diversity' in validation.statistical_tests
        assert validation.validation_details['validation_level'] == 'basic'
    
    def test_validate_cipher_detection_standard(self):
        """Test standard cipher detection validation"""
        validation = self.validator.validate_cipher_detection(
            self.sample_caesar_match, ValidationLevel.STANDARD
        )
        
        assert validation.confidence_score > 0
        assert 'chi_squared_validation' in validation.statistical_tests
        assert 'index_of_coincidence_validation' in validation.statistical_tests
        assert 'entropy_validation' in validation.statistical_tests
        assert 'english_patterns' in validation.statistical_tests
        assert validation.validation_details['validation_level'] == 'standard'
    
    def test_validate_cipher_detection_rigorous(self):
        """Test rigorous cipher detection validation"""
        validation = self.validator.validate_cipher_detection(
            self.sample_caesar_match, ValidationLevel.RIGOROUS
        )
        
        assert validation.confidence_score > 0
        assert 'random_text_comparison' in validation.statistical_tests
        assert 'caesar_significance' in validation.statistical_tests
        assert 'cross_validation_consistency' in validation.statistical_tests
        assert validation.validation_details['validation_level'] == 'rigorous'
    
    def test_validate_cipher_detection_comprehensive(self):
        """Test comprehensive cipher detection validation"""
        validation = self.validator.validate_cipher_detection(
            self.sample_caesar_match, ValidationLevel.COMPREHENSIVE
        )
        
        assert validation.confidence_score > 0
        assert 'syllable_structure' in validation.statistical_tests
        assert 'consonant_vowel_distribution' in validation.statistical_tests
        assert 'letter_combinations' in validation.statistical_tests
        assert 'historical_context' in validation.statistical_tests
        assert validation.validation_details['validation_level'] == 'comprehensive'
    
    def test_validate_high_confidence_match(self):
        """Test validation of high confidence cipher match"""
        high_confidence_match = self.sample_caesar_match
        high_confidence_match.confidence = 0.9
        
        validation = self.validator.validate_cipher_detection(high_confidence_match)
        
        assert validation.is_valid
        assert validation.confidence_score > 0.6
        assert "accept" in validation.recommendation.lower()
        assert validation.is_valid  # Should be valid with high confidence
    
    def test_validate_low_confidence_match(self):
        """Test validation of low confidence cipher match"""
        low_confidence_match = self.sample_caesar_match
        low_confidence_match.confidence = 0.2
        low_confidence_match.statistical_analysis['chi_squared'] = 100.0
        
        validation = self.validator.validate_cipher_detection(low_confidence_match)
        
        assert validation.confidence_score < 0.6  # Should have lower confidence due to low input confidence
        assert len(validation.potential_issues) > 0
        assert len(validation.potential_issues) > 0  # Should have issues identified
    
    def test_generate_cipher_recommendations_short_text(self):
        """Test cipher recommendations for short text"""
        short_text = "HELLO WORLD"
        
        recommendations = self.validator.generate_cipher_recommendations(short_text)
        
        assert isinstance(recommendations, CipherRecommendation)
        assert len(recommendations.recommended_methods) > 0
        assert "frequency_analysis" in recommendations.recommended_methods
        assert len(recommendations.analysis_priority) > 0
        assert 0 <= recommendations.expected_success_rate <= 1
        assert len(recommendations.reasoning) > 0
        assert len(recommendations.alternative_approaches) > 0
    
    def test_generate_cipher_recommendations_long_text(self):
        """Test cipher recommendations for long text"""
        long_text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG " * 10  # 430+ characters
        
        recommendations = self.validator.generate_cipher_recommendations(long_text)
        
        assert "caesar_cipher" in recommendations.recommended_methods
        assert "substitution_cipher" in recommendations.recommended_methods
        assert "skip_patterns" in recommendations.recommended_methods
        assert CipherType.CAESAR in recommendations.analysis_priority
        assert recommendations.expected_success_rate > 0.3  # Should be higher for longer text
    
    def test_generate_cipher_recommendations_with_context(self):
        """Test cipher recommendations with historical context"""
        text = "ANCIENT TEXT FOR ANALYSIS"
        context = {'historical_period': 'ancient'}
        
        recommendations = self.validator.generate_cipher_recommendations(text, context)
        
        assert "atbash_cipher" in recommendations.recommended_methods
        assert "simple_substitution" in recommendations.recommended_methods
        assert "ancient" in recommendations.reasoning.lower() or "historical" in recommendations.reasoning.lower()
    
    def test_explain_detection_results_single_match(self):
        """Test explanation of single cipher detection result"""
        matches = [self.sample_caesar_match]
        
        explanations = self.validator.explain_detection_results(matches)
        
        assert len(explanations) == 2  # One match + summary
        assert 'summary' in explanations
        
        match_key = f"{CipherType.CAESAR.value}_{self.sample_caesar_match.confidence:.2f}"
        assert match_key in explanations
        
        match_explanation = explanations[match_key]
        assert 'cipher_match' in match_explanation
        assert 'explanation' in match_explanation
        assert 'validation' in match_explanation
        assert 'detailed_analysis' in match_explanation
        assert 'next_steps' in match_explanation
    
    def test_explain_detection_results_multiple_matches(self):
        """Test explanation of multiple cipher detection results"""
        matches = [self.sample_caesar_match, self.sample_substitution_match]
        
        explanations = self.validator.explain_detection_results(matches)
        
        assert len(explanations) == 3  # Two matches + summary
        assert 'summary' in explanations
        
        summary = explanations['summary']
        assert summary['total_matches'] == 2
        assert len(summary['cipher_types_found']) == 2
        assert 'confidence_range' in summary
        assert 'best_match' in summary
        assert len(summary['recommendations']) > 0
    
    def test_explain_detection_results_no_matches(self):
        """Test explanation when no cipher matches are found"""
        matches = []
        
        explanations = self.validator.explain_detection_results(matches)
        
        assert 'summary' in explanations
        summary = explanations['summary']
        assert summary['total_matches'] == 0
        assert "no cipher patterns" in summary['recommendation'].lower()
    
    def test_basic_validation_tests(self):
        """Test basic validation test methods"""
        tests = self.validator._perform_basic_validation(self.sample_caesar_match)
        
        assert 'confidence_threshold' in tests
        assert 'text_length_adequacy' in tests
        assert 'character_diversity' in tests
        assert all(0 <= score <= 1 for score in tests.values())
    
    def test_frequency_validation_tests(self):
        """Test frequency validation test methods"""
        tests = self.validator._perform_frequency_validation(self.sample_caesar_match)
        
        assert 'chi_squared_validation' in tests
        assert 'index_of_coincidence_validation' in tests
        assert 'entropy_validation' in tests
        assert all(0 <= score <= 1 for score in tests.values())
    
    def test_pattern_validation_tests(self):
        """Test pattern validation test methods"""
        tests = self.validator._perform_pattern_validation(self.sample_caesar_match)
        
        assert 'english_patterns' in tests
        assert 'word_length_distribution' in tests
        assert 'reasonable_word_ratio' in tests
        assert all(0 <= score <= 1 for score in tests.values())
    
    def test_statistical_significance_tests(self):
        """Test statistical significance test methods"""
        tests = self.validator._perform_statistical_significance_tests(self.sample_caesar_match)
        
        assert 'random_text_comparison' in tests
        assert 'caesar_significance' in tests
        assert all(0 <= score <= 1 for score in tests.values())
    
    def test_cross_validation_tests(self):
        """Test cross-validation test methods"""
        # Create a longer text for cross-validation
        long_match = self.sample_caesar_match
        long_match.original_text = "KHOOR ZRUOG " * 10  # Make it long enough for splitting
        
        tests = self.validator._perform_cross_validation(long_match)
        
        assert 'cross_validation_consistency' in tests
        assert 0 <= tests['cross_validation_consistency'] <= 1
    
    def test_linguistic_validation_tests(self):
        """Test linguistic validation test methods"""
        tests = self.validator._perform_linguistic_validation(self.sample_caesar_match)
        
        assert 'syllable_structure' in tests
        assert 'consonant_vowel_distribution' in tests
        assert 'letter_combinations' in tests
        assert all(0 <= score <= 1 for score in tests.values())
    
    def test_contextual_validation_tests(self):
        """Test contextual validation test methods"""
        tests = self.validator._perform_contextual_validation(self.sample_caesar_match)
        
        assert 'historical_context' in tests
        assert 'complexity_appropriateness' in tests
        assert all(0 <= score <= 1 for score in tests.values())
    
    def test_calculate_validation_confidence(self):
        """Test validation confidence calculation"""
        test_scores = {
            'confidence_threshold': 1.0,
            'chi_squared_validation': 0.8,
            'english_patterns': 0.7,
            'text_length_adequacy': 1.0
        }
        
        confidence = self.validator._calculate_validation_confidence(test_scores)
        
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be reasonably high with good scores
    
    def test_assess_reliability_high_confidence(self):
        """Test reliability assessment for high confidence"""
        reliability = self.validator._assess_reliability(0.85, {})
        assert "high reliability" in reliability.lower()
    
    def test_assess_reliability_moderate_confidence(self):
        """Test reliability assessment for moderate confidence"""
        reliability = self.validator._assess_reliability(0.65, {})
        assert "moderate reliability" in reliability.lower()
    
    def test_assess_reliability_low_confidence(self):
        """Test reliability assessment for low confidence"""
        reliability = self.validator._assess_reliability(0.45, {})
        assert "low reliability" in reliability.lower()
    
    def test_assess_reliability_very_low_confidence(self):
        """Test reliability assessment for very low confidence"""
        reliability = self.validator._assess_reliability(0.25, {})
        assert "very low reliability" in reliability.lower()
    
    def test_identify_validation_issues(self):
        """Test identification of validation issues"""
        poor_tests = {
            'chi_squared_validation': 0.2,
            'english_patterns': 0.1,
            'text_length_adequacy': 0.4,
            'character_diversity': 0.2
        }
        
        issues = self.validator._identify_validation_issues(poor_tests, self.sample_caesar_match)
        
        assert len(issues) > 0
        assert any("frequency distribution" in issue.lower() for issue in issues)
        assert any("english word patterns" in issue.lower() for issue in issues)
    
    def test_generate_validation_recommendation_accept_high(self):
        """Test validation recommendation for high confidence valid result"""
        recommendation = self.validator._generate_validation_recommendation(
            True, 0.8, [], self.sample_caesar_match
        )
        assert "accept" in recommendation.lower()
        assert "high confidence" in recommendation.lower()
    
    def test_generate_validation_recommendation_accept_moderate(self):
        """Test validation recommendation for moderate confidence valid result"""
        recommendation = self.validator._generate_validation_recommendation(
            True, 0.6, [], self.sample_caesar_match
        )
        assert "accept" in recommendation.lower()
        assert "moderate confidence" in recommendation.lower()
    
    def test_generate_validation_recommendation_investigate(self):
        """Test validation recommendation for borderline result"""
        recommendation = self.validator._generate_validation_recommendation(
            False, 0.45, ["some issue"], self.sample_caesar_match
        )
        assert "consider" in recommendation.lower() or "investigate" in recommendation.lower()
    
    def test_generate_validation_recommendation_reject(self):
        """Test validation recommendation for low confidence result"""
        recommendation = self.validator._generate_validation_recommendation(
            False, 0.2, ["multiple issues"], self.sample_caesar_match
        )
        assert "reject" in recommendation.lower()
    
    def test_analyze_text_for_recommendations(self):
        """Test text analysis for recommendations"""
        text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
        
        analysis = self.validator._analyze_text_for_recommendations(text)
        
        assert 'length' in analysis
        assert 'clean_length' in analysis
        assert 'character_count' in analysis
        assert 'frequency_analysis' in analysis
        assert 'entropy' in analysis
        assert 'index_of_coincidence' in analysis
        assert analysis['length'] == len(text)
        assert analysis['clean_length'] > 0
    
    def test_determine_recommended_methods_short_text(self):
        """Test method recommendations for short text"""
        analysis = {'length': 30, 'entropy': 4.0, 'index_of_coincidence': 0.06}
        
        methods = self.validator._determine_recommended_methods(analysis)
        
        assert "frequency_analysis" in methods
        assert "pattern_detection" in methods
        assert "atbash_cipher" in methods
    
    def test_determine_recommended_methods_long_text(self):
        """Test method recommendations for long text"""
        analysis = {'length': 200, 'entropy': 4.2, 'index_of_coincidence': 0.065}
        
        methods = self.validator._determine_recommended_methods(analysis)
        
        assert "caesar_cipher" in methods
        assert "substitution_cipher" in methods
        assert "skip_patterns" in methods
    
    def test_determine_analysis_priority(self):
        """Test analysis priority determination"""
        analysis = {'length': 100, 'entropy': 4.0}
        methods = ["caesar_cipher", "atbash_cipher", "substitution_cipher"]
        
        priority = self.validator._determine_analysis_priority(analysis, methods)
        
        assert CipherType.CAESAR in priority
        assert CipherType.ATBASH in priority
        assert CipherType.SUBSTITUTION in priority
        # Caesar and Atbash should come before Substitution
        assert priority.index(CipherType.CAESAR) < priority.index(CipherType.SUBSTITUTION)
    
    def test_suggest_analysis_parameters(self):
        """Test analysis parameter suggestions"""
        analysis = {'length': 150}
        priority = [CipherType.CAESAR, CipherType.SKIP_PATTERN, CipherType.SUBSTITUTION]
        
        parameters = self.validator._suggest_analysis_parameters(analysis, priority)
        
        assert 'min_confidence' in parameters
        assert 'max_iterations' in parameters
        assert 'skip_pattern' in parameters
        assert 'substitution' in parameters
        assert parameters['min_confidence'] > 0
        assert parameters['max_iterations'] > 0
    
    def test_estimate_success_rate(self):
        """Test success rate estimation"""
        analysis = {
            'length': 200,
            'character_count': 20,
            'entropy': 4.1
        }
        methods = ["caesar_cipher", "substitution_cipher", "atbash_cipher"]
        
        success_rate = self.validator._estimate_success_rate(analysis, methods)
        
        assert 0.05 <= success_rate <= 0.95
        assert isinstance(success_rate, float)
    
    def test_generate_recommendation_reasoning(self):
        """Test recommendation reasoning generation"""
        analysis = {
            'length': 150,
            'entropy': 3.2,
            'index_of_coincidence': 0.07
        }
        methods = ["caesar_cipher"]
        priority = [CipherType.CAESAR]
        
        reasoning = self.validator._generate_recommendation_reasoning(analysis, methods, priority)
        
        assert "150 characters" in reasoning
        assert "entropy" in reasoning.lower()
        assert isinstance(reasoning, str)
        assert len(reasoning) > 20
    
    def test_suggest_alternative_approaches(self):
        """Test alternative approach suggestions"""
        analysis = {'length': 100, 'formatting_variations': True}
        methods = ["caesar_cipher", "substitution_cipher"]
        
        alternatives = self.validator._suggest_alternative_approaches(analysis, methods)
        
        assert len(alternatives) > 0
        assert any("typography" in alt.lower() for alt in alternatives)
        assert any("manual pattern" in alt.lower() for alt in alternatives)
    
    def test_helper_methods(self):
        """Test various helper methods"""
        # Test random text comparison
        english_text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
        random_score = self.validator._test_against_random_text(english_text)
        assert 0 <= random_score <= 1
        
        # Test Caesar significance
        caesar_score = self.validator._test_caesar_significance(self.sample_caesar_match)
        assert 0 <= caesar_score <= 1
        
        # Test substitution significance
        subst_score = self.validator._test_substitution_significance(self.sample_substitution_match)
        assert 0 <= subst_score <= 1
        
        # Test syllable structure analysis
        syllable_score = self.validator._analyze_syllable_structure(english_text)
        assert 0 <= syllable_score <= 1
        
        # Test consonant-vowel distribution
        cv_score = self.validator._analyze_consonant_vowel_distribution(english_text)
        assert 0 <= cv_score <= 1
        
        # Test letter combinations
        combo_score = self.validator._analyze_letter_combinations(english_text)
        assert 0 <= combo_score <= 1
    
    def test_create_detailed_analysis(self):
        """Test detailed analysis creation"""
        analysis = self.validator._create_detailed_analysis(self.sample_caesar_match)
        
        assert 'cipher_type' in analysis
        assert 'confidence_breakdown' in analysis
        assert 'text_analysis' in analysis
        assert 'quality_indicators' in analysis
        assert 'potential_improvements' in analysis
        assert analysis['cipher_type'] == 'caesar'
    
    def test_create_visual_examples(self):
        """Test visual examples creation"""
        examples = self.validator._create_visual_examples(self.sample_caesar_match)
        
        assert 'transformation_sample' in examples
        assert 'method_example' in examples
        assert "→" in examples['transformation_sample']
        assert "shift of 3" in examples['method_example']
    
    def test_suggest_next_steps_valid(self):
        """Test next steps suggestions for valid detection"""
        validation = ValidationResult(
            is_valid=True,
            confidence_score=0.8,
            statistical_tests={},
            validation_details={},
            reliability_assessment="High",
            potential_issues=[],
            recommendation="Accept"
        )
        
        steps = self.validator._suggest_next_steps(self.sample_caesar_match, validation)
        
        assert len(steps) > 0
        assert any("proceed" in step.lower() for step in steps)
        assert any("verify" in step.lower() for step in steps)
    
    def test_suggest_next_steps_invalid(self):
        """Test next steps suggestions for invalid detection"""
        validation = ValidationResult(
            is_valid=False,
            confidence_score=0.3,
            statistical_tests={},
            validation_details={},
            reliability_assessment="Low",
            potential_issues=["Low confidence"],
            recommendation="Reject"
        )
        
        steps = self.validator._suggest_next_steps(self.sample_caesar_match, validation)
        
        assert len(steps) > 0
        assert any("re-examine" in step.lower() for step in steps)
        assert any("alternative" in step.lower() for step in steps)
    
    def test_error_handling(self):
        """Test error handling for edge cases"""
        # Test with empty text
        empty_match = CipherMatch(
            cipher_type=CipherType.CAESAR,
            confidence=0.5,
            key_or_pattern="shift_1",
            decoded_text="",
            original_text="",
            position_start=0,
            position_end=0,
            method_details={},
            statistical_analysis={}
        )
        
        validation = self.validator.validate_cipher_detection(empty_match)
        assert isinstance(validation, ValidationResult)
        assert validation.confidence_score >= 0
        
        # Test with None values
        recommendations = self.validator.generate_cipher_recommendations("")
        assert isinstance(recommendations, CipherRecommendation)
        assert len(recommendations.recommended_methods) > 0
    
    def test_integration_with_cipher_detector(self):
        """Test integration with cipher detector"""
        # This test ensures the validator works with actual cipher detector output
        from app.services.cipher_detector import CipherDetector
        
        detector = CipherDetector()
        test_text = "KHOOR ZRUOG"  # "HELLO WORLD" with Caesar shift of 3
        
        matches = detector.detect_ciphers(test_text)
        
        if matches:  # If cipher detector finds matches
            for match in matches:
                validation = self.validator.validate_cipher_detection(match)
                assert isinstance(validation, ValidationResult)
                
                explanation = self.validator.explain_cipher_method(match.cipher_type)
                assert isinstance(explanation, CipherExplanation)
        
        # Test recommendations
        recommendations = self.validator.generate_cipher_recommendations(test_text)
        assert isinstance(recommendations, CipherRecommendation)
        assert len(recommendations.recommended_methods) > 0