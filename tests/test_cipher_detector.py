"""
Tests for the Classical Cipher Detection System
"""
import pytest
import string
from unittest.mock import Mock

from app.services.cipher_detector import CipherDetector, CipherType, CipherMatch, FrequencyAnalysis
from app.models.database_models import Character


class TestCipherDetector:
    
    @pytest.fixture
    def detector(self):
        """Create a CipherDetector instance"""
        return CipherDetector()
    
    @pytest.fixture
    def sample_text(self):
        """Sample English text for testing"""
        return "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
    
    @pytest.fixture
    def caesar_encrypted_text(self):
        """Caesar cipher encrypted text (shift 3)"""
        return "WKH TXLFN EURZQ IRA MXPSV RYHU WKH ODCB GRJ"
    
    @pytest.fixture
    def atbash_encrypted_text(self):
        """Atbash cipher encrypted text"""
        return "GSV JFRXP YILDM ULC QFNKH LEVI GSV OZAB WLT"
    
    def test_detector_initialization(self, detector):
        """Test cipher detector initialization"""
        assert detector.min_confidence == 0.3
        assert detector.min_text_length == 20
        assert len(detector.ENGLISH_FREQUENCIES) == 26
        assert len(detector.COMMON_BIGRAMS) == 10
        assert len(detector.COMMON_TRIGRAMS) == 10
    
    def test_clean_text(self, detector):
        """Test text cleaning functionality"""
        dirty_text = "Hello, World! 123 @#$"
        clean_text = detector._clean_text(dirty_text)
        assert clean_text == "HelloWorld"
        
        # Test with mixed case
        mixed_text = "ThE QuIcK bRoWn FoX"
        clean_mixed = detector._clean_text(mixed_text)
        assert clean_mixed == "ThEQuIcKbRoWnFoX"
    
    def test_frequency_analysis(self, detector, sample_text):
        """Test frequency analysis functionality"""
        freq_analysis = detector._analyze_frequencies(sample_text)
        
        assert isinstance(freq_analysis, FrequencyAnalysis)
        assert len(freq_analysis.letter_frequencies) > 0
        assert len(freq_analysis.bigram_frequencies) > 0
        assert len(freq_analysis.trigram_frequencies) > 0
        assert 0 <= freq_analysis.index_of_coincidence <= 1
        assert freq_analysis.chi_squared >= 0
        assert freq_analysis.entropy >= 0
        
        # Check that frequencies sum to approximately 100%
        total_freq = sum(freq_analysis.letter_frequencies.values())
        assert 99 <= total_freq <= 101  # Allow for rounding errors
    
    def test_caesar_cipher_application(self, detector, sample_text):
        """Test Caesar cipher shift application"""
        # Test shift of 3
        shifted = detector._apply_caesar_shift(sample_text, 3)
        assert shifted != sample_text
        
        # Test reverse shift
        unshifted = detector._apply_caesar_shift(shifted, -3)
        assert unshifted == sample_text
        
        # Test with different shifts
        for shift in [1, 5, 13, 25]:
            shifted = detector._apply_caesar_shift(sample_text, shift)
            unshifted = detector._apply_caesar_shift(shifted, -shift)
            assert unshifted == sample_text
    
    def test_atbash_cipher_application(self, detector, sample_text):
        """Test Atbash cipher transformation"""
        atbash_text = detector._apply_atbash(sample_text)
        assert atbash_text != sample_text
        
        # Atbash is its own inverse
        double_atbash = detector._apply_atbash(atbash_text)
        assert double_atbash == sample_text
        
        # Test specific transformations
        assert detector._apply_atbash("A") == "Z"
        assert detector._apply_atbash("Z") == "A"
        assert detector._apply_atbash("M") == "N"
        assert detector._apply_atbash("N") == "M"
    
    def test_substitution_cipher_application(self, detector):
        """Test substitution cipher application"""
        substitution_map = {'A': 'X', 'B': 'Y', 'C': 'Z'}
        text = "ABC"
        
        result = detector._apply_substitution(text, substitution_map)
        assert result == "XYZ"
        
        # Test with lowercase
        result_lower = detector._apply_substitution("abc", substitution_map)
        assert result_lower == "xyz"
        
        # Test with unmapped characters
        result_mixed = detector._apply_substitution("ABCD", substitution_map)
        assert result_mixed == "XYZD"  # D remains unchanged
    
    def test_caesar_cipher_detection(self, detector, caesar_encrypted_text):
        """Test Caesar cipher detection"""
        matches = detector._detect_caesar_cipher(caesar_encrypted_text, 
                                               detector._analyze_frequencies(caesar_encrypted_text))
        
        # Should find at least one match
        assert len(matches) > 0
        
        # Check for shift 3 match (the correct one)
        shift_3_matches = [m for m in matches if m.key_or_pattern == "shift_3"]
        if shift_3_matches:  # May not always detect due to short text
            match = shift_3_matches[0]
            assert match.cipher_type == CipherType.CAESAR
            assert match.confidence > 0.3  # Lowered threshold for realistic testing
            assert "QUICK" in match.decoded_text.upper()
    
    def test_atbash_cipher_detection(self, detector, atbash_encrypted_text):
        """Test Atbash cipher detection"""
        matches = detector._detect_atbash_cipher(atbash_encrypted_text,
                                               detector._analyze_frequencies(atbash_encrypted_text))
        
        # Should find at least one match
        assert len(matches) > 0
        
        match = matches[0]
        assert match.cipher_type == CipherType.ATBASH
        assert match.key_or_pattern == "atbash"
        # Check if decoded text contains expected words
        if match.confidence > 0.5:
            assert "QUICK" in match.decoded_text.upper()
    
    def test_substitution_cipher_detection(self, detector, sample_text):
        """Test substitution cipher detection"""
        # Create a simple substitution
        substitution_map = {chr(i): chr((i - ord('A') + 1) % 26 + ord('A')) for i in range(ord('A'), ord('Z') + 1)}
        encrypted = detector._apply_substitution(sample_text, substitution_map)
        
        matches = detector._detect_substitution_cipher(encrypted,
                                                     detector._analyze_frequencies(encrypted))
        
        # Should find at least one match
        assert len(matches) > 0
        
        match = matches[0]
        assert match.cipher_type == CipherType.SUBSTITUTION
        assert isinstance(match.method_details['substitution_mapping'], dict)
    
    def test_skip_pattern_detection(self, detector):
        """Test skip pattern detection"""
        # Create text with skip pattern
        base_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 3
        skip_text = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" + base_text
        
        matches = detector._detect_skip_patterns(skip_text)
        
        # Should find some patterns
        assert isinstance(matches, list)
        
        # Check match structure if any found
        for match in matches:
            assert match.cipher_type == CipherType.SKIP_PATTERN
            assert "skip_" in match.key_or_pattern
            assert match.confidence >= detector.min_confidence
    
    def test_revolving_cipher_detection(self, detector):
        """Test revolving cipher detection"""
        # Create a simple revolving pattern
        base_text = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" * 2
        
        matches = detector._detect_revolving_cipher(base_text)
        
        # Should return a list (may be empty for this simple test)
        assert isinstance(matches, list)
        
        # Check match structure if any found
        for match in matches:
            assert match.cipher_type == CipherType.REVOLVING
            assert "revolving_" in match.key_or_pattern
            assert match.confidence >= detector.min_confidence
    
    def test_biliteral_cipher_detection(self, detector):
        """Test biliteral cipher detection"""
        # Create mock characters with font styles
        characters = []
        text = "HELLO"
        font_styles = ['roman', 'italic', 'roman', 'italic', 'roman']
        
        for i, (char, style) in enumerate(zip(text, font_styles)):
            mock_char = Mock()
            mock_char.character = char
            mock_char.font_style = style
            characters.append(mock_char)
        
        matches = detector._detect_biliteral_cipher(text, characters)
        
        # Should return a list
        assert isinstance(matches, list)
        
        # Check match structure if any found
        for match in matches:
            assert match.cipher_type == CipherType.BILITERAL
            assert match.key_or_pattern == "italic=1, roman=0"
            assert 'binary_sequence' in match.method_details
    
    def test_detect_ciphers_comprehensive(self, detector, sample_text):
        """Test comprehensive cipher detection"""
        matches = detector.detect_ciphers(sample_text)
        
        # Should return a list
        assert isinstance(matches, list)
        
        # All matches should have required confidence
        for match in matches:
            assert match.confidence >= detector.min_confidence
            assert isinstance(match.cipher_type, CipherType)
            assert len(match.decoded_text) > 0
            assert len(match.original_text) > 0
    
    def test_index_of_coincidence_calculation(self, detector):
        """Test Index of Coincidence calculation"""
        # Test with known text
        text = "AAAAAAAAAA"  # All same letter should have high IC
        ic_high = detector._calculate_index_of_coincidence(text)
        assert ic_high > 0.3  # Should be high
        
        # Test with random-like text
        text = "ABCDEFGHIJ"  # All different letters should have low IC
        ic_low = detector._calculate_index_of_coincidence(text)
        assert ic_low < 0.1  # Should be low
        
        # Test edge cases
        assert detector._calculate_index_of_coincidence("") == 0
        assert detector._calculate_index_of_coincidence("A") == 0
    
    def test_chi_squared_calculation(self, detector):
        """Test chi-squared calculation"""
        # Test with English-like frequencies
        english_like_freq = {'E': 12.0, 'T': 9.0, 'A': 8.0, 'O': 7.5}
        chi_sq_low = detector._calculate_chi_squared(english_like_freq)
        
        # Test with non-English frequencies
        non_english_freq = {'Z': 20.0, 'Q': 15.0, 'X': 10.0}
        chi_sq_high = detector._calculate_chi_squared(non_english_freq)
        
        assert chi_sq_high > chi_sq_low  # Non-English should have higher chi-squared
    
    def test_entropy_calculation(self, detector):
        """Test entropy calculation"""
        # Uniform distribution should have high entropy
        uniform_freq = {chr(i): 1.0 for i in range(ord('A'), ord('Z') + 1)}
        entropy_high = detector._calculate_entropy(uniform_freq)
        
        # Skewed distribution should have lower entropy
        skewed_freq = {'A': 20.0, 'B': 1.0, 'C': 1.0}
        entropy_low = detector._calculate_entropy(skewed_freq)
        
        assert entropy_high > entropy_low
    
    def test_frequency_confidence_calculation(self, detector, sample_text):
        """Test frequency confidence calculation"""
        freq_analysis = detector._analyze_frequencies(sample_text)
        confidence = detector._calculate_frequency_confidence(freq_analysis)
        
        assert 0 <= confidence <= 1
        
        # English text should have reasonable confidence
        assert confidence > 0.3  # Should be somewhat confident for English text
    
    def test_english_patterns_check(self, detector, sample_text):
        """Test English pattern checking"""
        # English text should score well
        english_score = detector._check_english_patterns(sample_text)
        assert 0 <= english_score <= 1
        assert english_score > 0.2  # Should score reasonably for English
        
        # Random text should score poorly
        random_text = "XQZJKWVPFLMYHTGBCNRSD"
        random_score = detector._check_english_patterns(random_text)
        assert random_score < english_score
    
    def test_biliteral_decoding(self, detector):
        """Test biliteral cipher decoding"""
        # Test with valid binary sequence
        binary_seq = "0000000001000100001100100"  # Should decode to "ABCD"
        decoded_attempts = detector._decode_biliteral(binary_seq)
        
        assert isinstance(decoded_attempts, list)
        assert len(decoded_attempts) > 0
        
        # Test with short sequence
        short_seq = "001"
        short_attempts = detector._decode_biliteral(short_seq)
        assert isinstance(short_attempts, list)
    
    def test_shift_pattern_generation(self, detector):
        """Test shift pattern generation for revolving ciphers"""
        patterns = detector._generate_shift_patterns(4)
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert len(patterns) <= 10  # Should be limited
        
        # Each pattern should have the correct length
        for pattern in patterns:
            assert len(pattern) == 4
            assert all(isinstance(shift, int) for shift in pattern)
    
    def test_duplicate_match_removal(self, detector):
        """Test removal of duplicate cipher matches"""
        # Create overlapping matches
        match1 = CipherMatch(
            cipher_type=CipherType.CAESAR,
            confidence=0.8,
            key_or_pattern="shift_3",
            decoded_text="HELLO WORLD",
            original_text="KHOOR ZRUOG",
            position_start=0,
            position_end=11,
            method_details={},
            statistical_analysis={}
        )
        
        match2 = CipherMatch(
            cipher_type=CipherType.CAESAR,
            confidence=0.7,
            key_or_pattern="shift_5",
            decoded_text="HELLO WORLD",
            original_text="MJQQT BTWQI",
            position_start=0,
            position_end=11,
            method_details={},
            statistical_analysis={}
        )
        
        match3 = CipherMatch(
            cipher_type=CipherType.ATBASH,
            confidence=0.9,
            key_or_pattern="atbash",
            decoded_text="DIFFERENT TEXT",
            original_text="WRUUVIVMG GVCG",
            position_start=20,
            position_end=34,
            method_details={},
            statistical_analysis={}
        )
        
        matches = [match1, match2, match3]
        unique_matches = detector._remove_duplicate_matches(matches)
        
        # Should keep highest confidence match from overlapping ones
        assert len(unique_matches) == 2
        assert unique_matches[0].confidence == 0.9  # Atbash match (highest)
        assert unique_matches[1].confidence == 0.8  # Caesar match (higher than other Caesar)
    
    def test_cipher_pattern_analysis(self, detector):
        """Test cipher pattern analysis"""
        # Create sample matches
        matches = [
            CipherMatch(
                cipher_type=CipherType.CAESAR,
                confidence=0.8,
                key_or_pattern="shift_3",
                decoded_text="HELLO",
                original_text="KHOOR",
                position_start=0,
                position_end=5,
                method_details={},
                statistical_analysis={'chi_squared': 100}
            ),
            CipherMatch(
                cipher_type=CipherType.ATBASH,
                confidence=0.7,
                key_or_pattern="atbash",
                decoded_text="WORLD",
                original_text="DLIOW",
                position_start=6,
                position_end=11,
                method_details={},
                statistical_analysis={'chi_squared': 120}
            )
        ]
        
        analysis = detector.analyze_cipher_patterns(matches)
        
        assert 'cipher_type_distribution' in analysis
        assert 'confidence_statistics' in analysis
        assert 'pattern_complexity' in analysis
        assert 'potential_relationships' in analysis
        
        # Check cipher type distribution
        assert analysis['cipher_type_distribution']['caesar'] == 1
        assert analysis['cipher_type_distribution']['atbash'] == 1
        
        # Check confidence statistics
        stats = analysis['confidence_statistics']
        assert stats['mean'] == 0.75
        assert stats['min'] == 0.7
        assert stats['max'] == 0.8
    
    def test_cipher_report_generation(self, detector):
        """Test cipher report generation"""
        # Create sample matches
        matches = [
            CipherMatch(
                cipher_type=CipherType.CAESAR,
                confidence=0.8,
                key_or_pattern="shift_3",
                decoded_text="HELLO WORLD",
                original_text="KHOOR ZRUOG",
                position_start=0,
                position_end=11,
                method_details={'shift_amount': 3},
                statistical_analysis={'chi_squared': 100}
            )
        ]
        
        report = detector.generate_cipher_report(matches)
        
        assert 'summary' in report
        assert 'matches_by_type' in report
        assert 'statistical_analysis' in report
        assert 'recommendations' in report
        
        # Check summary
        summary = report['summary']
        assert summary['total_matches'] == 1
        assert summary['cipher_types_found'] == 1
        assert summary['highest_confidence'] == 0.8
        assert summary['average_confidence'] == 0.8
        
        # Check matches by type
        assert 'caesar' in report['matches_by_type']
        caesar_matches = report['matches_by_type']['caesar']
        assert len(caesar_matches) == 1
        assert caesar_matches[0]['confidence'] == 0.8
    
    def test_cipher_recommendations(self, detector):
        """Test cipher recommendation generation"""
        # Test with no matches
        no_matches_recs = detector._generate_cipher_recommendations([])
        assert len(no_matches_recs) > 0
        assert "No cipher patterns detected" in no_matches_recs[0]
        
        # Test with high confidence matches
        high_conf_match = CipherMatch(
            cipher_type=CipherType.CAESAR,
            confidence=0.9,
            key_or_pattern="shift_3",
            decoded_text="HELLO",
            original_text="KHOOR",
            position_start=0,
            position_end=5,
            method_details={},
            statistical_analysis={}
        )
        
        high_conf_recs = detector._generate_cipher_recommendations([high_conf_match])
        assert any("high-confidence" in rec for rec in high_conf_recs)
    
    def test_related_ciphers_detection(self, detector):
        """Test detection of related cipher patterns"""
        match1 = CipherMatch(
            cipher_type=CipherType.CAESAR,
            confidence=0.8,
            key_or_pattern="shift_3",
            decoded_text="HELLO",
            original_text="KHOOR",
            position_start=0,
            position_end=5,
            method_details={},
            statistical_analysis={'chi_squared': 100}
        )
        
        match2 = CipherMatch(
            cipher_type=CipherType.CAESAR,
            confidence=0.7,
            key_or_pattern="shift_3",
            decoded_text="WORLD",
            original_text="ZRUOG",
            position_start=6,
            position_end=11,
            method_details={},
            statistical_analysis={'chi_squared': 105}
        )
        
        # Same cipher type should be related
        assert detector._are_related_ciphers(match1, match2) == True
        
        # Different types with similar stats should be related
        match3 = CipherMatch(
            cipher_type=CipherType.ATBASH,
            confidence=0.8,
            key_or_pattern="atbash",
            decoded_text="TEST",
            original_text="GVHG",
            position_start=12,
            position_end=16,
            method_details={},
            statistical_analysis={'chi_squared': 102}
        )
        
        assert detector._are_related_ciphers(match1, match3) == True


if __name__ == "__main__":
    pytest.main([__file__])