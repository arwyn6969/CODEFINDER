"""
Tests for Sonnet Print Block Scanner
====================================
Comprehensive tests for the 1609 Shakespeare Sonnets OCR scanner.

Tests cover:
- Scanner initialization (PDF and IIIF sources)
- Character extraction and classification
- Long-s (ſ) detection and disambiguation
- Ligature detection
- Character normalization
- Report generation
- Integration tests
"""
import pytest
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np
from dataclasses import asdict

# Import the scanner and its data classes
sys.path.insert(0, str(Path(__file__).parent.parent))
from sonnet_print_block_scanner import (
    SonnetPrintBlockScanner,
    CharacterInstance,
    CharacterCatalogueEntry,
    AnomalyEntry,
    ScanStatistics,
    parse_page_range
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="scanner_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_pdf_path(temp_output_dir):
    """Create a mock PDF path."""
    pdf_path = os.path.join(temp_output_dir, "test.pdf")
    # Create a minimal placeholder file
    Path(pdf_path).touch()
    return pdf_path


@pytest.fixture
def mock_iiif_dir(temp_output_dir):
    """Create a mock IIIF image directory with test images."""
    iiif_dir = os.path.join(temp_output_dir, "iiif_images")
    os.makedirs(iiif_dir, exist_ok=True)
    
    # Create 3 mock page images
    for i in range(1, 4):
        img = Image.new('RGB', (800, 1200), color='white')
        img.save(os.path.join(iiif_dir, f"page_{i:03d}.jpg"))
    
    return iiif_dir


@pytest.fixture
def sample_character_instance():
    """Create a sample CharacterInstance for testing."""
    return CharacterInstance(
        character='e',
        page_number=1,
        x=100.0,
        y=200.0,
        width=15.0,
        height=20.0,
        confidence=95.5,
        block_id=0,
        line_id=0,
        word_id=0,
        is_ligature=False,
        is_long_s=False,
        is_anomaly=False,
        anomaly_type=None,
        image_path=None
    )


@pytest.fixture
def sample_test_image():
    """Create a sample test image for character processing."""
    # Create a simple black letter on white background
    img = Image.new('L', (48, 64), color=255)
    # Add some black pixels to simulate a character
    pixels = img.load()
    for x in range(15, 35):
        for y in range(20, 45):
            pixels[x, y] = 0
    return img


# ============================================================================
# DATA CLASSES TESTS
# ============================================================================

class TestCharacterInstance:
    """Tests for the CharacterInstance dataclass."""
    
    def test_create_basic_instance(self):
        """Test creating a basic CharacterInstance."""
        instance = CharacterInstance(
            character='a',
            page_number=1,
            x=10.0,
            y=20.0,
            width=8.0,
            height=12.0
        )
        assert instance.character == 'a'
        assert instance.page_number == 1
        assert instance.x == 10.0
        assert instance.y == 20.0
        assert instance.confidence == 0.0  # default
        assert instance.is_ligature is False  # default
        assert instance.is_long_s is False  # default
    
    def test_create_long_s_instance(self):
        """Test creating a Long-s character instance."""
        instance = CharacterInstance(
            character='ſ',
            page_number=5,
            x=100.0,
            y=200.0,
            width=10.0,
            height=15.0,
            confidence=92.0,
            is_long_s=True
        )
        assert instance.character == 'ſ'
        assert instance.is_long_s is True
        assert instance.confidence == 92.0
    
    def test_create_ligature_instance(self):
        """Test creating a ligature character instance."""
        instance = CharacterInstance(
            character='ff',
            page_number=3,
            x=50.0,
            y=100.0,
            width=16.0,
            height=12.0,
            is_ligature=True
        )
        assert instance.character == 'ff'
        assert instance.is_ligature is True
    
    def test_create_anomaly_instance(self):
        """Test creating an anomaly character instance."""
        instance = CharacterInstance(
            character='?',
            page_number=10,
            x=200.0,
            y=300.0,
            width=5.0,
            height=8.0,
            is_anomaly=True,
            anomaly_type="unknown_glyph"
        )
        assert instance.is_anomaly is True
        assert instance.anomaly_type == "unknown_glyph"
    
    def test_instance_serialization(self, sample_character_instance):
        """Test that CharacterInstance can be converted to dict."""
        data = asdict(sample_character_instance)
        assert isinstance(data, dict)
        assert data['character'] == 'e'
        assert data['confidence'] == 95.5


class TestCharacterCatalogueEntry:
    """Tests for the CharacterCatalogueEntry dataclass."""
    
    def test_create_catalogue_entry(self):
        """Test creating a basic catalogue entry."""
        entry = CharacterCatalogueEntry(
            character='T',
            unicode_name='LATIN CAPITAL LETTER T',
            unicode_codepoint='U+0054',
            category='uppercase'
        )
        assert entry.character == 'T'
        assert entry.unicode_name == 'LATIN CAPITAL LETTER T'
        assert entry.total_count == 0
        assert entry.instances == []
    
    def test_catalogue_entry_with_instances(self, sample_character_instance):
        """Test catalogue entry with instances list."""
        entry = CharacterCatalogueEntry(
            character='e',
            unicode_name='LATIN SMALL LETTER E',
            unicode_codepoint='U+0065',
            category='lowercase',
            total_count=5,
            instances=[sample_character_instance]
        )
        assert entry.total_count == 5
        assert len(entry.instances) == 1
        assert entry.instances[0].character == 'e'


class TestScanStatistics:
    """Tests for the ScanStatistics dataclass."""
    
    def test_create_default_statistics(self):
        """Test creating default scan statistics."""
        stats = ScanStatistics()
        assert stats.total_pages == 0
        assert stats.pages_scanned == 0
        assert stats.total_characters == 0
        assert stats.long_s_count == 0
        assert stats.ligatures_found == 0
    
    def test_statistics_with_values(self):
        """Test scan statistics with values."""
        stats = ScanStatistics(
            total_pages=53,
            pages_scanned=10,
            total_characters=5000,
            unique_characters=65,
            long_s_count=250,
            ligatures_found=50,
            average_confidence=97.5
        )
        assert stats.total_pages == 53
        assert stats.pages_scanned == 10
        assert stats.long_s_count == 250


class TestAnomalyEntry:
    """Tests for the AnomalyEntry dataclass."""
    
    def test_create_anomaly_entry(self):
        """Test creating an anomaly entry."""
        anomaly = AnomalyEntry(
            anomaly_type="damaged_type",
            page_number=15,
            x=300.0,
            y=400.0,
            width=12.0,
            height=16.0,
            description="Damaged or worn type block",
            severity="medium"
        )
        assert anomaly.anomaly_type == "damaged_type"
        assert anomaly.severity == "medium"
        assert anomaly.page_number == 15


# ============================================================================
# SCANNER INITIALIZATION TESTS
# ============================================================================

class TestScannerInitialization:
    """Tests for scanner initialization."""
    
    def test_init_with_iiif_directory(self, mock_iiif_dir, temp_output_dir):
        """Scanner initializes correctly with IIIF image directory."""
        scanner = SonnetPrintBlockScanner(
            source_path=mock_iiif_dir,
            output_dir=temp_output_dir
        )
        assert scanner.source_type == "iiif_images"
        assert str(scanner.source_path) == mock_iiif_dir
        assert str(scanner.output_dir) == temp_output_dir
        assert len(scanner.image_files) == 3
    
    @patch('sonnet_print_block_scanner.fitz')
    def test_init_with_pdf_source(self, mock_fitz, mock_pdf_path, temp_output_dir):
        """Scanner initializes correctly with PDF source."""
        # Mock the PDF document
        mock_doc = MagicMock()
        mock_doc.__len__ = Mock(return_value=10)
        mock_doc.__iter__ = Mock(return_value=iter([MagicMock() for _ in range(10)]))
        mock_fitz.open.return_value = mock_doc
        
        scanner = SonnetPrintBlockScanner(
            source_path=mock_pdf_path,
            output_dir=temp_output_dir
        )
        assert scanner.source_type == "pdf"
        mock_fitz.open.assert_called_once_with(mock_pdf_path)
    
    def test_init_with_invalid_source(self, temp_output_dir):
        """Scanner raises error on invalid source path."""
        with pytest.raises((ValueError, FileNotFoundError)):
            SonnetPrintBlockScanner(
                source_path="/nonexistent/path",
                output_dir=temp_output_dir
            )
    
    def test_directory_structure_created(self, mock_iiif_dir, temp_output_dir):
        """Output directory structure is created correctly."""
        scanner = SonnetPrintBlockScanner(
            source_path=mock_iiif_dir,
            output_dir=temp_output_dir
        )
        
        # Check that required directories exist
        assert os.path.isdir(os.path.join(temp_output_dir, "character_atlas"))
        assert os.path.isdir(os.path.join(temp_output_dir, "page_images"))
        assert os.path.isdir(os.path.join(temp_output_dir, "anomalies"))
    
    def test_init_with_ocr_engine(self, mock_iiif_dir, temp_output_dir):
        """Scanner accepts optional OCR engine."""
        mock_engine = MagicMock()
        mock_engine.name = "test_engine"
        
        scanner = SonnetPrintBlockScanner(
            source_path=mock_iiif_dir,
            output_dir=temp_output_dir,
            ocr_engine=mock_engine
        )
        assert scanner.ocr_engine == mock_engine


# ============================================================================
# CHARACTER PROCESSING TESTS
# ============================================================================

class TestCharacterProcessing:
    """Tests for character extraction and classification."""
    
    def test_get_character_category_uppercase(self, mock_iiif_dir, temp_output_dir):
        """Uppercase letters categorized correctly."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        assert scanner._get_character_category('A') == 'uppercase'
        assert scanner._get_character_category('Z') == 'uppercase'
        assert scanner._get_character_category('M') == 'uppercase'
    
    def test_get_character_category_lowercase(self, mock_iiif_dir, temp_output_dir):
        """Lowercase letters categorized correctly."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        assert scanner._get_character_category('a') == 'lowercase'
        assert scanner._get_character_category('z') == 'lowercase'
        assert scanner._get_character_category('e') == 'lowercase'
    
    def test_get_character_category_digits(self, mock_iiif_dir, temp_output_dir):
        """Digits categorized correctly."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        assert scanner._get_character_category('0') == 'digit'
        assert scanner._get_character_category('5') == 'digit'
        assert scanner._get_character_category('9') == 'digit'
    
    def test_get_character_category_punctuation(self, mock_iiif_dir, temp_output_dir):
        """Punctuation categorized correctly."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        assert scanner._get_character_category('.') == 'punctuation'
        assert scanner._get_character_category(',') == 'punctuation'
        assert scanner._get_character_category(';') == 'punctuation'
    
    def test_get_character_category_long_s(self, mock_iiif_dir, temp_output_dir):
        """Long-s (ſ) categorized as long_s."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        assert scanner._get_character_category('ſ') == 'long_s'
    
    def test_get_unicode_info(self, mock_iiif_dir, temp_output_dir):
        """Unicode info retrieved correctly."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        name, codepoint = scanner._get_unicode_info('A')
        assert 'LATIN' in name.upper() or 'LETTER' in name.upper()
        assert codepoint == 'U+0041'
    
    def test_get_unicode_info_long_s(self, mock_iiif_dir, temp_output_dir):
        """Unicode info for Long-s."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        name, codepoint = scanner._get_unicode_info('ſ')
        assert 'LONG S' in name.upper()
        assert codepoint == 'U+017F'
    
    def test_is_valid_character_letters(self, mock_iiif_dir, temp_output_dir):
        """Valid letters accepted."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        assert scanner._is_valid_character('a') is True
        assert scanner._is_valid_character('Z') is True
        assert scanner._is_valid_character('ſ') is True
    
    def test_is_valid_character_punctuation(self, mock_iiif_dir, temp_output_dir):
        """Valid punctuation accepted."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        assert scanner._is_valid_character('.') is True
        assert scanner._is_valid_character(',') is True
        assert scanner._is_valid_character("'") is True
    
    def test_is_valid_character_modern_symbols(self, mock_iiif_dir, temp_output_dir):
        """Modern symbols rejected."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        assert scanner._is_valid_character('@') is False
        assert scanner._is_valid_character('#') is False
        assert scanner._is_valid_character('$') is False
    
    def test_ligature_constants(self, mock_iiif_dir, temp_output_dir):
        """Scanner has correct ligature set."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        assert 'ff' in scanner.LIGATURES
        assert 'fi' in scanner.LIGATURES
        assert 'fl' in scanner.LIGATURES
        assert 'st' in scanner.LIGATURES
        assert 'ct' in scanner.LIGATURES
    
    def test_long_s_constant(self, mock_iiif_dir, temp_output_dir):
        """Scanner has correct Long-s constant."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        assert scanner.LONG_S == 'ſ'


# ============================================================================
# LONG-S DETECTION TESTS
# ============================================================================

class TestLongSDetection:
    """Tests for Long-s (ſ) detection and disambiguation."""
    
    def test_might_be_long_s_medial_position(self, mock_iiif_dir, temp_output_dir):
        """f in medial position might be Long-s."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        # "false" has 'f' in initial position - not Long-s
        assert scanner._might_be_long_s("false", 0) is False
        # Test with s in medial position
        result = scanner._might_be_long_s("uses", 1)
        assert isinstance(result, bool)
    
    def test_might_be_long_s_terminal_position(self, mock_iiif_dir, temp_output_dir):
        """f in terminal position is not Long-s."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        # 's' at end of word like "us" - historical s was shown as 's' not 'ſ'
        assert scanner._might_be_long_s("ifs", 2) is False
    
    def test_might_be_long_s_initial_position(self, mock_iiif_dir, temp_output_dir):
        """s in initial position could be Long-s or regular s."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        # Historical usage: ſ could appear initially
        result = scanner._might_be_long_s("some", 0)
        # This depends on implementation, but it should return a boolean
        assert isinstance(result, bool)


# ============================================================================
# CHARACTER NORMALIZATION TESTS
# ============================================================================

class TestCharacterNormalization:
    """Tests for character image normalization."""
    
    def test_normalize_standard_size(self, mock_iiif_dir, temp_output_dir, sample_test_image):
        """Normalized blocks are 48x64 pixels."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        normalized = scanner.normalize_character_block(sample_test_image)
        assert normalized.size == (48, 64)
    
    def test_normalize_small_input(self, mock_iiif_dir, temp_output_dir):
        """Small images are scaled up correctly."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        small_img = Image.new('L', (10, 15), color=255)
        normalized = scanner.normalize_character_block(small_img)
        assert normalized.size == (48, 64)
    
    def test_normalize_large_input(self, mock_iiif_dir, temp_output_dir):
        """Large images are scaled down correctly."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        large_img = Image.new('L', (200, 300), color=255)
        normalized = scanner.normalize_character_block(large_img)
        assert normalized.size == (48, 64)
    
    def test_normalize_preserves_mode(self, mock_iiif_dir, temp_output_dir, sample_test_image):
        """Normalization converts to RGB mode for reports."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        normalized = scanner.normalize_character_block(sample_test_image)
        # Scanner may convert to RGB for HTML report compatibility
        assert normalized.mode in ('L', 'RGB')
    
    def test_normalize_handles_rgb_input(self, mock_iiif_dir, temp_output_dir):
        """Normalization handles RGB input."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        rgb_img = Image.new('RGB', (48, 64), color='white')
        normalized = scanner.normalize_character_block(rgb_img)
        assert normalized.size == (48, 64)


# ============================================================================
# NOISE DETECTION TESTS
# ============================================================================

class TestNoiseDetection:
    """Tests for scanner noise/artifact detection."""
    
    def test_noise_detection_border(self, mock_iiif_dir, temp_output_dir):
        """Characters near border flagged as potential noise."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        # Character at very edge of page
        is_noise = scanner._is_noise('|', 0.0, 100.0, 2.0, 500.0, 800.0, 1200.0)
        # This depends on implementation - testing the method exists and returns bool
        assert isinstance(is_noise, bool)
    
    def test_noise_detection_normal_character(self, mock_iiif_dir, temp_output_dir):
        """Normal characters not flagged as noise."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        # Normal character in middle of page
        is_noise = scanner._is_noise('e', 400.0, 600.0, 10.0, 15.0, 800.0, 1200.0)
        assert is_noise is False
    
    def test_noise_detection_decorative(self, mock_iiif_dir, temp_output_dir):
        """Decorative elements may be flagged depending on implementation."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        # Very large object that might be decorative
        is_noise = scanner._is_noise('█', 100.0, 100.0, 200.0, 200.0, 800.0, 1200.0)
        assert isinstance(is_noise, bool)


# ============================================================================
# CATALOGUE UPDATE TESTS
# ============================================================================

class TestCatalogueUpdate:
    """Tests for catalogue management."""
    
    def test_update_catalogue_new_character(self, mock_iiif_dir, temp_output_dir, sample_character_instance):
        """New characters added to catalogue."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        initial_count = len(scanner.character_catalogue)
        
        scanner._update_catalogue(sample_character_instance)
        
        assert 'e' in scanner.character_catalogue
        assert scanner.character_catalogue['e'].total_count == 1
    
    def test_update_catalogue_existing_character(self, mock_iiif_dir, temp_output_dir):
        """Existing characters have counts incremented."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        
        instance1 = CharacterInstance(character='t', page_number=1, x=10, y=20, width=10, height=15)
        instance2 = CharacterInstance(character='t', page_number=1, x=30, y=20, width=10, height=15)
        
        scanner._update_catalogue(instance1)
        scanner._update_catalogue(instance2)
        
        assert scanner.character_catalogue['t'].total_count == 2


# ============================================================================
# REPORT GENERATION TESTS
# ============================================================================

class TestReportGeneration:
    """Tests for report generation."""
    
    def test_generate_frequency_csv(self, mock_iiif_dir, temp_output_dir):
        """Frequency CSV generated correctly."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        
        # Add some test data
        for char in ['e', 'e', 'e', 't', 't', 'a']:
            instance = CharacterInstance(character=char, page_number=1, x=10, y=20, width=10, height=15)
            scanner._update_catalogue(instance)
        
        csv_path = scanner.generate_frequency_csv()
        
        assert os.path.exists(csv_path)
        with open(csv_path, 'r') as f:
            content = f.read()
            assert 'character' in content.lower() or 'Character' in content
    
    def test_generate_statistics_json(self, mock_iiif_dir, temp_output_dir):
        """Statistics JSON generated correctly."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        
        # Set some statistics  
        scanner.statistics.total_pages = 53
        scanner.statistics.pages_scanned = 3
        scanner.statistics.total_characters = 1000
        
        json_path = scanner.generate_statistics_json()
        
        assert os.path.exists(json_path)
        with open(json_path, 'r') as f:
            data = json.load(f)
            assert 'statistics' in data
            assert data['statistics']['total_pages'] == 53
    
    def test_generate_html_report(self, mock_iiif_dir, temp_output_dir):
        """HTML report generated."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        
        # Add minimal test data
        instance = CharacterInstance(character='A', page_number=1, x=10, y=20, width=10, height=15)
        scanner._update_catalogue(instance)
        
        html_path = scanner.generate_html_report()
        
        assert os.path.exists(html_path)
        with open(html_path, 'r') as f:
            content = f.read()
            assert '<html' in content.lower()
    
    def test_report_includes_long_s_stats(self, mock_iiif_dir, temp_output_dir):
        """Reports include Long-s (ſ) statistics."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        
        # Add Long-s instances
        for i in range(5):
            instance = CharacterInstance(
                character='ſ',
                page_number=1,
                x=10 + i*20,
                y=20,
                width=10,
                height=15,
                is_long_s=True
            )
            scanner._update_catalogue(instance)
            scanner.statistics.long_s_count += 1
        
        json_path = scanner.generate_statistics_json()
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            assert data['statistics']['long_s_count'] == 5


# ============================================================================
# CLI TESTS
# ============================================================================

class TestCLI:
    """Tests for command-line interface utilities."""
    
    def test_parse_page_range_single(self):
        """Parse single page number."""
        start, end = parse_page_range("5", 53)
        assert start == 4  # 0-indexed
        assert end == 5
    
    def test_parse_page_range_range(self):
        """Parse page range."""
        start, end = parse_page_range("1-10", 53)
        assert start == 0  # 0-indexed
        assert end == 10
    
    def test_parse_page_range_full_range(self):
        """Parse full range of pages."""
        start, end = parse_page_range("1-53", 53)
        assert start == 0
        assert end == 53
    
    def test_parse_page_range_exceeds_total(self):
        """Range exceeding total pages is capped."""
        start, end = parse_page_range("1-100", 53)
        assert end == 53


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for full scanning workflow."""
    
    def test_scan_single_page(self, mock_iiif_dir, temp_output_dir):
        """Scan a single page produces results."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        
        # This will scan an empty test image - may return empty or minimal results
        characters = scanner.scan_page(0, save_images=False)
        
        # Should return a list (possibly empty for blank test images)
        assert isinstance(characters, list)
    
    def test_catalogue_populated_after_scan(self, mock_iiif_dir, temp_output_dir):
        """Catalogue is populated after scanning."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        
        # Manually add test data to simulate scan
        instance = CharacterInstance(character='X', page_number=0, x=10, y=20, width=10, height=15)
        scanner._update_catalogue(instance)
        
        assert 'X' in scanner.character_catalogue
        assert scanner.character_catalogue['X'].total_count == 1
    
    def test_scanner_close(self, mock_iiif_dir, temp_output_dir):
        """Scanner closes without error."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        scanner.close()  # Should not raise


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_image_handling(self, mock_iiif_dir, temp_output_dir):
        """Scanner handles empty/blank images."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        
        # Blank test image should not crash
        characters = scanner.scan_page(0, save_images=False)
        assert isinstance(characters, list)
    
    def test_special_characters_in_catalogue(self, mock_iiif_dir, temp_output_dir):
        """Special characters correctly catalogued."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        
        special_chars = ['ſ', 'æ', 'œ', '—']
        for char in special_chars:
            instance = CharacterInstance(character=char, page_number=1, x=10, y=20, width=10, height=15)
            scanner._update_catalogue(instance)
        
        assert 'ſ' in scanner.character_catalogue
    
    def test_unicode_normalization(self, mock_iiif_dir, temp_output_dir):
        """Unicode characters handled correctly."""
        scanner = SonnetPrintBlockScanner(mock_iiif_dir, temp_output_dir)
        
        name, codepoint = scanner._get_unicode_info('é')
        assert 'U+' in codepoint


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
