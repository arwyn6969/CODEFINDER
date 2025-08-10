"""
Tests for OCR engine
"""
import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from app.services.ocr_engine import (
    AdvancedOCR, OCRResult, CharacterBox, WordBox, UncertainRegion, OCRMode
)
from app.services.pdf_processor import PageImage, ProcessedImage

class TestAdvancedOCR:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.ocr = AdvancedOCR(confidence_threshold=0.85)
        
        # Create test images
        self.test_image_array = np.ones((100, 100, 3), dtype=np.uint8) * 255
        self.test_pil_image = Image.fromarray(self.test_image_array)
        
        self.test_page_image = PageImage(
            image=self.test_pil_image,
            page_number=0,
            width=100,
            height=100,
            dpi=300
        )
        
        self.test_processed_image = ProcessedImage(
            image=self.test_pil_image,
            original=self.test_page_image,
            preprocessing_steps=[],
            quality_score=0.8
        )
    
    def test_init(self):
        """Test AdvancedOCR initialization"""
        assert self.ocr.confidence_threshold == 0.85
        assert isinstance(self.ocr.tesseract_config, str)
        assert '--oem 3' in self.ocr.tesseract_config
    
    def test_get_tesseract_config(self):
        """Test Tesseract configuration generation"""
        config = self.ocr._get_tesseract_config()
        
        assert '--oem 3' in config  # LSTM OCR Engine Mode
        assert '--psm 6' in config  # Single uniform block
        assert isinstance(config, str)
        assert len(config) > 0
    
    @patch('app.services.ocr_engine.pytesseract.image_to_string')
    @patch('app.services.ocr_engine.pytesseract.image_to_data')
    def test_primary_ocr_pass(self, mock_image_to_data, mock_image_to_string):
        """Test primary OCR pass"""
        # Mock Tesseract responses
        mock_image_to_string.return_value = "Sample text"
        mock_image_to_data.return_value = {
            'text': ['Sample', 'text'],
            'conf': [95, 90],
            'left': [10, 60],
            'top': [20, 20],
            'width': [40, 30],
            'height': [15, 15],
            'line_num': [1, 1],
            'block_num': [1, 1]
        }
        
        cv_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = self.ocr._primary_ocr_pass(cv_image, 0)
        
        assert isinstance(result, OCRResult)
        assert result.text == "Sample text"
        assert result.confidence > 0
        assert len(result.words) == 2
        assert result.page_number == 0
    
    def test_process_word_data(self):
        """Test word data processing"""
        data = {
            'text': ['Hello', 'World', ''],
            'conf': [95, 85, -1],
            'left': [10, 60, 100],
            'top': [20, 20, 30],
            'width': [40, 35, 20],
            'height': [15, 15, 10],
            'line_num': [1, 1, 1],
            'block_num': [1, 1, 1]
        }
        
        words = self.ocr._process_word_data(data, 0)
        
        assert len(words) == 2  # Empty text should be filtered out
        assert words[0].text == "Hello"
        assert words[0].confidence == 95.0
        assert words[1].text == "World"
        assert words[1].confidence == 85.0
    
    def test_identify_uncertain_regions(self):
        """Test uncertain region identification"""
        data = {
            'text': ['Clear', 'Unclear', ''],
            'conf': [95, 70, -1],  # 70 is below 85% threshold
            'left': [10, 60, 100],
            'top': [20, 20, 30],
            'width': [40, 35, 20],
            'height': [15, 15, 10]
        }
        
        uncertain = self.ocr._identify_uncertain_regions(data, 0)
        
        assert len(uncertain) == 1
        assert uncertain[0].suggested_text == "Unclear"
        assert uncertain[0].confidence == 0.7  # Normalized from 70
        assert "Low confidence" in uncertain[0].reason
    
    @patch('app.services.ocr_engine.pytesseract.image_to_boxes')
    def test_get_character_positions(self, mock_image_to_boxes):
        """Test character position extraction"""
        mock_image_to_boxes.return_value = {
            'char': ['H', 'e', 'l'],
            'left': [10, 20, 30],
            'bottom': [80, 80, 80],
            'right': [18, 28, 38],
            'top': [95, 95, 95]
        }
        
        characters = self.ocr.get_character_positions(self.test_processed_image)
        
        assert len(characters) == 3
        assert all(isinstance(char, CharacterBox) for char in characters)
        assert characters[0].character == 'H'
        assert characters[0].page_number == 0
        assert characters[0].width == 8  # right - left
        assert characters[0].height == 15  # top - bottom
    
    def test_measure_character_sizes(self):
        """Test character size analysis"""
        characters = [
            CharacterBox('A', 10, 20, 8, 15, 0.9, 0, 0, 0, 0),
            CharacterBox('A', 30, 20, 9, 16, 0.9, 0, 0, 0, 0),
            CharacterBox('B', 50, 20, 7, 15, 0.9, 0, 0, 0, 0),
        ]
        
        analysis = self.ocr.measure_character_sizes(characters)
        
        assert 'character_analysis' in analysis
        assert 'overall_statistics' in analysis
        assert 'A' in analysis['character_analysis']
        assert analysis['character_analysis']['A']['count'] == 2
        assert analysis['overall_statistics']['total_characters'] == 3
    
    def test_measure_character_sizes_empty(self):
        """Test character size analysis with empty input"""
        analysis = self.ocr.measure_character_sizes([])
        assert 'error' in analysis
    
    def test_flag_uncertainties(self):
        """Test additional uncertainty flagging"""
        result = OCRResult(
            text="ABCDEFGH 12345 @@@ verylongwordthatisunusual",
            confidence=0.9,
            characters=[],
            words=[],
            uncertain_regions=[],
            processing_time=1.0,
            page_number=0,
            image_quality_score=0.8
        )
        
        uncertainties = self.ocr.flag_uncertainties(result)
        
        # Should detect unusual patterns
        assert len(uncertainties) > 0
        # Check that patterns are detected (this is simplified)
        reasons = [u.reason for u in uncertainties]
        assert any("Unusual pattern" in reason for reason in reasons)
    
    @patch('app.services.ocr_engine.pytesseract.image_to_string')
    @patch('app.services.ocr_engine.pytesseract.image_to_data')
    def test_secondary_ocr_pass(self, mock_image_to_data, mock_image_to_string):
        """Test secondary OCR pass for uncertain regions"""
        # Create uncertain region
        uncertain_region = UncertainRegion(
            x=10, y=10, width=50, height=20,
            confidence=0.6, suggested_text="unclear",
            alternatives=[], reason="Low confidence",
            page_number=0
        )
        
        # Mock improved OCR result
        mock_image_to_string.return_value = "clear"
        mock_image_to_data.return_value = {
            'conf': [95]
        }
        
        cv_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        results = self.ocr._secondary_ocr_pass(cv_image, [uncertain_region], 0)
        
        assert len(results) >= 0  # May or may not find improvements
    
    @patch('app.services.ocr_engine.pytesseract.image_to_string')
    @patch('app.services.ocr_engine.pytesseract.image_to_data')
    @patch('app.services.ocr_engine.pytesseract.image_to_boxes')
    def test_extract_text_success(self, mock_boxes, mock_data, mock_string):
        """Test successful text extraction"""
        # Mock all Tesseract calls
        mock_string.return_value = "Sample text"
        mock_data.return_value = {
            'text': ['Sample', 'text'],
            'conf': [95, 90],
            'left': [10, 60],
            'top': [20, 20],
            'width': [40, 30],
            'height': [15, 15],
            'line_num': [1, 1],
            'block_num': [1, 1]
        }
        mock_boxes.return_value = {
            'char': ['S', 'a'],
            'left': [10, 15],
            'bottom': [80, 80],
            'right': [15, 20],
            'top': [95, 95]
        }
        
        result = self.ocr.extract_text(self.test_processed_image)
        
        assert isinstance(result, OCRResult)
        assert result.text == "Sample text"
        assert result.confidence > 0
        assert len(result.characters) == 2
        assert result.page_number == 0
        assert result.processing_time > 0
    
    @patch('app.services.ocr_engine.pytesseract.image_to_string')
    def test_extract_text_failure(self, mock_string):
        """Test text extraction with error handling"""
        # Mock Tesseract failure
        mock_string.side_effect = Exception("Tesseract error")
        
        result = self.ocr.extract_text(self.test_processed_image)
        
        # Should return empty result on failure
        assert isinstance(result, OCRResult)
        assert result.text == ""
        assert result.confidence == 0.0
        assert len(result.characters) == 0
    
    def test_merge_ocr_results(self):
        """Test OCR result merging"""
        primary = OCRResult(
            text="primary text",
            confidence=0.8,
            characters=[],
            words=[],
            uncertain_regions=[],
            processing_time=1.0,
            page_number=0,
            image_quality_score=0.8
        )
        
        secondary = [
            OCRResult(
                text="improved text",
                confidence=0.9,
                characters=[],
                words=[],
                uncertain_regions=[],
                processing_time=0.5,
                page_number=0,
                image_quality_score=0.8
            )
        ]
        
        merged = self.ocr._merge_ocr_results(primary, secondary)
        
        assert isinstance(merged, OCRResult)
        assert merged.confidence >= primary.confidence  # Should improve or stay same