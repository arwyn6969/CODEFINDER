"""
Tesseract OCR Engine
====================
Wrapper for Tesseract OCR with the unified OCR interface.
"""

import logging
import time
from typing import List, Optional
from PIL import Image

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from app.services.ocr_interface import (
    BaseOCREngine, 
    OCRPageResult, 
    OCRCharacter, 
    LigatureStats,
    OCREngineType
)

logger = logging.getLogger(__name__)


class TesseractEngine(BaseOCREngine):
    """
    Tesseract OCR engine implementation.
    
    Good for:
    - Modern printed text
    - Cost-effective processing
    - Offline operation
    
    Limitations:
    - Cannot detect Long-s (ſ) reliably
    - Does not preserve ligatures
    - Lower accuracy on historical documents
    """
    
    name = "tesseract"
    version = pytesseract.get_tesseract_version().public if TESSERACT_AVAILABLE else "0.0.0"
    engine_type = OCREngineType.TESSERACT
    
    # Feature flags
    supports_long_s = False  # Tesseract misses Long-s
    supports_ligatures = False  # Tesseract splits ligatures
    supports_character_boxes = True
    requires_api_key = False
    
    def __init__(self, config: str = None, language: str = "eng"):
        """
        Initialize Tesseract engine.
        
        Args:
            config: Tesseract config string (e.g., "--psm 6 --oem 3")
            language: Language code for OCR
        """
        self.config = config or "--oem 3 --psm 6 -c preserve_interword_spaces=1"
        self.language = language
    
    def is_available(self) -> bool:
        """Check if Tesseract is installed and accessible."""
        if not TESSERACT_AVAILABLE:
            return False
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    
    def analyze_page(self, image: Image.Image, page_number: int = 1) -> OCRPageResult:
        """
        Analyze page with Tesseract OCR.
        
        Args:
            image: PIL Image of the page
            page_number: Page number for reference
            
        Returns:
            OCRPageResult with text and character boxes
        """
        start_time = time.time()
        
        if not self.is_available():
            return OCRPageResult(
                text="",
                engine_name=self.name,
                warnings=["Tesseract not available"],
                page_number=page_number,
            )
        
        # Get OCR data with character-level info
        try:
            ocr_data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config=self.config,
                lang=self.language
            )
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return OCRPageResult(
                text="",
                engine_name=self.name,
                warnings=[f"OCR failed: {str(e)}"],
                page_number=page_number,
            )
        
        # Process OCR data
        characters: List[OCRCharacter] = []
        full_text_parts = []
        confidence_sum = 0.0
        confidence_count = 0
        
        current_block = -1
        current_line = -1
        current_word = -1
        block_id = 0
        line_id = 0
        word_id = 0
        
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            text = ocr_data['text'][i]
            
            # Skip empty
            if not text or text.isspace():
                continue
            
            # Track layout structure
            if ocr_data['block_num'][i] != current_block:
                current_block = ocr_data['block_num'][i]
                block_id += 1
            if ocr_data['line_num'][i] != current_line:
                current_line = ocr_data['line_num'][i]
                line_id += 1
            if ocr_data['word_num'][i] != current_word:
                current_word = ocr_data['word_num'][i]
                word_id += 1
            
            # Get bounding box
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]
            
            # Get confidence
            raw_conf = ocr_data['conf'][i]
            if raw_conf == '-1' or raw_conf == -1:
                conf = 0.0
            else:
                conf = float(raw_conf)
            
            # Process each character in the word
            if len(text) > 0:
                char_width = w / len(text) if len(text) > 0 else w
                
                for char_idx, char in enumerate(text):
                    if char.isspace():
                        continue
                    
                    char_x = x + (char_idx * char_width)
                    
                    characters.append(OCRCharacter(
                        character=char,
                        x=char_x,
                        y=y,
                        width=char_width,
                        height=h,
                        confidence=conf,
                        block_id=block_id,
                        line_id=line_id,
                        word_id=word_id,
                        page_number=page_number,
                    ))
                    
                    confidence_sum += conf
                    confidence_count += 1
                
                full_text_parts.append(text)
        
        # Calculate stats
        avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0.0
        
        processing_time = (time.time() - start_time) * 1000
        
        return OCRPageResult(
            text=" ".join(full_text_parts),
            characters=characters,
            average_confidence=avg_confidence,
            long_s_count=0,  # Tesseract doesn't detect Long-s
            ligatures={},    # Tesseract doesn't detect ligatures
            total_ligatures=0,
            engine_name=self.name,
            engine_version=str(self.version),
            processing_time_ms=processing_time,
            page_number=page_number,
            image_width=image.width,
            image_height=image.height,
        )
