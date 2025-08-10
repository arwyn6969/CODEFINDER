"""
Advanced OCR Engine for Ancient Text Analyzer
Specialized for historical documents with ornate fonts and decorative elements
"""
import pytesseract
from PIL import Image
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import time

from app.services.pdf_processor import ProcessedImage
from app.core.config import settings

logger = logging.getLogger(__name__)

class OCRMode(Enum):
    """OCR processing modes"""
    SINGLE_BLOCK = 6  # Single uniform block of text
    SINGLE_COLUMN = 7  # Single text column
    SINGLE_WORD = 8   # Single word
    CIRCLE_WORD = 9   # Single word in a circle
    SINGLE_CHAR = 10  # Single character

@dataclass
class CharacterBox:
    """Container for character-level OCR data"""
    character: str
    x: float
    y: float
    width: float
    height: float
    confidence: float
    page_number: int
    word_id: int
    line_id: int
    block_id: int
    font_size: Optional[float] = None
    is_bold: bool = False
    is_italic: bool = False

@dataclass
class WordBox:
    """Container for word-level OCR data"""
    text: str
    x: float
    y: float
    width: float
    height: float
    confidence: float
    characters: List[CharacterBox]
    page_number: int
    line_id: int
    block_id: int

@dataclass
class UncertainRegion:
    """Container for uncertain OCR regions requiring manual review"""
    x: float
    y: float
    width: float
    height: float
    confidence: float
    suggested_text: str
    alternatives: List[str]
    reason: str
    page_number: int

@dataclass
class OCRResult:
    """Container for complete OCR results"""
    text: str
    confidence: float
    characters: List[CharacterBox]
    words: List[WordBox]
    uncertain_regions: List[UncertainRegion]
    processing_time: float
    page_number: int
    image_quality_score: float

class AdvancedOCR:
    """
    Advanced OCR engine with multi-pass processing and confidence scoring
    Optimized for historical documents and ornate fonts
    """
    
    def __init__(self, confidence_threshold: float = None):
        self.confidence_threshold = confidence_threshold or settings.ocr_confidence_threshold
        self.tesseract_config = self._get_tesseract_config()
        
    def _get_tesseract_config(self) -> str:
        """
        Get optimized Tesseract configuration for historical documents
        """
        # Simplified configuration to avoid quote issues
        config = '--oem 3 --psm 6'
        return config
    
    def extract_text(self, processed_image: ProcessedImage) -> OCRResult:
        """
        Extract text with high accuracy using multi-pass OCR
        
        Args:
            processed_image: Preprocessed image ready for OCR
            
        Returns:
            OCRResult with comprehensive text extraction data
        """
        logger.info(f"Starting OCR extraction for page {processed_image.original.page_number}")
        start_time = time.time()
        
        try:
            # Convert PIL image to format suitable for Tesseract
            cv_image = self._pil_to_cv2(processed_image.image)
            
            # Multi-pass OCR for better accuracy
            primary_result = self._primary_ocr_pass(cv_image, processed_image.original.page_number)
            
            # Secondary pass for uncertain regions
            if primary_result.uncertain_regions:
                secondary_results = self._secondary_ocr_pass(cv_image, primary_result.uncertain_regions, processed_image.original.page_number)
                primary_result = self._merge_ocr_results(primary_result, secondary_results)
            
            # Character-level analysis
            characters = self.get_character_positions(processed_image)
            
            # Update result with character data
            primary_result.characters = characters
            primary_result.processing_time = time.time() - start_time
            primary_result.image_quality_score = processed_image.quality_score
            
            logger.info(f"OCR completed in {primary_result.processing_time:.2f}s, confidence: {primary_result.confidence:.2f}")
            
            return primary_result
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            # Return empty result on failure
            return OCRResult(
                text="",
                confidence=0.0,
                characters=[],
                words=[],
                uncertain_regions=[],
                processing_time=time.time() - start_time,
                page_number=processed_image.original.page_number,
                image_quality_score=processed_image.quality_score
            )
    
    def _primary_ocr_pass(self, image: np.ndarray, page_number: int) -> OCRResult:
        """
        Primary OCR pass with standard configuration
        """
        try:
            # Extract text with detailed data
            data = pytesseract.image_to_data(
                image, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract plain text
            text = pytesseract.image_to_string(image, config=self.tesseract_config)
            
            # Process OCR data
            words = self._process_word_data(data, page_number)
            uncertain_regions = self._identify_uncertain_regions(data, page_number)
            
            # Calculate overall confidence
            confidences = [word.confidence for word in words if word.confidence > 0]
            overall_confidence = np.mean(confidences) if confidences else 0.0
            
            return OCRResult(
                text=text.strip(),
                confidence=overall_confidence,
                characters=[],  # Will be filled later
                words=words,
                uncertain_regions=uncertain_regions,
                processing_time=0.0,  # Will be set later
                page_number=page_number,
                image_quality_score=0.0  # Will be set later
            )
            
        except Exception as e:
            logger.error(f"Primary OCR pass failed: {e}")
            raise
    
    def _secondary_ocr_pass(self, image: np.ndarray, uncertain_regions: List[UncertainRegion], page_number: int) -> List[OCRResult]:
        """
        Secondary OCR pass for uncertain regions with different configurations
        """
        results = []
        
        for region in uncertain_regions:
            try:
                # Extract region from image
                x, y, w, h = int(region.x), int(region.y), int(region.width), int(region.height)
                roi = image[y:y+h, x:x+w]
                
                if roi.size == 0:
                    continue
                
                # Try different OCR modes for this region
                configs = [
                    '--oem 3 --psm 8',  # Single word
                    '--oem 3 --psm 7',  # Single text line
                    '--oem 3 --psm 13', # Raw line
                ]
                
                best_result = None
                best_confidence = 0.0
                
                for config in configs:
                    try:
                        text = pytesseract.image_to_string(roi, config=config).strip()
                        data = pytesseract.image_to_data(roi, config=config, output_type=pytesseract.Output.DICT)
                        
                        # Calculate confidence for this attempt
                        confidences = [conf for conf in data['conf'] if conf > 0]
                        avg_confidence = np.mean(confidences) if confidences else 0.0
                        
                        if avg_confidence > best_confidence and text:
                            best_confidence = avg_confidence
                            best_result = text
                            
                    except Exception as e:
                        logger.warning(f"Secondary OCR attempt failed: {e}")
                        continue
                
                if best_result and best_confidence > region.confidence:
                    # Create improved result
                    improved_result = OCRResult(
                        text=best_result,
                        confidence=best_confidence,
                        characters=[],
                        words=[],
                        uncertain_regions=[],
                        processing_time=0.0,
                        page_number=page_number,
                        image_quality_score=0.0
                    )
                    results.append(improved_result)
                    
            except Exception as e:
                logger.warning(f"Secondary OCR for region failed: {e}")
                continue
        
        return results
    
    def _process_word_data(self, data: Dict[str, List], page_number: int) -> List[WordBox]:
        """
        Process Tesseract word-level data into WordBox objects
        """
        words = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if not text:
                continue
            
            confidence = float(data['conf'][i])
            if confidence < 0:  # Invalid confidence
                confidence = 0.0
            
            word = WordBox(
                text=text,
                x=float(data['left'][i]),
                y=float(data['top'][i]),
                width=float(data['width'][i]),
                height=float(data['height'][i]),
                confidence=confidence,
                characters=[],  # Will be populated separately
                page_number=page_number,
                line_id=data['line_num'][i],
                block_id=data['block_num'][i]
            )
            
            words.append(word)
        
        return words
    
    def _identify_uncertain_regions(self, data: Dict[str, List], page_number: int) -> List[UncertainRegion]:
        """
        Identify regions with low confidence that need manual review
        """
        uncertain_regions = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            confidence = float(data['conf'][i])
            
            if confidence > 0 and confidence < self.confidence_threshold * 100:  # Tesseract uses 0-100 scale
                region = UncertainRegion(
                    x=float(data['left'][i]),
                    y=float(data['top'][i]),
                    width=float(data['width'][i]),
                    height=float(data['height'][i]),
                    confidence=confidence / 100.0,  # Normalize to 0-1
                    suggested_text=text,
                    alternatives=[],  # Could be populated with alternative suggestions
                    reason=f"Low confidence: {confidence:.1f}%",
                    page_number=page_number
                )
                uncertain_regions.append(region)
        
        return uncertain_regions
    
    def get_character_positions(self, processed_image: ProcessedImage) -> List[CharacterBox]:
        """
        Extract character-level positioning and sizing data
        
        Args:
            processed_image: Preprocessed image
            
        Returns:
            List of CharacterBox objects with detailed character data
        """
        try:
            cv_image = self._pil_to_cv2(processed_image.image)
            
            # Get character-level data from Tesseract
            data = pytesseract.image_to_boxes(
                cv_image,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            characters = []
            page_number = processed_image.original.page_number
            
            for i in range(len(data['char'])):
                char = data['char'][i]
                
                # Tesseract boxes use bottom-left origin, convert to top-left
                x = float(data['left'][i])
                y = float(cv_image.shape[0] - data['top'][i])  # Flip Y coordinate
                width = float(data['right'][i] - data['left'][i])
                height = float(data['top'][i] - data['bottom'][i])
                
                # Estimate confidence (boxes don't include confidence, use default)
                confidence = 0.8  # Default confidence for character boxes
                
                character = CharacterBox(
                    character=char,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=confidence,
                    page_number=page_number,
                    word_id=0,  # Would need additional processing to determine
                    line_id=0,  # Would need additional processing to determine
                    block_id=0, # Would need additional processing to determine
                    font_size=height  # Approximate font size from character height
                )
                
                characters.append(character)
            
            return characters
            
        except Exception as e:
            logger.error(f"Character position extraction failed: {e}")
            return []
    
    def measure_character_sizes(self, characters: List[CharacterBox]) -> Dict[str, Any]:
        """
        Analyze character sizes for variation detection
        
        Args:
            characters: List of character boxes
            
        Returns:
            Dictionary with size analysis results
        """
        if not characters:
            return {"error": "No characters provided"}
        
        try:
            # Group characters by type
            char_groups = {}
            for char in characters:
                if char.character not in char_groups:
                    char_groups[char.character] = []
                char_groups[char.character].append(char)
            
            # Analyze size variations
            size_analysis = {}
            
            for char_type, char_list in char_groups.items():
                heights = [c.height for c in char_list]
                widths = [c.width for c in char_list]
                
                size_analysis[char_type] = {
                    "count": len(char_list),
                    "height_mean": np.mean(heights),
                    "height_std": np.std(heights),
                    "height_min": np.min(heights),
                    "height_max": np.max(heights),
                    "width_mean": np.mean(widths),
                    "width_std": np.std(widths),
                    "width_min": np.min(widths),
                    "width_max": np.max(widths),
                    "size_variation_coefficient": np.std(heights) / np.mean(heights) if np.mean(heights) > 0 else 0
                }
            
            # Overall statistics
            all_heights = [c.height for c in characters]
            all_widths = [c.width for c in characters]
            
            overall_stats = {
                "total_characters": len(characters),
                "average_height": np.mean(all_heights),
                "average_width": np.mean(all_widths),
                "height_variation": np.std(all_heights),
                "width_variation": np.std(all_widths),
                "character_types": len(char_groups)
            }
            
            return {
                "character_analysis": size_analysis,
                "overall_statistics": overall_stats
            }
            
        except Exception as e:
            logger.error(f"Character size analysis failed: {e}")
            return {"error": str(e)}
    
    def flag_uncertainties(self, result: OCRResult) -> List[UncertainRegion]:
        """
        Additional uncertainty flagging based on text analysis
        
        Args:
            result: OCR result to analyze
            
        Returns:
            Updated list of uncertain regions
        """
        additional_uncertainties = []
        
        try:
            # Check for unusual character sequences
            unusual_patterns = [
                r'[A-Z]{5,}',  # Long sequences of capitals
                r'[0-9]{4,}',  # Long number sequences
                r'[^\w\s]{3,}', # Multiple special characters
                r'\b\w{15,}\b'  # Very long words
            ]
            
            for pattern in unusual_patterns:
                matches = re.finditer(pattern, result.text)
                for match in matches:
                    # Find corresponding word boxes
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # This is a simplified approach - in practice, you'd need
                    # to map text positions back to image coordinates
                    uncertainty = UncertainRegion(
                        x=0.0,  # Would need proper coordinate mapping
                        y=0.0,
                        width=0.0,
                        height=0.0,
                        confidence=0.5,  # Medium uncertainty
                        suggested_text=match.group(),
                        alternatives=[],
                        reason=f"Unusual pattern: {pattern}",
                        page_number=result.page_number
                    )
                    additional_uncertainties.append(uncertainty)
            
            # Combine with existing uncertainties
            all_uncertainties = result.uncertain_regions + additional_uncertainties
            
            return all_uncertainties
            
        except Exception as e:
            logger.error(f"Uncertainty flagging failed: {e}")
            return result.uncertain_regions
    
    def _merge_ocr_results(self, primary: OCRResult, secondary: List[OCRResult]) -> OCRResult:
        """
        Merge primary and secondary OCR results
        """
        # This is a simplified merge - in practice, you'd need sophisticated
        # logic to replace uncertain regions with better results
        merged_text = primary.text
        
        # Update confidence if secondary results are better
        if secondary:
            secondary_confidences = [r.confidence for r in secondary]
            if secondary_confidences:
                avg_secondary_confidence = np.mean(secondary_confidences)
                primary.confidence = max(primary.confidence, avg_secondary_confidence)
        
        return primary
    
    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL image to OpenCV format"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)