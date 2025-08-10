"""
Mock OCR engine for testing purposes
"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CharacterBox:
    """Mock CharacterBox for testing"""
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
class OCRResult:
    """Mock OCRResult for testing"""
    text: str
    confidence: float
    characters: List[CharacterBox]
    words: List
    uncertain_regions: List
    processing_time: float
    page_number: int
    image_quality_score: float