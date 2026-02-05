"""
OCR Engine Interface - Unified OCR Backend Abstraction
=======================================================
Provides a unified interface for multiple OCR backends:
- Tesseract (traditional OCR)
- Gemini Vision (VLM-based OCR)
- Legacy AdvancedOCR (existing implementation)

This module enables hot-swapping OCR engines with consistent output format.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Protocol
from pathlib import Path
from PIL import Image
from enum import Enum
import time


class OCREngineType(Enum):
    """Available OCR engine types."""
    TESSERACT = "tesseract"
    GEMINI = "gemini"
    LEGACY = "legacy"  # Existing AdvancedOCR
    AUTO = "auto"


@dataclass
class OCRCharacter:
    """
    Character-level OCR data with bounding box.
    
    Coordinates are in pixels relative to the source image.
    """
    character: str
    x: float
    y: float
    width: float
    height: float
    confidence: float = 0.0
    
    # Historical typography features
    is_long_s: bool = False
    is_ligature: bool = False
    ligature_type: Optional[str] = None  # "ct", "st", "ff", "fi", "fl"
    
    # Layout IDs
    block_id: int = 0
    line_id: int = 0
    word_id: int = 0
    
    # Page reference
    page_number: int = 1
    
    @property
    def right(self) -> float:
        """Right edge (x + width)."""
        return self.x + self.width
    
    @property
    def bottom(self) -> float:
        """Bottom edge (y + height)."""
        return self.y + self.height
    
    @property
    def center(self) -> tuple:
        """Center point (x, y)."""
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    @property
    def area(self) -> float:
        """Bounding box area."""
        return max(0, self.width * self.height)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "character": self.character,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "is_long_s": self.is_long_s,
            "is_ligature": self.is_ligature,
            "ligature_type": self.ligature_type,
            "page_number": self.page_number,
        }


@dataclass
class LigatureStats:
    """Statistics for a single ligature type."""
    ligature_type: str  # "ct", "st", "ff", etc.
    count: int = 0
    sample_indices: List[int] = field(default_factory=list)  # Indices in character list


@dataclass
class OCRPageResult:
    """
    Standardized result from any OCR engine for a single page.
    
    This is the common output format regardless of which backend is used.
    """
    # Core text output
    text: str
    
    # Character-level data (optional - some engines don't provide this)
    characters: List[OCRCharacter] = field(default_factory=list)
    
    # Confidence metrics
    average_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    
    # Historical typography detection
    long_s_count: int = 0
    long_s_indices: List[int] = field(default_factory=list)
    
    # Ligature detection
    ligatures: Dict[str, LigatureStats] = field(default_factory=dict)
    total_ligatures: int = 0
    
    # Anomalies and warnings
    anomalies: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Engine metadata
    engine_name: str = "unknown"
    engine_version: str = ""
    model_name: str = ""  # For VLMs
    
    # Performance
    processing_time_ms: float = 0.0
    
    # Page info
    page_number: int = 1
    image_width: int = 0
    image_height: int = 0
    
    # Raw data (for debugging)
    raw_response: Optional[Any] = None
    
    @property
    def character_count(self) -> int:
        """Total characters detected."""
        return len(self.characters)
    
    @property
    def unique_characters(self) -> set:
        """Set of unique characters."""
        return set(c.character for c in self.characters)
    
    @property
    def has_character_boxes(self) -> bool:
        """Whether character-level boxes are available."""
        return len(self.characters) > 0
    
    def get_ligature_count(self, ligature_type: str) -> int:
        """Get count for specific ligature type."""
        return self.ligatures.get(ligature_type, LigatureStats(ligature_type)).count
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "character_count": self.character_count,
            "average_confidence": self.average_confidence,
            "long_s_count": self.long_s_count,
            "ligatures": {k: {"count": v.count} for k, v in self.ligatures.items()},
            "total_ligatures": self.total_ligatures,
            "engine_name": self.engine_name,
            "model_name": self.model_name,
            "processing_time_ms": self.processing_time_ms,
            "page_number": self.page_number,
        }


class BaseOCREngine(ABC):
    """
    Abstract base class for OCR engines.
    
    All OCR implementations must inherit from this and implement:
    - analyze_page(): Main OCR method
    - is_available(): Check if engine can be used
    """
    
    # Engine metadata (override in subclasses)
    name: str = "base"
    version: str = "0.0.0"
    engine_type: OCREngineType = OCREngineType.TESSERACT
    
    # Feature flags
    supports_long_s: bool = False
    supports_ligatures: bool = False
    supports_character_boxes: bool = False
    requires_api_key: bool = False
    
    @abstractmethod
    def analyze_page(self, image: Image.Image, page_number: int = 1) -> OCRPageResult:
        """
        Analyze a page image and extract text with metadata.
        
        Args:
            image: PIL Image of the page
            page_number: Page number for reference
            
        Returns:
            OCRPageResult with text, character data, and statistics
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this engine is available and ready to use.
        
        Returns:
            True if engine can process pages
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get engine metadata and capabilities."""
        return {
            "name": self.name,
            "version": self.version,
            "type": self.engine_type.value,
            "supports_long_s": self.supports_long_s,
            "supports_ligatures": self.supports_ligatures,
            "supports_character_boxes": self.supports_character_boxes,
            "requires_api_key": self.requires_api_key,
            "available": self.is_available(),
        }
    
    def __repr__(self) -> str:
        status = "✓" if self.is_available() else "✗"
        return f"<{self.__class__.__name__} [{status}] {self.name} v{self.version}>"


class OCREngineProtocol(Protocol):
    """
    Protocol for type checking OCR engines.
    
    Use this for type hints when you want to accept any OCR engine.
    """
    
    name: str
    version: str
    
    def analyze_page(self, image: Image.Image, page_number: int = 1) -> OCRPageResult:
        ...
    
    def is_available(self) -> bool:
        ...
