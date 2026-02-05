"""
OCR Factory
===========
Factory for creating OCR engine instances with automatic fallback.
"""

import os
import logging
from typing import Optional, Dict, Any, List

from app.services.ocr_interface import BaseOCREngine, OCREngineType

logger = logging.getLogger(__name__)


def get_ocr_engine(
    engine_type: str = "auto",
    api_key: str = None,
    **kwargs
) -> BaseOCREngine:
    """
    Get an OCR engine instance.
    
    Args:
        engine_type: "gemini", "tesseract", or "auto"
        api_key: API key for Gemini (optional, uses env var if not provided)
        **kwargs: Additional engine-specific options
        
    Returns:
        Configured OCR engine instance
        
    Raises:
        ValueError: If requested engine is not available
        
    Examples:
        >>> engine = get_ocr_engine("gemini")
        >>> engine = get_ocr_engine("tesseract")
        >>> engine = get_ocr_engine("auto")  # Gemini if available, else Tesseract
    """
    engine_type = engine_type.lower()
    
    if engine_type == "gemini":
        from app.services.gemini_engine import GeminiEngine
        engine = GeminiEngine(api_key=api_key, **kwargs)
        if not engine.is_available():
            raise ValueError("Gemini engine not available. Set GEMINI_API_KEY.")
        return engine
    
    elif engine_type == "tesseract":
        from app.services.tesseract_engine import TesseractEngine
        engine = TesseractEngine(**kwargs)
        if not engine.is_available():
            raise ValueError("Tesseract not installed or not accessible.")
        return engine
    
    elif engine_type == "auto":
        # Try Gemini first, fall back to Tesseract
        try:
            from app.services.gemini_engine import GeminiEngine
            gemini = GeminiEngine(api_key=api_key, **kwargs)
            if gemini.is_available():
                logger.info("Using Gemini Vision OCR (auto-selected)")
                return gemini
        except Exception as e:
            logger.debug(f"Gemini not available: {e}")
        
        try:
            from app.services.tesseract_engine import TesseractEngine
            tesseract = TesseractEngine(**kwargs)
            if tesseract.is_available():
                logger.info("Using Tesseract OCR (auto-fallback)")
                return tesseract
        except Exception as e:
            logger.debug(f"Tesseract not available: {e}")
        
        raise ValueError("No OCR engines available. Install Tesseract or set GEMINI_API_KEY.")
    
    else:
        raise ValueError(f"Unknown engine type: {engine_type}. Use 'gemini', 'tesseract', or 'auto'.")


def list_available_engines() -> List[Dict[str, Any]]:
    """
    List all available OCR engines and their status.
    
    Returns:
        List of engine info dictionaries
    """
    engines = []
    
    # Check Gemini
    try:
        from app.services.gemini_engine import GeminiEngine
        gemini = GeminiEngine()
        engines.append(gemini.get_info())
    except Exception as e:
        engines.append({
            "name": "gemini",
            "available": False,
            "error": str(e)
        })
    
    # Check Tesseract
    try:
        from app.services.tesseract_engine import TesseractEngine
        tesseract = TesseractEngine()
        engines.append(tesseract.get_info())
    except Exception as e:
        engines.append({
            "name": "tesseract",
            "available": False,
            "error": str(e)
        })
    
    return engines


def get_best_engine_for_historical(api_key: str = None) -> BaseOCREngine:
    """
    Get the best available engine for historical document OCR.
    
    Prefers Gemini for its Long-s and ligature support.
    
    Args:
        api_key: Optional Gemini API key
        
    Returns:
        Best available OCR engine
    """
    try:
        from app.services.gemini_engine import GeminiEngine
        gemini = GeminiEngine(api_key=api_key)
        if gemini.is_available():
            return gemini
    except Exception:
        pass
    
    # Fallback
    return get_ocr_engine("tesseract")


# Convenience exports
__all__ = [
    "get_ocr_engine",
    "list_available_engines", 
    "get_best_engine_for_historical",
]
