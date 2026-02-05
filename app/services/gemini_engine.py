"""
Gemini Vision OCR Engine
========================
VLM-based OCR using Google Gemini for historical document analysis.

Advantages over Tesseract:
- Native Long-s (ſ) recognition
- Ligature detection (ct, st, ff, fi, fl)
- Semantic understanding of historical text
- Higher confidence on period typography
"""

import os
import json
import base64
import logging
import time
import re
from typing import List, Optional, Dict, Any
from pathlib import Path
from PIL import Image

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from app.services.ocr_interface import (
    BaseOCREngine,
    OCRPageResult,
    OCRCharacter,
    LigatureStats,
    OCREngineType
)

logger = logging.getLogger(__name__)


# Historical document OCR prompt
HISTORICAL_OCR_PROMPT = """You are an expert paleographer analyzing a page from Shakespeare's Sonnets (1609 Quarto).

Analyze this page and provide a JSON response with:

1. **transcription**: Transcribe exactly as printed:
   - Preserve Long-s (ſ) — do NOT modernize to 's'
   - Mark ligatures with [lig:XX] (e.g., [lig:ct], [lig:st], [lig:ff])
   - Keep original spelling and punctuation
   - Preserve line breaks with \\n

2. **long_s_count**: Number of Long-s (ſ) characters

3. **ligatures**: Object with counts for each type:
   {"ct": N, "st": N, "ff": N, "fi": N, "fl": N}

4. **confidence**: Your confidence in the transcription (0-100)

5. **anomalies**: Array of unusual marks, marginalia, or artifacts

Respond ONLY with valid JSON:
```json
{
  "transcription": "...",
  "long_s_count": N,
  "ligatures": {...},
  "confidence": N,
  "anomalies": [...]
}
```"""


class GeminiEngine(BaseOCREngine):
    """
    Gemini Vision OCR engine for historical documents.
    
    Features:
    - Native Long-s (ſ) detection
    - Ligature preservation
    - High accuracy on 17th-century typography
    
    Requirements:
    - GEMINI_API_KEY environment variable
    - google-generativeai package
    """
    
    name = "gemini"
    version = "2.5"
    engine_type = OCREngineType.GEMINI
    
    # Feature flags
    supports_long_s = True
    supports_ligatures = True
    supports_character_boxes = False  # VLMs don't provide pixel-level boxes
    requires_api_key = True
    
    # Rate limiting
    DEFAULT_RATE_LIMIT = 15  # requests per minute
    
    def __init__(
        self, 
        api_key: str = None, 
        model: str = "gemini-2.5-flash",
        rate_limit: int = None
    ):
        """
        Initialize Gemini engine.
        
        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
            model: Model name (gemini-2.5-flash, gemini-2.5-pro, etc.)
            rate_limit: Max requests per minute
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model
        self.rate_limit = rate_limit or self.DEFAULT_RATE_LIMIT
        self._model = None
        self._last_request_time = 0
    
    def is_available(self) -> bool:
        """Check if Gemini API is available."""
        if not GEMINI_AVAILABLE:
            logger.warning("google-generativeai package not installed")
            return False
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set")
            return False
        return True
    
    def _get_model(self):
        """Get or create Gemini model instance."""
        if self._model is None:
            if not self.is_available():
                raise RuntimeError("Gemini API not available")
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
        return self._model
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _rate_limit_wait(self):
        """Wait if needed to respect rate limit."""
        if self.rate_limit <= 0:
            return
        min_interval = 60.0 / self.rate_limit
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from Gemini response."""
        # Try to extract JSON from markdown code block
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        elif "{" in response_text:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]
        else:
            raise ValueError("No JSON found in response")
        
        return json.loads(json_str)
    
    def _extract_long_s_positions(self, text: str) -> List[int]:
        """Find positions of Long-s in transcription."""
        positions = []
        # Look for actual ſ character
        for i, char in enumerate(text):
            if char == 'ſ':
                positions.append(i)
        return positions
    
    def _parse_ligatures(self, ligatures_dict: dict) -> Dict[str, LigatureStats]:
        """Convert ligature counts to LigatureStats objects."""
        result = {}
        for lig_type, count in ligatures_dict.items():
            if count > 0:
                result[lig_type] = LigatureStats(
                    ligature_type=lig_type,
                    count=count
                )
        return result
    
    def analyze_page(self, image: Image.Image, page_number: int = 1) -> OCRPageResult:
        """
        Analyze page with Gemini Vision.
        
        Args:
            image: PIL Image of the page
            page_number: Page number for reference
            
        Returns:
            OCRPageResult with text, Long-s, and ligature detection
        """
        start_time = time.time()
        
        if not self.is_available():
            return OCRPageResult(
                text="",
                engine_name=self.name,
                model_name=self.model_name,
                warnings=["Gemini API not available"],
                page_number=page_number,
            )
        
        try:
            # Rate limiting
            self._rate_limit_wait()
            
            # Get model
            model = self._get_model()
            
            # Convert image
            image_data = self._image_to_base64(image)
            
            # Send to Gemini
            response = model.generate_content([
                HISTORICAL_OCR_PROMPT,
                {"mime_type": "image/jpeg", "data": image_data}
            ])
            
            # Parse response
            result_data = self._parse_response(response.text)
            
            # Extract data
            transcription = result_data.get("transcription", "")
            long_s_count = result_data.get("long_s_count", 0)
            ligatures_raw = result_data.get("ligatures", {})
            confidence = result_data.get("confidence", 0)
            anomalies = result_data.get("anomalies", [])
            
            # Process data
            long_s_positions = self._extract_long_s_positions(transcription)
            ligatures = self._parse_ligatures(ligatures_raw)
            total_ligatures = sum(lig.count for lig in ligatures.values())
            
            processing_time = (time.time() - start_time) * 1000
            
            return OCRPageResult(
                text=transcription,
                characters=[],  # VLMs don't provide character boxes
                average_confidence=float(confidence),
                long_s_count=long_s_count,
                long_s_indices=long_s_positions,
                ligatures=ligatures,
                total_ligatures=total_ligatures,
                anomalies=anomalies if isinstance(anomalies, list) else [str(anomalies)],
                engine_name=self.name,
                engine_version=self.version,
                model_name=self.model_name,
                processing_time_ms=processing_time,
                page_number=page_number,
                image_width=image.width,
                image_height=image.height,
                raw_response=response.text,
            )
            
        except Exception as e:
            logger.error(f"Gemini OCR failed: {e}")
            return OCRPageResult(
                text="",
                engine_name=self.name,
                model_name=self.model_name,
                warnings=[f"OCR failed: {str(e)}"],
                page_number=page_number,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
