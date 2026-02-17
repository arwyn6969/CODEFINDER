# app/services/flash_verifier.py
import os
import time
import json
import logging
import typing
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import google.generativeai as genai
from PIL import Image

logger = logging.getLogger(__name__)

class FlashVerifier:
    """
    Semantic anomaly verification using Gemini 1.5 Flash.
    Filters OpenCV candidates by distinguishing true anomalies from
    binding marks, paper damage, and printing artifacts.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def verify_candidate(self, 
                         image_path: str, 
                         anomaly_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify a single anomaly candidate.
        
        Args:
            image_path: Path to the character image
            anomaly_data: Dictionary containing anomaly details (score, type)
            
        Returns:
            Verification result dictionary
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Construct prompt
            prompt = self._build_prompt(anomaly_data)
            
            # Call Gemini
            response = self.model.generate_content([prompt, img])
            
            # Parse response
            result = self._parse_response(response.text)
            
            # Add metadata
            result['original_score'] = anomaly_data.get('score', 0)
            result['image_path'] = image_path
            
            return result
            
        except Exception as e:
            logger.error(f"Verification failed for {image_path}: {e}")
            return {
                "classification": "error",
                "is_anomaly": False,
                "confidence": 0.0,
                "reasoning": str(e)
            }

    def verify_batch(self, 
                     candidates: List[Dict[str, Any]], 
                     output_path: str = None) -> List[Dict[str, Any]]:
        """Verify a batch of candidates."""
        verified = []
        
        print(f"Verifying {len(candidates)} candidates with Gemini Flash...")
        
        for i, cand in enumerate(candidates):
            path = cand.get('path')
            if not path or not Path(path).exists():
                logger.warning(f"Image not found: {path}")
                continue
                
            print(f"[{i+1}/{len(candidates)}] Verifying {Path(path).name}...")
            
            result = self.verify_candidate(path, cand)
            verified.append(result)
            
            # Rate limit handling (Flash is high throughput but safe to pause slightly)
            time.sleep(1.0) 
            
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(verified, f, indent=2)
                
        return verified

    def _build_prompt(self, data: Dict[str, Any]) -> str:
        return """
        Analyze this image of a printed character from a 1609 document.
        Focus on the 'irregularities' or 'dots' detected in the ink.

        CLASSIFY the anomalies seen in this character image:
        1. "Printing Artifact": Ink splatter, over-inking, squeeze (common, ignore)
        2. "Binding/Paper": Binding shadows, paper holes, foxing, bleed-through (ignore)
        3. "Hand Interference": Pen marks, deliberate modifications (KEEP)
        4. "Interesting Anomaly": Unexplained meaningful irregularity (KEEP)

        Assess the likelihood that this is NOT just a printing mess-up or paper damage.

        Output purely a JSON object with this structure:
        {
            "classification": "one of the 4 categories above",
            "is_anomaly": boolean (true only for category 3 or 4),
            "confidence": float (0.0 to 1.0),
            "reasoning": "short explanation"
        }
        """

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Clean and parse JSON from model response."""
        try:
            # Strip markdown code blocks if present
            cleaned = text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {text[:100]}...")
            return {
                "classification": "unknown",
                "is_anomaly": False,
                "confidence": 0.0,
                "reasoning": "Response parsing failed"
            }
