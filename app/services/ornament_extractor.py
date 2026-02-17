"""
Ornament Extractor Service
==========================

Detects and extracts non-text ornamental elements (woodcuts, devices, decorative initials)
from page images using computer vision techniques.

Key features:
- Binarization and morphological operations to merge text blocks vs solid ornaments
- Contour detection with area filtering
- Aspect ratio and density analysis to distinguish text paragraphs from woodblocks
- High-resolution cropping

Usage:
    extractor = OrnamentExtractor()
    candidates = extractor.extract_from_page("path/to/page.jpg")
    extractor.save_candidates(candidates, "output/dir")
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import json
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class OrnamentCandidate:
    """A detected potential ornament."""
    source_path: str
    page_number: int  # Derived from filename if possible
    x: int
    y: int
    w: int
    h: int
    area: int
    aspect_ratio: float
    density: float  # Ink density (0-1)
    confidence: float # Heuristic score (0-1)
    type_guess: str = "unknown" # 'device', 'initial', 'border'
    crop_image: Optional[np.ndarray] = None # Not serialized

    def to_dict(self):
        d = asdict(self)
        if 'crop_image' in d:
            del d['crop_image']
        return d

class OrnamentExtractor:
    
    # Configuration for detection
    MIN_AREA = 5000       # Minimum pixel area (tuned for 2000px+ images)
    MAX_ASPECT = 4.0      # Skip very wide/tall strips (likely text lines)
    DENSITY_THRESH = 0.15 # Minimum ink density
    
    def __init__(self, debug_output: Optional[Path] = None):
        self.debug_output = debug_output
        if debug_output:
            debug_output.mkdir(parents=True, exist_ok=True)

    def extract_from_page(self, image_path: Path) -> List[OrnamentCandidate]:
        """
        Extract ornament candidates from a single page image.
        """
        path_str = str(image_path)
        img = cv2.imread(path_str)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return []
            
        h, w = img.shape[:2]
        if h < 500 or w < 500:
            logger.warning(f"Image too small for reliable extraction: {h}x{w} {image_path}")
            return []

        # 1. Preprocess
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binarize (Otsu)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. Morphological closing to merge text lines into blocks
        # We want text paragraphs to become solid blobs, which we then ignore if they look like text
        # Ornaments usually have different texture/density.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10)) # Wider to merge words
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 3. Find Contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        page_num = self._parse_page_number(image_path.name)
        
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            
            # 4. Filters
            if area < self.MIN_AREA:
                continue
            
            aspect = cw / ch
            if aspect > self.MAX_ASPECT or aspect < (1/self.MAX_ASPECT):
                continue
            
            # Ink density in original binary (not morphed)
            roi = binary[y:y+ch, x:x+cw]
            density = cv2.countNonZero(roi) / area
            
            if density < self.DENSITY_THRESH:
                continue
                
            # Heuristic Classification
            # Very simplistic for now
            confidence = 0.5
            guess = "unknown"
            
            # If fairly square and dense, maybe device or initial
            if 0.5 < aspect < 2.0 and density > 0.3:
                guess = "device/initial"
                confidence = 0.8
            # If wide and short, maybe headpiece/tailpiece
            elif aspect > 2.0:
                guess = "headpiece"
                confidence = 0.7
                
            # Crop
            # Add padding
            pad = 10
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(w, x+cw+pad), min(h, y+ch+pad)
            crop = img[y1:y2, x1:x2]
            
            cand = OrnamentCandidate(
                source_path=path_str,
                page_number=page_num,
                x=x, y=y, w=cw, h=ch,
                area=area,
                aspect_ratio=aspect,
                density=density,
                confidence=confidence,
                type_guess=guess,
                crop_image=crop
            )
            candidates.append(cand)
            
        logger.info(f"Detected {len(candidates)} candidates in {image_path.name}")
        return candidates

    def _parse_page_number(self, filename: str) -> int:
        """Attempt to extract page number from filename."""
        try:
            # Handle 000_0001.jpg -> 1
            # Handle 001_.jpg -> 1
            # Handle 00001.jpg -> 1
            # Just find first sequence of digits?
            import re
            nums = re.findall(r'\d+', filename)
            if nums:
                # Use the last number explicitly for HAB/BSB style '000_0001'
                # But for '001_page.jpg' logic?
                # BSB: 000_0001.jpg -> 1. 000 is likely batch/volume or dummy.
                # GDZ: 001_.jpg -> 1.
                # HAB: 00001.jpg -> 1.
                # Heuristic: verify if it looks like a page number (1-1000)
                # If multiple groups, maybe pick the one that increments
                
                # Default: use the last numeric group found
                return int(nums[-1])
        except:
            pass
        return 0

    def save_candidates(self, candidates: List[OrnamentCandidate], output_dir: Path):
        """Save candidate crops to disk with metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        meta_file = output_dir / "candidates.jsonl"
        
        with open(meta_file, 'a') as f:
            for cand in candidates:
                # Hash for unique filename
                h = hashlib.md5(f"{cand.source_path}_{cand.x}_{cand.y}".encode()).hexdigest()[:10]
                fname = f"orn_{cand.page_number:03d}_{h}.jpg"
                
                if cand.crop_image is not None:
                    cv2.imwrite(str(images_dir / fname), cand.crop_image)
                    
                meta = cand.to_dict()
                meta['filename'] = fname
                f.write(json.dumps(meta) + "\n")

    def run_batch(self, input_dir: Path, output_dir: Path, pattern: str = "*.jpg"):
        """Run extraction on a directory of images."""
        files = sorted(list(Path(input_dir).glob(pattern)))
        logger.info(f"Processing {len(files)} images in {input_dir}")
        
        for p in files:
            cands = self.extract_from_page(p)
            if cands:
                self.save_candidates(cands, output_dir)
