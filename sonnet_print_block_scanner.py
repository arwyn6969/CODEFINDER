#!/usr/bin/env python3
"""
Sonnet Print Block Scanner
==========================
Exhaustive OCR scanner for the 1609 Shakespeare Sonnets Quarto.
Exhaustive OCR scanner for the 1609 Shakespeare Sonnets Quarto.
Extracts, catalogues, and images all print blocks (typographical characters).
    
**PROTOCOL**: Follows the `AXIOM_OF_INTENT.md` - No anomaly is discarded. 
Every mark is preserved as a high-fidelity data point.

Usage:
    python3 sonnet_print_block_scanner.py                    # Full scan
    python3 sonnet_print_block_scanner.py --pages 1-5        # Test subset
    python3 sonnet_print_block_scanner.py --pages 1-5 --test # Smoke test
"""

import sys
import os
import json
import csv
import argparse
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Core dependencies
try:
    import fitz  # PyMuPDF
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip3 install pymupdf pillow numpy")
    sys.exit(1)

# Optional: Tesseract OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. Using PyMuPDF text extraction only.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CharacterInstance:
    """Single occurrence of a character in the document."""
    character: str
    page_number: int
    x: float
    y: float
    width: float
    height: float
    confidence: float = 0.0
    block_id: int = 0
    line_id: int = 0
    word_id: int = 0
    is_ligature: bool = False
    is_long_s: bool = False
    is_anomaly: bool = False
    anomaly_type: Optional[str] = None
    image_path: Optional[str] = None


@dataclass  
class CharacterCatalogueEntry:
    """Catalogue entry for a unique character."""
    character: str
    unicode_name: str
    unicode_codepoint: str
    category: str  # uppercase, lowercase, digit, punctuation, special, ligature
    total_count: int = 0
    instances: List[CharacterInstance] = field(default_factory=list)
    sample_images: List[str] = field(default_factory=list)
    variants_detected: int = 0
    average_width: float = 0.0
    average_height: float = 0.0


@dataclass
class AnomalyEntry:
    """Detected anomaly in the print."""
    anomaly_type: str  # misprint, broken_type, ink_blot, show_through, variant
    page_number: int
    x: float
    y: float
    width: float
    height: float
    description: str
    severity: str  # low, medium, high
    image_path: Optional[str] = None
    related_character: Optional[str] = None


@dataclass
class ScanStatistics:
    """Overall scan statistics."""
    total_pages: int = 0
    pages_scanned: int = 0
    total_characters: int = 0
    unique_characters: int = 0
    total_words: int = 0
    total_lines: int = 0
    total_blocks: int = 0
    ligatures_found: int = 0
    long_s_count: int = 0
    anomalies_detected: int = 0
    scan_duration_seconds: float = 0.0
    average_confidence: float = 0.0


# ============================================================================
# MAIN SCANNER CLASS
# ============================================================================

class SonnetPrintBlockScanner:
    """
    Exhaustive OCR scanner for 1609 Shakespeare Sonnets.
    Extracts, catalogues, and images all print blocks.
    """
    
    # Known ligatures in historical printing
    LIGATURES = {'ff', 'fi', 'fl', 'ffi', 'ffl', 'st', 'ct', 'Th', 'th'}
    
    # Long-s character (ſ) and common confusions
    LONG_S = 'ſ'
    LONG_S_CODEPOINT = 'U+017F'
    
    # Character categories
    CATEGORIES = {
        'uppercase': set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
        'lowercase': set('abcdefghijklmnopqrstuvwxyz'),
        'digit': set('0123456789'),
        'punctuation': set('.,;:!?\'"()-–—[]{}'),
        'special': {LONG_S, '&', '§', '¶', '†', '‡', '*', '/', '\\'},
    }
    
    def __init__(self, pdf_path: str, output_dir: str = None):
        """
        Initialize scanner with PDF path.
        
        Args:
            pdf_path: Path to the 1609 Sonnets PDF
            output_dir: Directory for output (default: reports/sonnet_print_block_analysis)
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("reports/sonnet_print_block_analysis")
        
        # Create directory structure
        self._setup_directories()
        
        # Initialize catalogues
        self.character_catalogue: Dict[str, CharacterCatalogueEntry] = {}
        self.anomalies: List[AnomalyEntry] = []
        self.statistics = ScanStatistics()
        
        # Open PDF
        self.doc = fitz.open(str(self.pdf_path))
        self.statistics.total_pages = len(self.doc)
        
        logger.info(f"Initialized scanner for: {self.pdf_path.name}")
        logger.info(f"Total pages: {self.statistics.total_pages}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_directories(self):
        """Create output directory structure."""
        dirs = [
            self.output_dir,
            self.output_dir / "character_atlas",
            self.output_dir / "character_atlas" / "uppercase",
            self.output_dir / "character_atlas" / "lowercase",
            self.output_dir / "character_atlas" / "digits",
            self.output_dir / "character_atlas" / "punctuation",
            self.output_dir / "character_atlas" / "special",
            self.output_dir / "character_atlas" / "ligatures",
            self.output_dir / "character_atlas" / "long_s",
            self.output_dir / "anomalies",
            self.output_dir / "page_images",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def _get_character_category(self, char: str) -> str:
        """Determine category for a character."""
        if char == self.LONG_S:
            return "long_s"
        for category, chars in self.CATEGORIES.items():
            if char in chars:
                return category
        if char.isspace():
            return "whitespace"
        return "other"
    
    def _get_unicode_info(self, char: str) -> Tuple[str, str]:
        """Get Unicode name and codepoint for a character."""
        import unicodedata
        try:
            name = unicodedata.name(char, f"UNKNOWN-{ord(char)}")
        except ValueError:
            name = f"UNKNOWN-{ord(char)}"
        codepoint = f"U+{ord(char):04X}"
        return name, codepoint
    
    def scan_page(self, page_num: int, save_images: bool = True) -> List[CharacterInstance]:
        """
        Scan a single page and extract all characters using Tesseract OCR.
        
        Args:
            page_num: 0-indexed page number
            save_images: Whether to save character images
            
        Returns:
            List of CharacterInstance objects
        """
        page = self.doc[page_num]
        instances = []
        
        logger.info(f"Scanning page {page_num + 1}/{self.statistics.total_pages}...")
        
        # Get page as high-resolution image for OCR
        # Use 3x zoom for better OCR accuracy on historical text
        mat = fitz.Matrix(3, 3)
        pix = page.get_pixmap(matrix=mat)
        page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Save full page image
        if save_images:
            page_img_path = self.output_dir / "page_images" / f"page_{page_num + 1:03d}.png"
            page_image.save(page_img_path, optimize=True)
        
        # Use Tesseract OCR to extract character-level data
        if not TESSERACT_AVAILABLE:
            logger.warning("Tesseract not available - skipping OCR")
            self.statistics.pages_scanned += 1
            return instances
        
        try:
            # Get character-level bounding boxes from Tesseract
            # PSM 6 = Assume a single uniform block of text
            # --oem 3 = Default, based on what is available
            ocr_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            ocr_data = pytesseract.image_to_data(
                page_image, 
                output_type=pytesseract.Output.DICT,
                config=ocr_config
            )
            
            n_boxes = len(ocr_data['text'])
            
            block_id = 0
            current_block = -1
            line_id = 0
            current_line = -1
            word_id = 0
            
            for i in range(n_boxes):
                text = ocr_data['text'][i]
                conf = int(ocr_data['conf'][i]) if ocr_data['conf'][i] != '-1' else 0
                
                # Skip empty results
                if not text or text.isspace():
                    continue
                
                # Track blocks and lines
                if ocr_data['block_num'][i] != current_block:
                    current_block = ocr_data['block_num'][i]
                    block_id += 1
                    self.statistics.total_blocks += 1
                
                if ocr_data['line_num'][i] != current_line:
                    current_line = ocr_data['line_num'][i]
                    line_id += 1
                    self.statistics.total_lines += 1
                
                word_id = ocr_data['word_num'][i]
                self.statistics.total_words += 1
                
                # Get bounding box (note: these are at 3x scale)
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # Process each character in the word
                if len(text) > 0:
                    char_width = w / len(text) if len(text) > 0 else w
                    
                    for char_idx, char in enumerate(text):
                        if char.isspace():
                            continue
                        
                        # Calculate character position
                        char_x = x + (char_idx * char_width)
                        char_y = y

                        # Check validation (filter modern/impossible chars)
                        # CRITICAL: Do NOT discard. Catalogue as Anomaly for investigation.
                        # PER AXIOM OF INTENT: These are High-Entropy Data Points, not errors.
                        is_anomaly = False
                        anomaly_type = None
                        
                        if not self._is_valid_character(char):
                            is_anomaly = True
                            anomaly_type = "invalid_character_symbol"
                        elif self._is_noise(char, char_x/3, char_y/3, char_width/3, h/3, page.rect.width, page.rect.height):
                            is_anomaly = True
                            anomaly_type = "suspicious_mark_or_noise"
                            
                        if is_anomaly:
                            # Create Anomaly Entry
                            # We need to save the image for proof
                            
                            # Temporarily create instance to use _save logic
                            temp_inst = CharacterInstance(
                                character=char,
                                page_number=page_num + 1,
                                x=char_x/3, y=char_y/3, width=char_width/3, height=h/3
                            )
                            # Custom save for anomalies
                            img_path = self._save_anomaly_image(page_image, temp_inst, page_num, char)
                            
                            self.anomalies.append(AnomalyEntry(
                                anomaly_type=anomaly_type,
                                page_number=page_num + 1,
                                x=char_x/3, y=char_y/3, width=char_width/3, height=h/3,
                                description=f"Suspicious print block identified as '{char}'",
                                severity="high",  # User considers these cornerstone
                                image_path=img_path,
                                related_character=char
                            ))
                            self.statistics.anomalies_detected += 1
                            continue # Skip adding to normal character text stream
                            
                        # Advanced F vs Long-s Check
                        is_long_s = False
                        if char == 'f':
                            # Crop for analysis
                            scale = 3
                            x1 = int(char_x) - 1
                            y1 = int(char_y) - 1
                            x2 = int(char_x + char_width) + 1
                            y2 = int(char_y + h) + 1
                            try:
                                analysis_crop = page_image.crop((x1, y1, x2, y2))
                                if self._analyze_f_vs_long_s(analysis_crop):
                                    is_long_s = True
                                    char = self.LONG_S
                            except Exception:
                                pass # formatting error
                        
                        if not is_long_s:
                            # Fallback to heuristic
                            is_long_s = (char == self.LONG_S or 
                                        (char == 'f' and self._might_be_long_s(text, char_idx)))
                            if is_long_s:
                                char = self.LONG_S
                        
                        # Check for ligatures
                        is_ligature = False
                        if char_idx < len(text) - 1:
                            digraph = text[char_idx:char_idx+2]
                            if digraph in self.LIGATURES:
                                is_ligature = True
                        
                        instance = CharacterInstance(
                            character=char,
                            page_number=page_num + 1,
                            x=char_x / 3,  # Convert back to original scale
                            y=char_y / 3,
                            width=char_width / 3,
                            height=h / 3,
                            confidence=conf,
                            block_id=block_id,
                            line_id=line_id,
                            word_id=word_id,
                            is_long_s=is_long_s,
                            is_ligature=is_ligature,
                        )
                        
                        # Save character image samples
                        if save_images and self._should_save_image(char):
                            instance.image_path = self._save_character_image(
                                page_image, instance, page_num, scale=3
                            )
                        
                        instances.append(instance)
                        self.statistics.total_characters += 1
                        
                        # Update catalogue
                        self._update_catalogue(instance)
                        
                        # Track Long-s and ligatures
                        if is_long_s:
                            self.statistics.long_s_count += 1
                        if is_ligature:
                            self.statistics.ligatures_found += 1
            
            # Calculate average confidence for this page
            page_confidences = [inst.confidence for inst in instances if inst.confidence > 0]
            if page_confidences:
                avg_conf = sum(page_confidences) / len(page_confidences)
                logger.info(f"  Page {page_num + 1}: {len(instances)} chars, avg confidence: {avg_conf:.1f}%")
            
        except Exception as e:
            logger.error(f"OCR failed on page {page_num + 1}: {e}")
        
        self.statistics.pages_scanned += 1
        return instances
    
    def _might_be_long_s(self, text: str, pos: int) -> bool:
        """
        Heuristic to detect if 'f' might actually be Long-s.
        In historical printing, Long-s appears in word-medial positions.
        """
        # Long-s typically appears at start or middle of words, not at end
        # Common patterns: "ſhall", "haſt", "ſo", but not "his" (ends with s)
        if pos == len(text) - 1:
            return False  # End of word - probably regular 's' or 'f'
        
        # Check if followed by common Long-s patterns
        if pos < len(text) - 1:
            next_char = text[pos + 1]
            # Long-s + h, t, i, etc. are common
            if next_char in 'htilo':
                return True
        
        return False
    
    def _analyze_f_vs_long_s(self, char_img: Image.Image) -> bool:
        """
        Visual analysis to distinguish 'f' from 'ſ' (Long-s).
        Returns True if likely 'ſ' (Long-s), False if 'f'.
        
        Heuristic: 'f' has a crossbar extending to the right. 'ſ' has a nub on left but empty on right.
        """
        # Convert to grayscale and threshold
        img = char_img.convert('L')
        # Simple binary threshold
        threshold = 128
        width, height = img.size
        
        # Define the "crossbar zone" (middle 40-60% of height)
        # and "right side" (right 40% of width)
        # If there is significant ink here, it's likely an 'f'
        
        mid_top = int(height * 0.40)
        mid_bot = int(height * 0.60)
        right_start = int(width * 0.60)
        
        ink_count = 0
        total_pixels = 0
        
        for y in range(mid_top, mid_bot):
            for x in range(right_start, width):
                pixel = img.getpixel((x, y))
                if pixel < threshold:  # Black/Ink
                    ink_count += 1
                total_pixels += 1
        
        if total_pixels == 0: return False
        
        density = ink_count / total_pixels
        
        # If right-side crossbar density is low, it's a Long-s
        # 'f' usually has > 20% density here (the bar)
        # 'ſ' usually has < 10% (empty space)
        return density < 0.15

    def _is_valid_character(self, char: str) -> bool:
        """
        Check if character is valid for 1609 English text.
        Filters out modern symbols and noise.
        """
        # Allow standard letters, digits
        if char.isalnum():
            return True
            
        # Allow specific punctuation and historical marks
        # ~ is allowed (nasal abbreviation)
        # & is allowed
        # Quotes and dashes allowed
        allowed = set('.,:;!?"\'()[]-–—&~§¶†‡')
        if char in allowed:
            return True
            
        # Allow Unicode quotes and variants
        if char in '‘’“”':
            return True
            
        # Allow Ligatures and Long-s
        if char == self.LONG_S:
            return True
        if char in ('æ', 'œ'):
            return True
            
        return False
        
    def _is_noise(self, char: str, x: float, y: float, w: float, h: float, page_w: float, page_h: float) -> bool:
        """
        Determine if a character is likely scanner noise/borders.
        """
        # ... existing checks ...
        
        # Check 1: Edge proximity (Margins)
        # If char implies a vertical line and is at the edge
        if char in '|/\\lI1':
            margin_threshold = 0.05 * page_w
            if x < margin_threshold or x > (page_w - margin_threshold):
                return True
                
        # Check 2: Aspect Ratio Extremes
        # If it's a "flat" char like - or _ but very tall -> Noise
        ratio = h / w if w > 0 else 0
        if ratio > 10: # Extremely tall and thin vertical line
             return True
             
        return False

    def _should_save_image(self, char: str) -> bool:
        """Determine if we should save an image for this character."""
        # Save first 10 samples of each character
        if char not in self.character_catalogue:
            return True
        entry = self.character_catalogue[char]
        return len(entry.sample_images) < 10
    
    def _save_character_image(self, page_image: Image.Image, 
                              instance: CharacterInstance, 
                              page_num: int,
                              scale: int = 3) -> str:
        """Crop and save character image."""
        # Scale coordinates (we used 3x zoom by default for OCR)
        x1 = int(instance.x * scale) - 2
        y1 = int(instance.y * scale) - 2
        x2 = int((instance.x + instance.width) * scale) + 2
        y2 = int((instance.y + instance.height) * scale) + 2
        
        # Ensure bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(page_image.width, x2)
        y2 = min(page_image.height, y2)
        
        # Crop
        try:
            char_img = page_image.crop((x1, y1, x2, y2))
        except Exception as e:
            logger.warning(f"Failed to crop: {e}")
            return None
            
        # Determine save directory
        category = self._get_character_category(instance.character)
        if category == "uppercase":
            subdir = "uppercase"
        elif category == "lowercase":
            subdir = "lowercase"
        elif category == "digit":
            subdir = "digits"
        elif category == "punctuation":
            subdir = "punctuation"
        elif category == "long_s" or instance.is_long_s:
            subdir = "long_s"
        elif instance.is_ligature:
            subdir = "ligatures"
        else:
            subdir = "special"
        
        # Generate filename
        char_safe = instance.character.replace('/', 'SLASH').replace('\\', 'BACKSLASH')
        if ord(instance.character) > 127:
            char_safe = f"U{ord(instance.character):04X}"
        
        # Use a simpler counter to avoid recounting
        count = int(time.time() * 1000) % 1000000
        
        filename = f"{char_safe}_{count:06d}_p{page_num+1}_x{int(instance.x)}_y{int(instance.y)}.png"
        filepath = self.output_dir / "character_atlas" / subdir / filename
        
        try:
            char_img.save(filepath)
            return str(filepath)
        except Exception as e:
            logger.warning(f"Failed to save character image: {e}")
            return None
    
    def _save_anomaly_image(self, page_image: Image.Image, 
                           instance: CharacterInstance, 
                           page_num: int,
                           char_label: str) -> str:
        """Save image of an anomaly."""
        scale = 3
        x1 = int(instance.x * scale) - 2
        y1 = int(instance.y * scale) - 2
        x2 = int((instance.x + instance.width) * scale) + 2
        y2 = int((instance.y + instance.height) * scale) + 2
        
        # Ensure bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(page_image.width, x2)
        y2 = min(page_image.height, y2)
        
        try:
            crop = page_image.crop((x1, y1, x2, y2))
            
            # Save to anomalies folder
            filename = f"ANOMALY_{char_label}_{int(time.time()*1000)%1000000}_p{page_num+1}.png"
            # Sanitize filename
            filename = filename.replace('/', 'SLASH').replace('\\', 'BACK')
            
            filepath = self.output_dir / "anomalies" / filename
            self.output_dir.joinpath("anomalies").mkdir(exist_ok=True)
            
            crop.save(filepath)
            return str(filepath)
        except Exception as e:
            logger.warning(f"Failed to save anomaly image: {e}")
            return None

    def _update_catalogue(self, instance: CharacterInstance):
        """Update character catalogue with new instance."""
        char = instance.character
        
        if char not in self.character_catalogue:
            unicode_name, unicode_codepoint = self._get_unicode_info(char)
            category = self._get_character_category(char)
            
            self.character_catalogue[char] = CharacterCatalogueEntry(
                character=char,
                unicode_name=unicode_name,
                unicode_codepoint=unicode_codepoint,
                category=category,
            )
        
        entry = self.character_catalogue[char]
        entry.total_count += 1
        
        # Keep limited instances for memory
        if len(entry.instances) < 100:
            entry.instances.append(instance)
        
        if instance.image_path:
            entry.sample_images.append(instance.image_path)
    
    def scan_all_pages(self, start_page: int = 0, end_page: int = None, 
                       save_images: bool = True) -> Dict[str, Any]:
        """
        Scan all pages (or a range) of the document.
        
        Args:
            start_page: Starting page (0-indexed)
            end_page: Ending page (exclusive), None for all
            save_images: Whether to save character sample images
            
        Returns:
            Complete scan results dictionary
        """
        start_time = time.time()
        
        if end_page is None:
            end_page = len(self.doc)
        
        logger.info(f"Starting scan of pages {start_page + 1} to {end_page}...")
        
        all_instances = []
        for page_num in range(start_page, end_page):
            instances = self.scan_page(page_num, save_images=save_images)
            all_instances.extend(instances)
            
            # Progress update
            progress = (page_num - start_page + 1) / (end_page - start_page) * 100
            logger.info(f"Progress: {progress:.1f}% ({len(all_instances)} characters found)")
        
        # Finalize statistics
        self.statistics.scan_duration_seconds = time.time() - start_time
        self.statistics.unique_characters = len(self.character_catalogue)
        
        # Calculate average dimensions
        for entry in self.character_catalogue.values():
            if entry.instances:
                entry.average_width = sum(i.width for i in entry.instances) / len(entry.instances)
                entry.average_height = sum(i.height for i in entry.instances) / len(entry.instances)
        
        logger.info(f"Scan complete in {self.statistics.scan_duration_seconds:.1f}s")
        logger.info(f"Total characters: {self.statistics.total_characters}")
        logger.info(f"Unique characters: {self.statistics.unique_characters}")
        
        return {
            "statistics": asdict(self.statistics),
            "catalogue": {k: asdict(v) for k, v in self.character_catalogue.items()},
            "anomalies": [asdict(a) for a in self.anomalies],
        }
    
    def generate_frequency_csv(self) -> str:
        """Generate CSV of character frequencies."""
        csv_path = self.output_dir / "character_frequency.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Character", "Unicode Name", "Codepoint", "Category", 
                "Count", "Percentage", "Avg Width", "Avg Height", "Samples"
            ])
            
            # Sort by count descending
            sorted_entries = sorted(
                self.character_catalogue.values(), 
                key=lambda x: x.total_count, 
                reverse=True
            )
            
            total = self.statistics.total_characters or 1
            
            for entry in sorted_entries:
                percentage = (entry.total_count / total) * 100
                writer.writerow([
                    entry.character,
                    entry.unicode_name,
                    entry.unicode_codepoint,
                    entry.category,
                    entry.total_count,
                    f"{percentage:.3f}%",
                    f"{entry.average_width:.2f}",
                    f"{entry.average_height:.2f}",
                    len(entry.sample_images)
                ])
        
        logger.info(f"Character frequency CSV saved to: {csv_path}")
        return str(csv_path)
    
    def generate_statistics_json(self) -> str:
        """Generate JSON statistics file."""
        json_path = self.output_dir / "statistics.json"
        
        data = {
            "scan_info": {
                "source_file": str(self.pdf_path),
                "scan_date": datetime.now().isoformat(),
                "scanner_version": "1.0.0",
            },
            "statistics": asdict(self.statistics),
            "character_summary": {
                "by_category": defaultdict(int),
                "top_20": [],
            }
        }
        
        # Summarize by category
        for entry in self.character_catalogue.values():
            data["character_summary"]["by_category"][entry.category] += entry.total_count
        
        # Top 20 characters
        sorted_entries = sorted(
            self.character_catalogue.items(), 
            key=lambda x: x[1].total_count, 
            reverse=True
        )[:20]
        
        for char, entry in sorted_entries:
            data["character_summary"]["top_20"].append({
                "character": char,
                "count": entry.total_count,
                "category": entry.category,
            })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Statistics JSON saved to: {json_path}")
        return str(json_path)
    
    def generate_html_report(self) -> str:
        """Generate comprehensive HTML report with embedded images."""
        html_path = self.output_dir / "print_block_report.html"
        
        # Sort characters for A-Z presentation
        sorted_chars = sorted(
            self.character_catalogue.items(),
            key=lambda x: (
                0 if x[1].category == 'uppercase' else
                1 if x[1].category == 'lowercase' else
                2 if x[1].category == 'digit' else
                3 if x[1].category == 'punctuation' else
                4 if x[1].category == 'long_s' else
                5,
                x[0].lower() if x[0].isalpha() else x[0]
            )
        )
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>1609 Shakespeare Sonnets - Print Block Analysis</title>
    <style>
        :root {{
            --bg-dark: #1a1a2e;
            --bg-card: #16213e;
            --accent: #e94560;
            --text: #eaeaea;
            --text-muted: #888;
            --border: #0f3460;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}
        
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, var(--accent), #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        
        h2 {{
            color: var(--accent);
            border-bottom: 2px solid var(--border);
            padding-bottom: 0.5rem;
            margin: 2rem 0 1rem;
        }}
        
        h3 {{ color: #ff6b6b; margin: 1.5rem 0 0.5rem; }}
        
        .subtitle {{
            color: var(--text-muted);
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }}
        
        .stat-card .value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--accent);
        }}
        
        .stat-card .label {{
            color: var(--text-muted);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .character-section {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }}
        
        .character-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }}
        
        .character-display {{
            font-size: 3rem;
            width: 80px;
            height: 80px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--bg-dark);
            border-radius: 8px;
            border: 2px solid var(--accent);
        }}
        
        .character-info {{
            flex: 1;
        }}
        
        .character-info .name {{
            font-size: 1.2rem;
            font-weight: bold;
        }}
        
        .character-info .details {{
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
        
        .character-info .count {{
            color: var(--accent);
            font-weight: bold;
        }}
        
        .sample-images {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }}
        
        .sample-images img {{
            height: 40px;
            background: white;
            border-radius: 4px;
            padding: 2px;
        }}
        
        .category-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .category-uppercase {{ background: #2ecc71; color: #000; }}
        .category-lowercase {{ background: #3498db; color: #fff; }}
        .category-digit {{ background: #9b59b6; color: #fff; }}
        .category-punctuation {{ background: #f39c12; color: #000; }}
        .category-special {{ background: #e74c3c; color: #fff; }}
        .category-long_s {{ background: #1abc9c; color: #000; }}
        .category-ligature {{ background: #e91e63; color: #fff; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        
        th {{ background: var(--bg-card); color: var(--accent); }}
        
        tr:hover {{ background: rgba(233, 69, 96, 0.1); }}
        
        .footer {{
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border);
            text-align: center;
            color: var(--text-muted);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 1609 Shakespeare Sonnets</h1>
        <p class="subtitle">Exhaustive Print Block Analysis Report</p>
        
        <h2>📊 Scan Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{self.statistics.pages_scanned}</div>
                <div class="label">Pages Scanned</div>
            </div>
            <div class="stat-card">
                <div class="value">{self.statistics.total_characters:,}</div>
                <div class="label">Total Characters</div>
            </div>
            <div class="stat-card">
                <div class="value">{self.statistics.unique_characters}</div>
                <div class="label">Unique Characters</div>
            </div>
            <div class="stat-card">
                <div class="value">{self.statistics.long_s_count}</div>
                <div class="label">Long-s (ſ) Found</div>
            </div>
            <div class="stat-card">
                <div class="value">{self.statistics.ligatures_found}</div>
                <div class="label">Ligatures</div>
            </div>
            <div class="stat-card">
                <div class="value">{self.statistics.scan_duration_seconds:.1f}s</div>
                <div class="label">Scan Duration</div>
            </div>
        </div>
        
        <h2>📝 Character Catalogue (A-Z)</h2>
"""
        
        current_category = None
        for char, entry in sorted_chars:
            if entry.category != current_category:
                current_category = entry.category
                category_title = current_category.replace('_', ' ').title()
                html += f"\n        <h3>{category_title}</h3>\n"
            
            # Get relative image paths
            sample_imgs = ""
            for img_path in entry.sample_images[:5]:
                rel_path = Path(img_path).relative_to(self.output_dir) if img_path else ""
                if rel_path:
                    sample_imgs += f'<img src="{rel_path}" alt="{char}" title="Sample from document">'
            
            percentage = (entry.total_count / max(self.statistics.total_characters, 1)) * 100
            
            html += f"""
        <div class="character-section">
            <div class="character-header">
                <div class="character-display">{char if char.isprintable() else f'U+{ord(char):04X}'}</div>
                <div class="character-info">
                    <div class="name">{entry.unicode_name}</div>
                    <div class="details">
                        {entry.unicode_codepoint} • 
                        <span class="category-badge category-{entry.category}">{entry.category}</span>
                    </div>
                    <div class="count">
                        {entry.total_count:,} occurrences ({percentage:.2f}%)
                    </div>
                </div>
            </div>
            {"<div class='sample-images'>" + sample_imgs + "</div>" if sample_imgs else ""}
        </div>
"""
        
        html += f"""
        
        <h2>📋 Frequency Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Character</th>
                    <th>Name</th>
                    <th>Count</th>
                    <th>Percentage</th>
                    <th>Category</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Top 30 frequency table
        for rank, (char, entry) in enumerate(sorted(
            self.character_catalogue.items(),
            key=lambda x: x[1].total_count,
            reverse=True
        )[:30], 1):
            percentage = (entry.total_count / max(self.statistics.total_characters, 1)) * 100
            html += f"""
                <tr>
                    <td>{rank}</td>
                    <td style="font-size: 1.5rem; font-weight: bold;">{char if char.isprintable() else '?'}</td>
                    <td>{entry.unicode_name}</td>
                    <td>{entry.total_count:,}</td>
                    <td>{percentage:.2f}%</td>
                    <td><span class="category-badge category-{entry.category}">{entry.category}</span></td>
                </tr>
"""
        
        html += f"""
            </tbody>
        </table>
        
        <h2>🚨 Anomalies & Suspicious Marks</h2>
        <p>The following print blocks were flagged as invalid characters, noise, or suspicious artifacts. These are documented for forensic analysis.</p>
        <div class="anomalies-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 1rem;">
    """
        
        for anomaly in self.anomalies[:200]: 
             rel_path = ""
             if anomaly.image_path:
                 try:
                    rel_path = Path(anomaly.image_path).relative_to(self.output_dir)
                 except ValueError:
                    rel_path = ""
             
             html += f"""
             <div class="anomaly-card" style="border: 1px solid #f5c6cb; padding: 10px; border-radius: 4px; background: #f8d7da;">
                <div class="anomaly-img">
                    {f'<img src="{rel_path}" style="max-width: 100%; height: auto;">' if rel_path else '(No Image)'}
                </div>
                <div class="anomaly-info" style="font-size: 0.8em; margin-top: 5px;">
                    <strong>{anomaly.related_character}</strong> (Pg {anomaly.page_number})<br>
                    <span style="color: #721c24;">{anomaly.anomaly_type}</span>
                </div>
             </div>
             """
             
        html += f"""
        </div>
        
        <div class="footer">
            <p>Generated by CODEFINDER Print Block Scanner</p>
            <p>Source: {self.pdf_path.name} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"HTML report saved to: {html_path}")
        return str(html_path)
    
    def close(self):
        """Close the PDF document."""
        if self.doc:
            self.doc.close()


# ============================================================================
# CLI INTERFACE
# ============================================================================

def parse_page_range(page_str: str, total_pages: int) -> Tuple[int, int]:
    """Parse page range string like '1-5' or '10'."""
    if '-' in page_str:
        parts = page_str.split('-')
        start = int(parts[0]) - 1  # Convert to 0-indexed
        end = int(parts[1])
    else:
        start = int(page_str) - 1
        end = start + 1
    
    # Clamp to valid range
    start = max(0, min(start, total_pages - 1))
    end = max(start + 1, min(end, total_pages))
    
    return start, end


def main():
    parser = argparse.ArgumentParser(
        description="Exhaustive OCR scanner for 1609 Shakespeare Sonnets"
    )
    parser.add_argument(
        "--pdf", 
        default="data/sources/SONNETS_QUARTO_1609_NET.pdf",
        help="Path to the Sonnets PDF"
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="Page range to scan (e.g., '1-5' or '10'). Default: all pages"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: reports/sonnet_print_block_analysis)"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip saving character images"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - scan first 3 pages only"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔍 SONNET PRINT BLOCK SCANNER")
    print("=" * 60)
    print(f"Source: {args.pdf}")
    print()
    
    try:
        scanner = SonnetPrintBlockScanner(args.pdf, args.output)
        
        # Determine page range
        if args.test:
            start_page, end_page = 0, min(3, scanner.statistics.total_pages)
            print(f"🧪 TEST MODE: Scanning pages 1-{end_page}")
        elif args.pages:
            start_page, end_page = parse_page_range(args.pages, scanner.statistics.total_pages)
            print(f"📄 Scanning pages {start_page + 1} to {end_page}")
        else:
            start_page, end_page = 0, scanner.statistics.total_pages
            print(f"📄 Scanning ALL {end_page} pages")
        
        print()
        
        # Run scan
        results = scanner.scan_all_pages(
            start_page=start_page,
            end_page=end_page,
            save_images=not args.no_images
        )
        
        # Generate outputs
        print("\n" + "=" * 60)
        print("📊 GENERATING REPORTS")
        print("=" * 60)
        
        csv_path = scanner.generate_frequency_csv()
        json_path = scanner.generate_statistics_json()
        html_path = scanner.generate_html_report()
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ SCAN COMPLETE")
        print("=" * 60)
        print(f"Pages scanned: {scanner.statistics.pages_scanned}")
        print(f"Total characters: {scanner.statistics.total_characters:,}")
        print(f"Unique characters: {scanner.statistics.unique_characters}")
        print(f"Long-s found: {scanner.statistics.long_s_count}")
        print(f"Ligatures found: {scanner.statistics.ligatures_found}")
        print(f"Duration: {scanner.statistics.scan_duration_seconds:.1f}s")
        print()
        print("📁 Output files:")
        print(f"   - {html_path}")
        print(f"   - {csv_path}")
        print(f"   - {json_path}")
        print(f"   - {scanner.output_dir}/character_atlas/")
        
        scanner.close()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Scan failed")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
