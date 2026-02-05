"""
Blackletter/Gothic Text Recognition Specialist Agent
=====================================================
Advanced agent for recognizing and processing blackletter/Gothic typography
specific to the 1611 King James Bible.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass, field
import logging
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BlackletterCharacter:
    """Represents a blackletter character with its variants"""
    modern: str
    blackletter_forms: List[str]
    common_ocr_errors: List[str]
    ligature_components: Optional[List[str]] = None
    contextual_rules: Dict[str, str] = field(default_factory=dict)


class BlackletterSpecialist:
    """
    Specialized agent for blackletter/Gothic text recognition.
    Handles the unique challenges of 1611 KJV typography.
    """
    
    def __init__(self):
        # Comprehensive blackletter character mapping
        self.character_map = self._initialize_character_map()
        
        # Common word patterns in 1611 KJV
        self.common_patterns = self._load_common_patterns()
        
        # Specialized OCR configurations
        self.ocr_configs = {
            'blackletter_standard': (
                '--oem 1 --psm 6 '
                '-l eng '
                '-c preserve_interword_spaces=1 '
                '-c textord_old_xheight=1 '
                '-c tessedit_char_blacklist=@#$%^&*()_+={}[]|\\<>?~ '
            ),
            'blackletter_single_word': '--oem 1 --psm 8 -l eng',
            'blackletter_single_char': '--oem 1 --psm 10 -l eng',
            'blackletter_sparse': '--oem 1 --psm 11 -l eng',
            'blackletter_uniform': '--oem 1 --psm 13 -l eng'
        }
        
        # Thresholding parameters optimized for old prints
        self.threshold_params = {
            'standard': {'method': cv2.THRESH_OTSU, 'block_size': 11, 'C': 2},
            'faded': {'method': cv2.THRESH_OTSU, 'block_size': 15, 'C': 5},
            'high_contrast': {'method': cv2.THRESH_BINARY, 'threshold': 127},
            'adaptive': {'method': cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 'block_size': 11, 'C': 2}
        }
        
    def _initialize_character_map(self) -> Dict[str, BlackletterCharacter]:
        """Initialize comprehensive blackletter character mappings"""
        
        char_map = {
            's': BlackletterCharacter(
                modern='s',
                blackletter_forms=['ſ', 'ʃ', 'f'],
                common_ocr_errors=['f', 'l', 'i', 't'],
                contextual_rules={'end_of_word': 's', 'before_s': 's', 'default': 'ſ'}
            ),
            'w': BlackletterCharacter(
                modern='w',
                blackletter_forms=['vv', 'uu', 'w'],
                common_ocr_errors=['vv', 'uu', 'm', 'iv'],
                ligature_components=['v', 'v']
            ),
            'th': BlackletterCharacter(
                modern='th',
                blackletter_forms=['þ', 'th', 'ð'],
                common_ocr_errors=['p', 'b', 'd', 'h'],
                contextual_rules={'initial': 'Þ', 'medial': 'þ', 'the': 'þe'}
            ),
            'y': BlackletterCharacter(
                modern='y',
                blackletter_forms=['y', 'ẏ', 'ÿ'],
                common_ocr_errors=['p', 'v', 'u'],
                contextual_rules={'the': 'ye', 'that': 'yt'}
            ),
            'u': BlackletterCharacter(
                modern='u',
                blackletter_forms=['u', 'v'],
                common_ocr_errors=['n', 'v', 'ii'],
                contextual_rules={'initial': 'v', 'medial': 'u', 'final': 'u'}
            ),
            'v': BlackletterCharacter(
                modern='v',
                blackletter_forms=['v', 'u'],
                common_ocr_errors=['u', 'n', 'r'],
                contextual_rules={'initial': 'v', 'medial': 'u'}
            ),
            'i': BlackletterCharacter(
                modern='i',
                blackletter_forms=['i', 'j', 'ı'],
                common_ocr_errors=['l', '1', 'j', 't'],
                contextual_rules={'initial_capital': 'I', 'numeral': 'j'}
            ),
            'j': BlackletterCharacter(
                modern='j',
                blackletter_forms=['j', 'i'],
                common_ocr_errors=['i', '1', 'l'],
                contextual_rules={'always': 'i'}
            ),
            # Ligatures
            'ff': BlackletterCharacter(
                modern='ff',
                blackletter_forms=['ff', 'ﬀ'],
                common_ocr_errors=['H', 'tt', 'ft'],
                ligature_components=['f', 'f']
            ),
            'fi': BlackletterCharacter(
                modern='fi',
                blackletter_forms=['fi', 'ﬁ'],
                common_ocr_errors=['h', 'n', 'A'],
                ligature_components=['f', 'i']
            ),
            'fl': BlackletterCharacter(
                modern='fl',
                blackletter_forms=['fl', 'ﬂ'],
                common_ocr_errors=['H', 'A', 'ft'],
                ligature_components=['f', 'l']
            ),
            'st': BlackletterCharacter(
                modern='st',
                blackletter_forms=['st', 'ﬆ'],
                common_ocr_errors=['ft', 'lt'],
                ligature_components=['s', 't']
            ),
            'ct': BlackletterCharacter(
                modern='ct',
                blackletter_forms=['ct', 'ꝉ'],
                common_ocr_errors=['d', 'a'],
                ligature_components=['c', 't']
            ),
            'ae': BlackletterCharacter(
                modern='ae',
                blackletter_forms=['æ', 'ae'],
                common_ocr_errors=['ce', 'oe', 'x'],
                ligature_components=['a', 'e']
            ),
            'oe': BlackletterCharacter(
                modern='oe',
                blackletter_forms=['œ', 'oe'],
                common_ocr_errors=['ce', 'ae', 'eo'],
                ligature_components=['o', 'e']
            )
        }
        
        return char_map
    
    def _load_common_patterns(self) -> Dict[str, List[str]]:
        """Load common word patterns from 1611 KJV"""
        
        patterns = {
            'articles': ['the', 'ye', 'þe', 'a', 'an'],
            'pronouns': ['thou', 'þou', 'thee', 'thy', 'thine', 'ye', 'you', 'your'],
            'verbs': ['hath', 'haþ', 'doth', 'doþ', 'saith', 'saiþ', 'shall', 'shalt'],
            'conjunctions': ['and', '&', 'but', 'for', 'if', 'that', 'þat'],
            'prepositions': ['of', 'in', 'to', 'unto', 'vnto', 'with', 'wiþ', 'from'],
            'religious': ['God', 'Lord', 'LORD', 'Jesus', 'Iesus', 'Christ', 'Spirit'],
            'numbers': ['one', 'two', 'three', 'foure', 'fiue', 'sixe', 'seuen', 'eight', 'nine', 'ten']
        }
        
        return patterns
    
    def enhance_image_for_blackletter(self, image: np.ndarray, 
                                     enhancement_level: str = 'standard') -> np.ndarray:
        """
        Apply specialized enhancement for blackletter text
        
        Args:
            image: Input image
            enhancement_level: 'light', 'standard', 'heavy', 'adaptive'
        """
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if enhancement_level == 'light':
            # Minimal processing for clean images
            enhanced = cv2.equalizeHist(gray)
            denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
            
        elif enhancement_level == 'standard':
            # Standard enhancement for typical pages
            enhanced = cv2.equalizeHist(gray)
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Sharpen to improve character edges
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            denoised = cv2.filter2D(denoised, -1, kernel)
            
        elif enhancement_level == 'heavy':
            # Heavy processing for poor quality images
            # CLAHE for local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Strong denoising
            denoised = cv2.fastNlMeansDenoising(enhanced, h=30)
            
            # Morphological operations to restore broken characters
            kernel = np.ones((2,2), np.uint8)
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
            
        elif enhancement_level == 'adaptive':
            # Analyze image quality and apply appropriate enhancement
            quality = self._assess_image_quality(gray)
            
            if quality['contrast'] < 50:
                # Low contrast - enhance aggressively
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
            else:
                enhanced = gray
            
            if quality['noise'] > 30:
                # High noise - denoise strongly
                denoised = cv2.fastNlMeansDenoising(enhanced, h=quality['noise'])
            else:
                denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
        else:
            denoised = gray
        
        return denoised
    
    def _assess_image_quality(self, gray: np.ndarray) -> Dict[str, float]:
        """Assess image quality metrics"""
        
        # Calculate contrast
        contrast = np.std(gray)
        
        # Estimate noise level
        kernel = np.array([[1, -2, 1],
                          [-2, 4, -2],
                          [1, -2, 1]])
        laplacian = cv2.filter2D(gray, -1, kernel)
        noise = np.std(laplacian)
        
        # Calculate sharpness
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sharpness = np.mean(np.sqrt(sobelx**2 + sobely**2))
        
        return {
            'contrast': contrast,
            'noise': noise,
            'sharpness': sharpness
        }
    
    def adaptive_threshold_for_blackletter(self, gray: np.ndarray, 
                                          method: str = 'auto') -> np.ndarray:
        """
        Apply adaptive thresholding optimized for blackletter text
        
        Args:
            gray: Grayscale image
            method: 'auto', 'standard', 'faded', 'high_contrast', 'adaptive'
        """
        
        if method == 'auto':
            # Automatically determine best method
            quality = self._assess_image_quality(gray)
            
            if quality['contrast'] < 30:
                method = 'faded'
            elif quality['contrast'] > 100:
                method = 'high_contrast'
            else:
                method = 'adaptive'
        
        if method == 'standard':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif method == 'faded':
            # For faded text, use adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 5
            )
            
        elif method == 'high_contrast':
            # Simple threshold for high contrast
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
        elif method == 'adaptive':
            # Adaptive with optimal parameters for blackletter
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up small noise
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Close gaps in characters (important for broken blackletter)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def segment_blackletter_characters(self, binary: np.ndarray, 
                                      min_char_width: int = 5,
                                      min_char_height: int = 10) -> List[Dict[str, Any]]:
        """
        Segment individual blackletter characters with special handling for ligatures
        """
        
        # Invert for connected components
        inverted = cv2.bitwise_not(binary)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            inverted, connectivity=8
        )
        
        characters = []
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # Filter noise
            if w < min_char_width or h < min_char_height or area < 20:
                continue
            
            # Check if this might be a ligature (wider than normal)
            is_ligature = w > h * 1.5 and w > 20
            
            # Check for broken characters (very small width relative to height)
            is_broken = w < h * 0.3
            
            char_info = {
                'bbox': (x, y, w, h),
                'area': area,
                'centroid': tuple(centroids[i]),
                'aspect_ratio': w / h if h > 0 else 0,
                'is_ligature': is_ligature,
                'is_broken': is_broken,
                'label': i
            }
            
            characters.append(char_info)
        
        # Sort by reading order (top to bottom, left to right)
        characters.sort(key=lambda c: (c['bbox'][1], c['bbox'][0]))
        
        # Merge broken characters if needed
        characters = self._merge_broken_characters(characters, labels)
        
        return characters
    
    def _merge_broken_characters(self, characters: List[Dict], 
                                labels: np.ndarray) -> List[Dict]:
        """Merge characters that are likely broken parts of the same letter"""
        
        merged = []
        skip_indices = set()
        
        for i, char in enumerate(characters):
            if i in skip_indices:
                continue
            
            if char['is_broken'] and i < len(characters) - 1:
                # Check if next character is close enough to merge
                next_char = characters[i + 1]
                x1, y1, w1, h1 = char['bbox']
                x2, y2, w2, h2 = next_char['bbox']
                
                # Characters should be horizontally close and vertically aligned
                horizontal_gap = x2 - (x1 + w1)
                vertical_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
                
                if horizontal_gap < 5 and vertical_overlap > h1 * 0.5:
                    # Merge the characters
                    merged_bbox = (
                        min(x1, x2),
                        min(y1, y2),
                        max(x1 + w1, x2 + w2) - min(x1, x2),
                        max(y1 + h1, y2 + h2) - min(y1, y2)
                    )
                    
                    merged_char = {
                        'bbox': merged_bbox,
                        'area': char['area'] + next_char['area'],
                        'centroid': ((x1 + x2) / 2, (y1 + y2) / 2),
                        'aspect_ratio': merged_bbox[2] / merged_bbox[3],
                        'is_ligature': False,
                        'is_broken': False,
                        'is_merged': True,
                        'label': char['label']
                    }
                    
                    merged.append(merged_char)
                    skip_indices.add(i + 1)
                else:
                    merged.append(char)
            else:
                merged.append(char)
        
        return merged
    
    def recognize_blackletter_text(self, image: np.ndarray, 
                                  char_segments: List[Dict],
                                  use_context: bool = True) -> str:
        """
        Recognize blackletter text from segmented characters
        """
        
        recognized_text = []
        
        for i, char_info in enumerate(char_segments):
            x, y, w, h = char_info['bbox']
            
            # Extract character image
            char_img = image[y:y+h, x:x+w]
            
            # Try OCR with single character mode
            char_text = pytesseract.image_to_string(
                char_img,
                config=self.ocr_configs['blackletter_single_char']
            ).strip()
            
            # Apply blackletter corrections
            if char_text:
                char_text = self._correct_blackletter_character(
                    char_text, 
                    char_info,
                    context=recognized_text if use_context else None
                )
            
            # Handle ligatures
            if char_info.get('is_ligature', False):
                char_text = self._recognize_ligature(char_img, char_text)
            
            recognized_text.append(char_text if char_text else '?')
        
        # Join and apply word-level corrections
        text = ''.join(recognized_text)
        
        if use_context:
            text = self._apply_contextual_corrections(text)
        
        return text
    
    def _correct_blackletter_character(self, char: str, 
                                      char_info: Dict,
                                      context: Optional[List[str]] = None) -> str:
        """Apply blackletter-specific character corrections"""
        
        # Common single character corrections
        corrections = {
            'f': 's',  # Long s often read as f
            'l': 'i',  # i often read as l
            '1': 'i',  # i often read as 1
            'vv': 'w',
            'VV': 'W',
            'u': 'v',  # Context dependent
            'j': 'i',  # j was i in 1611
            'J': 'I'
        }
        
        # Apply basic corrections
        if char in corrections:
            # Check context for u/v swap
            if char == 'u' and context:
                # If at beginning of word, likely 'v'
                if not context or context[-1] == ' ':
                    return 'v'
            return corrections.get(char, char)
        
        return char
    
    def _recognize_ligature(self, ligature_img: np.ndarray, 
                           ocr_result: str) -> str:
        """Recognize common blackletter ligatures"""
        
        # Try to match against known ligatures
        ligatures = {
            'ff': 'ff',
            'fi': 'fi',
            'fl': 'fl',
            'ffi': 'ffi',
            'ffl': 'ffl',
            'st': 'st',
            'ct': 'ct',
            'æ': 'ae',
            'œ': 'oe',
            'ß': 'ss'
        }
        
        # If OCR gave a result, check if it's a known ligature
        if ocr_result in ligatures:
            return ligatures[ocr_result]
        
        # Otherwise try to detect based on image features
        # This would use more sophisticated analysis in production
        h, w = ligature_img.shape[:2]
        aspect_ratio = w / h if h > 0 else 0
        
        # Guess based on aspect ratio
        if 1.5 < aspect_ratio < 2.0:
            return 'ff'  # Double character ligature
        elif 2.0 < aspect_ratio < 3.0:
            return 'ffi'  # Triple character ligature
        
        return ocr_result if ocr_result else '??'
    
    def _apply_contextual_corrections(self, text: str) -> str:
        """Apply word-level and contextual corrections"""
        
        # Common word replacements in 1611 KJV
        word_corrections = {
            'vvith': 'with',
            'vnto': 'unto',
            'vp': 'up',
            'vs': 'us',
            'haue': 'have',
            'euery': 'every',
            'ouer': 'over',
            'giue': 'give',
            'aboue': 'above',
            'loue': 'love',
            'ye': 'the',  # When ye = þe
            'yt': 'that',
            'yat': 'that',
            'ys': 'this'
        }
        
        # Apply word corrections
        for old, new in word_corrections.items():
            text = re.sub(r'\b' + old + r'\b', new, text, flags=re.IGNORECASE)
        
        # Fix common patterns
        text = re.sub(r'(?<=[a-z])u(?=[aeiou])', 'v', text)  # u→v before vowels
        text = re.sub(r'\bv(?=[^aeiou])', 'u', text)  # v→u before consonants at word start
        
        return text
    
    def process_blackletter_page(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Complete processing pipeline for a blackletter page
        """
        
        logger.info("Processing blackletter page...")
        
        # Enhance image
        enhanced = self.enhance_image_for_blackletter(image, 'adaptive')
        
        # Apply adaptive thresholding
        binary = self.adaptive_threshold_for_blackletter(enhanced, 'auto')
        
        # Segment characters
        char_segments = self.segment_blackletter_characters(binary)
        
        # Recognize text
        text = self.recognize_blackletter_text(binary, char_segments, use_context=True)
        
        # Calculate statistics
        stats = {
            'total_characters': len(char_segments),
            'ligatures_found': sum(1 for c in char_segments if c.get('is_ligature', False)),
            'broken_characters': sum(1 for c in char_segments if c.get('is_broken', False)),
            'merged_characters': sum(1 for c in char_segments if c.get('is_merged', False))
        }
        
        result = {
            'text': text,
            'character_count': len(char_segments),
            'statistics': stats,
            'segments': char_segments[:50]  # First 50 for review
        }
        
        logger.info(f"Processed {len(char_segments)} characters")
        
        return result