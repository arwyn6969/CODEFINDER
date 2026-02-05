"""
OCR Integration Layer
=====================
Integrates enhanced capabilities with existing OCR system,
providing fine-tuning and improved recognition without rebuilding.
"""

import cv2
import numpy as np
import pytesseract
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
import pickle
from datetime import datetime

# Import existing components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.services.ocr_engine import AdvancedOCR
from app.services.improved_ocr_engine import ImprovedOCREngine
from app.services.kjv_1611_specialist import KJV1611Specialist
from agents.blackletter_specialist import BlackletterSpecialist
from agents.column_layout_detector import ColumnLayoutDetector
from agents.training_data_generator import TrainingDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Enhanced OCR result with confidence and alternatives"""
    text: str
    confidence: float
    alternatives: List[Tuple[str, float]]  # (text, confidence) pairs
    method: str  # Which OCR method was used
    corrections_applied: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class OCRIntegrationLayer:
    """
    Integration layer that combines multiple OCR approaches
    and applies learned corrections.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        # Initialize all OCR components
        self.base_ocr = AdvancedOCR()
        self.improved_ocr = ImprovedOCREngine()
        self.kjv_specialist = KJV1611Specialist()
        self.blackletter_specialist = BlackletterSpecialist()
        self.column_detector = ColumnLayoutDetector()
        
        # Load or initialize correction model
        self.correction_model = self._load_correction_model(model_path)
        
        # Character confidence thresholds
        self.confidence_thresholds = {
            'high': 0.85,
            'medium': 0.60,
            'low': 0.30
        }
        
        # Ensemble weights (can be learned)
        self.ensemble_weights = {
            'tesseract': 0.3,
            'kjv_specialist': 0.35,
            'blackletter_specialist': 0.35
        }
        
        # Error patterns learned from training
        self.common_errors = self._load_error_patterns()
        
        # Index for fast lookup
        self.character_index = {}
        self.word_index = {}
        
    def _load_correction_model(self, model_path: Optional[str]) -> Optional[Any]:
        """Load trained correction model if available"""
        
        if model_path and Path(model_path).exists():
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        
        # Return default correction rules
        return {
            'character_substitutions': {
                'f': {'s': 0.7, 'f': 0.3},  # f often should be s (long s)
                'l': {'i': 0.4, 'l': 0.6},
                'vv': {'w': 0.9, 'vv': 0.1},
                'u': {'v': 0.3, 'u': 0.7},  # Context dependent
            },
            'word_corrections': {
                'haue': 'have',
                'vnto': 'unto',
                'vp': 'up',
                'vs': 'us',
                'euery': 'every',
                'ouer': 'over',
                'ye': 'the',  # When ye = þe
                'yt': 'that'
            },
            'context_rules': [
                {'pattern': r'\bu(?=[aeiou])', 'replacement': 'v'},
                {'pattern': r'^v(?=[^aeiou])', 'replacement': 'u'}
            ]
        }
    
    def _load_error_patterns(self) -> Dict[str, Any]:
        """Load common error patterns from training data"""
        
        patterns_file = Path("training_data/error_patterns.json")
        
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                return json.load(f)
        
        # Default patterns
        return {
            'ligature_splits': {
                'ﬀ': ['f', 'f'],
                'ﬁ': ['f', 'i'],
                'ﬂ': ['f', 'l'],
                'ﬆ': ['s', 't']
            },
            'common_misreads': {
                'rn': 'm',  # rn often read as m
                'cl': 'd',  # cl often read as d
                'li': 'h',  # li often read as h
            }
        }
    
    def process_page(self, image: np.ndarray, 
                    method: str = 'ensemble',
                    enhance: bool = True) -> Dict[str, Any]:
        """
        Process a full page with integrated OCR approach
        
        Args:
            image: Page image
            method: 'base', 'improved', 'specialist', 'ensemble'
            enhance: Whether to apply image enhancement
        """
        
        logger.info(f"Processing page with {method} method")
        
        # Step 1: Image Enhancement
        if enhance:
            image = self._enhance_image(image)
        
        # Step 2: Layout Analysis
        layout = self.column_detector.detect_columns(image, method='advanced')
        
        # Step 3: Process based on method
        if method == 'ensemble':
            result = self._ensemble_ocr(image, layout)
        elif method == 'specialist':
            result = self._specialist_ocr(image, layout)
        elif method == 'improved':
            result = self._improved_ocr(image, layout)
        else:
            result = self._base_ocr(image, layout)
        
        # Step 4: Apply corrections and post-processing
        result = self._apply_corrections(result)
        
        # Step 5: Build indices
        self._update_indices(result)
        
        return result
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive image enhancement"""
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Use blackletter specialist enhancement
        enhanced = self.blackletter_specialist.enhance_image_for_blackletter(
            gray, 'adaptive'
        )
        
        return enhanced
    
    def _ensemble_ocr(self, image: np.ndarray, 
                     layout: Any) -> Dict[str, Any]:
        """Ensemble OCR combining multiple methods"""
        
        results = []
        
        # Method 1: Tesseract
        try:
            tesseract_text = pytesseract.image_to_string(
                image,
                config='--oem 1 --psm 6 -l eng'
            )
            tesseract_conf = self._calculate_confidence(tesseract_text, 'tesseract')
            results.append({
                'method': 'tesseract',
                'text': tesseract_text,
                'confidence': tesseract_conf,
                'weight': self.ensemble_weights['tesseract']
            })
        except Exception as e:
            logger.warning(f"Tesseract failed: {e}")
        
        # Method 2: KJV Specialist
        try:
            blocks = self.kjv_specialist.identify_print_blocks(image)
            kjv_text = ' '.join([b.text for b in blocks if b.text])
            kjv_conf = np.mean([b.confidence for b in blocks if b.confidence > 0])
            results.append({
                'method': 'kjv_specialist',
                'text': kjv_text,
                'confidence': kjv_conf,
                'weight': self.ensemble_weights['kjv_specialist'],
                'blocks': blocks
            })
        except Exception as e:
            logger.warning(f"KJV specialist failed: {e}")
        
        # Method 3: Blackletter Specialist
        try:
            bl_result = self.blackletter_specialist.process_blackletter_page(image)
            results.append({
                'method': 'blackletter_specialist',
                'text': bl_result.get('text', ''),
                'confidence': 0.7,  # Default confidence
                'weight': self.ensemble_weights['blackletter_specialist'],
                'statistics': bl_result.get('statistics', {})
            })
        except Exception as e:
            logger.warning(f"Blackletter specialist failed: {e}")
        
        # Combine results
        combined = self._combine_results(results)
        combined['layout'] = layout
        
        return combined
    
    def _combine_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Combine multiple OCR results using weighted voting"""
        
        if not results:
            return {'text': '', 'confidence': 0.0, 'method': 'none'}
        
        if len(results) == 1:
            return results[0]
        
        # For now, use weighted average based on confidence
        # In production, use more sophisticated voting/combination
        best_result = max(results, key=lambda r: r['confidence'] * r['weight'])
        
        # Collect alternatives
        alternatives = []
        for r in results:
            if r['method'] != best_result['method']:
                alternatives.append((r['text'][:100], r['confidence']))
        
        combined = {
            'text': best_result['text'],
            'confidence': best_result['confidence'],
            'method': f"ensemble_{best_result['method']}",
            'alternatives': alternatives,
            'all_results': results
        }
        
        return combined
    
    def _specialist_ocr(self, image: np.ndarray, layout: Any) -> Dict[str, Any]:
        """Use KJV specialist OCR"""
        
        blocks = self.kjv_specialist.identify_print_blocks(image)
        
        text_parts = []
        total_confidence = []
        
        for block in blocks:
            if block.text:
                text_parts.append(block.text)
                total_confidence.append(block.confidence)
        
        return {
            'text': ' '.join(text_parts),
            'confidence': np.mean(total_confidence) if total_confidence else 0.0,
            'method': 'kjv_specialist',
            'blocks': blocks,
            'layout': layout
        }
    
    def _improved_ocr(self, image: np.ndarray, layout: Any) -> Dict[str, Any]:
        """Use improved OCR engine"""
        
        result = self.improved_ocr.process_image(image)
        
        return {
            'text': result.get('text', ''),
            'confidence': result.get('confidence', 0.0),
            'method': 'improved_ocr',
            'layout': layout,
            'metadata': result
        }
    
    def _base_ocr(self, image: np.ndarray, layout: Any) -> Dict[str, Any]:
        """Use base OCR engine"""
        
        result = self.base_ocr.process_image(image)
        
        return {
            'text': result.get('text', '') if isinstance(result, dict) else str(result),
            'confidence': self._calculate_confidence(result.get('text', '') if isinstance(result, dict) else str(result), 'base'),
            'method': 'base_ocr',
            'layout': layout
        }
    
    def _calculate_confidence(self, text: str, method: str) -> float:
        """Calculate confidence score for OCR result"""
        
        if not text:
            return 0.0
        
        # Basic heuristics
        confidence = 1.0
        
        # Check for too many special characters
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' .,;:')
        if special_chars > len(text) * 0.2:
            confidence *= 0.7
        
        # Check for reasonable word lengths
        words = text.split()
        if words:
            avg_word_len = np.mean([len(w) for w in words])
            if avg_word_len < 2 or avg_word_len > 15:
                confidence *= 0.8
        
        # Method-specific adjustments
        method_multipliers = {
            'tesseract': 0.9,
            'base': 0.85,
            'specialist': 0.95
        }
        
        confidence *= method_multipliers.get(method, 1.0)
        
        return min(confidence, 1.0)
    
    def _apply_corrections(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learned corrections to OCR result"""
        
        text = result.get('text', '')
        corrections_applied = []
        
        if not text or not self.correction_model:
            result['corrections_applied'] = corrections_applied
            return result
        
        # Apply word-level corrections
        word_corrections = self.correction_model.get('word_corrections', {})
        for old_word, new_word in word_corrections.items():
            if old_word in text:
                text = text.replace(old_word, new_word)
                corrections_applied.append(f"{old_word}->{new_word}")
        
        # Apply context-based corrections
        import re
        context_rules = self.correction_model.get('context_rules', [])
        for rule in context_rules:
            pattern = rule['pattern']
            replacement = rule['replacement']
            text = re.sub(pattern, replacement, text)
        
        result['text'] = text
        result['corrections_applied'] = corrections_applied
        
        return result
    
    def _update_indices(self, result: Dict[str, Any]):
        """Update character and word indices for fast lookup"""
        
        text = result.get('text', '')
        if not text:
            return
        
        # Update word index
        words = text.split()
        for i, word in enumerate(words):
            if word not in self.word_index:
                self.word_index[word] = []
            self.word_index[word].append({
                'position': i,
                'context': ' '.join(words[max(0, i-2):min(len(words), i+3)])
            })
        
        # Update character frequency (for analysis)
        for char in text:
            if char.isalpha():
                self.character_index[char] = self.character_index.get(char, 0) + 1
    
    def fine_tune_with_corrections(self, training_data: List[Tuple[str, str]]):
        """
        Fine-tune the correction model with manual corrections
        
        Args:
            training_data: List of (incorrect, correct) text pairs
        """
        
        logger.info(f"Fine-tuning with {len(training_data)} correction pairs")
        
        # Learn character substitutions
        char_substitutions = {}
        
        for incorrect, correct in training_data:
            if len(incorrect) == len(correct):
                for i, (inc_char, cor_char) in enumerate(zip(incorrect, correct)):
                    if inc_char != cor_char:
                        if inc_char not in char_substitutions:
                            char_substitutions[inc_char] = {}
                        char_substitutions[inc_char][cor_char] = \
                            char_substitutions[inc_char].get(cor_char, 0) + 1
        
        # Normalize to probabilities
        for inc_char, corrections in char_substitutions.items():
            total = sum(corrections.values())
            for cor_char in corrections:
                corrections[cor_char] /= total
        
        # Update correction model
        self.correction_model['character_substitutions'].update(char_substitutions)
        
        # Learn word corrections
        word_corrections = {}
        for incorrect, correct in training_data:
            inc_words = incorrect.split()
            cor_words = correct.split()
            
            if len(inc_words) == len(cor_words):
                for inc_word, cor_word in zip(inc_words, cor_words):
                    if inc_word != cor_word:
                        word_corrections[inc_word] = cor_word
        
        self.correction_model['word_corrections'].update(word_corrections)
        
        logger.info(f"Learned {len(char_substitutions)} character substitutions")
        logger.info(f"Learned {len(word_corrections)} word corrections")
    
    def save_model(self, path: str):
        """Save the correction model and indices"""
        
        model_data = {
            'correction_model': self.correction_model,
            'ensemble_weights': self.ensemble_weights,
            'character_index': self.character_index,
            'word_index': self.word_index,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a saved model"""
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.correction_model = model_data['correction_model']
        self.ensemble_weights = model_data['ensemble_weights']
        self.character_index = model_data['character_index']
        self.word_index = model_data['word_index']
        
        logger.info(f"Model loaded from {path}")
    
    def evaluate(self, test_images: List[np.ndarray], 
                ground_truth: List[str]) -> Dict[str, float]:
        """Evaluate OCR performance"""
        
        from difflib import SequenceMatcher
        
        scores = []
        
        for image, truth in zip(test_images, ground_truth):
            result = self.process_page(image, method='ensemble')
            predicted = result['text']
            
            # Calculate similarity
            similarity = SequenceMatcher(None, predicted, truth).ratio()
            scores.append(similarity)
        
        return {
            'accuracy': np.mean(scores),
            'min_accuracy': np.min(scores),
            'max_accuracy': np.max(scores),
            'std_accuracy': np.std(scores)
        }


class IndexingSystem:
    """
    Indexing system for fast retrieval of print blocks and characters
    """
    
    def __init__(self):
        self.block_index = {}  # block_type -> list of blocks
        self.character_map = {}  # character -> list of occurrences
        self.word_map = {}  # word -> list of occurrences
        self.page_index = {}  # page_num -> content
        
    def index_page(self, page_num: int, ocr_result: Dict[str, Any]):
        """Index a page's OCR results"""
        
        # Index by page
        self.page_index[page_num] = {
            'text': ocr_result.get('text', ''),
            'confidence': ocr_result.get('confidence', 0.0),
            'method': ocr_result.get('method', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        # Index blocks if available
        if 'blocks' in ocr_result:
            for block in ocr_result['blocks']:
                block_type = block.block_type.value if hasattr(block, 'block_type') else 'unknown'
                if block_type not in self.block_index:
                    self.block_index[block_type] = []
                
                self.block_index[block_type].append({
                    'page': page_num,
                    'bbox': block.bbox if hasattr(block, 'bbox') else None,
                    'text': block.text if hasattr(block, 'text') else '',
                    'confidence': block.confidence if hasattr(block, 'confidence') else 0.0
                })
        
        # Index words
        text = ocr_result.get('text', '')
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,;:!?')
            if word_lower:
                if word_lower not in self.word_map:
                    self.word_map[word_lower] = []
                
                self.word_map[word_lower].append({
                    'page': page_num,
                    'position': i,
                    'context': ' '.join(words[max(0, i-3):min(len(words), i+4)])
                })
    
    def search_word(self, word: str) -> List[Dict]:
        """Search for a word in the index"""
        return self.word_map.get(word.lower(), [])
    
    def get_block_type(self, block_type: str) -> List[Dict]:
        """Get all blocks of a specific type"""
        return self.block_index.get(block_type, [])
    
    def save_index(self, path: str):
        """Save index to disk"""
        
        index_data = {
            'block_index': self.block_index,
            'character_map': self.character_map,
            'word_map': self.word_map,
            'page_index': self.page_index,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
    
    def load_index(self, path: str):
        """Load index from disk"""
        
        with open(path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.block_index = index_data['block_index']
        self.character_map = index_data['character_map']
        self.word_map = index_data['word_map']
        self.page_index = index_data['page_index']