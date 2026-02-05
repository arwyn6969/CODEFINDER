"""
Training Data Generator for KJV 1611 OCR Model
===============================================
Generates high-quality training data from KJV 1611 pages with ground truth labels.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import fitz
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import hashlib
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CharacterSample:
    """Represents a single character training sample"""
    image: np.ndarray
    label: str
    confidence: float
    source_page: int
    bbox: Tuple[int, int, int, int]
    char_type: str  # 'blackletter', 'roman', 'italic', 'decorative'
    is_ligature: bool
    ligature_components: Optional[List[str]]
    context: str  # surrounding text for context
    hash_id: str
    

@dataclass
class PrintBlockSample:
    """Represents a print block training sample"""
    image: np.ndarray
    block_type: str
    text: str
    source_page: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    metadata: Dict[str, Any]
    

class TrainingDataGenerator:
    """
    Generates training data from KJV 1611 pages with manual annotations
    and semi-automatic labeling.
    """
    
    def __init__(self, pdf_path: str, output_dir: str):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.chars_dir = self.output_dir / "characters"
        self.blocks_dir = self.output_dir / "blocks"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir in [self.chars_dir, self.blocks_dir, self.metadata_dir]:
            dir.mkdir(exist_ok=True)
        
        # Known ground truth for specific pages (manually verified)
        self.ground_truth = self._load_ground_truth()
        
        # Character mapping for blackletter
        self.char_mapping = self._initialize_char_mapping()
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'characters': {},
            'ligatures': 0,
            'block_types': {},
            'pages_processed': 0
        }
    
    def _load_ground_truth(self) -> Dict[int, Dict[str, Any]]:
        """Load or create ground truth annotations"""
        
        ground_truth_file = self.metadata_dir / "ground_truth.json"
        
        if ground_truth_file.exists():
            with open(ground_truth_file, 'r') as f:
                return json.load(f)
        
        # Initialize with known text samples
        ground_truth = {
            10: {  # Genesis page
                'title': 'The first Booke of Moses',
                'subtitle': 'called Genesis',
                'chapter_1_start': 'In the beginning God created the Heaven, and the Earth',
                'known_words': ['beginning', 'God', 'created', 'Heaven', 'Earth'],
                'decorative_initial': 'I',
                'layout': 'two_column'
            },
            50: {  # Regular page
                'layout': 'two_column',
                'has_verse_numbers': True,
                'has_marginal_notes': True
            }
        }
        
        # Save initial ground truth
        with open(ground_truth_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        return ground_truth
    
    def _initialize_char_mapping(self) -> Dict[str, List[str]]:
        """Initialize character variations mapping"""
        
        return {
            's': ['s', 'ſ', 'ʃ'],
            'f': ['f', 'ſ'],
            'i': ['i', 'j', 'ı'],
            'u': ['u', 'v'],
            'v': ['v', 'u'],
            'w': ['w', 'vv', 'uu'],
            'th': ['th', 'þ', 'ð', 'y'],
            # Ligatures
            'ff': ['ff', 'ﬀ'],
            'fi': ['fi', 'ﬁ'],
            'fl': ['fl', 'ﬂ'],
            'ffi': ['ffi', 'ﬃ'],
            'ffl': ['ffl', 'ﬄ'],
            'st': ['st', 'ﬆ'],
            'ct': ['ct'],
            'ae': ['ae', 'æ'],
            'oe': ['oe', 'œ']
        }
    
    def extract_page_samples(self, page_num: int, 
                            dpi: int = 300) -> Tuple[List[CharacterSample], List[PrintBlockSample]]:
        """Extract training samples from a single page"""
        
        logger.info(f"Extracting samples from page {page_num}")
        
        # Open PDF and get page
        pdf_document = fitz.open(str(self.pdf_path))
        page = pdf_document[page_num]
        
        # Render at high DPI
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to numpy array
        img_data = np.frombuffer(pix.samples, dtype=np.uint8)
        img_data = img_data.reshape(pix.height, pix.width, pix.n)
        
        if pix.n == 4:  # RGBA
            image = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:  # RGB
            image = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        else:
            image = img_data
        
        pdf_document.close()
        
        # Extract character samples
        char_samples = self._extract_characters(image, page_num)
        
        # Extract print block samples
        block_samples = self._extract_print_blocks(image, page_num)
        
        return char_samples, block_samples
    
    def _extract_characters(self, image: np.ndarray, page_num: int) -> List[CharacterSample]:
        """Extract individual character samples from page"""
        
        from agents.blackletter_specialist import BlackletterSpecialist
        
        specialist = BlackletterSpecialist()
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Enhance and threshold
        enhanced = specialist.enhance_image_for_blackletter(gray, 'adaptive')
        binary = specialist.adaptive_threshold_for_blackletter(enhanced, 'auto')
        
        # Segment characters
        char_segments = specialist.segment_blackletter_characters(binary)
        
        samples = []
        
        for segment in char_segments:
            x, y, w, h = segment['bbox']
            
            # Extract character image
            char_img = binary[y:y+h, x:x+w]
            
            # Skip if too small
            if w < 5 or h < 10:
                continue
            
            # Generate unique ID
            hash_id = hashlib.md5(char_img.tobytes()).hexdigest()[:8]
            
            # Determine character type
            char_type = 'blackletter'  # Default
            if segment.get('is_ligature'):
                char_type = 'ligature'
            elif h > 50:
                char_type = 'decorative'
            
            # Create sample (label will be added later)
            sample = CharacterSample(
                image=char_img,
                label='',  # To be labeled
                confidence=0.0,
                source_page=page_num,
                bbox=(x, y, w, h),
                char_type=char_type,
                is_ligature=segment.get('is_ligature', False),
                ligature_components=None,
                context='',  # Add surrounding text context
                hash_id=hash_id
            )
            
            samples.append(sample)
        
        return samples
    
    def _extract_print_blocks(self, image: np.ndarray, page_num: int) -> List[PrintBlockSample]:
        """Extract print block samples from page"""
        
        from app.services.kjv_1611_specialist import KJV1611Specialist
        
        specialist = KJV1611Specialist()
        
        # Identify print blocks
        blocks = specialist.identify_print_blocks(image)
        
        samples = []
        
        for block in blocks:
            x, y, w, h = block.bbox
            
            # Extract block image
            block_img = image[y:y+h, x:x+w]
            
            # Create sample
            sample = PrintBlockSample(
                image=block_img,
                block_type=block.block_type.value,
                text=block.text,
                source_page=page_num,
                bbox=block.bbox,
                confidence=block.confidence,
                metadata={
                    'font_size': block.font_size_estimate,
                    'characteristics': block.characteristics
                }
            )
            
            samples.append(sample)
        
        return samples
    
    def create_synthetic_samples(self, base_samples: List[CharacterSample]) -> List[CharacterSample]:
        """Create synthetic training samples through augmentation"""
        
        synthetic_samples = []
        
        for sample in base_samples:
            # Original
            synthetic_samples.append(sample)
            
            # Rotation augmentation
            for angle in [-5, -2, 2, 5]:
                rotated = self._rotate_image(sample.image, angle)
                aug_sample = CharacterSample(
                    image=rotated,
                    label=sample.label,
                    confidence=sample.confidence * 0.9,  # Slightly lower confidence
                    source_page=sample.source_page,
                    bbox=sample.bbox,
                    char_type=sample.char_type,
                    is_ligature=sample.is_ligature,
                    ligature_components=sample.ligature_components,
                    context=sample.context,
                    hash_id=sample.hash_id + f"_rot{angle}"
                )
                synthetic_samples.append(aug_sample)
            
            # Noise augmentation
            noisy = self._add_noise(sample.image)
            noise_sample = CharacterSample(
                image=noisy,
                label=sample.label,
                confidence=sample.confidence * 0.85,
                source_page=sample.source_page,
                bbox=sample.bbox,
                char_type=sample.char_type,
                is_ligature=sample.is_ligature,
                ligature_components=sample.ligature_components,
                context=sample.context,
                hash_id=sample.hash_id + "_noise"
            )
            synthetic_samples.append(noise_sample)
            
            # Erosion/Dilation (simulate print quality variation)
            kernel = np.ones((2,2), np.uint8)
            
            # Eroded (thinner characters)
            eroded = cv2.erode(sample.image, kernel, iterations=1)
            eroded_sample = CharacterSample(
                image=eroded,
                label=sample.label,
                confidence=sample.confidence * 0.9,
                source_page=sample.source_page,
                bbox=sample.bbox,
                char_type=sample.char_type,
                is_ligature=sample.is_ligature,
                ligature_components=sample.ligature_components,
                context=sample.context,
                hash_id=sample.hash_id + "_eroded"
            )
            synthetic_samples.append(eroded_sample)
            
            # Dilated (thicker characters)
            dilated = cv2.dilate(sample.image, kernel, iterations=1)
            dilated_sample = CharacterSample(
                image=dilated,
                label=sample.label,
                confidence=sample.confidence * 0.9,
                source_page=sample.source_page,
                bbox=sample.bbox,
                char_type=sample.char_type,
                is_ligature=sample.is_ligature,
                ligature_components=sample.ligature_components,
                context=sample.context,
                hash_id=sample.hash_id + "_dilated"
            )
            synthetic_samples.append(dilated_sample)
        
        return synthetic_samples
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate
        rotated = cv2.warpAffine(image, M, (w, h), 
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=255)
        
        return rotated
    
    def _add_noise(self, image: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
        """Add salt and pepper noise to image"""
        
        noisy = image.copy()
        h, w = noisy.shape[:2]
        
        # Salt noise (white pixels)
        num_salt = int(noise_level * h * w)
        coords = [np.random.randint(0, i, num_salt) for i in (h, w)]
        noisy[coords[0], coords[1]] = 255
        
        # Pepper noise (black pixels)
        num_pepper = int(noise_level * h * w)
        coords = [np.random.randint(0, i, num_pepper) for i in (h, w)]
        noisy[coords[0], coords[1]] = 0
        
        return noisy
    
    def semi_automatic_labeling(self, samples: List[CharacterSample]) -> List[CharacterSample]:
        """Semi-automatic labeling using OCR with manual verification"""
        
        import pytesseract
        
        labeled_samples = []
        
        for sample in samples:
            # Try OCR first
            try:
                ocr_text = pytesseract.image_to_string(
                    sample.image,
                    config='--psm 10 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                ).strip()
                
                # Apply blackletter corrections
                if ocr_text:
                    # Common corrections
                    corrections = {
                        'f': 's',  # Long s
                        'l': 'i',
                        'j': 'i',
                        'vv': 'w'
                    }
                    
                    for old, new in corrections.items():
                        if ocr_text == old:
                            ocr_text = new
                            break
                
                sample.label = ocr_text if ocr_text else '?'
                sample.confidence = 0.7 if ocr_text else 0.0
                
            except Exception:
                sample.label = '?'
                sample.confidence = 0.0
            
            labeled_samples.append(sample)
        
        return labeled_samples
    
    def save_samples(self, char_samples: List[CharacterSample], 
                    block_samples: List[PrintBlockSample]):
        """Save samples to disk"""
        
        # Save character samples
        for i, sample in enumerate(char_samples):
            # Save image
            img_path = self.chars_dir / f"{sample.hash_id}.png"
            cv2.imwrite(str(img_path), sample.image)
            
            # Save metadata
            metadata = {
                'label': sample.label,
                'confidence': sample.confidence,
                'source_page': sample.source_page,
                'bbox': sample.bbox,
                'char_type': sample.char_type,
                'is_ligature': sample.is_ligature,
                'ligature_components': sample.ligature_components,
                'context': sample.context,
                'image_path': str(img_path.name)
            }
            
            meta_path = self.metadata_dir / f"char_{sample.hash_id}.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update statistics
            label = sample.label if sample.label else '?'
            self.stats['characters'][label] = self.stats['characters'].get(label, 0) + 1
            if sample.is_ligature:
                self.stats['ligatures'] += 1
        
        # Save block samples
        for i, sample in enumerate(block_samples):
            # Generate ID
            block_id = f"block_{sample.source_page}_{i}"
            
            # Save image
            img_path = self.blocks_dir / f"{block_id}.png"
            cv2.imwrite(str(img_path), sample.image)
            
            # Save metadata
            metadata = {
                'block_type': sample.block_type,
                'text': sample.text,
                'source_page': sample.source_page,
                'bbox': sample.bbox,
                'confidence': sample.confidence,
                'metadata': sample.metadata,
                'image_path': str(img_path.name)
            }
            
            meta_path = self.metadata_dir / f"{block_id}.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update statistics
            self.stats['block_types'][sample.block_type] = \
                self.stats['block_types'].get(sample.block_type, 0) + 1
        
        self.stats['total_samples'] += len(char_samples) + len(block_samples)
    
    def generate_dataset(self, page_numbers: List[int], 
                        augment: bool = True) -> Dict[str, Any]:
        """Generate complete training dataset from specified pages"""
        
        logger.info(f"Generating dataset from {len(page_numbers)} pages")
        
        all_char_samples = []
        all_block_samples = []
        
        for page_num in page_numbers:
            logger.info(f"Processing page {page_num}")
            
            # Extract samples
            char_samples, block_samples = self.extract_page_samples(page_num)
            
            # Semi-automatic labeling
            char_samples = self.semi_automatic_labeling(char_samples)
            
            # Augmentation
            if augment:
                char_samples = self.create_synthetic_samples(char_samples)
            
            all_char_samples.extend(char_samples)
            all_block_samples.extend(block_samples)
            
            self.stats['pages_processed'] += 1
        
        # Save all samples
        self.save_samples(all_char_samples, all_block_samples)
        
        # Save statistics
        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Dataset generation complete. Total samples: {self.stats['total_samples']}")
        
        return self.stats
    
    def create_train_val_split(self, train_ratio: float = 0.8) -> Dict[str, List[str]]:
        """Create training and validation splits"""
        
        # Get all character samples
        char_files = list(self.chars_dir.glob("*.png"))
        
        # Shuffle
        import random
        random.shuffle(char_files)
        
        # Split
        split_idx = int(len(char_files) * train_ratio)
        train_files = char_files[:split_idx]
        val_files = char_files[split_idx:]
        
        split = {
            'train': [str(f.name) for f in train_files],
            'val': [str(f.name) for f in val_files],
            'train_count': len(train_files),
            'val_count': len(val_files)
        }
        
        # Save split
        split_file = self.output_dir / "train_val_split.json"
        with open(split_file, 'w') as f:
            json.dump(split, f, indent=2)
        
        logger.info(f"Created train/val split: {len(train_files)}/{len(val_files)}")
        
        return split