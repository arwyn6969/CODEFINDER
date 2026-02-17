"""
Block Fingerprinter Service
============================

Computes forensic fingerprints for print blocks (character sorts, ornaments,
decorative initials). These fingerprints capture the unique wear patterns,
damage marks, and physical characteristics that identify a specific piece
of metal type or a woodblock.

Features extracted:
    1. Hu Moments        — rotation/scale-invariant shape descriptor
    2. Fourier Contour    — boundary shape encoding
    3. Edge Density       — Canny edge fraction (captures nicks/damage)
    4. Ink Density        — Black pixel ratio
    5. LBP Texture        — Local Binary Pattern histogram (ink micro-texture)
    6. Perceptual Hash    — pHash for fast pre-matching
    7. Damage Points      — Isolated defects (nicks, cracks, missing ink)

Usage:
    fp = BlockFingerprinter()
    features = fp.fingerprint(image_or_path)
    similarity = fp.compare(features1, features2)
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple, Union
import json

logger = logging.getLogger(__name__)

@dataclass
class BlockFingerprint:
    """Complete fingerprint for a single print block."""
    # Source info
    source_path: str = ""
    block_type: str = ""  # 'character', 'ornament', 'device', 'initial'
    
    # Shape descriptors
    hu_moments: List[float] = field(default_factory=list)       # 7 Hu moments (log-transformed)
    contour_fourier: List[float] = field(default_factory=list)  # First N Fourier descriptors
    
    # Damage indicators
    edge_density: float = 0.0       # Fraction of edge pixels
    ink_density: float = 0.0        # Fraction of black pixels
    damage_point_count: int = 0     # Number of isolated defects
    damage_points: List[Tuple[int, int]] = field(default_factory=list)  # (x,y) of damage
    
    # Texture
    lbp_histogram: List[float] = field(default_factory=list)    # LBP texture histogram
    
    # Fast match
    perceptual_hash: str = ""       # 64-bit pHash as hex string
    
    # Raw feature vector for ML
    feature_vector: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert tuples to lists for JSON
        d['damage_points'] = [list(p) for p in d['damage_points']]
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BlockFingerprint':
        d['damage_points'] = [tuple(p) for p in d.get('damage_points', [])]
        return cls(**d)


class BlockFingerprinter:
    """Compute forensic fingerprints for print blocks."""
    
    # Configuration
    NORMALIZE_SIZE = 128            # Normalize block to this square for comparison
    FOURIER_DESCRIPTORS = 32        # Number of Fourier contour descriptors
    LBP_RADIUS = 2                  # LBP radius
    LBP_POINTS = 16                 # LBP sampling points
    DAMAGE_MIN_AREA = 3             # Min area for a damage point
    DAMAGE_MAX_AREA = 200           # Max area for a damage point (vs text)
    
    def __init__(self):
        pass
    
    def fingerprint(self, image: Union[str, Path, np.ndarray], 
                    block_type: str = "unknown") -> BlockFingerprint:
        """
        Compute the complete forensic fingerprint for a print block image.
        
        Args:
            image: Path to image file or numpy array
            block_type: Type of block ('character', 'ornament', etc.)
            
        Returns:
            BlockFingerprint with all features computed
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            source_path = str(image)
        else:
            img = image.copy()
            source_path = "<array>"
            
        if img is None:
            logger.error(f"Could not read image: {source_path}")
            return BlockFingerprint(source_path=source_path, block_type=block_type)
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Normalize size for consistent comparison
        normalized = cv2.resize(gray, (self.NORMALIZE_SIZE, self.NORMALIZE_SIZE))
        
        # Binarize (Otsu)
        _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        fp = BlockFingerprint(source_path=source_path, block_type=block_type)
        
        # 1. Hu Moments
        fp.hu_moments = self._compute_hu_moments(binary)
        
        # 2. Fourier Contour Descriptors
        fp.contour_fourier = self._compute_fourier_descriptors(binary)
        
        # 3. Edge Density
        fp.edge_density = self._compute_edge_density(normalized)
        
        # 4. Ink Density
        fp.ink_density = self._compute_ink_density(binary)
        
        # 5. Damage Points
        fp.damage_point_count, fp.damage_points = self._detect_damage_points(binary)
        
        # 6. LBP Texture
        fp.lbp_histogram = self._compute_lbp(normalized)
        
        # 7. Perceptual Hash
        fp.perceptual_hash = self._compute_phash(normalized)
        
        # 8. Concatenated feature vector
        fp.feature_vector = self._build_feature_vector(fp)
        
        return fp
    
    def compare(self, fp1: BlockFingerprint, fp2: BlockFingerprint) -> Dict[str, float]:
        """
        Compare two fingerprints and return similarity scores.
        
        Returns dict with individual and aggregate scores (0-1, higher = more similar).
        """
        scores = {}
        
        # Hu Moments distance (lower = more similar)
        if fp1.hu_moments and fp2.hu_moments:
            hu_dist = np.sum(np.abs(np.array(fp1.hu_moments) - np.array(fp2.hu_moments)))
            scores['hu_similarity'] = max(0, 1.0 - hu_dist / 20.0)  # Normalize
        
        # Fourier descriptor distance
        if fp1.contour_fourier and fp2.contour_fourier:
            f1 = np.array(fp1.contour_fourier[:self.FOURIER_DESCRIPTORS])
            f2 = np.array(fp2.contour_fourier[:self.FOURIER_DESCRIPTORS])
            min_len = min(len(f1), len(f2))
            if min_len > 0:
                fourier_dist = np.linalg.norm(f1[:min_len] - f2[:min_len])
                scores['fourier_similarity'] = max(0, 1.0 - fourier_dist / 10.0)
        
        # Edge density difference
        scores['edge_similarity'] = 1.0 - abs(fp1.edge_density - fp2.edge_density)
        
        # Ink density difference
        scores['ink_similarity'] = 1.0 - abs(fp1.ink_density - fp2.ink_density)
        
        # LBP histogram comparison (chi-squared or correlation)
        if fp1.lbp_histogram and fp2.lbp_histogram:
            h1 = np.array(fp1.lbp_histogram, dtype=np.float32)
            h2 = np.array(fp2.lbp_histogram, dtype=np.float32)
            if len(h1) == len(h2) and len(h1) > 0:
                corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
                scores['lbp_similarity'] = max(0, corr)  # Correlation is -1 to 1
        
        # Perceptual hash Hamming distance
        if fp1.perceptual_hash and fp2.perceptual_hash:
            hamming = self._hamming_distance(fp1.perceptual_hash, fp2.perceptual_hash)
            scores['phash_similarity'] = max(0, 1.0 - hamming / 64.0)
        
        # Damage point count similarity
        max_dmg = max(fp1.damage_point_count, fp2.damage_point_count, 1)
        scores['damage_count_similarity'] = 1.0 - abs(fp1.damage_point_count - fp2.damage_point_count) / max_dmg
        
        # Aggregate score (weighted)
        weights = {
            'hu_similarity': 0.15,
            'fourier_similarity': 0.15,
            'edge_similarity': 0.10,
            'ink_similarity': 0.10,
            'lbp_similarity': 0.15,
            'phash_similarity': 0.25,
            'damage_count_similarity': 0.10,
        }
        
        total_weight = 0
        weighted_sum = 0
        for key, weight in weights.items():
            if key in scores:
                weighted_sum += scores[key] * weight
                total_weight += weight
        
        scores['aggregate'] = weighted_sum / total_weight if total_weight > 0 else 0
        
        return scores
    
    # -------------------------------------------------------------------------
    # Feature Extraction Methods
    # -------------------------------------------------------------------------
    
    def _compute_hu_moments(self, binary: np.ndarray) -> List[float]:
        """Compute 7 Hu moments (log-transformed for stability)."""
        moments = cv2.moments(binary)
        hu = cv2.HuMoments(moments).flatten()
        # Log-transform for better scaling
        log_hu = []
        for h in hu:
            if abs(h) > 0:
                log_hu.append(-np.sign(h) * np.log10(abs(h)))
            else:
                log_hu.append(0.0)
        return log_hu
    
    def _compute_fourier_descriptors(self, binary: np.ndarray) -> List[float]:
        """Compute Fourier descriptors of the largest contour."""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return [0.0] * self.FOURIER_DESCRIPTORS
        
        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Convert contour to complex numbers
        contour = contour.reshape(-1, 2).astype(np.float64)
        complex_contour = contour[:, 0] + 1j * contour[:, 1]
        
        # FFT
        fourier = np.fft.fft(complex_contour)
        
        # Take magnitudes, normalize by DC component
        magnitudes = np.abs(fourier)
        if magnitudes[0] > 0:
            magnitudes = magnitudes / magnitudes[0]
        
        # Return first N descriptors (skip DC)
        descriptors = magnitudes[1:self.FOURIER_DESCRIPTORS + 1].tolist()
        
        # Pad if contour was too short
        while len(descriptors) < self.FOURIER_DESCRIPTORS:
            descriptors.append(0.0)
        
        return descriptors
    
    def _compute_edge_density(self, gray: np.ndarray) -> float:
        """Compute fraction of edge pixels (Canny)."""
        edges = cv2.Canny(gray, 50, 150)
        return float(np.count_nonzero(edges)) / edges.size
    
    def _compute_ink_density(self, binary: np.ndarray) -> float:
        """Compute fraction of black (ink) pixels."""
        return float(np.count_nonzero(binary)) / binary.size
    
    def _detect_damage_points(self, binary: np.ndarray) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Detect isolated damage marks — small defects that indicate wear.
        
        These are tiny connected components that aren't part of the main
        character/ornament body. They represent:
        - Nicks in type edges
        - Cracks in woodblocks
        - Ink pooling artifacts
        - Missing ink regions (inverted)
        """
        # Invert to find white holes in black regions
        inverted = cv2.bitwise_not(binary)
        
        # Find small connected components in both positive and negative
        damage_points = []
        
        for target in [binary, inverted]:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                target, connectivity=8
            )
            
            for i in range(1, num_labels):  # Skip background
                area = stats[i, cv2.CC_STAT_AREA]
                if self.DAMAGE_MIN_AREA <= area <= self.DAMAGE_MAX_AREA:
                    cx, cy = int(centroids[i][0]), int(centroids[i][1])
                    damage_points.append((cx, cy))
        
        return len(damage_points), damage_points
    
    def _compute_lbp(self, gray: np.ndarray) -> List[float]:
        """Compute Local Binary Pattern histogram."""
        # Simple LBP implementation (not using skimage to avoid dependency)
        h, w = gray.shape
        r = self.LBP_RADIUS
        lbp = np.zeros((h - 2*r, w - 2*r), dtype=np.uint8)
        
        # 8-point LBP (simplified)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        
        for bit, (dy, dx) in enumerate(offsets):
            neighbor = gray[r+dy:h-r+dy, r+dx:w-r+dx]
            center = gray[r:h-r, r:w-r]
            lbp |= ((neighbor >= center).astype(np.uint8) << bit)
        
        # Histogram
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256), density=True)
        return hist.tolist()
    
    def _compute_phash(self, gray: np.ndarray) -> str:
        """Compute perceptual hash (pHash)."""
        # Resize to 32x32
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        
        # Convert to float
        img_float = resized.astype(np.float64)
        
        # DCT
        dct = cv2.dct(img_float)
        
        # Use top-left 8x8
        dct_low = dct[:8, :8]
        
        # Compute median (excluding DC)
        median = np.median(dct_low)
        
        # Binary hash
        hash_bits = (dct_low > median).flatten()
        
        # Convert to hex string
        hash_int = 0
        for bit in hash_bits:
            hash_int = (hash_int << 1) | int(bit)
        
        return format(hash_int, '016x')
    
    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """Compute Hamming distance between two hex hash strings."""
        try:
            val1 = int(hash1, 16)
            val2 = int(hash2, 16)
            xor = val1 ^ val2
            return bin(xor).count('1')
        except ValueError:
            return 64  # Max distance
    
    def _build_feature_vector(self, fp: BlockFingerprint) -> List[float]:
        """Concatenate all features into a single vector for ML."""
        vec = []
        vec.extend(fp.hu_moments)                       # 7
        vec.extend(fp.contour_fourier[:16])              # 16
        vec.append(fp.edge_density)                      # 1
        vec.append(fp.ink_density)                       # 1
        vec.append(float(fp.damage_point_count))         # 1 
        vec.extend(fp.lbp_histogram[:32])                # 32 (truncated)
        # Total: ~58 dimensions
        return vec
    
    def fingerprint_batch(self, image_dir: Union[str, Path], 
                          pattern: str = "*.jpg",
                          block_type: str = "unknown") -> List[BlockFingerprint]:
        """Fingerprint all images in a directory."""
        image_dir = Path(image_dir)
        files = sorted(image_dir.glob(pattern))
        
        results = []
        for f in files:
            fp = self.fingerprint(f, block_type)
            results.append(fp)
            
        logger.info(f"Fingerprinted {len(results)} blocks from {image_dir}")
        return results
