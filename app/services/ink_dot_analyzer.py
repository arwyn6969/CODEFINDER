"""
Ink Dot Analyzer: Sub-Character Micro-Texture Analysis
=======================================================
Forensic analysis of individual ink deposits within typeface characters.

This module provides post-processing analysis of extracted character images
from the Digital Type Case to detect:
- Individual ink dots/deposits
- Ink pooling at character edges
- Micro-pitting (gaps in ink coverage)
- Anomalous marks and artifacts

PROTOCOL: Non-disruptive extension - reads from existing Type Case output.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class InkDot:
    """Individual detected ink deposit."""
    x: int
    y: int
    radius: float
    area: float
    circularity: float
    intensity: float  # 0-255, lower = darker


@dataclass
class InkPoolingRegion:
    """Region of excess ink accumulation."""
    x: int
    y: int
    width: int
    height: int
    density: float  # 0-1
    edge_type: str  # 'top', 'bottom', 'left', 'right', 'corner'


@dataclass
class MicroPit:
    """Gap or hole in ink coverage."""
    x: int
    y: int
    width: int
    height: int
    area: float


@dataclass 
class AnomalousMark:
    """Unexpected mark not part of expected character form."""
    x: int
    y: int
    width: int
    height: int
    mark_type: str  # 'splatter', 'foreign', 'bleed', 'artifact'
    confidence: float


@dataclass
class InkTextureResult:
    """Complete analysis result for a single character."""
    character: str
    source_image: str
    total_dots: int
    dots: List[InkDot] = field(default_factory=list)
    pooling_regions: List[InkPoolingRegion] = field(default_factory=list)
    micro_pits: List[MicroPit] = field(default_factory=list)
    anomalous_marks: List[AnomalousMark] = field(default_factory=list)
    average_density: float = 0.0
    density_variance: float = 0.0
    irregularity_score: float = 0.0  # 0-1, higher = more irregular


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class InkDotAnalyzer:
    """
    Forensic analysis of ink deposits within extracted character images.
    
    Usage:
        analyzer = InkDotAnalyzer()
        result = analyzer.analyze_character("path/to/char_image.png")
        
        # Or analyze entire type case
        results = analyzer.analyze_type_case("reports/digital_type_case/")
    """
    
    def __init__(self, 
                 min_dot_area: int = 2,
                 max_dot_area: int = 100,
                 density_threshold: int = 128,
                 edge_margin: int = 3,
                 use_adaptive_threshold: bool = True,
                 generate_overlays: bool = True,
                 max_images_per_char: int = None):
        """
        Initialize the analyzer with detection parameters.
        
        Args:
            min_dot_area: Minimum area (pixels) for a detected dot
            max_dot_area: Maximum area (pixels) for a detected dot  
            density_threshold: Threshold for ink (0-255, lower = darker)
            edge_margin: Margin (pixels) to check for edge pooling
            use_adaptive_threshold: Use Otsu's method instead of fixed threshold
            generate_overlays: Create annotated images showing detections
            max_images_per_char: Limit images per character (None = all)
        """
        self.min_dot_area = min_dot_area
        self.max_dot_area = max_dot_area
        self.density_threshold = density_threshold
        self.edge_margin = edge_margin
        self.use_adaptive_threshold = use_adaptive_threshold
        self.generate_overlays = generate_overlays
        self.max_images_per_char = max_images_per_char
        
        # Configure blob detector for ink dots
        self.blob_params = cv2.SimpleBlobDetector_Params()
        self.blob_params.filterByArea = True
        self.blob_params.minArea = min_dot_area
        self.blob_params.maxArea = max_dot_area
        self.blob_params.filterByCircularity = False
        self.blob_params.filterByConvexity = False
        self.blob_params.filterByInertia = False
        self.blob_params.filterByColor = True
        self.blob_params.blobColor = 0  # Detect dark blobs
        
        self.blob_detector = cv2.SimpleBlobDetector_create(self.blob_params)
    
    def analyze_character(self, image_path: str) -> InkTextureResult:
        """
        Analyze a single character image for ink dot patterns.
        
        Args:
            image_path: Path to extracted character image
            
        Returns:
            InkTextureResult with all detected features
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and preprocess
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Extract character from filename (format: char_id_page_x_y.png)
        char = path.stem.split('_')[0] if '_' in path.stem else '?'
        
        # Run analysis pipeline
        dots = self._detect_dots(img)
        pooling = self._detect_pooling(img)
        pits = self._detect_micro_pits(img)
        anomalies = self._detect_anomalies(img)
        
        # Calculate density statistics
        avg_density, density_var = self._calculate_density_stats(img)
        
        # Calculate irregularity score (now size-normalized)
        irregularity = self._calculate_irregularity(
            img, dots, pooling, pits, anomalies, density_var
        )
        
        return InkTextureResult(
            character=char,
            source_image=str(path),
            total_dots=len(dots),
            dots=dots,
            pooling_regions=pooling,
            micro_pits=pits,
            anomalous_marks=anomalies,
            average_density=avg_density,
            density_variance=density_var,
            irregularity_score=irregularity
        )
    
    def _detect_dots(self, img: np.ndarray) -> List[InkDot]:
        """Detect individual ink dots using multi-scale blob detection."""
        dots = []
        
        # Multi-scale detection for better coverage
        scales = [1.0, 1.5, 2.0]
        
        for scale in scales:
            # Scale image if needed
            if scale != 1.0:
                scaled_img = cv2.resize(img, None, fx=scale, fy=scale, 
                                        interpolation=cv2.INTER_LINEAR)
            else:
                scaled_img = img
            
            # Invert for blob detection (blobs should be white)
            inverted = cv2.bitwise_not(scaled_img)
            
            # Use adaptive or fixed thresholding
            if self.use_adaptive_threshold:
                # Otsu's adaptive threshold
                _, binary = cv2.threshold(inverted, 0, 255, 
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                keypoints = self.blob_detector.detect(binary)
            else:
                # Multi-level fixed thresholding
                keypoints = []
                for thresh_level in [100, 150, 200]:
                    _, binary = cv2.threshold(inverted, thresh_level, 255, 
                                              cv2.THRESH_BINARY)
                    keypoints.extend(self.blob_detector.detect(binary))
            
            for kp in keypoints:
                # Scale coordinates back to original
                x = int(kp.pt[0] / scale)
                y = int(kp.pt[1] / scale)
                radius = (kp.size / 2) / scale
                area = np.pi * radius * radius
                
                # Get intensity at center (from original image)
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    intensity = float(img[y, x])
                else:
                    intensity = 128.0
                
                dots.append(InkDot(
                    x=x, y=y,
                    radius=radius,
                    area=area,
                    circularity=1.0,
                    intensity=intensity
                ))
        
        # Deduplicate (dots found at multiple scales/thresholds)
        unique_dots = self._deduplicate_dots(dots)
        return unique_dots
    
    def _deduplicate_dots(self, dots: List[InkDot], 
                          distance_thresh: float = 3.0) -> List[InkDot]:
        """Remove duplicate dots found at different threshold levels."""
        if not dots:
            return []
        
        unique = [dots[0]]
        for dot in dots[1:]:
            is_dup = False
            for existing in unique:
                dist = np.sqrt((dot.x - existing.x)**2 + (dot.y - existing.y)**2)
                if dist < distance_thresh:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(dot)
        return unique
    
    def _detect_pooling(self, img: np.ndarray) -> List[InkPoolingRegion]:
        """Detect ink pooling at character edges."""
        pooling = []
        h, w = img.shape
        
        # Apply morphological gradient to find edges
        kernel = np.ones((3, 3), np.uint8)
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        
        # Check edge regions
        margin = self.edge_margin
        edge_checks = [
            ('top', img[:margin, :]),
            ('bottom', img[-margin:, :]),
            ('left', img[:, :margin]),
            ('right', img[:, -margin:]),
        ]
        
        for edge_type, region in edge_checks:
            # Calculate ink density in this edge region
            ink_pixels = np.sum(region < self.density_threshold)
            total_pixels = region.size
            density = ink_pixels / total_pixels if total_pixels > 0 else 0
            
            # If density is significantly higher than average, it's pooling
            if density > 0.5:
                if edge_type == 'top':
                    pooling.append(InkPoolingRegion(
                        x=0, y=0, width=w, height=margin,
                        density=density, edge_type=edge_type
                    ))
                elif edge_type == 'bottom':
                    pooling.append(InkPoolingRegion(
                        x=0, y=h-margin, width=w, height=margin,
                        density=density, edge_type=edge_type
                    ))
                elif edge_type == 'left':
                    pooling.append(InkPoolingRegion(
                        x=0, y=0, width=margin, height=h,
                        density=density, edge_type=edge_type
                    ))
                elif edge_type == 'right':
                    pooling.append(InkPoolingRegion(
                        x=w-margin, y=0, width=margin, height=h,
                        density=density, edge_type=edge_type
                    ))
        
        return pooling
    
    def _detect_micro_pits(self, img: np.ndarray) -> List[MicroPit]:
        """Detect microscopic gaps in ink coverage."""
        pits = []
        
        # Threshold to binary (ink = black)
        _, binary = cv2.threshold(img, self.density_threshold, 255, cv2.THRESH_BINARY)
        
        # Invert so ink areas are white
        ink_mask = cv2.bitwise_not(binary)
        
        # Find contours of ink regions
        contours, _ = cv2.findContours(
            ink_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Look for small holes (inner contours)
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_dot_area < area < self.max_dot_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if this is inside a larger ink region
                # (simplified heuristic: not at image edge)
                if x > 2 and y > 2 and x + w < img.shape[1] - 2 and y + h < img.shape[0] - 2:
                    pits.append(MicroPit(
                        x=x, y=y, width=w, height=h, area=area
                    ))
        
        return pits
    
    def _detect_anomalies(self, img: np.ndarray) -> List[AnomalousMark]:
        """Detect anomalous marks outside expected character form."""
        anomalies = []
        h, w = img.shape
        
        # Use connected components to find isolated marks
        _, binary = cv2.threshold(img, self.density_threshold, 255, cv2.THRESH_BINARY_INV)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        # Find the main character component (largest)
        areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
        if len(areas) == 0:
            return anomalies
        
        main_idx = np.argmax(areas) + 1  # +1 because we skipped background
        main_area = stats[main_idx, cv2.CC_STAT_AREA]
        
        # Any component much smaller than main and isolated is likely anomalous
        for i in range(1, num_labels):
            if i == main_idx:
                continue
            
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Small isolated mark
            if area < main_area * 0.1 and area > self.min_dot_area:
                # Classify type based on position and shape
                if x < 3 or x + cw > w - 3 or y < 3 or y + ch > h - 3:
                    mark_type = 'bleed'  # At edge
                elif cw / ch > 3 or ch / cw > 3:
                    mark_type = 'splatter'  # Elongated
                else:
                    mark_type = 'artifact'
                
                confidence = 1.0 - (area / main_area)  # Higher confidence for smaller marks
                
                anomalies.append(AnomalousMark(
                    x=x, y=y, width=cw, height=ch,
                    mark_type=mark_type,
                    confidence=min(confidence, 0.95)
                ))
        
        return anomalies
    
    def _calculate_density_stats(self, img: np.ndarray) -> Tuple[float, float]:
        """Calculate ink density statistics."""
        # Only consider ink pixels (below threshold)
        ink_mask = img < self.density_threshold
        if not np.any(ink_mask):
            return 0.0, 0.0
        
        ink_values = img[ink_mask]
        avg_density = float(np.mean(255 - ink_values))  # Invert so higher = denser
        density_var = float(np.var(ink_values))
        
        return avg_density, density_var
    
    def _calculate_irregularity(self, 
                                img: np.ndarray,
                                dots: List[InkDot],
                                pooling: List[InkPoolingRegion],
                                pits: List[MicroPit],
                                anomalies: List[AnomalousMark],
                                density_var: float) -> float:
        """Calculate overall irregularity score (0-1), normalized by character area."""
        score = 0.0
        
        # Calculate character ink area for normalization
        if self.use_adaptive_threshold:
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(img, self.density_threshold, 255, cv2.THRESH_BINARY_INV)
        char_area = max(np.sum(binary > 0), 1)  # Avoid division by zero
        
        # Dots per 1000 ink pixels (normalized by character size)
        if len(dots) > 0:
            dot_density = len(dots) / char_area * 1000
            score += min(dot_density / 5.0, 0.25)
        
        # Pooling indicates irregular ink distribution
        if len(pooling) > 0:
            score += min(len(pooling) * 0.1, 0.25)
        
        # Micro-pits per 1000 ink pixels (normalized)
        if len(pits) > 0:
            pit_density = len(pits) / char_area * 1000
            score += min(pit_density / 2.0, 0.2)
        
        # Anomalies are direct irregularities
        if len(anomalies) > 0:
            score += min(len(anomalies) * 0.15, 0.3)
        
        # High density variance indicates uneven inking
        if density_var > 500:
            score += 0.1
        
        return min(score, 1.0)
    
    def _save_annotated_image(self, source_path: str, result: InkTextureResult, 
                              output_path: str) -> None:
        """
        Create annotated image showing detected dots and anomalies.
        
        Red circles = detected ink dots
        Yellow rectangles = anomalous marks
        Green rectangles = pooling regions
        """
        img = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        
        # Convert to color for annotations
        annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Scale up for better visibility of small features
        scale = 4
        h, w = annotated.shape[:2]
        annotated = cv2.resize(annotated, (w * scale, h * scale), 
                               interpolation=cv2.INTER_NEAREST)
        
        # Draw detected dots (red circles)
        for dot in result.dots:
            center = (int(dot.x * scale), int(dot.y * scale))
            radius = max(int(dot.radius * scale) + 2, 3)
            cv2.circle(annotated, center, radius, (0, 0, 255), 1)
        
        # Draw anomalous marks (yellow rectangles)
        for anom in result.anomalous_marks:
            pt1 = (int(anom.x * scale), int(anom.y * scale))
            pt2 = (int((anom.x + anom.width) * scale), 
                   int((anom.y + anom.height) * scale))
            cv2.rectangle(annotated, pt1, pt2, (0, 255, 255), 1)
        
        # Draw pooling regions (green rectangles)
        for pool in result.pooling_regions:
            pt1 = (int(pool.x * scale), int(pool.y * scale))
            pt2 = (int((pool.x + pool.width) * scale), 
                   int((pool.y + pool.height) * scale))
            cv2.rectangle(annotated, pt1, pt2, (0, 255, 0), 1)
        
        # Add legend
        cv2.putText(annotated, f"Dots:{result.total_dots} Anom:{len(result.anomalous_marks)}", 
                    (5, h * scale - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imwrite(output_path, annotated)
    
    def analyze_type_case(self, 
                          type_case_dir: str,
                          output_dir: str = None) -> Dict[str, Any]:
        """
        Analyze all characters in a Digital Type Case.
        
        Args:
            type_case_dir: Path to type case directory with images/
            output_dir: Optional output directory for results
            
        Returns:
            Summary dictionary with aggregated results
        """
        tc_path = Path(type_case_dir)
        images_dir = tc_path / "images"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = tc_path.parent / "ink_dot_analysis"
        out_path.mkdir(parents=True, exist_ok=True)
        
        results_by_char = defaultdict(list)
        anomaly_summary = []
        total_analyzed = 0
        
        # Process each character subdirectory
        for char_dir in sorted(images_dir.iterdir()):
            if not char_dir.is_dir():
                continue
            
            char_name = char_dir.name
            logger.info(f"Analyzing character: {char_name}")
            
            # Get all images (or limited if max_images_per_char set)
            images = list(char_dir.glob("*.png"))
            if self.max_images_per_char:
                images = images[:self.max_images_per_char]
            
            for img_path in images:
                try:
                    result = self.analyze_character(str(img_path))
                    results_by_char[char_name].append(result)
                    total_analyzed += 1
                    
                    # Generate overlay for high-irregularity cases
                    if result.irregularity_score > 0.5 and self.generate_overlays:
                        overlay_dir = out_path / "overlays" / char_name
                        overlay_dir.mkdir(parents=True, exist_ok=True)
                        overlay_path = overlay_dir / f"{img_path.stem}_annotated.png"
                        self._save_annotated_image(str(img_path), result, str(overlay_path))
                    
                    # Track high-irregularity cases
                    if result.irregularity_score > 0.5:
                        anomaly_summary.append({
                            "character": result.character,
                            "path": str(img_path),
                            "score": result.irregularity_score,
                            "dots": result.total_dots,
                            "anomalies": len(result.anomalous_marks)
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze {img_path}: {e}")
        
        # Generate summary
        summary = {
            "total_analyzed": total_analyzed,
            "characters": len(results_by_char),
            "high_irregularity_count": len(anomaly_summary),
            "anomalies": anomaly_summary[:100],  # Top 100
        }
        
        # Save results
        summary_path = out_path / "analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Analysis complete. {total_analyzed} images analyzed.")
        logger.info(f"Results saved to: {out_path}")
        
        return summary


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ink Dot Analyzer - Sub-character micro-texture analysis"
    )
    parser.add_argument(
        "source", 
        help="Path to character image or type case directory"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for results"
    )
    parser.add_argument(
        "--single", "-s",
        action="store_true",
        help="Analyze single image instead of type case"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    analyzer = InkDotAnalyzer()
    
    if args.single:
        result = analyzer.analyze_character(args.source)
        print(f"\n=== Ink Dot Analysis: {result.character} ===")
        print(f"Source: {result.source_image}")
        print(f"Total dots detected: {result.total_dots}")
        print(f"Pooling regions: {len(result.pooling_regions)}")
        print(f"Micro-pits: {len(result.micro_pits)}")
        print(f"Anomalous marks: {len(result.anomalous_marks)}")
        print(f"Average density: {result.average_density:.1f}")
        print(f"Irregularity score: {result.irregularity_score:.3f}")
    else:
        summary = analyzer.analyze_type_case(args.source, args.output)
        print(f"\n=== Type Case Analysis Summary ===")
        print(f"Total analyzed: {summary['total_analyzed']}")
        print(f"Characters: {summary['characters']}")
        print(f"High irregularity: {summary['high_irregularity_count']}")
