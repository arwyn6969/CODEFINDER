"""
Enhanced Column Layout Detection Agent
=======================================
Specialized agent for detecting and analyzing two-column layouts
typical of the 1611 King James Bible.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from scipy import signal, ndimage
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Detailed information about a text column"""
    index: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    width: int
    height: int
    text_density: float
    line_count: int
    avg_line_height: float
    confidence: float
    

@dataclass 
class LayoutAnalysis:
    """Complete layout analysis results"""
    layout_type: str  # 'single', 'two_column', 'multi_column', 'irregular'
    columns: List[ColumnInfo]
    gutter_info: Optional[Dict[str, Any]] = None
    margins: Dict[str, int] = field(default_factory=dict)
    text_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)
    confidence: float = 0.0
    

class ColumnLayoutDetector:
    """
    Advanced column detection for historical documents.
    Optimized for 1611 KJV two-column layout with decorative elements.
    """
    
    def __init__(self):
        self.min_column_width = 200
        self.max_column_width = 2000
        self.min_gutter_width = 20
        self.max_gutter_width = 200
        self.min_text_density = 0.1
        
    def detect_columns(self, image: np.ndarray, 
                       method: str = 'advanced') -> LayoutAnalysis:
        """
        Main entry point for column detection
        
        Args:
            image: Input page image
            method: 'simple', 'projection', 'advanced', 'ml'
        """
        
        logger.info(f"Detecting columns using {method} method...")
        
        if method == 'simple':
            return self._simple_column_detection(image)
        elif method == 'projection':
            return self._projection_based_detection(image)
        elif method == 'advanced':
            return self._advanced_column_detection(image)
        elif method == 'ml':
            return self._ml_based_detection(image)
        else:
            return self._advanced_column_detection(image)
    
    def _simple_column_detection(self, image: np.ndarray) -> LayoutAnalysis:
        """Simple vertical projection method"""
        
        gray = self._prepare_image(image)
        binary = self._binarize_image(gray)
        
        # Vertical projection
        vertical_sum = np.sum(binary, axis=0)
        
        # Find gaps
        gaps = self._find_gaps(vertical_sum)
        
        # Convert to columns
        columns = self._gaps_to_columns(gaps, image.shape)
        
        return LayoutAnalysis(
            layout_type=self._determine_layout_type(len(columns)),
            columns=columns,
            confidence=0.7
        )
    
    def _projection_based_detection(self, image: np.ndarray) -> LayoutAnalysis:
        """Enhanced projection with smoothing and peak detection"""
        
        gray = self._prepare_image(image)
        binary = self._binarize_image(gray)
        
        # Get vertical projection
        vertical_proj = np.sum(binary, axis=0)
        
        # Smooth the projection
        window_size = 21
        smoothed = signal.savgol_filter(vertical_proj, window_size, 3)
        
        # Find valleys (potential column gaps)
        valleys = self._find_valleys(smoothed)
        
        # Validate valleys as column boundaries
        valid_gaps = self._validate_column_gaps(valleys, image.shape[1])
        
        # Create column info
        columns = self._create_columns_from_gaps(valid_gaps, image.shape, binary)
        
        # Detect gutter information
        gutter_info = self._analyze_gutters(valid_gaps, smoothed)
        
        # Detect margins
        margins = self._detect_margins(binary)
        
        return LayoutAnalysis(
            layout_type=self._determine_layout_type(len(columns)),
            columns=columns,
            gutter_info=gutter_info,
            margins=margins,
            confidence=0.8
        )
    
    def _advanced_column_detection(self, image: np.ndarray) -> LayoutAnalysis:
        """
        Advanced detection using multiple techniques:
        - Projection profiles
        - Connected component analysis
        - Text line detection
        - Whitespace analysis
        """
        
        gray = self._prepare_image(image)
        binary = self._binarize_image(gray)
        
        # Step 1: Initial projection analysis
        vertical_proj = np.sum(binary, axis=0)
        smoothed_proj = signal.savgol_filter(vertical_proj, 31, 3)
        
        # Step 2: Find text regions using connected components
        text_regions = self._find_text_regions(binary)
        
        # Step 3: Cluster text regions into columns
        columns = self._cluster_text_regions(text_regions)
        
        # Step 4: Refine using whitespace analysis
        columns = self._refine_with_whitespace(columns, binary)
        
        # Step 5: Validate and score columns
        columns = self._validate_columns(columns, binary)
        
        # Step 6: Detect additional layout features
        gutter_info = self._analyze_gutters_advanced(columns, binary)
        margins = self._detect_margins_advanced(binary, columns)
        
        # Step 7: Calculate overall confidence
        confidence = self._calculate_confidence(columns, text_regions)
        
        return LayoutAnalysis(
            layout_type=self._determine_layout_type_advanced(columns),
            columns=columns,
            gutter_info=gutter_info,
            margins=margins,
            text_regions=[(r['x'], r['y'], r['w'], r['h']) for r in text_regions],
            confidence=confidence
        )
    
    def _ml_based_detection(self, image: np.ndarray) -> LayoutAnalysis:
        """Machine learning based detection (placeholder for future)"""
        # This would use a trained model for column detection
        # For now, fall back to advanced method
        return self._advanced_column_detection(image)
    
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for processing"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return gray
    
    def _binarize_image(self, gray: np.ndarray) -> np.ndarray:
        """Binarize image with optimal parameters for text"""
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to connect text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def _find_gaps(self, projection: np.ndarray, 
                   threshold_ratio: float = 0.1) -> List[Tuple[int, int]]:
        """Find gaps in projection profile"""
        
        threshold = np.max(projection) * threshold_ratio
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, val in enumerate(projection):
            if val < threshold and not in_gap:
                in_gap = True
                gap_start = i
            elif val >= threshold and in_gap:
                in_gap = False
                gap_width = i - gap_start
                if gap_width >= self.min_gutter_width:
                    gaps.append((gap_start, i))
        
        # Handle gap at end
        if in_gap:
            gaps.append((gap_start, len(projection)))
        
        return gaps
    
    def _find_valleys(self, smoothed: np.ndarray) -> List[int]:
        """Find valleys in smoothed projection"""
        
        # Find local minima
        valleys = signal.argrelmin(smoothed)[0]
        
        # Filter valleys by depth
        valley_depths = []
        for valley in valleys:
            left_peak = np.max(smoothed[max(0, valley-50):valley])
            right_peak = np.max(smoothed[valley:min(len(smoothed), valley+50)])
            depth = min(left_peak, right_peak) - smoothed[valley]
            valley_depths.append(depth)
        
        # Keep significant valleys
        if valley_depths:
            depth_threshold = np.mean(valley_depths) * 0.5
            significant_valleys = [v for v, d in zip(valleys, valley_depths) 
                                  if d > depth_threshold]
            return significant_valleys
        
        return []
    
    def _validate_column_gaps(self, valleys: List[int], 
                             page_width: int) -> List[Tuple[int, int]]:
        """Validate potential column gaps"""
        
        valid_gaps = []
        
        for i, valley in enumerate(valleys):
            # Check if valley is in reasonable position for column gap
            if self.min_gutter_width < valley < page_width - self.min_gutter_width:
                # Find gap boundaries
                gap_start = max(0, valley - self.max_gutter_width // 2)
                gap_end = min(page_width, valley + self.max_gutter_width // 2)
                
                # Validate gap width
                gap_width = gap_end - gap_start
                if self.min_gutter_width <= gap_width <= self.max_gutter_width:
                    valid_gaps.append((gap_start, gap_end))
        
        return valid_gaps
    
    def _find_text_regions(self, binary: np.ndarray) -> List[Dict[str, Any]]:
        """Find all text regions using connected components"""
        
        # Dilate to connect text into regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter small regions
            if area > 500 and w > 50 and h > 20:
                # Calculate text density
                region_binary = binary[y:y+h, x:x+w]
                text_density = np.sum(region_binary > 0) / (w * h)
                
                if text_density > self.min_text_density:
                    text_regions.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'area': area,
                        'density': text_density,
                        'centroid': (x + w//2, y + h//2)
                    })
        
        return text_regions
    
    def _cluster_text_regions(self, text_regions: List[Dict]) -> List[ColumnInfo]:
        """Cluster text regions into columns using DBSCAN"""
        
        if not text_regions:
            return []
        
        # Extract x-coordinates of centroids
        X = np.array([[r['centroid'][0]] for r in text_regions])
        
        # Cluster based on x-coordinate
        clustering = DBSCAN(eps=100, min_samples=3).fit(X)
        labels = clustering.labels_
        
        # Group regions by cluster
        columns = []
        for label in set(labels):
            if label == -1:  # Skip noise
                continue
            
            cluster_regions = [r for r, l in zip(text_regions, labels) if l == label]
            
            if cluster_regions:
                # Calculate column boundaries
                x_coords = [r['x'] for r in cluster_regions]
                x_coords_end = [r['x'] + r['w'] for r in cluster_regions]
                y_coords = [r['y'] for r in cluster_regions]
                y_coords_end = [r['y'] + r['h'] for r in cluster_regions]
                
                col = ColumnInfo(
                    index=label,
                    x_start=min(x_coords),
                    x_end=max(x_coords_end),
                    y_start=min(y_coords),
                    y_end=max(y_coords_end),
                    width=max(x_coords_end) - min(x_coords),
                    height=max(y_coords_end) - min(y_coords),
                    text_density=np.mean([r['density'] for r in cluster_regions]),
                    line_count=len(cluster_regions),
                    avg_line_height=np.mean([r['h'] for r in cluster_regions]),
                    confidence=0.0  # Will be calculated later
                )
                columns.append(col)
        
        # Sort columns by x position
        columns.sort(key=lambda c: c.x_start)
        
        # Re-index columns
        for i, col in enumerate(columns):
            col.index = i
        
        return columns
    
    def _refine_with_whitespace(self, columns: List[ColumnInfo], 
                               binary: np.ndarray) -> List[ColumnInfo]:
        """Refine column boundaries using whitespace analysis"""
        
        if len(columns) < 2:
            return columns
        
        refined = []
        
        for i, col in enumerate(columns):
            # Find precise boundaries using vertical projection
            col_region = binary[col.y_start:col.y_end, col.x_start:col.x_end]
            
            if col_region.size == 0:
                refined.append(col)
                continue
            
            # Vertical projection of column region
            vert_proj = np.sum(col_region, axis=0)
            
            # Find actual text boundaries
            text_indices = np.where(vert_proj > np.max(vert_proj) * 0.1)[0]
            
            if len(text_indices) > 0:
                # Update column boundaries
                col.x_start = col.x_start + text_indices[0]
                col.x_end = col.x_start + text_indices[-1]
                col.width = col.x_end - col.x_start
            
            refined.append(col)
        
        return refined
    
    def _validate_columns(self, columns: List[ColumnInfo], 
                         binary: np.ndarray) -> List[ColumnInfo]:
        """Validate and score detected columns"""
        
        validated = []
        
        for col in columns:
            # Check column width
            if not (self.min_column_width <= col.width <= self.max_column_width):
                continue
            
            # Check aspect ratio
            aspect_ratio = col.height / col.width if col.width > 0 else 0
            if aspect_ratio < 0.5:  # Too wide for a text column
                continue
            
            # Calculate confidence based on various factors
            confidence = 1.0
            
            # Factor 1: Text density
            confidence *= min(col.text_density / 0.3, 1.0)
            
            # Factor 2: Aspect ratio (ideal is around 2-3 for text columns)
            ideal_aspect = 2.5
            aspect_penalty = abs(aspect_ratio - ideal_aspect) / ideal_aspect
            confidence *= max(0, 1 - aspect_penalty * 0.5)
            
            # Factor 3: Line regularity (check if lines are evenly spaced)
            if col.line_count > 1:
                line_spacing_variance = 0.1  # Placeholder
                confidence *= max(0, 1 - line_spacing_variance)
            
            col.confidence = confidence
            
            if confidence > 0.5:
                validated.append(col)
        
        return validated
    
    def _analyze_gutters_advanced(self, columns: List[ColumnInfo], 
                                 binary: np.ndarray) -> Optional[Dict[str, Any]]:
        """Analyze gutter between columns"""
        
        if len(columns) < 2:
            return None
        
        gutters = []
        
        for i in range(len(columns) - 1):
            left_col = columns[i]
            right_col = columns[i + 1]
            
            gutter_start = left_col.x_end
            gutter_end = right_col.x_start
            gutter_width = gutter_end - gutter_start
            
            if gutter_width > 0:
                # Analyze gutter content
                gutter_region = binary[:, gutter_start:gutter_end]
                gutter_density = np.sum(gutter_region > 0) / gutter_region.size
                
                gutters.append({
                    'position': (gutter_start, gutter_end),
                    'width': gutter_width,
                    'density': gutter_density,
                    'is_clean': gutter_density < 0.05
                })
        
        if gutters:
            return {
                'count': len(gutters),
                'avg_width': np.mean([g['width'] for g in gutters]),
                'gutters': gutters
            }
        
        return None
    
    def _detect_margins_advanced(self, binary: np.ndarray, 
                                columns: List[ColumnInfo]) -> Dict[str, int]:
        """Detect page margins"""
        
        h, w = binary.shape
        
        # Find text boundaries
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        
        # Find first and last text pixels
        if np.any(rows):
            rmin, rmax = np.where(rows)[0][[0, -1]]
        else:
            rmin, rmax = 0, h
        
        if np.any(cols):
            cmin, cmax = np.where(cols)[0][[0, -1]]
        else:
            cmin, cmax = 0, w
        
        # Refine with column information
        if columns:
            cmin = min(cmin, min(c.x_start for c in columns))
            cmax = max(cmax, max(c.x_end for c in columns))
        
        margins = {
            'top': rmin,
            'bottom': h - rmax,
            'left': cmin,
            'right': w - cmax,
            'inner': cmin if cmin < w - cmax else w - cmax,
            'outer': w - cmax if cmin < w - cmax else cmin
        }
        
        return margins
    
    def _calculate_confidence(self, columns: List[ColumnInfo], 
                            text_regions: List[Dict]) -> float:
        """Calculate overall layout detection confidence"""
        
        if not columns:
            return 0.0
        
        # Base confidence on column detection
        column_confidence = np.mean([c.confidence for c in columns])
        
        # Check coverage of text regions by columns
        covered_regions = 0
        for region in text_regions:
            rx = region['x'] + region['w'] // 2
            for col in columns:
                if col.x_start <= rx <= col.x_end:
                    covered_regions += 1
                    break
        
        coverage_ratio = covered_regions / len(text_regions) if text_regions else 0
        
        # Combined confidence
        confidence = column_confidence * 0.7 + coverage_ratio * 0.3
        
        return min(confidence, 1.0)
    
    def _determine_layout_type(self, num_columns: int) -> str:
        """Determine layout type from column count"""
        if num_columns == 0:
            return 'undefined'
        elif num_columns == 1:
            return 'single'
        elif num_columns == 2:
            return 'two_column'
        elif num_columns == 3:
            return 'three_column'
        else:
            return 'multi_column'
    
    def _determine_layout_type_advanced(self, columns: List[ColumnInfo]) -> str:
        """Advanced layout type determination"""
        
        if not columns:
            return 'undefined'
        
        num_cols = len(columns)
        
        if num_cols == 1:
            return 'single'
        elif num_cols == 2:
            # Check if columns are balanced (typical for 1611 KJV)
            width_ratio = columns[0].width / columns[1].width if columns[1].width > 0 else 0
            if 0.8 <= width_ratio <= 1.2:
                return 'two_column_balanced'
            else:
                return 'two_column_unbalanced'
        elif num_cols == 3:
            return 'three_column'
        else:
            return 'multi_column'
    
    def _gaps_to_columns(self, gaps: List[Tuple[int, int]], 
                        shape: Tuple[int, int]) -> List[ColumnInfo]:
        """Convert gaps to column definitions"""
        
        columns = []
        h, w = shape[:2]
        
        if not gaps:
            # Single column
            columns.append(ColumnInfo(
                index=0,
                x_start=0,
                x_end=w,
                y_start=0,
                y_end=h,
                width=w,
                height=h,
                text_density=0,
                line_count=0,
                avg_line_height=0,
                confidence=0.5
            ))
        elif len(gaps) == 1:
            # Two columns
            gap = gaps[0]
            columns.append(ColumnInfo(
                index=0,
                x_start=0,
                x_end=gap[0],
                y_start=0,
                y_end=h,
                width=gap[0],
                height=h,
                text_density=0,
                line_count=0,
                avg_line_height=0,
                confidence=0.7
            ))
            columns.append(ColumnInfo(
                index=1,
                x_start=gap[1],
                x_end=w,
                y_start=0,
                y_end=h,
                width=w - gap[1],
                height=h,
                text_density=0,
                line_count=0,
                avg_line_height=0,
                confidence=0.7
            ))
        
        return columns
    
    def _create_columns_from_gaps(self, gaps: List[Tuple[int, int]], 
                                 shape: Tuple[int, int],
                                 binary: np.ndarray) -> List[ColumnInfo]:
        """Create detailed column info from gaps"""
        
        columns = self._gaps_to_columns(gaps, shape)
        
        # Enhance with actual text analysis
        for col in columns:
            if col.x_end > col.x_start and col.y_end > col.y_start:
                region = binary[col.y_start:col.y_end, col.x_start:col.x_end]
                col.text_density = np.sum(region > 0) / region.size if region.size > 0 else 0
                
                # Count text lines (simplified)
                horizontal_proj = np.sum(region, axis=1)
                text_lines = np.sum(horizontal_proj > np.max(horizontal_proj) * 0.1)
                col.line_count = text_lines
                
                if text_lines > 0:
                    col.avg_line_height = col.height / text_lines
        
        return columns
    
    def _analyze_gutters(self, gaps: List[Tuple[int, int]], 
                        projection: np.ndarray) -> Optional[Dict[str, Any]]:
        """Analyze gutter characteristics"""
        
        if not gaps:
            return None
        
        gutter_info = {
            'count': len(gaps),
            'positions': gaps,
            'widths': [g[1] - g[0] for g in gaps],
            'avg_width': np.mean([g[1] - g[0] for g in gaps]),
            'cleanness': []
        }
        
        # Check how clean the gutters are
        for gap in gaps:
            gap_proj = projection[gap[0]:gap[1]]
            if len(gap_proj) > 0:
                cleanness = 1.0 - (np.mean(gap_proj) / (np.max(projection) + 1e-6))
                gutter_info['cleanness'].append(cleanness)
        
        return gutter_info
    
    def _detect_margins(self, binary: np.ndarray) -> Dict[str, int]:
        """Basic margin detection"""
        
        h, w = binary.shape
        
        # Find text boundaries
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        
        if np.any(rows):
            rmin, rmax = np.where(rows)[0][[0, -1]]
        else:
            rmin, rmax = 0, h
        
        if np.any(cols):
            cmin, cmax = np.where(cols)[0][[0, -1]]
        else:
            cmin, cmax = 0, w
        
        margins = {
            'top': rmin,
            'bottom': h - rmax,
            'left': cmin,
            'right': w - cmax
        }
        
        return margins
    
    def visualize_layout(self, image: np.ndarray, 
                        layout: LayoutAnalysis,
                        save_path: Optional[str] = None) -> np.ndarray:
        """Visualize detected layout"""
        
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Draw columns
        for col in layout.columns:
            color = (0, 255, 0) if col.confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(vis_image, 
                         (col.x_start, col.y_start),
                         (col.x_end, col.y_end),
                         color, 2)
            
            # Add column index
            cv2.putText(vis_image, f"Col {col.index}", 
                       (col.x_start + 10, col.y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw gutters
        if layout.gutter_info:
            for gutter in layout.gutter_info.get('positions', []):
                cv2.rectangle(vis_image,
                            (gutter[0], 0),
                            (gutter[1], image.shape[0]),
                            (255, 0, 0), 1)
        
        # Draw margins
        if layout.margins:
            m = layout.margins
            h, w = image.shape[:2]
            # Top margin
            cv2.line(vis_image, (0, m['top']), (w, m['top']), (0, 0, 255), 1)
            # Bottom margin  
            cv2.line(vis_image, (0, h - m['bottom']), (w, h - m['bottom']), (0, 0, 255), 1)
            # Left margin
            cv2.line(vis_image, (m['left'], 0), (m['left'], h), (0, 0, 255), 1)
            # Right margin
            cv2.line(vis_image, (w - m['right'], 0), (w - m['right'], h), (0, 0, 255), 1)
        
        # Add layout info text
        info_text = f"Layout: {layout.layout_type} | Columns: {len(layout.columns)} | Confidence: {layout.confidence:.2f}"
        cv2.putText(vis_image, info_text,
                   (10, image.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image