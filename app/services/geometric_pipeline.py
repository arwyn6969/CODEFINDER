"""
Geometric Analysis Pipeline
Orchestrates coordinate extraction, spatial indexing, and pattern detection.
"""
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from sqlalchemy.orm import Session

from app.services.coordinate_extractor import (
    get_character_positions_for_page,
    positions_to_tuples,
    CharacterPosition,
    get_page_dimensions
)
from app.services.geometric_index import GeometricIndex


# Constants for pattern detection
GOLDEN_RATIO = 1.618033988749895
GOLDEN_RATIO_TOLERANCE = 0.005  # ±0.5%
RIGHT_ANGLE_DEGREES = 90.0
RIGHT_ANGLE_TOLERANCE = 1.0  # ±1°


@dataclass
class GeometricPattern:
    """Detected geometric pattern."""
    pattern_type: str  # 'golden_ratio', 'right_angle', 'isosceles'
    point_indices: List[int]  # Indices into the positions list
    positions: List[CharacterPosition]  # Actual character data
    measurement_value: float
    significance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeometricAnalysisResult:
    """Complete analysis result for a page."""
    document_id: int
    page_number: int
    total_characters: int
    significant_characters: int
    patterns_found: List[GeometricPattern]
    page_width: float
    page_height: float
    index: Optional[GeometricIndex] = None


def analyze_page_geometry(
    db: Session,
    document_id: int,
    page_number: int,
    min_confidence: float = 0.5,
    filter_significant: bool = True,
    max_patterns: int = 20
) -> GeometricAnalysisResult:
    """
    Run geometric analysis on a page using the optimized pipeline.
    
    Args:
        db: Database session
        document_id: Document ID
        page_number: Page number (1-indexed)
        min_confidence: Minimum OCR confidence
        filter_significant: If True, only analyze uppercase + punctuation
        max_patterns: Maximum number of patterns to return (by significance)
    
    Returns:
        GeometricAnalysisResult with detected patterns
    """
    # Step 1: Extract positions from database
    positions = get_character_positions_for_page(
        db=db,
        document_id=document_id,
        page_number=page_number,
        min_confidence=min_confidence,
        filter_significant=filter_significant
    )
    
    page_width, page_height = get_page_dimensions(db, document_id, page_number)
    
    if len(positions) < 3:
        return GeometricAnalysisResult(
            document_id=document_id,
            page_number=page_number,
            total_characters=len(positions),
            significant_characters=len(positions),
            patterns_found=[],
            page_width=page_width,
            page_height=page_height
        )
    
    # Step 2: Build KD-Tree index
    point_tuples = positions_to_tuples(positions)
    index = GeometricIndex(point_tuples)
    
    # Step 3: Detect patterns
    all_patterns = []
    
    # Detect Golden Ratio distances
    golden_patterns = _detect_golden_ratio_patterns(index, positions, page_width)
    all_patterns.extend(golden_patterns)
    
    # Detect Right Angle triangles
    right_angle_patterns = _detect_right_angle_patterns(index, positions)
    all_patterns.extend(right_angle_patterns)
    
    # Step 4: Sort by significance and limit
    all_patterns.sort(key=lambda p: p.significance_score, reverse=True)
    top_patterns = all_patterns[:max_patterns]
    
    return GeometricAnalysisResult(
        document_id=document_id,
        page_number=page_number,
        total_characters=len(positions),
        significant_characters=len([p for p in positions if p.is_uppercase or p.is_punctuation]),
        patterns_found=top_patterns,
        page_width=page_width,
        page_height=page_height,
        index=index
    )


def _detect_golden_ratio_patterns(
    index: GeometricIndex,
    positions: List[CharacterPosition],
    page_width: float
) -> List[GeometricPattern]:
    """
    Detect pairs of points whose distance approximates Golden Ratio multiples.
    Uses KD-Tree for efficient pair finding.
    """
    patterns = []
    
    # Check for Golden Ratio at various scales
    base_unit = page_width / 20  # Approximate 1/20th of page as base unit
    
    for scale in [1, 2, 3, 5, 8]:  # Fibonacci sequence scales
        target_distance = base_unit * scale * GOLDEN_RATIO
        tolerance = target_distance * GOLDEN_RATIO_TOLERANCE
        
        pairs = index.find_pairs_within_distance(target_distance + tolerance)
        
        for idx1, idx2 in pairs:
            # Get actual distance
            p1 = index.points[idx1]
            p2 = index.points[idx2]
            actual_distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # Check if within tolerance of golden ratio distance
            if abs(actual_distance - target_distance) <= tolerance:
                # Calculate significance (closer to exact = higher)
                deviation = abs(actual_distance - target_distance) / target_distance
                significance = 1.0 - (deviation / GOLDEN_RATIO_TOLERANCE)
                
                patterns.append(GeometricPattern(
                    pattern_type='golden_ratio',
                    point_indices=[idx1, idx2],
                    positions=[positions[idx1], positions[idx2]],
                    measurement_value=actual_distance / base_unit,
                    significance_score=significance,
                    metadata={
                        'scale': scale,
                        'target_distance': target_distance,
                        'actual_distance': actual_distance,
                        'ratio': actual_distance / base_unit
                    }
                ))
    
    return patterns


def _detect_right_angle_patterns(
    index: GeometricIndex,
    positions: List[CharacterPosition]
) -> List[GeometricPattern]:
    """
    Detect triplets of points forming right angles.
    Uses KD-Tree to efficiently find nearby points.
    """
    patterns = []
    n_points = index.n_points
    
    if n_points < 3:
        return patterns
    
    # For each point, check K nearest neighbors for right-angle relationships
    k = min(10, n_points)  # Check up to 10 nearest neighbors
    
    for i in range(n_points):
        neighbors = index.find_nearest_neighbors(tuple(index.points[i]), k=k)
        neighbor_indices = [idx for _, idx in neighbors if idx != i]
        
        # Check triplets formed by this point and pairs of neighbors
        for j_idx in range(len(neighbor_indices)):
            for k_idx in range(j_idx + 1, len(neighbor_indices)):
                j = neighbor_indices[j_idx]
                k = neighbor_indices[k_idx]
                
                # Calculate angle at vertex i
                angle = _calculate_angle(index.points[i], index.points[j], index.points[k])
                
                if abs(angle - RIGHT_ANGLE_DEGREES) <= RIGHT_ANGLE_TOLERANCE:
                    deviation = abs(angle - RIGHT_ANGLE_DEGREES)
                    significance = 1.0 - (deviation / RIGHT_ANGLE_TOLERANCE)
                    
                    patterns.append(GeometricPattern(
                        pattern_type='right_angle',
                        point_indices=[i, j, k],
                        positions=[positions[i], positions[j], positions[k]],
                        measurement_value=angle,
                        significance_score=significance,
                        metadata={
                            'vertex_index': i,
                            'angle_degrees': angle,
                            'vertex_char': positions[i].character
                        }
                    ))
    
    return patterns


def _calculate_angle(vertex: tuple, p1: tuple, p2: tuple) -> float:
    """Calculate angle in degrees at vertex formed by p1-vertex-p2."""
    # Vectors from vertex to each point
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
    
    # Dot product
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    
    # Magnitudes
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    # Clamp to avoid floating point errors with acos
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    angle_radians = math.acos(cos_angle)
    
    return math.degrees(angle_radians)
