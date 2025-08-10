"""
Geometric Analysis Service for Ancient Text Analyzer
Measures angles, distances, and geometric relationships in text layouts
"""
import numpy as np
import math
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import statistics

from app.services.ocr_engine import CharacterBox

logger = logging.getLogger(__name__)

class MeasurementType(Enum):
    """Types of geometric measurements"""
    ANGLE = "angle"
    DISTANCE = "distance"
    RATIO = "ratio"
    AREA = "area"
    PERIMETER = "perimeter"

class GeometricPattern(Enum):
    """Types of geometric patterns"""
    TRIANGLE = "triangle"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    LINE = "line"
    CROSS = "cross"
    SPIRAL = "spiral"

@dataclass(frozen=True)
class Point:
    """2D point with coordinates"""
    x: float
    y: float
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def angle_to(self, other: 'Point') -> float:
        """Calculate angle to another point in radians"""
        return math.atan2(other.y - self.y, other.x - self.x)
    
    def midpoint_to(self, other: 'Point') -> 'Point':
        """Calculate midpoint to another point"""
        return Point((self.x + other.x) / 2, (self.y + other.y) / 2)

@dataclass
class AngleMeasurement:
    """Container for angle measurement data"""
    angle_degrees: float
    angle_radians: float
    vertex: Point
    point1: Point
    point2: Point
    measurement_type: str
    confidence: float
    significance_score: float
    context: Optional[str] = None

@dataclass
class DistanceData:
    """Container for distance measurement data"""
    distance: float
    point1: Point
    point2: Point
    measurement_unit: str
    measurement_type: str
    confidence: float
    significance_score: float
    context: Optional[str] = None

@dataclass
class GeometricRelationship:
    """Container for geometric relationship data"""
    relationship_type: str
    elements: List[Point]
    measurements: Dict[str, float]
    confidence: float
    significance_score: float
    description: str
    pattern_type: Optional[GeometricPattern] = None

@dataclass
class TrigRelationship:
    """Container for trigonometric relationship data"""
    relationship_type: str
    angles: List[float]
    ratios: List[float]
    trigonometric_values: Dict[str, float]
    confidence: float
    significance_score: float
    description: str
    mathematical_constant: Optional[str] = None

class GeometricAnalyzer:
    """
    Advanced geometric analyzer for text layout analysis
    Specialized for detecting hidden geometric patterns and relationships
    """
    
    def __init__(self):
        # Mathematical constants for pattern recognition
        self.GOLDEN_RATIO = 1.618033988749
        self.PI = math.pi
        self.E = math.e
        self.SQRT_2 = math.sqrt(2)
        self.SQRT_3 = math.sqrt(3)
        self.SQRT_5 = math.sqrt(5)
        
        # Tolerance for floating point comparisons
        self.TOLERANCE = 1e-6
        
        # Significant angles in degrees
        self.SIGNIFICANT_ANGLES = [30, 45, 60, 90, 120, 135, 150, 180, 270, 360]
        
    def measure_angles(self, positions: List[Point]) -> List[AngleMeasurement]:
        """
        Measure angles between sets of three points
        
        Args:
            positions: List of Point objects
            
        Returns:
            List of AngleMeasurement objects
        """
        logger.info(f"Measuring angles for {len(positions)} points")
        
        if len(positions) < 3:
            logger.warning("Need at least 3 points to measure angles")
            return []
        
        angle_measurements = []
        
        # Measure angles for all combinations of three points
        for i in range(len(positions)):
            for j in range(len(positions)):
                for k in range(len(positions)):
                    if i != j and j != k and i != k:
                        angle = self._calculate_angle(positions[i], positions[j], positions[k])
                        if angle is not None:
                            angle_measurements.append(angle)
        
        # Sort by significance score
        angle_measurements.sort(key=lambda x: x.significance_score, reverse=True)
        
        logger.info(f"Measured {len(angle_measurements)} angles")
        return angle_measurements
    
    def _calculate_angle(self, p1: Point, vertex: Point, p2: Point) -> Optional[AngleMeasurement]:
        """
        Calculate angle at vertex formed by p1-vertex-p2
        
        Args:
            p1: First point
            vertex: Vertex point (angle measurement point)
            p2: Second point
            
        Returns:
            AngleMeasurement object or None if calculation fails
        """
        try:
            # Create vectors from vertex to other points
            v1 = np.array([p1.x - vertex.x, p1.y - vertex.y])
            v2 = np.array([p2.x - vertex.x, p2.y - vertex.y])
            
            # Calculate magnitudes
            mag1 = np.linalg.norm(v1)
            mag2 = np.linalg.norm(v2)
            
            if mag1 == 0 or mag2 == 0:
                return None
            
            # Calculate angle using dot product
            cos_angle = np.dot(v1, v2) / (mag1 * mag2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle floating point errors
            
            angle_radians = np.arccos(cos_angle)
            angle_degrees = np.degrees(angle_radians)
            
            # Calculate confidence based on point separation
            confidence = self._calculate_angle_confidence(p1, vertex, p2)
            
            # Calculate significance score
            significance = self._calculate_angle_significance(angle_degrees)
            
            return AngleMeasurement(
                angle_degrees=angle_degrees,
                angle_radians=angle_radians,
                vertex=vertex,
                point1=p1,
                point2=p2,
                measurement_type="three_point_angle",
                confidence=confidence,
                significance_score=significance,
                context=f"Angle at ({vertex.x:.1f}, {vertex.y:.1f})"
            )
            
        except Exception as e:
            logger.warning(f"Failed to calculate angle: {e}")
            return None
    
    def _calculate_angle_confidence(self, p1: Point, vertex: Point, p2: Point) -> float:
        """
        Calculate confidence score for angle measurement based on point geometry
        """
        # Distance from vertex to other points
        d1 = vertex.distance_to(p1)
        d2 = vertex.distance_to(p2)
        
        # Confidence is higher when points are well-separated
        min_distance = min(d1, d2)
        max_distance = max(d1, d2)
        
        if max_distance == 0:
            return 0.0
        
        # Base confidence from minimum distance
        base_confidence = min(min_distance / 50.0, 1.0)  # Normalize to reasonable pixel distance
        
        # Bonus for balanced distances
        balance_ratio = min_distance / max_distance
        balance_bonus = balance_ratio * 0.2
        
        return min(base_confidence + balance_bonus, 1.0)
    
    def _calculate_angle_significance(self, angle_degrees: float) -> float:
        """
        Calculate significance score based on how close angle is to meaningful values
        """
        # Check proximity to significant angles
        min_diff = min(abs(angle_degrees - sig_angle) for sig_angle in self.SIGNIFICANT_ANGLES)
        
        # Higher significance for angles close to meaningful values
        if min_diff < 1.0:  # Within 1 degree
            return 1.0
        elif min_diff < 5.0:  # Within 5 degrees
            return 0.8
        elif min_diff < 10.0:  # Within 10 degrees
            return 0.6
        else:
            return 0.3
    
    def calculate_distances(self, pos1: Point, pos2: Point) -> DistanceData:
        """
        Calculate distance between two points with analysis
        
        Args:
            pos1: First point
            pos2: Second point
            
        Returns:
            DistanceData object
        """
        distance = pos1.distance_to(pos2)
        
        # Calculate confidence (always high for distance measurements)
        confidence = 0.95
        
        # Calculate significance based on distance value
        significance = self._calculate_distance_significance(distance)
        
        return DistanceData(
            distance=distance,
            point1=pos1,
            point2=pos2,
            measurement_unit="pixels",
            measurement_type="euclidean_distance",
            confidence=confidence,
            significance_score=significance,
            context=f"Distance between ({pos1.x:.1f}, {pos1.y:.1f}) and ({pos2.x:.1f}, {pos2.y:.1f})"
        )
    
    def _calculate_distance_significance(self, distance: float) -> float:
        """
        Calculate significance of a distance measurement
        """
        # Significance based on whether distance relates to common ratios
        ratios_to_check = [
            self.GOLDEN_RATIO,
            self.PI,
            self.E,
            self.SQRT_2,
            self.SQRT_3,
            self.SQRT_5,
            2.0,
            3.0,
            5.0
        ]
        
        # Check if distance is a multiple of significant ratios
        for ratio in ratios_to_check:
            for multiplier in [1, 2, 3, 5, 10]:
                expected = ratio * multiplier
                if abs(distance - expected) < self.TOLERANCE * 100:  # Allow some tolerance
                    return 0.9
        
        # Check for simple integer relationships
        if abs(distance - round(distance)) < 0.1:
            return 0.7
        
        return 0.5  # Default significance
    
    def detect_geometric_patterns(self, measurements: List[AngleMeasurement]) -> List[GeometricRelationship]:
        """
        Detect geometric patterns from angle measurements
        
        Args:
            measurements: List of angle measurements
            
        Returns:
            List of detected geometric relationships
        """
        logger.info(f"Detecting geometric patterns from {len(measurements)} measurements")
        
        patterns = []
        
        # Group measurements by vertex to find patterns
        vertex_groups = self._group_by_vertex(measurements)
        
        for vertex, angles in vertex_groups.items():
            # Detect triangular patterns
            triangular_patterns = self._detect_triangular_patterns(vertex, angles)
            patterns.extend(triangular_patterns)
            
            # Detect right angle patterns
            right_angle_patterns = self._detect_right_angle_patterns(vertex, angles)
            patterns.extend(right_angle_patterns)
            
            # Detect cross patterns
            cross_patterns = self._detect_cross_patterns(vertex, angles)
            patterns.extend(cross_patterns)
        
        # Sort by significance
        patterns.sort(key=lambda x: x.significance_score, reverse=True)
        
        logger.info(f"Detected {len(patterns)} geometric patterns")
        return patterns
    
    def _group_by_vertex(self, measurements: List[AngleMeasurement]) -> Dict[Point, List[AngleMeasurement]]:
        """Group angle measurements by their vertex point"""
        groups = {}
        
        for measurement in measurements:
            vertex_key = (measurement.vertex.x, measurement.vertex.y)
            if vertex_key not in groups:
                groups[vertex_key] = []
            groups[vertex_key].append(measurement)
        
        # Convert keys back to Point objects
        point_groups = {}
        for (x, y), angles in groups.items():
            point_groups[Point(x, y)] = angles
        
        return point_groups
    
    def _detect_triangular_patterns(self, vertex: Point, angles: List[AngleMeasurement]) -> List[GeometricRelationship]:
        """Detect triangular patterns at a vertex"""
        patterns = []
        
        # Look for angles that sum to 180 degrees (triangle)
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                for k in range(j + 1, len(angles)):
                    angle_sum = angles[i].angle_degrees + angles[j].angle_degrees + angles[k].angle_degrees
                    
                    if abs(angle_sum - 180.0) < 5.0:  # Within 5 degrees of 180
                        pattern = GeometricRelationship(
                            relationship_type="triangular_pattern",
                            elements=[vertex, angles[i].point1, angles[j].point1, angles[k].point1],
                            measurements={
                                "angle1": angles[i].angle_degrees,
                                "angle2": angles[j].angle_degrees,
                                "angle3": angles[k].angle_degrees,
                                "angle_sum": angle_sum
                            },
                            confidence=min(angles[i].confidence, angles[j].confidence, angles[k].confidence),
                            significance_score=0.9,
                            description=f"Triangular pattern with angles {angles[i].angle_degrees:.1f}°, {angles[j].angle_degrees:.1f}°, {angles[k].angle_degrees:.1f}°",
                            pattern_type=GeometricPattern.TRIANGLE
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_right_angle_patterns(self, vertex: Point, angles: List[AngleMeasurement]) -> List[GeometricRelationship]:
        """Detect right angle patterns (90 degrees)"""
        patterns = []
        
        for angle in angles:
            if abs(angle.angle_degrees - 90.0) < 2.0:  # Within 2 degrees of 90
                pattern = GeometricRelationship(
                    relationship_type="right_angle_pattern",
                    elements=[vertex, angle.point1, angle.point2],
                    measurements={
                        "angle": angle.angle_degrees,
                        "deviation_from_90": abs(angle.angle_degrees - 90.0)
                    },
                    confidence=angle.confidence,
                    significance_score=0.95,
                    description=f"Right angle pattern: {angle.angle_degrees:.1f}°",
                    pattern_type=GeometricPattern.CROSS
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_cross_patterns(self, vertex: Point, angles: List[AngleMeasurement]) -> List[GeometricRelationship]:
        """Detect cross patterns (intersecting lines)"""
        patterns = []
        
        # Look for pairs of angles that are supplementary (sum to 180°)
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                angle_sum = angles[i].angle_degrees + angles[j].angle_degrees
                
                if abs(angle_sum - 180.0) < 5.0:  # Supplementary angles
                    pattern = GeometricRelationship(
                        relationship_type="cross_pattern",
                        elements=[vertex, angles[i].point1, angles[i].point2, angles[j].point1, angles[j].point2],
                        measurements={
                            "angle1": angles[i].angle_degrees,
                            "angle2": angles[j].angle_degrees,
                            "angle_sum": angle_sum
                        },
                        confidence=min(angles[i].confidence, angles[j].confidence),
                        significance_score=0.8,
                        description=f"Cross pattern with supplementary angles {angles[i].angle_degrees:.1f}° and {angles[j].angle_degrees:.1f}°",
                        pattern_type=GeometricPattern.CROSS
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def find_trigonometric_relationships(self, angles: List[float]) -> List[TrigRelationship]:
        """
        Find trigonometric relationships and ratios in angle data
        
        Args:
            angles: List of angles in degrees
            
        Returns:
            List of trigonometric relationships
        """
        logger.info(f"Finding trigonometric relationships in {len(angles)} angles")
        
        relationships = []
        
        # Convert to radians for calculations
        angles_rad = [math.radians(angle) for angle in angles]
        
        # Find golden ratio relationships
        golden_relationships = self._find_golden_ratio_relationships(angles, angles_rad)
        relationships.extend(golden_relationships)
        
        # Find pi relationships
        pi_relationships = self._find_pi_relationships(angles, angles_rad)
        relationships.extend(pi_relationships)
        
        # Find harmonic relationships
        harmonic_relationships = self._find_harmonic_relationships(angles, angles_rad)
        relationships.extend(harmonic_relationships)
        
        # Find trigonometric identities
        identity_relationships = self._find_trigonometric_identities(angles, angles_rad)
        relationships.extend(identity_relationships)
        
        # Sort by significance
        relationships.sort(key=lambda x: x.significance_score, reverse=True)
        
        logger.info(f"Found {len(relationships)} trigonometric relationships")
        return relationships
    
    def _find_golden_ratio_relationships(self, angles_deg: List[float], angles_rad: List[float]) -> List[TrigRelationship]:
        """Find relationships involving the golden ratio"""
        relationships = []
        
        for i, angle_deg in enumerate(angles_deg):
            angle_rad = angles_rad[i]
            
            # Check if angle relates to golden ratio
            # Golden angle ≈ 137.5°
            golden_angle_deg = 360.0 / (self.GOLDEN_RATIO + 1)
            
            if abs(angle_deg - golden_angle_deg) < 2.0:
                relationship = TrigRelationship(
                    relationship_type="golden_ratio_angle",
                    angles=[angle_deg],
                    ratios=[self.GOLDEN_RATIO],
                    trigonometric_values={
                        "sin": math.sin(angle_rad),
                        "cos": math.cos(angle_rad),
                        "tan": math.tan(angle_rad)
                    },
                    confidence=0.9,
                    significance_score=0.95,
                    description=f"Golden ratio angle: {angle_deg:.1f}° (expected: {golden_angle_deg:.1f}°)",
                    mathematical_constant="golden_ratio"
                )
                relationships.append(relationship)
        
        return relationships
    
    def _find_pi_relationships(self, angles_deg: List[float], angles_rad: List[float]) -> List[TrigRelationship]:
        """Find relationships involving pi"""
        relationships = []
        
        for i, angle_deg in enumerate(angles_deg):
            angle_rad = angles_rad[i]
            
            # Check for simple fractions of pi
            pi_fractions = [
                (self.PI / 6, "π/6", 30),    # 30°
                (self.PI / 4, "π/4", 45),    # 45°
                (self.PI / 3, "π/3", 60),    # 60°
                (self.PI / 2, "π/2", 90),    # 90°
                (2 * self.PI / 3, "2π/3", 120),  # 120°
                (3 * self.PI / 4, "3π/4", 135),  # 135°
                (self.PI, "π", 180),         # 180°
            ]
            
            for pi_value, pi_name, expected_deg in pi_fractions:
                if abs(angle_rad - pi_value) < 0.05:  # Small tolerance in radians
                    relationship = TrigRelationship(
                        relationship_type="pi_fraction_angle",
                        angles=[angle_deg],
                        ratios=[pi_value / self.PI],
                        trigonometric_values={
                            "sin": math.sin(angle_rad),
                            "cos": math.cos(angle_rad),
                            "tan": math.tan(angle_rad) if abs(math.cos(angle_rad)) > 1e-10 else float('inf')
                        },
                        confidence=0.95,
                        significance_score=0.9,
                        description=f"Pi fraction angle: {angle_deg:.1f}° = {pi_name}",
                        mathematical_constant="pi"
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def _find_harmonic_relationships(self, angles_deg: List[float], angles_rad: List[float]) -> List[TrigRelationship]:
        """Find harmonic relationships between angles"""
        relationships = []
        
        # Look for angles that are in harmonic ratios
        for i in range(len(angles_deg)):
            for j in range(i + 1, len(angles_deg)):
                ratio = angles_deg[i] / angles_deg[j] if angles_deg[j] != 0 else 0
                
                # Check for simple harmonic ratios
                harmonic_ratios = [1/2, 2/3, 3/4, 4/5, 3/2, 4/3, 5/4, 2/1, 3/1]
                
                for harmonic_ratio in harmonic_ratios:
                    if abs(ratio - harmonic_ratio) < 0.05:
                        relationship = TrigRelationship(
                            relationship_type="harmonic_angle_ratio",
                            angles=[angles_deg[i], angles_deg[j]],
                            ratios=[ratio, harmonic_ratio],
                            trigonometric_values={
                                "ratio": ratio,
                                "harmonic_ratio": harmonic_ratio,
                                "difference": abs(ratio - harmonic_ratio)
                            },
                            confidence=0.8,
                            significance_score=0.7,
                            description=f"Harmonic ratio: {angles_deg[i]:.1f}° : {angles_deg[j]:.1f}° ≈ {harmonic_ratio:.3f}",
                            mathematical_constant="harmonic"
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _find_trigonometric_identities(self, angles_deg: List[float], angles_rad: List[float]) -> List[TrigRelationship]:
        """Find trigonometric identities in the angle data"""
        relationships = []
        
        # Look for complementary angles (sum to 90°)
        for i in range(len(angles_deg)):
            for j in range(i + 1, len(angles_deg)):
                angle_sum = angles_deg[i] + angles_deg[j]
                
                if abs(angle_sum - 90.0) < 2.0:
                    relationship = TrigRelationship(
                        relationship_type="complementary_angles",
                        angles=[angles_deg[i], angles_deg[j]],
                        ratios=[angles_deg[i] / 90.0, angles_deg[j] / 90.0],
                        trigonometric_values={
                            "sin_1": math.sin(angles_rad[i]),
                            "cos_1": math.cos(angles_rad[i]),
                            "sin_2": math.sin(angles_rad[j]),
                            "cos_2": math.cos(angles_rad[j]),
                            "identity_check": abs(math.sin(angles_rad[i]) - math.cos(angles_rad[j]))
                        },
                        confidence=0.9,
                        significance_score=0.8,
                        description=f"Complementary angles: {angles_deg[i]:.1f}° + {angles_deg[j]:.1f}° = {angle_sum:.1f}°",
                        mathematical_constant="complementary"
                    )
                    relationships.append(relationship)
                
                # Look for supplementary angles (sum to 180°)
                elif abs(angle_sum - 180.0) < 2.0:
                    relationship = TrigRelationship(
                        relationship_type="supplementary_angles",
                        angles=[angles_deg[i], angles_deg[j]],
                        ratios=[angles_deg[i] / 180.0, angles_deg[j] / 180.0],
                        trigonometric_values={
                            "sin_1": math.sin(angles_rad[i]),
                            "sin_2": math.sin(angles_rad[j]),
                            "identity_check": abs(math.sin(angles_rad[i]) - math.sin(angles_rad[j]))
                        },
                        confidence=0.9,
                        significance_score=0.8,
                        description=f"Supplementary angles: {angles_deg[i]:.1f}° + {angles_deg[j]:.1f}° = {angle_sum:.1f}°",
                        mathematical_constant="supplementary"
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def analyze_character_geometry(self, characters: List[CharacterBox]) -> Dict[str, Any]:
        """
        Analyze geometric relationships in character positioning
        
        Args:
            characters: List of character boxes with position data
            
        Returns:
            Dictionary with geometric analysis results
        """
        logger.info(f"Analyzing geometry of {len(characters)} characters")
        
        if len(characters) < 3:
            return {"error": "Need at least 3 characters for geometric analysis"}
        
        # Convert character boxes to points
        points = [Point(char.x + char.width/2, char.y + char.height/2) for char in characters]
        
        # Measure angles
        angle_measurements = self.measure_angles(points)
        
        # Detect geometric patterns
        geometric_patterns = self.detect_geometric_patterns(angle_measurements)
        
        # Find trigonometric relationships
        angles_only = [measurement.angle_degrees for measurement in angle_measurements]
        trig_relationships = self.find_trigonometric_relationships(angles_only)
        
        # Calculate distance statistics
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance_data = self.calculate_distances(points[i], points[j])
                distances.append(distance_data.distance)
        
        distance_stats = {
            "mean": statistics.mean(distances) if distances else 0,
            "median": statistics.median(distances) if distances else 0,
            "std": statistics.stdev(distances) if len(distances) > 1 else 0,
            "min": min(distances) if distances else 0,
            "max": max(distances) if distances else 0
        }
        
        return {
            "character_count": len(characters),
            "angle_measurements": len(angle_measurements),
            "significant_angles": [a for a in angle_measurements if a.significance_score > 0.8],
            "geometric_patterns": geometric_patterns,
            "trigonometric_relationships": trig_relationships,
            "distance_statistics": distance_stats,
            "analysis_summary": {
                "total_patterns": len(geometric_patterns),
                "high_significance_patterns": len([p for p in geometric_patterns if p.significance_score > 0.8]),
                "trigonometric_relationships": len(trig_relationships),
                "mathematical_constants_found": list(set([r.mathematical_constant for r in trig_relationships if r.mathematical_constant]))
            }
        }