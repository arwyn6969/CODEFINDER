"""
BardCode Spatial Analysis Engine
Based on Alan Green's research into geometric patterns and mathematical constants in text layouts
"""
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from collections import defaultdict
import math
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import itertools

from app.models.database_models import Page, Character, Word, Pattern
from app.services.geometric_index import GeometricIndex


class Point(NamedTuple):
    """Represents a 2D coordinate point"""
    x: float
    y: float


@dataclass
class Triangle:
    """Represents a triangle with three points"""
    p1: Point
    p2: Point
    p3: Point
    
    @property
    def sides(self) -> Tuple[float, float, float]:
        """Calculate the three side lengths"""
        a = euclidean(self.p1, self.p2)
        b = euclidean(self.p2, self.p3)
        c = euclidean(self.p3, self.p1)
        return (a, b, c)
    
    @property
    def angles(self) -> Tuple[float, float, float]:
        """Calculate the three angles in degrees"""
        a, b, c = self.sides
        # Using law of cosines with numeric guards
        def _clamp_cos(x: float) -> float:
            return max(-1.0, min(1.0, x))
        if min(a, b, c) == 0:
            return (0.0, 0.0, 0.0)
        try:
            angle_a = math.degrees(math.acos(_clamp_cos((b**2 + c**2 - a**2) / (2 * b * c))))
            angle_b = math.degrees(math.acos(_clamp_cos((a**2 + c**2 - b**2) / (2 * a * c))))
        except ValueError:
            # Fallback to zeros on numerical instability
            return (0.0, 0.0, 0.0)
        angle_c = 180 - angle_a - angle_b
        return (angle_a, angle_b, angle_c)
    
    @property
    def ratios(self) -> Tuple[float, float, float]:
        """Calculate side ratios"""
        a, b, c = self.sides
        if min(a, b, c) == 0:
            return (0, 0, 0)
        return (a/min(a, b, c), b/min(a, b, c), c/min(a, b, c))


@dataclass
class MathematicalConstant:
    """Represents a detected mathematical constant"""
    name: str
    symbol: str
    expected_value: float
    detected_value: float
    accuracy: float
    confidence: float
    context: str
    evidence: Dict[str, Any]


@dataclass
class GeometricConstruction:
    """Represents a geometric construction on a page"""
    construction_type: str
    points: List[Point]
    measurements: Dict[str, float]
    constants_detected: List[MathematicalConstant]
    significance_score: float
    description: str


class BardCodeAnalyzer:
    """
    Analyzes page layouts as coordinate systems to detect hidden geometric patterns
    and mathematical constants like those found in Alan Green's BardCode research
    """
    
    # Mathematical constants to detect
    MATHEMATICAL_CONSTANTS = {
        'pi': 3.141592653589793,
        'e': 2.718281828459045,
        'phi': 1.618033988749895,  # Golden ratio
        'phi_inverse': 0.618033988749895,  # 1/φ
        'sqrt_2': 1.4142135623730951,
        'sqrt_3': 1.7320508075688772,
        'sqrt_5': 2.23606797749979,
        'bruns_constant': 1.9021605823,  # Twin prime constant
        'euler_mascheroni': 0.5772156649015329,
        'catalan': 0.9159655941772190,
        'apery': 1.2020569031595943,  # ζ(3)
    }
    
    def __init__(self):
        self.tolerance = 0.01  # Default tolerance for constant detection
        self.min_significance = 0.7  # Minimum significance for pattern detection
    
    def analyze_page_geometry(self, characters: List[Character], 
                            page_width: float, page_height: float) -> Dict[str, Any]:
        """
        Analyze a page as a 2D coordinate system
        """
        if not characters:
            return {'error': 'No characters provided'}
        
        # Convert characters to coordinate points
        points = [Point(char.x, char.y) for char in characters]
        
        # Normalize coordinates to page dimensions
        normalized_points = self._normalize_coordinates(points, page_width, page_height)
        
        analysis_results = {
            'page_dimensions': {'width': page_width, 'height': page_height},
            'total_points': len(points),
            'coordinate_bounds': self._calculate_bounds(points),
            'triangular_constructions': [],
            'mathematical_constants': [],
            'geometric_patterns': [],
            'significance_scores': {}
        }
        
        # Find triangular constructions using optimized search
        # Convert points relative to page dimensions for better indexing
        # Note: normalized_points are already Point objects (named tuples)
        point_tuples = [(p.x, p.y) for p in normalized_points]
        self.geometric_index = GeometricIndex(point_tuples)
        
        triangles = self._find_significant_triangles(normalized_points)
        analysis_results['triangular_constructions'] = triangles
        
        # Detect mathematical constants in triangles
        constants = self._detect_constants_in_triangles(triangles)
        analysis_results['mathematical_constants'] = constants
        
        # Find geometric patterns
        patterns = self._find_geometric_patterns(normalized_points)
        analysis_results['geometric_patterns'] = patterns
        
        # Calculate overall significance
        analysis_results['significance_scores'] = self._calculate_significance_scores(
            triangles, constants, patterns
        )
        
        return analysis_results
    
    def detect_mathematical_constants(self, measurements: List[float], 
                                    tolerance: float = None) -> List[MathematicalConstant]:
        """
        Detect mathematical constants in a list of measurements
        """
        if tolerance is None:
            tolerance = self.tolerance
        
        detected_constants = []
        
        for measurement in measurements:
            for const_name, const_value in self.MATHEMATICAL_CONSTANTS.items():
                # Check direct match
                accuracy = self._calculate_accuracy(measurement, const_value)
                if accuracy >= (1 - tolerance):
                    constant = MathematicalConstant(
                        name=const_name,
                        symbol=self._get_constant_symbol(const_name),
                        expected_value=const_value,
                        detected_value=measurement,
                        accuracy=accuracy,
                        confidence=self._calculate_confidence(accuracy),
                        context='direct_measurement',
                        evidence={'measurement': measurement, 'tolerance': tolerance}
                    )
                    detected_constants.append(constant)
                
                # Check reciprocal
                if measurement != 0:
                    reciprocal = 1 / measurement
                    accuracy_reciprocal = self._calculate_accuracy(reciprocal, const_value)
                    if accuracy_reciprocal >= (1 - tolerance):
                        constant = MathematicalConstant(
                            name=f"{const_name}_reciprocal",
                            symbol=f"1/{self._get_constant_symbol(const_name)}",
                            expected_value=const_value,
                            detected_value=reciprocal,
                            accuracy=accuracy_reciprocal,
                            confidence=self._calculate_confidence(accuracy_reciprocal),
                            context='reciprocal_measurement',
                            evidence={'original_measurement': measurement, 'reciprocal': reciprocal}
                        )
                        detected_constants.append(constant)
        
        return detected_constants
    
    def find_sacred_geometry(self, points: List[Point]) -> List[GeometricConstruction]:
        """
        Find sacred geometry patterns (Pythagorean triangles, golden ratio, etc.)
        Based on Alan Green's BardCode research into sacred geometric constructions
        """
        constructions = []
        
        # Find Pythagorean triangles (3:4:5 ratios and other Pythagorean triples)
        pythagorean_triangles = self._find_pythagorean_triangles(points)
        constructions.extend(pythagorean_triangles)
        
        # Find golden ratio rectangles and constructions
        golden_constructions = self._find_golden_ratio_constructions(points)
        constructions.extend(golden_constructions)
        
        # Find equilateral triangles (60° angles)
        equilateral_triangles = self._find_equilateral_triangles(points)
        constructions.extend(equilateral_triangles)
        
        # Find vesica piscis patterns (overlapping circles)
        vesica_patterns = self._find_vesica_piscis_patterns(points)
        constructions.extend(vesica_patterns)
        
        # Find pentagonal/pentagram patterns (related to golden ratio)
        pentagonal_patterns = self._find_pentagonal_patterns(points)
        constructions.extend(pentagonal_patterns)
        
        # Find sacred triangles (30-60-90, 45-45-90)
        sacred_triangles = self._find_sacred_triangles(points)
        constructions.extend(sacred_triangles)
        
        # Find circular patterns and mandala-like structures
        circular_patterns = self._find_sacred_circles(points)
        constructions.extend(circular_patterns)
        
        # Find cross patterns (tau cross, Greek cross, etc.)
        cross_patterns = self._find_cross_patterns(points)
        constructions.extend(cross_patterns)
        
        return constructions
    
    def extract_coordinates(self, angles: List[float]) -> List[Tuple[float, float]]:
        """
        Extract potential geographic coordinates from angle measurements
        Based on Alan Green's discovery of latitude/longitude in the Sonnets page
        """
        coordinates = []
        
        for angle in angles:
            # Check if angle could be a latitude (0-90 degrees)
            if 0 <= angle <= 90:
                # Check if it matches known significant latitudes
                significant_latitudes = [
                    29.9792,  # Great Pyramid of Giza
                    51.4769,  # Stonehenge
                    40.7589,  # New York
                    55.7558,  # Moscow
                ]
                
                for lat in significant_latitudes:
                    if abs(angle - lat) < 0.1:  # Within 0.1 degree
                        coordinates.append((lat, None))
            
            # Check if angle could be a longitude (0-180 degrees)
            if 0 <= angle <= 180:
                significant_longitudes = [
                    31.13,    # Great Pyramid of Giza (from Greenwich)
                    -1.8262,  # Stonehenge
                    -74.0060, # New York
                    37.6176,  # Moscow
                ]
                
                for lon in significant_longitudes:
                    if abs(angle - abs(lon)) < 0.1:  # Within 0.1 degree
                        coordinates.append((None, lon))
        
        return coordinates
    
    def extract_geographic_coordinates_advanced(self, geometric_measurements: List[Dict[str, Any]], 
                                              tolerance: float = 0.01) -> Dict[str, Any]:
        """
        Advanced geographic coordinate extraction from geometric measurements
        Based on Alan Green's BardCode research methodology
        """
        coordinate_analysis = {
            'potential_coordinates': [],
            'coordinate_pairs': [],
            'historical_sites': [],
            'accuracy_scores': [],
            'extraction_method': 'bardcode_geometric_analysis'
        }
        
        # Extract angles and ratios from geometric measurements
        angles = []
        ratios = []
        distances = []
        
        for measurement in geometric_measurements:
            if measurement.get('type') == 'angle':
                angles.append(measurement.get('value', 0))
            elif measurement.get('type') == 'ratio':
                ratios.append(measurement.get('value', 0))
            elif measurement.get('type') == 'distance':
                distances.append(measurement.get('value', 0))
        
        # Method 1: Direct angle interpretation as coordinates
        direct_coordinates = self._extract_direct_angle_coordinates(angles, tolerance)
        coordinate_analysis['potential_coordinates'].extend(direct_coordinates)
        
        # Method 2: Ratio-based coordinate extraction
        ratio_coordinates = self._extract_ratio_based_coordinates(ratios, tolerance)
        coordinate_analysis['potential_coordinates'].extend(ratio_coordinates)
        
        # Method 3: Triangulation-based coordinate extraction
        triangulation_coordinates = self._extract_triangulation_coordinates(angles, distances, tolerance)
        coordinate_analysis['potential_coordinates'].extend(triangulation_coordinates)
        
        # Method 4: Mathematical constant coordinate relationships
        constant_coordinates = self._extract_constant_based_coordinates(angles, ratios, tolerance)
        coordinate_analysis['potential_coordinates'].extend(constant_coordinates)
        
        # Find coordinate pairs (lat/lon combinations)
        coordinate_analysis['coordinate_pairs'] = self._find_coordinate_pairs(
            coordinate_analysis['potential_coordinates']
        )
        
        # Match against historical sites
        coordinate_analysis['historical_sites'] = self._match_historical_sites(
            coordinate_analysis['coordinate_pairs'], tolerance
        )
        
        # Calculate accuracy scores
        coordinate_analysis['accuracy_scores'] = self._calculate_coordinate_accuracy(
            coordinate_analysis['coordinate_pairs']
        )
        
        return coordinate_analysis
    
    def extract_geographic_coordinates_advanced(self, geometric_measurements: List[Dict[str, Any]], 
                                              tolerance: float = 0.01) -> Dict[str, Any]:
        """
        Advanced geographic coordinate extraction from geometric measurements
        Based on Alan Green's BardCode research methodology
        """
        coordinate_analysis = {
            'potential_coordinates': [],
            'coordinate_pairs': [],
            'historical_sites': [],
            'accuracy_scores': [],
            'extraction_method': 'bardcode_geometric_analysis'
        }
        
        # Extract angles and ratios from geometric measurements
        angles = []
        ratios = []
        distances = []
        
        for measurement in geometric_measurements:
            if measurement.get('type') == 'angle':
                angles.append(measurement.get('value', 0))
            elif measurement.get('type') == 'ratio':
                ratios.append(measurement.get('value', 0))
            elif measurement.get('type') == 'distance':
                distances.append(measurement.get('value', 0))
        
        # Method 1: Direct angle interpretation as coordinates
        direct_coordinates = self._extract_direct_angle_coordinates(angles, tolerance)
        coordinate_analysis['potential_coordinates'].extend(direct_coordinates)
        
        # Method 2: Ratio-based coordinate extraction
        ratio_coordinates = self._extract_ratio_based_coordinates(ratios, tolerance)
        coordinate_analysis['potential_coordinates'].extend(ratio_coordinates)
        
        # Method 3: Triangulation-based coordinate extraction
        triangulation_coordinates = self._extract_triangulation_coordinates(angles, distances, tolerance)
        coordinate_analysis['potential_coordinates'].extend(triangulation_coordinates)
        
        # Method 4: Mathematical constant coordinate relationships
        constant_coordinates = self._extract_constant_based_coordinates(angles, ratios, tolerance)
        coordinate_analysis['potential_coordinates'].extend(constant_coordinates)
        
        # Find coordinate pairs (lat/lon combinations)
        coordinate_analysis['coordinate_pairs'] = self._find_coordinate_pairs(
            coordinate_analysis['potential_coordinates']
        )
        
        # Match against historical sites
        coordinate_analysis['historical_sites'] = self._match_historical_sites(
            coordinate_analysis['coordinate_pairs'], tolerance
        )
        
        # Calculate accuracy scores
        coordinate_analysis['accuracy_scores'] = self._calculate_coordinate_accuracy(
            coordinate_analysis['coordinate_pairs']
        )
        
        return coordinate_analysis
    
    def _extract_direct_angle_coordinates(self, angles: List[float], tolerance: float) -> List[Dict[str, Any]]:
        """Extract coordinates by interpreting angles directly as lat/lon values"""
        coordinates = []
        
        for angle in angles:
            # Normalize angle to 0-360 range
            normalized_angle = angle % 360
            
            # Check for latitude (0-90 degrees)
            if 0 <= normalized_angle <= 90:
                coord = {
                    'type': 'latitude',
                    'value': normalized_angle,
                    'source_angle': angle,
                    'method': 'direct_angle_interpretation',
                    'confidence': self._calculate_coordinate_confidence(normalized_angle, 'latitude')
                }
                coordinates.append(coord)
            
            # Check for longitude (0-180 degrees, can be negative)
            if 0 <= normalized_angle <= 180:
                coord = {
                    'type': 'longitude',
                    'value': normalized_angle,
                    'source_angle': angle,
                    'method': 'direct_angle_interpretation',
                    'confidence': self._calculate_coordinate_confidence(normalized_angle, 'longitude')
                }
                coordinates.append(coord)
                
                # Also check negative longitude
                negative_coord = {
                    'type': 'longitude',
                    'value': -normalized_angle,
                    'source_angle': angle,
                    'method': 'direct_angle_interpretation_negative',
                    'confidence': self._calculate_coordinate_confidence(-normalized_angle, 'longitude')
                }
                coordinates.append(negative_coord)
        
        return coordinates
    
    def _extract_ratio_based_coordinates(self, ratios: List[float], tolerance: float) -> List[Dict[str, Any]]:
        """Extract coordinates from geometric ratios"""
        coordinates = []
        
        for ratio in ratios:
            # Convert ratios to potential coordinate values
            # Method 1: Ratio * 90 (for latitude scaling)
            lat_candidate = ratio * 90
            if 0 <= lat_candidate <= 90:
                coord = {
                    'type': 'latitude',
                    'value': lat_candidate,
                    'source_ratio': ratio,
                    'method': 'ratio_scaling_90',
                    'confidence': self._calculate_coordinate_confidence(lat_candidate, 'latitude')
                }
                coordinates.append(coord)
            
            # Method 2: Ratio * 180 (for longitude scaling)
            lon_candidate = ratio * 180
            if -180 <= lon_candidate <= 180:
                coord = {
                    'type': 'longitude',
                    'value': lon_candidate,
                    'source_ratio': ratio,
                    'method': 'ratio_scaling_180',
                    'confidence': self._calculate_coordinate_confidence(lon_candidate, 'longitude')
                }
                coordinates.append(coord)
            
            # Method 3: Mathematical constant relationships
            for const_name, const_value in self.MATHEMATICAL_CONSTANTS.items():
                if abs(ratio - const_value) < tolerance:
                    # Use constant to derive coordinates
                    derived_coords = self._derive_coordinates_from_constant(const_name, const_value)
                    coordinates.extend(derived_coords)
        
        return coordinates
    
    def _extract_triangulation_coordinates(self, angles: List[float], distances: List[float], 
                                         tolerance: float) -> List[Dict[str, Any]]:
        """Extract coordinates using triangulation methods"""
        coordinates = []
        
        # Look for angle combinations that could represent coordinate triangulation
        for i, angle1 in enumerate(angles):
            for j, angle2 in enumerate(angles[i+1:], i+1):
                # Check if angles could form a coordinate triangulation
                if self._could_be_coordinate_triangulation(angle1, angle2):
                    lat, lon = self._triangulate_coordinates(angle1, angle2, distances)
                    if lat is not None and lon is not None:
                        coord_pair = {
                            'type': 'coordinate_pair',
                            'latitude': lat,
                            'longitude': lon,
                            'source_angles': [angle1, angle2],
                            'method': 'triangulation',
                            'confidence': self._calculate_triangulation_confidence(angle1, angle2)
                        }
                        coordinates.append(coord_pair)
        
        return coordinates
    
    def _extract_constant_based_coordinates(self, angles: List[float], ratios: List[float], 
                                          tolerance: float) -> List[Dict[str, Any]]:
        """Extract coordinates based on mathematical constant relationships"""
        coordinates = []
        
        # Check for π-based coordinates (like π/180 * degrees)
        pi = self.MATHEMATICAL_CONSTANTS['pi']
        for angle in angles:
            pi_ratio = angle / 180 * pi
            if self._is_significant_coordinate_value(pi_ratio):
                coord = {
                    'type': 'derived_coordinate',
                    'value': pi_ratio,
                    'source_angle': angle,
                    'method': 'pi_ratio_conversion',
                    'mathematical_constant': 'pi',
                    'confidence': 0.7
                }
                coordinates.append(coord)
        
        # Check for φ (golden ratio) based coordinates
        phi = self.MATHEMATICAL_CONSTANTS['phi']
        for ratio in ratios:
            if abs(ratio - phi) < tolerance:
                # Golden ratio could encode special coordinates
                golden_coords = self._derive_golden_ratio_coordinates(ratio)
                coordinates.extend(golden_coords)
        
        return coordinates
    
    def _find_coordinate_pairs(self, potential_coordinates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find valid latitude/longitude pairs from potential coordinates"""
        pairs = []
        
        latitudes = [c for c in potential_coordinates if c.get('type') == 'latitude']
        longitudes = [c for c in potential_coordinates if c.get('type') == 'longitude']
        
        for lat_coord in latitudes:
            for lon_coord in longitudes:
                pair = {
                    'latitude': lat_coord['value'],
                    'longitude': lon_coord['value'],
                    'lat_source': lat_coord,
                    'lon_source': lon_coord,
                    'combined_confidence': (lat_coord['confidence'] + lon_coord['confidence']) / 2,
                    'methods': [lat_coord['method'], lon_coord['method']]
                }
                pairs.append(pair)
        
        return pairs
    
    def _match_historical_sites(self, coordinate_pairs: List[Dict[str, Any]], 
                               tolerance: float) -> List[Dict[str, Any]]:
        """Match coordinate pairs against known historical sites"""
        historical_sites = [
            {'name': 'Great Pyramid of Giza', 'lat': 29.9792, 'lon': 31.1342, 'significance': 'ancient_wonder'},
            {'name': 'Stonehenge', 'lat': 51.1789, 'lon': -1.8262, 'significance': 'megalithic_monument'},
            {'name': 'Parthenon, Athens', 'lat': 37.9715, 'lon': 23.7267, 'significance': 'classical_architecture'},
            {'name': 'Vatican City', 'lat': 41.9029, 'lon': 12.4534, 'significance': 'religious_center'},
            {'name': 'Jerusalem Temple Mount', 'lat': 31.7780, 'lon': 35.2354, 'significance': 'holy_site'},
            {'name': 'Delphi Oracle', 'lat': 38.4824, 'lon': 22.5012, 'significance': 'ancient_oracle'},
            {'name': 'Glastonbury Tor', 'lat': 51.1441, 'lon': -2.6987, 'significance': 'mystical_site'},
            {'name': 'Rosslyn Chapel', 'lat': 55.8552, 'lon': -3.1589, 'significance': 'templar_mystery'},
        ]
        
        matches = []
        
        for pair in coordinate_pairs:
            for site in historical_sites:
                lat_diff = abs(pair['latitude'] - site['lat'])
                lon_diff = abs(pair['longitude'] - site['lon'])
                
                # Check if coordinates are close to historical site
                if lat_diff < tolerance * 100 and lon_diff < tolerance * 100:  # Scale tolerance for coordinates
                    match = {
                        'site': site,
                        'detected_coordinates': pair,
                        'accuracy': {
                            'latitude_error': lat_diff,
                            'longitude_error': lon_diff,
                            'total_error': math.sqrt(lat_diff**2 + lon_diff**2)
                        },
                        'match_confidence': 1 - (lat_diff + lon_diff) / 2
                    }
                    matches.append(match)
        
        return matches
    
    def _calculate_coordinate_confidence(self, value: float, coord_type: str) -> float:
        """Calculate confidence score for a coordinate value"""
        if coord_type == 'latitude':
            # Latitude confidence based on how reasonable the value is (0-90)
            if 0 <= value <= 90:
                return 0.8 + 0.2 * (1 - abs(value - 45) / 45)  # Higher confidence near middle latitudes
            else:
                return 0.0
        elif coord_type == 'longitude':
            # Longitude confidence based on how reasonable the value is (-180 to 180)
            if -180 <= value <= 180:
                return 0.7 + 0.3 * (1 - abs(value) / 180)  # Higher confidence near prime meridian
            else:
                return 0.0
        
        return 0.5  # Default confidence
    
    def _calculate_coordinate_accuracy(self, coordinate_pairs: List[Dict[str, Any]]) -> List[float]:
        """Calculate accuracy scores for coordinate pairs"""
        return [pair.get('combined_confidence', 0.5) for pair in coordinate_pairs]
    
    def _could_be_coordinate_triangulation(self, angle1: float, angle2: float) -> bool:
        """Check if two angles could represent coordinate triangulation"""
        # Simple heuristic: angles should be different and within reasonable ranges
        return abs(angle1 - angle2) > 5 and 0 <= angle1 <= 180 and 0 <= angle2 <= 180
    
    def _triangulate_coordinates(self, angle1: float, angle2: float, 
                                distances: List[float]) -> Tuple[Optional[float], Optional[float]]:
        """Triangulate coordinates from angles and distances"""
        # Simplified triangulation - in practice this would be more complex
        # This is a placeholder implementation
        if distances:
            lat = (angle1 + angle2) / 2
            lon = distances[0] if distances else angle1
            
            # Ensure coordinates are in valid ranges
            if 0 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        
        return None, None
    
    def _calculate_triangulation_confidence(self, angle1: float, angle2: float) -> float:
        """Calculate confidence for triangulated coordinates"""
        # Higher confidence for angles that are more different (better triangulation)
        angle_diff = abs(angle1 - angle2)
        return min(0.9, 0.5 + angle_diff / 180)
    
    def _is_significant_coordinate_value(self, value: float) -> bool:
        """Check if a value could be a significant coordinate"""
        return (-180 <= value <= 180) and (abs(value) > 1)  # Exclude very small values
    
    def _derive_golden_ratio_coordinates(self, ratio: float) -> List[Dict[str, Any]]:
        """Derive coordinates from golden ratio relationships"""
        coordinates = []
        
        # Golden ratio could encode special latitudes/longitudes
        phi = self.MATHEMATICAL_CONSTANTS['phi']
        
        # Method 1: φ * 30 (could give latitude around 48.5°)
        lat_candidate = ratio * 30
        if 0 <= lat_candidate <= 90:
            coord = {
                'type': 'latitude',
                'value': lat_candidate,
                'source_ratio': ratio,
                'method': 'golden_ratio_scaling',
                'mathematical_constant': 'phi',
                'confidence': 0.75
            }
            coordinates.append(coord)
        
        return coordinates
    
    def _derive_coordinates_from_constant(self, const_name: str, const_value: float) -> List[Dict[str, Any]]:
        """Derive coordinates from mathematical constants"""
        coordinates = []
        
        if const_name == 'pi':
            # π could encode various coordinate relationships
            lat_candidate = const_value * 10  # π * 10 ≈ 31.4° (close to Giza latitude)
            if 0 <= lat_candidate <= 90:
                coord = {
                    'type': 'latitude',
                    'value': lat_candidate,
                    'method': 'pi_constant_derivation',
                    'mathematical_constant': const_name,
                    'confidence': 0.8
                }
                coordinates.append(coord)
        
        elif const_name == 'e':
            # e could encode longitude relationships
            lon_candidate = const_value * 10  # e * 10 ≈ 27.2°
            if -180 <= lon_candidate <= 180:
                coord = {
                    'type': 'longitude',
                    'value': lon_candidate,
                    'method': 'e_constant_derivation',
                    'mathematical_constant': const_name,
                    'confidence': 0.75
                }
                coordinates.append(coord)
        
        return coordinates
    
    def validate_coordinate_significance(self, coordinate_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the significance of extracted coordinate pairs
        Based on statistical analysis and historical site matching
        """
        validation_results = {
            'total_pairs': len(coordinate_pairs),
            'significant_pairs': [],
            'historical_matches': 0,
            'confidence_distribution': {},
            'validation_score': 0.0
        }
        
        if not coordinate_pairs:
            return validation_results
        
        # Analyze confidence distribution
        confidences = [pair.get('combined_confidence', 0) for pair in coordinate_pairs]
        validation_results['confidence_distribution'] = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        }
        
        # Identify significant pairs (high confidence)
        high_confidence_threshold = 0.7
        for pair in coordinate_pairs:
            if pair.get('combined_confidence', 0) >= high_confidence_threshold:
                validation_results['significant_pairs'].append(pair)
        
        # Check for historical site matches
        historical_matches = self._match_historical_sites(coordinate_pairs, 0.1)
        validation_results['historical_matches'] = len(historical_matches)
        
        # Calculate overall validation score
        confidence_score = validation_results['confidence_distribution']['mean']
        historical_score = min(1.0, validation_results['historical_matches'] / 3.0)  # Normalize by expected matches
        significance_score = len(validation_results['significant_pairs']) / max(1, len(coordinate_pairs))
        
        validation_results['validation_score'] = (
            confidence_score * 0.4 + 
            historical_score * 0.4 + 
            significance_score * 0.2
        )
        
        return validation_results
    
    def generate_coordinate_report(self, coordinate_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive report of coordinate extraction analysis
        """
        report = {
            'summary': {
                'extraction_method': coordinate_analysis.get('extraction_method', 'unknown'),
                'total_potential_coordinates': len(coordinate_analysis.get('potential_coordinates', [])),
                'coordinate_pairs_found': len(coordinate_analysis.get('coordinate_pairs', [])),
                'historical_site_matches': len(coordinate_analysis.get('historical_sites', []))
            },
            'coordinate_pairs': [],
            'historical_matches': [],
            'extraction_methods': {},
            'confidence_analysis': {},
            'recommendations': []
        }
        
        # Process coordinate pairs
        for pair in coordinate_analysis.get('coordinate_pairs', []):
            pair_info = {
                'latitude': pair['latitude'],
                'longitude': pair['longitude'],
                'confidence': pair['combined_confidence'],
                'extraction_methods': pair['methods'],
                'coordinate_string': f"{pair['latitude']:.4f}°, {pair['longitude']:.4f}°"
            }
            report['coordinate_pairs'].append(pair_info)
        
        # Process historical matches
        for match in coordinate_analysis.get('historical_sites', []):
            match_info = {
                'site_name': match['site']['name'],
                'site_significance': match['site']['significance'],
                'detected_coordinates': f"{match['detected_coordinates']['latitude']:.4f}°, {match['detected_coordinates']['longitude']:.4f}°",
                'actual_coordinates': f"{match['site']['lat']:.4f}°, {match['site']['lon']:.4f}°",
                'accuracy': match['accuracy'],
                'match_confidence': match['match_confidence']
            }
            report['historical_matches'].append(match_info)
        
        # Analyze extraction methods
        method_counts = {}
        for coord in coordinate_analysis.get('potential_coordinates', []):
            method = coord.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        report['extraction_methods'] = method_counts
        
        # Confidence analysis
        if coordinate_analysis.get('coordinate_pairs'):
            confidences = [pair['combined_confidence'] for pair in coordinate_analysis['coordinate_pairs']]
            report['confidence_analysis'] = {
                'average_confidence': np.mean(confidences),
                'confidence_range': [np.min(confidences), np.max(confidences)],
                'high_confidence_pairs': len([c for c in confidences if c >= 0.8])
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_coordinate_recommendations(coordinate_analysis)
        
        return report
    
    def _generate_coordinate_recommendations(self, coordinate_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on coordinate analysis results"""
        recommendations = []
        
        coordinate_pairs = coordinate_analysis.get('coordinate_pairs', [])
        historical_matches = coordinate_analysis.get('historical_sites', [])
        
        if not coordinate_pairs:
            recommendations.append("No coordinate pairs were extracted. Consider adjusting tolerance parameters or using different geometric measurements.")
        
        if len(coordinate_pairs) > 10:
            recommendations.append("Large number of coordinate pairs found. Consider filtering by confidence score to focus on most significant results.")
        
        if historical_matches:
            recommendations.append(f"Found {len(historical_matches)} matches with historical sites. These coordinates may have special significance.")
        
        high_confidence_pairs = [p for p in coordinate_pairs if p.get('combined_confidence', 0) >= 0.8]
        if high_confidence_pairs:
            recommendations.append(f"Focus on {len(high_confidence_pairs)} high-confidence coordinate pairs for further analysis.")
        
        if len(set(coord.get('method', '') for coord in coordinate_analysis.get('potential_coordinates', []))) > 3:
            recommendations.append("Multiple extraction methods yielded results. Cross-validate findings using different approaches.")
        
        return recommendations
    
    # Helper methods for geometric analysis
    def _normalize_coordinates(self, points: List[Point], width: float, height: float) -> List[Point]:
        """Normalize coordinates to 0-1 range"""
        if not points or width == 0 or height == 0:
            return points
        
        return [Point(p.x / width, p.y / height) for p in points]
    
    def _calculate_bounds(self, points: List[Point]) -> Dict[str, float]:
        """Calculate bounding box for points"""
        if not points:
            return {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}
        
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        
        return {
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords)
        }
    
    def _find_significant_triangles(self, points: List[Point]) -> List[Triangle]:
        """Find triangles with significant geometric properties"""
        triangles = []
        
        # Generate all possible triangles from points
        for point_combo in itertools.combinations(points, 3):
            triangle = Triangle(point_combo[0], point_combo[1], point_combo[2])
            
            # Check if triangle has significant properties
            if self._is_significant_triangle(triangle):
                triangles.append(triangle)
        
        return triangles
    
    def _is_significant_triangle(self, triangle: Triangle) -> bool:
        """Check if a triangle has significant geometric properties"""
        angles = triangle.angles
        sides = triangle.sides
        
        # Check for special angles (30, 45, 60, 90 degrees)
        special_angles = [30, 45, 60, 90]
        for angle in angles:
            for special in special_angles:
                if abs(angle - special) < 2:  # Within 2 degrees
                    return True
        
        # Check for Pythagorean relationships
        sorted_sides = sorted(sides)
        if len(sorted_sides) == 3 and sorted_sides[0] > 0:
            # Check if a² + b² ≈ c²
            if abs(sorted_sides[0]**2 + sorted_sides[1]**2 - sorted_sides[2]**2) < 0.01:
                return True
        
        return False
    
    def _detect_constants_in_triangles(self, triangles: List[Triangle]) -> List[MathematicalConstant]:
        """Detect mathematical constants in triangle measurements"""
        constants = []
        
        for triangle in triangles:
            # Check angles for constants
            angle_constants = self.detect_mathematical_constants(list(triangle.angles))
            constants.extend(angle_constants)
            
            # Check side ratios for constants
            ratio_constants = self.detect_mathematical_constants(list(triangle.ratios))
            constants.extend(ratio_constants)
        
        return constants
    
    def _find_geometric_patterns(self, points: List[Point]) -> List[Dict[str, Any]]:
        """Find geometric patterns in point arrangements"""
        patterns = []
        
        # Find linear patterns
        linear_patterns = self._find_linear_patterns(points)
        patterns.extend(linear_patterns)
        
        # Find circular patterns
        circular_patterns = self._find_circular_patterns(points)
        patterns.extend(circular_patterns)
        
        # Find grid patterns
        grid_patterns = self._find_grid_patterns(points)
        patterns.extend(grid_patterns)
        
        return patterns
    
    def _find_linear_patterns(self, points: List[Point]) -> List[Dict[str, Any]]:
        """Find points arranged in linear patterns"""
        patterns = []
        
        # Check for collinear points
        for point_combo in itertools.combinations(points, 3):
            if self._are_collinear(point_combo[0], point_combo[1], point_combo[2]):
                pattern = {
                    'type': 'linear',
                    'points': list(point_combo),
                    'description': 'Collinear points'
                }
                patterns.append(pattern)
        
        return patterns
    
    def _are_collinear(self, p1: Point, p2: Point, p3: Point) -> bool:
        """Check if three points are collinear"""
        # Calculate area of triangle formed by three points
        area = abs((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2)
        return area < 0.01  # Very small area indicates collinearity
    
    def _find_circular_patterns(self, points: List[Point]) -> List[Dict[str, Any]]:
        """Find points arranged in circular patterns"""
        patterns = []
        
        # Look for points equidistant from a center
        for center_point in points:
            distances = [euclidean(center_point, p) for p in points if p != center_point]
            
            # Group points by similar distances
            distance_groups = defaultdict(list)
            for i, distance in enumerate(distances):
                for group_dist in distance_groups.keys():
                    if abs(distance - group_dist) < 0.1:
                        distance_groups[group_dist].append(points[i])
                        break
                else:
                    distance_groups[distance] = [points[i]]
            
            # Check for circular patterns (multiple points at same distance)
            for distance, group_points in distance_groups.items():
                if len(group_points) >= 3:
                    pattern = {
                        'type': 'circular',
                        'center': center_point,
                        'radius': distance,
                        'points': group_points,
                        'description': f'Circular pattern with {len(group_points)} points'
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _find_grid_patterns(self, points: List[Point]) -> List[Dict[str, Any]]:
        """Find points arranged in grid patterns"""
        patterns = []
        
        # Group points by x and y coordinates
        x_groups = defaultdict(list)
        y_groups = defaultdict(list)
        
        for point in points:
            # Group by similar x coordinates
            for x_coord in x_groups.keys():
                if abs(point.x - x_coord) < 0.1:
                    x_groups[x_coord].append(point)
                    break
            else:
                x_groups[point.x] = [point]
            
            # Group by similar y coordinates
            for y_coord in y_groups.keys():
                if abs(point.y - y_coord) < 0.1:
                    y_groups[y_coord].append(point)
                    break
            else:
                y_groups[point.y] = [point]
        
        # Check for grid patterns (multiple points in both x and y groups)
        if len(x_groups) >= 2 and len(y_groups) >= 2:
            pattern = {
                'type': 'grid',
                'x_lines': len(x_groups),
                'y_lines': len(y_groups),
                'total_points': len(points),
                'description': f'Grid pattern with {len(x_groups)}x{len(y_groups)} structure'
            }
            patterns.append(pattern)
        
        return patterns
    
    def _calculate_significance_scores(self, triangles: List[Triangle], 
                                     constants: List[MathematicalConstant],
                                     patterns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate significance scores for different aspects of the analysis"""
        scores = {
            'triangle_significance': 0.0,
            'constant_significance': 0.0,
            'pattern_significance': 0.0,
            'overall_significance': 0.0
        }
        
        # Triangle significance
        if triangles:
            triangle_scores = []
            for triangle in triangles:
                # Score based on special angles and ratios
                angle_score = sum(1 for angle in triangle.angles if any(abs(angle - special) < 2 for special in [30, 45, 60, 90]))
                ratio_score = sum(1 for ratio in triangle.ratios if any(abs(ratio - const) < 0.1 for const in self.MATHEMATICAL_CONSTANTS.values()))
                triangle_scores.append((angle_score + ratio_score) / 6)  # Normalize
            
            scores['triangle_significance'] = np.mean(triangle_scores)
        
        # Constants significance
        if constants:
            constant_scores = [const.confidence for const in constants]
            scores['constant_significance'] = np.mean(constant_scores)
        
        # Pattern significance
        if patterns:
            pattern_scores = []
            for pattern in patterns:
                if pattern['type'] == 'circular':
                    pattern_scores.append(0.8)
                elif pattern['type'] == 'grid':
                    pattern_scores.append(0.9)
                elif pattern['type'] == 'linear':
                    pattern_scores.append(0.6)
                else:
                    pattern_scores.append(0.5)
            
            scores['pattern_significance'] = np.mean(pattern_scores)
        
        # Overall significance
        scores['overall_significance'] = (
            scores['triangle_significance'] * 0.4 +
            scores['constant_significance'] * 0.4 +
            scores['pattern_significance'] * 0.2
        )
        
        return scores
    
    def _calculate_accuracy(self, detected: float, expected: float) -> float:
        """Calculate accuracy between detected and expected values"""
        if expected == 0:
            return 1.0 if detected == 0 else 0.0
        
        error = abs(detected - expected) / abs(expected)
        return max(0.0, 1.0 - error)
    
    def _calculate_confidence(self, accuracy: float) -> float:
        """Calculate confidence score from accuracy"""
        return min(1.0, accuracy * 1.2)  # Boost confidence slightly
    
    def _get_constant_symbol(self, const_name: str) -> str:
        """Get symbol for mathematical constant"""
        symbols = {
            'pi': 'π',
            'e': 'e',
            'phi': 'φ',
            'phi_inverse': '1/φ',
            'sqrt_2': '√2',
            'sqrt_3': '√3',
            'sqrt_5': '√5',
            'bruns_constant': 'B₂',
            'euler_mascheroni': 'γ',
            'catalan': 'G',
            'apery': 'ζ(3)'
        }
        return symbols.get(const_name, const_name)
    
    def construct_triangles(self, points: List[Point]) -> List[Triangle]:
        """Construct all possible triangles from a list of points"""
        triangles = []
        
        for point_combo in itertools.combinations(points, 3):
            triangle = Triangle(point_combo[0], point_combo[1], point_combo[2])
            triangles.append(triangle)
        
        return triangles
    
    # Additional helper methods for sacred geometry patterns
    def _find_pythagorean_triangles(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find Pythagorean triangles (3:4:5 ratios and other Pythagorean triples)"""
        constructions = []
        triangles = self.construct_triangles(points)
        
        for triangle in triangles:
            sides = sorted(triangle.sides)
            if len(sides) == 3 and sides[0] > 0:
                # Check for Pythagorean relationship: a² + b² = c²
                if abs(sides[0]**2 + sides[1]**2 - sides[2]**2) < 0.01:
                    construction = GeometricConstruction(
                        construction_type='pythagorean_triangle',
                        points=[triangle.p1, triangle.p2, triangle.p3],
                        measurements={'sides': sides, 'pythagorean_error': abs(sides[0]**2 + sides[1]**2 - sides[2]**2)},
                        constants_detected=[],
                        significance_score=0.9,
                        description=f'Pythagorean triangle ({sides[0]:.2f}:{sides[1]:.2f}:{sides[2]:.2f})'
                    )
                    constructions.append(construction)
        
        return constructions
    
    def _find_equilateral_triangles(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find equilateral triangles (60° angles)"""
        constructions = []
        triangles = self.construct_triangles(points)
        
        for triangle in triangles:
            try:
                angles = triangle.angles
            except Exception:
                # Skip bad triangles due to numeric instability
                continue
            
            # Check if all angles are close to 60°
            if all(abs(angle - 60) < 2 for angle in angles):
                construction = GeometricConstruction(
                    construction_type='equilateral_triangle',
                    points=[triangle.p1, triangle.p2, triangle.p3],
                    measurements={'angles': angles, 'sides': triangle.sides},
                    constants_detected=[],
                    significance_score=0.85,
                    description='Equilateral triangle (60° angles)'
                )
                constructions.append(construction)
        
        return constructions
    
    def _find_sacred_circles(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find circular patterns and mandala-like structures"""
        constructions = []
        
        # Find circles with multiple points on circumference
        for center_candidate in points:
            # Find points at similar distances from center
            distances = [(p, euclidean(center_candidate, p)) for p in points if p != center_candidate]
            
            # Group by similar distances
            distance_groups = defaultdict(list)
            for point, distance in distances:
                for group_dist in distance_groups.keys():
                    if abs(distance - group_dist) < 0.1:
                        distance_groups[group_dist].append(point)
                        break
                else:
                    distance_groups[distance] = [point]
            
            # Look for groups with 3+ points (potential circles)
            for radius, circle_points in distance_groups.items():
                if len(circle_points) >= 3:
                    construction = GeometricConstruction(
                        construction_type='sacred_circle',
                        points=[center_candidate] + circle_points,
                        measurements={'radius': radius, 'circumference_points': len(circle_points)},
                        constants_detected=[],
                        significance_score=0.7 + 0.1 * len(circle_points),
                        description=f'Sacred circle with {len(circle_points)} points on circumference'
                    )
                    constructions.append(construction)
        
        return constructions
    
    def _find_cross_patterns(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find cross patterns (tau cross, Greek cross, etc.)"""
        constructions = []
        
        # Look for perpendicular line intersections
        for center_point in points:
            # Find points that could form cross arms
            other_points = [p for p in points if p != center_point]
            
            # Group points by direction from center
            directions = []
            for point in other_points:
                dx = point.x - center_point.x
                dy = point.y - center_point.y
                
                if abs(dx) > 0.01 or abs(dy) > 0.01:  # Not the same point
                    angle = math.degrees(math.atan2(dy, dx))
                    directions.append((point, angle))
            
            # Look for perpendicular pairs (90° apart)
            for i, (point1, angle1) in enumerate(directions):
                for j, (point2, angle2) in enumerate(directions[i+1:], i+1):
                    angle_diff = abs(angle1 - angle2)
                    if abs(angle_diff - 90) < 5 or abs(angle_diff - 270) < 5:  # Perpendicular
                        construction = GeometricConstruction(
                            construction_type='cross_pattern',
                            points=[center_point, point1, point2],
                            measurements={'angle_difference': angle_diff},
                            constants_detected=[],
                            significance_score=0.75,
                            description='Cross pattern (perpendicular lines)'
                        )
                        constructions.append(construction)
        
        return constructions
    
    def _is_regular_pentagon(self, five_points: Tuple[Point, ...]) -> bool:
        """Check if five points form a regular pentagon"""
        points = list(five_points)
        
        # Calculate center point
        center_x = sum(p.x for p in points) / 5
        center_y = sum(p.y for p in points) / 5
        center = Point(center_x, center_y)
        
        # Check if all points are equidistant from center
        distances = [euclidean(center, p) for p in points]
        avg_distance = sum(distances) / len(distances)
        
        # All distances should be similar
        if not all(abs(d - avg_distance) < 0.1 for d in distances):
            return False
        
        # Check angles between consecutive points
        angles = []
        for i in range(5):
            p1 = points[i]
            p2 = points[(i + 1) % 5]
            
            angle1 = math.degrees(math.atan2(p1.y - center.y, p1.x - center.x))
            angle2 = math.degrees(math.atan2(p2.y - center.y, p2.x - center.x))
            
            angle_diff = abs(angle2 - angle1)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            angles.append(angle_diff)
        
        # Regular pentagon should have 72° between consecutive points
        return all(abs(angle - 72) < 5 for angle in angles)
    
    def _calculate_pentagon_angles(self, five_points: Tuple[Point, ...]) -> List[float]:
        """Calculate internal angles of a pentagon"""
        points = list(five_points)
        angles = []
        
        for i in range(5):
            p1 = points[(i - 1) % 5]
            p2 = points[i]
            p3 = points[(i + 1) % 5]
            
            # Calculate angle at p2
            v1 = Point(p1.x - p2.x, p1.y - p2.y)
            v2 = Point(p3.x - p2.x, p3.y - p2.y)
            
            dot_product = v1.x * v2.x + v1.y * v2.y
            mag1 = math.sqrt(v1.x**2 + v1.y**2)
            mag2 = math.sqrt(v2.x**2 + v2.y**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                angle = math.degrees(math.acos(cos_angle))
                angles.append(angle)
        
        return angles
    
    def construct_triangles(self, points: List[Point]) -> List[Triangle]:
        """
        Construct all possible triangles from a set of points
        """
        triangles = []
        
        # Generate all combinations of 3 points
        for point_combo in itertools.combinations(points, 3):
            triangle = Triangle(point_combo[0], point_combo[1], point_combo[2])
            
            # Filter out degenerate triangles (collinear points)
            if self._is_valid_triangle(triangle):
                triangles.append(triangle)
        
        return triangles
    
    def validate_significance(self, construction: GeometricConstruction) -> Dict[str, float]:
        """
        Validate the statistical significance of a geometric construction
        """
        significance_tests = {}
        
        # Test 1: Number of constants detected
        constants_score = len(construction.constants_detected) / 5.0  # Max 5 constants
        significance_tests['constants_count'] = min(constants_score, 1.0)
        
        # Test 2: Average accuracy of detected constants
        if construction.constants_detected:
            avg_accuracy = np.mean([c.accuracy for c in construction.constants_detected])
            significance_tests['constants_accuracy'] = avg_accuracy
        else:
            significance_tests['constants_accuracy'] = 0.0
        
        # Test 3: Geometric complexity
        complexity_score = len(construction.points) / 10.0  # Normalize by max expected points
        significance_tests['geometric_complexity'] = min(complexity_score, 1.0)
        
        # Test 4: Pattern uniqueness (placeholder - would need comparison with random patterns)
        significance_tests['pattern_uniqueness'] = 0.8  # Default high uniqueness
        
        # Combined significance score
        weights = [0.3, 0.4, 0.2, 0.1]  # Weight the different tests
        combined_score = sum(score * weight for score, weight in 
                           zip(significance_tests.values(), weights))
        
        significance_tests['combined_significance'] = combined_score
        
        return significance_tests
    
    def _normalize_coordinates(self, points: List[Point], width: float, height: float) -> List[Point]:
        """Normalize coordinates to 0-1 range"""
        if not points or width == 0 or height == 0:
            return points
        
        return [Point(p.x / width, p.y / height) for p in points]
    
    def _calculate_bounds(self, points: List[Point]) -> Dict[str, float]:
        """Calculate bounding box of points"""
        if not points:
            return {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}
        
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        
        return {
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords)
        }
    
    def _find_significant_triangles(self, points: List[Point]) -> List[Dict[str, Any]]:
        """
        Find triangles with significant geometric properties using optimized spatial indexing
        """
        significant_triangles = []
        n_points = len(points)
        
        # Optimize: If too many points, limit search to significant points + neighbors
        # Limit processing for performance
        MAX_TRIANGLES = 2000
        
        checked_signatures = set()
        
        # Strategy: Iterate through points, find neighbors, form triangles locally
        for i in range(n_points):
            # Get neighbors from index (K=15 nearest neighbors + self)
            neighbors = self.geometric_index.find_nearest_neighbors((points[i].x, points[i].y), k=16)
            
            # Extract indices, excluding self
            neighbor_indices = [idx for dist, idx in neighbors if idx != i]
            
            # Also invoke "Far Search" -> check points that might form large Right Triangles?
            # For "Bard Code", we want local clusters for efficiency, but maybe specific "Line of Sight"
            # For now, local clustering + random sampling or specific grid points is better than N^3
            
            # Form pairs from neighbors
            for j in neighbor_indices:
                for k in neighbor_indices:
                    if j >= k: continue
                    
                    # Create signature to avoid duplicates (sorted indices)
                    sig = tuple(sorted([i, j, k]))
                    if sig in checked_signatures:
                        continue
                    checked_signatures.add(sig)
                    
                    # Create triangle
                    triangle = Triangle(points[i], points[j], points[k])
                    
                    # Check properties
                    # Skip degenerate triangles (collinear) early
                    if self._are_collinear(points[i], points[j], points[k], 0.001):
                        continue

                    # Analyze properties
                    properties = self._analyze_triangle_properties(triangle)
                    
                    if properties['significance_score'] >= self.min_significance:
                        triangle_data = {
                            'points': [{'x': p.x, 'y': p.y} for p in [triangle.p1, triangle.p2, triangle.p3]],
                            'sides': triangle.sides,
                            'angles': triangle.angles,
                            'ratios': triangle.ratios,
                            'properties': properties
                        }
                        significant_triangles.append(triangle_data)
                        
                        if len(significant_triangles) >= MAX_TRIANGLES:
                            return significant_triangles
        
        return significant_triangles
    
    def _analyze_triangle_properties(self, triangle: Triangle) -> Dict[str, Any]:
        """Analyze special properties of a triangle"""
        properties = {
            'is_right_triangle': False,
            'is_equilateral': False,
            'is_isosceles': False,
            'is_pythagorean': False,
            'contains_golden_ratio': False,
            'significance_score': 0.0
        }
        
        angles = triangle.angles
        sides = triangle.sides
        ratios = triangle.ratios
        
        # Check for right triangle (90-degree angle)
        if any(abs(angle - 90) < 1 for angle in angles):
            properties['is_right_triangle'] = True
            properties['significance_score'] += 0.3
        
        # Check for equilateral triangle
        if all(abs(angle - 60) < 1 for angle in angles):
            properties['is_equilateral'] = True
            properties['significance_score'] += 0.4
        
        # Check for isosceles triangle
        if len(set(round(side, 2) for side in sides)) == 2:
            properties['is_isosceles'] = True
            properties['significance_score'] += 0.2
        
        # Check for Pythagorean ratios (3:4:5)
        sorted_ratios = sorted(ratios)
        if (abs(sorted_ratios[0] - 3) < 0.1 and 
            abs(sorted_ratios[1] - 4) < 0.1 and 
            abs(sorted_ratios[2] - 5) < 0.1):
            properties['is_pythagorean'] = True
            properties['significance_score'] += 0.5
        
        # Check for golden ratio
        for ratio in ratios:
            if abs(ratio - self.MATHEMATICAL_CONSTANTS['phi']) < 0.05:
                properties['contains_golden_ratio'] = True
                properties['significance_score'] += 0.4
        
        return properties
    
    def _detect_constants_in_triangles(self, triangles: List[Dict[str, Any]]) -> List[MathematicalConstant]:
        """Detect mathematical constants in triangle measurements"""
        constants = []
        
        for triangle in triangles:
            # Check angles for constants
            angle_constants = self.detect_mathematical_constants(triangle['angles'])
            constants.extend(angle_constants)
            
            # Check side ratios for constants
            ratio_constants = self.detect_mathematical_constants(triangle['ratios'])
            constants.extend(ratio_constants)
            
            # Check side lengths for constants
            side_constants = self.detect_mathematical_constants(triangle['sides'])
            constants.extend(side_constants)
        
        return constants
    
    def _find_geometric_patterns(self, points: List[Point]) -> List[Dict[str, Any]]:
        """Find other geometric patterns beyond triangles"""
        patterns = []
        
        # Find collinear points (straight lines)
        collinear_groups = self._find_collinear_points(points)
        for group in collinear_groups:
            if len(group) >= 3:  # At least 3 points in a line
                patterns.append({
                    'type': 'collinear_points',
                    'points': [{'x': p.x, 'y': p.y} for p in group],
                    'count': len(group)
                })
        
        # Find circular patterns
        circular_patterns = self._find_circular_patterns(points)
        patterns.extend(circular_patterns)
        
        return patterns
    
    def _find_collinear_points(self, points: List[Point], tolerance: float = 0.01) -> List[List[Point]]:
        """Find groups of collinear points"""
        collinear_groups = []
        
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points[i+1:], i+1):
                group = [p1, p2]
                
                # Check all other points for collinearity with p1-p2
                for k, p3 in enumerate(points):
                    if k != i and k != j:
                        if self._are_collinear(p1, p2, p3, tolerance):
                            group.append(p3)
                
                if len(group) >= 3:
                    collinear_groups.append(group)
        
        return collinear_groups
    
    def _are_collinear(self, p1: Point, p2: Point, p3: Point, tolerance: float) -> bool:
        """Check if three points are collinear"""
        # Calculate cross product
        cross_product = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
        return abs(cross_product) < tolerance
    
    def _find_circular_patterns(self, points: List[Point]) -> List[Dict[str, Any]]:
        """Find points that form circular patterns"""
        # Placeholder for circular pattern detection
        # Would implement circle fitting algorithms
        return []
    
    def _calculate_significance_scores(self, triangles: List[Dict], constants: List[MathematicalConstant], 
                                     patterns: List[Dict]) -> Dict[str, float]:
        """Calculate overall significance scores"""
        scores = {}
        
        # Triangle significance
        if triangles:
            triangle_scores = [t['properties']['significance_score'] for t in triangles]
            scores['triangles'] = np.mean(triangle_scores)
        else:
            scores['triangles'] = 0.0
        
        # Constants significance
        if constants:
            constant_scores = [c.confidence for c in constants]
            scores['constants'] = np.mean(constant_scores)
        else:
            scores['constants'] = 0.0
        
        # Pattern significance
        scores['patterns'] = len(patterns) / 10.0  # Normalize by expected max patterns
        
        # Overall significance
        scores['overall'] = (scores['triangles'] * 0.4 + 
                           scores['constants'] * 0.4 + 
                           scores['patterns'] * 0.2)
        
        return scores
    
    def _find_pythagorean_triangles(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find triangles with Pythagorean ratios"""
        constructions = []
        triangles = self.construct_triangles(points)
        
        for triangle in triangles:
            sides = sorted(triangle.sides)
            # Check for 3:4:5 ratio by comparing normalized sides
            if len(sides) == 3 and sides[0] > 0:
                # Normalize to smallest side
                normalized = [s / sides[0] for s in sides]
                
                # Check for 3:4:5 ratio (1:1.33:1.67 when normalized)
                if (abs(normalized[0] - 1.0) < 0.1 and 
                    abs(normalized[1] - 4.0/3.0) < 0.1 and 
                    abs(normalized[2] - 5.0/3.0) < 0.1):
                    
                    construction = GeometricConstruction(
                        construction_type='pythagorean_triangle',
                        points=[triangle.p1, triangle.p2, triangle.p3],
                        measurements={'ratios': normalized, 'sides': triangle.sides},
                        constants_detected=[],
                        significance_score=0.9,
                        description='3:4:5 Pythagorean triangle'
                    )
                    constructions.append(construction)
                
                # Also check for other Pythagorean triples (5:12:13, 8:15:17, etc.)
                pythagorean_triples = [
                    (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29), (12, 35, 37)
                ]
                
                for triple in pythagorean_triples:
                    expected_normalized = [triple[0]/triple[0], triple[1]/triple[0], triple[2]/triple[0]]
                    if (abs(normalized[0] - expected_normalized[0]) < 0.1 and 
                        abs(normalized[1] - expected_normalized[1]) < 0.1 and 
                        abs(normalized[2] - expected_normalized[2]) < 0.1):
                        
                        construction = GeometricConstruction(
                            construction_type='pythagorean_triangle',
                            points=[triangle.p1, triangle.p2, triangle.p3],
                            measurements={'ratios': normalized, 'sides': triangle.sides},
                            constants_detected=[],
                            significance_score=0.8,
                            description=f'{triple[0]}:{triple[1]}:{triple[2]} Pythagorean triangle'
                        )
                        constructions.append(construction)
                        break
        
        return constructions
    
    def _find_golden_ratio_constructions(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find golden ratio constructions (rectangles, spirals, etc.)"""
        constructions = []
        
        # Find golden ratio rectangles
        rectangles = self._find_rectangles(points)
        for rect in rectangles:
            width = euclidean(rect[0], rect[1])
            height = euclidean(rect[1], rect[2])
            
            if width > 0 and height > 0:
                ratio = max(width, height) / min(width, height)
                
                # Check if ratio is close to golden ratio
                if abs(ratio - self.MATHEMATICAL_CONSTANTS['phi']) < 0.05:
                    phi_constant = MathematicalConstant(
                        name='phi',
                        symbol='φ',
                        expected_value=self.MATHEMATICAL_CONSTANTS['phi'],
                        detected_value=ratio,
                        accuracy=self._calculate_accuracy(ratio, self.MATHEMATICAL_CONSTANTS['phi']),
                        confidence=self._calculate_confidence(self._calculate_accuracy(ratio, self.MATHEMATICAL_CONSTANTS['phi'])),
                        context='golden_rectangle',
                        evidence={'width': width, 'height': height, 'ratio': ratio}
                    )
                    
                    construction = GeometricConstruction(
                        construction_type='golden_rectangle',
                        points=rect,
                        measurements={'width': width, 'height': height, 'ratio': ratio},
                        constants_detected=[phi_constant],
                        significance_score=0.9,
                        description=f'Golden ratio rectangle (φ = {ratio:.3f})'
                    )
                    constructions.append(construction)
        
        # Find golden ratio triangles
        triangles = self.construct_triangles(points)
        for triangle in triangles:
            sides = sorted(triangle.sides)
            if len(sides) == 3 and sides[0] > 0:
                # Check for golden ratio in side relationships
                ratios = [sides[1]/sides[0], sides[2]/sides[1], sides[2]/sides[0]]
                
                for ratio in ratios:
                    if abs(ratio - self.MATHEMATICAL_CONSTANTS['phi']) < 0.05:
                        phi_constant = MathematicalConstant(
                            name='phi',
                            symbol='φ',
                            expected_value=self.MATHEMATICAL_CONSTANTS['phi'],
                            detected_value=ratio,
                            accuracy=self._calculate_accuracy(ratio, self.MATHEMATICAL_CONSTANTS['phi']),
                            confidence=self._calculate_confidence(self._calculate_accuracy(ratio, self.MATHEMATICAL_CONSTANTS['phi'])),
                            context='golden_triangle',
                            evidence={'sides': sides, 'ratio': ratio}
                        )
                        
                        construction = GeometricConstruction(
                            construction_type='golden_triangle',
                            points=[triangle.p1, triangle.p2, triangle.p3],
                            measurements={'sides': sides, 'golden_ratio': ratio},
                            constants_detected=[phi_constant],
                            significance_score=0.85,
                            description=f'Golden ratio triangle (φ = {ratio:.3f})'
                        )
                        constructions.append(construction)
                        break
        
        return constructions
    
    def _find_rectangles(self, points: List[Point]) -> List[List[Point]]:
        """Find rectangular patterns in points"""
        rectangles = []
        
        # Generate all combinations of 4 points
        for point_combo in itertools.combinations(points, 4):
            if self._is_rectangle(point_combo):
                rectangles.append(list(point_combo))
        
        return rectangles
    
    def _is_rectangle(self, four_points: Tuple[Point, Point, Point, Point]) -> bool:
        """Check if four points form a rectangle"""
        points = list(four_points)
        
        # Calculate all distances
        distances = []
        for i in range(4):
            for j in range(i+1, 4):
                distances.append(euclidean(points[i], points[j]))
        
        distances.sort()
        
        # A rectangle should have 4 equal sides (2 pairs) and 2 equal diagonals
        # So we should have: 2 short sides, 2 long sides, 2 equal diagonals
        if (abs(distances[0] - distances[1]) < 0.1 and  # Two short sides
            abs(distances[2] - distances[3]) < 0.1 and  # Two long sides  
            abs(distances[4] - distances[5]) < 0.1):    # Two diagonals
            return True
        
        return False
    
    def _find_equilateral_triangles(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find equilateral triangles"""
        constructions = []
        triangles = self.construct_triangles(points)
        
        for triangle in triangles:
            # Guard against occasional numeric domain errors from acos
            try:
                angles = triangle.angles
            except Exception:
                continue
            # Check if all angles are approximately 60 degrees
            if all(abs(angle - 60) < 2 for angle in angles):
                construction = GeometricConstruction(
                    construction_type='equilateral_triangle',
                    points=[triangle.p1, triangle.p2, triangle.p3],
                    measurements={'angles': angles, 'sides': triangle.sides},
                    constants_detected=[],
                    significance_score=0.8,
                    description='Equilateral triangle (60° angles)'
                )
                constructions.append(construction)
        
        return constructions
    
    def _is_valid_triangle(self, triangle: Triangle) -> bool:
        """Check if triangle is valid (not degenerate)"""
        sides = triangle.sides
        # Check triangle inequality
        return (sides[0] + sides[1] > sides[2] and 
                sides[1] + sides[2] > sides[0] and 
                sides[2] + sides[0] > sides[1])
    
    def _calculate_accuracy(self, detected: float, expected: float) -> float:
        """Calculate accuracy of detected value vs expected"""
        if expected == 0:
            return 1.0 if detected == 0 else 0.0
        return 1 - abs(detected - expected) / expected
    
    def _calculate_confidence(self, accuracy: float) -> float:
        """Calculate confidence score from accuracy"""
        return accuracy ** 2  # Square to emphasize high accuracy
    
    def _get_constant_symbol(self, name: str) -> str:
        """Get mathematical symbol for constant"""
        symbols = {
            'pi': 'π',
            'e': 'e',
            'phi': 'φ',
            'phi_inverse': '1/φ',
            'sqrt_2': '√2',
            'sqrt_3': '√3',
            'sqrt_5': '√5',
            'bruns_constant': 'B₂',
            'euler_mascheroni': 'γ',
            'catalan': 'G',
            'apery': 'ζ(3)'
        }
        return symbols.get(name, name)
    
    def _find_vesica_piscis_patterns(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find vesica piscis patterns (overlapping circles creating almond shapes)"""
        constructions = []
        
        # Look for pairs of points that could be circle centers
        for i, center1 in enumerate(points):
            for j, center2 in enumerate(points[i+1:], i+1):
                distance = euclidean(center1, center2)
                
                # Find points that could form vesica piscis with these centers
                vesica_points = []
                for point in points:
                    dist1 = euclidean(center1, point)
                    dist2 = euclidean(center2, point)
                    
                    # Point should be equidistant from both centers (on both circles)
                    if abs(dist1 - distance) < 0.1 and abs(dist2 - distance) < 0.1:
                        vesica_points.append(point)
                
                # Vesica piscis should have exactly 2 intersection points
                if len(vesica_points) == 2:
                    construction = GeometricConstruction(
                        construction_type='vesica_piscis',
                        points=[center1, center2] + vesica_points,
                        measurements={'radius': distance, 'center_distance': distance},
                        constants_detected=[],
                        significance_score=0.75,
                        description='Vesica piscis (overlapping circles)'
                    )
                    constructions.append(construction)
        
        return constructions
    
    def _find_pentagonal_patterns(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find pentagonal and pentagram patterns (related to golden ratio)"""
        constructions = []
        
        # Look for 5-point patterns that could form pentagons
        for point_combo in itertools.combinations(points, 5):
            if self._is_regular_pentagon(point_combo):
                # Pentagon angles should be 108° (internal) or 36° (pentagram)
                pentagon_angles = self._calculate_pentagon_angles(point_combo)
                
                # Check for golden ratio relationships in pentagon
                phi_constants = []
                for angle in pentagon_angles:
                    if abs(angle - 36) < 2 or abs(angle - 72) < 2 or abs(angle - 108) < 2:
                        # These angles are related to golden ratio in pentagons
                        phi_constant = MathematicalConstant(
                            name='phi',
                            symbol='φ',
                            expected_value=self.MATHEMATICAL_CONSTANTS['phi'],
                            detected_value=self.MATHEMATICAL_CONSTANTS['phi'],
                            accuracy=0.95,
                            confidence=0.9,
                            context='pentagonal_geometry',
                            evidence={'angle': angle, 'pentagon_type': 'regular'}
                        )
                        phi_constants.append(phi_constant)
                        break
                
                construction = GeometricConstruction(
                    construction_type='regular_pentagon',
                    points=list(point_combo),
                    measurements={'angles': pentagon_angles},
                    constants_detected=phi_constants,
                    significance_score=0.85,
                    description='Regular pentagon (golden ratio geometry)'
                )
                constructions.append(construction)
        
        return constructions
    
    def _find_sacred_triangles(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find sacred triangles (30-60-90, 45-45-90)"""
        constructions = []
        triangles = self.construct_triangles(points)
        
        for triangle in triangles:
            try:
                angles = sorted(triangle.angles)
            except Exception:
                continue
            
            # Check for 30-60-90 triangle
            if (abs(angles[0] - 30) < 2 and 
                abs(angles[1] - 60) < 2 and 
                abs(angles[2] - 90) < 2):
                
                construction = GeometricConstruction(
                    construction_type='sacred_triangle_30_60_90',
                    points=[triangle.p1, triangle.p2, triangle.p3],
                    measurements={'angles': angles, 'sides': triangle.sides},
                    constants_detected=[],
                    significance_score=0.8,
                    description='Sacred 30-60-90 triangle'
                )
                constructions.append(construction)
            
            # Check for 45-45-90 triangle (isosceles right triangle)
            elif (abs(angles[0] - 45) < 2 and 
                  abs(angles[1] - 45) < 2 and 
                  abs(angles[2] - 90) < 2):
                
                construction = GeometricConstruction(
                    construction_type='sacred_triangle_45_45_90',
                    points=[triangle.p1, triangle.p2, triangle.p3],
                    measurements={'angles': angles, 'sides': triangle.sides},
                    constants_detected=[],
                    significance_score=0.8,
                    description='Sacred 45-45-90 triangle (isosceles right)'
                )
                constructions.append(construction)
        
        return constructions
    
    def _find_sacred_circles(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find circular patterns and mandala-like structures"""
        constructions = []
        
        # Look for points arranged in circular patterns
        for center_point in points:
            # Find points at similar distances from center (forming circles)
            distances = []
            circle_points = []
            
            for point in points:
                if point != center_point:
                    dist = euclidean(center_point, point)
                    distances.append((dist, point))
            
            # Group points by similar distances
            distance_groups = defaultdict(list)
            for dist, point in distances:
                # Round distance to group similar ones
                rounded_dist = round(dist, 1)
                distance_groups[rounded_dist].append(point)
            
            # Look for groups with multiple points (circular arrangements)
            for dist, group_points in distance_groups.items():
                if len(group_points) >= 3:  # At least 3 points on circle
                    # Check if points are evenly spaced around circle
                    if self._are_points_evenly_spaced_on_circle(center_point, group_points):
                        construction = GeometricConstruction(
                            construction_type='sacred_circle',
                            points=[center_point] + group_points,
                            measurements={'radius': dist, 'point_count': len(group_points)},
                            constants_detected=[],
                            significance_score=0.7,
                            description=f'Sacred circle with {len(group_points)} evenly spaced points'
                        )
                        constructions.append(construction)
        
        return constructions
    
    def _find_cross_patterns(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find cross patterns (tau cross, Greek cross, etc.)"""
        constructions = []
        
        # Look for 4-point cross patterns
        for point_combo in itertools.combinations(points, 4):
            if self._is_cross_pattern(point_combo):
                construction = GeometricConstruction(
                    construction_type='cross_pattern',
                    points=list(point_combo),
                    measurements={'cross_type': 'greek_cross'},
                    constants_detected=[],
                    significance_score=0.7,
                    description='Cross pattern (Greek cross)'
                )
                constructions.append(construction)
        
        # Look for 5-point tau cross patterns (T-shaped)
        for point_combo in itertools.combinations(points, 5):
            if self._is_tau_cross_pattern(point_combo):
                construction = GeometricConstruction(
                    construction_type='tau_cross',
                    points=list(point_combo),
                    measurements={'cross_type': 'tau_cross'},
                    constants_detected=[],
                    significance_score=0.75,
                    description='Tau cross pattern (T-shaped)'
                )
                constructions.append(construction)
        
        return constructions
    
    def _is_regular_pentagon(self, five_points: Tuple[Point, ...]) -> bool:
        """Check if five points form a regular pentagon"""
        points = list(five_points)
        
        # Calculate distances between consecutive points
        distances = []
        for i in range(5):
            next_i = (i + 1) % 5
            dist = euclidean(points[i], points[next_i])
            distances.append(dist)
        
        # All sides should be approximately equal
        avg_distance = np.mean(distances)
        return all(abs(dist - avg_distance) < 0.1 * avg_distance for dist in distances)
    
    def _calculate_pentagon_angles(self, five_points: Tuple[Point, ...]) -> List[float]:
        """Calculate internal angles of a pentagon"""
        points = list(five_points)
        angles = []
        
        for i in range(5):
            prev_i = (i - 1) % 5
            next_i = (i + 1) % 5
            
            # Calculate angle at point i
            v1 = (points[prev_i].x - points[i].x, points[prev_i].y - points[i].y)
            v2 = (points[next_i].x - points[i].x, points[next_i].y - points[i].y)
            
            # Calculate angle between vectors
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                angle = math.degrees(math.acos(cos_angle))
                angles.append(angle)
        
        return angles
    
    def _are_points_evenly_spaced_on_circle(self, center: Point, circle_points: List[Point]) -> bool:
        """Check if points are evenly spaced around a circle"""
        if len(circle_points) < 3:
            return False
        
        # Calculate angles from center to each point
        angles = []
        for point in circle_points:
            angle = math.atan2(point.y - center.y, point.x - center.x)
            angles.append(math.degrees(angle))
        
        # Sort angles
        angles.sort()
        
        # Calculate expected angle between points
        expected_angle = 360.0 / len(circle_points)
        
        # Check if actual angles are close to expected
        for i in range(len(angles)):
            next_i = (i + 1) % len(angles)
            actual_angle = (angles[next_i] - angles[i]) % 360
            if abs(actual_angle - expected_angle) > 10:  # 10 degree tolerance
                return False
        
        return True
    
    def _is_cross_pattern(self, four_points: Tuple[Point, ...]) -> bool:
        """Check if four points form a cross pattern"""
        points = list(four_points)
        
        # Find the center point (should be equidistant from other three)
        for i, potential_center in enumerate(points):
            other_points = [p for j, p in enumerate(points) if j != i]
            distances = [euclidean(potential_center, p) for p in other_points]
            
            # Check if all distances are similar (cross arms)
            avg_distance = np.mean(distances)
            if all(abs(dist - avg_distance) < 0.1 * avg_distance for dist in distances):
                # Check if the three points form right angles at center
                if self._form_right_angles_at_center(potential_center, other_points):
                    return True
        
        return False
    
    def _is_tau_cross_pattern(self, five_points: Tuple[Point, ...]) -> bool:
        """Check if five points form a tau (T-shaped) cross pattern"""
        points = list(five_points)
        
        # Look for a horizontal line of 3 points and a vertical line of 3 points
        # sharing a common center point
        for center_point in points:
            other_points = [p for p in points if p != center_point]
            
            # Try to find horizontal and vertical alignments
            horizontal_points = [center_point]
            vertical_points = [center_point]
            
            for point in other_points:
                # Check if point is horizontally aligned with center
                if abs(point.y - center_point.y) < 0.1:
                    horizontal_points.append(point)
                # Check if point is vertically aligned with center
                elif abs(point.x - center_point.x) < 0.1:
                    vertical_points.append(point)
            
            # Tau cross should have 3 horizontal points and 2 vertical points
            if len(horizontal_points) == 3 and len(vertical_points) == 2:
                return True
        
        return False
    
    def _form_right_angles_at_center(self, center: Point, three_points: List[Point]) -> bool:
        """Check if three points form right angles at a center point"""
        if len(three_points) != 3:
            return False
        
        # Calculate angles between each pair of points from center
        angles = []
        for i in range(3):
            for j in range(i+1, 3):
                v1 = (three_points[i].x - center.x, three_points[i].y - center.y)
                v2 = (three_points[j].x - center.x, three_points[j].y - center.y)
                
                # Calculate angle between vectors
                dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                
                if mag1 > 0 and mag2 > 0:
                    cos_angle = dot_product / (mag1 * mag2)
                    cos_angle = max(-1, min(1, cos_angle))
                    angle = math.degrees(math.acos(cos_angle))
                    angles.append(angle)
        
        # For a cross, we should have angles close to 90 degrees
        return any(abs(angle - 90) < 5 for angle in angles)
        
        return coordinates
    
    def validate_coordinate_significance(self, coordinate_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the significance of extracted coordinate pairs
        Based on statistical analysis and historical site matching
        """
        validation_results = {
            'total_pairs': len(coordinate_pairs),
            'significant_pairs': [],
            'historical_matches': 0,
            'confidence_distribution': {},
            'validation_score': 0.0
        }
        
        if not coordinate_pairs:
            return validation_results
        
        # Analyze confidence distribution
        confidences = [pair.get('combined_confidence', 0) for pair in coordinate_pairs]
        validation_results['confidence_distribution'] = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        }
        
        # Identify significant pairs (high confidence)
        high_confidence_threshold = 0.7
        for pair in coordinate_pairs:
            if pair.get('combined_confidence', 0) >= high_confidence_threshold:
                validation_results['significant_pairs'].append(pair)
        
        # Check for historical site matches
        historical_matches = self._match_historical_sites(coordinate_pairs, 0.1)
        validation_results['historical_matches'] = len(historical_matches)
        
        # Calculate overall validation score
        confidence_score = validation_results['confidence_distribution']['mean']
        historical_score = min(1.0, validation_results['historical_matches'] / 3.0)  # Normalize by expected matches
        significance_score = len(validation_results['significant_pairs']) / max(1, len(coordinate_pairs))
        
        validation_results['validation_score'] = (
            confidence_score * 0.4 + 
            historical_score * 0.4 + 
            significance_score * 0.2
        )
        
        return validation_results
    
    def generate_coordinate_report(self, coordinate_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive report of coordinate extraction analysis
        """
        report = {
            'summary': {
                'extraction_method': coordinate_analysis.get('extraction_method', 'unknown'),
                'total_potential_coordinates': len(coordinate_analysis.get('potential_coordinates', [])),
                'coordinate_pairs_found': len(coordinate_analysis.get('coordinate_pairs', [])),
                'historical_site_matches': len(coordinate_analysis.get('historical_sites', []))
            },
            'coordinate_pairs': [],
            'historical_matches': [],
            'extraction_methods': {},
            'confidence_analysis': {},
            'recommendations': []
        }
        
        # Process coordinate pairs
        for pair in coordinate_analysis.get('coordinate_pairs', []):
            pair_info = {
                'latitude': pair['latitude'],
                'longitude': pair['longitude'],
                'confidence': pair['combined_confidence'],
                'extraction_methods': pair['methods'],
                'coordinate_string': f"{pair['latitude']:.4f}°, {pair['longitude']:.4f}°"
            }
            report['coordinate_pairs'].append(pair_info)
        
        # Process historical matches
        for match in coordinate_analysis.get('historical_sites', []):
            match_info = {
                'site_name': match['site']['name'],
                'site_significance': match['site']['significance'],
                'detected_coordinates': f"{match['detected_coordinates']['latitude']:.4f}°, {match['detected_coordinates']['longitude']:.4f}°",
                'actual_coordinates': f"{match['site']['lat']:.4f}°, {match['site']['lon']:.4f}°",
                'accuracy': match['accuracy'],
                'match_confidence': match['match_confidence']
            }
            report['historical_matches'].append(match_info)
        
        # Analyze extraction methods
        method_counts = {}
        for coord in coordinate_analysis.get('potential_coordinates', []):
            method = coord.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        report['extraction_methods'] = method_counts
        
        # Confidence analysis
        if coordinate_analysis.get('coordinate_pairs'):
            confidences = [pair['combined_confidence'] for pair in coordinate_analysis['coordinate_pairs']]
            report['confidence_analysis'] = {
                'average_confidence': np.mean(confidences),
                'confidence_range': [np.min(confidences), np.max(confidences)],
                'high_confidence_pairs': len([c for c in confidences if c >= 0.8])
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_coordinate_recommendations(coordinate_analysis)
        
        return report
    
    def _generate_coordinate_recommendations(self, coordinate_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on coordinate analysis results"""
        recommendations = []
        
        coordinate_pairs = coordinate_analysis.get('coordinate_pairs', [])
        historical_matches = coordinate_analysis.get('historical_sites', [])
        
        if not coordinate_pairs:
            recommendations.append("No coordinate pairs were extracted. Consider adjusting tolerance parameters or using different geometric measurements.")
        
        if len(coordinate_pairs) > 10:
            recommendations.append("Large number of coordinate pairs found. Consider filtering by confidence score to focus on most significant results.")
        
        if historical_matches:
            recommendations.append(f"Found {len(historical_matches)} matches with historical sites. These coordinates may have special significance.")
        
        high_confidence_pairs = [p for p in coordinate_pairs if p.get('combined_confidence', 0) >= 0.8]
        if high_confidence_pairs:
            recommendations.append(f"Focus on {len(high_confidence_pairs)} high-confidence coordinate pairs for further analysis.")
        
        if len(set(coord.get('method', '') for coord in coordinate_analysis.get('potential_coordinates', []))) > 3:
            recommendations.append("Multiple extraction methods yielded results. Cross-validate findings using different approaches.")
        
        return recommendations 
   
    def _find_golden_ratio_constructions(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find golden ratio constructions (rectangles, spirals, etc.)"""
        constructions = []
        
        # Find golden ratio rectangles
        rectangles = self._find_rectangles(points)
        for rect in rectangles:
            width = euclidean(rect[0], rect[1])
            height = euclidean(rect[1], rect[2])
            
            if width > 0 and height > 0:
                ratio = max(width, height) / min(width, height)
                
                # Check if ratio is close to golden ratio
                if abs(ratio - self.MATHEMATICAL_CONSTANTS['phi']) < 0.05:
                    phi_constant = MathematicalConstant(
                        name='phi',
                        symbol='φ',
                        expected_value=self.MATHEMATICAL_CONSTANTS['phi'],
                        detected_value=ratio,
                        accuracy=self._calculate_accuracy(ratio, self.MATHEMATICAL_CONSTANTS['phi']),
                        confidence=self._calculate_confidence(self._calculate_accuracy(ratio, self.MATHEMATICAL_CONSTANTS['phi'])),
                        context='golden_rectangle',
                        evidence={'width': width, 'height': height, 'ratio': ratio}
                    )
                    
                    construction = GeometricConstruction(
                        construction_type='golden_rectangle',
                        points=rect,
                        measurements={'width': width, 'height': height, 'ratio': ratio},
                        constants_detected=[phi_constant],
                        significance_score=0.9,
                        description=f'Golden ratio rectangle (φ = {ratio:.3f})'
                    )
                    constructions.append(construction)
        
        return constructions
    
    def _find_rectangles(self, points: List[Point]) -> List[List[Point]]:
        """Find rectangular patterns in points"""
        rectangles = []
        
        # Generate all combinations of 4 points
        for point_combo in itertools.combinations(points, 4):
            if self._is_rectangle(point_combo):
                rectangles.append(list(point_combo))
        
        return rectangles
    
    def _is_rectangle(self, four_points: Tuple[Point, Point, Point, Point]) -> bool:
        """Check if four points form a rectangle"""
        points = list(four_points)
        
        # Calculate all distances
        distances = []
        for i in range(4):
            for j in range(i+1, 4):
                distances.append(euclidean(points[i], points[j]))
        
        distances.sort()
        
        # A rectangle should have 4 equal sides (2 pairs) and 2 equal diagonals
        # So we should have: 2 short sides, 2 long sides, 2 equal diagonals
        if (abs(distances[0] - distances[1]) < 0.1 and  # Two short sides
            abs(distances[2] - distances[3]) < 0.1 and  # Two long sides  
            abs(distances[4] - distances[5]) < 0.1):    # Two diagonals
            return True
        
        return False
    
    def _find_equilateral_triangles(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find equilateral triangles (60° angles)"""
        constructions = []
        triangles = self.construct_triangles(points)
        
        for triangle in triangles:
            try:
                angles = triangle.angles
            except Exception:
                continue
            
            # Check if all angles are close to 60°
            if all(abs(angle - 60) < 2 for angle in angles):
                construction = GeometricConstruction(
                    construction_type='equilateral_triangle',
                    points=[triangle.p1, triangle.p2, triangle.p3],
                    measurements={'angles': angles, 'sides': triangle.sides},
                    constants_detected=[],
                    significance_score=0.8,
                    description='Equilateral triangle (60° angles)'
                )
                constructions.append(construction)
        
        return constructions
    
    def _find_vesica_piscis_patterns(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find vesica piscis patterns (overlapping circles creating almond shapes)"""
        constructions = []
        
        # Look for pairs of points that could be circle centers
        for i, center1 in enumerate(points):
            for j, center2 in enumerate(points[i+1:], i+1):
                distance = euclidean(center1, center2)
                
                # Find points that could form vesica piscis with these centers
                vesica_points = []
                for point in points:
                    dist1 = euclidean(center1, point)
                    dist2 = euclidean(center2, point)
                    
                    # Point should be equidistant from both centers (on both circles)
                    if abs(dist1 - distance) < 0.1 and abs(dist2 - distance) < 0.1:
                        vesica_points.append(point)
                
                # Vesica piscis should have exactly 2 intersection points
                if len(vesica_points) == 2:
                    construction = GeometricConstruction(
                        construction_type='vesica_piscis',
                        points=[center1, center2] + vesica_points,
                        measurements={'radius': distance, 'center_distance': distance},
                        constants_detected=[],
                        significance_score=0.75,
                        description='Vesica piscis (overlapping circles)'
                    )
                    constructions.append(construction)
        
        return constructions
    
    def _find_pentagonal_patterns(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find pentagonal and pentagram patterns (related to golden ratio)"""
        constructions = []
        
        # Look for 5-point patterns that could form pentagons
        for point_combo in itertools.combinations(points, 5):
            if self._is_regular_pentagon(point_combo):
                # Pentagon angles should be 108° (internal) or 36° (pentagram)
                pentagon_angles = self._calculate_pentagon_angles(point_combo)
                
                # Check for golden ratio relationships in pentagon
                phi_constants = []
                for angle in pentagon_angles:
                    if abs(angle - 36) < 2 or abs(angle - 72) < 2 or abs(angle - 108) < 2:
                        # These angles are related to golden ratio in pentagons
                        phi_constant = MathematicalConstant(
                            name='phi',
                            symbol='φ',
                            expected_value=self.MATHEMATICAL_CONSTANTS['phi'],
                            detected_value=self.MATHEMATICAL_CONSTANTS['phi'],
                            accuracy=0.95,
                            confidence=0.9,
                            context='pentagonal_geometry',
                            evidence={'angle': angle, 'pentagon_type': 'regular'}
                        )
                        phi_constants.append(phi_constant)
                        break
                
                construction = GeometricConstruction(
                    construction_type='regular_pentagon',
                    points=list(point_combo),
                    measurements={'angles': pentagon_angles},
                    constants_detected=phi_constants,
                    significance_score=0.85,
                    description='Regular pentagon (golden ratio geometry)'
                )
                constructions.append(construction)
        
        return constructions
    
    def _find_sacred_triangles(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find sacred triangles (30-60-90, 45-45-90)"""
        constructions = []
        triangles = self.construct_triangles(points)
        
        for triangle in triangles:
            angles = sorted(triangle.angles)
            
            # Check for 30-60-90 triangle
            if (abs(angles[0] - 30) < 2 and 
                abs(angles[1] - 60) < 2 and 
                abs(angles[2] - 90) < 2):
                
                construction = GeometricConstruction(
                    construction_type='sacred_triangle_30_60_90',
                    points=[triangle.p1, triangle.p2, triangle.p3],
                    measurements={'angles': angles, 'sides': triangle.sides},
                    constants_detected=[],
                    significance_score=0.85,
                    description='Sacred triangle (30-60-90)'
                )
                constructions.append(construction)
            
            # Check for 45-45-90 triangle
            elif (abs(angles[0] - 45) < 2 and 
                  abs(angles[1] - 45) < 2 and 
                  abs(angles[2] - 90) < 2):
                
                construction = GeometricConstruction(
                    construction_type='sacred_triangle_45_45_90',
                    points=[triangle.p1, triangle.p2, triangle.p3],
                    measurements={'angles': angles, 'sides': triangle.sides},
                    constants_detected=[],
                    significance_score=0.85,
                    description='Sacred triangle (45-45-90)'
                )
                constructions.append(construction)
        
        return constructions
    
    def _find_sacred_circles(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find circular patterns and mandala-like structures"""
        constructions = []
        
        # Look for points arranged in circular patterns
        for center_point in points:
            # Find points at similar distances from center
            distances = []
            for point in points:
                if point != center_point:
                    dist = euclidean(center_point, point)
                    distances.append((point, dist))
            
            # Group points by similar distances
            distance_groups = {}
            for point, dist in distances:
                rounded_dist = round(dist, 1)
                if rounded_dist not in distance_groups:
                    distance_groups[rounded_dist] = []
                distance_groups[rounded_dist].append(point)
            
            # Look for groups with multiple points (potential circles)
            for dist, group_points in distance_groups.items():
                if len(group_points) >= 3:  # At least 3 points on circle
                    construction = GeometricConstruction(
                        construction_type='sacred_circle',
                        points=[center_point] + group_points,
                        measurements={'radius': dist, 'point_count': len(group_points)},
                        constants_detected=[],
                        significance_score=0.7,
                        description=f'Sacred circle (radius={dist:.2f}, {len(group_points)} points)'
                    )
                    constructions.append(construction)
        
        return constructions
    
    def _find_cross_patterns(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find cross patterns (tau cross, Greek cross, etc.)"""
        constructions = []
        
        # Look for simple Greek cross (center with 3 orthogonal arms)
        for center_point in points:
            right = any(abs(p.y - center_point.y) < 1e-6 and p.x > center_point.x for p in points if p != center_point)
            left = any(abs(p.y - center_point.y) < 1e-6 and p.x < center_point.x for p in points if p != center_point)
            up = any(abs(p.x - center_point.x) < 1e-6 and p.y > center_point.y for p in points if p != center_point)
            down = any(abs(p.x - center_point.x) < 1e-6 and p.y < center_point.y for p in points if p != center_point)

            # Accept a minimal cross with center and any three orthogonal directions
            orth_count = sum([right, left, up, down])
            if orth_count >= 3:
                arms = []
                for p in points:
                    if p == center_point:
                        continue
                    if abs(p.y - center_point.y) < 1e-6 or abs(p.x - center_point.x) < 1e-6:
                        arms.append(p)
                construction = GeometricConstruction(
                    construction_type='cross_pattern',
                    points=[center_point] + arms,
                    measurements={'cross_type': 'greek_cross'},
                    constants_detected=[],
                    significance_score=0.7,
                    description='Cross pattern (orthogonal arms)'
                )
                constructions.append(construction)
        
        return constructions
    
    def _find_pythagorean_triangles(self, points: List[Point]) -> List[GeometricConstruction]:
        """Find Pythagorean triangles (3:4:5 ratios and other Pythagorean triples)"""
        constructions = []
        triangles = self.construct_triangles(points)
        
        # Known Pythagorean triples
        pythagorean_triples = [
            (3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25),
            (20, 21, 29), (9, 40, 41), (12, 35, 37), (11, 60, 61)
        ]
        
        for triangle in triangles:
            sides = sorted(triangle.sides)
            
            # Check if sides match any Pythagorean triple (within tolerance)
            for triple in pythagorean_triples:
                scaled_triple = sorted(triple)
                
                # Check if triangle sides are proportional to Pythagorean triple
                if sides[0] > 0:
                    scale = sides[0] / scaled_triple[0]
                    expected_sides = [s * scale for s in scaled_triple]
                    
                    if all(abs(sides[i] - expected_sides[i]) < 0.1 for i in range(3)):
                        construction = GeometricConstruction(
                            construction_type='pythagorean_triangle',
                            points=[triangle.p1, triangle.p2, triangle.p3],
                            measurements={
                                'sides': sides,
                                'pythagorean_triple': triple,
                                'scale_factor': scale
                            },
                            constants_detected=[],
                            significance_score=0.9,
                            description=f'Pythagorean triangle ({triple[0]}:{triple[1]}:{triple[2]})'
                        )
                        constructions.append(construction)
                        break
        
        return constructions
    
    def _is_regular_pentagon(self, five_points: Tuple[Point, ...]) -> bool:
        """Check if five points form a regular pentagon"""
        points = list(five_points)
        
        # Calculate center point
        center_x = sum(p.x for p in points) / 5
        center_y = sum(p.y for p in points) / 5
        center = Point(center_x, center_y)
        
        # Check if all points are equidistant from center
        distances = [euclidean(center, point) for point in points]
        avg_distance = sum(distances) / len(distances)
        
        # All distances should be similar
        if all(abs(dist - avg_distance) < 0.1 for dist in distances):
            # Check if angles between adjacent points are close to 72°
            angles = []
            for i in range(5):
                p1 = points[i]
                p2 = points[(i + 1) % 5]
                
                # Calculate angle from center
                angle1 = math.atan2(p1.y - center.y, p1.x - center.x)
                angle2 = math.atan2(p2.y - center.y, p2.x - center.x)
                
                angle_diff = abs(math.degrees(angle2 - angle1))
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                angles.append(angle_diff)
            
            # Pentagon should have 72° angles between adjacent points
            return all(abs(angle - 72) < 5 for angle in angles)
        
        return False
    
    def _calculate_pentagon_angles(self, five_points: Tuple[Point, ...]) -> List[float]:
        """Calculate angles in a pentagon"""
        points = list(five_points)
        angles = []
        
        for i in range(5):
            p1 = points[(i - 1) % 5]
            p2 = points[i]
            p3 = points[(i + 1) % 5]
            
            # Calculate angle at p2
            v1 = Point(p1.x - p2.x, p1.y - p2.y)
            v2 = Point(p3.x - p2.x, p3.y - p2.y)
            
            dot_product = v1.x * v2.x + v1.y * v2.y
            mag1 = math.sqrt(v1.x**2 + v1.y**2)
            mag2 = math.sqrt(v2.x**2 + v2.y**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                angle = math.degrees(math.acos(cos_angle))
                angles.append(angle)
        
        return angles
    
    # Helper methods
    def construct_triangles(self, points: List[Point]) -> List[Triangle]:
        """Construct all possible triangles from points"""
        triangles = []
        
        for point_combo in itertools.combinations(points, 3):
            triangle = Triangle(point_combo[0], point_combo[1], point_combo[2])
            triangles.append(triangle)
        
        return triangles
    
    def _normalize_coordinates(self, points: List[Point], width: float, height: float) -> List[Point]:
        """Normalize coordinates to 0-1 range"""
        if not points or width <= 0 or height <= 0:
            return points
        
        return [Point(p.x / width, p.y / height) for p in points]
    
    def _calculate_bounds(self, points: List[Point]) -> Dict[str, float]:
        """Calculate bounding box of points"""
        if not points:
            return {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}
        
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        
        return {
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords)
        }
    
    def _find_significant_triangles(self, points: List[Point]) -> List[Dict[str, Any]]:
        """Find triangles with significant geometric properties"""
        significant_triangles = []
        triangles = self.construct_triangles(points)
        
        for triangle in triangles:
            significance_score = 0
            properties = {}
            
            # Check for special angle relationships
            angles = triangle.angles
            if any(abs(angle - 60) < 2 for angle in angles):
                significance_score += 0.3
                properties['has_60_degree_angle'] = True
            
            if any(abs(angle - 90) < 2 for angle in angles):
                significance_score += 0.3
                properties['has_90_degree_angle'] = True
            
            # Check for special side ratios
            sides = sorted(triangle.sides)
            if sides[0] > 0:
                ratios = [sides[1]/sides[0], sides[2]/sides[1], sides[2]/sides[0]]
                
                for ratio in ratios:
                    if abs(ratio - self.MATHEMATICAL_CONSTANTS['phi']) < 0.05:
                        significance_score += 0.4
                        properties['has_golden_ratio'] = True
                        break
            
            if significance_score >= self.min_significance:
                triangle_data = {
                    'points': [triangle.p1, triangle.p2, triangle.p3],
                    'angles': angles,
                    'sides': triangle.sides,
                    'ratios': triangle.ratios,
                    'significance_score': significance_score,
                    'properties': properties
                }
                significant_triangles.append(triangle_data)
        
        return significant_triangles
    
    def _detect_constants_in_triangles(self, triangles: List[Dict[str, Any]]) -> List[MathematicalConstant]:
        """Detect mathematical constants in triangle measurements"""
        constants = []
        
        for triangle in triangles:
            # Check angles for constants
            for angle in triangle['angles']:
                angle_constants = self.detect_mathematical_constants([angle])
                constants.extend(angle_constants)
            
            # Check side ratios for constants
            for ratio in triangle['ratios']:
                ratio_constants = self.detect_mathematical_constants([ratio])
                constants.extend(ratio_constants)
        
        return constants
    
    def _find_geometric_patterns(self, points: List[Point]) -> List[Dict[str, Any]]:
        """Find geometric patterns in point arrangements"""
        patterns = []
        
        # Find linear patterns
        linear_patterns = self._find_linear_patterns(points)
        patterns.extend(linear_patterns)
        
        # Find circular patterns
        circular_patterns = self._find_circular_patterns(points)
        patterns.extend(circular_patterns)
        
        # Find grid patterns
        grid_patterns = self._find_grid_patterns(points)
        patterns.extend(grid_patterns)
        
        return patterns
    
    def _find_linear_patterns(self, points: List[Point]) -> List[Dict[str, Any]]:
        """Find points arranged in linear patterns"""
        patterns = []
        
        # Check all combinations of 3+ points for collinearity
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                collinear_points = [points[i], points[j]]
                
                # Find other points on the same line
                for k in range(len(points)):
                    if k != i and k != j:
                        if self._are_collinear(points[i], points[j], points[k]):
                            collinear_points.append(points[k])
                
                if len(collinear_points) >= 3:
                    pattern = {
                        'type': 'linear',
                        'points': collinear_points,
                        'point_count': len(collinear_points),
                        'significance_score': min(1.0, len(collinear_points) / 10)
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _find_circular_patterns(self, points: List[Point]) -> List[Dict[str, Any]]:
        """Find points arranged in circular patterns"""
        patterns = []
        
        # For each potential center point
        for center in points:
            # Group points by distance from center
            distance_groups = {}
            for point in points:
                if point != center:
                    dist = round(euclidean(center, point), 1)
                    if dist not in distance_groups:
                        distance_groups[dist] = []
                    distance_groups[dist].append(point)
            
            # Find groups with multiple points (circular patterns)
            for dist, group_points in distance_groups.items():
                if len(group_points) >= 3:
                    pattern = {
                        'type': 'circular',
                        'center': center,
                        'radius': dist,
                        'points': group_points,
                        'point_count': len(group_points),
                        'significance_score': min(1.0, len(group_points) / 8)
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _find_grid_patterns(self, points: List[Point]) -> List[Dict[str, Any]]:
        """Find points arranged in grid patterns"""
        patterns = []
        
        # Group points by x and y coordinates
        x_groups = {}
        y_groups = {}
        
        for point in points:
            x_rounded = round(point.x, 1)
            y_rounded = round(point.y, 1)
            
            if x_rounded not in x_groups:
                x_groups[x_rounded] = []
            x_groups[x_rounded].append(point)
            
            if y_rounded not in y_groups:
                y_groups[y_rounded] = []
            y_groups[y_rounded].append(point)
        
        # Find grid intersections
        grid_points = []
        for x_coord, x_points in x_groups.items():
            for y_coord, y_points in y_groups.items():
                # Find points at this grid intersection
                intersection_points = []
                for point in points:
                    if abs(point.x - x_coord) < 0.1 and abs(point.y - y_coord) < 0.1:
                        intersection_points.append(point)
                
                if intersection_points:
                    grid_points.extend(intersection_points)
        
        if len(grid_points) >= 4:
            pattern = {
                'type': 'grid',
                'points': grid_points,
                'x_lines': len(x_groups),
                'y_lines': len(y_groups),
                'point_count': len(grid_points),
                'significance_score': min(1.0, len(grid_points) / 16)
            }
            patterns.append(pattern)
        
        return patterns
    
    def _are_collinear(self, p1: Point, p2: Point, p3: Point, tolerance: float = 0.1) -> bool:
        """Check if three points are collinear"""
        # Calculate cross product to check collinearity
        cross_product = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
        return abs(cross_product) < tolerance
    
    def _calculate_significance_scores(self, triangles: List[Dict[str, Any]], 
                                     constants: List[MathematicalConstant],
                                     patterns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall significance scores"""
        scores = {
            'triangle_significance': 0,
            'constant_significance': 0,
            'pattern_significance': 0,
            'overall_significance': 0
        }
        
        if triangles:
            scores['triangle_significance'] = sum(t['significance_score'] for t in triangles) / len(triangles)
        
        if constants:
            scores['constant_significance'] = sum(c.confidence for c in constants) / len(constants)
        
        if patterns:
            scores['pattern_significance'] = sum(p['significance_score'] for p in patterns) / len(patterns)
        
        scores['overall_significance'] = (
            scores['triangle_significance'] * 0.4 +
            scores['constant_significance'] * 0.4 +
            scores['pattern_significance'] * 0.2
        )
        
        return scores
    
    def _calculate_accuracy(self, detected: float, expected: float) -> float:
        """Calculate accuracy between detected and expected values"""
        if expected == 0:
            return 1.0 if detected == 0 else 0.0
        
        error = abs(detected - expected) / abs(expected)
        return max(0, 1 - error)
    
    def _calculate_confidence(self, accuracy: float) -> float:
        """Calculate confidence score from accuracy"""
        return accuracy ** 2  # Square to emphasize high accuracy
    
    def _get_constant_symbol(self, const_name: str) -> str:
        """Get symbol for mathematical constant"""
        symbols = {
            'pi': 'π',
            'e': 'e',
            'phi': 'φ',
            'phi_inverse': '1/φ',
            'sqrt_2': '√2',
            'sqrt_3': '√3',
            'sqrt_5': '√5',
            'bruns_constant': 'B₂',
            'euler_mascheroni': 'γ',
            'catalan': 'G',
            'apery': 'ζ(3)'
        }
        return symbols.get(const_name, const_name)