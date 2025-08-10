"""
Tests for geometric analyzer
"""
import pytest
import math
from app.services.geometric_analyzer import (
    GeometricAnalyzer, Point, AngleMeasurement, DistanceData, 
    GeometricRelationship, TrigRelationship, GeometricPattern
)

class TestPoint:
    
    def test_point_creation(self):
        """Test Point creation"""
        p = Point(10.0, 20.0)
        assert p.x == 10.0
        assert p.y == 20.0
    
    def test_distance_calculation(self):
        """Test distance calculation between points"""
        p1 = Point(0.0, 0.0)
        p2 = Point(3.0, 4.0)
        
        distance = p1.distance_to(p2)
        assert abs(distance - 5.0) < 1e-6  # 3-4-5 triangle
    
    def test_angle_calculation(self):
        """Test angle calculation between points"""
        p1 = Point(0.0, 0.0)
        p2 = Point(1.0, 0.0)  # 0 degrees
        
        angle = p1.angle_to(p2)
        assert abs(angle - 0.0) < 1e-6
        
        p3 = Point(0.0, 1.0)  # 90 degrees
        angle = p1.angle_to(p3)
        assert abs(angle - math.pi/2) < 1e-6
    
    def test_midpoint_calculation(self):
        """Test midpoint calculation"""
        p1 = Point(0.0, 0.0)
        p2 = Point(4.0, 6.0)
        
        midpoint = p1.midpoint_to(p2)
        assert midpoint.x == 2.0
        assert midpoint.y == 3.0

class TestGeometricAnalyzer:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = GeometricAnalyzer()
    
    def test_init(self):
        """Test GeometricAnalyzer initialization"""
        assert abs(self.analyzer.GOLDEN_RATIO - 1.618033988749) < 1e-6
        assert abs(self.analyzer.PI - math.pi) < 1e-6
        assert 90 in self.analyzer.SIGNIFICANT_ANGLES
        assert 45 in self.analyzer.SIGNIFICANT_ANGLES
    
    def test_calculate_angle_right_angle(self):
        """Test right angle calculation"""
        p1 = Point(0.0, 0.0)
        vertex = Point(1.0, 0.0)
        p2 = Point(1.0, 1.0)
        
        angle = self.analyzer._calculate_angle(p1, vertex, p2)
        
        assert angle is not None
        assert abs(angle.angle_degrees - 90.0) < 1e-6
        assert abs(angle.angle_radians - math.pi/2) < 1e-6
        assert angle.vertex == vertex
    
    def test_calculate_angle_45_degrees(self):
        """Test angle calculation with specific points"""
        # These points create a 135-degree angle (which is what the algorithm calculates)
        p1 = Point(0.0, 0.0)  # Left point
        vertex = Point(1.0, 0.0)  # Vertex on horizontal line
        p2 = Point(2.0, 1.0)  # Point creating the angle
        
        angle = self.analyzer._calculate_angle(p1, vertex, p2)
        
        assert angle is not None
        assert abs(angle.angle_degrees - 135.0) < 1e-6
    
    def test_calculate_angle_confidence(self):
        """Test angle confidence calculation"""
        # Well-separated points should have high confidence
        p1 = Point(0.0, 0.0)
        vertex = Point(50.0, 0.0)
        p2 = Point(50.0, 50.0)
        
        confidence = self.analyzer._calculate_angle_confidence(p1, vertex, p2)
        assert confidence > 0.8
        
        # Close points should have lower confidence
        p1_close = Point(0.0, 0.0)
        vertex_close = Point(1.0, 0.0)
        p2_close = Point(1.0, 1.0)
        
        confidence_close = self.analyzer._calculate_angle_confidence(p1_close, vertex_close, p2_close)
        assert confidence_close < confidence
    
    def test_calculate_angle_significance(self):
        """Test angle significance calculation"""
        # 90 degrees should be highly significant
        sig_90 = self.analyzer._calculate_angle_significance(90.0)
        assert sig_90 == 1.0
        
        # 45 degrees should be highly significant
        sig_45 = self.analyzer._calculate_angle_significance(45.0)
        assert sig_45 == 1.0
        
        # Random angle should be less significant
        sig_random = self.analyzer._calculate_angle_significance(73.5)
        assert sig_random < sig_90
    
    def test_measure_angles(self):
        """Test angle measurement for multiple points"""
        # Create a right triangle
        points = [
            Point(0.0, 0.0),
            Point(3.0, 0.0),
            Point(0.0, 4.0)
        ]
        
        angles = self.analyzer.measure_angles(points)
        
        assert len(angles) > 0
        # Should find a 90-degree angle
        right_angles = [a for a in angles if abs(a.angle_degrees - 90.0) < 1.0]
        assert len(right_angles) > 0
    
    def test_measure_angles_insufficient_points(self):
        """Test angle measurement with insufficient points"""
        points = [Point(0.0, 0.0), Point(1.0, 1.0)]
        
        angles = self.analyzer.measure_angles(points)
        assert len(angles) == 0
    
    def test_calculate_distances(self):
        """Test distance calculation"""
        p1 = Point(0.0, 0.0)
        p2 = Point(3.0, 4.0)
        
        distance_data = self.analyzer.calculate_distances(p1, p2)
        
        assert abs(distance_data.distance - 5.0) < 1e-6
        assert distance_data.point1 == p1
        assert distance_data.point2 == p2
        assert distance_data.measurement_unit == "pixels"
        assert distance_data.confidence > 0.9
    
    def test_distance_significance(self):
        """Test distance significance calculation"""
        # Distance close to golden ratio should be significant
        golden_distance = self.analyzer.GOLDEN_RATIO * 10
        sig_golden = self.analyzer._calculate_distance_significance(golden_distance)
        
        # Distance close to integer should be somewhat significant
        integer_distance = 10.05
        sig_integer = self.analyzer._calculate_distance_significance(integer_distance)
        
        # Random distance should be less significant
        random_distance = 7.3456
        sig_random = self.analyzer._calculate_distance_significance(random_distance)
        
        assert sig_golden > sig_random
        assert sig_integer > sig_random
    
    def test_detect_right_angle_patterns(self):
        """Test right angle pattern detection"""
        vertex = Point(1.0, 1.0)
        
        # Create a right angle measurement
        right_angle = AngleMeasurement(
            angle_degrees=90.0,
            angle_radians=math.pi/2,
            vertex=vertex,
            point1=Point(0.0, 1.0),
            point2=Point(1.0, 0.0),
            measurement_type="test",
            confidence=0.9,
            significance_score=0.9
        )
        
        patterns = self.analyzer._detect_right_angle_patterns(vertex, [right_angle])
        
        assert len(patterns) == 1
        assert patterns[0].relationship_type == "right_angle_pattern"
        assert patterns[0].pattern_type == GeometricPattern.CROSS
        assert patterns[0].significance_score == 0.95
    
    def test_detect_triangular_patterns(self):
        """Test triangular pattern detection"""
        vertex = Point(1.0, 1.0)
        
        # Create three angles that sum to 180 degrees
        angle1 = AngleMeasurement(60.0, math.pi/3, vertex, Point(0.0, 1.0), Point(1.0, 0.0), "test", 0.9, 0.9)
        angle2 = AngleMeasurement(60.0, math.pi/3, vertex, Point(1.0, 0.0), Point(2.0, 1.0), "test", 0.9, 0.9)
        angle3 = AngleMeasurement(60.0, math.pi/3, vertex, Point(2.0, 1.0), Point(0.0, 1.0), "test", 0.9, 0.9)
        
        patterns = self.analyzer._detect_triangular_patterns(vertex, [angle1, angle2, angle3])
        
        assert len(patterns) == 1
        assert patterns[0].relationship_type == "triangular_pattern"
        assert patterns[0].pattern_type == GeometricPattern.TRIANGLE
        assert abs(patterns[0].measurements["angle_sum"] - 180.0) < 5.0
    
    def test_detect_cross_patterns(self):
        """Test cross pattern detection"""
        vertex = Point(1.0, 1.0)
        
        # Create two supplementary angles (sum to 180°)
        angle1 = AngleMeasurement(120.0, 2*math.pi/3, vertex, Point(0.0, 1.0), Point(1.0, 0.0), "test", 0.9, 0.9)
        angle2 = AngleMeasurement(60.0, math.pi/3, vertex, Point(1.0, 0.0), Point(2.0, 1.0), "test", 0.9, 0.9)
        
        patterns = self.analyzer._detect_cross_patterns(vertex, [angle1, angle2])
        
        assert len(patterns) == 1
        assert patterns[0].relationship_type == "cross_pattern"
        assert patterns[0].pattern_type == GeometricPattern.CROSS
        assert abs(patterns[0].measurements["angle_sum"] - 180.0) < 5.0
    
    def test_find_pi_relationships(self):
        """Test finding pi-related angle relationships"""
        # Test angles that are fractions of pi
        angles_deg = [30.0, 45.0, 60.0, 90.0, 180.0]  # π/6, π/4, π/3, π/2, π
        angles_rad = [math.radians(a) for a in angles_deg]
        
        relationships = self.analyzer._find_pi_relationships(angles_deg, angles_rad)
        
        assert len(relationships) == 5  # All angles should be recognized as pi fractions
        
        # Check that 90° is recognized as π/2
        pi_half_relations = [r for r in relationships if "π/2" in r.description]
        assert len(pi_half_relations) == 1
        assert pi_half_relations[0].mathematical_constant == "pi"
    
    def test_find_golden_ratio_relationships(self):
        """Test finding golden ratio relationships"""
        # Golden angle ≈ 137.5°
        golden_angle = 360.0 / (self.analyzer.GOLDEN_RATIO + 1)
        angles_deg = [golden_angle]
        angles_rad = [math.radians(golden_angle)]
        
        relationships = self.analyzer._find_golden_ratio_relationships(angles_deg, angles_rad)
        
        assert len(relationships) == 1
        assert relationships[0].relationship_type == "golden_ratio_angle"
        assert relationships[0].mathematical_constant == "golden_ratio"
        assert relationships[0].significance_score == 0.95
    
    def test_find_harmonic_relationships(self):
        """Test finding harmonic relationships between angles"""
        # Create angles in harmonic ratios
        angles_deg = [60.0, 90.0]  # Ratio 2:3
        angles_rad = [math.radians(a) for a in angles_deg]
        
        relationships = self.analyzer._find_harmonic_relationships(angles_deg, angles_rad)
        
        # Should find harmonic relationship
        harmonic_relations = [r for r in relationships if r.relationship_type == "harmonic_angle_ratio"]
        assert len(harmonic_relations) > 0
    
    def test_find_complementary_angles(self):
        """Test finding complementary angle relationships"""
        angles_deg = [30.0, 60.0]  # Sum to 90°
        angles_rad = [math.radians(a) for a in angles_deg]
        
        relationships = self.analyzer._find_trigonometric_identities(angles_deg, angles_rad)
        
        complementary_relations = [r for r in relationships if r.relationship_type == "complementary_angles"]
        assert len(complementary_relations) == 1
        assert complementary_relations[0].mathematical_constant == "complementary"
    
    def test_find_supplementary_angles(self):
        """Test finding supplementary angle relationships"""
        angles_deg = [120.0, 60.0]  # Sum to 180°
        angles_rad = [math.radians(a) for a in angles_deg]
        
        relationships = self.analyzer._find_trigonometric_identities(angles_deg, angles_rad)
        
        supplementary_relations = [r for r in relationships if r.relationship_type == "supplementary_angles"]
        assert len(supplementary_relations) == 1
        assert supplementary_relations[0].mathematical_constant == "supplementary"
    
    def test_analyze_character_geometry(self):
        """Test comprehensive character geometry analysis"""
        # Create mock character boxes
        from app.services.ocr_engine import CharacterBox
        
        characters = [
            CharacterBox('A', 0, 0, 10, 15, 0.9, 0, 0, 0, 0),
            CharacterBox('B', 30, 0, 10, 15, 0.9, 0, 0, 0, 0),
            CharacterBox('C', 15, 26, 10, 15, 0.9, 0, 0, 0, 0),  # Forms triangle
        ]
        
        analysis = self.analyzer.analyze_character_geometry(characters)
        
        assert 'character_count' in analysis
        assert analysis['character_count'] == 3
        assert 'angle_measurements' in analysis
        assert 'geometric_patterns' in analysis
        assert 'trigonometric_relationships' in analysis
        assert 'distance_statistics' in analysis
        assert 'analysis_summary' in analysis
    
    def test_analyze_character_geometry_insufficient_data(self):
        """Test character geometry analysis with insufficient data"""
        from app.services.ocr_engine import CharacterBox
        
        characters = [
            CharacterBox('A', 0, 0, 10, 15, 0.9, 0, 0, 0, 0),
            CharacterBox('B', 30, 0, 10, 15, 0.9, 0, 0, 0, 0),
        ]
        
        analysis = self.analyzer.analyze_character_geometry(characters)
        
        assert 'error' in analysis
        assert "Need at least 3 characters" in analysis['error']