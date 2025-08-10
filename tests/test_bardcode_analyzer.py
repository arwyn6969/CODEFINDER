"""
Tests for the BardCodeAnalyzer
"""
import pytest
import math
from unittest.mock import Mock

from app.services.bardcode_analyzer import (
    BardCodeAnalyzer, Point, Triangle, MathematicalConstant, GeometricConstruction
)
from app.models.database_models import Character


class TestBardCodeAnalyzer:
    
    @pytest.fixture
    def analyzer(self):
        """Create a BardCodeAnalyzer instance"""
        return BardCodeAnalyzer()
    
    @pytest.fixture
    def sample_characters(self):
        """Create sample character data for testing"""
        characters = []
        # Create characters forming a right triangle (3-4-5)
        positions = [(0, 0), (3, 0), (0, 4)]
        
        for i, (x, y) in enumerate(positions):
            char = Mock()
            char.id = i + 1
            char.character = chr(65 + i)  # A, B, C
            char.x = float(x)
            char.y = float(y)
            char.confidence = 0.9
            characters.append(char)
        
        return characters
    
    def test_analyzer_initialization(self, analyzer):
        """Test BardCodeAnalyzer initializes correctly"""
        assert analyzer.tolerance == 0.01
        assert analyzer.min_significance == 0.7
        assert len(analyzer.MATHEMATICAL_CONSTANTS) > 0
        assert 'pi' in analyzer.MATHEMATICAL_CONSTANTS
        assert 'phi' in analyzer.MATHEMATICAL_CONSTANTS
    
    def test_point_creation(self):
        """Test Point creation and properties"""
        point = Point(3.0, 4.0)
        assert point.x == 3.0
        assert point.y == 4.0
    
    def test_triangle_creation_and_properties(self):
        """Test Triangle creation and property calculations"""
        # Create a 3-4-5 right triangle
        triangle = Triangle(Point(0, 0), Point(3, 0), Point(0, 4))
        
        # Test side calculations - sides are: P1-P2, P2-P3, P3-P1
        sides = triangle.sides
        expected_sides = sorted([3.0, 4.0, 5.0])  # 3-4-5 triangle
        actual_sides = sorted(sides)
        
        for expected, actual in zip(expected_sides, actual_sides):
            assert abs(expected - actual) < 0.001
        
        # Test angle calculations - should have one 90° angle and two others
        angles = triangle.angles
        sorted_angles = sorted(angles)
        
        # Should have angles approximately 37°, 53°, 90° for a 3-4-5 triangle
        assert abs(sorted_angles[0] - 36.87) < 1.0  # Smallest angle
        assert abs(sorted_angles[1] - 53.13) < 1.0  # Middle angle  
        assert abs(sorted_angles[2] - 90.0) < 1.0   # Right angle
        
        # Test ratios
        ratios = triangle.ratios
        expected_ratios = sorted([1.0, 4.0/3.0, 5.0/3.0])
        actual_ratios = sorted(ratios)
        for expected, actual in zip(expected_ratios, actual_ratios):
            assert abs(expected - actual) < 0.1
    
    def test_mathematical_constant_detection(self, analyzer):
        """Test detection of mathematical constants"""
        # Test with values close to known constants
        test_values = [
            3.14159,  # Close to π
            2.71828,  # Close to e
            1.61803,  # Close to φ
            1.41421,  # Close to √2
        ]
        
        constants = analyzer.detect_mathematical_constants(test_values, tolerance=0.01)
        
        # Should detect π, e, φ, and √2
        detected_names = [c.name for c in constants]
        assert 'pi' in detected_names
        assert 'e' in detected_names
        assert 'phi' in detected_names
        assert 'sqrt_2' in detected_names
        
        # Check accuracy
        pi_constant = next(c for c in constants if c.name == 'pi')
        assert pi_constant.accuracy > 0.99
        assert pi_constant.confidence > 0.98
    
    def test_mathematical_constant_reciprocal_detection(self, analyzer):
        """Test detection of reciprocal constants"""
        # Test with reciprocal of φ
        phi_reciprocal = 1 / analyzer.MATHEMATICAL_CONSTANTS['phi']
        constants = analyzer.detect_mathematical_constants([phi_reciprocal])
        
        # Should detect φ as reciprocal
        reciprocal_constants = [c for c in constants if 'reciprocal' in c.name]
        assert len(reciprocal_constants) > 0
    
    def test_analyze_page_geometry(self, analyzer, sample_characters):
        """Test page geometry analysis"""
        page_width = 100.0
        page_height = 100.0
        
        results = analyzer.analyze_page_geometry(sample_characters, page_width, page_height)
        
        # Check basic structure
        assert 'page_dimensions' in results
        assert 'total_points' in results
        assert 'triangular_constructions' in results
        assert 'mathematical_constants' in results
        assert 'significance_scores' in results
        
        # Check page dimensions
        assert results['page_dimensions']['width'] == page_width
        assert results['page_dimensions']['height'] == page_height
        
        # Check point count
        assert results['total_points'] == len(sample_characters)
        
        # Should find at least one significant triangle (3-4-5)
        assert len(results['triangular_constructions']) > 0
    
    def test_analyze_page_geometry_empty_input(self, analyzer):
        """Test page geometry analysis with empty input"""
        results = analyzer.analyze_page_geometry([], 100.0, 100.0)
        assert 'error' in results
    
    def test_construct_triangles(self, analyzer):
        """Test triangle construction from points"""
        points = [Point(0, 0), Point(3, 0), Point(0, 4), Point(1, 1)]
        triangles = analyzer.construct_triangles(points)
        
        # Should create 4 triangles from 4 points (C(4,3) = 4)
        assert len(triangles) == 4
        
        # All should be valid triangles
        for triangle in triangles:
            assert analyzer._is_valid_triangle(triangle)
    
    def test_find_sacred_geometry(self, analyzer):
        """Test sacred geometry pattern detection"""
        # Create points forming a 3-4-5 triangle
        points = [Point(0, 0), Point(3, 0), Point(0, 4)]
        constructions = analyzer.find_sacred_geometry(points)
        
        # Should find Pythagorean triangle
        pythagorean_constructions = [c for c in constructions if c.construction_type == 'pythagorean_triangle']
        assert len(pythagorean_constructions) > 0
    
    def test_extract_coordinates(self, analyzer):
        """Test geographic coordinate extraction from angles"""
        # Test with angle close to Great Pyramid latitude
        angles = [29.98, 45.0, 31.1]  # Close to Giza coordinates
        coordinates = analyzer.extract_coordinates(angles)
        
        # Should find coordinates
        assert len(coordinates) > 0
        
        # Check for Giza coordinates
        giza_coords = [(lat, lon) for lat, lon in coordinates 
                      if lat and abs(lat - 29.9792) < 0.1]
        assert len(giza_coords) > 0
    
    def test_validate_significance(self, analyzer):
        """Test significance validation"""
        # Create a construction with detected constants
        constants = [
            MathematicalConstant(
                name='pi', symbol='π', expected_value=math.pi, detected_value=3.14159,
                accuracy=0.99, confidence=0.98, context='test', evidence={}
            )
        ]
        
        construction = GeometricConstruction(
            construction_type='test_triangle',
            points=[Point(0, 0), Point(1, 0), Point(0, 1)],
            measurements={'test': 1.0},
            constants_detected=constants,
            significance_score=0.8,
            description='Test construction'
        )
        
        significance = analyzer.validate_significance(construction)
        
        # Check significance test results
        assert 'constants_count' in significance
        assert 'constants_accuracy' in significance
        assert 'geometric_complexity' in significance
        assert 'combined_significance' in significance
        
        # Should have reasonable significance scores
        assert 0 <= significance['combined_significance'] <= 1
    
    def test_normalize_coordinates(self, analyzer):
        """Test coordinate normalization"""
        points = [Point(0, 0), Point(50, 100), Point(100, 200)]
        normalized = analyzer._normalize_coordinates(points, 100, 200)
        
        # Check normalization
        assert normalized[0] == Point(0.0, 0.0)
        assert normalized[1] == Point(0.5, 0.5)
        assert normalized[2] == Point(1.0, 1.0)
    
    def test_calculate_bounds(self, analyzer):
        """Test bounding box calculation"""
        points = [Point(1, 2), Point(5, 8), Point(3, 4)]
        bounds = analyzer._calculate_bounds(points)
        
        assert bounds['min_x'] == 1
        assert bounds['max_x'] == 5
        assert bounds['min_y'] == 2
        assert bounds['max_y'] == 8
    
    def test_calculate_bounds_empty(self, analyzer):
        """Test bounding box calculation with empty points"""
        bounds = analyzer._calculate_bounds([])
        
        assert bounds['min_x'] == 0
        assert bounds['max_x'] == 0
        assert bounds['min_y'] == 0
        assert bounds['max_y'] == 0
    
    def test_analyze_triangle_properties(self, analyzer):
        """Test triangle property analysis"""
        # Test with right triangle
        right_triangle = Triangle(Point(0, 0), Point(3, 0), Point(0, 4))
        properties = analyzer._analyze_triangle_properties(right_triangle)
        
        assert properties['is_right_triangle'] == True
        assert properties['significance_score'] > 0
        
        # Test with equilateral triangle
        height = math.sqrt(3) / 2
        equilateral_triangle = Triangle(Point(0, 0), Point(1, 0), Point(0.5, height))
        properties = analyzer._analyze_triangle_properties(equilateral_triangle)
        
        assert properties['is_equilateral'] == True
    
    def test_are_collinear(self, analyzer):
        """Test collinearity detection"""
        # Test collinear points
        p1, p2, p3 = Point(0, 0), Point(1, 1), Point(2, 2)
        assert analyzer._are_collinear(p1, p2, p3, 0.01) == True
        
        # Test non-collinear points
        p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        assert analyzer._are_collinear(p1, p2, p3, 0.01) == False
    
    def test_find_collinear_points(self, analyzer):
        """Test finding groups of collinear points"""
        # Create points with some collinear groups
        points = [
            Point(0, 0), Point(1, 1), Point(2, 2),  # Collinear group
            Point(0, 1), Point(1, 2), Point(2, 3),  # Another collinear group
            Point(5, 5)  # Isolated point
        ]
        
        collinear_groups = analyzer._find_collinear_points(points)
        
        # Should find collinear groups
        assert len(collinear_groups) > 0
        
        # Each group should have at least 3 points
        for group in collinear_groups:
            assert len(group) >= 3
    
    def test_is_valid_triangle(self, analyzer):
        """Test triangle validity checking"""
        # Valid triangle
        valid_triangle = Triangle(Point(0, 0), Point(1, 0), Point(0, 1))
        assert analyzer._is_valid_triangle(valid_triangle) == True
        
        # Degenerate triangle (collinear points)
        degenerate_triangle = Triangle(Point(0, 0), Point(1, 0), Point(2, 0))
        assert analyzer._is_valid_triangle(degenerate_triangle) == False
    
    def test_calculate_accuracy(self, analyzer):
        """Test accuracy calculation"""
        # Perfect match
        assert analyzer._calculate_accuracy(3.14159, math.pi) > 0.99
        
        # No match
        assert analyzer._calculate_accuracy(1.0, math.pi) < 0.5
        
        # Zero values
        assert analyzer._calculate_accuracy(0.0, 0.0) == 1.0
    
    def test_calculate_confidence(self, analyzer):
        """Test confidence calculation"""
        # High accuracy should give high confidence
        high_confidence = analyzer._calculate_confidence(0.99)
        assert high_confidence > 0.98
        
        # Low accuracy should give low confidence
        low_confidence = analyzer._calculate_confidence(0.5)
        assert low_confidence < 0.3
    
    def test_get_constant_symbol(self, analyzer):
        """Test mathematical constant symbol retrieval"""
        assert analyzer._get_constant_symbol('pi') == 'π'
        assert analyzer._get_constant_symbol('phi') == 'φ'
        assert analyzer._get_constant_symbol('sqrt_2') == '√2'
        assert analyzer._get_constant_symbol('unknown') == 'unknown'
    
    def test_mathematical_constants_values(self, analyzer):
        """Test that mathematical constants have correct values"""
        constants = analyzer.MATHEMATICAL_CONSTANTS
        
        # Check key constants
        assert abs(constants['pi'] - math.pi) < 1e-10
        assert abs(constants['e'] - math.e) < 1e-10
        assert abs(constants['phi'] - 1.618033988749895) < 1e-10
        assert abs(constants['sqrt_2'] - math.sqrt(2)) < 1e-10
        assert abs(constants['sqrt_3'] - math.sqrt(3)) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])