"""
Simple test for geographic coordinate extraction
"""
import pytest
from app.services.bardcode_analyzer import BardCodeAnalyzer


def test_basic_coordinate_extraction():
    """Test basic coordinate extraction"""
    analyzer = BardCodeAnalyzer()
    angles = [29.98, 31.1, 45.0, 60.0]
    
    coordinates = analyzer.extract_coordinates(angles)
    
    # Should return a list
    assert isinstance(coordinates, list)


def test_advanced_coordinate_extraction():
    """Test advanced coordinate extraction"""
    analyzer = BardCodeAnalyzer()
    
    geometric_measurements = [
        {'type': 'angle', 'value': 29.98},
        {'type': 'angle', 'value': 31.13},
        {'type': 'ratio', 'value': 1.618},
        {'type': 'distance', 'value': 3.14159}
    ]
    
    results = analyzer.extract_geographic_coordinates_advanced(geometric_measurements)
    
    # Should return a dictionary with expected keys
    assert isinstance(results, dict)
    assert 'potential_coordinates' in results
    assert 'coordinate_pairs' in results
    assert 'historical_sites' in results
    assert 'accuracy_scores' in results


if __name__ == "__main__":
    pytest.main([__file__])