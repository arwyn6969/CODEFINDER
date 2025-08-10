"""
Tests for Geometric Analysis Visualization System
"""
import pytest
import math
import json
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from app.services.geometric_visualizer import (
    GeometricVisualizer, Point, GeometricElement, GeometricVisualization,
    AngleMeasurement, DistanceMeasurement, GeometricVisualizationType,
    SacredGeometryType
)
from app.models.database_models import Document, Pattern, Page


class TestGeometricVisualizer:
    """Test cases for GeometricVisualizer"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session"""
        return Mock(spec=Session)
    
    @pytest.fixture
    def sample_points(self):
        """Create sample points for testing"""
        return [
            Point(x=0, y=0, label="origin", character="A", pattern_id=1, confidence=0.9),
            Point(x=10, y=0, label="right", character="B", pattern_id=1, confidence=0.8),
            Point(x=10, y=10, label="top-right", character="C", pattern_id=2, confidence=0.7),
            Point(x=0, y=10, label="top-left", character="D", pattern_id=2, confidence=0.9),
            Point(x=5, y=5, label="center", character="E", pattern_id=3, confidence=0.6)
        ]
    
    @pytest.fixture
    def sample_patterns(self):
        """Create sample database patterns"""
        patterns = []
        
        # Pattern 1: Square corners
        pattern1 = Mock(spec=Pattern)
        pattern1.id = 1
        pattern1.document_id = 1
        pattern1.pattern_type = "geometric_shape"
        pattern1.coordinates = [
            {"x": 0, "y": 0, "character": "A"},
            {"x": 10, "y": 0, "character": "B"}
        ]
        pattern1.confidence = 0.9
        pattern1.description = "Square base"
        patterns.append(pattern1)
        
        # Pattern 2: Square top
        pattern2 = Mock(spec=Pattern)
        pattern2.id = 2
        pattern2.document_id = 1
        pattern2.pattern_type = "geometric_shape"
        pattern2.coordinates = [
            {"x": 10, "y": 10, "character": "C"},
            {"x": 0, "y": 10, "character": "D"}
        ]
        pattern2.confidence = 0.8
        pattern2.description = "Square top"
        patterns.append(pattern2)
        
        # Pattern 3: Center point
        pattern3 = Mock(spec=Pattern)
        pattern3.id = 3
        pattern3.document_id = 1
        pattern3.pattern_type = "center_point"
        pattern3.coordinates = [
            {"x": 5, "y": 5, "character": "E"}
        ]
        pattern3.confidence = 0.7
        pattern3.description = "Center point"
        patterns.append(pattern3)
        
        return patterns
    
    @pytest.fixture
    def visualizer(self, mock_db_session):
        """Create GeometricVisualizer instance"""
        with patch('app.services.geometric_visualizer.get_db') as mock_get_db:
            mock_get_db.return_value = mock_db_session
            return GeometricVisualizer(mock_db_session)
    
    def test_init(self, visualizer, mock_db_session):
        """Test GeometricVisualizer initialization"""
        assert visualizer.db == mock_db_session
        assert visualizer.golden_ratio == (1 + math.sqrt(5)) / 2
        assert visualizer.pi == math.pi
        assert 36 in visualizer.sacred_angles
        assert 'angles' in visualizer.color_schemes
        assert 'distances' in visualizer.color_schemes
        assert 'sacred_geometry' in visualizer.color_schemes
    
    def test_extract_geometric_points(self, visualizer, sample_patterns):
        """Test extracting geometric points from patterns"""
        # Mock database query
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = sample_patterns
        visualizer.db.query.return_value = mock_query
        
        points = visualizer._extract_geometric_points(document_id=1)
        
        assert len(points) == 5
        assert points[0].x == 0 and points[0].y == 0
        assert points[0].character == "A"
        assert points[0].pattern_id == 1
        assert points[0].confidence == 0.9
    
    def test_extract_geometric_points_with_pattern_filter(self, visualizer, sample_patterns):
        """Test extracting geometric points with pattern ID filter"""
        # Mock database query
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = [sample_patterns[0]]  # Only first pattern
        visualizer.db.query.return_value = mock_query
        
        points = visualizer._extract_geometric_points(document_id=1, pattern_ids=[1])
        
        assert len(points) == 2
        assert all(p.pattern_id == 1 for p in points)
    
    def test_calculate_angle(self, visualizer):
        """Test angle calculation between three points"""
        # Right angle test
        point1 = Point(x=1, y=0)
        vertex = Point(x=0, y=0)
        point2 = Point(x=0, y=1)
        
        angle = visualizer._calculate_angle(point1, vertex, point2)
        angle_degrees = math.degrees(angle)
        
        assert abs(angle_degrees - 90) < 0.1
    
    def test_calculate_angle_measurements(self, visualizer, sample_points):
        """Test calculating angle measurements"""
        angles = visualizer._calculate_angle_measurements(sample_points)
        
        assert len(angles) > 0
        assert all(isinstance(angle, AngleMeasurement) for angle in angles)
        assert all(0 <= angle.angle_degrees <= 360 for angle in angles)
        assert all(angle.angle_type in ['acute', 'right', 'obtuse', 'straight', 'reflex'] for angle in angles)
    
    def test_classify_angle(self, visualizer):
        """Test angle classification"""
        assert visualizer._classify_angle(45) == 'acute'
        assert visualizer._classify_angle(90) == 'right'
        assert visualizer._classify_angle(120) == 'obtuse'
        assert visualizer._classify_angle(180) == 'straight'
        assert visualizer._classify_angle(270) == 'reflex'
    
    def test_check_sacred_angle(self, visualizer):
        """Test sacred angle detection"""
        assert visualizer._check_sacred_angle(36) == "Pentagon/Pentagram"
        assert visualizer._check_sacred_angle(60) == "Hexagon/Triangle"
        assert visualizer._check_sacred_angle(90) == "Square/Right angle"
        assert visualizer._check_sacred_angle(45) == "Octagon"
        assert visualizer._check_sacred_angle(123) is None
    
    def test_calculate_distance_measurements(self, visualizer, sample_points):
        """Test calculating distance measurements"""
        distances = visualizer._calculate_distance_measurements(sample_points)
        
        assert len(distances) > 0
        assert all(isinstance(distance, DistanceMeasurement) for distance in distances)
        assert all(distance.distance >= 0 for distance in distances)
        assert all(distance.distance_type == 'euclidean' for distance in distances)
    
    def test_calculate_angle_significance(self, visualizer):
        """Test angle significance calculation"""
        # Sacred angle should have high significance
        significance = visualizer._calculate_angle_significance(36, "Pentagon/Pentagram")
        assert significance > 0.7
        
        # Right angle should have good significance
        significance = visualizer._calculate_angle_significance(90, None)
        assert significance > 0.5
        
        # Random angle should have low significance
        significance = visualizer._calculate_angle_significance(123, None)
        assert significance < 0.5
    
    def test_calculate_distance_significance(self, visualizer):
        """Test distance significance calculation"""
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        # Golden ratio distance should have high significance
        significance = visualizer._calculate_distance_significance(golden_ratio, 1.0)
        assert significance > 0.8
        
        # Fibonacci number should have good significance
        significance = visualizer._calculate_distance_significance(8, None)
        assert significance > 0.6
        
        # Random distance should have low significance
        significance = visualizer._calculate_distance_significance(7.3, None)
        assert significance < 0.5
    
    def test_is_fibonacci_related(self, visualizer):
        """Test Fibonacci number detection"""
        assert visualizer._is_fibonacci_related(1)
        assert visualizer._is_fibonacci_related(2)
        assert visualizer._is_fibonacci_related(3)
        assert visualizer._is_fibonacci_related(5)
        assert visualizer._is_fibonacci_related(8)
        assert visualizer._is_fibonacci_related(13)
        assert not visualizer._is_fibonacci_related(7)
        assert not visualizer._is_fibonacci_related(11)
    
    def test_create_coordinate_system(self, visualizer, sample_points):
        """Test coordinate system creation"""
        coord_system = visualizer._create_coordinate_system(sample_points)
        
        assert 'bounds' in coord_system
        assert 'grid' in coord_system
        assert 'axes' in coord_system
        
        bounds = coord_system['bounds']
        assert bounds['min_x'] < 0  # Should have padding
        assert bounds['max_x'] > 10  # Should have padding
        assert bounds['min_y'] < 0  # Should have padding
        assert bounds['max_y'] > 10  # Should have padding
    
    def test_create_coordinate_system_with_grid(self, visualizer, sample_points):
        """Test coordinate system creation with grid"""
        coord_system = visualizer._create_coordinate_system(sample_points, show_grid=True)
        
        assert coord_system['grid']['show'] is True
        assert 'spacing' in coord_system['grid']
        assert 'color' in coord_system['grid']
    
    def test_get_point_color(self, visualizer):
        """Test point color assignment based on confidence"""
        high_conf_point = Point(x=0, y=0, confidence=0.9)
        medium_conf_point = Point(x=0, y=0, confidence=0.7)
        low_conf_point = Point(x=0, y=0, confidence=0.4)
        
        assert visualizer._get_point_color(high_conf_point) == '#00FF00'
        assert visualizer._get_point_color(medium_conf_point) == '#FFFF00'
        assert visualizer._get_point_color(low_conf_point) == '#FF0000'
    
    def test_is_golden_rectangle(self, visualizer):
        """Test golden rectangle detection"""
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        # Create points for a golden rectangle
        points = [
            Point(x=0, y=0),
            Point(x=golden_ratio, y=0),
            Point(x=golden_ratio, y=1),
            Point(x=0, y=1)
        ]
        
        assert visualizer._is_golden_rectangle(points)
        
        # Create points for a regular square (not golden)
        square_points = [
            Point(x=0, y=0),
            Point(x=1, y=0),
            Point(x=1, y=1),
            Point(x=0, y=1)
        ]
        
        assert not visualizer._is_golden_rectangle(square_points)
    
    def test_follows_fibonacci_sequence(self, visualizer):
        """Test Fibonacci sequence detection"""
        # Perfect Fibonacci sequence
        fib_sequence = [1, 1, 2, 3, 5, 8]
        assert visualizer._follows_fibonacci_sequence(fib_sequence)
        
        # Scaled Fibonacci sequence
        scaled_fib = [2, 2, 4, 6, 10, 16]
        assert visualizer._follows_fibonacci_sequence(scaled_fib)
        
        # Non-Fibonacci sequence
        random_sequence = [1, 3, 7, 12, 20]
        assert not visualizer._follows_fibonacci_sequence(random_sequence)
    
    def test_is_regular_pentagon(self, visualizer):
        """Test regular pentagon detection"""
        # Create points for a regular pentagon
        center_x, center_y = 0, 0
        radius = 10
        pentagon_points = []
        
        for i in range(5):
            angle = 2 * math.pi * i / 5
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            pentagon_points.append(Point(x=x, y=y))
        
        assert visualizer._is_regular_pentagon(pentagon_points)
        
        # Create irregular points
        irregular_points = [
            Point(x=0, y=0),
            Point(x=1, y=0),
            Point(x=2, y=1),
            Point(x=1, y=2),
            Point(x=0, y=1)
        ]
        
        assert not visualizer._is_regular_pentagon(irregular_points)
    
    def test_is_regular_hexagon(self, visualizer):
        """Test regular hexagon detection"""
        # Create points for a regular hexagon
        center_x, center_y = 0, 0
        radius = 10
        hexagon_points = []
        
        for i in range(6):
            angle = 2 * math.pi * i / 6
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            hexagon_points.append(Point(x=x, y=y))
        
        assert visualizer._is_regular_hexagon(hexagon_points)
    
    def test_detect_golden_ratio_patterns(self, visualizer):
        """Test golden ratio pattern detection"""
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        # Create golden rectangle points
        points = [
            Point(x=0, y=0),
            Point(x=golden_ratio, y=0),
            Point(x=golden_ratio, y=1),
            Point(x=0, y=1)
        ]
        
        patterns = visualizer._detect_golden_ratio_patterns(points)
        
        assert len(patterns) > 0
        assert patterns[0]['type'] == 'golden_ratio'
        assert patterns[0]['element_type'] == 'polygon'
        assert patterns[0]['confidence'] > 0.5
    
    def test_detect_fibonacci_spiral_patterns(self, visualizer):
        """Test Fibonacci spiral pattern detection"""
        # Create points that follow Fibonacci distances from center
        center_x, center_y = 0, 0
        points = []
        
        fibonacci_distances = [1, 1, 2, 3, 5, 8]
        for i, dist in enumerate(fibonacci_distances):
            angle = 2 * math.pi * i / len(fibonacci_distances)
            x = center_x + dist * math.cos(angle)
            y = center_y + dist * math.sin(angle)
            points.append(Point(x=x, y=y))
        
        patterns = visualizer._detect_fibonacci_spiral_patterns(points)
        
        assert len(patterns) > 0
        assert patterns[0]['type'] == 'fibonacci_spiral'
        assert patterns[0]['element_type'] == 'spiral'
    
    def test_detect_vesica_piscis_patterns(self, visualizer):
        """Test Vesica Piscis pattern detection"""
        # Create two overlapping circles
        center1 = Point(x=0, y=0)
        center2 = Point(x=5, y=0)
        radius = 5
        
        # Add points on the circles
        points = [center1, center2]
        
        # Add points on first circle
        for i in range(4):
            angle = 2 * math.pi * i / 4
            x = center1.x + radius * math.cos(angle)
            y = center1.y + radius * math.sin(angle)
            points.append(Point(x=x, y=y))
        
        # Add points on second circle
        for i in range(4):
            angle = 2 * math.pi * i / 4
            x = center2.x + radius * math.cos(angle)
            y = center2.y + radius * math.sin(angle)
            points.append(Point(x=x, y=y))
        
        patterns = visualizer._detect_vesica_piscis_patterns(points)
        
        assert len(patterns) > 0
        assert patterns[0]['type'] == 'vesica_piscis'
        assert patterns[0]['element_type'] == 'circles'
    
    def test_detect_pentagram_patterns(self, visualizer):
        """Test pentagram pattern detection"""
        # Create regular pentagon points
        center_x, center_y = 0, 0
        radius = 10
        points = []
        
        for i in range(5):
            angle = 2 * math.pi * i / 5
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append(Point(x=x, y=y))
        
        patterns = visualizer._detect_pentagram_patterns(points)
        
        assert len(patterns) > 0
        assert patterns[0]['type'] == 'pentagram'
        assert patterns[0]['element_type'] == 'polygon'
    
    def test_detect_hexagram_patterns(self, visualizer):
        """Test hexagram pattern detection"""
        # Create regular hexagon points
        center_x, center_y = 0, 0
        radius = 10
        points = []
        
        for i in range(6):
            angle = 2 * math.pi * i / 6
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append(Point(x=x, y=y))
        
        patterns = visualizer._detect_hexagram_patterns(points)
        
        assert len(patterns) > 0
        assert patterns[0]['type'] == 'hexagram'
        assert patterns[0]['element_type'] == 'polygon'
    
    @patch('app.services.geometric_visualizer.GeometricVisualizer._extract_geometric_points')
    def test_create_angle_measurement_visualization(self, mock_extract_points, visualizer, sample_points):
        """Test creating angle measurement visualization"""
        mock_extract_points.return_value = sample_points
        
        visualization = visualizer.create_angle_measurement_visualization(document_id=1)
        
        assert isinstance(visualization, GeometricVisualization)
        assert visualization.visualization_type == GeometricVisualizationType.ANGLE_MEASUREMENT
        assert visualization.document_id == 1
        assert len(visualization.elements) > 0
        assert 'bounds' in visualization.coordinate_system
        assert 'total_angles' in visualization.measurements_summary
        assert 'chart_type' in visualization.d3_config
        assert 'filters' in visualization.interactive_features
    
    @patch('app.services.geometric_visualizer.GeometricVisualizer._extract_geometric_points')
    def test_create_distance_analysis_visualization(self, mock_extract_points, visualizer, sample_points):
        """Test creating distance analysis visualization"""
        mock_extract_points.return_value = sample_points
        
        visualization = visualizer.create_distance_analysis_visualization(document_id=1)
        
        assert isinstance(visualization, GeometricVisualization)
        assert visualization.visualization_type == GeometricVisualizationType.DISTANCE_ANALYSIS
        assert visualization.document_id == 1
        assert len(visualization.elements) > 0
        assert 'total_distances' in visualization.measurements_summary
        assert visualization.d3_config['chart_type'] == 'distance_analysis'
    
    @patch('app.services.geometric_visualizer.GeometricVisualizer._extract_geometric_points')
    def test_create_sacred_geometry_visualization(self, mock_extract_points, visualizer, sample_points):
        """Test creating sacred geometry visualization"""
        mock_extract_points.return_value = sample_points
        
        visualization = visualizer.create_sacred_geometry_visualization(
            document_id=1, 
            geometry_types=[SacredGeometryType.GOLDEN_RATIO, SacredGeometryType.PENTAGRAM]
        )
        
        assert isinstance(visualization, GeometricVisualization)
        assert visualization.visualization_type == GeometricVisualizationType.SACRED_GEOMETRY
        assert visualization.document_id == 1
        assert len(visualization.elements) > 0
        assert len(visualization.sacred_geometry_patterns) >= 0
        assert visualization.d3_config['chart_type'] == 'sacred_geometry'
    
    @patch('app.services.geometric_visualizer.GeometricVisualizer._extract_geometric_points')
    def test_create_interactive_coordinate_plot(self, mock_extract_points, visualizer, sample_points):
        """Test creating interactive coordinate plot"""
        mock_extract_points.return_value = sample_points
        
        visualization = visualizer.create_interactive_coordinate_plot(
            document_id=1, 
            show_grid=True, 
            show_measurements=True
        )
        
        assert isinstance(visualization, GeometricVisualization)
        assert visualization.visualization_type == GeometricVisualizationType.COORDINATE_PLOT
        assert visualization.document_id == 1
        assert len(visualization.elements) > 0
        assert 'points_count' in visualization.measurements_summary
        assert visualization.d3_config['chart_type'] == 'interactive_plot'
        assert 'navigation' in visualization.interactive_features
        assert 'layers' in visualization.interactive_features
    
    def test_generate_d3_javascript_config(self, visualizer, sample_points):
        """Test D3.js JavaScript configuration generation"""
        # Create a simple visualization
        elements = [
            GeometricElement(
                element_type='point',
                points=[sample_points[0]],
                measurements={},
                properties={'test': True},
                style={'fill': '#FF0000'},
                annotations=['test point']
            )
        ]
        
        visualization = GeometricVisualization(
            visualization_id='test_viz',
            document_id=1,
            visualization_type=GeometricVisualizationType.COORDINATE_PLOT,
            elements=elements,
            coordinate_system={'bounds': {'min_x': 0, 'max_x': 10, 'min_y': 0, 'max_y': 10}},
            measurements_summary={'test': 'summary'},
            sacred_geometry_patterns=[],
            interactive_features={'test': 'features'},
            d3_config={'test': 'config'},
            export_data={'test': 'data'}
        )
        
        js_config = visualizer.generate_d3_javascript_config(visualization)
        
        assert isinstance(js_config, str)
        config_dict = json.loads(js_config)
        assert config_dict['visualization_id'] == 'test_viz'
        assert config_dict['type'] == 'coordinate_plot'
        assert 'data' in config_dict
        assert 'elements' in config_dict['data']
        assert len(config_dict['data']['elements']) == 1
    
    def test_create_measurement_comparison_chart(self, visualizer):
        """Test creating measurement comparison chart"""
        # Mock document query
        mock_document = Mock(spec=Document)
        mock_document.id = 1
        mock_document.filename = "test_document.pdf"
        
        visualizer.db.query.return_value.filter.return_value.first.return_value = mock_document
        
        # Mock geometric data extraction
        with patch.object(visualizer, '_extract_geometric_points') as mock_extract, \
             patch.object(visualizer, '_calculate_angle_measurements') as mock_angles, \
             patch.object(visualizer, '_calculate_distance_measurements') as mock_distances, \
             patch.object(visualizer, '_detect_sacred_geometry_patterns') as mock_patterns:
            
            mock_extract.return_value = sample_points[:3]
            mock_angles.return_value = [Mock(angle_type='right', sacred_angle='Square')]
            mock_distances.return_value = [Mock(ratio_to_golden=1.0, distance=5)]
            mock_patterns.return_value = [{'type': 'golden_ratio'}]
            
            comparison_data = visualizer.create_measurement_comparison_chart([1])
            
            assert 'documents' in comparison_data
            assert 'statistical_comparisons' in comparison_data
            assert len(comparison_data['documents']) == 1
            
            doc_data = comparison_data['documents'][0]
            assert doc_data['id'] == 1
            assert doc_data['name'] == "test_document.pdf"
            assert 'angles' in doc_data
            assert 'distances' in doc_data
    
    def test_generate_angle_summary(self, visualizer):
        """Test angle summary generation"""
        angles = [
            AngleMeasurement(
                vertex=Point(x=0, y=0),
                point1=Point(x=1, y=0),
                point2=Point(x=0, y=1),
                angle_degrees=90,
                angle_radians=math.pi/2,
                angle_type='right',
                significance=0.8,
                sacred_angle='Square'
            ),
            AngleMeasurement(
                vertex=Point(x=0, y=0),
                point1=Point(x=1, y=0),
                point2=Point(x=1, y=1),
                angle_degrees=45,
                angle_radians=math.pi/4,
                angle_type='acute',
                significance=0.3,
                sacred_angle=None
            )
        ]
        
        summary = visualizer._generate_angle_summary(angles)
        
        assert summary['total_angles'] == 2
        assert summary['angle_types']['right'] == 1
        assert summary['angle_types']['acute'] == 1
        assert summary['sacred_angles'] == 1
        assert summary['significant_angles'] == 1
        assert summary['average_angle'] == 67.5
    
    def test_generate_distance_summary(self, visualizer):
        """Test distance summary generation"""
        distances = [
            DistanceMeasurement(
                point1=Point(x=0, y=0),
                point2=Point(x=1, y=0),
                distance=1.618,  # Golden ratio
                distance_type='euclidean',
                ratio_to_golden=1.0,
                significance=0.9,
                pattern_context='golden'
            ),
            DistanceMeasurement(
                point1=Point(x=0, y=0),
                point2=Point(x=5, y=0),
                distance=5,  # Fibonacci number
                distance_type='euclidean',
                ratio_to_golden=3.09,
                significance=0.7,
                pattern_context='fibonacci'
            )
        ]
        
        summary = visualizer._generate_distance_summary(distances)
        
        assert summary['total_distances'] == 2
        assert summary['golden_ratio_distances'] == 1
        assert summary['fibonacci_distances'] == 1
        assert summary['significant_distances'] == 2
        assert abs(summary['average_distance'] - 3.309) < 0.01
    
    def test_generate_sacred_geometry_summary(self, visualizer):
        """Test sacred geometry summary generation"""
        patterns = [
            {'type': 'golden_ratio', 'confidence': 0.8},
            {'type': 'pentagram', 'confidence': 0.9},
            {'type': 'golden_ratio', 'confidence': 0.6}
        ]
        
        summary = visualizer._generate_sacred_geometry_summary(patterns)
        
        assert summary['total_patterns'] == 3
        assert summary['pattern_types']['golden_ratio'] == 2
        assert summary['pattern_types']['pentagram'] == 1
        assert summary['high_confidence_patterns'] == 2
        assert abs(summary['average_confidence'] - 0.767) < 0.01
    
    def test_calculate_std_dev(self, visualizer):
        """Test standard deviation calculation"""
        values = [1, 2, 3, 4, 5]
        std_dev = visualizer._calculate_std_dev(values)
        
        # Expected standard deviation for [1,2,3,4,5] is sqrt(2)
        expected = math.sqrt(2)
        assert abs(std_dev - expected) < 0.01
        
        # Test empty list
        assert visualizer._calculate_std_dev([]) == 0
    
    def test_error_handling_insufficient_points(self, visualizer):
        """Test error handling with insufficient points"""
        with patch.object(visualizer, '_extract_geometric_points') as mock_extract:
            mock_extract.return_value = [Point(x=0, y=0)]  # Only one point
            
            with pytest.raises(ValueError, match="Need at least 3 points for angle measurements"):
                visualizer.create_angle_measurement_visualization(document_id=1)
            
            with pytest.raises(ValueError, match="Need at least 2 points for distance analysis"):
                visualizer.create_distance_analysis_visualization(document_id=1)
    
    def test_error_handling_database_error(self, visualizer):
        """Test error handling with database errors"""
        visualizer.db.query.side_effect = Exception("Database error")
        
        points = visualizer._extract_geometric_points(document_id=1)
        assert points == []
    
    def test_create_sacred_geometry_element(self, visualizer):
        """Test creating sacred geometry element"""
        pattern = {
            'type': 'golden_ratio',
            'element_type': 'polygon',
            'points': [
                {'x': 0, 'y': 0, 'label': 'p1'},
                {'x': 1, 'y': 0, 'label': 'p2'},
                {'x': 1, 'y': 1, 'label': 'p3'}
            ],
            'measurements': {'ratio': 1.618},
            'confidence': 0.8,
            'description': 'Golden Rectangle'
        }
        
        element = visualizer._create_sacred_geometry_element(pattern)
        
        assert isinstance(element, GeometricElement)
        assert element.element_type == 'polygon'
        assert len(element.points) == 3
        assert element.properties['sacred_geometry_type'] == 'golden_ratio'
        assert element.properties['confidence'] == 0.8
        assert 'Golden Rectangle' in element.annotations
    
    def test_d3_config_generation(self, visualizer, sample_points):
        """Test D3.js configuration generation for different visualization types"""
        coord_system = visualizer._create_coordinate_system(sample_points)
        
        # Test angle config
        angle_config = visualizer._create_d3_angle_config([], coord_system)
        assert angle_config['chart_type'] == 'angle_measurement'
        assert 'dimensions' in angle_config
        assert 'scales' in angle_config
        assert 'interactions' in angle_config
        
        # Test distance config
        distance_config = visualizer._create_d3_distance_config([], coord_system)
        assert distance_config['chart_type'] == 'distance_analysis'
        assert distance_config['elements']['highlight_golden_ratios'] is True
        
        # Test sacred geometry config
        sacred_config = visualizer._create_d3_sacred_geometry_config([], coord_system)
        assert sacred_config['chart_type'] == 'sacred_geometry'
        assert sacred_config['elements']['animate_construction'] is True
        
        # Test interactive plot config
        interactive_config = visualizer._create_d3_interactive_plot_config([], coord_system)
        assert interactive_config['chart_type'] == 'interactive_plot'
        assert 'layers' in interactive_config
        assert interactive_config['interactions']['pan_zoom'] is True
    
    def test_interactive_features_generation(self, visualizer):
        """Test interactive features generation"""
        # Test angle interactive features
        angle_features = visualizer._create_angle_interactive_features()
        assert 'filters' in angle_features
        assert 'tools' in angle_features
        assert 'animations' in angle_features
        assert 'angle_measurement' in angle_features['tools']
        
        # Test distance interactive features
        distance_features = visualizer._create_distance_interactive_features()
        assert 'distance_measurement' in distance_features['tools']
        assert 'ratio_calculator' in distance_features['tools']
        
        # Test sacred geometry interactive features
        sacred_features = visualizer._create_sacred_geometry_interactive_features()
        assert 'pattern_construction' in sacred_features['tools']
        assert 'construction_animation' in sacred_features['animations']
        
        # Test comprehensive interactive features
        comprehensive_features = visualizer._create_comprehensive_interactive_features()
        assert 'navigation' in comprehensive_features
        assert 'layers' in comprehensive_features
        assert 'measurements' in comprehensive_features
        assert 'export' in comprehensive_features
        assert 'analysis' in comprehensive_features


class TestGeometricVisualizationDataStructures:
    """Test the data structures used in geometric visualization"""
    
    def test_point_creation(self):
        """Test Point data structure"""
        point = Point(
            x=10.5, 
            y=20.3, 
            label="test_point", 
            character="A", 
            pattern_id=1, 
            confidence=0.8,
            metadata={"type": "corner"}
        )
        
        assert point.x == 10.5
        assert point.y == 20.3
        assert point.label == "test_point"
        assert point.character == "A"
        assert point.pattern_id == 1
        assert point.confidence == 0.8
        assert point.metadata["type"] == "corner"
    
    def test_geometric_element_creation(self):
        """Test GeometricElement data structure"""
        points = [Point(x=0, y=0), Point(x=1, y=1)]
        
        element = GeometricElement(
            element_type='line',
            points=points,
            measurements={'length': 1.414},
            properties={'is_diagonal': True},
            style={'stroke': '#FF0000', 'width': 2},
            annotations=['diagonal line']
        )
        
        assert element.element_type == 'line'
        assert len(element.points) == 2
        assert element.measurements['length'] == 1.414
        assert element.properties['is_diagonal'] is True
        assert element.style['stroke'] == '#FF0000'
        assert 'diagonal line' in element.annotations
    
    def test_angle_measurement_creation(self):
        """Test AngleMeasurement data structure"""
        vertex = Point(x=0, y=0)
        point1 = Point(x=1, y=0)
        point2 = Point(x=0, y=1)
        
        angle = AngleMeasurement(
            vertex=vertex,
            point1=point1,
            point2=point2,
            angle_degrees=90,
            angle_radians=math.pi/2,
            angle_type='right',
            significance=0.8,
            sacred_angle='Square'
        )
        
        assert angle.vertex == vertex
        assert angle.point1 == point1
        assert angle.point2 == point2
        assert angle.angle_degrees == 90
        assert angle.angle_radians == math.pi/2
        assert angle.angle_type == 'right'
        assert angle.significance == 0.8
        assert angle.sacred_angle == 'Square'
    
    def test_distance_measurement_creation(self):
        """Test DistanceMeasurement data structure"""
        point1 = Point(x=0, y=0)
        point2 = Point(x=3, y=4)
        
        distance = DistanceMeasurement(
            point1=point1,
            point2=point2,
            distance=5.0,
            distance_type='euclidean',
            ratio_to_golden=3.09,
            significance=0.7,
            pattern_context='triangle'
        )
        
        assert distance.point1 == point1
        assert distance.point2 == point2
        assert distance.distance == 5.0
        assert distance.distance_type == 'euclidean'
        assert distance.ratio_to_golden == 3.09
        assert distance.significance == 0.7
        assert distance.pattern_context == 'triangle'
    
    def test_geometric_visualization_creation(self):
        """Test GeometricVisualization data structure"""
        points = [Point(x=0, y=0), Point(x=1, y=1)]
        elements = [GeometricElement(
            element_type='point',
            points=[points[0]],
            measurements={},
            properties={}
        )]
        
        visualization = GeometricVisualization(
            visualization_id='test_viz_123',
            document_id=1,
            visualization_type=GeometricVisualizationType.COORDINATE_PLOT,
            elements=elements,
            coordinate_system={'bounds': {'min_x': 0, 'max_x': 10}},
            measurements_summary={'total_points': 2},
            sacred_geometry_patterns=[],
            interactive_features={'zoom': True},
            d3_config={'width': 800},
            export_data={'format': 'json'}
        )
        
        assert visualization.visualization_id == 'test_viz_123'
        assert visualization.document_id == 1
        assert visualization.visualization_type == GeometricVisualizationType.COORDINATE_PLOT
        assert len(visualization.elements) == 1
        assert visualization.coordinate_system['bounds']['min_x'] == 0
        assert visualization.measurements_summary['total_points'] == 2
        assert visualization.interactive_features['zoom'] is True
        assert visualization.d3_config['width'] == 800
        assert visualization.export_data['format'] == 'json'


if __name__ == '__main__':
    pytest.main([__file__])