"""
Tests for Interactive Text Grid Visualization System
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from sqlalchemy.orm import Session

from app.services.text_grid_visualizer import (
    TextGridVisualizer, VisualizationType, PatternHighlightStyle,
    GridCell, GridVisualization, VisualizationConfig
)
from app.models.database_models import Document, Grid


class TestTextGridVisualizer:
    """Test cases for TextGridVisualizer"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session"""
        return Mock(spec=Session)
    
    @pytest.fixture
    def visualizer(self, mock_db_session):
        """Create TextGridVisualizer instance with mock database"""
        return TextGridVisualizer(mock_db_session)
    
    @pytest.fixture
    def sample_grid(self):
        """Create sample grid for testing"""
        return Grid(
            id=1,
            document_id=1,
            rows=5,
            columns=5,
            grid_data=[['A', 'B', 'C', 'D', 'E'], ['F', 'G', 'H', 'I', 'J'], ['K', 'L', 'M', 'N', 'O'], ['P', 'Q', 'R', 'S', 'T'], ['U', 'V', 'W', 'X', 'Y']],
            source_text="ABCDEFGHIJKLMNOPQRSTUVWXY",
            created_at=datetime.now()
        )
    
    @pytest.fixture
    def sample_document(self):
        """Create sample document for testing"""
        return Document(
            id=1,
            filename="test_document.txt",
            original_filename="test_document.txt",
            file_path="/path/to/test_document.txt",
            file_size=1024,
            upload_date=datetime.now()
        )
    
    @pytest.fixture
    def sample_patterns(self):
        """Create sample patterns for testing"""
        return [
            {
                'id': 1,
                'type': 'cipher',
                'confidence': 0.85,
                'description': 'Caesar cipher pattern',
                'cells': [(0, 0), (0, 1), (0, 2)],
                'start_position': 0,
                'end_position': 2
            },
            {
                'id': 2,
                'type': 'geometric',
                'confidence': 0.92,
                'description': 'Triangular pattern',
                'cells': [(1, 1), (2, 2), (3, 3)],
                'start_position': 6,
                'end_position': 18
            }
        ]
    
    def test_visualizer_initialization(self, visualizer):
        """Test visualizer initialization"""
        assert visualizer is not None
        assert visualizer.default_config is not None
        assert isinstance(visualizer.pattern_colors, dict)
        assert isinstance(visualizer.css_classes, dict)
        assert len(visualizer.pattern_colors) > 0
    
    def test_visualization_config_defaults(self):
        """Test default visualization configuration"""
        config = VisualizationConfig()
        
        assert config.show_patterns is True
        assert config.show_annotations is True
        assert config.enable_zoom is True
        assert config.enable_selection is True
        assert config.enable_highlighting is True
        assert config.color_scheme == "default"
        assert config.cell_size == 30
        assert config.font_size == 12
        assert config.highlight_opacity == 0.7
        assert config.animation_enabled is True
        assert config.responsive_design is True
    
    def test_visualization_config_custom(self):
        """Test custom visualization configuration"""
        config = VisualizationConfig(
            show_patterns=False,
            cell_size=40,
            font_size=14,
            color_scheme="dark",
            enable_zoom=False
        )
        
        assert config.show_patterns is False
        assert config.cell_size == 40
        assert config.font_size == 14
        assert config.color_scheme == "dark"
        assert config.enable_zoom is False
    
    def test_grid_cell_creation(self):
        """Test GridCell dataclass creation"""
        cell = GridCell(
            row=1,
            col=2,
            character='A',
            original_position=5,
            is_pattern=True,
            pattern_ids=[1, 2],
            pattern_types=['cipher', 'geometric'],
            confidence_score=0.85,
            annotations=['Test annotation'],
            highlight_style=PatternHighlightStyle.BACKGROUND,
            css_classes=['grid-cell', 'pattern-cell']
        )
        
        assert cell.row == 1
        assert cell.col == 2
        assert cell.character == 'A'
        assert cell.original_position == 5
        assert cell.is_pattern is True
        assert cell.pattern_ids == [1, 2]
        assert cell.pattern_types == ['cipher', 'geometric']
        assert cell.confidence_score == 0.85
        assert cell.annotations == ['Test annotation']
        assert cell.highlight_style == PatternHighlightStyle.BACKGROUND
        assert 'grid-cell' in cell.css_classes
        assert 'pattern-cell' in cell.css_classes
    
    def test_create_interactive_grid_visualization(self, visualizer, mock_db_session, sample_grid):
        """Test creating interactive grid visualization"""
        # Mock database query
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_grid
        
        # Mock pattern retrieval
        with patch.object(visualizer, '_get_grid_patterns') as mock_get_patterns:
            mock_get_patterns.return_value = [
                {
                    'id': 1,
                    'type': 'cipher',
                    'confidence': 0.85,
                    'description': 'Test pattern',
                    'cells': [(0, 0), (0, 1)]
                }
            ]
            
            visualization = visualizer.create_interactive_grid_visualization(1)
            
            assert isinstance(visualization, GridVisualization)
            assert visualization.grid_id == 1
            assert visualization.document_id == 1
            assert visualization.grid_size == 5
            assert len(visualization.cells) == 5
            assert len(visualization.cells[0]) == 5
            assert len(visualization.patterns) == 1
            assert 'metadata' in visualization.__dict__
            assert 'interactive_features' in visualization.__dict__
    
    def test_create_pattern_overlay_visualization(self, visualizer, mock_db_session, sample_grid):
        """Test creating pattern overlay visualization"""
        # Mock database queries
        mock_db_session.query.return_value.filter.return_value.all.return_value = [sample_grid]
        
        # Mock pattern retrieval
        with patch.object(visualizer, '_get_grid_patterns') as mock_get_patterns:
            mock_get_patterns.return_value = [
                {
                    'id': 1,
                    'type': 'cipher',
                    'confidence': 0.85,
                    'description': 'Test pattern',
                    'cells': [(0, 0)]
                }
            ]
            
            visualizations = visualizer.create_pattern_overlay_visualization(1, ['cipher'])
            
            assert isinstance(visualizations, list)
            assert len(visualizations) == 1
            assert isinstance(visualizations[0], GridVisualization)
    
    def test_create_heatmap_visualization(self, visualizer, mock_db_session, sample_grid):
        """Test creating heatmap visualization"""
        # Mock database query
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_grid
        
        heatmap = visualizer.create_heatmap_visualization(1, "pattern_density")
        
        assert isinstance(heatmap, dict)
        assert heatmap['type'] == 'heatmap'
        assert heatmap['grid_id'] == 1
        assert heatmap['metric'] == 'pattern_density'
        assert 'data' in heatmap
        assert 'color_scale' in heatmap
        assert 'legend' in heatmap
        assert 'interactive_features' in heatmap
    
    def test_generate_character_frequency_chart(self, visualizer, mock_db_session, sample_document):
        """Test generating character frequency chart"""
        # Mock database query
        mock_db_session.query.return_value.filter.return_value.first.return_value = sample_document
        
        chart_data = visualizer.generate_character_frequency_chart(1)
        
        assert isinstance(chart_data, dict)
        assert chart_data['type'] == 'character_frequency'
        assert chart_data['document_id'] == 1
        assert 'data' in chart_data
        assert 'chart_config' in chart_data
        assert 'interactive_features' in chart_data
        assert 'statistical_info' in chart_data
        
        # Check data structure
        data = chart_data['data']
        assert 'labels' in data
        assert 'values' in data
        assert 'total_characters' in data
        assert len(data['labels']) == len(data['values'])
        
        # Check statistical info
        stats = chart_data['statistical_info']
        assert 'most_frequent' in stats
        assert 'least_frequent' in stats
        assert 'unique_characters' in stats
        assert 'entropy' in stats
    
    def test_create_pattern_timeline_visualization(self, visualizer, mock_db_session):
        """Test creating pattern timeline visualization"""
        # Mock pattern data
        mock_patterns = [
            Mock(
                id=1,
                pattern_type='cipher',
                coordinates=[{'x': 10, 'y': 5}, {'x': 20, 'y': 5}],
                confidence=0.85,
                description='Caesar cipher',
                detected_at=datetime.now()
            ),
            Mock(
                id=2,
                pattern_type='geometric',
                coordinates=[{'x': 30, 'y': 10}, {'x': 40, 'y': 15}],
                confidence=0.92,
                description='Triangle pattern',
                detected_at=datetime.now()
            )
        ]
        
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_patterns
        
        timeline = visualizer.create_pattern_timeline_visualization(1)
        
        assert isinstance(timeline, dict)
        assert timeline['type'] == 'pattern_timeline'
        assert timeline['document_id'] == 1
        assert 'data' in timeline
        assert 'visualization_config' in timeline
        assert 'pattern_statistics' in timeline
        
        # Check timeline data
        assert len(timeline['data']) == 2
        assert timeline['data'][0]['type'] == 'cipher'
        assert timeline['data'][1]['type'] == 'geometric'
        
        # Check statistics
        stats = timeline['pattern_statistics']
        assert stats['total_patterns'] == 2
        assert 'cipher' in stats['pattern_types']
        assert 'geometric' in stats['pattern_types']
    
    def test_generate_react_component_data(self, visualizer, sample_grid, sample_patterns):
        """Test generating React component data"""
        # Create a mock visualization
        cells = [
            [
                GridCell(0, 0, 'A', 0, True, [1], ['cipher'], 0.85, ['Test'], PatternHighlightStyle.BACKGROUND, ['grid-cell']),
                GridCell(0, 1, 'B', 1, False, [], [], 0.0, [], None, ['grid-cell'])
            ],
            [
                GridCell(1, 0, 'C', 2, False, [], [], 0.0, [], None, ['grid-cell']),
                GridCell(1, 1, 'D', 3, True, [2], ['geometric'], 0.92, ['Triangle'], PatternHighlightStyle.OUTLINE, ['grid-cell'])
            ]
        ]
        
        visualization = GridVisualization(
            grid_id=1,
            document_id=1,
            grid_size=2,
            cells=cells,
            patterns=sample_patterns,
            metadata={'test': 'metadata'},
            visualization_config={'cell_size': 30, 'font_size': 12},
            interactive_features={'selection': {'enabled': True}},
            export_data={'formats': ['png', 'svg']}
        )
        
        react_data = visualizer.generate_react_component_data(visualization)
        
        assert isinstance(react_data, dict)
        assert 'gridConfig' in react_data
        assert 'gridData' in react_data
        assert 'patterns' in react_data
        assert 'interactiveFeatures' in react_data
        
        # Check grid config
        grid_config = react_data['gridConfig']
        assert grid_config['gridId'] == 1
        assert grid_config['documentId'] == 1
        assert grid_config['gridSize'] == 2
        assert grid_config['cellSize'] == 30
        
        # Check grid data structure
        grid_data = react_data['gridData']
        assert len(grid_data['cells']) == 2
        assert len(grid_data['cells'][0]) == 2
        
        # Check first cell data
        first_cell = grid_data['cells'][0][0]
        assert first_cell['character'] == 'A'
        assert first_cell['isPattern'] is True
        assert first_cell['patternIds'] == [1]
        assert first_cell['confidence'] == 0.85
        
        # Check patterns data
        assert len(react_data['patterns']) == 2
        assert react_data['patterns'][0]['type'] == 'cipher'
        assert 'color' in react_data['patterns'][0]
    
    def test_create_zoom_navigation_config(self, visualizer):
        """Test creating zoom and navigation configuration"""
        # Test small grid
        small_config = visualizer.create_zoom_navigation_config(10)
        assert small_config['zoom']['min_zoom'] == 0.5
        assert small_config['zoom']['max_zoom'] == 3.0
        assert small_config['zoom']['default_zoom'] == 1.0
        assert small_config['minimap']['enabled'] is False
        
        # Test large grid
        large_config = visualizer.create_zoom_navigation_config(25)
        assert large_config['zoom']['min_zoom'] == 0.2
        assert large_config['zoom']['max_zoom'] == 1.5
        assert large_config['zoom']['default_zoom'] == 0.6
        assert large_config['minimap']['enabled'] is True
        
        # Check navigation features
        assert large_config['navigation']['enabled'] is True
        assert large_config['navigation']['pan_enabled'] is True
        assert 'keyboard_shortcuts' in large_config['navigation']
        assert 'mouse_controls' in large_config['navigation']
    
    def test_calculate_character_frequencies(self, visualizer):
        """Test character frequency calculation"""
        text = "AAABBC"
        frequencies = visualizer._calculate_character_frequencies(text)
        
        expected = {'A': 3/6, 'B': 2/6, 'C': 1/6}
        assert frequencies == expected
        
        # Test empty text
        empty_frequencies = visualizer._calculate_character_frequencies("")
        assert empty_frequencies == {}
    
    def test_calculate_entropy(self, visualizer):
        """Test entropy calculation"""
        # Test uniform distribution
        uniform_freq = {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25}
        entropy = visualizer._calculate_entropy(uniform_freq)
        assert abs(entropy - 2.0) < 0.001  # log2(4) = 2
        
        # Test single character
        single_freq = {'A': 1.0}
        entropy = visualizer._calculate_entropy(single_freq)
        assert entropy == 0.0
    
    def test_get_heatmap_color_scale(self, visualizer):
        """Test heatmap color scale generation"""
        # Test pattern density scale
        pattern_scale = visualizer._get_heatmap_color_scale('pattern_density')
        assert pattern_scale['type'] == 'sequential'
        assert len(pattern_scale['colors']) == 2
        assert pattern_scale['domain'] == [0, 1]
        
        # Test confidence scale
        confidence_scale = visualizer._get_heatmap_color_scale('confidence')
        assert confidence_scale['type'] == 'sequential'
        assert confidence_scale['domain'] == [0, 1]
        
        # Test unknown metric (should return default)
        unknown_scale = visualizer._get_heatmap_color_scale('unknown_metric')
        assert unknown_scale['type'] == 'sequential'
    
    def test_create_heatmap_legend(self, visualizer):
        """Test heatmap legend creation"""
        # Test pattern density legend
        pattern_legend = visualizer._create_heatmap_legend('pattern_density')
        assert pattern_legend['title'] == 'Pattern Density'
        assert pattern_legend['min_label'] == 'Low'
        assert pattern_legend['max_label'] == 'High'
        assert pattern_legend['format'] == 'percentage'
        
        # Test confidence legend
        confidence_legend = visualizer._create_heatmap_legend('confidence')
        assert confidence_legend['title'] == 'Confidence Score'
        assert confidence_legend['min_label'] == '0%'
        assert confidence_legend['max_label'] == '100%'
    
    def test_calculate_zoom_levels(self, visualizer):
        """Test zoom level calculation"""
        # Test small grid
        small_zoom = visualizer._calculate_zoom_levels(5)
        assert small_zoom['min'] == 0.5
        assert small_zoom['max'] == 3.0
        assert small_zoom['default'] == 1.0
        
        # Test medium grid
        medium_zoom = visualizer._calculate_zoom_levels(15)
        assert medium_zoom['min'] == 0.3
        assert medium_zoom['max'] == 2.0
        assert medium_zoom['default'] == 0.8
        
        # Test large grid
        large_zoom = visualizer._calculate_zoom_levels(25)
        assert large_zoom['min'] == 0.2
        assert large_zoom['max'] == 1.5
        assert large_zoom['default'] == 0.6
        
        # Test very large grid
        very_large_zoom = visualizer._calculate_zoom_levels(50)
        assert very_large_zoom['min'] == 0.1
        assert very_large_zoom['max'] == 1.0
        assert very_large_zoom['default'] == 0.4
    
    def test_pattern_highlight_styles(self):
        """Test pattern highlight style enumeration"""
        styles = list(PatternHighlightStyle)
        expected_styles = ['outline', 'fill', 'underline', 'background', 'border']
        
        style_values = [style.value for style in styles]
        for expected_style in expected_styles:
            assert expected_style in style_values
    
    def test_visualization_types(self):
        """Test visualization type enumeration"""
        types = list(VisualizationType)
        expected_types = ['basic_grid', 'pattern_overlay', 'heatmap', 'interactive_grid', 'annotated_grid']
        
        type_values = [vtype.value for vtype in types]
        for expected_type in expected_types:
            assert expected_type in type_values
    
    def test_error_handling_missing_grid(self, visualizer, mock_db_session):
        """Test error handling when grid is not found"""
        # Mock database to return None
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        with pytest.raises(ValueError, match="Grid 999 not found"):
            visualizer.create_interactive_grid_visualization(999)
    
    def test_error_handling_missing_document(self, visualizer, mock_db_session):
        """Test error handling when document is not found"""
        # Mock database to return None
        mock_db_session.query.return_value.filter.return_value.first.return_value = None
        
        chart_data = visualizer.generate_character_frequency_chart(999)
        assert chart_data == {}
    
    def test_pattern_color_mapping(self, visualizer):
        """Test pattern color mapping"""
        assert 'cipher' in visualizer.pattern_colors
        assert 'geometric' in visualizer.pattern_colors
        assert 'linguistic' in visualizer.pattern_colors
        assert 'structural' in visualizer.pattern_colors
        assert 'mathematical' in visualizer.pattern_colors
        assert 'anomaly' in visualizer.pattern_colors
        assert 'cross_reference' in visualizer.pattern_colors
        
        # Check that colors are valid hex codes
        for color in visualizer.pattern_colors.values():
            assert color.startswith('#')
            assert len(color) == 7
    
    def test_css_class_mapping(self, visualizer):
        """Test CSS class mapping"""
        expected_classes = [
            'pattern_cell', 'highlighted_cell', 'selected_cell', 'annotated_cell',
            'high_confidence', 'medium_confidence', 'low_confidence'
        ]
        
        for class_name in expected_classes:
            assert class_name in visualizer.css_classes
            assert isinstance(visualizer.css_classes[class_name], str)


if __name__ == "__main__":
    pytest.main([__file__])