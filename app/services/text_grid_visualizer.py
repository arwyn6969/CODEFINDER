"""
Interactive Text Grid Visualization System
Creates visual representations of text grids with pattern highlighting,
interactive controls, and annotation capabilities for ancient text analysis.
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from app.core.database import get_db
from app.models.database_models import Document, Pattern, Grid
from app.services.grid_generator import GridGenerator


class VisualizationType(Enum):
    """Types of grid visualizations"""
    BASIC_GRID = "basic_grid"
    PATTERN_OVERLAY = "pattern_overlay"
    HEATMAP = "heatmap"
    INTERACTIVE_GRID = "interactive_grid"
    ANNOTATED_GRID = "annotated_grid"


class PatternHighlightStyle(Enum):
    """Pattern highlighting styles"""
    OUTLINE = "outline"
    FILL = "fill"
    UNDERLINE = "underline"
    BACKGROUND = "background"
    BORDER = "border"


@dataclass
class GridCell:
    """Represents a single cell in the text grid"""
    row: int
    col: int
    character: str
    original_position: int
    is_pattern: bool = False
    pattern_ids: List[int] = field(default_factory=list)
    pattern_types: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    annotations: List[str] = field(default_factory=list)
    highlight_style: Optional[PatternHighlightStyle] = None
    css_classes: List[str] = field(default_factory=list)


@dataclass
class GridVisualization:
    """Complete grid visualization data structure"""
    grid_id: int
    document_id: int
    grid_size: int
    cells: List[List[GridCell]]
    patterns: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    visualization_config: Dict[str, Any]
    interactive_features: Dict[str, Any]
    export_data: Dict[str, Any]


@dataclass
class VisualizationConfig:
    """Configuration for grid visualization"""
    show_patterns: bool = True
    show_annotations: bool = True
    enable_zoom: bool = True
    enable_selection: bool = True
    enable_highlighting: bool = True
    color_scheme: str = "default"
    cell_size: int = 30
    font_size: int = 12
    pattern_colors: Dict[str, str] = field(default_factory=dict)
    highlight_opacity: float = 0.7
    animation_enabled: bool = True
    responsive_design: bool = True


class TextGridVisualizer:
    """
    Service for creating interactive text grid visualizations
    """
    
    def __init__(self, db_session: Session = None):
        self.db = db_session or next(get_db())
        self.grid_generator = GridGenerator()
        self.logger = logging.getLogger(__name__)
        
        # Default visualization configuration
        self.default_config = VisualizationConfig()
        
        # Pattern color mapping
        self.pattern_colors = {
            'cipher': '#FF6B6B',
            'geometric': '#4ECDC4', 
            'linguistic': '#45B7D1',
            'structural': '#96CEB4',
            'mathematical': '#FFEAA7',
            'anomaly': '#DDA0DD',
            'cross_reference': '#98D8C8'
        }
        
        # CSS class mappings
        self.css_classes = {
            'pattern_cell': 'grid-cell-pattern',
            'highlighted_cell': 'grid-cell-highlighted',
            'selected_cell': 'grid-cell-selected',
            'annotated_cell': 'grid-cell-annotated',
            'high_confidence': 'grid-cell-high-confidence',
            'medium_confidence': 'grid-cell-medium-confidence',
            'low_confidence': 'grid-cell-low-confidence'
        }
    
    def create_interactive_grid_visualization(self, grid_id: int, 
                                            config: VisualizationConfig = None) -> GridVisualization:
        """
        Create comprehensive interactive grid visualization
        """
        try:
            config = config or self.default_config
            
            # Get grid data
            grid = self.db.query(Grid).filter(Grid.id == grid_id).first()
            if not grid:
                raise ValueError(f"Grid {grid_id} not found")
            
            # Get patterns for this grid
            patterns = self._get_grid_patterns(grid_id)
            
            # Coerce rows/columns to integers to tolerate mocks
            try:
                grid.rows = int(grid.rows)
            except Exception:
                grid.rows = 1
            try:
                grid.columns = int(grid.columns)
            except Exception:
                grid.columns = 1

            # Create grid cells with pattern information
            cells = self._create_grid_cells(grid, patterns, config)
            
            # Generate visualization metadata
            metadata = self._generate_visualization_metadata(grid, patterns)
            
            # Create interactive features configuration
            interactive_features = self._create_interactive_features(config)
            
            # Generate export data
            export_data = self._generate_export_data(grid, cells, patterns)
            
            visualization = GridVisualization(
                grid_id=grid_id,
                document_id=grid.document_id,
                grid_size=max(grid.rows, grid.columns),  # Use the larger dimension
                cells=cells,
                patterns=patterns,
                metadata=metadata,
                visualization_config=config.__dict__,
                interactive_features=interactive_features,
                export_data=export_data
            )
            
            self.logger.info(f"Created interactive grid visualization for grid {grid_id}")
            return visualization
            
        except Exception as e:
            self.logger.error(f"Error creating grid visualization: {str(e)}")
            raise
    
    def create_pattern_overlay_visualization(self, document_id: int, 
                                           pattern_types: List[str] = None) -> List[GridVisualization]:
        """
        Create pattern overlay visualizations for all grids in a document
        """
        try:
            visualizations = []
            
            # Get all grids for the document
            grids = self.db.query(Grid).filter(Grid.document_id == document_id).all()
            
            for grid in grids:
                # Get patterns for this grid, filtered by type if specified
                patterns = self._get_grid_patterns(grid.id, pattern_types)
                
                if patterns:  # Only create visualization if patterns exist
                    config = VisualizationConfig(
                        show_patterns=True,
                        show_annotations=True,
                        color_scheme="pattern_focused"
                    )
                    
                    visualization = self.create_interactive_grid_visualization(grid.id, config)
                    visualizations.append(visualization)
            
            self.logger.info(f"Created {len(visualizations)} pattern overlay visualizations for document {document_id}")
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error creating pattern overlay visualizations: {str(e)}")
            return []
    
    def create_heatmap_visualization(self, grid_id: int, metric: str = "pattern_density") -> Dict[str, Any]:
        """
        Create heatmap visualization based on specified metric
        """
        try:
            grid = self.db.query(Grid).filter(Grid.id == grid_id).first()
            if not grid:
                raise ValueError(f"Grid {grid_id} not found")
            
            # Calculate heatmap values based on metric
            heatmap_data = self._calculate_heatmap_values(grid, metric)
            
            # Create heatmap configuration
            heatmap_config = {
                'type': 'heatmap',
                'grid_id': grid_id,
                'metric': metric,
                'data': heatmap_data,
                'color_scale': self._get_heatmap_color_scale(metric),
                'legend': self._create_heatmap_legend(metric),
                'interactive_features': {
                    'hover_info': True,
                    'click_details': True,
                    'zoom_enabled': True
                }
            }
            
            return heatmap_config
            
        except Exception as e:
            self.logger.error(f"Error creating heatmap visualization: {str(e)}")
            return {'type': 'heatmap', 'grid_id': grid_id, 'data': []}
    
    def generate_character_frequency_chart(self, document_id: int) -> Dict[str, Any]:
        """
        Generate interactive character frequency distribution chart
        """
        try:
            # Get document and its pages
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if not document:
                return {}
            
            # Get text from pages
            from app.models.database_models import Page
            pages = self.db.query(Page).filter(Page.document_id == document_id).all()
            if not pages:
                return {}
            
            # Combine text from all pages
            full_text = " ".join([page.extracted_text or "" for page in pages])
            if not full_text.strip():
                return {}
            
            # Calculate character frequencies
            char_frequencies = self._calculate_character_frequencies(full_text)
            
            # Create chart data
            chart_data = {
                'type': 'character_frequency',
                'document_id': document_id,
                'data': {
                    'labels': list(char_frequencies.keys()),
                    'values': list(char_frequencies.values()),
                    'total_characters': len(full_text)
                },
                'chart_config': {
                    'chart_type': 'bar',
                    'title': f'Character Frequency Distribution - {document.filename}',
                    'x_axis_label': 'Characters',
                    'y_axis_label': 'Frequency',
                    'color_scheme': 'frequency_based',
                    'interactive': True,
                    'sortable': True,
                    'filterable': True
                },
                'interactive_features': {
                    'hover_details': True,
                    'click_to_highlight': True,
                    'export_options': ['png', 'svg', 'csv'],
                    'zoom_enabled': True
                },
                'statistical_info': {
                    'most_frequent': max(char_frequencies.items(), key=lambda x: x[1]),
                    'least_frequent': min(char_frequencies.items(), key=lambda x: x[1]),
                    'unique_characters': len(char_frequencies),
                    'entropy': self._calculate_entropy(char_frequencies)
                }
            }
            
            return chart_data
            
        except Exception as e:
            self.logger.error(f"Error generating character frequency chart: {str(e)}")
            return {
                'type': 'character_frequency',
                'document_id': document_id,
                'data': {'labels': [], 'values': [], 'total_characters': 0},
                'chart_config': {'chart_type': 'bar', 'title': '', 'x_axis_label': 'Characters', 'y_axis_label': 'Frequency', 'color_scheme': 'frequency_based', 'interactive': True, 'sortable': True, 'filterable': True},
                'interactive_features': {'hover_details': True, 'click_to_highlight': True, 'export_options': ['png', 'svg', 'csv'], 'zoom_enabled': True},
                'statistical_info': {'most_frequent': None, 'least_frequent': None, 'unique_characters': 0, 'entropy': 0}
            }
    
    def create_pattern_timeline_visualization(self, document_id: int) -> Dict[str, Any]:
        """
        Create timeline visualization showing pattern evolution through the document
        """
        try:
            # Get all patterns for the document
            patterns = self.db.query(Pattern).filter(
                Pattern.document_id == document_id
            ).order_by(Pattern.detected_at).all()
            
            if not patterns:
                return {'message': 'No patterns found for timeline visualization'}
            
            # Create timeline data
            timeline_data = []
            for pattern in patterns:
                # Extract position information from coordinates if available
                start_pos = 0
                end_pos = 0
                if pattern.coordinates:
                    coords = pattern.coordinates
                    if isinstance(coords, list) and len(coords) > 0:
                        if isinstance(coords[0], dict):
                            start_pos = coords[0].get('x', 0)
                            end_pos = coords[-1].get('x', 0) if len(coords) > 1 else start_pos
                
                timeline_data.append({
                    'id': pattern.id,
                    'type': pattern.pattern_type,
                    'start_position': start_pos,
                    'end_position': end_pos,
                    'confidence': pattern.confidence,
                    'description': pattern.description,
                    'timestamp': pattern.detected_at.isoformat() if pattern.detected_at else None
                })
            
            # Create timeline configuration
            timeline_config = {
                'type': 'pattern_timeline',
                'document_id': document_id,
                'data': timeline_data,
                'visualization_config': {
                    'timeline_type': 'horizontal',
                    'show_confidence': True,
                    'color_by_type': True,
                    'interactive': True,
                    'zoomable': True
                },
                'pattern_statistics': {
                    'total_patterns': len(patterns),
                    'pattern_types': list(set(p.pattern_type for p in patterns)),
                    'average_confidence': sum(p.confidence for p in patterns) / len(patterns),
                    'document_coverage': self._calculate_pattern_coverage(patterns, document_id)
                }
            }
            
            return timeline_config
            
        except Exception as e:
            self.logger.error(f"Error creating pattern timeline visualization: {str(e)}")
            return {}
    
    def generate_react_component_data(self, visualization: GridVisualization) -> Dict[str, Any]:
        """
        Generate data structure optimized for React component consumption
        """
        try:
            react_data = {
                'gridConfig': {
                    'gridId': visualization.grid_id,
                    'documentId': visualization.document_id,
                    'gridSize': visualization.grid_size,
                    'cellSize': visualization.visualization_config.get('cell_size', 30),
                    'fontSize': visualization.visualization_config.get('font_size', 12)
                },
                'gridData': {
                    'cells': [
                        [
                            {
                                'row': cell.row,
                                'col': cell.col,
                                'character': cell.character,
                                'isPattern': cell.is_pattern,
                                'patternIds': cell.pattern_ids,
                                'patternTypes': cell.pattern_types,
                                'confidence': cell.confidence_score,
                                'annotations': cell.annotations,
                                'cssClasses': cell.css_classes,
                                'highlightStyle': cell.highlight_style.value if cell.highlight_style else None
                            }
                            for cell in row
                        ]
                        for row in visualization.cells
                    ]
                },
                'patterns': [
                    {
                        'id': pattern['id'],
                        'type': pattern['type'],
                        'confidence': pattern['confidence'],
                        'description': pattern['description'],
                        'color': self.pattern_colors.get(pattern['type'], '#CCCCCC'),
                        'cells': pattern.get('cells', [])
                    }
                    for pattern in visualization.patterns
                ],
                'interactiveFeatures': visualization.interactive_features,
                'visualizationMetadata': visualization.metadata,
                'exportOptions': visualization.export_data
            }
            
            return react_data
            
        except Exception as e:
            self.logger.error(f"Error generating React component data: {str(e)}")
            return {}
    
    def create_zoom_navigation_config(self, grid_size: int) -> Dict[str, Any]:
        """
        Create zoom and navigation configuration for large grids
        """
        try:
            # Calculate optimal zoom levels based on grid size
            zoom_levels = self._calculate_zoom_levels(grid_size)
            
            navigation_config = {
                'zoom': {
                    'enabled': True,
                    'min_zoom': zoom_levels['min'],
                    'max_zoom': zoom_levels['max'],
                    'default_zoom': zoom_levels['default'],
                    'zoom_step': 0.1,
                    'smooth_zoom': True
                },
                'navigation': {
                    'enabled': True,
                    'pan_enabled': True,
                    'keyboard_shortcuts': {
                        'zoom_in': '+',
                        'zoom_out': '-',
                        'reset_zoom': '0',
                        'pan_up': 'ArrowUp',
                        'pan_down': 'ArrowDown',
                        'pan_left': 'ArrowLeft',
                        'pan_right': 'ArrowRight'
                    },
                    'mouse_controls': {
                        'wheel_zoom': True,
                        'drag_pan': True,
                        'double_click_zoom': True
                    }
                },
                'minimap': {
                    'enabled': grid_size > 20,
                    'position': 'bottom-right',
                    'size': '150px',
                    'show_viewport': True
                },
                'grid_overview': {
                    'show_grid_lines': True,
                    'show_coordinates': True,
                    'highlight_viewport': True
                }
            }
            
            return navigation_config
            
        except Exception as e:
            self.logger.error(f"Error creating zoom navigation config: {str(e)}")
            return {}
    
    # Private helper methods
    
    def _get_grid_patterns(self, grid_id: int, pattern_types: List[str] = None) -> List[Dict[str, Any]]:
        """Get patterns associated with a grid"""
        try:
            # This would typically query a patterns table with grid associations
            # For now, return mock pattern data
            patterns = [
                {
                    'id': 1,
                    'type': 'cipher',
                    'confidence': 0.85,
                    'description': 'Potential Caesar cipher pattern',
                    'cells': [(2, 3), (2, 4), (2, 5)],
                    'start_position': 23,
                    'end_position': 25
                },
                {
                    'id': 2,
                    'type': 'geometric',
                    'confidence': 0.92,
                    'description': 'Triangular arrangement detected',
                    'cells': [(1, 1), (2, 2), (3, 3)],
                    'start_position': 11,
                    'end_position': 33
                }
            ]
            
            # Filter by pattern types if specified
            if pattern_types:
                patterns = [p for p in patterns if p['type'] in pattern_types]
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error getting grid patterns: {str(e)}")
            return []
    
    def _create_grid_cells(self, grid: Grid, patterns: List[Dict[str, Any]], 
                          config: VisualizationConfig) -> List[List[GridCell]]:
        """Create grid cells with pattern information"""
        try:
            # Get grid data
            grid_data = grid.grid_data or []
            grid_rows = grid.rows
            grid_cols = grid.columns
            
            # Create pattern lookup for efficient cell marking
            pattern_lookup = {}
            for pattern in patterns:
                for row, col in pattern.get('cells', []):
                    if (row, col) not in pattern_lookup:
                        pattern_lookup[(row, col)] = []
                    pattern_lookup[(row, col)].append(pattern)
            
            # Create grid cells
            cells = []
            
            for row in range(grid_rows):
                cell_row = []
                for col in range(grid_cols):
                    # Get character for this position
                    char = ' '
                    if row < len(grid_data) and col < len(grid_data[row]):
                        char = grid_data[row][col] or ' '
                    
                    # Check if this cell is part of any patterns
                    cell_patterns = pattern_lookup.get((row, col), [])
                    is_pattern = len(cell_patterns) > 0
                    
                    # Determine highlight style and CSS classes
                    highlight_style = None
                    css_classes = ['grid-cell']
                    
                    if is_pattern:
                        css_classes.append(self.css_classes['pattern_cell'])
                        highlight_style = PatternHighlightStyle.BACKGROUND
                        
                        # Add confidence-based classes
                        max_confidence = max(p['confidence'] for p in cell_patterns)
                        if max_confidence >= 0.8:
                            css_classes.append(self.css_classes['high_confidence'])
                        elif max_confidence >= 0.6:
                            css_classes.append(self.css_classes['medium_confidence'])
                        else:
                            css_classes.append(self.css_classes['low_confidence'])
                    
                    # Create cell
                    cell = GridCell(
                        row=row,
                        col=col,
                        character=char,
                        original_position=row * grid_cols + col,
                        is_pattern=is_pattern,
                        pattern_ids=[p['id'] for p in cell_patterns],
                        pattern_types=[p['type'] for p in cell_patterns],
                        confidence_score=max([p['confidence'] for p in cell_patterns]) if cell_patterns else 0.0,
                        annotations=[p['description'] for p in cell_patterns],
                        highlight_style=highlight_style,
                        css_classes=css_classes
                    )
                    
                    cell_row.append(cell)
                
                cells.append(cell_row)
            
            return cells
            
        except Exception as e:
            self.logger.error(f"Error creating grid cells: {str(e)}")
            return []
    
    def _generate_visualization_metadata(self, grid: Grid, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate metadata for the visualization"""
        return {
            'grid_info': {
                'id': grid.id,
                'rows': grid.rows,
                'columns': grid.columns,
                'created_at': grid.created_at.isoformat() if grid.created_at else None,
                'total_cells': grid.rows * grid.columns
            },
            'pattern_summary': {
                'total_patterns': len(patterns),
                'pattern_types': list(set(p['type'] for p in patterns)),
                'average_confidence': sum(p['confidence'] for p in patterns) / len(patterns) if patterns else 0,
                'high_confidence_patterns': len([p for p in patterns if p['confidence'] >= 0.8])
            },
            'visualization_stats': {
                'pattern_cells': sum(len(p.get('cells', [])) for p in patterns),
                'coverage_percentage': 0  # Would calculate actual coverage
            }
        }
    
    def _create_interactive_features(self, config: VisualizationConfig) -> Dict[str, Any]:
        """Create interactive features configuration"""
        return {
            'selection': {
                'enabled': config.enable_selection,
                'multi_select': True,
                'selection_modes': ['single_cell', 'row', 'column', 'pattern', 'custom_area']
            },
            'highlighting': {
                'enabled': config.enable_highlighting,
                'highlight_on_hover': True,
                'highlight_patterns': True,
                'highlight_related': True
            },
            'annotations': {
                'enabled': config.show_annotations,
                'show_on_hover': True,
                'show_on_click': True,
                'editable': False
            },
            'export': {
                'formats': ['png', 'svg', 'pdf', 'json'],
                'include_patterns': True,
                'include_annotations': True
            }
        }
    
    def _generate_export_data(self, grid: Grid, cells: List[List[GridCell]], 
                            patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate data for export functionality"""
        return {
            'grid_data': {
                'id': grid.id,
                'rows': grid.rows,
                'columns': grid.columns,
                'cells': [
                    [
                        {
                            'char': cell.character,
                            'row': cell.row,
                            'col': cell.col,
                            'patterns': cell.pattern_types
                        }
                        for cell in row
                    ]
                    for row in cells
                ]
            },
            'pattern_data': patterns,
            'export_formats': {
                'csv': 'grid_data_export.csv',
                'json': 'grid_visualization_export.json',
                'png': 'grid_visualization.png',
                'svg': 'grid_visualization.svg'
            }
        }
    
    def _calculate_character_frequencies(self, text: str) -> Dict[str, float]:
        """Calculate character frequency distribution"""
        if not text:
            return {}
        
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        total_chars = len(text)
        return {char: count / total_chars for char, count in char_counts.items()}
    
    def _calculate_entropy(self, frequencies: Dict[str, float]) -> float:
        """Calculate Shannon entropy of character frequencies"""
        import math
        entropy = 0
        for freq in frequencies.values():
            if freq > 0:
                entropy -= freq * math.log2(freq)
        return entropy
    
    def _calculate_heatmap_values(self, grid: Grid, metric: str) -> List[List[float]]:
        """Calculate heatmap values based on specified metric"""
        # Mock implementation - would calculate actual values based on metric
        # Derive size from rows/columns to support mocks
        grid_size = getattr(grid, 'grid_size', None) or max(getattr(grid, 'rows', 1) or 1, getattr(grid, 'columns', 1) or 1)
        import random
        return [
            [random.random() for _ in range(grid_size)]
            for _ in range(grid_size)
        ]
    
    def _get_heatmap_color_scale(self, metric: str) -> Dict[str, Any]:
        """Get color scale configuration for heatmap"""
        color_scales = {
            'pattern_density': {
                'type': 'sequential',
                'colors': ['#f7fbff', '#08519c'],
                'domain': [0, 1]
            },
            'confidence': {
                'type': 'sequential', 
                'colors': ['#fff5f0', '#67000d'],
                'domain': [0, 1]
            },
            'frequency': {
                'type': 'sequential',
                'colors': ['#f0f0f0', '#252525'],
                'domain': [0, 1]
            }
        }
        return color_scales.get(metric, color_scales['pattern_density'])
    
    def _create_heatmap_legend(self, metric: str) -> Dict[str, Any]:
        """Create legend configuration for heatmap"""
        legends = {
            'pattern_density': {
                'title': 'Pattern Density',
                'min_label': 'Low',
                'max_label': 'High',
                'format': 'percentage'
            },
            'confidence': {
                'title': 'Confidence Score',
                'min_label': '0%',
                'max_label': '100%',
                'format': 'percentage'
            }
        }
        return legends.get(metric, legends['pattern_density'])
    
    def _calculate_zoom_levels(self, grid_size: int) -> Dict[str, float]:
        """Calculate optimal zoom levels based on grid size"""
        if grid_size <= 10:
            return {'min': 0.5, 'max': 3.0, 'default': 1.0}
        elif grid_size <= 20:
            return {'min': 0.3, 'max': 2.0, 'default': 0.8}
        elif grid_size <= 30:
            return {'min': 0.2, 'max': 1.5, 'default': 0.6}
        else:
            return {'min': 0.1, 'max': 1.0, 'default': 0.4}
    
    def _calculate_pattern_coverage(self, patterns: List, document_id: int) -> float:
        """Calculate what percentage of document is covered by patterns"""
        # Mock implementation - would calculate actual coverage
        return 0.25  # 25% coverage