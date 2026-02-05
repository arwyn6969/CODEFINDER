"""
Geometric Visualization Adapter
Bridges the output of geometric_pipeline.py to D3.js rendering.
Uses lightweight standalone types to avoid heavy import chains.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import json

from app.services.geometric_pipeline import GeometricAnalysisResult, GeometricPattern


# Standalone lightweight types (mirror the visualizer types without heavy imports)
@dataclass
class Point:
    """2D point with metadata."""
    x: float
    y: float
    label: Optional[str] = None
    character: Optional[str] = None
    pattern_id: Optional[int] = None
    confidence: float = 1.0


@dataclass
class GeometricElement:
    """Geometric element for visualization."""
    element_type: str
    points: List[Point]
    measurements: Dict[str, float]
    properties: Dict[str, Any]
    style: Dict[str, Any] = field(default_factory=dict)
    annotations: List[str] = field(default_factory=list)


@dataclass
class GeometricVisualization:
    """Complete visualization data structure."""
    visualization_id: str
    document_id: int
    visualization_type: str
    elements: List[GeometricElement]
    coordinate_system: Dict[str, Any]
    measurements_summary: Dict[str, Any]
    sacred_geometry_patterns: List[Dict[str, Any]]
    interactive_features: Dict[str, Any]
    d3_config: Dict[str, Any]
    export_data: Dict[str, Any]


# Color scheme for pattern types
PATTERN_COLORS = {
    'golden_ratio': '#FFD700',  # Gold
    'right_angle': '#DC143C',    # Crimson
    'isosceles': '#4169E1',      # Royal Blue
    'fibonacci': '#FF8C00',      # Dark Orange
}


def convert_result_to_visualization(
    result: GeometricAnalysisResult
) -> GeometricVisualization:
    """
    Convert a GeometricAnalysisResult to a GeometricVisualization.
    
    Args:
        result: Output from analyze_page_geometry()
    
    Returns:
        GeometricVisualization ready for D3.js rendering
    """
    elements = []
    
    for pattern in result.patterns_found:
        element = _pattern_to_element(pattern)
        elements.append(element)
    
    # Create coordinate system based on page dimensions
    coordinate_system = {
        'width': result.page_width,
        'height': result.page_height,
        'origin': {'x': 0, 'y': 0},
        'scale': 1.0,
        'unit': 'pixels'
    }
    
    # Generate summary statistics
    measurements_summary = {
        'total_patterns': len(result.patterns_found),
        'golden_ratio_count': len([p for p in result.patterns_found if p.pattern_type == 'golden_ratio']),
        'right_angle_count': len([p for p in result.patterns_found if p.pattern_type == 'right_angle']),
        'avg_significance': (
            sum(p.significance_score for p in result.patterns_found) / len(result.patterns_found)
            if result.patterns_found else 0.0
        ),
        'characters_analyzed': result.significant_characters
    }
    
    # Create D3.js configuration
    d3_config = {
        'width': 800,
        'height': int(800 * (result.page_height / result.page_width)) if result.page_width > 0 else 1000,
        'margin': {'top': 40, 'right': 40, 'bottom': 40, 'left': 40},
        'zoom': {'min': 0.5, 'max': 5.0},
        'tooltip': True,
        'legend': True,
        'background_image': None  # Will be set by generate_visualization_with_background
    }
    
    # Interactive features
    interactive_features = {
        'pan': True,
        'zoom': True,
        'hover_highlight': True,
        'click_select': True,
        'filter_by_type': True,
        'filter_by_significance': True
    }
    
    return GeometricVisualization(
        visualization_id=f"geo_{result.document_id}_{result.page_number}_{int(datetime.now().timestamp())}",
        document_id=result.document_id,
        visualization_type=GeometricVisualizationType.PATTERN_OVERLAY,
        elements=elements,
        coordinate_system=coordinate_system,
        measurements_summary=measurements_summary,
        sacred_geometry_patterns=[],  # Could be populated from right angles, etc.
        interactive_features=interactive_features,
        d3_config=d3_config,
        export_data={
            'page_number': result.page_number,
            'total_characters': result.total_characters,
            'patterns': [_pattern_to_dict(p) for p in result.patterns_found]
        }
    )


def _pattern_to_element(pattern: GeometricPattern) -> GeometricElement:
    """Convert a GeometricPattern to a GeometricElement."""
    
    # Build Point objects from positions
    points = []
    for pos in pattern.positions:
        points.append(Point(
            x=pos.x + pos.width / 2,  # Use center point
            y=pos.y + pos.height / 2,
            label=pos.character,
            character=pos.character,
            pattern_id=pos.id,
            confidence=pos.confidence
        ))
    
    # Determine element type and styling
    if pattern.pattern_type == 'golden_ratio':
        element_type = 'line'
        stroke = PATTERN_COLORS['golden_ratio']
        stroke_width = 2 if pattern.significance_score > 0.8 else 1.5
    elif pattern.pattern_type == 'right_angle':
        element_type = 'triangle'
        stroke = PATTERN_COLORS['right_angle']
        stroke_width = 2 if pattern.significance_score > 0.8 else 1.5
    else:
        element_type = 'polygon'
        stroke = '#666666'
        stroke_width = 1
    
    return GeometricElement(
        element_type=element_type,
        points=points,
        measurements={
            'value': pattern.measurement_value,
            'significance': pattern.significance_score
        },
        properties={
            'pattern_type': pattern.pattern_type,
            'is_significant': pattern.significance_score > 0.8,
            **pattern.metadata
        },
        style={
            'stroke': stroke,
            'stroke_width': stroke_width,
            'fill': 'none' if element_type == 'line' else f"{stroke}22",  # Light fill for triangles
            'opacity': 0.7 + (0.3 * pattern.significance_score)
        },
        annotations=[f"{pattern.pattern_type}: {pattern.measurement_value:.2f}"]
    )


def _pattern_to_dict(pattern: GeometricPattern) -> Dict[str, Any]:
    """Convert pattern to exportable dict."""
    return {
        'type': pattern.pattern_type,
        'value': pattern.measurement_value,
        'significance': pattern.significance_score,
        'characters': [p.character for p in pattern.positions],
        'metadata': pattern.metadata
    }


def generate_visualization_with_background(
    result: GeometricAnalysisResult,
    page_image_url: str = None,
    page_image_opacity: float = 0.3
) -> GeometricVisualization:
    """
    Generate a visualization with a page image as background.
    
    Args:
        result: Output from analyze_page_geometry()
        page_image_url: URL or path to the page image (for rendering behind patterns)
        page_image_opacity: Opacity of the background image (0.0 - 1.0)
    
    Returns:
        GeometricVisualization with background image configured
    """
    visualization = convert_result_to_visualization(result)
    
    if page_image_url:
        visualization.d3_config['background_image'] = {
            'url': page_image_url,
            'opacity': page_image_opacity,
            'fit': 'contain'
        }
    
    return visualization


def generate_visualization_json(result: GeometricAnalysisResult) -> str:
    """
    Generate D3.js-compatible JSON for a geometric analysis result.
    
    Args:
        result: Output from analyze_page_geometry()
    
    Returns:
        JSON string ready to be consumed by the frontend
    """
    visualization = convert_result_to_visualization(result)
    
    # Serialize to D3.js-compatible format
    config = {
        'visualization_id': visualization.visualization_id,
        'type': visualization.visualization_type,
        'data': {
            'elements': [
                {
                    'type': element.element_type,
                    'points': [
                        {'x': p.x, 'y': p.y, 'label': p.label, 'character': p.character}
                        for p in element.points
                    ],
                    'measurements': element.measurements,
                    'properties': element.properties,
                    'style': element.style,
                    'annotations': element.annotations
                }
                for element in visualization.elements
            ],
            'coordinate_system': visualization.coordinate_system,
            'sacred_patterns': visualization.sacred_geometry_patterns
        },
        'config': visualization.d3_config,
        'interactive': visualization.interactive_features,
        'summary': visualization.measurements_summary,
        'export_data': visualization.export_data
    }
    
    return json.dumps(config, indent=2)
