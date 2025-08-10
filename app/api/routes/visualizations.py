"""
Visualizations API Routes
Interactive visualization data and configurations
"""
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging

from app.api.dependencies import get_current_active_user, get_database, User
from app.services.geometric_visualizer import GeometricVisualizer, GeometricVisualizationType

logger = logging.getLogger(__name__)
router = APIRouter()

class VisualizationRequest(BaseModel):
    visualization_type: str
    show_grid: bool = True
    show_measurements: bool = True

@router.get("/{document_id}/geometric")
async def get_geometric_visualization(
    document_id: int,
    viz_type: str = "coordinate_plot",
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """Get geometric visualization data"""
    try:
        visualizer = GeometricVisualizer(db)
        
        if viz_type == "coordinate_plot":
            viz = visualizer.create_interactive_coordinate_plot(document_id)
        elif viz_type == "angles":
            viz = visualizer.create_angle_measurement_visualization(document_id)
        elif viz_type == "distances":
            viz = visualizer.create_distance_analysis_visualization(document_id)
        elif viz_type == "sacred_geometry":
            viz = visualizer.create_sacred_geometry_visualization(document_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid visualization type")
        
        return {
            "visualization_id": viz.visualization_id,
            "type": viz.visualization_type.value,
            "d3_config": viz.d3_config,
            "data": viz.export_data,
            "interactive_features": viz.interactive_features
        }
        
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        raise HTTPException(status_code=500, detail="Visualization generation failed")

@router.get("/{document_id}/d3-config")
async def get_d3_config(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """Get D3.js configuration for visualizations"""
    try:
        visualizer = GeometricVisualizer(db)
        viz = visualizer.create_interactive_coordinate_plot(document_id)
        
        return {
            "config": visualizer.generate_d3_javascript_config(viz),
            "visualization_id": viz.visualization_id
        }
        
    except Exception as e:
        logger.error(f"D3 config error: {str(e)}")
        raise HTTPException(status_code=500, detail="D3 configuration generation failed")