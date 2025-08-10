"""
Reports API Routes
Report generation and export functionality
"""
from fastapi import APIRouter, HTTPException, Depends, status, Response
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
import logging

from app.api.dependencies import get_current_active_user, get_database, User
from app.services.report_generator import (
    ReportGenerator, ReportConfiguration, ReportFormat, 
    ReportType, ReportSection
)

logger = logging.getLogger(__name__)
router = APIRouter()

class ReportRequest(BaseModel):
    document_ids: List[int]
    report_type: str = "comprehensive"
    format: str = "html"
    sections: List[str] = ["executive_summary", "pattern_findings", "recommendations"]
    include_visualizations: bool = True
    confidence_threshold: float = 0.7

@router.post("/generate")
async def generate_report(
    request: ReportRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """Generate analysis report"""
    try:
        report_generator = ReportGenerator(db)
        
        # Convert string enums
        report_type = ReportType(request.report_type)
        report_format = ReportFormat(request.format)
        sections = [ReportSection(s) for s in request.sections]
        
        config = ReportConfiguration(
            report_type=report_type,
            format=report_format,
            sections=sections,
            include_visualizations=request.include_visualizations,
            confidence_threshold=request.confidence_threshold
        )
        
        if len(request.document_ids) == 1:
            report = report_generator.generate_report(request.document_ids[0], config)
        else:
            report = report_generator.generate_multi_document_report(request.document_ids, config)
        
        # Return appropriate response based on format
        if report_format == ReportFormat.HTML:
            return Response(content=report.content, media_type="text/html")
        elif report_format == ReportFormat.JSON:
            return Response(content=report.content, media_type="application/json")
        elif report_format == ReportFormat.CSV:
            return Response(content=report.content, media_type="text/csv")
        else:
            return Response(content=report.content, media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Report generation failed")

@router.get("/formats")
async def get_report_formats():
    """Get available report formats"""
    return {
        "formats": [f.value for f in ReportFormat],
        "types": [t.value for t in ReportType],
        "sections": [s.value for s in ReportSection]
    }