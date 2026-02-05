"""
Relationship Analysis API Routes
Exposes cross-document analysis capabilities including correlation matrices,
authorship profiling, and finding hidden connections between documents.
"""
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from app.api.dependencies import get_current_active_user, get_database, User, rate_limit_dependency
from app.models.database_models import Document
from app.services.relationship_analyzer import RelationshipAnalyzer

router = APIRouter()

# --- Pydantic Models for Responses ---

class NetworkNode(BaseModel):
    id: int
    title: str
    centrality: float
    community: int

class NetworkEdge(BaseModel):
    source: int
    target: int
    weight: float
    relationship_type: str
    evidence_count: int
    confidence: float

class NetworkGraph(BaseModel):
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]

class RelationshipNetworkResponse(BaseModel):
    network: NetworkGraph
    metrics: Dict[str, Any]
    communities: Dict[int, int]
    central_documents: List[int]
    summary: Dict[str, Any]

class CorrelationMatrixResponse(BaseModel):
    document_ids: List[int]
    matrix: List[List[float]]
    significance_matrix: List[List[float]]
    cluster_assignments: Dict[int, int]

class AuthorshipProfileResponse(BaseModel):
    document_count: int
    similarity_matrix: List[List[float]]
    author_groups: Dict[str, List[int]]
    confidence_scores: Dict[str, float]
    analysis_summary: str

class EvidencePoint(BaseModel):
    type: str
    strength: float
    details: Dict[str, Any]

class EvidenceTrailResponse(BaseModel):
    pattern_id: int
    document_chain: List[int]
    connection_strength: float
    confidence_score: float
    evidence_points: List[EvidencePoint]

class AnalysisRequest(BaseModel):
    document_ids: List[int]

# --- Endpoints ---

@router.post("/network", response_model=RelationshipNetworkResponse, dependencies=[Depends(rate_limit_dependency)])
async def get_relationship_network(
    request: AnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """
    Build a network graph showing relationships between selected documents.
    Used for visualization (node-link diagrams).
    """
    try:
        # Validate documents exist
        docs = db.query(Document).filter(Document.id.in_(request.document_ids)).all()
        if len(docs) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 2 documents are required for network analysis"
            )

        analyzer = RelationshipAnalyzer(db)
        result = analyzer.build_relationship_network(request.document_ids)
        
        if 'error' in result:
             raise HTTPException(status_code=500, detail=result['error'])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate relationship network: {str(e)}"
        )

@router.post("/matrix", response_model=CorrelationMatrixResponse, dependencies=[Depends(rate_limit_dependency)])
async def get_correlation_matrix(
    request: AnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """
    Generate a correlation matrix (heatmap) for a set of documents.
    Compares signatures across pattern, stylistic, temporal, and geometric dimensions.
    """
    try:
        if len(request.document_ids) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 2 documents are required for correlation analysis"
            )

        analyzer = RelationshipAnalyzer(db)
        matrix_obj = analyzer.generate_correlation_matrix(request.document_ids)
        
        # Convert numpy arrays to lists for JSON serialization
        return CorrelationMatrixResponse(
            document_ids=matrix_obj.document_ids,
            matrix=matrix_obj.matrix.tolist(),
            significance_matrix=matrix_obj.significance_matrix.tolist(),
            cluster_assignments=matrix_obj.cluster_assignments
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate correlation matrix: {str(e)}"
        )

@router.post("/authorship", response_model=AuthorshipProfileResponse, dependencies=[Depends(rate_limit_dependency)])
async def analyze_authorship(
    request: AnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """
    Analyze authorship patterns to detect if documents were likely created by the same entity.
    """
    try:
        if len(request.document_ids) < 2:
             raise HTTPException(status_code=400, detail="At least 2 documents required")

        analyzer = RelationshipAnalyzer(db)
        result = analyzer.analyze_authorship_patterns(request.document_ids)
        
        if 'error' in result:
             raise HTTPException(status_code=500, detail=result['error'])

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authorship analysis failed: {str(e)}"
        )

@router.get("/trails/{pattern_id}", response_model=List[EvidenceTrailResponse], dependencies=[Depends(rate_limit_dependency)])
async def track_evidence_trails(
    pattern_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """
    Trace the 'Evidence Trail' of a specific pattern across multiple documents.
    Shows how a specific symbol or cipher traverses the corpus.
    """
    try:
        analyzer = RelationshipAnalyzer(db)
        trails = analyzer.track_evidence_trails(pattern_id)
        
        # Convert dataclass objects to Pydantic-compatible dicts
        return [
            {
                "pattern_id": t.pattern_id,
                "document_chain": t.document_chain,
                "connection_strength": t.connection_strength,
                "confidence_score": t.confidence_score,
                "evidence_points": t.evidence_points
            }
            for t in trails
        ]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to track evidence trails: {str(e)}"
        )
