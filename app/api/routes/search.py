"""
Search API Routes
Advanced search and query capabilities
"""
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging

from app.api.dependencies import get_current_active_user, get_database, User
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)
router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    document_ids: Optional[List[int]] = None
    pattern_types: Optional[List[str]] = None
    min_confidence: float = 0.0
    max_results: int = 100

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_found: int
    query_time: float

@router.post("/", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """Perform advanced search across documents and patterns"""
    try:
        search_service = SearchService(db)
        results = search_service.search(
            query=request.query,
            document_ids=request.document_ids,
            pattern_types=request.pattern_types,
            min_confidence=request.min_confidence,
            max_results=request.max_results
        )
        
        return SearchResponse(
            results=results.results,
            total_found=results.total_found,
            query_time=results.query_time
        )
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Search failed")

@router.get("/suggestions")
async def get_search_suggestions(
    query: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """Get search suggestions and autocomplete"""
    # Implementation would provide search suggestions
    return {"suggestions": [f"{query}_suggestion_{i}" for i in range(5)]}