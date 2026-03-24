"""
Corpus registry API routes.
"""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.dependencies import User, get_current_active_user, get_database, rate_limit_dependency
from app.services.corpus_registry_service import CorpusRegistryService
from app.services.corpus_registry_types import CandidateType, CorpusLane, ReviewVocabulary


router = APIRouter()


class CorpusRecordPayload(BaseModel):
    work_key: Optional[str] = None
    witness_key: Optional[str] = None
    source_key: Optional[str] = None
    image_set_key: Optional[str] = None
    title: str
    short_title: Optional[str] = None
    author: Optional[str] = None
    language: Optional[str] = None
    script: Optional[str] = None
    print_or_manuscript: Optional[str] = None
    date_range: Optional[str] = None
    printer: Optional[str] = None
    publisher: Optional[str] = None
    seller: Optional[str] = None
    place: Optional[str] = None
    institution: Optional[str] = None
    acquisition_method: Optional[str] = None
    image_quality_tier: Optional[str] = None
    provenance_confidence: Optional[float] = None
    root_path: Optional[str] = None
    total_pages: Optional[int] = None
    source_type: Optional[str] = None
    mime_type: Optional[str] = None
    ingestion_mode: Optional[str] = None
    notes: Optional[str] = None
    metadata_json: Optional[dict[str, Any]] = None


class CorpusIngestRequest(BaseModel):
    records: list[CorpusRecordPayload]


class BootstrapRequest(BaseModel):
    config_path: str = "data/sources/config.yaml"
    source_keys: Optional[list[str]] = None


class CandidateBuildRequest(BaseModel):
    lane: CorpusLane = CorpusLane.PRINTED_EUROPEAN
    min_cluster_score: float = Field(default=0.65, ge=0.0, le=1.0)
    min_forensic_score: float = Field(default=0.85, ge=0.0, le=1.0)


class ReviewVerdictRequest(BaseModel):
    edge_id: int
    vocabulary: ReviewVocabulary
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_basis: str = "manual review"
    safe_for_external: bool = False
    notes: Optional[str] = None


@router.get("/summary", dependencies=[Depends(rate_limit_dependency)])
async def get_corpus_summary(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database),
):
    service = CorpusRegistryService(db)
    return service.summarize_registry()


@router.post("/ingest", dependencies=[Depends(rate_limit_dependency)])
async def ingest_corpus_records(
    request: CorpusIngestRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database),
):
    service = CorpusRegistryService(db)
    return service.ingest_records([record.model_dump(exclude_none=True) for record in request.records])


@router.post("/bootstrap/current-sources", dependencies=[Depends(rate_limit_dependency)])
async def bootstrap_current_sources(
    request: BootstrapRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database),
):
    service = CorpusRegistryService(db)
    return service.bootstrap_from_sources_config(
        config_path=request.config_path,
        source_keys=request.source_keys,
    )


@router.post("/candidates/build", dependencies=[Depends(rate_limit_dependency)])
async def build_candidates(
    request: CandidateBuildRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database),
):
    service = CorpusRegistryService(db)
    return service.build_similarity_candidates(
        lane=request.lane,
        min_cluster_score=request.min_cluster_score,
        min_forensic_score=request.min_forensic_score,
    )


@router.get("/candidates", dependencies=[Depends(rate_limit_dependency)])
async def list_candidates(
    candidate_type: Optional[CandidateType] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database),
):
    service = CorpusRegistryService(db)
    return {"candidates": service.list_candidates(candidate_type=candidate_type)}


@router.post("/reviews", dependencies=[Depends(rate_limit_dependency)])
async def record_review_verdict(
    request: ReviewVerdictRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database),
):
    service = CorpusRegistryService(db)
    try:
        return service.record_review_verdict(
            edge_id=request.edge_id,
            vocabulary=request.vocabulary,
            confidence=request.confidence,
            evidence_basis=request.evidence_basis,
            safe_for_external=request.safe_for_external,
            notes=request.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
