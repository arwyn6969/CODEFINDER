"""
Shared corpus-registry models for scalable corpus management.
"""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class CorpusWork(Base):
    __tablename__ = "corpus_works"

    id = Column(Integer, primary_key=True, index=True)
    work_key = Column(String(120), unique=True, nullable=False)
    title = Column(String(500), nullable=False)
    normalized_title = Column(String(500), nullable=False, index=True)
    author = Column(String(255))
    language = Column(String(120))
    script = Column(String(120))
    print_or_manuscript = Column(String(50), nullable=False)
    date_range = Column(String(120))
    notes = Column(Text)
    metadata_json = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    witnesses = relationship("CorpusWitness", back_populates="work", cascade="all, delete-orphan")


class CorpusWitness(Base):
    __tablename__ = "corpus_witnesses"

    id = Column(Integer, primary_key=True, index=True)
    work_id = Column(Integer, ForeignKey("corpus_works.id"), nullable=False)
    witness_key = Column(String(160), unique=True, nullable=False)
    short_title = Column(String(500))
    language = Column(String(120))
    script = Column(String(120))
    lane = Column(String(80), nullable=False, index=True)
    printer = Column(String(255))
    normalized_printer = Column(String(255), index=True)
    publisher = Column(String(255))
    seller = Column(String(255))
    place = Column(String(255))
    normalized_place = Column(String(255), index=True)
    institution = Column(String(255))
    acquisition_method = Column(String(120))
    image_quality_tier = Column(String(120))
    provenance_confidence = Column(Float, default=0.5)
    date_range = Column(String(120))
    metadata_json = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    work = relationship("CorpusWork", back_populates="witnesses")
    sources = relationship("CorpusSource", back_populates="witness", cascade="all, delete-orphan")
    image_sets = relationship("CorpusImageSet", back_populates="witness", cascade="all, delete-orphan")
    ocr_runs = relationship("CorpusOCRRun", back_populates="witness", cascade="all, delete-orphan")
    feature_runs = relationship("CorpusFeatureRun", back_populates="witness", cascade="all, delete-orphan")
    left_edges = relationship(
        "CorpusSimilarityEdge",
        foreign_keys="CorpusSimilarityEdge.left_witness_id",
        back_populates="left_witness",
    )
    right_edges = relationship(
        "CorpusSimilarityEdge",
        foreign_keys="CorpusSimilarityEdge.right_witness_id",
        back_populates="right_witness",
    )


class CorpusSource(Base):
    __tablename__ = "corpus_sources"

    id = Column(Integer, primary_key=True, index=True)
    witness_id = Column(Integer, ForeignKey("corpus_witnesses.id"), nullable=False)
    source_key = Column(String(160), unique=True, nullable=False)
    source_type = Column(String(120))
    institution = Column(String(255))
    external_id = Column(String(255))
    source_url = Column(String(1000))
    manifest_url = Column(String(1000))
    rights_url = Column(String(1000))
    acquisition_method = Column(String(120))
    metadata_json = Column(JSON)
    created_at = Column(DateTime, default=func.now())

    witness = relationship("CorpusWitness", back_populates="sources")


class CorpusImageSet(Base):
    __tablename__ = "corpus_image_sets"

    id = Column(Integer, primary_key=True, index=True)
    witness_id = Column(Integer, ForeignKey("corpus_witnesses.id"), nullable=False)
    image_set_key = Column(String(180), unique=True, nullable=False)
    root_path = Column(String(500))
    total_pages = Column(Integer)
    mime_type = Column(String(120))
    ingestion_mode = Column(String(120))
    image_quality_tier = Column(String(120))
    metadata_json = Column(JSON)
    created_at = Column(DateTime, default=func.now())

    witness = relationship("CorpusWitness", back_populates="image_sets")


class CorpusOCRRun(Base):
    __tablename__ = "corpus_ocr_runs"

    id = Column(Integer, primary_key=True, index=True)
    witness_id = Column(Integer, ForeignKey("corpus_witnesses.id"), nullable=False)
    lane = Column(String(80), nullable=False)
    ocr_profile = Column(String(120), nullable=False)
    engine_name = Column(String(120))
    status = Column(String(50), default="planned")
    page_count = Column(Integer)
    metrics_json = Column(JSON)
    created_at = Column(DateTime, default=func.now())

    witness = relationship("CorpusWitness", back_populates="ocr_runs")


class CorpusFeatureRun(Base):
    __tablename__ = "corpus_feature_runs"

    id = Column(Integer, primary_key=True, index=True)
    witness_id = Column(Integer, ForeignKey("corpus_witnesses.id"), nullable=False)
    lane = Column(String(80), nullable=False)
    feature_family = Column(String(120), nullable=False)
    status = Column(String(50), default="planned")
    metrics_json = Column(JSON)
    created_at = Column(DateTime, default=func.now())

    witness = relationship("CorpusWitness", back_populates="feature_runs")


class CorpusSimilarityEdge(Base):
    __tablename__ = "corpus_similarity_edges"

    id = Column(Integer, primary_key=True, index=True)
    left_witness_id = Column(Integer, ForeignKey("corpus_witnesses.id"), nullable=False)
    right_witness_id = Column(Integer, ForeignKey("corpus_witnesses.id"), nullable=False)
    lane = Column(String(80), nullable=False, index=True)
    candidate_type = Column(String(80), nullable=False, index=True)
    score = Column(Float, nullable=False)
    evidence_basis = Column(JSON)
    requires_manual_review = Column(Boolean, default=True)
    review_status = Column(String(50), default="pending")
    safe_for_external = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    left_witness = relationship("CorpusWitness", foreign_keys=[left_witness_id], back_populates="left_edges")
    right_witness = relationship("CorpusWitness", foreign_keys=[right_witness_id], back_populates="right_edges")
    review_verdicts = relationship("CorpusReviewVerdict", back_populates="similarity_edge", cascade="all, delete-orphan")

    __table_args__ = (
        Index(
            "idx_corpus_similarity_pair_type",
            "left_witness_id",
            "right_witness_id",
            "candidate_type",
            unique=True,
        ),
    )


class CorpusReviewVerdict(Base):
    __tablename__ = "corpus_review_verdicts"

    id = Column(Integer, primary_key=True, index=True)
    similarity_edge_id = Column(Integer, ForeignKey("corpus_similarity_edges.id"), nullable=False)
    vocabulary = Column(String(80), nullable=False)
    confidence = Column(Float, default=0.5)
    evidence_basis = Column(String(255))
    safe_for_external = Column(Boolean, default=False)
    notes = Column(Text)
    created_at = Column(DateTime, default=func.now())

    similarity_edge = relationship("CorpusSimilarityEdge", back_populates="review_verdicts")
