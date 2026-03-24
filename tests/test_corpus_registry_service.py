"""
Tests for corpus registry ingestion, routing, and candidate generation.
"""

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.corpus_registry_models import CorpusOCRRun, CorpusSimilarityEdge, CorpusWitness
from app.services.corpus_registry_service import CorpusRegistryService
from app.services.corpus_registry_types import CandidateType, CorpusLane, ReviewVocabulary


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_ingest_records_creates_entities_and_profiles(db_session):
    service = CorpusRegistryService(db_session)

    result = service.ingest_records(
        [
            {
                "title": "Vita, Non Vita",
                "work_key": "work:vita-non-vita",
                "witness_key": "witness:gdz_goettingen_ppn777246686",
                "source_key": "gdz_goettingen_ppn777246686",
                "language": "German",
                "script": "fraktur_blackletter",
                "printer": "Christoff Krausen",
                "place": "Kempten",
                "date_range": "1609",
                "print_or_manuscript": "printed",
                "acquisition_method": "iiif_images",
                "image_quality_tier": "full",
                "root_path": "data/sources/gdz_ppn777246686",
                "image_set_key": "images:gdz_goettingen_ppn777246686",
                "total_pages": 60,
            }
        ]
    )

    witness = db_session.query(CorpusWitness).one()
    ocr_run = db_session.query(CorpusOCRRun).one()

    assert result["works_created"] == 1
    assert result["witnesses_created"] == 1
    assert witness.lane == CorpusLane.PRINTED_EUROPEAN.value
    assert witness.normalized_printer == "christoph kraus"
    assert ocr_run.ocr_profile == "fraktur_blackletter"


def test_build_similarity_candidates_prefers_printer_shop_matches(db_session):
    service = CorpusRegistryService(db_session)

    service.ingest_records(
        [
            {
                "title": "Vita, Non Vita",
                "work_key": "work:vita-non-vita",
                "witness_key": "witness:gdz",
                "source_key": "gdz_goettingen_ppn777246686",
                "language": "German",
                "script": "fraktur_blackletter",
                "printer": "Christoff Krausen",
                "place": "Kempten",
                "date_range": "1609",
                "print_or_manuscript": "printed",
                "image_set_key": "images:gdz",
            },
            {
                "title": "Exercitationum theologicarum pars prima",
                "work_key": "work:exercitationum",
                "witness_key": "witness:hab",
                "source_key": "hab_wolfenbuettel_178_1_theol_1s",
                "language": "German",
                "script": "fraktur_blackletter",
                "printer": "Christoph Kraus",
                "place": "Campidoni",
                "date_range": "1616",
                "print_or_manuscript": "printed",
                "image_set_key": "images:hab",
            },
            {
                "title": "Het Schilder Boeck",
                "work_key": "work:het-schilder-boeck",
                "witness_key": "witness:negative-control",
                "source_key": "negative_control_bsb10222478",
                "language": "Dutch",
                "script": "mixed_latin_vernacular",
                "printer": "Jacob Pietersz Wachter",
                "place": "Amsterdam",
                "date_range": "1616",
                "print_or_manuscript": "printed",
                "image_set_key": "images:negative-control",
            },
        ]
    )

    result = service.build_similarity_candidates()

    cluster_pairs = {
        (item["left_witness_key"], item["right_witness_key"])
        for item in result["cluster_candidates"]
    }
    forensic_pairs = {
        (item["left_witness_key"], item["right_witness_key"])
        for item in result["forensic_candidates"]
    }

    assert ("witness:gdz", "witness:hab") in cluster_pairs
    assert ("witness:gdz", "witness:hab") in forensic_pairs
    assert ("witness:gdz", "witness:negative-control") not in cluster_pairs


def test_manuscript_records_route_to_separate_lane(db_session):
    service = CorpusRegistryService(db_session)

    service.ingest_records(
        [
            {
                "title": "Ancient Tibetan Scroll Fragment",
                "work_key": "work:tibetan-scroll-fragment",
                "witness_key": "witness:tibetan-scroll",
                "source_key": "tibetan_scroll_fragment",
                "language": "Tibetan",
                "script": "tibetan",
                "print_or_manuscript": "manuscript",
                "date_range": "1200-1300",
                "image_set_key": "images:tibetan-scroll",
            }
        ]
    )

    witness = db_session.query(CorpusWitness).filter(CorpusWitness.witness_key == "witness:tibetan-scroll").one()
    ocr_run = db_session.query(CorpusOCRRun).filter(CorpusOCRRun.witness_id == witness.id).one()

    assert witness.lane == CorpusLane.MANUSCRIPT_ANCIENT.value
    assert ocr_run.ocr_profile == "manuscript_layout_required"


def test_bootstrap_from_current_sources_config(db_session):
    service = CorpusRegistryService(db_session)

    result = service.bootstrap_from_sources_config(ROOT / "data/sources/config.yaml")
    summary = service.summarize_registry()

    assert result["records_prepared"] >= 8
    assert summary["witness_count"] >= 8
    assert summary["lanes"][CorpusLane.PRINTED_EUROPEAN.value] >= 8


def test_record_review_verdict_updates_edge_status(db_session):
    service = CorpusRegistryService(db_session)
    service.ingest_records(
        [
            {
                "title": "Vita, Non Vita",
                "work_key": "work:vita-non-vita",
                "witness_key": "witness:left",
                "source_key": "source:left",
                "language": "German",
                "script": "fraktur_blackletter",
                "printer": "Christoff Krausen",
                "place": "Kempten",
                "date_range": "1609",
                "print_or_manuscript": "printed",
                "image_set_key": "images:left",
            },
            {
                "title": "Exercitationum theologicarum pars prima",
                "work_key": "work:exercitationum",
                "witness_key": "witness:right",
                "source_key": "source:right",
                "language": "German",
                "script": "fraktur_blackletter",
                "printer": "Christoph Kraus",
                "place": "Campidoni",
                "date_range": "1610",
                "print_or_manuscript": "printed",
                "image_set_key": "images:right",
            },
        ]
    )
    candidates = service.build_similarity_candidates()
    edge_id = candidates["forensic_candidates"][0]["id"]

    response = service.record_review_verdict(
        edge_id=edge_id,
        vocabulary=ReviewVocabulary.POSSIBLE_SAME_SORT_BLOCK,
        confidence=0.7,
        safe_for_external=False,
    )
    edge = db_session.query(CorpusSimilarityEdge).filter(CorpusSimilarityEdge.id == edge_id).one()

    assert response["vocabulary"] == ReviewVocabulary.POSSIBLE_SAME_SORT_BLOCK.value
    assert edge.review_status == "reviewed"
    assert edge.candidate_type == CandidateType.FORENSIC_CANDIDATE.value
