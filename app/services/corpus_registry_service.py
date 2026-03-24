"""
Corpus registry ingestion, lane routing, and candidate generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import yaml
from sqlalchemy.orm import Session

from app.models.corpus_registry_models import (
    CorpusFeatureRun,
    CorpusImageSet,
    CorpusOCRRun,
    CorpusReviewVerdict,
    CorpusSimilarityEdge,
    CorpusSource,
    CorpusWitness,
    CorpusWork,
)
from app.services.corpus_registry_types import (
    CandidateType,
    CorpusLane,
    OCRProfile,
    PrintMaterialType,
    ReviewVocabulary,
    normalize_place,
    normalize_printer_name,
    normalize_token,
)


@dataclass
class IngestOutcome:
    works_created: int = 0
    works_updated: int = 0
    witnesses_created: int = 0
    witnesses_updated: int = 0
    sources_created: int = 0
    image_sets_created: int = 0
    ocr_runs_created: int = 0
    feature_runs_created: int = 0
    skipped: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "works_created": self.works_created,
            "works_updated": self.works_updated,
            "witnesses_created": self.witnesses_created,
            "witnesses_updated": self.witnesses_updated,
            "sources_created": self.sources_created,
            "image_sets_created": self.image_sets_created,
            "ocr_runs_created": self.ocr_runs_created,
            "feature_runs_created": self.feature_runs_created,
            "skipped": self.skipped,
        }


class CorpusLaneRouter:
    """Route records into stable analytical lanes."""

    manuscript_markers = (
        "manuscript",
        "scroll",
        "tibetan",
        "sanskrit",
        "palm leaf",
        "codex",
        "esoteric manuscript",
    )

    def classify(self, record: dict[str, Any]) -> CorpusLane:
        material_type = normalize_token(record.get("print_or_manuscript"))
        searchable = " ".join(
            normalize_token(str(record.get(field, "")))
            for field in ("title", "notes", "script", "language", "source_key")
        )

        if material_type == PrintMaterialType.MANUSCRIPT.value:
            return CorpusLane.MANUSCRIPT_ANCIENT
        if any(marker in searchable for marker in self.manuscript_markers):
            return CorpusLane.MANUSCRIPT_ANCIENT
        return CorpusLane.PRINTED_EUROPEAN


class OCRProfileSelector:
    """Select the first-pass OCR profile for a routed record."""

    def select(self, record: dict[str, Any], lane: CorpusLane) -> OCRProfile:
        if lane == CorpusLane.MANUSCRIPT_ANCIENT:
            return OCRProfile.MANUSCRIPT_LAYOUT_REQUIRED

        script = normalize_token(record.get("script"))
        language = normalize_token(record.get("language"))

        if any(marker in script for marker in ("fraktur", "blackletter", "gothic")):
            return OCRProfile.FRAKTUR_BLACKLETTER
        if any(marker in language for marker in ("german", "deutsch")) and not script:
            return OCRProfile.FRAKTUR_BLACKLETTER
        if any(marker in script for marker in ("roman", "italic")):
            return OCRProfile.ROMAN_ITALIC
        if any(marker in language for marker in ("latin", "english", "french", "italian", "spanish")):
            return OCRProfile.ROMAN_ITALIC
        return OCRProfile.MIXED_LATIN_VERNACULAR


class CorpusRegistryService:
    """
    Manage the shared corpus registry and build corpus-level candidates.
    """

    def __init__(self, db_session: Session):
        self.db = db_session
        self.lane_router = CorpusLaneRouter()
        self.ocr_profiles = OCRProfileSelector()

    def ingest_records(self, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        outcome = IngestOutcome()

        for raw_record in records:
            record = self._prepare_record(raw_record)
            if not record.get("title"):
                outcome.skipped += 1
                continue

            work = self._get_or_create_work(record, outcome)
            witness = self._get_or_create_witness(work, record, outcome)
            self._get_or_create_source(witness, record, outcome)
            self._get_or_create_image_set(witness, record, outcome)
            self._get_or_create_processing_runs(witness, record, outcome)

        self.db.commit()
        return outcome.to_dict()

    def bootstrap_from_sources_config(
        self,
        config_path: Path | str = Path("data/sources/config.yaml"),
        source_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        config_path = Path(config_path)
        with config_path.open() as handle:
            config = yaml.safe_load(handle) or {}

        requested = set(source_keys or [])
        records = []
        for source_key, source_config in (config.get("sources") or {}).items():
            if requested and source_key not in requested:
                continue
            record = self._record_from_sources_config(source_key, source_config)
            if record:
                records.append(record)

        ingest_summary = self.ingest_records(records)
        return {
            "config_path": str(config_path),
            "records_prepared": len(records),
            "ingest_summary": ingest_summary,
        }

    def summarize_registry(self) -> dict[str, Any]:
        witnesses = self.db.query(CorpusWitness).all()
        edges = self.db.query(CorpusSimilarityEdge).all()

        lane_summary: dict[str, int] = {}
        for witness in witnesses:
            lane_summary[witness.lane] = lane_summary.get(witness.lane, 0) + 1

        candidate_summary: dict[str, int] = {}
        for edge in edges:
            candidate_summary[edge.candidate_type] = candidate_summary.get(edge.candidate_type, 0) + 1

        return {
            "work_count": self.db.query(CorpusWork).count(),
            "witness_count": len(witnesses),
            "source_count": self.db.query(CorpusSource).count(),
            "image_set_count": self.db.query(CorpusImageSet).count(),
            "ocr_run_count": self.db.query(CorpusOCRRun).count(),
            "feature_run_count": self.db.query(CorpusFeatureRun).count(),
            "candidate_count": len(edges),
            "lanes": lane_summary,
            "candidates": candidate_summary,
        }

    def build_similarity_candidates(
        self,
        lane: CorpusLane = CorpusLane.PRINTED_EUROPEAN,
        min_cluster_score: float = 0.65,
        min_forensic_score: float = 0.85,
    ) -> dict[str, Any]:
        witnesses = (
            self.db.query(CorpusWitness)
            .filter(CorpusWitness.lane == lane.value)
            .order_by(CorpusWitness.id)
            .all()
        )

        cluster_candidates = []
        forensic_candidates = []

        for index, left in enumerate(witnesses):
            for right in witnesses[index + 1:]:
                score, evidence = self._score_witness_pair(left, right)
                if score >= min_cluster_score:
                    edge = self._upsert_similarity_edge(
                        left=left,
                        right=right,
                        lane=lane,
                        candidate_type=CandidateType.CLUSTER_CANDIDATE,
                        score=score,
                        evidence=evidence,
                    )
                    cluster_candidates.append(self._serialize_edge(edge))

                if self._qualifies_for_forensic_candidate(left, right, evidence, score, min_forensic_score):
                    edge = self._upsert_similarity_edge(
                        left=left,
                        right=right,
                        lane=lane,
                        candidate_type=CandidateType.FORENSIC_CANDIDATE,
                        score=score,
                        evidence=evidence,
                    )
                    forensic_candidates.append(self._serialize_edge(edge))

        self.db.commit()
        return {
            "lane": lane.value,
            "cluster_candidates": cluster_candidates,
            "forensic_candidates": forensic_candidates,
        }

    def list_candidates(self, candidate_type: CandidateType | None = None) -> list[dict[str, Any]]:
        query = self.db.query(CorpusSimilarityEdge).order_by(CorpusSimilarityEdge.score.desc())
        if candidate_type:
            query = query.filter(CorpusSimilarityEdge.candidate_type == candidate_type.value)
        return [self._serialize_edge(edge) for edge in query.all()]

    def record_review_verdict(
        self,
        edge_id: int,
        vocabulary: ReviewVocabulary,
        confidence: float = 0.5,
        evidence_basis: str = "manual review",
        safe_for_external: bool = False,
        notes: str | None = None,
    ) -> dict[str, Any]:
        edge = self.db.query(CorpusSimilarityEdge).filter(CorpusSimilarityEdge.id == edge_id).first()
        if not edge:
            raise ValueError(f"Similarity edge {edge_id} not found")

        verdict = CorpusReviewVerdict(
            similarity_edge_id=edge.id,
            vocabulary=vocabulary.value,
            confidence=confidence,
            evidence_basis=evidence_basis,
            safe_for_external=safe_for_external,
            notes=notes,
        )
        self.db.add(verdict)
        edge.review_status = "reviewed"
        edge.safe_for_external = safe_for_external
        self.db.commit()
        self.db.refresh(verdict)
        return {
            "edge_id": edge.id,
            "verdict_id": verdict.id,
            "vocabulary": verdict.vocabulary,
            "safe_for_external": verdict.safe_for_external,
        }

    def _prepare_record(self, raw_record: dict[str, Any]) -> dict[str, Any]:
        record = dict(raw_record)
        lane = self.lane_router.classify(record)
        record["lane"] = lane.value
        record["ocr_profile"] = self.ocr_profiles.select(record, lane).value

        printer = record.get("printer")
        record["normalized_printer"] = normalize_printer_name(printer)
        record["normalized_place"] = normalize_place(record.get("place"))
        record["normalized_title"] = normalize_token(record.get("title"))
        return record

    def _get_or_create_work(self, record: dict[str, Any], outcome: IngestOutcome) -> CorpusWork:
        work_key = record.get("work_key") or record.get("normalized_title")
        work = self.db.query(CorpusWork).filter(CorpusWork.work_key == work_key).first()
        if work:
            work.title = record["title"]
            work.normalized_title = record["normalized_title"]
            work.author = record.get("author")
            work.language = record.get("language")
            work.script = record.get("script")
            work.print_or_manuscript = record.get("print_or_manuscript", PrintMaterialType.PRINTED.value)
            work.date_range = record.get("date_range")
            work.notes = record.get("notes")
            work.metadata_json = record.get("metadata_json")
            outcome.works_updated += 1
            return work

        work = CorpusWork(
            work_key=work_key,
            title=record["title"],
            normalized_title=record["normalized_title"],
            author=record.get("author"),
            language=record.get("language"),
            script=record.get("script"),
            print_or_manuscript=record.get("print_or_manuscript", PrintMaterialType.PRINTED.value),
            date_range=record.get("date_range"),
            notes=record.get("notes"),
            metadata_json=record.get("metadata_json"),
        )
        self.db.add(work)
        self.db.flush()
        outcome.works_created += 1
        return work

    def _get_or_create_witness(
        self,
        work: CorpusWork,
        record: dict[str, Any],
        outcome: IngestOutcome,
    ) -> CorpusWitness:
        witness_key = record.get("witness_key") or record.get("source_key") or f"{work.work_key}:default"
        witness = self.db.query(CorpusWitness).filter(CorpusWitness.witness_key == witness_key).first()
        if witness:
            outcome.witnesses_updated += 1
        else:
            witness = CorpusWitness(work_id=work.id, witness_key=witness_key)
            self.db.add(witness)
            outcome.witnesses_created += 1

        witness.short_title = record.get("short_title") or record["title"]
        witness.language = record.get("language")
        witness.script = record.get("script")
        witness.lane = record["lane"]
        witness.printer = record.get("printer")
        witness.normalized_printer = record["normalized_printer"]
        witness.publisher = record.get("publisher")
        witness.seller = record.get("seller")
        witness.place = record.get("place")
        witness.normalized_place = record["normalized_place"]
        witness.institution = record.get("institution")
        witness.acquisition_method = record.get("acquisition_method")
        witness.image_quality_tier = record.get("image_quality_tier")
        witness.provenance_confidence = float(record.get("provenance_confidence", 0.5))
        witness.date_range = record.get("date_range")
        witness.metadata_json = record.get("metadata_json")
        self.db.flush()
        return witness

    def _get_or_create_source(self, witness: CorpusWitness, record: dict[str, Any], outcome: IngestOutcome) -> None:
        source_key = record.get("source_key")
        if not source_key:
            return

        source = self.db.query(CorpusSource).filter(CorpusSource.source_key == source_key).first()
        if source:
            return

        source = CorpusSource(
            witness_id=witness.id,
            source_key=source_key,
            source_type=record.get("source_type"),
            institution=record.get("institution"),
            external_id=record.get("external_id"),
            source_url=record.get("source_url"),
            manifest_url=record.get("manifest_url"),
            rights_url=record.get("rights_url"),
            acquisition_method=record.get("acquisition_method"),
            metadata_json=record.get("source_metadata"),
        )
        self.db.add(source)
        self.db.flush()
        outcome.sources_created += 1

    def _get_or_create_image_set(self, witness: CorpusWitness, record: dict[str, Any], outcome: IngestOutcome) -> None:
        image_set_key = record.get("image_set_key")
        if not image_set_key:
            return

        existing = self.db.query(CorpusImageSet).filter(CorpusImageSet.image_set_key == image_set_key).first()
        if existing:
            return

        image_set = CorpusImageSet(
            witness_id=witness.id,
            image_set_key=image_set_key,
            root_path=record.get("root_path"),
            total_pages=record.get("total_pages"),
            mime_type=record.get("mime_type"),
            ingestion_mode=record.get("ingestion_mode"),
            image_quality_tier=record.get("image_quality_tier"),
            metadata_json=record.get("image_set_metadata"),
        )
        self.db.add(image_set)
        self.db.flush()
        outcome.image_sets_created += 1

    def _get_or_create_processing_runs(
        self,
        witness: CorpusWitness,
        record: dict[str, Any],
        outcome: IngestOutcome,
    ) -> None:
        ocr_run = (
            self.db.query(CorpusOCRRun)
            .filter(
                CorpusOCRRun.witness_id == witness.id,
                CorpusOCRRun.ocr_profile == record["ocr_profile"],
            )
            .first()
        )
        if not ocr_run:
            ocr_run = CorpusOCRRun(
                witness_id=witness.id,
                lane=record["lane"],
                ocr_profile=record["ocr_profile"],
                engine_name="planned",
                status="planned",
                page_count=record.get("total_pages"),
                metrics_json={"source": "registry bootstrap"},
            )
            self.db.add(ocr_run)
            outcome.ocr_runs_created += 1

        feature_family = "metadata_cluster" if record["lane"] == CorpusLane.PRINTED_EUROPEAN.value else "manuscript_feasibility"
        feature_run = (
            self.db.query(CorpusFeatureRun)
            .filter(
                CorpusFeatureRun.witness_id == witness.id,
                CorpusFeatureRun.feature_family == feature_family,
            )
            .first()
        )
        if not feature_run:
            feature_run = CorpusFeatureRun(
                witness_id=witness.id,
                lane=record["lane"],
                feature_family=feature_family,
                status="planned",
                metrics_json={"ocr_profile": record["ocr_profile"]},
            )
            self.db.add(feature_run)
            outcome.feature_runs_created += 1

    def _score_witness_pair(self, left: CorpusWitness, right: CorpusWitness) -> tuple[float, dict[str, Any]]:
        evidence: dict[str, Any] = {
            "matched_features": [],
            "left": {"witness_key": left.witness_key},
            "right": {"witness_key": right.witness_key},
        }
        score = 0.0

        if left.normalized_printer and left.normalized_printer == right.normalized_printer:
            score += 0.45
            evidence["matched_features"].append("printer")

        if left.normalized_place and left.normalized_place == right.normalized_place:
            score += 0.20
            evidence["matched_features"].append("place")

        year_score = self._year_overlap_score(left.date_range, right.date_range)
        if year_score:
            score += 0.15 * year_score
            evidence["matched_features"].append("date_range")

        if left.script and right.script and left.script == right.script:
            score += 0.10
            evidence["matched_features"].append("script")

        if left.language and right.language and left.language == right.language:
            score += 0.10
            evidence["matched_features"].append("language")

        evidence["score_breakdown"] = {
            "printer_place": round(score, 4),
            "year_overlap": round(year_score, 4),
        }
        return round(min(score, 1.0), 4), evidence

    def _qualifies_for_forensic_candidate(
        self,
        left: CorpusWitness,
        right: CorpusWitness,
        evidence: dict[str, Any],
        score: float,
        min_forensic_score: float,
    ) -> bool:
        matched = set(evidence.get("matched_features") or [])
        return (
            score >= min_forensic_score
            and "printer" in matched
            and "place" in matched
            and left.lane == CorpusLane.PRINTED_EUROPEAN.value
            and right.lane == CorpusLane.PRINTED_EUROPEAN.value
        )

    def _upsert_similarity_edge(
        self,
        left: CorpusWitness,
        right: CorpusWitness,
        lane: CorpusLane,
        candidate_type: CandidateType,
        score: float,
        evidence: dict[str, Any],
    ) -> CorpusSimilarityEdge:
        edge = (
            self.db.query(CorpusSimilarityEdge)
            .filter(
                CorpusSimilarityEdge.left_witness_id == left.id,
                CorpusSimilarityEdge.right_witness_id == right.id,
                CorpusSimilarityEdge.candidate_type == candidate_type.value,
            )
            .first()
        )
        if not edge:
            edge = CorpusSimilarityEdge(
                left_witness_id=left.id,
                right_witness_id=right.id,
                lane=lane.value,
                candidate_type=candidate_type.value,
                score=score,
                evidence_basis=evidence,
                requires_manual_review=True,
                review_status="pending",
                safe_for_external=False,
            )
            self.db.add(edge)
            self.db.flush()
            return edge

        edge.score = score
        edge.evidence_basis = evidence
        edge.lane = lane.value
        return edge

    def _serialize_edge(self, edge: CorpusSimilarityEdge) -> dict[str, Any]:
        return {
            "id": edge.id,
            "lane": edge.lane,
            "candidate_type": edge.candidate_type,
            "score": edge.score,
            "left_witness_key": edge.left_witness.witness_key,
            "right_witness_key": edge.right_witness.witness_key,
            "review_status": edge.review_status,
            "safe_for_external": edge.safe_for_external,
            "evidence_basis": edge.evidence_basis,
        }

    def _record_from_sources_config(self, source_key: str, source_config: dict[str, Any]) -> dict[str, Any] | None:
        metadata = source_config.get("metadata") or {}
        provenance = source_config.get("provenance") or {}
        quality = source_config.get("quality") or {}
        source_type = source_config.get("type")

        if source_type == "text":
            return None

        title = metadata.get("title") or source_key
        language, script = self._infer_language_and_script(source_key, metadata)
        print_or_manuscript = PrintMaterialType.MANUSCRIPT.value if "manuscript" in normalize_token(title) else PrintMaterialType.PRINTED.value

        return {
            "work_key": f"work:{normalize_token(title)[:100]}",
            "witness_key": f"witness:{source_key}",
            "source_key": source_key,
            "image_set_key": f"images:{source_key}",
            "title": title,
            "short_title": title,
            "author": metadata.get("author"),
            "language": language,
            "script": script,
            "print_or_manuscript": print_or_manuscript,
            "date_range": str(metadata.get("date") or ""),
            "printer": metadata.get("printer"),
            "publisher": metadata.get("publisher"),
            "seller": metadata.get("seller"),
            "place": metadata.get("place"),
            "institution": provenance.get("name"),
            "acquisition_method": source_type,
            "image_quality_tier": quality.get("resolution") or quality.get("format"),
            "provenance_confidence": 0.9 if provenance else 0.5,
            "root_path": source_config.get("path"),
            "total_pages": quality.get("total_pages"),
            "mime_type": "application/pdf" if source_type in {"pdf", "pdf_images"} else "image/jpeg",
            "ingestion_mode": source_type,
            "notes": metadata.get("note") or source_config.get("notes"),
            "source_type": source_type,
            "external_id": provenance.get("id") or provenance.get("catalog_id") or provenance.get("ebook_id"),
            "source_url": provenance.get("url"),
            "manifest_url": provenance.get("manifest"),
            "rights_url": provenance.get("rights"),
            "metadata_json": metadata,
            "source_metadata": provenance,
            "image_set_metadata": quality,
        }

    def _infer_language_and_script(self, source_key: str, metadata: dict[str, Any]) -> tuple[str, str]:
        text = " ".join(
            [
                source_key,
                str(metadata.get("title") or ""),
                str(metadata.get("author") or ""),
                str(metadata.get("note") or ""),
            ]
        ).lower()

        if "shakespeare" in text or "sonnet" in text:
            return "English", "roman_italic"
        if any(marker in text for marker in ("haslmayr", "gedicht", "vita, non vita", "tractatus", "theolog", "bsb_", "gdz_", "hab_")):
            return "German,Latin", "fraktur_blackletter"
        if "negative_control" in source_key:
            return "European print", "mixed_latin_vernacular"
        return "Unknown", "mixed_latin_vernacular"

    def _year_overlap_score(self, left_range: str | None, right_range: str | None) -> float:
        left_year = self._extract_year(left_range)
        right_year = self._extract_year(right_range)
        if left_year is None or right_year is None:
            return 0.0

        difference = abs(left_year - right_year)
        if difference == 0:
            return 1.0
        if difference <= 3:
            return 0.75
        if difference <= 10:
            return 0.5
        if difference <= 25:
            return 0.25
        return 0.0

    def _extract_year(self, value: str | None) -> int | None:
        if not value:
            return None
        match = re.search(r"(1[0-9]{3}|20[0-9]{2})", str(value))
        if not match:
            return None
        return int(match.group(1))
