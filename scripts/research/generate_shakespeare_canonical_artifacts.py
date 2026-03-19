#!/usr/bin/env python3
"""
Generate the canonical Shakespeare lane artifacts from the local Folger corpus
and the surviving diagnostic report set.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "shakespeare"
DEFAULT_SHAREABLE_ROOT = ROOT / "reports" / "shareable"

GENERATOR_VERSION = "1.0.0"
REPORT_DATE = "March 9, 2026"
ALLOWED_MANUAL_ASSESSMENTS = (
    "same design",
    "possible same sort/block",
    "inconclusive",
)


WITNESSES = {
    "wright": {
        "catalog_id": "STC 22353a",
        "variant": "Wright",
        "short_label": "Wright",
        "display_name": "Folger Shakespeare Library STC 22353a (Wright)",
        "source_dir": ROOT / "data" / "sources" / "folger_sonnets_1609",
        "source_metadata_path": ROOT / "data" / "sources" / "folger_sonnets_1609" / "source_metadata.json",
        "stats_path": ROOT / "reports" / "scan_wright_fixed" / "statistics.json",
        "freq_path": ROOT / "reports" / "scan_wright_fixed" / "character_frequency.csv",
        "high_conf_stats_path": ROOT / "reports" / "wright_80conf" / "statistics.json",
        "high_conf_freq_path": ROOT / "reports" / "wright_80conf" / "character_frequency.csv",
        "catalog_url": "https://digitalcollections.folger.edu/bib169144-164315",
        "manifest_url": "https://digitalcollections.folger.edu/node/70076/manifest",
        "seller": "John Wright",
        "current_local_resolution": "2000px JPG cache",
    },
    "aspley": {
        "catalog_id": "STC 22353",
        "variant": "Aspley",
        "short_label": "Aspley",
        "display_name": "Folger Shakespeare Library STC 22353 (Aspley)",
        "source_dir": ROOT / "data" / "sources" / "folger_sonnets_1609_aspley",
        "source_metadata_path": ROOT / "data" / "sources" / "folger_sonnets_1609_aspley" / "source_metadata.json",
        "stats_path": ROOT / "reports" / "scan_aspley_fixed" / "statistics.json",
        "freq_path": ROOT / "reports" / "scan_aspley_fixed" / "character_frequency.csv",
        "high_conf_stats_path": ROOT / "reports" / "aspley_80conf" / "statistics.json",
        "high_conf_freq_path": ROOT / "reports" / "aspley_80conf" / "character_frequency.csv",
        "catalog_url": "https://digitalcollections.folger.edu/node/29467",
        "manifest_url": "https://digitalcollections.folger.edu/node/29467/manifest",
        "seller": "William Aspley",
        "current_local_resolution": "2000px JPG cache",
    },
}


LEGACY_INPUTS = {
    "page_forensics": ROOT / "reports" / "page_forensics" / "page_comparison.json",
    "legacy_mapping": ROOT / "reports" / "full_sonnet_mapping.json",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_frequency_csv(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            character = row["Character"]
            rows[character] = {
                "character": character,
                "unicode_name": row["Unicode Name"],
                "codepoint": row["Codepoint"],
                "category": row["Category"],
                "count": int(row["Count"]),
                "percentage": float(row["Percentage"].rstrip("%")),
                "avg_width": float(row["Avg Width"]),
                "avg_height": float(row["Avg Height"]),
            }
    return rows


def list_local_images(source_dir: Path) -> List[Path]:
    return sorted(source_dir.glob("*.jpg"))


def has_folio_signature(name: str) -> bool:
    lowered = name.lower()
    return bool(re.search(r"(^|[_-])[a-l]\d(?:[_-]|[.])", lowered) or re.search(r"leaf_[a-l]\d", lowered))


def humanize_filename(name: str) -> str:
    base = name.rsplit(".", 1)[0]
    base = re.sub(r"^\d+_", "", base)
    return base.replace("_", " ").replace("-", " ")


def build_legacy_page_map(mapping: dict) -> Dict[str, Dict[int, List[int]]]:
    page_map: Dict[str, Dict[int, List[int]]] = {
        "wright": defaultdict(list),
        "aspley": defaultdict(list),
    }
    for witness in ("wright", "aspley"):
        for sonnet_id, payload in mapping.get(witness, {}).items():
            page_map[witness][int(payload["page"])].append(int(sonnet_id))
    for witness in page_map:
        for page in page_map[witness]:
            page_map[witness][page].sort()
    return page_map


def build_page_equivalence_manifest() -> dict:
    legacy_mapping = load_json(LEGACY_INPUTS["legacy_mapping"])
    legacy_page_map = build_legacy_page_map(legacy_mapping)

    witness_pages: Dict[str, List[dict]] = {}
    pairings: List[dict] = []

    wright_files = list_local_images(WITNESSES["wright"]["source_dir"])
    aspley_files = list_local_images(WITNESSES["aspley"]["source_dir"])
    shared_length = min(len(wright_files), 53)

    for witness_key, files in (("wright", wright_files), ("aspley", aspley_files)):
        entries = []
        for file_path in files:
            sequence_number = int(file_path.name.split("_", 1)[0])
            shared_sequence_index = sequence_number if sequence_number <= shared_length else None

            if shared_sequence_index is None:
                status = "unmatched"
                reason = "Source-specific extra capture outside the shared 53-image Folger folio sequence."
            elif has_folio_signature(file_path.name):
                status = "included"
                reason = "Part of the shared 53-image Folger folio sequence and contains a signed folio opening or mixed folio/edge content."
            else:
                status = "excluded_paratext"
                reason = "Shared witness image but dominated by cover, endleaf, or spine material."

            counterpart = None
            if shared_sequence_index is not None:
                counterpart_files = aspley_files if witness_key == "wright" else wright_files
                counterpart_path = counterpart_files[shared_sequence_index - 1]
                counterpart = {
                    "witness": "aspley" if witness_key == "wright" else "wright",
                    "sequence_number": shared_sequence_index,
                    "filename": counterpart_path.name,
                }

            content_role = "body_or_mixed" if has_folio_signature(file_path.name) else "paratext"
            if status == "unmatched" and "title_page" in file_path.name.lower():
                content_role = "paratext"
            elif status == "unmatched" and "dedication" in file_path.name.lower():
                content_role = "paratext"
            elif status == "unmatched":
                content_role = "supplemental_capture"

            entries.append(
                {
                    "sequence_number": sequence_number,
                    "filename": file_path.name,
                    "human_label": humanize_filename(file_path.name),
                    "status": status,
                    "content_role": content_role,
                    "reason": reason,
                    "shared_sequence_index": shared_sequence_index,
                    "legacy_sonnet_ids": legacy_page_map[witness_key].get(sequence_number, []),
                    "counterpart": counterpart,
                }
            )
        witness_pages[witness_key] = entries

    for index in range(1, shared_length + 1):
        wright_page = witness_pages["wright"][index - 1]
        aspley_page = witness_pages["aspley"][index - 1]
        pair_status = "included" if wright_page["status"] == "included" and aspley_page["status"] == "included" else "excluded_paratext"
        pairings.append(
            {
                "shared_sequence_index": index,
                "pair_status": pair_status,
                "wright": {
                    "sequence_number": wright_page["sequence_number"],
                    "filename": wright_page["filename"],
                    "legacy_sonnet_ids": wright_page["legacy_sonnet_ids"],
                },
                "aspley": {
                    "sequence_number": aspley_page["sequence_number"],
                    "filename": aspley_page["filename"],
                    "legacy_sonnet_ids": aspley_page["legacy_sonnet_ids"],
                },
            }
        )

    summary = {
        witness_key: {
            "total_images": len(entries),
            "included": sum(entry["status"] == "included" for entry in entries),
            "excluded_paratext": sum(entry["status"] == "excluded_paratext" for entry in entries),
            "unmatched": sum(entry["status"] == "unmatched" for entry in entries),
        }
        for witness_key, entries in witness_pages.items()
    }
    summary["shared_sequence"] = {
        "paired_images": len(pairings),
        "included_pairs": sum(pair["pair_status"] == "included" for pair in pairings),
        "paratext_pairs": sum(pair["pair_status"] == "excluded_paratext" for pair in pairings),
    }
    summary["legacy_sonnet_mapping"] = legacy_mapping["summary"]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_date": REPORT_DATE,
        "generator_version": GENERATOR_VERSION,
        "status": "canonical",
        "normalization_rule": {
            "shared_sequence_definition": "Use the shared 53-image Folger folio sequence as the pairing spine for Wright and Aspley.",
            "included_definition": "Shared-sequence image with a signed folio opening or mixed folio/edge content.",
            "excluded_paratext_definition": "Shared-sequence image dominated by cover, endleaf, or spine material.",
            "unmatched_definition": "Source-specific extra capture outside the shared 53-image sequence.",
        },
        "summary": summary,
        "pairs": pairings,
        "witness_pages": witness_pages,
    }


def build_source_payload(witness_key: str, stats: dict, source_metadata: dict) -> dict:
    spec = WITNESSES[witness_key]
    return {
        "witness_key": witness_key,
        "name": spec["display_name"],
        "short_label": spec["short_label"],
        "catalog_id": spec["catalog_id"],
        "variant": spec["variant"],
        "seller": spec["seller"],
        "file": str(spec["source_dir"].relative_to(ROOT)),
        "catalog_url": spec["catalog_url"],
        "manifest_url": spec["manifest_url"],
        "local_total_images": source_metadata["download"]["total_pages"],
        "pages_scanned": stats["pages_scanned"],
        "total_characters": stats["total_characters"],
        "unique_characters": stats["unique_characters"],
        "average_confidence": round(stats["average_confidence"], 6),
        "long_s_count": stats["long_s_count"],
        "ligatures_found": stats["ligatures_found"],
        "anomalies_detected": stats["anomalies_detected"],
        "current_local_resolution": spec["current_local_resolution"],
        "local_download_size_param": source_metadata["download"]["size_param"],
        "local_download_format": source_metadata["download"]["format"],
    }


def build_unique_character_summary(left_rows: Dict[str, dict], right_rows: Dict[str, dict]) -> dict:
    left_only = sorted(character for character in left_rows if character not in right_rows)
    right_only = sorted(character for character in right_rows if character not in left_rows)
    return {
        "only_in_source1": left_only,
        "only_in_source2": right_only,
    }


def build_top_character_deltas(left_rows: Dict[str, dict], right_rows: Dict[str, dict], limit: int = 15) -> List[dict]:
    all_chars = set(left_rows) | set(right_rows)
    rows = []
    for character in all_chars:
        left = left_rows.get(character)
        right = right_rows.get(character)
        left_count = left["count"] if left else 0
        right_count = right["count"] if right else 0
        if not character:
            continue
        delta = right_count - left_count
        category = (left or right)["category"]
        delta_percentage = None
        if left_count:
            delta_percentage = round((delta / left_count) * 100, 2)
        rows.append(
            {
                "character": character,
                "category": category,
                "source1_count": left_count,
                "source2_count": right_count,
                "delta": delta,
                "delta_percentage": delta_percentage,
            }
        )
    rows.sort(key=lambda row: (abs(row["delta"]), row["source2_count"] + row["source1_count"]), reverse=True)
    return rows[:limit]


def build_comparison_payload(
    scope: str,
    source1_stats_path: Path,
    source2_stats_path: Path,
    source1_freq_path: Path,
    source2_freq_path: Path,
    page_manifest: dict,
) -> dict:
    source1_metadata = load_json(WITNESSES["wright"]["source_metadata_path"])
    source2_metadata = load_json(WITNESSES["aspley"]["source_metadata_path"])
    source1_stats = load_json(source1_stats_path)["statistics"]
    source2_stats = load_json(source2_stats_path)["statistics"]
    source1_freq = read_frequency_csv(source1_freq_path)
    source2_freq = read_frequency_csv(source2_freq_path)

    notes = [
        "Design similarity is not same-object proof.",
        "This packet is external-safe and diagnostic; it does not claim typographic identity.",
    ]

    if scope == "whole_witness":
        notes.append("Whole-witness inventory includes shared-sequence paratext and the source-specific Aspley extras.")
    elif scope == "matched_high_confidence":
        notes.append("High-confidence comparison uses the surviving 80% confidence scan outputs and is anchored to the shared 53-image sequence.")

    if source2_stats["pages_scanned"] != source1_stats["pages_scanned"]:
        notes.append("Page-scan totals remain asymmetric across witnesses and should be interpreted as an extraction constraint, not as proof of content difference.")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_date": REPORT_DATE,
        "generator_version": GENERATOR_VERSION,
        "scope": scope,
        "claim_level": "external_safe_diagnostic",
        "matching_basis": {
            "shared_sequence_images": page_manifest["summary"]["shared_sequence"]["paired_images"],
            "included_pairs": page_manifest["summary"]["shared_sequence"]["included_pairs"],
            "paratext_pairs": page_manifest["summary"]["shared_sequence"]["paratext_pairs"],
            "legacy_same_page_sonnets": page_manifest["summary"]["legacy_sonnet_mapping"]["same_page"],
        },
        "source1": build_source_payload("wright", source1_stats, source1_metadata),
        "source2": build_source_payload("aspley", source2_stats, source2_metadata),
        "deltas": {
            "local_image_delta": source2_metadata["download"]["total_pages"] - source1_metadata["download"]["total_pages"],
            "pages_scanned_delta": source2_stats["pages_scanned"] - source1_stats["pages_scanned"],
            "character_delta": source2_stats["total_characters"] - source1_stats["total_characters"],
            "confidence_delta": round(source2_stats["average_confidence"] - source1_stats["average_confidence"], 2),
            "long_s_delta": source2_stats["long_s_count"] - source1_stats["long_s_count"],
            "ligature_delta": source2_stats["ligatures_found"] - source1_stats["ligatures_found"],
            "anomaly_delta": source2_stats["anomalies_detected"] - source1_stats["anomalies_detected"],
        },
        "unique_characters": build_unique_character_summary(source1_freq, source2_freq),
        "top_character_deltas": build_top_character_deltas(source1_freq, source2_freq),
        "notes": notes,
    }


def build_page_variance_payload(page_manifest: dict) -> dict:
    payload = load_json(LEGACY_INPUTS["page_forensics"])
    rows = payload["results"]
    ranked = sorted(rows, key=lambda row: abs(row["delta"]), reverse=True)
    top_outliers = []
    for row in ranked[:10]:
        top_outliers.append(
            {
                "page": row["page"],
                "wright_chars": row["wright_chars"],
                "aspley_chars": row["aspley_chars"],
                "delta": row["delta"],
                "ratio": row["ratio"],
                "significant": row["significant"],
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_date": REPORT_DATE,
        "generator_version": GENERATOR_VERSION,
        "scope": "page_matched_diagnostic",
        "claim_level": "external_safe_diagnostic",
        "matching_basis": "Shared 53-image Folger sequence paired by local sequence index; diagnostic only because complete sonnet-level equivalence is not yet established.",
        "source1": WITNESSES["wright"]["display_name"],
        "source2": WITNESSES["aspley"]["display_name"],
        "min_confidence": payload["min_confidence"],
        "total_pairs": payload["total_pages"],
        "significant_pairs": payload["significant_pages"],
        "significant_pair_ratio": round(payload["significant_pages"] / payload["total_pages"], 4),
        "top_outliers": top_outliers,
        "notes": [
            "Flagged variance cannot be assigned to typography alone because the lane still lacks full sonnet-level page equivalence.",
            "Pages 1-6 and 48-53 remain paratext-heavy and are therefore useful mainly as control images.",
            "The output is preserved as a diagnostic artifact, not as an external verdict.",
        ],
        "results": rows,
        "page_manifest_summary": page_manifest["summary"],
    }


def build_manual_review_ledger(page_manifest: dict, page_variance: dict) -> List[dict]:
    top_variance_pages = {row["page"]: row for row in page_variance["top_outliers"]}
    ledger = [
        {
            "review_id": "legacy-mapping-page-009-sonnet-002",
            "artifact_type": "shared-sequence-sonnet-opening",
            "wright_page": 9,
            "aspley_page": 9,
            "sonnet_ids": [2],
            "supporting_artifact": "reports/full_sonnet_mapping.json",
            "manual_assessment": "same design",
            "manual_notes": "This is the only same-page sonnet opening preserved in the surviving canonical JSON. It is useful as a continuity check for design-level similarity, not as proof of shared physical type.",
        },
        {
            "review_id": "page-017-variance-cluster",
            "artifact_type": "page-variance-cluster",
            "wright_page": 17,
            "aspley_page": 17,
            "sonnet_ids": [],
            "supporting_artifact": "reports/page_forensics/page_comparison.json",
            "manual_assessment": "inconclusive",
            "manual_notes": "Large OCR-count divergence persists in the legacy diagnostic output, but the current packet cannot isolate whether the driver is content alignment, image quality, or typographic difference.",
        },
        {
            "review_id": "page-018-variance-cluster",
            "artifact_type": "page-variance-cluster",
            "wright_page": 18,
            "aspley_page": 18,
            "sonnet_ids": [43],
            "supporting_artifact": "reports/page_forensics/deep_analysis/page_18_summary.png",
            "manual_assessment": "inconclusive",
            "manual_notes": "The deep-analysis packet is still useful for showing how strongly page-level overlays can diverge, but it no longer supports a settled explanatory verdict.",
        },
        {
            "review_id": "page-033-variance-cluster",
            "artifact_type": "page-variance-cluster",
            "wright_page": 33,
            "aspley_page": 33,
            "sonnet_ids": [],
            "supporting_artifact": "reports/page_forensics/outlier_summary.txt",
            "manual_assessment": "inconclusive",
            "manual_notes": "This remains one of the largest delta pages in the legacy output. The current package keeps it as a diagnostic review target only.",
        },
        {
            "review_id": "page-041-variance-cluster",
            "artifact_type": "page-variance-cluster",
            "wright_page": 41,
            "aspley_page": 41,
            "sonnet_ids": [],
            "supporting_artifact": "reports/page_forensics/outlier_summary.txt",
            "manual_assessment": "inconclusive",
            "manual_notes": "The near-2x OCR ratio on this pair is too unstable to treat as design evidence without stronger page-equivalence control.",
        },
        {
            "review_id": "aspley-extra-page-056",
            "artifact_type": "unmatched-extra-capture",
            "wright_page": None,
            "aspley_page": 56,
            "sonnet_ids": [126],
            "supporting_artifact": "reports/full_sonnet_mapping.json",
            "manual_assessment": "same design",
            "manual_notes": "The Aspley-only Sonnet 126 extra capture is retained as source-specific evidence of the local corpus shape. It should not be forced into the paired sequence, but it does confirm that some Shakespeare content survives only in the unmatched Aspley extras.",
        },
    ]

    valid_pages = {row["page"] for row in page_variance["top_outliers"]}
    for item in ledger:
        if item["artifact_type"] == "page-variance-cluster" and item["wright_page"] not in valid_pages:
            raise ValueError(f"Manual review page {item['wright_page']} is not present in the normalized top outlier set.")

    return ledger


def render_manual_review_summary(rows: List[dict]) -> str:
    counts = Counter(row["manual_assessment"] for row in rows)
    lines = [
        "# Shakespeare Manual Review Summary",
        "",
        "> Status: Canonical internal review companion for the Shakespeare lane",
        f"> Date: {REPORT_DATE}",
        "",
        f"Reviewed rows: `{len(rows)}` / `{len(rows)}`",
        "",
        "## Assessment counts",
        "",
    ]
    for label in ALLOWED_MANUAL_ASSESSMENTS:
        lines.append(f"- `{label}`: `{counts.get(label, 0)}`")
    lines.extend(
        [
            "",
            "## Reading rule",
            "",
            "These labels are deliberately narrow. `same design` indicates a plausible design-level resemblance within the surviving diagnostic packet; it does not establish shared physical type. No Shakespeare row is presently rated `possible same sort/block` because the normalized packet still leaves page-equivalence and extraction constraints unresolved.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_manual_review_ledger_md(rows: List[dict]) -> str:
    lines = [
        "# Shakespeare Manual Review Ledger",
        "",
        "| Review ID | Type | Wright page | Aspley page | Sonnets | Assessment | Notes |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        sonnets = ", ".join(str(sonnet_id) for sonnet_id in row["sonnet_ids"]) or "-"
        wright_page = "-" if row["wright_page"] is None else str(row["wright_page"])
        aspley_page = "-" if row["aspley_page"] is None else str(row["aspley_page"])
        lines.append(
            f"| {row['review_id']} | {row['artifact_type']} | {wright_page} | {aspley_page} | {sonnets} | {row['manual_assessment']} | {row['manual_notes']} |"
        )
    return "\n".join(lines) + "\n"


def build_package_manifest(
    page_manifest: dict,
    whole_book: dict,
    high_conf: dict,
    page_variance: dict,
    manual_rows: List[dict],
    output_root: Path,
    shareable_root: Path,
) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_date": REPORT_DATE,
        "generator_version": GENERATOR_VERSION,
        "package_name": "CODEFINDER Shakespeare Canonical Packet",
        "claim_level": "external_safe_diagnostic",
        "primary_corpus": {
            "witnesses": [
                {
                    "name": WITNESSES["wright"]["display_name"],
                    "catalog_id": WITNESSES["wright"]["catalog_id"],
                    "local_images": page_manifest["summary"]["wright"]["total_images"],
                },
                {
                    "name": WITNESSES["aspley"]["display_name"],
                    "catalog_id": WITNESSES["aspley"]["catalog_id"],
                    "local_images": page_manifest["summary"]["aspley"]["total_images"],
                },
            ],
            "shared_sequence_images": page_manifest["summary"]["shared_sequence"]["paired_images"],
            "included_pairs": page_manifest["summary"]["shared_sequence"]["included_pairs"],
            "paratext_pairs": page_manifest["summary"]["shared_sequence"]["paratext_pairs"],
            "unmatched_aspley_extras": page_manifest["summary"]["aspley"]["unmatched"],
        },
        "key_metrics": {
            "whole_witness_characters": {
                "wright": whole_book["source1"]["total_characters"],
                "aspley": whole_book["source2"]["total_characters"],
            },
            "high_confidence_characters": {
                "wright": high_conf["source1"]["total_characters"],
                "aspley": high_conf["source2"]["total_characters"],
            },
            "page_variance_significant_pairs": page_variance["significant_pairs"],
            "legacy_same_page_sonnets": page_manifest["summary"]["legacy_sonnet_mapping"]["same_page"],
            "manual_review_counts": Counter(row["manual_assessment"] for row in manual_rows),
        },
        "artifacts": {
            "summary_txt": display_path(output_root / "summary.txt"),
            "page_equivalence_manifest": display_path(output_root / "page_equivalence_manifest.json"),
            "whole_book_comparison": display_path(output_root / "comparison" / "whole_book_comparison.json"),
            "high_confidence_comparison": display_path(output_root / "comparison" / "high_confidence_comparison.json"),
            "page_variance_diagnostic": display_path(output_root / "comparison" / "page_variance_diagnostic.json"),
            "manual_review_ledger": display_path(output_root / "manual_review" / "manual_review_ledger.json"),
            "manual_review_summary": display_path(output_root / "manual_review" / "manual_review_summary.md"),
            "archive_html": display_path(output_root / "shakespeare_archive.html"),
            "archive_pdf": display_path(output_root / "CODEFINDER_Shakespeare_Report.pdf"),
            "shareable_pdf": display_path(shareable_root / "CODEFINDER_Shakespeare_Summary.pdf"),
            "manuscript": "docs/SHAKESPEARE_MANUSCRIPT.md",
            "methods_appendix": "docs/SHAKESPEARE_METHODS_APPENDIX.md",
        },
    }


def render_summary_text(page_manifest: dict, high_conf: dict, page_variance: dict, manual_rows: List[dict]) -> str:
    manual_counts = Counter(row["manual_assessment"] for row in manual_rows)
    lines = [
        "CODEFINDER Shakespeare Canonical Archive Summary",
        "==============================================",
        "",
        f"Status date: {REPORT_DATE}",
        "Claim level: External-safe diagnostic packet",
        "",
        "This archive packet consolidates the Wright/Aspley Shakespeare lane into one cautious package.",
        "It preserves useful computational evidence while treating source-equivalence constraints as unresolved.",
        "",
        "Corpus normalization",
        "-------------------",
        f"- Wright local images: {page_manifest['summary']['wright']['total_images']}",
        f"- Aspley local images: {page_manifest['summary']['aspley']['total_images']}",
        f"- Shared folio-sequence images: {page_manifest['summary']['shared_sequence']['paired_images']}",
        f"- Included shared pairs: {page_manifest['summary']['shared_sequence']['included_pairs']}",
        f"- Shared paratext pairs: {page_manifest['summary']['shared_sequence']['paratext_pairs']}",
        f"- Unmatched Aspley extras: {page_manifest['summary']['aspley']['unmatched']}",
        "",
        "Key findings",
        "------------",
        f"- High-confidence OCR totals remain asymmetric: Wright {high_conf['source1']['total_characters']:,}, Aspley {high_conf['source2']['total_characters']:,}.",
        f"- Legacy page-variance diagnostics flag {page_variance['significant_pairs']} of {page_variance['total_pairs']} shared-sequence pairs as significant.",
        f"- The surviving canonical sonnet mapping remains partial: Wright {page_manifest['summary']['legacy_sonnet_mapping']['wright_found']} detections, Aspley {page_manifest['summary']['legacy_sonnet_mapping']['aspley_found']}, with {page_manifest['summary']['legacy_sonnet_mapping']['same_page']} same-page match.",
        f"- Manual review rows: {len(manual_rows)} total, `same design`={manual_counts.get('same design', 0)}, `possible same sort/block`={manual_counts.get('possible same sort/block', 0)}, `inconclusive`={manual_counts.get('inconclusive', 0)}.",
        "",
        "Interpretation",
        "--------------",
        "- The current packet supports provisional computational evidence and diagnostic comparison only.",
        "- Design similarity should not be read as proof of shared physical type or settled print-state identity.",
        "- Legacy Shakespeare report folders remain useful as inputs, but not as canonical verdicts.",
    ]
    return "\n".join(lines) + "\n"


def render_archive_html(summary_text: str, package_manifest: dict) -> str:
    paragraphs = "".join(f"<p>{escape(line)}</p>" for line in summary_text.splitlines() if line.strip())
    artifact_items = "".join(
        f"<li><strong>{escape(key)}</strong>: {escape(value)}</li>"
        for key, value in package_manifest["artifacts"].items()
    )
    return (
        "<html><head><meta charset='utf-8'><title>CODEFINDER Shakespeare Archive</title>"
        "<style>body{font-family:Georgia,serif;max-width:880px;margin:40px auto;line-height:1.45;color:#1b1b1b;}"
        "h1,h2{font-family:'Times New Roman',serif;} code{background:#f2f2f2;padding:2px 4px;}</style></head><body>"
        f"<h1>CODEFINDER Shakespeare Archive</h1><p><strong>Date:</strong> {escape(REPORT_DATE)}</p>"
        "<p><strong>Status:</strong> External-safe diagnostic packet</p>"
        f"{paragraphs}<h2>Artifact paths</h2><ul>{artifact_items}</ul></body></html>"
    )


def render_pdf(path: Path, title: str, sections: List[Tuple[str, Iterable[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    margin_x = 18 * mm
    y = height - 18 * mm

    def draw_line(text: str, font_name: str = "Helvetica", font_size: int = 10, leading: float = 14.0) -> None:
        nonlocal y
        if y < 20 * mm:
            pdf.showPage()
            y = height - 18 * mm
        pdf.setFont(font_name, font_size)
        pdf.drawString(margin_x, y, text)
        y -= leading

    draw_line(title, font_name="Helvetica-Bold", font_size=16, leading=18)
    draw_line(f"Date: {REPORT_DATE}", font_size=10)
    draw_line("Claim level: External-safe diagnostic packet", font_size=10)
    y -= 4

    for heading, lines in sections:
        draw_line(heading, font_name="Helvetica-Bold", font_size=12, leading=16)
        for line in lines:
            if not line:
                y -= 6
                continue
            wrapped = wrap_text(line, max_chars=95)
            for chunk in wrapped:
                draw_line(chunk, font_name="Helvetica", font_size=10)
        y -= 4

    pdf.save()


def wrap_text(text: str, max_chars: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate canonical Shakespeare lane artifacts.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--shareable-root", default=str(DEFAULT_SHAREABLE_ROOT))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    shareable_root = Path(args.shareable_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    if not shareable_root.is_absolute():
        shareable_root = ROOT / shareable_root
    comparison_root = output_root / "comparison"
    manual_root = output_root / "manual_review"

    output_root.mkdir(parents=True, exist_ok=True)
    comparison_root.mkdir(parents=True, exist_ok=True)
    manual_root.mkdir(parents=True, exist_ok=True)
    shareable_root.mkdir(parents=True, exist_ok=True)

    page_manifest = build_page_equivalence_manifest()
    write_json(output_root / "page_equivalence_manifest.json", page_manifest)

    whole_book = build_comparison_payload(
        "whole_witness",
        WITNESSES["wright"]["stats_path"],
        WITNESSES["aspley"]["stats_path"],
        WITNESSES["wright"]["freq_path"],
        WITNESSES["aspley"]["freq_path"],
        page_manifest,
    )
    write_json(comparison_root / "whole_book_comparison.json", whole_book)

    high_conf = build_comparison_payload(
        "matched_high_confidence",
        WITNESSES["wright"]["high_conf_stats_path"],
        WITNESSES["aspley"]["high_conf_stats_path"],
        WITNESSES["wright"]["high_conf_freq_path"],
        WITNESSES["aspley"]["high_conf_freq_path"],
        page_manifest,
    )
    write_json(comparison_root / "high_confidence_comparison.json", high_conf)

    page_variance = build_page_variance_payload(page_manifest)
    write_json(comparison_root / "page_variance_diagnostic.json", page_variance)

    manual_rows = build_manual_review_ledger(page_manifest, page_variance)
    write_json(manual_root / "manual_review_ledger.json", manual_rows)
    (manual_root / "manual_review_summary.md").write_text(render_manual_review_summary(manual_rows))
    (manual_root / "manual_review_ledger.md").write_text(render_manual_review_ledger_md(manual_rows))

    package_manifest = build_package_manifest(
        page_manifest,
        whole_book,
        high_conf,
        page_variance,
        manual_rows,
        output_root,
        shareable_root,
    )
    write_json(output_root / "package_manifest.json", package_manifest)

    summary_text = render_summary_text(page_manifest, high_conf, page_variance, manual_rows)
    (output_root / "summary.txt").write_text(summary_text)
    (output_root / "shakespeare_archive.html").write_text(render_archive_html(summary_text, package_manifest))

    archive_sections = [
        (
            "Corpus normalization",
            [
                f"Wright local images: {page_manifest['summary']['wright']['total_images']}",
                f"Aspley local images: {page_manifest['summary']['aspley']['total_images']}",
                f"Shared folio-sequence images: {page_manifest['summary']['shared_sequence']['paired_images']}",
                f"Included shared pairs: {page_manifest['summary']['shared_sequence']['included_pairs']}",
                f"Unmatched Aspley extras: {page_manifest['summary']['aspley']['unmatched']}",
            ],
        ),
        (
            "Key findings",
            [
                f"High-confidence OCR totals: Wright {high_conf['source1']['total_characters']:,}, Aspley {high_conf['source2']['total_characters']:,}.",
                f"Legacy page-variance diagnostic: {page_variance['significant_pairs']} of {page_variance['total_pairs']} pairs flagged.",
                f"Partial sonnet mapping: Wright {page_manifest['summary']['legacy_sonnet_mapping']['wright_found']}, Aspley {page_manifest['summary']['legacy_sonnet_mapping']['aspley_found']}, same-page {page_manifest['summary']['legacy_sonnet_mapping']['same_page']}.",
            ],
        ),
        (
            "Interpretation",
            [
                "This packet presents provisional computational evidence and diagnostic comparison only.",
                "Design similarity is not same-object proof.",
                "The normalized Shakespeare lane supersedes the older contradictory report cluster.",
            ],
        ),
    ]
    render_pdf(output_root / "CODEFINDER_Shakespeare_Report.pdf", "CODEFINDER Shakespeare Archive", archive_sections)

    shareable_sections = [
        (
            "Snapshot",
            [
                "Standalone Shakespeare packet built from the two local Folger witnesses.",
                f"Shared folio-sequence images: {page_manifest['summary']['shared_sequence']['paired_images']}",
                f"Included shared pairs: {page_manifest['summary']['shared_sequence']['included_pairs']}",
                f"Unmatched Aspley extras: {page_manifest['summary']['aspley']['unmatched']}",
            ],
        ),
        (
            "External-safe reading",
            [
                "The current Shakespeare packet offers provisional computational evidence, not a settled attribution.",
                "Source-equivalence constraints remain unresolved because the surviving sonnet mapping is still partial.",
                "Design similarity is not proof of shared physical type.",
            ],
        ),
        (
            "Current metrics",
            [
                f"High-confidence OCR totals: Wright {high_conf['source1']['total_characters']:,}; Aspley {high_conf['source2']['total_characters']:,}.",
                f"Legacy page-variance diagnostic: {page_variance['significant_pairs']} of {page_variance['total_pairs']} paired images flagged.",
                f"Manual review rows: {len(manual_rows)}; no rows currently rated possible same sort or block.",
            ],
        ),
    ]
    render_pdf(
        shareable_root / "CODEFINDER_Shakespeare_Summary.pdf",
        "CODEFINDER Shakespeare Summary",
        shareable_sections,
    )


if __name__ == "__main__":
    main()
