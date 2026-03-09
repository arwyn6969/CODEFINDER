#!/usr/bin/env python3
"""
Generate a Discord-safe summary PDF for the German/Kempten corpus.

Usage:
    python scripts/generate_discord_summary.py
    python scripts/generate_discord_summary.py --output reports/shareable/custom.pdf
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.services.shareable_report_content import (
    COVER_LABELS,
    HEADLINE_VERDICT,
    NEXT_STEPS,
    SOURCE_METADATA,
    SOURCE_ORDER,
    SUMMARY_SCOPE,
    VERDICT_CONTEXT,
    WHAT_WE_CAN_SAY,
    WHAT_WE_CANNOT_SAY,
    validate_shareable_text,
)

BASE_DIR = Path(__file__).resolve().parent.parent
SUMMARY_PATH = BASE_DIR / "reports" / "final_report" / "summary.txt"
SORT_RESULTS_PATH = BASE_DIR / "reports" / "character_sort_match" / "sort_comparison.json"
STATS_RESULTS_PATH = BASE_DIR / "reports" / "statistical_analysis" / "formal_stats.json"
GREENMAN_MATCHES_PATH = BASE_DIR / "reports" / "greenman_scan" / "matches.json"
PROOF_DIR = BASE_DIR / "reports" / "proof_images"
DEFAULT_OUTPUT_PATH = BASE_DIR / "reports" / "shareable" / "CODEFINDER_Discord_Summary.pdf"

NAVY = HexColor("#10324a")
TEAL = HexColor("#3f6e70")
GOLD = HexColor("#b48a39")
LIGHT_GREY = HexColor("#edf1f3")
MID_GREY = HexColor("#d7e0e6")
DARK = HexColor("#1f2a30")
GREEN = HexColor("#dfeee1")
RED = HexColor("#f6e2de")
WHITE = HexColor("#ffffff")

DISPLAY_TO_SOURCE = {
    "BSB Munich": "bsb_munich_10057380",
    "GDZ Göttingen": "gdz_goettingen_ppn777246686",
    "GDZ Goettingen": "gdz_goettingen_ppn777246686",
    "HAB Wolfenbüttel": "hab_wolfenbuettel_178_1_theol_1s",
    "HAB Wolfenbuettel": "hab_wolfenbuettel_178_1_theol_1s",
    "Google Books (Tractatus)": "google_books_tractatus_brevis",
}
SOURCE_ORDER_INDEX = {source_name: idx for idx, source_name in enumerate(SOURCE_ORDER)}


@dataclass(frozen=True)
class InventoryRow:
    source_name: str
    pages: int
    chars: int


@dataclass(frozen=True)
class PairwiseRow:
    pair_key: str
    left_source: str
    right_source: str
    avg_similarity: float
    characters_compared: int
    ks_width_verdict: str
    ks_height_verdict: str
    chi_verdict: str
    bootstrap_mean: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float


@dataclass(frozen=True)
class GreenmanRow:
    source_name: str
    page: int
    sift_matches: int
    aggregate_score: float
    match_id: str


@dataclass(frozen=True)
class DiscordSummaryData:
    inventory: list[InventoryRow]
    total_pages: int
    total_chars: int
    pairwise_rows: list[PairwiseRow]
    greenman_rows: list[GreenmanRow]
    all_ks_different: bool
    all_chi_different: bool
    bootstrap_min: float
    bootstrap_max: float


def build_styles():
    styles = getSampleStyleSheet()
    definitions = {
        "CoverTitle": dict(
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            textColor=DARK,
            spaceAfter=3 * mm,
        ),
        "CoverSub": dict(
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            textColor=TEAL,
            spaceAfter=4 * mm,
        ),
        "Section": dict(
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=NAVY,
            spaceBefore=8 * mm,
            spaceAfter=3 * mm,
        ),
        "Subsection": dict(
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=TEAL,
            spaceBefore=4 * mm,
            spaceAfter=2 * mm,
        ),
        "Body": dict(
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=DARK,
            alignment=TA_LEFT,
            spaceAfter=2 * mm,
        ),
        "BodyBold": dict(
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=13,
            textColor=DARK,
            alignment=TA_LEFT,
            spaceAfter=2 * mm,
        ),
        "Small": dict(
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=DARK,
            alignment=TA_LEFT,
            spaceAfter=1.5 * mm,
        ),
        "Caption": dict(
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=9,
            textColor=TEAL,
            alignment=TA_CENTER,
            spaceBefore=1 * mm,
            spaceAfter=3 * mm,
        ),
        "PanelTitle": dict(
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=1.5 * mm,
        ),
    }
    for name, config in definitions.items():
        styles.add(ParagraphStyle(name, parent=styles["Normal"], **config))
    return styles


def parse_summary_inventory(summary_path: Path) -> tuple[list[InventoryRow], int, int]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary artifact: {summary_path}")

    inventory_rows = []
    total_pages = None
    total_chars = None
    pattern = re.compile(r"^\s{2}(.*?)\s+(\d+)\s+pages\s+(\d+)\s+chars$")

    for raw_line in summary_path.read_text().splitlines():
        match = pattern.match(raw_line)
        if not match:
            continue
        display_name, pages_text, chars_text = match.groups()
        pages = int(pages_text)
        chars = int(chars_text)
        if display_name == "TOTAL":
            total_pages = pages
            total_chars = chars
            continue
        source_name = DISPLAY_TO_SOURCE.get(display_name)
        if not source_name:
            continue
        inventory_rows.append(InventoryRow(source_name=source_name, pages=pages, chars=chars))

    if total_pages is None or total_chars is None:
        raise ValueError("Could not parse total pages/chars from summary.txt")

    inventory_rows.sort(key=lambda row: SOURCE_ORDER_INDEX[row.source_name])
    return inventory_rows, total_pages, total_chars


def load_pairwise_rows(sort_results_path: Path, stats_results_path: Path) -> list[PairwiseRow]:
    if not sort_results_path.exists():
        raise FileNotFoundError(f"Missing sort artifact: {sort_results_path}")
    if not stats_results_path.exists():
        raise FileNotFoundError(f"Missing stats artifact: {stats_results_path}")

    raw_sort_results = json.loads(sort_results_path.read_text())
    stats_results = json.loads(stats_results_path.read_text())
    pair_scores = defaultdict(list)

    for result in raw_sort_results:
        for pair_key, metrics in result.get("pairwise", {}).items():
            pair_scores[pair_key].append(metrics["combined_score"])

    rows = []
    for pair_key, scores in pair_scores.items():
        left_source, right_source = pair_key.split(" vs ")
        ks_metrics = stats_results["ks_test"][pair_key]
        chi_metrics = stats_results["chi_squared"][pair_key]
        bootstrap_metrics = stats_results["bootstrap"][pair_key]
        rows.append(
            PairwiseRow(
                pair_key=pair_key,
                left_source=left_source,
                right_source=right_source,
                avg_similarity=statistics.fmean(scores),
                characters_compared=len(scores),
                ks_width_verdict=ks_metrics["verdict_width"],
                ks_height_verdict=ks_metrics["verdict_height"],
                chi_verdict=chi_metrics["verdict"],
                bootstrap_mean=bootstrap_metrics["mean_similarity"],
                bootstrap_ci_low=bootstrap_metrics["ci_95_lower"],
                bootstrap_ci_high=bootstrap_metrics["ci_95_upper"],
            )
        )

    rows.sort(key=lambda row: (
        SOURCE_ORDER_INDEX[row.left_source],
        SOURCE_ORDER_INDEX[row.right_source],
    ))
    return rows


def load_greenman_rows(matches_path: Path) -> list[GreenmanRow]:
    if not matches_path.exists():
        return []

    raw_matches = json.loads(matches_path.read_text())
    rows = []
    for match in raw_matches:
        rows.append(
            GreenmanRow(
                source_name=match["source"],
                page=int(match["page"]),
                sift_matches=int(match["sift_matches"]),
                aggregate_score=float(match["aggregate_score"]),
                match_id=match["match_id"],
            )
        )
    rows.sort(key=lambda row: (SOURCE_ORDER_INDEX.get(row.source_name, 999), row.page))
    return rows


def load_reviewed_pairs_manifest(proof_dir: Path = PROOF_DIR) -> list[dict]:
    path = proof_dir / "reviewed_pairs_manifest.json"
    if not path.exists():
        return []

    rows = json.loads(path.read_text())
    return [
        row for row in rows
        if (proof_dir / row.get("file", "")).exists()
    ]


def build_report_data(base_dir: Path = BASE_DIR) -> DiscordSummaryData:
    inventory_rows, total_pages, total_chars = parse_summary_inventory(base_dir / "reports" / "final_report" / "summary.txt")
    pairwise_rows = load_pairwise_rows(
        base_dir / "reports" / "character_sort_match" / "sort_comparison.json",
        base_dir / "reports" / "statistical_analysis" / "formal_stats.json",
    )
    greenman_rows = load_greenman_rows(base_dir / "reports" / "greenman_scan" / "matches.json")

    if len(greenman_rows) != 1:
        raise ValueError(f"Expected exactly one verified Greenman candidate, found {len(greenman_rows)}")

    all_ks_different = all(
        row.ks_width_verdict == "DIFFERENT" and row.ks_height_verdict == "DIFFERENT"
        for row in pairwise_rows
    )
    all_chi_different = all(row.chi_verdict == "DIFFERENT" for row in pairwise_rows)
    bootstrap_means = [row.bootstrap_mean for row in pairwise_rows]

    return DiscordSummaryData(
        inventory=inventory_rows,
        total_pages=total_pages,
        total_chars=total_chars,
        pairwise_rows=pairwise_rows,
        greenman_rows=greenman_rows,
        all_ks_different=all_ks_different,
        all_chi_different=all_chi_different,
        bootstrap_min=min(bootstrap_means),
        bootstrap_max=max(bootstrap_means),
    )


def pair_label(pair_row: PairwiseRow) -> str:
    left = SOURCE_METADATA[pair_row.left_source].display_name
    right = SOURCE_METADATA[pair_row.right_source].display_name
    return f"{left} vs {right}"


def render_validation_text(report_data: DiscordSummaryData) -> str:
    fragments = [
        "CODEFINDER Discord Summary",
        SUMMARY_SCOPE,
        HEADLINE_VERDICT,
        VERDICT_CONTEXT,
        *COVER_LABELS.values(),
        *WHAT_WE_CAN_SAY,
        *WHAT_WE_CANNOT_SAY,
        *NEXT_STEPS,
    ]

    for row in report_data.inventory:
        meta = SOURCE_METADATA[row.source_name]
        fragments.append(
            f"{meta.display_name} {meta.date} {meta.short_title} {meta.institution} "
            f"{row.pages} pages {row.chars} chars"
        )

    for row in report_data.greenman_rows:
        meta = SOURCE_METADATA[row.source_name]
        fragments.append(
            f"Foliate-head GDZ only {meta.display_name} page {row.page} "
            f"SIFT {row.sift_matches} fingerprint {row.aggregate_score:.3f}"
        )

    for row in report_data.pairwise_rows:
        fragments.append(
            f"{pair_label(row)} {row.avg_similarity:.3f} {row.characters_compared} "
            f"KS {row.ks_width_verdict}/{row.ks_height_verdict} "
            f"chi {row.chi_verdict} bootstrap {row.bootstrap_mean:.3f}"
        )

    return "\n".join(fragments)


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(TEAL)
    canvas.drawString(16 * mm, 10 * mm, "CODEFINDER Discord Summary")
    canvas.drawRightString(A4[0] - 16 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def safe_image(path: Path, max_width: float, max_height: float):
    if not path.exists():
        return None
    try:
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            return None
        height, width = image.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)
        return RLImage(str(path), width=width * scale, height=height * scale)
    except Exception:
        return None


def image_row(items: list[tuple[Path, str]], styles, max_width: float, max_height: float, col_width: float):
    cells = []
    for path, caption in items:
        image = safe_image(path, max_width, max_height)
        if image is None:
            continue
        cell = Table(
            [[image], [Paragraph(caption, styles["Caption"])]],
            colWidths=[col_width],
        )
        cell.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        cells.append(cell)

    if not cells:
        return None

    row = Table([cells], colWidths=[col_width] * len(cells))
    row.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return row


def panel_table(title: str, items: tuple[str, ...], styles, background):
    body = "<br/>".join(f"- {item}" for item in items)
    panel = Table(
        [[Paragraph(title, styles["PanelTitle"])], [Paragraph(body, styles["Body"])]],
        colWidths=[84 * mm],
    )
    panel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 0.6, MID_GREY),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return panel


def metadata_badges(styles):
    rows = []
    for label, value in COVER_LABELS.items():
        rows.append([
            Paragraph(f"<b>{label}</b>", styles["Small"]),
            Paragraph(value, styles["Small"]),
        ])

    badges = Table(rows, colWidths=[28 * mm, 124 * mm])
    badges.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY),
        ("BOX", (0, 0), (-1, -1), 0.6, MID_GREY),
        ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return badges


def build_pdf_story(report_data: DiscordSummaryData):
    styles = build_styles()
    story = []

    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("CODEFINDER Discord Summary", styles["CoverTitle"]))
    story.append(Paragraph("Standalone Kempten packet", styles["CoverSub"]))
    story.append(metadata_badges(styles))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(SUMMARY_SCOPE, styles["Body"]))
    story.append(Paragraph(
        f"The cleaned rerun summarized here covers {len(report_data.inventory)} sources, "
        f"{report_data.total_pages:,} pages, and {report_data.total_chars:,} extracted characters.",
        styles["Body"],
    ))

    story.append(Paragraph("1. Scope and Verdict", styles["Section"]))
    verdict_box = Table(
        [[Paragraph(HEADLINE_VERDICT, styles["BodyBold"])], [Paragraph(VERDICT_CONTEXT, styles["Body"])]],
        colWidths=[170 * mm],
    )
    verdict_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY),
        ("BOX", (0, 0), (-1, -1), 0.8, GOLD),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(verdict_box)
    story.append(Spacer(1, 4 * mm))
    story.append(Table(
        [[panel_table("What we can say", WHAT_WE_CAN_SAY, styles, GREEN),
          panel_table("What we cannot say yet", WHAT_WE_CANNOT_SAY, styles, RED)]],
        colWidths=[89 * mm, 89 * mm],
    ))

    story.append(Paragraph("2. Source inventory and corrected provenance", styles["Section"]))
    story.append(Paragraph(
        "The corrected source dates below supersede the approximate dates that still appear in the archived forensic PDF.",
        styles["Body"],
    ))
    inventory_headers = ["Source", "Date", "Work", "Holding / record", "Pages", "Chars"]
    inventory_rows = []
    for row in report_data.inventory:
        meta = SOURCE_METADATA[row.source_name]
        inventory_rows.append([
            Paragraph(meta.display_name, styles["Small"]),
            Paragraph(str(meta.date), styles["Small"]),
            Paragraph(meta.short_title, styles["Small"]),
            Paragraph(f"{meta.institution}<br/>{meta.record_id}", styles["Small"]),
            Paragraph(f"{row.pages:,}", styles["Small"]),
            Paragraph(f"{row.chars:,}", styles["Small"]),
        ])
    inventory_rows.append([
        Paragraph("<b>Total</b>", styles["Small"]),
        Paragraph("-", styles["Small"]),
        Paragraph("Corpus total", styles["Small"]),
        Paragraph("-", styles["Small"]),
        Paragraph(f"<b>{report_data.total_pages:,}</b>", styles["Small"]),
        Paragraph(f"<b>{report_data.total_chars:,}</b>", styles["Small"]),
    ])
    inventory_table = Table(
        [inventory_headers] + inventory_rows,
        colWidths=[25 * mm, 13 * mm, 52 * mm, 52 * mm, 18 * mm, 18 * mm],
        repeatRows=1,
    )
    inventory_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ("BOX", (0, 0), (-1, -1), 0.5, MID_GREY),
        ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(inventory_table)

    story.append(Paragraph("3. Differences and evidence", styles["Section"]))
    story.append(Paragraph("Foliate-head ornament evidence", styles["Subsection"]))
    greenman = report_data.greenman_rows[0]
    greenman_meta = SOURCE_METADATA[greenman.source_name]
    story.append(Paragraph(
        f"One verified foliate-head candidate is present in {greenman_meta.display_name} only. "
        "This is source-specific evidence and is not presented here as a cross-source woodblock claim.",
        styles["Body"],
    ))
    greenman_table = Table(
        [
            ["Source", "Page", "SIFT matches", "Fingerprint", "Reading"],
            [
                greenman_meta.display_name,
                str(greenman.page),
                f"{greenman.sift_matches:,}",
                f"{greenman.aggregate_score:.3f}",
                "Verified candidate under current thresholds",
            ],
        ],
        colWidths=[34 * mm, 16 * mm, 28 * mm, 24 * mm, 70 * mm],
    )
    greenman_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("BACKGROUND", (0, 1), (-1, -1), WHITE),
        ("BOX", (0, 0), (-1, -1), 0.5, MID_GREY),
        ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(greenman_table)

    greenman_images = image_row(
        [
            (PROOF_DIR / "greenman_context_gdz.jpg", "GDZ page context (candidate highlighted)"),
            (PROOF_DIR / "greenman_crop_gdz.jpg", "Extracted crop of the verified candidate"),
        ],
        styles,
        max_width=82 * mm,
        max_height=58 * mm,
        col_width=84 * mm,
    )
    if greenman_images:
        story.append(Spacer(1, 3 * mm))
        story.append(greenman_images)
        story.append(Paragraph("Page context and extracted crop for the only verified GDZ foliate-head candidate.", styles["Caption"]))

    reviewed_pairs = load_reviewed_pairs_manifest(PROOF_DIR)
    provisional_pair = next(
        (row for row in reviewed_pairs if row.get("assessment") == "possible same sort/block"),
        None,
    )
    control_pair = next(
        (row for row in reviewed_pairs if row.get("assessment") == "same design"),
        None,
    )
    exemplar_items = []
    if provisional_pair:
        exemplar_items.append((PROOF_DIR / provisional_pair["file"], provisional_pair["caption"]))
    if control_pair:
        exemplar_items.append((PROOF_DIR / control_pair["file"], control_pair["caption"]))

    if exemplar_items:
        story.append(Paragraph("Reviewed character exemplars", styles["Subsection"]))
        story.append(Paragraph(
            "These panels come from the manual-review ledger. They separate one provisional possible same "
            "sort/block reading from one same-design control so the reader can see the distinction used in the manuscript.",
            styles["Body"],
        ))
        exemplar_row = image_row(
            exemplar_items,
            styles,
            max_width=82 * mm,
            max_height=48 * mm,
            col_width=84 * mm,
        )
        if exemplar_row:
            story.append(exemplar_row)
            story.append(Paragraph(
                "Reviewed manual-review exemplars showing a provisional pair and a same-design control.",
                styles["Caption"],
            ))

    story.append(Paragraph("Character forms and formal tests", styles["Subsection"]))
    story.append(Paragraph(
        "Character-form similarity remains high across all six source pairs. "
        "The bootstrap values below are presented as similarity evidence only, not as proof on their own.",
        styles["Body"],
    ))
    pair_headers = ["Pair", "Sort avg", "Chars", "Bootstrap mean", "95% CI"]
    pair_rows = []
    for row in report_data.pairwise_rows:
        pair_rows.append([
            Paragraph(pair_label(row), styles["Small"]),
            Paragraph(f"{row.avg_similarity:.3f}", styles["Small"]),
            Paragraph(str(row.characters_compared), styles["Small"]),
            Paragraph(f"{row.bootstrap_mean:.3f}", styles["Small"]),
            Paragraph(f"{row.bootstrap_ci_low:.3f} - {row.bootstrap_ci_high:.3f}", styles["Small"]),
        ])
    pairwise_table = Table(
        [pair_headers] + pair_rows,
        colWidths=[62 * mm, 18 * mm, 16 * mm, 24 * mm, 44 * mm],
        repeatRows=1,
    )
    pairwise_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ("BOX", (0, 0), (-1, -1), 0.5, MID_GREY),
        ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(pairwise_table)

    stats_reading = (
        "All pairwise KS verdicts remain DIFFERENT for both width and height, and all pairwise "
        "chi-squared verdicts remain DIFFERENT. This is why the summary talks about similar printed "
        "forms rather than simple sameness."
    )
    stats_panel = Table(
        [[Paragraph(stats_reading, styles["Body"])]],
        colWidths=[170 * mm],
    )
    stats_panel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY),
        ("BOX", (0, 0), (-1, -1), 0.6, MID_GREY),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(Spacer(1, 3 * mm))
    story.append(stats_panel)

    story.append(Paragraph("4. Caveats and next steps", styles["Section"]))
    story.append(Paragraph(
        "Damage chronology is excluded from the main verdict in this PDF. The corrected rerun remains "
        "diagnostic only, so that line of evidence stays outside the shareable claim.",
        styles["Body"],
    ))
    next_steps_body = "<br/>".join(f"- {item}" for item in NEXT_STEPS)
    story.append(KeepTogether([
        Paragraph("Next steps before a stronger claim", styles["Subsection"]),
        Paragraph(next_steps_body, styles["Body"]),
        Paragraph(
            "This PDF stands on its own for informal sharing, but it is intentionally narrower than the full "
            "forensic report and keeps the Kempten packet self-contained.",
            styles["Body"],
        ),
    ]))

    return story


def generate_discord_summary(output_path: Path = DEFAULT_OUTPUT_PATH, base_dir: Path = BASE_DIR) -> Path:
    report_data = build_report_data(base_dir=base_dir)
    validate_shareable_text(render_validation_text(report_data))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="CODEFINDER Discord Summary",
        author="OpenAI Codex",
    )
    story = build_pdf_story(report_data)
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate the Discord-safe German/Kempten summary PDF.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output PDF path (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()
    output_path = generate_discord_summary(output_path=args.output)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
