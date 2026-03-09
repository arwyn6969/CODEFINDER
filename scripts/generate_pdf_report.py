#!/usr/bin/env python3
"""
CODEFINDER — Comprehensive Forensic PDF Report Generator (with Proof Images)
==============================================================================

Generates a professional, extensive PDF report with all evidence,
statistical proofs, embedded proof images, tables, methodology, and findings.

Usage:
    python scripts/generate_pdf_report.py
"""

import sys
import json
import sqlite3
import os
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm, inch
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage, HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import Flowable

from app.services.forensic_report_visuals import (
    CHART_FILES,
    build_bootstrap_rows,
    build_manual_review_rows,
    build_pairwise_similarity_rows,
    generate_report_charts,
    ordered_pair_keys,
    pair_short_label,
)

# ── Colour Palette ──────────────────────────────────────────────────────────
GOLD = HexColor('#c9a94e')
NAVY = HexColor('#0f3460')
GREEN = HexColor('#4caf50')
ORANGE = HexColor('#ff9800')
RED = HexColor('#f44336')
LIGHT_GREY = HexColor('#e0e0e0')
MED_GREY = HexColor('#888888')
DARK_GREY = HexColor('#333333')
WHITE = HexColor('#ffffff')

DISPLAY_NAMES = {
    'bsb_munich_10057380': 'BSB Munich',
    'gdz_goettingen_ppn777246686': 'GDZ Göttingen',
    'hab_wolfenbuettel_178_1_theol_1s': 'HAB Wolfenbüttel',
    'google_books_tractatus_brevis': 'Google Books (Tractatus)',
}

BASE_DIR = Path(__file__).parent.parent
PROOF_DIR = BASE_DIR / 'reports' / 'proof_images'
DEFAULT_DB_PATH = BASE_DIR / "data" / "forensic.db"
GREENMAN_MIN_AREA = 8_000
GREENMAN_MAX_AREA = 3_000_000


class HorizontalRule(Flowable):
    def __init__(self, width, thickness=1, color=GOLD):
        Flowable.__init__(self)
        self.width = width
        self.thickness = thickness
        self.color = color

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)

    def wrap(self, aW, aH):
        return (self.width, self.thickness + 2)


def build_styles():
    styles = getSampleStyleSheet()
    defs = {
        'CoverTitle':  dict(fontSize=28, leading=34, textColor=DARK_GREY, fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=6*mm),
        'CoverSub':    dict(fontSize=14, leading=18, textColor=MED_GREY, fontName='Helvetica', alignment=TA_CENTER, spaceAfter=4*mm),
        'SH':          dict(fontSize=18, leading=22, textColor=HexColor('#1a3a5c'), fontName='Helvetica-Bold', spaceBefore=12*mm, spaceAfter=6*mm),
        'SSH':         dict(fontSize=13, leading=16, textColor=HexColor('#2a5a8c'), fontName='Helvetica-Bold', spaceBefore=8*mm, spaceAfter=4*mm),
        'SSSH':        dict(fontSize=11, leading=14, textColor=HexColor('#3a6a9c'), fontName='Helvetica-Bold', spaceBefore=5*mm, spaceAfter=3*mm),
        'Body':        dict(fontSize=10, leading=14, textColor=DARK_GREY, fontName='Helvetica', alignment=TA_JUSTIFY, spaceAfter=3*mm),
        'Mono':        dict(fontSize=8.5, leading=11, textColor=DARK_GREY, fontName='Courier', spaceAfter=2*mm),
        'Caption':     dict(fontSize=8, leading=10, textColor=MED_GREY, fontName='Helvetica-Oblique', alignment=TA_CENTER, spaceBefore=2*mm, spaceAfter=4*mm),
        'VG':          dict(fontSize=14, leading=18, textColor=GREEN, fontName='Helvetica-Bold', alignment=TA_CENTER, spaceBefore=4*mm, spaceAfter=4*mm),
        'VO':          dict(fontSize=14, leading=18, textColor=ORANGE, fontName='Helvetica-Bold', alignment=TA_CENTER, spaceBefore=4*mm, spaceAfter=4*mm),
        'TOC':         dict(fontSize=11, leading=16, textColor=HexColor('#1a3a5c'), fontName='Helvetica', spaceAfter=2*mm, leftIndent=10*mm),
        'ImgCaption':  dict(fontSize=8, leading=10, textColor=MED_GREY, fontName='Helvetica-Oblique', alignment=TA_CENTER, spaceBefore=1*mm, spaceAfter=5*mm),
        'TableCell':   dict(fontSize=8.5, leading=10.5, textColor=DARK_GREY, fontName='Helvetica', alignment=TA_LEFT, spaceAfter=0),
        'TableCellSm': dict(fontSize=7.5, leading=9.2, textColor=DARK_GREY, fontName='Helvetica', alignment=TA_LEFT, spaceAfter=0),
    }
    for name, kw in defs.items():
        styles.add(ParagraphStyle(name, parent=styles['Normal'], **kw))
    return styles


def make_table(
    headers,
    rows,
    col_widths=None,
    *,
    styles=None,
    body_font_size=8.5,
    header_font_size=9,
    numeric_cols=None,
    wrap_cols=None,
    cell_style_name='TableCell',
    top_padding=4,
    bottom_padding=4,
    left_padding=6,
    right_padding=6,
):
    numeric_cols = set(numeric_cols or [])
    wrap_cols = set(wrap_cols or [])
    cell_style = styles[cell_style_name] if styles else None

    formatted_rows = []
    for row in rows:
        formatted = []
        for col_idx, value in enumerate(row):
            if isinstance(value, Paragraph):
                formatted.append(value)
            elif col_idx in wrap_cols and cell_style is not None:
                formatted.append(Paragraph(str(value), cell_style))
            else:
                formatted.append(value)
        formatted_rows.append(formatted)

    data = [headers] + formatted_rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    table_styles = [
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), header_font_size),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), body_font_size),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, LIGHT_GREY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, HexColor('#f5f5f5')]),
        ('TOPPADDING', (0, 0), (-1, -1), top_padding),
        ('BOTTOMPADDING', (0, 0), (-1, -1), bottom_padding),
        ('LEFTPADDING', (0, 0), (-1, -1), left_padding),
        ('RIGHTPADDING', (0, 0), (-1, -1), right_padding),
    ]
    for col_idx in numeric_cols:
        table_styles.append(('ALIGN', (col_idx, 1), (col_idx, -1), 'RIGHT'))
        table_styles.append(('ALIGN', (col_idx, 0), (col_idx, 0), 'CENTER'))
    t.setStyle(TableStyle(table_styles))
    return t


def format_p_value(value):
    value = float(value)
    if value < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def format_compact_int(value):
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{int(round(value))}"


def safe_image(path, max_width=None, max_height=None):
    """Return RLImage if path exists, else None."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        import cv2
        img = cv2.imread(str(p))
        if img is None:
            return None
        ih, iw = img.shape[:2]
        mw = max_width or 155*mm
        mh = max_height or 100*mm
        scale = min(mw / iw, mh / ih, 1.0)
        return RLImage(str(p), width=iw * scale, height=ih * scale)
    except Exception:
        return RLImage(str(p), width=max_width or 155*mm, height=max_height or 100*mm)


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(MED_GREY)
    canvas.drawCentredString(
        A4[0] / 2, 12*mm,
        f"CODEFINDER — Forensic Print Block Analysis  •  Page {doc.page}  •  {datetime.now().strftime('%B %d, %Y')}"
    )
    canvas.setStrokeColor(GOLD)
    canvas.setLineWidth(0.5)
    canvas.line(20*mm, A4[1] - 15*mm, A4[0] - 20*mm, A4[1] - 15*mm)
    canvas.restoreState()


def gather_data():
    conn = sqlite3.connect(str(DEFAULT_DB_PATH))
    conn.row_factory = sqlite3.Row

    sources = []
    for row in conn.execute("SELECT id, name FROM sources").fetchall():
        sid, name = row['id'], row['name']
        pages = conn.execute("SELECT count(*) FROM pages WHERE source_id=?", (sid,)).fetchone()[0]
        chars = conn.execute("""SELECT count(*) FROM character_instances ci
            JOIN pages p ON ci.page_id=p.id WHERE p.source_id=?""", (sid,)).fetchone()[0]
        crops = conn.execute("""SELECT count(*) FROM sort_images si
            JOIN character_instances ci ON si.character_instance_id=ci.id
            JOIN pages p ON ci.page_id=p.id WHERE p.source_id=?""", (sid,)).fetchone()[0]
        sources.append({'name': name, 'pages': pages, 'chars': chars, 'crops': crops})

    totals = {
        'pages': conn.execute("SELECT count(*) FROM pages").fetchone()[0],
        'chars': conn.execute("SELECT count(*) FROM character_instances").fetchone()[0],
        'crops': conn.execute("SELECT count(*) FROM sort_images").fetchone()[0],
    }

    char_freq = {}
    for src in sources:
        name = src['name']
        rows = conn.execute("""SELECT ci.character, count(*) as cnt
            FROM character_instances ci JOIN pages p ON ci.page_id=p.id
            JOIN sources s ON p.source_id=s.id WHERE s.name=?
            GROUP BY ci.character ORDER BY cnt DESC LIMIT 25""", (name,)).fetchall()
        char_freq[name] = [(r['character'], r['cnt']) for r in rows]
    conn.close()

    sort_path = BASE_DIR / "reports/character_sort_match/sort_comparison.json"
    sort_results = json.loads(sort_path.read_text()) if sort_path.exists() else []

    stats_path = BASE_DIR / "reports/statistical_analysis/formal_stats.json"
    stats_results = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    damage_path = BASE_DIR / "reports/damage_evolution/damage_evolution.json"
    damage_results = json.loads(damage_path.read_text()) if damage_path.exists() else {}

    return sources, totals, char_freq, sort_results, stats_results, damage_results


def load_greenman_results():
    """Load accepted foliate-head matches, filtering out page-sized artefacts."""
    path = BASE_DIR / "reports" / "greenman_scan" / "matches.json"
    if not path.exists():
        return {
            'valid_matches': [],
            'rejected_matches': [],
            'by_source': {},
            'best_match': None,
        }

    raw_matches = json.loads(path.read_text())
    valid_matches = []
    rejected_matches = []
    by_source = defaultdict(list)

    for match in raw_matches:
        bbox = match.get('bbox', {})
        area = int(bbox.get('w', 0)) * int(bbox.get('h', 0))
        if GREENMAN_MIN_AREA <= area <= GREENMAN_MAX_AREA:
            valid_matches.append(match)
            by_source[match.get('source', 'unknown')].append(match)
        else:
            rejected_matches.append(match)

    best_match = max(
        valid_matches,
        key=lambda match: (match.get('aggregate_score', 0), match.get('sift_matches', 0)),
        default=None,
    )

    return {
        'valid_matches': valid_matches,
        'rejected_matches': rejected_matches,
        'by_source': dict(by_source),
        'best_match': best_match,
    }


def load_reviewed_pairs_manifest():
    """Load reviewed proof panels selected for publication-facing figures."""
    path = PROOF_DIR / "reviewed_pairs_manifest.json"
    if not path.exists():
        return []

    rows = json.loads(path.read_text())
    return [
        row for row in rows
        if (PROOF_DIR / row.get('file', '')).exists()
    ]


def summarize_pair_scores(sort_results):
    """Aggregate pairwise character-sort scores into report-ready rows."""
    pair_scores = defaultdict(list)
    for result in sort_results:
        for pair, scores in result.get('pairwise', {}).items():
            pair_scores[pair].append(scores['combined_score'])

    summary = []
    for pair in sorted(pair_scores.keys()):
        scores = pair_scores[pair]
        avg = float(np.mean(scores))
        summary.append({
            'pair': pair,
            'avg': avg,
            'std': float(np.std(scores)),
            'characters': len(scores),
            'verdict': 'SIMILAR_FORMS' if avg > 0.6 else ('UNCLEAR' if avg > 0.5 else 'DIFFERENT'),
        })
    return summary


def collect_analysis_sources(pairwise_summary, stats_results, greenman_results):
    """Collect every source referenced by the comparison artefacts."""
    sources = set()

    for item in pairwise_summary:
        sources.update(item['pair'].split(' vs '))

    for section in ('bootstrap', 'ks_test', 'chi_squared', 'mann_whitney'):
        for pair in stats_results.get(section, {}):
            sources.update(pair.split(' vs '))

    for match in greenman_results['valid_matches']:
        source = match.get('source')
        if source:
            sources.add(source)

    return sources


def display_pair_name(pair: str) -> str:
    """Replace internal source keys with report display names."""
    pair_display = pair
    for raw_name, display_name in DISPLAY_NAMES.items():
        pair_display = pair_display.replace(raw_name, display_name)
    return pair_display


def add_image_row(story, images_and_captions, styles, max_height=70*mm):
    """Add a row of images with captions below each."""
    cells = []
    for img_path, caption in images_and_captions:
        rl = safe_image(img_path, max_width=70*mm, max_height=max_height)
        if rl:
            cells.append([rl, Paragraph(caption, styles['ImgCaption'])])
    if not cells:
        return

    # Layout: side by side
    n = len(cells)
    col_w = 155*mm / n if n > 0 else 155*mm
    row_data = [[c[0] for c in cells], [c[1] for c in cells]]
    t = Table(row_data, colWidths=[col_w] * n)
    t.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, 0), 'BOTTOM'),
        ('VALIGN', (0, 1), (-1, 1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t)


def add_chart_block(story, image_path, caption, max_width, max_height, styles):
    chart = safe_image(image_path, max_width=max_width, max_height=max_height)
    if chart:
        story.append(chart)
        story.append(Paragraph(caption, styles['ImgCaption']))


def build_pairwise_reference_rows(pairwise_rows):
    return [
        [
            row['label'],
            f"{row['avg_similarity']:.3f}",
            str(row['characters_compared']),
            row['verdict'].replace('_', ' '),
        ]
        for row in pairwise_rows
    ]


def build_top_character_rows(sort_results):
    char_best = []
    for result in sort_results:
        best = max((metrics['combined_score'] for metrics in result.get('pairwise', {}).values()), default=0)
        nsrc = len(result.get('sources', {}))
        char_best.append((result['character'], best, nsrc))
    char_best.sort(key=lambda item: item[1], reverse=True)
    return [[f"'{char}'", f"{score:.3f}", str(source_count)] for char, score, source_count in char_best[:20]]


def build_ks_dimension_rows(stats_results, verdict_key, stat_key, p_key):
    rows = []
    for pair_key in ordered_pair_keys(stats_results.get('ks_test', {}).keys()):
        data = stats_results['ks_test'][pair_key]
        rows.append([
            pair_short_label(pair_key),
            f"{data[stat_key]:.4f}",
            format_p_value(data[p_key]),
            data[verdict_key],
        ])
    return rows


def build_chi_rows(stats_results):
    rows = []
    for pair_key in ordered_pair_keys(stats_results.get('chi_squared', {}).keys()):
        data = stats_results['chi_squared'][pair_key]
        rows.append([
            pair_short_label(pair_key),
            f"{data['chi2']:.1f}",
            str(data['degrees_freedom']),
            format_p_value(data['p_value']),
            data['verdict'],
        ])
    return rows


def build_mw_rows(stats_results):
    rows = []
    for pair_key in ordered_pair_keys(stats_results.get('mann_whitney', {}).keys()):
        data = stats_results['mann_whitney'][pair_key]
        rows.append([
            pair_short_label(pair_key),
            format_compact_int(data['u_statistic']),
            format_p_value(data['p_value']),
            f"{data['effect_size']:.3f}",
            data['effect_magnitude'].upper(),
        ])
    return rows


def build_bootstrap_reference_rows(bootstrap_rows):
    rows = []
    for row in bootstrap_rows:
        rows.append([
            row['label'],
            f"{row['mean_similarity']:.3f}",
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]",
            'SIMILAR' if row['supports_threshold'] else 'UNCLEAR',
        ])
    return rows


def build_pdf(output_path=None):
    output_path = Path(output_path) if output_path else BASE_DIR / "reports" / "final_report" / "CODEFINDER_Forensic_Report.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=22*mm, bottomMargin=20*mm,
        title="CODEFINDER — Forensic Print Block Analysis",
        author="CODEFINDER Forensic Pipeline",
    )

    s = build_styles()
    story = []
    pw = A4[0] - 40*mm  # page width

    sources, totals, char_freq, sort_results, stats_results, damage_results = gather_data()
    greenman_results = load_greenman_results()
    reviewed_pairs_manifest = load_reviewed_pairs_manifest()
    manual_review_rows = json.loads((BASE_DIR / "reports" / "manual_review" / "manual_review_ledger.json").read_text())
    generate_report_charts(sort_results, stats_results, manual_review_rows, PROOF_DIR)
    pairwise_summary = build_pairwise_similarity_rows(sort_results)
    manual_review_summary = build_manual_review_rows(manual_review_rows)
    bootstrap_chart_rows = build_bootstrap_rows(stats_results)
    chart_paths = {key: PROOF_DIR / filename for key, filename in CHART_FILES.items()}
    manual_review_totals = {
        'same_design': sum(row['same_design'] for row in manual_review_summary),
        'possible_same_sort_block': sum(row['possible_same_sort_block'] for row in manual_review_summary),
        'inconclusive': sum(row['inconclusive'] for row in manual_review_summary),
    }
    similar_pair_count = sum(1 for item in pairwise_summary if item['verdict'] == 'SIMILAR_FORMS')
    analysis_sources = collect_analysis_sources(pairwise_summary, stats_results, greenman_results)
    inventory_sources = {src['name'] for src in sources}
    missing_inventory_sources = sorted(analysis_sources - inventory_sources)
    bootstrap_means = [
        data['mean_similarity']
        for data in stats_results.get('bootstrap', {}).values()
        if 'mean_similarity' in data
    ]
    bootstrap_range_label = (
        f"{min(bootstrap_means):.3f}–{max(bootstrap_means):.3f}"
        if bootstrap_means else "n/a"
    )
    best_bootstrap_line = "Bootstrap summary unavailable."
    if stats_results.get('bootstrap'):
        best_bootstrap_pair, best_bootstrap = max(
            stats_results['bootstrap'].items(),
            key=lambda item: item[1].get('mean_similarity', 0),
        )
        best_bootstrap_line = (
            f"{display_pair_name(best_bootstrap_pair)} bootstrap mean similarity "
            f"{best_bootstrap['mean_similarity']:.3f} "
            f"[{best_bootstrap['ci_95_lower']:.3f}, {best_bootstrap['ci_95_upper']:.3f}]"
        )
    best_mw_line = "Mann-Whitney summary unavailable."
    if stats_results.get('mann_whitney'):
        best_mw_pair, best_mw = min(
            stats_results['mann_whitney'].items(),
            key=lambda item: item[1].get('effect_size', 1),
        )
        best_mw_line = (
            f"{display_pair_name(best_mw_pair)} Mann-Whitney effect size "
            f"{best_mw['effect_size']:.3f} ({best_mw['effect_magnitude']})"
        )
    best_greenman = greenman_results['best_match']
    greenman_sources = sorted(greenman_results['by_source'].keys())
    damage_verdict = damage_results.get('verdict', {}) if damage_results else {}
    damage_strength = damage_verdict.get('evidence_strength', 'DIAGNOSTIC')
    damage_summary_value = (
        f"{damage_verdict.get('increasing_damage', 0)}/{damage_verdict.get('total_metrics', 0)} metrics increasing; diagnostic only"
        if damage_verdict else
        "Diagnostic only"
    )
    damage_summary_line = damage_verdict.get(
        'interpretation',
        "Damage evolution remains diagnostic and is excluded from the main verdict.",
    )

    if greenman_sources:
        if len(greenman_sources) == 1:
            woodblock_finding = (
                f"Verified only in {DISPLAY_NAMES.get(greenman_sources[0], greenman_sources[0])}"
            )
        else:
            woodblock_finding = (
                "Verified in "
                + ", ".join(DISPLAY_NAMES.get(src, src) for src in greenman_sources)
            )
    else:
        woodblock_finding = "No verified foliate-head match"

    if greenman_results['rejected_matches']:
        woodblock_finding += (
            f"; {len(greenman_results['rejected_matches'])} oversized artefact"
            f"{'' if len(greenman_results['rejected_matches']) == 1 else 's'} filtered"
        )

    if best_greenman:
        greenman_summary_line = (
            f"A verified foliate-head candidate is present only in "
            f"{DISPLAY_NAMES.get(best_greenman['source'], best_greenman['source'])} "
            f"(SIFT {best_greenman['sift_matches']:,}, fingerprint {best_greenman['aggregate_score']:.3f})."
        )
    else:
        greenman_summary_line = "No verified foliate-head candidate passed the current thresholds."

    inventory_note = None
    if missing_inventory_sources:
        inventory_note = (
            "Inventory note: the local OCR database covers "
            f"{len(inventory_sources)} source(s), while derived comparison artefacts also reference "
            f"{', '.join(DISPLAY_NAMES.get(src, src) for src in missing_inventory_sources)}. "
            "Treat the inventory table as database-backed only."
        )

    # ═════════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 40*mm))
    story.append(Paragraph("CODEFINDER", s['CoverTitle']))
    story.append(HorizontalRule(pw, 2, GOLD))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("Forensic Print Block Analysis", s['CoverSub']))
    story.append(Paragraph("Comprehensive Evidence Report", s['CoverSub']))
    story.append(Spacer(1, 12*mm))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%A, %B %d, %Y at %H:%M')}", s['CoverSub']))
    story.append(Paragraph(
        f"{totals['chars']:,} characters extracted from {totals['pages']} pages across {len(sources)} sources",
        s['CoverSub']))
    story.append(Spacer(1, 15*mm))

    summary_data = [
        ['Category', 'Finding', 'Confidence'],
        ['Foliate-Head Match', woodblock_finding, 'PROVISIONAL'],
        ['Character-Form Match', f'{similar_pair_count}/{len(pairwise_summary)} pairs = SIMILAR FORMS', 'STRONG'],
        ['Damage Evolution', damage_summary_value, damage_strength],
        ['Statistical Bootstrap', f'Mean similarity {bootstrap_range_label}', 'HIGH'],
    ]
    t = Table(summary_data, colWidths=[55*mm, 60*mm, 40*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('GRID', (0, 0), (-1, -1), 0.5, LIGHT_GREY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, HexColor('#f0f4f8')]),
        ('ALIGN', (2, 0), (2, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TEXTCOLOR', (2, 1), (2, 1), GREEN),
        ('TEXTCOLOR', (2, 2), (2, 2), GREEN),
        ('TEXTCOLOR', (2, 3), (2, 3), ORANGE),
        ('TEXTCOLOR', (2, 4), (2, 4), GREEN),
        ('FONTNAME', (2, 1), (2, -1), 'Helvetica-Bold'),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Table of Contents", s['SH']))
    story.append(HorizontalRule(pw, 1))
    story.append(Spacer(1, 4*mm))
    for item in [
        "1. Executive Summary", "2. Source Material & Acquisition",
        "3. OCR Pipeline & Methodology", "4. Foliate-Head Ornament Analysis",
        "5. Character Sort Matching", "6. Formal Statistical Tests",
        "7. Type Measurements", "8. Damage Evolution Tracking",
        "9. Combined Evidence & Final Verdict", "10. Methodology Notes",
        "11. Limitations & Caveats", "12. Appendices",
    ]:
        story.append(Paragraph(item, s['TOC']))
    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # 1. EXECUTIVE SUMMARY
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("1. Executive Summary", s['SH']))
    story.append(HorizontalRule(pw, 1))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "This report presents the results of a computational forensic analysis of four digitised "
        "early modern German and Latin printed books (1609–1616). The investigation asked whether "
        "these publications preserve evidence consistent with shared or transferred printing materials "
        "under a provisional computational reading.", s['Body']))
    story.append(Paragraph(
        "The present evidence supports a provisional shared-materials hypothesis. It is strongest at the "
        "level of recurring character forms and weaker at the level of ornament reuse, so it should not be "
        "read as proof of identical physical sorts or corpus-wide woodblock reuse.", s['Body']))

    story.append(Paragraph("Key Findings:", s['SSH']))
    for f in [
        f"<b>Foliate-Head Ornament (SOURCE-SPECIFIC):</b> {greenman_summary_line} "
        "This should be presented as source-specific evidence, not yet as proof of a shared block across all witnesses.",
        f"<b>Character-Form Similarity (STRONG):</b> {len(sort_results)} distinct character types were compared across "
        f"{len(analysis_sources)} sources. {similar_pair_count}/{len(pairwise_summary)} pairwise averages exceed the 0.60 threshold and are treated here as similar printed forms, not proof of identical sorts.",
        "<b>Statistical Tests (HIGH):</b> Bootstrap mean similarities "
        f"range from {bootstrap_range_label}. Mann-Whitney U tests remain most conservative when scan-resolution differences are large.",
        f"<b>Damage Evolution ({damage_strength}):</b> {damage_summary_line}"
    ]:
        story.append(Paragraph(f"• {f}", s['Body']))
    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # 2. SOURCE MATERIAL — with sample page images
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("2. Source Material & Acquisition", s['SH']))
    story.append(HorizontalRule(pw, 1))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "Source images were acquired from four European digital library services. Below are selected "
        "early text-bearing pages chosen for legibility and cross-source comparability; they are "
        "illustrative rather than statistically representative samples:", s['Body']))

    # Source page images — 2×2 grid
    page_imgs = [
        (PROOF_DIR / 'source_page_bsb_munich.jpg', 'BSB Munich (1616)'),
        (PROOF_DIR / 'source_page_gdz_gottingen.jpg', 'GDZ Göttingen (1609)'),
        (PROOF_DIR / 'source_page_hab_wolfenbuttel.jpg', 'HAB Wolfenbüttel (1616)'),
        (PROOF_DIR / 'source_page_google_books.jpg', 'Google Books (1613)'),
    ]
    # Row 1
    add_image_row(story, page_imgs[:2], s, max_height=60*mm)
    # Row 2
    add_image_row(story, page_imgs[2:], s, max_height=60*mm)
    story.append(Paragraph("Figure 1: Selected early text-bearing pages from each of the four source publications.", s['ImgCaption']))

    story.append(Paragraph("2.1 Inventory", s['SSH']))
    src_rows = []
    for src in sources:
        dn = DISPLAY_NAMES.get(src['name'], src['name'])
        src_rows.append([dn, str(src['pages']), f"{src['chars']:,}", f"{src['crops']:,}"])
    src_rows.append(['TOTAL', str(totals['pages']), f"{totals['chars']:,}", f"{totals['crops']:,}"])
    story.append(make_table(
        ['Source', 'Pages', 'Characters', 'Crop Images'], src_rows,
        col_widths=[50*mm, 25*mm, 30*mm, 30*mm]))
    story.append(Paragraph("Table 1: Database inventory.", s['Caption']))
    if inventory_note:
        story.append(Paragraph(inventory_note, s['Body']))

    story.append(Paragraph("2.2 Provenance", s['SSH']))
    story.append(make_table(
        ['Display Name', 'Institution', 'Catalogue ID', 'Date', 'Method'],
        [['BSB Munich', 'Bayerische Staatsbibliothek', 'bsb10057380', '1616', 'IIIF v2'],
         ['GDZ Göttingen', 'Göttinger Digitalisierungszentrum', 'PPN777246686', '1609', 'IIIF v2'],
         ['HAB Wolfenbüttel', 'Herzog August Bibliothek', '178-1-theol-1s', '1616', 'HTTP scrape'],
         ['Google Books', 'Google Books', 'uThoAAAAcAAJ', '1613', 'PDF extraction']],
        col_widths=[32*mm, 45*mm, 30*mm, 18*mm, 28*mm]))
    story.append(Paragraph("Table 2: Source provenance.", s['Caption']))

    story.append(Paragraph("2.3 DPI Normalisation", s['SSH']))
    story.append(Paragraph(
        "Source images varied significantly in resolution. All images were rescaled to a common height "
        "of 2400 pixels using Lanczos4 interpolation before OCR processing:", s['Body']))
    story.append(make_table(
        ['Source', 'Native Res.', 'Target', 'Scale', 'Note'],
        [['BSB Munich', '2557 × 1906', '2400px', '0.94×', 'Minimal downscale'],
         ['GDZ Göttingen', '2264 × 1848', '2400px', '1.06×', 'Slight upscale'],
         ['Google Books', '1682 × 1075', '2400px', '1.43×', 'Moderate upscale'],
         ['HAB Wolfenbüttel', '1225 × 1024', '2400px', '1.96×', 'Significant upscale']],
        col_widths=[32*mm, 30*mm, 20*mm, 18*mm, 38*mm]))
    story.append(Paragraph("Table 3: DPI normalisation applied.", s['Caption']))
    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # 3. OCR PIPELINE
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("3. OCR Pipeline & Methodology", s['SH']))
    story.append(HorizontalRule(pw, 1))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "Character extraction uses Tesseract OCR v5 with the LSTM engine (--oem 1) and a combined "
        "language model stack: <font face='Courier'>frk+deu+eng</font>.", s['Body']))
    story.append(make_table(
        ['Parameter', 'Value'],
        [['OCR Engine', 'Tesseract v5, LSTM (--oem 1)'],
         ['Language Model', 'frk + deu + eng (combined)'],
         ['Page Segmentation', 'PSM 6 (uniform block)'],
         ['Confidence Threshold', '50% (lowered for Fraktur model)'],
         ['DPI Normalisation', '2400px height (Lanczos4 interpolation)'],
         ['Ligatures', 'fi, fl, ff, ffi, ffl, ct, st, ſi, ſt, ſſ'],
         ['Dimension Storage', 'Normalised dims for comparison; original for crops']],
        col_widths=[45*mm, 110*mm]))
    story.append(Paragraph("Table 4: OCR pipeline configuration.", s['Caption']))

    story.append(Paragraph("3.1 V1 → V2 Improvement", s['SSH']))
    story.append(make_table(
        ['Metric', 'V1 (English)', 'V2 (Fraktur+DPI)', 'Change'],
        [['Total characters', '5,891', '14,165', '+2.4×'],
         ['Characters compared', '46', '84', '+1.8×'],
         ['Matching pairs', '4 / 6', '6 / 6', '+50%'],
         ['GDZ↔HAB similarity', '0.571 (UNCLEAR)', '0.655 (SIMILAR FORMS)', 'Fixed'],
         ['BSB↔HAB similarity', '0.597 (UNCLEAR)', '0.682 (SIMILAR FORMS)', 'Fixed']],
        col_widths=[38*mm, 38*mm, 42*mm, 25*mm]))
    story.append(Paragraph("Table 5: V1 vs V2 extraction comparison.", s['Caption']))
    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # 4. GREENMAN WOODBLOCK — with proof images
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("4. Foliate-Head Ornament Analysis", s['SH']))
    story.append(HorizontalRule(pw, 1))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "A foliate-head ornament was used as the reference motif. "
        "Under the current acceptance thresholds, only GDZ Göttingen yields a verified candidate. "
        "This is source-specific evidence only, and oversized page-level artefacts are filtered from the report even if they slipped into older result files.", s['Body']))

    # Show foliate-head page context plus crop
    gm_imgs = [
        (PROOF_DIR / 'greenman_context_gdz.jpg', 'GDZ Göttingen page context (candidate highlighted)'),
        (PROOF_DIR / 'greenman_crop_gdz.jpg', 'Extracted crop of the verified GDZ candidate'),
    ]
    add_image_row(story, gm_imgs, s, max_height=50*mm)
    story.append(Paragraph(
        "Figure 2: Page context and extracted crop for the only currently verified foliate-head candidate.",
        s['ImgCaption']))

    story.append(Spacer(1, 4*mm))

    # SIFT match visualisation
    story.append(Paragraph("4.1 Supplemental SIFT Diagnostic", s['SSH']))
    story.append(Paragraph(
        "SIFT (Scale-Invariant Feature Transform) detects distinctive keypoints in each image and matches "
        "them against the reference. The overlay below is retained as a diagnostic visual aid only: "
        "it shows local keypoint agreement, but it is too visually dense to function as a standalone proof figure.", s['Body']))

    sift_imgs = sorted(PROOF_DIR.glob('sift_match_*.jpg'))
    if sift_imgs:
        sift = safe_image(str(sift_imgs[0]), max_width=pw, max_height=80*mm)
        if sift:
            story.append(sift)
            story.append(Paragraph(
                "Figure 3: Supplemental SIFT keypoint overlay for the verified GDZ foliate-head candidate. "
                "Use this as diagnostic support for the page-context figure, not as independent proof of block identity.",
                s['ImgCaption']))

    story.append(Paragraph("4.2 Matching Results", s['SSH']))
    greenman_rows = []
    for source_name in ['gdz_goettingen_ppn777246686', 'hab_wolfenbuettel_178_1_theol_1s', 'bsb_munich_10057380']:
        label = DISPLAY_NAMES.get(source_name, source_name)
        source_matches = greenman_results['by_source'].get(source_name, [])
        if source_matches:
            best = max(
                source_matches,
                key=lambda match: (match.get('aggregate_score', 0), match.get('sift_matches', 0)),
            )
            greenman_rows.append([
                label,
                str(len(source_matches)),
                f"{best['sift_matches']:,}",
                f"{best['aggregate_score']:.3f}",
                'Verified under current thresholds',
            ])
        else:
            greenman_rows.append([label, '0', '—', '—', 'No verified match'])
    story.append(make_table(
        ['Source', 'Matches', 'Best SIFT', 'Best FP', 'Verdict'],
        greenman_rows,
        col_widths=[38*mm, 18*mm, 22*mm, 20*mm, 50*mm]))
    story.append(Paragraph("Table 6: Foliate-head matching results after filtering oversized artefacts.", s['Caption']))
    if greenman_results['rejected_matches']:
        story.append(Paragraph(
            f"Filtering note: {len(greenman_results['rejected_matches'])} oversized candidate"
            f"{'' if len(greenman_results['rejected_matches']) == 1 else 's'} "
            "were excluded at report time because the bounding box spanned most of a page.",
            s['Body']))
    story.append(Paragraph("⚠ SOURCE-SPECIFIC WOODBLOCK EVIDENCE", s['VO']))
    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # 5. CHARACTER SORT MATCHING — with proof images
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("5. Character Sort Matching", s['SH']))
    story.append(HorizontalRule(pw, 1))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "Individual character forms were extracted via Tesseract OCR, cropped at full resolution, and compared "
        "across sources using the BlockFingerprinter. The panels below are drawn from the manual-review ledger "
        "rather than from arbitrary first-instance crops:", s['Body']))

    # Character comparison proof images
    story.append(Paragraph("5.1 Reviewed Character Comparison Panels", s['SSH']))
    story.append(Paragraph(
        "These reviewed exemplars distinguish between provisional possible same sort/block readings and a "
        "same-design control. They support a cautious manual reading and are not presented as corpus-wide proof "
        "of identical physical sorts:", s['Body']))

    if reviewed_pairs_manifest:
        for start in range(0, len(reviewed_pairs_manifest), 2):
            chunk = reviewed_pairs_manifest[start:start + 2]
            add_image_row(
                story,
                [(PROOF_DIR / row['file'], row['caption']) for row in chunk],
                s,
                max_height=50*mm,
            )
        story.append(Paragraph(
            "Figure 4: Reviewed character-pair exemplars from the manual-review ledger. "
            "Provisional possible same sort/block pairs are shown separately from a same-design control.",
            s['ImgCaption']))
    else:
        story.append(Paragraph(
            "Reviewed comparison panels were not available in reports/proof_images/reviewed_pairs_manifest.json.",
            s['Body']))

    story.append(Paragraph("5.2 Pairwise Similarity Overview", s['SSH']))
    story.append(Paragraph(
        "The chart below keeps the focus on the six pairwise averages and the working threshold of 0.60. "
        "All six pairs remain above that line, but the narrow spread is easier to read here than in a dense table.",
        s['Body']))
    add_chart_block(
        story,
        chart_paths['pairwise_similarity'],
        "Figure 5: Pairwise average character-form similarity by source pair, with the working 0.60 threshold marked.",
        pw,
        78*mm,
        s,
    )

    story.append(Paragraph("5.3 Manual-Review Outcome Balance", s['SSH']))
    story.append(Paragraph(
        f"The fixed manual-review ledger totals {manual_review_totals['same_design']} same-design, "
        f"{manual_review_totals['possible_same_sort_block']} possible same sort/block, and "
        f"{manual_review_totals['inconclusive']} inconclusive rows. The distribution chart makes clear that the "
        "current packet is dominated by design-level resemblance, not by repeated same-object judgments.",
        s['Body']))
    add_chart_block(
        story,
        chart_paths['manual_review_outcomes'],
        "Figure 6: Manual-review outcomes by source pair. The review set is dominated by same-design and inconclusive readings, with only two provisional same-object candidates.",
        pw,
        82*mm,
        s,
    )

    story.append(Paragraph("✓ ALL 6 PAIRS: SIMILAR PRINTED FORMS", s['VG']))
    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # 6. FORMAL STATISTICAL TESTS
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("6. Formal Statistical Tests", s['SH']))
    story.append(HorizontalRule(pw, 1))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "Four formal statistical tests were applied to character dimension and frequency data:", s['Body']))

    if stats_results:
        story.append(Paragraph("6.1 Bootstrap Similarity Intervals", s['SSH']))
        story.append(Paragraph(
            "The bootstrap plot shows the mean similarity and 95% interval for each pair. Every interval remains above the "
            "working 0.60 threshold, which is why bootstrap continues to support strong formal resemblance.",
            s['Body']))
        add_chart_block(
            story,
            chart_paths['bootstrap_forest'],
            "Figure 7: Bootstrap mean similarity with 95% confidence intervals for each source pair.",
            pw,
            78*mm,
            s,
        )

        story.append(Paragraph("6.2 Distributional Verdict Matrix", s['SSH']))
        story.append(Paragraph(
            "The matrix below condenses the tension in the statistics: KS width, KS height, and chi-squared stay divergent "
            "across the whole corpus, while the Mann-Whitney effect sizes are mostly negligible. This is why the report argues "
            "for similar printed forms rather than a simple same/different verdict.",
            s['Body']))
        add_chart_block(
            story,
            chart_paths['formal_test_matrix'],
            "Figure 8: Summary matrix of KS width, KS height, chi-squared, and Mann-Whitney readings by source pair.",
            pw,
            78*mm,
            s,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # 7. TYPE MEASUREMENTS
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("7. Type Measurements", s['SH']))
    story.append(HorizontalRule(pw, 1))
    tm = stats_results.get('type_measurements', {})
    if tm:
        story.append(Paragraph(
            "Typographic measurements computed from OCR bounding boxes. The x-height to cap-height ratio "
            "is resolution-independent and characteristic of a specific typeface design:", s['Body']))
        tm_rows = []
        for src, data in tm.items():
            dn = DISPLAY_NAMES.get(src, src)
            tm_rows.append([dn, str(data['total_chars']),
                f"{data['body_height_mean']:.1f}±{data['body_height_std']:.1f}",
                f"{data['x_height_mean']:.1f}±{data['x_height_std']:.1f}",
                f"{data['set_width_mean']:.1f}±{data['set_width_std']:.1f}",
                f"{data.get('x_to_cap_ratio', 0):.3f}"])
        story.append(make_table(
            ['Source', 'N', 'Body H (px)', 'X-H (px)', 'Set W (px)', 'x/cap'],
            tm_rows, col_widths=[30*mm, 15*mm, 28*mm, 28*mm, 28*mm, 18*mm]))
        story.append(Paragraph("Table 13: Typographic measurements.", s['Caption']))
    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # 8. DAMAGE EVOLUTION — with explanation
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("8. Damage Evolution Tracking", s['SH']))
    story.append(HorizontalRule(pw, 1))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "If physical type was shared across publications, damage (nicks, cracks, ink spread) should "
        "accumulate over time. Five damage metrics were computed for character crops from each source, "
        "ordered chronologically:", s['Body']))

    if damage_results:
        sm = damage_results.get('source_means', {})
        mono = damage_results.get('monotonicity', {})
        verdict = damage_results.get('verdict', {})

        story.append(Paragraph("8.1 Metrics by Source (Chronological)", s['SSH']))
        dmg_rows = []
        for src in sorted(sm.keys(), key=lambda x: sm[x]['date']):
            d = sm[src]
            dn = DISPLAY_NAMES.get(src, src)
            dmg_rows.append([f"{dn} ({d['date']})", str(d['n_samples']),
                f"{d['edge_roughness']:.4f}", f"{d['edge_erosion']:.4f}",
                f"{d['ink_spread']:.4f}", f"{d['crack_density']:.4f}",
                f"{d['entropy']:.4f}"])
        story.append(make_table(
            ['Source (Date)', 'N', 'Edge Rough.', 'Edge Eros.', 'Ink Spread', 'Crack', 'Entropy'],
            dmg_rows, col_widths=[35*mm, 12*mm, 22*mm, 22*mm, 22*mm, 18*mm, 18*mm]))
        story.append(Paragraph("Table 14: Damage metrics in chronological order.", s['Caption']))

        story.append(Paragraph("8.2 Monotonicity Tests (Kendall's τ)", s['SSH']))
        mono_rows = []
        for metric, data in mono.items():
            mono_rows.append([metric.replace('_', ' ').title(), f"τ = {data['kendall_tau']:+.3f}",
                data['trend'], '✓' if data['supports_shared_type'] else '—'])
        story.append(make_table(
            ['Metric', "Kendall's τ", 'Trend', 'Supports?'], mono_rows,
            col_widths=[35*mm, 30*mm, 30*mm, 25*mm]))
        story.append(Paragraph("Table 15: Monotonicity tests for chronological damage.", s['Caption']))

        ev = verdict.get('evidence_strength', '')
        vs = s['VG'] if ev == 'STRONG' else s['VO']
        story.append(Paragraph(f"Damage Evolution: {ev} EVIDENCE", vs))
        story.append(Paragraph(verdict.get('interpretation', ''), s['Body']))
    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # 9. COMBINED EVIDENCE & VERDICT
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("9. Combined Evidence & Final Verdict", s['SH']))
    story.append(HorizontalRule(pw, 1))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph("9.1 Evidence Supporting a Shared-Materials Hypothesis", s['SSH']))
    for item in [
        greenman_summary_line,
        f"{similar_pair_count}/{len(pairwise_summary)} character sort pairs show similar printed forms (all above the 0.60 threshold)",
        best_bootstrap_line,
        best_mw_line,
        "Damage-evolution metrics remain exploratory and do not carry the main verdict on their own",
        f"{totals['chars']:,} extracted characters across {totals['pages']} locally inventoried pages provide the current database-backed footing",
    ]:
        story.append(Paragraph(f"✓  {item}", s['Body']))

    story.append(Paragraph("9.2 Evidence Requiring Caution", s['SSH']))
    for item in [
        "KS tests show DIFFERENT for dimensions (expected with large N)",
        "Chi-squared frequency differences expected (different texts)",
        "4/5 damage metrics do not show chronological increase",
        "Damage chronology remains metadata-sensitive and should be read as diagnostic only",
        "Manual review ledger is complete, but no publication-grade negative control has yet cleared the thresholds and the calibration sweep did not resolve the blocker",
    ]:
        story.append(Paragraph(f"⚠  {item}", s['Body']))

    story.append(Spacer(1, 8*mm))

    # FINAL VERDICT BOX
    vd = [['', ''], ['FORENSIC VERDICT', ''], ['', ''],
          ['', 'Current computational evidence supports a provisional shared-materials hypothesis.'],
          ['', 'Character-sort evidence is strong; foliate-head evidence is currently source-specific.'],
          ['', 'Damage evolution is currently diagnostic only and excluded from the main verdict.'],
          ['', ''],
          ['', 'Present this externally as a cleaned computational draft: promising, provisional, backed by a completed manual-review ledger, and still lacking an accepted publication-grade negative control because the current sort discriminator remains the live blocker.'],
          ['', '']]
    vt = Table(vd, colWidths=[10*mm, 135*mm])
    vt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#f0f8f0')),
        ('BOX', (0, 0), (-1, -1), 2, GREEN),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, 1), 16),
        ('TEXTCOLOR', (0, 1), (-1, 1), GREEN),
        ('ALIGN', (0, 1), (-1, 1), 'CENTER'),
        ('SPAN', (0, 1), (-1, 1)),
        ('FONTNAME', (1, 3), (1, 5), 'Helvetica'),
        ('FONTSIZE', (1, 3), (1, 7), 10),
        ('FONTNAME', (1, 7), (1, 7), 'Helvetica-Oblique'),
        ('TEXTCOLOR', (1, 7), (1, 7), MED_GREY),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(vt)
    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # 10. METHODOLOGY NOTES
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("10. Methodology Notes", s['SH']))
    story.append(HorizontalRule(pw, 1))
    for title, desc in [
        ("Data Acquisition",
         "Four sources acquired via IIIF manifests (BSB, GDZ), HTTP scraping (HAB), and PDF extraction "
         "(Google Books). Pages downloaded as JPEG at maximum resolution."),
        ("Character Extraction",
         "Tesseract OCR (v5, LSTM, PSM 6) with frk+deu+eng. Characters filtered by confidence "
         "(>50%), size (>5px), and aspect ratio. Each instance cropped from original image at full resolution."),
        ("DPI Normalisation",
         "All images rescaled to 2400px height using Lanczos4 interpolation before OCR. Bounding boxes "
         "recorded in normalised coordinates for cross-source comparison."),
        ("Forensic Fingerprinting",
         "BlockFingerprinter computes 7 feature families: Hu Moments, Fourier Contour Descriptors, "
         "Edge Density, Ink Density, Damage Points, LBP Texture Histogram, and Perceptual Hash."),
        ("Cross-Source Matching",
         "SIFT features matched via FLANN (k=2, ratio test 0.70). Sort matching uses per-character "
         "centroid comparison with pairwise fingerprint scoring."),
        ("Statistical Testing",
         "KS, Chi-Squared, Mann-Whitney U, and Bootstrap (1000 resamples, 95% CI). All implemented "
         "without scipy dependency."),
        ("Damage Evolution",
         "Five metrics (edge roughness, erosion, ink spread, crack density, entropy) computed for "
         "character crops. Kendall's τ is used as an exploratory chronology check, not as standalone attribution proof."),
    ]:
        story.append(Paragraph(f"<b>{title}:</b> {desc}", s['Body']))
    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # 11. LIMITATIONS
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("11. Limitations & Caveats", s['SH']))
    story.append(HorizontalRule(pw, 1))
    for title, desc in [
        ("<b>OCR Accuracy:</b>", "Fraktur model improves but doesn't eliminate mis-identifications. "
         "Dedicated models (Calamari, kraken) require Python 3.9–3.12."),
        ("<b>Resolution:</b>", "HAB images upscaled 1.96×, introducing interpolation artefacts."),
        ("<b>Artifact Consistency:</b>", "Some derived comparison artefacts reference a broader source set than the current local OCR inventory. Regenerate them together before formal submission."),
        ("<b>Text Content:</b>", "Character frequency differences reflect textual content, not type."),
        ("<b>Negative Control:</b>", "Three BSB candidates were processed as 61-page slices and rejected under the current sort thresholds; a publication-grade accepted control is still missing."),
        ("<b>Damage Status:</b>", "The chronology rerun now uses corrected local dates, but the result remains diagnostic only and stays out of the main verdict."),
        ("<b>Missing Source:</b>", "Academia.edu PDF (Haslmayr) excluded from analysis."),
    ]:
        story.append(Paragraph(f"{title} {desc}", s['Body']))

    # ═════════════════════════════════════════════════════════════════════════
    # 12. APPENDICES
    # ═════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("12. Appendices", s['SH']))
    story.append(HorizontalRule(pw, 1))

    story.append(Paragraph("A. Character Frequency (Top 25 per Source)", s['SSH']))
    for src_name, freqs in char_freq.items():
        dn = DISPLAY_NAMES.get(src_name, src_name)
        story.append(Paragraph(f"<b>{dn}</b>", s['SSSH']))
        if freqs:
            n = len(freqs[:25])
            mid = (n + 1) // 2
            combined = []
            for i in range(mid):
                row = [f"'{freqs[i][0]}'", str(freqs[i][1])]
                if i + mid < n:
                    row += [f"'{freqs[i+mid][0]}'", str(freqs[i+mid][1])]
                else:
                    row += ['', '']
                combined.append(row)
            story.append(make_table(
                ['Char', 'Count', 'Char', 'Count'],
                combined,
                col_widths=[20*mm, 20*mm, 20*mm, 20*mm],
                styles=s,
                numeric_cols={1, 3},
            ))

    story.append(Paragraph("B. Pairwise Character Reference Tables", s['SSH']))
    story.append(Paragraph(
        "Dense technical tables use abbreviated source-pair labels: BSB, GDZ, Google, and HAB.",
        s['Body']))
    story.append(make_table(
        ['Pair', 'Avg Score', 'Chars', 'Verdict'],
        build_pairwise_reference_rows(pairwise_summary),
        col_widths=[35*mm, 28*mm, 22*mm, 35*mm],
        styles=s,
        numeric_cols={1, 2},
    ))
    story.append(Paragraph("Table A1: Pairwise character-form averages in compact notation.", s['Caption']))

    story.append(make_table(
        ['Character', 'Best Score', 'Sources'],
        build_top_character_rows(sort_results),
        col_widths=[28*mm, 28*mm, 24*mm],
        styles=s,
        numeric_cols={1, 2},
    ))
    story.append(Paragraph("Table A2: Top 20 matching characters.", s['Caption']))

    story.append(PageBreak())
    story.append(Paragraph("C. Technical Statistical Tables", s['SSH']))
    story.append(Paragraph(
        "These appendix tables preserve the raw numerical outputs that are summarized by the chart suite in Sections 5 and 6.",
        s['Body']))

    story.append(make_table(
        ['Pair', 'KS(W)', 'p(W)', 'Verdict'],
        build_ks_dimension_rows(stats_results, 'verdict_width', 'ks_width', 'p_width'),
        col_widths=[34*mm, 24*mm, 28*mm, 36*mm],
        styles=s,
        numeric_cols={1, 2},
    ))
    story.append(Paragraph("Table A3: Kolmogorov-Smirnov width comparison.", s['Caption']))

    story.append(make_table(
        ['Pair', 'KS(H)', 'p(H)', 'Verdict'],
        build_ks_dimension_rows(stats_results, 'verdict_height', 'ks_height', 'p_height'),
        col_widths=[34*mm, 24*mm, 28*mm, 36*mm],
        styles=s,
        numeric_cols={1, 2},
    ))
    story.append(Paragraph("Table A4: Kolmogorov-Smirnov height comparison.", s['Caption']))

    story.append(make_table(
        ['Pair', 'χ²', 'df', 'p-value', 'Verdict'],
        build_chi_rows(stats_results),
        col_widths=[30*mm, 24*mm, 18*mm, 28*mm, 30*mm],
        styles=s,
        numeric_cols={1, 2, 3},
    ))
    story.append(Paragraph("Table A5: Chi-squared frequency comparison.", s['Caption']))

    story.append(make_table(
        ['Pair', 'U', 'p-value', 'Effect', 'Magnitude'],
        build_mw_rows(stats_results),
        col_widths=[26*mm, 22*mm, 26*mm, 24*mm, 32*mm],
        styles=s,
        body_font_size=7.8,
        header_font_size=8.5,
        numeric_cols={1, 2, 3},
    ))
    story.append(Paragraph("Table A6: Mann-Whitney U comparison, with compact U formatting.", s['Caption']))

    story.append(make_table(
        ['Pair', 'Mean', '95% CI', 'Verdict'],
        build_bootstrap_reference_rows(bootstrap_chart_rows),
        col_widths=[28*mm, 22*mm, 46*mm, 28*mm],
        styles=s,
        numeric_cols={1},
    ))
    story.append(Paragraph("Table A7: Bootstrap similarity intervals.", s['Caption']))

    story.append(PageBreak())
    story.append(Paragraph("D. Generated Reports & Scripts", s['SSH']))
    story.append(make_table(
        ['Path', 'Description'],
        [['reports/final_report/CODEFINDER_Forensic_Report.pdf', 'This report'],
         ['reports/final_report/final_report.html', 'Interactive HTML version'],
         ['reports/greenman_scan/greenman_report.html', 'Foliate-head matching with diagnostic SIFT overlays'],
         ['reports/character_sort_match/sort_report.html', 'Character sort comparison'],
         ['reports/statistical_analysis/stats_report.html', 'Statistical test results'],
         ['reports/damage_evolution/damage_report.html', 'Damage analysis'],
         [str(DEFAULT_DB_PATH), f'SQLite database ({totals["chars"]:,} chars)']],
        col_widths=[80*mm, 65*mm],
        styles=s,
        wrap_cols={0, 1},
        cell_style_name='TableCellSm',
        body_font_size=7.5,
        header_font_size=8.5,
    ))
    story.append(Paragraph("Table A8: Generated reports.", s['Caption']))

    # Build
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"✅ PDF report generated: {output_path}")
    print(f"   {size_kb:.0f} KB, {doc.page} pages")
    print(f"   Includes: selected source pages, foliate-head context/crop panels,")
    print(f"   reviewed character-pair exemplars, supplemental diagnostics")
    return output_path


if __name__ == "__main__":
    build_pdf()
