#!/usr/bin/env python3
"""
Damage Evolution Tracker
========================

Compares damage patterns (nicks, cracks, ink spread) across sources ordered
chronologically to test whether damage accumulates over time.

If physical type was shared, we expect:
- Damage metrics increase monotonically with publication date
- Edge erosion correlates with chronological distance from earliest use
- This constitutes strong evidence of shared physical materials

Dates (from config):
  BSB Munich    ~1609
  HAB Wolf.     ~1610 (est.)
  Tractatus     ~1613
  GDZ Göttingen ~1614

Usage:
    python scripts/damage_evolution.py
"""

import sys
import json
import sqlite3
import cv2
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("damage_evolution")

# Chronological ordering with approximate dates
SOURCE_DATES = {
    'bsb_munich_10057380': 1609,
    'hab_wolfenbuettel_178_1_theol_1s': 1610,  # estimated 1600-1620, using midpoint
    'google_books_tractatus_brevis': 1613,
    'gdz_goettingen_ppn777246686': 1614,
}


def compute_damage_metrics(crop_path: str) -> dict:
    """Compute damage metrics for a single character crop.
    
    Returns dict with:
      - edge_roughness: std of edge distances (nicks/chips cause high variance)
      - ink_spread: ratio of dilated vs original ink area 
      - crack_density: thin bright lines within ink area
      - edge_erosion: perimeter-to-area ratio (worn type has higher P/A)
      - entropy: texture entropy (damage increases disorder)
    """
    full = Path("reports/crops") / crop_path
    if not full.exists():
        return None
    
    img = cv2.imread(str(full), cv2.IMREAD_GRAYSCALE)
    if img is None or img.size < 25:
        return None
    
    h, w = img.shape[:2]
    
    # Binarize
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # --- Edge Roughness ---
    # Find contours and measure edge irregularity
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    
    main_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    
    if area < 10 or perimeter < 10:
        return None
    
    # Edge roughness: compare actual perimeter to convex hull perimeter
    hull = cv2.convexHull(main_contour)
    hull_perimeter = cv2.arcLength(hull, True)
    edge_roughness = (perimeter / hull_perimeter) - 1.0 if hull_perimeter > 0 else 0
    
    # --- Edge Erosion (P/A ratio) ---
    # Worn type has higher perimeter relative to area
    edge_erosion = perimeter / np.sqrt(area) if area > 0 else 0
    
    # --- Ink Spread ---
    # Compare ink area under different thresholds
    _, tight = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY_INV)
    _, loose = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)
    tight_area = np.sum(tight > 0)
    loose_area = np.sum(loose > 0)
    ink_spread = loose_area / tight_area if tight_area > 0 else 1.0
    
    # --- Crack Density ---
    # Morphological opening to detect thin gaps in ink
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cracks = binary.astype(float) - opened.astype(float)
    crack_pixels = np.sum(cracks > 128)
    crack_density = crack_pixels / area if area > 0 else 0
    
    # --- Texture Entropy ---
    # Shannon entropy of pixel intensities
    hist = cv2.calcHist([img], [0], None, [64], [0, 256])
    hist = hist.flatten() / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    return {
        'edge_roughness': float(edge_roughness),
        'edge_erosion': float(edge_erosion),
        'ink_spread': float(ink_spread),
        'crack_density': float(crack_density),
        'entropy': float(entropy),
    }


def gather_crop_paths(db_path="data/forensic.db"):
    """Get crop paths grouped by (source, character)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    results = defaultdict(lambda: defaultdict(list))
    
    rows = conn.execute("""
        SELECT s.name as source, ci.character, si.image_path
        FROM sort_images si
        JOIN character_instances ci ON si.character_instance_id = ci.id
        JOIN pages p ON ci.page_id = p.id
        JOIN sources s ON p.source_id = s.id
        WHERE si.image_path IS NOT NULL
    """).fetchall()
    
    for r in rows:
        results[r['character']][r['source']].append(r['image_path'])
    
    conn.close()
    return results


def run_evolution_analysis():
    logger.info("Gathering crop paths from database...")
    all_crops = gather_crop_paths()
    
    # Filter to characters present in 3+ sources
    multi_source_chars = {c: srcs for c, srcs in all_crops.items() 
                         if len(srcs) >= 3 and len(c) == 1 and c.isalpha()}
    
    logger.info(f"Found {len(multi_source_chars)} characters in 3+ sources")
    
    # Compute damage per source per character
    source_damage = defaultdict(lambda: defaultdict(list))
    
    for char, sources in multi_source_chars.items():
        logger.info(f"Analysing damage for '{char}' across {len(sources)} sources...")
        for source, paths in sources.items():
            # Sample up to 20 crops per source per character
            sample = paths[:20]
            for p in sample:
                metrics = compute_damage_metrics(p)
                if metrics:
                    source_damage[source][char].append(metrics)
    
    # Aggregate: per-source mean damage metrics
    source_means = {}
    for source in SOURCE_DATES:
        if source not in source_damage:
            continue
        all_metrics = []
        for char, metric_list in source_damage[source].items():
            all_metrics.extend(metric_list)
        
        if not all_metrics:
            continue
            
        source_means[source] = {
            'date': SOURCE_DATES[source],
            'n_samples': len(all_metrics),
            'n_characters': len(source_damage[source]),
            'edge_roughness': float(np.mean([m['edge_roughness'] for m in all_metrics])),
            'edge_erosion': float(np.mean([m['edge_erosion'] for m in all_metrics])),
            'ink_spread': float(np.mean([m['ink_spread'] for m in all_metrics])),
            'crack_density': float(np.mean([m['crack_density'] for m in all_metrics])),
            'entropy': float(np.mean([m['entropy'] for m in all_metrics])),
        }
    
    if len(source_means) < 2:
        logger.warning("Not enough sources with damage data for evolution analysis")
        return
    
    # Sort chronologically
    chrono = sorted(source_means.items(), key=lambda x: x[1]['date'])
    
    # Test monotonicity for each metric
    metrics_to_test = ['edge_roughness', 'edge_erosion', 'ink_spread', 'crack_density', 'entropy']
    monotonicity = {}
    
    for metric in metrics_to_test:
        values = [s[1][metric] for s in chrono]
        # Spearman-like: count concordant pairs
        n = len(values)
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i+1, n):
                if values[j] > values[i]:
                    concordant += 1
                elif values[j] < values[i]:
                    discordant += 1
        total = concordant + discordant
        tau = (concordant - discordant) / total if total > 0 else 0
        
        monotonicity[metric] = {
            'values': values,
            'dates': [s[1]['date'] for s in chrono],
            'sources': [s[0] for s in chrono],
            'kendall_tau': float(tau),
            'trend': 'INCREASING' if tau > 0.3 else ('DECREASING' if tau < -0.3 else 'NO TREND'),
            'supports_shared_type': tau > 0.3,
        }
    
    # Count how many metrics show increasing damage
    increasing_count = sum(1 for m in monotonicity.values() if m['supports_shared_type'])
    
    verdict = {
        'total_metrics': len(metrics_to_test),
        'increasing_damage': increasing_count,
        'evidence_strength': 'STRONG' if increasing_count >= 4 else (
            'MODERATE' if increasing_count >= 3 else (
            'WEAK' if increasing_count >= 2 else 'INSUFFICIENT')),
        'interpretation': (
            f"{increasing_count}/{len(metrics_to_test)} damage metrics increase chronologically. "
            "This is consistent with shared physical type experiencing progressive wear."
            if increasing_count >= 3
            else f"Only {increasing_count}/{len(metrics_to_test)} metrics show chronological increase. "
                 "Damage evolution is inconclusive — may reflect scan quality differences."
        )
    }
    
    # Save results
    output_dir = Path("reports/damage_evolution")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'source_means': source_means,
        'monotonicity': monotonicity,
        'verdict': verdict,
    }
    
    with open(output_dir / "damage_evolution.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate HTML report
    display_names = {
        'bsb_munich_10057380': 'BSB Munich',
        'gdz_goettingen_ppn777246686': 'GDZ Göttingen',
        'hab_wolfenbuettel_178_1_theol_1s': 'HAB Wolfenbüttel',
        'google_books_tractatus_brevis': 'Google Books (Tractatus)',
    }
    
    html = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Damage Evolution Analysis</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600&family=JetBrains+Mono:wght@300&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Crimson Pro', serif; background: #0a0a1a; color: #d0d0d0; line-height: 1.7; padding: 2em; }
.container { max-width: 1000px; margin: 0 auto; }
h1 { color: #c9a94e; font-size: 2em; margin-bottom: 0.5em; text-align: center; }
h2 { color: #c9a94e; font-size: 1.4em; margin: 2em 0 0.5em; border-left: 4px solid #c9a94e; padding-left: 0.7em; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; background: rgba(15,15,35,0.7); }
th { background: #0f3460; color: #c9a94e; padding: 10px 14px; text-align: left; }
td { padding: 8px 14px; border-bottom: 1px solid rgba(255,255,255,0.05); }
.trend-up { color: #4caf50; font-weight: 600; }
.trend-down { color: #f44336; }
.trend-none { color: #888; }
.verdict { padding: 1.5em; margin: 2em 0; border-radius: 8px; text-align: center; font-size: 1.2em; }
.strong { border: 2px solid #4caf50; color: #4caf50; background: rgba(76,175,80,0.1); }
.moderate { border: 2px solid #ff9800; color: #ff9800; background: rgba(255,152,0,0.1); }
.weak { border: 2px solid #f44336; color: #f44336; background: rgba(244,67,54,0.1); }
.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.85em; }
.bar { display: inline-block; height: 16px; background: #c9a94e; border-radius: 3px; }
</style></head><body><div class="container">
<h1>🔬 Damage Evolution Analysis</h1>
<p style="text-align:center;color:#888;">Chronological tracking of type wear across publications</p>
"""
    
    # Source summary table
    html += '<h2>Source Chronology & Sampling</h2>'
    html += '<table><tr><th>Source</th><th>Date</th><th>Samples</th><th>Characters</th></tr>'
    for src, data in sorted(source_means.items(), key=lambda x: x[1]['date']):
        dn = display_names.get(src, src)
        html += f'<tr><td>{dn}</td><td>{data["date"]}</td>'
        html += f'<td>{data["n_samples"]}</td><td>{data["n_characters"]}</td></tr>'
    html += '</table>'
    
    # Damage metrics table
    html += '<h2>Damage Metrics by Source (Chronological Order)</h2>'
    html += '<table><tr><th>Metric</th>'
    for src, data in sorted(source_means.items(), key=lambda x: x[1]['date']):
        dn = display_names.get(src, src)
        html += f'<th>{dn}<br><span class="mono">({data["date"]})</span></th>'
    html += '<th>Trend</th><th>τ</th></tr>'
    
    for metric in metrics_to_test:
        html += f'<tr><td>{metric.replace("_", " ").title()}</td>'
        vals = []
        for src, data in sorted(source_means.items(), key=lambda x: x[1]['date']):
            v = data[metric]
            vals.append(v)
            html += f'<td class="mono">{v:.4f}</td>'
        
        mono = monotonicity[metric]
        tcls = 'trend-up' if mono['trend'] == 'INCREASING' else ('trend-down' if mono['trend'] == 'DECREASING' else 'trend-none')
        html += f'<td class="{tcls}">{mono["trend"]}</td>'
        html += f'<td class="mono">{mono["kendall_tau"]:.3f}</td></tr>'
    html += '</table>'
    
    # Visual bar chart
    html += '<h2>Damage Progression</h2>'
    for metric in ['edge_roughness', 'edge_erosion', 'ink_spread']:
        html += f'<h3 style="color:#a08040;margin:1em 0 0.3em;">{metric.replace("_"," ").title()}</h3>'
        vals = []
        for src, data in sorted(source_means.items(), key=lambda x: x[1]['date']):
            vals.append((display_names.get(src, src), data['date'], data[metric]))
        max_v = max(v[2] for v in vals) if vals else 1
        for name, date, val in vals:
            bar_w = int(300 * val / max_v) if max_v > 0 else 0
            html += f'<div style="margin:4px 0;"><span class="mono" style="display:inline-block;width:200px;">{name} ({date})</span>'
            html += f'<span class="bar" style="width:{bar_w}px;"></span> <span class="mono">{val:.4f}</span></div>'
    
    # Verdict
    vcls = verdict['evidence_strength'].lower()
    if vcls not in ('strong', 'moderate', 'weak'):
        vcls = 'weak'
    html += f'<h2>Verdict</h2>'
    html += f'<div class="verdict {vcls}">'
    html += f'Evidence Strength: {verdict["evidence_strength"]}<br>'
    html += f'<span style="font-size:0.8em;font-weight:400;">{verdict["interpretation"]}</span></div>'
    
    html += '</div></body></html>'
    
    with open(output_dir / "damage_report.html", 'w') as f:
        f.write(html)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DAMAGE EVOLUTION ANALYSIS — SUMMARY")
    print("=" * 60)
    
    for src, data in sorted(source_means.items(), key=lambda x: x[1]['date']):
        dn = display_names.get(src, src)
        print(f"\n  {dn} ({data['date']}) — {data['n_samples']} samples:")
        print(f"    Edge roughness: {data['edge_roughness']:.4f}")
        print(f"    Edge erosion:   {data['edge_erosion']:.4f}")
        print(f"    Ink spread:     {data['ink_spread']:.4f}")
        print(f"    Crack density:  {data['crack_density']:.4f}")
        print(f"    Entropy:        {data['entropy']:.4f}")
    
    print(f"\n  Monotonicity tests:")
    for metric, data in monotonicity.items():
        symbol = "↑" if data['supports_shared_type'] else "—"
        print(f"    {metric:20s}: τ={data['kendall_tau']:+.3f}  {data['trend']:12s} {symbol}")
    
    print(f"\n  Verdict: {verdict['evidence_strength']}")
    print(f"  {verdict['interpretation']}")
    print(f"\n  Reports: reports/damage_evolution/damage_report.html")


if __name__ == "__main__":
    run_evolution_analysis()
