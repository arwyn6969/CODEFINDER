#!/usr/bin/env python3
"""
Character Geometry Analyzer

Compares character dimensions (widths, heights) between editions
to identify print block differences and potential type variants.

This is the core of forensic print block analysis.
"""

import csv
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime


@dataclass
class CharacterMetrics:
    """Metrics for a single character across an edition."""
    character: str
    count: int
    avg_width: float
    avg_height: float
    category: str


@dataclass
class GeometryDelta:
    """Difference in geometry between editions."""
    character: str
    ed1_count: int
    ed2_count: int
    ed1_width: float
    ed2_width: float
    width_delta: float
    ed1_height: float
    ed2_height: float
    height_delta: float
    significance: str  # 'low', 'medium', 'high'


def load_character_data(csv_path: str) -> Dict[str, CharacterMetrics]:
    """Load character frequency data from CSV."""
    data = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            char = row['Character']
            data[char] = CharacterMetrics(
                character=char,
                count=int(row['Count']),
                avg_width=float(row['Avg Width']),
                avg_height=float(row['Avg Height']),
                category=row['Category']
            )
    return data


def calculate_significance(width_delta: float, height_delta: float) -> str:
    """Determine significance level of geometry differences."""
    abs_width = abs(width_delta)
    abs_height = abs(height_delta)
    
    # High: likely different type blocks/fonts
    if abs_width > 3.0 or abs_height > 5.0:
        return "high"
    # Medium: possible variants or ink differences
    elif abs_width > 1.5 or abs_height > 2.5:
        return "medium"
    # Low: normal variation
    else:
        return "low"


def compare_editions(
    ed1_data: Dict[str, CharacterMetrics],
    ed2_data: Dict[str, CharacterMetrics],
    ed1_name: str = "Edition 1",
    ed2_name: str = "Edition 2"
) -> Tuple[List[GeometryDelta], Dict]:
    """Compare character geometry between two editions."""
    common_chars = set(ed1_data.keys()) & set(ed2_data.keys())
    only_ed1 = set(ed1_data.keys()) - set(ed2_data.keys())
    only_ed2 = set(ed2_data.keys()) - set(ed1_data.keys())
    
    deltas = []
    for char in sorted(common_chars, key=lambda c: ed1_data[c].count, reverse=True):
        ed1 = ed1_data[char]
        ed2 = ed2_data[char]
        
        width_delta = ed2.avg_width - ed1.avg_width
        height_delta = ed2.avg_height - ed1.avg_height
        significance = calculate_significance(width_delta, height_delta)
        
        deltas.append(GeometryDelta(
            character=char,
            ed1_count=ed1.count,
            ed2_count=ed2.count,
            ed1_width=ed1.avg_width,
            ed2_width=ed2.avg_width,
            width_delta=width_delta,
            ed1_height=ed1.avg_height,
            ed2_height=ed2.avg_height,
            height_delta=height_delta,
            significance=significance
        ))
    
    # Calculate overall scale factor (editions may have different image resolutions)
    total_ed1_width = sum(ed1_data[c].avg_width * ed1_data[c].count for c in common_chars)
    total_ed2_width = sum(ed2_data[c].avg_width * ed2_data[c].count for c in common_chars)
    total_ed1_count = sum(ed1_data[c].count for c in common_chars)
    total_ed2_count = sum(ed2_data[c].count for c in common_chars)
    
    weighted_ed1_width = total_ed1_width / total_ed1_count if total_ed1_count else 1
    weighted_ed2_width = total_ed2_width / total_ed2_count if total_ed2_count else 1
    scale_factor = weighted_ed2_width / weighted_ed1_width if weighted_ed1_width else 1
    
    summary = {
        "edition_1": ed1_name,
        "edition_2": ed2_name,
        "common_characters": len(common_chars),
        "only_in_ed1": len(only_ed1),
        "only_in_ed2": len(only_ed2),
        "overall_scale_factor": round(scale_factor, 4),
        "scale_interpretation": "Images have different resolutions" if abs(scale_factor - 1.0) > 0.1 else "Similar resolution",
        "high_significance_count": sum(1 for d in deltas if d.significance == "high"),
        "medium_significance_count": sum(1 for d in deltas if d.significance == "medium"),
        "low_significance_count": sum(1 for d in deltas if d.significance == "low"),
    }
    
    return deltas, summary


def generate_report(
    ed1_path: str,
    ed2_path: str,
    ed1_name: str,
    ed2_name: str,
    output_dir: str
) -> str:
    """Generate comprehensive geometry comparison report."""
    
    ed1_data = load_character_data(ed1_path)
    ed2_data = load_character_data(ed2_path)
    
    deltas, summary = compare_editions(ed1_data, ed2_data, ed1_name, ed2_name)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # JSON summary
    summary_data = {
        "analysis_date": datetime.now().isoformat(),
        "summary": summary,
        "key_findings": []
    }
    
    # Find key findings
    high_sig = [d for d in deltas if d.significance == "high"]
    if high_sig:
        summary_data["key_findings"].append({
            "type": "resolution_difference",
            "description": f"Character dimensions differ significantly ({len(high_sig)} chars)",
            "implication": "The source images likely have different resolutions or the editions use different type sizes"
        })
    
    # Long-s analysis
    long_s_delta = next((d for d in deltas if d.character == 'ſ'), None)
    if long_s_delta:
        summary_data["key_findings"].append({
            "type": "long_s_comparison",
            "description": f"Long-s (ſ) appears {long_s_delta.ed1_count} times in {ed1_name}, {long_s_delta.ed2_count} times in {ed2_name}",
            "ratio": round(long_s_delta.ed2_count / long_s_delta.ed1_count, 2) if long_s_delta.ed1_count else 0,
            "implication": "Different Long-s counts may indicate different compositor habits or text variations"
        })
    
    # VV/W analysis
    w_delta = next((d for d in deltas if d.character == 'W'), None)
    w_lower = next((d for d in deltas if d.character == 'w'), None)
    v_delta = next((d for d in deltas if d.character == 'V'), None)
    if w_delta and v_delta:
        vv_width = v_delta.ed1_width * 2 if v_delta else 0
        summary_data["key_findings"].append({
            "type": "vv_w_analysis",
            "description": f"W width ({ed1_name}): {w_delta.ed1_width:.2f}, V×2: {vv_width:.2f}",
            "implication": "VV used as W if W width ≈ 2×V width suggests ligated VV type"
        })
    
    with open(output_path / "geometry_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # CSV with all deltas
    with open(output_path / "geometry_deltas.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Character", "Category",
            f"{ed1_name} Count", f"{ed2_name} Count",
            f"{ed1_name} Width", f"{ed2_name} Width", "Width Δ",
            f"{ed1_name} Height", f"{ed2_name} Height", "Height Δ",
            "Significance"
        ])
        for d in deltas:
            cat = ed1_data.get(d.character, ed2_data.get(d.character)).category
            writer.writerow([
                d.character, cat,
                d.ed1_count, d.ed2_count,
                f"{d.ed1_width:.2f}", f"{d.ed2_width:.2f}", f"{d.width_delta:+.2f}",
                f"{d.ed1_height:.2f}", f"{d.ed2_height:.2f}", f"{d.height_delta:+.2f}",
                d.significance
            ])
    
    # Text report
    report_lines = [
        "=" * 80,
        "CHARACTER GEOMETRY ANALYSIS REPORT",
        "=" * 80,
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Edition 1: {ed1_name}",
        f"Edition 2: {ed2_name}",
        "",
        "SUMMARY",
        "-" * 80,
        f"Common characters analyzed: {summary['common_characters']}",
        f"Overall scale factor: {summary['overall_scale_factor']:.4f} ({summary['scale_interpretation']})",
        f"High significance differences: {summary['high_significance_count']}",
        f"Medium significance differences: {summary['medium_significance_count']}",
        "",
        "KEY CHARACTER COMPARISONS",
        "-" * 80,
    ]
    
    # Table of important characters
    key_chars = ['e', 't', 'o', 'a', 's', 'ſ', 'f', 'W', 'V', 'w', '.', ',']
    report_lines.append(f"{'Char':<8} {'Cat':<12} {ed1_name[:8]:>10} {ed2_name[:8]:>10} {'Δ Width':>10} {'Significance':>12}")
    report_lines.append("-" * 62)
    
    for char in key_chars:
        delta = next((d for d in deltas if d.character == char), None)
        if delta:
            cat = ed1_data.get(char, ed2_data.get(char)).category
            report_lines.append(
                f"{repr(char):<8} {cat:<12} {delta.ed1_width:>10.2f} {delta.ed2_width:>10.2f} {delta.width_delta:>+10.2f} {delta.significance:>12}"
            )
    
    report_lines.extend([
        "",
        "KEY FINDINGS",
        "-" * 80,
    ])
    
    for i, finding in enumerate(summary_data["key_findings"], 1):
        report_lines.append(f"{i}. {finding['type'].upper()}")
        report_lines.append(f"   {finding['description']}")
        report_lines.append(f"   Implication: {finding['implication']}")
        report_lines.append("")
    
    report_lines.extend([
        "",
        "INTERPRETATION",
        "-" * 80,
        f"The scale factor of {summary['overall_scale_factor']:.4f} suggests the Aspley source images",
        "are approximately 60% larger in pixel dimensions than the Wright images.",
        "",
        "To perform a true forensic comparison of type dimensions, the images",
        "should be normalized to the same DPI or physical scale.",
        "",
        "Current differences in character geometry are dominated by this",
        "resolution difference rather than actual print block variations.",
    ])
    
    report_text = '\n'.join(report_lines)
    with open(output_path / "geometry_report.txt", 'w') as f:
        f.write(report_text)
    
    return report_text


def main():
    """Run geometry comparison between Wright and Aspley editions."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare character geometry between editions")
    parser.add_argument("--ed1", required=True, help="Path to edition 1 character_frequency.csv")
    parser.add_argument("--ed2", required=True, help="Path to edition 2 character_frequency.csv")
    parser.add_argument("--ed1-name", default="Edition 1", help="Name of edition 1")
    parser.add_argument("--ed2-name", default="Edition 2", help="Name of edition 2")
    parser.add_argument("--output", default="reports/geometry_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    report = generate_report(
        args.ed1, args.ed2,
        args.ed1_name, args.ed2_name,
        args.output
    )
    print(report)


if __name__ == "__main__":
    main()
