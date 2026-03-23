#!/usr/bin/env python3
"""
Folio Comparison Engine - Comparative Analysis of Multiple OCR Scans

This script compares OCR scan results from two different source editions
and generates a comprehensive differential report.

Usage:
    python folio_comparison_engine.py --source1 reports/sonnet_print_block_analysis \
                                      --source2 reports/sonnets_pdf_analysis \
                                      --output reports/comparison
"""

import json
import csv
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import difflib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ScanStatistics:
    """Statistics from a single OCR scan."""
    source_name: str
    source_file: str
    total_pages: int
    total_characters: int
    unique_characters: int
    total_words: int
    total_lines: int
    ligatures_found: int
    long_s_count: int
    anomalies_detected: int
    average_confidence: float
    scan_duration: float


@dataclass
class CharacterDelta:
    """Difference in character frequency between two sources."""
    character: str
    category: str
    source1_count: int
    source2_count: int
    delta: int
    delta_percentage: float


@dataclass
class FolioComparison:
    """Complete comparison results between two folio scans."""
    source1_stats: ScanStatistics
    source2_stats: ScanStatistics
    
    # High-level deltas
    page_delta: int = 0
    character_delta: int = 0
    word_delta: int = 0
    confidence_delta: float = 0.0
    
    # Character frequency analysis
    character_deltas: List[CharacterDelta] = field(default_factory=list)
    
    # Unique to each source
    chars_only_in_source1: List[str] = field(default_factory=list)
    chars_only_in_source2: List[str] = field(default_factory=list)
    
    # Anomaly comparison
    anomaly_delta: int = 0
    
    # Comparison metadata
    comparison_date: str = ""
    notes: List[str] = field(default_factory=list)


class FolioComparisonEngine:
    """Engine for comparing two folio OCR scan results."""
    
    def __init__(self, source1_dir: str, source2_dir: str, output_dir: str = None):
        self.source1_path = Path(source1_dir)
        self.source2_path = Path(source2_dir)
        self.output_path = Path(output_dir) if output_dir else Path("reports/comparison")
        
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Load source data
        self.source1_stats = self._load_statistics(self.source1_path, "Source 1 (IIIF)")
        self.source2_stats = self._load_statistics(self.source2_path, "Source 2 (PDF)")
        
        self.source1_freq = self._load_character_frequency(self.source1_path)
        self.source2_freq = self._load_character_frequency(self.source2_path)
        
        self.comparison: Optional[FolioComparison] = None
    
    def _load_statistics(self, source_path: Path, name: str) -> ScanStatistics:
        """Load statistics.json from a scan output directory."""
        stats_file = source_path / "statistics.json"
        if not stats_file.exists():
            raise FileNotFoundError(f"Statistics file not found: {stats_file}")
        
        with open(stats_file) as f:
            data = json.load(f)
        
        stats = data.get("statistics", {})
        scan_info = data.get("scan_info", {})
        
        return ScanStatistics(
            source_name=name,
            source_file=scan_info.get("source_file", "unknown"),
            total_pages=stats.get("total_pages", 0),
            total_characters=stats.get("total_characters", 0),
            unique_characters=stats.get("unique_characters", 0),
            total_words=stats.get("total_words", 0),
            total_lines=stats.get("total_lines", 0),
            ligatures_found=stats.get("ligatures_found", 0),
            long_s_count=stats.get("long_s_count", 0),
            anomalies_detected=stats.get("anomalies_detected", 0),
            average_confidence=stats.get("average_confidence", 0.0),
            scan_duration=stats.get("scan_duration_seconds", 0.0)
        )
    
    def _load_character_frequency(self, source_path: Path) -> Dict[str, Dict]:
        """Load character_frequency.csv from a scan output directory."""
        freq_file = source_path / "character_frequency.csv"
        if not freq_file.exists():
            logger.warning(f"Character frequency file not found: {freq_file}")
            return {}
        
        frequency = {}
        with open(freq_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                char = row.get('character', '')
                frequency[char] = {
                    'count': int(row.get('count', 0)),
                    'category': row.get('category', 'unknown'),
                    'percentage': float(row.get('percentage', 0.0))
                }
        
        return frequency
    
    def run_comparison(self) -> FolioComparison:
        """Execute the full comparison analysis."""
        logger.info("=" * 60)
        logger.info("FOLIO COMPARISON ENGINE")
        logger.info("=" * 60)
        logger.info(f"Source 1: {self.source1_stats.source_file}")
        logger.info(f"Source 2: {self.source2_stats.source_file}")
        
        # Create comparison object
        self.comparison = FolioComparison(
            source1_stats=self.source1_stats,
            source2_stats=self.source2_stats,
            comparison_date=datetime.now().isoformat()
        )
        
        # Compute high-level deltas
        self._compute_high_level_deltas()
        
        # Compute character frequency deltas
        self._compute_character_deltas()
        
        # Find unique characters
        self._find_unique_characters()
        
        # Add notes
        self._add_comparison_notes()
        
        logger.info("Comparison analysis complete")
        return self.comparison
    
    def _compute_high_level_deltas(self):
        """Compute high-level statistical differences."""
        s1, s2 = self.source1_stats, self.source2_stats
        
        self.comparison.page_delta = s2.total_pages - s1.total_pages
        self.comparison.character_delta = s2.total_characters - s1.total_characters
        self.comparison.word_delta = s2.total_words - s1.total_words
        self.comparison.confidence_delta = s2.average_confidence - s1.average_confidence
        self.comparison.anomaly_delta = s2.anomalies_detected - s1.anomalies_detected
        
        logger.info(f"Page delta: {self.comparison.page_delta:+d}")
        logger.info(f"Character delta: {self.comparison.character_delta:+d}")
        logger.info(f"Word delta: {self.comparison.word_delta:+d}")
        logger.info(f"Confidence delta: {self.comparison.confidence_delta:+.2f}%")
    
    def _compute_character_deltas(self):
        """Compute per-character frequency differences."""
        all_chars = set(self.source1_freq.keys()) | set(self.source2_freq.keys())
        
        for char in sorted(all_chars):
            s1_data = self.source1_freq.get(char, {'count': 0, 'category': 'unknown'})
            s2_data = self.source2_freq.get(char, {'count': 0, 'category': 'unknown'})
            
            s1_count = s1_data['count']
            s2_count = s2_data['count']
            delta = s2_count - s1_count
            
            # Calculate percentage delta (relative to source1)
            delta_pct = (delta / s1_count * 100) if s1_count > 0 else (100.0 if s2_count > 0 else 0.0)
            
            char_delta = CharacterDelta(
                character=char,
                category=s1_data.get('category') or s2_data.get('category', 'unknown'),
                source1_count=s1_count,
                source2_count=s2_count,
                delta=delta,
                delta_percentage=delta_pct
            )
            self.comparison.character_deltas.append(char_delta)
        
        # Sort by absolute delta (largest differences first)
        self.comparison.character_deltas.sort(key=lambda x: abs(x.delta), reverse=True)
        
        logger.info(f"Analyzed {len(self.comparison.character_deltas)} unique characters")
    
    def _find_unique_characters(self):
        """Find characters unique to each source."""
        s1_chars = set(self.source1_freq.keys())
        s2_chars = set(self.source2_freq.keys())
        
        self.comparison.chars_only_in_source1 = sorted(s1_chars - s2_chars)
        self.comparison.chars_only_in_source2 = sorted(s2_chars - s1_chars)
        
        if self.comparison.chars_only_in_source1:
            logger.info(f"Characters only in Source 1: {self.comparison.chars_only_in_source1}")
        if self.comparison.chars_only_in_source2:
            logger.info(f"Characters only in Source 2: {self.comparison.chars_only_in_source2}")
    
    def _add_comparison_notes(self):
        """Add analytical notes based on comparison results."""
        notes = []
        
        # Page count difference
        if abs(self.comparison.page_delta) > 0:
            notes.append(
                f"Source 2 has {abs(self.comparison.page_delta)} "
                f"{'more' if self.comparison.page_delta > 0 else 'fewer'} pages than Source 1. "
                f"This indicates different digitization approaches or source coverage."
            )
        
        # Character density
        s1_chars_per_page = self.source1_stats.total_characters / self.source1_stats.total_pages
        s2_chars_per_page = self.source2_stats.total_characters / self.source2_stats.total_pages
        
        notes.append(
            f"Character density: Source 1 = {s1_chars_per_page:.1f} chars/page, "
            f"Source 2 = {s2_chars_per_page:.1f} chars/page"
        )
        
        # Confidence difference
        if abs(self.comparison.confidence_delta) > 5:
            better_source = "Source 1" if self.comparison.confidence_delta < 0 else "Source 2"
            notes.append(
                f"{better_source} has significantly higher OCR confidence "
                f"(Δ = {abs(self.comparison.confidence_delta):.1f}%)"
            )
        
        # Long-s comparison
        s1_long_s = self.source1_stats.long_s_count
        s2_long_s = self.source2_stats.long_s_count
        notes.append(f"Long-s (ſ) count: Source 1 = {s1_long_s}, Source 2 = {s2_long_s}")
        
        # Ligature comparison
        s1_lig = self.source1_stats.ligatures_found
        s2_lig = self.source2_stats.ligatures_found
        notes.append(f"Ligatures found: Source 1 = {s1_lig}, Source 2 = {s2_lig}")
        
        self.comparison.notes = notes
    
    def generate_reports(self):
        """Generate comparison report artifacts."""
        if not self.comparison:
            raise ValueError("Run comparison first before generating reports")
        
        # Generate JSON summary
        self._generate_json_report()
        
        # Generate CSV of character deltas
        self._generate_csv_report()
        
        # Generate HTML report
        self._generate_html_report()
        
        logger.info(f"Reports saved to: {self.output_path}")
    
    def _generate_json_report(self):
        """Generate JSON comparison summary."""
        output_file = self.output_path / "comparison_summary.json"
        
        data = {
            "comparison_date": self.comparison.comparison_date,
            "source1": {
                "name": self.source1_stats.source_name,
                "file": self.source1_stats.source_file,
                "total_pages": self.source1_stats.total_pages,
                "total_characters": self.source1_stats.total_characters,
                "unique_characters": self.source1_stats.unique_characters,
                "average_confidence": self.source1_stats.average_confidence,
                "long_s_count": self.source1_stats.long_s_count,
                "ligatures_found": self.source1_stats.ligatures_found,
                "anomalies_detected": self.source1_stats.anomalies_detected
            },
            "source2": {
                "name": self.source2_stats.source_name,
                "file": self.source2_stats.source_file,
                "total_pages": self.source2_stats.total_pages,
                "total_characters": self.source2_stats.total_characters,
                "unique_characters": self.source2_stats.unique_characters,
                "average_confidence": self.source2_stats.average_confidence,
                "long_s_count": self.source2_stats.long_s_count,
                "ligatures_found": self.source2_stats.ligatures_found,
                "anomalies_detected": self.source2_stats.anomalies_detected
            },
            "deltas": {
                "page_delta": self.comparison.page_delta,
                "character_delta": self.comparison.character_delta,
                "word_delta": self.comparison.word_delta,
                "confidence_delta": round(self.comparison.confidence_delta, 2),
                "anomaly_delta": self.comparison.anomaly_delta
            },
            "unique_characters": {
                "only_in_source1": self.comparison.chars_only_in_source1,
                "only_in_source2": self.comparison.chars_only_in_source2
            },
            "top_character_deltas": [
                {
                    "character": cd.character,
                    "category": cd.category,
                    "source1_count": cd.source1_count,
                    "source2_count": cd.source2_count,
                    "delta": cd.delta
                }
                for cd in self.comparison.character_deltas[:20]
            ],
            "notes": self.comparison.notes
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON report: {output_file}")
    
    def _generate_csv_report(self):
        """Generate CSV of character frequency deltas."""
        output_file = self.output_path / "character_deltas.csv"
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'character', 'category', 'source1_count', 'source2_count', 
                'delta', 'delta_percentage'
            ])
            
            for cd in self.comparison.character_deltas:
                writer.writerow([
                    cd.character, cd.category, cd.source1_count, cd.source2_count,
                    cd.delta, f"{cd.delta_percentage:.2f}"
                ])
        
        logger.info(f"CSV report: {output_file}")
    
    def _generate_html_report(self):
        """Generate HTML comparison visualization."""
        output_file = self.output_path / "comparison_report.html"
        
        html = self._build_html_report()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"HTML report: {output_file}")
    
    def _build_html_report(self) -> str:
        """Build the HTML report content."""
        s1 = self.source1_stats
        s2 = self.source2_stats
        comp = self.comparison
        
        # Build character delta table rows
        delta_rows = ""
        for cd in comp.character_deltas[:50]:  # Top 50
            delta_class = "positive" if cd.delta > 0 else ("negative" if cd.delta < 0 else "neutral")
            char_display = cd.character if cd.character not in [' ', '\n', '\t'] else repr(cd.character)
            delta_rows += f"""
            <tr class="{delta_class}">
                <td class="char-cell">{char_display}</td>
                <td>{cd.category}</td>
                <td class="num">{cd.source1_count:,}</td>
                <td class="num">{cd.source2_count:,}</td>
                <td class="num delta">{cd.delta:+,}</td>
                <td class="num">{cd.delta_percentage:+.1f}%</td>
            </tr>
            """
        
        # Build notes list
        notes_html = "\n".join(f"<li>{note}</li>" for note in comp.notes)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Folio Comparison Report</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
            --border-color: #30363d;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
            color: var(--accent-blue);
        }}
        
        h2 {{
            font-size: 1.5rem;
            margin: 2rem 0 1rem;
            color: var(--text-primary);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            margin-bottom: 2rem;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .stat-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
        }}
        
        .stat-card h3 {{
            color: var(--accent-blue);
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }}
        
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .stat-row:last-child {{
            border-bottom: none;
        }}
        
        .stat-label {{
            color: var(--text-secondary);
        }}
        
        .stat-value {{
            font-weight: 600;
            font-family: 'SF Mono', Monaco, monospace;
        }}
        
        .delta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .delta-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }}
        
        .delta-card .label {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }}
        
        .delta-card .value {{
            font-size: 1.5rem;
            font-weight: 700;
            font-family: 'SF Mono', Monaco, monospace;
        }}
        
        .delta-card .value.positive {{ color: var(--accent-green); }}
        .delta-card .value.negative {{ color: var(--accent-red); }}
        .delta-card .value.neutral {{ color: var(--text-secondary); }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background: var(--bg-tertiary);
            font-weight: 600;
            color: var(--text-secondary);
        }}
        
        .num {{ text-align: right; font-family: 'SF Mono', Monaco, monospace; }}
        
        .char-cell {{
            font-size: 1.2rem;
            font-family: 'SF Mono', Monaco, monospace;
            text-align: center;
            width: 60px;
        }}
        
        tr.positive .delta {{ color: var(--accent-green); }}
        tr.negative .delta {{ color: var(--accent-red); }}
        tr.neutral .delta {{ color: var(--text-secondary); }}
        
        .notes {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
        }}
        
        .notes ul {{
            margin-left: 1.5rem;
        }}
        
        .notes li {{
            margin-bottom: 0.5rem;
            color: var(--text-secondary);
        }}
        
        .timestamp {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-top: 2rem;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Folio Comparison Report</h1>
        <p class="subtitle">Comparative OCR Analysis of Shakespeare's Sonnets (1609)</p>
        
        <h2>Source Overview</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>📄 Source 1: IIIF Folger Images</h3>
                <div class="stat-row">
                    <span class="stat-label">Source File</span>
                    <span class="stat-value">{s1.source_file}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Total Pages</span>
                    <span class="stat-value">{s1.total_pages:,}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Total Characters</span>
                    <span class="stat-value">{s1.total_characters:,}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Unique Characters</span>
                    <span class="stat-value">{s1.unique_characters}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Avg Confidence</span>
                    <span class="stat-value">{s1.average_confidence:.1f}%</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Long-s (ſ) Count</span>
                    <span class="stat-value">{s1.long_s_count:,}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Ligatures Found</span>
                    <span class="stat-value">{s1.ligatures_found:,}</span>
                </div>
            </div>
            
            <div class="stat-card">
                <h3>📄 Source 2: Legacy PDF</h3>
                <div class="stat-row">
                    <span class="stat-label">Source File</span>
                    <span class="stat-value">{s2.source_file}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Total Pages</span>
                    <span class="stat-value">{s2.total_pages:,}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Total Characters</span>
                    <span class="stat-value">{s2.total_characters:,}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Unique Characters</span>
                    <span class="stat-value">{s2.unique_characters}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Avg Confidence</span>
                    <span class="stat-value">{s2.average_confidence:.1f}%</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Long-s (ſ) Count</span>
                    <span class="stat-value">{s2.long_s_count:,}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Ligatures Found</span>
                    <span class="stat-value">{s2.ligatures_found:,}</span>
                </div>
            </div>
        </div>
        
        <h2>Key Differences (Δ)</h2>
        <div class="delta-grid">
            <div class="delta-card">
                <div class="label">Page Count</div>
                <div class="value {'positive' if comp.page_delta > 0 else 'negative' if comp.page_delta < 0 else 'neutral'}">{comp.page_delta:+,}</div>
            </div>
            <div class="delta-card">
                <div class="label">Character Count</div>
                <div class="value {'positive' if comp.character_delta > 0 else 'negative' if comp.character_delta < 0 else 'neutral'}">{comp.character_delta:+,}</div>
            </div>
            <div class="delta-card">
                <div class="label">Word Count</div>
                <div class="value {'positive' if comp.word_delta > 0 else 'negative' if comp.word_delta < 0 else 'neutral'}">{comp.word_delta:+,}</div>
            </div>
            <div class="delta-card">
                <div class="label">Avg Confidence</div>
                <div class="value {'positive' if comp.confidence_delta > 0 else 'negative' if comp.confidence_delta < 0 else 'neutral'}">{comp.confidence_delta:+.1f}%</div>
            </div>
            <div class="delta-card">
                <div class="label">Anomalies</div>
                <div class="value {'positive' if comp.anomaly_delta > 0 else 'negative' if comp.anomaly_delta < 0 else 'neutral'}">{comp.anomaly_delta:+,}</div>
            </div>
        </div>
        
        <h2>Character Frequency Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Char</th>
                    <th>Category</th>
                    <th class="num">Source 1</th>
                    <th class="num">Source 2</th>
                    <th class="num">Delta</th>
                    <th class="num">Δ%</th>
                </tr>
            </thead>
            <tbody>
                {delta_rows}
            </tbody>
        </table>
        
        <h2>Analysis Notes</h2>
        <div class="notes">
            <ul>
                {notes_html}
            </ul>
        </div>
        
        <p class="timestamp">Generated: {comp.comparison_date}</p>
    </div>
</body>
</html>
"""
        return html


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare two folio OCR scan results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source1", "-s1",
        default="reports/sonnet_print_block_analysis",
        help="Path to first scan results directory"
    )
    parser.add_argument(
        "--source2", "-s2", 
        default="reports/sonnets_pdf_analysis",
        help="Path to second scan results directory"
    )
    parser.add_argument(
        "--output", "-o",
        default="reports/comparison",
        help="Output directory for comparison reports"
    )
    
    args = parser.parse_args()
    
    try:
        engine = FolioComparisonEngine(
            source1_dir=args.source1,
            source2_dir=args.source2,
            output_dir=args.output
        )
        
        comparison = engine.run_comparison()
        engine.generate_reports()
        
        print("\n" + "=" * 60)
        print("✅ COMPARISON COMPLETE")
        print("=" * 60)
        print(f"📁 Output files:")
        print(f"   - {args.output}/comparison_summary.json")
        print(f"   - {args.output}/character_deltas.csv")
        print(f"   - {args.output}/comparison_report.html")
        
    except FileNotFoundError as e:
        logger.error(f"Source file not found: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise


if __name__ == "__main__":
    main()
