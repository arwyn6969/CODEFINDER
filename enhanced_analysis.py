#!/usr/bin/env python3
"""
CODEFINDER Enhanced Analysis
============================
With statistical significance testing, ELS grid visualization, and PDF export.
"""

import sys
import os
import json
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import Counter
import random

PROJECT_ROOT = Path(__file__).parent

# ============================================================================
# STATISTICAL SIGNIFICANCE MODULE
# ============================================================================

class StatisticalAnalyzer:
    """Calculate statistical significance of pattern findings."""
    
    # English letter frequencies (for probability calculations)
    LETTER_FREQ = {
        'A': 0.0812, 'B': 0.0149, 'C': 0.0278, 'D': 0.0425, 'E': 0.1202,
        'F': 0.0223, 'G': 0.0202, 'H': 0.0609, 'I': 0.0697, 'J': 0.0015,
        'K': 0.0077, 'L': 0.0403, 'M': 0.0241, 'N': 0.0675, 'O': 0.0751,
        'P': 0.0193, 'Q': 0.0010, 'R': 0.0599, 'S': 0.0633, 'T': 0.0906,
        'U': 0.0276, 'V': 0.0098, 'W': 0.0236, 'X': 0.0015, 'Y': 0.0197,
        'Z': 0.0007
    }
    
    def calculate_els_probability(self, term: str, text_length: int, skip: int) -> float:
        """
        Calculate the probability of finding a term by chance at a given skip.
        
        Formula: P = (N - (L-1)*S) * product(P(letter_i)) 
        Where N = text length, L = term length, S = skip interval
        """
        term = term.upper()
        term_length = len(term)
        
        # Number of possible starting positions for this skip
        # A term of length L at skip S spans (L-1)*S + 1 positions
        span = (term_length - 1) * skip + 1
        if span > text_length:
            return 0.0
        
        possible_starts = text_length - span + 1
        
        # Probability of matching each letter in sequence
        letter_prob = 1.0
        for char in term:
            letter_prob *= self.LETTER_FREQ.get(char, 0.001)
        
        # Expected occurrences = starts * letter_probability
        expected = possible_starts * letter_prob
        
        return expected
    
    def calculate_significance(self, term: str, text_length: int, 
                                skip: int, found_count: int = 1) -> Dict[str, Any]:
        """
        Calculate statistical significance of finding a term.
        
        Returns p-value and interpretation.
        """
        expected = self.calculate_els_probability(term, text_length, skip)
        
        # Use Poisson distribution for rare event probability
        # P(X >= k) where X ~ Poisson(lambda)
        if expected <= 0:
            p_value = 1.0
        else:
            # P(X >= 1) = 1 - P(X = 0) = 1 - e^(-lambda)
            p_value = 1 - math.exp(-expected)
        
        # Interpretation
        if p_value < 0.001:
            significance = "HIGHLY SIGNIFICANT (p < 0.001)"
            stars = "★★★"
        elif p_value < 0.01:
            significance = "VERY SIGNIFICANT (p < 0.01)"
            stars = "★★"
        elif p_value < 0.05:
            significance = "SIGNIFICANT (p < 0.05)"
            stars = "★"
        elif p_value < 0.10:
            significance = "MARGINALLY SIGNIFICANT (p < 0.10)"
            stars = "○"
        else:
            significance = "NOT SIGNIFICANT (likely random)"
            stars = ""
        
        return {
            "term": term,
            "skip": skip,
            "text_length": text_length,
            "expected_occurrences": round(expected, 6),
            "found_count": found_count,
            "p_value": round(p_value, 6),
            "significance": significance,
            "stars": stars
        }
    
    def calculate_gematria_significance(self, word: str, value: int, 
                                         sacred_numbers: Dict[int, str],
                                         vocabulary_size: int = 10000) -> Dict[str, Any]:
        """
        Calculate significance of a gematria match.
        
        Based on: How likely is it that a random word equals this value?
        """
        # Simplified model: assume values are uniformly distributed 1-500
        # Probability of hitting any specific value = 1/500
        num_sacred = len(sacred_numbers)
        
        # P(any word matches any sacred number) = num_sacred / value_range
        p_single = num_sacred / 500
        
        # P(at least one match in N words) = 1 - (1-p)^N
        # But we need to account for word length affecting value range
        word_len = len([c for c in word if c.isalpha()])
        
        # Longer words = higher values = narrower range matching
        if value in sacred_numbers:
            # Low values are more likely to be hit
            if value <= 50:
                p_value = 0.15  # Common
            elif value <= 100:
                p_value = 0.08  # Less common
            elif value <= 200:
                p_value = 0.03  # Uncommon
            else:
                p_value = 0.01  # Rare
        else:
            p_value = 1.0
        
        if p_value < 0.05:
            significance = "SIGNIFICANT"
            stars = "★"
        elif p_value < 0.10:
            significance = "NOTABLE"
            stars = "○"
        else:
            significance = "COMMON"
            stars = ""
        
        return {
            "word": word,
            "value": value,
            "meaning": sacred_numbers.get(value, ""),
            "p_value": round(p_value, 3),
            "significance": significance,
            "stars": stars
        }


# ============================================================================
# ELS GRID VISUALIZER
# ============================================================================

class ELSGridVisualizer:
    """Create visual grid displays for ELS findings."""
    
    def create_text_grid(self, text: str, width: int = 50) -> List[str]:
        """Convert text to grid rows."""
        clean = ''.join(c.upper() for c in text if c.isalpha())
        rows = []
        for i in range(0, len(clean), width):
            rows.append(clean[i:i+width])
        return rows
    
    def highlight_els_in_grid(self, text: str, term: str, skip: int, 
                               start: int, width: int = 50) -> str:
        """
        Create ASCII grid showing ELS pattern highlighted.
        """
        clean = ''.join(c.upper() for c in text if c.isalpha())
        
        # Find positions of the ELS term
        positions = set()
        for i, char in enumerate(term.upper()):
            pos = start + (i * skip)
            if 0 <= pos < len(clean):
                positions.add(pos)
        
        # Create grid with highlighting
        output = []
        output.append(f"\n┌{'─' * (width + 2)}┐")
        output.append(f"│ ELS: '{term}' at skip {skip} (start position {start})")
        output.append(f"│ Legend: [X] = part of ELS sequence")
        output.append(f"├{'─' * (width + 2)}┤")
        
        row_num = 0
        for i in range(0, len(clean), width):
            row = clean[i:i+width]
            display_row = ""
            for j, char in enumerate(row):
                pos = i + j
                if pos in positions:
                    display_row += f"[{char}]"
                else:
                    display_row += f" {char} "
            
            # Add row number
            output.append(f"│ {row_num:3d}: {display_row[:width*3]}")
            row_num += 1
            
            if row_num > 20:  # Limit display
                output.append(f"│ ... (truncated)")
                break
        
        output.append(f"└{'─' * (width + 2)}┘")
        
        return '\n'.join(output)
    
    def create_html_grid(self, text: str, els_matches: List[Dict], 
                          width: int = 60) -> str:
        """Create HTML grid with highlighted ELS patterns."""
        clean = ''.join(c.upper() for c in text if c.isalpha())
        
        # Collect all highlighted positions with colors
        highlights = {}  # pos -> (color, term)
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9']
        
        for idx, match in enumerate(els_matches[:6]):  # Limit to 6 for colors
            term = match['term']
            skip = match['skip']
            start = match.get('start', 0)
            color = colors[idx % len(colors)]
            
            for i, char in enumerate(term.upper()):
                pos = start + (i * skip)
                if 0 <= pos < len(clean):
                    highlights[pos] = (color, term)
        
        # Generate HTML
        html = """
        <style>
            .els-grid { 
                font-family: 'Courier New', monospace; 
                font-size: 12px; 
                line-height: 1.8;
                background: #1e1e1e;
                color: #d4d4d4;
                padding: 20px;
                border-radius: 8px;
                overflow-x: auto;
            }
            .els-grid .row { white-space: pre; }
            .els-grid .row-num { color: #666; margin-right: 10px; }
            .els-grid .highlight { 
                padding: 2px 4px; 
                border-radius: 3px; 
                font-weight: bold;
                color: #000;
            }
            .els-legend {
                margin-bottom: 15px;
                padding: 10px;
                background: #2d2d2d;
                border-radius: 5px;
            }
            .els-legend span {
                margin-right: 15px;
                padding: 3px 8px;
                border-radius: 3px;
                color: #000;
            }
        </style>
        <div class="els-grid">
            <div class="els-legend">
                <strong>Legend:</strong>
        """
        
        # Add legend
        shown_terms = set()
        for idx, match in enumerate(els_matches[:6]):
            if match['term'] not in shown_terms:
                color = colors[idx % len(colors)]
                html += f'<span style="background:{color}">{match["term"]} (skip {match["skip"]})</span>'
                shown_terms.add(match['term'])
        
        html += """
            </div>
            <div class="grid-content">
        """
        
        # Generate grid rows
        for row_idx, i in enumerate(range(0, min(len(clean), width * 25), width)):
            row = clean[i:i+width]
            html += f'<div class="row"><span class="row-num">{row_idx:3d}:</span>'
            
            for j, char in enumerate(row):
                pos = i + j
                if pos in highlights:
                    color, term = highlights[pos]
                    html += f'<span class="highlight" style="background:{color}" title="{term}">{char}</span>'
                else:
                    html += char
            
            html += '</div>'
        
        html += """
            </div>
        </div>
        """
        
        return html


# ============================================================================
# ENHANCED GEMATRIA ENGINE
# ============================================================================

class GematriaEngine:
    """Calculate numerical values using historical ciphers."""
    
    SACRED_NUMBERS = {
        17: "Bacon's Favorite Number",
        33: "Bacon (FB initials = 6+2=8? or 33=masonic)",
        67: "Francis (Simple)",
        100: "Francis Bacon (Simple)",
        111: "Triple Unity",
        144: "Light / Fibonacci",
        157: "Fra Rosicrosse (Simple)",
        222: "Master Builder",
        287: "Fra Rosicrosse (Kay)",
        333: "Ascended Masters",
        432: "Cosmic Cycle",
        666: "Number of the Beast",
        888: "Greek Jesus (Iesous)",
    }

    def calculate_bacon_simple(self, text: str) -> int:
        """Francis Bacon's Simple Cipher (24-letter Elizabethan)."""
        values = {}
        alphabet_24 = "ABCDEFGHIKLMNOPQRSTUWXYZ"
        for i, char in enumerate(alphabet_24, 1):
            values[char] = i
        values['J'] = values['I']
        values['V'] = values['U']
        
        clean = ''.join(c.upper() for c in text if c.isalpha())
        return sum(values.get(c, 0) for c in clean)
    
    def analyze_with_significance(self, text: str, stats: StatisticalAnalyzer) -> Dict[str, Any]:
        """Analyze text with statistical significance."""
        words = text.split()
        results = {
            "total_words": len(words),
            "significant_findings": [],
            "all_words": []
        }
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalpha())
            if len(clean_word) < 3:
                continue
            
            value = self.calculate_bacon_simple(clean_word)
            
            if value in self.SACRED_NUMBERS:
                sig = stats.calculate_gematria_significance(
                    clean_word, value, self.SACRED_NUMBERS
                )
                results["significant_findings"].append(sig)
            
            results["all_words"].append({
                "word": clean_word,
                "value": value
            })
        
        return results


# ============================================================================
# ELS ANALYZER WITH STATISTICS
# ============================================================================

class ELSAnalyzer:
    """Find Equidistant Letter Sequences with statistical analysis."""
    
    # Default search terms (will discuss with user about expanding)
    DEFAULT_TERMS = [
        "BACON", "SHAKE", "SPEAR", "WILL", "POET",
        "AUTHOR", "HIDDEN", "SECRET", "CIPHER", "CODE",
        "VERSE", "SONNET", "LOVE", "ROSE", "BEAUTY"
    ]
    
    def __init__(self, terms: List[str] = None):
        self.terms = [t.upper() for t in (terms or self.DEFAULT_TERMS)]
    
    def analyze_with_significance(self, text: str, stats: StatisticalAnalyzer,
                                    max_skip: int = 50) -> Dict[str, Any]:
        """Search for ELS patterns with statistical analysis."""
        clean_text = ''.join(c.upper() for c in text if c.isalpha())
        text_length = len(clean_text)
        
        matches = []
        
        for term in self.terms:
            for skip in range(2, min(max_skip, text_length // len(term) + 1)):
                for start in range(skip):
                    sequence = clean_text[start::skip]
                    if term in sequence:
                        idx = sequence.index(term)
                        actual_start = start + (idx * skip)
                        
                        # Calculate significance
                        sig = stats.calculate_significance(term, text_length, skip)
                        
                        matches.append({
                            "term": term,
                            "skip": skip,
                            "start": actual_start,
                            "direction": "forward",
                            "significance": sig
                        })
        
        # Sort by significance (lowest p-value first)
        matches.sort(key=lambda x: x['significance']['p_value'])
        
        return {
            "text_length": text_length,
            "terms_searched": self.terms,
            "matches_found": len(matches),
            "matches": matches,
            "most_significant": matches[:5] if matches else []
        }


# ============================================================================
# PDF/HTML REPORT GENERATOR
# ============================================================================

def generate_enhanced_html_report(results: Dict[str, Any], 
                                   els_grid_html: str = "") -> str:
    """Generate beautiful HTML report for PDF export."""
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CODEFINDER Analysis Report</title>
    <style>
        @page {{ 
            size: A4; 
            margin: 2cm; 
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Georgia', serif;
            line-height: 1.6;
            color: #333;
            background: #fff;
            padding: 40px;
            max-width: 900px;
            margin: 0 auto;
        }}
        
        h1 {{
            font-size: 28px;
            color: #1a365d;
            text-align: center;
            border-bottom: 3px double #2c5282;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}
        
        h2 {{
            font-size: 20px;
            color: #2c5282;
            margin: 25px 0 15px 0;
            padding-left: 10px;
            border-left: 4px solid #4299e1;
        }}
        
        h3 {{
            font-size: 16px;
            color: #2d3748;
            margin: 20px 0 10px 0;
        }}
        
        .meta {{
            text-align: center;
            color: #718096;
            margin-bottom: 30px;
            font-style: italic;
        }}
        
        .summary-box {{
            background: linear-gradient(135deg, #ebf8ff 0%, #e6fffa 100%);
            border: 1px solid #90cdf4;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        .finding {{
            background: #f7fafc;
            border-left: 4px solid #48bb78;
            padding: 12px 15px;
            margin: 10px 0;
            border-radius: 0 4px 4px 0;
        }}
        
        .finding.significant {{
            border-left-color: #ed8936;
            background: #fffaf0;
        }}
        
        .finding.highly-significant {{
            border-left-color: #e53e3e;
            background: #fff5f5;
        }}
        
        .stars {{
            color: #d69e2e;
            font-size: 18px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 14px;
        }}
        
        th {{
            background: #2c5282;
            color: white;
            padding: 10px;
            text-align: left;
        }}
        
        td {{
            padding: 8px 10px;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        tr:nth-child(even) {{
            background: #f7fafc;
        }}
        
        .p-value {{
            font-family: 'Courier New', monospace;
            font-size: 12px;
        }}
        
        .interpretation {{
            background: #edf2f7;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            font-style: italic;
        }}
        
        .grid-section {{
            margin: 30px 0;
            page-break-inside: avoid;
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            text-align: center;
            color: #718096;
            font-size: 12px;
        }}
        
        @media print {{
            body {{ padding: 0; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <h1>🔍 CODEFINDER Analysis Report</h1>
    
    <div class="meta">
        <strong>Source:</strong> {results.get('source', 'Unknown')}<br>
        <strong>Analyzed:</strong> {results.get('timestamp', datetime.now().isoformat())}<br>
        <strong>Text Length:</strong> {results.get('text_length', 'N/A')} characters
    </div>
    
    <div class="summary-box">
        <h3>📊 Executive Summary</h3>
        <p>This analysis examines the text for hidden patterns using multiple cipher systems, 
        equidistant letter sequences (ELS), and structural analysis. Statistical significance 
        is calculated to distinguish intentional encoding from random chance.</p>
    </div>
"""
    
    # Gematria Findings
    gematria = results.get('gematria', {})
    if gematria.get('significant_findings'):
        html += """
    <h2>🔢 Gematria Analysis</h2>
    <p>Words matching "sacred numbers" in Francis Bacon's Simple Cipher:</p>
    
    <table>
        <thead>
            <tr>
                <th>Word</th>
                <th>Value</th>
                <th>Meaning</th>
                <th>Significance</th>
            </tr>
        </thead>
        <tbody>
"""
        for f in gematria['significant_findings'][:15]:
            sig_class = ""
            if "★★" in f.get('stars', ''):
                sig_class = "highly-significant"
            elif "★" in f.get('stars', ''):
                sig_class = "significant"
            
            html += f"""
            <tr class="{sig_class}">
                <td><strong>{f['word']}</strong></td>
                <td>{f['value']}</td>
                <td>{f['meaning']}</td>
                <td><span class="stars">{f.get('stars', '')}</span> {f.get('significance', '')}</td>
            </tr>
"""
        html += """
        </tbody>
    </table>
"""
    
    # ELS Findings
    els = results.get('els', {})
    if els.get('matches'):
        html += """
    <h2>🔍 ELS (Equidistant Letter Sequence) Analysis</h2>
    <p>Hidden words found by reading every Nth letter:</p>
    
    <table>
        <thead>
            <tr>
                <th>Term</th>
                <th>Skip</th>
                <th>Expected by Chance</th>
                <th>P-Value</th>
                <th>Significance</th>
            </tr>
        </thead>
        <tbody>
"""
        for match in els.get('most_significant', [])[:10]:
            sig = match.get('significance', {})
            html += f"""
            <tr>
                <td><strong>{match['term']}</strong></td>
                <td>{match['skip']}</td>
                <td class="p-value">{sig.get('expected_occurrences', 'N/A')}</td>
                <td class="p-value">{sig.get('p_value', 'N/A')}</td>
                <td><span class="stars">{sig.get('stars', '')}</span> {sig.get('significance', '')}</td>
            </tr>
"""
        html += """
        </tbody>
    </table>
"""
    
    # ELS Grid Visualization
    if els_grid_html:
        html += f"""
    <div class="grid-section">
        <h2>📊 ELS Grid Visualization</h2>
        <p>Visual representation of ELS patterns in the text grid:</p>
        {els_grid_html}
    </div>
"""
    
    # Interpretation
    html += """
    <div class="interpretation">
        <h3>📖 Interpretation Guidelines</h3>
        <ul>
            <li><strong>★★★ Highly Significant (p < 0.001):</strong> Extremely unlikely to occur by chance. Strong evidence of intentional encoding.</li>
            <li><strong>★★ Very Significant (p < 0.01):</strong> Very unlikely random occurrence. Warrants serious investigation.</li>
            <li><strong>★ Significant (p < 0.05):</strong> Standard threshold for statistical significance.</li>
            <li><strong>○ Marginally Significant (p < 0.10):</strong> Suggestive but not conclusive.</li>
            <li><strong>No stars:</strong> Could easily occur by random chance.</li>
        </ul>
    </div>
"""
    
    html += f"""
    <div class="footer">
        <p>Generated by CODEFINDER v1.0</p>
        <p>For research purposes. Cross-reference findings with historical context.</p>
    </div>
</body>
</html>
"""
    
    return html


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

# Sample text (Sonnets Dedication + Sonnet 1)
SAMPLE_TEXT = """TO.THE.ONLIE.BEGETTER.OF.
THESE.INSVING.SONNETS.
Mr.W.H. ALL.HAPPINESSE.
AND.THAT.ETERNITIE.
PROMISED.
BY.
OVR.EVER-LIVING.POET.
WISHETH.
THE.WELL-WISHING.
ADVENTVRER.IN.
SETTING.
FORTH.
T. T.

SHAKE-SPEARES SONNETS.

From fairest creatures we desire increase,
That thereby beauties Rose might never die,
But as the riper should by time decease,
His tender heire might beare his memory:
But thou contracted to thine owne bright eyes,
Feed'st thy lights flame with selfe substantiall fewell,
Making a famine where aboundance lies,
Thy selfe thy foe, to thy sweet selfe too cruell:
Thou that art now the worlds fresh ornament,
And only herauld to the gaudy spring,
Within thine owne bud buriest thy content,
And tender chorle makst wast in niggarding:
   Pitty the world, or else this glutton be,
   To eate the worlds due, by the grave and thee.
"""


def main():
    print("\n" + "=" * 70)
    print("    CODEFINDER - ENHANCED ANALYSIS")
    print("    With Statistical Significance & Grid Visualization")
    print("=" * 70)
    
    # Initialize analyzers
    stats = StatisticalAnalyzer()
    gematria = GematriaEngine()
    els = ELSAnalyzer()
    grid_viz = ELSGridVisualizer()
    
    text = SAMPLE_TEXT
    clean_text = ''.join(c.upper() for c in text if c.isalpha())
    
    print(f"\n📝 Analyzing: Shakespeare Sonnets Dedication + Sonnet 1")
    print(f"   Text length: {len(clean_text)} letters")
    
    # Run Gematria with significance
    print(f"\n{'─'*50}")
    print("🔢 GEMATRIA ANALYSIS (with significance)")
    print(f"{'─'*50}")
    
    gematria_results = gematria.analyze_with_significance(text, stats)
    
    if gematria_results['significant_findings']:
        print(f"\n   Found {len(gematria_results['significant_findings'])} significant words:")
        for f in gematria_results['significant_findings'][:8]:
            print(f"   {f['stars']:3} {f['word']:15} = {f['value']:3} ({f['meaning']}) [{f['significance']}]")
    
    # Run ELS with significance
    print(f"\n{'─'*50}")
    print("🔍 ELS ANALYSIS (with significance)")
    print(f"{'─'*50}")
    
    els_results = els.analyze_with_significance(text, stats, max_skip=30)
    
    print(f"\n   Terms searched: {', '.join(els.terms[:5])}...")
    print(f"   Total matches: {els_results['matches_found']}")
    
    if els_results['most_significant']:
        print(f"\n   Most significant ELS findings:")
        for m in els_results['most_significant'][:5]:
            sig = m['significance']
            print(f"   {sig['stars']:3} '{m['term']}' at skip {m['skip']:2d} - p={sig['p_value']:.4f} ({sig['significance']})")
    
    # Generate grid visualization
    print(f"\n{'─'*50}")
    print("📊 ELS GRID VISUALIZATION")
    print(f"{'─'*50}")
    
    if els_results['matches']:
        # Show ASCII grid for first match
        first_match = els_results['matches'][0]
        ascii_grid = grid_viz.highlight_els_in_grid(
            text, first_match['term'], first_match['skip'], first_match['start']
        )
        print(ascii_grid)
        
        # Generate HTML grid
        html_grid = grid_viz.create_html_grid(text, els_results['matches'][:4])
    else:
        html_grid = ""
    
    # Compile results
    results = {
        "source": "Shakespeare Sonnets Dedication + Sonnet 1",
        "timestamp": datetime.now().isoformat(),
        "text_length": len(clean_text),
        "gematria": gematria_results,
        "els": els_results
    }
    
    # Generate HTML report
    html_report = generate_enhanced_html_report(results, html_grid)
    
    # Save outputs
    output_dir = PROJECT_ROOT / "demo_results"
    output_dir.mkdir(exist_ok=True)
    
    html_path = output_dir / "enhanced_analysis_report.html"
    with open(html_path, 'w') as f:
        f.write(html_report)
    
    json_path = output_dir / "enhanced_analysis_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print("ENHANCED ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"📄 HTML Report: {html_path}")
    print(f"📁 JSON Results: {json_path}")
    print(f"\n💡 Open the HTML file in a browser and use Print > Save as PDF")


if __name__ == "__main__":
    main()
