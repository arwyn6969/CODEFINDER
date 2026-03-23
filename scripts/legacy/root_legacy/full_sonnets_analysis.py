#!/usr/bin/env python3
"""
CODEFINDER Full Sonnets Analysis
=================================
Complete analysis of Shakespeare's 154 Sonnets
with expanded search terms and statistical significance.
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

PROJECT_ROOT = Path(__file__).parent

# ============================================================================
# EXPANDED SEARCH TERMS
# ============================================================================

# ELS Search Terms - Expanded list per user request
ELS_TERMS = [
    # Original authorship terms
    "BACON", "SHAKE", "SPEAR", "WILL", "POET", "AUTHOR",
    
    # Cipher/Hidden terms
    "HIDDEN", "SECRET", "CIPHER", "CODE", "HIDE", "CLUE",
    
    # Poetry terms
    "VERSE", "SONNET", "LOVE", "ROSE", "BEAUTY", "WRITE",
    
    # User-requested personal terms
    "MARY", "MOTHER", "TWIN", "TWINS",
    
    # User-requested mystical terms
    "SALT", "STAB", "LIGHT", "DVER", "ARK", "TABOT",
    
    # Religious/Kabbalistic terms (with spelling variations)
    "KABBALAH", "KABALAH", "CABALA", "QABALA", "QABALAH",
    
    # Personal names
    "ARWYN", "KEVIN",
    
    # Elizabethan authorship candidates
    "MARLOWE", "OXFORD", "DERBY", "RUTLAND", "EDWARD",
    
    # Rosicrucian/Mystical
    "CROSS", "TEMPLE", "CHRIST", "JESUS", "ORDER",
    
    # Additional significant terms
    "TRUTH", "DARK", "SOUL", "HEART", "QUILL", "TRUE"
]

# Sacred Gematria Numbers - Expanded
SACRED_NUMBERS = {
    # Bacon-related
    17: "Bacon's Favorite Number",
    33: "FB initials / Masonic",
    53: "BACON (Simple)",
    67: "FRANCIS (Simple)",
    100: "FRANCIS BACON (Simple)",
    137: "Fine Structure Constant / Kabbalah",
    157: "FRA ROSICROSSE (Simple)",
    287: "FRA ROSICROSSE (Kay)",
    
    # Biblical/Religious
    5: "Grace / Pentateuch",
    7: "Divine Completion",
    12: "Tribes / Apostles",
    40: "Testing / Trial",
    144: "Light / 12x12",
    666: "Number of the Beast",
    888: "Greek Jesus (Iesous)",
    
    # Masonic/Mystery School
    3: "Trinity",
    9: "Completion",
    11: "Master Number",
    22: "Master Builder",
    111: "Triple Unity",
    222: "Manifestation",
    333: "Ascended Masters",
    444: "Protection",
    555: "Change",
    777: "Divine Perfection",
    
    # Mathematical constants (approximated)
    314: "Pi",
    161: "Phi (Golden Ratio)",
    271: "e (Euler's)",
    432: "Cosmic Frequency",
    
    # Shakespeare-specific
    74: "WILLIAM (Simple)",
    136: "SHAKESPEARE (partial)",
    
    # Kabbalistic
    26: "YHWH (Tetragrammaton)",
    72: "Shem HaMephorash",
    
    # User personal (corrected values)
    53: "MARY (Simple)",  # M=12 A=1 R=17 Y=23 = 53
    57: "KEVIN (Simple)", # K=10 E=5 V=20 I=9 N=13 = 57
    75: "ARWYN (Simple)", # A=1 R=17 W=21 Y=23 N=13 = 75
}

# ============================================================================
# ANALYSIS CLASSES
# ============================================================================

class StatisticalAnalyzer:
    """Calculate statistical significance."""
    
    LETTER_FREQ = {
        'A': 0.0812, 'B': 0.0149, 'C': 0.0278, 'D': 0.0425, 'E': 0.1202,
        'F': 0.0223, 'G': 0.0202, 'H': 0.0609, 'I': 0.0697, 'J': 0.0015,
        'K': 0.0077, 'L': 0.0403, 'M': 0.0241, 'N': 0.0675, 'O': 0.0751,
        'P': 0.0193, 'Q': 0.0010, 'R': 0.0599, 'S': 0.0633, 'T': 0.0906,
        'U': 0.0276, 'V': 0.0098, 'W': 0.0236, 'X': 0.0015, 'Y': 0.0197,
        'Z': 0.0007
    }
    
    def calculate_significance(self, term: str, text_length: int, 
                                skip: int, found_count: int = 1) -> Dict[str, Any]:
        term = term.upper()
        term_length = len(term)
        
        span = (term_length - 1) * skip + 1
        if span > text_length:
            return {"p_value": 1.0, "significance": "N/A", "stars": ""}
        
        possible_starts = text_length - span + 1
        
        letter_prob = 1.0
        for char in term:
            letter_prob *= self.LETTER_FREQ.get(char, 0.001)
        
        expected = possible_starts * letter_prob
        
        if expected <= 0:
            p_value = 1.0
        else:
            p_value = 1 - math.exp(-expected)
        
        if p_value < 0.001:
            significance = "HIGHLY SIGNIFICANT"
            stars = "★★★"
        elif p_value < 0.01:
            significance = "VERY SIGNIFICANT"
            stars = "★★"
        elif p_value < 0.05:
            significance = "SIGNIFICANT"
            stars = "★"
        elif p_value < 0.10:
            significance = "MARGINALLY SIGNIFICANT"
            stars = "○"
        else:
            significance = "NOT SIGNIFICANT"
            stars = ""
        
        return {
            "term": term,
            "skip": skip,
            "expected": round(expected, 6),
            "p_value": round(p_value, 6),
            "significance": significance,
            "stars": stars
        }


class GematriaEngine:
    """Calculate Gematria values."""
    
    def calculate_bacon_simple(self, text: str) -> int:
        values = {}
        alphabet_24 = "ABCDEFGHIKLMNOPQRSTUWXYZ"
        for i, char in enumerate(alphabet_24, 1):
            values[char] = i
        values['J'] = values['I']
        values['V'] = values['U']
        
        clean = ''.join(c.upper() for c in text if c.isalpha())
        return sum(values.get(c, 0) for c in clean)
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        words = text.split()
        findings = []
        word_values = []
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalpha())
            if len(clean_word) < 2:
                continue
            
            value = self.calculate_bacon_simple(clean_word)
            word_values.append({"word": clean_word, "value": value})
            
            if value in SACRED_NUMBERS:
                findings.append({
                    "word": clean_word,
                    "value": value,
                    "meaning": SACRED_NUMBERS[value]
                })
        
        return {
            "total_words": len(words),
            "significant_findings": findings,
            "word_values": word_values
        }


class ELSAnalyzer:
    """Find Equidistant Letter Sequences."""
    
    def __init__(self, terms: List[str]):
        self.terms = [t.upper() for t in terms]
    
    def analyze_text(self, text: str, stats: StatisticalAnalyzer,
                     max_skip: int = 100) -> Dict[str, Any]:
        clean_text = ''.join(c.upper() for c in text if c.isalpha())
        text_length = len(clean_text)
        
        matches = []
        term_counts = Counter()
        
        print(f"   Searching {len(self.terms)} terms in {text_length:,} letters...")
        
        for term_idx, term in enumerate(self.terms):
            if term_idx % 10 == 0:
                print(f"   ... processed {term_idx}/{len(self.terms)} terms", end='\r')
            
            term_len = len(term)
            max_possible_skip = min(max_skip, text_length // term_len)
            
            for skip in range(2, max_possible_skip + 1):
                for start in range(skip):
                    sequence = clean_text[start::skip]
                    if term in sequence:
                        idx = sequence.index(term)
                        actual_start = start + (idx * skip)
                        
                        sig = stats.calculate_significance(term, text_length, skip)
                        
                        if sig['p_value'] < 0.15:  # Record notable finds
                            matches.append({
                                "term": term,
                                "skip": skip,
                                "start": actual_start,
                                "significance": sig
                            })
                            term_counts[term] += 1
        
        print(f"   ... completed all {len(self.terms)} terms      ")
        
        # Sort by significance
        matches.sort(key=lambda x: x['significance']['p_value'])
        
        return {
            "text_length": text_length,
            "terms_searched": len(self.terms),
            "matches_found": len(matches),
            "term_counts": dict(term_counts),
            "most_significant": matches[:50],
            "all_matches": matches
        }


# ============================================================================
# HTML REPORT GENERATOR
# ============================================================================

def generate_full_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive HTML report."""
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CODEFINDER Full Sonnets Analysis</title>
    <style>
        @page {{ size: A4; margin: 1.5cm; }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Georgia', serif;
            line-height: 1.5;
            color: #333;
            background: #fff;
            padding: 30px;
            max-width: 1000px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 24px;
            color: #1a365d;
            text-align: center;
            border-bottom: 3px double #2c5282;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }}
        h2 {{
            font-size: 18px;
            color: #2c5282;
            margin: 20px 0 12px 0;
            padding-left: 10px;
            border-left: 4px solid #4299e1;
        }}
        .meta {{
            text-align: center;
            color: #666;
            margin-bottom: 25px;
            font-size: 14px;
        }}
        .summary {{
            background: linear-gradient(135deg, #ebf8ff, #e6fffa);
            border: 1px solid #90cdf4;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-box {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            color: #2c5282;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 13px;
        }}
        th {{
            background: #2c5282;
            color: white;
            padding: 10px 8px;
            text-align: left;
        }}
        td {{
            padding: 8px;
            border-bottom: 1px solid #e2e8f0;
        }}
        tr:nth-child(even) {{ background: #f7fafc; }}
        .stars {{ color: #d69e2e; font-size: 16px; }}
        .highlight {{ background: #fef3c7; }}
        .highly-sig {{ background: #fed7d7; }}
        .very-sig {{ background: #feebc8; }}
        .sig {{ background: #fefcbf; }}
        .footer {{
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #e2e8f0;
            text-align: center;
            color: #666;
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <h1>🔍 CODEFINDER Full Sonnets Analysis</h1>
    
    <div class="meta">
        <strong>Source:</strong> {results.get('source', 'Unknown')}<br>
        <strong>Analyzed:</strong> {results.get('timestamp', '')}<br>
        <strong>Total Letters:</strong> {results.get('text_length', 0):,}
    </div>
    
    <div class="summary">
        <h3>📊 Analysis Summary</h3>
        <div class="summary-grid">
            <div class="stat-box">
                <div class="stat-value">154</div>
                <div class="stat-label">Sonnets Analyzed</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{results.get('text_length', 0):,}</div>
                <div class="stat-label">Letters Searched</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(results.get('els', {}).get('most_significant', []))}</div>
                <div class="stat-label">Significant ELS</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(results.get('gematria', {}).get('significant_findings', []))}</div>
                <div class="stat-label">Sacred Numbers</div>
            </div>
        </div>
    </div>
"""
    
    # ELS Findings
    els = results.get('els', {})
    if els.get('most_significant'):
        html += """
    <h2>🔍 Most Significant ELS Findings</h2>
    <p>Equidistant Letter Sequences with statistical significance:</p>
    <table>
        <thead>
            <tr><th>Term</th><th>Skip</th><th>Expected</th><th>P-Value</th><th>Significance</th></tr>
        </thead>
        <tbody>
"""
        for m in els['most_significant'][:40]:
            sig = m['significance']
            p_val = sig['p_value']
            
            row_class = ""
            if p_val < 0.001:
                row_class = "highly-sig"
            elif p_val < 0.01:
                row_class = "very-sig"
            elif p_val < 0.05:
                row_class = "sig"
            
            html += f"""
            <tr class="{row_class}">
                <td><strong>{m['term']}</strong></td>
                <td>{m['skip']}</td>
                <td>{sig['expected']:.4f}</td>
                <td>{sig['p_value']:.6f}</td>
                <td><span class="stars">{sig['stars']}</span> {sig['significance']}</td>
            </tr>
"""
        html += "</tbody></table>"
    
    # Term frequency summary
    if els.get('term_counts'):
        html += """
    <h2>📈 Terms Found (Frequency)</h2>
    <table>
        <thead><tr><th>Term</th><th>Occurrences</th></tr></thead>
        <tbody>
"""
        sorted_terms = sorted(els['term_counts'].items(), key=lambda x: -x[1])
        for term, count in sorted_terms[:25]:
            html += f"<tr><td><strong>{term}</strong></td><td>{count}</td></tr>"
        html += "</tbody></table>"
    
    # Gematria Findings
    gematria = results.get('gematria', {})
    if gematria.get('significant_findings'):
        html += """
    <h2>🔢 Gematria Findings (Sacred Numbers)</h2>
    <table>
        <thead><tr><th>Word</th><th>Value</th><th>Meaning</th></tr></thead>
        <tbody>
"""
        # Group by value
        by_value = {}
        for f in gematria['significant_findings']:
            val = f['value']
            if val not in by_value:
                by_value[val] = []
            by_value[val].append(f['word'])
        
        for val, words in sorted(by_value.items()):
            meaning = SACRED_NUMBERS.get(val, "")
            unique_words = list(set(words))[:8]
            words_display = ", ".join(unique_words)
            if len(words) > 8:
                words_display += f" (+{len(words)-8} more)"
            html += f"<tr><td>{words_display}</td><td><strong>{val}</strong></td><td>{meaning}</td></tr>"
        
        html += "</tbody></table>"
    
    html += f"""
    <div class="footer">
        <p>Generated by CODEFINDER v1.0 • {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <p>Statistical significance based on expected probability of random occurrence</p>
    </div>
</body>
</html>
"""
    return html


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("    CODEFINDER - FULL SONNETS ANALYSIS")
    print("    With Expanded Terms & Statistical Significance")
    print("=" * 70)
    
    # Find text file
    text_path = PROJECT_ROOT / "data/sources/shakespeare_sonnets_text.txt"
    
    if not text_path.exists():
        print(f"\n❌ Text file not found: {text_path}")
        return
    
    print(f"\n📂 Found: {text_path.name}")
    
    # Load text
    with open(text_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # Remove Gutenberg header/footer
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    if start_marker in full_text:
        full_text = full_text.split(start_marker)[1]
    if end_marker in full_text:
        full_text = full_text.split(end_marker)[0]
    
    clean_text = ''.join(c.upper() for c in full_text if c.isalpha())
    print(f"   Total letters: {len(clean_text):,}")
    
    # Initialize analyzers
    stats = StatisticalAnalyzer()
    gematria = GematriaEngine()
    els = ELSAnalyzer(ELS_TERMS)
    
    # Run Gematria analysis
    print(f"\n{'─'*50}")
    print("🔢 GEMATRIA ANALYSIS")
    print(f"{'─'*50}")
    
    gematria_start = time.time()
    gematria_results = gematria.analyze_text(full_text)
    gematria_time = time.time() - gematria_start
    
    print(f"   Analyzed {gematria_results['total_words']:,} words")
    print(f"   Found {len(gematria_results['significant_findings'])} sacred number matches")
    print(f"   Time: {gematria_time:.1f}s")
    
    # Show top gematria findings
    if gematria_results['significant_findings']:
        print(f"\n   Top sacred number findings:")
        value_counts = Counter(f['value'] for f in gematria_results['significant_findings'])
        for value, count in value_counts.most_common(10):
            print(f"      {value:4d} ({SACRED_NUMBERS.get(value, '?')[:25]}): {count} occurrences")
    
    # Run ELS analysis
    print(f"\n{'─'*50}")
    print("🔍 ELS ANALYSIS")
    print(f"{'─'*50}")
    
    els_start = time.time()
    els_results = els.analyze_text(full_text, stats, max_skip=100)
    els_time = time.time() - els_start
    
    print(f"\n   Search complete!")
    print(f"   Total significant matches: {els_results['matches_found']}")
    print(f"   Time: {els_time:.1f}s")
    
    # Show most significant ELS
    if els_results['most_significant']:
        print(f"\n   🎯 TOP 20 MOST SIGNIFICANT ELS FINDINGS:")
        for i, m in enumerate(els_results['most_significant'][:20], 1):
            sig = m['significance']
            print(f"   {i:2d}. {sig['stars']:3} {m['term']:12} skip={m['skip']:3d}  p={sig['p_value']:.6f}")
    
    # Show term frequency
    if els_results['term_counts']:
        print(f"\n   📊 TERMS FOUND (top 15):")
        sorted_terms = sorted(els_results['term_counts'].items(), key=lambda x: -x[1])
        for term, count in sorted_terms[:15]:
            print(f"      {term:12}: {count} occurrences")
    
    # Compile results
    results = {
        "source": "Shakespeare Sonnets (Project Gutenberg)",
        "timestamp": datetime.now().isoformat(),
        "text_length": len(clean_text),
        "gematria": gematria_results,
        "gematria_time": gematria_time,
        "els": els_results,
        "els_time": els_time,
        "search_terms": ELS_TERMS,
        "sacred_numbers": {str(k): v for k, v in SACRED_NUMBERS.items()}
    }
    
    # Save results
    output_dir = PROJECT_ROOT / "demo_results"
    output_dir.mkdir(exist_ok=True)
    
    json_path = output_dir / "full_sonnets_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    html_path = output_dir / "full_sonnets_report.html"
    with open(html_path, 'w') as f:
        f.write(generate_full_report(results))
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"📁 JSON: {json_path}")
    print(f"📄 HTML: {html_path}")
    print(f"\n⏱️  Total time: {time.time() - els_start + gematria_time:.1f}s")
    print(f"\n💡 Open: file://{html_path.absolute()}")


if __name__ == "__main__":
    main()
