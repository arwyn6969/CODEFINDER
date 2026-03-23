#!/usr/bin/env python3
"""
Quote Character Deep Forensic Analysis

Investigates the extraordinary disparity in quote characters between
Wright (159 straight quotes) and Aspley (11 straight quotes).
"""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_data(json_path: str) -> list:
    """Load page comparison data."""
    with open(json_path, 'r') as f:
        return json.load(f)['results']

def analyze_quote_distribution(results: list, output_dir: str):
    """Analyze where quotes appear across pages."""
    
    output_path = Path(output_dir) / 'quote_analysis'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Quote characters to track
    quote_chars = {
        "'": "straight quote",
        "'": "left single quote",
        "'": "right single quote", 
        '"': "double quote",
        '"': "left double quote",
        '"': "right double quote",
    }
    
    # Track by page
    page_data = defaultdict(lambda: defaultdict(lambda: {'wright': 0, 'aspley': 0}))
    
    for result in results:
        page = result['page']
        for char, data in result.get('char_deltas', {}).items():
            if char in quote_chars:
                page_data[page][char]['wright'] += data['wright']
                page_data[page][char]['aspley'] += data['aspley']
    
    # Aggregate totals
    totals = defaultdict(lambda: {'wright': 0, 'aspley': 0})
    for page, chars in page_data.items():
        for char, counts in chars.items():
            totals[char]['wright'] += counts['wright']
            totals[char]['aspley'] += counts['aspley']
    
    print("="*80)
    print("QUOTE CHARACTER FORENSIC ANALYSIS")
    print("="*80)
    print()
    print("AGGREGATE QUOTE COUNTS:")
    print("-"*60)
    print(f"{'Character':<25}{'Wright':<12}{'Aspley':<12}{'Delta':<12}")
    print("-"*60)
    
    for char, name in quote_chars.items():
        w = totals[char]['wright']
        a = totals[char]['aspley']
        delta = a - w
        print(f"{name:<25}{w:<12}{a:<12}{delta:+<12}")
    
    # ============ Track Straight Quote by Page ============
    print("\n" + "="*80)
    print("STRAIGHT QUOTE (') DISTRIBUTION BY PAGE")
    print("="*80)
    
    pages = sorted(page_data.keys())
    wright_straight = []
    aspley_straight = []
    
    for page in pages:
        w = page_data[page]["'"]['wright']
        a = page_data[page]["'"]['aspley']
        wright_straight.append(w)
        aspley_straight.append(a)
        if w > 0 or a > 0:
            print(f"Page {page:>2}: Wright={w:>3}, Aspley={a:>3}, Delta={a-w:+>4}")
    
    # ============ Visualization ============
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Quote Character Forensic Analysis", fontsize=14, fontweight='bold')
    
    # 1. Quote types comparison
    ax1 = axes[0, 0]
    quote_names = list(quote_chars.values())
    wright_counts = [totals[c]['wright'] for c in quote_chars.keys()]
    aspley_counts = [totals[c]['aspley'] for c in quote_chars.keys()]
    
    x = np.arange(len(quote_names))
    width = 0.35
    
    ax1.bar(x - width/2, wright_counts, width, label='Wright', color='#2E86AB')
    ax1.bar(x + width/2, aspley_counts, width, label='Aspley', color='#A23B72')
    ax1.set_xticks(x)
    ax1.set_xticklabels(quote_names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Count')
    ax1.set_title('Quote Character Counts by Type')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Straight quote by page
    ax2 = axes[0, 1]
    pages_arr = np.array(pages)
    ax2.bar(pages_arr - 0.2, wright_straight, 0.4, label='Wright', color='#2E86AB', alpha=0.8)
    ax2.bar(pages_arr + 0.2, aspley_straight, 0.4, label='Aspley', color='#A23B72', alpha=0.8)
    ax2.set_xlabel('Page Number')
    ax2.set_ylabel('Straight Quote Count')
    ax2.set_title("Straight Quote (') Distribution by Page")
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Comparative pie chart  
    ax3 = axes[1, 0]
    
    # Wright edition quote composition
    wright_quote_types = ['straight', 'left single', 'right single', 'double', 'curly double']
    wright_quote_counts = [
        totals["'"]['wright'],
        totals["'"]['wright'],
        totals["'"]['wright'],
        totals['"']['wright'],
        totals['"']['wright'] + totals['"']['wright']
    ]
    
    # Filter out zeros
    nonzero_labels = []
    nonzero_values = []
    for label, value in zip(wright_quote_types, wright_quote_counts):
        if value > 0:
            nonzero_labels.append(label)
            nonzero_values.append(value)
    
    if nonzero_values:
        colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(nonzero_values)))
        ax3.pie(nonzero_values, labels=nonzero_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Wright Edition Quote Composition')
    
    # 4. Aspley edition quote composition
    ax4 = axes[1, 1]
    
    aspley_quote_counts = [
        totals["'"]['aspley'],
        totals["'"]['aspley'],
        totals["'"]['aspley'],
        totals['"']['aspley'],
        totals['"']['aspley'] + totals['"']['aspley']
    ]
    
    nonzero_labels_a = []
    nonzero_values_a = []
    for label, value in zip(wright_quote_types, aspley_quote_counts):
        if value > 0:
            nonzero_labels_a.append(label)
            nonzero_values_a.append(value)
    
    if nonzero_values_a:
        colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(nonzero_values_a)))
        ax4.pie(nonzero_values_a, labels=nonzero_labels_a, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Aspley Edition Quote Composition')
    
    plt.tight_layout()
    
    vis_path = output_path / 'quote_analysis.png'
    plt.savefig(vis_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nVisualization saved: {vis_path}")
    
    # ============ Forensic Interpretation ============
    print("\n" + "="*80)
    print("FORENSIC INTERPRETATION")
    print("="*80)
    
    straight_w = totals["'"]['wright']
    straight_a = totals["'"]['aspley']
    
    print(f"""
The extraordinary disparity in straight quote characters:
  - Wright: {straight_w} instances
  - Aspley: {straight_a} instances
  - Ratio: {straight_w / straight_a if straight_a > 0 else 'inf'}:1

POSSIBLE EXPLANATIONS:

1. OCR INTERPRETATION DIFFERENCE:
   The Tesseract engine may be interpreting the same typographical
   apostrophe/prime mark differently based on scan quality or
   pre-processing. Wright's lower-quality scan may trigger straight
   quote detection while Aspley's cleaner scan triggers curly quotes.

2. ACTUAL TYPOGRAPHICAL DIFFERENCE:
   The Wright and Aspley editions may have been set using different
   apostrophe sorts from the type case. This would be a significant
   bibliographical finding.

3. SCANNER/DIGITIZATION ARTIFACT:
   The Folger Library may have used different scanning equipment or
   settings for the two copies, introducing systematic character
   rendering differences.

RECOMMENDED NEXT STEP:
   Visual extraction and side-by-side comparison of apostrophe
   instances from pages with highest disparity.
""")
    
    # Find pages with highest straight quote disparity
    disparities = []
    for page in pages:
        w = page_data[page]["'"]['wright']
        a = page_data[page]["'"]['aspley']
        disparities.append((page, w, a, w - a))
    
    disparities.sort(key=lambda x: abs(x[3]), reverse=True)
    
    print("TOP 10 PAGES BY STRAIGHT QUOTE DISPARITY:")
    print("-"*50)
    for page, w, a, delta in disparities[:10]:
        if delta != 0:
            print(f"  Page {page:>2}: Wright={w:>2}, Aspley={a:>2}, Δ={delta:+>3}")
    
    return output_path

def main():
    json_path = "reports/page_forensics/page_comparison.json"
    output_dir = "reports/page_forensics"
    
    results = load_data(json_path)
    analyze_quote_distribution(results, output_dir)
    
    print("\n✅ Quote analysis complete!")

if __name__ == "__main__":
    main()
