#!/usr/bin/env python3
"""
Aspect Ratio Analysis (Task 7.1)

If the editions use the same type (just different scan resolution),
their aspect ratios (width/height) should be IDENTICAL.

Tests H5: Aspect ratios within ±5% for matched characters
"""

import sqlite3
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

# Configuration
DB_PATH = Path("data/codefinder.db")
REPORT_DIR = Path("reports/aspect_ratio")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

def get_aspect_ratios():
    """Get aspect ratio for all characters by edition."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            ci.character,
            s.name as source,
            ci.width,
            ci.height,
            CAST(ci.width AS REAL) / ci.height as aspect_ratio
        FROM character_instances ci
        JOIN pages p ON ci.page_id = p.id
        JOIN sources s ON p.source_id = s.id
        WHERE ci.confidence > 0.5
          AND ci.width > 3 AND ci.height > 3
    """)
    
    data = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return data

def analyze_by_character(data):
    """Group by character and compare aspect ratios."""
    
    # Group: {char: {'aspley': [ratios], 'wright': [ratios]}}
    by_char = defaultdict(lambda: {'aspley': [], 'wright': []})
    
    for d in data:
        by_char[d['character']][d['source']].append(d['aspect_ratio'])
    
    comparisons = []
    
    for char, editions in by_char.items():
        if not editions['aspley'] or not editions['wright']:
            continue
        if len(editions['aspley']) < 10 or len(editions['wright']) < 10:
            continue
        
        a_mean = np.mean(editions['aspley'])
        w_mean = np.mean(editions['wright'])
        
        # Ratio of ratios (should be ~1.0 if same type)
        ratio_ratio = a_mean / w_mean if w_mean > 0 else 0
        pct_diff = abs(ratio_ratio - 1.0) * 100
        
        comparisons.append({
            'character': char,
            'aspley_ratio': a_mean,
            'wright_ratio': w_mean,
            'ratio_of_ratios': ratio_ratio,
            'pct_difference': pct_diff,
            'aspley_count': len(editions['aspley']),
            'wright_count': len(editions['wright'])
        })
    
    return comparisons

def test_hypothesis(comparisons):
    """Test H5: Aspect ratios within ±5%."""
    
    print("="*60)
    print("ASPECT RATIO ANALYSIS")
    print("="*60)
    
    if not comparisons:
        print("No comparison data available.")
        return
    
    # Overall statistics
    ratio_ratios = [c['ratio_of_ratios'] for c in comparisons]
    pct_diffs = [c['pct_difference'] for c in comparisons]
    
    print(f"\nAnalyzed {len(comparisons)} characters.")
    
    print(f"\nRATIO OF RATIOS (Aspley AR / Wright AR):")
    print(f"  Mean: {np.mean(ratio_ratios):.4f}")
    print(f"  Median: {np.median(ratio_ratios):.4f}")
    print(f"  Std. Dev.: {np.std(ratio_ratios):.4f}")
    print(f"  Range: {np.min(ratio_ratios):.3f} - {np.max(ratio_ratios):.3f}")
    
    # H5 Test: Within ±5%
    threshold = 5.0  # percent
    within_threshold = [c for c in comparisons if c['pct_difference'] <= threshold]
    pct_within = len(within_threshold) / len(comparisons) * 100
    
    print("\n" + "="*60)
    print("HYPOTHESIS TESTING")
    print("="*60)
    
    print(f"\nH5: Aspect ratios within ±{threshold}%")
    print(f"  Within threshold: {len(within_threshold)}/{len(comparisons)} ({pct_within:.1f}%)")
    
    # For H5 to PASS, we want most characters to have matching aspect ratios
    h5_pass = pct_within >= 80  # 80% of characters should match
    print(f"  Result: {'PASS ✓' if h5_pass else 'FAIL ✗'}")
    
    # Show outliers
    outliers = [c for c in comparisons if c['pct_difference'] > threshold]
    if outliers:
        outliers.sort(key=lambda x: x['pct_difference'], reverse=True)
        print(f"\n  Top 10 Outliers (>±{threshold}%):")
        for o in outliers[:10]:
            print(f"    '{o['character']}': Aspley={o['aspley_ratio']:.3f}, Wright={o['wright_ratio']:.3f}, Diff={o['pct_difference']:.1f}%")
    
    return {
        'h5_pass': h5_pass,
        'pct_within': pct_within,
        'mean_ratio_of_ratios': np.mean(ratio_ratios)
    }

def save_results(comparisons):
    """Save detailed data."""
    csv_path = REPORT_DIR / "aspect_ratios.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=comparisons[0].keys())
        writer.writeheader()
        writer.writerows(comparisons)
    
    print(f"\nSaved data to: {csv_path}")

def main():
    data = get_aspect_ratios()
    print(f"Retrieved {len(data)} character instances.")
    
    comparisons = analyze_by_character(data)
    
    result = test_hypothesis(comparisons)
    
    save_results(comparisons)
    
    print("\n" + "="*60)
    print("ASPECT RATIO ANALYSIS COMPLETE")
    print("="*60)
    
    return result

if __name__ == "__main__":
    main()
