#!/usr/bin/env python3
"""
Character Geometry Census (Task 2)

Measures width/height of all characters and compares distributions
across Wright and Aspley editions.

Key Question: Is the 47% size difference seen in periods consistent
across ALL characters (indicating scan resolution) or variable
(indicating genuine typographical differences)?
"""

import sqlite3
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

# Configuration
DB_PATH = Path("data/codefinder.db")
REPORT_DIR = Path("reports/geometry_census")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

def get_character_stats():
    """Task 2.1-2.3: Query all chars, group by character and edition, calc means."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get aggregated stats per character per source
    cursor.execute("""
        SELECT 
            ci.character,
            s.name as source,
            COUNT(*) as count,
            AVG(ci.width) as mean_width,
            AVG(ci.height) as mean_height
        FROM character_instances ci
        JOIN pages p ON ci.page_id = p.id
        JOIN sources s ON p.source_id = s.id
        WHERE ci.confidence > 0.5
        GROUP BY ci.character, s.name
        HAVING COUNT(*) >= 10
        ORDER BY ci.character, s.name
    """)
    
    stats = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return stats

def analyze_size_ratios(stats):
    """Task 2.4-2.5: Compare sizes across editions, find outliers."""
    
    # Pivot: {character: {'aspley': {...}, 'wright': {...}}}
    by_char = defaultdict(dict)
    for s in stats:
        by_char[s['character']][s['source']] = s
    
    comparisons = []
    
    for char, editions in by_char.items():
        if 'aspley' not in editions or 'wright' not in editions:
            continue
        
        a = editions['aspley']
        w = editions['wright']
        
        # Calculate size ratio (Aspley / Wright)
        width_ratio = a['mean_width'] / w['mean_width'] if w['mean_width'] > 0 else 0
        height_ratio = a['mean_height'] / w['mean_height'] if w['mean_height'] > 0 else 0
        
        comparisons.append({
            'character': char,
            'aspley_width': a['mean_width'],
            'wright_width': w['mean_width'],
            'width_ratio': width_ratio,
            'aspley_height': a['mean_height'],
            'wright_height': w['mean_height'],
            'height_ratio': height_ratio,
            'aspley_count': a['count'],
            'wright_count': w['count']
        })
    
    return comparisons

def report_findings(comparisons):
    """Generate report with findings."""
    
    if not comparisons:
        print("No comparison data available.")
        return
    
    # Overall statistics
    width_ratios = [c['width_ratio'] for c in comparisons if c['width_ratio'] > 0]
    height_ratios = [c['height_ratio'] for c in comparisons if c['height_ratio'] > 0]
    
    print("="*60)
    print("CHARACTER GEOMETRY CENSUS")
    print("="*60)
    
    print(f"\nCompared {len(comparisons)} characters across editions.")
    
    print(f"\nWIDTH RATIO (Aspley / Wright):")
    print(f"  Mean: {np.mean(width_ratios):.3f}")
    print(f"  Median: {np.median(width_ratios):.3f}")
    print(f"  Std. Dev.: {np.std(width_ratios):.3f}")
    print(f"  Range: {np.min(width_ratios):.3f} - {np.max(width_ratios):.3f}")
    
    print(f"\nHEIGHT RATIO (Aspley / Wright):")
    print(f"  Mean: {np.mean(height_ratios):.3f}")
    print(f"  Median: {np.median(height_ratios):.3f}")
    print(f"  Std. Dev.: {np.std(height_ratios):.3f}")
    print(f"  Range: {np.min(height_ratios):.3f} - {np.max(height_ratios):.3f}")
    
    # Expected from periods: ratio ~1.47
    expected_ratio = 1.47
    tolerance = 0.15  # 10% tolerance
    
    # Find outliers (characters with ratio far from expected)
    outliers = [c for c in comparisons 
                if abs(c['width_ratio'] - expected_ratio) > tolerance
                or abs(c['height_ratio'] - expected_ratio) > tolerance]
    
    print(f"\nOUTLIER ANALYSIS (Expected ratio ≈ {expected_ratio:.2f} from period analysis):")
    print(f"  Characters within ±{tolerance*100:.0f}% of expected: {len(comparisons) - len(outliers)}/{len(comparisons)}")
    print(f"  Outliers: {len(outliers)}")
    
    if outliers:
        print("\n  Top 10 Outliers (by deviation):")
        outliers.sort(key=lambda x: abs(x['width_ratio'] - expected_ratio), reverse=True)
        for o in outliers[:10]:
            dev = abs(o['width_ratio'] - expected_ratio)
            print(f"    '{o['character']}': ratio={o['width_ratio']:.2f} (deviation={dev:.2f})")
    
    # Conclusion
    mean_ratio = np.mean(width_ratios)
    if abs(mean_ratio - expected_ratio) < 0.1:
        conclusion = "CONSISTENT: Global size ratio matches period analysis. Likely scan resolution difference."
    else:
        conclusion = "INCONSISTENT: Size ratios vary by character. May indicate genuine typographical differences."
    
    print(f"\nCONCLUSION: {conclusion}")
    
    return {
        'mean_width_ratio': np.mean(width_ratios),
        'mean_height_ratio': np.mean(height_ratios),
        'outlier_count': len(outliers),
        'total_compared': len(comparisons)
    }

def save_results(comparisons, summary):
    """Save detailed data to CSV."""
    csv_path = REPORT_DIR / "character_geometry.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'character', 'aspley_width', 'wright_width', 'width_ratio',
            'aspley_height', 'wright_height', 'height_ratio',
            'aspley_count', 'wright_count'
        ])
        writer.writeheader()
        writer.writerows(comparisons)
    
    print(f"\nSaved detailed data to: {csv_path}")

def main():
    # Task 2.1-2.3
    stats = get_character_stats()
    print(f"Retrieved stats for {len(stats)} character/edition pairs.")
    
    # Task 2.4-2.5
    comparisons = analyze_size_ratios(stats)
    
    # Task 2.6
    summary = report_findings(comparisons)
    
    # Save
    save_results(comparisons, summary)
    
    print("\n" + "="*60)
    print("GEOMETRY CENSUS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
