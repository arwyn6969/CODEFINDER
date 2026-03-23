#!/usr/bin/env python3
"""
Period Analysis (Task 1): Control Experiment for Forensic Methodology

Extracts all period (.) instances, measures their dimensions, and compares
distributions across Wright and Aspley editions to test:
- H1: Internal consistency (Std. Dev. < 2px per edition)
- H2: Cross-edition identity (Mean difference < 1px)
"""

import sqlite3
import csv
import numpy as np
from pathlib import Path

# Configuration
DB_PATH = Path("data/codefinder.db")
REPORT_DIR = Path("reports/period_analysis")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

def extract_periods():
    """Task 1.1: Extract all period instances from database."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            ci.id,
            ci.character,
            ci.width,
            ci.height,
            ci.confidence,
            p.page_number,
            s.name as source
        FROM character_instances ci
        JOIN pages p ON ci.page_id = p.id
        JOIN sources s ON p.source_id = s.id
        WHERE ci.character = '.'
        ORDER BY s.name, p.page_number
    """)
    
    periods = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    print(f"Extracted {len(periods)} period instances.")
    return periods

def calculate_metrics(periods):
    """Task 1.2: Calculate diameter (max of width/height) for each."""
    for p in periods:
        # Diameter = max dimension (periods should be roughly circular)
        p['diameter'] = max(p['width'], p['height'])
        # Aspect ratio (1.0 = perfect circle)
        p['aspect_ratio'] = p['width'] / p['height'] if p['height'] > 0 else 0
    return periods

def analyze_distribution(periods):
    """Task 1.3-1.4: Generate statistics per edition."""
    
    # Split by source
    aspley = [p for p in periods if p['source'] == 'aspley']
    wright = [p for p in periods if p['source'] == 'wright']
    
    # Apply size filter to remove OCR noise
    # True periods are typically 5-30px in diameter at our scan resolution
    MIN_DIAMETER = 5
    MAX_DIAMETER = 30
    
    aspley_clean = [p for p in aspley if MIN_DIAMETER <= p['diameter'] <= MAX_DIAMETER]
    wright_clean = [p for p in wright if MIN_DIAMETER <= p['diameter'] <= MAX_DIAMETER]
    
    print(f"\nApplied size filter ({MIN_DIAMETER}-{MAX_DIAMETER}px):")
    print(f"  Aspley: {len(aspley)} → {len(aspley_clean)} ({len(aspley)-len(aspley_clean)} filtered)")
    print(f"  Wright: {len(wright)} → {len(wright_clean)} ({len(wright)-len(wright_clean)} filtered)")
    
    results = {}
    
    for name, data in [('aspley', aspley_clean), ('wright', wright_clean)]:
        diameters = [p['diameter'] for p in data]
        
        if not diameters:
            print(f"WARNING: No periods found for {name}")
            continue
            
        results[name] = {
            'count': len(diameters),
            'mean': np.mean(diameters),
            'std_dev': np.std(diameters),
            'min': np.min(diameters),
            'max': np.max(diameters),
            'median': np.median(diameters)
        }
        
        print(f"\n{name.upper()} Periods (Filtered):")
        print(f"  Count: {results[name]['count']}")
        print(f"  Mean Diameter: {results[name]['mean']:.2f}px")
        print(f"  Std. Dev.: {results[name]['std_dev']:.2f}px")
        print(f"  Range: {results[name]['min']:.0f} - {results[name]['max']:.0f}px")
    
    return results, aspley_clean, wright_clean

def test_hypotheses(results):
    """Task 1.6: Report Pass/Fail on H1 and H2."""
    
    print("\n" + "="*60)
    print("HYPOTHESIS TESTING")
    print("="*60)
    
    # H1: Internal consistency (Std. Dev. < 2px per edition)
    h1_threshold = 2.0
    h1_aspley = results.get('aspley', {}).get('std_dev', 999) < h1_threshold
    h1_wright = results.get('wright', {}).get('std_dev', 999) < h1_threshold
    
    print(f"\nH1: Period sizes consistent WITHIN each edition (Std. Dev. < {h1_threshold}px)")
    print(f"  Aspley: {'PASS ✓' if h1_aspley else 'FAIL ✗'} (σ = {results.get('aspley', {}).get('std_dev', 0):.2f}px)")
    print(f"  Wright: {'PASS ✓' if h1_wright else 'FAIL ✗'} (σ = {results.get('wright', {}).get('std_dev', 0):.2f}px)")
    
    # H2: Cross-edition identity (Mean difference < 1px)
    h2_threshold = 1.0
    mean_diff = abs(results.get('aspley', {}).get('mean', 0) - results.get('wright', {}).get('mean', 0))
    h2_pass = mean_diff < h2_threshold
    
    print(f"\nH2: Period sizes identical ACROSS editions (Mean diff. < {h2_threshold}px)")
    print(f"  Mean Difference: {mean_diff:.2f}px")
    print(f"  Result: {'PASS ✓' if h2_pass else 'FAIL ✗'}")
    
    return {
        'H1_aspley': h1_aspley,
        'H1_wright': h1_wright,
        'H2': h2_pass,
        'mean_difference': mean_diff
    }

def save_results(periods, results, hypothesis_results):
    """Save all data to CSV for review."""
    
    # Raw data
    csv_path = REPORT_DIR / "period_instances.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'source', 'page_number', 'width', 'height', 'diameter', 'aspect_ratio', 'confidence'])
        writer.writeheader()
        for p in periods:
            writer.writerow({k: p.get(k) for k in writer.fieldnames})
    print(f"\nSaved raw data to: {csv_path}")
    
    # Summary
    summary_path = REPORT_DIR / "period_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("PERIOD ANALYSIS SUMMARY\n")
        f.write("="*40 + "\n\n")
        
        for name, stats in results.items():
            f.write(f"{name.upper()}:\n")
            for k, v in stats.items():
                f.write(f"  {k}: {v:.2f}\n" if isinstance(v, float) else f"  {k}: {v}\n")
            f.write("\n")
        
        f.write("HYPOTHESIS RESULTS:\n")
        for k, v in hypothesis_results.items():
            f.write(f"  {k}: {v}\n")
    
    print(f"Saved summary to: {summary_path}")

def main():
    print("="*60)
    print("PERIOD ANALYSIS (Control Experiment)")
    print("="*60)
    
    # Task 1.1
    periods = extract_periods()
    
    # Task 1.2
    periods = calculate_metrics(periods)
    
    # Task 1.3-1.4
    results, aspley, wright = analyze_distribution(periods)
    
    # Task 1.6
    hypothesis_results = test_hypotheses(results)
    
    # Save
    save_results(periods, results, hypothesis_results)
    
    print("\n" + "="*60)
    print("PERIOD ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
