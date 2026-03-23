#!/usr/bin/env python3
"""
Aligned Character Comparison (Task 4)

Reuses our alignment methodology to get matched character pairs,
then compares their width/height directly.

Tests H3: 95% of aligned chars within ±3px width
"""

import sqlite3
import csv
import numpy as np
import difflib
from pathlib import Path

# Configuration
DB_PATH = Path("data/codefinder.db")
REPORT_DIR = Path("reports/aligned_comparison")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

def get_page_instances(source, page_num, side):
    """Get instances for a specific page side (from forensic pipeline)."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT p.id, p.image_width 
        FROM pages p JOIN sources s ON p.source_id = s.id
        WHERE s.name = ? AND p.page_number = ?
    """, (source, page_num))
    page = cursor.fetchone()
    
    if not page:
        conn.close()
        return []
    
    midpoint = page['image_width'] / 2
    x_clause = f"x < {midpoint}" if side == 'left' else f"x >= {midpoint}"
    
    cursor.execute(f"""
        SELECT character, x, y, width, height 
        FROM character_instances 
        WHERE page_id = ? AND {x_clause}
        ORDER BY y, x
    """, (page['id'],))
    
    instances = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return instances

def align_and_compare_page(page_num, side):
    """Align page side and return matched pairs with size comparisons."""
    
    a_inst = get_page_instances('aspley', page_num, side)
    w_inst = get_page_instances('wright', page_num, side)
    
    if not a_inst or not w_inst or len(a_inst) < 50:
        return []
    
    # Text-based alignment
    a_text = "".join(i['character'] for i in a_inst)
    w_text = "".join(i['character'] for i in w_inst)
    
    matcher = difflib.SequenceMatcher(None, a_text, w_text, autojunk=False)
    
    # Calculate median offset
    y_diffs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                try:
                    y_diffs.append(a_inst[i1+k]['y'] - w_inst[j1+k]['y'])
                except:
                    pass
    
    if not y_diffs:
        return []
    
    median_offset = float(np.median(y_diffs))
    
    # Find pairs from 'equal' blocks
    pairs = []
    tolerance = 50
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                a = a_inst[i1+k]
                w = w_inst[j1+k]
                
                if abs(a['y'] - w['y'] - median_offset) > tolerance:
                    continue
                
                width_diff = a['width'] - w['width']
                height_diff = a['height'] - w['height']
                
                pairs.append({
                    'character': a['character'],
                    'aspley_width': a['width'],
                    'wright_width': w['width'],
                    'width_diff': width_diff,
                    'aspley_height': a['height'],
                    'wright_height': w['height'],
                    'height_diff': height_diff,
                    'page': page_num,
                    'side': side
                })
    
    return pairs

def main():
    print("="*60)
    print("ALIGNED CHARACTER COMPARISON")
    print("="*60)
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT p.page_number
        FROM pages p JOIN sources s ON p.source_id = s.id
        WHERE s.name = 'aspley'
        ORDER BY p.page_number
    """)
    pages = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    all_pairs = []
    
    for page_num in pages:
        for side in ['left', 'right']:
            pairs = align_and_compare_page(page_num, side)
            all_pairs.extend(pairs)
    
    print(f"\nTotal aligned pairs: {len(all_pairs)}")
    
    if not all_pairs:
        print("No pairs found!")
        return
    
    # Analyze width differences
    width_diffs = [p['width_diff'] for p in all_pairs]
    height_diffs = [p['height_diff'] for p in all_pairs]
    
    print(f"\nWIDTH DIFFERENCE (Aspley - Wright):")
    print(f"  Mean: {np.mean(width_diffs):.2f}px")
    print(f"  Std. Dev.: {np.std(width_diffs):.2f}px")
    print(f"  Range: {np.min(width_diffs):.0f} to {np.max(width_diffs):.0f}px")
    
    print(f"\nHEIGHT DIFFERENCE (Aspley - Wright):")
    print(f"  Mean: {np.mean(height_diffs):.2f}px")
    print(f"  Std. Dev.: {np.std(height_diffs):.2f}px")
    
    # H3 Test: 95% of aligned chars within ±3px width
    threshold = 3
    within_threshold = [p for p in all_pairs if abs(p['width_diff']) <= threshold]
    pct_within = len(within_threshold) / len(all_pairs) * 100
    
    print("\n" + "="*60)
    print("HYPOTHESIS TESTING")
    print("="*60)
    
    print(f"\nH3: 95% of aligned chars within ±{threshold}px width")
    print(f"  Within threshold: {len(within_threshold)}/{len(all_pairs)} ({pct_within:.1f}%)")
    print(f"  Result: {'PASS ✓' if pct_within >= 95 else 'FAIL ✗'}")
    
    # Save results
    csv_path = REPORT_DIR / "aligned_pairs.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_pairs[0].keys())
        writer.writeheader()
        writer.writerows(all_pairs)
    
    print(f"\nSaved {len(all_pairs)} pairs to: {csv_path}")
    
    print("\n" + "="*60)
    print("ALIGNED COMPARISON COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
