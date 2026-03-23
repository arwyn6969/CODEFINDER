#!/usr/bin/env python3
"""
Contour-Based Shape Analysis (Phase 8)

Extracts actual ink contours from character images and compares
their shapes using Hu moments — a scale/rotation invariant descriptor.

If editions use the same type, their Hu moments should be nearly identical.
"""

import sqlite3
import numpy as np
import cv2
import difflib
from pathlib import Path
from PIL import Image

# Configuration
DB_PATH = Path("data/codefinder.db")
REPORT_DIR = Path("reports/contour_analysis")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_PAGES = [16, 19, 23, 28, 29, 31, 33, 34, 35, 36]

def get_connection():
    return sqlite3.connect(DB_PATH)

def get_page_instances(source, page_num, side):
    """Get character instances for a page side."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT p.id, p.image_path, p.image_width 
        FROM pages p JOIN sources s ON p.source_id = s.id
        WHERE s.name = ? AND p.page_number = ?
    """, (source, page_num))
    page = cursor.fetchone()
    
    if not page:
        conn.close()
        return [], None
    
    midpoint = page['image_width'] / 2
    x_clause = f"x < {midpoint}" if side == 'left' else f"x >= {midpoint}"
    
    cursor.execute(f"""
        SELECT character, x, y, width, height 
        FROM character_instances 
        WHERE page_id = ? AND {x_clause}
          AND width > 8 AND height > 8
        ORDER BY y, x
    """, (page['id'],))
    
    instances = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return instances, page['image_path']

def extract_contour_moments(img_crop):
    """Extract the primary ink contour and compute Hu moments."""
    # Convert to grayscale numpy array
    gray = np.array(img_crop.convert('L'))
    
    # Binarize (ink = white, paper = black for contour detection)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # Get largest contour (main ink shape)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Calculate Hu moments (7 values, scale/rotation invariant)
    moments = cv2.moments(main_contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log-transform for better comparison (Hu moments span many orders of magnitude)
    hu_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    return main_contour, hu_log

def compare_hu_moments(hu1, hu2):
    """Compare two Hu moment vectors. Lower = more similar."""
    if hu1 is None or hu2 is None:
        return float('inf')
    
    # Euclidean distance in log-space
    return np.sqrt(np.sum((hu1 - hu2) ** 2))

def compare_page_contours(page_num, side, image_cache):
    """Compare contour shapes for aligned character pairs."""
    
    a_inst, a_path = get_page_instances('aspley', page_num, side)
    w_inst, w_path = get_page_instances('wright', page_num, side)
    
    if not a_inst or not w_inst or len(a_inst) < 50:
        return []
    
    # Load images
    if a_path not in image_cache:
        try:
            image_cache[a_path] = Image.open(a_path)
        except:
            return []
    if w_path not in image_cache:
        try:
            image_cache[w_path] = Image.open(w_path)
        except:
            return []
    
    a_img = image_cache[a_path]
    w_img = image_cache[w_path]
    
    # Text alignment
    a_text = "".join(i['character'] for i in a_inst)
    w_text = "".join(i['character'] for i in w_inst)
    
    matcher = difflib.SequenceMatcher(None, a_text, w_text, autojunk=False)
    
    # Get median Y offset
    y_diffs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for k in range(min(i2 - i1, 100)):
                try:
                    y_diffs.append(a_inst[i1+k]['y'] - w_inst[j1+k]['y'])
                except:
                    pass
    
    if not y_diffs:
        return []
    
    median_offset = float(np.median(y_diffs))
    
    # Compare contours for matched pairs
    results = []
    tolerance = 50
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                a = a_inst[i1+k]
                w = w_inst[j1+k]
                
                if abs(a['y'] - w['y'] - median_offset) > tolerance:
                    continue
                
                try:
                    # Crop characters
                    a_crop = a_img.crop((a['x'], a['y'], a['x']+a['width'], a['y']+a['height']))
                    w_crop = w_img.crop((w['x'], w['y'], w['x']+w['width'], w['y']+w['height']))
                    
                    # Extract contours and Hu moments
                    a_contour, a_hu = extract_contour_moments(a_crop)
                    w_contour, w_hu = extract_contour_moments(w_crop)
                    
                    if a_hu is None or w_hu is None:
                        continue
                    
                    # Compare
                    distance = compare_hu_moments(a_hu, w_hu)
                    
                    results.append({
                        'character': a['character'],
                        'hu_distance': distance,
                        'page': page_num,
                        'side': side
                    })
                except:
                    pass
    
    return results

def main():
    print("="*60)
    print("CONTOUR-BASED SHAPE ANALYSIS")
    print("="*60)
    print("\nUsing Hu Moments (scale/rotation invariant shape descriptors)")
    
    all_results = []
    image_cache = {}
    
    for page_num in SAMPLE_PAGES:
        print(f"Processing Page {page_num}...")
        for side in ['left', 'right']:
            results = compare_page_contours(page_num, side, image_cache)
            all_results.extend(results)
    
    print(f"\nTotal pairs analyzed: {len(all_results)}")
    
    if not all_results:
        print("No pairs found!")
        return
    
    # Analyze Hu moment distances
    distances = [r['hu_distance'] for r in all_results]
    
    print(f"\nHU MOMENT DISTANCE (lower = more similar):")
    print(f"  Mean: {np.mean(distances):.3f}")
    print(f"  Median: {np.median(distances):.3f}")
    print(f"  Std. Dev.: {np.std(distances):.3f}")
    print(f"  Range: {np.min(distances):.3f} - {np.max(distances):.3f}")
    
    # Threshold analysis
    # For identical type, we expect very low distances (< 1.0)
    thresholds = [0.5, 1.0, 1.5, 2.0]
    print(f"\nSIMILARITY THRESHOLDS:")
    for t in thresholds:
        count = sum(1 for d in distances if d < t)
        pct = count / len(distances) * 100
        print(f"  < {t}: {count}/{len(distances)} ({pct:.1f}%)")
    
    # By character breakdown
    from collections import defaultdict
    by_char = defaultdict(list)
    for r in all_results:
        by_char[r['character']].append(r['hu_distance'])
    
    print(f"\nBEST MATCHING CHARACTERS (lowest avg Hu distance):")
    sorted_chars = sorted(by_char.items(), key=lambda x: np.mean(x[1]))
    for char, dists in sorted_chars[:10]:
        if len(dists) >= 3:
            print(f"  '{char}': avg={np.mean(dists):.3f} (n={len(dists)})")
    
    print(f"\nWORST MATCHING CHARACTERS (highest avg Hu distance):")
    for char, dists in sorted_chars[-10:]:
        if len(dists) >= 3:
            print(f"  '{char}': avg={np.mean(dists):.3f} (n={len(dists)})")
    
    # Conclusion
    median_dist = np.median(distances)
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if median_dist < 1.0:
        print(f"✓ Ink shapes are HIGHLY SIMILAR (median distance = {median_dist:.3f})")
        print("  This supports typographical identity.")
    elif median_dist < 2.0:
        print(f"○ Ink shapes are MODERATELY similar (median distance = {median_dist:.3f})")
        print("  Minor variations may be due to ink spread or scan quality.")
    else:
        print(f"✗ Ink shapes show SIGNIFICANT differences (median distance = {median_dist:.3f})")
        print("  This may indicate different type or significant wear.")

if __name__ == "__main__":
    main()
