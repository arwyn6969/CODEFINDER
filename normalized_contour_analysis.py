#!/usr/bin/env python3
"""
Normalized Contour Analysis

Resizes character images to same size BEFORE extracting contours,
to eliminate resolution effects on contour quality.
"""

import sqlite3
import numpy as np
import cv2
import difflib
from pathlib import Path
from PIL import Image

DB_PATH = Path("data/codefinder.db")
NORMALIZE_SIZE = (64, 64)  # Resize all to this before contour extraction
SAMPLE_PAGES = [16, 19, 23, 28, 29, 31, 33, 34, 35, 36]

def get_connection():
    return sqlite3.connect(DB_PATH)

def get_page_instances(source, page_num, side):
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

def extract_normalized_contour(img_crop):
    """Normalize size, then extract contour and moments."""
    # Resize to standard size first
    resized = img_crop.resize(NORMALIZE_SIZE, Image.Resampling.LANCZOS)
    gray = np.array(resized.convert('L'))
    
    # Binarize
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    main_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(main_contour)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    return hu_log

def compare_page(page_num, side, image_cache):
    a_inst, a_path = get_page_instances('aspley', page_num, side)
    w_inst, w_path = get_page_instances('wright', page_num, side)
    
    if not a_inst or not w_inst or len(a_inst) < 50:
        return []
    
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
    
    a_img, w_img = image_cache[a_path], image_cache[w_path]
    
    a_text = "".join(i['character'] for i in a_inst)
    w_text = "".join(i['character'] for i in w_inst)
    
    matcher = difflib.SequenceMatcher(None, a_text, w_text, autojunk=False)
    
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
    results = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                a, w = a_inst[i1+k], w_inst[j1+k]
                
                if abs(a['y'] - w['y'] - median_offset) > 50:
                    continue
                
                try:
                    a_crop = a_img.crop((a['x'], a['y'], a['x']+a['width'], a['y']+a['height']))
                    w_crop = w_img.crop((w['x'], w['y'], w['x']+w['width'], w['y']+w['height']))
                    
                    a_hu = extract_normalized_contour(a_crop)
                    w_hu = extract_normalized_contour(w_crop)
                    
                    if a_hu is None or w_hu is None:
                        continue
                    
                    dist = np.sqrt(np.sum((a_hu - w_hu) ** 2))
                    results.append({'character': a['character'], 'distance': dist})
                except:
                    pass
    
    return results

def main():
    print("="*60)
    print("NORMALIZED CONTOUR ANALYSIS")
    print("="*60)
    print(f"All images resized to {NORMALIZE_SIZE} before contour extraction")
    
    all_results = []
    image_cache = {}
    
    for page_num in SAMPLE_PAGES:
        print(f"Processing Page {page_num}...")
        for side in ['left', 'right']:
            results = compare_page(page_num, side, image_cache)
            all_results.extend(results)
    
    print(f"\nTotal pairs: {len(all_results)}")
    
    if not all_results:
        return
    
    distances = [r['distance'] for r in all_results]
    
    print(f"\nNORMALIZED HU DISTANCE:")
    print(f"  Mean: {np.mean(distances):.3f}")
    print(f"  Median: {np.median(distances):.3f}")
    print(f"  Std. Dev.: {np.std(distances):.3f}")
    
    for t in [0.5, 1.0, 2.0, 3.0]:
        count = sum(1 for d in distances if d < t)
        print(f"  < {t}: {count}/{len(distances)} ({count/len(distances)*100:.1f}%)")
    
    print("\n" + "="*60)
    median = np.median(distances)
    if median < 2.0:
        print(f"✓ SIMILAR: Normalized contours match well (median={median:.2f})")
    else:
        print(f"○ DIFFERENT: Contour shapes differ (median={median:.2f})")

if __name__ == "__main__":
    main()
