#!/usr/bin/env python3
"""
Page 34 Deep Dive Visualizer (V4: Split-Spread Alignment)

Handles double-page spreads by splitting instances into Left/Right pages
before comparison. This prevents "gutter jumping" where OCR reads across
the fold, confusing the sequence aligner.
"""

import sqlite3
import numpy as np
import difflib
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Database path
DB_PATH = Path("data/codefinder.db")
OUTPUT_DIR = Path("reports/page_34_investigation_v4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

def get_page_instances(source_name: str, page_num: int, side: str = 'left'):
    """
    Get instances for a specific page side (left/right).
    side: 'left' (x < width/2) or 'right' (x >= width/2)
    """
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get page info
    cursor.execute("""
        SELECT p.id, p.image_path, p.image_width 
        FROM pages p JOIN sources s ON p.source_id = s.id
        WHERE s.name = ? AND p.page_number = ?
    """, (source_name, page_num))
    page = cursor.fetchone()
    
    if not page: return [], None
    
    midpoint = page['image_width'] / 2
    
    # Filter query
    if side == 'left':
        x_clause = f"x < {midpoint}"
    else:
        x_clause = f"x >= {midpoint}"
    
    cursor.execute(f"""
        SELECT character, x, y, width, height, confidence 
        FROM character_instances 
        WHERE page_id = ? AND {x_clause}
        ORDER BY y, x
    """, (page['id'],))
    
    instances = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return instances, page['image_path']

def align_sequences(aspley_inst, wright_inst):
    """
    Align sequences with Median Offset validation.
    """
    a_text = "".join(i['character'] for i in aspley_inst)
    w_text = "".join(i['character'] for i in wright_inst)
    
    # print(f"  Snippet A: {a_text[:50]}")
    # print(f"  Snippet W: {w_text[:50]}")
    
    matcher = difflib.SequenceMatcher(None, a_text, w_text, autojunk=False)
    
    # 1. Calc Offset
    y_diffs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                try:
                    y_diffs.append(aspley_inst[i1+k]['y'] - wright_inst[j1+k]['y'])
                except: pass

    if not y_diffs:
        return []

    median_offset = np.median(y_diffs)
    print(f"  Median Y-Offset: {median_offset:.1f}px")
    
    final_matches = []
    tolerance = 50 # Tight tolerance since we split pages
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                a = aspley_inst[i1+k]
                w = wright_inst[j1+k]
                if abs(a['y'] - w['y'] - median_offset) > tolerance: continue
                
                if a['character'] == 'ſ':
                    final_matches.append((a, w))
                    
        elif tag == 'replace':
            len_a = i2 - i1
            len_w = j2 - j1
            for k in range(len_a):
                a = aspley_inst[i1+k]
                if a['character'] == 'ſ':
                    rel = k / len_a
                    w_k = int(rel * len_w)
                    if w_k < len_w:
                        w = wright_inst[j1+w_k]
                        if abs(a['y'] - w['y'] - median_offset) > tolerance: continue
                        final_matches.append((a, w))
                        
    return final_matches

def create_sheet(matches, side, a_path, w_path): 
    if not matches: return None
    
    try:
        img_a = Image.open(a_path).convert('RGB')
        img_w = Image.open(w_path).convert('RGB')
    except: return None
    
    matches.sort(key=lambda x: x[0]['y'])
    
    # Filter context-mismatches (where chars don't match)
    divergences = [m for m in matches if m[0]['character'] != m[1]['character']]
    
    cell_w, cell_h = 100, 100
    padding = 20
    rows = len(divergences)
    if rows == 0: return None
    
    width = (cell_w * 3) + 150
    height = 60 + (rows * 120)
    
    sheet = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(sheet)
    
    # Render...
    y = 60
    for a, w in divergences:
        # Aspley
        c_a = img_a.crop((a['x'], a['y'], a['x']+a['width'], a['y']+a['height']))
        c_a.thumbnail((80, 80))
        sheet.paste(c_a, (20, y))
        
        # Wright
        c_w = img_w.crop((w['x'], w['y'], w['x']+w['width'], w['y']+w['height']))
        c_w.thumbnail((80, 80))
        sheet.paste(c_w, (140, y))
        
        draw.text((260, y+20), f"A: {a['character']}", fill='green')
        draw.text((260, y+40), f"W: {w['character']}", fill='red')
        y += 120
        
    out = OUTPUT_DIR / f"page_34_{side}_evidence.png"
    sheet.save(out)
    print(f"Saved {side} evidence: {out}")
    return out

def main():
    print("Running V4 Split-Page Analysis...")
    
    for side in ['left', 'right']:
        print(f"\nProcessing {side.upper()} page...")
        a_inst, a_path = get_page_instances('aspley', 34, side)
        w_inst, w_path = get_page_instances('wright', 34, side)
        
        print(f"  Counts: Aspley={len(a_inst)}, Wright={len(w_inst)}")
        if len(a_inst) < 100: continue
        
        matches = align_sequences(a_inst, w_inst)
        create_sheet(matches, side, a_path, w_path)

if __name__ == "__main__":
    main()
