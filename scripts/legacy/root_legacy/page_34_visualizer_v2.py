#!/usr/bin/env python3
"""
Page 34 Deep Dive Visualizer (Text-Anchored)

Fixed version that uses text sequence matching to align
characters between editions, handling coordinate drift.
"""

import sqlite3
import numpy as np
import difflib
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Database path
DB_PATH = Path("data/codefinder.db")
OUTPUT_DIR = Path("reports/page_34_investigation_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

def get_page_instances(source_name: str, page_num: int):
    """Get all instances for a specific page sorted by reading order."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get page info
    cursor.execute("""
        SELECT p.id, p.image_path 
        FROM pages p JOIN sources s ON p.source_id = s.id
        WHERE s.name = ? AND p.page_number = ?
    """, (source_name, page_num))
    page = cursor.fetchone()
    
    if not page: return [], None
    
    # Get instances sorted by reading order (Top->Bottom, Left->Right)
    # Using a Y-binning strategy to group lines
    cursor.execute("""
        SELECT character, x, y, width, height 
        FROM character_instances 
        WHERE page_id = ?
        ORDER BY y, x
    """, (page['id'],))
    
    instances = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return instances, page['image_path']

def align_sequences(aspley_inst, wright_inst):
    """
    Align two sequences of character instances using SequenceMatcher.
    Includes SPATIAL VETO with ROBUST MEDIAN OFFSET.
    """
    # Create text strings for alignment
    a_text = "".join(i['character'] for i in aspley_inst)
    w_text = "".join(i['character'] for i in wright_inst)
    
    print(f"Alignment Text Lengths: Aspley={len(a_text)}, Wright={len(w_text)}")
    
    matcher = difflib.SequenceMatcher(None, a_text, w_text, autojunk=False)
    
    # 1. First Pass: Collect high-confidence anchors to determine spatial offset
    y_diffs = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                try:
                    a = aspley_inst[i1+k]
                    w = wright_inst[j1+k]
                    y_diffs.append(a['y'] - w['y'])
                except IndexError:
                    pass
                    
    if not y_diffs:
        print("No matches found to determine offset!")
        return []
        
    # Calculate robust median offset
    base_y_diff = np.median(y_diffs)
    print(f"Robust Median Y-Offset (Aspley - Wright): {base_y_diff:.1f}px")
    
    # 2. Second Pass: Collect matches with spatial filtering
    final_matches = []
    spatial_tolerance = 60  # Pixels
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Direct matches
            for k in range(i2 - i1):
                try:
                    a = aspley_inst[i1+k]
                    w = wright_inst[j1+k]
                    
                    # SPATIAL VETO
                    if abs(a['y'] - w['y'] - base_y_diff) > spatial_tolerance:
                        continue
                        
                    if a['character'] == 'ſ':
                        final_matches.append((a, w))
                except IndexError: pass
                    
        elif tag == 'replace':
            # Mismatches (e.g. ſ -> f)
            len_a = i2 - i1
            len_w = j2 - j1
            
            # Linear map
            for k in range(len_a):
                try:
                    a_idx = i1 + k
                    a = aspley_inst[a_idx]
                    
                    if a['character'] == 'ſ':
                        rel_pos = k / len_a
                        w_k = int(rel_pos * len_w)
                        if w_k < len_w:
                            w_idx = j1 + w_k
                            w = wright_inst[w_idx]
                            
                            # SPATIAL VETO
                            if abs(a['y'] - w['y'] - base_y_diff) > spatial_tolerance:
                                continue
                                
                            final_matches.append((a, w))
                except IndexError: pass

    return final_matches

def create_investigation_sheet(matches, aspley_path, wright_path):
    """Create the visual evidence sheet."""
    try:
        img_a = Image.open(aspley_path).convert('RGB')
        img_w = Image.open(wright_path).convert('RGB')
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    # Image setup
    cell_w, cell_h = 100, 100
    padding = 20
    header_h = 60
    
    # Sort matches by Y position to follow page flow
    matches.sort(key=lambda x: x[0]['y'])
    
    divergent_matches = [m for m in matches if m[0]['character'] != m[1]['character']]
    print(f"Found {len(matches)} aligned Long-s characters.")
    print(f"Showing {len(divergent_matches)} divergences.")
    
    display_matches = divergent_matches
    title = f"Page 34: Corrected Alignment ({len(display_matches)} divergences)"
    
    rows = len(display_matches)
    width = (cell_w * 3) + (padding * 4) + 200
    height = header_h + (rows * (cell_h + padding)) + padding
    
    sheet = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(sheet)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
        
    draw.text((padding, 15), title, fill='black', font=font)
    draw.text((padding, 40), "Aspley (Long-s)", fill='green', font=font)
    draw.text((padding + cell_w + padding, 40), "Wright (Matched)", fill='red', font=font)
    
    for i, (a, w) in enumerate(display_matches):
        y = header_h + i * (cell_h + padding)
        
        # Crop Aspley
        pad = 8  # More context
        crop_a = img_a.crop((
            a['x']-pad, a['y']-pad, 
            a['x']+a['width']+pad, a['y']+a['height']+pad
        ))
        crop_a.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        sheet.paste(crop_a, (padding, y))
        
        # Crop Wright
        crop_w = img_w.crop((
            w['x']-pad, w['y']-pad, 
            w['x']+w['width']+pad, w['y']+w['height']+pad
        ))
        crop_w.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        sheet.paste(crop_w, (padding + cell_w + padding, y))
        
        # Info
        info_x = padding + (cell_w + padding) * 2
        draw.text((info_x, y + 20), f"A: {a['character']}", fill='green', font=font)
        draw.text((info_x, y + 50), f"W: {w['character']}", fill='red', font=font)
        
        # Draw context line
        draw.line([(0, y + cell_h + padding/2), (width, y + cell_h + padding/2)], fill='#eee', width=1)

    output_path = OUTPUT_DIR / "page_34_aligned_evidence.png"
    sheet.save(output_path)
    print(f"Saved aligned evidence to {output_path}")

def main():
    print("Running Text-Anchored Visualizer...")
    a_inst, a_path = get_page_instances('aspley', 34)
    w_inst, w_path = get_page_instances('wright', 34)
    
    if not a_inst or not w_inst:
        print("Missing data.")
        return
        
    matches = align_sequences(a_inst, w_inst)
    create_investigation_sheet(matches, a_path, w_path)

if __name__ == "__main__":
    main()
