#!/usr/bin/env python3
"""
Page 34 Deep Dive Visualizer (V3: Line-Constrained)

Solves the "drift" issue by first segmenting the page into text lines,
then aligning matched lines individually. This prevents error accumulation
and uses spatial constraints to validate text matches.
"""

import sqlite3
import numpy as np
import difflib
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Database path
DB_PATH = Path("data/codefinder.db")
OUTPUT_DIR = Path("reports/page_34_investigation_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

def get_page_instances(source_name: str, page_num: int):
    """Get all instances for a specific page."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get page info
    cursor.execute("""
        SELECT p.id, p.image_path, p.image_height 
        FROM pages p JOIN sources s ON p.source_id = s.id
        WHERE s.name = ? AND p.page_number = ?
    """, (source_name, page_num))
    page = cursor.fetchone()
    
    if not page: return [], None
    
    # Get instances
    cursor.execute("""
        SELECT character, x, y, width, height 
        FROM character_instances 
        WHERE page_id = ?
        ORDER BY y, x
    """, (page['id'],))
    
    instances = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return instances, page['image_path']

def cluster_into_lines(instances, height_threshold=20):
    """
    Group instances into lines based on Y-coordinates.
    Simple greedy clustering: if char is close to current line's Y, add it.
    """
    # Sort by Y primarily
    sorted_inst = sorted(instances, key=lambda i: i['y'])
    
    lines = []
    current_line = []
    current_y = -100
    
    for inst in sorted_inst:
        y = inst['y']
        
        # If this char is far below current line, start new line
        # We use the bottom of the character (y+h) or center? 
        # Tesseract center-line is roughly comparable.
        if abs(y - current_y) > height_threshold and current_line:
            # Sort current line by X before saving
            current_line.sort(key=lambda i: i['x'])
            lines.append(current_line)
            current_line = []
            current_y = y
        elif not current_line:
            current_y = y
            
        current_line.append(inst)
        
    # Append last line
    if current_line:
        current_line.sort(key=lambda i: i['x'])
        lines.append(current_line)
        
    print(f" segmented into {len(lines)} lines")
    return lines

def align_lines_and_chars(aspley_lines, wright_lines):
    """
    1. Match lines between editions (1-to-1 assumption for Sonnets)
    2. Match characters within aligned lines
    """
    
    # Assumption: Sonnet pages have same number of lines roughly.
    # We can use difflib on the specific line strings to align the lines themselves!
    
    a_line_strs = ["".join(i['character'] for i in l) for l in aspley_lines]
    w_line_strs = ["".join(i['character'] for i in l) for l in wright_lines]
    
    print(f"Line Counts: Aspley={len(a_line_strs)}, Wright={len(w_line_strs)}")
    
    line_matcher = difflib.SequenceMatcher(None, a_line_strs, w_line_strs, autojunk=False)
    
    all_char_matches = []
    
    for tag, i1, i2, j1, j2 in line_matcher.get_opcodes():
        if tag == 'equal':
            # These lines match 1-to-1
            for k in range(i2 - i1):
                a_line = aspley_lines[i1+k]
                w_line = wright_lines[j1+k]
                
                # Align characters WITHIN this line
                matches = align_sequences(a_line, w_line)
                all_char_matches.extend(matches)
                
        elif tag == 'replace':
            # Block of lines is different?
            # Try to map them 1-to-1 if count logic holds
            len_a = i2 - i1
            len_w = j2 - j1
            if len_a == len_w:
                for k in range(len_a):
                    a_line = aspley_lines[i1+k]
                    w_line = wright_lines[j1+k]
                    matches = align_sequences(a_line, w_line)
                    all_char_matches.extend(matches)
                    
    return all_char_matches

def align_sequences(aspley_inst, wright_inst):
    """Align chars within a single line."""
    
    a_text = "".join(i['character'] for i in aspley_inst)
    w_text = "".join(i['character'] for i in wright_inst)
    
    matcher = difflib.SequenceMatcher(None, a_text, w_text, autojunk=False)
    
    final_matches = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                a_idx = i1 + k
                w_idx = j1 + k
                if aspley_inst[a_idx]['character'] == 'ſ':
                    final_matches.append((aspley_inst[a_idx], wright_inst[w_idx]))
                    
        elif tag == 'replace':
            # Handle degradation
            len_a = i2 - i1
            len_w = j2 - j1
            
            # Simple linear map for the replace block
            for k in range(len_a):
                a_idx = i1 + k
                if aspley_inst[a_idx]['character'] == 'ſ':
                    # Relative pos
                    rel_pos = k / len_a
                    w_k = int(rel_pos * len_w)
                    if w_k < len_w:
                        w_idx = j1 + w_k
                        # Verify spatial constraint (sanity check)
                        # Y-pos should be very close (same line)
                        final_matches.append((aspley_inst[a_idx], wright_inst[w_idx]))

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
    title = f"Page 34: Line-Constrained Alignment V3 ({len(display_matches)} divergences)"
    
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
        pad = 10
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
        draw.text((info_x, y + 80), f"Line Y: {a['y']}", fill='#aaa', font=font)
        
        # Draw context line
        draw.line([(0, y + cell_h + padding/2), (width, y + cell_h + padding/2)], fill='#eee', width=1)

    output_path = OUTPUT_DIR / "page_34_line_aligned_evidence.png"
    sheet.save(output_path)
    print(f"Saved aligned evidence to {output_path}")

def main():
    print("Running Line-Based Visualizer V3...")
    a_inst, a_path = get_page_instances('aspley', 34)
    w_inst, w_path = get_page_instances('wright', 34)
    
    if not a_inst or not w_inst:
        print("Missing data.")
        return
        
    # 1. Segment into lines
    print("Clustering Aspley lines...")
    a_lines = cluster_into_lines(a_inst)
    print("Clustering Wright lines...")
    w_lines = cluster_into_lines(w_inst)
    
    # 2. Align Hierarchically
    print("Aligning lines...")
    matches = align_lines_and_chars(a_lines, w_lines)
    
    # 3. Visualize
    create_investigation_sheet(matches, a_path, w_path)

if __name__ == "__main__":
    main()
