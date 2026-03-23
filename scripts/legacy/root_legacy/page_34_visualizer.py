#!/usr/bin/env python3
"""
Page 34 Deep Dive Visualizer

Specifically investigates the 50% drop in Long-s counts on Page 34
by spatially matching characters between editions to see what the 
missing Long-s characters look like in the Wright edition.
"""

import sqlite3
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from typing import List, Dict, Tuple

# Database path
DB_PATH = Path("data/codefinder.db")
OUTPUT_DIR = Path("reports/page_34_investigation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

def get_page_instances(source_name: str, page_num: int) -> Tuple[List[Dict], str, Tuple[int, int]]:
    """Get all instances for a specific page."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get page info
    cursor.execute("""
        SELECT p.id, p.image_path, p.image_width, p.image_height 
        FROM pages p JOIN sources s ON p.source_id = s.id
        WHERE s.name = ? AND p.page_number = ?
    """, (source_name, page_num))
    page = cursor.fetchone()
    
    if not page:
        return [], None, (0,0)
        
    # Get instances
    cursor.execute("""
        SELECT character, x, y, width, height, confidence 
        FROM character_instances 
        WHERE page_id = ?
        ORDER BY y, x
    """, (page['id'],))
    
    instances = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return instances, page['image_path'], (page['image_width'], page['image_height'])

def match_instances(aspley_list: List[Dict], wright_list: List[Dict], 
                   w_dims: Tuple[int, int], a_dims: Tuple[int, int]) -> List[Tuple[Dict, Dict]]:
    """
    Spatially match instances between editions.
    
    Strategy:
    1. Normalize coordinates (0.0 - 1.0)
    2. For each Aspley Long-s, find closest Wright character
    """
    matches = []
    
    # Normalize Wright instances for fast lookup
    w_norm = []
    for w in wright_list:
        nx = w['x'] / w_dims[0]
        ny = w['y'] / w_dims[1]
        w_norm.append((nx, ny, w))
    
    # Iterate Aspley Long-s instances
    for a in aspley_list:
        if a['character'] != 'ſ':
            continue
            
        nx = a['x'] / a_dims[0]
        ny = a['y'] / a_dims[1]
        
        # Find closest match
        best_dist = 0.02  # Max distance threshold (2% of page)
        best_match = None
        
        for wnx, wny, w in w_norm:
            dist = np.sqrt((nx - wnx)**2 + (ny - wny)**2)
            if dist < best_dist:
                best_dist = dist
                best_match = w
                
        if best_match:
            matches.append((a, best_match))
            
    return matches

def create_investigation_sheet(matches: List[Tuple[Dict, Dict]], 
                             aspley_path: str, wright_path: str):
    """Create the visual evidence sheet."""
    
    # Load images
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
    
    # Filter for interesting matches (where identities differ)
    # matching_identities = [m for m in matches if m[0]['character'] == m[1]['character']]
    divergent_identities = [m for m in matches if m[0]['character'] != m[1]['character']]
    
    print(f"Found {len(matches)} matches for Aspley Long-s")
    print(f"Indices confirming Long-s: {len(matches) - len(divergent_identities)}")
    print(f"Indices showing divergence: {len(divergent_identities)}")
    
    # Sort divergent by Y position
    divergent_identities.sort(key=lambda x: x[0]['y'])
    
    if not divergent_identities:
        print("No divergences found! OCR might be normalizing everything.")
        # Fallback to showing all matches to verify quality
        display_matches = matches[:20]
        title = "Page 34: Long-s Verification (All Matches)"
    else:
        display_matches = divergent_identities
        title = f"Page 34 Forensic Evidence: The {len(divergent_identities)} Missing Long-s Characters"
    
    # Create canvas
    rows = len(display_matches)
    width = (cell_w * 3) + (padding * 4) + 150 # Extra space for text
    height = header_h + (rows * (cell_h + padding)) + padding
    
    sheet = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(sheet)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()
        title_font = font
        
    draw.text((padding, 15), title, fill='black', font=title_font)
    
    # Headers
    y_head = header_h - 25
    draw.text((padding, y_head), "Aspley (Source)", fill='green', font=font)
    draw.text((padding + cell_w + padding, y_head), "Wright (Degraded?)", fill='red', font=font)
    draw.text((padding + (cell_w + padding)*2, y_head), "Detection", fill='#333', font=font)
    
    for i, (a, w) in enumerate(display_matches):
        y = header_h + i * (cell_h + padding)
        
        # Crop Aspley
        pad = 5
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
        
        # Aspley Label
        a_char = a['character']
        if a_char == 'ſ': a_char = 'Long-s'
        
        # Wright Label
        w_char = w['character']
        if w_char == 'ſ': w_char = 'Long-s'
        
        draw.text((info_x, y + 10), f"A: {a_char}", fill='green', font=font)
        draw.text((info_x, y + 40), f"W: {w_char}", fill='red', font=font)
        
        # Highlight confusion
        if w['character'] == 'f':
            note = "Detected as 'f'"
        elif w['character'] == 's':
            note = "Detected as 's'"
        else:
            note = f"Detected as '{w['character']}'"
            
        draw.text((info_x, y + 70), note, fill='blue', font=font)
        
        # Separator line
        draw.line([(0, y + cell_h + padding/2), (width, y + cell_h + padding/2)], fill='#eee', width=1)

    output_path = OUTPUT_DIR / "page_34_evidence.png"
    sheet.save(output_path)
    print(f"Evidence saved to: {output_path}")
    return output_path

def main():
    print("Investigating Page 34...")
    
    # 1. Get Instances
    print("Loading Aspley instances...")
    aspley_inst, a_path, a_dims = get_page_instances('aspley', 34)
    print(f"Loading Wright instances...")
    wright_inst, w_path, w_dims = get_page_instances('wright', 34)
    
    if not aspley_inst or not wright_inst:
        print("Error: Could not load data. check database.")
        return
        
    print(f"Aspley chars: {len(aspley_inst)} (Dims: {a_dims})")
    print(f"Wright chars: {len(wright_inst)} (Dims: {w_dims})")
    
    # 2. Match
    print("Matching instances spatially...")
    matches = match_instances(aspley_inst, wright_inst, w_dims, a_dims)
    
    # 3. Visualize
    create_investigation_sheet(matches, a_path, w_path)

if __name__ == "__main__":
    main()
