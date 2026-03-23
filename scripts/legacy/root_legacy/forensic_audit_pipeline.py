#!/usr/bin/env python3
"""
Forensic Audit Pipeline (The "Book of Evidence")

Automates the V4 Split-Page alignment methodology across the entire
Sonnet corpus (Pages 9-67) to generate a comprehensive registry
of type degradation.
"""

import sqlite3
import numpy as np
import difflib
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import time

# Configuration
DB_PATH = Path("data/codefinder.db")
REPORT_DIR = Path("reports/full_audit")
IMAGES_DIR = REPORT_DIR / "images"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

class ForensicAligner:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def get_valid_pages(self):
        """Get distinct page numbers that exist in both sources."""
        cursor = self.conn.cursor()
        # Find pages present in both editions
        cursor.execute("""
            SELECT p.page_number
            FROM pages p JOIN sources s ON p.source_id = s.id
            WHERE s.name IN ('aspley', 'wright')
            GROUP BY p.page_number
            HAVING COUNT(DISTINCT s.name) = 2
            ORDER BY p.page_number
        """)
        return [row['page_number'] for row in cursor.fetchall()]

    def get_page_instances(self, source, page_num, side):
        """Get instances for a specific page side."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT p.id, p.image_path, p.image_width 
            FROM pages p JOIN sources s ON p.source_id = s.id
            WHERE s.name = ? AND p.page_number = ?
        """, (source, page_num))
        page = cursor.fetchone()
        
        if not page: return [], None
        
        midpoint = page['image_width'] / 2
        
        x_clause = f"x < {midpoint}" if side == 'left' else f"x >= {midpoint}"
        
        cursor.execute(f"""
            SELECT character, x, y, width, height 
            FROM character_instances 
            WHERE page_id = ? AND {x_clause}
            ORDER BY y, x
        """, (page['id'],))
        
        return [dict(row) for row in cursor.fetchall()], page['image_path']

    def align_page_side(self, page_num, side):
        """Perform V4 alignment for one side of a page."""
        a_inst, a_path = self.get_page_instances('aspley', page_num, side)
        w_inst, w_path = self.get_page_instances('wright', page_num, side)
        
        if not a_inst or not w_inst or len(a_inst) < 50:
            return [], 0.0, False, None, None

        # 1. Calculate Robust Median Offset
        a_text = "".join(i['character'] for i in a_inst)
        w_text = "".join(i['character'] for i in w_inst)
        
        matcher = difflib.SequenceMatcher(None, a_text, w_text, autojunk=False)
        
        y_diffs = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for k in range(i2 - i1):
                    try:
                        diff = a_inst[i1+k]['y'] - w_inst[j1+k]['y']
                        y_diffs.append(diff)
                    except: pass
        
        if not y_diffs:
            return [], 0.0, False, None, None
            
        # Robust Statistics: Filter Outliers using IQR (Interquartile Range)
        q1 = np.percentile(y_diffs, 25)
        q3 = np.percentile(y_diffs, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Keep only consistent offsets
        clean_diffs = [d for d in y_diffs if lower_bound <= d <= upper_bound]
        
        if not clean_diffs:
            print("    Skipping: No clean offsets after filtering")
            return [], 0.0, False, None, None

        median_offset = float(np.median(clean_diffs))
        
        # QUALITY GATE: Peak Density Check
        # Variance is useless because noise is high.
        # Check if we have a "solid core" of matches at the median.
        core_matches = [d for d in y_diffs if abs(d - median_offset) < 5.0]
        density_count = len(core_matches)
        
        print(f"    Offset: {median_offset:.1f}px, Core Matches: {density_count}")
        
        is_aligned = density_count > 20  # Require at least 20 locked-in characters
        
        if not is_aligned:
             print(f"    Skipping: Weak alignment signal (<20 core matches)")
             return [], median_offset, False, a_path, w_path
        
        # 2. Collect Valid Matches (with Spatial Veto)
        matches = []
        tolerance = 50
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for k in range(i2 - i1):
                    a = a_inst[i1+k]
                    w = w_inst[j1+k]
                    
                    if abs(a['y'] - w['y'] - median_offset) > tolerance: continue
                    
                    if a['character'] == 'ſ':
                        matches.append((a, w, 'matched'))
                        
            elif tag == 'replace':
                len_a = i2 - i1
                len_w = j2 - j1
                
                for k in range(len_a):
                    a = a_inst[i1+k]
                    if a['character'] == 'ſ':
                        # Mapping logic
                        rel = k / len_a
                        w_k = int(rel * len_w)
                        if w_k < len_w:
                            w = w_inst[j1+w_k]
                            
                            # Spatial Veto
                            if abs(a['y'] - w['y'] - median_offset) > tolerance: continue
                            
                            matches.append((a, w, 'degraded'))

        return matches, median_offset, is_aligned, a_path, w_path

def save_evidence_image(a_inst, w_inst, a_path, w_path, match_id):
    """Save cropped comparison image for a single match."""
    try:
        # Load images lazily or cache? For now, load per page is better but here we are deep in loop.
        # Ideally we pass open PIL images.
        pass # implemented in main loop to save I/O
    except: pass

def main():
    print("="*60)
    print("Running FORENSIC AUDIT PIPELINE (Full Corpus)")
    print("="*60)
    
    aligner = ForensicAligner(DB_PATH)
    pages = aligner.get_valid_pages()
    print(f"Found {len(pages)} valid pages for analysis.")
    
    # CSV Writer
    csv_file = REPORT_DIR / "forensic_registry.csv"
    
    total_long_s = 0
    total_degraded = 0
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['match_id', 'page', 'side', 'valid_alignment', 'offset_y', 
                         'aspley_char', 'wright_char', 'wright_identity', 'is_degraded', 'image_path'])
        
        for page_num in pages:
            print(f"Processing Page {page_num}...")
            
            for side in ['left', 'right']:
                try:
                    matches, offset, is_aligned, a_path, w_path = aligner.align_page_side(page_num, side)
                    
                    if not matches: continue
                    
                    # Load images once per side if needed
                    img_a = Image.open(a_path).convert('RGB')
                    img_w = Image.open(w_path).convert('RGB')
                    
                    for i, (a, w, status) in enumerate(matches):
                        match_id = f"{page_num}_{side}_{i}"
                        is_degraded = (status == 'degraded')
                        
                        # Save evidence crop
                        filename = f"{match_id}.jpg"
                        save_path = IMAGES_DIR / filename
                        
                        # Generate crop
                        # Composite image: Aspley Left | Wright Right
                        combo = Image.new('RGB', (120, 60))
                        
                        ca = img_a.crop((a['x'], a['y'], a['x']+a['width'], a['y']+a['height']))
                        ca.thumbnail((60, 60))
                        combo.paste(ca, (0, 0))
                        
                        cw = img_w.crop((w['x'], w['y'], w['x']+w['width'], w['y']+w['height']))
                        cw.thumbnail((60, 60))
                        combo.paste(cw, (60, 0))
                        
                        combo.save(save_path)
                        
                        # Log
                        writer.writerow([
                            match_id, page_num, side, is_aligned, f"{offset:.1f}",
                            'ſ', w['character'], w['character'], is_degraded, f"images/{filename}"
                        ])
                        
                        total_long_s += 1
                        if is_degraded: total_degraded += 1
                        
                except Exception as e:
                    print(f"  Error on {page_num} {side}: {e}")

    print("\n" + "="*60)
    print("AUDIT COMPLETE")
    print(f"Total Long-s Analyzed: {total_long_s}")
    print(f"Total Degraded: {total_degraded}")
    print(f"Global Degradation Rate: {(total_degraded/total_long_s)*100:.1f}%")
    print(f"Registry saved to: {csv_file}")
    print("="*60)

if __name__ == "__main__":
    main()
