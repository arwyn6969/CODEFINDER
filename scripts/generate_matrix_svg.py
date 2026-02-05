#!/usr/bin/env python3
"""
GENERATE MATRIX SVG
===================
Generates a standalone SVG vector graphic of the Torah matrix at Genesis 922.
Perfect for slide decks.
"""

from pathlib import Path

# Configuration (Same as HTML)
CENTER_POS = 922
ROWS = 15
COLS = 15
ROW_WIDTH = 12
CELL_SIZE = 40
GAP = 5
PADDING = 40

TERMS = [
    {"term": "פפי", "name": "PEPE", "skip": 12, "color": "#4CAF50", "stroke": "#2E7D32"},      # Green
    {"term": "מימי", "name": "MEME", "skip": 61, "color": "#9C27B0", "stroke": "#7B1FA2"},    # Purple
    {"term": "צפר", "name": "FROG", "skip": 10, "color": "#009688", "stroke": "#00695C"},     # Teal
]

def load_torah():
    path = Path(__file__).parent / "app/data/torah.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def generate_svg(text):
    # Calculate grid bounds
    center_row = CENTER_POS // ROW_WIDTH
    center_col = CENTER_POS % ROW_WIDTH
    start_row = center_row - (ROWS // 2)
    start_col = center_col - (COLS // 2)

    # Identify highlights (Logic from visualize_matrix.py)
    highlights = {}
    for t in TERMS:
        search_start = max(0, CENTER_POS - 1000)
        search_end = min(len(text), CENTER_POS + 1000)
        
        for i in range(search_start, search_end):
            if t["name"] == "PEPE":
                if text[i] == t["term"][0]:
                    match = True
                    for k in range(len(t["term"])):
                        if text[i + k*t["skip"]] != t["term"][k]:
                            match = False; break
                    if match:
                        mid = i + (len(t["term"])//2 * t["skip"])
                        if abs(mid - CENTER_POS) < 150: 
                            for k in range(len(t["term"])):
                                highlights[i + k*t["skip"]] = t
                            break
            elif t["name"] == "FROG":
                 if text[i] == t["term"][0]:
                    match = True
                    for k in range(len(t["term"])):
                        if text[i + k*t["skip"]] != t["term"][k]:
                            match = False; break
                    if match:
                        mid = i + (len(t["term"])//2 * t["skip"])
                        if abs(mid - CENTER_POS) < 150: 
                            for k in range(len(t["term"])):
                                highlights[i + k*t["skip"]] = t
                            break
            elif t["name"] == "MEME":
                 if text[i] == t["term"][0]:
                    match = True
                    for k in range(len(t["term"])):
                        if text[i + k*t["skip"]] != t["term"][k]:
                            match = False; break
                    if match:
                        mid = i + (len(t["term"])//2 * t["skip"])
                        if abs(mid - CENTER_POS) < 150: 
                            for k in range(len(t["term"])):
                                highlights[i + k*t["skip"]] = t
                            break

    # SVG Construction
    width = (COLS * (CELL_SIZE + GAP)) + (PADDING * 2)
    height = (ROWS * (CELL_SIZE + GAP)) + (PADDING * 2) + 60 # Extra for legend
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background-color: #0d0d14; font-family: monospace;">']
    
    # Title
    svg.append(f'<text x="{width/2}" y="30" text-anchor="middle" fill="#ffffff" font-size="16" font-family="Arial" font-weight="bold">GENESIS 1 CONVERGENCE (POS 922)</text>')
    
    # Grid
    for r in range(ROWS):
        for c in range(COLS):
            curr_row = start_row + r
            curr_col = start_col + c
            idx = (curr_row * ROW_WIDTH) + curr_col
            
            x = PADDING + (c * (CELL_SIZE + GAP))
            y = PADDING + 30 + (r * (CELL_SIZE + GAP)) # Offset for title
            
            char = text[idx] if 0 <= idx < len(text) else ""
            
            fill = "#15151e"
            stroke = "#333333"
            text_fill = "#555555"
            stroke_width = "1"
            font_weight = "normal"
            
            if idx in highlights:
                t = highlights[idx]
                fill = t["color"]
                stroke = t["stroke"]
                text_fill = "#ffffff"
                stroke_width = "2"
                font_weight = "bold"
            
            # Cell rect
            svg.append(f'<rect x="{x}" y="{y}" width="{CELL_SIZE}" height="{CELL_SIZE}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>')
            # Text
            svg.append(f'<text x="{x + CELL_SIZE/2}" y="{y + CELL_SIZE/2 + 6}" text-anchor="middle" fill="{text_fill}" font-size="20" font-weight="{font_weight}">{char}</text>')

    # Legend
    legend_y = height - 30
    legend_start_x = width / 2 - 150
    
    # Pepe Legend
    svg.append(f'<rect x="{legend_start_x}" y="{legend_y}" width="80" height="25" rx="10" fill="#4CAF50"/>')
    svg.append(f'<text x="{legend_start_x + 40}" y="{legend_y + 17}" text-anchor="middle" fill="white" font-size="12" font-weight="bold" font-family="Arial">PEPE</text>')
    
    # Frog Legend
    svg.append(f'<rect x="{legend_start_x + 90}" y="{legend_y}" width="80" height="25" rx="10" fill="#009688"/>')
    svg.append(f'<text x="{legend_start_x + 130}" y="{legend_y + 17}" text-anchor="middle" fill="white" font-size="12" font-weight="bold" font-family="Arial">FROG</text>')
    
    # Meme Legend
    svg.append(f'<rect x="{legend_start_x + 180}" y="{legend_y}" width="80" height="25" rx="10" fill="#9C27B0"/>')
    svg.append(f'<text x="{legend_start_x + 220}" y="{legend_y + 17}" text-anchor="middle" fill="white" font-size="12" font-weight="bold" font-family="Arial">MEME</text>')

    svg.append('</svg>')
    
    output_path = "pepe_matrix.svg"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"SVG saved to {output_path}")

if __name__ == "__main__":
    text = load_torah()
    generate_svg(text)
