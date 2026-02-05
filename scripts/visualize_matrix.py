#!/usr/bin/env python3
"""
VISUALIZE MATRIX: GENESIS 1 CONVERGENCE
=======================================
Generates a high-fidelity HTML visualization of the actual Torah matrix
at the "Best Convergence" point (Genesis 1, position 922).

Highlights:
- PEPE (Green)
- MEME (Purple)
- FROG (Teal)
- TRUE LETTERS from the Torah text
"""

from pathlib import Path
import math

# Torah loading
def load_torah():
    path = Path(__file__).parent / "app/data/torah.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# Configuration for the Best Convergence (from previous analysis)
CENTER_POS = 922
TERMS = [
    {"term": "פפי", "name": "PEPE", "skip": 12, "color": "#4CAF50", "glow": "0 0 10px #4CAF50"},      # Green
    {"term": "מימי", "name": "MEME", "skip": 61, "color": "#9C27B0", "glow": "0 0 10px #9C27B0"},    # Purple
    {"term": "צפר", "name": "FROG", "skip": 10, "color": "#009688", "glow": "0 0 10px #009688"},     # Teal
]

# Grid settings
ROWS = 15
COLS = 15
ROW_WIDTH = 12  # Based on one of the skips or arbitrary for best view?
# Let's try to find a row width that makes at least one term look straight or diagonal.
# PEPE is skip 12. If row width is 12, PEPE will be a vertical column. Perfect.
ROW_WIDTH = 12 

def generate_html(text):
    # Calculate grid bounds
    # Center pos is 922.
    # Grid coordinate of center: row = 922 // 12, col = 922 % 12
    center_row = CENTER_POS // ROW_WIDTH
    center_col = CENTER_POS % ROW_WIDTH
    
    start_row = center_row - (ROWS // 2)
    start_col = center_col - (COLS // 2)
    
    # Identify highlighted indices
    highlights = {}
    for t in TERMS:
        # Find start index that places the term near the center
        # For this demo, let's just search for the term near the center 
        # using the known skip to ensure we get the exact same hit.
        
        # Search range around center
        found = False
        search_start = max(0, CENTER_POS - 1000)
        search_end = min(len(text), CENTER_POS + 1000)
        
        # We need the exact hit from the analysis
        # Analysis said: 
        # #1: Center 922
        # PEPE: פפי skip 12
        # FROG: צפר skip 10
        # MEME: מימי skip 61
        
        # Let's brute force find the start index that crosses near 922
        for i in range(search_start, search_end):
            # Check PEPE
            if t["name"] == "PEPE":
                # Check if this index + skip*n forms the word
                if text[i] == t["term"][0]:
                    match = True
                    for k in range(len(t["term"])):
                        if text[i + k*t["skip"]] != t["term"][k]:
                            match = False
                            break
                    if match:
                        # Is it the one near 922?
                        # Check distance of middle letter to 922
                        mid = i + (len(t["term"])//2 * t["skip"])
                        if abs(mid - CENTER_POS) < 20: # Close enough
                            # Mark indices
                            for k in range(len(t["term"])):
                                idx = i + k*t["skip"]
                                highlights[idx] = t
                            found = True
                            print(f"Found PEPE at {i}")
                            break
                            
            # Check FROG
            elif t["name"] == "FROG":
                if text[i] == t["term"][0]:
                    match = True
                    for k in range(len(t["term"])):
                        if text[i + k*t["skip"]] != t["term"][k]:
                            match = False
                            break
                    if match:
                        mid = i + (len(t["term"])//2 * t["skip"])
                        if abs(mid - CENTER_POS) < 20: 
                            for k in range(len(t["term"])):
                                idx = i + k*t["skip"]
                                highlights[idx] = t
                            found = True
                            print(f"Found FROG at {i}")
                            break

            # Check MEME
            elif t["name"] == "MEME":
                if text[i] == t["term"][0]:
                    match = True
                    for k in range(len(t["term"])):
                        if text[i + k*t["skip"]] != t["term"][k]:
                            match = False
                            break
                    if match:
                        mid = i + (len(t["term"])//2 * t["skip"])
                        print(f"DEBUG: Found MEME at start {i}, mid {mid}, dist {abs(mid - CENTER_POS)}")
                        if abs(mid - CENTER_POS) < 150: 
                            for k in range(len(t["term"])):
                                idx = i + k*t["skip"]
                                highlights[idx] = t
                            found = True
                            print(f"Found MEME at {i}")
                            break
    
    # Generate HTML content
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                background-color: #0d0d14;
                color: #333;
                font-family: 'Courier New', Courier, monospace;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify_content: center;
                height: 100vh;
                margin: 0;
            }}
            .container {{
                text-align: center;
            }}
            h1 {{
                color: #fff;
                margin-bottom: 10px;
                font-family: Arial, sans-serif;
                letter-spacing: 2px;
            }}
            .matrix {{
                display: grid;
                grid-template-columns: repeat({COLS}, 40px);
                gap: 5px;
                padding: 20px;
                background: #15151e;
                border-radius: 10px;
                box-shadow: 0 0 30px rgba(0,0,0,0.5);
                border: 1px solid #333;
            }}
            .cell {{
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                color: #555;
                font-weight: bold;
                border-radius: 4px;
                transition: transform 0.3s;
                cursor: default;
            }}
            .cell:hover {{
                background: #222;
                transform: scale(1.1);
            }}
            .highlight {{
                color: #fff !important;
                font-weight: 900;
            }}
            .legend {{
                margin-top: 20px;
                display: flex;
                gap: 20px;
                justify-content: center;
            }}
            .legend-item {{
                padding: 5px 15px;
                border-radius: 20px;
                color: white;
                font-weight: bold;
                font-family: Arial, sans-serif;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>GENESIS 1 CONVERGENCE (POS 922)</h1>
            <h3 style="color:#888; margin-top:0;">Skip Width: {ROW_WIDTH} (PEPE alignment)</h3>
            <div class="matrix">
    """
    
    # Generate cells
    for r in range(ROWS):
        curr_row = start_row + r
        for c in range(COLS):
            curr_col = start_col + c
            
            # Calculate actual text index
            # Index = (row * width) + col
            # Note: Biblical Hebrew reads right-to-left, but standard ELS arrays usually 
            # map index 0 to top-right or top-left depending on implementation.
            # Here we map strictly by index: 0, 1, 2...
            
            # Since we display left-to-right in HTML grid, 
            # if we want right-to-left reading, we might flip columns?
            # Standard "CodeFinder" displays usually read right-to-left.
            # Let's stick to simple index mapping first.
            
            idx = (curr_row * ROW_WIDTH) + curr_col
            
            char = text[idx] if 0 <= idx < len(text) else ""
            
            style = ""
            cls = "cell"
            
            if idx in highlights:
                t = highlights[idx]
                style = f"background-color: {t['color']}22; color: {t['color']}; text-shadow: {t['glow']}; border: 1px solid {t['color']};"
                cls += " highlight"
                
            html += f'<div class="{cls}" style="{style}" title="Index: {idx}">{char}</div>'
            
    html += """
            </div>
            <div class="legend">
                <div class="legend-item" style="background:#4CAF50; box-shadow: 0 0 10px #4CAF50;">PEPE (Skip 12)</div>
                <div class="legend-item" style="background:#009688; box-shadow: 0 0 10px #009688;">FROG (Skip 10)</div>
                <div class="legend-item" style="background:#9C27B0; box-shadow: 0 0 10px #9C27B0;">MEME (Skip 61)</div>
            </div>
        </div>
    </body>
    </html>
    """
    
    output_path = "pepe_matrix_visualization.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    text = load_torah()
    generate_html(text)
