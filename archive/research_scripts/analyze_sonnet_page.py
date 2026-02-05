import sys
import os
import fitz # PyMuPDF
import pytesseract
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Any
import math

# ----------------------------------------------------------------------------
# 1. SETUP
# ----------------------------------------------------------------------------
@dataclass
class CharacterBox:
    character: str
    x: float
    y: float
    width: float
    height: float

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Pattern:
    type: str # 'right_angle', 'golden_ratio'
    points: List[Point]

# ----------------------------------------------------------------------------
# 2. LIGHTWEIGHT GEOMETRY ENGINE
# ----------------------------------------------------------------------------
def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def find_patterns(chars: List[CharacterBox]) -> List[Pattern]:
    points = [Point(c.x + c.width/2, c.y + c.height/2) for c in chars]
    patterns = []
    
    n = len(points)
    print(f"   ⚡ Fast-Scanning {n} points ({n*(n-1)*(n-2)//6} triangles)...")
    
    # 1. Golden Ratio (Distance checks) - O(N^2)
    # Check 500 pairs max for speed
    pairs_checked = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = dist(points[i], points[j])
            if d < 10: continue # Skip close
            
            # Simple check: Is this vertical Golden Ratio to page height?
            # Or is it related to another distance?
            # Let's just create line for specific relationships
            
            # GOLDEN RATIO CHECK: Segment A vs Segment B? 
            # Simplified: Just check if Dy / Dx is roughly Phi (1.618)
            dx = abs(points[i].x - points[j].x)
            dy = abs(points[i].y - points[j].y)
            if dx > 1:
                ratio = dy / dx
                if 1.5 < ratio < 1.7:
                    patterns.append(Pattern("golden_ratio", [points[i], points[j]]))
            
            pairs_checked += 1
            if pairs_checked > 2000: break
        if pairs_checked > 2000: break

    # 2. Right Angles - O(N^3)
    # Limit to first 20 points for cubic check
    limit_cub = min(n, 20) 
    print(f"   📐 Checking angles for top {limit_cub} points...")
    
    for i in range(limit_cub):
        for j in range(i + 1, limit_cub):
            for k in range(j + 1, limit_cub):
                p1, p2, p3 = points[i], points[j], points[k]
                
                # Check angle at p1, p2, p3
                # Vector a = p2-p1, b = p3-p1
                # cos theta = dot(a,b) / (|a||b|)
                
                def check_angle(v, a, b):
                    vx, vy = v.x, v.y
                    ax, ay = a.x - vx, a.y - vy
                    bx, by = b.x - vx, b.y - vy
                    dot = ax*bx + ay*by
                    mag_a = math.hypot(ax, ay)
                    mag_b = math.hypot(bx, by)
                    if mag_a * mag_b == 0: return False
                    cosine = dot / (mag_a * mag_b)
                    angle = math.degrees(math.acos(max(-1, min(1, cosine))))
                    return 88 < angle < 92
                
                if check_angle(p1, p2, p3):
                    patterns.append(Pattern("right_angle", [p1, p2, p3]))
                elif check_angle(p2, p1, p3):
                    patterns.append(Pattern("right_angle", [p2, p1, p3]))
                elif check_angle(p3, p1, p2):
                    patterns.append(Pattern("right_angle", [p3, p1, p2]))

    return patterns

# ----------------------------------------------------------------------------
# 3. SVG GENERATOR
# ----------------------------------------------------------------------------
class SVGGenerator:
    def __init__(self, width, height, filename="sonnet_12_geometry.svg"):
        self.width = width
        self.height = height
        self.filename = filename
        self.elements = []

    def add_line(self, x1, y1, x2, y2, color="blue", width=1, dash=None):
        dash_attr = f'stroke-dasharray="{dash}"' if dash else ''
        self.elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" {dash_attr} opacity="0.6" />')

    def add_text(self, x, y, text, color="black", size=10):
        # Escape XML chars
        safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.elements.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="Courier" font-weight="bold" font-size="{size}">{safe_text}</text>')

    def save(self):
        with open(self.filename, "w") as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" style="background-color: white;">\n')
            f.write("\n".join(self.elements))
            f.write('\n</svg>')
        print(f"✅ Saved visualization to {self.filename}")

# ----------------------------------------------------------------------------
# 4. MAIN LOGIC
# ----------------------------------------------------------------------------
def analyze_page():
    pdf_path = "data/sources/archive/SONNETS_QUARTO_1609_NET.pdf"
    page_index = 11 # "Page 12"
    
    print(f"📖 Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    
    print(f"📷 Rasterizing Page {page_index + 1}...")
    zoom = 2
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    print("🔍 Running OCR...")
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config='--psm 6')
    
    chars = []
    n_boxes = len(data['text'])
    svg = SVGGenerator(pix.width, pix.height)
    
    for i in range(n_boxes):
        text = data['text'][i]
        if not text or text.isspace(): continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        char_w = w / len(text)
        for idx, char in enumerate(text):
            if char.isalnum() or char in ".,:;?!-&":
                cx = x + (idx * char_w)
                cy = y 
                box = CharacterBox(char, cx, cy, char_w, h)
                chars.append(box)
                svg.add_text(cx, cy + h*0.8, char, size=h)

    print(f"   Extracted {len(chars)} valid characters.")

    # Optimized Selection
    print("✨ Filtering for TOP 80 Capital Letters & Punctuation...")
    significant = [c for c in chars if c.character.isupper() or c.character in ".!?,"]
    points = significant[:80] # Hard limit
    print(f"   Analyzing {len(points)} points...")

    # Geometric Analysis
    patterns = find_patterns(points)
    print(f"   Found {len(patterns)} patterns.")
    
    for pat in patterns:
        c = "blue"
        w = 1
        if pat.type == "right_angle":
            c = "red"
            w = 2
            # Draw triangle
            p1, p2, p3 = pat.points
            svg.add_line(p1.x, p1.y, p2.x, p2.y, c, w)
            svg.add_line(p2.x, p2.y, p3.x, p3.y, c, w)
            svg.add_line(p3.x, p3.y, p1.x, p1.y, c, w, dash="2,2")
            
        elif pat.type == "golden_ratio":
            c = "gold"
            w = 3
            p1, p2 = pat.points
            svg.add_line(p1.x, p1.y, p2.x, p2.y, c, w)

    svg.save()

if __name__ == "__main__":
    analyze_page()
