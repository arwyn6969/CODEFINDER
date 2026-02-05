import sys
import os
import math
from dataclasses import dataclass
from unittest.mock import MagicMock
from typing import List, Tuple

# ----------------------------------------------------------------------------
# 1. MOCKING & SETUP
# ----------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class CharacterBox:
    character: str
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0

# Mock ocr_engine
ocr_mock = MagicMock()
ocr_mock.CharacterBox = CharacterBox
sys.modules['app.services.ocr_engine'] = ocr_mock

try:
    from app.services.geometric_analyzer import GeometricAnalyzer, Point
except ImportError:
    print("❌ Import failed.")
    sys.exit(1)

# ----------------------------------------------------------------------------
# 2. SVG GENERATOR
# ----------------------------------------------------------------------------
class SVGGenerator:
    def __init__(self, width=1000, height=1000, filename="geometry_demo.svg"):
        self.width = width
        self.height = height
        self.filename = filename
        self.elements = []
        
    def add_line(self, x1, y1, x2, y2, color="black", width=2, opacity=1.0, dash=None):
        dash_attr = f'stroke-dasharray="{dash}"' if dash else ''
        self.elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}" {dash_attr} />')

    def add_circle(self, cx, cy, r, color="black", fill="none", opacity=1.0):
        self.elements.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" stroke="{color}" fill="{fill}" fill-opacity="{opacity}" />')

    def add_text(self, x, y, text, color="black", size=12):
        self.elements.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial" font-size="{size}">{text}</text>')

    def add_rect(self, x, y, w, h, color="none", fill="none"):
        self.elements.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" stroke="{color}" fill="{fill}" />')

    def save(self):
        with open(self.filename, "w") as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" style="background-color: #f8f9fa;">\n')
            f.write(f'  <rect x="0" y="0" width="{self.width}" height="{self.height}" fill="#white" stroke="#ccc" />\n')
            # Grid
            for i in range(0, self.width, 100):
                f.write(f'  <line x1="{i}" y1="0" x2="{i}" y2="{self.height}" stroke="#eee" stroke-width="1" />\n')
            for i in range(0, self.height, 100):
                f.write(f'  <line x1="0" y1="{i}" x2="{self.width}" y2="{i}" stroke="#eee" stroke-width="1" />\n')
            
            f.write("\n".join(self.elements))
            f.write('\n</svg>')
        print(f"✅ Saved visualization to {self.filename}")

# ----------------------------------------------------------------------------
# 3. LOGIC
# ----------------------------------------------------------------------------
def main():
    print("🎨 Generating Geometric Visualization...")
    svg = SVGGenerator(800, 1000, "geometry_demo.svg")
    
    # 1. Create Data (Same as Demo)
    chars = []
    def add_char(c, x, y):
        chars.append(CharacterBox(character=c, x=x, y=y, width=20, height=20))
        # Draw the point immediately
        svg.add_circle(x, y, 4, color="#333", fill="#333")
        svg.add_text(x + 10, y - 10, f"{c} ({x},{y})", color="#666", size=10)

    # Pythagorean (A, B, C)
    add_char("A", 100, 100)
    add_char("B", 500, 100)
    add_char("C", 100, 400)
    
    # Golden Ratio (D, E)
    phi = 1.61803398875
    add_char("D", 100, 600)
    add_char("E", 100 + (100 * phi), 600)
    
    # Equilateral (F, G, H)
    add_char("F", 100, 800)
    add_char("G", 300, 800)
    add_char("H", 200, 800 - 173.205)

    # 2. Analyze
    analyzer = GeometricAnalyzer()
    results = analyzer.analyze_character_geometry(chars)

    # 3. Visualize Results
    
    # Draw Patterns
    colors = {
        "triangular_pattern": "blue",
        "right_angle_pattern": "red",
        "cross_pattern": "purple"
    }

    print(f"   found {len(results['geometric_patterns'])} patterns")

    for pat in results['geometric_patterns']:
        pts = pat.elements
        c = colors.get(pat.relationship_type, "green")
        
        # Connect the dots
        if "right_angle" in pat.relationship_type:
            # Vertex is usually first in our logic? Check implementation: elements=[vertex, p1, p2]
            v, p1, p2 = pts[0], pts[1], pts[2]
            svg.add_line(v.x, v.y, p1.x, p1.y, color=c, width=3)
            svg.add_line(v.x, v.y, p2.x, p2.y, color=c, width=3)
            svg.add_text(v.x - 20, v.y - 20, "Right Angle (90°)", color=c, size=14)
            
        elif "triangular" in pat.relationship_type:
            # elements=[vertex, p1, p1(dup from loops), p2]... 
            # The logic in _detect_triangular_patterns returns elements=[vertex, angles[i].point1, angles[j].point1, angles[k].point1]
            # Wait, looking at source: elements=[vertex, angles[i].point1, angles[j].point1, angles[k].point1]
            # Actually simplest to just draw lines between all unique points in elements
            unique_pts = list(set([(p.x, p.y) for p in pts]))
            
            # Simple polygon draw
            for i in range(len(unique_pts)):
                p1 = unique_pts[i]
                p2 = unique_pts[(i+1) % len(unique_pts)]
                svg.add_line(p1[0], p1[1], p2[0], p2[1], color="blue", width=1, dash="5,5")
            
            # Label center
            cx = sum(p[0] for p in unique_pts)/len(unique_pts)
            cy = sum(p[1] for p in unique_pts)/len(unique_pts)
            svg.add_text(cx, cy, "Triangular", color="blue", size=10)

    # Draw Trig Relationships (Golden Ratio)
    for trig in results['trigonometric_relationships']:
        if trig.mathematical_constant == "golden_ratio":
            # We don't have the points directly in TrigRelationship (it stores angles/ratios), 
            # but we know D and E are the golden ratio ones from our setup.
            # Manually highlight D-E for the demo visual
            svg.add_line(100, 600, 100 + (100 * phi), 600, color="gold", width=4)
            svg.add_text(150, 580, "GOLDEN RATIO (φ)", color="#d4af37", size=16)

    svg.save()

if __name__ == "__main__":
    main()
