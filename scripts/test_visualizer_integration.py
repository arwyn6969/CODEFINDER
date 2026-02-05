#!/usr/bin/env python3
"""
Verify Visualizer Integration
=============================
Smoke test for ELSVisualizer new methods.
"""
import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())

from app.services.els_visualizer import ELSVisualizer

def main():
    print("Testing ELSVisualizer SVG Generation...")
    
    text = "X" * 1000
    # Create simple pattern
    # ABC at 500, skip 10
    text = text[:500] + "A" + text[501:]
    text = text[:510] + "B" + text[511:]
    text = text[:520] + "C" + text[521:]
    
    terms = [
        {"term": "ABC", "name": "Test", "skip": 10, "color": "#FF0000"}
    ]
    
    svg = ELSVisualizer.generate_svg_matrix(text, 510, terms)
    
    if "<svg" in svg and "ABC" not in svg: # ABC terms are rendered as individual letters
        pass
        
    print(f"SVG Generated: {len(svg)} chars")
    
    with open("test_output.svg", "w") as f:
        f.write(svg)
        
    print("Saved test_output.svg")

if __name__ == "__main__":
    main()
