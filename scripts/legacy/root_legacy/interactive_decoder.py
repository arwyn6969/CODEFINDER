#!/usr/bin/env python3
"""
THE PROPHETIC DECODER RING
==========================
"Fun and Easy" English-First ELS Search.
Enter a word -> Get a Prophetic Card.
"""

import sys
import os
import time
import webbrowser
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.transliteration_service import TransliterationService
from app.services.els_visualizer import ELSVisualizer

# Load Torah text
def load_torah():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, "app/data/torah.txt"),
        os.path.join(current_dir, "data/torah.txt"),
        os.path.join(current_dir, "torah.txt")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read().strip()
    return None

def find_best_skip(text, term, limit=5000):
    """Find the best skip (smallest skip) for a term."""
    best_skip = 0
    min_dist = float('inf')
    
    # Try efficient search
    # This is a simplified version for the interactive tool
    # We prioritize finding *any* occurrence quickly
    
    # Check direct first
    if term in text:
        return 1, text.index(term)
        
    # Check skips
    for skip in range(2, 10000):
        # Scan text
        for start in range(len(text) // 2 - 40000, len(text) // 2 + 40000): # Scan middle first
             if start + (len(term)-1)*skip < len(text):
                match = True
                for i in range(len(term)):
                     if text[start + i*skip] != term[i]:
                        match = False
                        break
                if match:
                    return skip, start
                    
    # Broaden search if not found in middle
    for skip in range(2, 10000):
        for start in range(0, len(text), 20): # Finer scan
             if start + (len(term)-1)*skip < len(text):
                match = True
                for i in range(len(term)):
                     if text[start + i*skip] != term[i]:
                        match = False
                        break
                if match:
                    return skip, start
                    
    return 0, 0

def generate_card(word: str, hebrew: str, skip: int, idx: int, text: str):
    """Generate a cool HTML card."""
    
    # Get Matrix Visualization
    # We want a visual of the term in the matrix
    
    terms = [
        {"term": hebrew, "name": word.upper(), "color": "#00FF00", "skip": skip} # Neon Green
    ]
    
    # Add context terms if they exist nearby
    # PEPE = פפי
    pepe_skip, pepe_idx = find_best_skip(text, "פפי") # Just find one for context
    if pepe_idx and abs(pepe_idx - idx) < 2000:
        terms.append({"term": "פפי", "name": "PEPE", "color": "#9D00FF", "skip": pepe_skip})
        
    matrix_svg = ELSVisualizer.generate_svg_matrix(
        text=text,
        center_index=idx + (len(hebrew)*skip)//2, # Center on the word
        terms=terms,
        rows=15,
        cols=15,
        row_width=abs(skip) if abs(skip) < 50 else 20
    )
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prophetic Decoder: {word}</title>
        <style>
            body {{ background: #000; color: #fff; font-family: 'Courier New', monospace; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }}
            .card {{
                background: #111;
                border: 2px solid #00FF00;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 0 30px rgba(0, 255, 0, 0.2);
                max-width: 600px;
            }}
            h1 {{ font-size: 3em; margin: 0; color: #00FF00; text-shadow: 0 0 10px #00FF00; }}
            h2 {{ font-size: 1.5em; color: #fff; margin-bottom: 30px; }}
            .hebrew {{ font-size: 4em; color: #FFD700; font-family: 'Arial', sans-serif; margin: 20px 0; }}
            .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0; }}
            .stat {{ background: #222; padding: 15px; border-radius: 10px; }}
            .stat-label {{ color: #888; font-size: 0.8em; }}
            .stat-value {{ font-size: 1.5em; font-weight: bold; color: #00FF00; }}
            .matrix {{ margin-top: 30px; border: 1px solid #333; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>PROPHETIC DECODED</h1>
            <h2>{word.upper()}</h2>
            
            <div class="hebrew">{hebrew}</div>
            
            <div class="stat-grid">
                <div class="stat">
                    <div class="stat-label">ENCODING SKIP</div>
                    <div class="stat-value">{skip}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">MATRIX INDEX</div>
                    <div class="stat-value">{idx}</div>
                </div>
            </div>
            
            <div>{matrix_svg}</div>
            
            <p style="color: #666; margin-top: 20px;">VERIFIED IN THE ANCIENT CODE</p>
        </div>
    </body>
    </html>
    """
    
    filename = f"prophetic_card_{word.lower()}.html"
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    return out_path

def main():
    print("\n🔮 PROPHETIC DECODER RING INITIALIZED 🔮")
    print("---------------------------------------")
    
    text = load_torah()
    if not text:
        print("Error: Torah text not found.")
        return
        
    ts = TransliterationService()
    
    # Check for CLI args
    if len(sys.argv) > 1:
        words_to_process = [sys.argv[1]]
        interactive = False
    else:
        words_to_process = []
        interactive = True
    
    while True:
        if interactive:
            word = input("\nEnter an English word to decode (or 'q' to quit): ").strip()
            if word.lower() == 'q':
                break
        else:
            if not words_to_process:
                break
            word = words_to_process.pop(0)

        print(f"\nScanning the matrix for '{word.upper()}'...")
        
        # 1. Transliterate
        candidates = ts.get_hebrew_candidates(word)
        if not candidates:
            # Auto
            auto = ts.auto_transliterate(word)
            if auto:
                candidates = [(auto[0], "Auto-Transliterated")]
            else:
                print(f"Could not transliterate '{word}'. Try a simpler word.")
                if interactive: continue
                else: break
                
        hebrew = candidates[0][0]
        desc = candidates[0][1]
        print(f"-> Hebrew: {hebrew} ({desc})")
        
        # 2. Find in Torah
        skip, idx = find_best_skip(text, hebrew)
        
        if skip == 0:
            print("Not found in the matrix with simple search.")
        else:
            print(f"-> FOUND! Skip: {skip} at Index: {idx}")
            print("-> Generating Prophetic Card...")
            
            path = generate_card(word, hebrew, skip, idx, text)
            print(f"-> Card generated: {path}")

if __name__ == "__main__":
    main()
