#!/usr/bin/env python3
"""
Generate Forensic Proof HTML (The "Book of Evidence")
"""

import csv
from pathlib import Path

REPORT_DIR = Path("reports/full_audit")
HTML_PATH = REPORT_DIR / "forensic_proof.html"

def generate_html():
    registry_path = REPORT_DIR / "forensic_registry.csv"
    if not registry_path.exists():
        print("Registry not found!")
        return

    # Read Data
    matches = []
    with open(registry_path, 'r') as f:
        reader = csv.DictReader(f)
        matches = list(reader)

    if not matches:
        print("No matches to report.")
        return

    total = len(matches)
    degraded = len([m for m in matches if m['is_degraded'] == 'True'])
    rate = (degraded / total) * 100 if total > 0 else 0
    
    # HTML Template
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CodeFinder Forensic Proof: The "Book of Evidence"</title>
        <style>
            body {{ font-family: sans-serif; background: #1a1a1a; color: #ddd; padding: 20px; }}
            h1, h2 {{ color: #fff; }}
            .stats {{ background: #333; padding: 15px; border-radius: 8px; margin-bottom: 30px; display: flex; gap: 40px; }}
            .stat-box {{ text-align: center; }}
            .stat-val {{ font-size: 2em; font-weight: bold; color: #4CAF50; }}
            .stat-val.bad {{ color: #F44336; }}
            
            .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 15px; }}
            .card {{ background: #2a2a2a; border-radius: 6px; padding: 10px; text-align: center; border: 1px solid #444; }}
            .card.degraded {{ border-color: #F44336; background: #2a1a1a; }}
            .card img {{ width: 100%; border-radius: 4px; display: block; margin-bottom: 8px; }}
            .meta {{ font-size: 0.8em; color: #888; }}
            .identity {{ font-size: 1.2em; font-weight: bold; margin: 5px 0; }}
            .aspley {{ color: #4CAF50; }}
            .wright {{ color: #F44336; }}
        </style>
    </head>
    <body>
        <h1>Forensic Proof: The "Book of Evidence"</h1>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-label">Verified Samples</div>
                <div class="stat-val">{total}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Degraded (Wright)</div>
                <div class="stat-val bad">{degraded}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Degradation Rate</div>
                <div class="stat-val bad">{rate:.1f}%</div>
            </div>
        </div>

        <h2>Rogues Gallery (All {total} Matches)</h2>
        <div class="grid">
    """
    
    # Sort: Degraded first, then Page order
    matches.sort(key=lambda x: (x['is_degraded'] != 'True', int(x['page'])))
    
    for m in matches:
        is_bad = m['is_degraded'] == 'True'
        css_class = "degraded" if is_bad else "clean"
        w_char = m['wright_identity']
        
        # Determine strict display
        # The wright_char in CSV is from instance['character']
        
        html += f"""
            <div class="card {css_class}">
                <div class="meta">Page {m['page']} ({m['side']})</div>
                <img src="{m['image_path']}" loading="lazy">
                <div class="identity">
                    <span class="aspley">ſ</span> &rarr; 
                    <span class="wright">{w_char}</span>
                </div>
            </div>
        """
        
    html += """
        </div>
    </body>
    </html>
    """
    
    with open(HTML_PATH, 'w') as f:
        f.write(html)
        
    print(f"Report generated: {HTML_PATH}")

if __name__ == "__main__":
    generate_html()
