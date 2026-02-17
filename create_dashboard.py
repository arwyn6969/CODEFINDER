#!/usr/bin/env python3
"""
Forensic Analysis Dashboard

Creates a comprehensive visual summary of the Wright vs Aspley
edition comparison findings.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_dashboard(output_dir: str):
    """Create a comprehensive dashboard image."""
    
    # Dashboard dimensions
    width = 2400
    height = 1600
    
    # Create base image with dark theme
    dashboard = Image.new('RGB', (width, height), color='#1a1a2e')
    draw = ImageDraw.Draw(dashboard)
    
    # Load fonts
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        header_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        body_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        title_font = header_font = body_font = small_font = ImageFont.load_default()
    
    # ============ TITLE ============
    draw.text((width//2 - 450, 30), "FORENSIC PRINT BLOCK ANALYSIS DASHBOARD", 
              fill='white', font=title_font)
    draw.text((width//2 - 280, 90), "Shakespeare's Sonnets (1609) - Wright vs Aspley", 
              fill='#e94560', font=header_font)
    
    # ============ KEY METRICS PANEL ============
    # Left panel - metrics
    panel_y = 160
    panel_height = 200
    
    # Draw panel background
    draw.rectangle([(50, panel_y), (750, panel_y + panel_height)], 
                   fill='#16213e', outline='#0f3460', width=2)
    draw.text((70, panel_y + 10), "KEY METRICS", fill='#e94560', font=header_font)
    
    metrics = [
        ("Total Pages Compared", "53"),
        ("Significant Pages", "40 (75.5%)"),
        ("Overall Char Ratio", "1.316:1"),
        ("Critical Outliers", "4 pages"),
        ("Primary Cause", "OCR Quality")
    ]
    
    y_offset = panel_y + 50
    for label, value in metrics:
        draw.text((70, y_offset), f"{label}:", fill='#a2a2a2', font=small_font)
        draw.text((300, y_offset), value, fill='white', font=body_font)
        y_offset += 30
    
    # ============ STATISTICAL SIGNIFICANCE PANEL ============
    draw.rectangle([(780, panel_y), (1620, panel_y + panel_height)], 
                   fill='#16213e', outline='#0f3460', width=2)
    draw.text((800, panel_y + 10), "STATISTICAL SIGNIFICANCE", fill='#e94560', font=header_font)
    
    sig_data = [
        ("🔴 Critical (p<0.001)", "3 characters", "#E74C3C"),
        ("🟠 High (p<0.01)", "3 characters", "#F39C12"),
        ("🟡 Moderate (p<0.05)", "8 characters", "#F4D03F"),
        ("🟢 Not Significant", "All others", "#27AE60")
    ]
    
    y_offset = panel_y + 50
    for label, value, color in sig_data:
        draw.text((800, y_offset), label, fill=color, font=small_font)
        draw.text((1100, y_offset), value, fill='white', font=body_font)
        y_offset += 35
    
    # ============ CONCLUSION PANEL ============
    draw.rectangle([(1650, panel_y), (2350, panel_y + panel_height)], 
                   fill='#0f3460', outline='#e94560', width=3)
    draw.text((1670, panel_y + 10), "CONCLUSION", fill='#e94560', font=header_font)
    
    conclusion_lines = [
        "H₀ ACCEPTED:",
        "",
        "Editions printed from",
        "SAME TYPE SETTING",
        "",
        "All variance explained by",
        "OCR & scan quality"
    ]
    
    y_offset = panel_y + 55
    for line in conclusion_lines:
        if line == "SAME TYPE SETTING":
            draw.text((1670, y_offset), line, fill='#27AE60', font=header_font)
            y_offset += 35
        elif line:
            draw.text((1670, y_offset), line, fill='white', font=body_font)
            y_offset += 25
        else:
            y_offset += 10
    
    # ============ VISUALIZATION THUMBNAILS ============
    thumb_y = 400
    thumb_width = 700
    thumb_height = 400
    
    # Load and paste visualizations
    viz_files = [
        ("reports/page_forensics/page_variance_visualization.png", "Page Variance"),
        ("reports/page_forensics/statistical_analysis/statistical_analysis.png", "Statistical Analysis"),
        ("reports/page_forensics/deep_analysis/page_18_color_overlay.png", "Page 18 Overlay")
    ]
    
    x_offset = 50
    for viz_path, title in viz_files:
        try:
            viz = Image.open(viz_path)
            # Resize to thumbnail
            viz.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            
            # Create frame
            frame_x = x_offset
            frame_y = thumb_y
            draw.rectangle([(frame_x - 5, frame_y - 5), 
                           (frame_x + thumb_width + 5, frame_y + thumb_height + 35)],
                          fill='#16213e', outline='#0f3460', width=2)
            
            # Paste image centered
            paste_x = frame_x + (thumb_width - viz.size[0]) // 2
            paste_y = frame_y + (thumb_height - viz.size[1]) // 2
            dashboard.paste(viz, (paste_x, paste_y))
            
            # Title
            draw.text((frame_x + 10, frame_y + thumb_height + 5), 
                     title, fill='white', font=body_font)
        except Exception as e:
            draw.text((x_offset + 10, thumb_y + 10), 
                     f"[{title}]", fill='gray', font=body_font)
        
        x_offset += thumb_width + 30
    
    # ============ OUTLIER PAGES TABLE ============
    table_y = 850
    draw.rectangle([(50, table_y), (1150, table_y + 350)], 
                   fill='#16213e', outline='#0f3460', width=2)
    draw.text((70, table_y + 10), "TOP OUTLIER PAGES", fill='#e94560', font=header_font)
    
    # Table headers
    headers = ["Page", "Wright", "Aspley", "Delta", "Ratio", "Priority"]
    x_positions = [80, 200, 350, 500, 650, 800]
    
    draw.text((80, table_y + 55), "Page", fill='#e94560', font=small_font)
    draw.text((200, table_y + 55), "Wright", fill='#e94560', font=small_font)
    draw.text((350, table_y + 55), "Aspley", fill='#e94560', font=small_font)
    draw.text((500, table_y + 55), "Delta", fill='#e94560', font=small_font)
    draw.text((650, table_y + 55), "Ratio", fill='#e94560', font=small_font)
    draw.text((800, table_y + 55), "Priority", fill='#e94560', font=small_font)
    
    # Draw separator
    draw.line([(70, table_y + 80), (1130, table_y + 80)], fill='#0f3460', width=1)
    
    # Top outliers data
    outliers = [
        (18, 997, 1705, "+708", "1.71x", "🔴 CRITICAL"),
        (33, 1319, 1987, "+668", "1.51x", "🔴 CRITICAL"),
        (17, 1208, 1851, "+643", "1.53x", "🔴 CRITICAL"),
        (41, 668, 1303, "+635", "1.95x", "🔴 CRITICAL"),
        (27, 1333, 1920, "+587", "1.44x", "🟠 HIGH"),
        (39, 1104, 1680, "+576", "1.52x", "🟠 HIGH"),
        (16, 1245, 1812, "+567", "1.46x", "🟠 HIGH"),
        (19, 1259, 1821, "+562", "1.45x", "🟠 HIGH"),
    ]
    
    y_offset = table_y + 95
    for page, wright, aspley, delta, ratio, priority in outliers:
        draw.text((80, y_offset), str(page), fill='white', font=small_font)
        draw.text((200, y_offset), str(wright), fill='#2E86AB', font=small_font)
        draw.text((350, y_offset), str(aspley), fill='#A23B72', font=small_font)
        draw.text((500, y_offset), delta, fill='#F4D03F', font=small_font)
        draw.text((650, y_offset), ratio, fill='white', font=small_font)
        draw.text((800, y_offset), priority, fill='white', font=small_font)
        y_offset += 30
    
    # ============ CATEGORY ANALYSIS PANEL ============
    draw.rectangle([(1180, table_y), (2350, table_y + 350)], 
                   fill='#16213e', outline='#0f3460', width=2)
    draw.text((1200, table_y + 10), "CHARACTER CATEGORY ANALYSIS", fill='#e94560', font=header_font)
    
    categories = [
        ("Lowercase", "1.31x", "✅ Normal", "#27AE60"),
        ("Uppercase", "1.52x", "⚠️ +16%", "#F4D03F"),
        ("Punctuation", "1.47x", "⚠️ +12%", "#F4D03F"),
        ("Digits", "1.84x", "🔴 +40%", "#E74C3C"),
        ("Special", "0.54x", "🔴 -59%", "#E74C3C"),
    ]
    
    y_offset = table_y + 60
    for cat, ratio, status, color in categories:
        draw.text((1200, y_offset), cat, fill='white', font=body_font)
        draw.text((1450, y_offset), ratio, fill='#3498DB', font=body_font)
        draw.text((1570, y_offset), status, fill=color, font=body_font)
        y_offset += 45
    
    # Interpretation
    draw.line([(1200, y_offset + 10), (2330, y_offset + 10)], fill='#0f3460', width=1)
    y_offset += 25
    
    interp_lines = [
        "Digit variance (1.84x) suggests OCR",
        "struggles with numerals in Wright scan.",
        "",
        "Special char inversion (0.54x) reflects",
        "different character encoding artifacts."
    ]
    
    for line in interp_lines:
        if line:
            draw.text((1200, y_offset), line, fill='#a2a2a2', font=small_font)
        y_offset += 22
    
    # ============ FOOTER ============
    draw.rectangle([(0, height - 60), (width, height)], fill='#0f3460')
    draw.text((50, height - 45), "CODEFINDER Forensic Analysis Suite | February 5, 2026 | Wright (STC 22353a) vs Aspley (STC 22353)", 
              fill='#a2a2a2', font=small_font)
    draw.text((width - 400, height - 45), "Shakespeare's Sonnets, London, 1609", 
              fill='#e94560', font=small_font)
    
    # Save dashboard
    output_path = Path(output_dir) / 'FORENSIC_DASHBOARD.png'
    dashboard.save(output_path, quality=95)
    print(f"Dashboard saved: {output_path}")
    
    return str(output_path)

def main():
    output_dir = "reports/page_forensics"
    create_dashboard(output_dir)
    print("✅ Dashboard generation complete!")

if __name__ == "__main__":
    main()
