import csv
import cv2
import numpy as np
from pathlib import Path
from fpdf import FPDF

# Constants
CSV_PATH = Path("reports/census_aligned/census_data.csv")
ATLAS_DIR = Path("reports/census_aligned/atlas") 
SONNET_DIR = Path("reports/sonnet_isolation_aligned")
OUTPUT_PDF = Path("reports/census_aligned/Variance_Atlas.pdf")

def generate_atlas():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return

    print("Loading Census Data...")
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    print(f"Loaded {len(data)} records.")

    # Convert scores to float
    for row in data:
        row['diff_score'] = float(row['diff_score'])
        try:
            row['ncc_score'] = float(row['ncc_score'])
        except:
            row['ncc_score'] = 0.0

    # Sort by Diff Score (Descending)
    # Filter out very small boxes (noise)
    valid_data = [x for x in data if int(x['w']) > 8 and int(x['h']) > 10]
    
    valid_data.sort(key=lambda x: x['diff_score'], reverse=True)
    
    # Take Top 50
    top_50 = valid_data[:50]
    
    print(f"Generating Atlas for Top {len(top_50)} variants...")
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title Page
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 20, "The Variance Atlas", ln=True, align='C')
    pdf.set_font("Arial", '', 14)
    pdf.cell(0, 10, "Top 50 Typographical Variants (Wright vs Aspley)", ln=True, align='C')
    pdf.ln(20)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, "This document highlights the characters with the greatest physical difference between the two copies. Use this to identify potential font changes, broken type, or unique corrections.")
    
    # Process
    # Ensure temp dir for extracts
    temp_dir = Path("reports/census/temp_atlas")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    for i, item in enumerate(top_50):
        if i % 4 == 0:
            pdf.add_page()
            
        s_id = f"{int(item['sonnet_id']):03d}"
        char = item['char']
        x = int(item['x'])
        y = int(item['y']) # Tesseract Y is flipped? No, my script flipped it.
        w = int(item['w'])
        h = int(item['h'])
        
        # Load Images
        w_path = SONNET_DIR / f"sonnet_{s_id}_wright.png"
        a_path = SONNET_DIR / f"sonnet_{s_id}_aspley.png"
        
        if not w_path.exists() or not a_path.exists():
            continue
            
        img_w = cv2.imread(str(w_path))
        img_a = cv2.imread(str(a_path)) # Using raw Aspley, or should I verify alignment?
        
        # Note: Census used ECC aligned Wright.
        # But for visualization, raw might be better if alignment failed?
        # No, show Aligned if possible. But I didn't save aligned images.
        # I'll re-align just this snippet? Expensive.
        # I'll show RAW for now. The user wants to see strict Comparison.
        # Ideally I should replicate the transformation.
        
        # Let's crop with context
        pad_x = 20
        pad_y = 10
        
        # Coords
        # Wait, the CSV coords (x,y) are relative to the Aspley Image (Ref).
        # So I can crop Aspley directly.
        # For Wright, if not aligned, the coords might be off.
        # But 'isolate_sonnets.py' centered them. They should be close.
        # I'll show a larger crop for Wright to ensure we catch it.
        
        y1 = max(0, y - pad_y)
        y2 = min(img_a.shape[0], y + h + pad_y)
        x1 = max(0, x - pad_x)
        x2 = min(img_a.shape[1], x + w + pad_x)
        
        crop_a = img_a[y1:y2, x1:x2]
        
        # For Wright, take same region
        wy2 = min(img_w.shape[0], y2)
        wx2 = min(img_w.shape[1], x2)
        crop_w = img_w[y1:wy2, x1:wx2]
        
        if crop_w.size == 0 or crop_a.size == 0:
            continue
            
        # Save temp
        t_a = temp_dir / f"{i}_aspley.png"
        t_w = temp_dir / f"{i}_wright.png"
        cv2.imwrite(str(t_a), crop_a)
        cv2.imwrite(str(t_w), crop_w)
        
        # Add to PDF
        y_pos = pdf.get_y()
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"#{i+1}: Sonnet {item['sonnet_id']} - '{char}' (Diff: {item['diff_score']:.1f})", ln=True)
        
        # Images
        pdf.image(str(t_w), x=20, y=y_pos+10, h=30)
        pdf.image(str(t_a), x=110, y=y_pos+10, h=30)
        
        pdf.set_y(y_pos + 45)
        pdf.set_font("Arial", '', 10)
        pdf.text(35, y_pos + 45, "Wright (Target)")
        pdf.text(125, y_pos + 45, "Aspley (Reference)")
        
        pdf.ln(10)
        
    pdf.output(str(OUTPUT_PDF))
    print(f"Atlas saved to {OUTPUT_PDF}")

if __name__ == "__main__":
    generate_atlas()
