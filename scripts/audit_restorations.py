import json
import cv2
import random
import numpy as np
from pathlib import Path
from fpdf import FPDF

# Constants
JSON_PATH = Path("reports/ink_dot_analysis_v3/candidate_restorations.json")
IMG_DIR = Path("reports/ink_dot_analysis_v3/evidence_audit")
IMG_DIR.mkdir(parents=True, exist_ok=True)
WRIGHT_DIR = Path("reports/scan_wright_fixed/page_images")
ASPLEY_DIR = Path("reports/scan_aspley_fixed/page_images")

def audit_candidates():
    if not JSON_PATH.exists():
        print(f"Error: {JSON_PATH} not found.")
        return

    with open(JSON_PATH, 'r') as f:
        candidates = json.load(f)

    print(f"Loaded {len(candidates)} candidates.")
    
    # Sample 20 random candidates (or top 20 by area)
    # Let's take top 10 by area + 10 random for variety
    candidates.sort(key=lambda x: x['area'], reverse=True)
    top_10 = candidates[:10]
    remaining = candidates[10:]
    random_10 = random.sample(remaining, min(len(remaining), 10))
    
    audit_set = top_10 + random_10
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 10, "Handwriting Restoration Audit", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Sample of {len(audit_set)} candidates from {len(candidates)} total.", ln=True, align='C')
    pdf.ln(5)

    for i, c in enumerate(audit_set):
        page = c['page']
        x, y, w, h = c['x'], c['y'], c['w'], c['h']
        
        # Load Images
        w_path = WRIGHT_DIR / (page + ".png")
        a_path = ASPLEY_DIR / (page + ".png")
        
        if not w_path.exists() or not a_path.exists():
            continue
            
        img_w = cv2.imread(str(w_path))
        img_a = cv2.imread(str(a_path))
        
        # Crop context (larger than anomaly)
        pad = 50
        y1 = max(0, y - pad)
        y2 = min(img_w.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img_w.shape[1], x + w + pad)
        
        crop_w = img_w[y1:y2, x1:x2].copy()
        crop_a = img_a[y1:y2, x1:x2].copy()
        
        # Draw box on W
        # Local coords
        lx, ly = x - x1, y - y1
        cv2.rectangle(crop_w, (lx, ly), (lx+w, ly+h), (0, 0, 255), 2)
        
        # Save temp
        t_w = IMG_DIR / f"audit_{i}_wright.jpg"
        t_a = IMG_DIR / f"audit_{i}_aspley.jpg"
        cv2.imwrite(str(t_w), crop_w)
        cv2.imwrite(str(t_a), crop_a)
        
        # Add to PDF
        # Check space
        if pdf.get_y() > 250:
            pdf.add_page()

        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 8, f"Candidate #{i+1}: {page} (Area: {c['area']:.0f})", ln=True)
        
        # Image layout
        curr_y = pdf.get_y()
        # Scale to max width 80
        pdf.image(str(t_w), x=15, y=curr_y, w=80)
        pdf.image(str(t_a), x=105, y=curr_y, w=80)
        
        # Move cursor down based on image height aspect
        aspect = crop_w.shape[0] / crop_w.shape[1]
        h_pdf = 80 * aspect
        pdf.set_y(curr_y + h_pdf + 5)
        
        pdf.set_font("Arial", '', 8)
        pdf.cell(90, 5, "Wright (Marked)", align='C')
        pdf.cell(90, 5, "Aspley (Reference)", align='C')
        pdf.ln(10)

    out_file = Path("reports/ink_dot_analysis_v3/Restoration_Audit.pdf")
    pdf.output(str(out_file))
    print(f"Audit PDF saved to {out_file}")

if __name__ == "__main__":
    audit_candidates()
