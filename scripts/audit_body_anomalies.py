import json
import cv2
import random
import numpy as np
from pathlib import Path
from fpdf import FPDF

# Constants
JSON_PATH = Path("reports/ink_dot_analysis_v3/scan_v3_results.json")
IMG_DIR = Path("reports/ink_dot_analysis_v3/evidence_body")
IMG_DIR.mkdir(parents=True, exist_ok=True)
WRIGHT_DIR = Path("reports/scan_wright_fixed/page_images")
ASPLEY_DIR = Path("reports/scan_aspley_fixed/page_images")

def audit_body():
    if not JSON_PATH.exists():
        print(f"Error: {JSON_PATH} not found.")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # Filter: Body Only + Normal Page Size
    candidates = []
    
    # Cache page sizes
    page_sizes = {}
    
    for c in data:
        page = c['page']
        
        # 1. Height Check (Skip corrupted strips)
        if page not in page_sizes:
            p_path = WRIGHT_DIR / (page + ".png")
            if not p_path.exists():
                page_sizes[page] = 99999
            else:
                img = cv2.imread(str(p_path))
                if img is None:
                    page_sizes[page] = 99999
                else:
                    page_sizes[page] = img.shape[0]
        
        if page_sizes[page] > 3000:
            continue
            
        # 2. Body Check (Ignore top margin)
        if c['rel_y'] < 0.15:
            continue
            
        # 3. Area Check (Ignore noise)
        if c['area'] < 100:
            continue
            
        candidates.append(c)

    print(f"Filtered down to {len(candidates)} valid body anomalies.")
    
    # Sort by Area
    candidates.sort(key=lambda x: x['area'], reverse=True)
    
    # Take Top 20
    audit_set = candidates[:20]
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 10, "Forensic Body Audit (Top 20)", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Excluding headers and corrupted pages. {len(candidates)} candidates found.", ln=True, align='C')
    pdf.ln(5)

    for i, c in enumerate(audit_set):
        page = c['page']
        x, y, w, h = c['x'], c['y'], c['w'], c['h']
        
        w_path = WRIGHT_DIR / (page + ".png")
        a_path = ASPLEY_DIR / (page + ".png")
        
        if not w_path.exists() or not a_path.exists():
            continue
            
        img_w = cv2.imread(str(w_path))
        img_a = cv2.imread(str(a_path))
        
        # Crop context
        pad = 80
        y1 = max(0, y - pad)
        y2 = min(img_w.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img_w.shape[1], x + w + pad)
        
        crop_w = img_w[y1:y2, x1:x2].copy()
        crop_a = img_a[y1:y2, x1:x2].copy()
        
        # Draw box
        lx, ly = x - x1, y - y1
        cv2.rectangle(crop_w, (lx, ly), (lx+w, ly+h), (0, 0, 255), 2)
        
        # Save temp
        t_w = IMG_DIR / f"body_{i}_wright.jpg"
        t_a = IMG_DIR / f"body_{i}_aspley.jpg"
        cv2.imwrite(str(t_w), crop_w)
        cv2.imwrite(str(t_a), crop_a)
        
        if pdf.get_y() > 240:
            pdf.add_page()

        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 8, f"#{i+1}: {page} (Area: {c['area']:.0f})", ln=True)
        
        curr_y = pdf.get_y()
        pdf.image(str(t_w), x=15, y=curr_y, w=80)
        pdf.image(str(t_a), x=105, y=curr_y, w=80)
        
        # Height adjust
        aspect = crop_w.shape[0] / crop_w.shape[1]
        h_pdf = 80 * aspect
        pdf.set_y(curr_y + h_pdf + 5)
        
        pdf.set_font("Arial", '', 8)
        pdf.cell(90, 5, "Wright (Marked)", align='C')
        pdf.cell(90, 5, "Aspley (Reference)", align='C')
        pdf.ln(10)

    out_file = Path("reports/ink_dot_analysis_v3/Body_Anomaly_Audit.pdf")
    pdf.output(str(out_file))
    print(f"Body Audit PDF saved to {out_file}")

if __name__ == "__main__":
    audit_body()
