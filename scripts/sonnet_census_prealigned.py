import cv2
import numpy as np
import pytesseract
import csv
from pathlib import Path
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SONNET_DIR = Path("reports/sonnet_isolation_aligned") # NEW DIR
OUTPUT_DIR = Path("reports/census_aligned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "census_data.csv"

def run_census():
    # Prepare CSV
    with open(CSV_PATH, 'w', newline='') as csvfile:
        fieldnames = ['sonnet_id', 'char', 'x', 'y', 'w', 'h', 'ncc_score', 'diff_score', 'image_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Iterate Sonnets 1-154
        for i in range(1, 155):
            s_id = f"{i:03d}"
            w_path = SONNET_DIR / f"sonnet_{s_id}_wright.png"
            a_path = SONNET_DIR / f"sonnet_{s_id}_aspley.png"
            
            if not w_path.exists() or not a_path.exists():
                # logger.warning(f"Missing {s_id}")
                continue
                
            img_w = cv2.imread(str(w_path))
            img_a = cv2.imread(str(a_path))
            
            if img_w is None or img_a is None:
                continue
                
            # Convert to Gray
            gray_w = cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY)
            gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
            
            # NO ALIGNMENT NEEDED (Already done globally)
            
            # 2. Extract Characters from Aspley (Reference)
            h_img, w_img = gray_a.shape
            try:
                boxes = pytesseract.image_to_boxes(gray_a)
            except Exception as e:
                # logger.error(f"Tesseract failed: {e}")
                continue
                
            for line in boxes.splitlines():
                parts = line.split()
                if len(parts) < 6: continue
                
                char = parts[0]
                if not char.isalnum(): continue
                
                x1 = int(parts[1])
                y1_inv = int(parts[2])
                x2 = int(parts[3])
                y2_inv = int(parts[4])
                
                y1 = h_img - y2_inv
                y2 = h_img - y1_inv
                
                w_box = x2 - x1
                h_box = y2 - y1
                
                if w_box < 4 or h_box < 4: continue
                
                # Check bounds
                if x1 < 0 or y1 < 0 or x2 > w_img or y2 > h_img:
                    continue
                
                # Extract ROIs
                roi_a = gray_a[y1:y2, x1:x2]
                roi_w = gray_w[y1:y2, x1:x2] # Direct mapping
                
                if roi_a.shape != roi_w.shape:
                    continue
                    
                # 3. Compare
                try:
                    res = cv2.matchTemplate(roi_w, roi_a, cv2.TM_CCOEFF_NORMED)
                    ncc = res[0][0]
                except:
                    ncc = 0
                
                diff = cv2.absdiff(roi_a, roi_w)
                diff_score = np.mean(diff)
                
                rec = {
                    'sonnet_id': i,
                    'char': char,
                    'x': x1, 'y': y1, 'w': w_box, 'h': h_box,
                    'ncc_score': ncc,
                    'diff_score': diff_score,
                    'image_path': str(a_path)
                }
                writer.writerow(rec)
            
            if i % 10 == 0:
                logger.info(f"Processed Sonnet {s_id}")

if __name__ == "__main__":
    run_census()
