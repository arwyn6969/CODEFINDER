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
SONNET_DIR = Path("reports/sonnet_isolation")
OUTPUT_DIR = Path("reports/census")
ATLAS_DIR = OUTPUT_DIR / "diff_atlas"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ATLAS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "census_data.csv"

def align_orb(im1_gray, im2_gray):
    """Align im1 to im2 using ORB Feature Matching + Homography."""
    # ORB Detector
    orb = cv2.ORB_create(nfeatures=5000)
    
    # Detect features
    kp1, des1 = orb.detectAndCompute(im1_gray, None)
    kp2, des2 = orb.detectAndCompute(im2_gray, None)
    
    if des1 is None or des2 is None:
        return im1_gray, None
        
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Take top matches (robustness)
    good_matches = matches[:int(len(matches) * 0.5)]
    if len(good_matches) < 4:
        return im1_gray, None
        
    # Extract points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find Homography
    try:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return im1_gray, None
            
        # Warp
        h, w = im2_gray.shape
        aligned = cv2.warpPerspective(im1_gray, M, (w, h))
        return aligned, M
    except Exception as e:
        logger.warning(f"Homography failed: {e}")
        return im1_gray, None

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
                continue
                
            img_w = cv2.imread(str(w_path))
            img_a = cv2.imread(str(a_path))
            
            if img_w is None or img_a is None:
                continue
                
            # Convert to Gray
            gray_w_raw = cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY)
            gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
            
            # Resize Wright to match Aspley dimensions
            h_a, w_a = gray_a.shape
            gray_w = cv2.resize(gray_w_raw, (w_a, h_a))
            
            # 1. Align Wright to Aspley (Reference)
            # ORB is robust for warped paper
            aligned_w, matrix = align_orb(gray_w, gray_a)
            
            # If alignment failed (matrix is None), aligned_w is just gray_w
            # which is now correct size.
            
            # 2. Extract Characters from Aspley (Reference)
            # Using pytesseract image_to_boxes
            # Format: 'char x1 y1 x2 y2 page' (y from bottom!)
            h_img, w_img = gray_a.shape
            try:
                boxes = pytesseract.image_to_boxes(gray_a)
            except Exception as e:
                logger.error(f"Tesseract failed on Sonnet {s_id}: {e}")
                continue
                
            for line in boxes.splitlines():
                parts = line.split()
                if len(parts) < 6: continue
                
                char = parts[0]
                if not char.isalnum(): continue # Skip punctuation/noise for now
                
                # Coords (Tesseract y is from bottom)
                x1 = int(parts[1])
                y1_inv = int(parts[2])
                x2 = int(parts[3])
                y2_inv = int(parts[4])
                
                # Flip Y
                y1 = h_img - y2_inv
                y2 = h_img - y1_inv
                
                # Width/Height
                w_box = x2 - x1
                h_box = y2 - y1
                
                # Padding for context match
                pad = 2
                px1 = max(0, x1 - pad)
                py1 = max(0, y1 - pad)
                px2 = min(w_img, x2 + pad)
                py2 = min(h_img, y2 + pad)
                
                # Extract ROIs
                roi_a = gray_a[py1:py2, px1:px2]
                roi_w = aligned_w[py1:py2, px1:px2]
                
                if roi_a.size == 0 or roi_w.size == 0:
                    continue
                
                # 3. Compare
                # NCC (Normalized Cross Correlation)
                try:
                    res = cv2.matchTemplate(roi_w, roi_a, cv2.TM_CCOEFF_NORMED)
                    ncc = res[0][0]
                except:
                    ncc = 0
                
                # Abs Diff Mean
                diff = cv2.absdiff(roi_a, roi_w)
                diff_score = np.mean(diff)
                
                # Record
                rec = {
                    'sonnet_id': i,
                    'char': char,
                    'x': x1, 'y': y1, 'w': w_box, 'h': h_box,
                    'ncc_score': ncc,
                    'diff_score': diff_score,
                    'image_path': str(a_path)
                }
                writer.writerow(rec)
                
                # 4. Save Atlas Image if Difference is High
                # Low NCC (< 0.6) or High Diff (> 50)
                if ncc < 0.6 and w_box > 5 and h_box > 5:
                    atlas_name = f"diff_S{s_id}_{char}_{x1}_{y1}.png"
                    # Create side-by-side
                    vis = np.hstack((roi_a, roi_w))
                    # Resize for visibility?
                    vis = cv2.resize(vis, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(str(ATLAS_DIR / atlas_name), vis)
            
            logger.info(f"Processed Sonnet {s_id}")

if __name__ == "__main__":
    run_census()
