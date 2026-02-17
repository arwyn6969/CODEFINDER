import cv2
import numpy as np
from pathlib import Path
import json
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
WRIGHT_DIR = Path("reports/scan_wright_fixed/page_images")
ASPLEY_DIR = Path("reports/scan_aspley_fixed/page_images")
OUTPUT_DIR = Path("reports/ink_dot_analysis_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_OUTPUT_DIR = OUTPUT_DIR / "evidence"
IMG_OUTPUT_DIR.mkdir(exist_ok=True)

HANDWRITING_ZONE_RATIO = 0.15  # Top 15% is suspect for handwriting restoration

def align_images(img_src, img_ref):
    """Register img_src to img_ref using ORB + Homography."""
    try:
        # Convert to grayscale
        gray1 = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        
        # Max features for robustness
        orb = cv2.ORB_create(nfeatures=5000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return None

        # Match (Hamming for ORB)
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(des1, des2, None)
        
        # Sort and keep top 20%
        matches = list(matches)
        matches.sort(key=lambda x: x.distance, reverse=False)
        good_matches = matches[:int(len(matches) * 0.2)]
        
        if len(good_matches) < 10:
            logger.warning("Not enough matches")
            return None
            
        # Extract points
        pts1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        pts2 = np.zeros((len(good_matches), 2), dtype=np.float32)
        
        for i, m in enumerate(good_matches):
            pts1[i, :] = kp1[m.queryIdx].pt
            pts2[i, :] = kp2[m.trainIdx].pt
            
        # Find Homography
        h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        
        # Warp
        height, width = img_ref.shape[:2]
        aligned = cv2.warpPerspective(img_src, h, (width, height))
        
        return aligned
    except Exception as e:
        logger.error(f"Alignment Failed: {e}")
        return None

def analyze_page(filename):
    """Process a single page pair."""
    page_id = filename.stem
    w_path = WRIGHT_DIR / filename.name
    a_path = ASPLEY_DIR / filename.name
    
    if not a_path.exists():
        logger.warning(f"Skipping {page_id} - Aspley counterpart missing")
        return []
        
    logger.info(f"Processing {page_id}...")
    
    wright = cv2.imread(str(w_path))
    aspley = cv2.imread(str(a_path))
    
    if wright is None or aspley is None:
        return []

    # 1. Register
    wright_aligned = align_images(wright, aspley)
    if wright_aligned is None:
        return []
        
    # 2. Subtract
    gray_w = cv2.cvtColor(wright_aligned, cv2.COLOR_BGR2GRAY)
    gray_a = cv2.cvtColor(aspley, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray_w, gray_a)
    
    # 3. Mask Borders (Crop artifacts)
    h, w = diff.shape
    border_x = int(w * 0.05)
    border_y = int(h * 0.05)
    mask = np.zeros_like(diff)
    mask[border_y:h-border_y, border_x:w-border_x] = 255
    diff = cv2.bitwise_and(diff, diff, mask=mask)
    
    # 4. Threshold & Detection
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    
    # Clean noise (morphology)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Filter & Classify candidates
    anomalies = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20: continue # Too small noise
        if area > 5000: continue # Too big (likely alignment error/binding)
        
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        
        # Calculate centrality
        rel_y = y / h
        
        # Classification Logic
        classification = "Unknown Anomaly"
        
        # A. Handwriting Logic (Top Margin)
        if rel_y < HANDWRITING_ZONE_RATIO:
            classification = "Handwriting Restoration"
        
        # B. Binding Logic (Center gutter for spreads)
        # Assuming single pages here? Or spreads? Folder name says 'page_images'.
        # Usually spread split? If spread, binding is center.
        # But let's stick to the Handwriting logic.
        
        anomaly = {
            "page": page_id,
            "x": x, "y": y, "w": w_box, "h": h_box,
            "area": area,
            "rel_y": rel_y,
            "classification": classification
        }
        anomalies.append(anomaly)
        
        # Save snapshot for significant ones
        if area > 50:
            snapshot_name = f"{page_id}_{x}_{y}_{classification.replace(' ', '_')}.jpg"
            # Draw rect on aligned wright
            vis = wright_aligned.copy()
            color = (0, 0, 255) if "Handwriting" in classification else (0, 255, 0)
            cv2.rectangle(vis, (x, y), (x+w_box, y+h_box), color, 2)
            # Crop context
            pad = 50
            roi = vis[max(0, y-pad):min(h, y+h_box+pad), max(0, x-pad):min(w, x+w_box+pad)]
            cv2.imwrite(str(IMG_OUTPUT_DIR / snapshot_name), roi)
            
    return anomalies

def main():
    all_results = []
    
    # Process all common files
    files = sorted(list(WRIGHT_DIR.glob("*.png")))
    
    for f in files:
        results = analyze_page(f)
        all_results.extend(results)
        
    # Save Report
    with open(OUTPUT_DIR / "scan_v3_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"Scan complete. Found {len(all_results)} anomalies.")
    print(f"Results saved to {OUTPUT_DIR}/scan_v3_results.json")
    
    # Text Summary
    handwriting_count = len([x for x in all_results if x['classification'] == 'Handwriting Restoration'])
    project_summary = (
        f"Scan V3 Complete\n"
        f"Total Anomalies: {len(all_results)}\n"
        f"Handwriting Restorations Detected: {handwriting_count}\n"
    )
    with open(OUTPUT_DIR / "summary.txt", "w") as f:
        f.write(project_summary)

if __name__ == "__main__":
    main()
