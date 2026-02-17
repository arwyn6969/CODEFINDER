import cv2
import numpy as np
import json
from pathlib import Path
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAPPING_JSON = Path("reports/full_sonnet_mapping.json")
OUTPUT_DIR = Path("reports/sonnet_isolation_aligned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WRIGHT_DIR = Path("reports/scan_wright_fixed/page_images")
ASPLEY_DIR = Path("reports/scan_aspley_fixed/page_images")

# Sonnet Layout
SONNET_WIDTH_PCT = 0.7  # Center 70%
SONNET_HEIGHT = 900     # Fixed pixel height for extraction (generous)

def align_images(img_src, img_ref):
    """Register img_src (Aspley) to img_ref (Wright) using ORB + Homography."""
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
            return None
            
        # Extract points
        pts1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        pts2 = np.zeros((len(good_matches), 2), dtype=np.float32)
        
        for i, m in enumerate(good_matches):
            pts1[i, :] = kp1[m.queryIdx].pt
            pts2[i, :] = kp2[m.trainIdx].pt
            
        # Find Homography
        h_matrix, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        
        if h_matrix is None:
            return None
            
        # Warp src to match ref shape
        height, width = img_ref.shape[:2]
        aligned = cv2.warpPerspective(img_src, h_matrix, (width, height))
        
        return aligned
    except Exception as e:
        logger.error(f"Alignment Failed: {e}")
        return None

def get_image_path(edition_dir, page_num):
    return edition_dir / f"page_{int(page_num):03d}.png"

def run_extraction():
    if not MAPPING_JSON.exists():
        logger.error("Mapping file not found.")
        return

    with open(MAPPING_JSON, 'r') as f:
        map_data = json.load(f)
        
    wright_map = map_data.get('wright', {})
    
    # We need to process by PAGE to avoid reloading/aligning same page multiple times
    # Group sonnets by page
    page_to_sonnets = {}
    
    for s_id, info in wright_map.items():
        page = info.get('page')
        if not page: continue
        if page not in page_to_sonnets:
            page_to_sonnets[page] = []
        page_to_sonnets[page].append(s_id)
        
    # Iterate Pages
    sorted_pages = sorted(page_to_sonnets.keys())
    
    for page in sorted_pages:
        sonnets = page_to_sonnets[page]
        logger.info(f"Processing Page {page} (Sonnets: {sonnets})...")
        
        # Load Images
        w_path = get_image_path(WRIGHT_DIR, page)
        a_path = get_image_path(ASPLEY_DIR, page) # Assuming mapping says same page number
        
        if not w_path.exists() or not a_path.exists():
            logger.warning(f"Missing images for Page {page}")
            continue
            
        img_w = cv2.imread(str(w_path))
        img_a = cv2.imread(str(a_path))
        
        if img_w is None or img_a is None:
            continue
        
        # 1. REGISTER PAGE (Aspley -> Wright)
        # Note: We align Aspley TO Wright. 
        # So Wright is reference.
        aligned_a = align_images(img_a, img_w)
        
        if aligned_a is None:
            logger.warning(f"Failed to register Page {page}. Skipping sonnets {sonnets}.")
            continue
            
        # 2. EXTRACT SONNETS
        for s_id in sonnets:
            info = wright_map[s_id]
            y = int(info.get('y', 0))
            
            # Dimensions
            h, w = img_w.shape[:2]
            
            # Crop Box
            # Center X
            margin = (1.0 - SONNET_WIDTH_PCT) / 2
            x1 = int(w * margin)
            x2 = int(w * (1 - margin))
            
            # Y Range
            y1 = max(0, y - 50) # Padding
            y2 = min(h, y + SONNET_HEIGHT)
            
            if y2 <= y1:
                continue
                
            # Crop BOTH from aligned versions
            crop_w = img_w[y1:y2, x1:x2]
            crop_a = aligned_a[y1:y2, x1:x2]
            
            if crop_w.size == 0 or crop_a.size == 0:
                continue
                
            # Save
            cv2.imwrite(str(OUTPUT_DIR / f"sonnet_{int(s_id):03d}_wright.png"), crop_w)
            cv2.imwrite(str(OUTPUT_DIR / f"sonnet_{int(s_id):03d}_aspley.png"), crop_a)

    logger.info("Global Registration & Extraction Complete.")

if __name__ == "__main__":
    run_extraction()
