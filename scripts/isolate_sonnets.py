import json
import cv2
import numpy as np
from pathlib import Path
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAPPING_JSON = Path("reports/full_sonnet_mapping.json")
OUTPUT_DIR = Path("reports/sonnet_isolation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WRIGHT_DIR = Path("reports/scan_wright_fixed/page_images")
ASPLEY_DIR = Path("reports/scan_aspley_fixed/page_images")

# Layout Constants
# Margins to ignore
X_MARGIN_PCT = 0.15
# Approx Sonnet Height (14 lines + spacing)
# Based on typical line height ~25-30px
SONNET_HEIGHT = 500 # Generous crop

def get_image_path(edition_dir, page_num):
    # Try different formats
    # page_XXX.png or page_X.png
    # Assuming page number matches filename index? 
    # MAPPING USES INT page number.
    # FILENAMES are page_001.png etc.
    # BUT page 9 in mapping might be page_010.png if blank pages exist?
    # Let's assume direct mapping for now: page 9 -> page_009.png
    return edition_dir / f"page_{int(page_num):03d}.png"

def isolate_sonnets():
    if not MAPPING_JSON.exists():
        logger.error(f"{MAPPING_JSON} not found.")
        return

    with open(MAPPING_JSON, 'r') as f:
        data = json.load(f)
        
    wright_map = data.get('wright', {})
    aspley_map = data.get('aspley', {})
    

    # Define interpolation logic
    def interpolate_missing(map_data):
        # Convert to list of (id, info) sorted by id
        # Filled map
        filled = map_data.copy()
        
        # We need to fill 1..154
        # Multi-pass filling
        changed = True
        while changed:
            changed = False
            for i in range(1, 155):
                s_id = str(i)
                if s_id in filled:
                    continue
                
                # Look for neighbors
                prev_id = str(i - 1)
                next_id = str(i + 1)
                
                prev_info = filled.get(prev_id)
                next_info = filled.get(next_id)
                
                # Case 1: Sandwiched on same page
                if prev_info and next_info and prev_info['page'] == next_info['page']:
                    # Interpolate Y
                    new_y = (prev_info['y'] + next_info['y']) / 2
                    filled[s_id] = {'page': prev_info['page'], 'y': new_y, 'inferred': True}
                    changed = True
                    continue
                
                # Case 2: Prev is known, Next is unknown (or diff page)
                # Helper: Standard layout is 2 sonnets per page usually.
                # Y positions approx: 300 (Top), 1100 (Bottom)
                if prev_info:
                    p_page = prev_info['page']
                    p_y = prev_info['y']
                    
                    # If prev is Top, Curr is likely Bottom SAME page
                    if p_y < 800:
                        filled[s_id] = {'page': p_page, 'y': p_y + 700, 'inferred': True}
                        changed = True
                        continue
                    # If prev is Bottom, Curr is likely Top NEXT page
                    else:
                        filled[s_id] = {'page': p_page + 1, 'y': 300, 'inferred': True}
                        changed = True
                        continue
                        
        return filled

    print("Filling gaps in Wright map...")
    wright_map = interpolate_missing(wright_map)
    print("Filling gaps in Aspley map...")
    aspley_map = interpolate_missing(aspley_map)
    
    count = 0
    inferred_count = 0
    
    for sonnet_id in range(1, 155):
        s_id = str(sonnet_id)
        
        # Check if mapped (now likely mostly yes)
        if s_id not in wright_map or s_id not in aspley_map:
            continue
            
        w_info = wright_map[s_id]
        a_info = aspley_map[s_id]
        
        # Get Info
        w_page = w_info.get('page')
        a_page = a_info.get('page')
        w_y = w_info.get('y')
        a_y = a_info.get('y')
        
        # Safety for inferred
        if not w_page or not a_page:
            continue
            
        is_inferred = w_info.get('inferred', False) or a_info.get('inferred', False)

        # Load Images
        w_path = get_image_path(WRIGHT_DIR, w_page)
        a_path = get_image_path(ASPLEY_DIR, a_page)
        
        if not w_path.exists() or not a_path.exists():
            continue
            
        img_w = cv2.imread(str(w_path))
        img_a = cv2.imread(str(a_path))
        
        if img_w is None or img_a is None:
            continue
            
        # Calc Crop W
        h, w = img_w.shape[:2]
        x1 = int(w * X_MARGIN_PCT)
        x2 = int(w * (1 - X_MARGIN_PCT))
        
        # Calc Crop Y (Wright)
        y_w = int(w_y)
        y_start_w = max(0, min(h - 10, y_w - 50))
        y_end_w = max(y_start_w + 10, min(h, y_w + SONNET_HEIGHT))
        
        # Calc Crop Y (Aspley)
        y_a = int(a_y)
        y_start_a = max(0, min(img_a.shape[0] - 10, y_a - 50))
        y_end_a = max(y_start_a + 10, min(img_a.shape[0], y_a + SONNET_HEIGHT))
        
        # Crop
        crop_w = img_w[y_start_w:y_end_w, x1:x2]
        crop_a = img_a[y_start_a:y_end_a, x1:x2]
        
        if crop_w.size == 0 or crop_a.size == 0:
            logger.warning(f"Empty crop for Sonnet {s_id}. Skipping.")
            continue
        
        # Save
        # Mark inferred in filename? No, keep clean.
        validity = "INFERRED" if is_inferred else "VALID"
        
        cv2.imwrite(str(OUTPUT_DIR / f"sonnet_{int(s_id):03d}_wright.png"), crop_w)
        cv2.imwrite(str(OUTPUT_DIR / f"sonnet_{int(s_id):03d}_aspley.png"), crop_a)
        
        count += 1
        if is_inferred:
            inferred_count += 1
        
        if count % 20 == 0:
            logger.info(f"Isolated {count} Sonnets ({inferred_count} inferred)...")

    logger.info(f"Complete. Isolated {count} Sonnets ({inferred_count} inferred).")

if __name__ == "__main__":
    isolate_sonnets()
