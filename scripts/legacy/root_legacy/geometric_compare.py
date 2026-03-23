
import logging
import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw
from app.services.geometric_analysis import GeometricPoint
from app.services.ocr_factory import get_ocr_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scan_page_get_points(image_path: str) -> list[GeometricPoint]:
    """Scans a page and returns geometric points."""
    engine = get_ocr_engine("tesseract")
    
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to open image {image_path}: {e}")
        return []

    result = engine.analyze_page(img)
    
    points = []
    for char in result.characters:
        if char.confidence < 40.0:
            continue
        if char.confidence < 40.0:
            continue
        # Relaxing filter to allow punctuation, but avoiding pure whitespace/noise
        if char.character.isspace() or not char.character.isprintable(): 
            continue
            
        points.append(GeometricPoint(
            x=char.x,
            y=char.y,
            character=char.character,
            confidence=char.confidence
        ))
    return points

def derive_affine_transform(ref_points: list[GeometricPoint], target_points: list[GeometricPoint]):
    """
    Derives the optimal Affine Transform (Scale, Rotation, Translation) 
    using OpenCV's RANSAC on character centroids.
    """
    
    # 1. Bucket points by character to find correspondences
    ref_map = {}
    for p in ref_points:
        if p.character not in ref_map: ref_map[p.character] = []
        ref_map[p.character].append(p)
        
    target_map = {}
    for p in target_points:
        if p.character not in target_map: target_map[p.character] = []
        target_map[p.character].append(p)
        
    # 2. match closest points for each character type
    # (Naive matching: purely based on spatial rank? No, that assumes alignment)
    # Better: Brute force match distinctive characters first?
    # Or just dump ALL "e" to "e" matches, ALL "W" to "W" matches into the localized RANSAC stew?
    
    # RANSAC is robust to outliers, so we can feed it noisy matches.
    # Let's create pairs: for each character 'C', pair every Ref 'C' with every Target 'C' that is "roughly" in the same relative area? 
    # Actually, RANSAC can handle a lot of trash matches.
    # Let's try pairing EVERY Ref(C) with the NEAREST Target(C) in normalized space? No, we don't know the space yet.
    
    # Let's pair ALL Ref(C) with ALL Target(C) only if the counts are small (unique chars).
    # If counts are large (like 'e'), just skip for alignment?
    # Let's use "Rare" characters for alignment: Q, Z, J, X, K, maybe Capitals?
    
    src_pts = [] # Target (Aspley)
    dst_pts = [] # Ref (Wright)
    
    match_count = 0
    
    unique_chars = set(ref_map.keys()) & set(target_map.keys())
    
    for char in unique_chars:
        r_list = ref_map[char]
        t_list = target_map[char]
        
        # If too many instances, it's ambiguous. Skip 'e', 't', etc.
        # Arbitrary threshold: only use characters appearing < 10 times?
        if len(r_list) > 20 or len(t_list) > 20: 
            continue
            
        # For small groups, do a cross-product? Or just 1-to-1 if len == 1?
        if len(r_list) == 1 and len(t_list) == 1:
             src_pts.append([t_list[0].x, t_list[0].y])
             dst_pts.append([r_list[0].x, r_list[0].y])
             match_count += 1
        elif len(r_list) < 5 and len(t_list) < 5:
            # Cross product limited
            for r in r_list:
                for t in t_list:
                    src_pts.append([t.x, t.y])
                    dst_pts.append([r.x, r.y])
                    
    logger.info(f"RANSAC Input: {len(src_pts)} pairs from rare characters")
    
    if len(src_pts) < 3:
        logger.warning("Not enough points for RANSAC")
        return None
        
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    
    # Estimate Affine Transform (Allows Rotation/Scale/Translation)
    # estimateAffinePartial2D restricts to Rotation + Uniform Scale + Translation (no skew/shear)
    # reliable = True creates a refined model
    transform_matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    
    if transform_matrix is None:
        logger.error("RANSAC failed to find a transform")
        return None
        
    # Extract scale from the matrix
    # Matrix is [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
    s_cos = transform_matrix[0,0]
    s_sin = transform_matrix[1,0]
    scale = np.sqrt(s_cos**2 + s_sin**2)
    rotation = np.arctan2(s_sin, s_cos)
    tx = transform_matrix[0,2]
    ty = transform_matrix[1,2]
    
    logger.info(f"RANSAC Result: Inliers={np.sum(inliers)}/{len(src_pts)}")
    logger.info(f"Transformation: Scale={scale:.4f}, Rotation={np.degrees(rotation):.2f}deg, Offset=({tx:.1f}, {ty:.1f})")
    
    return transform_matrix

def split_pages(points: list[GeometricPoint], image_width: int) -> tuple[list[GeometricPoint], list[GeometricPoint]]:
    """Splits points into Left (Verso) and Right (Recto) pages based on X coordinate."""
    midpoint = image_width / 2
    left_page = []
    right_page = []
    
    for p in points:
        if p.x < midpoint:
            left_page.append(p)
        else:
            right_page.append(p)
            
    return left_page, right_page

def align_single_page(ref_points, target_points, page_name, output_image_path, base_image_path):
    logger.info(f"--- Aligning {page_name} ---")
    logger.info(f"Points: Ref={len(ref_points)}, Target={len(target_points)}")
    
    if len(ref_points) < 10 or len(target_points) < 10:
        logger.warning(f"Not enough points for {page_name}")
        return

    # Derive Transform
    M = derive_affine_transform(ref_points, target_points)
    
    if M is None:
        logger.error(f"Could not align {page_name}")
        return
        
    # Apply Transform
    src_coords = np.array([[p.x, p.y] for p in target_points], dtype=np.float32)
    transformed_coords = cv2.transform(np.array([src_coords]), M)[0]
    
    # Draw Overlay
    try:
        if Path(base_image_path).exists():
            img = Image.open(base_image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            # 1. Draw Reference (Wright) points as Green Crosses
            # These are already in the coordinate space of the base image
            for p in ref_points:
                x, y = p.x, p.y
                r = 2
                # Draw '+'
                draw.line((x-r, y, x+r, y), fill="lime", width=1)
                draw.line((x, y-r, x, y+r), fill="lime", width=1)
            
            # 2. Draw Transformed Target (Aspley) points as Red Circles
            for i in range(len(transformed_coords)):
                x, y = transformed_coords[i]
                r = 3
                # Draw 'o'
                draw.ellipse((x-r, y-r, x+r, y+r), outline="red", width=2)
                
            img.save(output_image_path)
            logger.info(f"Saved {output_image_path}")
    except Exception as e:
        logger.error(f"Visualization error: {e}")

def main():
    # Exhaustive Batch Processing
    # Pages to test: 011 to 015
    page_prefixes = ["011", "012", "013", "014", "015"]
    
    wright_dir = Path("data/sources/folger_sonnets_1609")
    aspley_dir = Path("data/sources/folger_sonnets_1609_aspley")
    
    results = []
    
    logger.info(f"Starting Exhaustive Geometric Proof on {len(page_prefixes)} spreads...")
    
    for prefix in page_prefixes:
        # Find files
        wright_files = list(wright_dir.glob(f"{prefix}_*.jpg"))
        aspley_files = list(aspley_dir.glob(f"{prefix}_*.jpg"))
        
        if not wright_files or not aspley_files:
            logger.warning(f"Could not find pair for prefix {prefix}")
            continue
            
        wright_page = str(wright_files[0])
        aspley_page = str(aspley_files[0])
        
        logger.info(f"\n=== Processing Page {prefix} ===")
        logger.info(f"Wright: {wright_files[0].name}")
        logger.info(f"Aspley: {aspley_files[0].name}")

        # Get Image dimensions for splitting
        try:
            w_img = Image.open(wright_page)
            w_width = w_img.size[0]
            
            a_img = Image.open(aspley_page)
            a_width = a_img.size[0]
        except Exception as e:
            logger.error(f"Image load error: {e}")
            continue

        wright_points = scan_page_get_points(wright_page)
        aspley_points = scan_page_get_points(aspley_page)
        
        # Split Pages
        w_left, w_right = split_pages(wright_points, w_width)
        a_left, a_right = split_pages(aspley_points, a_width)
        
        # Filter Left Page Gutter Noise
        w_left = [p for p in w_left if p.x > 50]
        
        # Run Alignments & Save Overlays
        # We process both, but for the "Proof" report we focus on stability
        
        # Recto
        align_single_page(w_right, a_right, f"Page {prefix} Recto", f"reports/overlay_{prefix}_recto.png", wright_page)
        
        # Verso
        align_single_page(w_left, a_left, f"Page {prefix} Verso", f"reports/overlay_{prefix}_verso.png", wright_page)

if __name__ == "__main__":
    main()
