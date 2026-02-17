import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def align_images(img1_path, img2_path, output_dir):
    """
    Align img1 (source/to_be_warped) to match img2 (reference).
    Using ORB feature detection and Homography.
    """
    # 1. Load images
    # We load in grayscale for feature detection
    img1 = cv2.imread(img1_path) # Wright
    img2 = cv2.imread(img2_path) # Aspley (Reference)
    
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    h, w = img2.shape[:2] # Reference dimensions
    
    # 2. Detect ORB features
    MAX_FEATURES = 5000
    orb = cv2.ORB_create(MAX_FEATURES)
    
    keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)
    
    # 3. Match features
    # Use Hamming distance for binary descriptors (ORB)
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort matches by score
    matches = list(matches)
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # Keep top X% matches
    num_good_matches = int(len(matches) * 0.15)
    matches = matches[:num_good_matches]
    
    logger.info(f"Using top {num_good_matches} matches for alignment")

    # Draw top matches (optional visual debug)
    imMatches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
    cv2.imwrite(str(Path(output_dir) / "alignment_matches.jpg"), imMatches)
    
    # 4. Find Homography
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
        
    # Find homography matrix
    # RANSAC is robust to outliers
    h_matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # 5. Warp image
    # Use the matrix to warp img1 to match img2's perspective
    img1_aligned = cv2.warpPerspective(img1, h_matrix, (w, h))

    return img1_aligned, img2

def create_comparison(img_aligned, img_reference, output_dir):
    """Generate visual comparisons: Difference map and Side-by-Side."""
    output_path = Path(output_dir)
    
    # Save aligned image
    cv2.imwrite(str(output_path / "wright_aligned_to_aspley.png"), img_aligned)
    
    # Difference Map
    # Convert to gray
    gray1 = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)
    
    # Absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Enhance contrast of difference
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    cv2.imwrite(str(output_path / "difference_aligned.png"), diff_thresh)
    
    # Color overlay (Red = Wright, Blue = Aspley)
    # Put Wright (aligned) in Red channel, Aspley in Blue channel
    overlay = np.zeros_like(img_reference)
    overlay[:,:,2] = gray1 # Red
    overlay[:,:,0] = gray2 # Blue
    # Mix green for overlap
    overlay[:,:,1] = cv2.addWeighted(gray1, 0.5, gray2, 0.5, 0)
    
    cv2.imwrite(str(output_path / "overlay_comparison.png"), overlay)
    
    # Create a focused crop around the Page 18 anomaly
    # Approximate location on the LEFT page (verso)
    # Full spread is typically 4000x3000-ish
    # Verso is left half. Anomaly is near top-left of text block.
    # We'll just crop the left page for now.
    h, w = img_reference.shape[:2]
    
    # Crop Right Page (Recto) where Anomaly 8 is (Sonnet 39)
    crop_center = w // 2
    recto_aligned = img_aligned[:, crop_center:]
    recto_ref = img_reference[:, crop_center:]
    recto_diff = diff_thresh[:, crop_center:]
    
    cv2.imwrite(str(output_path / "recto_aligned_wright.png"), recto_aligned)
    cv2.imwrite(str(output_path / "recto_ref_aspley.png"), recto_ref)
    cv2.imwrite(str(output_path / "recto_difference.png"), recto_diff)


if __name__ == "__main__":
    # Define paths
    # Using the 'fixed' scans which are the best quality we have locally
    wright_src = "reports/scan_wright_fixed/page_images/page_017.png"
    aspley_src = "reports/scan_aspley_fixed/page_images/page_017.png"
    
    out_dir = "reports/ink_dot_analysis_v2/page17_registered"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Aligning {wright_src} -> {aspley_src}...")
    
    try:
        aligned, ref = align_images(wright_src, aspley_src, out_dir)
        create_comparison(aligned, ref, out_dir)
        print(f"Success! Registered images saved to {out_dir}")
    except Exception as e:
        print(f"Error during alignment: {e}")
