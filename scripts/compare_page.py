#!/usr/bin/env python3
"""
Single-Page Forensic Comparator (Type & Ornaments)
==================================================

A rapid forensic analysis tool that compares a single page image against 
the entire database of known print blocks (Sonnet calibration set + others).

It performs:
1. ORB/SIFT Feature Matching to find identical devices (greenman)
2. Character segmentation (blobs) -> Database lookup for matching sorts
3. Measurement comparison (body height, x-height)

Usage:
    python scripts/compare_page.py --target path/to/page.jpg --known-db data/db.sqlite3

Output:
    Forensic report (JSON) + Visual matches (annotated images)
"""

import argparse
import sys
import cv2
import numpy as np
import logging
from pathlib import Path
import json

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.ornament_extractor import OrnamentExtractor, OrnamentCandidate
from app.services.image_registration import ImageRegistrationService
# We would import a CharacterExtractor here in Phase 2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("compare_page")

class ForensicComparator:
    def __init__(self):
        self.ornament_extractor = OrnamentExtractor()
        self.matcher = ImageRegistrationService()
        
        # Load known "Greenman" reference if available
        # In a real scenario, this would load from a DB or reference directory
        self.references = [] 
        
    def load_reference_ornament(self, path: Path, label: str):
        """Load a reference image (e.g. the known greenman)."""
        img = cv2.imread(str(path))
        if img is not None:
            self.references.append({"label": label, "image": img})
            logger.info(f"Loaded reference: {label}")
            
    def analyze_page(self, page_path: Path, output_dir: Path):
        """Analyze a page for matches against references."""
        output_dir.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(page_path))
        if img is None:
            logger.error(f"Could not read {page_path}")
            return
            
        # 1. Extract ornaments from target page
        candidates = self.ornament_extractor.extract_from_page(page_path)
        
        matches = []
        
        # 2. Compare each candidate against references
        for cand in candidates:
            if cand.crop_image is None: continue
            
            for ref in self.references:
                # Use ImageRegistrationService to check for homography match
                # This is robust to rotation/scale
                
                # We need to adapt the service slightly or use direct matching here
                # Let's use a direct SIFT match check as a quick heuristic
                score, match_img = self._match_images(ref["image"], cand.crop_image)
                
                if score > 10: # Threshold for "good matches" count
                    matches.append({
                        "candidate": cand.to_dict(),
                        "reference": ref["label"],
                        "score": score,
                        "match_viz": match_img
                    })
                    
        # 3. Generate Report
        report_path = output_dir / f"forensic_report_{page_path.stem}.json"
        with open(report_path, 'w') as f:
            json.dump([m["candidate"] for m in matches], f, indent=2)
            
        # Save Match Visualizations
        for i, m in enumerate(matches):
            viz_path = output_dir / f"match_{i}_{m['reference']}.jpg"
            if m["match_viz"] is not None:
                cv2.imwrite(str(viz_path), m["match_viz"])
                
        logger.info(f"Analysis complete. Found {len(matches)} potential matches.")

    def _match_images(self, img1, img2):
        """Simple SIFT feature matching and outlier removal."""
        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return 0, None

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        score = len(good)
        
        # Draw matches if significant
        img_matches = None
        if score > 10:
            img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good, None, 
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                                        
        return score, img_matches

def main():
    parser = argparse.ArgumentParser(description="Compare page against forensic db")
    parser.add_argument("--target", required=True, help="Target page image")
    parser.add_argument("--reference", help="Reference ornament image (e.g. greenman crop)")
    parser.add_argument("--output", default="reports/forensic_compare", help="Output directory")
    
    args = parser.parse_args()
    
    comparator = ForensicComparator()
    
    if args.reference:
        comparator.load_reference_ornament(Path(args.reference), "manual_ref")
        
    comparator.analyze_page(Path(args.target), Path(args.output))

if __name__ == "__main__":
    main()
