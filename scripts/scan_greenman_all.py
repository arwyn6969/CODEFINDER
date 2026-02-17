#!/usr/bin/env python3
"""
Cross-Source Greenman Scanner
=============================

Scans ALL acquired sources for the Greenman woodblock device by comparing
every ornament candidate against a reference crop using SIFT feature matching
AND damage fingerprint comparison.

This is the main forensic tool — it answers:
"Does the same physical woodblock appear in multiple books?"

Usage:
    python scripts/scan_greenman_all.py [--reference PATH] [--threshold 0.6]

Output:
    reports/greenman_scan/
        matches.json           — All confirmed matches with scores
        match_viz_*.jpg        — SIFT match visualizations
        damage_comparison.json — Fingerprint comparison data
        greenman_report.html   — Human-readable visual report
"""

import sys
import json
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from app.services.ornament_extractor import OrnamentExtractor
from app.services.block_fingerprinter import BlockFingerprinter, BlockFingerprint

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("greenman_scan")


class GreenmanScanner:
    """Cross-source scanner for matching a specific woodblock across books."""
    
    SIFT_GOOD_THRESHOLD = 0.7       # Lowe's ratio test threshold
    MIN_SIFT_MATCHES = 8            # Minimum good SIFT matches to consider
    FINGERPRINT_THRESHOLD = 0.55    # Minimum aggregate fingerprint similarity
    
    def __init__(self, reference_path: str):
        self.reference_img = cv2.imread(reference_path)
        if self.reference_img is None:
            raise FileNotFoundError(f"Reference image not found: {reference_path}")
        
        self.fingerprinter = BlockFingerprinter()
        self.reference_fp = self.fingerprinter.fingerprint(
            self.reference_img, block_type="device"
        )
        
        # Pre-compute SIFT for reference
        self.sift = cv2.SIFT_create()
        gray_ref = cv2.cvtColor(self.reference_img, cv2.COLOR_BGR2GRAY)
        self.ref_kp, self.ref_des = self.sift.detectAndCompute(gray_ref, None)
        
        logger.info(f"Loaded reference: {reference_path} "
                     f"({self.reference_img.shape[1]}x{self.reference_img.shape[0]}, "
                     f"{len(self.ref_kp)} SIFT keypoints)")
    
    def scan_source(self, source_name: str, image_dir: Path, 
                     output_dir: Path) -> List[Dict[str, Any]]:
        """
        Scan all pages of a source for the greenman.
        
        Returns list of match results.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all ornament candidates from cached results if available
        cached_candidates = output_dir.parent.parent / "ornaments" / source_name / "candidates.jsonl"
        
        matches = []
        
        if cached_candidates.exists():
            logger.info(f"Using cached ornament candidates from {cached_candidates}")
            matches = self._scan_from_cached_candidates(
                source_name, cached_candidates, image_dir, output_dir
            )
        else:
            logger.info(f"No cached candidates found. Running fresh extraction on {image_dir}")
            extractor = OrnamentExtractor()
            image_files = sorted(list(image_dir.glob("*.jpg")))
            
            for img_path in image_files:
                candidates = extractor.extract_from_page(img_path)
                for cand in candidates:
                    if cand.crop_image is not None:
                        result = self._evaluate_candidate(
                            cand.crop_image, source_name, 
                            img_path.name, cand.page_number,
                            cand.x, cand.y, cand.w, cand.h,
                            output_dir
                        )
                        if result:
                            matches.append(result)
        
        logger.info(f"Source {source_name}: Found {len(matches)} greenman matches")
        return matches
    
    def _scan_from_cached_candidates(self, source_name: str, 
                                      candidates_path: Path,
                                      image_dir: Path,
                                      output_dir: Path) -> List[Dict]:
        """Scan using pre-extracted ornament candidates."""
        matches = []
        ornament_images_dir = candidates_path.parent / "images"
        
        with open(candidates_path) as f:
            for line in f:
                try:
                    cand = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                
                # Load the candidate crop
                crop_path = ornament_images_dir / cand.get('filename', '')
                if not crop_path.exists():
                    continue
                
                crop = cv2.imread(str(crop_path))
                if crop is None:
                    continue
                
                # Skip full-page captures (too large, definitely not greenman)
                if cand.get('area', 0) > 3_000_000:
                    continue
                
                # Skip very tiny blobs
                if cand.get('area', 0) < 8_000:
                    continue
                
                result = self._evaluate_candidate(
                    crop, source_name,
                    Path(cand.get('source_path', '')).name,
                    cand.get('page_number', 0),
                    cand.get('x', 0), cand.get('y', 0),
                    cand.get('w', 0), cand.get('h', 0),
                    output_dir
                )
                if result:
                    matches.append(result)
        
        return matches
    
    def _evaluate_candidate(self, crop_img: np.ndarray,
                              source_name: str, filename: str,
                              page_num: int, x: int, y: int, w: int, h: int,
                              output_dir: Path) -> Dict[str, Any]:
        """
        Evaluate a single candidate against the reference using:
        1. SIFT feature matching
        2. Damage fingerprint comparison
        
        Returns match dict if significant, None otherwise.
        """
        # 1. SIFT Match
        sift_score, match_viz = self._sift_match(crop_img)
        
        # 2. Fingerprint comparison
        cand_fp = self.fingerprinter.fingerprint(crop_img, block_type="device")
        fp_scores = self.fingerprinter.compare(self.reference_fp, cand_fp)
        
        # Decision: match if either SIFT is strong OR fingerprint aggregate is high
        is_match = (sift_score >= self.MIN_SIFT_MATCHES or 
                    fp_scores.get('aggregate', 0) >= self.FINGERPRINT_THRESHOLD)
        
        if not is_match:
            return None
        
        # Save visualization
        match_id = f"{source_name}_p{page_num}_{x}_{y}"
        if match_viz is not None:
            viz_path = output_dir / f"match_viz_{match_id}.jpg"
            cv2.imwrite(str(viz_path), match_viz)
        
        # Save candidate crop
        crop_path = output_dir / f"crop_{match_id}.jpg"
        cv2.imwrite(str(crop_path), crop_img)
        
        result = {
            'source': source_name,
            'filename': filename,
            'page': page_num,
            'bbox': {'x': x, 'y': y, 'w': w, 'h': h},
            'sift_matches': sift_score,
            'fingerprint_scores': fp_scores,
            'aggregate_score': fp_scores.get('aggregate', 0),
            'match_id': match_id,
            'timestamp': datetime.now().isoformat(),
        }
        
        logger.info(f"  ✓ MATCH: {source_name} page {page_num} — "
                     f"SIFT={sift_score}, FP={fp_scores.get('aggregate', 0):.3f}")
        
        return result
    
    def _sift_match(self, candidate_img: np.ndarray) -> Tuple[int, np.ndarray]:
        """SIFT feature matching against reference."""
        if len(candidate_img.shape) == 3:
            gray = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = candidate_img
        
        kp, des = self.sift.detectAndCompute(gray, None)
        
        if des is None or self.ref_des is None:
            return 0, None
        
        # FLANN matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        if len(des) < 2 or len(self.ref_des) < 2:
            return 0, None
        
        matches = flann.knnMatch(self.ref_des, des, k=2)
        
        # Ratio test
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.SIFT_GOOD_THRESHOLD * n.distance:
                    good.append(m)
        
        # Draw matches visualization
        viz = None
        if len(good) >= self.MIN_SIFT_MATCHES:
            viz = cv2.drawMatches(
                self.reference_img, self.ref_kp,
                candidate_img, kp,
                good, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
        
        return len(good), viz


def generate_html_report(all_matches: List[Dict], output_dir: Path):
    """Generate an HTML report with all greenman matches."""
    html = """<!DOCTYPE html>
<html>
<head>
<title>Greenman Woodblock — Cross-Source Forensic Report</title>
<style>
    body { font-family: 'Georgia', serif; background: #1a1a2e; color: #e0e0e0; margin: 2em; }
    h1 { color: #c9a94e; border-bottom: 2px solid #c9a94e; padding-bottom: 0.5em; }
    h2 { color: #8b7355; margin-top: 2em; }
    .match-card {
        background: #16213e; border: 1px solid #333; border-radius: 8px;
        padding: 1.5em; margin: 1em 0; display: flex; gap: 1em;
    }
    .match-card img { max-width: 300px; border: 1px solid #555; border-radius: 4px; }
    .match-info { flex: 1; }
    .score { font-size: 1.5em; font-weight: bold; }
    .score.high { color: #4caf50; }
    .score.medium { color: #ff9800; }
    .score.low { color: #f44336; }
    .detail { color: #aaa; font-size: 0.9em; margin: 0.3em 0; }
    table { border-collapse: collapse; width: 100%; margin: 1em 0; }
    th, td { border: 1px solid #444; padding: 8px; text-align: left; }
    th { background: #0f3460; }
    .verdict { font-size: 1.3em; padding: 1em; margin: 1em 0;
               border: 2px solid; border-radius: 8px; text-align: center; }
    .verdict.positive { border-color: #4caf50; color: #4caf50; }
    .verdict.negative { border-color: #f44336; color: #f44336; }
    .verdict.inconclusive { border-color: #ff9800; color: #ff9800; }
</style>
</head>
<body>
<h1>🌿 Greenman Woodblock — Cross-Source Forensic Report</h1>
<p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
"""
    
    # Summary table
    sources = {}
    for m in all_matches:
        src = m['source']
        if src not in sources:
            sources[src] = []
        sources[src].append(m)
    
    html += "<h2>Summary</h2>"
    html += "<table><tr><th>Source</th><th>Matches Found</th><th>Best Score</th><th>Avg Score</th></tr>"
    for src, matches in sources.items():
        best = max(m['aggregate_score'] for m in matches)
        avg = sum(m['aggregate_score'] for m in matches) / len(matches)
        html += f"<tr><td>{src}</td><td>{len(matches)}</td><td>{best:.3f}</td><td>{avg:.3f}</td></tr>"
    html += "</table>"
    
    # Verdict
    if len(sources) >= 2:
        # Check if there are strong matches across multiple sources
        strong_sources = [s for s, m in sources.items() 
                         if any(x['aggregate_score'] > 0.6 for x in m)]
        if len(strong_sources) >= 2:
            html += '<div class="verdict positive">✅ SAME WOODBLOCK DETECTED across ' + \
                    f'{len(strong_sources)} sources: {", ".join(strong_sources)}' + \
                    '<br>Evidence strongly suggests common printer origin.</div>'
        else:
            html += '<div class="verdict inconclusive">⚠️ INCONCLUSIVE — Matches found but ' + \
                    'cross-source evidence insufficient for definitive attribution.</div>'
    elif len(sources) == 1:
        html += '<div class="verdict inconclusive">⚠️ Only one source scanned so far.</div>'
    else:
        html += '<div class="verdict negative">❌ No matches found.</div>'
    
    # Individual matches
    html += "<h2>Individual Matches</h2>"
    for m in sorted(all_matches, key=lambda x: x['aggregate_score'], reverse=True):
        score = m['aggregate_score']
        score_class = 'high' if score > 0.7 else ('medium' if score > 0.5 else 'low')
        
        html += f"""
        <div class="match-card">
            <div>
                <img src="crop_{m['match_id']}.jpg" alt="Candidate">
                <img src="match_viz_{m['match_id']}.jpg" alt="SIFT Match" onerror="this.style.display='none'">
            </div>
            <div class="match-info">
                <div class="score {score_class}">Score: {score:.3f}</div>
                <div class="detail"><b>Source:</b> {m['source']}</div>
                <div class="detail"><b>Page:</b> {m['page']}</div>
                <div class="detail"><b>SIFT Matches:</b> {m['sift_matches']}</div>
                <div class="detail"><b>Fingerprint:</b></div>
                <ul class="detail">
        """
        for key, val in m.get('fingerprint_scores', {}).items():
            if key != 'aggregate':
                html += f"<li>{key}: {val:.3f}</li>"
        
        html += """
                </ul>
            </div>
        </div>
        """
    
    html += "</body></html>"
    
    report_path = output_dir / "greenman_report.html"
    with open(report_path, 'w') as f:
        f.write(html)
    
    logger.info(f"HTML report saved: {report_path}")
    return report_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cross-source Greenman scanner")
    parser.add_argument("--reference", default="data/greenman_reference.jpg",
                        help="Path to reference greenman crop")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Fingerprint similarity threshold")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "reports" / "greenman_scan"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize scanner
    scanner = GreenmanScanner(args.reference)
    scanner.FINGERPRINT_THRESHOLD = args.threshold
    
    # Define sources to scan
    source_dirs = {
        'gdz_goettingen_ppn777246686': base_dir / "data/sources/gdz_ppn777246686",
        'hab_wolfenbuettel_178_1_theol_1s': base_dir / "data/sources/hab_178_1_theol_1s",
        'bsb_munich_10057380': base_dir / "data/sources/bsb_10057380",
    }
    
    all_matches = []
    
    for source_name, image_dir in source_dirs.items():
        if not image_dir.exists():
            logger.warning(f"Source directory not found: {image_dir}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Scanning: {source_name}")
        logger.info(f"{'='*60}")
        
        matches = scanner.scan_source(source_name, image_dir, output_dir)
        all_matches.extend(matches)
    
    # Save JSON results
    results_path = output_dir / "matches.json"
    with open(results_path, 'w') as f:
        json.dump(all_matches, f, indent=2, default=str)
    logger.info(f"Results saved: {results_path}")
    
    # Generate HTML report
    if all_matches:
        generate_html_report(all_matches, output_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"GREENMAN SCAN COMPLETE")
    print(f"{'='*60}")
    print(f"Total matches: {len(all_matches)}")
    sources_with_matches = set(m['source'] for m in all_matches)
    print(f"Sources with matches: {', '.join(sources_with_matches) if sources_with_matches else 'None'}")
    if all_matches:
        best = max(all_matches, key=lambda x: x['aggregate_score'])
        print(f"Best match: {best['source']} page {best['page']} "
              f"(score={best['aggregate_score']:.3f})")


if __name__ == "__main__":
    main()
