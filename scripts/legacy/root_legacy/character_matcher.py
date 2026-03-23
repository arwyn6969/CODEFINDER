#!/usr/bin/env python3
"""
Character Matcher Module
========================
Exhaustive per-character matching between Wright and Aspley editions.
Uses KD-Tree for efficient nearest-neighbor lookup after RANSAC alignment.

Usage:
    python3 character_matcher.py
"""

import logging
import json
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from character_database import CharacterDatabase, CharacterRecord


class CharacterMatcher:
    """
    Matches characters between Wright and Aspley editions.
    
    Workflow:
    1. Load characters for a page from both editions
    2. Derive RANSAC alignment from rare characters
    3. Apply transform to Aspley coordinates
    4. Use KD-Tree to find nearest Wright neighbor for each Aspley char
    5. Record matches and residual distances
    """
    
    MAX_MATCH_DISTANCE = 15.0  # pixels - max distance to consider a match
    
    def __init__(self, db: CharacterDatabase):
        self.db = db
    
    def derive_ransac_transform(
        self, 
        wright_chars: List[CharacterRecord], 
        aspley_chars: List[CharacterRecord],
        min_inliers: int = 10
    ) -> Optional[np.ndarray]:
        """
        Derive RANSAC affine transform from rare characters.
        
        Uses characters with low frequency as they're more reliable for matching.
        """
        # Build frequency map
        wright_freq = {}
        for c in wright_chars:
            wright_freq[c.character] = wright_freq.get(c.character, 0) + 1
        
        aspley_freq = {}
        for c in aspley_chars:
            aspley_freq[c.character] = aspley_freq.get(c.character, 0) + 1
        
        # Find rare characters (frequency < 5) that exist in both
        rare_chars = set()
        for char, count in wright_freq.items():
            if count < 5 and char in aspley_freq and aspley_freq[char] < 5:
                rare_chars.add(char)
        
        if len(rare_chars) < 3:
            # Fallback to less rare characters
            for char, count in wright_freq.items():
                if count < 15 and char in aspley_freq:
                    rare_chars.add(char)
        
        # Build point sets from rare characters
        wright_points = []
        aspley_points = []
        
        for char in rare_chars:
            w_instances = [c for c in wright_chars if c.character == char]
            a_instances = [c for c in aspley_chars if c.character == char]
            
            # Match by position (closest pairs)
            for w in w_instances:
                for a in a_instances:
                    wright_points.append([w.x, w.y])
                    aspley_points.append([a.x, a.y])
        
        if len(wright_points) < 4:
            logger.warning("Not enough rare character matches for RANSAC")
            return None
        
        wright_pts = np.array(wright_points, dtype=np.float32)
        aspley_pts = np.array(aspley_points, dtype=np.float32)
        
        # Estimate affine transform with RANSAC
        try:
            M, inliers = cv2.estimateAffinePartial2D(
                aspley_pts, wright_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            
            if M is None:
                logger.warning("RANSAC failed to find transform")
                return None
            
            inlier_count = np.sum(inliers) if inliers is not None else 0
            logger.info(f"RANSAC: {inlier_count} inliers from {len(wright_points)} candidates")
            
            if inlier_count < min_inliers:
                logger.warning(f"Only {inlier_count} inliers, below threshold {min_inliers}")
            
            return M
            
        except Exception as e:
            logger.error(f"RANSAC error: {e}")
            return None
    
    def transform_points(self, points: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Apply affine transform to points."""
        return cv2.transform(np.array([points]), M)[0]
    
    def match_page(self, page: int) -> dict:
        """
        Match all characters on a specific page between editions.
        
        Returns:
            dict with match statistics
        """
        logger.info(f"Matching page {page}...")
        
        # Get characters
        wright_chars = self.db.get_characters_for_page("wright", page)
        aspley_chars = self.db.get_characters_for_page("aspley", page)
        
        if not wright_chars or not aspley_chars:
            logger.warning(f"Page {page}: Missing data (W={len(wright_chars)}, A={len(aspley_chars)})")
            return {'page': page, 'status': 'missing_data', 'matches': 0}
        
        # Derive transform
        M = self.derive_ransac_transform(wright_chars, aspley_chars)
        
        if M is None:
            return {'page': page, 'status': 'ransac_failed', 'matches': 0}
        
        # Transform Aspley coordinates to Wright space
        aspley_coords = np.array([[c.x, c.y] for c in aspley_chars], dtype=np.float32)
        transformed_aspley = self.transform_points(aspley_coords, M)
        
        # Build KD-Tree from Wright points
        wright_coords = np.array([[c.x, c.y] for c in wright_chars])
        tree = cKDTree(wright_coords)
        
        # Find matches
        matches = 0
        unmatched = 0
        residuals = []
        matched_wright_ids = set()
        
        for i, (tx, ty) in enumerate(transformed_aspley):
            distance, idx = tree.query([tx, ty])
            
            if distance <= self.MAX_MATCH_DISTANCE:
                # Check if this Wright char is already matched
                wright_id = wright_chars[idx].id
                if wright_id in matched_wright_ids:
                    # Already matched - find next closest
                    k = min(5, len(wright_chars))  # Don't query more than available
                    distances, indices = tree.query([tx, ty], k=k)
                    found_alt = False
                    for d, j in zip(distances, indices):
                        if j < len(wright_chars) and wright_chars[j].id not in matched_wright_ids and d <= self.MAX_MATCH_DISTANCE:
                            wright_id = wright_chars[j].id
                            distance = d
                            idx = j
                            found_alt = True
                            break
                    if not found_alt:
                        unmatched += 1
                        continue
                
                matched_wright_ids.add(wright_id)
                
                # Record match
                self.db.update_match(aspley_chars[i].id, wright_id, distance)
                self.db.update_match(wright_id, aspley_chars[i].id, distance)
                
                matches += 1
                residuals.append(distance)
            else:
                unmatched += 1
        
        self.db.commit()
        
        # Calculate stats
        mean_residual = np.mean(residuals) if residuals else 0
        max_residual = np.max(residuals) if residuals else 0
        
        result = {
            'page': page,
            'status': 'matched',
            'wright_count': len(wright_chars),
            'aspley_count': len(aspley_chars),
            'matches': matches,
            'unmatched': unmatched,
            'mean_residual': float(mean_residual),
            'max_residual': float(max_residual),
            'coverage': matches / len(aspley_chars) * 100 if aspley_chars else 0
        }
        
        logger.info(f"  Page {page}: {matches} matches, mean={mean_residual:.2f}px, max={max_residual:.2f}px")
        
        return result
    
    def match_all_pages(self) -> List[dict]:
        """Match all comparable pages between editions."""
        wright_pages = set(self.db.get_page_numbers("wright"))
        aspley_pages = set(self.db.get_page_numbers("aspley"))
        
        # Find common pages (by number)
        common_pages = sorted(wright_pages & aspley_pages)
        
        logger.info(f"Found {len(common_pages)} common pages to match")
        
        results = []
        for page in common_pages:
            result = self.match_page(page)
            results.append(result)
        
        return results


def main():
    db = CharacterDatabase("reports/characters.db")
    matcher = CharacterMatcher(db)
    
    # Match all pages
    results = matcher.match_all_pages()
    
    # Calculate aggregate stats
    total_matches = sum(r.get('matches', 0) for r in results)
    total_unmatched = sum(r.get('unmatched', 0) for r in results)
    all_residuals = [r.get('mean_residual', 0) for r in results if r.get('matches', 0) > 0]
    
    summary = {
        'total_pages_matched': len([r for r in results if r.get('status') == 'matched']),
        'total_character_matches': total_matches,
        'total_unmatched': total_unmatched,
        'mean_residual': float(np.mean(all_residuals)) if all_residuals else 0,
        'max_residual': float(max(r.get('max_residual', 0) for r in results)),
        'per_page_results': results
    }
    
    # Save results
    output_path = Path("reports/match_results.json")
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("MATCHING COMPLETE")
    logger.info("="*60)
    logger.info(f"Pages matched: {summary['total_pages_matched']}")
    logger.info(f"Total character matches: {total_matches:,}")
    logger.info(f"Unmatched characters: {total_unmatched:,}")
    logger.info(f"Mean residual: {summary['mean_residual']:.2f} pixels")
    logger.info(f"Max residual: {summary['max_residual']:.2f} pixels")
    logger.info(f"Results saved to: {output_path}")
    
    db.close()


if __name__ == "__main__":
    main()
