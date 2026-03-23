#!/usr/bin/env python3
"""
Mismatch Investigation Script
==============================
Analyzes pages with high residuals to identify specific mismatched characters.

Usage:
    python3 investigate_mismatches.py
"""

import logging
import json
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from character_database import CharacterDatabase


def investigate_page(db: CharacterDatabase, page: int, threshold: float = 10.0) -> dict:
    """
    Investigate mismatches on a specific page.
    
    Returns dict with:
    - high_residual_chars: characters with residual > threshold
    - unmatched_wright: Wright characters without Aspley match
    - unmatched_aspley: Aspley characters without Wright match
    """
    wright_chars = db.get_characters_for_page("wright", page)
    aspley_chars = db.get_characters_for_page("aspley", page)
    
    # Find high residual matches
    high_residual = []
    for c in wright_chars:
        if c.residual_distance is not None and c.residual_distance > threshold:
            high_residual.append({
                'char': c.character,
                'position': (c.x, c.y),
                'residual': c.residual_distance
            })
    
    # Find unmatched characters
    unmatched_wright = [c for c in wright_chars if c.matched_id is None]
    unmatched_aspley = [c for c in aspley_chars if c.matched_id is None]
    
    # Count unmatched by character type
    wright_unmatched_counts = Counter(c.character for c in unmatched_wright)
    aspley_unmatched_counts = Counter(c.character for c in unmatched_aspley)
    
    return {
        'page': page,
        'wright_total': len(wright_chars),
        'aspley_total': len(aspley_chars),
        'wright_matched': len(wright_chars) - len(unmatched_wright),
        'aspley_matched': len(aspley_chars) - len(unmatched_aspley),
        'high_residual_count': len(high_residual),
        'high_residual_chars': high_residual[:10],  # Top 10
        'wright_unmatched_top': wright_unmatched_counts.most_common(10),
        'aspley_unmatched_top': aspley_unmatched_counts.most_common(10),
        'delta_count': len(wright_chars) - len(aspley_chars)
    }


def main():
    db = CharacterDatabase("reports/characters.db")
    
    # Pages with highest residuals to investigate
    problem_pages = [18, 40, 33, 26, 20, 17]
    
    print("="*70)
    print("SONNET PAGE MISMATCH INVESTIGATION")
    print("="*70)
    
    results = []
    for page in problem_pages:
        result = investigate_page(db, page)
        results.append(result)
        
        print(f"\n--- Page {page} ---")
        print(f"Wright: {result['wright_total']} chars | Aspley: {result['aspley_total']} chars")
        print(f"Delta: {result['delta_count']:+d} chars (Wright - Aspley)")
        print(f"Matched: Wright {result['wright_matched']}, Aspley {result['aspley_matched']}")
        print(f"High residual matches (>10px): {result['high_residual_count']}")
        
        print(f"\nMost common UNMATCHED in Wright:")
        for char, count in result['wright_unmatched_top'][:5]:
            print(f"  '{char}': {count}")
        
        print(f"\nMost common UNMATCHED in Aspley:")
        for char, count in result['aspley_unmatched_top'][:5]:
            print(f"  '{char}': {count}")
    
    # Save detailed results
    output_path = Path("reports/mismatch_investigation.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_path}")
    
    db.close()


if __name__ == "__main__":
    main()
