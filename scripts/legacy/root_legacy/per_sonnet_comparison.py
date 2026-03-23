#!/usr/bin/env python3
"""
Per-Sonnet Character Comparison
===============================
For Sonnets found on the same page in both editions, compare character-by-character.

Usage:
    python3 per_sonnet_comparison.py
"""

import logging
import json
from pathlib import Path
from collections import Counter
from character_database import CharacterDatabase

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Sonnets confirmed on same pages (from first-line matching)
# Format: sonnet_number: (wright_page, aspley_page)
MATCHED_SONNETS = {
    2: (9, 9),
    9: (11, 11),
    14: (12, 12),
}


def compare_page_characters(db: CharacterDatabase, page: int) -> dict:
    """Compare characters on a specific page between editions."""
    wright = db.get_characters_for_page("wright", page)
    aspley = db.get_characters_for_page("aspley", page)
    
    # Count characters
    w_counts = Counter(c.character for c in wright)
    a_counts = Counter(c.character for c in aspley)
    
    # Find differences
    all_chars = set(w_counts.keys()) | set(a_counts.keys())
    differences = {}
    
    for char in all_chars:
        w_count = w_counts.get(char, 0)
        a_count = a_counts.get(char, 0)
        if w_count != a_count:
            differences[char] = {
                'wright': w_count,
                'aspley': a_count,
                'diff': a_count - w_count
            }
    
    return {
        'page': page,
        'wright_total': len(wright),
        'aspley_total': len(aspley),
        'char_differences': differences
    }


def main():
    db = CharacterDatabase("reports/characters.db")
    
    print("PER-SONNET CHARACTER COMPARISON")
    print("="*70)
    print()
    print("Comparing pages where the same Sonnet appears in both editions:")
    print()
    
    all_results = []
    
    for sonnet_num, (w_page, a_page) in MATCHED_SONNETS.items():
        print(f"--- SONNET {sonnet_num} (Page {w_page} in both) ---")
        
        if w_page != a_page:
            print(f"  ⚠️ Different pages! Wright={w_page}, Aspley={a_page}")
            continue
        
        result = compare_page_characters(db, w_page)
        all_results.append(result)
        
        print(f"Wright: {result['wright_total']} chars | Aspley: {result['aspley_total']} chars")
        print(f"Total diff: {result['aspley_total'] - result['wright_total']:+d} chars")
        
        # Show significant differences
        diffs = result['char_differences']
        sig_diffs = [(c, d) for c, d in diffs.items() if abs(d['diff']) > 5]
        
        if sig_diffs:
            print("Significant character differences (|diff| > 5):")
            for char, d in sorted(sig_diffs, key=lambda x: abs(x[1]['diff']), reverse=True)[:10]:
                print(f"  '{char}': Wright={d['wright']}, Aspley={d['aspley']} (diff={d['diff']:+d})")
        else:
            print("No significant character differences!")
        
        print()
    
    # Also check the core Sonnet pages that PASSED our residual test
    print("="*70)
    print("CHECKING PAGES THAT PASSED RESIDUAL TEST")
    print("="*70)
    
    passing_pages = [11, 12, 13, 14, 15, 27, 28, 29, 31, 32]
    
    for page in passing_pages:
        result = compare_page_characters(db, page)
        
        diff_count = len([d for d in result['char_differences'].values() if abs(d['diff']) > 5])
        total_diff = result['aspley_total'] - result['wright_total']
        
        status = "✅" if abs(total_diff) < 200 else "⚠️"
        
        print(f"Page {page}: Wright={result['wright_total']}, Aspley={result['aspley_total']}, "
              f"diff={total_diff:+d} {status}")
    
    # Save results
    output_path = Path("reports/per_sonnet_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    db.close()


if __name__ == "__main__":
    main()
