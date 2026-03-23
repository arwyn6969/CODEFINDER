#!/usr/bin/env python3
"""
Front Matter Investigation
==========================
Analyzes the early pages of both editions to understand structural differences.

Usage:
    python3 investigate_front_matter.py
"""

import logging
from collections import defaultdict
from character_database import CharacterDatabase

logging.basicConfig(level=logging.INFO, format='%(message)s')


def extract_page_text(db: CharacterDatabase, edition: str, page: int, max_chars: int = 500) -> str:
    """Extract readable text from a page."""
    chars = db.get_characters_for_page(edition, page)
    if not chars:
        return "(empty page)"
    
    # Sort by position
    lines = defaultdict(list)
    for c in chars:
        line_y = c.y // 30
        lines[line_y].append(c)
    
    text_lines = []
    for y_key in sorted(lines.keys()):
        line_chars = sorted(lines[y_key], key=lambda x: x.x)
        text_lines.append(''.join(c.character for c in line_chars))
    
    full_text = '\n'.join(text_lines)
    return full_text[:max_chars]


def main():
    db = CharacterDatabase("reports/characters.db")
    
    print("FRONT MATTER INVESTIGATION")
    print("="*70)
    print("\nComparing early pages to understand edition structure differences\n")
    
    # Get page counts
    wright_pages = db.get_page_numbers("wright")
    aspley_pages = db.get_page_numbers("aspley")
    
    print(f"Wright edition: Pages {min(wright_pages)} to {max(wright_pages)} ({len(wright_pages)} total)")
    print(f"Aspley edition: Pages {min(aspley_pages)} to {max(aspley_pages)} ({len(aspley_pages)} total)")
    print(f"Page count difference: {len(aspley_pages) - len(wright_pages)}")
    print()
    
    # Compare first 10 pages
    print("="*70)
    print("FIRST 10 PAGES COMPARISON")
    print("="*70)
    
    for page in range(1, 11):
        print(f"\n--- PAGE {page} ---")
        
        wright_chars = db.get_characters_for_page("wright", page)
        aspley_chars = db.get_characters_for_page("aspley", page)
        
        w_count = len(wright_chars)
        a_count = len(aspley_chars)
        
        print(f"Wright: {w_count} chars | Aspley: {a_count} chars | Delta: {a_count - w_count:+d}")
        
        if w_count > 0:
            w_text = extract_page_text(db, "wright", page, 150)
            print(f"  Wright content: {w_text[:100].replace(chr(10), ' ')}...")
        
        if a_count > 0:
            a_text = extract_page_text(db, "aspley", page, 150)
            print(f"  Aspley content: {a_text[:100].replace(chr(10), ' ')}...")
    
    # Character count by page range
    print("\n" + "="*70)
    print("PAGE RANGE CHARACTER TOTALS")
    print("="*70)
    
    ranges = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 40), (41, 50)]
    print(f"\n{'Range':<12} {'Wright':<12} {'Aspley':<12} {'Difference':<12}")
    print("-"*50)
    
    for start, end in ranges:
        w_total = 0
        a_total = 0
        for p in range(start, end + 1):
            w_total += len(db.get_characters_for_page("wright", p))
            a_total += len(db.get_characters_for_page("aspley", p))
        
        print(f"{start}-{end:<10} {w_total:<12} {a_total:<12} {a_total - w_total:+d}")
    
    # Analyze what's happening
    print("\n" + "="*70)
    print("ANALYSIS")  
    print("="*70)
    
    # Count total chars in each edition
    w_total = sum(len(db.get_characters_for_page("wright", p)) for p in wright_pages)
    a_total = sum(len(db.get_characters_for_page("aspley", p)) for p in aspley_pages)
    
    print(f"\nTotal characters:")
    print(f"  Wright: {w_total:,}")
    print(f"  Aspley: {a_total:,}")
    print(f"  Difference: {a_total - w_total:+,} ({100*(a_total - w_total)/w_total:+.1f}%)")
    
    print("\nPossible explanations:")
    print("  1. Different scan quality (Aspley may have more noise detected as chars)")
    print("  2. Different page structure (Aspley has more running headers)")
    print("  3. One edition may include additional content (e.g., dedication, prefaces)")
    print("  4. OCR calibration differences between sources")
    
    db.close()


if __name__ == "__main__":
    main()
