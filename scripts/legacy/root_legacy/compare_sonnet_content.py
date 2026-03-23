#!/usr/bin/env python3
"""
Sonnet Content Comparison
=========================
Extracts readable text from pages to identify which Sonnets are on each page.

Usage:
    python3 compare_sonnet_content.py
"""

import logging
from character_database import CharacterDatabase

logging.basicConfig(level=logging.INFO, format='%(message)s')

def extract_text_sample(chars, max_chars=300):
    """Extract readable text from characters, sorted by reading order."""
    # Sort by y (with line grouping), then x
    chars.sort(key=lambda c: (c.y // 25, c.x))
    
    # Join characters
    text = ''.join(c.character for c in chars[:max_chars])
    
    # Add spaces at likely word boundaries (rough heuristic)
    return text


def main():
    db = CharacterDatabase('reports/characters.db')
    
    print("SONNET PAGE CONTENT COMPARISON")
    print("="*80)
    print()
    print("Goal: Identify which Sonnets are on each page in each edition")
    print()
    
    # Check pages 9-20 to see content alignment
    for page in range(9, 25):
        wright = db.get_characters_for_page('wright', page)
        aspley = db.get_characters_for_page('aspley', page)
        
        if not wright or not aspley:
            continue
        
        print(f"--- PAGE {page} ---")
        print(f"Wright ({len(wright):4d} chars): {extract_text_sample(wright)[:60]}...")
        print(f"Aspley ({len(aspley):4d} chars): {extract_text_sample(aspley)[:60]}...")
        
        # Look for Sonnet numbers (roman numerals or arabic)
        wright_nums = [c.character for c in wright if c.character.isdigit()]
        aspley_nums = [c.character for c in aspley if c.character.isdigit()]
        
        print(f"  Numbers found - Wright: {''.join(wright_nums[:10])} | Aspley: {''.join(aspley_nums[:10])}")
        print()
    
    db.close()


if __name__ == "__main__":
    main()
