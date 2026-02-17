#!/usr/bin/env python3
"""
Same-Page Sonnet Deep Comparison
=================================
For Sonnets on the same page in both editions, perform detailed character analysis.

Usage:
    python3 same_page_deep_comparison.py
"""

import json
import logging
from collections import Counter
from character_database import CharacterDatabase

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    # Load mapping
    with open("reports/full_sonnet_mapping.json") as f:
        data = json.load(f)
    
    wright = data['wright']
    aspley = data['aspley']
    
    # Find same-page Sonnets
    same_page_sonnets = []
    for num in range(1, 155):
        w_page = wright.get(str(num), {}).get('page')
        a_page = aspley.get(str(num), {}).get('page')
        
        if w_page and a_page and w_page == a_page:
            same_page_sonnets.append((num, w_page))
    
    print("SAME-PAGE SONNET DEEP COMPARISON")
    print("="*70)
    print(f"Found {len(same_page_sonnets)} Sonnets on the same page in both editions:\n")
    
    db = CharacterDatabase("reports/characters.db")
    
    all_results = []
    
    for sonnet_num, page in same_page_sonnets:
        print(f"--- SONNET {sonnet_num} (Page {page}) ---")
        
        wright_chars = db.get_characters_for_page("wright", page)
        aspley_chars = db.get_characters_for_page("aspley", page)
        
        # Character frequency comparison
        w_freq = Counter(c.character for c in wright_chars)
        a_freq = Counter(c.character for c in aspley_chars)
        
        # Find differences
        all_chars = set(w_freq.keys()) | set(a_freq.keys())
        diffs = []
        for char in all_chars:
            w_count = w_freq.get(char, 0)
            a_count = a_freq.get(char, 0)
            if w_count != a_count:
                diffs.append((char, w_count, a_count, a_count - w_count))
        
        # Sort by absolute difference
        diffs.sort(key=lambda x: abs(x[3]), reverse=True)
        
        print(f"  Wright: {len(wright_chars)} chars | Aspley: {len(aspley_chars)} chars")
        print(f"  Delta: {len(aspley_chars) - len(wright_chars):+d} chars")
        
        if diffs:
            print(f"  Top character differences:")
            for char, w, a, d in diffs[:5]:
                if char == ' ':
                    char_display = '(space)'
                elif char == '\n':
                    char_display = '(newline)'
                else:
                    char_display = f"'{char}'"
                print(f"    {char_display:<10}: Wright={w:<3}, Aspley={a:<3}, diff={d:+d}")
        else:
            print("  ✓ No character differences!")
        
        # Identify likely cause
        total_diff = len(aspley_chars) - len(wright_chars)
        if abs(total_diff) < 50:
            cause = "Near-identical (minor OCR variance)"
        elif abs(total_diff) < 150:
            cause = "Header difference (running header present/absent)"
        else:
            cause = "Significant difference (possible content mismatch)"
        
        print(f"  Likely cause: {cause}\n")
        
        all_results.append({
            'sonnet': sonnet_num,
            'page': page,
            'wright_chars': len(wright_chars),
            'aspley_chars': len(aspley_chars),
            'delta': total_diff,
            'top_diffs': diffs[:5],
            'likely_cause': cause
        })
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    identical = sum(1 for r in all_results if abs(r['delta']) < 50)
    header_diff = sum(1 for r in all_results if 50 <= abs(r['delta']) < 150)
    significant = sum(1 for r in all_results if abs(r['delta']) >= 150)
    
    print(f"Nearly identical (diff < 50): {identical}")
    print(f"Header difference (50-150): {header_diff}")
    print(f"Significant difference (150+): {significant}")
    
    # Save results
    with open("reports/same_page_comparison.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: reports/same_page_comparison.json")
    
    db.close()


if __name__ == "__main__":
    main()
