#!/usr/bin/env python3
"""
Sonnet-Level Comparison
=======================
Extracts and compares individual Sonnets between Wright and Aspley editions.
Eliminates page-level confusion by matching Sonnets by their number.

Usage:
    python3 sonnet_level_comparison.py
"""

import logging
import json
import re
from pathlib import Path
from collections import defaultdict
from character_database import CharacterDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SonnetExtractor:
    """
    Extracts individual Sonnets from OCR character data.
    
    Strategy:
    - Look for Roman numeral or Arabic number patterns
    - Group characters between Sonnet markers
    - Build text reconstruction for each Sonnet
    """
    
    # Sonnet patterns - Roman numerals I-CLIV (1-154)
    ROMAN_PATTERN = re.compile(r'^[IVXLCivxlc]+$')
    
    def __init__(self, db: CharacterDatabase):
        self.db = db
    
    def extract_text_lines(self, edition: str) -> dict:
        """
        Extract full text organized by page and line.
        Returns: {page: [(y_pos, line_text), ...]}
        """
        pages = self.db.get_page_numbers(edition)
        result = {}
        
        for page in pages:
            chars = self.db.get_characters_for_page(edition, page)
            if not chars:
                continue
            
            # Group by y-position (line grouping)
            lines = defaultdict(list)
            for c in chars:
                line_y = c.y // 30  # 30px line height approximation
                lines[line_y].append(c)
            
            # Build line texts
            page_lines = []
            for y_key in sorted(lines.keys()):
                line_chars = sorted(lines[y_key], key=lambda x: x.x)
                line_text = ''.join(c.character for c in line_chars)
                avg_y = sum(c.y for c in line_chars) / len(line_chars)
                page_lines.append((avg_y, line_text))
            
            result[page] = page_lines
        
        return result
    
    def find_sonnet_markers(self, text_lines: dict) -> list:
        """
        Find lines that appear to be Sonnet number markers.
        Returns: [(page, y_pos, sonnet_number), ...]
        """
        markers = []
        
        for page, lines in text_lines.items():
            for y_pos, text in lines:
                # Clean text
                clean = text.strip().upper()
                
                # Check for standalone numbers (Arabic or Roman)
                # Sonnets 1-154
                
                # Arabic numbers
                if clean.isdigit():
                    num = int(clean)
                    if 1 <= num <= 154:
                        markers.append((page, y_pos, num))
                        continue
                
                # Roman numerals - look for common patterns
                if len(clean) <= 6 and self.ROMAN_PATTERN.match(clean):
                    try:
                        num = self.roman_to_int(clean)
                        if 1 <= num <= 154:
                            markers.append((page, y_pos, num))
                    except:
                        pass
        
        return sorted(markers, key=lambda x: (x[0], x[1]))
    
    def roman_to_int(self, s: str) -> int:
        """Convert Roman numeral to integer."""
        values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        s = s.upper()
        total = 0
        prev = 0
        for char in reversed(s):
            val = values.get(char, 0)
            if val < prev:
                total -= val
            else:
                total += val
            prev = val
        return total
    
    def extract_sonnets(self, edition: str) -> dict:
        """
        Extract individual Sonnets from an edition.
        Returns: {sonnet_number: {'chars': [...], 'page': int, 'text': str}}
        """
        text_lines = self.extract_text_lines(edition)
        markers = self.find_sonnet_markers(text_lines)
        
        logger.info(f"Found {len(markers)} potential Sonnet markers in {edition}")
        
        # For now, return character data grouped by page
        # (Full Sonnet segmentation would require more sophisticated parsing)
        sonnets = {}
        for page, y_pos, num in markers:
            if num not in sonnets:
                sonnets[num] = {
                    'page': page,
                    'marker_y': y_pos,
                    'edition': edition
                }
        
        return sonnets


def compare_sonnet_locations(wright_sonnets: dict, aspley_sonnets: dict) -> dict:
    """Compare where each Sonnet appears in both editions."""
    comparison = {}
    
    all_sonnets = set(wright_sonnets.keys()) | set(aspley_sonnets.keys())
    
    for num in sorted(all_sonnets):
        w_page = wright_sonnets.get(num, {}).get('page', None)
        a_page = aspley_sonnets.get(num, {}).get('page', None)
        
        comparison[num] = {
            'wright_page': w_page,
            'aspley_page': a_page,
            'page_diff': (a_page - w_page) if (w_page and a_page) else None,
            'in_both': w_page is not None and a_page is not None
        }
    
    return comparison


def main():
    db = CharacterDatabase("reports/characters.db")
    extractor = SonnetExtractor(db)
    
    print("SONNET-LEVEL COMPARISON")
    print("="*70)
    
    # Extract Sonnet markers from both editions
    print("\nExtracting Sonnets from Wright edition...")
    wright_sonnets = extractor.extract_sonnets("wright")
    print(f"  Found: {len(wright_sonnets)} Sonnets")
    
    print("\nExtracting Sonnets from Aspley edition...")
    aspley_sonnets = extractor.extract_sonnets("aspley")
    print(f"  Found: {len(aspley_sonnets)} Sonnets")
    
    # Compare locations
    print("\n" + "="*70)
    print("SONNET LOCATION COMPARISON")
    print("="*70)
    
    comparison = compare_sonnet_locations(wright_sonnets, aspley_sonnets)
    
    print(f"\n{'Sonnet':<8} {'Wright':<10} {'Aspley':<10} {'Page Diff':<10}")
    print("-"*40)
    
    pagination_diffs = []
    for num in sorted(comparison.keys()):
        data = comparison[num]
        w_page = data['wright_page'] or '-'
        a_page = data['aspley_page'] or '-'
        diff = data['page_diff']
        
        if diff is not None:
            pagination_diffs.append(diff)
            diff_str = f"{diff:+d}"
        else:
            diff_str = '-'
        
        # Only show if there's a page difference
        if diff is not None and diff != 0:
            print(f"{num:<8} {w_page:<10} {a_page:<10} {diff_str:<10}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total Sonnets found in Wright: {len(wright_sonnets)}")
    print(f"Total Sonnets found in Aspley: {len(aspley_sonnets)}")
    
    if pagination_diffs:
        from collections import Counter
        diff_counts = Counter(pagination_diffs)
        print(f"\nPage offset distribution:")
        for diff, count in sorted(diff_counts.items()):
            print(f"  Offset {diff:+d}: {count} Sonnets")
    
    # Save results
    output = {
        'wright_sonnets': {k: {'page': v['page']} for k, v in wright_sonnets.items()},
        'aspley_sonnets': {k: {'page': v['page']} for k, v in aspley_sonnets.items()},
        'comparison': comparison
    }
    
    output_path = Path("reports/sonnet_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    db.close()


if __name__ == "__main__":
    main()
