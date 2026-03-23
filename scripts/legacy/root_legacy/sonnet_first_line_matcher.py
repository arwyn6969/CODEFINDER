#!/usr/bin/env python3
"""
Sonnet First-Line Matcher
=========================
Uses known Sonnet first lines to identify and match Sonnets between editions.
This is more reliable than number detection due to OCR noise.

Usage:
    python3 sonnet_first_line_matcher.py
"""

import logging
import json
import re
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from character_database import CharacterDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# First 20 Sonnet first lines for matching (canonical spellings)
SONNET_FIRST_LINES = {
    1: "From fairest creatures we desire increase",
    2: "When forty winters shall besiege thy brow",
    3: "Look in thy glass and tell the face thou viewest",
    4: "Unthrifty loveliness why dost thou spend",
    5: "Those hours that with gentle work did frame",
    6: "Then let not winters ragged hand deface",
    7: "Lo in the orient when the gracious light",
    8: "Music to hear why hearst thou music sadly",
    9: "Is it for fear to wet a widows eye",
    10: "For shame deny that thou bearst love to any",
    11: "As fast as thou shalt wane so fast thou growst",
    12: "When I do count the clock that tells the time",
    13: "O that you were your self but love you are",
    14: "Not from the stars do I my judgement pluck",
    15: "When I consider every thing that grows",
    16: "But wherefore do not you a mightier way",
    17: "Who will believe my verse in time to come",
    18: "Shall I compare thee to a summers day",
    19: "Devouring Time blunt thou the lions paws",
    20: "A womans face with natures own hand painted",
}


def normalize_text(text: str) -> str:
    """Normalize text for matching - lowercase, remove punctuation, collapse spaces."""
    # Convert long-s to regular s
    text = text.replace('ſ', 's')
    # Remove non-alphanumeric
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase and collapse spaces
    text = ' '.join(text.lower().split())
    return text


def similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


class SonnetFirstLineMatcher:
    """Match Sonnets by their first lines."""
    
    def __init__(self, db: CharacterDatabase):
        self.db = db
    
    def extract_lines(self, edition: str) -> list:
        """
        Extract all text lines with position info.
        Returns: [(page, y_pos, text), ...]
        """
        pages = self.db.get_page_numbers(edition)
        all_lines = []
        
        for page in pages:
            chars = self.db.get_characters_for_page(edition, page)
            if not chars:
                continue
            
            # Group by y-position
            lines = defaultdict(list)
            for c in chars:
                line_y = c.y // 25  # 25px line height
                lines[line_y].append(c)
            
            for y_key in sorted(lines.keys()):
                line_chars = sorted(lines[y_key], key=lambda x: x.x)
                line_text = ''.join(c.character for c in line_chars)
                avg_y = sum(c.y for c in line_chars) / len(line_chars)
                all_lines.append((page, avg_y, line_text))
        
        return all_lines
    
    def find_sonnets(self, edition: str) -> dict:
        """
        Find Sonnets by matching first lines.
        Returns: {sonnet_num: {'page': int, 'y_pos': float, 'matched_text': str, 'confidence': float}}
        """
        lines = self.extract_lines(edition)
        found = {}
        
        for sonnet_num, canonical_first_line in SONNET_FIRST_LINES.items():
            normalized_canonical = normalize_text(canonical_first_line)
            
            best_match = None
            best_score = 0
            
            for page, y_pos, line_text in lines:
                # Normalize and check similarity
                normalized_line = normalize_text(line_text)
                
                # Skip very short lines
                if len(normalized_line) < 20:
                    continue
                    
                # Check if this line contains the first ~30 chars of the Sonnet
                first_words = normalized_canonical.split()[:6]  # First 6 words
                search_phrase = ' '.join(first_words)
                
                score = similarity(normalized_line[:len(search_phrase) + 10], search_phrase)
                
                if score > best_score and score > 0.5:
                    best_score = score
                    best_match = {
                        'page': page,
                        'y_pos': y_pos,
                        'matched_text': line_text[:60],
                        'confidence': score
                    }
            
            if best_match:
                found[sonnet_num] = best_match
        
        return found


def main():
    db = CharacterDatabase("reports/characters.db")
    matcher = SonnetFirstLineMatcher(db)
    
    print("SONNET FIRST-LINE MATCHING")
    print("="*70)
    print("Matching Sonnets 1-20 by their canonical first lines...")
    print()
    
    # Find Sonnets in Wright
    print("--- WRIGHT EDITION ---")
    wright_sonnets = matcher.find_sonnets("wright")
    for num in sorted(wright_sonnets.keys()):
        s = wright_sonnets[num]
        print(f"Sonnet {num:2d}: Page {s['page']:2d} (conf: {s['confidence']:.2f}) - {s['matched_text'][:50]}...")
    
    print()
    print("--- ASPLEY EDITION ---")
    aspley_sonnets = matcher.find_sonnets("aspley")
    for num in sorted(aspley_sonnets.keys()):
        s = aspley_sonnets[num]
        print(f"Sonnet {num:2d}: Page {s['page']:2d} (conf: {s['confidence']:.2f}) - {s['matched_text'][:50]}...")
    
    # Compare page locations
    print()
    print("="*70)
    print("PAGINATION COMPARISON")
    print("="*70)
    print(f"{'Sonnet':<8} {'Wright Pg':<12} {'Aspley Pg':<12} {'Difference':<12}")
    print("-"*50)
    
    for num in sorted(SONNET_FIRST_LINES.keys()):
        w_page = wright_sonnets.get(num, {}).get('page', '-')
        a_page = aspley_sonnets.get(num, {}).get('page', '-')
        
        if isinstance(w_page, int) and isinstance(a_page, int):
            diff = a_page - w_page
            diff_str = f"{diff:+d}"
        else:
            diff_str = "-"
        
        print(f"{num:<8} {str(w_page):<12} {str(a_page):<12} {diff_str:<12}")
    
    # Save results
    output = {
        'wright': {k: {'page': v['page'], 'confidence': v['confidence']} for k, v in wright_sonnets.items()},
        'aspley': {k: {'page': v['page'], 'confidence': v['confidence']} for k, v in aspley_sonnets.items()}
    }
    
    output_path = Path("reports/sonnet_first_line_matches.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    db.close()


if __name__ == "__main__":
    main()
