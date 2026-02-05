#!/usr/bin/env python3
"""
ELS DISCOVERY MODE: SATOSHI + NAKAMOTO
========================================
Philosophy: "Discovering new ideas... giving context in the deepest spiritual sense."

Algorithm:
1. Find all SATOSHI and NAKAMOTO hits.
2. Identify CONVERGENCE ZONES (where they are close).
3. "X-RAY" scanning of these zones:
   - Calculate the Gematria of the surrounding text.
   - Scan for statistically significant 3-4 letter Hebrew roots.
   - Map ANY existing English/Hebrew keywords that appear spontaneously.
"""

import sys
import os
import time
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

sys.setrecursionlimit(2000)

@dataclass
class ELSHit:
    term: str
    skip: int
    start_index: int
    indices: Set[int]

# Common meaningful Hebrew roots/words to scan for in context
# This acts as our "Discovery Dictionary"
DISCOVERY_DICT = {
    "אמת": "TRUTH",
    "זהב": "GOLD",
    "כסף": "SILVER/MONEY",
    "מטבע": "COIN",
    "עתיד": "FUTURE",
    "משיח": "MESSIAH",
    "נביא": "PROPHET",
    "ידע": "KNOWLEDGE",
    "אור": "LIGHT",
    "צופן": "CODE",
    "סוד": "SECRET",
    "חכמה": "WISDOM",
    "מחשב": "COMPUTER (Calculator)",
    "רשת": "NETWORK/NET",
    "חדש": "NEW",
    "גאולה": "REDEMPTION",
    "עולם": "WORLD",
    "כלכלה": "ECONOMY",
    "בנק": "BANK",
    "חופש": "FREEDOM",
    "רוח": "SPIRIT"
}

def load_torah():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, "app/data/torah.txt"),
        os.path.join(current_dir, "data/torah.txt"),
        os.path.join(current_dir, "torah.txt"),
        os.path.join(current_dir, "../torah.txt")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read().strip()
    return ""

def find_els_indexed(text: str, term: str, min_skip=2, max_skip=2000) -> List[ELSHit]:
    """Optimized search (kept moderate for discovery speed)."""
    results = []
    term_len = len(term)
    text_len = len(text)
    if term_len > text_len: return results
    
    first_char = term[0]
    starts = [i for i, c in enumerate(text) if c == first_char]
    
    for skip in range(min_skip, max_skip + 1):
        limit = text_len - ((term_len - 1) * skip)
        if limit <= 0: break
        
        for start in range(min(skip, limit)):
             sequence = text[start::skip]
             sub_idx = -1
             while True:
                 try:
                     sub_idx = sequence.index(term, sub_idx + 1)
                     real_start = start + (sub_idx * skip)
                     if real_start + (term_len-1)*skip < text_len:
                         indices = set(real_start + i*skip for i in range(term_len))
                         results.append(ELSHit(term, skip, real_start, indices))
                 except ValueError:
                     break
    return results

def get_context_words(text: str, center: int, radius: int = 200) -> List[str]:
    """Scan the immediate surface text around a center for known meaningful words."""
    start = max(0, center - radius)
    end = min(len(text), center + radius)
    segment = text[start:end]
    
    found = []
    for hebrew, english in DISCOVERY_DICT.items():
        if hebrew in segment:
            found.append(f"{english} ({hebrew})")
    
    return found

def main():
    print("\n🔍 ELS DISCOVERY MODE: SATOSHI + NAKAMOTO 🔍")
    print("============================================")
    print("Objective: Find where they meet, then listen to what the text says.")
    
    text = load_torah()
    if not text:
        print("Torah text not found.")
        return
        
    # 1. Find the Anchor Terms
    anchors = {
        "SATOSHI": "סטושי",
        "NAKAMOTO": "נקמוטו"
    }
    
    results = {}
    
    for name, hebrew in anchors.items():
        print(f"\nLocating {name} ({hebrew})...")
        hits = find_els_indexed(text, hebrew, max_skip=5000)
        results[name] = hits
        print(f"-> Found {len(hits)} occurrences.")

    if not all(results.values()):
        print("Could not find both terms.")
        return

    # 2. Find Convergence Zones
    print("\n📍 Identifying Convergence Zones...")
    zones = []
    radius = 1000 # Search radius
    
    for s in results["SATOSHI"]:
        for n in results["NAKAMOTO"]:
            dist = abs(s.start_index - n.start_index)
            if dist <= radius:
                center = (s.start_index + n.start_index) // 2
                zones.append({
                    "center": center,
                    "spread": dist,
                    "satoshi": s,
                    "nakamoto": n
                })
                
    zones.sort(key=lambda x: x["spread"])
    
    print(f"-> Found {len(zones)} places where Satoshi meets Nakamoto.")
    
    if not zones:
        print("No close convergence found.")
        return

    # 3. THE DISCOVERY SCAN
    print("\n👁️  DISCOVERY SCAN: Reading the Context 👁️")
    print("Scanning the top 5 tightest meetings for spiritual/scientific context...\n")
    
    for i, z in enumerate(zones[:5]):
        center = z["center"]
        print(f"ZONE #{i+1}: Index {center} (Spread: {z['spread']} letters)")
        print(f"   SATOSHI skip: {z['satoshi'].skip}")
        print(f"   NAKAMOTO skip: {z['nakamoto'].skip}")
        
        # Scan context
        context_hits = get_context_words(text, center, radius=300)
        
        if context_hits:
            print(f"   ✨ AUTOMATIC DISCOVERY (Nearby Words):")
            for word in context_hits:
                print(f"      - {word}")
        else:
            print("   (No primary keywords found in surface text)")
            
        print("-" * 50)

if __name__ == "__main__":
    main()
