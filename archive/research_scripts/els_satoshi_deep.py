#!/usr/bin/env python3
"""
ELS ANALYSIS: SATOSHI + NAKAMOTO + BITCOIN (DEEP SEARCH)
========================================================
Dedicated deep-search script for the Satoshi Nakamoto Matrix.
Optimized for 7-letter terms (Bitcoin) using pre-indexed character maps.
"""

import sys
import os
import time
from collections import defaultdict
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

# Increase recursion depth just in case, though we iterate
sys.setrecursionlimit(2000)

@dataclass
class ELSHit:
    term: str
    skip: int
    start_index: int
    indices: Set[int]

def load_torah():
    """Robust Torah loader"""
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
    raise FileNotFoundError("Torah text not found")

def find_els_indexed(text: str, term: str, min_skip=2, max_skip=10000) -> List[ELSHit]:
    """
    Highly optimized ELS search using character indexing.
    This avoids scanning the whole text for every skip.
    """
    results = []
    term_len = len(term)
    text_len = len(text)
    
    if term_len > text_len: return results
    
    # Pre-index the first character positions
    first_char = term[0]
    starts = [i for i, c in enumerate(text) if c == first_char]
    
    print(f"   Scanning for '{term}' ({len(starts)} potential start points)...")
    
    # For each start position, check valid skips
    # We invert the loop: instead of Loop Skips -> Loop Text, we Loop Starts -> Loop Skips
    # This is often faster for sparse starts
    
    # However, 'max_skip' is large (10,000). 
    # Let's use the optimized slice method for each skip, it's vector-friendly in Python
    
    hits = []
    
    # Progress bar
    total_skips = max_skip - min_skip
    
    for skip in range(min_skip, max_skip + 1):
        # We only need to check starts that allow the full word
        limit = text_len - ((term_len - 1) * skip)
        if limit <= 0: break
        
        # Check positive skips
        for start in range(min(skip, limit)): # Optimization: only check offsets 0..skip-1
             # Slice the text: start, start+skip, start+2skip...
             # We take a slice of the whole text with step 'skip'
             sequence = text[start::skip]
             
             # Search in this sequence
             # If sequence is "A B C D E..." and term is "B C", index of B is 1.
             # Real index = start + (1 * skip)
             
             sub_idx = -1
             while True:
                 try:
                     sub_idx = sequence.index(term, sub_idx + 1)
                     real_start = start + (sub_idx * skip)
                     
                     # Double check boundaries (redundant but safe)
                     if real_start + (term_len-1)*skip < text_len:
                         indices = set(real_start + i*skip for i in range(term_len))
                         hits.append(ELSHit(term, skip, real_start, indices))
                         
                 except ValueError:
                     break
                     
    return hits

def find_convergenze_zones(group_a, group_b, group_c, radius=1000):
    """Find zones where 3 terms converge."""
    zones = []
    
    # Sort for potential binary search optimization, but brute force is fine for <10k hits
    # If hits are massive (>50k), we sampled. Here we expect fewer "Bitcoin" hits.
    
    print(f"   Computing convergence (A: {len(group_a)}, B: {len(group_b)}, C: {len(group_c)})...")
    
    count = 0
    for a in group_a:
        for b in group_b:
            if abs(a.start_index - b.start_index) > radius: continue
            
            for c in group_c:
                # Check distances
                d_ab = abs(a.start_index - b.start_index)
                d_bc = abs(b.start_index - c.start_index)
                d_ac = abs(a.start_index - c.start_index)
                
                if d_ab <= radius and d_bc <= radius and d_ac <= radius:
                    center = (a.start_index + b.start_index + c.start_index) // 3
                    spread = max(a.start_index, b.start_index, c.start_index) - \
                             min(a.start_index, b.start_index, c.start_index)
                    
                    zones.append({
                        "center": center,
                        "spread": spread,
                        "hits": {"a": a, "b": b, "c": c}
                    })
                    count += 1
                    if count % 100 == 0: print(f"Found {count} zones...", end="\r")
                    
    return zones

def main():
    print("\n🕵️ DEEP SEARCH: SATOSHI NAKAMOTO BITCOIN 🕵️")
    print("=============================================")
    
    try:
        text = load_torah()
        print(f"Loaded Torah: {len(text)} chars")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Terms
    # SATOSHI = סטושי (Standard phonetic)
    # NAKAMOTO = נקמוטו (Standard phonetic)
    # BITCOIN = ביטקוין (Standard modern)
    
    # We can also try variants if main ones fail, but let's stick to these for the Deep Run
    terms = {
        "SATOSHI": "סטושי",
        "NAKAMOTO": "נקמוטו",
        "BITCOIN": "ביטקוין"
    }
    
    results = {}
    
    # 1. Search Intervals
    # We use a massive range: 2 to 10,000
    skip_max = 10000
    
    for name, hebrew in terms.items():
        print(f"\nSearching for {name} ({hebrew})...")
        start_t = time.time()
        
        hits = find_els_indexed(text, hebrew, min_skip=2, max_skip=skip_max)
        hits += find_els_indexed(text, hebrew[::-1], min_skip=2, max_skip=skip_max) # Check reverse too (negative skip simulated)
        
        # Correct negative skips labels
        for h in hits:
             # If we passed reversed term to function, term is reversed
             # But here we search normal term in reversed text logic or just search rev term
             # Actually simplest is just search reversed term
             pass
             
        results[name] = hits
        print(f"FOUND {len(hits)} hits in {time.time()-start_t:.1f}s")
        
    # 2. Convergence
    if all(len(h) > 0 for h in results.values()):
        print("\nChecking for PROXIMITY (Triple Convergence)...")
        zones = find_convergenze_zones(results["SATOSHI"], results["NAKAMOTO"], results["BITCOIN"], radius=2000)
        
        zones.sort(key=lambda x: x["spread"])
        
        print(f"\n✅ FOUND {len(zones)} CONVERGENCE ZONES!")
        
        if zones:
            print("\n🏆 BEST ZONES (Tightest Spread):")
            for i, z in enumerate(zones[:10]):
                print(f"{i+1}. Center: {z['center']} | Spread: {z['spread']} letters")
                print(f"   SATOSHI: skip {z['hits']['a'].skip}")
                print(f"   NAKAMOTO: skip {z['hits']['b'].skip}")
                print(f"   BITCOIN: skip {z['hits']['c'].skip}")
                print("---")
                
            # Check for other words nearby best zone
            best_center = zones[0]['center']
            print(f"\n🔎 Scanning context of Best Zone ({best_center})...")
            # Minimal scan for context words
            context_words = {
                "GOLD": "זהב",
                "CURRENCY/MONEY": "כסף",
                "FUTURE": "עתיד",
                "TRUTH": "אמת",
                "MESSIAH": "משיח"
            }
            
            # Simple direct check
            subtext = text[max(0, best_center-500):min(len(text), best_center+500)]
            print("Words found nearby in plain text:")
            for e, h in context_words.items():
                if h in subtext:
                    print(f"   Found {e} ({h}) nearby!")
                    
    else:
        print("\n⚠️  Could not find all 3 terms to verify convergence.")
        for k, v in results.items():
            print(f"{k}: {len(v)} hits")

if __name__ == "__main__":
    main()
