#!/usr/bin/env python3
"""
ELS Advanced Search: "WE ARE ALL KEVIN" in Torah
=================================================
- Searches for full phrase as one term
- Extended skip range (up to 5000)
- Matrix crossing detection between KEVIN and "WE ARE ALL"
"""

from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time

@dataclass
class ELSHit:
    term: str
    skip: int
    start_index: int
    end_index: int
    direction: str
    indices: Set[int]  # All letter positions used

def get_els_indices(start: int, skip: int, term_len: int) -> Set[int]:
    """Get all indices covered by an ELS hit."""
    return set(start + i * skip for i in range(term_len))

def find_els_with_indices(text: str, term: str, min_skip: int = 2, max_skip: int = 5000) -> List[ELSHit]:
    """Find all ELS occurrences with their index positions."""
    results = []
    term_len = len(term)
    text_len = len(text)
    
    if term_len > text_len:
        return results
    
    # Forward and backward skips
    skips = list(range(min_skip, max_skip + 1)) + list(range(-max_skip, -min_skip + 1))
    
    for skip in skips:
        if skip == 0:
            continue
            
        required_span = (term_len - 1) * abs(skip)
        if required_span >= text_len:
            continue
        
        for start in range(min(abs(skip), text_len)):
            sequence = text[start::skip]
            if term in sequence:
                idx = 0
                while True:
                    try:
                        found_idx = sequence.index(term, idx)
                        abs_start = start + (found_idx * skip)
                        abs_end = abs_start + (term_len - 1) * skip
                        indices = get_els_indices(abs_start, skip, term_len)
                        
                        results.append(ELSHit(
                            term=term,
                            skip=skip,
                            start_index=abs_start,
                            end_index=abs_end,
                            direction="forward" if skip > 0 else "backward",
                            indices=indices
                        ))
                        idx = found_idx + 1
                    except ValueError:
                        break
    
    return results

def find_crossings(hits_a: List[ELSHit], hits_b: List[ELSHit]) -> List[Tuple[ELSHit, ELSHit, Set[int]]]:
    """Find where two sets of ELS hits share common letter positions (crossings)."""
    crossings = []
    
    for a in hits_a:
        for b in hits_b:
            shared = a.indices & b.indices
            if shared:
                crossings.append((a, b, shared))
    
    return crossings

def main():
    print("\n" + "=" * 80)
    print("    ADVANCED ELS SEARCH: 'WE ARE ALL KEVIN' IN TORAH")
    print("    Matrix Crossings + Extended Skip Range + Full Phrase")
    print("=" * 80)
    
    # Load Torah
    torah_path = Path(__file__).parent / "app/data/torah.txt"
    
    if not torah_path.exists():
        print(f"\n❌ Torah not found at: {torah_path}")
        return
    
    with open(torah_path, 'r', encoding='utf-8') as f:
        torah_text = f.read().strip()
    
    print(f"\n📜 Torah loaded: {len(torah_text):,} Hebrew letters")
    
    # ========================================================================
    # PART 1: Search for FULL PHRASE as one term (extended skip range)
    # ========================================================================
    print(f"\n{'─' * 80}")
    print("PART 1: FULL PHRASE SEARCH (Extended Skip Range up to 5000)")
    print(f"{'─' * 80}")
    
    full_phrases = [
        "כולנוקווין",      # kulanu kevin (10 letters)
        "כלנוקווין",       # kulnu kevin (9 letters)
        "אנחנוכלקווין",    # anachnu kol kevin
        "כולנוכאוינ",      # kulanu + KEVIN letter-by-letter
        "כלנוכאוינ",       # kulnu + KEVIN letter-by-letter
    ]
    
    full_phrase_results = []
    start_time = time.time()
    
    for phrase in full_phrases:
        print(f"\n   🔍 Searching: {phrase} ({len(phrase)} letters) up to skip 5000...", end=" ", flush=True)
        results = find_els_with_indices(torah_text, phrase, min_skip=2, max_skip=5000)
        print(f"✓ Found {len(results)}")
        full_phrase_results.extend(results)
    
    print(f"\n   ⏱️  Full phrase search took: {time.time() - start_time:.1f}s")
    
    if full_phrase_results:
        print(f"\n   🎯 FULL PHRASE HITS:")
        full_phrase_results.sort(key=lambda x: abs(x.skip))
        for hit in full_phrase_results[:20]:
            print(f"      {hit.term} | skip={hit.skip:>5} | start={hit.start_index:>7}")
    else:
        print(f"\n   ❌ No full phrase hits found at skips 2-5000")
    
    # ========================================================================
    # PART 2: Find KEVIN occurrences
    # ========================================================================
    print(f"\n{'─' * 80}")
    print("PART 2: KEVIN SEARCH (for crossing analysis)")
    print(f"{'─' * 80}")
    
    kevin_terms = [
        "קווין",   # standard
        "כאוינ",   # letter-by-letter with kaf
        "קאוינ",   # letter-by-letter with qof
    ]
    
    kevin_hits = []
    for term in kevin_terms:
        print(f"   Searching: {term}...", end=" ", flush=True)
        hits = find_els_with_indices(torah_text, term, min_skip=2, max_skip=1000)
        print(f"Found {len(hits)}")
        kevin_hits.extend(hits)
    
    print(f"\n   Total KEVIN hits: {len(kevin_hits)}")
    
    # ========================================================================
    # PART 3: Find "WE ARE ALL" occurrences  
    # ========================================================================
    print(f"\n{'─' * 80}")
    print("PART 3: 'WE ARE ALL' SEARCH (for crossing analysis)")
    print(f"{'─' * 80}")
    
    weareall_terms = [
        "כולנו",   # kulanu (we are all / all of us)
        "כלנו",    # short form
        "אנחנו",   # we
    ]
    
    weareall_hits = []
    for term in weareall_terms:
        print(f"   Searching: {term}...", end=" ", flush=True)
        # Limit to skip 500 for crossing analysis (too many otherwise)
        hits = find_els_with_indices(torah_text, term, min_skip=2, max_skip=500)
        print(f"Found {len(hits)}")
        weareall_hits.extend(hits)
    
    print(f"\n   Total 'WE ARE ALL' hits: {len(weareall_hits)}")
    
    # ========================================================================
    # PART 4: MATRIX CROSSINGS
    # ========================================================================
    print(f"\n{'─' * 80}")
    print("PART 4: MATRIX CROSSINGS (KEVIN × 'WE ARE ALL')")
    print(f"{'─' * 80}")
    
    print(f"\n   Analyzing {len(kevin_hits)} KEVIN hits × {len(weareall_hits)} 'WE ARE ALL' hits...")
    
    crossings = find_crossings(kevin_hits, weareall_hits)
    
    print(f"\n   🎯 CROSSINGS FOUND: {len(crossings)}")
    
    if crossings:
        # Sort by lowest combined skip (most significant)
        crossings.sort(key=lambda x: abs(x[0].skip) + abs(x[1].skip))
        
        print(f"\n   TOP 25 MOST SIGNIFICANT CROSSINGS:")
        print(f"   {'KEVIN term':<12} {'skip':>6} {'WE ARE ALL':<10} {'skip':>6} {'shared positions':>20}")
        print("   " + "-" * 60)
        
        for kevin_hit, weareall_hit, shared in crossings[:25]:
            shared_str = ",".join(str(s) for s in sorted(shared)[:3])
            if len(shared) > 3:
                shared_str += f"... ({len(shared)} total)"
            print(f"   {kevin_hit.term:<12} {kevin_hit.skip:>6} {weareall_hit.term:<10} {weareall_hit.skip:>6} {shared_str:>20}")
        
        # Show the absolute best crossing detail
        if crossings:
            best = crossings[0]
            print(f"\n   ⭐ BEST CROSSING DETAIL:")
            print(f"      KEVIN: '{best[0].term}' at skip {best[0].skip}, start {best[0].start_index}")
            print(f"      WE ARE ALL: '{best[1].term}' at skip {best[1].skip}, start {best[1].start_index}")
            print(f"      Shared letter position(s): {sorted(best[2])}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("SEARCH COMPLETE - SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n   📊 Full phrase hits:     {len(full_phrase_results)}")
    print(f"   📊 KEVIN hits:           {len(kevin_hits)}")
    print(f"   📊 'WE ARE ALL' hits:    {len(weareall_hits)}")
    print(f"   📊 Matrix crossings:     {len(crossings)}")
    print()

if __name__ == "__main__":
    main()
