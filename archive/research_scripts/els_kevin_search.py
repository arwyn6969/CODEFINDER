#!/usr/bin/env python3
"""
ELS Search: "WE ARE ALL KEVIN" in Hebrew on Torah
==================================================
Searches for Hebrew transliteration in the Torah text.
"""

from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

# Hebrew translations/transliterations of "WE ARE ALL KEVIN"
SEARCH_TERMS = [
    # Full phrase variations
    "כולנוקווין",  # kulanu kevin (we are all kevin)
    "אנחנוכולנוקווין",  # anachnu kulanu kevin
    
    # Just "KEVIN" in Hebrew
    "קווין",  # kevin (standard transliteration)
    "קוין",   # kevin (alternate)
    "כוין",   # kevin (alternate with kaf)
    
    # Individual letters K-E-V-I-N mapped to Hebrew
    # K=כ/ק, E=א, V=ו, I=י, N=נ
    "קאוינ",  # KEVIN letter by letter
    "כאוינ",  # KEVIN with kaf
    
    # "We are all" variations
    "כלנו",   # kulanu (all of us)
    "אנחנו",  # anachnu (we)
    "כולנו",  # kulanu (full)
]

@dataclass
class ELSResult:
    term: str
    skip: int
    start_index: int
    direction: str

def find_els(text: str, term: str, min_skip: int = 2, max_skip: int = 500) -> List[ELSResult]:
    """Find all ELS occurrences of a term in text."""
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
                        results.append(ELSResult(
                            term=term,
                            skip=skip,
                            start_index=abs_start,
                            direction="forward" if skip > 0 else "backward"
                        ))
                        idx = found_idx + 1
                    except ValueError:
                        break
    
    return results

def main():
    print("\n" + "=" * 70)
    print("    ELS SEARCH: 'WE ARE ALL KEVIN' IN TORAH")
    print("    Hebrew Transliteration Search")
    print("=" * 70)
    
    # Load Torah
    torah_path = Path(__file__).parent / "app/data/torah.txt"
    
    if not torah_path.exists():
        print(f"\n❌ Torah not found at: {torah_path}")
        return
    
    with open(torah_path, 'r', encoding='utf-8') as f:
        torah_text = f.read().strip()
    
    print(f"\n📜 Torah loaded: {len(torah_text):,} Hebrew letters")
    print(f"\n🔍 Searching for {len(SEARCH_TERMS)} term(s)...\n")
    
    all_results = []
    
    for term in SEARCH_TERMS:
        print(f"   Searching: {term} ({len(term)} letters)...", end=" ")
        results = find_els(torah_text, term, min_skip=2, max_skip=500)
        print(f"Found {len(results)} occurrences")
        
        for r in results:
            all_results.append({
                "term": term,
                "skip": r.skip,
                "start": r.start_index,
                "direction": r.direction
            })
    
    # Sort by absolute skip value (lower = more significant)
    all_results.sort(key=lambda x: abs(x["skip"]))
    
    print(f"\n{'─' * 70}")
    print(f"📊 RESULTS SUMMARY")
    print(f"{'─' * 70}")
    
    if all_results:
        print(f"\n🎯 Total ELS occurrences found: {len(all_results)}\n")
        
        # Show top 30 results
        print("Top findings (sorted by skip value - lower = more significant):\n")
        print(f"{'Term':<20} {'Skip':>8} {'Start':>10} {'Direction':>12}")
        print("-" * 52)
        
        for i, r in enumerate(all_results[:30], 1):
            print(f"{r['term']:<20} {r['skip']:>8} {r['start']:>10} {r['direction']:>12}")
        
        if len(all_results) > 30:
            print(f"\n... and {len(all_results) - 30} more occurrences")
    else:
        print("\n❌ No ELS occurrences found for любой из the search terms.")
    
    print(f"\n{'=' * 70}")
    print("SEARCH COMPLETE")
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    main()
