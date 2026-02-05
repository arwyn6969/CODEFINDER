#!/usr/bin/env python3
"""
KEVIN & ARWYN ELS ANALYSIS
==========================
Scans for Kevin and Arwyn variants, identifies crossings,
and analyzes 'coagulating' terms around these intersections.
"""

import time
import sys
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Tuple
from datetime import datetime

# Add app to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.transliteration_service import TransliterationService
from app.services.gematria_engine import GematriaEngine
from app.services.els_analyzer import ELSAnalyzer, ELSResult

# Torah book structure
TORAH_BOOKS = [
    ("Genesis / בראשית", 0, 78064),
    ("Exodus / שמות", 78064, 141064),
    ("Leviticus / ויקרא", 141064, 185413),
    ("Numbers / במדבר", 185413, 249878),
    ("Deuteronomy / דברים", 249878, 304805),
]

def get_book_ref(position: int) -> str:
    for name, start, end in TORAH_BOOKS:
        if start <= position < end:
            return name
    return "Unknown"

def main():
    start_time = time.time()
    
    # Initialize Services
    transliteration = TransliterationService()
    gematria = GematriaEngine()
    els_analyzer = ELSAnalyzer()
    
    print("\n" + "✨" * 40)
    print("✨  ELS SCAN: KEVIN & ARWYN COAGULATION  ✨".center(80))
    print("✨" * 40 + "\n")
    
    # Load Torah
    torah_path = Path(__file__).parent / "app/data/torah.txt"
    try:
        with open(torah_path, 'r', encoding='utf-8') as f:
            torah_text = f.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: Torah file not found at {torah_path}")
        return

    # Normalize Hebrew (Convert Final forms to Medial for consistency)
    def normalize_hebrew(text):
        replacements = {
            'ך': 'כ',
            'ם': 'מ',
            'ן': 'נ',
            'ף': 'פ',
            'ץ': 'צ'
        }
        for final, medial in replacements.items():
            text = text.replace(final, medial)
        return text

    original_len = len(torah_text)
    torah_text = normalize_hebrew(torah_text)
    print(f"📜 Torah Loaded & Normalized: {len(torah_text):,} letters (Original: {original_len})")
    print(f"   Debug sample: {torah_text[:50]}...")

    
    # 1. Define Search Groups
    groups = {
        "KEVIN": transliteration.get_hebrew_candidates("KEVIN"),
        "ARWYN": transliteration.get_hebrew_candidates("ARWYN"),
        "COAGULANTS": [] # Will populate below
    }
    
    # Populate Coagulants (Context terms)
    coagulant_terms = ["PEPE", "FROG", "MEME", "TRUTH", "GOD", "SATOSHI", "BITCOIN"]
    for term in coagulant_terms:
        groups["COAGULANTS"].extend(transliteration.get_hebrew_candidates(term))
    
    # 2. Run Searches
    results = {}
    
    print(f"\n🔍 PHASE 1: SEARCHING TERMS")
    
    # Helper to search a list of (hebrew, desc)
    def search_group(group_name, candidates):
        group_hits = []
        print(f"   Searching {group_name} variants: {len(candidates)} defined")
        for hebrew, desc in candidates:
            # Normalize term
            orig_hebrew = hebrew
            hebrew = normalize_hebrew(hebrew)
            
            # Using reasonable skip range 2-500
            print(f"      Scanning for: {desc} [{hebrew}] (orig: {orig_hebrew})")
            hits = els_analyzer._find_term(torah_text, hebrew, 2, 500)
            if hits:
                print(f"      -> Found {len(hits):<4} for {desc}")
                group_hits.extend(hits)
            else:
                print(f"      -> No hits for {desc}")
        return group_hits

    kevin_hits = search_group("KEVIN", groups["KEVIN"])
    arwyn_hits = search_group("ARWYN", groups["ARWYN"])
    
    print(f"\n   Total KEVIN hits: {len(kevin_hits)}")
    print(f"   Total ARWYN hits: {len(arwyn_hits)}")

    # Filter Arwyn hits: Exclude "ארנ" (Normalized from ארן)
    arwyn_hits = [h for h in arwyn_hits if h.term != normalize_hebrew("ארן")]
    print(f"   Filtered ARWYN hits (removed Short): {len(arwyn_hits)}")
    
    # SEARCH COAGULANTS (Moved from Phase 3)
    coagulant_hits = search_group("COAGULANTS (Context)", groups["COAGULANTS"])
    print(f"   Total Context hits: {len(coagulant_hits)}")

    # Filter Coagulants: Exclude "ממ", "אל"
    bad_coagulants = {normalize_hebrew("מם"), normalize_hebrew("אל")}
    
    coagulant_hits = [h for h in coagulant_hits if h.term not in bad_coagulants]
    print(f"   Filtered Context hits (removed Short MEME/GOD): {len(coagulant_hits)}")
    
    if not kevin_hits or not arwyn_hits:
        print("❌ Insufficient data for crossings after filtering.")
        return

    # 3. Find Crossings
    print(f"\n⚡ PHASE 2: CALCULATING CROSSINGS")
    
    crossings = []
    
    # Crossing logic involves finding where the 'bounding box' of the terms intersect or are close
    # We'll use a simple proximity check: do the ranges of indices overlap or come within X chars?
    
    for k in kevin_hits:
        k_indices = set(range(k.start_index, k.end_index + 1, k.skip))
        k_min = min(k.start_index, k.end_index)
        k_max = max(k.start_index, k.end_index)
        
        for a in arwyn_hits:
            a_indices = set(range(a.start_index, a.end_index + 1, a.skip))
            
            # Intersection Check (Direct shared letter)
            intersection = k_indices & a_indices
            
            # Proximity Check (Are they in the same 'neighborhood'?)
            # Let's say within 100 letters of each other in the linear text
            a_min = min(a.start_index, a.end_index)
            a_max = max(a.start_index, a.end_index)
            
            # Check overlap of ranges [min, max]
            # range1_start <= range2_end and range1_end >= range2_start
            # Expand ranges by padding to find 'loose' crossings
            padding = 50
            start_overlap = max(k_min - padding, a_min - padding)
            end_overlap = min(k_max + padding, a_max + padding)
            
            is_close = start_overlap <= end_overlap
            
            if intersection or is_close:
                crossings.append({
                    "kevin": k,
                    "arwyn": a,
                    "intersection": intersection,
                    "center": (k_min + a_min) // 2
                })

    print(f"   Found {len(crossings)} potential crossings/proximities.")
    
    # 4. Analyze Coagulation
    print(f"\n🧪 PHASE 3: COAGULATION ANALYSIS")
    
    # Coagulant hits already computed and filtered in Phase 1
    print(f"   Using {len(coagulant_hits)} filtered context hits for analysis.")
    
    # Analyze best crossings
    # Sort by 'tightness' (intersection is best, then distance)
    
    def score_crossing(c):
        if c['intersection']: return 1000 # High score for direct intersection
        # Inverse distance
        k = c['kevin']
        a = c['arwyn']
        dist = abs(k.start_index - a.start_index)
        return 500 - dist # approximate
        
    crossings.sort(key=score_crossing, reverse=True)
    
    top_crossings = crossings[:10] # Top 10
    
    for i, cross in enumerate(top_crossings):
        k = cross['kevin']
        a = cross['arwyn']
        center = cross['center']
        book = get_book_ref(center)
        
        print(f"\n   📍 CROSSING #{i+1} in {book}")
        print(f"      KEVIN: {k.text_segment} (Skip {k.skip}) @ {k.start_index}")
        print(f"      ARWYN: {a.text_segment} (Skip {a.skip}) @ {a.start_index}")
        
        if cross['intersection']:
            print(f"      🔗 DIRECT INTERSECTION at indices: {cross['intersection']}")
        
        # Check for Coagulants in window +/- 200 chars
        window_start = center - 200
        window_end = center + 200
        
        found_coagulants = []
        for c in coagulant_hits:
            # Check if ANY part of the term falls in window
            c_min = min(c.start_index, c.end_index)
            c_max = max(c.start_index, c.end_index)
            
            if (c_min >= window_start and c_min <= window_end) or \
               (c_max >= window_start and c_max <= window_end):
                   found_coagulants.append(c)
        
        if found_coagulants:
            # Group by term base name (e.g. PEPE, FROG, GOD)
            from collections import Counter
            term_counts = Counter()
            term_details = defaultdict(list)
            
            # Map valid Hebrew to English label (reverse lookup tricky, we'll try desc)
            # Since we don't have the desc easily attached to the hit object (it just has 'term' which is Hebrew),
            # we rely on the fact we printed candidates earlier? No.
            # We can just use the Hebrew term itself as key.
            
            for fc in found_coagulants:
                term_counts[fc.term] += 1
                term_details[fc.term].append(fc)
                
            print("      🦠 COAGULATING TERMS:")
            for term, count in term_counts.most_common():
                print(f"         - {term} (x{count})")
        else:
            print("      (No major coagulants in immediate vicinity)")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Done in {elapsed:.1f}s")

if __name__ == "__main__":
    main()
