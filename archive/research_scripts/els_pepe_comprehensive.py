#!/usr/bin/env python3
"""
COMPREHENSIVE ELS ANALYSIS: PEPE THE FROG
==========================================
Full analysis powered by CODEFINDER App Services:
1. TransliterationService (for variants)
2. GematriaEngine (for numerology)
3. ELSAnalyzer (for search)
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
from app.services.els_analyzer import ELSAnalyzer

# Torah book structure (Metadata)
TORAH_BOOKS = [
    ("Genesis / בראשית", 0, 78064),
    ("Exodus / שמות", 78064, 141064),
    ("Leviticus / ויקרא", 141064, 185413),
    ("Numbers / במדבר", 185413, 249878),
    ("Deuteronomy / דברים", 249878, 304805),
]

def get_book_pct(position: int) -> Tuple[str, float]:
    for name, start, end in TORAH_BOOKS:
        if start <= position < end:
            pct = ((position - start) / (end - start)) * 100
            return name, pct
    return "Unknown", 0

def main():
    start_time = time.time()
    
    # Initialize Services
    transliteration = TransliterationService()
    gematria = GematriaEngine()
    els_analyzer = ELSAnalyzer() # initialized with default terms, but we'll use custom ones
    
    print("\n" + "🐸" * 40)
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  COMPREHENSIVE ELS ANALYSIS: PEPE THE FROG IN TORAH  ".center(78) + "█")
    print("█" + "  (Powered by CODEFINDER Application Services)  ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # Load Torah
    torah_path = Path(__file__).parent / "app/data/torah.txt"
    try:
        with open(torah_path, 'r', encoding='utf-8') as f:
            torah_text = f.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: Torah file not found at {torah_path}")
        return

    print(f"\n📜 Torah loaded: {len(torah_text):,} Hebrew letters")
    print(f"⏱️  Analysis started: {datetime.now().strftime('%H:%M:%S')}")
    
    results_summary = {}
    
    # ========================================================================
    # SECTION 1: PEPE SEARCH (Using Transliteration)
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("🐸 SECTION 1: PEPE VARIANTS SEARCH")
    print(f"{'═' * 80}")
    
    # Get candidates from service
    pepe_candidates = transliteration.get_hebrew_candidates("PEPE")
    all_pepe_hits = []
    
    for hebrew, desc in pepe_candidates:
        print(f"   Searching: {desc} ({hebrew})...", end=" ", flush=True)
        # Use ELSAnalyzer service logic manually for control, or create instance per term
        # For simplicity here, we inspect the internal _find_term for raw list access since we are a "power script"
        hits = els_analyzer._find_term(torah_text, hebrew, 2, 500)
        
        print(f"Found {len(hits)}")
        all_pepe_hits.extend(hits)
        results_summary[f"PEPE_{hebrew}"] = len(hits)
    
    print(f"\n   🐸 Total PEPE hits: {len(all_pepe_hits)}")

    # ========================================================================
    # SECTION 2: FROG SEARCH
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("🐸 SECTION 2: FROG VARIANTS SEARCH")
    print(f"{'═' * 80}")
    
    frog_candidates = transliteration.get_hebrew_candidates("FROG")
    all_frog_hits = []
    
    for hebrew, desc in frog_candidates:
        print(f"   Searching: {desc} ({hebrew})...", end=" ", flush=True)
        hits = els_analyzer._find_term(torah_text, hebrew, 2, 500)
        print(f"Found {len(hits)}")
        all_frog_hits.extend(hits)
        results_summary[f"FROG_{hebrew}"] = len(hits)
        
    print(f"\n   🐸 Total FROG hits: {len(all_frog_hits)}")

    # ========================================================================
    # SECTION 3: GEMATRIA
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("🐸 SECTION 3: GEMATRIA ANALYSIS (via GematriaEngine)")
    print(f"{'═' * 80}")
    
    terms_to_check = []
    terms_to_check.extend(pepe_candidates)
    terms_to_check.extend(frog_candidates)
    
    print(f"\n   {'Term':<15} {'Hebrew':<10} {'Gematria':>10}")
    print(f"   {'─' * 40}")
    
    for hebrew, desc in terms_to_check:
        res = gematria.calculate_all(hebrew)
        val = res['hebrew_standard']['score']
        print(f"   {desc[:15]:<15} {hebrew:<10} {val:>10}")
        
    # ========================================================================
    # SECTION 4: CROSSING SUMMARY (Simplified)
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("🐸 SECTION 4: CROSSING ANALYSIS")
    print(f"{'═' * 80}")
    
    cross_count = 0
    # Naive O(N^2) crossing check for demo
    for p in all_pepe_hits:
        p_indices = set(range(p.start_index, p.end_index + 1, p.skip))
        for f in all_frog_hits:
            f_indices = set(range(f.start_index, f.end_index + 1, f.skip))
            if p_indices & f_indices:
                cross_count += 1
                
                # Only show top few
                if cross_count <= 3:
                     pos = min(p.start_index, f.start_index)
                     book, pct = get_book_pct(pos)
                     print(f"   ✨ Match #{cross_count} in {book}: PEPE({p.skip}) x FROG({f.skip})")
    
    print(f"\n   Total Crossings Found: {cross_count}")
        
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total analysis time: {elapsed:.1f} seconds")
    print("🐸" * 40)

if __name__ == "__main__":
    main()
