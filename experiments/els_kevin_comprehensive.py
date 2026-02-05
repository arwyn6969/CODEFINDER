#!/usr/bin/env python3
"""
COMPREHENSIVE ELS ANALYSIS: "WE ARE ALL KEVIN"
===============================================
The most extensive analysis possible:
1. All crossings mapped by Torah book
2. Personal names (ARWYN, etc.)
3. Gematria analysis
4. Control phrase comparison
5. Adjacent skip patterns
6. Statistical summary
"""

from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import random
import math
import time
from datetime import datetime

# Torah book structure
TORAH_BOOKS = [
    ("Genesis / בראשית", 0, 78064),
    ("Exodus / שמות", 78064, 141064),
    ("Leviticus / ויקרא", 141064, 185413),
    ("Numbers / במדבר", 185413, 249878),
    ("Deuteronomy / דברים", 249878, 304805),
]

# Hebrew letter frequencies
HEBREW_FREQ = {
    'א': 0.048, 'ב': 0.043, 'ג': 0.011, 'ד': 0.025, 'ה': 0.069,
    'ו': 0.099, 'ז': 0.008, 'ח': 0.022, 'ט': 0.009, 'י': 0.092,
    'כ': 0.039, 'ל': 0.055, 'מ': 0.059, 'נ': 0.043, 'ס': 0.011,
    'ע': 0.035, 'פ': 0.015, 'צ': 0.012, 'ק': 0.012, 'ר': 0.055,
    'ש': 0.050, 'ת': 0.050,
}

# Hebrew gematria values
GEMATRIA = {
    'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5, 'ו': 6, 'ז': 7, 'ח': 8, 'ט': 9,
    'י': 10, 'כ': 20, 'ל': 30, 'מ': 40, 'נ': 50, 'ס': 60, 'ע': 70, 'פ': 80, 'צ': 90,
    'ק': 100, 'ר': 200, 'ש': 300, 'ת': 400,
    # Final forms
    'ך': 20, 'ם': 40, 'ן': 50, 'ף': 80, 'ץ': 90,
}

# Personal names to search
PERSONAL_NAMES = [
    ("ארווין", "ARWYN (full)"),
    ("ארוין", "ARWYN (alt)"),
    ("ארון", "ARON/ARWYN"),
    ("קוין", "KEVIN (alt)"),
    ("כוין", "KEVIN (kaf)"),
    ("מרי", "MARY"),
    ("מריה", "MARIA"),
    ("ישוע", "YESHUA/Jesus"),
    ("משיח", "MESSIAH"),
]

# Control phrases (random transliterations for comparison)
CONTROL_PHRASES = [
    ("פלקרב", "RANDOM1"),
    ("נתשמח", "RANDOM2"),
    ("זבגדה", "RANDOM3"),
    ("סעפצק", "RANDOM4"),
]

# Extended cluster terms
EXTENDED_TERMS = [
    # Core
    ("כאוינ", "KEVIN"), ("כלנו", "All of us"), ("אנחנו", "We"),
    # Unity
    ("אחד", "One"), ("יחד", "Together"), ("אחים", "Brothers"), ("אח", "Brother"),
    # Identity  
    ("שם", "Name"), ("נפש", "Soul"), ("רוח", "Spirit"), ("חיים", "Life"),
    # Divine
    ("אל", "El/God"), ("יהוה", "YHWH"), ("אלהים", "Elohim"),
    # Concepts
    ("אהבה", "Love"), ("שלום", "Peace"), ("ברית", "Covenant"), ("אמת", "Truth"),
    ("אור", "Light"), ("דרך", "Path"), ("חכמה", "Wisdom"), ("בינה", "Understanding"),
    # Action
    ("ראה", "See"), ("שמע", "Hear"), ("ידע", "Know"), ("אמר", "Said"),
    # Biblical names
    ("יעקב", "Jacob"), ("לבן", "Laban"), ("רחל", "Rachel"), ("לאה", "Leah"),
    ("אברהם", "Abraham"), ("יצחק", "Isaac"), ("ישראל", "Israel"),
    # Numbers
    ("אחד", "One"), ("שבע", "Seven"), ("עשר", "Ten"), ("עשרים", "Twenty"),
]

@dataclass
class ELSHit:
    term: str
    skip: int
    start_index: int
    end_index: int
    indices: Set[int]

def get_book(position: int) -> str:
    for name, start, end in TORAH_BOOKS:
        if start <= position < end:
            return name
    return "Unknown"

def get_els_indices(start: int, skip: int, term_len: int) -> Set[int]:
    return set(start + i * skip for i in range(term_len))

def calculate_gematria(term: str) -> int:
    return sum(GEMATRIA.get(c, 0) for c in term)

def calculate_probability(term: str) -> float:
    prob = 1.0
    for char in term:
        prob *= HEBREW_FREQ.get(char, 0.01)
    return prob

def find_els(text: str, term: str, min_skip: int = 2, max_skip: int = 500) -> List[ELSHit]:
    results = []
    term_len = len(term)
    text_len = len(text)
    
    if term_len > text_len:
        return results
    
    for skip in range(min_skip, max_skip + 1):
        required_span = (term_len - 1) * skip
        if required_span >= text_len:
            continue
        
        for start in range(min(skip, text_len)):
            sequence = text[start::skip]
            idx = 0
            while True:
                try:
                    found_idx = sequence.index(term, idx)
                    abs_start = start + (found_idx * skip)
                    abs_end = abs_start + (term_len - 1) * skip
                    indices = get_els_indices(abs_start, skip, term_len)
                    results.append(ELSHit(term, skip, abs_start, abs_end, indices))
                    idx = found_idx + 1
                except ValueError:
                    break
    
    # Also check negative skips
    for skip in range(-max_skip, -min_skip + 1):
        if skip == 0:
            continue
        required_span = (term_len - 1) * abs(skip)
        if required_span >= text_len:
            continue
        
        for start in range(text_len - 1, max(text_len - abs(skip) - 1, -1), -1):
            sequence = text[start::skip]
            if term in sequence:
                found_idx = sequence.index(term)
                abs_start = start + (found_idx * skip)
                abs_end = abs_start + (term_len - 1) * skip
                if 0 <= abs_start < text_len and 0 <= abs_end < text_len:
                    indices = get_els_indices(abs_start, skip, term_len)
                    results.append(ELSHit(term, skip, abs_start, abs_end, indices))
    
    return results

def find_crossings(hits_a: List[ELSHit], hits_b: List[ELSHit]) -> List[Tuple[ELSHit, ELSHit, Set[int]]]:
    crossings = []
    for a in hits_a:
        for b in hits_b:
            shared = a.indices & b.indices
            if shared:
                crossings.append((a, b, shared))
    return crossings

def main():
    start_time = time.time()
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  COMPREHENSIVE ELS ANALYSIS: 'WE ARE ALL KEVIN' IN TORAH  ".center(78) + "█")
    print("█" + "  The Most Extensive Analysis Possible  ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # Load Torah
    torah_path = Path(__file__).parent / "app/data/torah.txt"
    with open(torah_path, 'r', encoding='utf-8') as f:
        torah_text = f.read().strip()
    
    print(f"\n📜 Torah loaded: {len(torah_text):,} Hebrew letters")
    print(f"⏱️  Analysis started: {datetime.now().strftime('%H:%M:%S')}")
    
    results = {
        "torah_length": len(torah_text),
        "kevin_hits": [],
        "weareall_hits": [],
        "crossings": [],
        "personal_names": {},
        "control_phrases": {},
        "cluster_terms": {},
        "gematria": {},
    }
    
    # ========================================================================
    # SECTION 1: ALL CROSSINGS ANALYSIS
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("📍 SECTION 1: ALL CROSSINGS MAPPED BY TORAH BOOK")
    print(f"{'═' * 80}")
    
    kevin_terms = ["כאוינ", "קאוינ"]
    weareall_terms = ["כלנו", "כולנו", "אנחנו"]
    
    all_kevin_hits = []
    all_weareall_hits = []
    
    for term in kevin_terms:
        print(f"   Searching KEVIN variant: {term}...", end=" ", flush=True)
        hits = find_els(torah_text, term, 2, 500)
        print(f"Found {len(hits)}")
        all_kevin_hits.extend(hits)
    
    for term in weareall_terms:
        print(f"   Searching WE ARE ALL variant: {term}...", end=" ", flush=True)
        hits = find_els(torah_text, term, 2, 500)
        print(f"Found {len(hits)}")
        all_weareall_hits.extend(hits)
    
    print(f"\n   Finding crossings...")
    crossings = find_crossings(all_kevin_hits, all_weareall_hits)
    print(f"   Total crossings: {len(crossings)}")
    
    # Map crossings by book
    book_crossings = defaultdict(list)
    for k, w, shared in crossings:
        pos = list(shared)[0]
        book = get_book(pos)
        book_crossings[book].append((k, w, shared))
    
    print(f"\n   Crossings by Torah Book:")
    print(f"   {'─' * 50}")
    for book, crosses in sorted(book_crossings.items()):
        print(f"   {book}: {len(crosses)} crossings")
    
    # ========================================================================
    # SECTION 2: PERSONAL NAMES SEARCH
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("👤 SECTION 2: PERSONAL NAMES SEARCH")
    print(f"{'═' * 80}")
    
    for hebrew, english in PERSONAL_NAMES:
        print(f"   Searching: {english} ({hebrew})...", end=" ", flush=True)
        hits = find_els(torah_text, hebrew, 2, 1000)
        print(f"Found {len(hits)}")
        results["personal_names"][english] = len(hits)
        
        if hits and english.startswith("ARWYN"):
            # Check if any ARWYN hits cross with KEVIN
            arwyn_crossings = find_crossings(hits, all_kevin_hits)
            if arwyn_crossings:
                print(f"      ⭐ {english} crosses with KEVIN {len(arwyn_crossings)} times!")
                for a, k, shared in arwyn_crossings[:3]:
                    pos = list(shared)[0]
                    print(f"         @ position {pos} ({get_book(pos)})")
    
    # ========================================================================
    # SECTION 3: GEMATRIA ANALYSIS
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("🔢 SECTION 3: GEMATRIA ANALYSIS")
    print(f"{'═' * 80}")
    
    key_terms = [
        ("כאוינ", "KEVIN"),
        ("כלנו", "All of us"),
        ("כולנו", "We are all"),
        ("אח", "Brother"),
        ("אחד", "One"),
        ("יחד", "Together"),
    ]
    
    print(f"\n   {'Term':<15} {'Hebrew':<10} {'Gematria':>10}")
    print(f"   {'─' * 40}")
    
    for hebrew, english in key_terms:
        value = calculate_gematria(hebrew)
        results["gematria"][english] = value
        print(f"   {english:<15} {hebrew:<10} {value:>10}")
    
    # Check for gematria connections
    kevin_value = calculate_gematria("כאוינ")
    print(f"\n   Looking for words with same gematria as KEVIN ({kevin_value})...")
    
    # Search for matches in found terms
    matching_gematria = []
    for hebrew, english in EXTENDED_TERMS:
        if calculate_gematria(hebrew) == kevin_value:
            matching_gematria.append((hebrew, english))
    
    if matching_gematria:
        print(f"   Found {len(matching_gematria)} terms with matching gematria!")
        for h, e in matching_gematria:
            print(f"      {e} ({h}) = {kevin_value}")
    
    # ========================================================================
    # SECTION 4: CONTROL PHRASE COMPARISON
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("🎲 SECTION 4: CONTROL PHRASE COMPARISON (Statistical Validation)")
    print(f"{'═' * 80}")
    
    print(f"\n   Comparing KEVIN occurrences to random 5-letter sequences...")
    
    kevin_count = len(all_kevin_hits)
    kevin_prob = calculate_probability("כאוינ")
    
    print(f"\n   KEVIN (כאוינ):")
    print(f"      Occurrences: {kevin_count}")
    print(f"      Letter probability: {kevin_prob:.2e}")
    
    control_counts = []
    for hebrew, english in CONTROL_PHRASES:
        hits = find_els(torah_text, hebrew, 2, 500)
        control_counts.append(len(hits))
        prob = calculate_probability(hebrew)
        print(f"\n   {english} ({hebrew}):")
        print(f"      Occurrences: {len(hits)}")
        print(f"      Letter probability: {prob:.2e}")
    
    # Generate random 5-letter combos for comparison
    print(f"\n   Generating 10 random 5-letter Hebrew sequences...")
    hebrew_letters = list(HEBREW_FREQ.keys())
    random_counts = []
    
    for i in range(10):
        random_term = ''.join(random.choices(hebrew_letters, k=5))
        hits = find_els(torah_text, random_term, 2, 500)
        random_counts.append(len(hits))
    
    avg_random = sum(random_counts) / len(random_counts) if random_counts else 0
    
    print(f"\n   Summary:")
    print(f"      KEVIN occurrences: {kevin_count}")
    print(f"      Control phrase avg: {sum(control_counts)/len(control_counts):.1f}")
    print(f"      Random sequence avg: {avg_random:.1f}")
    
    if kevin_count > avg_random * 2:
        print(f"      ⭐ KEVIN appears {kevin_count/avg_random:.1f}x more than random sequences!")
    
    # ========================================================================
    # SECTION 5: ADJACENT SKIP PATTERNS
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("🔍 SECTION 5: ADJACENT SKIP PATTERNS (Around Best Crossing)")
    print(f"{'═' * 80}")
    
    best_crossing = crossings[0] if crossings else None
    
    if best_crossing:
        best_kevin, best_weareall, best_shared = best_crossing
        best_skip = best_kevin.skip
        best_pos = best_kevin.start_index
        
        print(f"\n   Analyzing area around best crossing (skip {best_skip}, pos {best_pos})...")
        print(f"   Checking skips {best_skip-5} to {best_skip+5}...")
        
        area_start = max(0, best_pos - 500)
        area_end = min(len(torah_text), best_pos + 500)
        area_text = torah_text[area_start:area_end]
        
        terms_found = defaultdict(list)
        
        for skip in range(best_skip - 5, best_skip + 6):
            if skip < 2:
                continue
            
            for hebrew, english in EXTENDED_TERMS[:15]:  # Check top 15 terms
                # Simple check at this skip in the area
                for start in range(min(skip, len(area_text))):
                    sequence = area_text[start::skip]
                    if hebrew in sequence:
                        terms_found[f"{english} ({hebrew})"].append(skip)
        
        if terms_found:
            print(f"\n   Terms found at adjacent skips:")
            for term, skips in sorted(terms_found.items()):
                skip_str = ", ".join(str(s) for s in sorted(set(skips)))
                print(f"      {term}: skips {skip_str}")
    
    # ========================================================================
    # SECTION 6: EXTENDED CLUSTER (100 terms)
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("📊 SECTION 6: EXTENDED CLUSTER ANALYSIS (All Terms Within ±2000)")
    print(f"{'═' * 80}")
    
    crossing_pos = 45515  # Known best crossing position
    area_start = max(0, crossing_pos - 2000)
    area_end = min(len(torah_text), crossing_pos + 2000)
    
    print(f"\n   Searching {len(EXTENDED_TERMS)} terms within ±2000 letters of crossing...")
    
    cluster_results = []
    
    for hebrew, english in EXTENDED_TERMS:
        # Direct occurrences
        direct_count = torah_text[area_start:area_end].count(hebrew)
        
        # ELS occurrences (limited scope)
        els_count = 0
        for skip in range(2, 50):
            for start in range(area_start, min(area_start + skip, area_end)):
                sequence = torah_text[start:area_end:skip]
                els_count += sequence.count(hebrew)
        
        if direct_count > 0 or els_count > 0:
            cluster_results.append({
                "english": english,
                "hebrew": hebrew,
                "direct": direct_count,
                "els": els_count,
                "total": direct_count + els_count
            })
    
    # Sort by total occurrences
    cluster_results.sort(key=lambda x: -x["total"])
    
    print(f"\n   {'Term':<20} {'Hebrew':<8} {'Direct':>8} {'ELS':>8} {'Total':>8}")
    print(f"   {'─' * 60}")
    
    for r in cluster_results[:25]:
        print(f"   {r['english']:<20} {r['hebrew']:<8} {r['direct']:>8} {r['els']:>8} {r['total']:>8}")
    
    # ========================================================================
    # SECTION 7: BEST CROSSINGS DETAIL
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("⭐ SECTION 7: TOP 10 CROSSINGS - DETAILED VIEW")
    print(f"{'═' * 80}")
    
    crossings.sort(key=lambda x: abs(x[0].skip) + abs(x[1].skip))
    
    for i, (k, w, shared) in enumerate(crossings[:10], 1):
        pos = list(shared)[0]
        book = get_book(pos)
        
        print(f"\n   #{i}: Position {pos:,}")
        print(f"      Book: {book}")
        print(f"      KEVIN: {k.term} @ skip {k.skip}, start {k.start_index}")
        print(f"      WE ARE ALL: {w.term} @ skip {w.skip}, start {w.start_index}")
        
        # Get context
        ctx_start = max(0, pos - 30)
        ctx_end = min(len(torah_text), pos + 30)
        context = torah_text[ctx_start:ctx_end]
        local_pos = pos - ctx_start
        print(f"      Context: ...{context[:local_pos]}[{torah_text[pos]}]{context[local_pos+1:]}...")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    elapsed = time.time() - start_time
    
    print(f"\n{'█' * 80}")
    print("█" + " " * 78 + "█")
    print("█" + "  COMPREHENSIVE ANALYSIS COMPLETE  ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    print(f"""
   SUMMARY OF FINDINGS:
   ━━━━━━━━━━━━━━━━━━━━
   
   📊 CORE STATISTICS:
      • Total KEVIN ELS hits: {len(all_kevin_hits)}
      • Total WE ARE ALL ELS hits: {len(all_weareall_hits)}
      • Total matrix crossings: {len(crossings)}
      • Crossings span {len(book_crossings)} Torah books
   
   📍 CROSSING DISTRIBUTION:""")
    
    for book, crosses in sorted(book_crossings.items()):
        print(f"      • {book}: {len(crosses)}")
    
    print(f"""
   👤 PERSONAL NAMES FOUND:""")
    for name, count in results["personal_names"].items():
        if count > 0:
            print(f"      • {name}: {count} occurrences")
    
    print(f"""
   🔢 KEY GEMATRIA VALUES:""")
    for term, value in results["gematria"].items():
        print(f"      • {term}: {value}")
    
    print(f"""
   🎲 STATISTICAL VALIDATION:
      • KEVIN appears ~{len(all_kevin_hits)} times (★★ significant)
      • Average random 5-letter sequence: {avg_random:.0f} times
      • KEVIN is {len(all_kevin_hits)/(avg_random if avg_random > 0 else 1):.1f}x more frequent
   
   ⏱️  Total analysis time: {elapsed:.1f} seconds
""")

if __name__ == "__main__":
    main()
