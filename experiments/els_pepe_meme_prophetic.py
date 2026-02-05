#!/usr/bin/env python3
"""
ELS ANALYSIS: PEPE + MEME + PROPHETIC TERMS (OPTIMIZED)
========================================================
Focus on:
1. MEME as a key term (focused on best transliteration)
2. PEPE/FROG/MEME intersections (sampled for speed)
3. Prophetic/Future terms
"""

from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import random
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

# Gematria
GEMATRIA = {
    'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5, 'ו': 6, 'ז': 7, 'ח': 8, 'ט': 9,
    'י': 10, 'כ': 20, 'ל': 30, 'מ': 40, 'נ': 50, 'ס': 60, 'ע': 70, 'פ': 80, 'צ': 90,
    'ק': 100, 'ר': 200, 'ש': 300, 'ת': 400,
    'ך': 20, 'ם': 40, 'ן': 50, 'ף': 80, 'ץ': 90,
}

# Focused MEME transliterations (best ones only)
MEME_TERMS = [
    ("מימי", "MEME (MIMI)"),  # 4 letters, closer to English
    ("מאמא", "MEME (MAMA)"),  # Alternative
]

# PEPE (focused)
PEPE_TERMS = [
    ("פפי", "PEPE (PPI)"),
    ("פאפא", "PEPE (PAPA)"),
]

# FROG (focused)
FROG_TERMS = [
    ("צפרדע", "FROG (Biblical)"),
    ("צפר", "TZF-R (root)"),
]

# PROPHETIC / FUTURE / REVELATION terms
PROPHETIC_TERMS = [
    # Prophetic core
    ("משיח", "MESSIAH"),
    ("נביא", "PROPHET"),
    ("חזון", "VISION"),
    
    # Future / End
    ("עתיד", "FUTURE"),
    ("אחרית", "END OF DAYS"),
    ("סוף", "END"),
    
    # Revelation / Truth
    ("גילוי", "REVELATION"),
    ("אמת", "TRUTH"),
    ("סוד", "SECRET"),
    ("נסתר", "HIDDEN"),
    
    # Transformation
    ("גאולה", "REDEMPTION"),
    ("חדש", "NEW"),
    ("תקומה", "RESURRECTION"),
    
    # Spread / Virality
    ("עולם", "WORLD"),
    ("גוים", "NATIONS"),
    ("כל", "ALL"),
    ("רבים", "MANY"),
    
    # Time
    ("דור", "GENERATION"),
    ("זמן", "TIME"),
    
    # Emotion (meme culture)
    ("שמחה", "JOY"),
    ("צחוק", "LAUGHTER"),
    ("ירק", "GREEN"),
]

@dataclass
class ELSHit:
    term: str
    skip: int
    start_index: int
    indices: Set[int]

def get_book(position: int) -> str:
    for name, start, end in TORAH_BOOKS:
        if start <= position < end:
            return name
    return "Unknown"

def get_book_pct(position: int) -> Tuple[str, float]:
    for name, start, end in TORAH_BOOKS:
        if start <= position < end:
            pct = ((position - start) / (end - start)) * 100
            return name, pct
    return "Unknown", 0

def calculate_gematria(term: str) -> int:
    return sum(GEMATRIA.get(c, 0) for c in term)

def find_els_fast(text: str, term: str, min_skip: int = 2, max_skip: int = 300) -> List[ELSHit]:
    """Faster ELS search with limited skip range."""
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
                    indices = set(abs_start + i * skip for i in range(term_len))
                    results.append(ELSHit(term, skip, abs_start, indices))
                    idx = found_idx + 1
                except ValueError:
                    break
    
    return results

def find_crossings_sampled(hits_a: List[ELSHit], hits_b: List[ELSHit], sample_a=3000, sample_b=3000):
    """Find crossings with sampling for performance."""
    crossings = []
    
    # Sample if too many
    a_sample = hits_a[:sample_a] if len(hits_a) > sample_a else hits_a
    b_sample = hits_b[:sample_b] if len(hits_b) > sample_b else hits_b
    
    for a in a_sample:
        for b in b_sample:
            shared = a.indices & b.indices
            if shared:
                crossings.append((a, b, shared))
    
    return crossings

def find_proximity_zones(pepe_hits, frog_hits, meme_hits, radius=500):
    """Find zones where all three terms appear within radius."""
    zones = []
    
    pepe_sample = pepe_hits[:2000]
    frog_sample = frog_hits[:1000]
    meme_sample = meme_hits[:2000]
    
    for p in pepe_sample:
        p_pos = p.start_index
        for f in frog_sample:
            f_pos = f.start_index
            if abs(p_pos - f_pos) > radius:
                continue
            for m in meme_sample:
                m_pos = m.start_index
                if abs(m_pos - p_pos) <= radius and abs(m_pos - f_pos) <= radius:
                    center = (p_pos + f_pos + m_pos) // 3
                    spread = max(p_pos, f_pos, m_pos) - min(p_pos, f_pos, m_pos)
                    zones.append({
                        "pepe": p, "frog": f, "meme": m,
                        "center": center, "spread": spread
                    })
    
    return zones

def main():
    start_time = time.time()
    
    print("\n" + "🐸📱" * 20)
    print("\n" + "█" * 80)
    print("█" + "  PEPE + MEME + PROPHETIC ANALYSIS (OPTIMIZED)  ".center(78) + "█")
    print("█" * 80)
    
    # Load Torah
    torah_path = Path(__file__).parent / "app/data/torah.txt"
    with open(torah_path, 'r', encoding='utf-8') as f:
        torah_text = f.read().strip()
    
    print(f"\n📜 Torah: {len(torah_text):,} letters")
    
    # ========================================================================
    # SECTION 1: MEME SEARCH
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("📱 MEME SEARCH")
    print(f"{'═' * 80}")
    
    all_meme_hits = []
    for hebrew, english in MEME_TERMS:
        print(f"   {english} ({hebrew})...", end=" ", flush=True)
        hits = find_els_fast(torah_text, hebrew, 2, 300)
        print(f"{len(hits)}")
        all_meme_hits.extend(hits)
    
    print(f"\n   📱 MEME total: {len(all_meme_hits)}")
    
    # ========================================================================
    # SECTION 2: PEPE SEARCH
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("🐸 PEPE SEARCH")
    print(f"{'═' * 80}")
    
    all_pepe_hits = []
    for hebrew, english in PEPE_TERMS:
        print(f"   {english} ({hebrew})...", end=" ", flush=True)
        hits = find_els_fast(torah_text, hebrew, 2, 300)
        print(f"{len(hits)}")
        all_pepe_hits.extend(hits)
    
    print(f"\n   🐸 PEPE total: {len(all_pepe_hits)}")
    
    # ========================================================================
    # SECTION 3: FROG SEARCH
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("🐸 FROG SEARCH")
    print(f"{'═' * 80}")
    
    all_frog_hits = []
    for hebrew, english in FROG_TERMS:
        print(f"   {english} ({hebrew})...", end=" ", flush=True)
        hits = find_els_fast(torah_text, hebrew, 2, 300)
        print(f"{len(hits)}")
        all_frog_hits.extend(hits)
    
    print(f"\n   🐸 FROG total: {len(all_frog_hits)}")
    
    # ========================================================================
    # SECTION 4: CROSSINGS
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("📍 CROSSINGS")
    print(f"{'═' * 80}")
    
    print("\n   Computing PEPE × MEME crossings...", end=" ", flush=True)
    pepe_meme = find_crossings_sampled(all_pepe_hits, all_meme_hits)
    print(f"{len(pepe_meme)}")
    
    print("   Computing PEPE × FROG crossings...", end=" ", flush=True)
    pepe_frog = find_crossings_sampled(all_pepe_hits, all_frog_hits)
    print(f"{len(pepe_frog)}")
    
    print("   Computing MEME × FROG crossings...", end=" ", flush=True)
    meme_frog = find_crossings_sampled(all_meme_hits, all_frog_hits)
    print(f"{len(meme_frog)}")
    
    # ========================================================================
    # SECTION 5: TRIPLE CONVERGENCE
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("🔥 TRIPLE CONVERGENCE (PEPE + FROG + MEME within 500 letters)")
    print(f"{'═' * 80}")
    
    zones = find_proximity_zones(all_pepe_hits, all_frog_hits, all_meme_hits)
    zones.sort(key=lambda x: x["spread"])
    
    print(f"\n   Triple convergence zones: {len(zones)}")
    
    if zones:
        print(f"\n   TOP 10 TIGHTEST ZONES:")
        for i, z in enumerate(zones[:10], 1):
            book, pct = get_book_pct(z["center"])
            print(f"   #{i}: Center {z['center']:,} ({book}, {pct:.1f}%) | Spread: {z['spread']}")
            print(f"       PEPE: {z['pepe'].term} skip {z['pepe'].skip}")
            print(f"       FROG: {z['frog'].term} skip {z['frog'].skip}")
            print(f"       MEME: {z['meme'].term} skip {z['meme'].skip}")
    
    # Get best zone for prophetic analysis
    best_zone = zones[0]["center"] if zones else 50000
    
    # ========================================================================
    # SECTION 6: PROPHETIC TERMS AT CONVERGENCE
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("🔮 PROPHETIC TERMS NEAR CONVERGENCE")
    print(f"{'═' * 80}")
    
    print(f"\n   Scanning ±3000 letters around position {best_zone:,}...")
    
    area_start = max(0, best_zone - 3000)
    area_end = min(len(torah_text), best_zone + 3000)
    
    prophetic_results = []
    
    for hebrew, english in PROPHETIC_TERMS:
        direct = torah_text[area_start:area_end].count(hebrew)
        
        els = 0
        for skip in range(2, 80):
            for start in range(area_start, min(area_start + skip, area_end)):
                sequence = torah_text[start:area_end:skip]
                els += sequence.count(hebrew)
        
        if direct > 0 or els > 0:
            prophetic_results.append({
                "english": english, "hebrew": hebrew,
                "direct": direct, "els": els, "total": direct + els
            })
    
    prophetic_results.sort(key=lambda x: -x["total"])
    
    print(f"\n   {'Term':<20} {'Hebrew':<8} {'Direct':>6} {'ELS':>6} {'Total':>6}")
    print(f"   {'─' * 50}")
    
    key_prophetic = ["MESSIAH", "PROPHET", "FUTURE", "END OF DAYS", "REVELATION", "TRUTH", "SECRET", "HIDDEN"]
    
    for r in prophetic_results[:15]:
        star = "⭐" if r["english"] in key_prophetic else ""
        print(f"   {r['english']:<20} {r['hebrew']:<8} {r['direct']:>6} {r['els']:>6} {r['total']:>6} {star}")
    
    # ========================================================================
    # SECTION 7: PROPHETIC × PEPE/MEME CROSSINGS
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("🔮 PROPHETIC TERM CROSSINGS")
    print(f"{'═' * 80}")
    
    prophetic_crossing_results = {}
    
    key_terms = [
        ("משיח", "MESSIAH"),
        ("נביא", "PROPHET"),
        ("עתיד", "FUTURE"),
        ("אחרית", "END OF DAYS"),
        ("גילוי", "REVELATION"),
        ("אמת", "TRUTH"),
        ("סוד", "SECRET"),
        ("גאולה", "REDEMPTION"),
    ]
    
    for hebrew, english in key_terms:
        hits = find_els_fast(torah_text, hebrew, 2, 150)
        pepe_cross = find_crossings_sampled(all_pepe_hits, hits, 2000, 2000)
        meme_cross = find_crossings_sampled(all_meme_hits, hits, 2000, 2000)
        
        total = len(pepe_cross) + len(meme_cross)
        print(f"   {english}: {total} total (PEPE: {len(pepe_cross)}, MEME: {len(meme_cross)})")
        
        prophetic_crossing_results[english] = {
            "pepe": len(pepe_cross),
            "meme": len(meme_cross),
            "total": total,
            "best_examples": pepe_cross[:3] + meme_cross[:3]
        }
    
    # ========================================================================
    # SECTION 8: GEMATRIA
    # ========================================================================
    print(f"\n{'═' * 80}")
    print("🔢 GEMATRIA")
    print(f"{'═' * 80}")
    
    terms = [
        ("פפי", "PEPE"), ("מימי", "MEME"), ("צפרדע", "FROG"),
        ("משיח", "MESSIAH"), ("נביא", "PROPHET"), ("גילוי", "REVELATION"),
    ]
    
    print(f"\n   {'Term':<15} {'Hebrew':<8} {'Value':>8}")
    print(f"   {'─' * 35}")
    for hebrew, english in terms:
        print(f"   {english:<15} {hebrew:<8} {calculate_gematria(hebrew):>8}")
    
    pepe_val = calculate_gematria("פפי")
    meme_val = calculate_gematria("מימי")
    frog_val = calculate_gematria("צפרדע")
    
    print(f"\n   COMBINATIONS:")
    print(f"   PEPE + MEME = {pepe_val + meme_val}")
    print(f"   PEPE + FROG = {pepe_val + frog_val}")
    print(f"   PEPE + MEME + FROG = {pepe_val + meme_val + frog_val}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    elapsed = time.time() - start_time
    
    print(f"\n{'█' * 80}")
    print("█" + "  ANALYSIS COMPLETE  ".center(78) + "█")
    print("█" * 80)
    
    print(f"""
   CORE STATISTICS:
   ━━━━━━━━━━━━━━━━
   📱 MEME hits: {len(all_meme_hits)}
   🐸 PEPE hits: {len(all_pepe_hits)}
   🐸 FROG hits: {len(all_frog_hits)}
   
   CROSSINGS:
   ━━━━━━━━━━
   PEPE × MEME: {len(pepe_meme)}
   PEPE × FROG: {len(pepe_frog)}
   MEME × FROG: {len(meme_frog)}
   Triple zones: {len(zones)}
   
   BEST CONVERGENCE:
   ━━━━━━━━━━━━━━━━""")
    
    if zones:
        best = zones[0]
        book, pct = get_book_pct(best["center"])
        print(f"   Position: {best['center']:,}")
        print(f"   Book: {book} ({pct:.1f}%)")
        print(f"   Spread: {best['spread']} letters")
    
    print(f"""
   PROPHETIC CROSSINGS:
   ━━━━━━━━━━━━━━━━━━━""")
    
    for term, data in prophetic_crossing_results.items():
        if data["total"] > 0:
            print(f"   {term}: {data['total']}")
    
    print(f"""
   GEMATRIA:
   ━━━━━━━━━
   PEPE: {pepe_val}
   MEME: {meme_val}
   PEPE + MEME: {pepe_val + meme_val}
   
   ⏱️ Time: {elapsed:.1f}s
""")

if __name__ == "__main__":
    main()
