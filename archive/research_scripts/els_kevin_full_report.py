#!/usr/bin/env python3
"""
ELS Full Report: "WE ARE ALL KEVIN" in Torah
=============================================
1. Matrix Grid Visualization
2. Statistical Significance Analysis
3. Torah Context Identification (Book/Chapter)
"""

from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
import math
from datetime import datetime

# Torah book structure (approximate letter counts)
TORAH_BOOKS = [
    ("Genesis / בראשית", 0, 78064),
    ("Exodus / שמות", 78064, 141064),  
    ("Leviticus / ויקרא", 141064, 185413),
    ("Numbers / במדבר", 185413, 249878),
    ("Deuteronomy / דברים", 249878, 304805),
]

# Hebrew letter frequencies (approximate)
HEBREW_FREQ = {
    'א': 0.048, 'ב': 0.043, 'ג': 0.011, 'ד': 0.025, 'ה': 0.069,
    'ו': 0.099, 'ז': 0.008, 'ח': 0.022, 'ט': 0.009, 'י': 0.092,
    'כ': 0.039, 'ל': 0.055, 'מ': 0.059, 'נ': 0.043, 'ס': 0.011,
    'ע': 0.035, 'פ': 0.015, 'צ': 0.012, 'ק': 0.012, 'ר': 0.055,
    'ש': 0.050, 'ת': 0.050,
}

@dataclass
class ELSHit:
    term: str
    skip: int
    start_index: int
    end_index: int
    direction: str
    indices: Set[int]

def get_book_and_location(position: int) -> Tuple[str, float]:
    """Determine which Torah book contains the given position."""
    for book_name, start, end in TORAH_BOOKS:
        if start <= position < end:
            pct = ((position - start) / (end - start)) * 100
            return book_name, pct
    return "Unknown", 0

def get_els_indices(start: int, skip: int, term_len: int) -> Set[int]:
    return set(start + i * skip for i in range(term_len))

def find_els_with_indices(text: str, term: str, min_skip: int, max_skip: int) -> List[ELSHit]:
    results = []
    term_len = len(term)
    text_len = len(text)
    
    if term_len > text_len:
        return results
    
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
                            term=term, skip=skip, start_index=abs_start,
                            end_index=abs_end, direction="forward" if skip > 0 else "backward",
                            indices=indices
                        ))
                        idx = found_idx + 1
                    except ValueError:
                        break
    return results

def calculate_term_probability(term: str) -> float:
    """Calculate probability of term appearing by chance."""
    prob = 1.0
    for char in term:
        prob *= HEBREW_FREQ.get(char, 0.01)
    return prob

def calculate_statistical_significance(text_len: int, term: str, skip: int, found_count: int) -> Dict:
    """Calculate p-value and significance of ELS finding."""
    term_len = len(term)
    term_prob = calculate_term_probability(term)
    
    # Expected occurrences at this specific skip
    possible_starts = text_len // skip
    expected = possible_starts * term_prob
    
    # Poisson approximation for p-value
    if expected <= 0:
        p_value = 1.0
    else:
        p_value = 1 - math.exp(-expected) * sum(
            (expected ** k) / math.factorial(k) for k in range(found_count)
        )
    
    if p_value < 0.001:
        significance = "HIGHLY SIGNIFICANT ★★★"
    elif p_value < 0.01:
        significance = "VERY SIGNIFICANT ★★"
    elif p_value < 0.05:
        significance = "SIGNIFICANT ★"
    elif p_value < 0.10:
        significance = "MARGINALLY SIGNIFICANT"
    else:
        significance = "NOT SIGNIFICANT"
    
    return {
        "term_prob": term_prob,
        "expected": expected,
        "p_value": p_value,
        "significance": significance
    }

def create_matrix_visualization(text: str, kevin_hit: ELSHit, weareall_hit: ELSHit, 
                                 shared_pos: int, context_size: int = 15) -> str:
    """Create ASCII matrix grid showing the crossing."""
    
    # Determine the width (skip of one term) and find intersection
    width = abs(kevin_hit.skip)
    
    # Find the row that contains the shared position
    min_pos = min(kevin_hit.start_index, weareall_hit.start_index)
    max_pos = max(kevin_hit.end_index, weareall_hit.end_index)
    
    # Calculate display bounds
    start_row = max(0, (min_pos // width) - context_size)
    end_row = min(len(text) // width, (max_pos // width) + context_size)
    
    lines = []
    lines.append(f"\n{'╔' + '═' * (width + 4) + '╗'}")
    lines.append(f"║ Matrix Grid (width={width}) ║")
    lines.append(f"{'╚' + '═' * (width + 4) + '╝'}")
    lines.append("")
    
    # Create header with column numbers
    header = "     "
    for col in range(width):
        header += f"{col % 10}"
    lines.append(header)
    lines.append("     " + "─" * width)
    
    for row in range(start_row, min(end_row, start_row + 35)):
        row_start = row * width
        row_end = min(row_start + width, len(text))
        
        if row_start >= len(text):
            break
        
        row_text = text[row_start:row_end]
        
        # Mark special positions
        display_chars = []
        for col, char in enumerate(row_text):
            pos = row_start + col
            
            if pos == shared_pos:
                # Crossing point - highlight
                display_chars.append(f"[{char}]")
            elif pos in kevin_hit.indices:
                # KEVIN letter
                display_chars.append(f"«{char}»")
            elif pos in weareall_hit.indices:
                # WE ARE ALL letter
                display_chars.append(f"‹{char}›")
            else:
                display_chars.append(f" {char} ")
        
        # Row number
        row_label = f"{row:4d} "
        lines.append(row_label + "".join(display_chars))
    
    lines.append("")
    lines.append("Legend: [X] = CROSSING POINT (both terms)")
    lines.append("        «X» = KEVIN letter")
    lines.append("        ‹X› = WE ARE ALL letter")
    
    return "\n".join(lines)

def generate_html_report(torah_text: str, kevin_hit: ELSHit, weareall_hit: ELSHit,
                          shared_positions: Set[int], all_crossings: int, stats: Dict) -> str:
    """Generate full HTML report."""
    
    shared_pos = list(shared_positions)[0]
    book, pct = get_book_and_location(shared_pos)
    
    # Create visual matrix
    width = abs(kevin_hit.skip)
    min_pos = min(kevin_hit.start_index, weareall_hit.start_index)
    max_pos = max(kevin_hit.end_index, weareall_hit.end_index)
    
    start_row = max(0, (min_pos // width) - 8)
    end_row = min(len(torah_text) // width, (max_pos // width) + 8)
    
    matrix_rows = []
    for row in range(start_row, min(end_row, start_row + 25)):
        row_start = row * width
        row_end = min(row_start + width, len(torah_text))
        if row_start >= len(torah_text):
            break
        
        cells = []
        for col in range(row_start, row_end):
            char = torah_text[col]
            pos = col
            
            if pos in shared_positions:
                cells.append(f'<td class="cross">{char}</td>')
            elif pos in kevin_hit.indices:
                cells.append(f'<td class="kevin">{char}</td>')
            elif pos in weareall_hit.indices:
                cells.append(f'<td class="weareall">{char}</td>')
            else:
                cells.append(f'<td>{char}</td>')
        
        matrix_rows.append(f'<tr><th>{row}</th>{"".join(cells)}</tr>')
    
    html = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>ELS Report: WE ARE ALL KEVIN in Torah</title>
    <style>
        body {{ font-family: 'David', 'Times New Roman', serif; background: #1a1a2e; color: #eee; padding: 20px; }}
        h1 {{ color: #ffd700; text-align: center; }}
        h2 {{ color: #4fc3f7; border-bottom: 2px solid #4fc3f7; padding-bottom: 5px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .summary {{ background: linear-gradient(135deg, #16213e, #1f4068); padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .stat-box {{ background: #0f3460; padding: 15px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 28px; color: #ffd700; font-weight: bold; }}
        .stat-label {{ font-size: 12px; color: #aaa; }}
        .matrix {{ overflow-x: auto; margin: 20px 0; }}
        table {{ border-collapse: collapse; font-family: 'Courier New', monospace; font-size: 14px; direction: rtl; }}
        th, td {{ width: 24px; height: 24px; text-align: center; border: 1px solid #333; }}
        th {{ background: #0f3460; color: #4fc3f7; }}
        .kevin {{ background: #e91e63 !important; color: white !important; font-weight: bold; }}
        .weareall {{ background: #4caf50 !important; color: white !important; font-weight: bold; }}
        .cross {{ background: #ffd700 !important; color: #000 !important; font-weight: bold; font-size: 18px; animation: pulse 1s infinite; }}
        @keyframes pulse {{ 0%, 100% {{ transform: scale(1); }} 50% {{ transform: scale(1.2); }} }}
        .legend {{ display: flex; gap: 20px; justify-content: center; margin: 20px 0; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .legend-box {{ width: 20px; height: 20px; border-radius: 3px; }}
        .context {{ background: #16213e; padding: 15px; border-radius: 8px; margin: 10px 0; }}
        .highlight {{ color: #ffd700; font-weight: bold; }}
        .sig-stars {{ color: #ffd700; font-size: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔯 ELS Analysis: "WE ARE ALL KEVIN" in Torah</h1>
        <p style="text-align: center; color: #888;">כולנו קווין • Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        
        <div class="summary">
            <h2>📊 Summary Statistics</h2>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-value">{all_crossings}</div>
                    <div class="stat-label">Matrix Crossings Found</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{abs(kevin_hit.skip)}/{abs(weareall_hit.skip)}</div>
                    <div class="stat-label">Best Skip Values</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{shared_pos:,}</div>
                    <div class="stat-label">Crossing Position</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats['p_value']:.4f}</div>
                    <div class="stat-label">P-Value (KEVIN)</div>
                </div>
            </div>
        </div>
        
        <h2>📍 Torah Location</h2>
        <div class="context">
            <p><strong>Book:</strong> <span class="highlight">{book}</span></p>
            <p><strong>Position:</strong> Letter {shared_pos:,} of 304,805 ({pct:.1f}% through the book)</p>
            <p><strong>KEVIN (כאוינ):</strong> Skip {kevin_hit.skip}, starting at position {kevin_hit.start_index:,}</p>
            <p><strong>WE ARE ALL (כלנו):</strong> Skip {weareall_hit.skip}, starting at position {weareall_hit.start_index:,}</p>
        </div>
        
        <h2>📈 Statistical Significance</h2>
        <div class="context">
            <p><strong>KEVIN term probability:</strong> {stats['term_prob']:.2e} per position</p>
            <p><strong>Expected occurrences at skip {kevin_hit.skip}:</strong> {stats['expected']:.4f}</p>
            <p><strong>P-Value:</strong> {stats['p_value']:.6f}</p>
            <p><strong>Assessment:</strong> <span class="sig-stars">{stats['significance']}</span></p>
        </div>
        
        <h2>🔲 Matrix Visualization</h2>
        <div class="legend">
            <div class="legend-item"><div class="legend-box" style="background:#e91e63;"></div> KEVIN (כאוינ)</div>
            <div class="legend-item"><div class="legend-box" style="background:#4caf50;"></div> WE ARE ALL (כלנו)</div>
            <div class="legend-item"><div class="legend-box" style="background:#ffd700;"></div> CROSSING POINT</div>
        </div>
        <div class="matrix">
            <table>
                {''.join(matrix_rows)}
            </table>
        </div>
        
        <p style="text-align: center; color: #666; margin-top: 30px;">
            Generated by CODEFINDER • Torah Text: 304,805 Hebrew Letters
        </p>
    </div>
</body>
</html>"""
    return html

def main():
    print("\n" + "=" * 80)
    print("    ELS FULL REPORT: 'WE ARE ALL KEVIN' IN TORAH")
    print("    Visualization • Statistics • Torah Context")
    print("=" * 80)
    
    # Load Torah
    torah_path = Path(__file__).parent / "app/data/torah.txt"
    with open(torah_path, 'r', encoding='utf-8') as f:
        torah_text = f.read().strip()
    
    print(f"\n📜 Torah loaded: {len(torah_text):,} Hebrew letters")
    
    # Find the best crossing (from previous analysis)
    print("\n🔍 Finding best crossing...")
    
    kevin_term = "כאוינ"
    weareall_term = "כלנו"
    
    kevin_hits = find_els_with_indices(torah_text, kevin_term, 2, 500)
    weareall_hits = find_els_with_indices(torah_text, weareall_term, 2, 500)
    
    # Find crossings
    crossings = []
    for k in kevin_hits:
        for w in weareall_hits:
            shared = k.indices & w.indices
            if shared:
                crossings.append((k, w, shared))
    
    crossings.sort(key=lambda x: abs(x[0].skip) + abs(x[1].skip))
    
    print(f"   Found {len(crossings)} crossings")
    
    if not crossings:
        print("❌ No crossings found!")
        return
    
    best = crossings[0]
    kevin_hit, weareall_hit, shared_positions = best
    shared_pos = list(shared_positions)[0]
    
    # ========================================================================
    # 1. TORAH CONTEXT
    # ========================================================================
    print(f"\n{'─' * 80}")
    print("1. TORAH CONTEXT")
    print(f"{'─' * 80}")
    
    book, pct = get_book_and_location(shared_pos)
    print(f"\n   📖 Book: {book}")
    print(f"   📍 Position: Letter {shared_pos:,} ({pct:.1f}% through the book)")
    print(f"   🔤 KEVIN start: {kevin_hit.start_index:,} | skip: {kevin_hit.skip}")
    print(f"   🔤 WE ARE ALL start: {weareall_hit.start_index:,} | skip: {weareall_hit.skip}")
    
    # Show surrounding text
    context_start = max(0, shared_pos - 50)
    context_end = min(len(torah_text), shared_pos + 50)
    context_text = torah_text[context_start:context_end]
    crossing_offset = shared_pos - context_start
    
    print(f"\n   Context around crossing point:")
    print(f"   ...{context_text[:crossing_offset]}[{torah_text[shared_pos]}]{context_text[crossing_offset+1:]}...")
    
    # ========================================================================
    # 2. STATISTICAL SIGNIFICANCE
    # ========================================================================
    print(f"\n{'─' * 80}")
    print("2. STATISTICAL SIGNIFICANCE")
    print(f"{'─' * 80}")
    
    kevin_stats = calculate_statistical_significance(
        len(torah_text), kevin_term, kevin_hit.skip, 1
    )
    
    weareall_stats = calculate_statistical_significance(
        len(torah_text), weareall_term, weareall_hit.skip, 1
    )
    
    print(f"\n   KEVIN ({kevin_term}):")
    print(f"      Term probability: {kevin_stats['term_prob']:.2e}")
    print(f"      Expected at skip {kevin_hit.skip}: {kevin_stats['expected']:.4f}")
    print(f"      P-value: {kevin_stats['p_value']:.6f}")
    print(f"      Assessment: {kevin_stats['significance']}")
    
    print(f"\n   WE ARE ALL ({weareall_term}):")
    print(f"      Term probability: {weareall_stats['term_prob']:.2e}")
    print(f"      Expected at skip {weareall_hit.skip}: {weareall_stats['expected']:.4f}")
    print(f"      P-value: {weareall_stats['p_value']:.6f}")
    print(f"      Assessment: {weareall_stats['significance']}")
    
    # Combined crossing probability
    crossing_prob = kevin_stats['p_value'] * weareall_stats['p_value'] * (1 / len(torah_text))
    print(f"\n   COMBINED CROSSING:")
    print(f"      Approximate crossing probability: {crossing_prob:.2e}")
    print(f"      Total crossings found: {len(crossings)}")
    
    # ========================================================================
    # 3. MATRIX VISUALIZATION
    # ========================================================================
    print(f"\n{'─' * 80}")
    print("3. MATRIX VISUALIZATION")
    print(f"{'─' * 80}")
    
    matrix = create_matrix_visualization(torah_text, kevin_hit, weareall_hit, shared_pos)
    print(matrix)
    
    # ========================================================================
    # SAVE HTML REPORT
    # ========================================================================
    html = generate_html_report(
        torah_text, kevin_hit, weareall_hit, shared_positions,
        len(crossings), kevin_stats
    )
    
    output_dir = Path(__file__).parent / "demo_results"
    output_dir.mkdir(exist_ok=True)
    html_path = output_dir / "els_kevin_torah_report.html"
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n{'=' * 80}")
    print("REPORT COMPLETE")
    print(f"{'=' * 80}")
    print(f"\n📄 HTML Report saved: {html_path}")
    print(f"💡 Open: file://{html_path.absolute()}")

if __name__ == "__main__":
    main()
