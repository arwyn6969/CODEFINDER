import json
import re
from pathlib import Path
from collections import defaultdict

"""
Analyzes 'V', 'VV', and 'W' usage in the 1609 Sonnets.

PROTOCOL: Adheres to `AXIOM_OF_INTENT.md`.
'VV' is treated as a deliberate substitution cipher symbol, not a type shortage.
"""

def analyze_vv_w():
    db_path = "reports/digital_type_case/type_case_db.json"
    if not Path(db_path).exists():
        print("Database not found.")
        return

    with open(db_path) as f:
        data = json.load(f)

    # Reconstruct text flow
    # Structure: page -> word_idx -> list of (char_idx, char, path)
    pages = defaultdict(lambda: defaultdict(list))

    total_w = 0
    total_v = 0
    
    print("Parsing database...")
    for entry in data:
        char = entry['char']
        path = entry['path']
        page = entry['page']
        
        # Filename format: {safe_char}_{page}_{word_idx}_{char_idx}.png
        # Example: S_upper_1_4_0.png
        filename = Path(path).stem
        parts = filename.split('_')
        
        # We need to find where the numeric parts start. 
        # Safe char can have underscores (e.g. S_upper, long_s).
        # We know the last 3 parts are ALWAYS page, word_idx, char_idx
        try:
            char_idx = int(parts[-1])
            word_idx = int(parts[-2])
            page_from_fname = int(parts[-3])
            
            # Verification
            if page != page_from_fname:
                # mismatch, trust filename if db has offset? No, usually trust DB. 
                # But let's stick to consistent parsing.
                pass
            
            pages[page][word_idx].append((char_idx, char, path))
            
            if char.lower() == 'w':
                total_w += 1
            if char.lower() == 'v':
                total_v += 1
                
        except ValueError:
            continue

    print(f"Total 'W/w' found: {total_w}")
    print(f"Total 'V/v' found: {total_v}")
    
    vv_sequences = []
    
    # Scan for VV sequences
    sorted_pages = sorted(pages.keys())
    for p in sorted_pages:
        words = pages[p]
        sorted_words = sorted(words.keys())
        for w_idx in sorted_words:
            chars = sorted(words[w_idx], key=lambda x: x[0])
            
            # Reconstruct word string
            word_str = "".join([c[1] for c in chars])
            
            # Check for VV (case insensitive)
            # We want to find adjacent V's
            
            for i in range(len(chars) - 1):
                c1 = chars[i][1]
                c2 = chars[i+1][1]
                
                if c1.lower() == 'v' and c2.lower() == 'v':
                    # Found a VV!
                    vv_sequences.append({
                        "page": p,
                        "word": word_str,
                        "char_1": chars[i],
                        "char_2": chars[i+1]
                    })

    print(f"Found {len(vv_sequences)} 'VV' sequences.")
    
    # Generate Report
    report_path = "reports/sonnet_print_block_analysis/vv_w_analysis.txt"
    with open(report_path, "w") as f:
        f.write("# V vs VV vs W Analysis\n\n")
        f.write(f"Total 'W' blocks (single sort): {total_w}\n")
        f.write(f"Total 'V' blocks: {total_v}\n")
        f.write(f"Total 'VV' sequences (adjacent sorts): {len(vv_sequences)}\n\n")
        
        if vv_sequences:
            f.write("## 'VV' Sequences Found\n")
            f.write("These are instances where two 'V' blocks appear immediately next to each other within the same word context.\n\n")
            for seq in vv_sequences:
                f.write(f"- Page {seq['page']}: Word '{seq['word']}' (Chars: {seq['char_1'][1]}{seq['char_2'][1]})\n")
                f.write(f"  - Image 1: {seq['char_1'][2]}\n")
                f.write(f"  - Image 2: {seq['char_2'][2]}\n")
        else:
            f.write("## No 'VV' Sequences Found\n")
            f.write("No instances of immediate adjacency (V followed by V within same Tesseract word box) were detected.\n")
            
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    analyze_vv_w()
