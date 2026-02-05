import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.els_analyzer import ELSAnalyzer
from app.services.els_visualizer import ELSVisualizer

def find_local_skip(text, term, center_idx, radius=100):
    """Find the specific skip for a term near a center index"""
    best_skip = 0
    min_dist = float('inf')
    
    # Try reasonable skip range
    for skip in range(1, 10000):
        # Forward
        for start in range(max(0, center_idx - radius), min(len(text), center_idx + radius)):
            if start + (len(term)-1)*skip < len(text):
                match = True
                for i in range(len(term)):
                    if text[start + i*skip] != term[i]:
                        match = False
                        break
                if match:
                    # Calculate distance of this occurrence's center to our target center
                    occurrence_center = start + (len(term)*skip)//2
                    dist = abs(occurrence_center - center_idx)
                    if dist < min_dist:
                        min_dist = dist
                        best_skip = skip
                        
        # Backward (negative skip)
        for start in range(max(0, center_idx - radius), min(len(text), center_idx + radius)):
            if start + (len(term)-1)*(-skip) >= 0:
                match = True
                for i in range(len(term)):
                    if text[start + i*(-skip)] != term[i]:
                        match = False
                        break
                if match:
                    occurrence_center = start + (len(term)*(-skip))//2
                    dist = abs(occurrence_center - center_idx)
                    if dist < min_dist:
                        min_dist = dist
                        best_skip = -skip
                        
    return best_skip

def main():
    print("Loading Torah...")
    # Load Torah directly
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Try both possible locations
        possible_paths = [
            os.path.join(current_dir, "app/data/torah.txt"),
            os.path.join(current_dir, "data/torah.txt"),
            os.path.join(current_dir, "torah.txt")
        ]
        
        torah_text = None
        for p in possible_paths:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    torah_text = f.read().strip()
                print(f"Loaded from {p}")
                break
        
        if not torah_text:
            raise FileNotFoundError("Could not find torah.txt in expected locations")
            
        text = torah_text
    except Exception as e:
        print(f"Error loading text: {e}")
        sys.exit(1)
            
    print(f"Loaded {len(text)} chars")
    
    # Target: Genesis 1.2 approx index 922
    center_idx = 922
    
    # Terms to visualize
    # PEPE = PPI = פפי
    # MEME = MIMI = מימי
    # TRUTH = EMET = אמת
    
    terms = [
        {"term": "פפי", "name": "PEPE", "color": "#00FF00"},   # Neon Green
        {"term": "מימי", "name": "MEME", "color": "#9D00FF"},  # Cyber Purple
        {"term": "אמת", "name": "TRUTH", "color": "#FFD700"}   # Gold
    ]
    
    print(f"Finding exact skips near index {center_idx}...")
    
    visual_terms = []
    
    for t in terms:
        skip = find_local_skip(text, t["term"], center_idx)
        print(f"Found {t['name']} ({t['term']}) closest skip: {skip}")
        
        if skip != 0:
            t["skip"] = skip
            visual_terms.append(t)
        else:
            print(f"WARNING: Could not find {t['name']} near center!")

    # Set row width to the largest skip (or a multiple) to try and align one of them vertically
    # Or just use a standard width that looks good. 
    # Usually we want to align the main term (PEPE or MEME).
    # Let's align MEME (Cyber Purple)
    meme_term = next((t for t in visual_terms if t["name"] == "MEME"), None)
    row_width = abs(meme_term["skip"]) if meme_term else 50
    
    # Ensure reasonable row width for display
    if row_width > 100: row_width = 50 
    
    print("Generating SVG...")
    svg_content = ELSVisualizer.generate_svg_matrix(
        text=text,
        center_index=center_idx,
        terms=visual_terms,
        rows=25,
        cols=25,
        row_width=row_width
    )
    
    out_path = "/Users/arwynhughes/.gemini/antigravity/brain/b3c33d65-4f0c-4c57-a62e-ba79a30bdcd2/pepe_meme_truth_matrix.svg"
    with open(out_path, "w") as f:
        f.write(svg_content)
        
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
