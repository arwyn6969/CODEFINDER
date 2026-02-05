from app.services.els_visualizer import ELSVisualizer

def test_visualization():
    # Create pattern: "01234..." repeated
    # Should align if row width matches period
    pattern = "ABCDE" # Len 5
    text = pattern * 10 # 50 chars. 
    # ABCDEABCDE...
    # At skip 5, we should see columns of same letters.
    
    print("=== Testing ELS Visualization ===")
    
    # Request: Center on index 7 ('C', second occurrence), skip 5
    # Row width = 5.
    # Grid should look like:
    # A B C D E
    # A B C D E
    
    res = ELSVisualizer.generate_grid(text, center_index=7, skip=5, rows=4, cols=5)
    
    grid = res['grid']
    print(f"Generated Grid ({len(grid)} rows):")
    for row in grid:
        line = "".join([c['char'] for c in row])
        print(f"[{line}]")
        
    # Check if column 2 (index 0,1,2 'C') consists of 'C's
    # Note: center index 7 matches 'C'.
    # 7 // 5 = row 1, col 2.
    # So center of viewport should be 'C'.
    
    center_row_idx = len(grid) // 2
    center_col_idx = len(grid[0]) // 2
    
    center_char = grid[center_row_idx][center_col_idx]['char']
    print(f"Center Char (Row {center_row_idx}, Col {center_col_idx}): {center_char}")
    
    if center_char == 'C':
        print("SUCCESS: Center alignment correct.")
    else:
        print(f"FAILURE: Center alignment incorrect. Expected C, got {center_char}")
        
    # Check vertical alignment
    # All chars in column 2 should be 'C', ignoring padding
    col_correct = True
    for r in range(len(grid)):
        char = grid[r][2]['char']
        if char != 'C' and char != ' ': # Ignore padding
            col_correct = False
            print(f"Row {r} mismatch: got '{char}'")
    
    if col_correct:
         print("SUCCESS: Vertical alignment (skip) confirmed.")
    else:
         print("FAILURE: Vertical alignment issue.")

    # Test Path Calculation
    print("\n--- Testing Path Calculation ---")
    # Term "C" at index 7. Skip 5. Length 1.
    path = ELSVisualizer.get_term_path(start_index=7, skip=5, length=1, grid_config=res)
    print(f"Path for index 7: {path}")
    
    # Expected: Row 1, Col 2 (Relative to viewport) -> But wait, in viewport row 2 corresponds to C.
    # res['viewport'] logic:
    # Target Row (abs) = 1. Target Col = 2.
    # Start Row = -1. Start Col = 0.
    # Rel Row = 1 - (-1) = 2.
    # Rel Col = 2 - 0 = 2.
    # My manual calc said Row 2.
    # Let's verify.
    
    p = path[0]
    if p['grid_row'] == 2 and p['grid_col'] == 2 and p['visible']:
        print("SUCCESS: Path calculation correct.")
    else:
        print(f"FAILURE: Path calc mismatch. Got {p}")

if __name__ == "__main__":
    test_visualization()
