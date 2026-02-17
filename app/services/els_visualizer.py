"""
ELS Visualizer Service
Generates 2D character grids to visualize Equidistant Letter Sequence matches.
Standard "Bible Code" style visualization.
"""
from typing import List, Dict, Any, Tuple
import math

class ELSVisualizer:
    @staticmethod
    def generate_grid(
        text: str, 
        center_index: int, 
        skip: int, 
        rows: int = 20, 
        cols: int = 20
    ) -> Dict[str, Any]:
        """
        Generate a 2D grid of characters centered on a specific index.
        Display Logic:
        - The 'row length' is usually the 'skip' value (or similar) to make the sequence vertical.
        - Text wraps row by row.
        
        Args:
            text: The full source text (e.g. Torah).
            center_index: The index in the text to center the view on.
            skip: The skip value of the ELS pattern (often sets the row width).
            rows: Number of rows to display.
            cols: Number of columns to display.
            
        Returns:
            grid: List[List[str]] (2D array of chars)
            highlights: List[Dict] (Coordinates of highlighted cells)
            metadata: Info about the view.
        """
        text_len = len(text)
        
        # Default strategy: Row width = abs(skip)
        # This aligns the skip-sequence vertically.
        row_width = abs(skip)
        
        # If skip is massive (e.g. 5000), a grid of 5000 width is unusable.
        # But Bible Code software usually shows the wider grid or a zoomed window.
        # Here we generate a "viewport" window.
        
        # Calculate start index for the grid
        # We want (center_index) to be in the middle of our (rows x cols) viewport.
        # But coordinate space is: row = index // row_width, col = index % row_width.
        
        target_row = center_index // row_width
        target_col = center_index % row_width
        
        start_row = target_row - (rows // 2)
        start_col = target_col - (cols // 2)
        
        grid = []
        highlights = []
        
        for r in range(rows):
            current_row_idx = start_row + r
            row_chars = []
            
            for c in range(cols):
                current_col_idx = start_col + c
                
                # Calculate actual text index: (row * row_width) + col
                # Note: This logic assumes the "scroll" is continuous.
                
                # Handling "Column wrapping" for viewports?
                # Usually standard view is:
                # Text is broken into lines of length 'row_width'.
                # We view lines [start_row ... start_row+rows]
                # And columns [start_col ... start_col+cols]
                
                idx = (current_row_idx * row_width) + current_col_idx
                
                char = ""
                if 0 <= idx < text_len:
                    char = text[idx]
                else:
                    char = " " # Padding
                    
                row_chars.append({
                    "char": char,
                    "index": idx,
                    "row": r,
                })
                
            grid.append(row_chars)
            
        return {
            "grid": grid,
            "dimensions": {"rows": rows, "cols": cols, "row_width": row_width},
            "center_index": center_index,
            "skip": skip,
            "viewport": {
                "start_row": start_row,
                "start_col": start_col
            }
        }

    @staticmethod
    def get_term_path(
        start_index: int, 
        skip: int, 
        length: int, 
        grid_config: Dict[str, Any]
    ) -> List[Dict[str, int]]:
        """
        Calculate relative grid coordinates (row, col) for a specific term.
        """
        path = []
        row_width = grid_config['dimensions']['row_width']
        start_row = grid_config['viewport']['start_row']
        start_col = grid_config['viewport']['start_col']
        rows = grid_config['dimensions']['rows']
        cols = grid_config['dimensions']['cols']

        current_idx = start_index
        for _ in range(length):
            # Calculate absolute grid coordinates
            abs_row = current_idx // row_width
            abs_col = current_idx % row_width
            
            # Convert to relative viewport coordinates
            rel_row = abs_row - start_row
            rel_col = abs_col - start_col
            
            # Check if point is visible in viewport
            is_visible = (0 <= rel_row < rows) and (0 <= rel_col < cols)
            
            path.append({
                "index": current_idx,
                "grid_row": rel_row,
                "grid_col": rel_col,
                "visible": is_visible
            })
            
            current_idx += skip
        
        return path

    @staticmethod
    def generate_html_matrix(
        text: str, 
        center_index: int, 
        terms: List[Dict[str, Any]], 
        rows: int = 15, 
        cols: int = 15, 
        row_width: int = 12
    ) -> str:
        """
        Generate a standalone HTML string visualizing the matrix.
        Terms input format: [{"term": "ABC", "name": "Name", "skip": 12, "color": "#HEX"}]
        """
        center_row = center_index // row_width
        center_col = center_index % row_width
        
        start_row = center_row - (rows // 2)
        start_col = center_col - (cols // 2)
        
        # Identify highlights
        highlights = {}
        text_len = len(text)
        
        for t in terms:
            term_str = t["term"]
            skip = t["skip"]
            
            # Search around center to find the specific instance intended
            # This logic assumes the caller knows the terms converge here.
            # We search for the term pattern centered roughly on center_index
            search_start = max(0, center_index - 1000)
            search_end = min(text_len, center_index + 1000)
            
            for i in range(search_start, search_end):
                if text[i] == term_str[0]:
                    match = True
                    for k in range(len(term_str)):
                        if i + k*skip >= text_len or text[i + k*skip] != term_str[k]:
                            match = False
                            break
                    if match:
                        mid = i + (len(term_str)//2 * skip)
                        # Heuristic: Match if mid-point is within 150 chars of center
                        if abs(mid - center_index) < 150: 
                            for k in range(len(term_str)):
                                idx = i + k*skip
                                if idx not in highlights: highlights[idx] = []
                                highlights[idx].append(t)
                            break
                            
        # Generate HTML
        matrix_html = f'''
        <div class="matrix" style="display: grid; grid-template-columns: repeat({cols}, 40px); gap: 5px; background: #15151e; padding: 20px; border-radius: 10px; width: fit-content;">
        '''
        
        for r in range(rows):
            curr_row = start_row + r
            for c in range(cols):
                curr_col = start_col + c
                idx = (curr_row * row_width) + curr_col
                
                char = text[idx] if 0 <= idx < text_len else ""
                
                style = 'width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; font-size: 24px; color: #555; border-radius: 4px;'
                
                if idx in highlights:
                    # Use color of first term found at this index
                    t = highlights[idx][0]
                    style += f' background-color: {t["color"]}22; color: {t["color"]}; font-weight: bold; border: 1px solid {t["color"]}; text-shadow: 0 0 5px {t["color"]};'
                
                matrix_html += f'<div style="{style}" title="Index: {idx}">{char}</div>'
                
        matrix_html += '</div>'
        return matrix_html

    @staticmethod
    def generate_svg_matrix(
        text: str, 
        center_index: int, 
        terms: List[Dict[str, Any]], 
        rows: int = 15, 
        cols: int = 15, 
        row_width: int = 12
    ) -> str:
        """
        Generate a standalone SVG string visualizing the matrix.
        """
        cell_size = 40
        gap = 5
        padding = 40
        
        width = (cols * (cell_size + gap)) + (padding * 2)
        height = (rows * (cell_size + gap)) + (padding * 2)
        
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background-color: #0d0d14; font-family: monospace;">']
        
        # Grid calculation same as HTML
        center_row = center_index // row_width
        center_col = center_index % row_width
        start_row = center_row - (rows // 2)
        start_col = center_col - (cols // 2)

        # Highlight logic (duplicated for now to keep independent)
        highlights = {}
        text_len = len(text)
        for t in terms:
            term_str = t["term"]
            skip = t["skip"]
            search_start = max(0, center_index - 1000)
            search_end = min(text_len, center_index + 1000)
            for i in range(search_start, search_end):
                if text[i] == term_str[0]:
                    match = True
                    for k in range(len(term_str)):
                        if i + k*skip >= text_len or text[i + k*skip] != term_str[k]:
                            match = False; break
                    if match:
                        mid = i + (len(term_str)//2 * skip)
                        if abs(mid - center_index) < 150: 
                            for k in range(len(term_str)):
                                idx = i + k*skip
                                if idx not in highlights: highlights[idx] = []
                                highlights[idx].append(t)
                            break

        # Render cells
        for r in range(rows):
            for c in range(cols):
                curr_row = start_row + r
                curr_col = start_col + c
                idx = (curr_row * row_width) + curr_col
                
                x = padding + (c * (cell_size + gap))
                y = padding + (r * (cell_size + gap))
                
                char = text[idx] if 0 <= idx < text_len else ""
                
                fill = "#15151e"
                stroke = "#333333"
                text_fill = "#555555"
                stroke_width = "1"
                font_weight = "normal"
                
                if idx in highlights:
                    t = highlights[idx][0]
                    fill = t["color"]
                    stroke = t.get("stroke", t["color"]) # Use stroke if present, else color
                    text_fill = "#ffffff"
                    stroke_width = "2"
                    font_weight = "bold"
                
                svg.append(f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>')
                svg.append(f'<text x="{x + cell_size/2}" y="{y + cell_size/2 + 6}" text-anchor="middle" fill="{text_fill}" font-size="20" font-weight="{font_weight}">{char}</text>')
        
        svg.append('</svg>')
        return "\n".join(svg)
