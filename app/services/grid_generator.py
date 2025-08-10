"""
Grid Generation Service for Ancient Text Analyzer
Creates customizable text grids for cryptographic pattern analysis
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass
from enum import Enum
import re
import math

logger = logging.getLogger(__name__)

@dataclass
class GridCell:
    """Container for grid cell data"""
    character: str
    row: int
    column: int
    source_position: Optional[int] = None  # Position in original text
    original_character: Optional[str] = None  # Before normalization

@dataclass
class TextGrid:
    """Container for text grid with metadata"""
    rows: int
    columns: int
    cells: List[List[str]]
    source_text: str
    options: Dict[str, bool]
    character_positions: Dict[Tuple[int, int], int]  # Grid pos -> source pos
    total_characters: int
    filtered_characters: int

@dataclass
class GridOptions:
    """Configuration options for grid generation"""
    include_spaces: bool = True
    include_punctuation: bool = True
    include_numbers: bool = True
    normalize_case: bool = False
    remove_accents: bool = False
    custom_filter: Optional[str] = None  # Regex pattern for custom filtering

@dataclass
class PatternMatch:
    """Container for pattern search results"""
    pattern: str
    start_position: Tuple[int, int]
    end_position: Tuple[int, int]
    path: List[Tuple[int, int]]
    confidence: float
    direction: str
    context_before: str
    context_after: str
    is_connected: bool

@dataclass
class ConnectionRules:
    """Rules for connected sequence detection"""
    allow_horizontal: bool = True
    allow_vertical: bool = True
    allow_diagonal: bool = False
    max_gap: int = 0  # Maximum gap between characters
    require_continuous: bool = True
    case_sensitive: bool = False

class GridGenerator:
    """
    Advanced grid generator for text analysis
    Supports flexible filtering and multiple grid configurations
    """
    
    def __init__(self):
        self.punctuation_chars = set('.,;:!?()[]{}"\'-/')
        self.number_chars = set('0123456789')
        self.space_chars = set(' \t\n\r')
        
    def create_grid(self, text: str, rows: int, cols: int, options: GridOptions = None) -> TextGrid:
        """
        Create a text grid with specified dimensions and options
        
        Args:
            text: Source text to convert to grid
            rows: Number of grid rows
            cols: Number of grid columns
            options: Grid generation options
            
        Returns:
            TextGrid object with grid data and metadata
        """
        if options is None:
            options = GridOptions()
        
        logger.info(f"Creating {rows}x{cols} grid from {len(text)} characters")
        
        try:
            # Filter and process text
            filtered_text, char_positions = self._filter_text(text, options)
            
            if not filtered_text:
                raise ValueError("No characters remaining after filtering")
            
            # Calculate grid size needed
            total_cells = rows * cols
            if len(filtered_text) > total_cells:
                logger.warning(f"Text length ({len(filtered_text)}) exceeds grid capacity ({total_cells}). Text will be truncated.")
                filtered_text = filtered_text[:total_cells]
                char_positions = char_positions[:total_cells]
            
            # Create grid cells
            cells = self._create_grid_cells(filtered_text, rows, cols)
            
            # Create position mapping
            position_mapping = {}
            for i, source_pos in enumerate(char_positions):
                row = i // cols
                col = i % cols
                if row < rows:  # Ensure we don't exceed grid bounds
                    position_mapping[(row, col)] = source_pos
            
            grid = TextGrid(
                rows=rows,
                columns=cols,
                cells=cells,
                source_text=text,
                options={
                    'include_spaces': options.include_spaces,
                    'include_punctuation': options.include_punctuation,
                    'include_numbers': options.include_numbers,
                    'normalize_case': options.normalize_case,
                    'remove_accents': options.remove_accents
                },
                character_positions=position_mapping,
                total_characters=len(text),
                filtered_characters=len(filtered_text)
            )
            
            logger.info(f"Grid created successfully: {len(filtered_text)} characters placed in {rows}x{cols} grid")
            return grid
            
        except Exception as e:
            logger.error(f"Grid creation failed: {e}")
            raise
    
    def _filter_text(self, text: str, options: GridOptions) -> Tuple[str, List[int]]:
        """
        Filter text based on options and return filtered text with position mapping
        
        Returns:
            Tuple of (filtered_text, list_of_original_positions)
        """
        filtered_chars = []
        positions = []
        
        for i, char in enumerate(text):
            include_char = True
            
            # Check space filtering
            if char in self.space_chars and not options.include_spaces:
                include_char = False
            
            # Check punctuation filtering
            elif char in self.punctuation_chars and not options.include_punctuation:
                include_char = False
            
            # Check number filtering
            elif char in self.number_chars and not options.include_numbers:
                include_char = False
            
            # Apply custom filter if provided
            elif options.custom_filter and not re.match(options.custom_filter, char):
                include_char = False
            
            if include_char:
                # Apply transformations
                processed_char = char
                
                if options.normalize_case:
                    processed_char = processed_char.lower()
                
                if options.remove_accents:
                    processed_char = self._remove_accents(processed_char)
                
                filtered_chars.append(processed_char)
                positions.append(i)
        
        return ''.join(filtered_chars), positions
    
    def _remove_accents(self, text: str) -> str:
        """
        Remove accents from text (simplified implementation)
        """
        # This is a basic implementation - in production you'd use a proper library
        accent_map = {
            'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a', 'ā': 'a', 'ã': 'a',
            'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e', 'ē': 'e',
            'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i', 'ī': 'i',
            'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', 'ō': 'o', 'õ': 'o',
            'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u', 'ū': 'u',
            'ñ': 'n', 'ç': 'c'
        }
        
        result = []
        for char in text:
            result.append(accent_map.get(char.lower(), char))
        
        return ''.join(result)
    
    def _create_grid_cells(self, text: str, rows: int, cols: int) -> List[List[str]]:
        """
        Create 2D grid from filtered text
        """
        cells = []
        
        for row in range(rows):
            row_cells = []
            for col in range(cols):
                index = row * cols + col
                if index < len(text):
                    row_cells.append(text[index])
                else:
                    row_cells.append('')  # Empty cell if text is shorter than grid
            cells.append(row_cells)
        
        return cells
    
    def get_grid_text(self, grid: TextGrid, include_spaces: bool = True) -> str:
        """
        Convert grid back to text format
        """
        lines = []
        for row in grid.cells:
            if include_spaces:
                lines.append(' '.join(row))
            else:
                lines.append(''.join(row))
        
        return '\n'.join(lines)
    
    def get_grid_statistics(self, grid: TextGrid) -> Dict[str, Any]:
        """
        Calculate statistics for the grid
        """
        # Count non-empty cells
        non_empty_cells = 0
        character_counts = {}
        
        for row in grid.cells:
            for cell in row:
                if cell:
                    non_empty_cells += 1
                    character_counts[cell] = character_counts.get(cell, 0) + 1
        
        # Calculate fill percentage
        total_cells = grid.rows * grid.columns
        fill_percentage = (non_empty_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Find most common characters
        most_common = sorted(character_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_cells': total_cells,
            'filled_cells': non_empty_cells,
            'empty_cells': total_cells - non_empty_cells,
            'fill_percentage': fill_percentage,
            'unique_characters': len(character_counts),
            'most_common_characters': most_common,
            'character_distribution': character_counts,
            'compression_ratio': len(grid.source_text) / grid.filtered_characters if grid.filtered_characters > 0 else 0
        }
    
    def search_patterns(self, grid: TextGrid, pattern: str, rules: ConnectionRules = None) -> List[PatternMatch]:
        """
        Search for patterns in the grid following connection rules
        
        Args:
            grid: TextGrid to search in
            pattern: Pattern string to find
            rules: Connection rules for pattern detection
            
        Returns:
            List of PatternMatch objects
        """
        if rules is None:
            rules = ConnectionRules()
        
        if not pattern:
            return []
        
        logger.info(f"Searching for pattern '{pattern}' in {grid.rows}x{grid.columns} grid")
        
        directions = {
            'horizontal_forward': (0, 1),
            'horizontal_backward': (0, -1),
            'vertical_down': (1, 0),
            'vertical_up': (-1, 0),
            'diagonal_down_right': (1, 1),
            'diagonal_down_left': (1, -1),
            'diagonal_up_right': (-1, 1),
            'diagonal_up_left': (-1, -1)
        }
        
        matches = []
        
        # Search in all allowed directions
        for direction_name, direction_vector in directions.items():
            if not self._is_direction_allowed(direction_name, rules):
                continue
            
            direction_matches = self._search_direction(grid, pattern, direction_vector, direction_name, rules)
            matches.extend(direction_matches)
        
        # Remove duplicates and sort by confidence
        unique_matches = self._remove_duplicate_matches(matches)
        unique_matches.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Found {len(unique_matches)} unique matches for pattern '{pattern}'")
        return unique_matches
    
    def _is_direction_allowed(self, direction_name: str, rules: ConnectionRules) -> bool:
        """Check if search direction is allowed by rules"""
        if 'horizontal' in direction_name:
            return rules.allow_horizontal
        elif 'vertical' in direction_name:
            return rules.allow_vertical
        elif 'diagonal' in direction_name:
            return rules.allow_diagonal
        return False
    
    def _search_direction(self, grid: TextGrid, pattern: str, direction: Tuple[int, int], 
                         direction_name: str, rules: ConnectionRules) -> List[PatternMatch]:
        """
        Search for pattern in a specific direction
        """
        matches = []
        rows, cols = grid.rows, grid.columns
        
        # Try starting from each cell
        for start_row in range(rows):
            for start_col in range(cols):
                match = self._try_match_from_position(
                    grid, pattern, start_row, start_col, direction, direction_name, rules
                )
                if match:
                    matches.append(match)
        
        return matches
    
    def _try_match_from_position(self, grid: TextGrid, pattern: str, start_row: int, start_col: int,
                                direction: Tuple[int, int], direction_name: str, 
                                rules: ConnectionRules) -> Optional[PatternMatch]:
        """
        Try to match pattern starting from a specific position
        """
        dr, dc = direction
        path = []
        
        row, col = start_row, start_col
        
        for i, target_char in enumerate(pattern):
            # Check bounds
            if not (0 <= row < grid.rows and 0 <= col < grid.columns):
                return None
            
            cell_char = grid.cells[row][col]
            
            # Check character match
            if not self._characters_match(cell_char, target_char, rules.case_sensitive):
                return None
            
            path.append((row, col))
            
            # Move to next position (except for last character)
            if i < len(pattern) - 1:
                row += dr
                col += dc
        
        # Calculate confidence
        confidence = self._calculate_match_confidence(grid, path, pattern, direction_name)
        
        # Get context
        context_before, context_after = self._get_match_context(grid, path, direction)
        
        return PatternMatch(
            pattern=pattern,
            start_position=(start_row, start_col),
            end_position=(row, col),
            path=path,
            confidence=confidence,
            direction=direction_name,
            context_before=context_before,
            context_after=context_after,
            is_connected=rules.require_continuous
        )
    
    def _characters_match(self, cell_char: str, target_char: str, case_sensitive: bool) -> bool:
        """Check if two characters match according to rules"""
        if not cell_char:  # Empty cell
            return False
        
        if case_sensitive:
            return cell_char == target_char
        else:
            return cell_char.lower() == target_char.lower()
    
    def _calculate_match_confidence(self, grid: TextGrid, path: List[Tuple[int, int]], 
                                  pattern: str, direction: str) -> float:
        """
        Calculate confidence score for a pattern match
        """
        base_confidence = 0.8  # Base confidence for any match
        
        # Bonus for longer patterns
        length_bonus = min(len(pattern) * 0.02, 0.15)
        
        # Bonus for straight-line patterns
        direction_bonus = 0.05 if 'horizontal' in direction or 'vertical' in direction else 0.0
        
        confidence = base_confidence + length_bonus + direction_bonus
        return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
    
    def _get_match_context(self, grid: TextGrid, path: List[Tuple[int, int]], 
                          direction: Tuple[int, int]) -> Tuple[str, str]:
        """
        Get context characters before and after the match
        """
        return "", ""  # Simplified for now
    
    def _remove_duplicate_matches(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """
        Remove duplicate matches (same path, different direction names)
        """
        unique_matches = []
        seen_paths = set()
        
        for match in matches:
            path_key = tuple(match.path)
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                unique_matches.append(match)
        
        return unique_matches
    
    def find_connected_sequences(self, grid: TextGrid, rules: ConnectionRules = None) -> List[PatternMatch]:
        """
        Find all connected sequences in the grid following cryptographic rules
        """
        if rules is None:
            rules = ConnectionRules()
        
        logger.info(f"Finding connected sequences in {grid.rows}x{grid.columns} grid")
        
        sequences = []
        # Simplified implementation - just return horizontal sequences for now
        
        for row in range(grid.rows):
            sequence = ""
            start_col = 0
            
            for col in range(grid.columns):
                if grid.cells[row][col]:
                    sequence += grid.cells[row][col]
                else:
                    if len(sequence) > 1:
                        path = [(row, start_col + i) for i in range(len(sequence))]
                        match = PatternMatch(
                            pattern=sequence,
                            start_position=(row, start_col),
                            end_position=(row, start_col + len(sequence) - 1),
                            path=path,
                            confidence=0.7,
                            direction="horizontal_sequence",
                            context_before="",
                            context_after="",
                            is_connected=True
                        )
                        sequences.append(match)
                    sequence = ""
                    start_col = col + 1
            
            # Handle sequence at end of row
            if len(sequence) > 1:
                path = [(row, start_col + i) for i in range(len(sequence))]
                match = PatternMatch(
                    pattern=sequence,
                    start_position=(row, start_col),
                    end_position=(row, start_col + len(sequence) - 1),
                    path=path,
                    confidence=0.7,
                    direction="horizontal_sequence",
                    context_before="",
                    context_after="",
                    is_connected=True
                )
                sequences.append(match)
        
        logger.info(f"Found {len(sequences)} connected sequences")
        return sequences
    
    def create_multiple_grids(self, text: str, grid_configs: List[Dict[str, Any]]) -> List[TextGrid]:
        """
        Create multiple grids with different configurations
        
        Args:
            text: Source text
            grid_configs: List of grid configuration dictionaries
                         Each should contain 'rows', 'cols', and optionally 'options'
        
        Returns:
            List of TextGrid objects
        """
        grids = []
        
        for config in grid_configs:
            try:
                rows = config['rows']
                cols = config['cols']
                options = config.get('options', GridOptions())
                
                grid = self.create_grid(text, rows, cols, options)
                grids.append(grid)
                
            except Exception as e:
                logger.error(f"Failed to create grid with config {config}: {e}")
                continue
        
        logger.info(f"Created {len(grids)} grids from {len(grid_configs)} configurations")
        return grids
    
    def validate_grid_dimensions(self, text: str, rows: int, cols: int, options: GridOptions = None) -> Dict[str, Any]:
        """
        Validate grid dimensions against text length and provide recommendations
        """
        if options is None:
            options = GridOptions()
        
        # Filter text to get actual character count
        filtered_text, _ = self._filter_text(text, options)
        filtered_length = len(filtered_text)
        
        total_cells = rows * cols
        
        validation = {
            'is_valid': True,
            'warnings': [],
            'recommendations': [],
            'text_length': len(text),
            'filtered_length': filtered_length,
            'grid_capacity': total_cells,
            'utilization': (filtered_length / total_cells) * 100 if total_cells > 0 else 0
        }
        
        # Check if grid is too small
        if filtered_length > total_cells:
            validation['is_valid'] = False
            validation['warnings'].append(f"Text length ({filtered_length}) exceeds grid capacity ({total_cells})")
            validation['recommendations'].append(f"Increase grid size to at least {math.ceil(math.sqrt(filtered_length))}x{math.ceil(math.sqrt(filtered_length))}")
        
        # Check if grid is much larger than needed
        elif total_cells > filtered_length * 2:
            validation['warnings'].append(f"Grid is much larger than needed (only {validation['utilization']:.1f}% utilized)")
            optimal_size = math.ceil(math.sqrt(filtered_length))
            validation['recommendations'].append(f"Consider using a smaller grid like {optimal_size}x{optimal_size}")
        
        # Check for very unbalanced dimensions
        aspect_ratio = max(rows, cols) / min(rows, cols)
        if aspect_ratio > 5:
            validation['warnings'].append(f"Grid has extreme aspect ratio ({rows}:{cols})")
            validation['recommendations'].append("Consider more balanced dimensions for better pattern detection")
        
        return validation
    
    def suggest_optimal_dimensions(self, text: str, options: GridOptions = None) -> List[Tuple[int, int]]:
        """
        Suggest optimal grid dimensions based on text length
        """
        if options is None:
            options = GridOptions()
        
        filtered_text, _ = self._filter_text(text, options)
        length = len(filtered_text)
        
        if length == 0:
            return [(1, 1)]
        
        suggestions = []
        
        # Perfect square
        sqrt_length = math.sqrt(length)
        if sqrt_length == int(sqrt_length):
            size = int(sqrt_length)
            suggestions.append((size, size))
        else:
            # Closest squares
            lower_size = int(sqrt_length)
            upper_size = lower_size + 1
            suggestions.append((lower_size, math.ceil(length / lower_size)))
            suggestions.append((upper_size, math.ceil(length / upper_size)))
        
        # Common cryptographic grid sizes
        common_sizes = [
            (12, 26),  # Half alphabet
            (13, 24),  # Common in historical ciphers
            (24, 24),  # Square grid for larger texts
            (26, 26),  # Full alphabet square
        ]
        
        for rows, cols in common_sizes:
            if rows * cols >= length:
                suggestions.append((rows, cols))
        
        # Remove duplicates and sort by total size
        unique_suggestions = list(set(suggestions))
        unique_suggestions.sort(key=lambda x: x[0] * x[1])
        
        return unique_suggestions[:5]  # Return top 5 suggestions
    
    def export_grid(self, grid: TextGrid, format_type: str = "csv") -> str:
        """
        Export grid in various formats
        
        Args:
            grid: TextGrid to export
            format_type: Export format ("csv", "json", "txt", "html")
        
        Returns:
            Formatted string representation
        """
        if format_type.lower() == "csv":
            return self._export_csv(grid)
        elif format_type.lower() == "json":
            return self._export_json(grid)
        elif format_type.lower() == "txt":
            return self._export_txt(grid)
        elif format_type.lower() == "html":
            return self._export_html(grid)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_csv(self, grid: TextGrid) -> str:
        """Export grid as CSV"""
        lines = []
        for row in grid.cells:
            # Escape commas and quotes in cell values
            escaped_row = []
            for cell in row:
                if ',' in cell or '"' in cell:
                    escaped_row.append(f'"{cell.replace('"', '""')}"')
                else:
                    escaped_row.append(cell)
            lines.append(','.join(escaped_row))
        
        return '\n'.join(lines)
    
    def _export_json(self, grid: TextGrid) -> str:
        """Export grid as JSON"""
        import json
        
        grid_data = {
            'rows': grid.rows,
            'columns': grid.columns,
            'cells': grid.cells,
            'options': grid.options,
            'statistics': self.get_grid_statistics(grid)
        }
        
        return json.dumps(grid_data, indent=2)
    
    def _export_txt(self, grid: TextGrid) -> str:
        """Export grid as formatted text"""
        lines = [f"Grid: {grid.rows}x{grid.columns}"]
        lines.append("=" * 40)
        
        for i, row in enumerate(grid.cells):
            line = f"{i:2d}: " + " ".join(f"{cell:>2}" for cell in row)
            lines.append(line)
        
        return '\n'.join(lines)
    
    def _export_html(self, grid: TextGrid) -> str:
        """Export grid as HTML table"""
        html = ['<table border="1" cellpadding="5" cellspacing="0">']
        
        for row in grid.cells:
            html.append('  <tr>')
            for cell in row:
                html.append(f'    <td>{cell if cell else "&nbsp;"}</td>')
            html.append('  </tr>')
        
        html.append('</table>')
        
        return '\n'.join(html)

@dataclass

class CombinedGrid:
    """Container for combined grid analysis results"""
    grids: List[TextGrid]
    combination_method: str
    combined_dimensions: Tuple[int, int]
    alignment_strategy: str
    cross_grid_patterns: List[Dict[str, Any]]
    correlation_score: float

class GridCombiner:
    """
    Advanced grid combination and cross-analysis capabilities
    Supports multiple grid alignment and pattern correlation strategies
    """
    
    def __init__(self, generator: 'GridGenerator'):
        self.generator = generator
    
    def combine_grids(self, grids: List[TextGrid], method: str = "horizontal", 
                     alignment: str = "top") -> CombinedGrid:
        """
        Combine multiple grids using specified method and alignment
        
        Args:
            grids: List of TextGrid objects to combine
            method: Combination method ("horizontal", "vertical", "overlay", "interleave")
            alignment: Alignment strategy ("top", "center", "bottom", "left", "right")
            
        Returns:
            CombinedGrid object with combined analysis
        """
        if not grids:
            raise ValueError("No grids provided for combination")
        
        logger.info(f"Combining {len(grids)} grids using {method} method with {alignment} alignment")
        
        if method == "horizontal":
            combined_result = self._combine_horizontal(grids, alignment)
        elif method == "vertical":
            combined_result = self._combine_vertical(grids, alignment)
        elif method == "overlay":
            combined_result = self._combine_overlay(grids, alignment)
        elif method == "interleave":
            combined_result = self._combine_interleave(grids, alignment)
        else:
            raise ValueError(f"Unsupported combination method: {method}")
        
        # Analyze cross-grid patterns
        cross_patterns = self._analyze_cross_grid_patterns(grids, combined_result)
        
        # Calculate correlation score
        correlation_score = self._calculate_grid_correlation(grids)
        
        return CombinedGrid(
            grids=grids,
            combination_method=method,
            combined_dimensions=combined_result['dimensions'],
            alignment_strategy=alignment,
            cross_grid_patterns=cross_patterns,
            correlation_score=correlation_score
        )
    
    def _combine_horizontal(self, grids: List[TextGrid], alignment: str) -> Dict[str, Any]:
        """
        Combine grids horizontally (side by side)
        """
        if not grids:
            return {'dimensions': (0, 0), 'combined_cells': []}
        
        # Calculate combined dimensions
        max_rows = max(grid.rows for grid in grids)
        total_cols = sum(grid.columns for grid in grids)
        
        # Create combined cell matrix
        combined_cells = []
        
        for row in range(max_rows):
            combined_row = []
            
            for grid in grids:
                # Handle alignment for grids with different heights
                if alignment == "top":
                    grid_row = row if row < grid.rows else grid.rows - 1
                elif alignment == "center":
                    offset = (max_rows - grid.rows) // 2
                    grid_row = row - offset if offset <= row < offset + grid.rows else 0
                elif alignment == "bottom":
                    offset = max_rows - grid.rows
                    grid_row = row - offset if row >= offset else 0
                else:
                    grid_row = min(row, grid.rows - 1)
                
                # Add cells from this grid to the combined row
                if 0 <= grid_row < grid.rows:
                    combined_row.extend(grid.cells[grid_row])
                else:
                    combined_row.extend([''] * grid.columns)  # Empty cells for padding
            
            combined_cells.append(combined_row)
        
        return {
            'dimensions': (max_rows, total_cols),
            'combined_cells': combined_cells
        }
    
    def _combine_vertical(self, grids: List[TextGrid], alignment: str) -> Dict[str, Any]:
        """
        Combine grids vertically (stacked)
        """
        if not grids:
            return {'dimensions': (0, 0), 'combined_cells': []}
        
        # Calculate combined dimensions
        total_rows = sum(grid.rows for grid in grids)
        max_cols = max(grid.columns for grid in grids)
        
        # Create combined cell matrix
        combined_cells = []
        
        for grid in grids:
            for row in range(grid.rows):
                combined_row = []
                
                # Handle alignment for grids with different widths
                if alignment == "left":
                    combined_row.extend(grid.cells[row])
                    combined_row.extend([''] * (max_cols - grid.columns))  # Right padding
                elif alignment == "center":
                    left_padding = (max_cols - grid.columns) // 2
                    right_padding = max_cols - grid.columns - left_padding
                    combined_row.extend([''] * left_padding)
                    combined_row.extend(grid.cells[row])
                    combined_row.extend([''] * right_padding)
                elif alignment == "right":
                    combined_row.extend([''] * (max_cols - grid.columns))  # Left padding
                    combined_row.extend(grid.cells[row])
                else:
                    combined_row.extend(grid.cells[row])
                    if len(combined_row) < max_cols:
                        combined_row.extend([''] * (max_cols - len(combined_row)))
                
                combined_cells.append(combined_row)
        
        return {
            'dimensions': (total_rows, max_cols),
            'combined_cells': combined_cells
        }
    
    def _combine_overlay(self, grids: List[TextGrid], alignment: str) -> Dict[str, Any]:
        """
        Combine grids by overlaying them (superimposition)
        """
        if not grids:
            return {'dimensions': (0, 0), 'combined_cells': []}
        
        # Use dimensions of the largest grid
        max_rows = max(grid.rows for grid in grids)
        max_cols = max(grid.columns for grid in grids)
        
        # Create combined cell matrix
        combined_cells = [['' for _ in range(max_cols)] for _ in range(max_rows)]
        
        for grid in grids:
            # Calculate offset based on alignment
            if alignment == "center":
                row_offset = (max_rows - grid.rows) // 2
                col_offset = (max_cols - grid.columns) // 2
            elif alignment == "top":
                row_offset = 0
                col_offset = (max_cols - grid.columns) // 2
            elif alignment == "bottom":
                row_offset = max_rows - grid.rows
                col_offset = (max_cols - grid.columns) // 2
            elif alignment == "left":
                row_offset = (max_rows - grid.rows) // 2
                col_offset = 0
            elif alignment == "right":
                row_offset = (max_rows - grid.rows) // 2
                col_offset = max_cols - grid.columns
            else:
                row_offset = 0
                col_offset = 0
            
            # Overlay this grid onto the combined matrix
            for row in range(grid.rows):
                for col in range(grid.columns):
                    target_row = row + row_offset
                    target_col = col + col_offset
                    
                    if (0 <= target_row < max_rows and 0 <= target_col < max_cols):
                        cell_value = grid.cells[row][col]
                        if cell_value:  # Only overlay non-empty cells
                            if combined_cells[target_row][target_col]:
                                # Handle collision - combine characters
                                combined_cells[target_row][target_col] += f"/{cell_value}"
                            else:
                                combined_cells[target_row][target_col] = cell_value
        
        return {
            'dimensions': (max_rows, max_cols),
            'combined_cells': combined_cells
        }
    
    def _combine_interleave(self, grids: List[TextGrid], alignment: str) -> Dict[str, Any]:
        """
        Combine grids by interleaving their rows or columns
        """
        if not grids:
            return {'dimensions': (0, 0), 'combined_cells': []}
        
        # For simplicity, interleave rows
        max_cols = max(grid.columns for grid in grids)
        combined_cells = []
        
        # Find the maximum number of rows across all grids
        max_grid_rows = max(grid.rows for grid in grids)
        
        # Interleave rows from each grid
        for row_idx in range(max_grid_rows):
            for grid in grids:
                if row_idx < grid.rows:
                    row = grid.cells[row_idx][:]
                    # Pad row to max_cols if necessary
                    while len(row) < max_cols:
                        row.append('')
                    combined_cells.append(row)
        
        total_rows = len(combined_cells)
        
        return {
            'dimensions': (total_rows, max_cols),
            'combined_cells': combined_cells
        }
    
    def _analyze_cross_grid_patterns(self, grids: List[TextGrid], 
                                   combined_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze patterns that span across multiple grids
        """
        cross_patterns = []
        
        # Look for patterns that appear in multiple grids
        all_patterns = {}
        
        for i, grid in enumerate(grids):
            # Find patterns in this grid
            sequences = self.generator.find_connected_sequences(grid)
            
            for sequence in sequences:
                pattern_text = sequence.pattern
                if pattern_text not in all_patterns:
                    all_patterns[pattern_text] = []
                
                all_patterns[pattern_text].append({
                    'grid_index': i,
                    'sequence': sequence,
                    'grid_id': id(grid)
                })
        
        # Find patterns that appear in multiple grids
        for pattern_text, occurrences in all_patterns.items():
            if len(occurrences) > 1:  # Pattern appears in multiple grids
                cross_pattern = {
                    'pattern': pattern_text,
                    'occurrences': len(occurrences),
                    'grids_involved': [occ['grid_index'] for occ in occurrences],
                    'positions': [occ['sequence'].start_position for occ in occurrences],
                    'average_confidence': sum(occ['sequence'].confidence for occ in occurrences) / len(occurrences),
                    'pattern_type': 'cross_grid_repetition'
                }
                cross_patterns.append(cross_pattern)
        
        # Look for complementary patterns (patterns that complete each other across grids)
        cross_patterns.extend(self._find_complementary_patterns(grids))
        
        return cross_patterns
    
    def _find_complementary_patterns(self, grids: List[TextGrid]) -> List[Dict[str, Any]]:
        """
        Find patterns that complement each other across different grids
        """
        complementary_patterns = []
        
        # Simple implementation: look for patterns that when combined form known words
        if len(grids) >= 2:
            grid1_sequences = self.generator.find_connected_sequences(grids[0])
            grid2_sequences = self.generator.find_connected_sequences(grids[1])
            
            for seq1 in grid1_sequences:
                for seq2 in grid2_sequences:
                    # Check if combining these sequences forms a meaningful pattern
                    combined_pattern = seq1.pattern + seq2.pattern
                    
                    if self._is_meaningful_combination(combined_pattern):
                        complementary_pattern = {
                            'pattern': combined_pattern,
                            'component_patterns': [seq1.pattern, seq2.pattern],
                            'grid_indices': [0, 1],
                            'combined_confidence': (seq1.confidence + seq2.confidence) / 2,
                            'pattern_type': 'complementary_combination'
                        }
                        complementary_patterns.append(complementary_pattern)
        
        return complementary_patterns
    
    def _is_meaningful_combination(self, pattern: str) -> bool:
        """
        Simple heuristic to determine if a combined pattern is meaningful
        """
        # Basic checks for meaningful combinations
        if len(pattern) < 3:
            return False
        
        # Check for reasonable letter distribution
        vowels = set('aeiouAEIOU')
        vowel_count = sum(1 for c in pattern if c in vowels)
        consonant_count = sum(1 for c in pattern if c.isalpha() and c not in vowels)
        
        if vowel_count == 0 or consonant_count == 0:
            return False
        
        # Check for reasonable vowel/consonant ratio
        if len(pattern) > 0:
            vowel_ratio = vowel_count / len(pattern)
            return 0.1 <= vowel_ratio <= 0.6
        
        return False
    
    def _calculate_grid_correlation(self, grids: List[TextGrid]) -> float:
        """
        Calculate correlation score between grids
        """
        if len(grids) < 2:
            return 0.0
        
        total_correlation = 0.0
        comparisons = 0
        
        # Compare each pair of grids
        for i in range(len(grids)):
            for j in range(i + 1, len(grids)):
                correlation = self._calculate_pairwise_correlation(grids[i], grids[j])
                total_correlation += correlation
                comparisons += 1
        
        return total_correlation / comparisons if comparisons > 0 else 0.0
    
    def _calculate_pairwise_correlation(self, grid1: TextGrid, grid2: TextGrid) -> float:
        """
        Calculate correlation between two grids
        """
        # Simple correlation based on character frequency similarity
        stats1 = self.generator.get_grid_statistics(grid1)
        stats2 = self.generator.get_grid_statistics(grid2)
        
        dist1 = stats1['character_distribution']
        dist2 = stats2['character_distribution']
        
        # Get all unique characters from both grids
        all_chars = set(dist1.keys()) | set(dist2.keys())
        
        if not all_chars:
            return 0.0
        
        # Calculate frequency vectors
        freq1 = [dist1.get(char, 0) for char in all_chars]
        freq2 = [dist2.get(char, 0) for char in all_chars]
        
        # Normalize frequencies
        total1 = sum(freq1) if sum(freq1) > 0 else 1
        total2 = sum(freq2) if sum(freq2) > 0 else 1
        
        norm_freq1 = [f / total1 for f in freq1]
        norm_freq2 = [f / total2 for f in freq2]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(norm_freq1, norm_freq2))
        magnitude1 = math.sqrt(sum(a * a for a in norm_freq1))
        magnitude2 = math.sqrt(sum(a * a for a in norm_freq2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def analyze_grid_relationships(self, grids: List[TextGrid]) -> Dict[str, Any]:
        """
        Comprehensive analysis of relationships between multiple grids
        """
        if len(grids) < 2:
            return {"error": "Need at least 2 grids for relationship analysis"}
        
        analysis = {
            'grid_count': len(grids),
            'dimension_analysis': self._analyze_grid_dimensions(grids),
            'content_similarity': self._analyze_content_similarity(grids),
            'pattern_overlap': self._analyze_pattern_overlap(grids),
            'structural_relationships': self._analyze_structural_relationships(grids),
            'optimal_combinations': self._suggest_optimal_combinations(grids)
        }
        
        return analysis
    
    def _analyze_grid_dimensions(self, grids: List[TextGrid]) -> Dict[str, Any]:
        """Analyze dimensional relationships between grids"""
        dimensions = [(grid.rows, grid.columns) for grid in grids]
        
        return {
            'dimensions': dimensions,
            'uniform_size': len(set(dimensions)) == 1,
            'total_cells': [rows * cols for rows, cols in dimensions],
            'aspect_ratios': [cols / rows if rows > 0 else 0 for rows, cols in dimensions],
            'size_relationships': self._find_size_relationships(dimensions)
        }
    
    def _find_size_relationships(self, dimensions: List[Tuple[int, int]]) -> List[str]:
        """Find mathematical relationships between grid dimensions"""
        relationships = []
        
        for i, (r1, c1) in enumerate(dimensions):
            for j, (r2, c2) in enumerate(dimensions[i+1:], i+1):
                # Check for simple ratios
                if r1 * c1 == r2 * c2:
                    relationships.append(f"Grid {i} and {j} have equal areas")
                elif r1 == r2 and c1 * 2 == c2:
                    relationships.append(f"Grid {j} is twice as wide as Grid {i}")
                elif r1 * 2 == r2 and c1 == c2:
                    relationships.append(f"Grid {j} is twice as tall as Grid {i}")
        
        return relationships
    
    def _analyze_content_similarity(self, grids: List[TextGrid]) -> Dict[str, Any]:
        """Analyze content similarity between grids"""
        similarities = {}
        
        for i in range(len(grids)):
            for j in range(i + 1, len(grids)):
                similarity = self._calculate_pairwise_correlation(grids[i], grids[j])
                similarities[f"grid_{i}_vs_{j}"] = similarity
        
        return {
            'pairwise_similarities': similarities,
            'average_similarity': sum(similarities.values()) / len(similarities) if similarities else 0.0,
            'most_similar_pair': max(similarities.items(), key=lambda x: x[1]) if similarities else None,
            'least_similar_pair': min(similarities.items(), key=lambda x: x[1]) if similarities else None
        }
    
    def _analyze_pattern_overlap(self, grids: List[TextGrid]) -> Dict[str, Any]:
        """Analyze pattern overlap between grids"""
        all_patterns = {}
        
        for i, grid in enumerate(grids):
            sequences = self.generator.find_connected_sequences(grid)
            for seq in sequences:
                pattern = seq.pattern
                if pattern not in all_patterns:
                    all_patterns[pattern] = []
                all_patterns[pattern].append(i)
        
        # Find patterns that appear in multiple grids
        shared_patterns = {pattern: grids for pattern, grids in all_patterns.items() if len(grids) > 1}
        
        return {
            'total_unique_patterns': len(all_patterns),
            'shared_patterns': len(shared_patterns),
            'shared_pattern_details': shared_patterns,
            'pattern_sharing_ratio': len(shared_patterns) / len(all_patterns) if all_patterns else 0.0
        }
    
    def _analyze_structural_relationships(self, grids: List[TextGrid]) -> Dict[str, Any]:
        """Analyze structural relationships between grids"""
        structures = []
        
        for grid in grids:
            stats = self.generator.get_grid_statistics(grid)
            structure = {
                'fill_percentage': stats['fill_percentage'],
                'unique_characters': stats['unique_characters'],
                'character_diversity': stats['unique_characters'] / stats['filled_cells'] if stats['filled_cells'] > 0 else 0
            }
            structures.append(structure)
        
        return {
            'individual_structures': structures,
            'structural_consistency': self._calculate_structural_consistency(structures)
        }
    
    def _calculate_structural_consistency(self, structures: List[Dict[str, Any]]) -> float:
        """Calculate how structurally consistent the grids are"""
        if len(structures) < 2:
            return 1.0
        
        # Calculate variance in key structural metrics
        fill_percentages = [s['fill_percentage'] for s in structures]
        diversities = [s['character_diversity'] for s in structures]
        
        fill_variance = self._calculate_variance(fill_percentages)
        diversity_variance = self._calculate_variance(diversities)
        
        # Lower variance = higher consistency
        consistency = 1.0 / (1.0 + fill_variance + diversity_variance)
        return consistency
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _suggest_optimal_combinations(self, grids: List[TextGrid]) -> List[Dict[str, Any]]:
        """Suggest optimal ways to combine the grids"""
        suggestions = []
        
        # Suggest based on dimensions
        dimensions = [(grid.rows, grid.columns) for grid in grids]
        
        # If grids have same height, suggest horizontal combination
        if len(set(rows for rows, cols in dimensions)) == 1:
            suggestions.append({
                'method': 'horizontal',
                'reason': 'All grids have the same height',
                'alignment': 'top',
                'expected_dimensions': (dimensions[0][0], sum(cols for rows, cols in dimensions))
            })
        
        # If grids have same width, suggest vertical combination
        if len(set(cols for rows, cols in dimensions)) == 1:
            suggestions.append({
                'method': 'vertical',
                'reason': 'All grids have the same width',
                'alignment': 'left',
                'expected_dimensions': (sum(rows for rows, cols in dimensions), dimensions[0][1])
            })
        
        # Always suggest overlay for pattern analysis
        max_rows = max(rows for rows, cols in dimensions)
        max_cols = max(cols for rows, cols in dimensions)
        suggestions.append({
            'method': 'overlay',
            'reason': 'Good for finding overlapping patterns',
            'alignment': 'center',
            'expected_dimensions': (max_rows, max_cols)
        })
        
        return suggestions

# Add the GridCombiner to the GridGenerator class
GridGenerator.GridCombiner = GridCombiner

# Add method to GridGenerator class
def get_combiner(self) -> 'GridCombiner':
    """Get a GridCombiner instance for this generator"""
    return self.GridCombiner(self)

GridGenerator.get_combiner = get_combiner