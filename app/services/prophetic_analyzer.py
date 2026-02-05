"""
Prophetic Analyzer Service
==========================
Service for analyzing triple thematic convergences and prophetic term crossings in the Torah.
Migrated from ad-hoc analysis scripts.
"""

from typing import List, Dict, Set, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import time
from pathlib import Path

# Local imports
# Assuming ELSAnalyzer is available or we reimplement the fast search
from app.services.els_analyzer import ELSAnalyzer, ELSResult

@dataclass
class ConvergenceZone:
    center_index: int
    spread: int
    terms: Dict[str, ELSResult]
    book: str
    position_percentage: float

class PropheticAnalyzerService:
    def __init__(self):
        self.els_analyzer = ELSAnalyzer() # Use existing analyzer for base search
        self.torah_books = [
            ("Genesis / בראשית", 0, 78064),
            ("Exodus / שמות", 78064, 141064),
            ("Leviticus / ויקרא", 141064, 185413),
            ("Numbers / במדבר", 185413, 249878),
            ("Deuteronomy / דברים", 249878, 304805),
        ]

    def _get_book_info(self, position: int) -> Tuple[str, float]:
        for name, start, end in self.torah_books:
            if start <= position < end:
                pct = ((position - start) / (end - start)) * 100
                return name, pct
        return "Unknown", 0.0

    def find_triple_convergence(
        self, 
        text: str, 
        term_groups: List[Dict[str, Any]], 
        max_spread: int = 500
    ) -> List[ConvergenceZone]:
        """
        Find zones where one term from each group appears within max_spread letters.
        OPTIMIZED: Uses binary search instead of cubic loops.
        """
        import bisect
        
        # Build index map once for this text
        char_indices = defaultdict(list)
        for i, char in enumerate(text):
            char_indices[char].append(i)
            
        # 1. Search for all terms using index
        all_hits: Dict[str, List[ELSResult]] = {}
        
        for group in term_groups:
            group_name = group["name"]
            group_hits = []
            for term in group["terms"]:
                hits = self._indexed_search(text, char_indices, term, min_skip=2, max_skip=300)
                group_hits.extend(hits)
            
            # Sort hits by start_index for binary search
            group_hits.sort(key=lambda x: x.start_index)
            all_hits[group_name] = group_hits

        # 2. Find convergences
        if len(term_groups) != 3:
            raise NotImplementedError("Currently optimized for exactly 3 term groups")
            
        names = [g["name"] for g in term_groups]
        hits_a = all_hits[names[0]]
        hits_b = all_hits[names[1]]
        hits_c = all_hits[names[2]]
        
        # Create position lists for binary search lookup
        pos_b = [h.start_index for h in hits_b]
        pos_c = [h.start_index for h in hits_c]
        
        zones = []
        
        # Iterate through A
        for a in hits_a:
            p_a = a.start_index
            
            # Find B candidates
            min_b = p_a - max_spread
            max_b = p_a + max_spread
            
            start_idx_b = bisect.bisect_left(pos_b, min_b)
            end_idx_b = bisect.bisect_right(pos_b, max_b)
            
            for i_b in range(start_idx_b, end_idx_b):
                b = hits_b[i_b]
                p_b = b.start_index
                
                # Find C candidates
                min_c = max(p_a - max_spread, p_b - max_spread)
                max_c = min(p_a + max_spread, p_b + max_spread)
                
                if min_c > max_c: continue
                    
                start_idx_c = bisect.bisect_left(pos_c, min_c)
                end_idx_c = bisect.bisect_right(pos_c, max_c)
                
                for i_c in range(start_idx_c, end_idx_c):
                    c = hits_c[i_c]
                    p_c = c.start_index
                    
                    if abs(p_c - p_a) <= max_spread and abs(p_c - p_b) <= max_spread:
                        positions = [p_a, p_b, p_c]
                        center = sum(positions) // 3
                        spread = max(positions) - min(positions)
                        book, pct = self._get_book_info(center)
                        
                        zones.append(ConvergenceZone(
                            center_index=center, spread=spread,
                            terms={names[0]: a, names[1]: b, names[2]: c},
                            book=book, position_percentage=pct
                        ))
        
        zones.sort(key=lambda x: x.spread)
        return zones

    def _indexed_search(self, text: str, char_indices: Dict[str, List[int]], term: str, min_skip=2, max_skip=300) -> List[ELSResult]:
        """Flattened search using character index for speed."""
        results = []
        if not term: return []
        
        first_char = term[0]
        term_len = len(term)
        text_len = len(text)
        
        starts = char_indices.get(first_char, [])
        
        for start in starts:
            # Check skips
            # We can iterate skips. 2 to 300 is 298 checks.
            # Start positions are ~13k. Total 4M checks.
            for skip in range(min_skip, max_skip + 1):
                # Check bounds
                if start + (term_len - 1) * skip >= text_len:
                    break # larger skips will also fail
                
                # Check rest of chars
                match = True
                for i in range(1, term_len):
                    if text[start + i * skip] != term[i]:
                        match = False
                        break
                
                if match:
                    results.append(ELSResult(
                        term=term, skip=skip, start_index=start,
                        end_index=start + (term_len - 1) * skip,
                        text_segment=term, direction=1
                    ))
        return results

    def _fast_search(self, text: str, term: str, min_skip=2, max_skip=300) -> List[ELSResult]:
        """Legacy helper kept for compatibility if needed."""
        # Simple wrapper to build index on fly
        char_indices = defaultdict(list)
        for i, char in enumerate(text):
            char_indices[char].append(i)
        return self._indexed_search(text, char_indices, term, min_skip, max_skip)

    def find_crossings(self, text: str, term_a: str, term_b: str) -> List[Tuple[ELSResult, ELSResult]]:
        """Find direct index crossings between two terms."""
        hits_a = self._fast_search(text, term_a)
        hits_b = self._fast_search(text, term_b)
        
        crossings = []
        
        # Convert to sets of indices for fast intersection
        for a in hits_a:
            indices_a = set(range(a.start_index, a.end_index + 1, a.skip))
            for b in hits_b:
                indices_b = set(range(b.start_index, b.end_index + 1, b.skip))
                
                if not indices_a.isdisjoint(indices_b):
                    crossings.append((a, b))
                    
        return crossings
