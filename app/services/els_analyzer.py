"""
ELS Analyzer (Equidistant Letter Sequences)
Searches for hidden messages encoded via skip patterns (c.f. Bible Codes).
"""
from typing import List, Dict, Any, Tuple
import re
from dataclasses import dataclass

@dataclass
class ELSResult:
    term: str
    skip: int
    start_index: int
    end_index: int
    text_segment: str  # The actual letters found
    direction: int # 1 (forward) or -1 (backward)

class ELSAnalyzer:
    """
    Analyzer for finding Equidistant Letter Sequences in text.
    """
    
    # Default list of interesting terms to search for
    DEFAULT_TERMS = [
        "JESUS", "CHRIST", "GOD", "YHWH", "TORAH",
        "SHAKESPEARE", "BACON", "MARLOWE", "OXFORD",
        "ROSE", "CROSS", "TEMPLE", "GIZA",
        "EDEN", "ARK", "GRAIL"
    ]

    def __init__(self, terms: List[str] = None):
        self.terms = [t.upper() for t in (terms or self.DEFAULT_TERMS)]
        # Lazy import to avoid circular dependency if any (none expected)
        from app.services.transliteration_service import TransliterationService
        self.transliteration = TransliterationService()
        
        # Expand terms using transliteration? 
        # For now, we keep the original terms list, but analyze_text can optionally expand.

    def analyze_text(self, text: str, min_skip: int = 2, max_skip: int = 150, auto_transliterate: bool = False) -> Dict[str, Any]:
        """
        Analyze text for ELS terms.
        If auto_transliterate is True, English terms in self.terms will be expanded to Hebrew candidates.
        """
        # 1. Clean text (remove spaces/punctuation)
        clean_text = ''.join(c.upper() for c in text if c.isalnum())
        N = len(clean_text)
        
        matches = []
        
        # Prepare search terms
        search_terms = []
        for term in self.terms:
            search_terms.append(term) # Always search original
            
            if auto_transliterate:
                # Get candidates
                candidates = self.transliteration.get_hebrew_candidates(term)
                if not candidates:
                    # Try naive auto
                    candidates = [(h, "Auto") for h in self.transliteration.auto_transliterate(term)]
                
                for hebrew, _ in candidates:
                    search_terms.append(hebrew)
        
        # Remove duplicates
        search_terms = list(set(search_terms))
        
        # 2. Search for each term
        for term in search_terms:
            term_matches = self._find_term(clean_text, term, min_skip, max_skip)
            matches.extend(term_matches)
            
        return {
            "total_length": N,
            "found_count": len(matches),
            "matches": [
                {
                    "term": m.term,
                    "skip": m.skip,
                    "location": [m.start_index, m.end_index],
                    "direction": "forward" if m.skip > 0 else "backward"
                }
                for m in matches
            ]
        }
        
    def _find_term(self, text: str, term: str, min_skip: int, max_skip: int) -> List[ELSResult]:
        """Find all occurrences of a specific term at various skips"""
        results = []
        N = len(text)
        term_len = len(term)
        
        if term_len > N:
            return []
            
        # Iterate skips
        # Note: ELS usually includes negative skips (reverse reading).
        # We will check skips in range [min, max] AND [-max, -min]
        
        skips = list(range(min_skip, max_skip + 1)) + list(range(-max_skip, -min_skip + 1))
        
        for skip in skips:
            if skip == 0: continue
            
            # Max index check to avoid unnecessary slicing
            # We need (term_len - 1) * abs(skip) space
            required_span = (term_len - 1) * abs(skip)
            if required_span >= N:
                continue
                
            # Iterate starting positions
            # We can check specific chars first to fail fast
            # Optimization: Try matching first char, then check rest?
            # Or just brute slice if N < 10000
            
            # Python slicing with step is very fast
            # text[start::skip]
            
            # We need to find the term in the sliced strings
            # But text[i::skip] gives the WHOLE sequence. We need to find 'term' in it.
            
            for start in range(min(abs(skip), N)): # Only need to check starts up to skip-1
                sequence = text[start::skip]
                if term in sequence:
                    # Found it! Now locate exactly where
                    # Occurrences in the sequence
                    seq_index = 0
                    while True:
                        try:
                            found_idx = sequence.index(term, seq_index)
                            # Convert back to original indices
                            abs_start = start + (found_idx * skip)
                            # ELS end is start + (len-1)*skip
                            abs_end = abs_start + (term_len - 1) * skip
                            
                            # Validation bound check (should be implied by slice)
                            if 0 <= abs_start < N and 0 <= abs_end < N:
                                results.append(ELSResult(
                                    term=term,
                                    skip=skip,
                                    start_index=abs_start,
                                    end_index=abs_end,
                                    text_segment=term,
                                    direction=1 if skip > 0 else -1
                                ))
                            
                            seq_index = found_idx + 1
                        except ValueError:
                            break
                            
        return results
