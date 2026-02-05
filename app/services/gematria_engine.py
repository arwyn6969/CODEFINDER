"""
Gematria Engine
Calculates numerological values for text using various historical ciphers.
"""
from typing import Dict, List, Any, Optional
import re
from dataclasses import dataclass

@dataclass
class GematriaResult:
    cipher: str
    score: int
    breakdown: List[int]  # Value of each character

class GematriaEngine:
    """
    Engine for calculating Gematria scores using various historical and modern ciphers.
    """
    
    # Sacred/Significant numbers to watch for
    SACRED_NUMBERS = {
        888: "Greek 'Jesus' (Iesous)",
        666: "Number of the Beast",
        144: "Fibonacci / Light",
        432: "Cosmic Cycle",
        314: "Pi (approx)",
        161: "Phi (approx)",
        271: "e (approx)",
        # Add more as needed
    }

    def __init__(self):
        self.ciphers = {
            'english_ordinal': self._cipher_english_ordinal,
            'english_standard': self._cipher_english_standard,
            'english_sumerian': self._cipher_english_sumerian,
            'francis_bacon_simple': self._cipher_francis_bacon_simple,
            'francis_bacon_reverse': self._cipher_francis_bacon_reverse,
            'francis_bacon_kay': self._cipher_francis_bacon_kay,
            'hebrew_standard': self._cipher_hebrew_standard,
            'greek_isopsephy': self._cipher_greek_isopsephy
        }

    def calculate_all(self, text: str) -> Dict[str, Any]:
        """Calculate scores for all known ciphers"""
        results = {}
        cleaned_text = self._clean_text(text)
        
        for name, cipher_func in self.ciphers.items():
            score, breakdown = cipher_func(cleaned_text)
            
            significance = self.SACRED_NUMBERS.get(score)
            
            results[name] = {
                'score': score,
                'breakdown': breakdown,
                'significance': significance
            }
        
        return results

    def _clean_text(self, text: str) -> str:
        """Remove non-alphanumeric characters, keep case if needed but usually ciphers are case-insensitive or specific"""
        # For simple Gematria, we usually ignore spaces and punctuation
        # But some ciphers might depend on exact characters. 
        # Standard approach: upper case, ignore non-letters/numbers
        return ''.join(c.upper() for c in text if c.isalnum())

    def _cipher_english_ordinal(self, text: str) -> (int, List[int]):
        """A=1, B=2... Z=26"""
        breakdown = []
        score = 0
        for char in text:
            if 'A' <= char <= 'Z':
                val = ord(char) - ord('A') + 1
                score += val
                breakdown.append(val)
        return score, breakdown

    def _cipher_english_standard(self, text: str) -> (int, List[int]):
        """A=1..I=9, J=10..R=90, S=100..Z=800"""
        # Mapping based on Agrippa code or similar
        values = {
            'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'I':9,
            'J':10, 'K':20, 'L':30, 'M':40, 'N':50, 'O':60, 'P':70, 'Q':80, 'R':90,
            'S':100, 'T':200, 'U':300, 'V':400, 'W':500, 'X':600, 'Y':700, 'Z':800
        }
        return self._mapped_cipher(text, values)

    def _cipher_english_sumerian(self, text: str) -> (int, List[int]):
        """A=6, B=12... (Ordinal * 6)"""
        score, breakdown = self._cipher_english_ordinal(text)
        return score * 6, [x * 6 for x in breakdown]

    def _cipher_francis_bacon_simple(self, text: str) -> (int, List[int]):
        """
        Francis Bacon's Simple Cipher (Elizabethan 24-letter alphabet).
        I=J (9), U=V (20).
        A=1...I=9, K=10...T=19, U=20, W=21...Z=24.
        """
        # Mapping for 24-letter alphabet
        values = {}
        alphabet_24 = "ABCDEFGHIKLMNOPQRSTUWXYZ" # Note: No J, No V (merged into I/U logic)
        
        # Build standard 1-24 map
        for i, char in enumerate(alphabet_24, 1):
            values[char] = i
        
        # Add the merged letters
        values['J'] = values['I'] # J becomes I (9)
        values['V'] = values['U'] # V becomes U (20)
        
        return self._mapped_cipher(text, values)

    def _cipher_francis_bacon_reverse(self, text: str) -> (int, List[int]):
        """
        Francis Bacon's Reverse Cipher (Elizabethan 24-letter alphabet).
        Z=1...A=24.
        """
        # Mapping for 24-letter alphabet reverse
        values = {}
        alphabet_24 = "ABCDEFGHIKLMNOPQRSTUWXYZ"
        
        # Build standard 24-1 map
        for i, char in enumerate(alphabet_24, 1):
            values[char] = 25 - i # 24 -> 1
            
        # Add the merged letters
        values['J'] = values['I']
        values['V'] = values['U']
        
        return self._mapped_cipher(text, values)

    def _cipher_francis_bacon_kay(self, text: str) -> (int, List[int]):
        """
        Francis Bacon's 'Key' (Kay) Cipher (A=27).
        Calculated as Simple Cipher Value + 26.
        A=27, B=28... I=35, K=36... Z=50.
        """
        # Get Simple cipher values first
        _, simple_breakdown = self._cipher_francis_bacon_simple(text)
        
        # Add 26 to each value
        kay_breakdown = [val + 26 for val in simple_breakdown]
        kay_score = sum(kay_breakdown)
        
        return kay_score, kay_breakdown

    def _cipher_hebrew_standard(self, text: str) -> (int, List[int]):
        """Standard Mispar Ragil (Hebrew)"""
        values = {
            'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5, 'ו': 6, 'ז': 7, 'ח': 8, 'ט': 9,
            'י': 10, 'כ': 20, 'ל': 30, 'מ': 40, 'נ': 50, 'ס': 60, 'ע': 70, 'פ': 80, 'צ': 90,
            'ק': 100, 'ר': 200, 'ש': 300, 'ת': 400,
            # Final forms (same values as non-final in standard encoding, sometimes different in Soffits)
            # Standard Mispar Ragil treats finals as equal to their non-final counterparts usually,
            # but sometimes they continue the hundreds (500, 600, etc).
            # The script uses the standard non-final values for finals.
            'ך': 20, 'ם': 40, 'ן': 50, 'ף': 80, 'ץ': 90,
        }
        return self._mapped_cipher(text, values)

    def _cipher_greek_isopsephy(self, text: str) -> (int, List[int]):
        """Standard Greek Isopsephy"""
        values = {
            'Α':1, 'Β':2, 'Γ':3, 'Δ':4, 'Ε':5, 'Ϛ':6, 'Ζ':7, 'Η':8, 'Θ':9,
            'Ι':10, 'Κ':20, 'Λ':30, 'Μ':40, 'Ν':50, 'Ξ':60, 'Ο':70, 'Π':80, 'Ϙ':90,
            'Ρ':100, 'Σ':200, 'Τ':300, 'Υ':400, 'Φ':500, 'Χ':600, 'Ψ':700, 'Ω':800,
            # Handle unicode variants/lowercase if needed, simpler to normalize to upper
        }
        # Add lowercase map
        upper_values = values.copy()
        for k, v in upper_values.items():
            values[k.lower()] = v
            
        return self._mapped_cipher(text, values)

    def _mapped_cipher(self, text: str, mapping: Dict[str, int]) -> (int, List[int]):
        breakdown = []
        score = 0
        for char in text:
            if char in mapping:
                val = mapping[char]
                score += val
                breakdown.append(val)
        return score, breakdown
