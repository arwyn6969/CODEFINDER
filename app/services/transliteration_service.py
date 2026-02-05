from typing import Dict, List, Tuple

class TransliterationService:
    """
    Service for transliterating English terms into Hebrew for ELS searching.
    Currently supports a manual dictionary of known mappings (e.g. PEPE -> פפי).
    Future expansion: Algorithmically map phonetic sounds.
    """
    
    # Static dictionary migrated from els_pepe_comprehensive.py
    # Format: ENGLISH_KEY: [(HEBREW_STR, DESCRIPTION)]
    KNOWN_MAPPINGS = {
        "PEPE": [
            ("פפי", "Standard (PPI)"),
            ("פאפא", "PAPA"),
            ("פפפ", "PPP"),
            ("פיפי", "PIPI"),
            ("פאפי", "PAPI"),
            ("פפא", "PPA"),
            ("פאפא", "Letter-by-Letter (P-E-P-E)"),
            ("פא", "PE (Short)"),
        ],
        "FROG": [
            ("צפרדע", "Biblical (Tzfardea)"),
            ("הצפרדע", "The Frog"),
            ("צפרדעים", "Frogs (Plural)"),
            ("צפר", "Root (TZF-R)"),
            ("פרוג", "Phonetic (Frog)"),
            ("פראג", "Phonetic Alt"),
        ],
        "KEVIN": [
            ("קבין", "Standard (QBIN)"),
            ("כבין", "Alt (KBIN)"),
            ("קעבין", "Yiddish-style"),
        ],
        "JESUS": [
            ("ישוע", "Yeshua"),
            ("ישו", "Yeshu"),
        ],
        "GOD": [
            ("יהוה", "YHWH"),
            ("אל", "El"),
            ("אלהים", "Elohim"),
        ],
        "SATOSHI": [
            ("סטושי", "Phonetic (Satoshi)"),
            ("סאטושי", "Phonetic Long (Satoshi)"),
        ],
        "NAKAMOTO": [
            ("נקמוטו", "Phonetic (Nakamoto)"),
            ("נאקאמוטו", "Phonetic Long (Nakamoto)"),
        ],
        "BITCOIN": [
            ("ביטקוין", "Modern Hebrew (Bitcoin)"),
            ("ביטקיין", "Alt Spelling"),
        ],
        "ARWYN": [
            ("ארוין", "Standard (ARWIN)"),
            ("ארווין", "Variant (ARVVIN)"),
            ("ארן", "Short (ARN)"),
        ],
        "MEME": [
            ("מם", "Standard (Mem)"),
            ("מיים", "Meme (Phonetic)"),
        ],
        "TRUTH": [
            ("אמת", "Emet (Standard)"),
        ]
    }

    def get_hebrew_candidates(self, term: str) -> List[Tuple[str, str]]:
        """
        Get a list of Hebrew candidates for a given English term.
        Returns list of (hebrew_string, description).
        """
        term_upper = term.upper()
        
        # 1. Direct dictionary lookup
        if term_upper in self.KNOWN_MAPPINGS:
            return self.KNOWN_MAPPINGS[term_upper]
            
        # 2. Fallback: Return empty list (caller can handle or try algorithmic)
        return []

    def auto_transliterate(self, text: str) -> List[str]:
        """
        Simple helper to try and transliterate a string if it's not in the dictionary.
        Very basic implementation for now.
        """
        # Dictionary of letter mappings
        # This is a 'naive' reversal of the standard phonetic map
        char_map = {
            'A': 'א', 'B': 'ב', 'C': 'ק', 'D': 'ד', 'E': 'ה', 
            'F': 'פ', 'G': 'ג', 'H': 'ה', 'I': 'י', 'J': 'ג', 
            'K': 'ק', 'L': 'ל', 'M': 'מ', 'N': 'נ', 'O': 'ו', 
            'P': 'פ', 'Q': 'ק', 'R': 'ר', 'S': 'ס', 'T': 'ת', 
            'U': 'ו', 'V': 'ו', 'W': 'ו', 'X': 'קס', 'Y': 'י', 'Z': 'ז'
        }
        
        result = []
        hebrew = ""
        valid = True
        for char in text.upper():
            if char in char_map:
                hebrew += char_map[char]
            elif char.isalnum():
                valid = False
                break
        
        if valid and hebrew:
            result.append(hebrew)
            
        return result
