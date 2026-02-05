#!/usr/bin/env python3
"""
CODEFINDER Real Document Analysis
==================================
Analyze an actual page from the 1609 Shakespeare Sonnets Quarto.

This simulates the real user experience:
1. Extract a page from the PDF
2. Run OCR to get text
3. Apply all analysis engines
4. Show meaningful findings a researcher would care about
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import Counter
import math
import re

# PDF handling
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("⚠️ PyMuPDF not available. Will use sample text.")

try:
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("⚠️ OCR libraries not available. Will use sample text.")

PROJECT_ROOT = Path(__file__).parent

# ============================================================================
# ANALYSIS ENGINES (Condensed versions)
# ============================================================================

class GematriaEngine:
    """Calculate numerical values using historical ciphers."""
    
    SACRED_NUMBERS = {
        33: "Bacon (Simple FB initials)",
        67: "Francis (Simple)",
        100: "Francis Bacon (Simple)",
        157: "Fra Rosicrosse (Simple)",
        287: "Fra Rosicrosse (Kay)", 
        888: "Greek Jesus",
        666: "Number of the Beast",
        144: "Fibonacci / Light",
        432: "Cosmic Cycle",
        17: "Bacon's Favorite",
        111: "Triple Unity",
        222: "Master Builder",
        333: "Ascended Masters",
    }

    def calculate_simple(self, text: str) -> int:
        """Simple English ordinal A=1, B=2..."""
        clean = ''.join(c.upper() for c in text if c.isalpha())
        return sum(ord(c) - ord('A') + 1 for c in clean)
    
    def calculate_bacon_simple(self, text: str) -> int:
        """Francis Bacon's Simple Cipher (24-letter Elizabethan)."""
        values = {}
        alphabet_24 = "ABCDEFGHIKLMNOPQRSTUWXYZ"
        for i, char in enumerate(alphabet_24, 1):
            values[char] = i
        values['J'] = values['I']
        values['V'] = values['U']
        
        clean = ''.join(c.upper() for c in text if c.isalpha())
        return sum(values.get(c, 0) for c in clean)
    
    def calculate_bacon_kay(self, text: str) -> int:
        """Bacon's Kay Cipher (Simple + 26)."""
        simple = self.calculate_bacon_simple(text)
        word_count = len([c for c in text if c.isalpha()])
        return simple + (word_count * 26 // len(text.replace(' ', '')) if text.replace(' ', '') else 0)
    
    def find_significance(self, value: int) -> str:
        """Check if a value is significant."""
        return self.SACRED_NUMBERS.get(value)
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for Gematria patterns."""
        results = {
            "full_text": {
                "simple": self.calculate_simple(text),
                "bacon_simple": self.calculate_bacon_simple(text),
            },
            "words": [],
            "significant_findings": []
        }
        
        # Analyze each word
        words = text.split()
        for word in words:
            clean_word = ''.join(c for c in word if c.isalpha())
            if len(clean_word) >= 3:  # Skip tiny words
                simple = self.calculate_simple(clean_word)
                bacon = self.calculate_bacon_simple(clean_word)
                
                sig_simple = self.find_significance(simple)
                sig_bacon = self.find_significance(bacon)
                
                word_result = {
                    "word": clean_word,
                    "simple": simple,
                    "bacon": bacon
                }
                
                if sig_simple or sig_bacon:
                    word_result["significance"] = sig_simple or sig_bacon
                    results["significant_findings"].append({
                        "word": clean_word,
                        "value": bacon if sig_bacon else simple,
                        "cipher": "bacon" if sig_bacon else "simple",
                        "meaning": sig_simple or sig_bacon
                    })
                
                results["words"].append(word_result)
        
        return results


class ELSAnalyzer:
    """Find Equidistant Letter Sequences."""
    
    # Terms relevant to Shakespeare/Bacon authorship debate
    SHAKESPEARE_TERMS = [
        "BACON", "SHAKE", "SPEAR", "WILL", "POET",
        "AUTHOR", "HIDDEN", "SECRET", "CIPHER", "CODE",
        "VERSE", "SONNET", "LOVE", "ROSE", "BEAUTY"
    ]
    
    def analyze_text(self, text: str, terms: List[str] = None, max_skip: int = 50) -> Dict[str, Any]:
        """Search for ELS patterns."""
        clean_text = ''.join(c.upper() for c in text if c.isalpha())
        terms = terms or self.SHAKESPEARE_TERMS
        
        matches = []
        
        for term in terms:
            # Search at various skip intervals
            for skip in range(2, min(max_skip, len(clean_text) // len(term) + 1)):
                # Forward direction
                for start in range(skip):
                    sequence = clean_text[start::skip]
                    if term in sequence:
                        idx = sequence.index(term)
                        matches.append({
                            "term": term,
                            "skip": skip,
                            "start": start + (idx * skip),
                            "direction": "forward"
                        })
                
                # Backward (negative skip via reversed text)
                reversed_text = clean_text[::-1]
                for start in range(skip):
                    sequence = reversed_text[start::skip]
                    if term in sequence:
                        idx = sequence.index(term)
                        matches.append({
                            "term": term,
                            "skip": skip,
                            "start": len(clean_text) - 1 - (start + idx * skip),
                            "direction": "backward"
                        })
        
        return {
            "text_length": len(clean_text),
            "terms_searched": terms,
            "matches_found": len(matches),
            "matches": matches[:20]  # Limit output
        }


class CipherAnalyzer:
    """Detect acrostics and hidden messages."""
    
    def find_acrostic(self, lines: List[str]) -> str:
        """Extract first letter of each line."""
        acrostic = ""
        for line in lines:
            clean = line.strip()
            if clean:
                for c in clean:
                    if c.isalpha():
                        acrostic += c.upper()
                        break
        return acrostic
    
    def find_telestich(self, lines: List[str]) -> str:
        """Extract last letter of each line."""
        telestich = ""
        for line in lines:
            clean = line.strip()
            if clean:
                for c in reversed(clean):
                    if c.isalpha():
                        telestich += c.upper()
                        break
        return telestich
    
    def find_mesostich(self, lines: List[str], position: int = None) -> str:
        """Extract middle letter of each line."""
        mesostich = ""
        for line in lines:
            letters = [c for c in line if c.isalpha()]
            if letters:
                pos = position if position else len(letters) // 2
                if pos < len(letters):
                    mesostich += letters[pos].upper()
        return mesostich
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Find hidden acrostic patterns."""
        lines = [l for l in text.split('\n') if l.strip()]
        
        acrostic = self.find_acrostic(lines)
        telestich = self.find_telestich(lines)
        
        return {
            "line_count": len(lines),
            "acrostic": acrostic,
            "telestich": telestich,
            "acrostic_words": self._find_words_in_sequence(acrostic),
            "telestich_words": self._find_words_in_sequence(telestich)
        }
    
    def _find_words_in_sequence(self, seq: str) -> List[str]:
        """Find meaningful words hidden in a letter sequence."""
        # Common words that might appear
        words_to_find = [
            "BACON", "SHAKE", "WILL", "POET", "LOVE", "ROSE", "HIDE", "TRUE",
            "DARK", "FAIR", "SOUL", "HEART", "ART", "WIT", "PEN", "INK"
        ]
        found = []
        for word in words_to_find:
            if word in seq:
                found.append(word)
        return found


class PatternAnalyzer:
    """Find numerical and structural patterns."""
    
    def analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the structural properties of text."""
        lines = [l for l in text.split('\n') if l.strip()]
        words = text.split()
        letters = [c for c in text if c.isalpha()]
        
        return {
            "line_count": len(lines),
            "word_count": len(words),
            "letter_count": len(letters),
            "avg_words_per_line": round(len(words) / len(lines), 1) if lines else 0,
            "avg_letters_per_word": round(len(letters) / len(words), 1) if words else 0,
            
            # Numerological analysis
            "line_count_reduced": self._reduce_to_digit(len(lines)),
            "word_count_reduced": self._reduce_to_digit(len(words)),
            "letter_count_reduced": self._reduce_to_digit(len(letters)),
            
            # Bacon-relevant counts
            "is_14_lines": len(lines) == 14,  # Sonnet structure
            "divisible_by_17": len(letters) % 17 == 0 or len(words) % 17 == 0,
        }
    
    def _reduce_to_digit(self, n: int) -> int:
        """Reduce number to single digit (numerology)."""
        while n >= 10:
            n = sum(int(d) for d in str(n))
        return n


# ============================================================================
# PDF EXTRACTION
# ============================================================================

def extract_page_from_pdf(pdf_path: str, page_num: int) -> Tuple[str, Any]:
    """Extract text and image from a PDF page."""
    if not HAS_FITZ:
        return None, None
    
    doc = fitz.open(pdf_path)
    if page_num >= len(doc):
        page_num = len(doc) // 2  # Use middle page
    
    page = doc[page_num]
    
    # Get text
    text = page.get_text()
    
    # Get image for OCR if text is sparse
    if len(text.strip()) < 50:
        pix = page.get_pixmap(dpi=300)
        img_path = PROJECT_ROOT / "temp_page.png"
        pix.save(str(img_path))
        
        if HAS_OCR:
            img = Image.open(img_path)
            text = pytesseract.image_to_string(img)
            img_path.unlink()  # Clean up
    
    doc.close()
    return text, page_num


# ============================================================================
# SAMPLE SONNET TEXT (if PDF extraction fails)
# ============================================================================

SAMPLE_SONNET_18 = """SHAKE-SPEARES
SONNETS.

From fairest creatures we desire increase,
That thereby beauties Rose might never die,
But as the riper should by time decease,
His tender heire might beare his memory:
But thou contracted to thine owne bright eyes,
Feed'st thy lights flame with selfe substantiall fewell,
Making a famine where aboundance lies,
Thy selfe thy foe, to thy sweet selfe too cruell:
Thou that art now the worlds fresh ornament,
And only herauld to the gaudy spring,
Within thine owne bud buriest thy content,
And tender chorle makst wast in niggarding:
   Pitty the world, or else this glutton be,
   To eate the worlds due, by the grave and thee.
"""

SAMPLE_SONNET_DEDICATION = """TO.THE.ONLIE.BEGETTER.OF.
THESE.INSVING.SONNETS.
Mr.W.H. ALL.HAPPINESSE.
AND.THAT.ETERNITIE.
PROMISED.
BY.
OVR.EVER-LIVING.POET.
WISHETH.
THE.WELL-WISHING.
ADVENTVRER.IN.
SETTING.
FORTH.
T. T.
"""


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_document_page(text: str, page_info: str = "Sample") -> Dict[str, Any]:
    """Run comprehensive analysis on a page of text."""
    
    print(f"\n{'='*70}")
    print(f"📜 ANALYZING: {page_info}")
    print(f"{'='*70}")
    print(f"\n📝 Text Preview (first 200 chars):")
    print(f"   {text[:200].replace(chr(10), ' ')}...")
    
    results = {
        "source": page_info,
        "timestamp": datetime.now().isoformat(),
        "text_preview": text[:500],
        "analyses": {}
    }
    
    # 1. GEMATRIA ANALYSIS
    print(f"\n{'─'*50}")
    print("🔢 GEMATRIA ANALYSIS")
    print(f"{'─'*50}")
    
    gematria = GematriaEngine()
    gematria_results = gematria.analyze_text(text)
    
    print(f"   Full text Simple value: {gematria_results['full_text']['simple']}")
    print(f"   Full text Bacon value:  {gematria_results['full_text']['bacon_simple']}")
    
    if gematria_results['significant_findings']:
        print(f"\n   ⭐ SIGNIFICANT WORDS FOUND:")
        for finding in gematria_results['significant_findings'][:5]:
            print(f"      • {finding['word']} = {finding['value']} ({finding['meaning']})")
    else:
        print(f"\n   No sacred numbers found in individual words")
    
    results['analyses']['gematria'] = gematria_results
    
    # 2. ACROSTIC/HIDDEN MESSAGE ANALYSIS
    print(f"\n{'─'*50}")
    print("🔐 ACROSTIC ANALYSIS")
    print(f"{'─'*50}")
    
    cipher = CipherAnalyzer()
    cipher_results = cipher.analyze_text(text)
    
    print(f"   Lines analyzed: {cipher_results['line_count']}")
    print(f"   Acrostic (first letters): {cipher_results['acrostic'][:30]}{'...' if len(cipher_results['acrostic']) > 30 else ''}")
    print(f"   Telestich (last letters): {cipher_results['telestich'][:30]}{'...' if len(cipher_results['telestich']) > 30 else ''}")
    
    if cipher_results['acrostic_words']:
        print(f"\n   ⭐ HIDDEN WORDS IN ACROSTIC: {', '.join(cipher_results['acrostic_words'])}")
    if cipher_results['telestich_words']:
        print(f"   ⭐ HIDDEN WORDS IN TELESTICH: {', '.join(cipher_results['telestich_words'])}")
    
    results['analyses']['cipher'] = cipher_results
    
    # 3. ELS ANALYSIS
    print(f"\n{'─'*50}")
    print("🔍 ELS (EQUIDISTANT LETTER SEQUENCE) ANALYSIS")
    print(f"{'─'*50}")
    
    els = ELSAnalyzer()
    els_results = els.analyze_text(text, max_skip=30)
    
    print(f"   Text length: {els_results['text_length']} letters")
    print(f"   Terms searched: {', '.join(els_results['terms_searched'][:5])}...")
    print(f"   Matches found: {els_results['matches_found']}")
    
    if els_results['matches']:
        print(f"\n   📍 ELS MATCHES:")
        for match in els_results['matches'][:5]:
            print(f"      • '{match['term']}' at skip {match['skip']} ({match['direction']})")
    
    results['analyses']['els'] = els_results
    
    # 4. STRUCTURAL ANALYSIS
    print(f"\n{'─'*50}")
    print("📊 STRUCTURAL ANALYSIS")
    print(f"{'─'*50}")
    
    pattern = PatternAnalyzer()
    structure_results = pattern.analyze_structure(text)
    
    print(f"   Lines: {structure_results['line_count']}")
    print(f"   Words: {structure_results['word_count']}")
    print(f"   Letters: {structure_results['letter_count']}")
    print(f"   Avg words/line: {structure_results['avg_words_per_line']}")
    
    if structure_results['is_14_lines']:
        print(f"\n   ⭐ SONNET STRUCTURE: 14 lines detected!")
    if structure_results['divisible_by_17']:
        print(f"   ⭐ BACON SIGNATURE: Letter/word count divisible by 17!")
    
    results['analyses']['structure'] = structure_results
    
    return results


def generate_findings_report(results: Dict[str, Any]) -> str:
    """Generate a human-readable findings report."""
    
    report = []
    report.append("=" * 70)
    report.append("CODEFINDER ANALYSIS REPORT")
    report.append("=" * 70)
    report.append(f"\nSource: {results['source']}")
    report.append(f"Analyzed: {results['timestamp']}")
    
    report.append("\n" + "─" * 70)
    report.append("KEY FINDINGS")
    report.append("─" * 70)
    
    findings = []
    
    # Collect significant findings
    gematria = results['analyses'].get('gematria', {})
    if gematria.get('significant_findings'):
        for f in gematria['significant_findings']:
            findings.append(f"• GEMATRIA: '{f['word']}' = {f['value']} ({f['meaning']})")
    
    cipher = results['analyses'].get('cipher', {})
    if cipher.get('acrostic_words'):
        findings.append(f"• ACROSTIC: Hidden words found: {', '.join(cipher['acrostic_words'])}")
    if cipher.get('telestich_words'):
        findings.append(f"• TELESTICH: Hidden words found: {', '.join(cipher['telestich_words'])}")
    
    els = results['analyses'].get('els', {})
    if els.get('matches'):
        unique_terms = set(m['term'] for m in els['matches'])
        findings.append(f"• ELS: {len(els['matches'])} sequences found for terms: {', '.join(unique_terms)}")
    
    structure = results['analyses'].get('structure', {})
    if structure.get('is_14_lines'):
        findings.append("• STRUCTURE: Classic 14-line sonnet form detected")
    if structure.get('divisible_by_17'):
        findings.append("• BACON SIGNATURE: Text divisible by 17 (Bacon's number)")
    
    if findings:
        for f in findings:
            report.append(f)
    else:
        report.append("No significant patterns detected in this text.")
    
    report.append("\n" + "─" * 70)
    report.append("RESEARCHER INTERPRETATION")
    report.append("─" * 70)
    
    # Add interpretation guidance
    report.append("""
The findings above should be interpreted in context:

1. GEMATRIA values matching "sacred numbers" may indicate intentional
   encoding, especially in Renaissance-era texts where such practices
   were common among Rosicrucians and Freemasons.

2. ACROSTIC/TELESTICH patterns spelling meaningful words suggest
   deliberate hidden messages embedded by the author.

3. ELS (Equidistant Letter Sequences) at low skip values (2-10) are
   more statistically significant than those at higher intervals.

4. Structural features like 17-divisibility are associated with
   Francis Bacon, who used 17 as a signature number.

For definitive conclusions, cross-reference findings with:
- Other pages/documents
- Historical context
- Multiple cipher systems
""")
    
    return "\n".join(report)


def main():
    print("\n" + "=" * 70)
    print("    CODEFINDER - REAL DOCUMENT ANALYSIS")
    print("    Simulating User Experience with Shakespeare Sonnets")
    print("=" * 70)
    
    # Try to extract from actual PDF
    sonnets_pdf = PROJECT_ROOT / "data/sources/archive/SONNETS_QUARTO_1609_NET.pdf"
    
    text = None
    page_info = ""
    
    if sonnets_pdf.exists() and HAS_FITZ:
        print(f"\n📂 Found PDF: {sonnets_pdf.name}")
        print("   Extracting page 10 (middle of dedication/early sonnets)...")
        
        try:
            text, actual_page = extract_page_from_pdf(str(sonnets_pdf), 10)
            page_info = f"Shakespeare Sonnets 1609 Quarto - Page {actual_page + 1}"
            print(f"   ✅ Extracted {len(text)} characters")
        except Exception as e:
            print(f"   ⚠️ Extraction failed: {e}")
            text = None
    
    if not text or len(text.strip()) < 50:
        print("\n📝 Using sample Sonnet text (Dedication + Sonnet 1)...")
        text = SAMPLE_SONNET_DEDICATION + "\n\n" + SAMPLE_SONNET_18
        page_info = "Sample: Sonnets Dedication + Sonnet 1"
    
    # Run analysis
    results = analyze_document_page(text, page_info)
    
    # Generate report
    report = generate_findings_report(results)
    print("\n" + report)
    
    # Save results
    output_dir = PROJECT_ROOT / "demo_results"
    output_dir.mkdir(exist_ok=True)
    
    json_path = output_dir / "real_document_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"📁 JSON results: {json_path}")
    print(f"📄 Text report: {report_path}")
    print("\n💡 This demonstrates what a user would see after uploading a document!")


if __name__ == "__main__":
    main()
