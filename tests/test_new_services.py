from app.services.transliteration_service import TransliterationService
from app.services.gematria_engine import GematriaEngine
from app.services.els_analyzer import ELSAnalyzer

def test_transliteration():
    print("\n--- Testing Transliteration Service ---")
    ts = TransliterationService()
    
    # Test 1: Dictionary Lookup
    pepe = ts.get_hebrew_candidates("PEPE")
    print(f"PEPE lookup: {len(pepe)} variants found")
    assert len(pepe) > 0
    assert ("פפי", "Standard (PPI)") in pepe
    
    # Test 2: Auto Transliteration
    random_term = "ABBA"
    auto = ts.auto_transliterate(random_term)
    print(f"ABBA auto: {auto}")
    assert "אבבא" in auto

def test_gematria():
    print("\n--- Testing Gematria Engine ---")
    ge = GematriaEngine()
    
    # Test Hebrew Standard (Mispar Ragil)
    # Aleph (1) + Bet (2) + Bet (2) + Aleph (1) = 6
    res = ge.calculate_all("אבבא")
    hebrew_score = res['hebrew_standard']['score']
    print(f"Gematria of 'אבבא': {hebrew_score}")
    assert hebrew_score == 6
    
    # Test PEPE (80+80+10 = 170? No, Pe is 80, Yud is 10. Pe-Pe-Yud = 80+80+10 = 170)
    # Wait, Pe (80) + Pe (80) + Yud (10)?
    # Script said: PEPE (פפי): 170? Let's check.
    # Pe=80. 80+80+10 = 170.
    res2 = ge.calculate_all("פפי")
    print(f"Gematria of 'פפי': {res2['hebrew_standard']['score']}")
    assert res2['hebrew_standard']['score'] == 170

def test_els_integration():
    print("\n--- Testing ELS Integration ---")
    # Small test text: "ABCPEPEXYZ" mapped to Hebrew? 
    # No, we need Hebrew text.
    # Text: "אבגפפידהו" (Aleph Bet Gimel PEPE Dalet He Vav)
    # "פפי" starts at index 3 (0-based)
    
    hebrew_text = "אבגפפידהו"
    analyzer = ELSAnalyzer(terms=["PEPE"]) # We search for ENGLISH 'PEPE'
    
    # Test with auto_transliterate=True
    results = analyzer.analyze_text(hebrew_text, min_skip=1, max_skip=1, auto_transliterate=True)
    
    print(f"Found {results['found_count']} matches")
    matches = results['matches']
    found_terms = [m['term'] for m in matches]
    print(f"Terms found: {found_terms}")
    
    assert "פפי" in found_terms
    assert results['found_count'] >= 1

if __name__ == "__main__":
    test_transliteration()
    test_gematria()
    test_els_integration()
    print("\n✅ All Tests Passed!")
