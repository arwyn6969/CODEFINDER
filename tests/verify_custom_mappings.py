import sys
import os

# Add app to path
sys.path.append(os.getcwd())

from app.services.transliteration_service import TransliterationService

def test_custom_mappings():
    print("🧪 Testing Custom Transliteration Mappings...")
    service = TransliterationService()
    
    test_cases = [
        ("BITCOIN", ["ביטקוין", "ביטקיין"]),
        ("ARWYN", ["ארוין", "ארווין", "ארן"]),
        ("MEME", ["מם", "מיים"]),
        ("TRUTH", ["אמת"])
    ]
    
    all_passed = True
    
    for term, expected_hebrew_parts in test_cases:
        candidates = service.get_hebrew_candidates(term)
        # candidates is list of (hebrew, description)
        hebrew_values = [h for h, d in candidates]
        
        print(f"\nChecking '{term}':")
        found_any = False
        for expected in expected_hebrew_parts:
            if expected in hebrew_values:
                print(f"  ✅ Found '{expected}'")
                found_any = True
            else:
                print(f"  ❌ Missing '{expected}'")
                all_passed = False
        
        if not found_any:
            print(f"  🔴 No mappings found for {term}!")
            all_passed = False
            
    if all_passed:
        print("\n✨ All custom mappings verified successfully!")
    else:
        print("\n⚠️  Some mappings were missing or incorrect.")

if __name__ == "__main__":
    test_custom_mappings()
