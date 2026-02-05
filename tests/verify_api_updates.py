
import sys
import os
from pydantic import BaseModel

# Add app to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app.api.routes.research import TransliterateRequest, ELSRequest, transliterate_term
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_models():
    # Test TransliterateRequest
    try:
        req = TransliterateRequest(text="PEPE")
        print("✅ TransliterateRequest model valid")
    except Exception as e:
        print(f"❌ TransliterateRequest failed: {e}")

    # Test ELSRequest for new field
    try:
        req = ELSRequest(text="ABC", auto_transliterate=True)
        if req.auto_transliterate is True:
            print("✅ ELSRequest has auto_transliterate field")
        else:
            print("❌ ELSRequest auto_transliterate field missing or invalid")
    except Exception as e:
        print(f"❌ ELSRequest failed: {e}")

if __name__ == "__main__":
    test_models()
