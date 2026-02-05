import sys
import os
sys.path.append(os.getcwd())
from app.services.gematria_engine import GematriaEngine

def verify():
    engine = GematriaEngine()
    word = "CodeFinder"
    res = engine.calculate_all(word)
    print(f"Gematria for '{word}':")
    for cipher, val in res.items():
        score = val['score'] if isinstance(val, dict) else val
        print(f"- {cipher}: {score}")

if __name__ == "__main__":
    verify()
