import pytesseract
from PIL import Image
import os

# Configuration
LONG_S = 'ſ'
image_path = "/Users/arwynhughes/Documents/CODEFINDER_PUBLISH/data/sources/folger_sonnets_1609/011_b2_verso_b3_recto.jpg"
config = "--oem 3 --psm 6 -c preserve_interword_spaces=1"

print(f"Testing OCR on: {image_path}")
img = Image.open(image_path).convert("RGB")

# Get data
data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)

counts = {
    'long_s_literal': 0,
    'f_potential': 0,
    's_potential': 0
}

found_examples = []

for text in data['text']:
    if not text: continue
    
    if LONG_S in text:
        counts['long_s_literal'] += text.count(LONG_S)
        found_examples.append(text)
    
    # Check for words that SHOULD have long-s but might have f/s
    if 'f' in text.lower() or 's' in text.lower():
        # Heuristic: 'ſhame' -> 'fhame' or 'shame'
        if 'fhame' in text.lower() or 'felfe' in text.lower():
             counts['f_potential'] += 1
        if 'shame' in text.lower() or 'selfe' in text.lower():
             counts['s_potential'] += 1

print("\n--- Results ---")
print(f"Literal LONG_S ('{LONG_S}') found: {counts['long_s_literal']}")
print(f"Potential 'f' confusions (e.g. fhame, felfe): {counts['f_potential']}")
print(f"Potential 's' confusions (e.g. shame, selfe): {counts['s_potential']}")

if found_examples:
    print(f"\nExamples of literal matches: {found_examples[:5]}")
else:
    print("\nNo literal long-s found in Tesseract output.")
    
# Check Unicode of LONG_S
print(f"\nLONG_S Unicode: {hex(ord(LONG_S))}")
