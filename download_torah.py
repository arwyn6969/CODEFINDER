import requests
import os
from pathlib import Path

# URL for the Koren Torah text (Rips version)
URL = "https://github.com/TorahBibleCodes/TorahBibleCodes/raw/refs/heads/master/texts/text_koren_torah_fromrips.txt"
OUTPUT_DIR = Path("app/data")
OUTPUT_FILE = OUTPUT_DIR / "torah.txt"

# Michigan-Claremont to Hebrew Unicode Mapping
MC_MAP = {
    ')': 'א',
    'B': 'ב',
    'G': 'ג',
    'D': 'ד',
    'H': 'ה',
    'W': 'ו',
    'Z': 'ז',
    'X': 'ח',
    '+': 'ט',
    'Y': 'י',
    'K': 'כ',
    'L': 'ל',
    'M': 'מ',
    'N': 'נ',
    'S': 'ס',
    '(': 'ע',
    'P': 'פ',
    'C': 'צ',
    'Q': 'ק',
    'R': 'ר',
    '$': 'ש',
    'T': 'ת',
    # Final forms (if present in source, usually MC uses context or specific chars like M for mem, but let's check if source distinguishes final)
    # The Rips text usually uses standard letters. 
    # If final forms are needed for display, we can map, but pure ELS usually ignores final distinctions or treats them as equal.
    # However, Hebrew Unicode distinguishes. 
    # Let's assume standard mapping for now.
}

def download_and_convert():
    print(f"Downloading from {URL}...")
    try:
        response = requests.get(URL)
        response.raise_for_status()
        content_mc = response.text
        
        # Clean newlines/spaces if any (Rips text file might have them)
        # Usually it's a continuous string or line breaks.
        # We want a continuous string of letters.
        
        # Filter only valid MC chars
        valid_mc = set(MC_MAP.keys())
        cleaned_mc = "".join([c for c in content_mc if c in valid_mc])
        
        print(f"Downloaded {len(content_mc)} chars. Cleaned to {len(cleaned_mc)} MC chars.")
        
        # Convert to Hebrew
        hebrew_text = "".join([MC_MAP[c] for c in cleaned_mc])
        
        # Ensure output dir exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(hebrew_text)
            
        print(f"Saved Hebrew Torah ({len(hebrew_text)} letters) to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_and_convert()
