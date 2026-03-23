from digital_type_case_builder import DigitalTypeCaseBuilder
from pathlib import Path
import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recover():
    builder = DigitalTypeCaseBuilder("data/sources/archive/SONNETS_QUARTO_1609_NET.pdf")
    
    # Check if images exist
    if not builder.images_dir.exists():
        logger.error("Images directory not found. Cannot recover.")
        return

    logger.info("Scanning disk for extracted images...")
    metadata = []
    
    # Iterate subfolders
    for char_dir in builder.images_dir.iterdir():
        if not char_dir.is_dir(): continue
        
        safe_char = char_dir.name
        
        # Reverse map safe_char to real char
        char = safe_char
        if safe_char == 'period': char = '.'
        elif safe_char == 'colon': char = ':'
        elif safe_char == 'comma': char = ','
        elif safe_char == 'semicolon': char = ';'
        elif safe_char == 'slash': char = '/'
        elif safe_char.endswith('_upper') and len(safe_char) == 7: # c_upper
             char = safe_char[0]
        
        for img_path in char_dir.glob("*.png"):
            # Filename: safe_char_PAGE_BLOCK_IDX.png
            # We just need path and page number for metadata
            try:
                parts = img_path.stem.split('_')
                # Last 3 parts are PAGE, BLOCK, IDX
                # Preceding parts are safe_char
                page = int(parts[-3])
            except:
                page = 0
            
            metadata.append({
                "char": char,
                "safe_char": safe_char,
                "path": str(img_path),
                "page": page
            })
            
    logger.info(f"Recovered {len(metadata)} items from disk.")
    
    # Save DB
    builder.metadata = metadata
    with open(builder.output_dir / "type_case_db.json", 'w') as f:
        json.dump(metadata, f, indent=2)
        
    # Run Clustering
    builder.cluster_and_report_full()

if __name__ == "__main__":
    recover()
