#!/usr/bin/env python3
"""
Database Population Script

Runs OCR on source images and populates the database with 
character instances including bounding boxes.
"""

import sys
from pathlib import Path
from PIL import Image
import pytesseract
import unicodedata
from datetime import datetime
from typing import List, Dict

from db_persistence import (
    init_database, get_connection, get_or_create_source,
    save_page, save_character_instances, update_source_stats
)


def get_character_category(char: str) -> str:
    """Determine category for a character."""
    if char.islower():
        return 'lowercase'
    elif char.isupper():
        return 'uppercase'
    elif char.isdigit():
        return 'digit'
    elif char == 'ſ':
        return 'long_s'
    elif char in '.,;:!?\'"()-[]{}':
        return 'punctuation'
    else:
        return 'other'


def preprocess_image(image: Image.Image, upscale_factor: int = 2) -> Image.Image:
    """
    Preprocess image for optimal OCR accuracy.
    
    Based on empirical testing:
    - 2x upscaling improves high-confidence detection from 66.7% to 80.0%
    """
    # Upscale for better character recognition
    if upscale_factor > 1:
        new_size = (image.width * upscale_factor, image.height * upscale_factor)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image


def analyze_f_vs_long_s(char_img: Image.Image) -> bool:
    """
    Visual analysis to distinguish 'f' from 'ſ' (Long-s).
    Returns True if likely 'ſ' (Long-s), False if 'f'.
    
    Heuristic: 'f' has a crossbar extending to the right. 
    'ſ' has a nub on left but empty on right.
    """
    # Convert to grayscale
    img = char_img.convert('L')
    # Simple binary threshold
    threshold = 128
    width, height = img.size
    
    # Define the "crossbar zone" (middle 40-60% of height)
    # and "right side" (right 40% of width)
    # If there is significant ink here, it's likely an 'f'
    
    mid_top = int(height * 0.40)
    mid_bot = int(height * 0.60)
    right_start = int(width * 0.60)
    
    ink_count = 0
    total_pixels = 0
    
    for y in range(mid_top, mid_bot):
        for x in range(right_start, width):
            pixel = img.getpixel((x, y))
            if pixel < threshold:  # Black/Ink
                ink_count += 1
            total_pixels += 1
    
    if total_pixels == 0: return False
    
    density = ink_count / total_pixels
    
    # If right-side crossbar density is low, it's a Long-s
    # 'f' usually has > 20% density here (the bar)
    # 'ſ' usually has < 10% (empty space)
    return density < 0.15


def scan_page_to_instances(image_path: Path, min_confidence: float = 60.0,
                           upscale: bool = True) -> List[Dict]:
    """
    Scan a page and return character instances with bounding boxes.
    
    Optimized OCR configuration:
    - PSM 3 (Auto) instead of PSM 6 (Block) - better for multi-column layouts
    - 2x upscaling for improved character recognition
    - Visual analysis for f/Long-s distinction
    """
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Preprocessing: upscale for better OCR
    upscale_factor = 2 if upscale else 1
    if upscale:
        image = preprocess_image(image, upscale_factor)
    
    # Optimized OCR config: PSM 3 (Auto) beats PSM 6 (Block)
    # Testing showed: PSM 3 = 66.7% high-conf vs PSM 6 = 49.9%
    ocr_config = r'--oem 3 --psm 3'
    ocr_data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config=ocr_config
    )
    
    instances = []
    n_boxes = len(ocr_data['text'])
    
    for i in range(n_boxes):
        text = ocr_data['text'][i]
        
        # Skip empty
        if not text or text.isspace():
            continue
        
        # Get confidence
        raw_conf = ocr_data['conf'][i]
        if raw_conf == '-1' or raw_conf == -1:
            continue
        conf = float(raw_conf)
        
        # Skip low confidence
        if conf < min_confidence:
            continue
        
        # Get bounding box (in upscaled coordinates)
        x = ocr_data['left'][i]
        y = ocr_data['top'][i]
        w = ocr_data['width'][i]
        h = ocr_data['height'][i]
        
        # Scale coordinates back to original image size
        final_x = x // upscale_factor
        final_y = y // upscale_factor
        final_w = w // upscale_factor
        final_h = h // upscale_factor
        
        # Process each character in the word
        if len(text) > 0:
            char_width = w / len(text) if len(text) > 0 else w
            final_char_width = char_width / upscale_factor
            
            for char_idx, char in enumerate(text):
                if char.isspace():
                    continue
                
                # Calculate character position in upscaled image
                curr_x = x + int(char_idx * char_width)
                curr_y = y
                curr_w = max(1, int(char_width))
                curr_h = max(1, h)
                
                # Final coordinates (original scale)
                final_curr_x = final_x + int(char_idx * final_char_width)
                final_curr_y = final_y
                final_curr_w = max(1, int(final_char_width))
                final_curr_h = max(1, final_h)
                
                # Visual Check for Long-s
                is_long_s = False
                if char == 'f':
                    # Extract crop from UPSCALED image for better resolution
                    try:
                        # Add 1px padding
                        crop = image.crop((
                            max(0, curr_x - 1),
                            max(0, curr_y - 1),
                            min(image.width, curr_x + curr_w + 1),
                            min(image.height, curr_y + curr_h + 1)
                        ))
                        if analyze_f_vs_long_s(crop):
                            char = 'ſ'
                            is_long_s = True
                    except Exception as e:
                        pass
                
                instances.append({
                    'character': char,
                    'category': get_character_category(char),
                    'x': final_curr_x,
                    'y': final_curr_y,
                    'width': final_curr_w,
                    'height': final_curr_h,
                    'confidence': conf
                })
    
    return instances


def populate_source(source_path: str, source_name: str, 
                    min_confidence: float = 60.0, max_pages: int = None):
    """Populate database from a source directory."""
    
    source_dir = Path(source_path)
    if not source_dir.exists():
        print(f"Source not found: {source_path}")
        return
    
    # Get image files
    images = sorted(
        list(source_dir.glob("*.jpg")) +
        list(source_dir.glob("*.png")) +
        list(source_dir.glob("*.tif"))
    )
    
    if max_pages:
        images = images[:max_pages]
    
    print(f"\n{'='*60}")
    print(f"POPULATING: {source_name}")
    print(f"{'='*60}")
    print(f"Source: {source_path}")
    print(f"Images: {len(images)}")
    print(f"Min confidence: {min_confidence}%")
    
    # Initialize database
    init_database()
    conn = get_connection()
    
    # Create/get source
    source_id = get_or_create_source(conn, source_name, str(source_path), len(images))
    
    total_chars = 0
    total_conf = 0
    conf_count = 0
    
    for page_num, image_path in enumerate(images, 1):
        print(f"  Processing page {page_num}/{len(images)}: {image_path.name}...", end=' ')
        
        try:
            # Get image dimensions
            img = Image.open(image_path)
            img_width, img_height = img.size
            img.close()
            
            # Scan page
            instances = scan_page_to_instances(image_path, min_confidence)
            
            # Calculate stats
            char_count = len(instances)
            avg_conf = sum(i['confidence'] for i in instances) / char_count if char_count > 0 else 0
            long_s_count = sum(1 for i in instances if i['character'] == 'ſ')
            
            # Save page
            page_id = save_page(
                conn, source_id, page_num,
                str(image_path), img_width, img_height,
                char_count, avg_conf, long_s_count, 0
            )
            
            # Save character instances
            save_character_instances(conn, page_id, instances)
            
            total_chars += char_count
            total_conf += avg_conf * char_count
            conf_count += char_count
            
            print(f"{char_count} chars")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Update source stats
    avg_confidence = total_conf / conf_count if conf_count > 0 else 0
    update_source_stats(conn, source_id, total_chars, avg_confidence)
    
    conn.close()
    
    print(f"\n✅ Completed: {total_chars:,} characters extracted")
    return total_chars


def main():
    """Main entry point."""
    print("="*60)
    print("CODEFINDER DATABASE POPULATION")
    print("="*60)
    print(f"Started: {datetime.now()}")
    
    # Define sources
    sources = [
        ("data/sources/folger_sonnets_1609", "wright"),
        ("data/sources/folger_sonnets_1609_aspley", "aspley"),
    ]
    
    # Check if we want quick mode (fewer pages)
    quick_mode = '--quick' in sys.argv
    max_pages = 10 if quick_mode else None
    
    if quick_mode:
        print("\n⚡ QUICK MODE: Processing only 10 pages per source")
    
    total = 0
    for source_path, source_name in sources:
        count = populate_source(source_path, source_name, 
                               min_confidence=60.0, max_pages=max_pages)
        if count:
            total += count
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {total:,} total characters in database")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
