
import fitz
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def generate_proof_sheet(pdf_path: str, page_num: int, output_path: str):
    """
    Scans a specific page and draws bounding boxes around all detected characters.
    Red box = low confidence (<60%)
    Green box = high confidence
    Blue box = Long-s
    """
    logger.info(f"Generating proof sheet for Page {page_num}...")
    
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]  # 0-indexed
    
    # 1. Get High-Res Image (3x zoom matches the scanner)
    mat = fitz.Matrix(3, 3)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img, "RGBA")
    
    # 2. Run OCR
    ocr_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=ocr_config)
    
    # 3. Draw Boxes
    n_boxes = len(data['text'])
    box_count = 0
    
    for i in range(n_boxes):
        text = data['text'][i]
        conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
        
        if not text or text.isspace():
            continue
            
        # Get word bounding box
        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]
        
        # Estimate character positions (same logic as scanner)
        char_width = w / len(text)
        
        for char_idx, char in enumerate(text):
            if char.isspace():
                continue
            
            cx = x + (char_idx * char_width)
            cy = y
            cw = char_width
            ch = h
            
            # Determine Color
            # Default: Green (good)
            color = (0, 255, 0, 60)       # Light Green fill
            outline = (0, 255, 0, 180)    # Green outline
            
            if conf < 60:
                color = (255, 0, 0, 60)   # Red fill (Low Conf)
                outline = (255, 0, 0, 180)
            
            # Long-s detection (basic heuristic for visual check)
            if char == 'f' or char == 'ſ':
                 # Check simple heuristic: if 'f' is followed by 'h', 'l', 't' -> likely Long-s
                if char_idx < len(text) - 1 and text[char_idx+1] in 'hlti':
                     color = (0, 100, 255, 60) # Blue (Long-s candidate)
                     outline = (0, 100, 255, 200)

            # Draw
            draw.rectangle([cx, cy, cx+cw, cy+ch], fill=color, outline=outline, width=2)
            box_count += 1

    # Save
    img.save(output_path)
    logger.info(f"Proof sheet saved to: {output_path} ({box_count} characters marked)")
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("page", type=int, help="Page number to generate proof for")
    parser.add_argument("--pdf", default="data/sources/archive/SONNETS_QUARTO_1609_NET.pdf", help="Path to PDF")
    args = parser.parse_args()
    
    out_path = f"reports/sonnet_print_block_analysis/visual_proof_page_{args.page:03d}.png"
    generate_proof_sheet(args.pdf, args.page, out_path)
