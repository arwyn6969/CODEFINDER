import fitz
import pytesseract
from PIL import Image
import sys
import os

def analyze_resolution(page_num):
    pdf_path = "data/sources/archive/SONNETS_QUARTO_1609_NET.pdf"
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # 1. First, find a target char (e.g. 'e') using standard scan
    mat = fitz.Matrix(3, 3)
    pix = page.get_pixmap(matrix=mat)
    img_3x = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    ocr_data = pytesseract.image_to_data(img_3x, output_type=pytesseract.Output.DICT)
    
    target_idx = -1
    for i, text in enumerate(ocr_data['text']):
        if 'e' in text:
            target_idx = i
            break
            
    if target_idx == -1:
        print("Could not find an 'e' on the page.")
        return

    # Get relative coordinates (0-1)
    x = ocr_data['left'][target_idx]
    y = ocr_data['top'][target_idx]
    w = ocr_data['width'][target_idx]
    h = ocr_data['height'][target_idx]
    
    # Text block might have multiple chars. Let's assume 'e' is the char or part of it.
    # Refine finding actual 'e' char?
    # Simpler: Just crop the word.
    
    # Coordinates in 3x image space
    w3 = img_3x.width
    h3 = img_3x.height
    
    norm_x = x / w3
    norm_y = y / h3
    norm_w = w / w3
    norm_h = h / h3
    
    resolutions = [3, 6, 9, 12]
    
    print(f"Analyzing block '{ocr_data['text'][target_idx]}'...")
    
    out_dir = "reports/resolution_test"
    os.makedirs(out_dir, exist_ok=True)
    
    for res in resolutions:
        mat = fitz.Matrix(res, res)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Calculate pixel crop
        x1 = int(norm_x * img.width)
        y1 = int(norm_y * img.height)
        x2 = int((norm_x + norm_w) * img.width)
        y2 = int((norm_y + norm_h) * img.height)
        
        crop = img.crop((x1, y1, x2, y2))
        
        fname = f"{out_dir}/sample_e_res_{res}x.png"
        crop.save(fname)
        print(f"Saved {res}x: {crop.size} pixels. Path: {fname}")

if __name__ == "__main__":
    analyze_resolution(int(sys.argv[1]))
