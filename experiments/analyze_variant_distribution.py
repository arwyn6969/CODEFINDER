import fitz
import pytesseract
from PIL import Image
import io
import json
import numpy as np
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from pathlib import Path
from sonnet_print_block_scanner import SonnetPrintBlockScanner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VariantMetric:
    char: str
    page: int
    width: int
    height: int
    ink_density: float
    aspect_ratio: float
    cluster_id: int = -1  # To be assigned
    description: str = ""

class VariantAnalyzer(SonnetPrintBlockScanner):
    def __init__(self, pdf_path: str):
        super().__init__(pdf_path, output_dir="reports/variant_analysis")
        self.targets = ['.', 'T', 't']
        self.metrics: List[VariantMetric] = []
        
    def scan_page_variants(self, page_num: int):
        """Scans page specifically for variants of target characters."""
        page = self.doc[page_num]
        mat = fitz.Matrix(3, 3) # 3x zoom for metric precision
        pix = page.get_pixmap(matrix=mat)
        page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Run Tesseract
        try:
            ocr_data = pytesseract.image_to_data(
                page_image, 
                lang='eng', 
                config='--psm 6', 
                output_type=pytesseract.Output.DICT
            )
        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")
            return

        text_blocks = ocr_data['text']
        
        for i, text in enumerate(text_blocks):
            if not text or text.isspace():
                continue
            
            # Bounding box
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]
            
            char_width = w / len(text)
            
            for char_idx, char in enumerate(text):
                if char not in self.targets:
                    continue
                
                # Check valid logic from scanner? 
                # Just raw for now
                
                # Calculate pos
                cx = int(x + (char_idx * char_width))
                cy = int(y)
                cw = int(char_width)
                ch = int(h)
                
                # Crop and measure
                # Expand slightly to capture edges? No, strict box better for density
                # But density needs tight crop.
                # Let's crop exactly (+1 px margin)
                try:
                    crop = page_image.crop((max(0, cx-1), max(0, cy-1), min(page_image.width, cx+cw+1), min(page_image.height, cy+ch+1)))
                    
                    # Convert to binary
                    bn = crop.convert('L')
                    # Count ink (assuming black ink on white)
                    # Threshold
                    ink_pixels = 0
                    total_pixels = 0
                    width, height = bn.size
                    for py in range(height):
                        for px in range(width):
                            if bn.getpixel((px, py)) < 128:
                                ink_pixels += 1
                            total_pixels += 1
                    
                    density = ink_pixels / max(1, total_pixels)
                    ratio = height / max(1, width)
                    
                    self.metrics.append(VariantMetric(
                        char=char,
                        page=page_num+1,
                        width=width,
                        height=height,
                        ink_density=density,
                        aspect_ratio=ratio
                    ))
                    
                except Exception as e:
                    pass

    def analyze_clusters(self):
        """Simple clustering logic."""
        # Simple clustering based on Height and Density
        # For '.', we classify by Density (Boldness) and Size
        logger.info("Clustering collected metrics...")
        
        clusters = {}
        for target in self.targets:
            items = [m for m in self.metrics if m.char == target]
            if not items: continue
            
            # Simple stats
            avg_w = np.mean([m.width for m in items])
            avg_h = np.mean([m.height for m in items])
            avg_d = np.mean([m.ink_density for m in items])
            
            logger.info(f"Stats for '{target}': Count={len(items)}, AvgW={avg_w:.1f}, AvgH={avg_h:.1f}, AvgDens={avg_d:.2f}")
            
            # Define clusters based on deviation
            type_counts = defaultdict(int)
            
            for m in items:
                # Classify '.'
                if target == '.':
                    # Size check
                    if m.width > avg_w * 1.5: strength = "Large"
                    elif m.width < avg_w * 0.8: strength = "Small"
                    else: strength = "Medium"
                    
                    # Density check
                    if m.ink_density > avg_d * 1.2: weight = "Heavy"
                    elif m.ink_density < avg_d * 0.8: weight = "Faint"
                    else: weight = "Normal"
                    
                    m.description = f"{strength} {weight} Period"
                    
                elif target == 'T':
                    # Check broadness
                    if m.aspect_ratio < 1.2: style = "Wide/Squat"
                    elif m.aspect_ratio > 1.8: style = "Tall/Condensed"
                    else: style = "Standard"
                    
                    m.description = f"{style} T"
                
                elif target == 't':
                     # Lowercase t checks
                    start_desc = "Standard"
                    if m.height > avg_h * 1.2: start_desc = "Tall Ascender"
                    
                    m.description = f"{start_desc} t"
                
                type_counts[m.description] += 1
            
            clusters[target] = dict(type_counts)
            
        return clusters

if __name__ == "__main__":
    from collections import defaultdict
    analyzer = VariantAnalyzer("data/sources/SONNETS_QUARTO_1609_NET.pdf")
    # Scan first 10 pages for sample
    print("Scanning Pages 1-10 for variations...")
    for p in range(10):
        analyzer.scan_page_variants(p)
        print(f"Scanned page {p+1}...")
        
    results = analyzer.analyze_clusters()
    
    print("\n VARIANT ANALYSIS REPORT")
    print("=======================")
    for char, counts in results.items():
        print(f"\nCharacter '{char}' Variations:")
        total = sum(counts.values())
        for desc, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {desc}: {count} ({count/total*100:.1f}%)")
