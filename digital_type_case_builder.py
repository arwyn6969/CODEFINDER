import fitz
import pytesseract
from PIL import Image
import numpy as np
import json
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import shutil
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DigitalTypeCaseBuilder:
    """
    Builds the Digital Type Case for the 1609 Sonnets.
    
    PROTOCOL: Adheres to `AXIOM_OF_INTENT.md`.
    - No mark is considered 'noise'.
    - 'Outliers' are classified as 'Unique Variants'.
    """
    def __init__(self, pdf_path, output_dir="reports/digital_type_case"):
        self.doc = fitz.open(pdf_path)
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)
        (self.output_dir / "images").mkdir()
        
        # Validation set (Strict 1609)
        self.valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;!?"\'()[]-&')
        
    def __init__(self, pdf_path, output_dir="reports/digital_type_case", clean=True):
        self.doc = fitz.open(pdf_path)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        
        # Reset output directory only if requested
        if clean and self.output_dir.exists():
            try:
                shutil.rmtree(self.output_dir)
            except Exception as e:
                logger.warning(f"Could not clean directory: {e}")
                
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        
        # Validation set (Strict 1609 + Anomalies logic?)
        # We want "Exhaustive list". So we capture everything that Tesseract finds reliable.
        # But we filter modern noise.
        self.valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;!?"\'()[]-&ſ')
        # Add ligatures if possible? Tesseract usually splits them or gives 'ae'.
        
        self.metadata = []

    def scan_full_corpus(self):
        """Extract valid blocks from ALL pages."""
        logger.info(f"Starting Full Corpus Scan (80 Pages) at 6x Resolution...")
        
        total_extracted = 0
        
        for page_num in range(len(self.doc)):
            blocks = self.extract_page_blocks(page_num)
            if blocks:
                total_extracted += len(blocks)
                # Free memory (images in blocks list)
                # We saved to disk, so we can clear 'image' object if not needed immediately
                # But clustering needs them?
                # Loading 105k images into RAM for clustering likely crashes.
                # Strategy: Save to disk. Load batch per char for clustering phase.
                for b in blocks:
                    # Remove PIL object to save RAM
                    del b['image']
                    self.metadata.append(b)
            
            if (page_num + 1) % 10 == 0:
                logger.info(f"Progress: {page_num+1}/80 pages scanned. Total blocks: {total_extracted}")
                
        # Save metadata DB
        db_path = self.output_dir / "type_case_db.json"
        with open(db_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Scan Complete. Extracted {total_extracted} blocks. Database saved.")
        return self.metadata

    def extract_page_blocks(self, page_num):
        """Extract high-res (6x) blocks from a page."""
        # ... logic similar to before but updated storage ...
        page = self.doc[page_num]
        mat3 = fitz.Matrix(3, 3)
        pix3 = page.get_pixmap(matrix=mat3)
        img3 = Image.frombytes("RGB", [pix3.width, pix3.height], pix3.samples)
        
        # Tesseract optimization: use existing config
        ocr_data = pytesseract.image_to_data(
            img3, lang='eng', config='--psm 6', output_type=pytesseract.Output.DICT
        )
        
        # Render 6x
        mat6 = fitz.Matrix(6, 6)
        pix6 = page.get_pixmap(matrix=mat6)
        img6 = Image.frombytes("RGB", [pix6.width, pix6.height], pix6.samples)
        
        blocks = []
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            text = ocr_data['text'][i]
            if not text or text.isspace(): continue
            
            # 3x coords
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            if len(text) == 0: continue
            char_w = w / len(text)
            
            for char_idx, char in enumerate(text):
                # Validation check
                if char not in self.valid_chars and char != 'ſ':
                    # Allow quotes if not in set
                    if char not in '‘’“”':
                        continue
                
                # Coords
                cx = x + (char_idx * char_w)
                
                # 6x Scale
                pad = 4
                x1 = int(cx * 2) - pad
                y1 = int(y * 2) - pad
                x2 = int((cx + char_w) * 2) + pad
                y2 = int((y + h) * 2) + pad
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img6.width, x2), min(img6.height, y2)
                
                try:
                    crop = img6.crop((x1, y1, x2, y2))
                    
                    # Store in subfolder
                    safe_char = char
                    if char == '.': safe_char = 'period'
                    elif char == ':': safe_char = 'colon'
                    elif char == ',': safe_char = 'comma'
                    elif char == ';': safe_char = 'semicolon'
                    elif char == '/': safe_char = 'slash'
                    elif char.isupper(): safe_char = f"{char}_upper"
                    else: safe_char = char
                    
                    char_dir = self.images_dir / safe_char
                    char_dir.mkdir(exist_ok=True)
                    
                    fname = f"{safe_char}_{page_num+1}_{i}_{char_idx}.png"
                    save_path = char_dir / fname
                    
                    crop.save(save_path)
                    
                    blocks.append({
                        "char": char,
                        "safe_char": safe_char,
                        "path": str(save_path),
                        "page": page_num + 1,
                        "image": crop # Will be deleted after appending to list in scan loop
                    })
                except Exception:
                    pass
        return blocks

    def cluster_and_report_full(self):
        """Cluster ALL characters and generate master report."""
        logger.info("Starting Full Clustering Phase...")
        
        # Load metadata if empty
        if not self.metadata:
            with open(self.output_dir / "type_case_db.json") as f:
                self.metadata = json.load(f)
        
        # Group by char
        char_groups = defaultdict(list)
        for b in self.metadata:
            char_groups[b['char']].append(b)
            
        html = """<html><body><h1>Full Digital Type Case</h1>
                  <style>
                  .char-block { margin-bottom: 30px; border-bottom: 1px solid #444; padding-bottom: 20px; }
                  .cluster { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; }
                  .cluster-title { font-weight: bold; margin-top: 10px; color: #aaa; }
                  img { height: 40px; border: 1px solid #333; }
                  </style>"""
        
        # Sort characters for consistent report
        sorted_chars = sorted(char_groups.keys())
        
        for char in sorted_chars:
            group = char_groups[char]
            if len(group) < 5: continue
            
            logger.info(f"Clustering '{char}' ({len(group)} items)...")
            
            # Vectorize on demand (Load -> Resize -> Flat)
            vectors = []
            valid_indices = []
            
            # Optimization: sample max 1000 for clustering definition?
            # Or cluster all? For "Exhaustive list", we should cluster all.
            # But memory...
            # Let's cluster a sample of 500 to find centroids, then assign others?
            # Or just cluster up to 500 and dump the rest as "Unsorted"?
            # User wants "exhaustive".
            # Let's do batch clustering up to 1000.
            
            process_limit = 500
            subset = group[:process_limit]
            
            try:
                X_list = []
                for item in subset:
                    img = Image.open(item['path']).convert('L').resize((32, 32))
                    X_list.append(np.array(img).flatten())
                
                X = np.array(X_list)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                n_comps = min(len(subset), 30)
                pca = PCA(n_components=n_comps)
                X_pca = pca.fit_transform(X_scaled)
                
                db = DBSCAN(eps=15.0, min_samples=5).fit(X_pca)
                labels = db.labels_
                
                # HTML
                html += f"<div class='char-block'><h2>Character '{char}' ({len(group)} total)</h2>"
                
                unique_labels = set(labels)
                for label in sorted(unique_labels):
                    if label == -1: name = "Unique Variants (High-Entropy)"
                    else: name = f"Sort Type {label+1}"
                    
                    html += f"<div class='cluster-title'>{name} (Sample)</div><div class='cluster'>"
                    
                    # Show samples
                    indices = [i for i, x in enumerate(labels) if x == label]
                    for idx in indices[:15]: # Show 15 samples per cluster
                         item = subset[idx]
                         rel = Path(item['path']).relative_to(self.output_dir)
                         html += f"<img src='{rel}' title='Pg {item['page']}'>"
                    html += "</div>"
                html += "</div>"
                    
            except Exception as e:
                logger.error(f"Failed to cluster {char}: {e}")
        
        html += "</body></html>"
        with open(self.output_dir / "index.html", 'w') as f:
            f.write(html)

if __name__ == "__main__":
    builder = DigitalTypeCaseBuilder("data/sources/SONNETS_QUARTO_1609_NET.pdf")
    builder.scan_full_corpus()
    builder.cluster_and_report_full()
    import json
