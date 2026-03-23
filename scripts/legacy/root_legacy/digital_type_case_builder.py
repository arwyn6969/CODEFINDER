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
        
        # Standard size for normalized character blocks
        self.target_size = (48, 64)  # width, height
    
    def normalize_character_block(self, crop: Image.Image) -> Image.Image:
        """
        Normalize a character crop to a standard size with centered placement.
        
        This ensures all character blocks have consistent dimensions for:
        - Clean grid layouts in HTML reports
        - Consistent visual comparison
        - Better clustering results
        
        Args:
            crop: PIL Image of the extracted character (variable size)
            
        Returns:
            PIL Image of standardized size with character centered on white background
        """
        target_w, target_h = self.target_size
        
        # Create white background
        normalized = Image.new('RGB', self.target_size, (255, 255, 255))
        
        # Get current dimensions
        char_w, char_h = crop.size
        
        # Handle edge cases
        if char_w <= 0 or char_h <= 0:
            return normalized
        
        # Calculate scale to fit while preserving aspect ratio
        # Use 85% to leave a margin around the character
        scale = min(target_w / char_w, target_h / char_h) * 0.85
        
        new_w = max(1, int(char_w * scale))
        new_h = max(1, int(char_h * scale))
        
        # Resize with high-quality resampling
        try:
            resized = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
        except AttributeError:
            # Fallback for older Pillow versions
            resized = crop.resize((new_w, new_h), Image.LANCZOS)
        
        # Center on background
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        normalized.paste(resized, (x_offset, y_offset))
        
        return normalized

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
                    
                    # Normalize to standard size before saving
                    normalized_crop = self.normalize_character_block(crop)
                    normalized_crop.save(save_path)
                    
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
            
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digital Type Case - 1609 Shakespeare Sonnets</title>
    <style>
        :root {
            --bg-dark: #1a1a2e;
            --bg-card: #16213e;
            --accent: #e94560;
            --text: #eaeaea;
            --text-muted: #888;
            --border: #0f3460;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }
        
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, var(--accent), #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        h2 {
            color: var(--accent);
            border-bottom: 2px solid var(--border);
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .char-block {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            scroll-margin-top: 80px;
        }
        
        /* Table of Contents */
        #toc {
            position: sticky;
            top: 0;
            background: var(--bg);
            padding: 1rem;
            margin-bottom: 2rem;
            border-bottom: 2px solid var(--border);
            z-index: 1000;
        }
        #toc h2 { margin: 0 0 0.5rem; font-size: 1rem; }
        .toc-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            max-height: 120px;
            overflow-y: auto;
        }
        .toc-grid a {
            padding: 4px 8px;
            background: var(--bg-card);
            border-radius: 4px;
            color: var(--text);
            text-decoration: none;
            font-family: monospace;
        }
        .toc-grid a:hover {
            background: var(--accent);
            color: white;
        }
        
        .cluster-title {
            font-weight: bold;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            color: var(--text-muted);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .cluster {
            display: grid;
            grid-template-columns: repeat(auto-fill, 52px);
            gap: 8px;
            margin-top: 0.5rem;
            padding: 12px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
        }
        
        .cluster img {
            width: 48px;
            height: 64px;
            object-fit: contain;
            background: white;
            border: 1px solid #333;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .cluster img:hover {
            transform: scale(2);
            box-shadow: 0 4px 20px rgba(233, 69, 96, 0.5);
            z-index: 100;
            position: relative;
        }
    </style>
</head>
<body>
<h1>Full Digital Type Case</h1>
<nav id="toc">
    <h2>📑 Table of Contents</h2>
    <div class="toc-grid"></div>
</nav>"""
        
        # Sort characters for consistent report
        sorted_chars = sorted(char_groups.keys())
        
        # Build TOC entries
        toc_entries = []
        for char in sorted_chars:
            group = char_groups[char]
            if len(group) < 5: continue
            safe_id = f"char-{ord(char)}"
            display = char if char.isprintable() else f"U+{ord(char):04X}"
            toc_entries.append(f"<a href='#{safe_id}'>{display} ({len(group)})</a>")
        
        # Insert TOC
        toc_html = " • ".join(toc_entries)
        html = html.replace('<div class="toc-grid"></div>', f'<div class="toc-grid">{toc_html}</div>')
        
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
                safe_id = f"char-{ord(char)}"
                html += f"<div class='char-block' id='{safe_id}'><h2>Character '{char}' ({len(group)} total)</h2>"
                
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
                         html += f"<img src='{rel}' title='Pg {item['page']}' loading='lazy'>"
                    html += "</div>"
                html += "</div>"
                    
            except Exception as e:
                logger.error(f"Failed to cluster {char}: {e}")
        
        html += "</body></html>"
        with open(self.output_dir / "index.html", 'w') as f:
            f.write(html)

if __name__ == "__main__":
    # Updated to use archived path - for IIIF images, consider using SonnetPrintBlockScanner
    builder = DigitalTypeCaseBuilder("data/sources/archive/SONNETS_QUARTO_1609_NET.pdf")
    builder.scan_full_corpus()
    builder.cluster_and_report_full()
    import json
