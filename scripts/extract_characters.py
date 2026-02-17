#!/usr/bin/env python3
"""
Character Extractor Service (v2 — Fraktur + DPI Normalisation)
===============================================================

Extracts individual character sorts from page images using OCR bounding boxes.
Core of Phase 2A "Complete Typographic Inventory".

Improvements over v1:
- Fraktur OCR: Uses Fraktur+frk+deu+eng combined model for blackletter
- DPI Normalisation: Rescales all images to common height before OCR
- Expanded PSM: Configurable page segmentation mode
- Lower threshold: 50% confidence (Fraktur model is lower-confidence but correct)

Usage:
    python scripts/extract_characters.py --source <source_key> --limit <N>
    python scripts/extract_characters.py --source bsb_munich_10057380 --lang Fraktur+frk+eng
    python scripts/extract_characters.py --source hab_wolfenbuettel_178_1_theol_1s --normalize-height 2400
"""

import sys
import logging
import cv2
import pytesseract
import numpy as np
import argparse
import yaml
import re
from pathlib import Path
from dataclasses import dataclass

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extract_characters")

# Known ligatures in early modern printing
KNOWN_LIGATURES = {'ct', 'st', 'fi', 'fl', 'ff', 'ffi', 'ffl', 'ſi', 'ſt', 'ſſ'}

@dataclass
class CharacterInstance:
    character: str
    x: int; y: int; w: int; h: int
    confidence: float
    category: str = 'letter'


class CharacterExtractor:
    def __init__(self, db_path: str = "data/forensic.db",
                 lang: str = "Fraktur+frk+eng",
                 normalize_height: int = 0,
                 psm: int = 6,
                 min_confidence: float = 50.0):
        # Initialize DB
        if not Path(db_path).exists():
            from db_persistence import init_database
            init_database(Path(db_path))

        self.db_path = db_path
        self.lang = lang
        self.normalize_height = normalize_height  # 0 = no normalisation
        self.psm = psm
        self.min_confidence = min_confidence

    def get_connection(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _normalize_image(self, img):
        """Resize image to standard height, preserving aspect ratio.
        Returns (normalised_image, scale_factor).
        scale_factor is used to convert bounding boxes back to original coords.
        """
        if self.normalize_height <= 0:
            return img, 1.0

        h, w = img.shape[:2]
        if h == self.normalize_height:
            return img, 1.0

        scale = self.normalize_height / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, self.normalize_height), interpolation=cv2.INTER_LANCZOS4)
        return resized, scale

    def _build_ocr_config(self):
        """Build Tesseract config string."""
        config = f'--psm {self.psm} --oem 1'
        return config

    def process_page(self, image_path: Path, source_name: str):
        """Process a single page: OCR -> Extract -> DB."""
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return

        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            return

        orig_h, orig_w = img.shape[:2]

        # DPI normalisation
        norm_img, scale = self._normalize_image(img)
        norm_h, norm_w = norm_img.shape[:2]

        # 1. Get/Create Source & Page in DB
        from db_persistence import save_page, save_character_instances, save_sort_image

        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM sources WHERE name = ?", (source_name,))
            row = cursor.fetchone()
            if row:
                source_id = row[0]
            else:
                cursor.execute("INSERT INTO sources (name, path, total_pages) VALUES (?, ?, ?)",
                             (source_name, str(image_path.parent), 0))
                source_id = cursor.lastrowid
                conn.commit()

            page_num = self._parse_page_number(image_path.name)

            # Run OCR with Fraktur model on normalised image
            logger.info(f"Running OCR on {image_path.name} (lang={self.lang}, psm={self.psm}, "
                       f"norm={'%dpx' % self.normalize_height if self.normalize_height else 'off'})...")
            try:
                ocr_config = self._build_ocr_config()
                data = pytesseract.image_to_data(
                    norm_img,
                    lang=self.lang,
                    output_type=pytesseract.Output.DICT,
                    config=ocr_config
                )
            except Exception as e:
                logger.error(f"OCR failed on {image_path.name}: {e}")
                return

            n_boxes = len(data['text'])

            # Prepare instances
            instances_to_save = []
            crops_to_save = []

            for i in range(n_boxes):
                char = data['text'][i]
                conf = float(data['conf'][i])

                if not char.strip() or conf < self.min_confidence:
                    continue
                # Allow single chars and known ligatures
                if len(char) > 1 and char not in KNOWN_LIGATURES:
                    continue

                # Bounding box in normalised space
                nx, ny, ncw, nch = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                if ncw < 5 or nch < 5 or ncw > norm_w / 2 or nch > norm_h / 2:
                    continue

                # Convert back to original coords
                if scale != 1.0:
                    ox = int(nx / scale)
                    oy = int(ny / scale)
                    ow = int(ncw / scale)
                    oh = int(nch / scale)
                else:
                    ox, oy, ow, oh = nx, ny, ncw, nch

                inst = {
                    'character': char,
                    'category': 'letter',
                    'x': ox, 'y': oy,
                    'width': ncw, 'height': nch,       # Normalised dims for comparison
                    'width_orig': ow, 'height_orig': oh,  # Original dims
                    'confidence': conf
                }
                instances_to_save.append(inst)

                # Crop from ORIGINAL image at full resolution
                roi = img[max(0, oy):min(orig_h, oy + oh), max(0, ox):min(orig_w, ox + ow)]
                crops_to_save.append(roi)

            if not instances_to_save:
                return

            # Save Page
            avg_conf = sum(i['confidence'] for i in instances_to_save) / len(instances_to_save)
            page_id = save_page(conn, source_id, page_num, str(image_path), orig_w, orig_h,
                              len(instances_to_save), avg_conf)

            # Save instances (uses normalised width/height for cross-source comparison)
            save_character_instances(conn, page_id, instances_to_save)

            # Fetch back IDs to save images
            cursor.execute("SELECT id, x, y FROM character_instances WHERE page_id = ?", (page_id,))
            saved_instances = cursor.fetchall()
            coord_map = {(r[1], r[2]): r[0] for r in saved_instances}

            saved_sorts = 0
            for i, inst in enumerate(instances_to_save):
                key = (inst['x'], inst['y'])
                if key in coord_map:
                    cid = coord_map[key]
                    roi = crops_to_save[i]
                    if roi.size == 0:
                        continue

                    crop_rel_path = f"{source_name}/{page_num:03d}_{cid}_{inst['character']}.png"
                    full_crop_path = Path("reports/crops") / crop_rel_path
                    full_crop_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(full_crop_path), roi)

                    save_sort_image(conn, cid, str(crop_rel_path),
                                  width=inst['width'], height=inst['height'])
                    saved_sorts += 1

            logger.info(f"Saved {saved_sorts} character sorts for page {page_num}")

        finally:
            conn.close()

    def _parse_page_number(self, filename: str) -> int:
        nums = re.findall(r'\d+', filename)
        return int(nums[-1]) if nums else 0

    def run_batch(self, input_dir: Path, source_name: str, limit: int = None):
        files = sorted(list(input_dir.glob("*.jpg")))
        if limit:
            files = files[:limit]

        for f in files:
            self.process_page(f, source_name)


def main():
    parser = argparse.ArgumentParser(description="Extract characters from pages (v2 — Fraktur + DPI Norm)")
    parser.add_argument("--source", required=True, help="Source key from config")
    parser.add_argument("--limit", type=int, help="Limit number of pages")
    parser.add_argument("--lang", default="Fraktur+frk+eng",
                       help="Tesseract language model(s) (default: Fraktur+frk+eng)")
    parser.add_argument("--normalize-height", type=int, default=2400,
                       help="Normalise image height in pixels before OCR (0=off, default=2400)")
    parser.add_argument("--psm", type=int, default=6,
                       help="Tesseract page segmentation mode (default: 6)")
    parser.add_argument("--min-confidence", type=float, default=50.0,
                       help="Minimum OCR confidence threshold (default: 50)")
    args = parser.parse_args()

    # Load config to get path
    config_path = Path("data/sources/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    src_config = config["sources"].get(args.source)
    if not src_config:
        logger.error(f"Source {args.source} not found")
        return

    input_dir = Path("data/sources") / src_config["path"]

    extractor = CharacterExtractor(
        lang=args.lang,
        normalize_height=args.normalize_height,
        psm=args.psm,
        min_confidence=args.min_confidence,
    )
    extractor.run_batch(input_dir, args.source, args.limit)


if __name__ == "__main__":
    main()
