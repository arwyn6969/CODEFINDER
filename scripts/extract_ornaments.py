#!/usr/bin/env python3
"""
Ornament Extraction Script
==========================

Runs the OrnamentExtractor service on downloaded source images to locate
potential greenman woodblocks, decorative initials, and other non-text elements.

Usage:
    python scripts/extract_ornaments.py [--source SOURCE_KEY] [--limit N]

Output:
    reports/ornaments/<source_name>/images/
    reports/ornaments/<source_name>/candidates.jsonl
"""

import argparse
import sys
import yaml
import logging
import shutil
import re
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.ornament_extractor import OrnamentExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("extract_ornaments")

def load_config(config_path: Path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def run_extraction(source_key: str, config: dict, limit: int = None, output_root: Path = None):
    """Run extraction for a specific source."""
    source_path = config.get("path")
    if not source_path:
        logger.warning(f"No path for {source_key}")
        return

    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "data/sources" / source_path
    output_root = output_root or (base_dir / "reports/ornaments")
    output_dir = output_root / source_key
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)

    logger.info(f"Processing {source_key} from {input_dir}")
    
    extractor = OrnamentExtractor(debug_output=output_dir / "debug")
    image_files = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        image_files.extend(input_dir.glob(pattern))
    image_files = sorted(
        set(image_files),
        key=lambda path: (
            int(re.findall(r"\d+", path.name)[-1]) if re.findall(r"\d+", path.name) else -1,
            path.name,
        ),
    )
    if limit:
        image_files = image_files[:limit]

    logger.info(f"Scanning {len(image_files)} page images")
    for image_path in image_files:
        candidates = extractor.extract_from_page(image_path)
        if candidates:
            extractor.save_candidates(candidates, output_dir)
    logger.info(f"Finished extraction for {source_key}. Results in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract ornaments from source images")
    parser.add_argument("--source", help="Specific source key")
    parser.add_argument("--limit", type=int, help="Limit number of pages")
    parser.add_argument("--output-root", default="reports/ornaments",
                        help="Root output directory for extracted ornament candidates")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "data/sources/config.yaml"
    
    if not config_path.exists():
        logger.error("Config not found")
        return
        
    config = load_config(config_path)
    sources = config.get("sources", {})
    
    greenman_sources = [
        "bsb_munich_10057380",
        "gdz_goettingen_ppn777246686",
        "hab_wolfenbuettel_178_1_theol_1s"
    ]
    
    keys = [args.source] if args.source else greenman_sources
    
    for key in keys:
        if key not in sources:
            logger.warning(f"Source {key} not found in config")
            continue
            
        src_config = sources[key]
        if src_config.get("type") in ["iiif_images", "html_scrape"]:
            run_extraction(key, src_config, args.limit, base_dir / args.output_root)
        else:
            logger.info(f"Skipping {key} (type {src_config.get('type')} not supported yet)")

if __name__ == "__main__":
    main()
