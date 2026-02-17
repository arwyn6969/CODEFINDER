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

def run_extraction(source_key: str, config: dict, limit: int = None):
    """Run extraction for a specific source."""
    source_path = config.get("path")
    if not source_path:
        logger.warning(f"No path for {source_key}")
        return

    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "data/sources" / source_path
    output_dir = base_dir / "reports/ornaments" / source_key
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    logger.info(f"Processing {source_key} from {input_dir}")
    
    extractor = OrnamentExtractor(debug_output=output_dir / "debug")
    extractor.run_batch(input_dir, output_dir)
    logger.info(f"Finished extraction for {source_key}. Results in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract ornaments from source images")
    parser.add_argument("--source", help="Specific source key")
    parser.add_argument("--limit", type=int, help="Limit number of pages")
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
            run_extraction(key, src_config, args.limit)
        else:
            logger.info(f"Skipping {key} (type {src_config.get('type')} not supported yet)")

if __name__ == "__main__":
    main()
