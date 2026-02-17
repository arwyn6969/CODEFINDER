#!/usr/bin/env python3
"""
Greenman Source Acquisition Script
==================================

Downloads all source materials for the Greenman Woodblock Provenance Analysis.
Handles IIIF downloads (BSB, GDZ) and HAB scraping.
Provides instructions for manual PDF downloads.

Usage:
    python scripts/acquire_sources.py [--source SOURCE_KEY] [--force]

Options:
    --source    Download only a specific source (default: all)
    --force     Re-download existing files
"""

import argparse
import sys
import yaml
import requests
import logging
from pathlib import Path
from typing import Dict, Any

# Add app to path to import services
sys.path.append(str(Path(__file__).parent.parent))

from app.services.folger_iiif_client import IIIFClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("acquire_sources")

def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)

def download_iiif_source(source_key: str, config: Dict[str, Any], output_root: Path, force: bool):
    """Download a source using IIIF."""
    logger.info(f"Starting IIIF download for {source_key}...")
    
    provenance = config.get("provenance", {})
    manifest_url = provenance.get("manifest")
    
    if not manifest_url:
        logger.error(f"No manifest URL for {source_key}")
        return

    # Initialize client with specific manifest
    client = IIIFClient(manifest_url=manifest_url)
    
    # Output directory
    source_path = config.get("path")
    output_dir = output_root / source_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    # Use max resolution
    client.download_all_pages(output_dir, size=IIIFClient.SIZE_FULL)
    logger.info(f"Finished IIIF download: {source_key}")

def download_hab_scrape(source_key: str, config: Dict[str, Any], output_root: Path, force: bool):
    """Scrape images from HAB Wolfenbüttel."""
    logger.info(f"Starting HAB scrape for {source_key}...")
    
    # Specific logic for HAB 178-1-theol-1s
    base_url = "http://diglib.hab.de/drucke/178-1-theol-1s"
    
    # Create output directory
    source_path = config.get("path")
    output_dir = output_root / source_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # HAB formatting: 5 digits, e.g., 00001.jpg
    # Try getting high-res directly. Usually it's in the same dir or 'max' or 'original'.
    # Looking at similar HAB projects, full images are often at:
    # http://diglib.hab.de/drucke/178-1-theol-1s/00001.jpg
    
    session = requests.Session()
    session.headers.update({'User-Agent': 'CODEFINDER-Research/1.0'})
    
    success_count = 0
    fail_count = 0
    max_sequential_fails = 10
    
    # Iterate pages
    for i in range(1, 1000): # Safety limit
        if fail_count >= max_sequential_fails:
            logger.info(f"Stopping after {max_sequential_fails} sequential failures. Total downloaded: {success_count}")
            break
            
        filename = f"{i:05d}.jpg"
        url = f"{base_url}/{filename}"
        output_path = output_dir / filename
        
        if output_path.exists() and not force:
            logger.debug(f"Skipping existing: {filename}")
            success_count += 1 # Count as success for logic
            fail_count = 0
            continue
            
        try:
            # Try head first
            curr_url = url
            resp = session.head(curr_url, timeout=10)
            
            if resp.status_code == 404:
                # Try fallback: sometimes they are in 'max' subfolder
                # But typically main dir works for simple views
                logger.debug(f"Not found: {curr_url}")
                fail_count += 1
                continue
            
            if resp.status_code == 200:
                # Download
                logger.info(f"Downloading HAB: {filename}")
                resp = session.get(curr_url, stream=True, timeout=30)
                resp.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                success_count += 1
                fail_count = 0
            else:
                logger.warning(f"Unexpected status {resp.status_code} for {curr_url}")
                fail_count += 1
                
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            fail_count += 1

def handle_manual_download(source_key: str, config: Dict[str, Any], output_root: Path):
    """Output instructions for manual downloads."""
    provenance = config.get("provenance", {})
    url = provenance.get("url", "Unknown URL")
    path = config.get("path")
    target_path = output_root / path
    
    print("\n" + "="*60)
    print(f"MANUAL ACTION REQUIRED: {source_key}")
    print("="*60)
    print(f"Source Type: {config.get('type')}")
    print(f"URL: {url}")
    print(f"Target Path: {target_path}")
    print("-" * 60)
    print("Please download the PDF/file manually and place it at the target path.")
    if "academia.edu" in url:
        print("Note: Academia.edu requires login. Use the 'Download PDF' button on the page.")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Acquire Greenman sources")
    parser.add_argument("--source", help="Specific source key to download")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "data/sources/config.yaml"
    data_sources_dir = base_dir / "data/sources"
    
    if not config_path.exists():
        logger.error(f"Config not found at {config_path}")
        return
        
    config = load_config(config_path)
    sources = config.get("sources", {})
    
    keys_to_process = [args.source] if args.source else sources.keys()
    
    for key in keys_to_process:
        if key not in sources:
            logger.warning(f"Source '{key}' not found in configuration.")
            continue
            
        src_config = sources[key]
        src_type = src_config.get("type")
        
        # Determine strict processing
        # Skip legacy PDF or gutenberg unless explicitly requested
        if key in ["legacy_pdf", "gutenberg_text"] and not args.source:
             continue
             
        try:
            if src_type == "iiif_images":
                download_iiif_source(key, src_config, data_sources_dir, args.force)
            elif src_type == "html_scrape":
                download_hab_scrape(key, src_config, data_sources_dir, args.force)
            elif src_type == "pdf":
                handle_manual_download(key, src_config, data_sources_dir)
            else:
                logger.warning(f"Unknown source type '{src_type}' for {key}")
        except Exception as e:
            logger.error(f"Failed to process {key}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
