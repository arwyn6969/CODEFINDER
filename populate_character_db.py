#!/usr/bin/env python3
"""
Populate Character Database
===========================
Scans both Wright and Aspley editions and populates the SQLite database
for exhaustive per-character matching.

Usage:
    python3 populate_character_db.py
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from character_database import CharacterDatabase
from sonnet_print_block_scanner import SonnetPrintBlockScanner


def populate_edition(db: CharacterDatabase, source_path: str, edition_name: str):
    """Scan an edition and populate the database."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Scanning {edition_name.upper()} edition...")
    logger.info(f"{'='*60}")
    
    # Create scanner (no image saving for speed)
    scanner = SonnetPrintBlockScanner(
        source_path=source_path,
        output_dir=f"reports/db_scan_{edition_name}"
    )
    
    # Clear existing data for this edition
    db.clear_edition(edition_name)
    
    # Scan all pages
    batch = []
    for page_num in range(scanner.statistics.total_pages):
        instances = scanner.scan_page(page_num, save_images=False)
        
        for inst in instances:
            batch.append({
                'edition': edition_name,
                'page': inst.page_number,
                'character': inst.character,
                'x': inst.x,
                'y': inst.y,
                'width': inst.width,
                'height': inst.height,
                'confidence': inst.confidence,
                'is_long_s': 1 if inst.is_long_s else 0,
                'is_ligature': 1 if inst.is_ligature else 0
            })
        
        # Batch insert every 10 pages
        if (page_num + 1) % 10 == 0:
            db.insert_batch(batch)
            batch = []
            logger.info(f"  Progress: {page_num + 1}/{scanner.statistics.total_pages} pages")
    
    # Insert remaining
    if batch:
        db.insert_batch(batch)
    
    db.commit()
    logger.info(f"Completed {edition_name}: {scanner.statistics.total_characters} characters")
    return scanner.statistics.total_characters


def main():
    # Initialize database
    db = CharacterDatabase("reports/characters.db")
    
    # Scan both editions
    wright_count = populate_edition(
        db, 
        "data/sources/folger_sonnets_1609", 
        "wright"
    )
    
    aspley_count = populate_edition(
        db,
        "data/sources/folger_sonnets_1609_aspley",
        "aspley"
    )
    
    # Print summary
    stats = db.get_stats()
    logger.info("\n" + "="*60)
    logger.info("DATABASE POPULATION COMPLETE")
    logger.info("="*60)
    for edition, data in stats.items():
        logger.info(f"  {edition}: {data['count']:,} chars across {data['pages']} pages")
    
    db.close()


if __name__ == "__main__":
    main()
