#!/usr/bin/env python3
"""
Character Database Module
=========================
SQLite-based storage for per-character OCR data.
Enables fast querying for exhaustive matching and residual analysis.

Usage:
    from character_database import CharacterDatabase
    
    db = CharacterDatabase("characters.db")
    db.insert_character(edition="wright", page=11, char="A", x=100, y=200, ...)
    db.get_characters_for_page(edition="wright", page=11)
"""

import sqlite3
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Iterator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CharacterRecord:
    """Database record for a character instance."""
    id: int
    edition: str  # 'wright' or 'aspley'
    page: int
    character: str
    x: float
    y: float
    width: float
    height: float
    confidence: float
    is_long_s: bool
    is_ligature: bool
    matched_id: Optional[int] = None  # ID of matched char in other edition
    residual_distance: Optional[float] = None


class CharacterDatabase:
    """SQLite database for character storage and matching."""
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS characters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        edition TEXT NOT NULL,
        page INTEGER NOT NULL,
        character TEXT NOT NULL,
        x REAL NOT NULL,
        y REAL NOT NULL,
        width REAL NOT NULL,
        height REAL NOT NULL,
        confidence REAL DEFAULT 0.0,
        is_long_s INTEGER DEFAULT 0,
        is_ligature INTEGER DEFAULT 0,
        matched_id INTEGER DEFAULT NULL,
        residual_distance REAL DEFAULT NULL
    );
    
    CREATE INDEX IF NOT EXISTS idx_edition_page ON characters(edition, page);
    CREATE INDEX IF NOT EXISTS idx_character ON characters(character);
    CREATE INDEX IF NOT EXISTS idx_matched ON characters(matched_id);
    """
    
    def __init__(self, db_path: str = "reports/characters.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info(f"Character database initialized: {self.db_path}")
    
    def _init_schema(self):
        """Create tables if they don't exist."""
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()
    
    def clear_edition(self, edition: str):
        """Remove all characters for an edition."""
        self.conn.execute("DELETE FROM characters WHERE edition = ?", (edition,))
        self.conn.commit()
        logger.info(f"Cleared all characters for edition: {edition}")
    
    def insert_character(
        self,
        edition: str,
        page: int,
        character: str,
        x: float,
        y: float,
        width: float,
        height: float,
        confidence: float = 0.0,
        is_long_s: bool = False,
        is_ligature: bool = False
    ) -> int:
        """Insert a single character and return its ID."""
        cursor = self.conn.execute(
            """
            INSERT INTO characters 
            (edition, page, character, x, y, width, height, confidence, is_long_s, is_ligature)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (edition, page, character, x, y, width, height, confidence, 
             1 if is_long_s else 0, 1 if is_ligature else 0)
        )
        return cursor.lastrowid
    
    def insert_batch(self, characters: List[dict]):
        """Insert multiple characters at once."""
        self.conn.executemany(
            """
            INSERT INTO characters 
            (edition, page, character, x, y, width, height, confidence, is_long_s, is_ligature)
            VALUES (:edition, :page, :character, :x, :y, :width, :height, :confidence, :is_long_s, :is_ligature)
            """,
            characters
        )
        self.conn.commit()
    
    def commit(self):
        """Commit pending transactions."""
        self.conn.commit()
    
    def get_characters_for_page(self, edition: str, page: int) -> List[CharacterRecord]:
        """Get all characters for a specific page."""
        cursor = self.conn.execute(
            "SELECT * FROM characters WHERE edition = ? AND page = ? ORDER BY y, x",
            (edition, page)
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]
    
    def get_page_numbers(self, edition: str) -> List[int]:
        """Get list of all page numbers for an edition."""
        cursor = self.conn.execute(
            "SELECT DISTINCT page FROM characters WHERE edition = ? ORDER BY page",
            (edition,)
        )
        return [row[0] for row in cursor.fetchall()]
    
    def get_stats(self) -> dict:
        """Get summary statistics."""
        cursor = self.conn.execute("""
            SELECT edition, COUNT(*) as count, COUNT(DISTINCT page) as pages
            FROM characters GROUP BY edition
        """)
        stats = {}
        for row in cursor.fetchall():
            stats[row['edition']] = {'count': row['count'], 'pages': row['pages']}
        return stats
    
    def update_match(self, char_id: int, matched_id: int, residual_distance: float):
        """Update a character with its match information."""
        self.conn.execute(
            "UPDATE characters SET matched_id = ?, residual_distance = ? WHERE id = ?",
            (matched_id, residual_distance, char_id)
        )
    
    def get_unmatched_count(self, edition: str) -> int:
        """Count characters without matches."""
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM characters WHERE edition = ? AND matched_id IS NULL",
            (edition,)
        )
        return cursor.fetchone()[0]
    
    def get_residual_stats(self, edition: str) -> dict:
        """Get residual distance statistics."""
        cursor = self.conn.execute("""
            SELECT 
                AVG(residual_distance) as mean,
                MAX(residual_distance) as max,
                MIN(residual_distance) as min,
                COUNT(*) as matched_count
            FROM characters 
            WHERE edition = ? AND residual_distance IS NOT NULL
        """, (edition,))
        row = cursor.fetchone()
        return {
            'mean': row['mean'],
            'max': row['max'],
            'min': row['min'],
            'matched_count': row['matched_count']
        }
    
    def _row_to_record(self, row) -> CharacterRecord:
        """Convert a database row to a CharacterRecord."""
        return CharacterRecord(
            id=row['id'],
            edition=row['edition'],
            page=row['page'],
            character=row['character'],
            x=row['x'],
            y=row['y'],
            width=row['width'],
            height=row['height'],
            confidence=row['confidence'],
            is_long_s=bool(row['is_long_s']),
            is_ligature=bool(row['is_ligature']),
            matched_id=row['matched_id'],
            residual_distance=row['residual_distance']
        )
    
    def close(self):
        """Close the database connection."""
        self.conn.close()


def import_from_scanner(scanner, edition: str, db: CharacterDatabase):
    """
    Import character data from a completed scanner run into the database.
    
    Args:
        scanner: SonnetPrintBlockScanner instance (after scan)
        edition: Edition name ('wright' or 'aspley')
        db: CharacterDatabase instance
    """
    db.clear_edition(edition)
    
    batch = []
    for char, entry in scanner.character_catalogue.items():
        for instance in entry.instances:
            batch.append({
                'edition': edition,
                'page': instance.page_number,
                'character': instance.character,
                'x': instance.x,
                'y': instance.y,
                'width': instance.width,
                'height': instance.height,
                'confidence': instance.confidence,
                'is_long_s': 1 if instance.is_long_s else 0,
                'is_ligature': 1 if instance.is_ligature else 0
            })
    
    db.insert_batch(batch)
    logger.info(f"Imported {len(batch)} characters for edition '{edition}'")


if __name__ == "__main__":
    # Test the database
    db = CharacterDatabase("reports/test_characters.db")
    
    # Insert test data
    db.insert_character("wright", 11, "A", 100.0, 200.0, 20.0, 30.0, 85.0)
    db.insert_character("wright", 11, "B", 120.0, 200.0, 18.0, 30.0, 90.0)
    db.commit()
    
    # Query
    chars = db.get_characters_for_page("wright", 11)
    print(f"Found {len(chars)} characters on page 11")
    for c in chars:
        print(f"  {c.character} at ({c.x}, {c.y})")
    
    # Stats
    print(f"Stats: {db.get_stats()}")
    
    db.close()
    print("Database test complete.")
