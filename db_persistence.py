#!/usr/bin/env python3
"""
Database Persistence Layer for CODEFINDER

Provides functions to save OCR results to SQLite database
with full bounding box data for character extraction.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import unicodedata

# Database path
DB_PATH = Path(__file__).parent / "data" / "codefinder.db"


def get_connection(db_path: Path = None) -> sqlite3.Connection:
    """Get database connection."""
    path = db_path if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: Path = None):
    """Initialize database schema if not exists."""
    schema_path = Path(__file__).parent / "schema.sql"
    if schema_path.exists():
        conn = get_connection(db_path)
        with open(schema_path, 'r') as f:
            conn.executescript(f.read())
        conn.commit()
        conn.close()


def get_or_create_source(conn: sqlite3.Connection, name: str, path: str, 
                          total_pages: int = 0) -> int:
    """Get source ID, creating if needed."""
    cursor = conn.cursor()
    
    # Check if exists
    cursor.execute("SELECT id FROM sources WHERE name = ?", (name,))
    row = cursor.fetchone()
    
    if row:
        return row['id']
    
    # Create new source
    cursor.execute("""
        INSERT INTO sources (name, path, total_pages)
        VALUES (?, ?, ?)
    """, (name, path, total_pages))
    conn.commit()
    return cursor.lastrowid


def update_source_stats(conn: sqlite3.Connection, source_id: int,
                         total_characters: int, avg_confidence: float):
    """Update source statistics after scan completes."""
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE sources 
        SET total_characters = ?, avg_confidence = ?
        WHERE id = ?
    """, (total_characters, avg_confidence, source_id))
    conn.commit()


def save_page(conn: sqlite3.Connection, source_id: int, page_number: int,
              image_path: str, image_width: int, image_height: int,
              char_count: int, avg_confidence: float,
              long_s_count: int = 0, ligature_count: int = 0) -> int:
    """Save page data, returns page_id."""
    cursor = conn.cursor()
    
    # Check if page exists (update if so)
    cursor.execute("""
        SELECT id FROM pages WHERE source_id = ? AND page_number = ?
    """, (source_id, page_number))
    row = cursor.fetchone()
    
    if row:
        # Update existing
        cursor.execute("""
            UPDATE pages SET
                image_path = ?, image_width = ?, image_height = ?,
                char_count = ?, avg_confidence = ?,
                long_s_count = ?, ligature_count = ?
            WHERE id = ?
        """, (image_path, image_width, image_height, char_count, avg_confidence,
              long_s_count, ligature_count, row['id']))
        conn.commit()
        return row['id']
    
    # Insert new
    cursor.execute("""
        INSERT INTO pages (source_id, page_number, image_path, image_width, image_height,
                          char_count, avg_confidence, long_s_count, ligature_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (source_id, page_number, image_path, image_width, image_height,
          char_count, avg_confidence, long_s_count, ligature_count))
    conn.commit()
    return cursor.lastrowid


def save_character_instances(conn: sqlite3.Connection, page_id: int, 
                              instances: List[Dict[str, Any]]):
    """
    Batch save character instances.
    
    Each instance dict should have:
        character, category, x, y, width, height, confidence
    """
    cursor = conn.cursor()
    
    # Clear existing instances for this page
    cursor.execute("DELETE FROM character_instances WHERE page_id = ?", (page_id,))
    
    # Batch insert
    data = []
    for inst in instances:
        char = inst['character']
        try:
            if len(char) == 1:
                unicode_name = unicodedata.name(char, f"U+{ord(char):04X}")
            else:
                unicode_name = f"LIGATURE_{'_'.join(f'U+{ord(c):04X}' for c in char)}"
        except:
            unicode_name = f"U+{ord(char[0]):04X}" if char else "UNKNOWN"
        
        data.append((
            page_id,
            char,
            unicode_name,
            inst.get('category', 'other'),
            int(inst['x']),
            int(inst['y']),
            int(inst['width']),
            int(inst['height']),
            float(inst.get('confidence', 0.0))
        ))
    
    cursor.executemany("""
        INSERT INTO character_instances 
        (page_id, character, unicode_name, category, x, y, width, height, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()


def get_character_instances(conn: sqlite3.Connection, 
                            source_name: str = None,
                            character: str = None,
                            category: str = None,
                            min_confidence: float = 0,
                            limit: int = 1000) -> List[Dict]:
    """Query character instances with filters."""
    cursor = conn.cursor()
    
    query = """
        SELECT ci.*, p.page_number, p.image_path, s.name as source_name
        FROM character_instances ci
        JOIN pages p ON ci.page_id = p.id
        JOIN sources s ON p.source_id = s.id
        WHERE 1=1
    """
    params = []
    
    if source_name:
        query += " AND s.name = ?"
        params.append(source_name)
    if character:
        query += " AND ci.character = ?"
        params.append(character)
    if category:
        query += " AND ci.category = ?"
        params.append(category)
    if min_confidence > 0:
        query += " AND ci.confidence >= ?"
        params.append(min_confidence)
    
    query += f" ORDER BY p.page_number, ci.y, ci.x LIMIT {limit}"
    
    cursor.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


def get_source_stats(conn: sqlite3.Connection) -> List[Dict]:
    """Get statistics for all sources."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, 
               COUNT(DISTINCT p.id) as page_count,
               COUNT(ci.id) as instance_count
        FROM sources s
        LEFT JOIN pages p ON p.source_id = s.id
        LEFT JOIN character_instances ci ON ci.page_id = p.id
        GROUP BY s.id
    """)
    return [dict(row) for row in cursor.fetchall()]


# ============================================================================
# SORT-LEVEL CRUD (Phase 2)
# ============================================================================

def save_sort_image(conn: sqlite3.Connection, character_instance_id: int,
                    image_path: str, normalized_path: str = None,
                    width: int = 0, height: int = 0,
                    scale_factor: float = 1.0) -> int:
    """Save a sort image record, returns sort_image_id."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sort_images 
        (character_instance_id, image_path, normalized_path, width, height, scale_factor)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (character_instance_id, image_path, normalized_path, width, height, scale_factor))
    conn.commit()
    return cursor.lastrowid


def save_sort_fingerprint(conn: sqlite3.Connection, sort_image_id: int,
                          fingerprint: Dict[str, Any]) -> int:
    """Save a sort fingerprint. fingerprint dict should contain feature keys."""
    import json
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sort_fingerprints
        (sort_image_id, hu_moments, contour_descriptor, edge_density, ink_density,
         texture_lbp, perceptual_hash, feature_vector)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        sort_image_id,
        json.dumps(fingerprint.get('hu_moments')),
        json.dumps(fingerprint.get('contour_descriptor')),
        fingerprint.get('edge_density', 0.0),
        fingerprint.get('ink_density', 0.0),
        json.dumps(fingerprint.get('texture_lbp')),
        fingerprint.get('perceptual_hash', ''),
        json.dumps(fingerprint.get('feature_vector')),
    ))
    conn.commit()
    return cursor.lastrowid


def save_sort_match(conn: sqlite3.Connection, sort_image_id_1: int,
                    sort_image_id_2: int, similarity_score: float,
                    match_type: str, confidence: float = 0.0,
                    notes: str = None) -> int:
    """Save a sort match result."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sort_matches
        (sort_image_id_1, sort_image_id_2, similarity_score, match_type, confidence, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (sort_image_id_1, sort_image_id_2, similarity_score, match_type, confidence, notes))
    conn.commit()
    return cursor.lastrowid


def get_sort_images_for_character(conn: sqlite3.Connection, character: str,
                                  source_name: str = None,
                                  limit: int = 100) -> List[Dict]:
    """Get all sort images for a given character, optionally filtered by source."""
    cursor = conn.cursor()
    query = """
        SELECT si.*, ci.character, ci.category, ci.confidence,
               p.page_number, p.image_path as page_image_path,
               s.name as source_name
        FROM sort_images si
        JOIN character_instances ci ON si.character_instance_id = ci.id
        JOIN pages p ON ci.page_id = p.id
        JOIN sources s ON p.source_id = s.id
        WHERE ci.character = ?
    """
    params = [character]
    if source_name:
        query += " AND s.name = ?"
        params.append(source_name)
    query += f" ORDER BY p.page_number LIMIT {limit}"
    cursor.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


# ============================================================================
# SCANNER INTEGRATION
# ============================================================================

def persist_scan_results(source_name: str, source_path: str,
                         all_instances: list, statistics: dict,
                         page_map: Dict[int, Dict[str, Any]] = None) -> Dict[str, int]:
    """
    Persist complete scan results to the database.
    
    This is the bridge between SonnetPrintBlockScanner and the DB layer.
    
    Args:
        source_name: Human-readable source identifier (e.g., 'wright', 'aspley')
        source_path: Path to the source file/directory
        all_instances: List of CharacterInstance (or dicts with character, x, y, etc.)
        statistics: Statistics dict from the scanner
        page_map: Optional dict mapping page_number -> {image_path, width, height, ...}
        
    Returns:
        Dict with counts: {sources, pages, characters}
    """
    init_database()
    conn = get_connection()
    
    try:
        # 1. Create or get the source
        total_pages = statistics.get('total_pages', statistics.get('pages_scanned', 0))
        source_id = get_or_create_source(conn, source_name, source_path, total_pages)
        
        # Update source-level stats
        update_source_stats(
            conn, source_id,
            total_characters=statistics.get('total_characters', 0),
            avg_confidence=statistics.get('average_confidence', 0.0)
        )
        
        # 2. Group instances by page
        pages: Dict[int, list] = {}
        for inst in all_instances:
            # Support both CharacterInstance objects and dicts
            if hasattr(inst, 'page_number'):
                pn = inst.page_number
            else:
                pn = inst.get('page_number', 0)
            if pn not in pages:
                pages[pn] = []
            pages[pn].append(inst)
        
        total_chars_saved = 0
        total_pages_saved = 0
        
        # 3. Save each page and its character instances
        for page_number, instances in sorted(pages.items()):
            # Build page metadata
            image_path = ""
            image_width = 0
            image_height = 0
            
            if page_map and page_number in page_map:
                pm = page_map[page_number]
                image_path = pm.get('image_path', '')
                image_width = pm.get('width', 0)
                image_height = pm.get('height', 0)
            
            # Compute page-level stats
            confidences = []
            long_s_count = 0
            ligature_count = 0
            
            for inst in instances:
                if hasattr(inst, 'confidence'):
                    conf = inst.confidence
                else:
                    conf = inst.get('confidence', 0.0)
                if conf > 0:
                    confidences.append(conf)
                
                if hasattr(inst, 'is_long_s'):
                    if inst.is_long_s:
                        long_s_count += 1
                    if inst.is_ligature:
                        ligature_count += 1
                else:
                    if inst.get('is_long_s', False):
                        long_s_count += 1
                    if inst.get('is_ligature', False):
                        ligature_count += 1
            
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            
            page_id = save_page(
                conn, source_id, page_number,
                image_path=image_path,
                image_width=image_width,
                image_height=image_height,
                char_count=len(instances),
                avg_confidence=avg_conf,
                long_s_count=long_s_count,
                ligature_count=ligature_count
            )
            
            # Convert instances to dicts for save_character_instances
            inst_dicts = []
            for inst in instances:
                if hasattr(inst, 'character'):
                    inst_dicts.append({
                        'character': inst.character,
                        'category': getattr(inst, 'category', None) or 'other',
                        'x': inst.x,
                        'y': inst.y,
                        'width': inst.width,
                        'height': inst.height,
                        'confidence': inst.confidence,
                    })
                else:
                    inst_dicts.append(inst)
            
            save_character_instances(conn, page_id, inst_dicts)
            total_chars_saved += len(inst_dicts)
            total_pages_saved += 1
        
        result = {
            'source_id': source_id,
            'pages_saved': total_pages_saved,
            'characters_saved': total_chars_saved,
        }
        
        return result
        
    finally:
        conn.close()


if __name__ == "__main__":
    # Test the database
    init_database()
    conn = get_connection()
    
    print("Database initialized!")
    print("\nSource stats:")
    for stat in get_source_stats(conn):
        print(f"  {stat['name']}: {stat['instance_count']} characters")
    
    conn.close()
