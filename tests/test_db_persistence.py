"""
Tests for db_persistence module.

Verifies the Phase 0 infrastructure: schema creation, CRUD operations
for sources/pages/character_instances, sort-level tables, and the
persist_scan_results bridge function.
"""

import sqlite3
import tempfile
import os
import pytest
from pathlib import Path
from unittest.mock import patch
from dataclasses import dataclass, field
from typing import Optional


# ── Fixture: temp DB ────────────────────────────────────────────────────────

@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_codefinder.db"
    
    # Patch DB_PATH so all db_persistence functions use our temp DB
    with patch("db_persistence.DB_PATH", db_path):
        import db_persistence
        db_persistence.init_database()
        yield db_path


@pytest.fixture
def conn(temp_db):
    """Get a connection to the temp database."""
    with patch("db_persistence.DB_PATH", temp_db):
        import db_persistence
        connection = db_persistence.get_connection()
        yield connection
        connection.close()


# ── Test: Schema Initialization ─────────────────────────────────────────────

class TestSchemaInit:
    """Verify all tables and indexes are created correctly."""
    
    def test_tables_created(self, conn):
        """All expected tables exist after init."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row['name'] for row in cursor.fetchall()}
        
        expected = {
            'sources', 'pages', 'character_instances',
            'sort_images', 'sort_fingerprints', 'sort_matches',
        }
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"
    
    def test_indexes_created(self, conn):
        """Critical indexes exist."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indexes = {row['name'] for row in cursor.fetchall()}
        
        expected_indexes = {
            'idx_char_instances_char',
            'idx_char_instances_page',
            'idx_sort_images_instance',
            'idx_sort_fingerprints_image',
            'idx_sort_matches_pair',
        }
        assert expected_indexes.issubset(indexes), f"Missing indexes: {expected_indexes - indexes}"
    
    def test_views_created(self, conn):
        """Views for common queries exist."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='view'"
        )
        views = {row['name'] for row in cursor.fetchall()}
        assert 'character_frequency' in views
        assert 'page_stats' in views


# ── Test: Source CRUD ───────────────────────────────────────────────────────

class TestSourceCRUD:
    
    def test_create_source(self, temp_db):
        with patch("db_persistence.DB_PATH", temp_db):
            import db_persistence
            conn = db_persistence.get_connection()
            
            source_id = db_persistence.get_or_create_source(
                conn, "wright", "/data/wright", total_pages=80
            )
            assert source_id is not None
            assert source_id > 0
            conn.close()
    
    def test_idempotent_source(self, temp_db):
        """Creating the same source twice returns the same ID."""
        with patch("db_persistence.DB_PATH", temp_db):
            import db_persistence
            conn = db_persistence.get_connection()
            
            id1 = db_persistence.get_or_create_source(conn, "wright", "/data/wright")
            id2 = db_persistence.get_or_create_source(conn, "wright", "/data/wright")
            assert id1 == id2
            conn.close()
    
    def test_update_source_stats(self, temp_db):
        with patch("db_persistence.DB_PATH", temp_db):
            import db_persistence
            conn = db_persistence.get_connection()
            
            source_id = db_persistence.get_or_create_source(conn, "aspley", "/data/aspley")
            db_persistence.update_source_stats(conn, source_id, 50000, 85.5)
            
            cursor = conn.cursor()
            cursor.execute("SELECT total_characters, avg_confidence FROM sources WHERE id = ?", (source_id,))
            row = cursor.fetchone()
            assert row['total_characters'] == 50000
            assert abs(row['avg_confidence'] - 85.5) < 0.01
            conn.close()


# ── Test: Page CRUD ─────────────────────────────────────────────────────────

class TestPageCRUD:
    
    def test_save_page(self, temp_db):
        with patch("db_persistence.DB_PATH", temp_db):
            import db_persistence
            conn = db_persistence.get_connection()
            
            source_id = db_persistence.get_or_create_source(conn, "wright", "/data/wright")
            page_id = db_persistence.save_page(
                conn, source_id, 1,
                image_path="/images/page_001.png",
                image_width=7000, image_height=9000,
                char_count=500, avg_confidence=88.0,
                long_s_count=12, ligature_count=3
            )
            assert page_id > 0
            conn.close()
    
    def test_page_upsert(self, temp_db):
        """Saving same page again updates rather than duplicates."""
        with patch("db_persistence.DB_PATH", temp_db):
            import db_persistence
            conn = db_persistence.get_connection()
            
            source_id = db_persistence.get_or_create_source(conn, "wright", "/data/wright")
            
            id1 = db_persistence.save_page(
                conn, source_id, 1, "/img/p1.png", 7000, 9000, 500, 88.0
            )
            id2 = db_persistence.save_page(
                conn, source_id, 1, "/img/p1.png", 7000, 9000, 505, 89.0
            )
            assert id1 == id2
            
            # Verify updated values
            cursor = conn.cursor()
            cursor.execute("SELECT char_count FROM pages WHERE id = ?", (id1,))
            assert cursor.fetchone()['char_count'] == 505
            conn.close()


# ── Test: Character Instance CRUD ───────────────────────────────────────────

class TestCharacterInstanceCRUD:
    
    def _create_page(self, conn, db_persistence):
        source_id = db_persistence.get_or_create_source(conn, "wright", "/data/wright")
        return db_persistence.save_page(
            conn, source_id, 1, "/img/p1.png", 7000, 9000, 0, 0.0
        )
    
    def test_save_character_instances(self, temp_db):
        with patch("db_persistence.DB_PATH", temp_db):
            import db_persistence
            conn = db_persistence.get_connection()
            page_id = self._create_page(conn, db_persistence)
            
            instances = [
                {'character': 'S', 'category': 'uppercase', 'x': 100, 'y': 200, 'width': 30, 'height': 40, 'confidence': 92.0},
                {'character': 'h', 'category': 'lowercase', 'x': 130, 'y': 200, 'width': 25, 'height': 38, 'confidence': 88.0},
                {'character': 'ſ', 'category': 'long_s', 'x': 155, 'y': 200, 'width': 20, 'height': 42, 'confidence': 75.0},
            ]
            
            db_persistence.save_character_instances(conn, page_id, instances)
            
            # Verify
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as cnt FROM character_instances WHERE page_id = ?", (page_id,))
            assert cursor.fetchone()['cnt'] == 3
            conn.close()
    
    def test_get_character_instances_filter(self, temp_db):
        with patch("db_persistence.DB_PATH", temp_db):
            import db_persistence
            conn = db_persistence.get_connection()
            page_id = self._create_page(conn, db_persistence)
            
            instances = [
                {'character': 'e', 'category': 'lowercase', 'x': 100, 'y': 200, 'width': 25, 'height': 35, 'confidence': 90.0},
                {'character': 'e', 'category': 'lowercase', 'x': 200, 'y': 200, 'width': 25, 'height': 35, 'confidence': 85.0},
                {'character': 't', 'category': 'lowercase', 'x': 300, 'y': 200, 'width': 20, 'height': 38, 'confidence': 91.0},
            ]
            db_persistence.save_character_instances(conn, page_id, instances)
            
            # Filter by character
            results = db_persistence.get_character_instances(conn, character='e')
            assert len(results) == 2
            
            # Filter by min confidence
            results = db_persistence.get_character_instances(conn, min_confidence=89.0)
            assert len(results) == 2  # 'e' at 90 and 't' at 91
            conn.close()


# ── Test: Sort-Level CRUD ───────────────────────────────────────────────────

class TestSortLevelCRUD:
    
    def _setup_character(self, conn, db_persistence):
        """Create a source, page, and character instance."""
        source_id = db_persistence.get_or_create_source(conn, "wright", "/data/wright")
        page_id = db_persistence.save_page(
            conn, source_id, 1, "/img/p1.png", 7000, 9000, 1, 90.0
        )
        db_persistence.save_character_instances(conn, page_id, [
            {'character': 'A', 'category': 'uppercase', 'x': 100, 'y': 200, 'width': 30, 'height': 40, 'confidence': 95.0}
        ])
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM character_instances WHERE page_id = ? LIMIT 1", (page_id,))
        return cursor.fetchone()['id']
    
    def test_save_sort_image(self, temp_db):
        with patch("db_persistence.DB_PATH", temp_db):
            import db_persistence
            conn = db_persistence.get_connection()
            ci_id = self._setup_character(conn, db_persistence)
            
            sort_id = db_persistence.save_sort_image(
                conn, ci_id, "/sorts/A_001.png",
                normalized_path="/sorts/A_001_norm.png",
                width=48, height=64, scale_factor=3.0
            )
            assert sort_id > 0
            conn.close()
    
    def test_save_sort_fingerprint(self, temp_db):
        with patch("db_persistence.DB_PATH", temp_db):
            import db_persistence
            conn = db_persistence.get_connection()
            ci_id = self._setup_character(conn, db_persistence)
            sort_id = db_persistence.save_sort_image(conn, ci_id, "/sorts/A_001.png")
            
            fp_id = db_persistence.save_sort_fingerprint(conn, sort_id, {
                'hu_moments': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                'edge_density': 0.35,
                'ink_density': 0.42,
                'perceptual_hash': 'a1b2c3d4',
            })
            assert fp_id > 0
            conn.close()
    
    def test_save_sort_match(self, temp_db):
        with patch("db_persistence.DB_PATH", temp_db):
            import db_persistence
            conn = db_persistence.get_connection()
            ci_id = self._setup_character(conn, db_persistence)
            
            sort_id_1 = db_persistence.save_sort_image(conn, ci_id, "/sorts/A_001.png")
            sort_id_2 = db_persistence.save_sort_image(conn, ci_id, "/sorts/A_002.png")
            
            match_id = db_persistence.save_sort_match(
                conn, sort_id_1, sort_id_2,
                similarity_score=0.92,
                match_type='same_sort',
                confidence=0.88,
                notes='Strong wear-mark correlation'
            )
            assert match_id > 0
            conn.close()


# ── Test: persist_scan_results ──────────────────────────────────────────────

class TestPersistScanResults:
    
    @dataclass
    class MockCharacterInstance:
        """Mimics the scanner's CharacterInstance dataclass."""
        character: str
        page_number: int
        x: float
        y: float
        width: float
        height: float
        confidence: float = 0.0
        is_long_s: bool = False
        is_ligature: bool = False
        category: Optional[str] = None
    
    def test_persist_scan_results(self, temp_db):
        with patch("db_persistence.DB_PATH", temp_db):
            import db_persistence
            
            instances = [
                self.MockCharacterInstance('T', 1, 100, 200, 30, 40, 95.0, category='uppercase'),
                self.MockCharacterInstance('h', 1, 130, 200, 25, 38, 88.0, category='lowercase'),
                self.MockCharacterInstance('e', 1, 155, 200, 25, 35, 90.0, category='lowercase'),
                self.MockCharacterInstance('ſ', 2, 100, 200, 20, 42, 75.0, is_long_s=True, category='long_s'),
                self.MockCharacterInstance('e', 2, 120, 200, 25, 35, 91.0, category='lowercase'),
            ]
            
            statistics = {
                'total_pages': 2,
                'pages_scanned': 2,
                'total_characters': 5,
                'average_confidence': 87.8,
            }
            
            result = db_persistence.persist_scan_results(
                source_name='test_wright',
                source_path='/test/wright',
                all_instances=instances,
                statistics=statistics,
            )
            
            assert result['pages_saved'] == 2
            assert result['characters_saved'] == 5
            assert result['source_id'] > 0
            
            # Verify data in DB
            conn = db_persistence.get_connection()
            
            # Check source
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sources WHERE name = 'test_wright'")
            source = cursor.fetchone()
            assert source is not None
            assert source['total_characters'] == 5
            
            # Check pages
            cursor.execute("SELECT COUNT(*) as cnt FROM pages")
            assert cursor.fetchone()['cnt'] == 2
            
            # Check character instances
            cursor.execute("SELECT COUNT(*) as cnt FROM character_instances")
            assert cursor.fetchone()['cnt'] == 5
            
            # Check long-s was tracked
            cursor.execute("SELECT long_s_count FROM pages WHERE page_number = 2")
            assert cursor.fetchone()['long_s_count'] == 1
            
            conn.close()
    
    def test_persist_with_dicts(self, temp_db):
        """persist_scan_results also accepts plain dicts."""
        with patch("db_persistence.DB_PATH", temp_db):
            import db_persistence
            
            instances = [
                {'character': 'A', 'page_number': 1, 'x': 100, 'y': 200,
                 'width': 30, 'height': 40, 'confidence': 95.0, 'category': 'uppercase'},
            ]
            
            result = db_persistence.persist_scan_results(
                source_name='test_dict',
                source_path='/test/dict',
                all_instances=instances,
                statistics={'total_pages': 1, 'total_characters': 1, 'average_confidence': 95.0},
            )
            
            assert result['characters_saved'] == 1
