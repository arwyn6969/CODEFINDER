"""
Tests for database models and data access layer
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import tempfile
import os

from app.core.database import Base
from app.models.database_models import (
    Document, Page, Character, Word, UncertainRegion, Pattern,
    Grid, GridPattern, GeometricMeasurement, EtymologyCache
)
from app.models.data_access import (
    DocumentRepository, PageRepository, CharacterRepository,
    WordRepository, PatternRepository, GridRepository,
    EtymologyRepository, AnalysisRepository
)

class TestDatabaseModels:
    
    @pytest.fixture
    def db_session(self):
        """Create a test database session"""
        # Create in-memory SQLite database for testing
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        yield session
        
        session.close()
    
    def test_document_model(self, db_session):
        """Test Document model creation and relationships"""
        document = Document(
            filename="test.pdf",
            original_filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_size=1024,
            page_count=5,
            processing_status="uploaded"
        )
        
        db_session.add(document)
        db_session.commit()
        
        assert document.id is not None
        assert document.filename == "test.pdf"
        assert document.processing_status == "uploaded"
        assert document.upload_date is not None
    
    def test_page_model(self, db_session):
        """Test Page model creation and relationships"""
        # Create document first
        document = Document(
            filename="test.pdf",
            original_filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_size=1024
        )
        db_session.add(document)
        db_session.commit()
        
        # Create page
        page = Page(
            document_id=document.id,
            page_number=1,
            width=612.0,
            height=792.0,
            extracted_text="Sample text",
            ocr_confidence=0.95
        )
        
        db_session.add(page)
        db_session.commit()
        
        assert page.id is not None
        assert page.document_id == document.id
        assert page.page_number == 1
        assert page.ocr_confidence == 0.95
        
        # Test relationship
        assert page.document == document
        assert document.pages[0] == page
    
    def test_character_model(self, db_session):
        """Test Character model creation"""
        # Create document and page
        document = Document(
            filename="test.pdf",
            original_filename="test.pdf", 
            file_path="/path/to/test.pdf",
            file_size=1024
        )
        db_session.add(document)
        db_session.commit()
        
        page = Page(
            document_id=document.id,
            page_number=1,
            width=612.0,
            height=792.0
        )
        db_session.add(page)
        db_session.commit()
        
        # Create character
        character = Character(
            page_id=page.id,
            character='A',
            x=10.0,
            y=20.0,
            width=8.0,
            height=15.0,
            confidence=0.95,
            word_id=None,
            line_id=1,
            block_id=1
        )
        
        db_session.add(character)
        db_session.commit()
        
        assert character.id is not None
        assert character.character == 'A'
        assert character.confidence == 0.95
        assert character.page == page
    
    def test_word_model(self, db_session):
        """Test Word model creation"""
        # Create document and page
        document = Document(
            filename="test.pdf",
            original_filename="test.pdf",
            file_path="/path/to/test.pdf", 
            file_size=1024
        )
        db_session.add(document)
        db_session.commit()
        
        page = Page(
            document_id=document.id,
            page_number=1,
            width=612.0,
            height=792.0
        )
        db_session.add(page)
        db_session.commit()
        
        # Create word
        word = Word(
            page_id=page.id,
            text="hello",
            x=10.0,
            y=20.0,
            width=40.0,
            height=15.0,
            confidence=0.92,
            length=5,
            is_palindrome=False
        )
        
        db_session.add(word)
        db_session.commit()
        
        assert word.id is not None
        assert word.text == "hello"
        assert word.length == 5
        assert word.page == page
    
    def test_pattern_model(self, db_session):
        """Test Pattern model creation"""
        # Create document
        document = Document(
            filename="test.pdf",
            original_filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_size=1024
        )
        db_session.add(document)
        db_session.commit()
        
        # Create pattern
        pattern = Pattern(
            document_id=document.id,
            pattern_type="character_size_variation",
            description="Character 'A' shows size variation",
            confidence=0.85,
            severity=0.7,
            significance_score=0.8,
            pattern_data={"character": "A", "variation": 0.2}
        )
        
        db_session.add(pattern)
        db_session.commit()
        
        assert pattern.id is not None
        assert pattern.pattern_type == "character_size_variation"
        assert pattern.confidence == 0.85
        assert pattern.document == document
    
    def test_grid_model(self, db_session):
        """Test Grid model creation"""
        # Create document
        document = Document(
            filename="test.pdf",
            original_filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_size=1024
        )
        db_session.add(document)
        db_session.commit()
        
        # Create grid
        grid_data = [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']]
        grid = Grid(
            document_id=document.id,
            name="Test Grid",
            rows=3,
            columns=3,
            grid_data=grid_data,
            source_text="ABCDEFGHI"
        )
        
        db_session.add(grid)
        db_session.commit()
        
        assert grid.id is not None
        assert grid.name == "Test Grid"
        assert grid.rows == 3
        assert grid.columns == 3
        assert grid.grid_data == grid_data
        assert grid.document == document

class TestDocumentRepository:
    
    @pytest.fixture
    def db_session(self):
        """Create a test database session"""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        yield session
        
        session.close()
    
    @pytest.fixture
    def doc_repo(self, db_session):
        """Create DocumentRepository instance"""
        return DocumentRepository(db_session)
    
    def test_create_document(self, doc_repo):
        """Test document creation"""
        document = doc_repo.create_document(
            filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_size=1024,
            page_count=5
        )
        
        assert document.id is not None
        assert document.filename == "test.pdf"
        assert document.file_size == 1024
        assert document.page_count == 5
        assert document.processing_status == "uploaded"
    
    def test_get_document(self, doc_repo):
        """Test document retrieval"""
        # Create document
        document = doc_repo.create_document(
            filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_size=1024
        )
        
        # Retrieve document
        retrieved = doc_repo.get_document(document.id)
        
        assert retrieved is not None
        assert retrieved.id == document.id
        assert retrieved.filename == "test.pdf"
    
    def test_update_processing_status(self, doc_repo):
        """Test processing status update"""
        # Create document
        document = doc_repo.create_document(
            filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_size=1024
        )
        
        # Update status
        doc_repo.update_processing_status(document.id, "processing")
        
        # Verify update
        updated = doc_repo.get_document(document.id)
        assert updated.processing_status == "processing"
        assert updated.processing_started_at is not None
    
    def test_delete_document(self, doc_repo):
        """Test document deletion"""
        # Create document
        document = doc_repo.create_document(
            filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_size=1024
        )
        
        # Delete document
        result = doc_repo.delete_document(document.id)
        assert result == True
        
        # Verify deletion
        deleted = doc_repo.get_document(document.id)
        assert deleted is None

class TestCharacterRepository:
    
    @pytest.fixture
    def db_session(self):
        """Create a test database session"""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        yield session
        
        session.close()
    
    @pytest.fixture
    def char_repo(self, db_session):
        """Create CharacterRepository instance"""
        return CharacterRepository(db_session)
    
    @pytest.fixture
    def test_page(self, db_session):
        """Create a test page"""
        document = Document(
            filename="test.pdf",
            original_filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_size=1024
        )
        db_session.add(document)
        db_session.commit()
        
        page = Page(
            document_id=document.id,
            page_number=1,
            width=612.0,
            height=792.0
        )
        db_session.add(page)
        db_session.commit()
        
        return page
    
    def test_bulk_create_characters(self, char_repo, test_page):
        """Test bulk character creation"""
        characters_data = [
            {
                'page_id': test_page.id,
                'character': 'A',
                'x': 10.0,
                'y': 20.0,
                'width': 8.0,
                'height': 15.0,
                'confidence': 0.95,
                'line_id': 1,
                'block_id': 1
            },
            {
                'page_id': test_page.id,
                'character': 'B',
                'x': 20.0,
                'y': 20.0,
                'width': 8.0,
                'height': 15.0,
                'confidence': 0.92,
                'line_id': 1,
                'block_id': 1
            }
        ]
        
        characters = char_repo.bulk_create_characters(characters_data)
        
        assert len(characters) == 2
        assert characters[0].character == 'A'
        assert characters[1].character == 'B'
    
    def test_get_characters_by_type(self, char_repo, test_page):
        """Test getting characters by type"""
        # Create test characters
        characters_data = [
            {
                'page_id': test_page.id,
                'character': 'A',
                'x': 10.0,
                'y': 20.0,
                'width': 8.0,
                'height': 15.0,
                'confidence': 0.95,
                'line_id': 1,
                'block_id': 1
            },
            {
                'page_id': test_page.id,
                'character': 'A',
                'x': 30.0,
                'y': 20.0,
                'width': 9.0,
                'height': 16.0,
                'confidence': 0.90,
                'line_id': 1,
                'block_id': 1
            },
            {
                'page_id': test_page.id,
                'character': 'B',
                'x': 50.0,
                'y': 20.0,
                'width': 8.0,
                'height': 15.0,
                'confidence': 0.92,
                'line_id': 1,
                'block_id': 1
            }
        ]
        
        char_repo.bulk_create_characters(characters_data)
        
        # Get all 'A' characters
        a_chars = char_repo.get_characters_by_type(test_page.id, 'A')
        
        assert len(a_chars) == 2
        assert all(char.character == 'A' for char in a_chars)
    
    def test_get_character_statistics(self, char_repo, test_page):
        """Test character statistics calculation"""
        # Create test characters
        characters_data = [
            {
                'page_id': test_page.id,
                'character': 'A',
                'x': 10.0,
                'y': 20.0,
                'width': 8.0,
                'height': 15.0,
                'confidence': 0.95,
                'line_id': 1,
                'block_id': 1
            },
            {
                'page_id': test_page.id,
                'character': 'B',
                'x': 20.0,
                'y': 20.0,
                'width': 8.0,
                'height': 15.0,
                'confidence': 0.85,
                'line_id': 1,
                'block_id': 1
            }
        ]
        
        char_repo.bulk_create_characters(characters_data)
        
        # Get statistics
        stats = char_repo.get_character_statistics(test_page.id)
        
        assert stats['total_characters'] == 2
        assert stats['unique_characters'] == 2
        assert abs(stats['average_confidence'] - 0.9) < 0.001  # (0.95 + 0.85) / 2
        assert stats['min_confidence'] == 0.85
        assert stats['max_confidence'] == 0.95
    
    def test_get_character_frequency(self, char_repo, test_page):
        """Test character frequency calculation"""
        # Create test characters with duplicates
        characters_data = [
            {
                'page_id': test_page.id,
                'character': 'A',
                'x': 10.0,
                'y': 20.0,
                'width': 8.0,
                'height': 15.0,
                'confidence': 0.95,
                'line_id': 1,
                'block_id': 1
            },
            {
                'page_id': test_page.id,
                'character': 'A',
                'x': 30.0,
                'y': 20.0,
                'width': 8.0,
                'height': 15.0,
                'confidence': 0.90,
                'line_id': 1,
                'block_id': 1
            },
            {
                'page_id': test_page.id,
                'character': 'B',
                'x': 50.0,
                'y': 20.0,
                'width': 8.0,
                'height': 15.0,
                'confidence': 0.92,
                'line_id': 1,
                'block_id': 1
            }
        ]
        
        char_repo.bulk_create_characters(characters_data)
        
        # Get frequency
        frequency = char_repo.get_character_frequency(test_page.id)
        
        # Should be sorted by frequency (descending)
        assert len(frequency) == 2
        assert frequency[0][0] == 'A'  # Most frequent character
        assert frequency[0][1] == 2    # Frequency count
        assert frequency[1][0] == 'B'
        assert frequency[1][1] == 1


class TestGeometricRepository:
    
    @pytest.fixture
    def db_session(self):
        """Create a test database session"""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        yield session
        
        session.close()
    
    @pytest.fixture
    def geom_repo(self, db_session):
        """Create GeometricRepository instance"""
        from app.models.data_access import GeometricRepository
        return GeometricRepository(db_session)
    
    @pytest.fixture
    def test_document(self, db_session):
        """Create a test document"""
        document = Document(
            filename="test.pdf",
            original_filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_size=1024
        )
        db_session.add(document)
        db_session.commit()
        return document
    
    def test_create_measurement(self, geom_repo, test_document):
        """Test geometric measurement creation"""
        measurement = geom_repo.create_measurement(
            document_id=test_document.id,
            measurement_type="angle",
            measurement_value=90.0,
            measurement_unit="degrees",
            element_type="character",
            coordinates=[10.0, 20.0, 30.0, 40.0],
            description="Right angle between characters",
            is_significant=True,
            significance_score=0.95
        )
        
        assert measurement.id is not None
        assert measurement.measurement_type == "angle"
        assert measurement.measurement_value == 90.0
        assert measurement.is_significant == True
        assert measurement.significance_score == 0.95
    
    def test_get_measurements_by_document(self, geom_repo, test_document):
        """Test retrieving measurements by document"""
        # Create test measurements
        geom_repo.create_measurement(
            document_id=test_document.id,
            measurement_type="angle",
            measurement_value=90.0,
            significance_score=0.9
        )
        
        geom_repo.create_measurement(
            document_id=test_document.id,
            measurement_type="distance",
            measurement_value=100.0,
            significance_score=0.7
        )
        
        measurements = geom_repo.get_measurements_by_document(test_document.id)
        
        assert len(measurements) == 2
        # Should be sorted by significance score (descending)
        assert measurements[0].significance_score >= measurements[1].significance_score
    
    def test_get_measurements_by_type(self, geom_repo, test_document):
        """Test retrieving measurements by type"""
        geom_repo.create_measurement(
            document_id=test_document.id,
            measurement_type="angle",
            measurement_value=90.0
        )
        
        geom_repo.create_measurement(
            document_id=test_document.id,
            measurement_type="distance",
            measurement_value=100.0
        )
        
        angle_measurements = geom_repo.get_measurements_by_type(test_document.id, "angle")
        distance_measurements = geom_repo.get_measurements_by_type(test_document.id, "distance")
        
        assert len(angle_measurements) == 1
        assert len(distance_measurements) == 1
        assert angle_measurements[0].measurement_type == "angle"
        assert distance_measurements[0].measurement_type == "distance"
    
    def test_get_significant_measurements(self, geom_repo, test_document):
        """Test retrieving significant measurements"""
        # Create measurements with different significance scores
        geom_repo.create_measurement(
            document_id=test_document.id,
            measurement_type="angle",
            measurement_value=90.0,
            significance_score=0.9
        )
        
        geom_repo.create_measurement(
            document_id=test_document.id,
            measurement_type="angle",
            measurement_value=45.0,
            significance_score=0.6
        )
        
        significant = geom_repo.get_significant_measurements(test_document.id, min_significance=0.7)
        
        assert len(significant) == 1
        assert significant[0].significance_score == 0.9
    
    def test_find_measurements_near_value(self, geom_repo, test_document):
        """Test finding measurements near a target value"""
        geom_repo.create_measurement(
            document_id=test_document.id,
            measurement_type="angle",
            measurement_value=89.5
        )
        
        geom_repo.create_measurement(
            document_id=test_document.id,
            measurement_type="angle",
            measurement_value=45.0
        )
        
        near_90 = geom_repo.find_measurements_near_value(
            test_document.id, "angle", 90.0, tolerance=1.0
        )
        
        assert len(near_90) == 1
        assert near_90[0].measurement_value == 89.5
    
    def test_get_golden_ratio_measurements(self, geom_repo, test_document):
        """Test finding golden ratio measurements"""
        golden_ratio = 1.618033988749
        
        geom_repo.create_measurement(
            document_id=test_document.id,
            measurement_type="ratio",
            measurement_value=golden_ratio + 0.05  # Close to golden ratio
        )
        
        geom_repo.create_measurement(
            document_id=test_document.id,
            measurement_type="ratio",
            measurement_value=2.0  # Not golden ratio
        )
        
        golden_measurements = geom_repo.get_golden_ratio_measurements(test_document.id)
        
        assert len(golden_measurements) == 1
        assert abs(golden_measurements[0].measurement_value - golden_ratio) < 0.1
    
    def test_get_right_angle_measurements(self, geom_repo, test_document):
        """Test finding right angle measurements"""
        geom_repo.create_measurement(
            document_id=test_document.id,
            measurement_type="angle",
            measurement_value=89.5  # Close to 90°
        )
        
        geom_repo.create_measurement(
            document_id=test_document.id,
            measurement_type="angle",
            measurement_value=45.0  # Not a right angle
        )
        
        right_angles = geom_repo.get_right_angle_measurements(test_document.id)
        
        assert len(right_angles) == 1
        assert abs(right_angles[0].measurement_value - 90.0) < 2.0
    
    def test_get_measurement_statistics(self, geom_repo, test_document):
        """Test measurement statistics calculation"""
        # Create various measurements
        geom_repo.create_measurement(
            document_id=test_document.id,
            measurement_type="angle",
            measurement_value=90.0,
            significance_score=0.9,
            is_significant=True
        )
        
        geom_repo.create_measurement(
            document_id=test_document.id,
            measurement_type="angle",
            measurement_value=45.0,
            significance_score=0.8,
            is_significant=True
        )
        
        geom_repo.create_measurement(
            document_id=test_document.id,
            measurement_type="distance",
            measurement_value=100.0,
            significance_score=0.6,
            is_significant=False
        )
        
        stats = geom_repo.get_measurement_statistics(test_document.id)
        
        assert 'angle' in stats
        assert 'distance' in stats
        assert 'overall' in stats
        
        assert stats['angle']['count'] == 2
        assert stats['angle']['average_value'] == 67.5  # (90 + 45) / 2
        assert stats['distance']['count'] == 1
        
        assert stats['overall']['total_measurements'] == 3
        assert stats['overall']['significant_measurements'] == 2
        assert stats['overall']['significance_ratio'] == 2/3
    
    def test_bulk_create_measurements(self, geom_repo, test_document):
        """Test bulk measurement creation"""
        measurements_data = [
            {
                'document_id': test_document.id,
                'measurement_type': 'angle',
                'measurement_value': 90.0,
                'measurement_unit': 'degrees'
            },
            {
                'document_id': test_document.id,
                'measurement_type': 'angle',
                'measurement_value': 45.0,
                'measurement_unit': 'degrees'
            },
            {
                'document_id': test_document.id,
                'measurement_type': 'distance',
                'measurement_value': 100.0,
                'measurement_unit': 'pixels'
            }
        ]
        
        measurements = geom_repo.bulk_create_measurements(measurements_data)
        
        assert len(measurements) == 3
        assert all(m.document_id == test_document.id for m in measurements)
    
    def test_update_significance(self, geom_repo, test_document):
        """Test updating measurement significance"""
        measurement = geom_repo.create_measurement(
            document_id=test_document.id,
            measurement_type="angle",
            measurement_value=90.0,
            is_significant=False,
            significance_score=0.5
        )
        
        geom_repo.update_significance(
            measurement.id, 
            is_significant=True, 
            significance_score=0.95,
            pattern_relationship="right_angle_pattern"
        )
        
        # Refresh the measurement
        updated = geom_repo.db.query(GeometricMeasurement).filter(
            GeometricMeasurement.id == measurement.id
        ).first()
        
        assert updated.is_significant == True
        assert updated.significance_score == 0.95
        assert updated.pattern_relationship == "right_angle_pattern"