"""
Database models for Ancient Text Analyzer
SQLAlchemy models for storing text analysis results
"""
from sqlalchemy import Column, Integer, String, Float, Text, Boolean, DateTime, ForeignKey, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any
import uuid

from app.core.database import Base

class Document(Base):
    """
    Document model for storing uploaded documents and their metadata
    """
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), default="application/pdf")
    
    # Document metadata
    page_count = Column(Integer)
    creation_date = Column(DateTime)
    upload_date = Column(DateTime, default=func.now())
    
    # Processing status
    processing_status = Column(String(50), default="uploaded")  # uploaded, processing, completed, failed
    processing_started_at = Column(DateTime)
    processing_completed_at = Column(DateTime)
    error_message = Column(Text)
    
    # Document analysis summary
    total_characters = Column(Integer)
    total_words = Column(Integer)
    average_confidence = Column(Float)
    # Added fields for pipeline and routes alignment
    content = Column(Text)  # Concatenated OCR text for the document
    total_pages = Column(Integer)
    analysis_complete = Column(Boolean, default=False)
    uploaded_by = Column(String(100))
    
    # Relationships
    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")
    patterns = relationship("Pattern", back_populates="document", cascade="all, delete-orphan")
    grids = relationship("Grid", back_populates="document", cascade="all, delete-orphan")
    # Cross-document relationships (imported from cross_document_models)
    # cross_patterns = relationship("CrossDocumentPattern", secondary="pattern_document_associations", back_populates="documents")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.processing_status}')>"

class Page(Base):
    """
    Page model for storing individual page data and OCR results
    """
    __tablename__ = "pages"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    
    # Page dimensions and metadata
    width = Column(Float)
    height = Column(Float)
    dpi = Column(Integer, default=300)
    rotation = Column(Integer, default=0)
    
    # OCR results
    extracted_text = Column(Text)
    ocr_confidence = Column(Float)
    processing_time = Column(Float)
    image_quality_score = Column(Float)
    image_path = Column(String(500))
    
    # Processing metadata
    preprocessing_steps = Column(JSON)  # Store preprocessing steps as JSON
    ocr_engine_version = Column(String(50))
    processed_at = Column(DateTime, default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="pages")
    characters = relationship("Character", back_populates="page", cascade="all, delete-orphan")
    words = relationship("Word", back_populates="page", cascade="all, delete-orphan")
    uncertain_regions = relationship("UncertainRegion", back_populates="page", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_page_document_number', 'document_id', 'page_number'),
    )
    
    def __repr__(self):
        return f"<Page(id={self.id}, document_id={self.document_id}, page_number={self.page_number})>"

class Character(Base):
    """
    Character model for storing character-level OCR data with positions
    """
    __tablename__ = "characters"
    
    id = Column(Integer, primary_key=True, index=True)
    page_id = Column(Integer, ForeignKey("pages.id"), nullable=False)
    
    # Character data
    character = Column(String(10), nullable=False)  # Support for multi-byte characters
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    
    # Hierarchical structure
    word_id = Column(Integer, ForeignKey("words.id"))
    line_id = Column(Integer)
    block_id = Column(Integer)
    
    # Font and style information
    font_size = Column(Float)
    is_bold = Column(Boolean, default=False)
    is_italic = Column(Boolean, default=False)
    
    # Analysis flags
    is_anomaly = Column(Boolean, default=False)
    anomaly_type = Column(String(50))
    anomaly_severity = Column(Float)
    
    # Relationships
    page = relationship("Page", back_populates="characters")
    word = relationship("Word", back_populates="characters")
    
    # Indexes for spatial and character queries
    __table_args__ = (
        Index('idx_character_page_char', 'page_id', 'character'),
        Index('idx_character_position', 'page_id', 'x', 'y'),
        Index('idx_character_confidence', 'page_id', 'confidence'),
    )
    
    def __repr__(self):
        return f"<Character(id={self.id}, char='{self.character}', x={self.x}, y={self.y})>"

class Word(Base):
    """
    Word model for storing word-level analysis data
    """
    __tablename__ = "words"
    
    id = Column(Integer, primary_key=True, index=True)
    page_id = Column(Integer, ForeignKey("pages.id"), nullable=False)
    
    # Word data
    text = Column(String(200), nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    
    # Hierarchical structure
    line_id = Column(Integer)
    block_id = Column(Integer)
    word_order = Column(Integer)  # Order within page
    
    # Word analysis
    length = Column(Integer)
    is_palindrome = Column(Boolean, default=False)
    is_archaic = Column(Boolean, default=False)
    is_anomaly = Column(Boolean, default=False)
    frequency_score = Column(Float)  # Frequency within document
    
    # Pattern analysis
    character_pattern = Column(String(100))  # e.g., "CVCVCV" for consonant/vowel pattern
    has_repeated_sequences = Column(Boolean, default=False)
    
    # Relationships
    page = relationship("Page", back_populates="words")
    characters = relationship("Character", back_populates="word")
    
    # Indexes for text analysis queries
    __table_args__ = (
        Index('idx_word_page_text', 'page_id', 'text'),
        Index('idx_word_confidence', 'page_id', 'confidence'),
        Index('idx_word_anomaly', 'page_id', 'is_anomaly'),
        Index('idx_word_length', 'page_id', 'length'),
    )
    
    def __repr__(self):
        return f"<Word(id={self.id}, text='{self.text}', confidence={self.confidence})>"

class UncertainRegion(Base):
    """
    Model for storing regions with low OCR confidence requiring manual review
    """
    __tablename__ = "uncertain_regions"
    
    id = Column(Integer, primary_key=True, index=True)
    page_id = Column(Integer, ForeignKey("pages.id"), nullable=False)
    
    # Region coordinates
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    
    # OCR data
    confidence = Column(Float, nullable=False)
    suggested_text = Column(Text)
    alternatives = Column(JSON)  # Store alternative suggestions as JSON array
    reason = Column(String(200))
    
    # Review status
    review_status = Column(String(50), default="pending")  # pending, reviewed, corrected
    corrected_text = Column(Text)
    reviewed_by = Column(String(100))
    reviewed_at = Column(DateTime)
    
    # Relationships
    page = relationship("Page", back_populates="uncertain_regions")
    
    # Indexes
    __table_args__ = (
        Index('idx_uncertain_page_confidence', 'page_id', 'confidence'),
        Index('idx_uncertain_review_status', 'review_status'),
    )
    
    def __repr__(self):
        return f"<UncertainRegion(id={self.id}, confidence={self.confidence}, status='{self.review_status}')>"

class Pattern(Base):
    """
    Model for storing detected patterns and anomalies
    """
    __tablename__ = "patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Pattern identification
    pattern_type = Column(String(100), nullable=False)  # e.g., "character_size_variation", "grid_pattern"
    pattern_name = Column(String(200))
    description = Column(Text, nullable=False)
    
    # Pattern location and scope
    page_numbers = Column(JSON)  # Pages where pattern appears
    # Optional scalar for common case of single-page patterns
    page_number = Column(Integer)
    coordinates = Column(JSON)  # Bounding boxes or coordinate sets
    
    # Pattern significance
    confidence = Column(Float, nullable=False)
    severity = Column(Float, nullable=False)
    significance_score = Column(Float)
    statistical_p_value = Column(Float)
    
    # Pattern data
    pattern_data = Column(JSON)  # Flexible storage for pattern-specific data
    context_data = Column(JSON)  # Additional context information
    
    # Analysis metadata
    detection_method = Column(String(100))
    detected_at = Column(DateTime, default=func.now())
    analysis_version = Column(String(50))
    
    # Review and validation
    is_validated = Column(Boolean, default=False)
    validation_notes = Column(Text)
    validated_by = Column(String(100))
    validated_at = Column(DateTime)
    
    # Relationships
    document = relationship("Document", back_populates="patterns")
    
    # Indexes for pattern queries
    __table_args__ = (
        Index('idx_pattern_document_type', 'document_id', 'pattern_type'),
        Index('idx_pattern_significance', 'document_id', 'significance_score'),
        Index('idx_pattern_confidence', 'document_id', 'confidence'),
    )
    
    def __repr__(self):
        return f"<Pattern(id={self.id}, type='{self.pattern_type}', confidence={self.confidence})>"

class Grid(Base):
    """
    Model for storing text grid configurations and analysis results
    """
    __tablename__ = "grids"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Grid configuration
    name = Column(String(200))
    rows = Column(Integer, nullable=False)
    columns = Column(Integer, nullable=False)
    include_spaces = Column(Boolean, default=True)
    include_punctuation = Column(Boolean, default=True)
    
    # Grid data
    grid_data = Column(JSON, nullable=False)  # 2D array of characters
    source_text = Column(Text)
    character_positions = Column(JSON)  # Mapping of grid positions to original coordinates
    
    # Grid analysis results
    patterns_found = Column(JSON)  # Array of detected patterns
    pattern_count = Column(Integer, default=0)
    analysis_completed = Column(Boolean, default=False)
    
    # Creation metadata
    created_at = Column(DateTime, default=func.now())
    created_by = Column(String(100))
    analysis_parameters = Column(JSON)  # Parameters used for pattern analysis
    
    # Relationships
    document = relationship("Document", back_populates="grids")
    grid_patterns = relationship("GridPattern", back_populates="grid", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_grid_document_size', 'document_id', 'rows', 'columns'),
        Index('idx_grid_pattern_count', 'document_id', 'pattern_count'),
    )
    
    def __repr__(self):
        return f"<Grid(id={self.id}, name='{self.name}', size={self.rows}x{self.columns})>"

class GridPattern(Base):
    """
    Model for storing patterns found within text grids
    """
    __tablename__ = "grid_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    grid_id = Column(Integer, ForeignKey("grids.id"), nullable=False)
    
    # Pattern identification
    pattern_text = Column(String(500), nullable=False)
    pattern_type = Column(String(100))  # horizontal, vertical, cross, diagonal
    
    # Pattern location within grid
    start_row = Column(Integer, nullable=False)
    start_column = Column(Integer, nullable=False)
    end_row = Column(Integer, nullable=False)
    end_column = Column(Integer, nullable=False)
    path_coordinates = Column(JSON)  # Full path of the pattern
    
    # Pattern analysis
    confidence = Column(Float, nullable=False)
    significance_score = Column(Float)
    context_before = Column(String(200))
    context_after = Column(String(200))
    
    # Pattern metadata
    length = Column(Integer)
    direction = Column(String(50))  # forward, backward, up, down, diagonal
    is_connected = Column(Boolean, default=True)
    connection_rules = Column(JSON)  # Rules used for pattern detection
    
    # Discovery metadata
    discovered_at = Column(DateTime, default=func.now())
    discovery_method = Column(String(100))
    
    # Relationships
    grid = relationship("Grid", back_populates="grid_patterns")
    
    # Indexes
    __table_args__ = (
        Index('idx_grid_pattern_grid_type', 'grid_id', 'pattern_type'),
        Index('idx_grid_pattern_significance', 'grid_id', 'significance_score'),
        Index('idx_grid_pattern_text', 'grid_id', 'pattern_text'),
    )
    
    def __repr__(self):
        return f"<GridPattern(id={self.id}, text='{self.pattern_text}', type='{self.pattern_type}')>"

class GeometricMeasurement(Base):
    """
    Model for storing geometric measurements and relationships
    """
    __tablename__ = "geometric_measurements"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    page_id = Column(Integer, ForeignKey("pages.id"))
    
    # Measurement type and data
    measurement_type = Column(String(100), nullable=False)  # angle, distance, ratio, area
    measurement_value = Column(Float, nullable=False)
    measurement_unit = Column(String(50))  # degrees, pixels, ratio
    
    # Geometric elements involved
    element_type = Column(String(100))  # character, word, punctuation, line
    element_ids = Column(JSON)  # IDs of elements involved in measurement
    coordinates = Column(JSON)  # Coordinate data for the measurement
    
    # Measurement context
    description = Column(Text)
    calculation_method = Column(String(100))
    precision = Column(Float)  # Measurement precision/error bounds
    
    # Analysis results
    is_significant = Column(Boolean, default=False)
    significance_score = Column(Float)
    pattern_relationship = Column(String(200))  # Relationship to known patterns
    
    # Metadata
    measured_at = Column(DateTime, default=func.now())
    measurement_tool = Column(String(100))
    
    # Indexes
    __table_args__ = (
        Index('idx_geometric_document_type', 'document_id', 'measurement_type'),
        Index('idx_geometric_significance', 'document_id', 'is_significant'),
        Index('idx_geometric_value', 'document_id', 'measurement_value'),
    )
    
    def __repr__(self):
        return f"<GeometricMeasurement(id={self.id}, type='{self.measurement_type}', value={self.measurement_value})>"

class EtymologyCache(Base):
    """
    Model for caching etymology and linguistic analysis results
    """
    __tablename__ = "etymology_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Word identification
    word = Column(String(200), nullable=False)
    language = Column(String(50), nullable=False)  # hebrew, greek, latin, english
    normalized_form = Column(String(200))  # Normalized/root form
    
    # Etymology data
    root_words = Column(JSON)  # Array of root word information
    morphological_analysis = Column(JSON)  # Morphological breakdown
    historical_forms = Column(JSON)  # Historical variations
    etymology_source = Column(String(100))  # Source of etymology data
    
    # Translation and meaning
    definitions = Column(JSON)  # Array of definitions
    translations = Column(JSON)  # Translations to other languages
    usage_examples = Column(JSON)  # Historical usage examples
    
    # Frequency and usage data
    frequency_score = Column(Float)
    usage_period = Column(String(100))  # Time period of usage
    geographic_usage = Column(JSON)  # Geographic distribution
    
    # Cache metadata
    cached_at = Column(DateTime, default=func.now())
    cache_source = Column(String(100))  # API or database source
    last_updated = Column(DateTime, default=func.now())
    cache_version = Column(String(50))
    
    # Validation
    is_verified = Column(Boolean, default=False)
    verification_source = Column(String(100))
    confidence_score = Column(Float)
    
    # Indexes for etymology queries
    __table_args__ = (
        Index('idx_etymology_word_lang', 'word', 'language'),
        Index('idx_etymology_normalized', 'normalized_form', 'language'),
        Index('idx_etymology_updated', 'last_updated'),
    )
    
    def __repr__(self):
        return f"<EtymologyCache(id={self.id}, word='{self.word}', language='{self.language}')>"