"""
Data Access Layer for Ancient Text Analyzer
Provides efficient database operations and complex queries
"""
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy import and_, or_, func, text, desc, asc
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta, timezone

from app.models.database_models import (
    Document, Page, Character, Word, UncertainRegion, Pattern, 
    Grid, GridPattern, GeometricMeasurement, EtymologyCache
)
from app.core.database import get_db

logger = logging.getLogger(__name__)

class DocumentRepository:
    """Repository for document-related database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_document(self, filename: str, file_path: str, file_size: int, **kwargs) -> Document:
        """Create a new document record"""
        document = Document(
            filename=filename,
            original_filename=kwargs.get('original_filename', filename),
            file_path=file_path,
            file_size=file_size,
            mime_type=kwargs.get('mime_type', 'application/pdf'),
            page_count=kwargs.get('page_count'),
            creation_date=kwargs.get('creation_date'),
            processing_status='uploaded'
        )
        
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        
        logger.info(f"Created document: {document.id} - {filename}")
        return document
    
    def get_document(self, document_id: int) -> Optional[Document]:
        """Get document by ID with related data"""
        return self.db.query(Document).options(
            selectinload(Document.pages),
            selectinload(Document.patterns),
            selectinload(Document.grids)
        ).filter(Document.id == document_id).first()
    
    def get_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        """Get list of documents with pagination"""
        return self.db.query(Document).order_by(desc(Document.upload_date)).offset(offset).limit(limit).all()
    
    def update_processing_status(self, document_id: int, status: str, error_message: str = None):
        """Update document processing status"""
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.processing_status = status
            if status == 'processing':
                document.processing_started_at = datetime.now(timezone.utc)
            elif status in ['completed', 'failed']:
                document.processing_completed_at = datetime.now(timezone.utc)
            if error_message:
                document.error_message = error_message
            
            self.db.commit()
            logger.info(f"Updated document {document_id} status to {status}")
    
    def update_analysis_summary(self, document_id: int, total_characters: int, 
                              total_words: int, average_confidence: float):
        """Update document analysis summary"""
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.total_characters = total_characters
            document.total_words = total_words
            document.average_confidence = average_confidence
            self.db.commit()
    
    def delete_document(self, document_id: int) -> bool:
        """Delete document and all related data"""
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if document:
            self.db.delete(document)
            self.db.commit()
            logger.info(f"Deleted document: {document_id}")
            return True
        return False

class PageRepository:
    """Repository for page-related database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_page(self, document_id: int, page_number: int, **kwargs) -> Page:
        """Create a new page record"""
        page = Page(
            document_id=document_id,
            page_number=page_number,
            width=kwargs.get('width'),
            height=kwargs.get('height'),
            dpi=kwargs.get('dpi', 300),
            rotation=kwargs.get('rotation', 0),
            extracted_text=kwargs.get('extracted_text'),
            ocr_confidence=kwargs.get('ocr_confidence'),
            processing_time=kwargs.get('processing_time'),
            image_quality_score=kwargs.get('image_quality_score'),
            preprocessing_steps=kwargs.get('preprocessing_steps'),
            ocr_engine_version=kwargs.get('ocr_engine_version')
        )
        
        self.db.add(page)
        self.db.commit()
        self.db.refresh(page)
        
        logger.info(f"Created page: {page.id} - Document {document_id}, Page {page_number}")
        return page
    
    def get_page(self, page_id: int) -> Optional[Page]:
        """Get page by ID with related data"""
        return self.db.query(Page).options(
            selectinload(Page.characters),
            selectinload(Page.words),
            selectinload(Page.uncertain_regions)
        ).filter(Page.id == page_id).first()
    
    def get_pages_by_document(self, document_id: int) -> List[Page]:
        """Get all pages for a document"""
        return self.db.query(Page).filter(Page.document_id == document_id).order_by(Page.page_number).all()
    
    def get_page_by_number(self, document_id: int, page_number: int) -> Optional[Page]:
        """Get specific page by document and page number"""
        return self.db.query(Page).filter(
            and_(Page.document_id == document_id, Page.page_number == page_number)
        ).first()

class CharacterRepository:
    """Repository for character-level database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def bulk_create_characters(self, characters_data: List[Dict[str, Any]]) -> List[Character]:
        """Bulk create character records for efficiency"""
        characters = [Character(**char_data) for char_data in characters_data]
        
        self.db.bulk_save_objects(characters, return_defaults=True)
        self.db.commit()
        
        logger.info(f"Bulk created {len(characters)} characters")
        return characters
    
    def get_characters_by_page(self, page_id: int) -> List[Character]:
        """Get all characters for a page"""
        return self.db.query(Character).filter(Character.page_id == page_id).order_by(Character.x, Character.y).all()
    
    def get_characters_by_type(self, page_id: int, character: str) -> List[Character]:
        """Get all instances of a specific character on a page"""
        return self.db.query(Character).filter(
            and_(Character.page_id == page_id, Character.character == character)
        ).all()
    
    def get_characters_in_region(self, page_id: int, x: float, y: float, 
                                width: float, height: float) -> List[Character]:
        """Get characters within a specific region"""
        return self.db.query(Character).filter(
            and_(
                Character.page_id == page_id,
                Character.x >= x,
                Character.x <= x + width,
                Character.y >= y,
                Character.y <= y + height
            )
        ).all()
    
    def get_low_confidence_characters(self, page_id: int, threshold: float = 0.8) -> List[Character]:
        """Get characters with confidence below threshold"""
        return self.db.query(Character).filter(
            and_(Character.page_id == page_id, Character.confidence < threshold)
        ).all()
    
    def get_character_statistics(self, page_id: int) -> Dict[str, Any]:
        """Get character statistics for a page"""
        stats = self.db.query(
            func.count(Character.id).label('total_characters'),
            func.count(func.distinct(Character.character)).label('unique_characters'),
            func.avg(Character.confidence).label('average_confidence'),
            func.min(Character.confidence).label('min_confidence'),
            func.max(Character.confidence).label('max_confidence')
        ).filter(Character.page_id == page_id).first()
        
        return {
            'total_characters': stats.total_characters,
            'unique_characters': stats.unique_characters,
            'average_confidence': float(stats.average_confidence) if stats.average_confidence else 0.0,
            'min_confidence': float(stats.min_confidence) if stats.min_confidence else 0.0,
            'max_confidence': float(stats.max_confidence) if stats.max_confidence else 0.0
        }
    
    def get_character_frequency(self, page_id: int) -> List[Tuple[str, int]]:
        """Get character frequency distribution for a page"""
        return self.db.query(
            Character.character,
            func.count(Character.id).label('frequency')
        ).filter(Character.page_id == page_id).group_by(Character.character).order_by(desc('frequency')).all()

class WordRepository:
    """Repository for word-level database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def bulk_create_words(self, words_data: List[Dict[str, Any]]) -> List[Word]:
        """Bulk create word records"""
        words = [Word(**word_data) for word_data in words_data]
        
        self.db.bulk_save_objects(words, return_defaults=True)
        self.db.commit()
        
        logger.info(f"Bulk created {len(words)} words")
        return words
    
    def get_words_by_page(self, page_id: int) -> List[Word]:
        """Get all words for a page"""
        return self.db.query(Word).filter(Word.page_id == page_id).order_by(Word.word_order).all()
    
    def search_words(self, document_id: int, search_term: str, case_sensitive: bool = False) -> List[Word]:
        """Search for words containing search term"""
        query = self.db.query(Word).join(Page).filter(Page.document_id == document_id)
        
        if case_sensitive:
            query = query.filter(Word.text.contains(search_term))
        else:
            query = query.filter(Word.text.ilike(f'%{search_term}%'))
        
        return query.all()
    
    def get_word_frequency(self, document_id: int, limit: int = 100) -> List[Tuple[str, int]]:
        """Get word frequency distribution for a document"""
        return self.db.query(
            Word.text,
            func.count(Word.id).label('frequency')
        ).join(Page).filter(Page.document_id == document_id).group_by(Word.text).order_by(desc('frequency')).limit(limit).all()
    
    def get_anomalous_words(self, document_id: int) -> List[Word]:
        """Get words marked as anomalous"""
        return self.db.query(Word).join(Page).filter(
            and_(Page.document_id == document_id, Word.is_anomaly == True)
        ).all()
    
    def get_palindromes(self, document_id: int) -> List[Word]:
        """Get palindromic words"""
        return self.db.query(Word).join(Page).filter(
            and_(Page.document_id == document_id, Word.is_palindrome == True)
        ).all()
    
    def get_archaic_words(self, document_id: int) -> List[Word]:
        """Get words marked as archaic"""
        return self.db.query(Word).join(Page).filter(
            and_(Page.document_id == document_id, Word.is_archaic == True)
        ).all()

class PatternRepository:
    """Repository for pattern-related database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_pattern(self, document_id: int, pattern_type: str, description: str, 
                      confidence: float, severity: float, **kwargs) -> Pattern:
        """Create a new pattern record"""
        pattern = Pattern(
            document_id=document_id,
            pattern_type=pattern_type,
            pattern_name=kwargs.get('pattern_name'),
            description=description,
            page_numbers=kwargs.get('page_numbers'),
            coordinates=kwargs.get('coordinates'),
            confidence=confidence,
            severity=severity,
            significance_score=kwargs.get('significance_score'),
            statistical_p_value=kwargs.get('statistical_p_value'),
            pattern_data=kwargs.get('pattern_data'),
            context_data=kwargs.get('context_data'),
            detection_method=kwargs.get('detection_method'),
            analysis_version=kwargs.get('analysis_version')
        )
        
        self.db.add(pattern)
        self.db.commit()
        self.db.refresh(pattern)
        
        logger.info(f"Created pattern: {pattern.id} - {pattern_type}")
        return pattern
    
    def get_patterns_by_document(self, document_id: int) -> List[Pattern]:
        """Get all patterns for a document"""
        return self.db.query(Pattern).filter(Pattern.document_id == document_id).order_by(desc(Pattern.significance_score)).all()
    
    def get_patterns_by_type(self, document_id: int, pattern_type: str) -> List[Pattern]:
        """Get patterns of a specific type"""
        return self.db.query(Pattern).filter(
            and_(Pattern.document_id == document_id, Pattern.pattern_type == pattern_type)
        ).all()
    
    def get_significant_patterns(self, document_id: int, min_significance: float = 0.7) -> List[Pattern]:
        """Get patterns above significance threshold"""
        return self.db.query(Pattern).filter(
            and_(Pattern.document_id == document_id, Pattern.significance_score >= min_significance)
        ).order_by(desc(Pattern.significance_score)).all()
    
    def get_validated_patterns(self, document_id: int) -> List[Pattern]:
        """Get patterns that have been validated"""
        return self.db.query(Pattern).filter(
            and_(Pattern.document_id == document_id, Pattern.is_validated == True)
        ).all()

class GridRepository:
    """Repository for grid-related database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_grid(self, document_id: int, name: str, rows: int, columns: int, 
                   grid_data: List[List[str]], **kwargs) -> Grid:
        """Create a new grid record"""
        grid = Grid(
            document_id=document_id,
            name=name,
            rows=rows,
            columns=columns,
            include_spaces=kwargs.get('include_spaces', True),
            include_punctuation=kwargs.get('include_punctuation', True),
            grid_data=grid_data,
            source_text=kwargs.get('source_text'),
            character_positions=kwargs.get('character_positions'),
            created_by=kwargs.get('created_by'),
            analysis_parameters=kwargs.get('analysis_parameters')
        )
        
        self.db.add(grid)
        self.db.commit()
        self.db.refresh(grid)
        
        logger.info(f"Created grid: {grid.id} - {name} ({rows}x{columns})")
        return grid
    
    def get_grid(self, grid_id: int) -> Optional[Grid]:
        """Get grid by ID with patterns"""
        return self.db.query(Grid).options(
            selectinload(Grid.grid_patterns)
        ).filter(Grid.id == grid_id).first()
    
    def get_grids_by_document(self, document_id: int) -> List[Grid]:
        """Get all grids for a document"""
        return self.db.query(Grid).filter(Grid.document_id == document_id).order_by(desc(Grid.created_at)).all()
    
    def create_grid_pattern(self, grid_id: int, pattern_text: str, pattern_type: str,
                           start_row: int, start_column: int, end_row: int, end_column: int,
                           confidence: float, **kwargs) -> GridPattern:
        """Create a new grid pattern record"""
        pattern = GridPattern(
            grid_id=grid_id,
            pattern_text=pattern_text,
            pattern_type=pattern_type,
            start_row=start_row,
            start_column=start_column,
            end_row=end_row,
            end_column=end_column,
            path_coordinates=kwargs.get('path_coordinates'),
            confidence=confidence,
            significance_score=kwargs.get('significance_score'),
            context_before=kwargs.get('context_before'),
            context_after=kwargs.get('context_after'),
            length=len(pattern_text),
            direction=kwargs.get('direction'),
            is_connected=kwargs.get('is_connected', True),
            connection_rules=kwargs.get('connection_rules'),
            discovery_method=kwargs.get('discovery_method')
        )
        
        self.db.add(pattern)
        self.db.commit()
        self.db.refresh(pattern)
        
        return pattern
    
    def get_grid_patterns(self, grid_id: int) -> List[GridPattern]:
        """Get all patterns for a grid"""
        return self.db.query(GridPattern).filter(GridPattern.grid_id == grid_id).order_by(desc(GridPattern.significance_score)).all()
    
    def search_grid_patterns(self, grid_id: int, search_text: str) -> List[GridPattern]:
        """Search for patterns containing specific text"""
        return self.db.query(GridPattern).filter(
            and_(GridPattern.grid_id == grid_id, GridPattern.pattern_text.ilike(f'%{search_text}%'))
        ).all()

class EtymologyRepository:
    """Repository for etymology cache operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_etymology(self, word: str, language: str) -> Optional[EtymologyCache]:
        """Get cached etymology data"""
        return self.db.query(EtymologyCache).filter(
            and_(EtymologyCache.word == word.lower(), EtymologyCache.language == language)
        ).first()
    
    def cache_etymology(self, word: str, language: str, etymology_data: Dict[str, Any]) -> EtymologyCache:
        """Cache etymology data"""
        # Check if already exists
        existing = self.get_etymology(word, language)
        if existing:
            # Update existing record
            for key, value in etymology_data.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            existing.last_updated = datetime.now(timezone.utc)
            self.db.commit()
            return existing
        
        # Create new record
        etymology = EtymologyCache(
            word=word.lower(),
            language=language,
            **etymology_data
        )
        
        self.db.add(etymology)
        self.db.commit()
        self.db.refresh(etymology)
        
        return etymology
    
    def search_etymology(self, search_term: str, language: str = None) -> List[EtymologyCache]:
        """Search etymology cache"""
        query = self.db.query(EtymologyCache).filter(
            or_(
                EtymologyCache.word.ilike(f'%{search_term}%'),
                EtymologyCache.normalized_form.ilike(f'%{search_term}%')
            )
        )
        
        if language:
            query = query.filter(EtymologyCache.language == language)
        
        return query.limit(50).all()

class AnalysisRepository:
    """Repository for complex analysis queries across multiple tables"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_document_analysis_summary(self, document_id: int) -> Dict[str, Any]:
        """Get comprehensive analysis summary for a document"""
        # Get basic document info
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return {}
        
        # Get page count and processing status
        page_count = self.db.query(func.count(Page.id)).filter(Page.document_id == document_id).scalar()
        
        # Get character statistics
        char_stats = self.db.query(
            func.count(Character.id).label('total_characters'),
            func.count(func.distinct(Character.character)).label('unique_characters'),
            func.avg(Character.confidence).label('avg_confidence')
        ).join(Page).filter(Page.document_id == document_id).first()
        
        # Get word statistics
        word_stats = self.db.query(
            func.count(Word.id).label('total_words'),
            func.count(func.distinct(Word.text)).label('unique_words'),
            func.sum(func.case([(Word.is_anomaly == True, 1)], else_=0)).label('anomalous_words')
        ).join(Page).filter(Page.document_id == document_id).first()
        
        # Get pattern count
        pattern_count = self.db.query(func.count(Pattern.id)).filter(Pattern.document_id == document_id).scalar()
        
        # Get grid count
        grid_count = self.db.query(func.count(Grid.id)).filter(Grid.document_id == document_id).scalar()
        
        return {
            'document_id': document_id,
            'filename': document.filename,
            'processing_status': document.processing_status,
            'page_count': page_count,
            'total_characters': char_stats.total_characters or 0,
            'unique_characters': char_stats.unique_characters or 0,
            'average_confidence': float(char_stats.avg_confidence) if char_stats.avg_confidence else 0.0,
            'total_words': word_stats.total_words or 0,
            'unique_words': word_stats.unique_words or 0,
            'anomalous_words': word_stats.anomalous_words or 0,
            'pattern_count': pattern_count or 0,
            'grid_count': grid_count or 0
        }
    
    def get_cross_page_patterns(self, document_id: int) -> List[Dict[str, Any]]:
        """Find patterns that span multiple pages"""
        # This is a complex query that would need to be implemented based on specific pattern types
        # For now, return patterns that appear on multiple pages
        patterns = self.db.query(Pattern).filter(
            and_(
                Pattern.document_id == document_id,
                func.json_array_length(Pattern.page_numbers) > 1
            )
        ).all()
        
        return [
            {
                'pattern_id': p.id,
                'pattern_type': p.pattern_type,
                'description': p.description,
                'page_count': len(p.page_numbers) if p.page_numbers else 0,
                'confidence': p.confidence,
                'significance_score': p.significance_score
            }
            for p in patterns
        ]


class GeometricRepository:
    """Repository for geometric measurement database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_measurement(self, document_id: int, measurement_type: str, 
                          measurement_value: float, **kwargs) -> GeometricMeasurement:
        """Create a new geometric measurement record"""
        measurement = GeometricMeasurement(
            document_id=document_id,
            page_id=kwargs.get('page_id'),
            measurement_type=measurement_type,
            measurement_value=measurement_value,
            measurement_unit=kwargs.get('measurement_unit', 'pixels'),
            element_type=kwargs.get('element_type'),
            element_ids=kwargs.get('element_ids'),
            coordinates=kwargs.get('coordinates'),
            description=kwargs.get('description'),
            calculation_method=kwargs.get('calculation_method'),
            precision=kwargs.get('precision'),
            is_significant=kwargs.get('is_significant', False),
            significance_score=kwargs.get('significance_score'),
            pattern_relationship=kwargs.get('pattern_relationship'),
            measurement_tool=kwargs.get('measurement_tool', 'geometric_analyzer')
        )
        
        self.db.add(measurement)
        self.db.commit()
        self.db.refresh(measurement)
        
        logger.info(f"Created geometric measurement: {measurement.id} - {measurement_type}")
        return measurement
    
    def get_measurements_by_document(self, document_id: int) -> List[GeometricMeasurement]:
        """Get all geometric measurements for a document"""
        return self.db.query(GeometricMeasurement).filter(
            GeometricMeasurement.document_id == document_id
        ).order_by(desc(GeometricMeasurement.significance_score)).all()
    
    def get_measurements_by_type(self, document_id: int, measurement_type: str) -> List[GeometricMeasurement]:
        """Get measurements of a specific type"""
        return self.db.query(GeometricMeasurement).filter(
            and_(
                GeometricMeasurement.document_id == document_id,
                GeometricMeasurement.measurement_type == measurement_type
            )
        ).all()
    
    def get_significant_measurements(self, document_id: int, min_significance: float = 0.7) -> List[GeometricMeasurement]:
        """Get measurements above significance threshold"""
        return self.db.query(GeometricMeasurement).filter(
            and_(
                GeometricMeasurement.document_id == document_id,
                GeometricMeasurement.significance_score >= min_significance
            )
        ).order_by(desc(GeometricMeasurement.significance_score)).all()
    
    def get_measurements_by_value_range(self, document_id: int, measurement_type: str,
                                      min_value: float, max_value: float) -> List[GeometricMeasurement]:
        """Get measurements within a value range"""
        return self.db.query(GeometricMeasurement).filter(
            and_(
                GeometricMeasurement.document_id == document_id,
                GeometricMeasurement.measurement_type == measurement_type,
                GeometricMeasurement.measurement_value >= min_value,
                GeometricMeasurement.measurement_value <= max_value
            )
        ).all()
    
    def get_angle_measurements(self, document_id: int) -> List[GeometricMeasurement]:
        """Get all angle measurements for a document"""
        return self.get_measurements_by_type(document_id, "angle")
    
    def get_distance_measurements(self, document_id: int) -> List[GeometricMeasurement]:
        """Get all distance measurements for a document"""
        return self.get_measurements_by_type(document_id, "distance")
    
    def get_ratio_measurements(self, document_id: int) -> List[GeometricMeasurement]:
        """Get all ratio measurements for a document"""
        return self.get_measurements_by_type(document_id, "ratio")
    
    def find_measurements_near_value(self, document_id: int, measurement_type: str,
                                   target_value: float, tolerance: float = 1.0) -> List[GeometricMeasurement]:
        """Find measurements near a target value"""
        return self.get_measurements_by_value_range(
            document_id, measurement_type, 
            target_value - tolerance, target_value + tolerance
        )
    
    def get_golden_ratio_measurements(self, document_id: int, tolerance: float = 0.1) -> List[GeometricMeasurement]:
        """Find measurements close to the golden ratio"""
        golden_ratio = 1.618033988749
        return self.find_measurements_near_value(document_id, "ratio", golden_ratio, tolerance)
    
    def get_right_angle_measurements(self, document_id: int, tolerance: float = 2.0) -> List[GeometricMeasurement]:
        """Find angle measurements close to 90 degrees"""
        return self.find_measurements_near_value(document_id, "angle", 90.0, tolerance)
    
    def get_pi_related_measurements(self, document_id: int, tolerance: float = 2.0) -> List[GeometricMeasurement]:
        """Find angle measurements related to pi fractions"""
        import math
        
        pi_angles = [30, 45, 60, 90, 120, 135, 150, 180]  # Common pi fraction angles
        pi_measurements = []
        
        for angle in pi_angles:
            measurements = self.find_measurements_near_value(document_id, "angle", angle, tolerance)
            pi_measurements.extend(measurements)
        
        return pi_measurements
    
    def get_measurement_statistics(self, document_id: int) -> Dict[str, Any]:
        """Get statistics for geometric measurements"""
        stats = self.db.query(
            GeometricMeasurement.measurement_type,
            func.count(GeometricMeasurement.id).label('count'),
            func.avg(GeometricMeasurement.measurement_value).label('avg_value'),
            func.min(GeometricMeasurement.measurement_value).label('min_value'),
            func.max(GeometricMeasurement.measurement_value).label('max_value'),
            func.avg(GeometricMeasurement.significance_score).label('avg_significance')
        ).filter(GeometricMeasurement.document_id == document_id).group_by(
            GeometricMeasurement.measurement_type
        ).all()
        
        statistics = {}
        for stat in stats:
            statistics[stat.measurement_type] = {
                'count': stat.count,
                'average_value': float(stat.avg_value) if stat.avg_value else 0.0,
                'min_value': float(stat.min_value) if stat.min_value else 0.0,
                'max_value': float(stat.max_value) if stat.max_value else 0.0,
                'average_significance': float(stat.avg_significance) if stat.avg_significance else 0.0
            }
        
        # Overall statistics
        total_measurements = self.db.query(func.count(GeometricMeasurement.id)).filter(
            GeometricMeasurement.document_id == document_id
        ).scalar()
        
        significant_measurements = self.db.query(func.count(GeometricMeasurement.id)).filter(
            and_(
                GeometricMeasurement.document_id == document_id,
                GeometricMeasurement.is_significant == True
            )
        ).scalar()
        
        statistics['overall'] = {
            'total_measurements': total_measurements or 0,
            'significant_measurements': significant_measurements or 0,
            'significance_ratio': (significant_measurements / total_measurements) if total_measurements > 0 else 0.0,
            'measurement_types': len(statistics)
        }
        
        return statistics
    
    def bulk_create_measurements(self, measurements_data: List[Dict[str, Any]]) -> List[GeometricMeasurement]:
        """Bulk create geometric measurements for efficiency"""
        measurements = [GeometricMeasurement(**data) for data in measurements_data]
        
        self.db.bulk_save_objects(measurements, return_defaults=True)
        self.db.commit()
        
        logger.info(f"Bulk created {len(measurements)} geometric measurements")
        return measurements
    
    def update_significance(self, measurement_id: int, is_significant: bool, 
                          significance_score: float, pattern_relationship: str = None):
        """Update significance data for a measurement"""
        measurement = self.db.query(GeometricMeasurement).filter(
            GeometricMeasurement.id == measurement_id
        ).first()
        
        if measurement:
            measurement.is_significant = is_significant
            measurement.significance_score = significance_score
            if pattern_relationship:
                measurement.pattern_relationship = pattern_relationship
            
            self.db.commit()
            logger.info(f"Updated significance for measurement {measurement_id}")
    
    def delete_measurements_by_document(self, document_id: int) -> int:
        """Delete all geometric measurements for a document"""
        deleted_count = self.db.query(GeometricMeasurement).filter(
            GeometricMeasurement.document_id == document_id
        ).delete()
        
        self.db.commit()
        logger.info(f"Deleted {deleted_count} geometric measurements for document {document_id}")
        return deleted_count
    
    def search_measurements_by_pattern(self, document_id: int, pattern_type: str) -> List[GeometricMeasurement]:
        """Search measurements by pattern relationship"""
        return self.db.query(GeometricMeasurement).filter(
            and_(
                GeometricMeasurement.document_id == document_id,
                GeometricMeasurement.pattern_relationship.ilike(f'%{pattern_type}%')
            )
        ).all()
    
    def get_measurements_by_coordinates(self, document_id: int, x: float, y: float, 
                                     radius: float = 50.0) -> List[GeometricMeasurement]:
        """Get measurements near specific coordinates"""
        # This is a simplified implementation - in practice, you'd use spatial queries
        measurements = self.db.query(GeometricMeasurement).filter(
            GeometricMeasurement.document_id == document_id
        ).all()
        
        nearby_measurements = []
        for measurement in measurements:
            if measurement.coordinates:
                # Simple distance check (would be more sophisticated in practice)
                coords = measurement.coordinates
                if isinstance(coords, list) and len(coords) >= 2:
                    mx, my = coords[0], coords[1]
                    distance = ((x - mx) ** 2 + (y - my) ** 2) ** 0.5
                    if distance <= radius:
                        nearby_measurements.append(measurement)
        
        return nearby_measurements