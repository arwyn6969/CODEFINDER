"""
Coordinate Extraction Utility for Geometric Analysis
Provides database queries to extract character positions for spatial indexing.
"""
from typing import List, Tuple, Optional
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.core.database import get_db
from app.models.database_models import Character, Page, Document


@dataclass
class CharacterPosition:
    """Character position data for geometric analysis."""
    id: int
    character: str
    x: float
    y: float
    width: float
    height: float
    confidence: float
    is_uppercase: bool
    is_punctuation: bool


def get_character_positions_for_page(
    db: Session,
    document_id: int,
    page_number: int,
    min_confidence: float = 0.5,
    filter_significant: bool = False
) -> List[CharacterPosition]:
    """
    Extract character positions from database for a specific page.
    
    Args:
        db: Database session
        document_id: Document ID
        page_number: Page number (1-indexed)
        min_confidence: Minimum OCR confidence threshold
        filter_significant: If True, only return uppercase + punctuation
    
    Returns:
        List of CharacterPosition objects suitable for GeometricIndex
    """
    # Get page ID
    page = db.query(Page).filter(
        and_(
            Page.document_id == document_id,
            Page.page_number == page_number
        )
    ).first()
    
    if not page:
        return []
    
    # Query characters
    query = db.query(Character).filter(
        and_(
            Character.page_id == page.id,
            Character.confidence >= min_confidence
        )
    ).order_by(Character.y, Character.x)
    
    characters = query.all()
    
    positions = []
    for char in characters:
        is_upper = char.character.isupper() if char.character else False
        is_punct = not char.character.isalnum() if char.character else False
        
        # Skip if filtering and not significant
        if filter_significant and not (is_upper or is_punct):
            continue
        
        positions.append(CharacterPosition(
            id=char.id,
            character=char.character,
            x=char.x,
            y=char.y,
            width=char.width,
            height=char.height,
            confidence=char.confidence,
            is_uppercase=is_upper,
            is_punctuation=is_punct
        ))
    
    return positions


def positions_to_tuples(positions: List[CharacterPosition]) -> List[Tuple[float, float]]:
    """
    Convert CharacterPositions to (x, y) tuples for GeometricIndex.
    Uses character center point.
    """
    return [(p.x + p.width / 2, p.y + p.height / 2) for p in positions]


def get_page_dimensions(db: Session, document_id: int, page_number: int) -> Tuple[float, float]:
    """Get page width and height from database."""
    page = db.query(Page).filter(
        and_(
            Page.document_id == document_id,
            Page.page_number == page_number
        )
    ).first()
    
    if page and page.width and page.height:
        return (page.width, page.height)
    
    # Default to standard page size if not stored
    return (612.0, 792.0)  # Letter size at 72 DPI
