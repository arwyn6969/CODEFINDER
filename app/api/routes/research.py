from fastapi import APIRouter, Depends, HTTPException, Body
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel
from pathlib import Path

from app.services.gematria_engine import GematriaEngine
from app.services.els_analyzer import ELSAnalyzer
from app.services.els_visualizer import ELSVisualizer
from app.api.dependencies import get_current_active_user, User
# from app.models.user import User  <-- Removed incorrect import
from sqlalchemy.orm import Session, joinedload
from app.core.database import get_db
from app.models.database_models import Pattern, Document

router = APIRouter()


# Initialize engines once (stateless)
gematria_engine = GematriaEngine()
els_analyzer = ELSAnalyzer()
from app.services.transliteration_service import TransliterationService
transliteration_service = TransliterationService()
from app.services.prophetic_analyzer import PropheticAnalyzerService



# ... (Previous imports)

# Cache for Torah text
TORAH_TEXT_CACHE = None
TORAH_PATH = Path("app/data/torah.txt")

# Interesting Gematria values (Baconian/Rosicrucian)
INTERESTING_NUMBERS = {
    33: "Bacon (Simple)",
    67: "Francis (Simple)",
    100: "Francis Bacon (Simple)",
    157: "Fra Rosicrosse (Simple)",
    287: "Fra Rosicrosse (Kay)",
    888: "Jesus (Greek)"
}

class GematriaRequest(BaseModel):
    text: str
    document_id: Optional[int] = None
    save: bool = False

class ELSRequest(BaseModel):
    text: Optional[str] = None
    source: str = "custom"  # "custom", "torah", or "document"
    document_id: Optional[int] = None
    terms: Optional[List[str]] = None
    min_skip: int = 2
    max_skip: int = 150
    auto_transliterate: bool = False
    save: bool = False

class TransliterateRequest(BaseModel):
    text: str

@router.post("/gematria", response_model=Dict[str, Any])
async def calculate_gematria(
    request: GematriaRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Calculate Gematria values. Optionally save significant findings to a document.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        results = gematria_engine.calculate_all(request.text)
        
        # Persistence Logic
        if request.save and request.document_id:
            doc = db.query(Document).get(request.document_id)
            if not doc:
                # Log warning but don't fail? Or fail? 
                # Let's proceed returning results but add a warning in logs (not possible here easily)
                pass 
            else:
                saved_count = 0
                for cipher_name, val_data in results.items():
                    score = val_data.get('score')
                    if score in INTERESTING_NUMBERS:
                        desc = INTERESTING_NUMBERS[score]
                        pattern = Pattern(
                            document_id=request.document_id,
                            pattern_type="gematria_match",
                            pattern_name=f"Gematria: {score} ({cipher_name})",
                            description=f"Significant Gematria value found in text selection: {score} ({desc})",
                            confidence=1.0,
                            severity=0.8,
                            significance_score=0.8,
                            pattern_data={
                                "cipher": cipher_name, 
                                "score": score, 
                                "text": request.text[:50],
                                "meaning": desc
                            }
                        )
                        db.add(pattern)
                        saved_count += 1
                
                if saved_count > 0:
                    db.commit()
                    results['persisted_patterns'] = saved_count
                    
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/transliterate", response_model=List[Dict[str, str]])
async def transliterate_term(
    request: TransliterateRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get Hebrew candidates for an English term.
    """
    try:
        candidates = transliteration_service.get_hebrew_candidates(request.text)
        # If no dict hits, try auto
        if not candidates:
            auto_hits = transliteration_service.auto_transliterate(request.text)
            candidates = [(h, "Auto-generated") for h in auto_hits]
            
        return [{"hebrew": c[0], "description": c[1]} for c in candidates]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_torah_text() -> str:
    global TORAH_TEXT_CACHE
    if TORAH_TEXT_CACHE:
        return TORAH_TEXT_CACHE
    
    if not TORAH_PATH.exists():
        raise FileNotFoundError(f"Torah text not found at {TORAH_PATH}")
        
    with open(TORAH_PATH, "r", encoding="utf-8") as f:
        TORAH_TEXT_CACHE = f.read().strip()
    return TORAH_TEXT_CACHE

@router.post("/els", response_model=Dict[str, Any])
async def find_els(
    request: ELSRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Search for ELS. Optionally persist matches to a document.
    """
    try:
        search_text = ""
        
        if request.source == "torah":
            try:
                search_text = get_torah_text()
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="Torah text file not found on server.")
        elif request.source == "document":
            if not request.document_id:
                raise HTTPException(status_code=400, detail="document_id is required for document source")
            doc = db.query(Document).options(joinedload(Document.pages)).get(request.document_id)
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document {request.document_id} not found")
            search_text = "".join([p.extracted_text or "" for p in doc.pages])
            if not search_text.strip():
                raise HTTPException(status_code=400, detail="Document has no extracted text")
        else:
            if not request.text:
                raise HTTPException(status_code=400, detail="Text is required for custom source")
            search_text = request.text
            
        analyzer = els_analyzer
        if request.terms:
            analyzer = ELSAnalyzer(terms=request.terms)
            
        results = analyzer.analyze_text(
            search_text, 
            min_skip=request.min_skip, 
            max_skip=request.max_skip,
            auto_transliterate=request.auto_transliterate
        )
        
        # Persistence Logic
        if request.save and request.document_id and results.get('matches'):
            doc = db.query(Document).get(request.document_id)
            if doc:
                for match in results['matches']:
                    pattern = Pattern(
                        document_id=request.document_id,
                        pattern_type="els_match",
                        pattern_name=f"ELS: {match['term']}",
                        description=f"ELS Found: {match['term']} at skip {match['skip']} ({match['direction']})",
                        confidence=1.0,
                        severity=0.5,
                        significance_score=0.7,
                        pattern_data=match
                    )
                    db.add(pattern)
                db.commit()
                results['persisted_patterns'] = len(results['matches'])
                
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ELSVisRequest(BaseModel):
    text: Optional[str] = None
    source: str = "custom"  # "custom", "torah", or "document"
    document_id: Optional[int] = None
    center_index: int
    skip: int
    rows: int = 20
    cols: int = 20
    term_length: int = 5  # Added for highlighting

@router.post("/els/visualize", response_model=Dict[str, Any])
async def visualize_els(
    request: ELSVisRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Generate a 2D character grid to visualize an ELS match.
    """
    try:
        search_text = ""
        if request.source == "torah":
            try:
                search_text = get_torah_text()
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="Torah text file not found")
        elif request.source == "document":
            if not request.document_id:
                raise HTTPException(status_code=400, detail="document_id is required for document source")
            doc = db.query(Document).options(joinedload(Document.pages)).get(request.document_id)
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document {request.document_id} not found")
            search_text = "".join([p.extracted_text or "" for p in doc.pages])
            if not search_text.strip():
                raise HTTPException(status_code=400, detail="Document has no extracted text")
        else:
            if not request.text:
                raise HTTPException(status_code=400, detail="Text is required for custom source")
            search_text = request.text
            
        # Generate Grid (centered on the requested index)
        grid_data = ELSVisualizer.generate_grid(
            text=search_text,
            center_index=request.center_index,
            skip=request.skip,
            rows=request.rows,
            cols=request.cols
        )
        
        # Calculate Highlights
        # We assume the user wants to see the term starting at 'center_index' (which is passed as start_index from frontend)
        term_start_index = request.center_index 
        
        highlights = ELSVisualizer.get_term_path(
            start_index=term_start_index,
            skip=request.skip,
            length=request.term_length,
            grid_config=grid_data
        )
        
        grid_data['highlights'] = highlights
        return grid_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualizer Error: {str(e)}")




class CipherSolveRequest(BaseModel):
    text: str
    method: str  # 'substitution', 'caesar', 'atbash', 'reverse'
    key: Optional[Union[str, int, Dict[str, str]]] = None

@router.post("/cipher/solve", response_model=Dict[str, Any])
async def solve_cipher(
    request: CipherSolveRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Solve a cipher using a user-provided key.
    Supported methods:
    - 'substitution': Requires 'key' (dict mapping or 26-char string)
    - 'caesar': Requires 'key' (int shift)
    - 'atbash': No key needed (Hebrew or English)
    - 'reverse': No key needed
    """
    text = request.text
    method = request.method.lower()
    result = ""
    error = None

    try:
        if method == 'reverse':
            result = text[::-1]
            
        elif method == 'atbash':
            # English Atbash: A<->Z, B<->Y...
            def atbash_char(c):
                if 'A' <= c <= 'Z':
                    return chr(ord('Z') - (ord(c) - ord('A')))
                elif 'a' <= c <= 'z':
                    return chr(ord('z') - (ord(c) - ord('a')))
                return c
            result = "".join(atbash_char(c) for c in text)
            
        elif method == 'caesar':
            shift = int(request.key) if request.key is not None else 0
            def shift_char(c, s):
                if 'A' <= c <= 'Z':
                    return chr((ord(c) - ord('A') + s) % 26 + ord('A'))
                elif 'a' <= c <= 'z':
                    return chr((ord(c) - ord('a') + s) % 26 + ord('a'))
                return c
            result = "".join(shift_char(c, shift) for c in text)
            
        elif method == 'substitution':
            key = request.key
            mapping = {}
            if isinstance(key, dict):
                mapping = {k.upper(): v.upper() for k, v in key.items()}
            elif isinstance(key, str) and len(key) == 26:
                # Key string represents A-Z mapping
                # e.g. "QWERTY..." means A->Q, B->W...
                base = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                mapping = {base[i]: key[i].upper() for i in range(26)}
            else:
                raise ValueError("Invalid substitution key. Must be dict or 26-char string.")
            
            result = "".join([mapping.get(c.upper(), c) if c.isalpha() else c for c in text])
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    return {
        "original": text,
        "method": method,
        "key": request.key,
        "result": result
    }

class PropheticConvergenceRequest(BaseModel):
    terms: List[Dict[str, Any]]
    max_spread: int = 500
    generate_visual: bool = True

@router.post("/prophetic/convergence", response_model=Dict[str, Any])
async def find_prophetic_convergence(
    request: PropheticConvergenceRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Find triple/multiple convergence zones in the Torah.
    Returns top zones and optional visualization for the best one.
    """
    try:
        text = get_torah_text()
        analyzer = PropheticAnalyzerService()
        
        zones = analyzer.find_triple_convergence(
            text, 
            request.terms, 
            max_spread=request.max_spread
        )
        
        response = {
            "total_zones_found": len(zones),
            "top_zones": []
        }
        
        # Process top 5
        for z in zones[:5]:
            zone_data = {
                "center_index": z.center_index,
                "spread": z.spread,
                "book": z.book,
                "position_percentage": z.position_percentage,
                "terms": [
                    {
                        "name": name,
                        "term": res.term,
                        "skip": res.skip,
                        "start_index": res.start_index
                    }
                    for name, res in z.terms.items()
                ]
            }
            
            if request.generate_visual and z == zones[0]:
                # Assign colors for visual
                colors = ["#4CAF50", "#9C27B0", "#009688", "#FF5722", "#3F51B5"]
                terms_for_vis = []
                for i, (name, res) in enumerate(z.terms.items()):
                    terms_for_vis.append({
                        "term": res.term,
                        "name": name,
                        "skip": res.skip,
                        "color": colors[i % len(colors)]
                    })
                
                svg = ELSVisualizer.generate_svg_matrix(
                    text, 
                    z.center_index, 
                    terms_for_vis,
                    row_width=12 # Should be dynamic based on best term skip?
                )
                zone_data["visualization_svg"] = svg
                
            response["top_zones"].append(zone_data)
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
