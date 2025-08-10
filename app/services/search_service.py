"""
Advanced search service for complex queries across multiple dimensions
"""
from typing import Dict, List, Any, Optional, Union
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
from collections import defaultdict
import re

from app.models.database_models import (
    Document, Page, Character, Word, Pattern, Grid, GeometricMeasurement
)
from app.models.data_access import (
    DocumentRepository, PageRepository, CharacterRepository, 
    WordRepository, PatternRepository, GridRepository, 
    GeometricRepository, EtymologyRepository
)
from app.services.etymology_engine import EtymologyEngine, Language


class SearchService:
    """
    Advanced search service for complex queries across multiple dimensions
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.document_repo = DocumentRepository(db)
        self.page_repo = PageRepository(db)
        self.character_repo = CharacterRepository(db)
        self.word_repo = WordRepository(db)
        self.pattern_repo = PatternRepository(db)
        self.grid_repo = GridRepository(db)
        self.geometric_repo = GeometricRepository(db)
        self.etymology_repo = EtymologyRepository(db)
    
    def search_text(self, query: str, document_id: Optional[int] = None, 
                   case_sensitive: bool = False, whole_word: bool = False) -> Dict[str, Any]:
        """
        Search for text across documents or within a specific document
        """
        results = {
            'query': query,
            'document_id': document_id,
            'case_sensitive': case_sensitive,
            'whole_word': whole_word,
            'word_matches': [],
            'character_matches': [],
            'pattern_matches': [],
            'total_matches': 0
        }
        
        # Build the search condition
        if whole_word:
            if case_sensitive:
                text_filter = Word.text == query
            else:
                text_filter = func.lower(Word.text) == query.lower()
        else:
            if case_sensitive:
                text_filter = Word.text.contains(query)
            else:
                text_filter = func.lower(Word.text).contains(query.lower())
        
        # Apply document filter if specified
        if document_id:
            word_query = self.db.query(Word).join(Page).filter(
                and_(Page.document_id == document_id, text_filter)
            )
        else:
            word_query = self.db.query(Word).join(Page).filter(text_filter)
        
        # Execute query and process results
        word_matches = word_query.all()
        results['word_matches'] = [
            {
                'word_id': word.id,
                'text': word.text,
                'page_id': word.page_id,
                'page_number': self.db.query(Page).filter(Page.id == word.page_id).first().page_number,
                'document_id': self.db.query(Page).filter(Page.id == word.page_id).first().document_id,
                'position': {'x': word.x, 'y': word.y},
                'confidence': word.confidence
            }
            for word in word_matches
        ]
        
        # Search for patterns containing the query
        pattern_matches = self.db.query(Pattern).filter(
            Pattern.description.ilike(f'%{query}%')
        ).all()
        
        results['pattern_matches'] = [
            {
                'pattern_id': pattern.id,
                'pattern_type': pattern.pattern_type,
                'description': pattern.description,
                'document_id': pattern.document_id,
                'confidence': pattern.confidence,
                'significance': pattern.significance_score
            }
            for pattern in pattern_matches
        ]
        
        results['total_matches'] = len(results['word_matches']) + len(results['pattern_matches'])
        return results
    
    def search_patterns(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search for patterns matching specific criteria
        """
        query = self.db.query(Pattern)
        
        # Apply filters based on criteria
        if 'document_id' in criteria:
            query = query.filter(Pattern.document_id == criteria['document_id'])
        
        if 'pattern_type' in criteria:
            query = query.filter(Pattern.pattern_type == criteria['pattern_type'])
        
        if 'min_confidence' in criteria:
            query = query.filter(Pattern.confidence >= criteria['min_confidence'])
        
        if 'min_significance' in criteria:
            query = query.filter(Pattern.significance_score >= criteria['min_significance'])
        
        if 'description_contains' in criteria:
            query = query.filter(Pattern.description.ilike(f"%{criteria['description_contains']}%"))
        
        # Sort results
        if criteria.get('sort_by') == 'confidence':
            query = query.order_by(desc(Pattern.confidence))
        elif criteria.get('sort_by') == 'significance':
            query = query.order_by(desc(Pattern.significance_score))
        else:
            query = query.order_by(desc(Pattern.significance_score))
        
        # Apply pagination
        limit = criteria.get('limit', 100)
        offset = criteria.get('offset', 0)
        query = query.limit(limit).offset(offset)
        
        # Execute query
        patterns = query.all()
        
        return [
            {
                'pattern_id': p.id,
                'pattern_type': p.pattern_type,
                'description': p.description,
                'document_id': p.document_id,
                'confidence': p.confidence,
                'significance': p.significance_score,
                'page_numbers': p.page_numbers,
                'coordinates': p.coordinates,
                'pattern_data': p.pattern_data
            }
            for p in patterns
        ]
    
    def search_geometric_relationships(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search for geometric relationships matching specific criteria
        """
        query = self.db.query(GeometricMeasurement)
        
        # Apply filters
        if 'document_id' in criteria:
            query = query.filter(GeometricMeasurement.document_id == criteria['document_id'])
        
        if 'measurement_type' in criteria:
            query = query.filter(GeometricMeasurement.measurement_type == criteria['measurement_type'])
        
        if 'min_value' in criteria and 'max_value' in criteria:
            query = query.filter(
                and_(
                    GeometricMeasurement.measurement_value >= criteria['min_value'],
                    GeometricMeasurement.measurement_value <= criteria['max_value']
                )
            )
        
        if 'is_significant' in criteria:
            query = query.filter(GeometricMeasurement.is_significant == criteria['is_significant'])
        
        # Sort and paginate
        query = query.order_by(desc(GeometricMeasurement.significance_score))
        limit = criteria.get('limit', 100)
        offset = criteria.get('offset', 0)
        query = query.limit(limit).offset(offset)
        
        measurements = query.all()
        
        return [
            {
                'measurement_id': m.id,
                'measurement_type': m.measurement_type,
                'measurement_value': m.measurement_value,
                'measurement_unit': m.measurement_unit,
                'document_id': m.document_id,
                'page_id': m.page_id,
                'significance': m.significance_score,
                'is_significant': m.is_significant,
                'coordinates': m.coordinates,
                'description': m.description
            }
            for m in measurements
        ]
    
    def search_etymology(self, word: str, languages: List[Language] = None) -> Dict[str, Any]:
        """
        Search for etymology data across multiple languages
        """
        if languages is None:
            languages = [Language.HEBREW, Language.GREEK, Language.LATIN, Language.ENGLISH]
        
        results = {
            'query': word,
            'results': {},
            'total_matches': 0
        }
        
        for language in languages:
            # Check cache first
            cached = self.etymology_repo.get_etymology(word, language.value)
            if cached:
                results['results'][language.value] = {
                    'cached': True,
                    'word': cached.word,
                    'normalized_form': cached.normalized_form,
                    'root_words': cached.root_words,
                    'definitions': cached.definitions,
                    'translations': cached.translations,
                    'confidence': cached.confidence_score
                }
                results['total_matches'] += 1
            else:
                # Perform live analysis
                engine = EtymologyEngine()
                analysis = engine.analyze_word(word, language)
                
                if analysis.confidence > 0.5:  # Only include meaningful results
                    results['results'][language.value] = {
                        'cached': False,
                        'word': analysis.word,
                        'root_words': [
                            {'root': r.root, 'meaning': r.meaning}
                            for r in analysis.root_words
                        ],
                        'definitions': analysis.definitions,
                        'confidence': analysis.confidence
                    }
                    results['total_matches'] += 1
                    
                    # Cache the result for future queries
                    etymology_data = {
                        'normalized_form': word.lower(),
                        'root_words': [
                            {'root': r.root, 'meaning': r.meaning}
                            for r in analysis.root_words
                        ],
                        'definitions': analysis.definitions,
                        'confidence_score': analysis.confidence
                    }
                    self.etymology_repo.cache_etymology(word, language.value, etymology_data)
        
        return results
    
    def search_cross_document_patterns(self, document_ids: List[int], 
                                     min_significance: float = 0.7) -> Dict[str, Any]:
        """
        Search for patterns that appear across multiple documents
        """
        results = {
            'document_ids': document_ids,
            'common_patterns': [],
            'unique_patterns': {},
            'correlation_score': 0.0
        }
        
        # Get patterns for each document
        document_patterns = {}
        for doc_id in document_ids:
            patterns = self.pattern_repo.get_significant_patterns(doc_id, min_significance)
            document_patterns[doc_id] = patterns
        
        # Find common patterns (by description similarity)
        pattern_groups = defaultdict(list)
        for doc_id, patterns in document_patterns.items():
            for pattern in patterns:
                # Use pattern type and a simplified description as key
                key = f"{pattern.pattern_type}_{self._simplify_description(pattern.description)}"
                pattern_groups[key].append({
                    'document_id': doc_id,
                    'pattern_id': pattern.id,
                    'pattern_type': pattern.pattern_type,
                    'description': pattern.description,
                    'significance': pattern.significance_score
                })
        
        # Extract common patterns (appearing in multiple documents)
        for key, patterns in pattern_groups.items():
            doc_ids = set(p['document_id'] for p in patterns)
            if len(doc_ids) > 1:  # Pattern appears in multiple documents
                results['common_patterns'].append({
                    'pattern_key': key,
                    'occurrences': patterns,
                    'document_count': len(doc_ids),
                    'average_significance': sum(p['significance'] for p in patterns) / len(patterns)
                })
        
        # Extract unique patterns (appearing in only one document)
        for doc_id in document_ids:
            results['unique_patterns'][doc_id] = []
            for key, patterns in pattern_groups.items():
                pattern_doc_ids = set(p['document_id'] for p in patterns)
                if len(pattern_doc_ids) == 1 and doc_id in pattern_doc_ids:
                    results['unique_patterns'][doc_id].extend(patterns)
        
        # Calculate correlation score based on common patterns
        if document_ids:
            common_pattern_count = len(results['common_patterns'])
            total_pattern_count = sum(len(patterns) for patterns in document_patterns.values())
            if total_pattern_count > 0:
                results['correlation_score'] = common_pattern_count / total_pattern_count
        
        return results
    
    def complex_query(self, query_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute complex multi-dimensional queries
        """
        results = {
            'query_spec': query_spec,
            'results': {},
            'total_matches': 0,
            'execution_time': 0
        }
        
        # Text search component
        if 'text' in query_spec:
            text_results = self.search_text(
                query_spec['text'].get('query', ''),
                query_spec['text'].get('document_id'),
                query_spec['text'].get('case_sensitive', False),
                query_spec['text'].get('whole_word', False)
            )
            results['results']['text'] = text_results
            results['total_matches'] += text_results['total_matches']
        
        # Pattern search component
        if 'patterns' in query_spec:
            pattern_results = self.search_patterns(query_spec['patterns'])
            results['results']['patterns'] = pattern_results
            results['total_matches'] += len(pattern_results)
        
        # Geometric search component
        if 'geometric' in query_spec:
            geometric_results = self.search_geometric_relationships(query_spec['geometric'])
            results['results']['geometric'] = geometric_results
            results['total_matches'] += len(geometric_results)
        
        # Etymology search component
        if 'etymology' in query_spec:
            etymology_results = self.search_etymology(
                query_spec['etymology'].get('word', ''),
                query_spec['etymology'].get('languages')
            )
            results['results']['etymology'] = etymology_results
            results['total_matches'] += etymology_results['total_matches']
        
        # Cross-document analysis component
        if 'cross_document' in query_spec:
            cross_doc_results = self.search_cross_document_patterns(
                query_spec['cross_document'].get('document_ids', []),
                query_spec['cross_document'].get('min_significance', 0.7)
            )
            results['results']['cross_document'] = cross_doc_results
            results['total_matches'] += len(cross_doc_results['common_patterns'])
        
        return results
    
    def get_search_suggestions(self, partial_query: str, search_type: str = 'all') -> List[str]:
        """
        Get search suggestions based on partial query
        """
        suggestions = []
        
        if search_type in ['all', 'words']:
            # Get word suggestions
            word_suggestions = self.db.query(Word.text).filter(
                Word.text.ilike(f'{partial_query}%')
            ).distinct().limit(10).all()
            suggestions.extend([w.text for w in word_suggestions])
        
        if search_type in ['all', 'patterns']:
            # Get pattern type suggestions
            pattern_suggestions = self.db.query(Pattern.pattern_type).filter(
                Pattern.pattern_type.ilike(f'{partial_query}%')
            ).distinct().limit(10).all()
            suggestions.extend([p.pattern_type for p in pattern_suggestions])
        
        return list(set(suggestions))  # Remove duplicates
    
    def _simplify_description(self, description: str) -> str:
        """Simplify pattern description for comparison"""
        # Remove specific measurements and coordinates
        simplified = re.sub(r'\d+\.\d+', 'X.X', description)
        simplified = re.sub(r'\d+', 'N', simplified)
        # Keep only alphabetic characters and spaces
        simplified = re.sub(r'[^a-zA-Z\s]', '', simplified)
        # Convert to lowercase and remove extra spaces
        simplified = ' '.join(simplified.lower().split())
        return simplified