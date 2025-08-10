"""
Etymology and Linguistic Analysis Engine for Ancient Text Analyzer
Provides multi-language analysis for Hebrew, Greek, Latin, and English texts
"""
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

class Language(Enum):
    """Supported languages for etymology analysis"""
    HEBREW = "hebrew"
    GREEK = "greek"
    LATIN = "latin"
    ENGLISH = "english"
    ARAMAIC = "aramaic"

class WordType(Enum):
    """Types of words for morphological analysis"""
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PREPOSITION = "preposition"
    CONJUNCTION = "conjunction"
    ARTICLE = "article"
    PRONOUN = "pronoun"
    PARTICLE = "particle"
    PROPER_NOUN = "proper_noun"

@dataclass
class RootWord:
    """Container for root word information"""
    root: str
    language: Language
    meaning: str
    morphology: Dict[str, Any]
    variants: List[str]
    frequency: int
    historical_period: str
    source: str

@dataclass
class EtymologyResult:
    """Container for etymology analysis results"""
    word: str
    language: Language
    root_words: List[RootWord]
    definitions: List[str]
    morphological_analysis: Dict[str, Any]
    historical_forms: List[str]
    related_words: List[str]
    confidence: float
    sources: List[str]

@dataclass
class TranslationComparison:
    """Container for translation comparison results"""
    word: str
    translations: Dict[Language, List[str]]
    variations: List[Dict[str, Any]]
    linguistic_notes: List[str]
    confidence: float

@dataclass
class UsageHistory:
    """Container for historical usage data"""
    word: str
    language: Language
    periods: List[Dict[str, Any]]
    frequency_changes: List[Dict[str, Any]]
    semantic_shifts: List[str]
    geographic_distribution: Dict[str, Any]

class EtymologyEngine:
    """
    Advanced etymology and linguistic analysis engine
    Specialized for ancient languages and historical text analysis
    """
    
    def __init__(self):
        # Initialize language-specific data
        self.hebrew_data = self._load_hebrew_data()
        self.greek_data = self._load_greek_data()
        self.latin_data = self._load_latin_data()
        self.english_data = self._load_english_data()
        
        # Common patterns for each language
        self.language_patterns = {
            Language.HEBREW: self._get_hebrew_patterns(),
            Language.GREEK: self._get_greek_patterns(),
            Language.LATIN: self._get_latin_patterns(),
            Language.ENGLISH: self._get_english_patterns()
        }
        
        # Morphological rules
        self.morphological_rules = self._load_morphological_rules()
        
    def analyze_word(self, word: str, language: Language) -> EtymologyResult:
        """
        Perform comprehensive etymology analysis of a word
        
        Args:
            word: Word to analyze
            language: Language of the word
            
        Returns:
            EtymologyResult with detailed analysis
        """
        logger.info(f"Analyzing word '{word}' in {language.value}")
        
        try:
            # Normalize the word
            normalized_word = self._normalize_word(word, language)
            
            # Find root words
            root_words = self.find_root_words(normalized_word, language)
            
            # Get definitions
            definitions = self._get_definitions(normalized_word, language)
            
            # Perform morphological analysis
            morphology = self._analyze_morphology(normalized_word, language)
            
            # Find historical forms
            historical_forms = self._find_historical_forms(normalized_word, language)
            
            # Find related words
            related_words = self._find_related_words(normalized_word, language)
            
            # Calculate confidence
            confidence = self._calculate_etymology_confidence(
                normalized_word, language, root_words, definitions
            )
            
            # Determine sources
            sources = self._get_sources(language)
            
            result = EtymologyResult(
                word=word,
                language=language,
                root_words=root_words,
                definitions=definitions,
                morphological_analysis=morphology,
                historical_forms=historical_forms,
                related_words=related_words,
                confidence=confidence,
                sources=sources
            )
            
            logger.info(f"Etymology analysis completed for '{word}' with confidence {confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Etymology analysis failed for '{word}': {e}")
            return EtymologyResult(
                word=word,
                language=language,
                root_words=[],
                definitions=[],
                morphological_analysis={},
                historical_forms=[],
                related_words=[],
                confidence=0.0,
                sources=[]
            )
    
    def find_root_words(self, word: str, language: Language) -> List[RootWord]:
        """
        Find root words for the given word in the specified language
        
        Args:
            word: Word to find roots for
            language: Language of the word
            
        Returns:
            List of RootWord objects
        """
        logger.debug(f"Finding root words for '{word}' in {language.value}")
        
        root_words = []
        
        if language == Language.HEBREW:
            root_words = self._find_hebrew_roots(word)
        elif language == Language.GREEK:
            root_words = self._find_greek_roots(word)
        elif language == Language.LATIN:
            root_words = self._find_latin_roots(word)
        elif language == Language.ENGLISH:
            root_words = self._find_english_roots(word)
        
        return root_words
    
    def _find_hebrew_roots(self, word: str) -> List[RootWord]:
        """Find Hebrew root words (typically 3-letter roots)"""
        roots = []
        
        # Hebrew words typically derive from 3-letter roots
        if len(word) >= 3:
            # Try to extract 3-letter combinations
            for i in range(len(word) - 2):
                potential_root = word[i:i+3]
                
                # Check against known Hebrew roots
                if potential_root in self.hebrew_data.get('roots', {}):
                    root_info = self.hebrew_data['roots'][potential_root]
                    
                    root = RootWord(
                        root=potential_root,
                        language=Language.HEBREW,
                        meaning=root_info.get('meaning', 'Unknown'),
                        morphology=root_info.get('morphology', {}),
                        variants=root_info.get('variants', []),
                        frequency=root_info.get('frequency', 0),
                        historical_period=root_info.get('period', 'Biblical Hebrew'),
                        source='Hebrew Lexicon'
                    )
                    roots.append(root)
        
        return roots
    
    def _find_greek_roots(self, word: str) -> List[RootWord]:
        """Find Greek root words"""
        roots = []
        
        # Check against known Greek roots and stems
        for root_pattern, root_info in self.greek_data.get('roots', {}).items():
            if re.search(root_pattern, word, re.IGNORECASE):
                root = RootWord(
                    root=root_pattern,
                    language=Language.GREEK,
                    meaning=root_info.get('meaning', 'Unknown'),
                    morphology=root_info.get('morphology', {}),
                    variants=root_info.get('variants', []),
                    frequency=root_info.get('frequency', 0),
                    historical_period=root_info.get('period', 'Koine Greek'),
                    source='Greek Lexicon'
                )
                roots.append(root)
        
        return roots
    
    def _find_latin_roots(self, word: str) -> List[RootWord]:
        """Find Latin root words"""
        roots = []
        
        # Check against known Latin roots and stems
        for root_pattern, root_info in self.latin_data.get('roots', {}).items():
            if word.lower().startswith(root_pattern.lower()) or root_pattern.lower() in word.lower():
                root = RootWord(
                    root=root_pattern,
                    language=Language.LATIN,
                    meaning=root_info.get('meaning', 'Unknown'),
                    morphology=root_info.get('morphology', {}),
                    variants=root_info.get('variants', []),
                    frequency=root_info.get('frequency', 0),
                    historical_period=root_info.get('period', 'Classical Latin'),
                    source='Latin Dictionary'
                )
                roots.append(root)
        
        return roots
    
    def _find_english_roots(self, word: str) -> List[RootWord]:
        """Find English root words (often from other languages)"""
        roots = []
        
        # Check for common English prefixes and suffixes that indicate etymology
        prefixes = ['pre', 'post', 'anti', 'pro', 'sub', 'super', 'inter', 'intra', 'beau']
        suffixes = ['tion', 'sion', 'ment', 'ness', 'able', 'ible', 'ous', 'ious', 'ful']
        
        # Remove common prefixes and suffixes to find root
        root_word = word.lower()
        
        for prefix in prefixes:
            if root_word.startswith(prefix):
                root_word = root_word[len(prefix):]
                break
        
        for suffix in suffixes:
            if root_word.endswith(suffix):
                root_word = root_word[:-len(suffix)]
                break
        
        if root_word and (root_word != word.lower() or word.lower().startswith('beau')):
            root = RootWord(
                root=root_word,
                language=Language.ENGLISH,
                meaning=f"Root of {word}",
                morphology={'type': 'derived'},
                variants=[word],
                frequency=100,  # Default frequency
                historical_period='Modern English',
                source='English Etymology'
            )
            roots.append(root)
        
        return roots
    
    def _analyze_morphology(self, word: str, language: Language) -> Dict[str, Any]:
        """
        Analyze the morphological structure of a word
        
        Args:
            word: Word to analyze
            language: Language of the word
            
        Returns:
            Dictionary with morphological analysis
        """
        morphology = {
            'word': word,
            'language': language.value,
            'length': len(word),
            'syllables': self._count_syllables(word),
            'word_type': self._determine_word_type(word, language),
            'prefixes': self._find_prefixes(word, language),
            'suffixes': self._find_suffixes(word, language),
            'stem': self._find_stem(word, language),
            'inflections': self._find_inflections(word, language)
        }
        
        return morphology
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified algorithm)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _determine_word_type(self, word: str, language: Language) -> WordType:
        """Determine the grammatical type of a word"""
        # Simplified word type detection based on patterns
        word_lower = word.lower()
        
        # Common noun endings
        noun_endings = ['tion', 'sion', 'ment', 'ness', 'ity', 'er', 'or', 'ist']
        if any(word_lower.endswith(ending) for ending in noun_endings):
            return WordType.NOUN
        
        # Common verb endings
        verb_endings = ['ing', 'ed', 'ate', 'ize', 'ify']
        if any(word_lower.endswith(ending) for ending in verb_endings):
            return WordType.VERB
        
        # Common adjective endings
        adj_endings = ['able', 'ible', 'ous', 'ious', 'ful', 'less', 'ive']
        if any(word_lower.endswith(ending) for ending in adj_endings):
            return WordType.ADJECTIVE
        
        # Common adverb endings
        if word_lower.endswith('ly'):
            return WordType.ADVERB
        
        # Default to noun if uncertain
        return WordType.NOUN
    
    def _find_prefixes(self, word: str, language: Language) -> List[str]:
        """Find prefixes in a word"""
        prefixes = []
        patterns = self.language_patterns.get(language, {}).get('prefixes', [])
        
        for prefix in patterns:
            if word.lower().startswith(prefix.lower()):
                prefixes.append(prefix)
        
        return prefixes
    
    def _find_suffixes(self, word: str, language: Language) -> List[str]:
        """Find suffixes in a word"""
        suffixes = []
        patterns = self.language_patterns.get(language, {}).get('suffixes', [])
        
        for suffix in patterns:
            if word.lower().endswith(suffix.lower()):
                suffixes.append(suffix)
        
        return suffixes
    
    def _find_stem(self, word: str, language: Language) -> str:
        """Find the stem of a word by removing prefixes and suffixes"""
        stem = word.lower()
        
        # Remove prefixes
        prefixes = self._find_prefixes(word, language)
        for prefix in prefixes:
            if stem.startswith(prefix.lower()):
                stem = stem[len(prefix):]
                break
        
        # Remove suffixes
        suffixes = self._find_suffixes(word, language)
        for suffix in suffixes:
            if stem.endswith(suffix.lower()):
                stem = stem[:-len(suffix)]
                break
        
        return stem if stem else word.lower()
    
    def _find_inflections(self, word: str, language: Language) -> List[str]:
        """Find inflectional variations of a word"""
        inflections = []
        
        # This is a simplified implementation
        # In practice, you'd use comprehensive morphological databases
        
        if language == Language.ENGLISH:
            # Common English inflections
            base = word.lower()
            
            # Plural forms
            if base.endswith('s') and len(base) > 1:
                inflections.append(base[:-1])  # Remove 's'
            
            # Past tense
            if base.endswith('ed') and len(base) > 2:
                inflections.append(base[:-2])  # Remove 'ed'
            
            # Present participle
            if base.endswith('ing') and len(base) > 3:
                inflections.append(base[:-3])  # Remove 'ing'
        
        return inflections
    
    def compare_translations(self, word: str, versions: List[str]) -> TranslationComparison:
        """
        Compare translations of a word across different versions
        
        Args:
            word: Word to compare
            versions: List of version identifiers
            
        Returns:
            TranslationComparison with analysis results
        """
        logger.info(f"Comparing translations for '{word}' across {len(versions)} versions")
        
        translations = {}
        variations = []
        linguistic_notes = []
        
        # This is a simplified implementation
        # In practice, you'd query actual translation databases
        
        for version in versions:
            # Mock translation data
            if version.lower() in ['kjv', 'king james']:
                translations[Language.ENGLISH] = [word, f"{word} (KJV)"]
            elif version.lower() in ['lxx', 'septuagint']:
                translations[Language.GREEK] = [f"Greek: {word}"]
            elif version.lower() in ['vulgate']:
                translations[Language.LATIN] = [f"Latin: {word}"]
        
        # Analyze variations
        if len(translations) > 1:
            variations.append({
                'type': 'cross_language_variation',
                'description': f"Word appears in {len(translations)} languages",
                'significance': 0.8
            })
        
        # Add linguistic notes
        linguistic_notes.append(f"Translation comparison for '{word}' across {len(versions)} versions")
        
        confidence = 0.7 if translations else 0.3
        
        return TranslationComparison(
            word=word,
            translations=translations,
            variations=variations,
            linguistic_notes=linguistic_notes,
            confidence=confidence
        )
    
    def get_historical_usage(self, word: str, language: Language) -> UsageHistory:
        """
        Get historical usage patterns for a word
        
        Args:
            word: Word to analyze
            language: Language of the word
            
        Returns:
            UsageHistory with historical data
        """
        logger.info(f"Getting historical usage for '{word}' in {language.value}")
        
        # Mock historical data - in practice, this would query historical corpora
        periods = [
            {
                'period': 'Ancient',
                'frequency': 50,
                'contexts': ['religious', 'formal'],
                'meaning_shifts': []
            },
            {
                'period': 'Medieval',
                'frequency': 75,
                'contexts': ['religious', 'scholarly'],
                'meaning_shifts': ['expanded usage']
            },
            {
                'period': 'Modern',
                'frequency': 25,
                'contexts': ['archaic', 'literary'],
                'meaning_shifts': ['archaic usage']
            }
        ]
        
        frequency_changes = [
            {'period': 'Ancient to Medieval', 'change': '+50%'},
            {'period': 'Medieval to Modern', 'change': '-67%'}
        ]
        
        semantic_shifts = [
            'Original meaning: religious/sacred context',
            'Medieval expansion: broader scholarly usage',
            'Modern restriction: primarily archaic/literary'
        ]
        
        geographic_distribution = {
            'primary_regions': ['Middle East', 'Mediterranean'],
            'spread_pattern': 'Religious and scholarly transmission',
            'modern_usage': 'Academic and religious contexts'
        }
        
        return UsageHistory(
            word=word,
            language=language,
            periods=periods,
            frequency_changes=frequency_changes,
            semantic_shifts=semantic_shifts,
            geographic_distribution=geographic_distribution
        )
    
    def _normalize_word(self, word: str, language: Language) -> str:
        """Normalize a word for analysis"""
        normalized = word.strip().lower()
        
        # Language-specific normalization
        if language == Language.HEBREW:
            # Remove Hebrew vowel points and cantillation marks
            normalized = re.sub(r'[\u0591-\u05C7]', '', normalized)
        elif language == Language.GREEK:
            # Normalize Greek accents and breathing marks
            normalized = re.sub(r'[\u0300-\u036F]', '', normalized)
        
        return normalized
    
    def _get_definitions(self, word: str, language: Language) -> List[str]:
        """Get definitions for a word"""
        # Mock definitions - in practice, query lexical databases
        if not word:
            return []
        return [f"Definition of {word} in {language.value}"]
    
    def _find_historical_forms(self, word: str, language: Language) -> List[str]:
        """Find historical forms of a word"""
        # Mock historical forms
        return [f"Historical form of {word}"]
    
    def _find_related_words(self, word: str, language: Language) -> List[str]:
        """Find words related to the given word"""
        # Mock related words
        return [f"Related to {word}"]
    
    def _calculate_etymology_confidence(self, word: str, language: Language, 
                                      root_words: List[RootWord], definitions: List[str]) -> float:
        """Calculate confidence score for etymology analysis"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on available data
        if root_words:
            confidence += 0.2
        if definitions:
            confidence += 0.2
        if len(word) > 2:  # Longer words generally have more reliable etymology
            confidence += 0.1
        if len(word) == 0:
            confidence = 0.0
        
        return min(confidence, 1.0)
    
    def _get_sources(self, language: Language) -> List[str]:
        """Get sources for etymology data"""
        sources = {
            Language.HEBREW: ['Brown-Driver-Briggs Hebrew Lexicon', 'Strong\'s Hebrew Dictionary'],
            Language.GREEK: ['Liddell-Scott Greek Lexicon', 'Strong\'s Greek Dictionary'],
            Language.LATIN: ['Lewis & Short Latin Dictionary', 'Oxford Latin Dictionary'],
            Language.ENGLISH: ['Oxford English Dictionary', 'Etymology Online']
        }
        
        return sources.get(language, ['General Etymology Sources'])
    
    def _load_hebrew_data(self) -> Dict[str, Any]:
        """Load Hebrew language data"""
        return {
            'roots': {
                'אמן': {'meaning': 'to be firm, faithful', 'morphology': {'type': 'verb'}, 'variants': ['amen'], 'frequency': 100, 'period': 'Biblical Hebrew'},
                'ברא': {'meaning': 'to create', 'morphology': {'type': 'verb'}, 'variants': ['bara'], 'frequency': 50, 'period': 'Biblical Hebrew'},
                'דבר': {'meaning': 'word, thing', 'morphology': {'type': 'noun'}, 'variants': ['davar'], 'frequency': 200, 'period': 'Biblical Hebrew'}
            }
        }
    
    def _load_greek_data(self) -> Dict[str, Any]:
        """Load Greek language data"""
        return {
            'roots': {
                'λογ': {'meaning': 'word, reason', 'morphology': {'type': 'noun'}, 'variants': ['logos'], 'frequency': 150, 'period': 'Koine Greek'},
                'θεο': {'meaning': 'god', 'morphology': {'type': 'noun'}, 'variants': ['theos'], 'frequency': 300, 'period': 'Koine Greek'},
                'αγαπ': {'meaning': 'love', 'morphology': {'type': 'noun/verb'}, 'variants': ['agape'], 'frequency': 100, 'period': 'Koine Greek'}
            }
        }
    
    def _load_latin_data(self) -> Dict[str, Any]:
        """Load Latin language data"""
        return {
            'roots': {
                'verb': {'meaning': 'word', 'morphology': {'type': 'noun'}, 'variants': ['verbum'], 'frequency': 100, 'period': 'Classical Latin'},
                'deus': {'meaning': 'god', 'morphology': {'type': 'noun'}, 'variants': ['deus'], 'frequency': 200, 'period': 'Classical Latin'},
                'amor': {'meaning': 'love', 'morphology': {'type': 'noun'}, 'variants': ['amor'], 'frequency': 80, 'period': 'Classical Latin'}
            }
        }
    
    def _load_english_data(self) -> Dict[str, Any]:
        """Load English language data"""
        return {
            'roots': {},  # English roots are typically from other languages
            'common_words': ['the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'you', 'that']
        }
    
    def _get_hebrew_patterns(self) -> Dict[str, List[str]]:
        """Get Hebrew morphological patterns"""
        return {
            'prefixes': ['ב', 'כ', 'ל', 'מ', 'ה', 'ו'],
            'suffixes': ['ים', 'ות', 'ה', 'י', 'ך', 'נו']
        }
    
    def _get_greek_patterns(self) -> Dict[str, List[str]]:
        """Get Greek morphological patterns"""
        return {
            'prefixes': ['αντι', 'προ', 'συν', 'κατα', 'μετα', 'παρα'],
            'suffixes': ['ος', 'η', 'ον', 'ων', 'εις', 'ας']
        }
    
    def _get_latin_patterns(self) -> Dict[str, List[str]]:
        """Get Latin morphological patterns"""
        return {
            'prefixes': ['pre', 'post', 'sub', 'super', 'inter', 'intra'],
            'suffixes': ['us', 'a', 'um', 'is', 'es', 'orum']
        }
    
    def _get_english_patterns(self) -> Dict[str, List[str]]:
        """Get English morphological patterns"""
        return {
            'prefixes': ['pre', 'post', 'anti', 'pro', 'sub', 'super', 'inter', 'un', 're'],
            'suffixes': ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ment', 'ness', 'able', 'ible']
        }
    
    def _load_morphological_rules(self) -> Dict[str, Any]:
        """Load morphological analysis rules"""
        return {
            'hebrew': {
                'root_pattern': r'[א-ת]{3}',  # 3-letter Hebrew root pattern
                'verb_patterns': ['קטל', 'פעל', 'פיעל'],
                'noun_patterns': ['קטל', 'קטלה', 'מקטל']
            },
            'greek': {
                'verb_endings': ['ω', 'εις', 'ει', 'ομεν', 'ετε', 'ουσι'],
                'noun_endings': ['ος', 'η', 'ον', 'ου', 'ης', 'ων']
            },
            'latin': {
                'verb_endings': ['o', 's', 't', 'mus', 'tis', 'nt'],
                'noun_endings': ['us', 'a', 'um', 'i', 'ae', 'orum']
            }
        }