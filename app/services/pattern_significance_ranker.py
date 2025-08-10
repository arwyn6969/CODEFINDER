"""
Pattern Significance Ranking System
Implements pattern scoring algorithms based on statistical significance,
ranking systems for detected patterns and anomalies, confidence intervals,
and pattern interpretation suggestions.
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import math
import numpy as np
from scipy import stats
from scipy.stats import chi2, norm, t
import logging
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

from app.models.database_models import Document, Pattern
from app.models.cross_document_models import CrossDocumentPattern, CrossPatternInstance
from app.services.anomaly_detector import AnomalyDetector, AnomalyResult
from app.core.database import get_db


@dataclass
class PatternScore:
    """Represents a pattern's significance score"""
    pattern_id: int
    pattern_type: str
    significance_score: float
    confidence_interval: Tuple[float, float]
    reliability_measure: float
    statistical_measures: Dict[str, float]
    ranking_factors: Dict[str, float]


@dataclass
class RankedPattern:
    """Represents a ranked pattern with interpretation suggestions"""
    pattern: Any  # Pattern object
    score: PatternScore
    rank: int
    interpretation_suggestions: List[str]
    evidence_strength: float
    recommendation_priority: str


@dataclass
class RankingResult:
    """Container for ranking results"""
    ranked_patterns: List[RankedPattern]
    ranking_metadata: Dict[str, Any]
    summary_statistics: Dict[str, float]


class PatternSignificanceRanker:
    """
    Service for ranking patterns and anomalies by statistical significance
    """
    
    def __init__(self, db_session: Session = None):
        self.db = db_session or next(get_db())
        self.anomaly_detector = AnomalyDetector(self.db)
        self.logger = logging.getLogger(__name__)
        
        # Ranking parameters
        self.min_significance_threshold = 0.05
        self.confidence_level = 0.95
        self.reliability_threshold = 0.7
        
        # Weighting factors for different ranking criteria
        self.ranking_weights = {
            'statistical_significance': 0.3,
            'frequency_rarity': 0.2,
            'cross_document_presence': 0.2,
            'complexity_score': 0.15,
            'consistency_measure': 0.15
        }   
 
    def rank_patterns_by_significance(self, document_id: int = None, 
                                     pattern_types: List[str] = None) -> RankingResult:
        """
        Rank patterns by statistical significance and other criteria
        """
        try:
            # Get patterns to rank
            if document_id:
                patterns = self._get_document_patterns(document_id)
            else:
                patterns = self._get_all_patterns(pattern_types)
            
            if not patterns:
                return RankingResult(
                    ranked_patterns=[],
                    ranking_metadata={'message': 'No patterns found'},
                    summary_statistics={}
                )
            
            # Calculate significance scores for each pattern
            scored_patterns = []
            for pattern in patterns:
                score = self._calculate_pattern_significance_score(pattern)
                if score.significance_score >= self.min_significance_threshold:
                    scored_patterns.append((pattern, score))
            
            # Sort by significance score
            scored_patterns.sort(key=lambda x: x[1].significance_score, reverse=True)
            
            # Create ranked patterns with interpretations
            ranked_patterns = []
            for rank, (pattern, score) in enumerate(scored_patterns, 1):
                interpretation_suggestions = self._generate_interpretation_suggestions(pattern, score)
                evidence_strength = self._calculate_evidence_strength(pattern, score)
                priority = self._determine_recommendation_priority(score, evidence_strength)
                
                ranked_pattern = RankedPattern(
                    pattern=pattern,
                    score=score,
                    rank=rank,
                    interpretation_suggestions=interpretation_suggestions,
                    evidence_strength=evidence_strength,
                    recommendation_priority=priority
                )
                ranked_patterns.append(ranked_pattern)
            
            # Generate ranking metadata and summary statistics
            ranking_metadata = self._generate_ranking_metadata(ranked_patterns)
            summary_statistics = self._calculate_summary_statistics(ranked_patterns)
            
            return RankingResult(
                ranked_patterns=ranked_patterns,
                ranking_metadata=ranking_metadata,
                summary_statistics=summary_statistics
            )
            
        except Exception as e:
            self.logger.error(f"Error ranking patterns by significance: {str(e)}")
            return RankingResult(
                ranked_patterns=[],
                ranking_metadata={'error': str(e)},
                summary_statistics={}
            )
    
    def rank_anomalies_by_significance(self, document_id: int) -> RankingResult:
        """
        Rank detected anomalies by statistical significance
        """
        try:
            # Get anomalies from the anomaly detector
            anomaly_results = self.anomaly_detector.detect_all_anomalies(document_id)
            
            if 'error' in anomaly_results:
                return RankingResult(
                    ranked_patterns=[],
                    ranking_metadata={'error': anomaly_results['error']},
                    summary_statistics={}
                )
            
            # Collect all anomalies
            all_anomalies = []
            for anomaly_type, anomaly_list in anomaly_results.items():
                if anomaly_type != 'summary' and isinstance(anomaly_list, list):
                    all_anomalies.extend(anomaly_list)
            
            if not all_anomalies:
                return RankingResult(
                    ranked_patterns=[],
                    ranking_metadata={'message': 'No anomalies found'},
                    summary_statistics={}
                )
            
            # Calculate significance scores for anomalies
            scored_anomalies = []
            for anomaly in all_anomalies:
                score = self._calculate_anomaly_significance_score(anomaly)
                scored_anomalies.append((anomaly, score))
            
            # Sort by significance
            scored_anomalies.sort(key=lambda x: x[1].significance_score, reverse=True)
            
            # Create ranked anomalies
            ranked_anomalies = []
            for rank, (anomaly, score) in enumerate(scored_anomalies, 1):
                interpretation_suggestions = self._generate_anomaly_interpretations(anomaly, score)
                evidence_strength = self._calculate_anomaly_evidence_strength(anomaly, score)
                priority = self._determine_recommendation_priority(score, evidence_strength)
                
                ranked_anomaly = RankedPattern(
                    pattern=anomaly,
                    score=score,
                    rank=rank,
                    interpretation_suggestions=interpretation_suggestions,
                    evidence_strength=evidence_strength,
                    recommendation_priority=priority
                )
                ranked_anomalies.append(ranked_anomaly)
            
            # Generate metadata and statistics
            ranking_metadata = self._generate_anomaly_ranking_metadata(ranked_anomalies)
            summary_statistics = self._calculate_anomaly_summary_statistics(ranked_anomalies)
            
            return RankingResult(
                ranked_patterns=ranked_anomalies,
                ranking_metadata=ranking_metadata,
                summary_statistics=summary_statistics
            )
            
        except Exception as e:
            self.logger.error(f"Error ranking anomalies by significance: {str(e)}")
            return RankingResult(
                ranked_patterns=[],
                ranking_metadata={'error': str(e)},
                summary_statistics={}
            )
    
    def calculate_confidence_intervals(self, pattern_scores: List[float], 
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence intervals for pattern significance scores
        """
        try:
            if not pattern_scores or len(pattern_scores) < 2:
                return (0.0, 1.0)
            
            # Calculate sample statistics
            mean_score = np.mean(pattern_scores)
            std_score = np.std(pattern_scores, ddof=1)
            n = len(pattern_scores)
            
            # Calculate confidence interval using t-distribution
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
            margin_of_error = t_critical * (std_score / math.sqrt(n))
            
            lower_bound = max(0.0, mean_score - margin_of_error)
            upper_bound = min(1.0, mean_score + margin_of_error)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence intervals: {str(e)}")
            return (0.0, 1.0)
    
    def assess_pattern_reliability(self, pattern: Any) -> float:
        """
        Assess the reliability of a pattern based on multiple factors
        """
        try:
            reliability_factors = []
            
            # Factor 1: Consistency across instances
            if hasattr(pattern, 'consistency_score') and pattern.consistency_score:
                reliability_factors.append(pattern.consistency_score)
            
            # Factor 2: Cross-document presence
            if hasattr(pattern, 'document_count') and pattern.document_count:
                cross_doc_reliability = min(1.0, pattern.document_count / 5.0)  # Normalize to max 5 docs
                reliability_factors.append(cross_doc_reliability)
            
            # Factor 3: Statistical significance
            if hasattr(pattern, 'significance_score') and pattern.significance_score:
                stat_reliability = 1.0 - pattern.significance_score  # Lower p-value = higher reliability
                reliability_factors.append(stat_reliability)
            
            # Factor 4: Pattern complexity (more complex = potentially more reliable)
            if hasattr(pattern, 'pattern_complexity') and pattern.pattern_complexity:
                complexity_reliability = pattern.pattern_complexity
                reliability_factors.append(complexity_reliability)
            
            # Factor 5: Average confidence of instances
            if hasattr(pattern, 'average_confidence') and pattern.average_confidence:
                reliability_factors.append(pattern.average_confidence)
            
            # Calculate overall reliability
            if reliability_factors:
                return np.mean(reliability_factors)
            else:
                return 0.5  # Default reliability
                
        except Exception as e:
            self.logger.error(f"Error assessing pattern reliability: {str(e)}")
            return 0.5
    
    def suggest_pattern_interpretations(self, pattern: Any, 
                                      significance_score: float) -> List[str]:
        """
        Generate interpretation suggestions for a pattern based on its characteristics
        """
        try:
            suggestions = []
            
            # Get pattern type and characteristics
            pattern_type = getattr(pattern, 'pattern_type', 'unknown')
            pattern_subtype = getattr(pattern, 'pattern_subtype', None)
            complexity = getattr(pattern, 'pattern_complexity', 0)
            document_count = getattr(pattern, 'document_count', 1)
            
            # Base suggestions by pattern type
            if pattern_type == 'cipher':
                suggestions.extend(self._generate_cipher_interpretations(pattern, significance_score))
            elif pattern_type == 'geometric':
                suggestions.extend(self._generate_geometric_interpretations(pattern, significance_score))
            elif pattern_type == 'linguistic':
                suggestions.extend(self._generate_linguistic_interpretations(pattern, significance_score))
            elif pattern_type == 'structural':
                suggestions.extend(self._generate_structural_interpretations(pattern, significance_score))
            
            # Add significance-based suggestions
            if significance_score > 0.95:
                suggestions.append("Extremely high significance - priority investigation recommended")
            elif significance_score > 0.8:
                suggestions.append("High significance - detailed analysis warranted")
            elif significance_score > 0.6:
                suggestions.append("Moderate significance - consider for further study")
            
            # Add cross-document suggestions
            if document_count > 3:
                suggestions.append("Cross-document pattern - potential shared authorship or encoding scheme")
            
            # Add complexity-based suggestions
            if complexity > 0.8:
                suggestions.append("High complexity pattern - may indicate sophisticated encoding")
            
            return suggestions[:5]  # Limit to top 5 suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating pattern interpretations: {str(e)}")
            return ["Pattern analysis completed - manual review recommended"]
    
    # Private helper methods
    
    def _get_document_patterns(self, document_id: int) -> List[Any]:
        """Get all patterns for a specific document"""
        try:
            # Get cross-document patterns for this document
            instances = self.db.query(CrossPatternInstance).filter(
                CrossPatternInstance.document_id == document_id
            ).all()
            
            pattern_ids = [inst.cross_pattern_id for inst in instances]
            patterns = self.db.query(CrossDocumentPattern).filter(
                CrossDocumentPattern.id.in_(pattern_ids)
            ).all()
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error getting document patterns: {str(e)}")
            return []
    
    def _get_all_patterns(self, pattern_types: List[str] = None) -> List[Any]:
        """Get all patterns, optionally filtered by type"""
        try:
            query = self.db.query(CrossDocumentPattern)
            
            if pattern_types:
                query = query.filter(CrossDocumentPattern.pattern_type.in_(pattern_types))
            
            return query.all()
            
        except Exception as e:
            self.logger.error(f"Error getting all patterns: {str(e)}")
            return []
    
    def _calculate_pattern_significance_score(self, pattern: Any) -> PatternScore:
        """Calculate comprehensive significance score for a pattern"""
        try:
            # Initialize scoring components
            scoring_components = {}
            
            # Statistical significance component
            if hasattr(pattern, 'statistical_p_value') and pattern.statistical_p_value:
                stat_significance = 1.0 - pattern.statistical_p_value
                scoring_components['statistical_significance'] = stat_significance
            elif hasattr(pattern, 'significance_score') and pattern.significance_score:
                scoring_components['statistical_significance'] = pattern.significance_score
            else:
                scoring_components['statistical_significance'] = 0.5
            
            # Frequency rarity component
            if hasattr(pattern, 'rarity_score') and pattern.rarity_score:
                scoring_components['frequency_rarity'] = pattern.rarity_score
            else:
                scoring_components['frequency_rarity'] = 0.5
            
            # Cross-document presence component
            if hasattr(pattern, 'document_count') and pattern.document_count:
                cross_doc_score = min(1.0, pattern.document_count / 10.0)  # Normalize
                scoring_components['cross_document_presence'] = cross_doc_score
            else:
                scoring_components['cross_document_presence'] = 0.1
            
            # Complexity component
            if hasattr(pattern, 'pattern_complexity') and pattern.pattern_complexity:
                scoring_components['complexity_score'] = pattern.pattern_complexity
            else:
                scoring_components['complexity_score'] = 0.5
            
            # Consistency component
            if hasattr(pattern, 'consistency_score') and pattern.consistency_score:
                scoring_components['consistency_measure'] = pattern.consistency_score
            else:
                scoring_components['consistency_measure'] = 0.5
            
            # Calculate weighted significance score
            significance_score = sum(
                score * self.ranking_weights.get(component, 0.2)
                for component, score in scoring_components.items()
            )
            
            # Calculate confidence interval
            component_scores = list(scoring_components.values())
            confidence_interval = self.calculate_confidence_intervals(component_scores, self.confidence_level)
            
            # Calculate reliability measure
            reliability_measure = self.assess_pattern_reliability(pattern)
            
            # Prepare statistical measures
            statistical_measures = {
                'p_value': getattr(pattern, 'statistical_p_value', None),
                'chi_square': getattr(pattern, 'chi_square_statistic', None),
                'z_score': getattr(pattern, 'z_score', None),
                'effect_size': getattr(pattern, 'effect_size', None)
            }
            
            return PatternScore(
                pattern_id=getattr(pattern, 'id', 0),
                pattern_type=getattr(pattern, 'pattern_type', 'unknown'),
                significance_score=significance_score,
                confidence_interval=confidence_interval,
                reliability_measure=reliability_measure,
                statistical_measures=statistical_measures,
                ranking_factors=scoring_components
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern significance score: {str(e)}")
            return PatternScore(
                pattern_id=0,
                pattern_type='unknown',
                significance_score=0.0,
                confidence_interval=(0.0, 1.0),
                reliability_measure=0.0,
                statistical_measures={},
                ranking_factors={}
            ) 
   
    def _calculate_anomaly_significance_score(self, anomaly: Any) -> PatternScore:
        """Calculate significance score for an anomaly"""
        try:
            # Extract anomaly characteristics
            if hasattr(anomaly, 'deviation_score'):
                deviation = anomaly.deviation_score
            elif hasattr(anomaly, 'anomaly_score'):
                deviation = anomaly.anomaly_score
            else:
                deviation = 1.0
            
            # Statistical significance
            if hasattr(anomaly, 'significance_level'):
                stat_significance = 1.0 - anomaly.significance_level
            elif hasattr(anomaly, 'statistical_significance'):
                stat_significance = 1.0 - anomaly.statistical_significance
            else:
                stat_significance = 0.5
            
            # Normalize deviation score to 0-1 range
            normalized_deviation = min(1.0, deviation / 5.0)
            
            # Calculate overall significance
            significance_score = (stat_significance * 0.6 + normalized_deviation * 0.4)
            
            # Simple confidence interval based on score
            margin = 0.1 * (1 - significance_score)
            confidence_interval = (
                max(0.0, significance_score - margin),
                min(1.0, significance_score + margin)
            )
            
            # Reliability based on consistency of the anomaly
            reliability_measure = min(1.0, significance_score + 0.2)
            
            return PatternScore(
                pattern_id=getattr(anomaly, 'id', 0),
                pattern_type=getattr(anomaly, 'anomaly_type', 'anomaly'),
                significance_score=significance_score,
                confidence_interval=confidence_interval,
                reliability_measure=reliability_measure,
                statistical_measures={
                    'deviation_score': deviation,
                    'statistical_significance': stat_significance
                },
                ranking_factors={
                    'deviation_magnitude': normalized_deviation,
                    'statistical_significance': stat_significance
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating anomaly significance score: {str(e)}")
            return PatternScore(
                pattern_id=0,
                pattern_type='anomaly',
                significance_score=0.0,
                confidence_interval=(0.0, 1.0),
                reliability_measure=0.0,
                statistical_measures={},
                ranking_factors={}
            )
    
    def _generate_interpretation_suggestions(self, pattern: Any, score: PatternScore) -> List[str]:
        """Generate interpretation suggestions for a pattern"""
        return self.suggest_pattern_interpretations(pattern, score.significance_score)
    
    def _generate_anomaly_interpretations(self, anomaly: Any, score: PatternScore) -> List[str]:
        """Generate interpretation suggestions for an anomaly"""
        try:
            suggestions = []
            
            anomaly_type = getattr(anomaly, 'anomaly_type', 'unknown')
            
            # Type-specific suggestions
            if 'character' in anomaly_type.lower():
                suggestions.append("Character distribution anomaly - potential cipher or encoding")
                if score.significance_score > 0.8:
                    suggestions.append("Strong character pattern - investigate substitution ciphers")
            elif 'word' in anomaly_type.lower():
                suggestions.append("Word pattern anomaly - unusual placement or frequency")
                if score.significance_score > 0.8:
                    suggestions.append("Significant word pattern - check for skip ciphers or acrostics")
            elif 'punctuation' in anomaly_type.lower():
                suggestions.append("Punctuation anomaly - size or positioning irregularity")
                if score.significance_score > 0.7:
                    suggestions.append("Punctuation pattern may indicate steganographic encoding")
            
            # General significance suggestions
            if score.significance_score > 0.9:
                suggestions.append("Extremely significant anomaly - immediate investigation recommended")
            elif score.significance_score > 0.7:
                suggestions.append("Highly significant anomaly - detailed analysis warranted")
            
            return suggestions[:4]  # Limit suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating anomaly interpretations: {str(e)}")
            return ["Anomaly detected - manual review recommended"]
    
    def _calculate_evidence_strength(self, pattern: Any, score: PatternScore) -> float:
        """Calculate evidence strength for a pattern"""
        try:
            evidence_factors = []
            
            # Significance score contributes to evidence
            evidence_factors.append(score.significance_score)
            
            # Reliability contributes to evidence
            evidence_factors.append(score.reliability_measure)
            
            # Cross-document presence strengthens evidence
            if hasattr(pattern, 'document_count') and pattern.document_count:
                cross_doc_strength = min(1.0, pattern.document_count / 5.0)
                evidence_factors.append(cross_doc_strength)
            
            # Pattern complexity can strengthen evidence
            if hasattr(pattern, 'pattern_complexity') and pattern.pattern_complexity:
                evidence_factors.append(pattern.pattern_complexity)
            
            return np.mean(evidence_factors) if evidence_factors else 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating evidence strength: {str(e)}")
            return 0.5
    
    def _calculate_anomaly_evidence_strength(self, anomaly: Any, score: PatternScore) -> float:
        """Calculate evidence strength for an anomaly"""
        try:
            # Base evidence on significance and deviation
            base_evidence = score.significance_score
            
            # Add deviation magnitude
            if hasattr(anomaly, 'deviation_score'):
                deviation_strength = min(1.0, anomaly.deviation_score / 3.0)
                base_evidence = (base_evidence + deviation_strength) / 2
            
            return base_evidence
            
        except Exception as e:
            self.logger.error(f"Error calculating anomaly evidence strength: {str(e)}")
            return 0.5
    
    def _determine_recommendation_priority(self, score: PatternScore, evidence_strength: float) -> str:
        """Determine recommendation priority based on score and evidence"""
        try:
            combined_score = (score.significance_score + evidence_strength) / 2
            
            if combined_score >= 0.9:
                return "critical"
            elif combined_score >= 0.7:
                return "high"
            elif combined_score >= 0.5:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            self.logger.error(f"Error determining recommendation priority: {str(e)}")
            return "medium"
    
    def _generate_ranking_metadata(self, ranked_patterns: List[RankedPattern]) -> Dict[str, Any]:
        """Generate metadata for pattern ranking results"""
        try:
            if not ranked_patterns:
                return {'message': 'No patterns to rank'}
            
            metadata = {
                'total_patterns': len(ranked_patterns),
                'ranking_method': 'statistical_significance_weighted',
                'confidence_level': self.confidence_level,
                'ranking_weights': self.ranking_weights,
                'priority_distribution': Counter(p.recommendation_priority for p in ranked_patterns),
                'pattern_type_distribution': Counter(p.score.pattern_type for p in ranked_patterns),
                'top_pattern': {
                    'rank': 1,
                    'pattern_type': ranked_patterns[0].score.pattern_type,
                    'significance_score': ranked_patterns[0].score.significance_score,
                    'priority': ranked_patterns[0].recommendation_priority
                } if ranked_patterns else None
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error generating ranking metadata: {str(e)}")
            return {'error': str(e)}
    
    def _generate_anomaly_ranking_metadata(self, ranked_anomalies: List[RankedPattern]) -> Dict[str, Any]:
        """Generate metadata for anomaly ranking results"""
        try:
            if not ranked_anomalies:
                return {'message': 'No anomalies to rank'}
            
            metadata = {
                'total_anomalies': len(ranked_anomalies),
                'ranking_method': 'anomaly_significance',
                'priority_distribution': Counter(a.recommendation_priority for a in ranked_anomalies),
                'anomaly_type_distribution': Counter(a.score.pattern_type for a in ranked_anomalies),
                'highest_significance': max(a.score.significance_score for a in ranked_anomalies),
                'average_significance': np.mean([a.score.significance_score for a in ranked_anomalies])
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error generating anomaly ranking metadata: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_summary_statistics(self, ranked_patterns: List[RankedPattern]) -> Dict[str, float]:
        """Calculate summary statistics for ranked patterns"""
        try:
            if not ranked_patterns:
                return {}
            
            significance_scores = [p.score.significance_score for p in ranked_patterns]
            evidence_strengths = [p.evidence_strength for p in ranked_patterns]
            reliability_measures = [p.score.reliability_measure for p in ranked_patterns]
            
            return {
                'mean_significance': np.mean(significance_scores),
                'median_significance': np.median(significance_scores),
                'std_significance': np.std(significance_scores),
                'mean_evidence_strength': np.mean(evidence_strengths),
                'mean_reliability': np.mean(reliability_measures),
                'high_priority_percentage': len([p for p in ranked_patterns if p.recommendation_priority in ['critical', 'high']]) / len(ranked_patterns) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating summary statistics: {str(e)}")
            return {}
    
    def _calculate_anomaly_summary_statistics(self, ranked_anomalies: List[RankedPattern]) -> Dict[str, float]:
        """Calculate summary statistics for ranked anomalies"""
        try:
            if not ranked_anomalies:
                return {}
            
            significance_scores = [a.score.significance_score for a in ranked_anomalies]
            evidence_strengths = [a.evidence_strength for a in ranked_anomalies]
            
            return {
                'mean_significance': np.mean(significance_scores),
                'median_significance': np.median(significance_scores),
                'std_significance': np.std(significance_scores),
                'mean_evidence_strength': np.mean(evidence_strengths),
                'critical_anomaly_count': len([a for a in ranked_anomalies if a.recommendation_priority == 'critical']),
                'high_priority_percentage': len([a for a in ranked_anomalies if a.recommendation_priority in ['critical', 'high']]) / len(ranked_anomalies) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating anomaly summary statistics: {str(e)}")
            return {}
    
    # Pattern-specific interpretation methods
    
    def _generate_cipher_interpretations(self, pattern: Any, significance_score: float) -> List[str]:
        """Generate cipher-specific interpretation suggestions"""
        try:
            suggestions = []
            pattern_subtype = getattr(pattern, 'pattern_subtype', None)
            
            if pattern_subtype == 'caesar':
                suggestions.append("Caesar cipher detected - try shift values 1-25")
                if significance_score > 0.8:
                    suggestions.append("Strong Caesar cipher pattern - high confidence in shift cipher")
            elif pattern_subtype == 'substitution':
                suggestions.append("Substitution cipher detected - analyze frequency patterns")
                if significance_score > 0.8:
                    suggestions.append("Clear substitution pattern - attempt frequency analysis")
            elif pattern_subtype == 'skip':
                suggestions.append("Skip cipher detected - examine character intervals")
                if significance_score > 0.8:
                    suggestions.append("Strong skip pattern - investigate regular intervals")
            elif pattern_subtype == 'vigenere':
                suggestions.append("Vigenère cipher detected - determine key length")
                if significance_score > 0.8:
                    suggestions.append("Clear polyalphabetic pattern - use Kasiski examination")
            else:
                suggestions.append("Cipher pattern detected - requires detailed cryptanalysis")
                if significance_score > 0.8:
                    suggestions.append("Strong cipher indicators - priority cryptographic analysis")
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating cipher interpretations: {str(e)}")
            return ["Cipher pattern detected - manual analysis recommended"]
    
    def _generate_geometric_interpretations(self, pattern: Any, significance_score: float) -> List[str]:
        """Generate geometric-specific interpretation suggestions"""
        try:
            suggestions = []
            pattern_subtype = getattr(pattern, 'pattern_subtype', None)
            
            if pattern_subtype == 'triangle':
                suggestions.append("Triangle geometric pattern - examine angles and proportions")
                if significance_score > 0.8:
                    suggestions.append("Strong triangle arrangement - check for sacred geometry")
            elif pattern_subtype == 'circle':
                suggestions.append("Circle geometric pattern - analyze radius and center points")
                if significance_score > 0.8:
                    suggestions.append("Clear circle arrangement - investigate astronomical alignments")
            elif pattern_subtype == 'golden_ratio':
                suggestions.append("Golden ratio pattern detected - examine proportional relationships")
                if significance_score > 0.8:
                    suggestions.append("Strong golden ratio evidence - investigate divine proportion usage")
            elif pattern_subtype == 'spiral':
                suggestions.append("Spiral pattern detected - analyze growth ratios")
                if significance_score > 0.8:
                    suggestions.append("Clear spiral arrangement - examine Fibonacci relationships")
            else:
                suggestions.append("Geometric pattern detected - analyze spatial relationships")
                if significance_score > 0.8:
                    suggestions.append("Significant geometric arrangement - detailed measurement recommended")
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating geometric interpretations: {str(e)}")
            return ["Geometric pattern detected - spatial analysis recommended"]
    
    def _generate_linguistic_interpretations(self, pattern: Any, significance_score: float) -> List[str]:
        """Generate linguistic-specific interpretation suggestions"""
        try:
            suggestions = []
            pattern_subtype = getattr(pattern, 'pattern_subtype', None)
            
            if pattern_subtype == 'etymology':
                suggestions.append("Etymology pattern detected - examine word origins and relationships")
                if significance_score > 0.8:
                    suggestions.append("Strong etymological connections - investigate historical linguistics")
            elif pattern_subtype == 'acrostic':
                suggestions.append("Acrostic pattern detected - examine first letters of lines/words")
                if significance_score > 0.8:
                    suggestions.append("Clear acrostic structure - decode hidden message")
            elif pattern_subtype == 'anagram':
                suggestions.append("Anagram pattern detected - rearrange letters for hidden meaning")
                if significance_score > 0.8:
                    suggestions.append("Strong anagram evidence - systematic letter rearrangement needed")
            elif pattern_subtype == 'wordplay':
                suggestions.append("Wordplay pattern detected - examine puns and double meanings")
                if significance_score > 0.8:
                    suggestions.append("Significant wordplay - investigate multiple interpretation layers")
            else:
                suggestions.append("Linguistic pattern detected - examine language structure")
                if significance_score > 0.8:
                    suggestions.append("Strong linguistic anomaly - detailed philological analysis needed")
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating linguistic interpretations: {str(e)}")
            return ["Linguistic pattern detected - language analysis recommended"]
    
    def _generate_structural_interpretations(self, pattern: Any, significance_score: float) -> List[str]:
        """Generate structural-specific interpretation suggestions"""
        try:
            suggestions = []
            pattern_subtype = getattr(pattern, 'pattern_subtype', None)
            
            if pattern_subtype == 'grid':
                suggestions.append("Grid structure pattern detected - examine row/column relationships")
                if significance_score > 0.8:
                    suggestions.append("Strong grid pattern - investigate matrix-based encoding")
            elif pattern_subtype == 'sequence':
                suggestions.append("Sequence pattern detected - analyze ordering and progression")
                if significance_score > 0.8:
                    suggestions.append("Clear sequential structure - examine mathematical progressions")
            elif pattern_subtype == 'symmetry':
                suggestions.append("Symmetry pattern detected - examine mirror and rotational properties")
                if significance_score > 0.8:
                    suggestions.append("Strong symmetrical arrangement - investigate intentional design")
            elif pattern_subtype == 'repetition':
                suggestions.append("Repetition pattern detected - analyze recurring elements")
                if significance_score > 0.8:
                    suggestions.append("Significant repetitive structure - examine cyclical encoding")
            else:
                suggestions.append("Structural pattern detected - examine organizational principles")
                if significance_score > 0.8:
                    suggestions.append("Strong structural anomaly - detailed architectural analysis needed")
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating structural interpretations: {str(e)}")
            return ["Structural pattern detected - organizational analysis recommended"]
    
    def _calculate_anomaly_summary_statistics(self, ranked_anomalies: List[RankedPattern]) -> Dict[str, float]:
        """Calculate summary statistics for ranked anomalies"""
        return self._calculate_summary_statistics(ranked_anomalies)
    
    # Pattern-specific interpretation methods
    
    def _generate_cipher_interpretations(self, pattern: Any, significance_score: float) -> List[str]:
        """Generate cipher-specific interpretation suggestions"""
        suggestions = []
        
        pattern_subtype = getattr(pattern, 'pattern_subtype', None)
        
        if pattern_subtype == 'caesar':
            suggestions.append("Caesar cipher pattern detected - check for letter shift encoding")
        elif pattern_subtype == 'substitution':
            suggestions.append("Substitution cipher pattern - analyze character mapping")
        elif pattern_subtype == 'skip':
            suggestions.append("Skip cipher pattern - investigate interval-based encoding")
        else:
            suggestions.append("Cipher pattern detected - apply cryptographic analysis methods")
        
        if significance_score > 0.8:
            suggestions.append("High confidence cipher detection - priority decryption candidate")
        
        return suggestions
    

    
    def _generate_linguistic_interpretations(self, pattern: Any, significance_score: float) -> List[str]:
        """Generate linguistic-specific interpretation suggestions"""
        suggestions = []
        
        suggestions.append("Linguistic pattern detected - analyze word origins and relationships")
        
        if significance_score > 0.8:
            suggestions.append("Strong linguistic correlation - investigate etymology connections")
        
        return suggestions
    
    def _generate_structural_interpretations(self, pattern: Any, significance_score: float) -> List[str]:
        """Generate structural-specific interpretation suggestions"""
        suggestions = []
        
        suggestions.append("Structural pattern detected - analyze document organization")
        
        if significance_score > 0.7:
            suggestions.append("Significant structural pattern - investigate layout encoding")
        
        return suggestions