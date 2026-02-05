"""
Document Relationship Analysis Service
Implements correlation matrix generation, evidence trail tracking, authorship pattern analysis,
and shared cryptographic knowledge detection across multiple documents.
"""
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import math
import numpy as np
from scipy.stats import pearsonr, chi2_contingency, spearmanr
from scipy.spatial.distance import cosine, euclidean
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import networkx as nx
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
import logging
from datetime import datetime, timedelta

from app.models.cross_document_models import (
    CrossDocumentPattern, CrossPatternInstance, PatternRelationship,
    DocumentCluster, DocumentClusterMembership, SharedConstruction
)
from app.models.database_models import Document, Pattern, Page
from app.services.cross_document_pattern_database import CrossDocumentPatternDatabase
from app.core.database import get_db


@dataclass
class DocumentRelationship:
    """Represents a relationship between two documents"""
    document1_id: int
    document2_id: int
    relationship_type: str
    strength: float
    confidence: float
    evidence_count: int
    shared_patterns: List[str]
    temporal_correlation: float
    authorship_similarity: float


@dataclass
class CorrelationMatrix:
    """Represents a correlation matrix between documents"""
    document_ids: List[int]
    matrix: np.ndarray
    correlation_types: Dict[str, np.ndarray]
    significance_matrix: np.ndarray
    cluster_assignments: Dict[int, int]


@dataclass
class EvidenceTrail:
    """Represents an evidence trail for pattern connections"""
    pattern_id: int
    document_chain: List[int]
    connection_strength: float
    evidence_points: List[Dict[str, Any]]
    temporal_sequence: List[datetime]
    confidence_score: float


@dataclass
class AuthorshipProfile:
    """Represents an authorship pattern profile"""
    document_id: int
    stylistic_features: Dict[str, float]
    cipher_preferences: Dict[str, float]
    complexity_signature: Dict[str, float]
    temporal_patterns: Dict[str, Any]
    geometric_preferences: Dict[str, float]


class RelationshipAnalyzer:
    """
    Service for analyzing relationships between documents based on shared patterns,
    authorship analysis, and cryptographic knowledge detection.
    """
    
    def __init__(self, db_session: Session = None):
        self.db = db_session or next(get_db())
        self.pattern_db = CrossDocumentPatternDatabase(self.db)
        self.logger = logging.getLogger(__name__)
        
        # Analysis thresholds
        self.min_relationship_strength = 0.3
        self.min_evidence_count = 3
        self.significance_threshold = 0.05
        self.authorship_similarity_threshold = 0.6
        
        # Weighting factors for different relationship types
        self.relationship_weights = {
            'pattern_based': 0.4,
            'stylistic': 0.25,
            'temporal': 0.15,
            'geometric': 0.1,
            'cipher_method': 0.1
        }
    
    def generate_correlation_matrix(self, document_ids: List[int]) -> CorrelationMatrix:
        """
        Generate a comprehensive correlation matrix for document relationships
        """
        try:
            if len(document_ids) < 2:
                raise ValueError("Need at least 2 documents for correlation analysis")
            
            n_docs = len(document_ids)
            
            # Initialize matrices
            overall_matrix = np.zeros((n_docs, n_docs))
            pattern_matrix = np.zeros((n_docs, n_docs))
            stylistic_matrix = np.zeros((n_docs, n_docs))
            temporal_matrix = np.zeros((n_docs, n_docs))
            geometric_matrix = np.zeros((n_docs, n_docs))
            cipher_matrix = np.zeros((n_docs, n_docs))
            significance_matrix = np.zeros((n_docs, n_docs))
            
            # Fill diagonal with perfect correlation
            np.fill_diagonal(overall_matrix, 1.0)
            np.fill_diagonal(pattern_matrix, 1.0)
            np.fill_diagonal(stylistic_matrix, 1.0)
            np.fill_diagonal(temporal_matrix, 1.0)
            np.fill_diagonal(geometric_matrix, 1.0)
            np.fill_diagonal(cipher_matrix, 1.0)
            
            # Calculate pairwise correlations
            for i in range(n_docs):
                for j in range(i + 1, n_docs):
                    doc1_id, doc2_id = document_ids[i], document_ids[j]
                    
                    # Calculate different types of correlations
                    pattern_corr = self._calculate_pattern_correlation(doc1_id, doc2_id)
                    stylistic_corr = self._calculate_stylistic_correlation(doc1_id, doc2_id)
                    temporal_corr = self._calculate_temporal_correlation(doc1_id, doc2_id)
                    geometric_corr = self._calculate_geometric_correlation(doc1_id, doc2_id)
                    cipher_corr = self._calculate_cipher_method_correlation(doc1_id, doc2_id)
                    
                    # Store individual correlations
                    pattern_matrix[i, j] = pattern_matrix[j, i] = pattern_corr
                    stylistic_matrix[i, j] = stylistic_matrix[j, i] = stylistic_corr
                    temporal_matrix[i, j] = temporal_matrix[j, i] = temporal_corr
                    geometric_matrix[i, j] = geometric_matrix[j, i] = geometric_corr
                    cipher_matrix[i, j] = cipher_matrix[j, i] = cipher_corr
                    
                    # Calculate overall correlation
                    overall_corr = (
                        pattern_corr * self.relationship_weights['pattern_based'] +
                        stylistic_corr * self.relationship_weights['stylistic'] +
                        temporal_corr * self.relationship_weights['temporal'] +
                        geometric_corr * self.relationship_weights['geometric'] +
                        cipher_corr * self.relationship_weights['cipher_method']
                    )
                    
                    overall_matrix[i, j] = overall_matrix[j, i] = overall_corr
                    
                    # Calculate significance
                    significance = self._calculate_correlation_significance(
                        doc1_id, doc2_id, overall_corr
                    )
                    significance_matrix[i, j] = significance_matrix[j, i] = significance
            
            # Perform clustering
            cluster_assignments = self._cluster_documents_by_correlation(
                overall_matrix, document_ids
            )
            
            correlation_types = {
                'pattern_based': pattern_matrix,
                'stylistic': stylistic_matrix,
                'temporal': temporal_matrix,
                'geometric': geometric_matrix,
                'cipher_method': cipher_matrix
            }
            
            self.logger.info(f"Generated correlation matrix for {n_docs} documents")
            
            return CorrelationMatrix(
                document_ids=document_ids,
                matrix=overall_matrix,
                correlation_types=correlation_types,
                significance_matrix=significance_matrix,
                cluster_assignments=cluster_assignments
            )
            
        except Exception as e:
            self.logger.error(f"Error generating correlation matrix: {str(e)}")
            raise
    
    def track_evidence_trails(self, pattern_id: int) -> List[EvidenceTrail]:
        """
        Track evidence trails for pattern connections across documents
        """
        try:
            # Get pattern and its instances
            pattern = self.db.query(CrossDocumentPattern).get(pattern_id)
            if not pattern:
                return []
            
            instances = self.db.query(CrossPatternInstance).filter(
                CrossPatternInstance.cross_pattern_id == pattern_id
            ).all()
            
            if len(instances) < 2:
                return []
            
            # Group instances by document
            doc_instances = defaultdict(list)
            for instance in instances:
                doc_instances[instance.document_id].append(instance)
            
            document_ids = list(doc_instances.keys())
            trails = []
            
            # Create evidence trails for each document chain
            for i in range(len(document_ids)):
                for j in range(i + 1, len(document_ids)):
                    doc1_id, doc2_id = document_ids[i], document_ids[j]
                    
                    # Calculate connection strength
                    connection_strength = self._calculate_connection_strength(
                        doc_instances[doc1_id], doc_instances[doc2_id]
                    )
                    
                    if connection_strength >= 0.3:
                        # Build evidence points
                        evidence_points = self._build_evidence_points(
                            doc_instances[doc1_id], doc_instances[doc2_id]
                        )
                        
                        # Get temporal sequence
                        temporal_sequence = self._extract_temporal_sequence(
                            doc_instances[doc1_id] + doc_instances[doc2_id]
                        )
                        
                        # Calculate confidence
                        confidence_score = self._calculate_trail_confidence(
                            evidence_points, connection_strength
                        )
                        
                        trail = EvidenceTrail(
                            pattern_id=pattern_id,
                            document_chain=[doc1_id, doc2_id],
                            connection_strength=connection_strength,
                            evidence_points=evidence_points,
                            temporal_sequence=temporal_sequence,
                            confidence_score=confidence_score
                        )
                        trails.append(trail)
            
            # Extend trails to multi-document chains
            extended_trails = self._extend_evidence_trails(trails, doc_instances)
            trails.extend(extended_trails)
            
            # Sort by confidence and connection strength
            trails.sort(key=lambda x: (x.confidence_score, x.connection_strength), reverse=True)
            
            self.logger.info(f"Generated {len(trails)} evidence trails for pattern {pattern_id}")
            return trails
            
        except Exception as e:
            self.logger.error(f"Error tracking evidence trails: {str(e)}")
            return []
    
    def analyze_authorship_patterns(self, document_ids: List[int]) -> Dict[str, Any]:
        """
        Analyze authorship patterns across multiple documents
        """
        try:
            if len(document_ids) < 2:
                return {'error': 'Need at least 2 documents for authorship analysis'}
            
            # Create authorship profiles for each document
            profiles = []
            for doc_id in document_ids:
                profile = self._create_authorship_profile(doc_id)
                if profile:
                    profiles.append(profile)
            
            if len(profiles) < 2:
                return {'error': 'Could not create sufficient authorship profiles'}
            
            # Calculate authorship similarities
            similarity_matrix = self._calculate_authorship_similarity_matrix(profiles)
            
            # Identify potential author groups
            author_groups = self._identify_author_groups(profiles, similarity_matrix)
            
            # Analyze stylistic evolution
            stylistic_evolution = self._analyze_stylistic_evolution(profiles)
            
            # Detect shared knowledge indicators
            shared_knowledge = self._detect_shared_knowledge_indicators(profiles)
            
            # Calculate authorship confidence scores
            confidence_scores = self._calculate_authorship_confidence_scores(
                profiles, similarity_matrix
            )
            
            return {
                'document_count': len(document_ids),
                'authorship_profiles': profiles,
                'similarity_matrix': similarity_matrix.tolist(),
                'author_groups': author_groups,
                'stylistic_evolution': stylistic_evolution,
                'shared_knowledge_indicators': shared_knowledge,
                'confidence_scores': confidence_scores,
                'analysis_summary': self._generate_authorship_summary(
                    author_groups, shared_knowledge
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing authorship patterns: {str(e)}")
            return {'error': str(e)}
    
    def detect_shared_cryptographic_knowledge(self, document_ids: List[int]) -> Dict[str, Any]:
        """
        Detect evidence of shared cryptographic knowledge across documents
        """
        try:
            # Get sophisticated patterns from each document
            sophisticated_patterns = self._identify_sophisticated_patterns(document_ids)
            
            # Analyze cipher method progression
            cipher_progression = self._analyze_cipher_method_progression(document_ids)
            
            # Detect mathematical knowledge indicators
            mathematical_indicators = self._detect_mathematical_knowledge_indicators(document_ids)
            
            # Analyze pattern complexity evolution
            complexity_evolution = self._analyze_pattern_complexity_evolution(document_ids)
            
            # Calculate knowledge sharing probability
            knowledge_sharing_prob = self._calculate_knowledge_sharing_probability(
                sophisticated_patterns, cipher_progression, mathematical_indicators
            )
            
            # Identify knowledge transfer patterns
            transfer_patterns = self._identify_knowledge_transfer_patterns(
                document_ids, sophisticated_patterns
            )
            
            return {
                'document_count': len(document_ids),
                'sophisticated_patterns': sophisticated_patterns,
                'cipher_method_progression': cipher_progression,
                'mathematical_knowledge_indicators': mathematical_indicators,
                'complexity_evolution': complexity_evolution,
                'knowledge_sharing_probability': knowledge_sharing_prob,
                'knowledge_transfer_patterns': transfer_patterns,
                'evidence_strength': self._calculate_evidence_strength(
                    sophisticated_patterns, mathematical_indicators
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting shared cryptographic knowledge: {str(e)}")
            return {'error': str(e)}
    
    def build_relationship_network(self, document_ids: List[int]) -> Dict[str, Any]:
        """
        Build a network graph of document relationships
        """
        try:
            # Create network graph
            G = nx.Graph()
            
            # Add nodes (documents)
            for doc_id in document_ids:
                doc = self.db.query(Document).get(doc_id)
                G.add_node(doc_id, 
                          title=doc.title if doc else f"Document {doc_id}",
                          type='document')
            
            # Add edges (relationships)
            for i in range(len(document_ids)):
                for j in range(i + 1, len(document_ids)):
                    doc1_id, doc2_id = document_ids[i], document_ids[j]
                    
                    # Calculate relationship strength
                    relationship = self._calculate_document_relationship(doc1_id, doc2_id)
                    
                    if relationship.strength >= self.min_relationship_strength:
                        G.add_edge(doc1_id, doc2_id,
                                 weight=relationship.strength,
                                 relationship_type=relationship.relationship_type,
                                 evidence_count=relationship.evidence_count,
                                 confidence=relationship.confidence)
            
            # Calculate network metrics
            network_metrics = self._calculate_network_metrics(G)
            
            # Identify communities
            communities = self._identify_communities(G)
            
            # Find central documents
            central_documents = self._identify_central_documents(G)
            
            # Convert to serializable format
            network_data = {
                'nodes': [
                    {
                        'id': node,
                        'title': G.nodes[node].get('title', f'Document {node}'),
                        'centrality': network_metrics['centrality'].get(node, 0),
                        'community': communities.get(node, 0)
                    }
                    for node in G.nodes()
                ],
                'edges': [
                    {
                        'source': edge[0],
                        'target': edge[1],
                        'weight': G.edges[edge]['weight'],
                        'relationship_type': G.edges[edge]['relationship_type'],
                        'evidence_count': G.edges[edge]['evidence_count'],
                        'confidence': G.edges[edge]['confidence']
                    }
                    for edge in G.edges()
                ]
            }
            
            return {
                'network': network_data,
                'metrics': network_metrics,
                'communities': communities,
                'central_documents': central_documents,
                'summary': {
                    'total_documents': len(document_ids),
                    'total_relationships': len(G.edges()),
                    'average_relationship_strength': np.mean([
                        G.edges[edge]['weight'] for edge in G.edges()
                    ]) if G.edges() else 0,
                    'community_count': len(set(communities.values()))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error building relationship network: {str(e)}")
            return {'error': str(e)}
    
    # Private helper methods
    
    def _calculate_pattern_correlation(self, doc1_id: int, doc2_id: int) -> float:
        """Calculate correlation based on shared patterns"""
        try:
            # Get patterns for both documents
            patterns1 = self._get_document_pattern_signatures(doc1_id)
            patterns2 = self._get_document_pattern_signatures(doc2_id)
            
            if not patterns1 or not patterns2:
                return 0.0
            
            # Calculate Jaccard similarity
            set1 = set(patterns1.keys())
            set2 = set(patterns2.keys())
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            if union == 0:
                return 0.0
            
            jaccard_similarity = intersection / union
            
            # Weight by pattern significance
            weighted_similarity = 0.0
            total_weight = 0.0
            
            for pattern in set1 & set2:
                weight = (patterns1[pattern] + patterns2[pattern]) / 2
                weighted_similarity += weight * jaccard_similarity
                total_weight += weight
            
            return weighted_similarity / total_weight if total_weight > 0 else jaccard_similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern correlation: {str(e)}")
            return 0.0
    
    def _calculate_stylistic_correlation(self, doc1_id: int, doc2_id: int) -> float:
        """Calculate correlation based on stylistic features"""
        try:
            # Get stylistic features for both documents
            features1 = self._extract_stylistic_features(doc1_id)
            features2 = self._extract_stylistic_features(doc2_id)
            
            if not features1 or not features2:
                return 0.0
            
            # Create feature vectors
            all_features = set(features1.keys()) | set(features2.keys())
            vec1 = [features1.get(feature, 0) for feature in all_features]
            vec2 = [features2.get(feature, 0) for feature in all_features]
            
            # Calculate cosine similarity
            try:
                correlation = 1 - cosine(vec1, vec2)
                return max(0.0, correlation)
            except:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating stylistic correlation: {str(e)}")
            return 0.0
    
    def _calculate_temporal_correlation(self, doc1_id: int, doc2_id: int) -> float:
        """Calculate correlation based on temporal patterns"""
        try:
            # Get temporal features for both documents
            temporal1 = self._extract_temporal_features(doc1_id)
            temporal2 = self._extract_temporal_features(doc2_id)
            
            if not temporal1 or not temporal2:
                return 0.0
            
            # Calculate temporal proximity
            time_diff = abs(temporal1.get('creation_time', 0) - temporal2.get('creation_time', 0))
            max_time_diff = 365 * 24 * 3600  # 1 year in seconds
            
            temporal_proximity = max(0.0, 1.0 - (time_diff / max_time_diff))
            
            # Calculate pattern timing correlation
            pattern_timing_corr = self._calculate_pattern_timing_correlation(
                temporal1.get('pattern_timings', []),
                temporal2.get('pattern_timings', [])
            )
            
            return (temporal_proximity + pattern_timing_corr) / 2
            
        except Exception as e:
            self.logger.error(f"Error calculating temporal correlation: {str(e)}")
            return 0.0
    
    def _calculate_geometric_correlation(self, doc1_id: int, doc2_id: int) -> float:
        """Calculate correlation based on geometric patterns"""
        try:
            # Get geometric patterns for both documents
            geo1 = self._extract_geometric_features(doc1_id)
            geo2 = self._extract_geometric_features(doc2_id)
            
            if not geo1 or not geo2:
                return 0.0
            
            # Compare geometric pattern types
            types1 = set(geo1.get('pattern_types', []))
            types2 = set(geo2.get('pattern_types', []))
            
            type_similarity = len(types1 & types2) / len(types1 | types2) if (types1 | types2) else 0
            
            # Compare mathematical constants
            constants1 = set(geo1.get('mathematical_constants', []))
            constants2 = set(geo2.get('mathematical_constants', []))
            
            constant_similarity = len(constants1 & constants2) / len(constants1 | constants2) if (constants1 | constants2) else 0
            
            return (type_similarity + constant_similarity) / 2
            
        except Exception as e:
            self.logger.error(f"Error calculating geometric correlation: {str(e)}")
            return 0.0
    
    def _calculate_cipher_method_correlation(self, doc1_id: int, doc2_id: int) -> float:
        """Calculate correlation based on cipher methods"""
        try:
            # Get cipher methods for both documents
            ciphers1 = self._extract_cipher_methods(doc1_id)
            ciphers2 = self._extract_cipher_methods(doc2_id)
            
            if not ciphers1 or not ciphers2:
                return 0.0
            
            # Compare cipher types
            types1 = Counter(ciphers1.get('cipher_types', []))
            types2 = Counter(ciphers2.get('cipher_types', []))
            
            # Calculate weighted Jaccard similarity
            all_types = set(types1.keys()) | set(types2.keys())
            intersection_weight = sum(min(types1.get(t, 0), types2.get(t, 0)) for t in all_types)
            union_weight = sum(max(types1.get(t, 0), types2.get(t, 0)) for t in all_types)
            
            return intersection_weight / union_weight if union_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating cipher method correlation: {str(e)}")
            return 0.0
    
    def _calculate_correlation_significance(self, doc1_id: int, doc2_id: int, correlation: float) -> float:
        """Calculate statistical significance of correlation"""
        try:
            # Get sample sizes (number of patterns)
            patterns1 = self._get_document_patterns(doc1_id)
            patterns2 = self._get_document_patterns(doc2_id)
            
            n1, n2 = len(patterns1), len(patterns2)
            
            if n1 < 3 or n2 < 3:
                return 1.0  # Not significant with small samples
            
            # Simple significance calculation based on sample size and correlation strength
            min_n = min(n1, n2)
            significance = max(0.001, 1.0 - (correlation * math.sqrt(min_n) / 10))
            
            return min(1.0, significance)
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation significance: {str(e)}")
            return 1.0
    
    def _cluster_documents_by_correlation(self, correlation_matrix: np.ndarray, 
                                        document_ids: List[int]) -> Dict[int, int]:
        """Cluster documents based on correlation matrix"""
        try:
            if correlation_matrix.shape[0] < 2:
                return {doc_id: 0 for doc_id in document_ids}
            
            # Convert correlation to distance
            distance_matrix = 1 - correlation_matrix
            
            # Perform hierarchical clustering
            condensed_distances = []
            n = distance_matrix.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    condensed_distances.append(distance_matrix[i, j])
            
            if not condensed_distances:
                return {doc_id: 0 for doc_id in document_ids}
            
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Determine number of clusters
            n_clusters = max(2, min(len(document_ids) // 2, 5))
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            return {document_ids[i]: cluster_labels[i] - 1 for i in range(len(document_ids))}
            
        except Exception as e:
            self.logger.error(f"Error clustering documents: {str(e)}")
            return {doc_id: 0 for doc_id in document_ids}
    
    def _calculate_connection_strength(self, instances1: List, instances2: List) -> float:
        """Calculate connection strength between pattern instances"""
        try:
            if not instances1 or not instances2:
                return 0.0
            
            # Calculate average confidence
            avg_conf1 = np.mean([inst.confidence for inst in instances1])
            avg_conf2 = np.mean([inst.confidence for inst in instances2])
            confidence_factor = (avg_conf1 + avg_conf2) / 2
            
            # Calculate instance count factor
            count_factor = min(1.0, (len(instances1) + len(instances2)) / 10)
            
            # Calculate quality factor
            avg_quality1 = np.mean([inst.quality_score or 0.5 for inst in instances1])
            avg_quality2 = np.mean([inst.quality_score or 0.5 for inst in instances2])
            quality_factor = (avg_quality1 + avg_quality2) / 2
            
            return (confidence_factor + count_factor + quality_factor) / 3
            
        except Exception as e:
            self.logger.error(f"Error calculating connection strength: {str(e)}")
            return 0.0
    
    def _build_evidence_points(self, instances1: List, instances2: List) -> List[Dict[str, Any]]:
        """Build evidence points for pattern connections"""
        evidence_points = []
        
        try:
            # Evidence from instance similarities
            for inst1 in instances1:
                for inst2 in instances2:
                    similarity = self._calculate_instance_similarity(inst1, inst2)
                    if similarity > 0.5:
                        evidence_points.append({
                            'type': 'instance_similarity',
                            'strength': similarity,
                            'document1_instance': inst1.id,
                            'document2_instance': inst2.id,
                            'details': {
                                'confidence_match': abs(inst1.confidence - inst2.confidence) < 0.2,
                                'quality_match': abs((inst1.quality_score or 0.5) - (inst2.quality_score or 0.5)) < 0.2
                            }
                        })
            
            # Evidence from parameter consistency
            params1 = [inst.instance_data for inst in instances1 if inst.instance_data]
            params2 = [inst.instance_data for inst in instances2 if inst.instance_data]
            
            if params1 and params2:
                param_consistency = self._calculate_parameter_consistency(params1, params2)
                if param_consistency > 0.3:
                    evidence_points.append({
                        'type': 'parameter_consistency',
                        'strength': param_consistency,
                        'details': {
                            'consistent_parameters': self._find_consistent_parameters(params1, params2)
                        }
                    })
            
            return evidence_points
            
        except Exception as e:
            self.logger.error(f"Error building evidence points: {str(e)}")
            return []
    
    def _extract_temporal_sequence(self, instances: List) -> List[datetime]:
        """Extract temporal sequence from instances"""
        try:
            timestamps = []
            for instance in instances:
                if hasattr(instance, 'detected_at') and instance.detected_at:
                    timestamps.append(instance.detected_at)
            
            return sorted(timestamps)
            
        except Exception as e:
            self.logger.error(f"Error extracting temporal sequence: {str(e)}")
            return []
    
    def _calculate_trail_confidence(self, evidence_points: List[Dict[str, Any]], 
                                  connection_strength: float) -> float:
        """Calculate confidence score for evidence trail"""
        try:
            if not evidence_points:
                return 0.0
            
            # Base confidence from connection strength
            base_confidence = connection_strength
            
            # Boost from evidence count
            evidence_boost = min(0.3, len(evidence_points) * 0.05)
            
            # Boost from evidence strength
            avg_evidence_strength = np.mean([ep.get('strength', 0) for ep in evidence_points])
            strength_boost = avg_evidence_strength * 0.2
            
            return min(1.0, base_confidence + evidence_boost + strength_boost)
            
        except Exception as e:
            self.logger.error(f"Error calculating trail confidence: {str(e)}")
            return 0.0
    
    def _extend_evidence_trails(self, trails: List[EvidenceTrail], 
                              doc_instances: Dict[int, List]) -> List[EvidenceTrail]:
        """Extend evidence trails to multi-document chains"""
        extended_trails = []
        
        try:
            # Look for chains of 3+ documents
            document_ids = list(doc_instances.keys())
            
            for i in range(len(document_ids)):
                for j in range(i + 1, len(document_ids)):
                    for k in range(j + 1, len(document_ids)):
                        doc_chain = [document_ids[i], document_ids[j], document_ids[k]]
                        
                        # Check if we have evidence for all pairs
                        pairs = [(doc_chain[0], doc_chain[1]), (doc_chain[1], doc_chain[2]), (doc_chain[0], doc_chain[2])]
                        pair_strengths = []
                        
                        for pair in pairs:
                            strength = self._calculate_connection_strength(
                                doc_instances[pair[0]], doc_instances[pair[1]]
                            )
                            pair_strengths.append(strength)
                        
                        # If all pairs have reasonable strength, create extended trail
                        if all(s >= 0.2 for s in pair_strengths):
                            avg_strength = np.mean(pair_strengths)
                            
                            # Build combined evidence points
                            combined_evidence = []
                            for pair in pairs:
                                evidence = self._build_evidence_points(
                                    doc_instances[pair[0]], doc_instances[pair[1]]
                                )
                                combined_evidence.extend(evidence)
                            
                            # Create extended trail
                            extended_trail = EvidenceTrail(
                                pattern_id=trails[0].pattern_id if trails else 0,
                                document_chain=doc_chain,
                                connection_strength=avg_strength,
                                evidence_points=combined_evidence,
                                temporal_sequence=self._extract_temporal_sequence(
                                    [inst for doc_id in doc_chain for inst in doc_instances[doc_id]]
                                ),
                                confidence_score=self._calculate_trail_confidence(
                                    combined_evidence, avg_strength
                                )
                            )
                            extended_trails.append(extended_trail)
            
            return extended_trails
            
        except Exception as e:
            self.logger.error(f"Error extending evidence trails: {str(e)}")
            return []
    
    def _create_authorship_profile(self, document_id: int) -> Optional[AuthorshipProfile]:
        """Create authorship profile for a document"""
        try:
            # Get document
            document = self.db.query(Document).get(document_id)
            if not document:
                return None
            
            # Extract stylistic features
            stylistic_features = self._extract_stylistic_features(document_id)
            
            # Extract cipher preferences
            cipher_preferences = self._extract_cipher_preferences(document_id)
            
            # Extract complexity signature
            complexity_signature = self._extract_complexity_signature(document_id)
            
            # Extract temporal patterns
            temporal_patterns = self._extract_temporal_features(document_id)
            
            # Extract geometric preferences
            geometric_preferences = self._extract_geometric_preferences(document_id)
            
            return AuthorshipProfile(
                document_id=document_id,
                stylistic_features=stylistic_features,
                cipher_preferences=cipher_preferences,
                complexity_signature=complexity_signature,
                temporal_patterns=temporal_patterns,
                geometric_preferences=geometric_preferences
            )
            
        except Exception as e:
            self.logger.error(f"Error creating authorship profile: {str(e)}")
            return None
    
    def _calculate_authorship_similarity_matrix(self, profiles: List[AuthorshipProfile]) -> np.ndarray:
        """Calculate similarity matrix between authorship profiles"""
        try:
            n = len(profiles)
            similarity_matrix = np.zeros((n, n))
            
            # Fill diagonal
            np.fill_diagonal(similarity_matrix, 1.0)
            
            # Calculate pairwise similarities
            for i in range(n):
                for j in range(i + 1, n):
                    similarity = self._calculate_profile_similarity(profiles[i], profiles[j])
                    similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
            
            return similarity_matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating authorship similarity matrix: {str(e)}")
            return np.eye(len(profiles))
    
    def _identify_author_groups(self, profiles: List[AuthorshipProfile], 
                              similarity_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Identify potential author groups based on similarity"""
        try:
            # Perform clustering on similarity matrix
            n = similarity_matrix.shape[0]
            if n < 2:
                return []
            
            # Convert similarity to distance
            distance_matrix = 1 - similarity_matrix
            
            # Hierarchical clustering
            condensed_distances = []
            for i in range(n):
                for j in range(i + 1, n):
                    condensed_distances.append(distance_matrix[i, j])
            
            if not condensed_distances:
                return []
            
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Determine clusters
            n_clusters = max(2, min(n // 2, 4))
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Build author groups
            groups = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                groups[label].append(profiles[i])
            
            author_groups = []
            for group_id, group_profiles in groups.items():
                if len(group_profiles) >= 2:  # Only groups with multiple documents
                    group_info = {
                        'group_id': group_id,
                        'document_count': len(group_profiles),
                        'document_ids': [p.document_id for p in group_profiles],
                        'average_similarity': self._calculate_group_average_similarity(
                            group_profiles, similarity_matrix, profiles
                        ),
                        'shared_characteristics': self._identify_shared_characteristics(group_profiles),
                        'confidence': self._calculate_group_confidence(group_profiles)
                    }
                    author_groups.append(group_info)
            
            return author_groups
            
        except Exception as e:
            self.logger.error(f"Error identifying author groups: {str(e)}")
            return []
    
    # Additional helper methods for feature extraction and analysis
    
    def _get_document_pattern_signatures(self, document_id: int) -> Dict[str, float]:
        """Get pattern signatures for a document, merging CrossPatternInstance and local Pattern"""
        try:
            signatures = {}
            
            # 1. Get CrossPatternInstances (Existing logic)
            instances = self.db.query(CrossPatternInstance).filter(
                CrossPatternInstance.document_id == document_id
            ).all()
            
            for instance in instances:
                pattern = instance.cross_pattern
                if pattern:
                    signature_key = f"{pattern.pattern_type}_{pattern.pattern_subtype or 'default'}"
                    signatures[signature_key] = pattern.significance_score or 0.5
            
            # 2. Get Local Patterns (New Logic for Gematria/ELS)
            local_patterns = self.db.query(Pattern).filter(
                Pattern.document_id == document_id
            ).all()
            
            for p in local_patterns:
                # Include specific pattern types that are significant for direct matching
                if p.pattern_type in ['gematria_match', 'els_match']:
                    # Use pattern_name as key to ensure exact match across documents
                    # e.g., "Gematria: 100 (francis_bacon_simple)"
                    signature_key = p.pattern_name
                    # Use higher weight if available
                    signatures[signature_key] = p.significance_score or 0.8
            
            return signatures
            
        except Exception as e:
            self.logger.error(f"Error getting document pattern signatures: {str(e)}")
            return {}
    
    def _extract_stylistic_features(self, document_id: int) -> Dict[str, float]:
        """Extract stylistic features from document"""
        try:
            # Get document text
            document = self.db.query(Document).get(document_id)
            if not document:
                return {}
            
            # Get pages
            pages = self.db.query(Page).filter(Page.document_id == document_id).all()
            text_content = ' '.join([page.text for page in pages if page.text])
            
            if not text_content:
                return {}
            
            # Calculate stylistic features
            features = {}
            
            # Text length features
            features['text_length'] = len(text_content)
            features['word_count'] = len(text_content.split())
            features['sentence_count'] = text_content.count('.') + text_content.count('!') + text_content.count('?')
            
            # Average word/sentence length
            words = text_content.split()
            if words:
                features['avg_word_length'] = np.mean([len(word) for word in words])
            
            if features['sentence_count'] > 0:
                features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
            
            # Punctuation features
            features['comma_frequency'] = text_content.count(',') / len(text_content) if text_content else 0
            features['semicolon_frequency'] = text_content.count(';') / len(text_content) if text_content else 0
            features['colon_frequency'] = text_content.count(':') / len(text_content) if text_content else 0
            
            # Character frequency features
            char_counts = Counter(text_content.upper())
            total_chars = sum(char_counts.values())
            
            for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                features[f'char_freq_{char}'] = char_counts.get(char, 0) / total_chars if total_chars else 0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting stylistic features: {str(e)}")
            return {}
    
    def _extract_cipher_preferences(self, document_id: int) -> Dict[str, float]:
        """Extract cipher method preferences from document"""
        try:
            # Get cipher patterns for document
            instances = self.db.query(CrossPatternInstance).filter(
                CrossPatternInstance.document_id == document_id
            ).all()
            
            cipher_counts = Counter()
            total_patterns = 0
            
            for instance in instances:
                pattern = instance.cross_pattern
                if pattern and pattern.pattern_type in ['cipher', 'cryptographic']:
                    cipher_type = pattern.pattern_subtype or 'unknown'
                    cipher_counts[cipher_type] += 1
                    total_patterns += 1
            
            # Convert to preferences (frequencies)
            preferences = {}
            for cipher_type, count in cipher_counts.items():
                preferences[cipher_type] = count / total_patterns if total_patterns > 0 else 0
            
            return preferences
            
        except Exception as e:
            self.logger.error(f"Error extracting cipher preferences: {str(e)}")
            return {}
    
    def _extract_complexity_signature(self, document_id: int) -> Dict[str, float]:
        """Extract complexity signature from document patterns"""
        try:
            instances = self.db.query(CrossPatternInstance).filter(
                CrossPatternInstance.document_id == document_id
            ).all()
            
            complexities = []
            for instance in instances:
                pattern = instance.cross_pattern
                if pattern and pattern.pattern_complexity:
                    complexities.append(pattern.pattern_complexity)
            
            if not complexities:
                return {}
            
            return {
                'avg_complexity': np.mean(complexities),
                'max_complexity': np.max(complexities),
                'complexity_variance': np.var(complexities),
                'complexity_range': np.max(complexities) - np.min(complexities)
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting complexity signature: {str(e)}")
            return {}
    
    def _extract_temporal_features(self, document_id: int) -> Dict[str, Any]:
        """Extract temporal features from document"""
        try:
            document = self.db.query(Document).get(document_id)
            if not document:
                return {}
            
            features = {}
            
            # Document creation time
            if hasattr(document, 'created_at') and document.created_at:
                features['creation_time'] = document.created_at.timestamp()
            
            # Pattern detection timings
            instances = self.db.query(CrossPatternInstance).filter(
                CrossPatternInstance.document_id == document_id
            ).all()
            
            pattern_timings = []
            for instance in instances:
                if hasattr(instance, 'detected_at') and instance.detected_at:
                    pattern_timings.append(instance.detected_at.timestamp())
            
            features['pattern_timings'] = pattern_timings
            
            if pattern_timings:
                features['first_pattern_time'] = min(pattern_timings)
                features['last_pattern_time'] = max(pattern_timings)
                features['pattern_time_span'] = max(pattern_timings) - min(pattern_timings)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting temporal features: {str(e)}")
            return {}
    
    def _extract_geometric_preferences(self, document_id: int) -> Dict[str, float]:
        """Extract geometric pattern preferences from document"""
        try:
            instances = self.db.query(CrossPatternInstance).filter(
                CrossPatternInstance.document_id == document_id
            ).all()
            
            geometric_counts = Counter()
            total_geometric = 0
            
            for instance in instances:
                pattern = instance.cross_pattern
                if pattern and pattern.pattern_type == 'geometric':
                    geo_type = pattern.pattern_subtype or 'unknown'
                    geometric_counts[geo_type] += 1
                    total_geometric += 1
            
            # Convert to preferences
            preferences = {}
            for geo_type, count in geometric_counts.items():
                preferences[geo_type] = count / total_geometric if total_geometric > 0 else 0
            
            return preferences
            
        except Exception as e:
            self.logger.error(f"Error extracting geometric preferences: {str(e)}")
            return {}
    
    def _calculate_profile_similarity(self, profile1: AuthorshipProfile, 
                                    profile2: AuthorshipProfile) -> float:
        """Calculate similarity between two authorship profiles"""
        try:
            similarities = []
            
            # Stylistic similarity
            stylistic_sim = self._calculate_feature_similarity(
                profile1.stylistic_features, profile2.stylistic_features
            )
            similarities.append(stylistic_sim * 0.3)
            
            # Cipher preference similarity
            cipher_sim = self._calculate_feature_similarity(
                profile1.cipher_preferences, profile2.cipher_preferences
            )
            similarities.append(cipher_sim * 0.25)
            
            # Complexity similarity
            complexity_sim = self._calculate_feature_similarity(
                profile1.complexity_signature, profile2.complexity_signature
            )
            similarities.append(complexity_sim * 0.2)
            
            # Geometric preference similarity
            geometric_sim = self._calculate_feature_similarity(
                profile1.geometric_preferences, profile2.geometric_preferences
            )
            similarities.append(geometric_sim * 0.15)
            
            # Temporal similarity
            temporal_sim = self._calculate_temporal_similarity(
                profile1.temporal_patterns, profile2.temporal_patterns
            )
            similarities.append(temporal_sim * 0.1)
            
            return sum(similarities)
            
        except Exception as e:
            self.logger.error(f"Error calculating profile similarity: {str(e)}")
            return 0.0
    
    def _calculate_feature_similarity(self, features1: Dict[str, float], 
                                    features2: Dict[str, float]) -> float:
        """Calculate similarity between feature dictionaries"""
        try:
            if not features1 or not features2:
                return 0.0
            
            all_features = set(features1.keys()) | set(features2.keys())
            if not all_features:
                return 0.0
            
            vec1 = [features1.get(feature, 0) for feature in all_features]
            vec2 = [features2.get(feature, 0) for feature in all_features]
            
            # Calculate cosine similarity
            try:
                similarity = 1 - cosine(vec1, vec2)
                return max(0.0, similarity)
            except:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating feature similarity: {str(e)}")
            return 0.0
    
    def _calculate_temporal_similarity(self, temporal1: Dict[str, Any], 
                                     temporal2: Dict[str, Any]) -> float:
        """Calculate temporal similarity between profiles"""
        try:
            if not temporal1 or not temporal2:
                return 0.0
            
            # Compare creation times
            time1 = temporal1.get('creation_time', 0)
            time2 = temporal2.get('creation_time', 0)
            
            if time1 and time2:
                time_diff = abs(time1 - time2)
                max_time_diff = 365 * 24 * 3600  # 1 year
                temporal_proximity = max(0.0, 1.0 - (time_diff / max_time_diff))
                return temporal_proximity
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating temporal similarity: {str(e)}")
            return 0.0
    
    # Additional helper methods would continue here...
    # For brevity, I'll include the key remaining methods
    
    def _get_document_patterns(self, document_id: int) -> List:
        """Get all patterns for a document"""
        try:
            instances = self.db.query(CrossPatternInstance).filter(
                CrossPatternInstance.document_id == document_id
            ).all()
            return [instance.cross_pattern for instance in instances if instance.cross_pattern]
        except:
            return []
    
    def _calculate_document_relationship(self, doc1_id: int, doc2_id: int) -> DocumentRelationship:
        """Calculate comprehensive relationship between two documents"""
        try:
            # Calculate various correlation types
            pattern_corr = self._calculate_pattern_correlation(doc1_id, doc2_id)
            stylistic_corr = self._calculate_stylistic_correlation(doc1_id, doc2_id)
            temporal_corr = self._calculate_temporal_correlation(doc1_id, doc2_id)
            
            # Overall strength
            strength = (
                pattern_corr * 0.4 +
                stylistic_corr * 0.35 +
                temporal_corr * 0.25
            )
            
            # Determine relationship type
            if pattern_corr > 0.7:
                relationship_type = "pattern_based"
            elif stylistic_corr > 0.7:
                relationship_type = "stylistic"
            elif temporal_corr > 0.7:
                relationship_type = "temporal"
            else:
                relationship_type = "mixed"
            
            # Count shared patterns
            patterns1 = set(self._get_document_pattern_signatures(doc1_id).keys())
            patterns2 = set(self._get_document_pattern_signatures(doc2_id).keys())
            shared_patterns = list(patterns1 & patterns2)
            
            return DocumentRelationship(
                document1_id=doc1_id,
                document2_id=doc2_id,
                relationship_type=relationship_type,
                strength=strength,
                confidence=min(0.9, strength + 0.1),
                evidence_count=len(shared_patterns),
                shared_patterns=shared_patterns,
                temporal_correlation=temporal_corr,
                authorship_similarity=stylistic_corr
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating document relationship: {str(e)}")
            return DocumentRelationship(
                document1_id=doc1_id,
                document2_id=doc2_id,
                relationship_type="unknown",
                strength=0.0,
                confidence=0.0,
                evidence_count=0,
                shared_patterns=[],
                temporal_correlation=0.0,
                authorship_similarity=0.0
            )
    
    def _calculate_network_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        """Calculate network metrics for document relationship graph"""
        try:
            metrics = {}
            
            # Basic metrics
            metrics['node_count'] = G.number_of_nodes()
            metrics['edge_count'] = G.number_of_edges()
            metrics['density'] = nx.density(G)
            
            # Centrality measures
            if G.number_of_nodes() > 0:
                metrics['centrality'] = nx.degree_centrality(G)
                metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
                metrics['closeness_centrality'] = nx.closeness_centrality(G)
            
            # Clustering coefficient
            if G.number_of_nodes() > 2:
                metrics['clustering_coefficient'] = nx.average_clustering(G)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating network metrics: {str(e)}")
            return {}
    
    def _identify_communities(self, G: nx.Graph) -> Dict[int, int]:
        """Identify communities in the document network"""
        try:
            if G.number_of_nodes() < 2:
                return {}
            
            # Simple community detection based on connected components
            communities = {}
            for i, component in enumerate(nx.connected_components(G)):
                for node in component:
                    communities[node] = i
            
            return communities
            
        except Exception as e:
            self.logger.error(f"Error identifying communities: {str(e)}")
            return {}
    
    def _identify_central_documents(self, G: nx.Graph) -> List[Dict[str, Any]]:
        """Identify central documents in the network"""
        try:
            if G.number_of_nodes() == 0:
                return []
            
            centrality = nx.degree_centrality(G)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            
            central_docs = []
            for node, centrality_score in sorted_nodes[:5]:  # Top 5
                central_docs.append({
                    'document_id': node,
                    'centrality_score': centrality_score,
                    'degree': G.degree(node),
                    'connections': list(G.neighbors(node))
                })
            
            return central_docs
            
        except Exception as e:
            self.logger.error(f"Error identifying central documents: {str(e)}")
            return []    

    # Additional missing helper methods
    
    def _calculate_pattern_timing_correlation(self, timings1: List[float], timings2: List[float]) -> float:
        """Calculate correlation between pattern timing sequences"""
        try:
            if not timings1 or not timings2:
                return 0.0
            
            # Simple correlation based on timing proximity
            min_len = min(len(timings1), len(timings2))
            if min_len == 0:
                return 0.0
            
            # Calculate average time differences
            avg_diff1 = np.mean(np.diff(timings1[:min_len])) if min_len > 1 else 0
            avg_diff2 = np.mean(np.diff(timings2[:min_len])) if min_len > 1 else 0
            
            # Similarity based on timing patterns
            if avg_diff1 == 0 and avg_diff2 == 0:
                return 1.0
            
            max_diff = max(abs(avg_diff1), abs(avg_diff2), 1)
            similarity = 1.0 - abs(avg_diff1 - avg_diff2) / max_diff
            
            return max(0.0, similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern timing correlation: {str(e)}")
            return 0.0
    
    def _calculate_instance_similarity(self, instance1, instance2) -> float:
        """Calculate similarity between two pattern instances"""
        try:
            # Compare confidence scores
            conf_similarity = 1.0 - abs(instance1.confidence - instance2.confidence)
            
            # Compare quality scores
            quality1 = getattr(instance1, 'quality_score', 0.5) or 0.5
            quality2 = getattr(instance2, 'quality_score', 0.5) or 0.5
            quality_similarity = 1.0 - abs(quality1 - quality2)
            
            # Compare instance data if available
            data_similarity = 0.5  # Default
            if hasattr(instance1, 'instance_data') and hasattr(instance2, 'instance_data'):
                data1 = instance1.instance_data or {}
                data2 = instance2.instance_data or {}
                
                if data1 and data2:
                    common_keys = set(data1.keys()) & set(data2.keys())
                    total_keys = set(data1.keys()) | set(data2.keys())
                    
                    if total_keys:
                        data_similarity = len(common_keys) / len(total_keys)
            
            # Weighted average
            return (conf_similarity * 0.4 + quality_similarity * 0.3 + data_similarity * 0.3)
            
        except Exception as e:
            self.logger.error(f"Error calculating instance similarity: {str(e)}")
            return 0.0
    
    def _calculate_parameter_consistency(self, params1: List[Dict], params2: List[Dict]) -> float:
        """Calculate consistency between parameter sets"""
        try:
            if not params1 or not params2:
                return 0.0
            
            # Flatten parameter dictionaries
            all_params1 = {}
            for param_dict in params1:
                if isinstance(param_dict, dict):
                    all_params1.update(param_dict)
            
            all_params2 = {}
            for param_dict in params2:
                if isinstance(param_dict, dict):
                    all_params2.update(param_dict)
            
            if not all_params1 or not all_params2:
                return 0.0
            
            # Calculate parameter overlap
            common_keys = set(all_params1.keys()) & set(all_params2.keys())
            total_keys = set(all_params1.keys()) | set(all_params2.keys())
            
            if not total_keys:
                return 0.0
            
            # Base consistency from key overlap
            key_consistency = len(common_keys) / len(total_keys)
            
            # Value consistency for common keys
            value_consistency = 0.0
            if common_keys:
                consistent_values = 0
                for key in common_keys:
                    val1, val2 = all_params1[key], all_params2[key]
                    if val1 == val2:
                        consistent_values += 1
                    elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        # Numerical similarity
                        max_val = max(abs(val1), abs(val2), 1)
                        if abs(val1 - val2) / max_val < 0.1:  # Within 10%
                            consistent_values += 0.8
                
                value_consistency = consistent_values / len(common_keys)
            
            return (key_consistency + value_consistency) / 2
            
        except Exception as e:
            self.logger.error(f"Error calculating parameter consistency: {str(e)}")
            return 0.0
    
    def _find_consistent_parameters(self, params1: List[Dict], params2: List[Dict]) -> List[str]:
        """Find parameters that are consistent between two sets"""
        try:
            consistent_params = []
            
            # Flatten parameter dictionaries
            all_params1 = {}
            for param_dict in params1:
                if isinstance(param_dict, dict):
                    all_params1.update(param_dict)
            
            all_params2 = {}
            for param_dict in params2:
                if isinstance(param_dict, dict):
                    all_params2.update(param_dict)
            
            # Find consistent parameters
            common_keys = set(all_params1.keys()) & set(all_params2.keys())
            
            for key in common_keys:
                val1, val2 = all_params1[key], all_params2[key]
                if val1 == val2:
                    consistent_params.append(key)
                elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    max_val = max(abs(val1), abs(val2), 1)
                    if abs(val1 - val2) / max_val < 0.1:  # Within 10%
                        consistent_params.append(key)
            
            return consistent_params
            
        except Exception as e:
            self.logger.error(f"Error finding consistent parameters: {str(e)}")
            return []
    
    def _analyze_stylistic_evolution(self, profiles: List[AuthorshipProfile]) -> Dict[str, Any]:
        """Analyze stylistic evolution across authorship profiles"""
        try:
            if len(profiles) < 2:
                return {}
            
            # Sort profiles by document creation time if available
            sorted_profiles = sorted(profiles, key=lambda p: p.temporal_patterns.get('creation_time', 0))
            
            evolution_data = {
                'profile_count': len(profiles),
                'feature_trends': {},
                'complexity_trend': [],
                'cipher_preference_evolution': {},
                'temporal_span': 0
            }
            
            # Analyze feature trends
            feature_keys = set()
            for profile in sorted_profiles:
                feature_keys.update(profile.stylistic_features.keys())
            
            for feature in feature_keys:
                values = []
                for profile in sorted_profiles:
                    if feature in profile.stylistic_features:
                        values.append(profile.stylistic_features[feature])
                
                if len(values) >= 2:
                    evolution_data['feature_trends'][feature] = {
                        'start_value': values[0],
                        'end_value': values[-1],
                        'change': values[-1] - values[0],
                        'trend': 'increasing' if values[-1] > values[0] else 'decreasing'
                    }
            
            # Analyze complexity evolution
            for profile in sorted_profiles:
                complexity = profile.complexity_signature.get('avg_complexity', 0)
                evolution_data['complexity_trend'].append(complexity)
            
            # Calculate temporal span
            if len(sorted_profiles) >= 2:
                first_time = sorted_profiles[0].temporal_patterns.get('creation_time', 0)
                last_time = sorted_profiles[-1].temporal_patterns.get('creation_time', 0)
                evolution_data['temporal_span'] = last_time - first_time
            
            return evolution_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing stylistic evolution: {str(e)}")
            return {}
    
    def _detect_shared_knowledge_indicators(self, profiles: List[AuthorshipProfile]) -> Dict[str, Any]:
        """Detect indicators of shared knowledge across profiles"""
        try:
            indicators = {
                'shared_cipher_preferences': {},
                'common_complexity_patterns': [],
                'synchronized_features': [],
                'knowledge_sharing_score': 0.0
            }
            
            if len(profiles) < 2:
                return indicators
            
            # Analyze shared cipher preferences
            all_ciphers = set()
            for profile in profiles:
                all_ciphers.update(profile.cipher_preferences.keys())
            
            for cipher in all_ciphers:
                preferences = []
                for profile in profiles:
                    if cipher in profile.cipher_preferences:
                        preferences.append(profile.cipher_preferences[cipher])
                
                if len(preferences) >= 2:
                    indicators['shared_cipher_preferences'][cipher] = {
                        'profile_count': len(preferences),
                        'average_preference': np.mean(preferences),
                        'consistency': 1.0 - np.std(preferences) if len(preferences) > 1 else 1.0
                    }
            
            # Analyze complexity patterns
            complexities = []
            for profile in profiles:
                complexity = profile.complexity_signature.get('avg_complexity', 0)
                if complexity > 0:
                    complexities.append(complexity)
            
            if len(complexities) >= 2:
                indicators['common_complexity_patterns'] = {
                    'average_complexity': np.mean(complexities),
                    'complexity_consistency': 1.0 - np.std(complexities),
                    'high_complexity_count': sum(1 for c in complexities if c > 0.7)
                }
            
            # Calculate overall knowledge sharing score
            cipher_score = len(indicators['shared_cipher_preferences']) / max(len(all_ciphers), 1)
            complexity_score = indicators['common_complexity_patterns'].get('complexity_consistency', 0) if indicators['common_complexity_patterns'] else 0
            
            indicators['knowledge_sharing_score'] = (cipher_score + complexity_score) / 2
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error detecting shared knowledge indicators: {str(e)}")
            return {}
    
    def _calculate_authorship_confidence_scores(self, profiles: List[AuthorshipProfile], 
                                              similarity_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate confidence scores for authorship analysis"""
        try:
            confidence_data = {
                'individual_confidences': {},
                'group_confidences': {},
                'overall_confidence': 0.0
            }
            
            # Individual profile confidences
            for i, profile in enumerate(profiles):
                # Base confidence on feature completeness
                feature_completeness = len(profile.stylistic_features) / 20  # Assume 20 typical features
                cipher_completeness = len(profile.cipher_preferences) / 5  # Assume 5 typical ciphers
                
                # Confidence from similarity with others
                if similarity_matrix.shape[0] > i:
                    similarities = similarity_matrix[i, :]
                    avg_similarity = np.mean([s for j, s in enumerate(similarities) if j != i])
                else:
                    avg_similarity = 0.5
                
                individual_confidence = (feature_completeness + cipher_completeness + avg_similarity) / 3
                confidence_data['individual_confidences'][profile.document_id] = min(1.0, individual_confidence)
            
            # Overall confidence
            if confidence_data['individual_confidences']:
                confidence_data['overall_confidence'] = np.mean(list(confidence_data['individual_confidences'].values()))
            
            return confidence_data
            
        except Exception as e:
            self.logger.error(f"Error calculating authorship confidence scores: {str(e)}")
            return {}
    
    def _generate_authorship_summary(self, author_groups: List[Dict[str, Any]], 
                                   shared_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of authorship analysis"""
        try:
            summary = {
                'total_groups': len(author_groups),
                'largest_group_size': 0,
                'shared_knowledge_strength': shared_knowledge.get('knowledge_sharing_score', 0),
                'key_findings': []
            }
            
            if author_groups:
                summary['largest_group_size'] = max(group['document_count'] for group in author_groups)
                
                # Key findings
                if summary['largest_group_size'] >= 3:
                    summary['key_findings'].append("Strong authorship clustering detected")
                
                if summary['shared_knowledge_strength'] > 0.7:
                    summary['key_findings'].append("High level of shared cryptographic knowledge")
                
                high_confidence_groups = [g for g in author_groups if g.get('confidence', 0) > 0.8]
                if high_confidence_groups:
                    summary['key_findings'].append(f"{len(high_confidence_groups)} high-confidence author groups")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating authorship summary: {str(e)}")
            return {}
    
    def _identify_sophisticated_patterns(self, document_ids: List[int]) -> List[Dict[str, Any]]:
        """Identify sophisticated cryptographic patterns across documents"""
        try:
            sophisticated_patterns = []
            
            for doc_id in document_ids:
                # Get patterns for document
                instances = self.db.query(CrossPatternInstance).filter(
                    CrossPatternInstance.document_id == doc_id
                ).all()
                
                for instance in instances:
                    pattern = instance.cross_pattern
                    if pattern and pattern.pattern_complexity and pattern.pattern_complexity > 0.7:
                        sophisticated_patterns.append({
                            'document_id': doc_id,
                            'pattern_id': pattern.id,
                            'pattern_type': pattern.pattern_type,
                            'complexity': pattern.pattern_complexity,
                            'significance': pattern.significance_score or 0,
                            'description': pattern.description
                        })
            
            # Sort by complexity and significance
            sophisticated_patterns.sort(key=lambda x: (x['complexity'], x['significance']), reverse=True)
            
            return sophisticated_patterns
            
        except Exception as e:
            self.logger.error(f"Error identifying sophisticated patterns: {str(e)}")
            return []
    
    def _analyze_cipher_method_progression(self, document_ids: List[int]) -> Dict[str, Any]:
        """Analyze progression of cipher methods across documents"""
        try:
            progression_data = {
                'method_timeline': [],
                'complexity_progression': [],
                'method_evolution': {},
                'innovation_points': []
            }
            
            # Get documents with temporal information
            documents_with_time = []
            for doc_id in document_ids:
                doc = self.db.query(Document).get(doc_id)
                if doc and hasattr(doc, 'created_at') and doc.created_at:
                    documents_with_time.append((doc_id, doc.created_at))
            
            # Sort by time
            documents_with_time.sort(key=lambda x: x[1])
            
            # Analyze progression
            for doc_id, timestamp in documents_with_time:
                cipher_methods = self._extract_cipher_methods(doc_id)
                
                progression_data['method_timeline'].append({
                    'document_id': doc_id,
                    'timestamp': timestamp.isoformat(),
                    'methods': cipher_methods.get('cipher_types', []),
                    'complexity': np.mean([0.5] + [0.8 if 'advanced' in method else 0.3 for method in cipher_methods.get('cipher_types', [])])
                })
            
            return progression_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing cipher method progression: {str(e)}")
            return {}
    
    def _detect_mathematical_knowledge_indicators(self, document_ids: List[int]) -> Dict[str, Any]:
        """Detect indicators of mathematical knowledge in documents"""
        try:
            indicators = {
                'mathematical_constants': [],
                'geometric_sophistication': [],
                'numerical_patterns': [],
                'knowledge_level': 'basic'
            }
            
            for doc_id in document_ids:
                # Get geometric patterns
                geometric_features = self._extract_geometric_features(doc_id)
                
                # Check for mathematical constants
                constants = geometric_features.get('mathematical_constants', [])
                indicators['mathematical_constants'].extend(constants)
                
                # Check geometric sophistication
                pattern_types = geometric_features.get('pattern_types', [])
                if 'sacred_geometry' in pattern_types or 'golden_ratio' in pattern_types:
                    indicators['geometric_sophistication'].append({
                        'document_id': doc_id,
                        'sophistication_level': 'advanced',
                        'patterns': pattern_types
                    })
            
            # Determine overall knowledge level
            if len(indicators['mathematical_constants']) > 5:
                indicators['knowledge_level'] = 'advanced'
            elif len(indicators['mathematical_constants']) > 2:
                indicators['knowledge_level'] = 'intermediate'
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error detecting mathematical knowledge indicators: {str(e)}")
            return {}
    
    def _analyze_pattern_complexity_evolution(self, document_ids: List[int]) -> Dict[str, Any]:
        """Analyze evolution of pattern complexity across documents"""
        try:
            evolution_data = {
                'complexity_timeline': [],
                'average_complexity': 0.0,
                'complexity_trend': 'stable',
                'innovation_rate': 0.0
            }
            
            # Get complexity data for each document
            complexities = []
            for doc_id in document_ids:
                doc_complexity = self._calculate_document_complexity(doc_id)
                complexities.append({
                    'document_id': doc_id,
                    'complexity': doc_complexity
                })
            
            evolution_data['complexity_timeline'] = complexities
            
            if complexities:
                complexity_values = [c['complexity'] for c in complexities]
                evolution_data['average_complexity'] = np.mean(complexity_values)
                
                # Determine trend
                if len(complexity_values) >= 2:
                    if complexity_values[-1] > complexity_values[0] * 1.2:
                        evolution_data['complexity_trend'] = 'increasing'
                    elif complexity_values[-1] < complexity_values[0] * 0.8:
                        evolution_data['complexity_trend'] = 'decreasing'
            
            return evolution_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing pattern complexity evolution: {str(e)}")
            return {}
    
    def _calculate_knowledge_sharing_probability(self, sophisticated_patterns: List[Dict], 
                                               cipher_progression: Dict[str, Any], 
                                               mathematical_indicators: Dict[str, Any]) -> float:
        """Calculate probability of knowledge sharing between documents"""
        try:
            # Base probability from sophisticated patterns
            pattern_score = min(1.0, len(sophisticated_patterns) / 10)
            
            # Boost from cipher progression
            progression_score = 0.0
            if cipher_progression.get('method_timeline'):
                timeline = cipher_progression['method_timeline']
                if len(timeline) >= 2:
                    # Check for method consistency
                    all_methods = set()
                    for entry in timeline:
                        all_methods.update(entry.get('methods', []))
                    
                    if len(all_methods) > 1:  # Multiple methods used
                        progression_score = 0.3
                    
                    # Check for complexity progression
                    complexities = [entry.get('complexity', 0) for entry in timeline]
                    if len(complexities) >= 2 and complexities[-1] > complexities[0]:
                        progression_score += 0.2
            
            # Boost from mathematical knowledge
            math_score = 0.0
            knowledge_level = mathematical_indicators.get('knowledge_level', 'basic')
            if knowledge_level == 'advanced':
                math_score = 0.4
            elif knowledge_level == 'intermediate':
                math_score = 0.2
            
            # Combined probability
            probability = (pattern_score * 0.5 + progression_score * 0.3 + math_score * 0.2)
            
            return min(1.0, probability)
            
        except Exception as e:
            self.logger.error(f"Error calculating knowledge sharing probability: {str(e)}")
            return 0.0
    
    def _identify_knowledge_transfer_patterns(self, document_ids: List[int], 
                                            sophisticated_patterns: List[Dict]) -> List[Dict[str, Any]]:
        """Identify patterns of knowledge transfer between documents"""
        try:
            transfer_patterns = []
            
            # Group patterns by type and complexity
            pattern_groups = defaultdict(list)
            for pattern in sophisticated_patterns:
                key = f"{pattern['pattern_type']}_{pattern.get('complexity', 0):.1f}"
                pattern_groups[key].append(pattern)
            
            # Look for transfer patterns
            for pattern_type, patterns in pattern_groups.items():
                if len(patterns) >= 2:
                    # Sort by document order (assuming document IDs reflect chronology)
                    patterns.sort(key=lambda x: x['document_id'])
                    
                    transfer_pattern = {
                        'pattern_type': pattern_type,
                        'document_chain': [p['document_id'] for p in patterns],
                        'transfer_strength': np.mean([p['significance'] for p in patterns]),
                        'complexity_evolution': [p['complexity'] for p in patterns],
                        'evidence_count': len(patterns)
                    }
                    
                    transfer_patterns.append(transfer_pattern)
            
            return transfer_patterns
            
        except Exception as e:
            self.logger.error(f"Error identifying knowledge transfer patterns: {str(e)}")
            return []
    
    def _calculate_evidence_strength(self, sophisticated_patterns: List[Dict], 
                                   mathematical_indicators: Dict[str, Any]) -> float:
        """Calculate overall evidence strength for shared knowledge"""
        try:
            # Evidence from pattern count and complexity
            pattern_evidence = 0.0
            if sophisticated_patterns:
                avg_complexity = np.mean([p['complexity'] for p in sophisticated_patterns])
                pattern_count_factor = min(1.0, len(sophisticated_patterns) / 5)
                pattern_evidence = (avg_complexity + pattern_count_factor) / 2
            
            # Evidence from mathematical sophistication
            math_evidence = 0.0
            knowledge_level = mathematical_indicators.get('knowledge_level', 'basic')
            if knowledge_level == 'advanced':
                math_evidence = 0.8
            elif knowledge_level == 'intermediate':
                math_evidence = 0.5
            elif knowledge_level == 'basic':
                math_evidence = 0.2
            
            # Combined evidence strength
            return (pattern_evidence * 0.6 + math_evidence * 0.4)
            
        except Exception as e:
            self.logger.error(f"Error calculating evidence strength: {str(e)}")
            return 0.0
    
    def _extract_cipher_methods(self, document_id: int) -> Dict[str, Any]:
        """Extract cipher methods used in a document"""
        try:
            instances = self.db.query(CrossPatternInstance).filter(
                CrossPatternInstance.document_id == document_id
            ).all()
            
            cipher_types = []
            for instance in instances:
                pattern = instance.cross_pattern
                if pattern and pattern.pattern_type in ['cipher', 'cryptographic']:
                    cipher_types.append(pattern.pattern_subtype or 'unknown')
            
            return {
                'cipher_types': cipher_types,
                'method_count': len(set(cipher_types)),
                'total_instances': len(cipher_types)
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting cipher methods: {str(e)}")
            return {}
    
    def _extract_geometric_features(self, document_id: int) -> Dict[str, Any]:
        """Extract geometric features from a document"""
        try:
            instances = self.db.query(CrossPatternInstance).filter(
                CrossPatternInstance.document_id == document_id
            ).all()
            
            pattern_types = []
            mathematical_constants = []
            
            for instance in instances:
                pattern = instance.cross_pattern
                if pattern and pattern.pattern_type == 'geometric':
                    pattern_types.append(pattern.pattern_subtype or 'unknown')
                    
                    # Check for mathematical constants in pattern data
                    if pattern.pattern_parameters:
                        params = pattern.pattern_parameters
                        if 'mathematical_constant' in params:
                            mathematical_constants.append(params['mathematical_constant'])
            
            return {
                'pattern_types': pattern_types,
                'mathematical_constants': mathematical_constants,
                'geometric_complexity': len(set(pattern_types)) / 5  # Normalize by expected types
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting geometric features: {str(e)}")
            return {}
    
    def _calculate_document_complexity(self, document_id: int) -> float:
        """Calculate overall complexity score for a document"""
        try:
            instances = self.db.query(CrossPatternInstance).filter(
                CrossPatternInstance.document_id == document_id
            ).all()
            
            if not instances:
                return 0.0
            
            complexities = []
            for instance in instances:
                pattern = instance.cross_pattern
                if pattern and pattern.pattern_complexity:
                    complexities.append(pattern.pattern_complexity)
            
            return np.mean(complexities) if complexities else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating document complexity: {str(e)}")
            return 0.0
    
    def _calculate_group_average_similarity(self, group_profiles: List[AuthorshipProfile], 
                                          similarity_matrix: np.ndarray, 
                                          all_profiles: List[AuthorshipProfile]) -> float:
        """Calculate average similarity within an author group"""
        try:
            if len(group_profiles) < 2:
                return 0.0
            
            # Get indices of group profiles in the full list
            group_indices = []
            for group_profile in group_profiles:
                for i, profile in enumerate(all_profiles):
                    if profile.document_id == group_profile.document_id:
                        group_indices.append(i)
                        break
            
            if len(group_indices) < 2:
                return 0.0
            
            # Calculate average pairwise similarity within group
            similarities = []
            for i in range(len(group_indices)):
                for j in range(i + 1, len(group_indices)):
                    idx1, idx2 = group_indices[i], group_indices[j]
                    if idx1 < similarity_matrix.shape[0] and idx2 < similarity_matrix.shape[1]:
                        similarities.append(similarity_matrix[idx1, idx2])
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating group average similarity: {str(e)}")
            return 0.0
    
    def _identify_shared_characteristics(self, group_profiles: List[AuthorshipProfile]) -> Dict[str, Any]:
        """Identify shared characteristics within an author group"""
        try:
            shared_chars = {
                'common_stylistic_features': {},
                'shared_cipher_preferences': {},
                'similar_complexity_patterns': {},
                'temporal_clustering': False
            }
            
            if len(group_profiles) < 2:
                return shared_chars
            
            # Find common stylistic features
            all_features = set()
            for profile in group_profiles:
                all_features.update(profile.stylistic_features.keys())
            
            for feature in all_features:
                values = []
                for profile in group_profiles:
                    if feature in profile.stylistic_features:
                        values.append(profile.stylistic_features[feature])
                
                if len(values) >= 2:
                    std_dev = np.std(values)
                    if std_dev < 0.1:  # Low variation indicates shared characteristic
                        shared_chars['common_stylistic_features'][feature] = {
                            'average_value': np.mean(values),
                            'consistency': 1.0 - std_dev
                        }
            
            # Find shared cipher preferences
            all_ciphers = set()
            for profile in group_profiles:
                all_ciphers.update(profile.cipher_preferences.keys())
            
            for cipher in all_ciphers:
                preferences = []
                for profile in group_profiles:
                    if cipher in profile.cipher_preferences:
                        preferences.append(profile.cipher_preferences[cipher])
                
                if len(preferences) >= len(group_profiles) * 0.8:  # Most profiles have this cipher
                    shared_chars['shared_cipher_preferences'][cipher] = {
                        'average_preference': np.mean(preferences),
                        'profile_coverage': len(preferences) / len(group_profiles)
                    }
            
            return shared_chars
            
        except Exception as e:
            self.logger.error(f"Error identifying shared characteristics: {str(e)}")
            return {}
    
    def _calculate_group_confidence(self, group_profiles: List[AuthorshipProfile]) -> float:
        """Calculate confidence score for an author group"""
        try:
            if len(group_profiles) < 2:
                return 0.0
            
            # Base confidence on group size
            size_factor = min(1.0, len(group_profiles) / 5)
            
            # Confidence from feature completeness
            total_features = 0
            complete_features = 0
            
            for profile in group_profiles:
                total_features += 20  # Expected number of features
                complete_features += len(profile.stylistic_features)
            
            completeness_factor = complete_features / total_features if total_features > 0 else 0
            
            # Confidence from cipher diversity
            all_ciphers = set()
            for profile in group_profiles:
                all_ciphers.update(profile.cipher_preferences.keys())
            
            cipher_factor = min(1.0, len(all_ciphers) / 5)
            
            # Combined confidence
            return (size_factor * 0.4 + completeness_factor * 0.4 + cipher_factor * 0.2)
            
        except Exception as e:
            self.logger.error(f"Error calculating group confidence: {str(e)}")
            return 0.0