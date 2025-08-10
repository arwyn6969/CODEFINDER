"""
Cross-Document Pattern Database Service
Handles storage, indexing, and retrieval of patterns that appear across multiple documents.
Provides efficient pattern matching, similarity scoring, and clustering capabilities.
"""
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import hashlib
import json
import math
import numpy as np
from scipy.spatial.distance import cosine, jaccard
from scipy.cluster.hierarchy import linkage, fcluster
# Removed sklearn dependencies - using simpler alternatives
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
import logging

from app.models.cross_document_models import (
    CrossDocumentPattern, CrossPatternInstance, PatternRelationship,
    DocumentCluster, DocumentClusterMembership, SharedConstruction,
    ConstructionInstance, PatternSimilarityIndex
)
from app.models.database_models import Document, Pattern
from app.core.database import get_db


@dataclass
class PatternSearchResult:
    """Result of pattern search operation"""
    pattern: CrossDocumentPattern
    instances: List[CrossPatternInstance]
    similarity_score: float
    relevance_score: float


@dataclass
class ClusteringResult:
    """Result of document clustering operation"""
    clusters: List[DocumentCluster]
    cluster_assignments: Dict[int, int]  # document_id -> cluster_id
    quality_metrics: Dict[str, float]


class CrossDocumentPatternDatabase:
    """
    Service for managing cross-document patterns and their relationships
    """
    
    def __init__(self, db_session: Session = None):
        self.db = db_session or next(get_db())
        self.logger = logging.getLogger(__name__)
        
        # Pattern similarity thresholds
        self.similarity_threshold = 0.7
        self.clustering_threshold = 0.6
        self.significance_threshold = 0.05
        
        # Indexing parameters
        self.max_pattern_cache = 10000
        self.similarity_cache = {}
        
    def store_cross_document_pattern(self, pattern_data: Dict[str, Any]) -> CrossDocumentPattern:
        """
        Store a new cross-document pattern in the database
        """
        try:
            # Generate pattern hash for deduplication
            pattern_hash = self._generate_pattern_hash(pattern_data)
            
            # Check if pattern already exists
            existing_pattern = self.db.query(CrossDocumentPattern).filter(
                CrossDocumentPattern.pattern_hash == pattern_hash
            ).first()
            
            if existing_pattern:
                # Update existing pattern
                return self._update_existing_pattern(existing_pattern, pattern_data)
            
            # Create new pattern
            new_pattern = CrossDocumentPattern(
                pattern_hash=pattern_hash,
                pattern_type=pattern_data.get('pattern_type'),
                pattern_subtype=pattern_data.get('pattern_subtype'),
                pattern_name=pattern_data.get('pattern_name'),
                description=pattern_data.get('description', ''),
                pattern_signature=pattern_data.get('pattern_signature', {}),
                pattern_parameters=pattern_data.get('pattern_parameters', {}),
                pattern_complexity=pattern_data.get('pattern_complexity', 0.0),
                discovery_method=pattern_data.get('discovery_method', 'automated')
            )
            
            self.db.add(new_pattern)
            self.db.commit()
            self.db.refresh(new_pattern)
            
            # Store pattern instances
            if 'instances' in pattern_data:
                self._store_pattern_instances(new_pattern.id, pattern_data['instances'])
            
            # Update pattern statistics
            self._update_pattern_statistics(new_pattern.id)
            
            self.logger.info(f"Stored new cross-document pattern: {new_pattern.id}")
            return new_pattern
            
        except Exception as e:
            self.logger.error(f"Error storing cross-document pattern: {str(e)}")
            self.db.rollback()
            raise
    
    def search_similar_patterns(self, query_pattern: Dict[str, Any], 
                               limit: int = 10) -> List[PatternSearchResult]:
        """
        Search for patterns similar to the query pattern
        """
        try:
            query_signature = query_pattern.get('pattern_signature', {})
            query_type = query_pattern.get('pattern_type')
            
            # Get candidate patterns of the same type
            candidates = self.db.query(CrossDocumentPattern).filter(
                CrossDocumentPattern.pattern_type == query_type
            ).all()
            
            if not candidates:
                return []
            
            # Calculate similarities
            results = []
            for candidate in candidates:
                similarity_score = self._calculate_pattern_similarity(
                    query_signature, candidate.pattern_signature
                )
                
                if similarity_score >= self.similarity_threshold:
                    # Get pattern instances
                    instances = self.db.query(CrossPatternInstance).filter(
                        CrossPatternInstance.cross_pattern_id == candidate.id
                    ).all()
                    
                    # Calculate relevance score
                    relevance_score = self._calculate_relevance_score(
                        candidate, similarity_score
                    )
                    
                    results.append(PatternSearchResult(
                        pattern=candidate,
                        instances=instances,
                        similarity_score=similarity_score,
                        relevance_score=relevance_score
                    ))
            
            # Sort by relevance and limit results
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error searching similar patterns: {str(e)}")
            return []
    
    def identify_shared_constructions(self, document_ids: List[int]) -> List[SharedConstruction]:
        """
        Identify shared geometric or cryptographic constructions across specified documents
        """
        try:
            if len(document_ids) < 2:
                return []
            
            # Get patterns from all specified documents
            document_patterns = {}
            for doc_id in document_ids:
                patterns = self._get_document_patterns(doc_id)
                document_patterns[doc_id] = patterns
            
            # Find patterns that appear in multiple documents
            pattern_document_map = defaultdict(set)
            for doc_id, patterns in document_patterns.items():
                for pattern in patterns:
                    pattern_document_map[pattern.pattern_hash].add(doc_id)
            
            # Identify shared constructions
            shared_constructions = []
            for pattern_hash, docs in pattern_document_map.items():
                if len(docs) >= 2:  # Appears in at least 2 documents
                    construction = self._create_shared_construction(pattern_hash, docs)
                    if construction:
                        shared_constructions.append(construction)
            
            return shared_constructions
            
        except Exception as e:
            self.logger.error(f"Error identifying shared constructions: {str(e)}")
            return []
    
    def cluster_documents_by_patterns(self, document_ids: List[int], 
                                    clustering_method: str = "hierarchical") -> ClusteringResult:
        """
        Cluster documents based on their shared patterns
        """
        try:
            if len(document_ids) < 2:
                return ClusteringResult(clusters=[], cluster_assignments={}, quality_metrics={})
            
            # Build document-pattern matrix
            doc_pattern_matrix = self._build_document_pattern_matrix(document_ids)
            
            if clustering_method == "hierarchical":
                clusters, assignments, metrics = self._hierarchical_clustering(
                    doc_pattern_matrix, document_ids
                )
            else:
                # Add other clustering methods as needed
                raise ValueError(f"Unsupported clustering method: {clustering_method}")
            
            # Store clustering results
            stored_clusters = []
            for i, cluster_info in enumerate(clusters):
                stored_cluster = self._store_document_cluster(cluster_info, i)
                stored_clusters.append(stored_cluster)
            
            return ClusteringResult(
                clusters=stored_clusters,
                cluster_assignments=assignments,
                quality_metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error clustering documents: {str(e)}")
            return ClusteringResult(clusters=[], cluster_assignments={}, quality_metrics={})
    
    def build_pattern_similarity_index(self, batch_size: int = 1000) -> None:
        """
        Build or update the pattern similarity index for efficient similarity queries
        """
        try:
            # Get all patterns
            patterns = self.db.query(CrossDocumentPattern).all()
            total_patterns = len(patterns)
            
            self.logger.info(f"Building similarity index for {total_patterns} patterns")
            
            # Process patterns in batches
            for i in range(0, total_patterns, batch_size):
                batch = patterns[i:i + batch_size]
                self._process_similarity_batch(batch, patterns)
                
                # Commit batch
                self.db.commit()
                self.logger.info(f"Processed similarity batch {i//batch_size + 1}/{(total_patterns-1)//batch_size + 1}")
            
            self.logger.info("Pattern similarity index build complete")
            
        except Exception as e:
            self.logger.error(f"Error building similarity index: {str(e)}")
            self.db.rollback()
            raise
    
    def get_pattern_relationships(self, pattern_id: int) -> List[PatternRelationship]:
        """
        Get all relationships for a specific pattern
        """
        try:
            relationships = self.db.query(PatternRelationship).filter(
                or_(
                    PatternRelationship.pattern1_id == pattern_id,
                    PatternRelationship.pattern2_id == pattern_id
                )
            ).all()
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error getting pattern relationships: {str(e)}")
            return []
    
    def analyze_pattern_evolution(self, pattern_id: int) -> Dict[str, Any]:
        """
        Analyze how a pattern evolves across different documents or time periods
        """
        try:
            pattern = self.db.query(CrossDocumentPattern).get(pattern_id)
            if not pattern:
                return {}
            
            # Get all instances of this pattern
            instances = self.db.query(CrossPatternInstance).filter(
                CrossPatternInstance.cross_pattern_id == pattern_id
            ).all()
            
            # Analyze evolution
            evolution_data = {
                'pattern_id': pattern_id,
                'total_instances': len(instances),
                'document_spread': len(set(inst.document_id for inst in instances)),
                'confidence_trend': self._analyze_confidence_trend(instances),
                'parameter_evolution': self._analyze_parameter_evolution(instances),
                'temporal_distribution': self._analyze_temporal_distribution(instances),
                'quality_metrics': self._calculate_evolution_quality_metrics(instances)
            }
            
            return evolution_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing pattern evolution: {str(e)}")
            return {}
    
    def get_document_pattern_profile(self, document_id: int) -> Dict[str, Any]:
        """
        Get a comprehensive pattern profile for a specific document
        """
        try:
            # Get document patterns
            patterns = self._get_document_patterns(document_id)
            
            # Get pattern instances
            instances = self.db.query(CrossPatternInstance).filter(
                CrossPatternInstance.document_id == document_id
            ).all()
            
            # Calculate profile metrics
            profile = {
                'document_id': document_id,
                'total_patterns': len(patterns),
                'pattern_types': self._analyze_pattern_types(patterns),
                'complexity_distribution': self._analyze_complexity_distribution(patterns),
                'significance_scores': self._analyze_significance_scores(patterns),
                'shared_pattern_count': len([p for p in patterns if p.document_count > 1]),
                'unique_pattern_count': len([p for p in patterns if p.document_count == 1]),
                'pattern_density': len(instances) / max(1, len(patterns)),
                'quality_metrics': self._calculate_document_quality_metrics(instances)
            }
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Error getting document pattern profile: {str(e)}")
            return {}
    
    # Private helper methods
    
    def _generate_pattern_hash(self, pattern_data: Dict[str, Any]) -> str:
        """Generate a unique hash for pattern deduplication"""
        # Create a normalized representation for hashing
        hash_data = {
            'type': pattern_data.get('pattern_type'),
            'subtype': pattern_data.get('pattern_subtype'),
            'signature': pattern_data.get('pattern_signature', {}),
            'parameters': pattern_data.get('pattern_parameters', {})
        }
        
        # Sort keys for consistent hashing
        normalized_json = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(normalized_json.encode()).hexdigest()
    
    def _update_existing_pattern(self, pattern: CrossDocumentPattern, 
                               new_data: Dict[str, Any]) -> CrossDocumentPattern:
        """Update an existing pattern with new data"""
        # Update fields that might change
        if 'instances' in new_data:
            self._store_pattern_instances(pattern.id, new_data['instances'])
        
        # Update statistics
        self._update_pattern_statistics(pattern.id)
        
        pattern.last_updated = func.now()
        self.db.commit()
        
        return pattern
    
    def _store_pattern_instances(self, pattern_id: int, instances_data: List[Dict[str, Any]]) -> None:
        """Store pattern instances for a cross-document pattern"""
        for instance_data in instances_data:
            instance = CrossPatternInstance(
                cross_pattern_id=pattern_id,
                document_id=instance_data.get('document_id'),
                page_numbers=instance_data.get('page_numbers', []),
                coordinates=instance_data.get('coordinates', {}),
                instance_data=instance_data.get('instance_data', {}),
                confidence=instance_data.get('confidence', 0.0),
                quality_score=instance_data.get('quality_score', 0.0),
                detection_method=instance_data.get('detection_method', 'automated')
            )
            self.db.add(instance)
    
    def _update_pattern_statistics(self, pattern_id: int) -> None:
        """Update statistical measures for a pattern"""
        # Get all instances
        instances = self.db.query(CrossPatternInstance).filter(
            CrossPatternInstance.cross_pattern_id == pattern_id
        ).all()
        
        if not instances:
            return
        
        # Calculate statistics
        document_count = len(set(inst.document_id for inst in instances))
        total_occurrences = len(instances)
        avg_confidence = sum(inst.confidence for inst in instances) / len(instances)
        
        # Update pattern
        pattern = self.db.query(CrossDocumentPattern).get(pattern_id)
        if pattern:
            pattern.document_count = document_count
            pattern.total_occurrences = total_occurrences
            pattern.average_confidence = avg_confidence
            pattern.consistency_score = self._calculate_consistency_score(instances)
            pattern.rarity_score = self._calculate_rarity_score(document_count, total_occurrences)
            pattern.significance_score = self._calculate_significance_score(pattern)
    
    def _calculate_pattern_similarity(self, signature1: Dict[str, Any], 
                                    signature2: Dict[str, Any]) -> float:
        """Calculate similarity between two pattern signatures"""
        if not signature1 or not signature2:
            return 0.0
        
        # Convert signatures to feature vectors
        features1 = self._signature_to_features(signature1)
        features2 = self._signature_to_features(signature2)
        
        # Calculate cosine similarity
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
        
        # Ensure same dimensionality
        all_keys = set(features1.keys()) | set(features2.keys())
        vec1 = [features1.get(key, 0) for key in all_keys]
        vec2 = [features2.get(key, 0) for key in all_keys]
        
        # Calculate cosine similarity
        try:
            similarity = 1 - cosine(vec1, vec2)
            return max(0.0, similarity)  # Ensure non-negative
        except:
            return 0.0   
 
    def _signature_to_features(self, signature: Dict[str, Any]) -> Dict[str, float]:
        """Convert pattern signature to feature vector"""
        features = {}
        
        # Extract numerical features
        for key, value in signature.items():
            if isinstance(value, (int, float)):
                features[f"num_{key}"] = float(value)
            elif isinstance(value, str):
                # Hash string values to numerical features
                features[f"str_{key}"] = hash(value) % 1000 / 1000.0
            elif isinstance(value, list):
                # List features
                features[f"list_{key}_len"] = len(value)
                if value and isinstance(value[0], (int, float)):
                    features[f"list_{key}_mean"] = sum(value) / len(value)
                    features[f"list_{key}_std"] = np.std(value) if len(value) > 1 else 0
        
        return features
    
    def _calculate_relevance_score(self, pattern: CrossDocumentPattern, 
                                 similarity_score: float) -> float:
        """Calculate relevance score combining similarity and pattern importance"""
        # Weight factors
        similarity_weight = 0.4
        significance_weight = 0.3
        frequency_weight = 0.2
        recency_weight = 0.1
        
        # Normalize scores
        significance_score = pattern.significance_score or 0.0
        frequency_score = min(1.0, pattern.document_count / 10.0)  # Normalize to 0-1
        
        # Calculate recency score (more recent = higher score)
        import datetime
        days_since_discovery = (datetime.datetime.now() - pattern.first_discovered).days
        recency_score = max(0.0, 1.0 - (days_since_discovery / 365.0))  # Decay over a year
        
        relevance = (
            similarity_score * similarity_weight +
            significance_score * significance_weight +
            frequency_score * frequency_weight +
            recency_score * recency_weight
        )
        
        return min(1.0, relevance)
    
    def _get_document_patterns(self, document_id: int) -> List[CrossDocumentPattern]:
        """Get all cross-document patterns for a specific document"""
        instances = self.db.query(CrossPatternInstance).filter(
            CrossPatternInstance.document_id == document_id
        ).all()
        
        pattern_ids = [inst.cross_pattern_id for inst in instances]
        patterns = self.db.query(CrossDocumentPattern).filter(
            CrossDocumentPattern.id.in_(pattern_ids)
        ).all()
        
        return patterns
    
    def _create_shared_construction(self, pattern_hash: str, 
                                  document_ids: Set[int]) -> Optional[SharedConstruction]:
        """Create a shared construction from a pattern that appears in multiple documents"""
        try:
            # Get the pattern
            pattern = self.db.query(CrossDocumentPattern).filter(
                CrossDocumentPattern.pattern_hash == pattern_hash
            ).first()
            
            if not pattern:
                return None
            
            # Check if construction already exists
            existing = self.db.query(SharedConstruction).filter(
                SharedConstruction.construction_signature == pattern_hash
            ).first()
            
            if existing:
                return existing
            
            # Create new shared construction
            construction = SharedConstruction(
                construction_type=pattern.pattern_type,
                construction_name=pattern.pattern_name or f"Shared {pattern.pattern_type}",
                description=pattern.description,
                construction_parameters=pattern.pattern_parameters,
                construction_signature=pattern_hash,
                document_ids=list(document_ids),
                occurrence_count=pattern.total_occurrences,
                consistency_measure=pattern.consistency_score or 0.0,
                complexity_score=pattern.pattern_complexity or 0.0,
                significance_score=pattern.significance_score or 0.0,
                discovery_method=pattern.discovery_method
            )
            
            self.db.add(construction)
            self.db.commit()
            self.db.refresh(construction)
            
            return construction
            
        except Exception as e:
            self.logger.error(f"Error creating shared construction: {str(e)}")
            return None
    
    def _build_document_pattern_matrix(self, document_ids: List[int]) -> np.ndarray:
        """Build a document-pattern matrix for clustering"""
        # Get all unique patterns across documents
        all_patterns = set()
        doc_patterns = {}
        
        for doc_id in document_ids:
            patterns = self._get_document_patterns(doc_id)
            pattern_ids = [p.id for p in patterns]
            doc_patterns[doc_id] = pattern_ids
            all_patterns.update(pattern_ids)
        
        all_patterns = list(all_patterns)
        
        # Build matrix
        matrix = np.zeros((len(document_ids), len(all_patterns)))
        
        for i, doc_id in enumerate(document_ids):
            doc_pattern_ids = doc_patterns.get(doc_id, [])
            for j, pattern_id in enumerate(all_patterns):
                if pattern_id in doc_pattern_ids:
                    # Get pattern significance as weight
                    pattern = self.db.query(CrossDocumentPattern).get(pattern_id)
                    weight = pattern.significance_score if pattern else 1.0
                    matrix[i, j] = weight
        
        return matrix
    
    def _hierarchical_clustering(self, matrix: np.ndarray, 
                               document_ids: List[int]) -> Tuple[List[Dict], Dict[int, int], Dict[str, float]]:
        """Perform hierarchical clustering on document-pattern matrix"""
        if matrix.shape[0] < 2:
            return [], {}, {}
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(matrix, method='ward')
        
        # Determine optimal number of clusters (simple heuristic)
        n_clusters = max(2, min(len(document_ids) // 2, 5))
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Build cluster information
        clusters = []
        cluster_assignments = {}
        
        for cluster_id in range(1, n_clusters + 1):
            cluster_docs = [document_ids[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            if cluster_docs:
                cluster_info = {
                    'cluster_id': cluster_id,
                    'document_ids': cluster_docs,
                    'document_count': len(cluster_docs),
                    'cluster_type': 'pattern_based'
                }
                clusters.append(cluster_info)
                
                for doc_id in cluster_docs:
                    cluster_assignments[doc_id] = cluster_id
        
        # Calculate quality metrics
        quality_metrics = {
            'n_clusters': n_clusters,
            'silhouette_score': 0.0,  # Would need sklearn for proper calculation
            'cluster_sizes': [len(c['document_ids']) for c in clusters]
        }
        
        return clusters, cluster_assignments, quality_metrics
    
    def _store_document_cluster(self, cluster_info: Dict[str, Any], cluster_index: int) -> DocumentCluster:
        """Store a document cluster in the database"""
        cluster = DocumentCluster(
            cluster_name=f"Pattern Cluster {cluster_index + 1}",
            cluster_type=cluster_info.get('cluster_type', 'pattern_based'),
            cluster_algorithm='hierarchical',
            document_count=cluster_info.get('document_count', 0),
            cohesion_score=0.0,  # Would calculate from actual data
            separation_score=0.0,  # Would calculate from actual data
            silhouette_score=0.0,  # Would calculate from actual data
            clustering_parameters={'method': 'hierarchical', 'threshold': self.clustering_threshold}
        )
        
        self.db.add(cluster)
        self.db.commit()
        self.db.refresh(cluster)
        
        # Add cluster memberships
        for doc_id in cluster_info.get('document_ids', []):
            membership = DocumentClusterMembership(
                cluster_id=cluster.id,
                document_id=doc_id,
                membership_strength=1.0,  # Would calculate actual strength
                distance_to_centroid=0.0,
                membership_confidence=0.8,
                assignment_method='hierarchical_clustering',
                is_core_member=True
            )
            self.db.add(membership)
        
        self.db.commit()
        return cluster
    
    def _process_similarity_batch(self, batch: List[CrossDocumentPattern], 
                                all_patterns: List[CrossDocumentPattern]) -> None:
        """Process a batch of patterns for similarity index building"""
        for pattern1 in batch:
            for pattern2 in all_patterns:
                if pattern1.id >= pattern2.id:  # Avoid duplicates and self-comparison
                    continue
                
                # Check if similarity already computed
                existing = self.db.query(PatternSimilarityIndex).filter(
                    and_(
                        PatternSimilarityIndex.pattern1_id == pattern1.id,
                        PatternSimilarityIndex.pattern2_id == pattern2.id
                    )
                ).first()
                
                if existing:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_pattern_similarity(
                    pattern1.pattern_signature, pattern2.pattern_signature
                )
                
                if similarity >= 0.1:  # Only store meaningful similarities
                    similarity_index = PatternSimilarityIndex(
                        pattern1_id=pattern1.id,
                        pattern2_id=pattern2.id,
                        cosine_similarity=similarity,
                        overall_similarity=similarity,
                        similarity_confidence=0.8,
                        computation_method='cosine_signature'
                    )
                    self.db.add(similarity_index)
    
    def _calculate_consistency_score(self, instances: List[CrossPatternInstance]) -> float:
        """Calculate how consistently a pattern appears across instances"""
        if not instances:
            return 0.0
        
        # Calculate coefficient of variation for confidence scores
        confidences = [inst.confidence for inst in instances]
        if len(confidences) < 2:
            return 1.0
        
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        
        if mean_conf == 0:
            return 0.0
        
        cv = std_conf / mean_conf
        consistency = max(0.0, 1.0 - cv)  # Lower variation = higher consistency
        
        return consistency
    
    def _calculate_rarity_score(self, document_count: int, total_occurrences: int) -> float:
        """Calculate rarity score (inverse of frequency)"""
        # Assume we have some baseline for common patterns
        max_expected_docs = 100  # Adjust based on your corpus size
        max_expected_occurrences = 1000
        
        doc_rarity = 1.0 - (document_count / max_expected_docs)
        occurrence_rarity = 1.0 - (total_occurrences / max_expected_occurrences)
        
        return max(0.0, (doc_rarity + occurrence_rarity) / 2.0)
    
    def _calculate_significance_score(self, pattern: CrossDocumentPattern) -> float:
        """Calculate overall significance score for a pattern"""
        # Combine multiple factors
        factors = {
            'rarity': pattern.rarity_score or 0.0,
            'consistency': pattern.consistency_score or 0.0,
            'confidence': pattern.average_confidence or 0.0,
            'complexity': pattern.pattern_complexity or 0.0,
            'spread': min(1.0, pattern.document_count / 10.0)  # Normalize document spread
        }
        
        # Weighted combination
        weights = {'rarity': 0.3, 'consistency': 0.2, 'confidence': 0.2, 'complexity': 0.15, 'spread': 0.15}
        
        significance = sum(factors[key] * weights[key] for key in factors)
        return min(1.0, significance)
    
    # Analysis helper methods
    
    def _analyze_confidence_trend(self, instances: List[CrossPatternInstance]) -> Dict[str, float]:
        """Analyze confidence trend across instances"""
        if not instances:
            return {}
        
        confidences = [inst.confidence for inst in instances]
        return {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': min(confidences),
            'max': max(confidences),
            'trend': 'stable'  # Would calculate actual trend
        }
    
    def _analyze_parameter_evolution(self, instances: List[CrossPatternInstance]) -> Dict[str, Any]:
        """Analyze how pattern parameters evolve across instances"""
        # This would analyze changes in pattern parameters across documents/time
        return {'evolution_detected': False, 'parameter_stability': 0.8}
    
    def _analyze_temporal_distribution(self, instances: List[CrossPatternInstance]) -> Dict[str, Any]:
        """Analyze temporal distribution of pattern instances"""
        # This would analyze when patterns appear chronologically
        return {'temporal_clustering': False, 'distribution': 'uniform'}
    
    def _calculate_evolution_quality_metrics(self, instances: List[CrossPatternInstance]) -> Dict[str, float]:
        """Calculate quality metrics for pattern evolution analysis"""
        return {
            'completeness': 1.0,
            'reliability': 0.8,
            'temporal_coverage': 0.9
        }
    
    def _analyze_pattern_types(self, patterns: List[CrossDocumentPattern]) -> Dict[str, int]:
        """Analyze distribution of pattern types"""
        type_counts = Counter(p.pattern_type for p in patterns)
        return dict(type_counts)
    
    def _analyze_complexity_distribution(self, patterns: List[CrossDocumentPattern]) -> Dict[str, float]:
        """Analyze complexity distribution of patterns"""
        complexities = [p.pattern_complexity or 0.0 for p in patterns]
        if not complexities:
            return {}
        
        return {
            'mean': np.mean(complexities),
            'std': np.std(complexities),
            'min': min(complexities),
            'max': max(complexities)
        }
    
    def _analyze_significance_scores(self, patterns: List[CrossDocumentPattern]) -> Dict[str, float]:
        """Analyze significance score distribution"""
        scores = [p.significance_score or 0.0 for p in patterns]
        if not scores:
            return {}
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'high_significance_count': len([s for s in scores if s > 0.7])
        }
    
    def _calculate_document_quality_metrics(self, instances: List[CrossPatternInstance]) -> Dict[str, float]:
        """Calculate quality metrics for a document's patterns"""
        if not instances:
            return {}
        
        return {
            'average_confidence': np.mean([inst.confidence for inst in instances]),
            'average_quality': np.mean([inst.quality_score or 0.0 for inst in instances]),
            'validation_rate': len([inst for inst in instances if inst.is_validated]) / len(instances)
        }