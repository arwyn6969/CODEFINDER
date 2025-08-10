"""
Cross-Document Pattern Database Models
SQLAlchemy models for storing cross-document patterns, relationships, and shared constructions
"""
from sqlalchemy import Column, Integer, String, Float, Text, Boolean, DateTime, ForeignKey, JSON, Index, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any, List

from app.core.database import Base


# Association table for many-to-many relationship between patterns and documents
pattern_document_association = Table(
    'pattern_document_associations',
    Base.metadata,
    Column('cross_pattern_id', Integer, ForeignKey('cross_document_patterns.id'), primary_key=True),
    Column('document_id', Integer, ForeignKey('documents.id'), primary_key=True),
    Column('relevance_score', Float, default=1.0),
    Column('first_detected', DateTime, default=func.now())
)


class CrossDocumentPattern(Base):
    """
    Model for storing patterns that appear across multiple documents
    """
    __tablename__ = "cross_document_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Pattern identification
    pattern_hash = Column(String(64), unique=True, nullable=False)  # SHA-256 hash for deduplication
    pattern_type = Column(String(100), nullable=False)  # cipher, geometric, linguistic, structural
    pattern_subtype = Column(String(100))  # caesar_shift, skip_pattern, triangle_construction, etc.
    pattern_name = Column(String(200))
    description = Column(Text, nullable=False)
    
    # Pattern characteristics
    pattern_signature = Column(JSON, nullable=False)  # Normalized pattern representation
    pattern_parameters = Column(JSON)  # Parameters that define the pattern
    pattern_complexity = Column(Float)  # Complexity score (0-1)
    
    # Cross-document statistics
    document_count = Column(Integer, default=0)  # Number of documents containing this pattern
    total_occurrences = Column(Integer, default=0)  # Total occurrences across all documents
    average_confidence = Column(Float)  # Average confidence across all occurrences
    consistency_score = Column(Float)  # How consistently the pattern appears
    
    # Pattern significance
    rarity_score = Column(Float)  # How rare this pattern is (inverse of frequency)
    significance_score = Column(Float)  # Overall significance score
    statistical_p_value = Column(Float)  # Statistical significance
    
    # Pattern clustering and similarity
    cluster_id = Column(String(50))  # Cluster assignment for similar patterns
    similarity_hash = Column(String(32))  # Hash for similarity grouping
    
    # Discovery and validation
    first_discovered = Column(DateTime, default=func.now())
    last_updated = Column(DateTime, default=func.now())
    discovery_method = Column(String(100))
    validation_status = Column(String(50), default="pending")  # pending, validated, rejected
    validation_notes = Column(Text)
    
    # Relationships (Document relationship handled separately to avoid circular imports)
    pattern_instances = relationship("CrossPatternInstance", back_populates="cross_pattern", cascade="all, delete-orphan")
    pattern_relationships = relationship("PatternRelationship", 
                                       foreign_keys="PatternRelationship.pattern1_id",
                                       back_populates="pattern1")
    related_patterns = relationship("PatternRelationship",
                                  foreign_keys="PatternRelationship.pattern2_id", 
                                  back_populates="pattern2")
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_cross_pattern_type', 'pattern_type', 'pattern_subtype'),
        Index('idx_cross_pattern_significance', 'significance_score'),
        Index('idx_cross_pattern_documents', 'document_count'),
        Index('idx_cross_pattern_cluster', 'cluster_id'),
        Index('idx_cross_pattern_hash', 'pattern_hash'),
    )
    
    def __repr__(self):
        return f"<CrossDocumentPattern(id={self.id}, type='{self.pattern_type}', docs={self.document_count})>"


class CrossPatternInstance(Base):
    """
    Model for storing specific instances of cross-document patterns within individual documents
    """
    __tablename__ = "cross_pattern_instances"
    
    id = Column(Integer, primary_key=True, index=True)
    cross_pattern_id = Column(Integer, ForeignKey("cross_document_patterns.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Instance location and context
    page_numbers = Column(JSON)  # Pages where this instance appears
    coordinates = Column(JSON)  # Specific coordinates within the document
    context_before = Column(Text)  # Text/context before the pattern
    context_after = Column(Text)  # Text/context after the pattern
    
    # Instance characteristics
    instance_data = Column(JSON, nullable=False)  # Specific data for this instance
    confidence = Column(Float, nullable=False)  # Confidence of this specific instance
    quality_score = Column(Float)  # Quality/clarity of this instance
    
    # Instance analysis
    variations = Column(JSON)  # Any variations from the standard pattern
    anomalies = Column(JSON)  # Anomalies specific to this instance
    surrounding_patterns = Column(JSON)  # Other patterns found nearby
    
    # Discovery metadata
    detected_at = Column(DateTime, default=func.now())
    detection_method = Column(String(100))
    detection_parameters = Column(JSON)
    
    # Validation and review
    is_validated = Column(Boolean, default=False)
    validation_confidence = Column(Float)
    reviewer_notes = Column(Text)
    
    # Relationships
    cross_pattern = relationship("CrossDocumentPattern", back_populates="pattern_instances")
    document = relationship("Document")
    
    # Indexes
    __table_args__ = (
        Index('idx_instance_pattern_doc', 'cross_pattern_id', 'document_id'),
        Index('idx_instance_confidence', 'document_id', 'confidence'),
        Index('idx_instance_validated', 'is_validated'),
    )
    
    def __repr__(self):
        return f"<CrossPatternInstance(id={self.id}, pattern_id={self.cross_pattern_id}, doc_id={self.document_id})>"


class PatternRelationship(Base):
    """
    Model for storing relationships between different cross-document patterns
    """
    __tablename__ = "pattern_relationships"
    
    id = Column(Integer, primary_key=True, index=True)
    pattern1_id = Column(Integer, ForeignKey("cross_document_patterns.id"), nullable=False)
    pattern2_id = Column(Integer, ForeignKey("cross_document_patterns.id"), nullable=False)
    
    # Relationship characteristics
    relationship_type = Column(String(100), nullable=False)  # similar, complementary, sequential, hierarchical
    relationship_strength = Column(Float, nullable=False)  # 0-1 strength of relationship
    relationship_direction = Column(String(50))  # bidirectional, pattern1_to_pattern2, pattern2_to_pattern1
    
    # Relationship evidence
    evidence_data = Column(JSON)  # Evidence supporting this relationship
    co_occurrence_frequency = Column(Float)  # How often they appear together
    correlation_coefficient = Column(Float)  # Statistical correlation
    
    # Relationship context
    context_description = Column(Text)
    shared_documents = Column(JSON)  # Documents where both patterns appear
    relationship_significance = Column(Float)
    
    # Discovery and validation
    discovered_at = Column(DateTime, default=func.now())
    discovery_method = Column(String(100))
    is_validated = Column(Boolean, default=False)
    validation_notes = Column(Text)
    
    # Relationships
    pattern1 = relationship("CrossDocumentPattern", foreign_keys=[pattern1_id], back_populates="pattern_relationships")
    pattern2 = relationship("CrossDocumentPattern", foreign_keys=[pattern2_id], back_populates="related_patterns")
    
    # Indexes
    __table_args__ = (
        Index('idx_relationship_patterns', 'pattern1_id', 'pattern2_id'),
        Index('idx_relationship_type', 'relationship_type'),
        Index('idx_relationship_strength', 'relationship_strength'),
    )
    
    def __repr__(self):
        return f"<PatternRelationship(id={self.id}, type='{self.relationship_type}', strength={self.relationship_strength})>"


class DocumentCluster(Base):
    """
    Model for storing clusters of related documents based on shared patterns
    """
    __tablename__ = "document_clusters"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Cluster identification
    cluster_name = Column(String(200))
    cluster_type = Column(String(100))  # cipher_based, author_based, temporal, thematic
    cluster_algorithm = Column(String(100))  # Algorithm used for clustering
    
    # Cluster characteristics
    document_count = Column(Integer, default=0)
    cohesion_score = Column(Float)  # How tightly clustered the documents are
    separation_score = Column(Float)  # How well separated from other clusters
    silhouette_score = Column(Float)  # Overall clustering quality metric
    
    # Cluster patterns
    defining_patterns = Column(JSON)  # Patterns that define this cluster
    shared_characteristics = Column(JSON)  # Common characteristics of documents in cluster
    cluster_centroid = Column(JSON)  # Mathematical centroid of the cluster
    
    # Cluster analysis
    temporal_span = Column(JSON)  # Time range of documents in cluster
    geographic_distribution = Column(JSON)  # Geographic distribution if applicable
    authorship_analysis = Column(JSON)  # Authorship patterns within cluster
    
    # Clustering metadata
    created_at = Column(DateTime, default=func.now())
    clustering_parameters = Column(JSON)  # Parameters used for clustering
    last_updated = Column(DateTime, default=func.now())
    
    # Relationships
    cluster_memberships = relationship("DocumentClusterMembership", back_populates="cluster", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_cluster_type', 'cluster_type'),
        Index('idx_cluster_quality', 'silhouette_score'),
        Index('idx_cluster_size', 'document_count'),
    )
    
    def __repr__(self):
        return f"<DocumentCluster(id={self.id}, name='{self.cluster_name}', docs={self.document_count})>"


class DocumentClusterMembership(Base):
    """
    Model for storing document membership in clusters with membership strength
    """
    __tablename__ = "document_cluster_memberships"
    
    id = Column(Integer, primary_key=True, index=True)
    cluster_id = Column(Integer, ForeignKey("document_clusters.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Membership characteristics
    membership_strength = Column(Float, nullable=False)  # 0-1 strength of membership
    distance_to_centroid = Column(Float)  # Distance from cluster centroid
    membership_confidence = Column(Float)  # Confidence in cluster assignment
    
    # Membership evidence
    contributing_patterns = Column(JSON)  # Patterns that led to this clustering
    similarity_scores = Column(JSON)  # Similarity scores with other cluster members
    outlier_score = Column(Float)  # How much of an outlier this document is
    
    # Assignment metadata
    assigned_at = Column(DateTime, default=func.now())
    assignment_method = Column(String(100))
    is_core_member = Column(Boolean, default=False)  # Whether this is a core cluster member
    
    # Relationships
    cluster = relationship("DocumentCluster", back_populates="cluster_memberships")
    document = relationship("Document")
    
    # Indexes
    __table_args__ = (
        Index('idx_membership_cluster_doc', 'cluster_id', 'document_id'),
        Index('idx_membership_strength', 'cluster_id', 'membership_strength'),
        Index('idx_membership_core', 'cluster_id', 'is_core_member'),
    )
    
    def __repr__(self):
        return f"<DocumentClusterMembership(cluster_id={self.cluster_id}, doc_id={self.document_id}, strength={self.membership_strength})>"


class SharedConstruction(Base):
    """
    Model for storing shared geometric or cryptographic constructions across documents
    """
    __tablename__ = "shared_constructions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Construction identification
    construction_type = Column(String(100), nullable=False)  # geometric, cipher, linguistic, structural
    construction_name = Column(String(200))
    description = Column(Text, nullable=False)
    
    # Construction definition
    construction_parameters = Column(JSON, nullable=False)  # Mathematical/structural parameters
    construction_rules = Column(JSON)  # Rules that define the construction
    construction_signature = Column(String(128))  # Unique signature for deduplication
    
    # Cross-document presence
    document_ids = Column(JSON, nullable=False)  # Documents containing this construction
    occurrence_count = Column(Integer, default=0)  # Total occurrences
    consistency_measure = Column(Float)  # How consistently it appears
    
    # Construction analysis
    complexity_score = Column(Float)  # Complexity of the construction
    sophistication_level = Column(String(50))  # basic, intermediate, advanced, expert
    historical_significance = Column(Float)  # Historical importance score
    
    # Statistical analysis
    probability_random = Column(Float)  # Probability of random occurrence
    significance_score = Column(Float)  # Overall significance
    correlation_strength = Column(Float)  # Strength of cross-document correlation
    
    # Discovery and validation
    discovered_at = Column(DateTime, default=func.now())
    discovery_method = Column(String(100))
    validation_status = Column(String(50), default="pending")
    expert_validation = Column(Boolean, default=False)
    
    # Relationships
    construction_instances = relationship("ConstructionInstance", back_populates="shared_construction", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_construction_type', 'construction_type'),
        Index('idx_construction_significance', 'significance_score'),
        Index('idx_construction_signature', 'construction_signature'),
        Index('idx_construction_documents', 'document_ids'),  # GIN index for JSON array
    )
    
    def __repr__(self):
        return f"<SharedConstruction(id={self.id}, type='{self.construction_type}', docs={len(self.document_ids) if self.document_ids else 0})>"


class ConstructionInstance(Base):
    """
    Model for storing specific instances of shared constructions within documents
    """
    __tablename__ = "construction_instances"
    
    id = Column(Integer, primary_key=True, index=True)
    shared_construction_id = Column(Integer, ForeignKey("shared_constructions.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Instance location
    page_numbers = Column(JSON)  # Pages where this instance appears
    coordinates = Column(JSON)  # Specific coordinates
    bounding_box = Column(JSON)  # Bounding box of the construction
    
    # Instance data
    instance_parameters = Column(JSON, nullable=False)  # Specific parameters for this instance
    measurements = Column(JSON)  # Measurements and calculations
    quality_metrics = Column(JSON)  # Quality assessment of this instance
    
    # Instance analysis
    deviation_from_ideal = Column(Float)  # How much this deviates from the ideal construction
    confidence = Column(Float, nullable=False)  # Confidence in this instance
    completeness = Column(Float)  # How complete this instance is (0-1)
    
    # Context and relationships
    related_constructions = Column(JSON)  # Other constructions found nearby
    contextual_elements = Column(JSON)  # Surrounding textual/geometric elements
    
    # Discovery metadata
    detected_at = Column(DateTime, default=func.now())
    detection_algorithm = Column(String(100))
    detection_confidence = Column(Float)
    
    # Relationships
    shared_construction = relationship("SharedConstruction", back_populates="construction_instances")
    document = relationship("Document")
    
    # Indexes
    __table_args__ = (
        Index('idx_construction_instance_doc', 'shared_construction_id', 'document_id'),
        Index('idx_construction_instance_confidence', 'document_id', 'confidence'),
        Index('idx_construction_instance_quality', 'document_id', 'completeness'),
    )
    
    def __repr__(self):
        return f"<ConstructionInstance(id={self.id}, construction_id={self.shared_construction_id}, doc_id={self.document_id})>"


class PatternSimilarityIndex(Base):
    """
    Model for storing precomputed similarity scores between patterns for efficient retrieval
    """
    __tablename__ = "pattern_similarity_index"
    
    id = Column(Integer, primary_key=True, index=True)
    pattern1_id = Column(Integer, ForeignKey("cross_document_patterns.id"), nullable=False)
    pattern2_id = Column(Integer, ForeignKey("cross_document_patterns.id"), nullable=False)
    
    # Similarity metrics
    cosine_similarity = Column(Float)  # Cosine similarity score
    jaccard_similarity = Column(Float)  # Jaccard similarity score
    edit_distance = Column(Float)  # Normalized edit distance
    structural_similarity = Column(Float)  # Structural similarity score
    
    # Combined similarity
    overall_similarity = Column(Float, nullable=False)  # Weighted combination of metrics
    similarity_confidence = Column(Float)  # Confidence in similarity calculation
    
    # Similarity context
    similarity_basis = Column(JSON)  # What aspects are similar
    difference_analysis = Column(JSON)  # Key differences between patterns
    
    # Index metadata
    computed_at = Column(DateTime, default=func.now())
    computation_method = Column(String(100))
    last_updated = Column(DateTime, default=func.now())
    
    # Indexes for fast similarity queries
    __table_args__ = (
        Index('idx_similarity_patterns', 'pattern1_id', 'pattern2_id'),
        Index('idx_similarity_score', 'overall_similarity'),
        Index('idx_similarity_pattern1', 'pattern1_id', 'overall_similarity'),
        Index('idx_similarity_pattern2', 'pattern2_id', 'overall_similarity'),
    )
    
    def __repr__(self):
        return f"<PatternSimilarityIndex(pattern1_id={self.pattern1_id}, pattern2_id={self.pattern2_id}, similarity={self.overall_similarity})>"


# Add the cross_patterns relationship to the Document model
# This would be added to the existing Document model in database_models.py
# documents = relationship("CrossDocumentPattern", secondary=pattern_document_association, back_populates="documents")