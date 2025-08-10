"""
Cross-Reference Visualization System
Generates visualization data for document relationships, pattern connections,
timeline analysis, and comparative dashboards.
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import json
import math
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
import logging

from app.models.cross_document_models import (
    CrossDocumentPattern, CrossPatternInstance, PatternRelationship,
    DocumentCluster, DocumentClusterMembership, SharedConstruction
)
from app.models.database_models import Document, Pattern, Page
from app.services.relationship_analyzer import RelationshipAnalyzer
from app.services.cross_document_pattern_database import CrossDocumentPatternDatabase
from app.core.database import get_db


@dataclass
class NetworkNode:
    """Represents a node in the network visualization"""
    id: str
    label: str
    type: str
    size: float
    color: str
    metadata: Dict[str, Any]


@dataclass
class NetworkEdge:
    """Represents an edge in the network visualization"""
    source: str
    target: str
    weight: float
    type: str
    color: str
    metadata: Dict[str, Any]


@dataclass
class TimelineEvent:
    """Represents an event in the timeline visualization"""
    timestamp: datetime
    event_type: str
    title: str
    description: str
    document_ids: List[int]
    pattern_ids: List[int]
    significance: float


@dataclass
class VisualizationData:
    """Container for visualization data"""
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]
    timeline: List[TimelineEvent]
    metadata: Dict[str, Any]


class CrossReferenceVisualizer:
    """
    Service for generating cross-reference visualization data
    """
    
    def __init__(self, db_session: Session = None):
        self.db = db_session or next(get_db())
        self.relationship_analyzer = RelationshipAnalyzer(self.db)
        self.pattern_db = CrossDocumentPatternDatabase(self.db)
        self.logger = logging.getLogger(__name__)
        
        # Visualization configuration
        self.node_colors = {
            'document': '#3498db',
            'pattern': '#e74c3c',
            'cluster': '#2ecc71',
            'construction': '#f39c12'
        }
        
        self.edge_colors = {
            'pattern_based': '#e74c3c',
            'stylistic': '#9b59b6',
            'temporal': '#1abc9c',
            'geometric': '#f39c12',
            'cipher_method': '#34495e'
        }
    
    def generate_document_relationship_network(self, document_ids: List[int]) -> Dict[str, Any]:
        """
        Generate network visualization data for document relationships
        """
        try:
            # Get correlation matrix
            correlation_matrix = self.relationship_analyzer.generate_correlation_matrix(document_ids)
            
            # Create nodes for documents
            nodes = []
            for i, doc_id in enumerate(document_ids):
                doc = self.db.query(Document).get(doc_id)
                
                # Calculate node size based on pattern count
                pattern_count = self._get_document_pattern_count(doc_id)
                node_size = max(10, min(50, pattern_count * 2))
                
                # Determine cluster color
                cluster_id = correlation_matrix.cluster_assignments.get(doc_id, 0)
                color = self._get_cluster_color(cluster_id)
                
                node = NetworkNode(
                    id=f"doc_{doc_id}",
                    label=doc.title if doc else f"Document {doc_id}",
                    type="document",
                    size=node_size,
                    color=color,
                    metadata={
                        'document_id': doc_id,
                        'pattern_count': pattern_count,
                        'cluster_id': cluster_id,
                        'created_at': doc.created_at.isoformat() if doc and doc.created_at else None
                    }
                )
                nodes.append(node)
            
            # Create edges for relationships
            edges = []
            n_docs = len(document_ids)
            
            for i in range(n_docs):
                for j in range(i + 1, n_docs):
                    doc1_id, doc2_id = document_ids[i], document_ids[j]
                    
                    # Get correlation strength
                    correlation = correlation_matrix.matrix[i, j]
                    
                    if correlation >= 0.3:  # Only show significant relationships
                        # Determine relationship type
                        relationship_type = self._determine_primary_relationship_type(
                            correlation_matrix.correlation_types, i, j
                        )
                        
                        edge = NetworkEdge(
                            source=f"doc_{doc1_id}",
                            target=f"doc_{doc2_id}",
                            weight=correlation,
                            type=relationship_type,
                            color=self.edge_colors.get(relationship_type, '#95a5a6'),
                            metadata={
                                'correlation': correlation,
                                'significance': correlation_matrix.significance_matrix[i, j],
                                'relationship_details': self._get_relationship_details(
                                    correlation_matrix.correlation_types, i, j
                                )
                            }
                        )
                        edges.append(edge)
            
            # Calculate network statistics
            network_stats = self._calculate_network_statistics(nodes, edges)
            
            return {
                'nodes': [self._node_to_dict(node) for node in nodes],
                'edges': [self._edge_to_dict(edge) for edge in edges],
                'statistics': network_stats,
                'layout_config': self._get_network_layout_config(),
                'legend': self._generate_network_legend()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating document relationship network: {str(e)}")
            return {'error': str(e)}
    
    def generate_pattern_connection_network(self, pattern_ids: List[int] = None) -> Dict[str, Any]:
        """
        Generate network visualization showing pattern connections across documents
        """
        try:
            # Get patterns to visualize
            if pattern_ids:
                patterns = [self.db.query(CrossDocumentPattern).get(pid) for pid in pattern_ids]
                patterns = [p for p in patterns if p]
            else:
                # Get top patterns by significance
                patterns = self.db.query(CrossDocumentPattern).filter(
                    CrossDocumentPattern.significance_score > 0.5
                ).order_by(desc(CrossDocumentPattern.significance_score)).limit(50).all()
            
            if not patterns:
                return {'nodes': [], 'edges': [], 'message': 'No significant patterns found'}
            
            # Create nodes for patterns
            nodes = []
            for pattern in patterns:
                node_size = max(15, min(60, pattern.document_count * 5))
                
                node = NetworkNode(
                    id=f"pattern_{pattern.id}",
                    label=pattern.pattern_name or f"{pattern.pattern_type}_{pattern.id}",
                    type="pattern",
                    size=node_size,
                    color=self._get_pattern_color(pattern.pattern_type),
                    metadata={
                        'pattern_id': pattern.id,
                        'pattern_type': pattern.pattern_type,
                        'pattern_subtype': pattern.pattern_subtype,
                        'document_count': pattern.document_count,
                        'significance_score': pattern.significance_score,
                        'complexity': pattern.pattern_complexity
                    }
                )
                nodes.append(node)
            
            # Add document nodes
            document_nodes = self._create_document_nodes_for_patterns(patterns)
            nodes.extend(document_nodes)
            
            # Create edges for pattern-document relationships
            edges = []
            for pattern in patterns:
                instances = self.db.query(CrossPatternInstance).filter(
                    CrossPatternInstance.cross_pattern_id == pattern.id
                ).all()
                
                for instance in instances:
                    edge = NetworkEdge(
                        source=f"pattern_{pattern.id}",
                        target=f"doc_{instance.document_id}",
                        weight=instance.confidence,
                        type="contains",
                        color='#bdc3c7',
                        metadata={
                            'instance_id': instance.id,
                            'confidence': instance.confidence,
                            'quality_score': instance.quality_score
                        }
                    )
                    edges.append(edge)
            
            # Add pattern-pattern relationships
            pattern_edges = self._create_pattern_relationship_edges(patterns)
            edges.extend(pattern_edges)
            
            return {
                'nodes': [self._node_to_dict(node) for node in nodes],
                'edges': [self._edge_to_dict(edge) for edge in edges],
                'statistics': self._calculate_pattern_network_statistics(patterns),
                'layout_config': self._get_pattern_layout_config(),
                'legend': self._generate_pattern_legend()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating pattern connection network: {str(e)}")
            return {'error': str(e)}
    
    def generate_timeline_analysis(self, document_ids: List[int]) -> Dict[str, Any]:
        """
        Generate timeline visualization for shared pattern evolution
        """
        try:
            timeline_events = []
            
            # Get documents with timestamps
            documents_with_time = []
            for doc_id in document_ids:
                doc = self.db.query(Document).get(doc_id)
                if doc and hasattr(doc, 'created_at') and doc.created_at:
                    documents_with_time.append((doc_id, doc, doc.created_at))
            
            if not documents_with_time:
                return {'events': [], 'message': 'No temporal data available'}
            
            # Sort by time
            documents_with_time.sort(key=lambda x: x[2])
            
            # Create document creation events
            for doc_id, doc, timestamp in documents_with_time:
                pattern_count = self._get_document_pattern_count(doc_id)
                
                event = TimelineEvent(
                    timestamp=timestamp,
                    event_type="document_creation",
                    title=f"Document Created: {doc.title or f'Doc {doc_id}'}",
                    description=f"Document with {pattern_count} patterns",
                    document_ids=[doc_id],
                    pattern_ids=[],
                    significance=0.5
                )
                timeline_events.append(event)
            
            # Add pattern discovery events
            pattern_events = self._create_pattern_timeline_events(document_ids)
            timeline_events.extend(pattern_events)
            
            # Add shared construction events
            construction_events = self._create_construction_timeline_events(document_ids)
            timeline_events.extend(construction_events)
            
            # Sort all events by timestamp
            timeline_events.sort(key=lambda x: x.timestamp)
            
            # Generate timeline statistics
            timeline_stats = self._calculate_timeline_statistics(timeline_events)
            
            return {
                'events': [self._timeline_event_to_dict(event) for event in timeline_events],
                'statistics': timeline_stats,
                'time_range': {
                    'start': timeline_events[0].timestamp.isoformat() if timeline_events else None,
                    'end': timeline_events[-1].timestamp.isoformat() if timeline_events else None
                },
                'visualization_config': self._get_timeline_config()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating timeline analysis: {str(e)}")
            return {'error': str(e)}
    
    def generate_comparative_dashboard(self, document_ids: List[int]) -> Dict[str, Any]:
        """
        Generate comparative analysis dashboard data
        """
        try:
            dashboard_data = {
                'document_profiles': [],
                'comparison_matrices': {},
                'pattern_distributions': {},
                'relationship_summary': {},
                'key_insights': []
            }
            
            # Generate document profiles
            for doc_id in document_ids:
                profile = self._create_document_profile_for_dashboard(doc_id)
                dashboard_data['document_profiles'].append(profile)
            
            # Generate comparison matrices
            dashboard_data['comparison_matrices'] = self._generate_comparison_matrices(document_ids)
            
            # Generate pattern distributions
            dashboard_data['pattern_distributions'] = self._generate_pattern_distributions(document_ids)
            
            # Generate relationship summary
            dashboard_data['relationship_summary'] = self._generate_relationship_summary(document_ids)
            
            # Generate key insights
            dashboard_data['key_insights'] = self._generate_key_insights(document_ids)
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Error generating comparative dashboard: {str(e)}")
            return {'error': str(e)}
    
    def generate_evidence_trail_visualization(self, pattern_id: int) -> Dict[str, Any]:
        """
        Generate visualization for evidence trails of a specific pattern
        """
        try:
            # Get evidence trails
            trails = self.relationship_analyzer.track_evidence_trails(pattern_id)
            
            if not trails:
                return {'trails': [], 'message': 'No evidence trails found'}
            
            # Convert trails to visualization format
            trail_visualizations = []
            
            for trail in trails:
                # Create nodes for documents in the trail
                trail_nodes = []
                for doc_id in trail.document_chain:
                    doc = self.db.query(Document).get(doc_id)
                    node = {
                        'id': f"doc_{doc_id}",
                        'label': doc.title if doc else f"Document {doc_id}",
                        'type': 'document',
                        'metadata': {
                            'document_id': doc_id,
                            'connection_strength': trail.connection_strength
                        }
                    }
                    trail_nodes.append(node)
                
                # Create edges for connections
                trail_edges = []
                for i in range(len(trail.document_chain) - 1):
                    edge = {
                        'source': f"doc_{trail.document_chain[i]}",
                        'target': f"doc_{trail.document_chain[i + 1]}",
                        'weight': trail.connection_strength,
                        'type': 'evidence_connection',
                        'metadata': {
                            'confidence': trail.confidence_score,
                            'evidence_count': len(trail.evidence_points)
                        }
                    }
                    trail_edges.append(edge)
                
                trail_viz = {
                    'pattern_id': pattern_id,
                    'trail_id': f"trail_{len(trail_visualizations)}",
                    'nodes': trail_nodes,
                    'edges': trail_edges,
                    'evidence_points': [
                        {
                            'type': ep.get('type', 'unknown'),
                            'strength': ep.get('strength', 0),
                            'details': ep.get('details', {})
                        }
                        for ep in trail.evidence_points
                    ],
                    'temporal_sequence': [ts.isoformat() for ts in trail.temporal_sequence],
                    'confidence_score': trail.confidence_score,
                    'connection_strength': trail.connection_strength
                }
                trail_visualizations.append(trail_viz)
            
            return {
                'pattern_id': pattern_id,
                'trails': trail_visualizations,
                'summary': {
                    'total_trails': len(trails),
                    'average_confidence': np.mean([t.confidence_score for t in trails]),
                    'max_chain_length': max(len(t.document_chain) for t in trails),
                    'total_evidence_points': sum(len(t.evidence_points) for t in trails)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating evidence trail visualization: {str(e)}")
            return {'error': str(e)}
    
    # Private helper methods
    
    def _get_document_pattern_count(self, document_id: int) -> int:
        """Get the number of patterns in a document"""
        try:
            count = self.db.query(CrossPatternInstance).filter(
                CrossPatternInstance.document_id == document_id
            ).count()
            return count
        except:
            return 0
    
    def _get_cluster_color(self, cluster_id: int) -> str:
        """Get color for a cluster"""
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        return colors[cluster_id % len(colors)]
    
    def _get_pattern_color(self, pattern_type: str) -> str:
        """Get color for a pattern type"""
        color_map = {
            'cipher': '#e74c3c',
            'geometric': '#f39c12',
            'linguistic': '#9b59b6',
            'structural': '#2ecc71',
            'cryptographic': '#34495e'
        }
        return color_map.get(pattern_type, '#95a5a6')
    
    def _determine_primary_relationship_type(self, correlation_types: Dict[str, np.ndarray], 
                                           i: int, j: int) -> str:
        """Determine the primary relationship type between two documents"""
        max_correlation = 0
        primary_type = 'mixed'
        
        for rel_type, matrix in correlation_types.items():
            if matrix[i, j] > max_correlation:
                max_correlation = matrix[i, j]
                primary_type = rel_type
        
        return primary_type
    
    def _get_relationship_details(self, correlation_types: Dict[str, np.ndarray], 
                                i: int, j: int) -> Dict[str, float]:
        """Get detailed relationship scores"""
        details = {}
        for rel_type, matrix in correlation_types.items():
            details[rel_type] = float(matrix[i, j])
        return details
    
    def _calculate_network_statistics(self, nodes: List[NetworkNode], 
                                    edges: List[NetworkEdge]) -> Dict[str, Any]:
        """Calculate network statistics"""
        return {
            'node_count': len(nodes),
            'edge_count': len(edges),
            'average_degree': (2 * len(edges)) / len(nodes) if nodes else 0,
            'density': (2 * len(edges)) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0,
            'average_weight': np.mean([e.weight for e in edges]) if edges else 0,
            'node_types': Counter(n.type for n in nodes),
            'edge_types': Counter(e.type for e in edges)
        }
    
    def _get_network_layout_config(self) -> Dict[str, Any]:
        """Get configuration for network layout"""
        return {
            'algorithm': 'force_directed',
            'iterations': 1000,
            'node_repulsion': 100,
            'edge_attraction': 0.1,
            'gravity': 0.01,
            'damping': 0.9
        }
    
    def _generate_network_legend(self) -> Dict[str, Any]:
        """Generate legend for network visualization"""
        return {
            'node_types': [
                {'type': 'document', 'color': self.node_colors['document'], 'description': 'Document'},
                {'type': 'pattern', 'color': self.node_colors['pattern'], 'description': 'Pattern'},
                {'type': 'cluster', 'color': self.node_colors['cluster'], 'description': 'Cluster'}
            ],
            'edge_types': [
                {'type': 'pattern_based', 'color': self.edge_colors['pattern_based'], 'description': 'Pattern-based relationship'},
                {'type': 'stylistic', 'color': self.edge_colors['stylistic'], 'description': 'Stylistic similarity'},
                {'type': 'temporal', 'color': self.edge_colors['temporal'], 'description': 'Temporal correlation'}
            ],
            'size_encoding': 'Node size represents pattern count or significance',
            'weight_encoding': 'Edge thickness represents relationship strength'
        }
    
    def _create_document_nodes_for_patterns(self, patterns: List[CrossDocumentPattern]) -> List[NetworkNode]:
        """Create document nodes for pattern visualization"""
        document_ids = set()
        for pattern in patterns:
            instances = self.db.query(CrossPatternInstance).filter(
                CrossPatternInstance.cross_pattern_id == pattern.id
            ).all()
            document_ids.update(inst.document_id for inst in instances)
        
        nodes = []
        for doc_id in document_ids:
            doc = self.db.query(Document).get(doc_id)
            pattern_count = self._get_document_pattern_count(doc_id)
            
            node = NetworkNode(
                id=f"doc_{doc_id}",
                label=doc.title if doc else f"Document {doc_id}",
                type="document",
                size=max(10, min(40, pattern_count)),
                color=self.node_colors['document'],
                metadata={
                    'document_id': doc_id,
                    'pattern_count': pattern_count
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _create_pattern_relationship_edges(self, patterns: List[CrossDocumentPattern]) -> List[NetworkEdge]:
        """Create edges for pattern-pattern relationships"""
        edges = []
        
        for pattern in patterns:
            relationships = self.db.query(PatternRelationship).filter(
                or_(
                    PatternRelationship.pattern1_id == pattern.id,
                    PatternRelationship.pattern2_id == pattern.id
                )
            ).all()
            
            for rel in relationships:
                other_pattern_id = rel.pattern2_id if rel.pattern1_id == pattern.id else rel.pattern1_id
                
                # Check if other pattern is in our list
                if any(p.id == other_pattern_id for p in patterns):
                    edge = NetworkEdge(
                        source=f"pattern_{rel.pattern1_id}",
                        target=f"pattern_{rel.pattern2_id}",
                        weight=rel.relationship_strength,
                        type=rel.relationship_type,
                        color=self._get_relationship_color(rel.relationship_type),
                        metadata={
                            'relationship_id': rel.id,
                            'relationship_type': rel.relationship_type,
                            'strength': rel.relationship_strength,
                            'evidence': rel.evidence_data
                        }
                    )
                    edges.append(edge)
        
        return edges
    
    def _get_relationship_color(self, relationship_type: str) -> str:
        """Get color for relationship type"""
        color_map = {
            'similar': '#3498db',
            'complementary': '#2ecc71',
            'sequential': '#f39c12',
            'hierarchical': '#9b59b6'
        }
        return color_map.get(relationship_type, '#95a5a6')
    
    def _calculate_pattern_network_statistics(self, patterns: List[CrossDocumentPattern]) -> Dict[str, Any]:
        """Calculate statistics for pattern network"""
        return {
            'pattern_count': len(patterns),
            'pattern_types': Counter(p.pattern_type for p in patterns),
            'average_significance': np.mean([p.significance_score or 0 for p in patterns]),
            'average_complexity': np.mean([p.pattern_complexity or 0 for p in patterns]),
            'total_document_coverage': sum(p.document_count for p in patterns),
            'average_document_count': np.mean([p.document_count for p in patterns])
        }
    
    def _get_pattern_layout_config(self) -> Dict[str, Any]:
        """Get layout configuration for pattern network"""
        return {
            'algorithm': 'hierarchical',
            'levels': ['pattern', 'document'],
            'spacing': {
                'level_separation': 200,
                'node_separation': 100
            }
        }
    
    def _generate_pattern_legend(self) -> Dict[str, Any]:
        """Generate legend for pattern network"""
        return {
            'pattern_types': [
                {'type': 'cipher', 'color': self._get_pattern_color('cipher'), 'description': 'Cipher patterns'},
                {'type': 'geometric', 'color': self._get_pattern_color('geometric'), 'description': 'Geometric patterns'},
                {'type': 'linguistic', 'color': self._get_pattern_color('linguistic'), 'description': 'Linguistic patterns'}
            ],
            'relationship_types': [
                {'type': 'similar', 'color': self._get_relationship_color('similar'), 'description': 'Similar patterns'},
                {'type': 'complementary', 'color': self._get_relationship_color('complementary'), 'description': 'Complementary patterns'}
            ]
        }
    
    def _create_pattern_timeline_events(self, document_ids: List[int]) -> List[TimelineEvent]:
        """Create timeline events for pattern discoveries"""
        events = []
        
        for doc_id in document_ids:
            instances = self.db.query(CrossPatternInstance).filter(
                CrossPatternInstance.document_id == doc_id
            ).all()
            
            for instance in instances:
                if hasattr(instance, 'detected_at') and instance.detected_at:
                    pattern = instance.cross_pattern
                    
                    event = TimelineEvent(
                        timestamp=instance.detected_at,
                        event_type="pattern_discovery",
                        title=f"Pattern Discovered: {pattern.pattern_type if pattern else 'Unknown'}",
                        description=f"Pattern in document {doc_id} with confidence {instance.confidence:.2f}",
                        document_ids=[doc_id],
                        pattern_ids=[instance.cross_pattern_id],
                        significance=instance.confidence
                    )
                    events.append(event)
        
        return events
    
    def _create_construction_timeline_events(self, document_ids: List[int]) -> List[TimelineEvent]:
        """Create timeline events for shared constructions"""
        events = []
        
        # Get shared constructions involving these documents
        constructions = self.db.query(SharedConstruction).all()
        
        for construction in constructions:
            if construction.document_ids:
                # Check if any of our documents are involved
                involved_docs = [doc_id for doc_id in document_ids if doc_id in construction.document_ids]
                
                if involved_docs and hasattr(construction, 'discovered_at') and construction.discovered_at:
                    event = TimelineEvent(
                        timestamp=construction.discovered_at,
                        event_type="shared_construction",
                        title=f"Shared Construction: {construction.construction_type}",
                        description=f"Construction found across {len(involved_docs)} documents",
                        document_ids=involved_docs,
                        pattern_ids=[],
                        significance=construction.significance_score or 0.5
                    )
                    events.append(event)
        
        return events
    
    def _calculate_timeline_statistics(self, events: List[TimelineEvent]) -> Dict[str, Any]:
        """Calculate timeline statistics"""
        if not events:
            return {}
        
        return {
            'total_events': len(events),
            'event_types': Counter(e.event_type for e in events),
            'time_span_days': (events[-1].timestamp - events[0].timestamp).days,
            'average_significance': np.mean([e.significance for e in events]),
            'events_per_month': self._calculate_events_per_month(events),
            'peak_activity_period': self._find_peak_activity_period(events)
        }
    
    def _calculate_events_per_month(self, events: List[TimelineEvent]) -> Dict[str, int]:
        """Calculate events per month"""
        monthly_counts = defaultdict(int)
        
        for event in events:
            month_key = event.timestamp.strftime('%Y-%m')
            monthly_counts[month_key] += 1
        
        return dict(monthly_counts)
    
    def _find_peak_activity_period(self, events: List[TimelineEvent]) -> Dict[str, Any]:
        """Find the period with peak activity"""
        if not events:
            return {}
        
        # Group events by week
        weekly_counts = defaultdict(int)
        for event in events:
            week_start = event.timestamp - timedelta(days=event.timestamp.weekday())
            week_key = week_start.strftime('%Y-%m-%d')
            weekly_counts[week_key] += 1
        
        if not weekly_counts:
            return {}
        
        peak_week = max(weekly_counts.items(), key=lambda x: x[1])
        
        return {
            'week_start': peak_week[0],
            'event_count': peak_week[1],
            'percentage_of_total': (peak_week[1] / len(events)) * 100
        }
    
    def _get_timeline_config(self) -> Dict[str, Any]:
        """Get timeline visualization configuration"""
        return {
            'time_format': 'YYYY-MM-DD',
            'event_height': 30,
            'lane_height': 50,
            'zoom_levels': ['year', 'month', 'week', 'day'],
            'color_scheme': {
                'document_creation': '#3498db',
                'pattern_discovery': '#e74c3c',
                'shared_construction': '#2ecc71'
            }
        }
    
    def _node_to_dict(self, node: NetworkNode) -> Dict[str, Any]:
        """Convert NetworkNode to dictionary"""
        return {
            'id': node.id,
            'label': node.label,
            'type': node.type,
            'size': node.size,
            'color': node.color,
            'metadata': node.metadata
        }
    
    def _edge_to_dict(self, edge: NetworkEdge) -> Dict[str, Any]:
        """Convert NetworkEdge to dictionary"""
        return {
            'source': edge.source,
            'target': edge.target,
            'weight': edge.weight,
            'type': edge.type,
            'color': edge.color,
            'metadata': edge.metadata
        }
    
    def _timeline_event_to_dict(self, event: TimelineEvent) -> Dict[str, Any]:
        """Convert TimelineEvent to dictionary"""
        return {
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type,
            'title': event.title,
            'description': event.description,
            'document_ids': event.document_ids,
            'pattern_ids': event.pattern_ids,
            'significance': event.significance
        }
    
    # Dashboard helper methods
    
    def _create_document_profile_for_dashboard(self, document_id: int) -> Dict[str, Any]:
        """Create document profile for dashboard"""
        try:
            doc = self.db.query(Document).get(document_id)
            if not doc:
                return {'error': f'Document {document_id} not found'}
            
            # Get pattern statistics
            instances = self.db.query(CrossPatternInstance).filter(
                CrossPatternInstance.document_id == document_id
            ).all()
            
            pattern_types = Counter()
            total_confidence = 0
            
            for instance in instances:
                if instance.cross_pattern:
                    pattern_types[instance.cross_pattern.pattern_type] += 1
                    total_confidence += instance.confidence
            
            avg_confidence = total_confidence / len(instances) if instances else 0
            
            return {
                'document_id': document_id,
                'title': doc.title,
                'created_at': doc.created_at.isoformat() if doc.created_at else None,
                'pattern_count': len(instances),
                'pattern_types': dict(pattern_types),
                'average_confidence': avg_confidence,
                'unique_pattern_types': len(pattern_types)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating document profile: {str(e)}")
            return {'error': str(e)}
    
    def _generate_comparison_matrices(self, document_ids: List[int]) -> Dict[str, Any]:
        """Generate comparison matrices for dashboard"""
        try:
            correlation_matrix = self.relationship_analyzer.generate_correlation_matrix(document_ids)
            
            return {
                'overall_correlation': correlation_matrix.matrix.tolist(),
                'correlation_types': {
                    name: matrix.tolist() 
                    for name, matrix in correlation_matrix.correlation_types.items()
                },
                'significance_matrix': correlation_matrix.significance_matrix.tolist(),
                'document_ids': document_ids,
                'cluster_assignments': correlation_matrix.cluster_assignments
            }
            
        except Exception as e:
            self.logger.error(f"Error generating comparison matrices: {str(e)}")
            return {'error': str(e)}
    
    def _generate_pattern_distributions(self, document_ids: List[int]) -> Dict[str, Any]:
        """Generate pattern distribution data for dashboard"""
        try:
            distributions = {
                'by_type': defaultdict(int),
                'by_document': {},
                'by_complexity': defaultdict(int),
                'by_significance': defaultdict(int)
            }
            
            for doc_id in document_ids:
                instances = self.db.query(CrossPatternInstance).filter(
                    CrossPatternInstance.document_id == doc_id
                ).all()
                
                doc_patterns = defaultdict(int)
                
                for instance in instances:
                    pattern = instance.cross_pattern
                    if pattern:
                        # By type
                        distributions['by_type'][pattern.pattern_type] += 1
                        doc_patterns[pattern.pattern_type] += 1
                        
                        # By complexity
                        if pattern.pattern_complexity:
                            complexity_bin = f"{int(pattern.pattern_complexity * 10) / 10:.1f}"
                            distributions['by_complexity'][complexity_bin] += 1
                        
                        # By significance
                        if pattern.significance_score:
                            sig_bin = f"{int(pattern.significance_score * 10) / 10:.1f}"
                            distributions['by_significance'][sig_bin] += 1
                
                distributions['by_document'][doc_id] = dict(doc_patterns)
            
            # Convert defaultdicts to regular dicts
            return {
                'by_type': dict(distributions['by_type']),
                'by_document': distributions['by_document'],
                'by_complexity': dict(distributions['by_complexity']),
                'by_significance': dict(distributions['by_significance'])
            }
            
        except Exception as e:
            self.logger.error(f"Error generating pattern distributions: {str(e)}")
            return {'error': str(e)}
    
    def _generate_relationship_summary(self, document_ids: List[int]) -> Dict[str, Any]:
        """Generate relationship summary for dashboard"""
        try:
            correlation_matrix = self.relationship_analyzer.generate_correlation_matrix(document_ids)
            
            # Calculate summary statistics
            correlations = []
            n_docs = len(document_ids)
            
            for i in range(n_docs):
                for j in range(i + 1, n_docs):
                    correlations.append(correlation_matrix.matrix[i, j])
            
            if not correlations:
                return {'message': 'No relationships to analyze'}
            
            return {
                'total_relationships': len(correlations),
                'average_correlation': np.mean(correlations),
                'max_correlation': np.max(correlations),
                'min_correlation': np.min(correlations),
                'strong_relationships': len([c for c in correlations if c >= 0.7]),
                'moderate_relationships': len([c for c in correlations if 0.4 <= c < 0.7]),
                'weak_relationships': len([c for c in correlations if c < 0.4]),
                'cluster_count': len(set(correlation_matrix.cluster_assignments.values())),
                'relationship_types': self._analyze_relationship_types(correlation_matrix.correlation_types)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating relationship summary: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_relationship_types(self, correlation_types: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze relationship types for summary"""
        type_analysis = {}
        
        for rel_type, matrix in correlation_types.items():
            # Get upper triangle (avoid double counting)
            n = matrix.shape[0]
            upper_triangle = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    upper_triangle.append(matrix[i, j])
            
            if upper_triangle:
                type_analysis[rel_type] = {
                    'average': np.mean(upper_triangle),
                    'max': np.max(upper_triangle),
                    'strong_count': len([c for c in upper_triangle if c >= 0.7])
                }
        
        return type_analysis
    
    def _generate_key_insights(self, document_ids: List[int]) -> List[Dict[str, Any]]:
        """Generate key insights for dashboard"""
        try:
            insights = []
            
            # Analyze document relationships
            correlation_matrix = self.relationship_analyzer.generate_correlation_matrix(document_ids)
            
            # Insight 1: Strongest relationship
            max_corr = np.max(correlation_matrix.matrix)
            if max_corr > 0.8:
                insights.append({
                    'type': 'strong_relationship',
                    'title': 'Strong Document Relationship Detected',
                    'description': f'Found correlation of {max_corr:.2f} between documents',
                    'significance': 'high',
                    'action': 'Investigate shared patterns and authorship'
                })
            
            # Insight 2: Cluster analysis
            cluster_count = len(set(correlation_matrix.cluster_assignments.values()))
            if cluster_count > 1:
                insights.append({
                    'type': 'clustering',
                    'title': f'{cluster_count} Document Clusters Identified',
                    'description': 'Documents group into distinct clusters based on patterns',
                    'significance': 'medium',
                    'action': 'Analyze cluster characteristics for authorship patterns'
                })
            
            # Insight 3: Pattern diversity
            total_patterns = 0
            unique_types = set()
            
            for doc_id in document_ids:
                instances = self.db.query(CrossPatternInstance).filter(
                    CrossPatternInstance.document_id == doc_id
                ).all()
                total_patterns += len(instances)
                
                for instance in instances:
                    if instance.cross_pattern:
                        unique_types.add(instance.cross_pattern.pattern_type)
            
            if len(unique_types) >= 4:
                insights.append({
                    'type': 'pattern_diversity',
                    'title': 'High Pattern Diversity',
                    'description': f'Found {len(unique_types)} different pattern types across {total_patterns} patterns',
                    'significance': 'medium',
                    'action': 'Investigate sophisticated encoding techniques'
                })
            
            # Insight 4: Shared constructions
            shared_constructions = self.pattern_db.identify_shared_constructions(document_ids)
            if len(shared_constructions) >= 3:
                insights.append({
                    'type': 'shared_knowledge',
                    'title': 'Shared Cryptographic Knowledge',
                    'description': f'Found {len(shared_constructions)} shared constructions',
                    'significance': 'high',
                    'action': 'Analyze for evidence of collaboration or common source'
                })
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating key insights: {str(e)}")
            return [{'type': 'error', 'title': 'Analysis Error', 'description': str(e)}]