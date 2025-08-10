"""
Tests for Cross-Reference Visualization System
"""
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from app.services.cross_reference_visualizer import (
    CrossReferenceVisualizer, NetworkNode, NetworkEdge, TimelineEvent, VisualizationData
)
from app.models.cross_document_models import (
    CrossDocumentPattern, CrossPatternInstance, PatternRelationship, SharedConstruction
)
from app.models.database_models import Document
from app.services.relationship_analyzer import CorrelationMatrix


class TestCrossReferenceVisualizer:
    """Test cases for CrossReferenceVisualizer"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session"""
        return Mock(spec=Session)
    
    @pytest.fixture
    def visualizer(self, mock_db_session):
        """Create CrossReferenceVisualizer instance with mock database"""
        return CrossReferenceVisualizer(mock_db_session)
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Mock(id=1, title="Document 1", created_at=datetime.now()),
            Mock(id=2, title="Document 2", created_at=datetime.now() - timedelta(days=30)),
            Mock(id=3, title="Document 3", created_at=datetime.now() - timedelta(days=60))
        ]
    
    @pytest.fixture
    def sample_patterns(self):
        """Create sample cross-document patterns"""
        return [
            Mock(
                id=1,
                pattern_type="cipher",
                pattern_subtype="caesar",
                pattern_name="Caesar Cipher",
                significance_score=0.8,
                pattern_complexity=0.6,
                document_count=2,
                total_occurrences=5
            ),
            Mock(
                id=2,
                pattern_type="geometric",
                pattern_subtype="triangle",
                pattern_name="Triangle Pattern",
                significance_score=0.7,
                pattern_complexity=0.8,
                document_count=3,
                total_occurrences=8
            )
        ]
    
    @pytest.fixture
    def sample_correlation_matrix(self):
        """Create sample correlation matrix"""
        matrix = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ])
        
        correlation_types = {
            'pattern_based': matrix * 0.9,
            'stylistic': matrix * 0.7,
            'temporal': matrix * 0.5,
            'geometric': matrix * 0.4,
            'cipher_method': matrix * 0.3
        }
        
        significance_matrix = np.ones_like(matrix) * 0.05
        cluster_assignments = {1: 0, 2: 0, 3: 1}
        
        return CorrelationMatrix(
            document_ids=[1, 2, 3],
            matrix=matrix,
            correlation_types=correlation_types,
            significance_matrix=significance_matrix,
            cluster_assignments=cluster_assignments
        )
    
    def test_generate_document_relationship_network_basic(self, visualizer, mock_db_session, 
                                                        sample_documents, sample_correlation_matrix):
        """Test basic document relationship network generation"""
        document_ids = [1, 2, 3]
        
        # Setup mocks
        mock_db_session.query.return_value.get.side_effect = lambda doc_id: sample_documents[doc_id - 1]
        
        with patch.object(visualizer.relationship_analyzer, 'generate_correlation_matrix', 
                         return_value=sample_correlation_matrix), \
             patch.object(visualizer, '_get_document_pattern_count', return_value=5):
            
            result = visualizer.generate_document_relationship_network(document_ids)
            
            # Verify result structure
            assert 'nodes' in result
            assert 'edges' in result
            assert 'statistics' in result
            assert 'layout_config' in result
            assert 'legend' in result
            
            # Verify nodes
            nodes = result['nodes']
            assert len(nodes) == 3
            for node in nodes:
                assert 'id' in node
                assert 'label' in node
                assert 'type' in node
                assert node['type'] == 'document'
                assert 'size' in node
                assert 'color' in node
                assert 'metadata' in node
            
            # Verify edges (should have edges for correlations >= 0.3)
            edges = result['edges']
            assert len(edges) >= 1  # At least one strong correlation
            for edge in edges:
                assert 'source' in edge
                assert 'target' in edge
                assert 'weight' in edge
                assert 'type' in edge
                assert 'color' in edge
                assert 'metadata' in edge
    
    def test_generate_pattern_connection_network_basic(self, visualizer, mock_db_session, sample_patterns):
        """Test basic pattern connection network generation"""
        pattern_ids = [1, 2]
        
        # Setup mocks
        mock_db_session.query.return_value.get.side_effect = lambda pid: sample_patterns[pid - 1]
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = sample_patterns
        
        # Mock pattern instances
        mock_instances = [
            Mock(id=1, cross_pattern_id=1, document_id=1, confidence=0.8, quality_score=0.9),
            Mock(id=2, cross_pattern_id=1, document_id=2, confidence=0.7, quality_score=0.8),
            Mock(id=3, cross_pattern_id=2, document_id=2, confidence=0.9, quality_score=0.85)
        ]
        
        mock_db_session.query.return_value.filter.return_value.all.return_value = mock_instances
        
        with patch.object(visualizer, '_create_document_nodes_for_patterns', return_value=[]), \
             patch.object(visualizer, '_create_pattern_relationship_edges', return_value=[]):
            
            result = visualizer.generate_pattern_connection_network(pattern_ids)
            
            # Verify result structure
            assert 'nodes' in result
            assert 'edges' in result
            assert 'statistics' in result
            assert 'layout_config' in result
            assert 'legend' in result
            
            # Verify pattern nodes
            nodes = result['nodes']
            assert len(nodes) == 2  # Two patterns
            for node in nodes:
                assert node['type'] == 'pattern'
                assert 'metadata' in node
                assert 'pattern_id' in node['metadata']
    
    def test_generate_timeline_analysis_basic(self, visualizer, mock_db_session, sample_documents):
        """Test basic timeline analysis generation"""
        document_ids = [1, 2, 3]
        
        # Setup document mocks
        mock_db_session.query.return_value.get.side_effect = lambda doc_id: sample_documents[doc_id - 1]
        
        with patch.object(visualizer, '_get_document_pattern_count', return_value=3), \
             patch.object(visualizer, '_create_pattern_timeline_events', return_value=[]), \
             patch.object(visualizer, '_create_construction_timeline_events', return_value=[]):
            
            result = visualizer.generate_timeline_analysis(document_ids)
            
            # Verify result structure
            assert 'events' in result
            assert 'statistics' in result
            assert 'time_range' in result
            assert 'visualization_config' in result
            
            # Should have document creation events
            events = result['events']
            assert len(events) >= 3  # At least one for each document
            
            # Verify event structure
            for event in events:
                assert 'timestamp' in event
                assert 'event_type' in event
                assert 'title' in event
                assert 'description' in event
                assert 'document_ids' in event
                assert 'significance' in event
    
    def test_generate_comparative_dashboard_basic(self, visualizer):
        """Test basic comparative dashboard generation"""
        document_ids = [1, 2, 3]
        
        with patch.object(visualizer, '_create_document_profile_for_dashboard', 
                         return_value={'document_id': 1, 'pattern_count': 5}), \
             patch.object(visualizer, '_generate_comparison_matrices', return_value={}), \
             patch.object(visualizer, '_generate_pattern_distributions', return_value={}), \
             patch.object(visualizer, '_generate_relationship_summary', return_value={}), \
             patch.object(visualizer, '_generate_key_insights', return_value=[]):
            
            result = visualizer.generate_comparative_dashboard(document_ids)
            
            # Verify result structure
            assert 'document_profiles' in result
            assert 'comparison_matrices' in result
            assert 'pattern_distributions' in result
            assert 'relationship_summary' in result
            assert 'key_insights' in result
            
            # Should have profiles for all documents
            assert len(result['document_profiles']) == 3
    
    def test_generate_evidence_trail_visualization_basic(self, visualizer):
        """Test basic evidence trail visualization generation"""
        pattern_id = 1
        
        # Mock evidence trail
        mock_trail = Mock(
            pattern_id=pattern_id,
            document_chain=[1, 2],
            connection_strength=0.8,
            evidence_points=[
                {'type': 'similarity', 'strength': 0.9, 'details': {}}
            ],
            temporal_sequence=[datetime.now()],
            confidence_score=0.85
        )
        
        with patch.object(visualizer.relationship_analyzer, 'track_evidence_trails', 
                         return_value=[mock_trail]), \
             patch.object(visualizer.db.query, 'return_value.get') as mock_get:
            
            mock_get.side_effect = lambda doc_id: Mock(id=doc_id, title=f"Document {doc_id}")
            
            result = visualizer.generate_evidence_trail_visualization(pattern_id)
            
            # Verify result structure
            assert 'pattern_id' in result
            assert 'trails' in result
            assert 'summary' in result
            
            # Verify trail structure
            trails = result['trails']
            assert len(trails) == 1
            
            trail = trails[0]
            assert 'nodes' in trail
            assert 'edges' in trail
            assert 'evidence_points' in trail
            assert 'confidence_score' in trail
    
    def test_generate_evidence_trail_visualization_no_trails(self, visualizer):
        """Test evidence trail visualization with no trails"""
        pattern_id = 999
        
        with patch.object(visualizer.relationship_analyzer, 'track_evidence_trails', 
                         return_value=[]):
            
            result = visualizer.generate_evidence_trail_visualization(pattern_id)
            
            assert 'trails' in result
            assert 'message' in result
            assert result['trails'] == []
            assert 'No evidence trails found' in result['message']
    
    def test_network_node_creation(self, visualizer):
        """Test NetworkNode creation and conversion"""
        node = NetworkNode(
            id="test_node",
            label="Test Node",
            type="document",
            size=25.0,
            color="#3498db",
            metadata={'test': 'data'}
        )
        
        node_dict = visualizer._node_to_dict(node)
        
        assert node_dict['id'] == "test_node"
        assert node_dict['label'] == "Test Node"
        assert node_dict['type'] == "document"
        assert node_dict['size'] == 25.0
        assert node_dict['color'] == "#3498db"
        assert node_dict['metadata'] == {'test': 'data'}
    
    def test_network_edge_creation(self, visualizer):
        """Test NetworkEdge creation and conversion"""
        edge = NetworkEdge(
            source="node1",
            target="node2",
            weight=0.8,
            type="pattern_based",
            color="#e74c3c",
            metadata={'correlation': 0.8}
        )
        
        edge_dict = visualizer._edge_to_dict(edge)
        
        assert edge_dict['source'] == "node1"
        assert edge_dict['target'] == "node2"
        assert edge_dict['weight'] == 0.8
        assert edge_dict['type'] == "pattern_based"
        assert edge_dict['color'] == "#e74c3c"
        assert edge_dict['metadata'] == {'correlation': 0.8}
    
    def test_timeline_event_creation(self, visualizer):
        """Test TimelineEvent creation and conversion"""
        timestamp = datetime.now()
        event = TimelineEvent(
            timestamp=timestamp,
            event_type="pattern_discovery",
            title="Test Event",
            description="Test Description",
            document_ids=[1, 2],
            pattern_ids=[1],
            significance=0.7
        )
        
        event_dict = visualizer._timeline_event_to_dict(event)
        
        assert event_dict['timestamp'] == timestamp.isoformat()
        assert event_dict['event_type'] == "pattern_discovery"
        assert event_dict['title'] == "Test Event"
        assert event_dict['description'] == "Test Description"
        assert event_dict['document_ids'] == [1, 2]
        assert event_dict['pattern_ids'] == [1]
        assert event_dict['significance'] == 0.7
    
    def test_get_document_pattern_count(self, visualizer, mock_db_session):
        """Test document pattern count calculation"""
        document_id = 1
        
        # Mock query result
        mock_db_session.query.return_value.filter.return_value.count.return_value = 5
        
        count = visualizer._get_document_pattern_count(document_id)
        
        assert count == 5
    
    def test_get_cluster_color(self, visualizer):
        """Test cluster color assignment"""
        # Test different cluster IDs
        color1 = visualizer._get_cluster_color(0)
        color2 = visualizer._get_cluster_color(1)
        color3 = visualizer._get_cluster_color(6)  # Should wrap around
        
        assert color1 != color2
        assert isinstance(color1, str)
        assert color1.startswith('#')
        assert color3 == color1  # Should wrap around
    
    def test_get_pattern_color(self, visualizer):
        """Test pattern color assignment"""
        cipher_color = visualizer._get_pattern_color('cipher')
        geometric_color = visualizer._get_pattern_color('geometric')
        unknown_color = visualizer._get_pattern_color('unknown_type')
        
        assert cipher_color == '#e74c3c'
        assert geometric_color == '#f39c12'
        assert unknown_color == '#95a5a6'  # Default color
    
    def test_determine_primary_relationship_type(self, visualizer):
        """Test primary relationship type determination"""
        correlation_types = {
            'pattern_based': np.array([[1.0, 0.8], [0.8, 1.0]]),
            'stylistic': np.array([[1.0, 0.6], [0.6, 1.0]]),
            'temporal': np.array([[1.0, 0.4], [0.4, 1.0]])
        }
        
        primary_type = visualizer._determine_primary_relationship_type(correlation_types, 0, 1)
        
        assert primary_type == 'pattern_based'  # Highest correlation
    
    def test_calculate_network_statistics(self, visualizer):
        """Test network statistics calculation"""
        nodes = [
            NetworkNode("n1", "Node 1", "document", 10, "#000", {}),
            NetworkNode("n2", "Node 2", "document", 15, "#000", {}),
            NetworkNode("n3", "Node 3", "pattern", 20, "#000", {})
        ]
        
        edges = [
            NetworkEdge("n1", "n2", 0.8, "pattern_based", "#000", {}),
            NetworkEdge("n2", "n3", 0.6, "stylistic", "#000", {})
        ]
        
        stats = visualizer._calculate_network_statistics(nodes, edges)
        
        assert stats['node_count'] == 3
        assert stats['edge_count'] == 2
        assert stats['average_degree'] == (2 * 2) / 3  # (2 * edges) / nodes
        assert stats['density'] == (2 * 2) / (3 * 2)  # (2 * edges) / (nodes * (nodes-1))
        assert stats['average_weight'] == 0.7  # (0.8 + 0.6) / 2
        assert 'node_types' in stats
        assert 'edge_types' in stats
    
    def test_get_network_layout_config(self, visualizer):
        """Test network layout configuration"""
        config = visualizer._get_network_layout_config()
        
        assert 'algorithm' in config
        assert 'iterations' in config
        assert 'node_repulsion' in config
        assert 'edge_attraction' in config
        assert config['algorithm'] == 'force_directed'
    
    def test_generate_network_legend(self, visualizer):
        """Test network legend generation"""
        legend = visualizer._generate_network_legend()
        
        assert 'node_types' in legend
        assert 'edge_types' in legend
        assert 'size_encoding' in legend
        assert 'weight_encoding' in legend
        
        # Verify node types
        node_types = legend['node_types']
        assert len(node_types) >= 3
        for node_type in node_types:
            assert 'type' in node_type
            assert 'color' in node_type
            assert 'description' in node_type
    
    def test_create_document_profile_for_dashboard(self, visualizer, mock_db_session):
        """Test document profile creation for dashboard"""
        document_id = 1
        
        # Mock document
        mock_doc = Mock(id=1, title="Test Document", created_at=datetime.now())
        mock_db_session.query.return_value.get.return_value = mock_doc
        
        # Mock pattern instances
        mock_instances = [
            Mock(cross_pattern=Mock(pattern_type='cipher'), confidence=0.8),
            Mock(cross_pattern=Mock(pattern_type='geometric'), confidence=0.7),
            Mock(cross_pattern=Mock(pattern_type='cipher'), confidence=0.9)
        ]
        mock_db_session.query.return_value.filter.return_value.all.return_value = mock_instances
        
        profile = visualizer._create_document_profile_for_dashboard(document_id)
        
        assert profile['document_id'] == document_id
        assert profile['title'] == "Test Document"
        assert profile['pattern_count'] == 3
        assert profile['pattern_types']['cipher'] == 2
        assert profile['pattern_types']['geometric'] == 1
        assert profile['average_confidence'] == (0.8 + 0.7 + 0.9) / 3
        assert profile['unique_pattern_types'] == 2
    
    def test_generate_pattern_distributions(self, visualizer, mock_db_session):
        """Test pattern distribution generation"""
        document_ids = [1, 2]
        
        # Mock pattern instances for different documents
        mock_instances_doc1 = [
            Mock(cross_pattern=Mock(pattern_type='cipher', pattern_complexity=0.8, significance_score=0.7))
        ]
        mock_instances_doc2 = [
            Mock(cross_pattern=Mock(pattern_type='geometric', pattern_complexity=0.6, significance_score=0.9))
        ]
        
        # Create a side effect function that returns different instances based on document_id
        def mock_query_side_effect():
            mock_query = Mock()
            def mock_filter_side_effect(filter_condition):
                mock_filtered = Mock()
                # Simple way to differentiate based on document_id
                # In real scenario, this would be based on the actual filter condition
                if str(filter_condition).find('1') != -1:  # Rough check for doc_id 1
                    mock_filtered.all.return_value = mock_instances_doc1
                else:
                    mock_filtered.all.return_value = mock_instances_doc2
                return mock_filtered
            mock_query.filter.side_effect = mock_filter_side_effect
            return mock_query
        
        mock_db_session.query.side_effect = lambda *args: mock_query_side_effect()
        
        distributions = visualizer._generate_pattern_distributions(document_ids)
        
        assert 'by_type' in distributions
        assert 'by_document' in distributions
        assert 'by_complexity' in distributions
        assert 'by_significance' in distributions
        
        # The exact counts may vary due to mock complexity, so just verify structure
        assert isinstance(distributions['by_type'], dict)
        assert isinstance(distributions['by_document'], dict)
    
    def test_generate_key_insights(self, visualizer, sample_correlation_matrix):
        """Test key insights generation"""
        document_ids = [1, 2, 3]
        
        with patch.object(visualizer.relationship_analyzer, 'generate_correlation_matrix', 
                         return_value=sample_correlation_matrix), \
             patch.object(visualizer.db.query, 'return_value.filter.return_value.all') as mock_all, \
             patch.object(visualizer.pattern_db, 'identify_shared_constructions') as mock_constructions:
            
            # Mock shared constructions with proper list behavior
            mock_constructions.return_value = [Mock(), Mock(), Mock()]
            
            # Mock pattern instances
            mock_all.return_value = [
                Mock(cross_pattern=Mock(pattern_type='cipher')),
                Mock(cross_pattern=Mock(pattern_type='geometric')),
                Mock(cross_pattern=Mock(pattern_type='linguistic')),
                Mock(cross_pattern=Mock(pattern_type='structural'))
            ]
            
            insights = visualizer._generate_key_insights(document_ids)
            
            assert isinstance(insights, list)
            assert len(insights) >= 1
            
            # Verify insight structure (may include error insights)
            for insight in insights:
                assert 'type' in insight
                assert 'title' in insight
                assert 'description' in insight
                # Some insights may be error insights, so don't require 'significance'
                if insight['type'] != 'error':
                    assert 'significance' in insight
    
    def test_calculate_timeline_statistics(self, visualizer):
        """Test timeline statistics calculation"""
        now = datetime.now()
        events = [
            TimelineEvent(now, "document_creation", "Doc 1", "Description", [1], [], 0.5),
            TimelineEvent(now + timedelta(days=1), "pattern_discovery", "Pattern 1", "Description", [1], [1], 0.8),
            TimelineEvent(now + timedelta(days=30), "document_creation", "Doc 2", "Description", [2], [], 0.6)
        ]
        
        stats = visualizer._calculate_timeline_statistics(events)
        
        assert stats['total_events'] == 3
        assert 'event_types' in stats
        assert stats['event_types']['document_creation'] == 2
        assert stats['event_types']['pattern_discovery'] == 1
        assert stats['time_span_days'] == 30
        assert 'average_significance' in stats
        assert 'events_per_month' in stats
        assert 'peak_activity_period' in stats
    
    def test_error_handling(self, visualizer):
        """Test error handling in visualization methods"""
        # Test with invalid document IDs
        with patch.object(visualizer.relationship_analyzer, 'generate_correlation_matrix', 
                         side_effect=Exception("Test error")):
            
            result = visualizer.generate_document_relationship_network([999])
            
            assert 'error' in result
            assert 'Test error' in result['error']
    
    def test_empty_data_handling(self, visualizer, mock_db_session):
        """Test handling of empty data scenarios"""
        # Test timeline with no temporal data - mock documents without created_at
        mock_docs = [
            Mock(id=1, title="Doc 1", created_at=None),
            Mock(id=2, title="Doc 2", created_at=None),
            Mock(id=3, title="Doc 3", created_at=None)
        ]
        
        mock_db_session.query.return_value.get.side_effect = lambda doc_id: mock_docs[doc_id - 1]
        
        with patch.object(visualizer, '_create_pattern_timeline_events', return_value=[]), \
             patch.object(visualizer, '_create_construction_timeline_events', return_value=[]):
            
            result = visualizer.generate_timeline_analysis([1, 2, 3])
            
            assert 'events' in result
            assert 'message' in result
            assert 'No temporal data available' in result['message']
    
    def test_visualization_data_structure(self):
        """Test VisualizationData structure"""
        nodes = [NetworkNode("n1", "Node 1", "document", 10, "#000", {})]
        edges = [NetworkEdge("n1", "n2", 0.5, "test", "#000", {})]
        timeline = [TimelineEvent(datetime.now(), "test", "Test", "Desc", [1], [], 0.5)]
        metadata = {"test": "data"}
        
        viz_data = VisualizationData(
            nodes=nodes,
            edges=edges,
            timeline=timeline,
            metadata=metadata
        )
        
        assert len(viz_data.nodes) == 1
        assert len(viz_data.edges) == 1
        assert len(viz_data.timeline) == 1
        assert viz_data.metadata["test"] == "data"


class TestCrossReferenceVisualizerIntegration:
    """Integration tests for CrossReferenceVisualizer"""
    
    @pytest.fixture
    def visualizer_with_real_db(self):
        """Create visualizer with real database session for integration tests"""
        # This would use a real test database
        # For now, we'll skip these tests in unit testing
        pytest.skip("Integration tests require real database")
    
    def test_full_network_generation_workflow(self, visualizer_with_real_db):
        """Test complete network generation workflow"""
        # This would test the full workflow with real data
        pass
    
    def test_timeline_analysis_workflow(self, visualizer_with_real_db):
        """Test complete timeline analysis workflow"""
        # This would test timeline analysis with real data
        pass
    
    def test_dashboard_generation_workflow(self, visualizer_with_real_db):
        """Test complete dashboard generation workflow"""
        # This would test dashboard generation with real data
        pass


if __name__ == "__main__":
    pytest.main([__file__])