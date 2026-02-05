import React, { useState, useEffect } from 'react';
import { Layout, Button, Card, Spin, message, Statistic, Row, Col, Empty } from 'antd';
import { ReloadOutlined, ClusterOutlined, ShareAltOutlined } from '@ant-design/icons';
import ForceGraph2D from 'react-force-graph-2d';
import researchService from '../../../services/researchService';

const { Content, Sider } = Layout;

const TheMap = ({ onOpenDocument }) => {
    const [loading, setLoading] = useState(false);
    const [networkData, setNetworkData] = useState({ nodes: [], links: [] });
    const [metrics, setMetrics] = useState({});
    const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

    useEffect(() => {
        // Adjust dimensions to fit container
        const updateDimensions = () => {
             const container = document.getElementById('map-container');
             if (container) {
                 setDimensions({
                     width: container.offsetWidth,
                     height: container.offsetHeight
                 });
             }
        };

        window.addEventListener('resize', updateDimensions);
        updateDimensions();
        
        // Initial Fetch
        fetchNetwork();

        return () => window.removeEventListener('resize', updateDimensions);
    }, []);

    const fetchNetwork = async () => {
        setLoading(true);
        try {
            // 1. Get all documents first
            const docsData = await researchService.listDocuments();
            const allDocs = docsData.documents || [];
            
            if (allDocs.length < 2) {
                message.warning("Need at least 2 documents for network analysis");
                setNetworkData({ nodes: [], links: [] });
                return;
            }

            const docIds = allDocs.map(d => d.id);
            
            // 2. Get Network
            const data = await researchService.getRelationshipNetwork(docIds);
            
            // Transform data for react-force-graph
            // API returns: network: { nodes: [], edges: [] }
            // Graph expects: { nodes, links } where links have source/target
            
            const graphData = {
                nodes: data.network.nodes.map(n => ({ 
                    id: n.id, 
                    name: n.title, 
                    val: n.centrality * 10, // radius based on centrality
                    group: n.community 
                })),
                links: data.network.edges.map(e => ({
                    source: e.source,
                    target: e.target,
                    value: e.weight
                }))
            };
            
            setNetworkData(graphData);
            setMetrics(data.metrics || {});
            
        } catch (error) {
            console.error("Network fetch error", error);
            message.error("Failed to load relationship network");
        } finally {
            setLoading(false);
        }
    };

    return (
        <Layout style={{ height: 'calc(100vh - 120px)', background: '#000' }}>
            <Content id="map-container" style={{ position: 'relative', overflow: 'hidden' }}>
                {loading ? (
                    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                        <Spin size="large" tip="Calculating Relationships..." />
                    </div>
                ) : networkData.nodes.length === 0 ? (
                    <Empty 
                        image={Empty.PRESENTED_IMAGE_SIMPLE} 
                        description={<span style={{ color: '#666' }}>Not enough data for network visualization</span>} 
                        style={{ marginTop: 100 }}
                    >
                        <Button type="primary" onClick={fetchNetwork}>Retry</Button>
                    </Empty>
                ) : (
                    <ForceGraph2D
                        width={dimensions.width}
                        height={dimensions.height}
                        graphData={networkData}
                        nodeLabel="name"
                        nodeAutoColorBy="group"
                        linkDirectionalParticles={2}
                        linkDirectionalParticleSpeed={d => d.value * 0.001}
                        backgroundColor="#000000"
                        onNodeClick={node => {
                            if (onOpenDocument) {
                                onOpenDocument(node.id);
                                message.info(`Opening ${node.name}`);
                            }
                        }}
                        nodeCanvasObject={(node, ctx, globalScale) => {
                            const label = node.name;
                            const fontSize = 12/globalScale;
                            ctx.font = `${fontSize}px Sans-Serif`;
                            const textWidth = ctx.measureText(label).width;
                            const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2); // some padding

                            ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
                            ctx.fillRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] / 2, ...bckgDimensions);

                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillStyle = node.color;
                            ctx.fillText(label, node.x, node.y);
                            
                            node.__bckgDimensions = bckgDimensions; // to re-use in nodePointerAreaPaint
                        }}
                    />
                )}
                
                <div style={{ position: 'absolute', top: 20, right: 20 }}>
                    <Button icon={<ReloadOutlined />} onClick={fetchNetwork}>Refresh</Button>
                </div>
            </Content>
            
            <Sider width={300} style={{ background: '#141414', borderLeft: '1px solid #333', padding: 16 }}>
                <h3 style={{ color: '#fff' }}><ShareAltOutlined /> Network Stats</h3>
                <Row gutter={[16, 16]}>
                    <Col span={12}>
                        <Card size="small" style={{ background: '#222', borderColor: '#444' }}>
                            <Statistic title={<span style={{ color: '#888' }}>Nodes</span>} value={networkData.nodes.length} valueStyle={{ color: '#fff' }} />
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card size="small" style={{ background: '#222', borderColor: '#444' }}>
                            <Statistic title={<span style={{ color: '#888' }}>Edges</span>} value={networkData.links.length} valueStyle={{ color: '#fff' }} />
                        </Card>
                    </Col>
                    <Col span={24}>
                         <Card size="small" style={{ background: '#222', borderColor: '#444' }}>
                            <Statistic title={<span style={{ color: '#888' }}>Density</span>} value={metrics.density || 0} precision={4} valueStyle={{ color: '#52c41a' }} />
                        </Card>
                    </Col>
                     <Col span={24}>
                         <Card size="small" style={{ background: '#222', borderColor: '#444' }}>
                            <Statistic title={<span style={{ color: '#888' }}>Modularity</span>} value={metrics.modularity || 0} precision={4} valueStyle={{ color: '#1890ff' }} />
                        </Card>
                    </Col>
                </Row>
                
                <div style={{ marginTop: 24, color: '#aaa' }}>
                   <p><ClusterOutlined /> Detected Communities: {metrics.community_count || 0}</p>
                </div>
            </Sider>
        </Layout>
    );
};

export default TheMap;
