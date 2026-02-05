import React, { useState, useEffect } from 'react';
import { Layout, Table, Card, Input, Tabs, Tag, message } from 'antd';
import { BookOutlined, FileSearchOutlined, RobotOutlined } from '@ant-design/icons';
import researchService from '../../../services/researchService';

const { Content } = Layout;
const { TabPane } = Tabs;
const { Search } = Input;

const TheLibrary = () => {
    const [patterns, setPatterns] = useState([]);
    const [loading, setLoading] = useState(false);
    
    useEffect(() => {
        fetchPatterns();
    }, []);

    const fetchPatterns = async (type = null) => {
        setLoading(true);
        try {
            const params = type ? { pattern_type: type } : {};
            const data = await researchService.listAllPatterns(params);
            setPatterns(data || []);
        } catch (error) {
            message.error("Failed to load patterns library");
        } finally {
            setLoading(false);
        }
    };

    const columns = [
        {
            title: 'ID',
            dataIndex: 'id',
            key: 'id',
            width: 70,
        },
        {
            title: 'Pattern Name',
            dataIndex: 'pattern_name',
            key: 'name',
            render: (text) => <span style={{ fontWeight: 'bold', color: '#fff' }}>{text || 'Unnamed Pattern'}</span>
        },
        {
            title: 'Type',
            dataIndex: 'pattern_type',
            key: 'type',
            render: (type) => {
                let color = 'geekblue';
                if (type === 'gematria_match') color = 'gold';
                if (type === 'els_match') color = 'purple';
                if (type === 'geometric') color = 'cyan';
                return <Tag color={color}>{type.toUpperCase()}</Tag>;
            }
        },
        {
            title: 'Doc ID',
            dataIndex: 'document_id',
            key: 'docId',
            width: 80
        },
        {
            title: 'Description',
            dataIndex: 'description',
            key: 'desc',
            ellipsis: true
        },
        {
            title: 'Confidence',
            dataIndex: 'confidence',
            key: 'conf',
            render: (val) => <span style={{ color: val > 0.8 ? '#52c41a' : '#faad14' }}>{(val * 100).toFixed(0)}%</span>
        }
    ];

    return (
        <Layout style={{ height: 'calc(100vh - 120px)', background: '#000' }}>
            <Content style={{ padding: 24 }}>
                <Card style={{ background: '#1f1f1f', border: 'none', height: '100%' }} bodyStyle={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'space-between' }}>
                        <h2 style={{ color: '#fff', margin: 0 }}><BookOutlined /> The Library</h2>
                        <Search placeholder="Search patterns..." style={{ width: 300 }} onSearch={val => message.info("Client-side search to be implemented")} />
                    </div>
                    
                    <Tabs defaultActiveKey="all" onChange={(key) => fetchPatterns(key === 'all' ? null : key)}>
                        <TabPane tab={<span><FileSearchOutlined /> All Patterns</span>} key="all" />
                        <TabPane tab="ELS Matches" key="els_match" />
                        <TabPane tab="Gematria" key="gematria_match" />
                        <TabPane tab={<span><RobotOutlined /> Authorship Profiles</span>} key="authorship">
                            {/* Placeholder for Authorship Radar Charts */}
                            <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>
                                <RobotOutlined style={{ fontSize: 48, marginBottom: 16 }} />
                                <p>Authorship Analysis and Radar Charts coming in Phase 3.</p>
                            </div>
                        </TabPane>
                    </Tabs>

                    <Table 
                        dataSource={patterns} 
                        columns={columns} 
                        rowKey="id"
                        loading={loading}
                        pagination={{ pageSize: 10 }}
                        scroll={{ y: 'calc(100vh - 350px)' }}
                        style={{ marginTop: 16 }}
                    />
                </Card>
            </Content>
        </Layout>
    );
};

export default TheLibrary;
