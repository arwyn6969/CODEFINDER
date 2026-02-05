
import React, { useState } from 'react';
import { Card, Button, Input, Form, InputNumber, Divider, Row, Col, Typography, Space, Table, Alert, Tag } from 'antd';
import { SearchOutlined, PlusOutlined, DeleteOutlined } from '@ant-design/icons';
import axios from 'axios';

const { Title, Text, Paragraph } = Typography;

const PropheticConvergence = () => {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  // Default triple convergence setup
  const [groups, setGroups] = useState([
    { name: 'PEPE', terms: 'פפי' },
    { name: 'MEME', terms: 'מימי' },
    { name: 'FROG', terms: 'צפר' }
  ]);
  
  const [maxSpread, setMaxSpread] = useState(500);

  const handleAnalyze = async () => {
    try {
      setLoading(true);
      setError('');
      setResults(null);
      
      // Transform groups for API
      const apiTerms = groups.map(g => ({
        name: g.name,
        terms: g.terms.split(',').map(t => t.trim())
      }));
      
      const payload = {
        terms: apiTerms,
        max_spread: maxSpread,
        generate_visual: true
      };
      
      const response = await axios.post('/api/research/prophetic/convergence', payload);
      setResults(response.data);
      
    } catch (err) {
      console.error(err);
      setError('Analysis failed. Ensure server is running and terms are valid Hebrew.');
    } finally {
      setLoading(false);
    }
  };

  const addGroup = () => {
    setGroups([...groups, { name: '', terms: '' }]);
  };

  const removeGroup = (idx) => {
    const newGroups = [...groups];
    newGroups.splice(idx, 1);
    setGroups(newGroups);
  };

  const updateGroup = (idx, field, val) => {
    const newGroups = [...groups];
    newGroups[idx][field] = val;
    setGroups(newGroups);
  };

  // Result Columns
  const columns = [
    { title: 'Book', dataIndex: 'book', key: 'book' },
    { title: 'Position', dataIndex: 'center_index', key: 'center_index', render: (val, record) => `${val} (${record.position_percentage.toFixed(1)}%)` },
    { title: 'Spread', dataIndex: 'spread', key: 'spread', sorter: (a,b) => a.spread - b.spread },
    { 
      title: 'Terms Found', 
      key: 'terms', 
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          {record.terms.map((t, i) => (
            <Tag key={i} color={i===0?'green':i===1?'purple':'blue'}>
              {t.name}: {t.term} (Skip {t.skip})
            </Tag>
          ))}
        </Space>
      )
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>Prophetic Convergence Analyzer</Title>
      <Paragraph>
        Find triple convergence zones where multiple thematic concepts intersect in the Torah.
      </Paragraph>

      <Row gutter={[24, 24]}>
        <Col xs={24} lg={8}>
          <Card title="Analysis Configuration">
            <Form layout="vertical">
              <Form.Item label="Maximum Spread (Letters)">
                <InputNumber 
                  style={{ width: '100%' }} 
                  value={maxSpread} 
                  onChange={setMaxSpread} 
                  min={1} 
                  max={2000} 
                />
              </Form.Item>
              
              <Divider orientation="left">Term Groups</Divider>
              
              {groups.map((group, idx) => (
                <Card size="small" key={idx} style={{ marginBottom: 16 }} bodyStyle={{ padding: 12 }}>
                  <Space style={{ display: 'flex', marginBottom: 8 }} align="baseline">
                    <Input 
                      placeholder="Group Name (e.g. PEPE)" 
                      value={group.name} 
                      onChange={e => updateGroup(idx, 'name', e.target.value)} 
                    />
                    <Button 
                      danger 
                      icon={<DeleteOutlined />} 
                      onClick={() => removeGroup(idx)} 
                      disabled={groups.length <= 2}
                    />
                  </Space>
                  <Input 
                    placeholder="Hebrew Terms (comma iter)" 
                    value={group.terms} 
                    onChange={e => updateGroup(idx, 'terms', e.target.value)} 
                  />
                </Card>
              ))}
              
              <Button type="dashed" onClick={addGroup} block icon={<PlusOutlined />} style={{ marginBottom: 16 }}>
                Add Term Group
              </Button>
              
              <Button 
                type="primary" 
                block 
                size="large" 
                icon={<SearchOutlined />} 
                onClick={handleAnalyze} 
                loading={loading}
              >
                Analyze Convergence
              </Button>
            </Form>
          </Card>
        </Col>

        <Col xs={24} lg={16}>
          {error && <Alert message={error} type="error" showIcon style={{ marginBottom: 16 }} />}
          
          {!results && !loading && (
            <Card style={{ textAlign: 'center', color: '#999', paddingTop: 40, paddingBottom: 40 }}>
              <p>Configure terms and click Analyze to see convergence zones.</p>
            </Card>
          )}

          {results && (
            <Space direction="vertical" style={{ width: '100%' }} size="large">
              <Card title={`Found ${results.total_zones_found} Convergence Zones`}>
                 <Table 
                   dataSource={results.top_zones} 
                   columns={columns} 
                   rowKey="center_index" 
                   pagination={false}
                   size="small"
                 />
              </Card>

              {results.top_zones.length > 0 && results.top_zones[0].visualization_svg && (
                <Card title="Best Convergence Visualization">
                  <div 
                    style={{ 
                      background: '#15151e', 
                      padding: 20, 
                      borderRadius: 8, 
                      textAlign: 'center',
                      overflowX: 'auto'
                    }}
                    dangerouslySetInnerHTML={{ __html: results.top_zones[0].visualization_svg }} 
                  />
                  <div style={{ marginTop: 12, textAlign: 'center' }}>
                    <Text type="secondary">
                       Convergence at position {results.top_zones[0].center_index} (Book: {results.top_zones[0].book})
                    </Text>
                  </div>
                </Card>
              )}
            </Space>
          )}
        </Col>
      </Row>
    </div>
  );
};

export default PropheticConvergence;
