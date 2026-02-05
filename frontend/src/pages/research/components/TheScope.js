import React, { useState } from 'react';
import { Card, Input, Button, Table, Row, Col, Switch, Statistic, Tag, message, Tabs, Space, Modal, Spin } from 'antd';
import { SearchOutlined, RadarChartOutlined, EyeOutlined } from '@ant-design/icons';
import researchService from '../../../services/researchService';

const { TabPane } = Tabs;

const TheScope = () => {
  const [activeTab, setActiveTab] = useState('els');

  // Simple ELS State
  const [searchText, setSearchText] = useState('');
  const [loading, setLoading] = useState(false);
  const [elsResults, setElsResults] = useState([]);
  const [autoTransliterate, setAutoTransliterate] = useState(true);
  
  // Stats State
  const [stats, setStats] = useState({ totalLength: 0, foundCount: 0 });
  // Matrix View State
  const [matrixVisible, setMatrixVisible] = useState(false);
  const [matrixData, setMatrixData] = useState(null);
  const [matrixLoading, setMatrixLoading] = useState(false);

  // Prophetic State
  const [propheticTerms, setPropheticTerms] = useState([{ name: 'Term 1', term: '' }, { name: 'Term 2', term: '' }, { name: 'Term 3', term: '' }]);
  const [convergenceZones, setConvergenceZones] = useState([]);
  const [convergenceLoading, setConvergenceLoading] = useState(false);

  const handlePropheticSearch = async () => {
    // Validate inputs
    const validTerms = propheticTerms.filter(t => t.term.trim() !== '');
    if (validTerms.length < 3) {
      message.warning('Please enter at least 3 terms for convergence analysis');
      return;
    }

    setConvergenceLoading(true);
    try {
      // Auto-transliterate terms if they look English
      const processedTerms = await Promise.all(validTerms.map(async (t) => {
        // Simple check: if has latin chars, transliterate
        if (/[a-zA-Z]/.test(t.term)) {
           const variants = await researchService.transliterate(t.term);
           return { name: t.name, terms: [variants[0]?.hebrew || t.term] }; // Just take top variant for simplicity in this mode
        }
        return { name: t.name, terms: [t.term] };
      }));

      const data = await researchService.findPropheticConvergence({
        terms: processedTerms,
        maxSpread: 500,
        generateVisual: true
      });

      setConvergenceZones(data.top_zones);
      message.success(`Found ${data.total_zones_found} convergence zones!`);

    } catch (error) {
       message.error('Convergence analysis failed: ' + error.message);
    } finally {
       setConvergenceLoading(false);
    }
  };

  const handleTermChange = (index, value) => {
    const newTerms = [...propheticTerms];
    newTerms[index].term = value;
    setPropheticTerms(newTerms);
  };

  const handleSearch = async () => {
    if (!searchText) {
      message.warning('Please enter a term to search');
      return;
    }

    setLoading(true);
    try {
      // 1. Transliterate first (for visibility of what's happening)
      let terms = [searchText];
      if (autoTransliterate) {
        const variants = await researchService.transliterate(searchText);
        const hebrewTerms = variants.map(v => v.hebrew);
        terms = [...terms, ...hebrewTerms];
        message.info(`Searching for: ${terms.join(', ')}`);
      }

      // 2. Run ELS Search
      const data = await researchService.findELS({
        terms: terms,
        source: 'torah',
        minSkip: 2,
        maxSkip: 200,
        autoTransliterate: false
      });

      setElsResults(data.matches);
      setStats({
        totalLength: data.total_length,
        foundCount: data.found_count
      });
      message.success(`Found ${data.found_count} matches!`);
    } catch (error) {
      message.error('Search failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleViewMatrix = async (record) => {
    setMatrixLoading(true);
    setMatrixVisible(true);
    try {
      const data = await researchService.getELSVisualization({
        source: record.source || 'torah',
        centerIndex: record.start_index, // Using start index as center
        skip: record.skip,
        termLength: record.term.length
      });
      setMatrixData(data);
    } catch (error) {
       message.error('Failed to load matrix: ' + error.message);
    } finally {
       setMatrixLoading(false);
    }
  };

  const regularColumns = [
    { title: 'Term', dataIndex: 'term', key: 'term' },
    { title: 'Skip', dataIndex: 'skip', key: 'skip', render: (val) => <Tag color={val > 0 ? "blue" : "red"}>{val}</Tag> },
    { title: 'Start', dataIndex: 'start_index', key: 'start' },
    { title: 'Direction', dataIndex: 'direction', key: 'dir' },
    { 
      title: 'Action', 
      key: 'action',
      render: (_, record) => (
        <Button 
          icon={<EyeOutlined />} 
          size="small" 
          onClick={() => handleViewMatrix(record)}
        >
          View
        </Button>
      )
    }
  ];


  return (
    <div style={{ background: '#1f1f1f', padding: '24px', borderRadius: '8px' }}>
      <Tabs activeKey={activeTab} onChange={setActiveTab} type="card">
        <TabPane tab={<span><SearchOutlined /> Single ELS Search</span>} key="els">
          <Row gutter={24}>
            <Col span={8}>
              <Card title="ELS Search Console" bordered={false} style={{ height: '100%' }}>
                <div style={{ marginBottom: 16 }}>
                  <Input 
                    placeholder="Enter English Term (e.g., PEPE)" 
                    size="large"
                    value={searchText}
                    onChange={e => setSearchText(e.target.value)}
                    onPressEnter={handleSearch}
                    prefix={<SearchOutlined />}
                  />
                </div>
                <div style={{ marginBottom: 24 }}>
                   <Switch 
                     checked={autoTransliterate} 
                     onChange={setAutoTransliterate} 
                     checkedChildren="Auto-Hebrew" 
                     unCheckedChildren="English Only" 
                   />
                   <span style={{ marginLeft: 12, color: '#888' }}>
                     {autoTransliterate ? "Searching Hebrew variants via TransliterationService" : "Searching exact text only"}
                   </span>
                </div>
                <Button type="primary" size="large" onClick={handleSearch} loading={loading} block>
                  Initiate Scan
                </Button>
                
                <div style={{ marginTop: 32 }}>
                  <Statistic title="Total Text Length" value={stats.totalLength} />
                  <Statistic title="Matches Found" value={stats.foundCount} style={{ marginTop: 16 }} />
                </div>
              </Card>
            </Col>
            <Col span={16}>
              <Card title="Analysis Results" bordered={false} style={{ height: '100%' }}>
                <Table 
                  dataSource={elsResults} 
                  columns={regularColumns} 
                  rowKey={(r) => `${r.start_index}_${r.skip}`}
                  pagination={{ pageSize: 8 }}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>
        
        <TabPane tab={<span><RadarChartOutlined /> Prophetic Convergence</span>} key="convergence">
           <Row gutter={24}>
             <Col span={8}>
               <Card title="Convergence Parameters" bordered={false}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    {propheticTerms.map((t, idx) => (
                      <Input 
                        key={idx} 
                        placeholder={`Term ${idx + 1} (e.g. TRUMP)`} 
                        value={t.term}
                        onChange={(e) => handleTermChange(idx, e.target.value)}
                      />
                    ))}
                    <div style={{ marginTop: 16 }}>
                      <Button type="primary" onClick={handlePropheticSearch} loading={convergenceLoading} block icon={<RadarChartOutlined />}>
                        Find Convergence Zones
                      </Button>
                    </div>
                    <div style={{ color: '#aaa', fontSize: 12, marginTop: 8 }}>
                       * Automatically transliterates inputs to Hebrew. Looks for zones where all 3 intersect within 500 letters.
                    </div>
                  </Space>
               </Card>
             </Col>
             <Col span={16}>
                <Card title="Top Convergence Zones" bordered={false}>
                   {convergenceZones.length === 0 ? (
                      <div style={{ textAlign: 'center', padding: 40, color: '#666' }}>
                        No convergence zones analyzed yet.
                      </div>
                   ) : (
                      <div style={{ maxHeight: '60vh', overflowY: 'auto' }}>
                         {convergenceZones.map((zone, i) => (
                            <Card key={i} size="small" title={`Zone #${i+1}: ${zone.book} (${zone.position_percentage.toFixed(1)}%)`} style={{ marginBottom: 16 }}>
                               <Row>
                                 <Col span={12}>
                                    <Statistic title="Spread" value={zone.spread} suffix="letters" />
                                    <ul>
                                      {zone.terms.map((t, k) => (
                                        <li key={k}><b>{t.name}:</b> {t.term} (Skip {t.skip})</li>
                                      ))}
                                    </ul>
                                 </Col>
                                 <Col span={12}>
                                    {zone.visualization_svg && (
                                       <div dangerouslySetInnerHTML={{ __html: zone.visualization_svg }} style={{ border: '1px solid #333' }} />
                                    )}
                                 </Col>
                               </Row>
                            </Card>
                         ))}
                      </div>
                   )}
                </Card>
             </Col>
           </Row>
        </TabPane>
      </Tabs>
      
      <Modal
        title="ELS Matrix View"
        open={matrixVisible}
        onCancel={() => setMatrixVisible(false)}
        footer={null}
        width={800}
        bodyStyle={{ background: '#111', color: '#fff', minHeight: '400px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}
      >
         {matrixLoading ? <Spin size="large" /> : matrixData && (
            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${matrixData.dimensions.cols}, 30px)`, gap: 2, padding: 20 }}>
               {matrixData.grid.flatMap((row, rIdx) => 
                  row.map((cell, cIdx) => {
                     // Check highlights
                     const isHighlight = matrixData.highlights.some(h => h.index === cell.index);
                     return (
                       <div key={`${rIdx}-${cIdx}`} style={{
                          width: 30, height: 30, 
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          background: isHighlight ? '#faad14' : '#222',
                          color: isHighlight ? '#000' : '#888',
                          border: isHighlight ? '1px solid #ffe58f' : '1px solid #333',
                          fontWeight: isHighlight ? 'bold' : 'normal',
                          borderRadius: 2,
                          cursor: 'pointer'
                       }} title={`Index: ${cell.index}`}>
                          {cell.char}
                       </div>
                     );
                  })
               )}
            </div>
         )}
      </Modal>
    </div>
  );
};

export default TheScope;
