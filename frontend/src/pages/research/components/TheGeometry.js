import React, { useEffect, useMemo, useState } from 'react';
import { Alert, Button, Card, Col, Empty, List, Row, Select, Space, Spin, Statistic, Tag, Typography } from 'antd';
import { AimOutlined, CompassOutlined, NodeIndexOutlined } from '@ant-design/icons';

import researchService from '../../../services/researchService';

const { Paragraph, Text, Title } = Typography;

const TheGeometry = ({ currentDocument, onOpenDocument }) => {
  const [documents, setDocuments] = useState([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState(currentDocument?.id ?? null);
  const [analysis, setAnalysis] = useState(null);
  const [loadingDocuments, setLoadingDocuments] = useState(false);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDocuments = async () => {
      setLoadingDocuments(true);
      try {
        const data = await researchService.listDocuments();
        setDocuments(data.documents || []);
      } catch (fetchError) {
        setError(fetchError.message || 'Failed to load documents for geometry analysis.');
      } finally {
        setLoadingDocuments(false);
      }
    };

    fetchDocuments();
  }, []);

  useEffect(() => {
    if (currentDocument?.id) {
      setSelectedDocumentId(currentDocument.id);
    }
  }, [currentDocument]);

  useEffect(() => {
    if (!selectedDocumentId) {
      setAnalysis(null);
      return;
    }

    const fetchAnalysis = async () => {
      setLoadingAnalysis(true);
      setError(null);
      try {
        const data = await researchService.getGeometricAnalysis(selectedDocumentId);
        setAnalysis(data);
      } catch (fetchError) {
        setAnalysis(null);
        setError(fetchError.message || 'Failed to load legacy geometry analysis.');
      } finally {
        setLoadingAnalysis(false);
      }
    };

    fetchAnalysis();
  }, [selectedDocumentId]);

  const selectedDocument = useMemo(() => {
    return documents.find((document) => document.id === selectedDocumentId) || currentDocument || null;
  }, [currentDocument, documents, selectedDocumentId]);

  const documentOptions = documents.map((document) => ({
    label: `${document.filename} (#${document.id})`,
    value: document.id,
  }));

  const renderWarnings = () => {
    if (!analysis?.warnings?.length) {
      return null;
    }

    return (
      <Alert
        type={analysis.status === 'no_data' ? 'info' : 'warning'}
        showIcon
        style={{ marginBottom: 16 }}
        message={analysis.status === 'no_data' ? 'No geometry data available' : 'Geometry analysis warnings'}
        description={
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {analysis.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        }
      />
    );
  };

  return (
    <div style={{ background: '#1f1f1f', padding: '24px', borderRadius: '8px', minHeight: 'calc(100vh - 220px)' }}>
      <Card
        bordered={false}
        title={<span style={{ color: '#fff' }}><CompassOutlined /> Legacy Geometry</span>}
        extra={
          <Space>
            <Select
              style={{ minWidth: 320 }}
              placeholder={loadingDocuments ? 'Loading documents...' : 'Select a document'}
              loading={loadingDocuments}
              options={documentOptions}
              value={selectedDocumentId}
              onChange={setSelectedDocumentId}
              allowClear
            />
            {selectedDocument && onOpenDocument && (
              <Button onClick={() => onOpenDocument(selectedDocument)}>Open In Desk</Button>
            )}
          </Space>
        }
        style={{ background: '#141414', border: '1px solid #303030' }}
        headStyle={{ color: '#fff', borderBottom: '1px solid #303030' }}
        bodyStyle={{ background: '#141414' }}
      >
        <Paragraph style={{ color: '#aaa' }}>
          This internal geometry view uses stored pattern coordinates, derives angle and distance measurements, and
          runs BardCode-style geographic extraction without promoting the lane as a primary workflow.
        </Paragraph>

        {!selectedDocumentId && (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={<span style={{ color: '#777' }}>Choose a document to inspect its preserved geometry lane.</span>}
          />
        )}

        {selectedDocumentId && loadingAnalysis && (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '40px 0' }}>
            <Spin size="large" />
          </div>
        )}

        {selectedDocumentId && !loadingAnalysis && error && (
          <Alert type="error" showIcon message="Geometry analysis failed" description={error} />
        )}

        {selectedDocumentId && !loadingAnalysis && analysis && (
          <React.Fragment>
            {renderWarnings()}

            <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
              <Col xs={24} md={12} xl={6}>
                <Card size="small" style={{ background: '#222', borderColor: '#333' }}>
                  <Statistic title={<span style={{ color: '#888' }}>Total Measurements</span>} value={analysis.total_measurements} valueStyle={{ color: '#fff' }} />
                </Card>
              </Col>
              <Col xs={24} md={12} xl={6}>
                <Card size="small" style={{ background: '#222', borderColor: '#333' }}>
                  <Statistic title={<span style={{ color: '#888' }}>Angles</span>} value={analysis.angle_measurements} valueStyle={{ color: '#fff' }} />
                </Card>
              </Col>
              <Col xs={24} md={12} xl={6}>
                <Card size="small" style={{ background: '#222', borderColor: '#333' }}>
                  <Statistic title={<span style={{ color: '#888' }}>Distances</span>} value={analysis.distance_measurements} valueStyle={{ color: '#fff' }} />
                </Card>
              </Col>
              <Col xs={24} md={12} xl={6}>
                <Card size="small" style={{ background: '#222', borderColor: '#333' }}>
                  <Statistic title={<span style={{ color: '#888' }}>Coordinate Pairs</span>} value={analysis.coordinate_pairs.length} valueStyle={{ color: '#fff' }} />
                </Card>
              </Col>
            </Row>

            <Row gutter={[16, 16]}>
              <Col xs={24} xl={10}>
                <Card
                  size="small"
                  title={<span style={{ color: '#fff' }}><NodeIndexOutlined /> Constants And Scores</span>}
                  style={{ background: '#1b1b1b', borderColor: '#303030', height: '100%' }}
                  headStyle={{ color: '#fff', borderBottom: '1px solid #303030' }}
                >
                  <Space wrap style={{ marginBottom: 16 }}>
                    {analysis.mathematical_constants_found.length > 0 ? (
                      analysis.mathematical_constants_found.map((constantName) => (
                        <Tag key={constantName} color="gold">{constantName}</Tag>
                      ))
                    ) : (
                      <Text style={{ color: '#777' }}>No mathematical constants detected.</Text>
                    )}
                  </Space>

                  <List
                    dataSource={Object.entries(analysis.significance_scores || {})}
                    locale={{ emptyText: <span style={{ color: '#777' }}>No significance metrics available.</span> }}
                    renderItem={([metric, value]) => (
                      <List.Item style={{ borderBottom: '1px solid #2a2a2a' }}>
                        <Text style={{ color: '#aaa' }}>{metric.replace(/_/g, ' ')}</Text>
                        <Text style={{ color: '#fff' }}>{value}</Text>
                      </List.Item>
                    )}
                  />
                </Card>
              </Col>

              <Col xs={24} xl={14}>
                <Card
                  size="small"
                  title={<span style={{ color: '#fff' }}><AimOutlined /> Candidate Coordinates</span>}
                  style={{ background: '#1b1b1b', borderColor: '#303030', marginBottom: 16 }}
                  headStyle={{ color: '#fff', borderBottom: '1px solid #303030' }}
                >
                  {analysis.coordinate_pairs.length === 0 ? (
                    <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={<span style={{ color: '#777' }}>No plausible coordinate pairs were derived.</span>} />
                  ) : (
                    <List
                      dataSource={analysis.coordinate_pairs.slice(0, 8)}
                      renderItem={(pair) => (
                        <List.Item style={{ borderBottom: '1px solid #2a2a2a' }}>
                          <div>
                            <Title level={5} style={{ color: '#fff', margin: 0 }}>
                              {pair.latitude.toFixed(4)}°, {pair.longitude.toFixed(4)}°
                            </Title>
                            <Text style={{ color: '#888' }}>
                              Confidence {pair.combined_confidence.toFixed(3)} · {pair.methods.join(', ')}
                            </Text>
                          </div>
                        </List.Item>
                      )}
                    />
                  )}
                </Card>

                <Card
                  size="small"
                  title={<span style={{ color: '#fff' }}><CompassOutlined /> Historical Site Matches</span>}
                  style={{ background: '#1b1b1b', borderColor: '#303030' }}
                  headStyle={{ color: '#fff', borderBottom: '1px solid #303030' }}
                >
                  {analysis.historical_sites.length === 0 ? (
                    <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={<span style={{ color: '#777' }}>No historical site matches met the current tolerance.</span>} />
                  ) : (
                    <List
                      dataSource={analysis.historical_sites.slice(0, 6)}
                      renderItem={(match) => (
                        <List.Item style={{ borderBottom: '1px solid #2a2a2a' }}>
                          <div>
                            <Title level={5} style={{ color: '#fff', margin: 0 }}>{match.site.name}</Title>
                            <Text style={{ color: '#888', display: 'block' }}>
                              {match.detected_coordinates.latitude.toFixed(4)}°, {match.detected_coordinates.longitude.toFixed(4)}°
                            </Text>
                            <Text style={{ color: '#888' }}>
                              Error {match.accuracy.total_error.toFixed(3)} · Match confidence {match.match_confidence.toFixed(3)}
                            </Text>
                          </div>
                        </List.Item>
                      )}
                    />
                  )}
                </Card>
              </Col>
            </Row>
          </React.Fragment>
        )}
      </Card>
    </div>
  );
};

export default TheGeometry;
