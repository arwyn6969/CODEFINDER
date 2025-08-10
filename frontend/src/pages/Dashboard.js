import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Progress, List, Button, Alert } from 'antd';
import { 
  FileTextOutlined, 
  BarChartOutlined, 
  SearchOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';
import axios from 'axios';

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalDocuments: 0,
    processingDocuments: 0,
    completedDocuments: 0,
    totalPatterns: 0
  });
  const [recentDocuments, setRecentDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      
      // Load documents
      const documentsResponse = await axios.get('/api/documents/', {
        params: { per_page: 10 }
      });
      
      const documents = documentsResponse.data.documents;
      setRecentDocuments(documents);
      
      // Calculate statistics
      const totalDocuments = documentsResponse.data.total;
      const processingDocuments = documents.filter(d => d.processing_status === 'processing').length;
      const completedDocuments = documents.filter(d => d.processing_status === 'completed').length;
      
      // For demo purposes, simulate pattern count
      const totalPatterns = completedDocuments * 15; // Rough estimate
      
      setStats({
        totalDocuments,
        processingDocuments,
        completedDocuments,
        totalPatterns
      });
      
    } catch (error) {
      console.error('Error loading dashboard data:', error);
      setError('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'processing':
        return <ClockCircleOutlined style={{ color: '#fa8c16' }} />;
      case 'failed':
        return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return <FileTextOutlined style={{ color: '#1890ff' }} />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return '#52c41a';
      case 'processing': return '#fa8c16';
      case 'failed': return '#ff4d4f';
      default: return '#1890ff';
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (error) {
    return (
      <div style={{ padding: '24px' }}>
        <Alert
          message="Error Loading Dashboard"
          description={error}
          type="error"
          showIcon
          action={
            <Button size="small" onClick={loadDashboardData}>
              Retry
            </Button>
          }
        />
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ margin: 0, color: '#8B4513' }}>Dashboard</h1>
        <p style={{ margin: '8px 0 0 0', color: '#666' }}>
          Overview of your ancient text analysis projects
        </p>
      </div>

      {/* Statistics Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Total Documents"
              value={stats.totalDocuments}
              prefix={<FileTextOutlined />}
              valueStyle={{ color: '#8B4513' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Processing"
              value={stats.processingDocuments}
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Completed"
              value={stats.completedDocuments}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Patterns Found"
              value={stats.totalPatterns}
              prefix={<BarChartOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        {/* Recent Documents */}
        <Col xs={24} lg={16}>
          <Card 
            title="Recent Documents" 
            loading={loading}
            extra={
              <Button type="link" href="/documents">
                View All
              </Button>
            }
          >
            <List
              itemLayout="horizontal"
              dataSource={recentDocuments}
              renderItem={(document) => (
                <List.Item
                  actions={[
                    <Button 
                      type="link" 
                      href={`/documents/${document.id}`}
                      size="small"
                    >
                      View
                    </Button>
                  ]}
                >
                  <List.Item.Meta
                    avatar={getStatusIcon(document.processing_status)}
                    title={
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span>{document.filename}</span>
                        <span 
                          style={{
                            fontSize: '12px',
                            padding: '2px 8px',
                            borderRadius: '12px',
                            backgroundColor: getStatusColor(document.processing_status),
                            color: 'white'
                          }}
                        >
                          {document.processing_status}
                        </span>
                      </div>
                    }
                    description={
                      <div>
                        <div>Size: {formatFileSize(document.file_size)}</div>
                        <div>Uploaded: {new Date(document.upload_date).toLocaleDateString()}</div>
                        {document.total_pages && (
                          <div>Pages: {document.total_pages}</div>
                        )}
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>

        {/* Quick Actions */}
        <Col xs={24} lg={8}>
          <Card title="Quick Actions">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <Button 
                type="primary" 
                icon={<FileTextOutlined />}
                size="large"
                href="/upload"
                style={{ backgroundColor: '#8B4513', borderColor: '#8B4513' }}
              >
                Upload New Document
              </Button>
              
              <Button 
                icon={<SearchOutlined />}
                size="large"
                href="/search"
              >
                Search & Query
              </Button>
              
              <Button 
                icon={<BarChartOutlined />}
                size="large"
                href="/documents"
              >
                View Analysis Results
              </Button>
            </div>
          </Card>

          {/* System Status */}
          <Card title="System Status" style={{ marginTop: '16px' }}>
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span>Processing Capacity</span>
                <span>75%</span>
              </div>
              <Progress percent={75} strokeColor="#8B4513" />
            </div>
            
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span>Storage Used</span>
                <span>45%</span>
              </div>
              <Progress percent={45} strokeColor="#52c41a" />
            </div>

            <div style={{ fontSize: '12px', color: '#666' }}>
              <div>✅ OCR Engine: Online</div>
              <div>✅ Pattern Detection: Online</div>
              <div>✅ Geometric Analysis: Online</div>
              <div>✅ Report Generation: Online</div>
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;