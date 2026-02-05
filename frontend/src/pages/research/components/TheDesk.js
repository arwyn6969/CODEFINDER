import React, { useState, useEffect } from 'react';
import { Layout, Card, Empty, Button, Upload, List, Typography, Spin, message, Row, Col, Tabs, Statistic, Tag, Input } from 'antd';
import { FileImageOutlined, UploadOutlined, CloseOutlined, CalculatorOutlined, FileTextOutlined } from '@ant-design/icons';
import researchService from '../../../services/researchService';

import DocumentViewer from './DocumentViewer';

const { Content, Sider } = Layout;
const { Title, Paragraph, Text } = Typography;
const { TabPane } = Tabs;

const TheDesk = ({ currentDocument, onSetDocument }) => {
  const [documents, setDocuments] = useState([]);
  const [loadingList, setLoadingList] = useState(false);
  const [docContent, setDocContent] = useState('');
  const [loadingDoc, setLoadingDoc] = useState(false);
  
  // Selection & Analysis State
  const [selectedText, setSelectedText] = useState('');
  const [gematriaData, setGematriaData] = useState(null);
  const [loadingGematria, setLoadingGematria] = useState(false);

  // Initial Load of Documents
  useEffect(() => {
    if (!currentDocument) {
      fetchDocuments();
    }
  }, [currentDocument]);

  // Load Content when Document Selected
  useEffect(() => {
    if (currentDocument) {
      loadDocumentContent(currentDocument.id);
    }
  }, [currentDocument]);

  const fetchDocuments = async () => {
    setLoadingList(true);
    try {
      const data = await researchService.listDocuments();
      setDocuments(data.documents || []);
    } catch (error) {
      message.error('Failed to load documents list');
    } finally {
      setLoadingList(false);
    }
  };

  const loadDocumentContent = async (id) => {
    setLoadingDoc(true);
    setDocContent('');
    try {
      const data = await researchService.getDocumentContent(id);
      setDocContent(data.content);
    } catch (error) {
      message.error('Failed to load document content');
    } finally {
      setLoadingDoc(false);
    }
  };

  const handleUpload = async (file) => {
    try {
      const response = await researchService.uploadDocument(file);
      message.success('Upload successful');
      fetchDocuments();
      return false; // Prevent auto upload by antd
    } catch (error) {
       message.error('Upload failed');
    }
  };

  const handleTextSelection = async () => {
     const selection = window.getSelection().toString().trim();
     if (selection && selection.length > 0 && selection.length < 500) {
        setSelectedText(selection);
        setLoadingGematria(true);
        try {
           const data = await researchService.calculateGematria(selection);
           setGematriaData(data);
        } catch (error) {
           console.error("Gematria calc error", error);
        } finally {
           setLoadingGematria(false);
        }
     }
  };

  const renderDocumentList = () => (
    <div style={{ padding: 40, maxWidth: 800, margin: '0 auto', color: '#fff' }}>
      <div style={{ textAlign: 'center', marginBottom: 40 }}>
        <FileImageOutlined style={{ fontSize: 64, color: '#1890ff', marginBottom: 16 }} />
        <Title level={2} style={{ color: '#fff' }}>The Scholar's Desk</Title>
        <Paragraph style={{ color: '#aaa' }}>
          Select a document from the library or upload a new text for analysis.
        </Paragraph>
        <Upload beforeUpload={handleUpload} showUploadList={false}>
           <Button type="primary" icon={<UploadOutlined />} size="large">Upload Document</Button>
        </Upload>
      </div>

      <Card title="Available Documents" bordered={false} style={{ background: '#1f1f1f' }} headStyle={{ color: '#fff' }}>
         {loadingList ? <Spin /> : (
            <List
              itemLayout="horizontal"
              dataSource={documents}
              renderItem={item => (
                <List.Item 
                  actions={[<Button type="link" onClick={() => onSetDocument(item)}>Open</Button>]}
                >
                  <List.Item.Meta
                    avatar={<FileTextOutlined style={{ fontSize: 24, color: '#aaa' }} />}
                    title={<span style={{ color: '#fff' }}>{item.filename}</span>}
                    description={<span style={{ color: '#666' }}>ID: {item.id} | Status: {item.processing_status}</span>}
                  />
                </List.Item>
              )}
            />
         )}
      </Card>
    </div>
  );

  const renderWorkspace = () => {
     // Construct image path if document has a filename
     // Assuming document images are stored as {filename} (e.g. page_01.img)
     // For PDFs, we might need a specific page renderer, but for now let's assume direct image access or a placeholder.
     // If it's a PDF, we might treat it differently.
     
     const docImage = currentDocument?.filename ? `/static/uploads/${currentDocument.filename}` : null;
     const isImage = docImage && (docImage.endsWith('.png') || docImage.endsWith('.jpg') || docImage.endsWith('.jpeg'));

     return (
    <Layout style={{ height: 'calc(100vh - 120px)', background: '#000' }}>
       <Content style={{ padding: '0 24px', overflowY: 'auto', borderRight: '1px solid #333' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, paddingTop: 16 }}>
             <Title level={4} style={{ color: '#fff', margin: 0 }}>{currentDocument.filename}</Title>
             <Button icon={<CloseOutlined />} onClick={() => onSetDocument(null)}>Close</Button>
          </div>
          
          <Row gutter={16} style={{ height: '80%' }}>
             {/* Left Panel: Text Content */}
             <Col span={isImage ? 12 : 24} style={{ height: '100%' }}>
                  <Card 
                    style={{ height: '100%', background: '#1f1f1f', border: 'none', overflowY: 'auto' }} 
                    bodyStyle={{ padding: 24, color: '#ddd', fontSize: '16px', lineHeight: '1.8', whiteSpace: 'pre-wrap', fontFamily: 'Georgia, serif' }}
                    onMouseUp={handleTextSelection}
                    title={<span style={{ color: '#aaa' }}><FileTextOutlined /> Parsed Text</span>}
                  >
                     {loadingDoc ? <div style={{ textAlign: 'center', padding: 50 }}><Spin size="large" tip="Loading Content..." /></div> : (docContent || <Empty description="No text content found" />)}
                  </Card>
             </Col>
             
             {/* Right Panel: Image Viewer (if applicable) */}
             {isImage && (
                 <Col span={12} style={{ height: '100%' }}>
                    <Card
                        style={{ height: '100%', background: '#1f1f1f', border: 'none', overflowY: 'auto' }}
                        bodyStyle={{ padding: 0, height: '100%', background: '#000' }}
                        title={<span style={{ color: '#aaa' }}><FileImageOutlined /> Original Source</span>}
                    >
                        <DocumentViewer src={docImage} alt="Document Source" />
                    </Card>
                 </Col>
             )}
          </Row>
       </Content>
       
       <Sider width={350} style={{ background: '#141414', borderLeft: '1px solid #333', padding: 16, overflowY: 'auto' }}>
          {/* ... Stats & Gematria ... */}
          <Title level={5} style={{ color: '#faad14', marginTop: 0 }}><CalculatorOutlined /> Analysis Workspace</Title>
          
          {selectedText ? (
             <div style={{ marginBottom: 24 }}>
                <Text style={{ color: '#fff', display: 'block', marginBottom: 8, padding: 8, background: '#222', borderRadius: 4 }}>
                   "{selectedText}"
                </Text>
                {loadingGematria ? <Spin /> : (
                   gematriaData && Object.entries(gematriaData).map(([cipher, value]) => (
                      <Card key={cipher} size="small" style={{ background: '#222', borderColor: '#444', marginBottom: 8 }}>
                         <Statistic 
                            title={<span style={{ color: '#888' }}>{cipher.replace(/_/g, ' ')}</span>}
                            value={value.score || value}
                            valueStyle={{ color: '#faad14' }}
                         />
                      </Card>
                   ))
                )}
             </div>
          ) : (
             <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={<span style={{ color: '#666' }}>Select text to analyze</span>} />
          )}

          <div style={{ borderTop: '1px solid #333', marginTop: 24, paddingTop: 24 }}>
             <Title level={5} style={{ color: '#fff' }}>Document Stats</Title>
             <p style={{ color: '#888' }}>ID: <span style={{ color: '#fff' }}>{currentDocument.id}</span></p>
             <p style={{ color: '#888' }}>Pages: <span style={{ color: '#fff' }}>{currentDocument.page_count}</span></p>
             <p style={{ color: '#888' }}>Status: <Tag color="green">{currentDocument.c_status || 'Ready'}</Tag></p>
          </div>
       </Sider>
    </Layout>
  );
  };

  return (
    <div style={{ height: '100%', background: '#000' }}>
      {!currentDocument ? renderDocumentList() : renderWorkspace()}
    </div>
  );
};

export default TheDesk;
