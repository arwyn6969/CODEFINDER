import React, { useState, useCallback } from 'react';
import { 
  Card, 
  Upload, 
  Button, 
  Progress, 
  Alert, 
  List, 
  Typography,
  Space,
  Tag
} from 'antd';
import { 
  InboxOutlined, 
  FileTextOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined
} from '@ant-design/icons';
import axios from 'axios';
import WebSocketService from '../services/WebSocketService';

const { Dragger } = Upload;
const { Title, Text } = Typography;

const DocumentUpload = () => {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleUpload = useCallback(async (file) => {
    setError('');
    setSuccess('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/api/documents/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          // Update progress for the current file
          setUploadedFiles(prev => 
            prev.map(f => 
              f.uid === file.uid 
                ? { ...f, progress: percentCompleted }
                : f
            )
          );
        },
      });

      const { document_id, filename } = response.data;

      // Add to uploaded files list
      const newFile = {
        uid: file.uid,
        name: filename,
        document_id,
        status: 'processing',
        progress: 100,
        upload_time: new Date().toISOString()
      };

      setUploadedFiles(prev => {
        const existing = prev.find(f => f.uid === file.uid);
        if (existing) {
          return prev.map(f => f.uid === file.uid ? newFile : f);
        }
        return [...prev, newFile];
      });

      // Subscribe to document processing updates
      WebSocketService.subscribeToDocument(document_id);
      
      // Listen for processing updates
      const unsubscribe = WebSocketService.subscribe('processing_update', (message) => {
        if (message.document_id === document_id) {
          setUploadedFiles(prev =>
            prev.map(f =>
              f.document_id === document_id
                ? { 
                    ...f, 
                    status: message.status,
                    processing_step: message.current_step,
                    processing_progress: message.progress
                  }
                : f
            )
          );
        }
      });

      // Listen for completion
      const unsubscribeComplete = WebSocketService.subscribe('analysis_complete', (message) => {
        if (message.document_id === document_id) {
          setUploadedFiles(prev =>
            prev.map(f =>
              f.document_id === document_id
                ? { ...f, status: 'completed', processing_progress: 100 }
                : f
            )
          );
          unsubscribe();
          unsubscribeComplete();
        }
      });

      setSuccess(`${filename} uploaded successfully! Processing started.`);

    } catch (error) {
      console.error('Upload error:', error);
      setError(error.response?.data?.detail || 'Upload failed');
      
      // Update file status to failed
      setUploadedFiles(prev =>
        prev.map(f =>
          f.uid === file.uid
            ? { ...f, status: 'failed' }
            : f
        )
      );
    }
  }, []);

  const uploadProps = {
    name: 'file',
    multiple: true,
    accept: '.pdf,.txt,.docx,.png,.jpg,.jpeg,.tiff',
    beforeUpload: (file) => {
      // Validate file size (100MB limit)
      const isLt100M = file.size / 1024 / 1024 < 100;
      if (!isLt100M) {
        setError('File must be smaller than 100MB!');
        return false;
      }

      // Validate file type
      const allowedTypes = [
        'application/pdf',
        'text/plain',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'image/png',
        'image/jpeg',
        'image/tiff'
      ];
      
      if (!allowedTypes.includes(file.type)) {
        setError('Please upload PDF, TXT, DOCX, or image files only!');
        return false;
      }

      // Add file to the list immediately
      const newFile = {
        uid: file.uid,
        name: file.name,
        status: 'uploading',
        progress: 0,
        upload_time: new Date().toISOString()
      };
      
      setUploadedFiles(prev => [...prev, newFile]);

      // Start upload
      handleUpload(file);
      
      return false; // Prevent default upload
    },
    showUploadList: false,
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'failed':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return <FileTextOutlined style={{ color: '#1890ff' }} />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'success';
      case 'processing': return 'processing';
      case 'uploading': return 'default';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2} style={{ margin: 0, color: '#8B4513' }}>
          Upload Document
        </Title>
        <Text type="secondary">
          Upload ancient texts, manuscripts, or historical documents for analysis
        </Text>
      </div>

      {error && (
        <Alert
          message="Upload Error"
          description={error}
          type="error"
          closable
          onClose={() => setError('')}
          style={{ marginBottom: '16px' }}
        />
      )}

      {success && (
        <Alert
          message="Upload Successful"
          description={success}
          type="success"
          closable
          onClose={() => setSuccess('')}
          style={{ marginBottom: '16px' }}
        />
      )}

      <Card style={{ marginBottom: '24px' }}>
        <Dragger {...uploadProps} style={{ padding: '40px' }}>
          <p className="ant-upload-drag-icon">
            <InboxOutlined style={{ fontSize: '48px', color: '#8B4513' }} />
          </p>
          <p className="ant-upload-text" style={{ fontSize: '18px' }}>
            Click or drag files to this area to upload
          </p>
          <p className="ant-upload-hint" style={{ fontSize: '14px' }}>
            Support for PDF, TXT, DOCX, and image files (PNG, JPG, TIFF).
            <br />
            Maximum file size: 100MB per file.
          </p>
        </Dragger>

        <div style={{ marginTop: '16px', textAlign: 'center' }}>
          <Space>
            <Tag color="blue">PDF Documents</Tag>
            <Tag color="green">Text Files</Tag>
            <Tag color="orange">Word Documents</Tag>
            <Tag color="purple">Images</Tag>
          </Space>
        </div>
      </Card>

      {uploadedFiles.length > 0 && (
        <Card title="Upload Progress" style={{ marginBottom: '24px' }}>
          <List
            itemLayout="horizontal"
            dataSource={uploadedFiles}
            renderItem={(file) => (
              <List.Item
                actions={[
                  file.document_id && file.status === 'completed' && (
                    <Button 
                      type="link" 
                      href={`/documents/${file.document_id}`}
                      size="small"
                    >
                      View Analysis
                    </Button>
                  )
                ]}
              >
                <List.Item.Meta
                  avatar={getStatusIcon(file.status)}
                  title={
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span>{file.name}</span>
                      <Tag color={getStatusColor(file.status)}>
                        {file.status}
                      </Tag>
                    </div>
                  }
                  description={
                    <div>
                      {file.status === 'uploading' && (
                        <Progress 
                          percent={file.progress || 0} 
                          size="small"
                          strokeColor="#8B4513"
                        />
                      )}
                      {file.status === 'processing' && (
                        <div>
                          <div style={{ marginBottom: '4px' }}>
                            {file.processing_step || 'Processing...'}
                          </div>
                          <Progress 
                            percent={file.processing_progress || 0} 
                            size="small"
                            strokeColor="#fa8c16"
                          />
                        </div>
                      )}
                      {file.status === 'completed' && (
                        <Text type="success">
                          ✅ Analysis completed successfully
                        </Text>
                      )}
                      {file.status === 'failed' && (
                        <Text type="danger">
                          ❌ Processing failed
                        </Text>
                      )}
                      <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
                        Uploaded: {new Date(file.upload_time).toLocaleString()}
                      </div>
                    </div>
                  }
                />
              </List.Item>
            )}
          />
        </Card>
      )}

      <Card title="Supported File Types" size="small">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
          <div>
            <Text strong>📄 PDF Documents</Text>
            <div style={{ fontSize: '12px', color: '#666' }}>
              Scanned or digital PDFs of ancient texts, manuscripts, historical documents
            </div>
          </div>
          <div>
            <Text strong>📝 Text Files</Text>
            <div style={{ fontSize: '12px', color: '#666' }}>
              Plain text transcriptions, UTF-8 encoded text files
            </div>
          </div>
          <div>
            <Text strong>📋 Word Documents</Text>
            <div style={{ fontSize: '12px', color: '#666' }}>
              DOCX format documents with text content
            </div>
          </div>
          <div>
            <Text strong>🖼️ Images</Text>
            <div style={{ fontSize: '12px', color: '#666' }}>
              High-resolution scans (PNG, JPG, TIFF) for OCR processing
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default DocumentUpload;
