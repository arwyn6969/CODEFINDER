import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Table, 
  Button, 
  Tag, 
  Space, 
  Input, 
  Select, 
  Popconfirm,
  message,
  Progress
} from 'antd';
import { 
  EyeOutlined, 
  DeleteOutlined, 
  ReloadOutlined,
  SearchOutlined,
  DownloadOutlined
} from '@ant-design/icons';
import axios from 'axios';

const { Search } = Input;
const { Option } = Select;

const DocumentList = () => {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [pagination, setPagination] = useState({
    current: 1,
    pageSize: 10,
    total: 0
  });
  const [filters, setFilters] = useState({
    status: null,
    search: ''
  });

  useEffect(() => {
    loadDocuments();
  }, [pagination.current, pagination.pageSize, filters]);

  const loadDocuments = async () => {
    try {
      setLoading(true);
      const params = {
        page: pagination.current,
        per_page: pagination.pageSize,
        ...(filters.status && { status_filter: filters.status })
      };

      const response = await axios.get('/api/documents/', { params });
      
      setDocuments(response.data.documents);
      setPagination(prev => ({
        ...prev,
        total: response.data.total
      }));
    } catch (error) {
      console.error('Error loading documents:', error);
      message.error('Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (documentId) => {
    try {
      await axios.delete(`/api/documents/${documentId}`);
      message.success('Document deleted successfully');
      loadDocuments();
    } catch (error) {
      console.error('Error deleting document:', error);
      message.error('Failed to delete document');
    }
  };

  const handleReprocess = async (documentId) => {
    try {
      await axios.post(`/api/documents/${documentId}/reprocess`);
      message.success('Document reprocessing started');
      loadDocuments();
    } catch (error) {
      console.error('Error reprocessing document:', error);
      message.error('Failed to start reprocessing');
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'success';
      case 'processing': return 'processing';
      case 'uploaded': return 'default';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const columns = [
    {
      title: 'Document',
      dataIndex: 'filename',
      key: 'filename',
      render: (filename, record) => (
        <div>
          <div style={{ fontWeight: 500 }}>{filename}</div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            {formatFileSize(record.file_size)} • {record.total_pages ? `${record.total_pages} pages` : 'Processing...'}
          </div>
        </div>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'processing_status',
      key: 'status',
      render: (status) => (
        <Tag color={getStatusColor(status)}>
          {status.toUpperCase()}
        </Tag>
      ),
      filters: [
        { text: 'Uploaded', value: 'uploaded' },
        { text: 'Processing', value: 'processing' },
        { text: 'Completed', value: 'completed' },
        { text: 'Failed', value: 'failed' },
      ],
    },
    {
      title: 'Upload Date',
      dataIndex: 'upload_date',
      key: 'upload_date',
      render: (date) => new Date(date).toLocaleDateString(),
      sorter: true,
    },
    {
      title: 'Progress',
      key: 'progress',
      render: (_, record) => {
        if (record.processing_status === 'completed') {
          return <Progress percent={100} size="small" strokeColor="#52c41a" />;
        } else if (record.processing_status === 'processing') {
          return <Progress percent={50} size="small" strokeColor="#fa8c16" />;
        } else if (record.processing_status === 'failed') {
          return <Progress percent={0} size="small" strokeColor="#ff4d4f" />;
        }
        return <Progress percent={0} size="small" />;
      },
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button
            type="primary"
            size="small"
            icon={<EyeOutlined />}
            href={`/documents/${record.id}`}
            disabled={record.processing_status !== 'completed'}
          >
            View
          </Button>
          
          {record.processing_status === 'failed' && (
            <Button
              size="small"
              icon={<ReloadOutlined />}
              onClick={() => handleReprocess(record.id)}
            >
              Retry
            </Button>
          )}
          
          <Popconfirm
            title="Are you sure you want to delete this document?"
            onConfirm={() => handleDelete(record.id)}
            okText="Yes"
            cancelText="No"
          >
            <Button
              size="small"
              danger
              icon={<DeleteOutlined />}
            >
              Delete
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ margin: 0, color: '#8B4513' }}>My Documents</h1>
        <p style={{ margin: '8px 0 0 0', color: '#666' }}>
          Manage and analyze your uploaded documents
        </p>
      </div>

      <Card>
        <div style={{ marginBottom: '16px', display: 'flex', gap: '16px', alignItems: 'center' }}>
          <Search
            placeholder="Search documents..."
            allowClear
            style={{ width: 300 }}
            onSearch={(value) => setFilters(prev => ({ ...prev, search: value }))}
          />
          
          <Select
            placeholder="Filter by status"
            allowClear
            style={{ width: 150 }}
            onChange={(value) => setFilters(prev => ({ ...prev, status: value }))}
          >
            <Option value="uploaded">Uploaded</Option>
            <Option value="processing">Processing</Option>
            <Option value="completed">Completed</Option>
            <Option value="failed">Failed</Option>
          </Select>

          <Button
            icon={<ReloadOutlined />}
            onClick={loadDocuments}
          >
            Refresh
          </Button>
        </div>

        <Table
          columns={columns}
          dataSource={documents}
          rowKey="id"
          loading={loading}
          pagination={{
            ...pagination,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) =>
              `${range[0]}-${range[1]} of ${total} documents`,
            onChange: (page, pageSize) => {
              setPagination(prev => ({
                ...prev,
                current: page,
                pageSize
              }));
            },
          }}
        />
      </Card>
    </div>
  );
};

export default DocumentList;