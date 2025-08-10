import React from 'react';
import { Card, Typography } from 'antd';

const { Title } = Typography;

const DocumentAnalysis = () => {
  return (
    <div style={{ padding: '24px' }}>
      <Title level={2} style={{ color: '#8B4513' }}>
        Document Analysis
      </Title>
      <Card>
        <p>Document analysis page - detailed analysis results and visualizations will be displayed here.</p>
      </Card>
    </div>
  );
};

export default DocumentAnalysis;