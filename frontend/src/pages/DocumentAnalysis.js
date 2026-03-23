import React from 'react';
import { Card, Typography } from 'antd';

const { Title } = Typography;

const DocumentAnalysis = () => {
  return (
    <div style={{ padding: '24px' }}>
      <Title level={2} style={{ color: '#8B4513' }}>
        Analysis Results
      </Title>
      <Card>
        <p>Detailed OCR output, extracted text, findings, and export/report tools will be displayed here.</p>
      </Card>
    </div>
  );
};

export default DocumentAnalysis;
