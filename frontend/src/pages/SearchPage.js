import React from 'react';
import { Card, Typography } from 'antd';

const { Title } = Typography;

const SearchPage = () => {
  return (
    <div style={{ padding: '24px' }}>
      <Title level={2} style={{ color: '#8B4513' }}>
        Search & Query
      </Title>
      <Card>
        <p>Advanced search and query interface will be implemented here.</p>
      </Card>
    </div>
  );
};

export default SearchPage;