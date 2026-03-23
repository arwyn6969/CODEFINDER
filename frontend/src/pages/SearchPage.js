import React from 'react';
import { Card, Typography } from 'antd';

const { Title } = Typography;

const SearchPage = () => {
  return (
    <div style={{ padding: '24px' }}>
      <Title level={2} style={{ color: '#8B4513' }}>
        Catalog Search
      </Title>
      <Card>
        <p>Search, filtering, and report-oriented retrieval will be consolidated here.</p>
        <p>The exploratory ELS and gematria tooling remains available through the Legacy Lab.</p>
      </Card>
    </div>
  );
};

export default SearchPage;
