import React, { useState } from 'react';
import { Alert, Layout, Tabs } from 'antd';
import { SearchOutlined, ShareAltOutlined, BookOutlined, DesktopOutlined } from '@ant-design/icons';

import TheScope from './components/TheScope';
import TheDesk from './components/TheDesk';
import TheMap from './components/TheMap';
import TheLibrary from './components/TheLibrary';
import TheGeometry from './components/TheGeometry';

const { Content } = Layout;

const ResearchDashboard = () => {
  const [activeTab, setActiveTab] = useState('scope');
  const [currentDocument, setCurrentDocument] = useState(null);

  const handleOpenDocument = (doc) => {
    setCurrentDocument(doc);
    setActiveTab('desk');
  };
  
  return (
    <Layout style={{ minHeight: '100vh', background: '#141414' }}>
      <Content style={{ padding: '24px' }}>
        <h1 style={{ color: '#fff', marginBottom: '16px' }}>Legacy Exploratory Lab</h1>
        <Alert
          type="warning"
          showIcon
          style={{ marginBottom: '24px' }}
          message="Internal legacy surface"
          description="This area preserves ELS, gematria, prophetic, and geographic-style exploratory tools for auditability and secondary analysis. It is not the primary product workflow."
        />
        
        <Tabs 
          activeKey={activeTab} 
          onChange={setActiveTab} 
          type="card"
          theme="dark"
          items={[
            {
              label: <span><SearchOutlined />The Scope</span>,
              key: 'scope',
              children: <TheScope onOpenDocument={handleOpenDocument} />
            },
            {
              label: <span><DesktopOutlined />The Desk</span>,
              key: 'desk',
              children: <TheDesk currentDocument={currentDocument} onSetDocument={setCurrentDocument} />
            },
            {
              label: <span><DesktopOutlined />Geometry</span>,
              key: 'geometry',
              children: <TheGeometry currentDocument={currentDocument} onOpenDocument={handleOpenDocument} />
            },
            {
              label: <span><ShareAltOutlined />The Map</span>,
              key: 'map',
              children: <TheMap onOpenDocument={handleOpenDocument} />
            },
             {
              label: <span><BookOutlined />The Library</span>,
              key: 'library',
              children: <TheLibrary />
            }
          ]}
        />
      </Content>
    </Layout>
  );
};

export default ResearchDashboard;
