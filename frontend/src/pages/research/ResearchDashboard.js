import React, { useState } from 'react';
import { Layout, Tabs } from 'antd';
import { SearchOutlined, TranslationOutlined, ShareAltOutlined, BookOutlined, DesktopOutlined } from '@ant-design/icons';

import TheScope from './components/TheScope';
import TheDesk from './components/TheDesk';
import TheMap from './components/TheMap';
import TheLibrary from './components/TheLibrary';

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
        <h1 style={{ color: '#fff', marginBottom: '24px' }}>🔬 Research Dashboard</h1>
        
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
