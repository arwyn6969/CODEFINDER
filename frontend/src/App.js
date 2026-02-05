import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate, Link } from 'react-router-dom';
import { Layout, Menu, Avatar, Dropdown, notification } from 'antd';
import { 
  FileTextOutlined, 
  BarChartOutlined, 
  SearchOutlined, 
  SettingOutlined,
  UserOutlined,
  LogoutOutlined,
  HomeOutlined
} from '@ant-design/icons';

import Dashboard from './pages/Dashboard';
import DocumentUpload from './pages/DocumentUpload';
import DocumentList from './pages/DocumentList';
import DocumentAnalysis from './pages/DocumentAnalysis';
import SearchPage from './pages/SearchPage';
import Login from './pages/Login';
import ResearchDashboard from './pages/research/ResearchDashboard';
import AuthService from './services/AuthService';
import WebSocketService from './services/WebSocketService';

const { Header, Sider, Content } = Layout;

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    // Check if user is already logged in
    const checkAuth = async () => {
      try {
        const currentUser = await AuthService.getCurrentUser();
        if (currentUser) {
          setUser(currentUser);
          // Initialize WebSocket connection
          WebSocketService.connect(currentUser.username);
        }
      } catch (error) {
        console.log('No active session');
      } finally {
        setLoading(false);
      }
    };

    checkAuth();

    // Cleanup WebSocket on unmount
    return () => {
      WebSocketService.disconnect();
    };
  }, []);

  const handleLogin = async (credentials) => {
    try {
      const response = await AuthService.login(credentials);
      setUser(response.user);
      
      // Initialize WebSocket connection
      WebSocketService.connect(response.user.username);
      
      notification.success({
        message: 'Login Successful',
        description: `Welcome back, ${response.user.username}!`
      });
      
      return true;
    } catch (error) {
      notification.error({
        message: 'Login Failed',
        description: error.message || 'Invalid credentials'
      });
      return false;
    }
  };

  const handleLogout = async () => {
    try {
      await AuthService.logout();
      setUser(null);
      WebSocketService.disconnect();
      
      notification.success({
        message: 'Logged Out',
        description: 'You have been successfully logged out'
      });
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  const userMenu = (
    <Menu>
      <Menu.Item key="profile" icon={<UserOutlined />}>
        Profile
      </Menu.Item>
      <Menu.Item key="settings" icon={<SettingOutlined />}>
        Settings
      </Menu.Item>
      <Menu.Divider />
      <Menu.Item key="logout" icon={<LogoutOutlined />} onClick={handleLogout}>
        Logout
      </Menu.Item>
    </Menu>
  );

  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh' 
      }}>
        <div className="loading-spinner"></div>
      </div>
    );
  }

  if (!user) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider 
        collapsible 
        collapsed={collapsed} 
        onCollapse={setCollapsed}
        theme="light"
        style={{
          boxShadow: '2px 0 6px rgba(0,21,41,.35)'
        }}
      >
        <div style={{ 
          padding: '16px', 
          textAlign: 'center',
          borderBottom: '1px solid #f0f0f0'
        }}>
          <h3 style={{ 
            margin: 0, 
            color: '#8B4513',
            fontSize: collapsed ? '14px' : '16px'
          }}>
            {collapsed ? 'ATA' : 'Ancient Text Analyzer'}
          </h3>
        </div>
        
        <Menu
          mode="inline"
          defaultSelectedKeys={['dashboard']}
          style={{ borderRight: 0 }}
        >
          <Menu.Item key="dashboard" icon={<HomeOutlined />}>
            <Link to="/">Dashboard</Link>
          </Menu.Item>
          <Menu.Item key="upload" icon={<FileTextOutlined />}>
            <Link to="/upload">Upload Document</Link>
          </Menu.Item>
          <Menu.Item key="documents" icon={<FileTextOutlined />}>
            <Link to="/documents">My Documents</Link>
          </Menu.Item>
          <Menu.Item key="search" icon={<SearchOutlined />}>
            <Link to="/search">Search & Query</Link>
          </Menu.Item>
          <Menu.Item key="analysis" icon={<BarChartOutlined />}>
            <Link to="/analysis">Analysis Tools</Link>
          </Menu.Item>
        </Menu>
      </Sider>
      
      <Layout>
        <Header style={{ 
          background: '#fff', 
          padding: '0 24px',
          boxShadow: '0 1px 4px rgba(0,21,41,.08)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <h2 style={{ margin: 0, color: '#8B4513' }}>
            Ancient Text Analysis System
          </h2>
          
          <Dropdown overlay={userMenu} placement="bottomRight">
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              cursor: 'pointer',
              padding: '8px'
            }}>
              <Avatar 
                size="small" 
                icon={<UserOutlined />} 
                style={{ backgroundColor: '#8B4513', marginRight: '8px' }}
              />
              <span>{user.username}</span>
            </div>
          </Dropdown>
        </Header>
        
        <Content style={{ 
          margin: '24px',
          background: '#fff',
          borderRadius: '8px',
          overflow: 'auto'
        }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<DocumentUpload />} />
            <Route path="/documents" element={<DocumentList />} />
            <Route path="/documents/:id" element={<DocumentAnalysis />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/research" element={<ResearchDashboard />} />
            <Route path="/analysis" element={<Navigate to="/documents" />} />
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  );
}

export default App;