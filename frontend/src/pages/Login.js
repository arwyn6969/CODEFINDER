import React, { useState, useEffect } from 'react';
import { Form, Input, Button, Card, Alert, Divider, Tag } from 'antd';
import { UserOutlined, LockOutlined } from '@ant-design/icons';
import AuthService from '../services/AuthService';

const Login = ({ onLogin }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [demoUsers, setDemoUsers] = useState([]);

  useEffect(() => {
    // Load demo users for development
    const loadDemoUsers = async () => {
      try {
        const users = await AuthService.getDemoUsers();
        setDemoUsers(users);
      } catch (error) {
        console.error('Error loading demo users:', error);
      }
    };
    
    loadDemoUsers();
  }, []);

  const handleSubmit = async (values) => {
    setLoading(true);
    setError('');
    
    try {
      const success = await onLogin(values);
      if (!success) {
        setError('Login failed. Please check your credentials.');
      }
    } catch (error) {
      setError(error.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  const handleDemoLogin = (username, password) => {
    handleSubmit({ username, password });
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #8B4513 0%, #D2691E 100%)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px'
    }}>
      <Card
        style={{
          width: '100%',
          maxWidth: '400px',
          boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
        }}
        title={
          <div style={{ textAlign: 'center' }}>
            <h2 style={{ margin: 0, color: '#8B4513' }}>
              Ancient Text Analyzer
            </h2>
            <p style={{ margin: '8px 0 0 0', color: '#666' }}>
              Sign in to your account
            </p>
          </div>
        }
      >
        {error && (
          <Alert
            message={error}
            type="error"
            style={{ marginBottom: '16px' }}
            closable
            onClose={() => setError('')}
          />
        )}

        <Form
          name="login"
          onFinish={handleSubmit}
          autoComplete="off"
          size="large"
        >
          <Form.Item
            name="username"
            rules={[
              { required: true, message: 'Please input your username!' }
            ]}
          >
            <Input
              prefix={<UserOutlined />}
              placeholder="Username"
            />
          </Form.Item>

          <Form.Item
            name="password"
            rules={[
              { required: true, message: 'Please input your password!' }
            ]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="Password"
            />
          </Form.Item>

          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              loading={loading}
              style={{
                width: '100%',
                backgroundColor: '#8B4513',
                borderColor: '#8B4513'
              }}
            >
              Sign In
            </Button>
          </Form.Item>
        </Form>

        {demoUsers.length > 0 && (
          <>
            <Divider>Demo Accounts</Divider>
            <div style={{ textAlign: 'center' }}>
              <p style={{ marginBottom: '12px', color: '#666', fontSize: '14px' }}>
                Click to login with demo credentials:
              </p>
              {demoUsers.map((user, index) => (
                <Tag
                  key={index}
                  color="blue"
                  style={{
                    cursor: 'pointer',
                    margin: '4px',
                    padding: '4px 8px'
                  }}
                  onClick={() => handleDemoLogin(user.username, user.password)}
                >
                  {user.username}
                </Tag>
              ))}
            </div>
          </>
        )}

        <div style={{ 
          textAlign: 'center', 
          marginTop: '24px',
          padding: '16px',
          background: '#f9f9f9',
          borderRadius: '6px'
        }}>
          <h4 style={{ margin: '0 0 8px 0', color: '#8B4513' }}>
            🔍 Features
          </h4>
          <ul style={{ 
            margin: 0, 
            padding: 0, 
            listStyle: 'none',
            fontSize: '13px',
            color: '#666'
          }}>
            <li>📄 Document Upload & OCR</li>
            <li>🔍 Pattern Detection</li>
            <li>📐 Geometric Analysis</li>
            <li>🔐 Cipher Detection</li>
            <li>📊 Interactive Visualizations</li>
            <li>📈 Comprehensive Reports</li>
          </ul>
        </div>
      </Card>
    </div>
  );
};

export default Login;