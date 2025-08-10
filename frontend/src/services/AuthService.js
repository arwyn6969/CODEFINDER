import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

class AuthService {
  constructor() {
    this.token = localStorage.getItem('auth_token');
    this.setupAxiosInterceptors();
  }

  setupAxiosInterceptors() {
    // Request interceptor to add auth token
    axios.interceptors.request.use(
      (config) => {
        if (this.token) {
          config.headers.Authorization = `Bearer ${this.token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor to handle auth errors
    axios.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          this.logout();
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  async login(credentials) {
    try {
      const response = await axios.post(`${API_BASE_URL}/auth/login`, credentials);
      const { access_token, user } = response.data;
      
      this.token = access_token;
      localStorage.setItem('auth_token', access_token);
      localStorage.setItem('user', JSON.stringify(user));
      
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Login failed');
    }
  }

  async register(userData) {
    try {
      const response = await axios.post(`${API_BASE_URL}/auth/register`, userData);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Registration failed');
    }
  }

  async logout() {
    try {
      if (this.token) {
        await axios.post(`${API_BASE_URL}/auth/logout`);
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      this.token = null;
      localStorage.removeItem('auth_token');
      localStorage.removeItem('user');
    }
  }

  async getCurrentUser() {
    try {
      if (!this.token) {
        throw new Error('No token available');
      }
      
      const response = await axios.get(`${API_BASE_URL}/auth/me`);
      return response.data;
    } catch (error) {
      // Try to get user from localStorage as fallback
      const storedUser = localStorage.getItem('user');
      if (storedUser) {
        return JSON.parse(storedUser);
      }
      throw error;
    }
  }

  async refreshToken() {
    try {
      const response = await axios.post(`${API_BASE_URL}/auth/refresh`);
      const { access_token } = response.data;
      
      this.token = access_token;
      localStorage.setItem('auth_token', access_token);
      
      return access_token;
    } catch (error) {
      this.logout();
      throw error;
    }
  }

  isAuthenticated() {
    return !!this.token;
  }

  getToken() {
    return this.token;
  }

  async getDemoUsers() {
    try {
      const response = await axios.get(`${API_BASE_URL}/auth/demo-users`);
      return response.data.demo_users;
    } catch (error) {
      console.error('Error fetching demo users:', error);
      return [];
    }
  }
}

export default new AuthService();