import axios from 'axios';

const API_URL = '/api/research'; // Proxy setup in package.json handles the host
const REL_API_URL = '/api/relationships';
const ANALYSIS_API_URL = '/api/analysis';

const researchService = {
  // --- Documents ---
  getDocumentContent: async (documentId) => {
    // Note: Assuming documents.py is mounted at /api/documents
    // We might need to adjust API_URL or hardcode the path
    const response = await axios.get(`/api/documents/${documentId}/content`);
    return response.data;
  },

  listDocuments: async () => {
     const response = await axios.get('/api/documents/');
     return response.data;
  },
  
  listAllPatterns: async (params = {}) => {
     const response = await axios.get('/api/patterns/', { params });
     return response.data;
  },

  uploadDocument: async (file) => {
     const formData = new FormData();
     formData.append('file', file);
     const response = await axios.post('/api/documents/upload', formData, {
        headers: {
           'Content-Type': 'multipart/form-data'
        }
     });
     return response.data;
  },

  // --- ELS & Transliteration ---
  transliterate: async (text) => {
    const response = await axios.post(`${API_URL}/transliterate`, { text });
    return response.data;
  },

  findELS: async ({ text, terms, minSkip, maxSkip, autoTransliterate, source, documentId }) => {
    const response = await axios.post(`${API_URL}/els`, {
      text,
      terms,
      min_skip: minSkip,
      max_skip: maxSkip,
      auto_transliterate: autoTransliterate,
      source,
      document_id: documentId,
      save: true
    });
    return response.data;
  },

  getELSVisualization: async ({ source, documentId, text, centerIndex, skip, rows = 20, cols = 20, termLength }) => {
    const response = await axios.post(`${API_URL}/els/visualize`, {
      source,
      document_id: documentId,
      text,
      center_index: centerIndex,
      skip,
      rows,
      cols,
      term_length: termLength
    });
    return response.data;
  },

  // --- Gematria ---
  calculateGematria: async (text) => {
    const response = await axios.post(`${API_URL}/gematria`, { text });
    return response.data;
  },

  // --- Network ---
  getRelationshipNetwork: async (documentIds) => {
    const response = await axios.post(`${REL_API_URL}/network`, { document_ids: documentIds });
    return response.data;
  },
  
  // --- Geometric ---
  getGeometricAnalysis: async (documentId) => {
    const response = await axios.get(`${ANALYSIS_API_URL}/${documentId}/geometric`);
    return response.data;
  },

  // --- Prophetic ---
  findPropheticConvergence: async ({ terms, maxSpread, generateVisual }) => {
    const response = await axios.post(`${API_URL}/prophetic/convergence`, {
      terms,
      max_spread: maxSpread,
      generate_visual: generateVisual
    });
    return response.data;
  }
};

export default researchService;
