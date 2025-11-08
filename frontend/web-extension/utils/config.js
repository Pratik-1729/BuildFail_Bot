/**
 * Configuration for Web Extension
 * API URL defaults to localhost - can be changed in code if needed
 */
// Default API URL - change this if your backend runs on a different URL
const API_BASE_URL = 'http://localhost:8000';

// Export API endpoints
export const API_ENDPOINTS = {
  PREDICT: `${API_BASE_URL}/predict`,
  MANUAL: `${API_BASE_URL}/api/logs/manual`,
  FEEDBACK: `${API_BASE_URL}/api/feedback`,
};

export default API_BASE_URL;

