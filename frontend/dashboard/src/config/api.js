/**
 * API Configuration for Dashboard
 * Uses environment variables with fallback to localhost
 */
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
  DASHBOARD_LOGS: `${API_BASE_URL}/api/dashboard/logs`,
  FEEDBACK: `${API_BASE_URL}/api/feedback`,
  RETRAIN: `${API_BASE_URL}/api/retrain`,
  RETRAIN_HISTORY: `${API_BASE_URL}/api/retrain/history`,
  PREDICT: `${API_BASE_URL}/predict`,
};

export default API_BASE_URL;

