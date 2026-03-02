/**
 * API Service Layer
 * Handles all communication with the Flask backend
 */

import axios from 'axios';

// API base URL - Flask backend
const API_BASE_URL = 'http://localhost:5000/api';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minute timeout for analysis
  headers: {
    'Accept': 'application/json',
  },
});

/**
 * Upload X-ray image and get analysis results
 * @param {File} imageFile - The X-ray image file to analyze
 * @param {Function} onProgress - Progress callback (optional)
 * @returns {Promise} Analysis results
 */
export const uploadAndAnalyze = async (imageFile, onProgress = null) => {
  const formData = new FormData();
  formData.append('xray_image', imageFile);

  try {
    const response = await apiClient.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onProgress(percentCompleted);
        }
      },
    });

    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Get analysis by ID
 * @param {number} analysisId - The analysis ID
 * @returns {Promise} Analysis details
 */
export const getAnalysis = async (analysisId) => {
  try {
    const response = await apiClient.get(`/analysis/${analysisId}`);
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Get analysis history
 * @returns {Promise} List of previous analyses
 */
export const getHistory = async () => {
  try {
    const response = await apiClient.get('/history');
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Delete an analysis
 * @param {number} analysisId - The analysis ID to delete
 * @returns {Promise} Deletion result
 */
export const deleteAnalysis = async (analysisId) => {
  try {
    const response = await apiClient.delete(`/analysis/${analysisId}`);
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Health check endpoint
 * @returns {Promise} API health status
 */
export const healthCheck = async () => {
  try {
    const response = await apiClient.get('/health');
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Get image URL for display
 * @param {string} filename - Image filename
 * @returns {string} Full image URL
 */
export const getImageUrl = (filename) => {
  return `${API_BASE_URL}/images/${filename}`;
};

/**
 * Handle API errors consistently
 * @param {Error} error - The error object
 * @returns {Error} Formatted error
 */
const handleApiError = (error) => {
  if (error.response) {
    // Server responded with error
    const message = error.response.data?.error || 'An error occurred';
    const statusCode = error.response.status;

    if (statusCode === 413) {
      return new Error('File too large. Maximum size is 16MB.');
    } else if (statusCode === 400) {
      return new Error(message);
    } else if (statusCode === 404) {
      return new Error('Resource not found.');
    } else if (statusCode >= 500) {
      return new Error('Server error. Please try again later.');
    }

    return new Error(message);
  } else if (error.request) {
    // Request made but no response
    return new Error('Unable to connect to server. Please ensure the backend is running.');
  } else {
    // Error setting up request
    return new Error(error.message || 'An unexpected error occurred.');
  }
};

export default {
  uploadAndAnalyze,
  getAnalysis,
  getHistory,
  deleteAnalysis,
  healthCheck,
  getImageUrl,
};
