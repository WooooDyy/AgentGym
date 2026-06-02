/**
 * BabyAI Environment Client
 */

import BaseEnvClient from '../../../shared/services/baseClient.js';

class BabyAIClient extends BaseEnvClient {
  constructor() {
    super('/api/babyai');  // Uses vite proxy
    console.log('🔨 BabyAIClient initialized');
  }

  /**
   * Test connection to the BabyAI server
   * @returns {Promise<Object>} Connection test result
   */
  async testConnection() {
    try {
      const response = await this.request('/');
      console.log('✅ BabyAI connection test successful:', response);
      return { 
        success: true, 
        message: typeof response === 'string' ? response : 'Connected',
        connected: true 
      };
    } catch (error) {
      console.error('❌ BabyAI connection test failed:', error);
      return {
        success: false,
        error: error.message || 'Connection failed',
        connected: false
      };
    }
  }

  /**
   * Create a new BabyAI environment
   * @returns {Promise<Object>} Creation result
   */
  async createEnvironment() {
    try {
      const response = await this.request('/create', {
        method: 'POST'
      });
      
      if (!response || response.error) {
        throw new Error(response?.error || 'Failed to create environment');
      }
      
      console.log(`✅ BabyAI environment created successfully! ID: ${response.id}`);
      
      // Automatically reset the environment after creation
      const envId = response.id;
      const resetResult = await this.reset(envId, 0);
      if (!resetResult.success) {
        throw new Error(resetResult.error || 'Failed to initialize environment after creation');
      }
      
      return {
        success: true,
        data: {
          id: envId,
          environmentId: envId,
          ...resetResult.data // Include the reset data in the response
        }
      };
    } catch (error) {
      console.error('❌ Failed to create BabyAI environment:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Execute an action in the environment
   * @param {string|number} envId - Environment ID
   * @param {string} action - Action to execute
   * @returns {Promise<Object>} Step result
   */
  async step(envId, action) {
    try {
      const response = await this.request('/step', {
        method: 'POST',
        body: JSON.stringify({
          id: parseInt(envId),
          action: action
        })
      });
      
      if (!response || response.error) {
        throw new Error(response?.error || 'Failed to execute action');
      }
      
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('❌ BabyAI step failed:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Reset the environment
   * @param {string|number} envId - Environment ID
   * @param {number} dataIdx - Data index for task selection
   * @returns {Promise<Object>} Reset result
   */
  async reset(envId, dataIdx = 0) {
    try {
      console.log(`🔄 Resetting BabyAI environment ${envId} with dataIdx ${dataIdx}`);
      
      const response = await this.request('/reset', {
        method: 'POST',
        body: JSON.stringify({
          id: parseInt(envId),
          data_idx: parseInt(dataIdx)
        })
      });
      
      console.log('🔄 Reset response:', response);
      
      if (!response || response.error) {
        throw new Error(response?.error || 'Failed to reset environment');
      }
      
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('❌ BabyAI reset failed:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get current observation from the environment
   * @param {string|number} envId - Environment ID
   * @returns {Promise<Object>} Observation result
   */
  async getObservation(envId) {
    try {
      const response = await this.request(`/observation?id=${parseInt(envId)}`);
      
      if (!response || response.error) {
        throw new Error(response?.error || 'Failed to get observation');
      }
      
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('❌ Failed to get BabyAI observation:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Render the current state as an image
   * @param {string|number} envId - Environment ID
   * @returns {Promise<Object>} Render result with image data
   */
  async render(envId) {
    try {
      // First check if the environment is done to avoid errors
      const obsResult = await this.getObservation(envId);
      if (obsResult.success && obsResult.data.done) {
        console.warn('Cannot render BabyAI environment when done');
        return {
          success: false,
          error: 'Cannot render completed environment',
          data: null
        };
      }
      
      console.log(`🖼️ Rendering BabyAI environment ${envId}`);
      
      const response = await this.request('/render', {
        method: 'POST',
        body: JSON.stringify({
          id: parseInt(envId)
        })
      });
      
      // Log the full response for debugging
      console.log('🖼️ Render raw response:', response);
      
      // Check the structure of the response
      if (response) {
        console.log('🖼️ Render response type:', typeof response);
        if (typeof response === 'object') {
          console.log('🖼️ Render response keys:', Object.keys(response));
          
          if (response.image) {
            console.log('🖼️ Image data found, starting with:', response.image.substring(0, 50) + '...');
            console.log('🖼️ Image data length:', response.image.length);
          } else {
            console.log('🖼️ No image data in response');
          }
        }
      }
      
      if (!response || response.error) {
        throw new Error(response?.error || 'Failed to render environment');
      }
      
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('❌ BabyAI render failed:', error);
      console.error('Error details:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Close the environment
   * @param {string|number} envId - Environment ID
   * @returns {Promise<Object>} Close result
   */
  async close(envId) {
    try {
      const response = await this.request('/close', {
        method: 'POST',
        body: JSON.stringify({
          id: parseInt(envId)
        })
      });
      
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('❌ BabyAI close failed:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }
}

export default BabyAIClient;