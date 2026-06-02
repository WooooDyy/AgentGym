/**
 * SciWorld Environment Client
 */

import BaseEnvClient from '../../../shared/services/baseClient.js';

class SciWorldClient extends BaseEnvClient {
  constructor() {
    // Use the vite proxy path for API calls
    super('/api/sciworld');
    console.log('🔬 SciWorldClient initialized');
  }

  /**
   * Test connection to the SciWorld server
   * @returns {Promise<Object>} Connection test result
   */
  async testConnection() {
    try {
      const response = await this.request('/');
      console.log('✅ SciWorld connection test successful:', response);
      return { 
        success: true, 
        message: response?.message || 'Connected',
        connected: true 
      };
    } catch (error) {
      console.error('❌ SciWorld connection test failed:', error);
      return {
        success: false,
        error: error.message || 'Connection failed',
        connected: false
      };
    }
  }

  /**
   * Create a new SciWorld environment
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
      
      console.log(`✅ SciWorld environment created successfully! ID: ${response.id}`);
      
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
      console.error('❌ Failed to create SciWorld environment:', error);
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
      const response = await this.request('/step_visual', {
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
      console.error('❌ SciWorld step failed:', error);
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
      console.log(`🔄 Resetting SciWorld environment ${envId} with dataIdx ${dataIdx}`);
      
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
      console.error('❌ SciWorld reset failed:', error);
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
      console.error('❌ Failed to get SciWorld observation:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get environment state for visualization
   * @param {string|number} envId - Environment ID
   * @returns {Promise<Object>} State result
   */
  async getState(envId) {
    try {
      const response = await this.request(`/state?id=${parseInt(envId)}`);
      
      if (!response || response.error) {
        throw new Error(response?.error || 'Failed to get state');
      }
      
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('❌ Failed to get SciWorld state:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get object tree structure for visualization
   * @param {string|number} envId - Environment ID
   * @returns {Promise<Object>} Object tree result
   */
  async getObjectTree(envId) {
    try {
      const response = await this.request(`/object_tree?id=${parseInt(envId)}`);
      
      if (!response || response.error) {
        throw new Error(response?.error || 'Failed to get object tree');
      }
      
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('❌ Failed to get SciWorld object tree:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get inventory information
   * @param {string|number} envId - Environment ID
   * @returns {Promise<Object>} Inventory result
   */
  async getInventory(envId) {
    try {
      const response = await this.request(`/inventory?id=${parseInt(envId)}`);
      
      if (!response || response.error) {
        throw new Error(response?.error || 'Failed to get inventory');
      }
      
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('❌ Failed to get SciWorld inventory:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get task description
   * @param {string|number} envId - Environment ID
   * @returns {Promise<Object>} Task description result
   */
  async getTaskDescription(envId) {
    try {
      const response = await this.request(`/task_description?id=${parseInt(envId)}`);
      
      if (!response || response.error) {
        throw new Error(response?.error || 'Failed to get task description');
      }
      
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('❌ Failed to get SciWorld task description:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get possible actions
   * @param {string|number} envId - Environment ID
   * @returns {Promise<Object>} Possible actions result
   */
  async getPossibleActions(envId) {
    try {
      const response = await this.request(`/possible_actions?id=${parseInt(envId)}`);
      
      if (!response || response.error) {
        throw new Error(response?.error || 'Failed to get possible actions');
      }
      
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('❌ Failed to get SciWorld possible actions:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get action hints
   * @param {string|number} envId - Environment ID
   * @returns {Promise<Object>} Action hints result
   */
  async getActionHints(envId) {
    try {
      const response = await this.request(`/hints?id=${parseInt(envId)}`);
      
      if (!response || response.error) {
        throw new Error(response?.error || 'Failed to get action hints');
      }
      
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('❌ Failed to get SciWorld action hints:', error);
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
      console.error('❌ SciWorld close failed:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }
}

export default SciWorldClient; 