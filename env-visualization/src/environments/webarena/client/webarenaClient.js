/**
 * WebArena Environment Client
 * 
 * Handles communication with WebArena server with special handling for its unique response format.
 * 
 * IMPORTANT: WebArena uses 'terminated' instead of 'done' to indicate environment completion.
 * This client automatically maps 'terminated' to 'done' for consistency with other environments.
 */

import BaseEnvClient from '../../../shared/services/baseClient.js';

class WebArenaClient extends BaseEnvClient {
  constructor() {
    super('/api/webarena');  // Uses vite proxy
    console.log('🔨 WebArenaClient initialized');
  }

  /**
   * Test connection to the WebArena server
   * @returns {Promise<Object>} Connection test result
   */
  async testConnection() {
    try {
      const response = await this.request('/');
      console.log('✅ WebArena connection test successful:', response);
      return { 
        success: true, 
        message: typeof response === 'string' ? response : 'Connected',
        connected: true 
      };
    } catch (error) {
      console.error('❌ WebArena connection test failed:', error);
      return {
        success: false,
        error: error.message || 'Connection failed',
        connected: false
      };
    }
  }

  /**
   * Create a new WebArena environment
   * @returns {Promise<Object>} Creation result
   */
  async createEnvironment() {
    try {
      const response = await this.request('/create', {  // Fixed endpoint
        method: 'POST'
      },{
        timeout: 60000 // 60 second timeout for create operations
      });
      
      if (!response || response.error) {
        throw new Error(response?.error || 'Failed to create environment');
      }
      
      console.log(`✅ WebArena environment created successfully! ID: ${response.env_idx}`);
      
      // Automatically reset the environment after creation
      const envId = response.env_idx;
      const resetResult = await this.reset(envId, 42, 0);
      
      if (!resetResult.success) {
        throw new Error(resetResult.error || 'Failed to initialize environment after creation');
      }
      
      const finalData = {
        id: envId,
        environmentId: envId,
        ...resetResult.data // Include the reset data in the response
      };
      
      console.log('✅ WebArena environment initialized with observation');
      
      return {
        success: true,
        data: finalData
      };
    } catch (error) {
      console.error('❌ Failed to create WebArena environment:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Execute an action in the environment
   * @param {string|number} envId - Environment ID (env_idx)
   * @param {string} action - Action to execute
   * @returns {Promise<Object>} Step result
   */
  async step(envId, action) {
    try {
      console.log(`🚀 WebArena step - EnvID: ${envId}, Action: "${action}"`);
      
      const stepData = {
        env_idx: parseInt(envId),
        action: action
      };
      
      console.log(`📤 Sending step request:`, stepData);
      
      const response = await this.request('/step', {
        method: 'POST',
        body: JSON.stringify(stepData),
        timeout: 120000 // 120 second timeout for step operations
      });
      
      console.log(`📥 Received step response:`, response);
      
      if (!response || response.error) {
        throw new Error(response?.error || 'Failed to execute action');
      }
      
      // Process WebArena-specific response format
      const processedResponse = this.processStepResponse(response);
      
      return {
        success: true,
        data: processedResponse
      };
    } catch (error) {
      console.error('❌ WebArena step failed:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Process WebArena step response to handle terminated -> done mapping
   * WebArena returns (prompt, reward, terminated, truncated, info)
   * @param {*} response - Raw response from server
   * @returns {Object} Processed response with consistent format
   */
  processStepResponse(response) {
    // WebArena server returns tuple: (prompt, reward, terminated, truncated, info)
    if (Array.isArray(response) && response.length >= 5) {
      const [prompt, reward, terminated, truncated, info] = response;
      
      return {
        observation: prompt,
        reward: reward || 0,
        done: terminated || false,  // Map terminated to done for consistency
        terminated: terminated || false,  // Keep original field too
        truncated: truncated || false,
        info: info || {},
        action: null // Will be set by calling code if needed
      };
    }
    
    // Handle case where response is already an object
    if (typeof response === 'object' && response !== null) {
      // If it already has terminated field, map it to done
      if ('terminated' in response) {
        return {
          observation: response.observation || response.prompt || '',
          reward: response.reward || 0,
          done: response.terminated || false,  // Map terminated to done
          terminated: response.terminated || false,
          truncated: response.truncated || false,
          info: response.info || {},
          action: response.action || null
        };
      }
      
      // If it already has done field, keep as is
      if ('done' in response) {
        return {
          observation: response.observation || '',
          reward: response.reward || 0,
          done: response.done || false,
          terminated: response.done || false,  // Map done to terminated too
          truncated: response.truncated || false,
          info: response.info || {},
          action: response.action || null
        };
      }
      
      // Default processing for object responses
      return {
        observation: response.observation || response.prompt || JSON.stringify(response),
        reward: response.reward || 0,
        done: false,  // Default to not done if no indication
        terminated: false,
        truncated: false,
        info: response.info || response,
        action: response.action || null
      };
    }
    
    // Handle string responses (treat as observation)
    if (typeof response === 'string') {
      return {
        observation: response,
        reward: 0,
        done: false,
        terminated: false,
        truncated: false,
        info: {},
        action: null
      };
    }
    
    // Fallback for unexpected response format
    console.warn('⚠️ Unexpected WebArena step response format:', response);
    return {
      observation: String(response),
      reward: 0,
      done: false,
      terminated: false,
      truncated: false,
      info: {},
      action: null
    };
  }

  /**
   * Reset the environment
   * @param {string|number} envId - Environment ID
   * @param {number} seed - Random seed
   * @param {number} idx - Configuration index
   * @returns {Promise<Object>} Reset result
   */
  async reset(envId, seed = 0, idx = 0) {
    try {
      console.log(`🔄 Resetting WebArena environment ${envId} with seed ${seed}, idx ${idx}`);
      
      const response = await this.request('/reset', {
        method: 'POST',
        body: JSON.stringify({
          env_idx: parseInt(envId),
          seed: idx,
          idx: seed,
          options: null  // Added options field as expected by server
        }),
        timeout: 120000 // 120 second timeout for reset operations
      });
      
      console.log('🔄 Reset response received:', response);
      
      if (!response || response.error) {
        throw new Error(response?.error || 'Failed to reset environment');
      }
      
      // Process the reset response to ensure proper data structure
      let processedData = response;
      
      // If response is a string (observation), wrap it in proper structure
      if (typeof response === 'string') {
        processedData = {
          initialObservation: response,
          observation: response,
          done: false,
          terminated: false
        };
      } else if (typeof response === 'object' && response !== null) {
        // If response is an object, ensure it has the right structure
        processedData = {
          ...response,
          // If there's an observation field but no initialObservation, copy it
          initialObservation: response.initialObservation || response.observation || response.prompt,
          observation: response.observation || response.prompt || response.initialObservation,
          done: response.done || response.terminated || false,
          terminated: response.terminated || response.done || false
        };
      }
      
      console.log('🔄 Processed reset data:', processedData);
      
      return {
        success: true,
        data: processedData
      };
    } catch (error) {
      console.error('❌ WebArena reset failed:', error);
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
      const response = await this.request(`/observation?env_idx=${parseInt(envId)}`);
      
      console.log('🔍 WebArena getObservation response type:', typeof response);
      console.log('🔍 WebArena getObservation response:', response);
      
      if (!response || response.error) {
        throw new Error(response?.error || 'Failed to get observation');
      }
      
      // Return the observation data as-is for proper processing by the frontend
      // The frontend will handle parsing OBJECTIVE from the observation text
      return {
        success: true,
        data: response  // Keep the original response format
      };
    } catch (error) {
      console.error('❌ Failed to get WebArena observation:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get observation metadata (including screenshot if available)
   * @param {string|number} envId - Environment ID
   * @returns {Promise<Object>} Metadata result
   */
  async getObservationMetadata(envId) {
    try {
      const response = await this.request(`/observation_metadata?env_idx=${parseInt(envId)}`);
      
      if (!response || response.error) {
        throw new Error(response?.error || 'Failed to get observation metadata');
      }
      
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('❌ Failed to get WebArena observation metadata:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get screenshot from the environment
   * @param {string|number} envId - Environment ID
   * @returns {Promise<Object>} Screenshot result
   */
  async getScreenshot(envId) {
    try {
      // WebArena doesn't have a dedicated screenshot endpoint,
      // so we get it through observation metadata
      const metadataResult = await this.getObservationMetadata(envId);
      
      if (metadataResult.success && metadataResult.data && metadataResult.data.image) {
        return {
          success: true,
          data: metadataResult.data.image
        };
      }
      
      return {
        success: true,
        data: null // No image available
      };
    } catch (error) {
      console.error('❌ Failed to get WebArena screenshot:', error);
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
      const response = await this.request('/close', {  // Now using the actual close endpoint
        method: 'POST',
        body: JSON.stringify({
          env_idx: parseInt(envId)
        })
      });
      
      console.log(`WebArena environment ${envId} close response:`, response);
      return {
        success: true,
        data: response
      };
    } catch (error) {
      console.error('❌ WebArena close failed:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }
}

export default WebArenaClient;


