/**
 * TextCraft Environment Client
 * 
 * Client implementation for TextCraft environment that extends BaseEnvClient
 */

import BaseEnvClient from '../../../shared/services/baseClient.js';
import assetManager from '../../../shared/services/assetManager.js';

class TextCraftClient extends BaseEnvClient {
  constructor() {
    super('/api/textcraft');  // Uses vite proxy
    this.inventoryCache = new Map();
    this.interactionHistory = new Map();
    this.goalCache = new Map();
    
    // Event listeners
    this.eventListeners = {
      'state-changed': [],
      'inventory-updated': [],
      'goal-achieved': []
    };
    
    // Initialize asset manager
    this.initializeAssets();
    
    console.log('🔨 TextCraftClient initialized');
  }

  /**
   * Initialize asset manager
   */
  async initializeAssets() {
    try {
      await assetManager.loadManifest();
      console.log('✅ Asset manifest loaded');
    } catch (error) {
      console.error('❌ Error loading asset manifest:', error);
    }
  }

  /**
   * Test connection to the TextCraft server
   * @returns {Promise<Object>} Connection test result
   */
  async testConnection() {
    try {
      console.log('🔍 Testing TextCraft server connection...');
      
      const response = await this.request('/');
      
      console.log('✅ TextCraft connection test successful:', response);
      return { 
        success: true, 
        message: typeof response === 'string' ? response : 'Connected',
        connected: true 
      };
    } catch (error) {
      console.error('❌ TextCraft connection test failed:', error);
      return {
        success: false,
        error: error.message || 'Connection failed',
        connected: false
      };
    }
  }

  /**
   * Create a new TextCraft environment
   * @param {Object} config - Environment configuration
   * @returns {Promise<Object>} Creation result
   */
  async createEnvironment(config = {}) {
    try {
      console.log('🏗️ Creating TextCraft environment with config:', config);
      
      const requestBody = {
        commands: config.commands || null,
        goal: config.goal || null
      };
      
      const response = await this.request('/create', {
        method: 'POST',
        body: JSON.stringify(requestBody)
      });
      
      // Extract environment ID
      let environmentId = null;
      if (typeof response === 'object' && response !== null) {
        environmentId = response.id || response.environmentId || response.env_id;
      } else if (typeof response === 'number') {
        environmentId = response;
      }
      
      if (environmentId === null || environmentId === undefined) {
        throw new Error('No environment ID returned from server');
      }
      
      const numericId = parseInt(environmentId);
      
      // Get initial observation
      let observationText = '';
      if (typeof response === 'object' && response !== null) {
        observationText = this.processObservation(response);
        
        // Cache the goal if it exists in the response
        this.extractAndCacheGoal(numericId, observationText);
      }
      
      // Initialize inventory cache
      this.inventoryCache.set(numericId, []);
      
      // Initialize interaction history
      this.interactionHistory.set(numericId, []);
      
      console.log(`✅ TextCraft environment created successfully! ID: ${numericId}`);
      
      return {
        success: true,
        data: {
          id: numericId,
          environmentId: numericId,
          observation: observationText,
          reward: response.reward || 0,
          done: response.done || false,
          size: 1
        }
      };
    } catch (error) {
      console.error('❌ Failed to create TextCraft environment:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get current observation from the environment
   * @param {string|number} envId - Environment ID
   * @returns {Promise<Object>} Observation with success flag and data
   */
  async getObservation(envId) {
    try {
      const numericId = parseInt(envId);
      
      console.log(`👁️ Getting TextCraft observation for environment ${numericId}`);
      
      const response = await this.request(`/observation?id=${numericId}`);
      
      const observationText = this.processObservation(response);
      
      console.log(`✅ TextCraft observation retrieved successfully`);
      
      return {
        success: true,
        data: observationText
      };
    } catch (error) {
      console.error(`❌ Failed to get TextCraft observation:`, error);
      return {
        success: false,
        error: error.message,
        data: `Error: ${error.message}`
      };
    }
  }

  /**
   * Execute an action in the environment
   * @param {string|number} envId - Environment ID
   * @param {string} action - Action to execute
   * @returns {Promise<Object>} Step result with success flag and data
   */
  async step(envId, action) {
    try {
      const numericId = parseInt(envId);
      const cleanedAction = String(action).trim();
      
      console.log(`🎮 Executing action: "${cleanedAction}" in environment ${numericId}`);
      
      const requestBody = {
        id: numericId,
        action: cleanedAction
      };
      
      const response = await this.request('/step', {
        method: 'POST',
        body: JSON.stringify(requestBody)
      });
      
      // Process observation
      const observationText = this.processObservation(response);
      
      // Extract reward and done state
      let reward = 0;
      let done = false;
      
      if (typeof response === 'object' && response !== null) {
        reward = response.reward || 0;
        done = response.done || false;
      }
      
      // 修改: 对所有动作都更新库存，而不仅限于inventory命令
      this.updateInventoryFromObservation(numericId, observationText);
      
      // Add to interaction history
      this.addToHistory(numericId, cleanedAction, observationText);
      
      // Check if goal is achieved
      if (done && reward > 0) {
        console.log(`🎉 Goal achieved in environment ${numericId}!`);
        this.emit('goal-achieved', { environmentId: numericId, observation: observationText });
      }
      
      console.log(`✅ Action executed successfully. Reward: ${reward}, Done: ${done}`);
      
      return {
        success: true,
        data: {
          observation: observationText,
          reward: reward,
          done: done,
          state: observationText
        }
      };
    } catch (error) {
      console.error('❌ TextCraft step failed:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Reset the environment
   * @param {string|number} envId - Environment ID
   * @param {number} idx - Data index for task selection
   * @returns {Promise<Object>} Reset result with success flag and data
   */
  async reset(envId, idx = 0) {
    try {
      const numericId = parseInt(envId);
      
      // Ensure idx is a number but don't limit the range
      const dataIdx = parseInt(idx) || 0;
      
      console.log(`🔄 Resetting TextCraft environment ${numericId} with data_idx=${dataIdx}`);
      
      const requestBody = {
        id: numericId,
        data_idx: dataIdx
      };
      
      const response = await this.request('/reset', {
        method: 'POST',
        body: JSON.stringify(requestBody)
      });
      
      const observationText = this.processObservation(response);
      
      // Extract and cache goal
      this.extractAndCacheGoal(numericId, observationText);
      
      // Clear inventory and history
      this.inventoryCache.set(numericId, []);
      this.interactionHistory.set(numericId, []);
      
      console.log(`✅ TextCraft environment reset successfully`);
      console.log(`📋 Reset response:`, observationText);
      
      // Emit state changed event
      this.emit('state-changed', {
        environmentId: numericId,
        observation: observationText,
        reset: true
      });
      
      return {
        success: true,
        data: observationText
      };
    } catch (error) {
      console.error('❌ TextCraft reset failed:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Extract and cache goal from observation
   * @param {number} environmentId - Environment ID
   * @param {string} observation - Observation text
   * @returns {string|null} Extracted goal
   */
  extractAndCacheGoal(environmentId, observation) {
    try {
      if (!observation) return null;
      
      // Extract goal using regex
      const goalMatch = observation.match(/Goal:\s*(?:craft\s+)?([^.]+)\.?/i);
      if (goalMatch && goalMatch[1]) {
        const goal = goalMatch[1].trim();
        this.goalCache.set(environmentId, goal);
        console.log(`📝 Extracted and cached goal for env ${environmentId}: ${goal}`);
        return goal;
      }
      
      return null;
    } catch (error) {
      console.error('❌ Error extracting goal:', error);
      return null;
    }
  }

  /**
   * Get cached goal for an environment
   * @param {number} environmentId - Environment ID
   * @returns {string} Cached goal or empty string
   */
  getGoal(environmentId) {
    return this.goalCache.get(parseInt(environmentId)) || '';
  }

  /**
   * Update inventory cache from observation text
   * @param {number} environmentId - Environment ID
   * @param {string} observation - Observation text
   */
  updateInventoryFromObservation(environmentId, observation) {
    try {
      // 如果是物品使用或合成相关的观察结果，需要检查库存信息
      const obsStr = typeof observation === 'string' 
        ? observation 
        : JSON.stringify(observation);
      
      // 1. 尝试直接从观察文本解析库存
      let inventory = [];
      const inventoryMatch = obsStr.match(/Inventory:\s*(.+?)(?:\n|$)/i);
      
      if (inventoryMatch && inventoryMatch[1]) {
        const inventoryText = inventoryMatch[1].trim();
        
        // 检查是否为空库存
        if (inventoryText.includes("You are not carrying anything") || 
            inventoryText.trim() === '' || 
            inventoryText.includes('empty') ||
            inventoryText.includes('nothing')) {
          console.log('🔄 Empty inventory detected');
          inventory = [];
        } else {
          // 解析物品格式：[item name] (count)
          const itemMatches = Array.from(inventoryText.matchAll(/\[([^\]]+)\]\s*\((\d+)\)/g));
          
          for (const match of itemMatches) {
            const itemName = match[1].trim();
            const count = parseInt(match[2]);
            const normalizedName = this.normalizeItemName(itemName);
            
            // 获取物品数据
            const itemData = assetManager.getItem(normalizedName);
            
            inventory.push({
              count,
              name: itemName,
              itemName: normalizedName,
              normalizedName,
              ...itemData
            });
          }
          
          console.log(`📦 Parsed inventory: ${inventory.length} items`);
        }
      }
      
      // 2. 如果直接解析失败，尝试使用assetManager
      if (inventory.length === 0) {
        inventory = assetManager.parseInventory(obsStr);
        console.log(`📦 Using assetManager parsed inventory: ${inventory.length} items`);
      }
      
      // 3. 保存到缓存
      this.inventoryCache.set(environmentId, inventory);
      
      // 4. 发送事件通知
      this.emit('inventory-updated', {
        environmentId,
        inventory
      });
      
      return inventory;
    } catch (error) {
      console.error('❌ Error parsing inventory:', error);
      return [];
    }
  }

  /**
   * 规范化物品名称，处理特殊情况
   * @param {string} itemName - 物品名称
   * @returns {string} 规范化的物品名称
   */
  normalizeItemName(itemName) {
    if (!itemName) return '';
    
    // 转换为小写，替换空格为下划线
    let normalized = itemName.toLowerCase().replace(/\s+/g, '_');
    
    // 特殊物品名称映射
    const specialItems = {
      'crimson_stem': 'crimson_stem',
      'crimson_hyphae': 'crimson_hyphae',
      'cocoa_bean': 'cocoa_beans',
      'cocoa_beans': 'cocoa_beans',
      'cookie': 'cookie',
      'wheat': 'wheat'
    };
    
    // 检查特殊物品名称映射
    if (specialItems[normalized]) {
      normalized = specialItems[normalized];
    }
    
    return normalized;
  }

  /**
   * Get inventory for an environment
   * @param {number} environmentId - Environment ID
   * @returns {Promise<Array>} Inventory items
   */
  async getInventory(environmentId) {
    const cachedInventory = this.inventoryCache.get(parseInt(environmentId)) || [];
    
    // If cache is empty, try to get current observation and parse inventory
    if (cachedInventory.length === 0) {
      try {
        const observationResult = await this.getObservation(environmentId);
        if (observationResult.success) {
          return this.updateInventoryFromObservation(
            environmentId, 
            observationResult.data
          );
        }
      } catch (error) {
        console.error('❌ Error fetching inventory:', error);
      }
    }
    
    return cachedInventory;
  }

  /**
   * Add interaction to history
   * @param {number} environmentId - Environment ID
   * @param {string} action - Action
   * @param {string} observation - Observation
   */
  addToHistory(environmentId, action, observation) {
    if (!this.interactionHistory.has(environmentId)) {
      this.interactionHistory.set(environmentId, []);
    }
    
    const history = this.interactionHistory.get(environmentId);
    history.push({
      timestamp: new Date().toISOString(),
      action: action,
      observation: observation
    });
    
    // Limit history size
    if (history.length > 100) {
      history.splice(0, history.length - 100);
    }
  }

  /**
   * Get interaction history for an environment
   * @param {number} environmentId - Environment ID
   * @returns {Array} Interaction history
   */
  getHistory(environmentId) {
    return this.interactionHistory.get(parseInt(environmentId)) || [];
  }

  /**
   * Event handling methods
   */
  on(event, callback) {
    if (this.eventListeners[event]) {
      this.eventListeners[event].push(callback);
    }
  }

  off(event, callback) {
    if (this.eventListeners[event]) {
      const index = this.eventListeners[event].indexOf(callback);
      if (index > -1) {
        this.eventListeners[event].splice(index, 1);
      }
    }
  }

  emit(event, data) {
    if (this.eventListeners[event]) {
      this.eventListeners[event].forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`❌ Error in event listener for ${event}:`, error);
        }
      });
    }
  }

  /**
   * Process observation response from server
   * @param {any} response - Server response
   * @returns {string} Processed observation text
   */
  processObservation(response) {
    if (!response) {
      return 'No observation received';
    }
    
    // If it's already a string, return it
    if (typeof response === 'string') {
      return response;
    }
    
    // If it's an object with observation property
    if (typeof response === 'object' && response !== null) {
      if (response.observation) {
        if (typeof response.observation === 'string') {
          return response.observation;
        } else {
          return JSON.stringify(response.observation, null, 2);
        }
      }
      
      // Try to find other potential observation properties
      const potentialFields = ['state', 'text', 'output', 'message'];
      for (const field of potentialFields) {
        if (response[field] && typeof response[field] === 'string') {
          return response[field];
        }
      }
      
      // If we can't find a specific field, return the whole object as JSON
      return JSON.stringify(response, null, 2);
    }
    
    // Fallback for other types
    return String(response);
  }
}

// Export the class and instance
export default TextCraftClient;

// Create an instance for convenience
export const textcraftClient = new TextCraftClient();