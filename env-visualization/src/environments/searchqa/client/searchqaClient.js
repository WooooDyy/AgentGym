/**
 * SearchQA Environment Client
 * 
 * Handles all communication with SearchQA server
 * Based on agentenv-searchqa/server.py API
 */

import BaseEnvClient from '../../../shared/services/baseClient.js';

class SearchQAClient extends BaseEnvClient {
  constructor() {
    super('/api/searchqa');  // Uses vite proxy
    this.interactionHistory = new Map();
    this.environmentState = new Map();
    this.lastActionByEnv = new Map(); // Track last action per environment
    
    // Event listeners for state changes
    this.eventListeners = {
      'state-changed': [],
      'action-executed': [],
      'question-answered': [],
      'search-completed': []
    };
    
    console.log('🔍 SearchQAClient initialized');
  }
  
  async testConnection() {
    try {
      const response = await this.request('/');
      console.log('✅ SearchQA server connection test successful');
      return { success: true, message: response };
    } catch (error) {
      console.error('❌ Failed to connect to SearchQA server:', error);
      return { success: false, error: error.message };
    }
  }
  
  async len() {
    try {
      // Note: len endpoint is not defined in the provided server code
      // This might need to be implemented or we can return a default value
      console.warn('⚠️ len endpoint not available on server, returning default');
      return { success: true, data: 51713 }; // Total test items from README
    } catch (error) {
      console.error('❌ Failed to get SearchQA environment length:', error);
      return { success: false, error: error.message };
    }
  }
  
  async createEnvironment(config = {}) {
    try {
      // Extract item ID from config, default to 0 if not provided
      const itemId = config.itemId || config.id || 0;
      
      const response = await this.request('/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: itemId })
      });
      
      // Server returns environment ID directly as integer
      const environmentId = response;
      this.environmentId = environmentId;
      
      // Initialize environment state
      this.environmentState.set(environmentId, {
        rounds: 0,
        totalReward: 0,
        isCompleted: false,
        question: null,
        lastSearchResults: null,
        actionHistory: []
      });
      
      console.log(`🏗️ Created SearchQA environment with ID: ${environmentId}`);
      
      return { 
        success: true, 
        data: { id: environmentId },
        environmentId: environmentId
      };
    } catch (error) {
      console.error('❌ Failed to create SearchQA environment:', error);
      return { success: false, error: error.message };
    }
  }
  
  async observe(envId) {
    return this.getObservation(envId);
  }
  
  async getObservation(envId) {
    try {
      const response = await this.request(`/observation?env_idx=${envId}`);
      
      // Server returns observation string directly
      let processedObservation = this.processObservationData(response);
      // Update environment state
      this.updateEnvironmentState(envId, { observation: processedObservation });
      
      return { 
        success: true, 
        data: processedObservation
      };
    } catch (error) {
      console.error('❌ Failed to get SearchQA observation:', error);
      return { success: false, error: error.message };
    }
  }
  
  async step(envId, action) {
    try {
      // Parse and validate action
      const parsedAction = this.parseAction(action);
      
      // Store the last action for this environment
      this.lastActionByEnv.set(envId, parsedAction);
      
      const response = await this.request(`/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          env_idx: envId,
          action: action
        })
      });
      
      // Process step response - server returns StepResponse format
      const processedData = this.processStepResponse(response, parsedAction);
      
      // Update interaction history
      this.addToHistory(envId, parsedAction, processedData);
      
      // Update environment state
      this.updateEnvironmentState(envId, {
        rounds: this.getEnvironmentState(envId)?.rounds + (parsedAction.type === 'search' ? 1 : 0),
        totalReward: processedData.reward || 0,
        isCompleted: processedData.done || false,
        lastAction: parsedAction,
        lastResponse: processedData
      });
      
      // Emit events
      this.emit('action-executed', { envId, action: parsedAction, response: processedData });
      
      if (parsedAction.type === 'search') {
        this.emit('search-completed', { envId, query: parsedAction.content, results: processedData.searchResults });
      }
      
      if (parsedAction.type === 'answer') {
        this.emit('question-answered', { envId, answer: parsedAction.content, correct: processedData.reward > 0 });
      }
      
      console.log(`🎯 SearchQA Step executed: ${parsedAction.type} - Reward: ${processedData.reward}`);
      
      return { 
        success: true, 
        data: processedData
      };
    } catch (error) {
      console.error('❌ Failed to step SearchQA environment:', error);
      return { success: false, error: error.message };
    }
  }
  
  async reset(envId, idx = 0) {
    try {
      const response = await this.request('/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          env_idx: envId,
          id: idx
        })
      });
      
      // Clear history and state
      this.interactionHistory.set(envId, []);
      this.lastActionByEnv.delete(envId); // Clear last action
      this.environmentState.set(envId, {
        rounds: 0,
        totalReward: 0,
        isCompleted: false,
        question: null,
        lastSearchResults: null,
        actionHistory: []
      });
      
      // Server returns observation string directly
      const processedObservation = this.processObservationData(response);
      
      // Update state with new question
      this.updateEnvironmentState(envId, { observation: processedObservation });
      
      console.log(`🔄 SearchQA environment ${envId} reset to task ${idx}`);
      
      return { 
        success: true, 
        data: {
          observation: processedObservation,
          reward: 0,
          done: false,
          steps: 0
        }
      };
    } catch (error) {
      console.error('❌ Failed to reset SearchQA environment:', error);
      return { success: false, error: error.message };
    }
  }
  
  // Additional method for direct search (if needed)
  async search(envId, query) {
    try {
      // This would be equivalent to step with search action
      return await this.step(envId, `<search>${query}</search>`);
    } catch (error) {
      console.error('❌ Failed to perform SearchQA search:', error);
      return { success: false, error: error.message };
    }
  }
  
  // Parse action string to extract type and content
  parseAction(action) {
    if (typeof action !== 'string') {
      return { type: 'unknown', content: action, raw: action };
    }
    
    // Check for search action
    const searchMatch = action.match(/<search>(.*?)<\/search>/s);
    if (searchMatch) {
      return {
        type: 'search',
        content: searchMatch[1].trim(),
        raw: action
      };
    }
    
    // Check for answer action
    const answerMatch = action.match(/<answer>(.*?)<\/answer>/s);
    if (answerMatch) {
      return {
        type: 'answer',
        content: answerMatch[1].trim(),
        raw: action
      };
    }
    
    // Check for think action
    const thinkMatch = action.match(/<think>(.*?)<\/think>/s);
    if (thinkMatch) {
      return {
        type: 'think',
        content: thinkMatch[1].trim(),
        raw: action
      };
    }
    
    return {
      type: 'invalid',
      content: action,
      raw: action
    };
  }
  
  // Process observation data to extract structured information
  processObservationData(observation) {
    let content = observation;
    if (typeof observation === 'object') {
      content = JSON.stringify(observation);
    }
    content = content.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const processed = {
      raw: content,
      question: null,
      searchResults: null,
      feedback: null,
      type: 'observation'
    };
    
    // Extract question - improved regex to handle the actual server format
    // Server format: "Question: {question}" at the end of the prompt
    const questionMatch = content.match(/Question:\s*(.+?)(?:\s*$)/s);
    if (questionMatch) {
      processed.question = questionMatch[1].trim();
    }
    
    // Also try alternative question patterns
    if (!processed.question) {
      // Try to find question after the instruction text
      const altQuestionMatch = content.match(/Follow this process every time\.\s*\n\s*Question:\s*(.+?)(?:\s*$)/s);
      if (altQuestionMatch) {
        processed.question = altQuestionMatch[1].trim();
      }
    }
    
    // Extract search results - server format: \n\n<information>content</information>\n\n
    const infoMatch = content.match(/<information>(.*?)<\/information>/s);
    if (infoMatch) {
      processed.searchResults = infoMatch[1].trim();
      processed.type = 'search_result';
    }
    
    // Extract feedback messages
    if (content.includes('Congratulations! You have answered the question correctly')) {
      processed.feedback = 'correct';
      processed.type = 'answer_feedback';
    } else if (content.includes('Sorry, your answer is incorrect')) {
      processed.feedback = 'incorrect';
      processed.type = 'answer_feedback';
    } else if (content.includes('Your previous action is invalid')) {
      processed.feedback = 'invalid';
      processed.type = 'error_feedback';
    }
    
    // If this is just a question observation (no search results or feedback)
    if (!processed.searchResults && !processed.feedback && processed.question) {
      processed.type = 'question';
    }
    
    return processed;
  }
  
  // Process step response according to StepResponse model
  processStepResponse(response, action) {
    const processed = {
      observation: response.observation.replace(/</g, '&lt;').replace(/>/g, '&gt;') || '',
      reward: response.reward || 0,
      done: response.done || false,
      info: response.info || null,
      action: action,
      searchResults: null
    };
    
    // Process observation
    if (processed.observation) {
      const observationData = this.processObservationData(processed.observation);
      processed.observationData = observationData;
      
      if (observationData.searchResults) {
        processed.searchResults = observationData.searchResults;
      }
    }
    
    return processed;
  }
  
  // Environment state management
  getEnvironmentState(envId) {
    return this.environmentState.get(envId) || {
      rounds: 0,
      totalReward: 0,
      isCompleted: false,
      question: null,
      lastSearchResults: null,
      actionHistory: []
    };
  }
  
  updateEnvironmentState(envId, updates) {
    const currentState = this.getEnvironmentState(envId);
    const newState = { ...currentState, ...updates };
    this.environmentState.set(envId, newState);
    
    this.emit('state-changed', { envId, state: newState });
  }
  
  // Get the last action executed for a specific environment
  getLastAction(envId) {
    return this.lastActionByEnv.get(envId) || null;
  }
  
  // Event system
  on(event, callback) {
    if (!this.eventListeners[event]) {
      this.eventListeners[event] = [];
    }
    this.eventListeners[event].push(callback);
  }
  
  emit(event, data) {
    if (this.eventListeners[event]) {
      this.eventListeners[event].forEach(callback => callback(data));
    }
  }
  
  // History management
  addToHistory(envId, action, response) {
    if (!this.interactionHistory.has(envId)) {
      this.interactionHistory.set(envId, []);
    }
    
    const history = this.interactionHistory.get(envId);
    
    const historyItem = {
      timestamp: new Date().toISOString(),
      action: action,
      response: response,
      processedObservation: this.processObservation(response.observation || response)
    };
    
    history.push(historyItem);
    
    // Keep history to a reasonable size
    if (history.length > 100) {
      history.shift();
    }
    
    // Update environment state
    const state = this.getEnvironmentState(envId);
    state.actionHistory = history;
    this.environmentState.set(envId, state);
  }
  
  getHistory(envId) {
    return this.interactionHistory.get(envId) || [];
  }
  
  // Get statistics for an environment
  getStats(envId) {
    const history = this.getHistory(envId);
    const state = this.getEnvironmentState(envId);
    
    return {
      totalActions: history.length,
      searchActions: history.filter(h => h.action.type === 'search').length,
      answerActions: history.filter(h => h.action.type === 'answer').length,
      invalidActions: history.filter(h => h.action.type === 'invalid').length,
      totalReward: state.totalReward,
      isCompleted: state.isCompleted,
      rounds: state.rounds
    };
  }
}

export default SearchQAClient;
export const searchqaClient = new SearchQAClient();
