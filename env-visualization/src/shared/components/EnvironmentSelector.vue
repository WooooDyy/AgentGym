<template>
  <div class="environment-selector">
    <!-- 动态背景粒子 -->
    <div class="background-particles">
      <div 
        v-for="n in 15" 
        :key="n" 
        class="particle"
        :style="getParticleStyle(n)"
      ></div>
    </div>

    <div class="selector-header">
      <h1 class="welcome-title">Welcome to AgentGym Hub</h1>
    </div>
    
    <div class="environment-grid">
      <div
        v-for="(env, index) in environments"
        :key="env.id"
        :class="[
          'environment-card',
          { 'active': currentEnvironment?.id === env.id },
          { 'hover-effect': !currentEnvironment || currentEnvironment.id !== env.id }
        ]"
        :style="{ '--delay': index * 0.1 + 's' }"
        @click="selectEnvironment(env)"
        @mouseenter="onCardHover(env)"
        @mouseleave="onCardLeave(env)"
      >
        <!-- 环境图标和状态 -->
        <div class="env-icon-container">
          <div class="env-icon" :style="{ '--icon-color': env.color }">
            {{ env.icon }}
          </div>
          <div class="icon-glow" :style="{ '--glow-color': env.color }"></div>
          
          <!-- 服务器可用性状态指示器 -->
          <div 
            class="availability-indicator"
            :class="{
              'available': isEnvAvailable(env.id),
              'unavailable': availabilityChecked && !isEnvAvailable(env.id),
              'checking': !availabilityChecked
            }"
            :title="getAvailabilityStatus(env.id)"
          >
            <span class="status-dot"></span>
            <span class="status-text">{{ getAvailabilityStatus(env.id) }}</span>
          </div>
          
          <!-- 添加难度指示器 -->
          <div class="difficulty-indicator">
            <div class="difficulty-stars">
              <span 
                v-for="star in 5" 
                :key="star"
                class="star"
                :class="{ 'filled': star <= getDifficulty(env.id) }"
              >
                ⭐
              </span>
            </div>
            <span class="difficulty-label">{{ getDifficultyLabel(env.id) }}</span>
          </div>
        </div>
        
        <!-- 环境信息 -->
        <div class="env-info">
          <h4 class="env-name">
            {{ env.name }}
            <span class="new-badge" v-if="isNewEnvironment(env.id)">NEW!</span>
          </h4>
          <p class="env-description">{{ env.description }}</p>
          
          <!-- 环境特性标签 -->
          <div class="env-tags">
            <span 
              v-for="tag in env.tags" 
              :key="tag" 
              class="env-tag"
              :style="{ '--tag-color': env.color }"
            >
              {{ tag }}
            </span>
          </div>

          <!-- 添加环境详细信息 -->
          <div class="env-details">
            <div class="detail-item">
              <span class="detail-icon">🎯</span>
              <span class="detail-text">{{ getObjective(env.id) }}</span>
            </div>
            <div class="detail-item">
              <span class="detail-icon">⏱️</span>
              <span class="detail-text">{{ getEstimatedTime(env.id) }}</span>
            </div>
          </div>
        </div>
        
        <!-- 选择按钮和预览 -->
        <div class="env-actions">
          <button 
            class="btn btn-select"
            :style="{ '--btn-color': env.color }"
            @click.stop="selectEnvironment(env)"
          >
            <span class="btn-icon">
              {{ currentEnvironment?.id === env.id ? '✅' : '🎯' }}
            </span>
            <span class="btn-text">
              {{ currentEnvironment?.id === env.id ? 'Selected' : 'Select' }}
            </span>
            <div class="btn-sparkle"></div>
          </button>
        </div>
        
        <!-- 活跃状态指示器 -->
        <div v-if="currentEnvironment?.id === env.id" class="active-indicator">
          <div class="pulse-ring"></div>
          <div class="pulse-dot"></div>
        </div>

        <!-- 悬停时显示的额外信息 -->
        <div class="env-overlay" :class="{ 'visible': hoveredEnv === env.id }">
          <div class="overlay-content">
            <h5>✨ Quick Facts</h5>
            <ul class="fact-list">
              <li v-for="fact in getEnvironmentFacts(env.id)" :key="fact">{{ fact }}</li>
            </ul>
          </div>
        </div>

        <!-- 卡片装饰元素 -->
        <div class="card-decoration">
          <div class="deco-circle" :style="{ '--deco-color': env.color }"></div>
          <div class="deco-triangle" :style="{ '--deco-color': env.color }"></div>
        </div>
      </div>
    </div>
    
    <!-- 预览模态框 -->
    <div v-if="previewVisible" class="preview-modal" @click="closePreview">
      <div class="preview-content" @click.stop>
        <div class="preview-header">
          <h3>{{ previewEnv?.name }} Preview</h3>
          <button class="close-btn" @click="closePreview">✕</button>
        </div>
        <div class="preview-body">
          <div class="preview-image">
            <div class="mock-preview" :style="{ '--env-color': previewEnv?.color }">
              <span class="preview-icon">{{ previewEnv?.icon }}</span>
              <div class="preview-text">{{ previewEnv?.name }} Environment</div>
            </div>
          </div>
          <div class="preview-info">
            <p><strong>Description:</strong> {{ previewEnv?.description }}</p>
            <p><strong>Difficulty:</strong> {{ getDifficultyLabel(previewEnv?.id) }}</p>
            <p><strong>Objective:</strong> {{ getObjective(previewEnv?.id) }}</p>
            <p><strong>Features:</strong> {{ previewEnv?.tags.join(', ') }}</p>
          </div>
        </div>
        <div class="preview-actions">
          <button class="btn btn-primary" @click="selectEnvironmentFromPreview">
            🎯 Select This Environment
          </button>
        </div>
      </div>
    </div>

    <!-- Footer Section -->
    <div class="selector-footer">
      <p>AgentGym is a visualization platform for multiple AI environments</p>
      <router-link to="/about" class="about-link">Learn more</router-link>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, onUnmounted } from 'vue'
import { environmentList, checkAllEnvironmentsAvailability } from '../../environments/index.js'

export default {
  name: 'EnvironmentSelector',
  emits: ['environment-selected'],
  setup(props, { emit }) {
    const environments = ref([])
    const currentEnvironment = ref(null)
    const hoveredEnv = ref(null)
    const previewVisible = ref(false)
    const previewEnv = ref(null)
    const helpVisible = ref(false)
    const currentFactIndex = ref(0)
    
    // Environment availability state
    const envAvailability = ref({})
    const availabilityChecked = ref(false)

    // 有趣的事实轮播
    const funFacts = [
      "Each environment offers unique challenges and experiences! 🌟",
      "AI agents learn differently in each environment! 🤖",
      "You can switch environments anytime! 🔄",
      "Some environments are easier than others! 📈",
      "Explore all environments to become an expert! 🏆"
    ]

    // 环境配置数据
    const environmentConfig = {
      textcraft: {
        difficulty: 2,
        objective: "Craft and build amazing structures",
        facts: [
          "🏗️ Build anything you can imagine",
          "🌳 Gather resources from the world",
          "⚒️ Craft tools and materials",
          "🎯 Complete building challenges"
        ],
        isNew: false
      },
      babyai: {
        difficulty: 2,
        objective: "Navigate and follow instructions",
        facts: [
          "🧸 Perfect for beginners",
          "🗺️ Simple grid-based navigation",
          "📋 Clear instruction following",
          "🎮 Easy to understand mechanics"
        ],
        isNew: false
      },
      sciworld: {
        difficulty: 5,
        objective: "Conduct scientific experiments",
        facts: [
          "🧪 Real scientific simulations",
          "🔬 Use laboratory equipment",
          "📊 Collect and analyze data",
          "🏆 Discover scientific principles"
        ],
        isNew: false
      },
      webarena: {
        difficulty: 5,
        objective: "Navigate websites and complete tasks",
        facts: [
          "🌐 Real web browsing challenges",
          "🖱️ Click, scroll, and interact",
          "📱 Modern web interface",
          "🎯 Complete realistic web tasks"
        ],
        isNew: false
      },
      searchqa: {
        difficulty: 3,
        objective: "Research and answer questions",
        facts: [
          "🔍 Search for information",
          "📚 Analyze multiple sources",
          "💡 Provide accurate answers",
          "🧠 Develop research skills"
        ],
        isNew: false
      }
    }

    const getEnvironmentIcon = (envId) => {
      const icons = {
        textcraft: '⚒️',
        babyai: '🧸',
        sciworld: '🧪',
        webarena: '🌐',
        searchqa: '🔍'
      }
      return icons[envId] || '📦'
    }

    const getEnvironmentColor = (envId) => {
      const colors = {
        textcraft: '#27ae60',
        babyai: '#3498db',
        sciworld: '#9b59b6', 
        webarena: '#e67e22',
        searchqa: '#34495e'
      }
      return colors[envId] || '#3498db'
    }

    const getEnvironmentTags = (envId) => {
      const tags = {
        textcraft: ['Crafting', 'Strategy', 'Sandbox'],
        babyai: ['Learning', 'Navigation', 'Simple'],
        sciworld: ['Science', 'Simulation', 'Physics'],
        webarena: ['Web', 'Browsing', 'Interactive'],
        searchqa: ['Search', 'Q&A', 'Knowledge']
      }
      return tags[envId] || ['General']
    }

    // Check if environment is available (server running)
    const isEnvAvailable = (envId) => {
      return envAvailability.value[envId]?.available || false
    }

    // Get availability status text
    const getAvailabilityStatus = (envId) => {
      if (!availabilityChecked.value) return 'Checking...'
      const status = envAvailability.value[envId]
      if (!status) return 'Unknown'
      return status.available ? 'Online' : 'Offline'
    }

    // Check all environment availability
    const checkAvailability = async () => {
      try {
        console.log('🔍 Checking environment availability...')
        const results = await checkAllEnvironmentsAvailability()
        envAvailability.value = results
        availabilityChecked.value = true
        console.log('✅ Availability check complete:', results)
      } catch (error) {
        console.error('❌ Failed to check availability:', error)
        availabilityChecked.value = true
      }
    }

    const getDifficulty = (envId) => {
      return environmentConfig[envId]?.difficulty || 3
    }

    const getDifficultyLabel = (envId) => {
      const difficulty = getDifficulty(envId)
      const labels = ['Easy', 'Easy', 'Moderate', 'Moderate', 'Hard', 'Expert']
      return labels[difficulty] || 'Medium'
    }

    const getObjective = (envId) => {
      return environmentConfig[envId]?.objective || 'Explore and learn'
    }

    const getEstimatedTime = (envId) => {
      return environmentConfig[envId]?.estimatedTime || '10-20 min'
    }

    const isNewEnvironment = (envId) => {
      return environmentConfig[envId]?.isNew || false
    }

    const getEnvironmentFacts = (envId) => {
      return environmentConfig[envId]?.facts || ['Fun environment to explore!']
    }

    const getParticleStyle = (index) => {
      return {
        left: Math.random() * 100 + '%',
        top: Math.random() * 100 + '%',
        animationDelay: Math.random() * 3 + 's',
        animationDuration: (3 + Math.random() * 4) + 's'
      }
    }

    const loadEnvironments = () => {
      environments.value = environmentList.map(env => ({
        ...env,
        icon: getEnvironmentIcon(env.id),
        color: getEnvironmentColor(env.id),
        tags: getEnvironmentTags(env.id)
      }))
    }

    const selectEnvironment = (env) => {
      try {
        currentEnvironment.value = env
        emit('environment-selected', env)
        
        // 添加选择反馈效果
        const card = event.currentTarget
        card.style.transform = 'scale(0.95)'
        setTimeout(() => {
          card.style.transform = ''
        }, 150)

        // 关闭预览如果打开
        previewVisible.value = false
      } catch (error) {
        console.error('Failed to select environment:', error)
      }
    }

    const onCardHover = (env) => {
      hoveredEnv.value = env.id
    }

    const onCardLeave = (env) => {
      hoveredEnv.value = null
    }

    const previewEnvironment = (env) => {
      previewEnv.value = env
      previewVisible.value = true
    }

    const closePreview = () => {
      previewVisible.value = false
      previewEnv.value = null
    }

    const selectEnvironmentFromPreview = () => {
      if (previewEnv.value) {
        selectEnvironment(previewEnv.value)
      }
    }

    const showHelp = () => {
      helpVisible.value = !helpVisible.value
    }

    const selectRandomEnvironment = () => {
      const randomIndex = Math.floor(Math.random() * environments.value.length)
      const randomEnv = environments.value[randomIndex]
      selectEnvironment(randomEnv)
    }

    // 定期更换有趣事实
    let factInterval
    let availabilityInterval
    onMounted(() => {
      loadEnvironments()
      
      // Check availability immediately and periodically
      checkAvailability()
      availabilityInterval = setInterval(checkAvailability, 30000) // Check every 30s
      
      factInterval = setInterval(() => {
        currentFactIndex.value = (currentFactIndex.value + 1) % funFacts.length
      }, 4000)
    })

    onUnmounted(() => {
      if (factInterval) {
        clearInterval(factInterval)
      }
      if (availabilityInterval) {
        clearInterval(availabilityInterval)
      }
    })

    return {
      environments,
      currentEnvironment,
      hoveredEnv,
      previewVisible,
      previewEnv,
      helpVisible,
      currentFactIndex,
      funFacts,
      envAvailability,
      availabilityChecked,
      selectEnvironment,
      onCardHover,
      onCardLeave,
      previewEnvironment,
      closePreview,
      selectEnvironmentFromPreview,
      showHelp,
      selectRandomEnvironment,
      getDifficulty,
      getDifficultyLabel,
      getObjective,
      getEstimatedTime,
      isNewEnvironment,
      getEnvironmentFacts,
      getParticleStyle,
      isEnvAvailable,
      getAvailabilityStatus,
      checkAvailability
    }
  }
}
</script>

<style scoped>
.environment-selector {
  padding: 1rem 1rem 0.5rem 1rem;
  margin: 0;
  background: linear-gradient(135deg, #fefefe 0%, #f8f9fa 100%);
  min-height: 100vh;
  width: 100%;
  height: 100vh;
  position: relative;
  overflow: hidden;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
}

/* 动态背景粒子 */
.background-particles {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 0;
}

.particle {
  position: absolute;
  width: 4px;
  height: 4px;
  background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
  border-radius: 50%;
  animation: float infinite ease-in-out;
  opacity: 0.7;
}

@keyframes float {
  0%, 100% { 
    transform: translateY(0px) rotate(0deg);
    opacity: 0.6;
  }
  50% { 
    transform: translateY(-20px) rotate(180deg);
    opacity: 1;
  }
}

.environment-selector::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: 
    radial-gradient(circle at 20% 20%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
    radial-gradient(circle at 40% 70%, rgba(255, 255, 255, 0.05) 0%, transparent 50%);
  pointer-events: none;
  z-index: 1;
}

.selector-header {
  text-align: center;
  margin-bottom: 1.5rem;
  position: relative;
  z-index: 2;
}

.welcome-title {
  color: #2c3e50;
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
  font-weight: 800;
  text-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.environment-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1.5rem;
  margin-bottom: 1rem;
  position: relative;
  z-index: 2;
  flex: 1;
  overflow-y: auto;
  padding: 0 1rem 1rem 1rem;
  justify-content: center;
  max-width: 1600px;
  margin-left: auto;
  margin-right: auto;
  height: calc(100vh - 180px);
}

.environment-grid::-webkit-scrollbar {
  width: 10px;
}

.environment-grid::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 10px;
}

.environment-grid::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.3);
  border-radius: 10px;
}

.environment-grid::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.5);
}

.environment-card {
  background: #ffffff;
  border-radius: 20px;
  padding: 1.8rem;
  cursor: pointer;
  transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  position: relative;
  overflow: hidden;
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(0, 0, 0, 0.05);
  animation: cardSlideIn 0.6s ease-out var(--delay, 0s);
  transform: translateY(20px);
  opacity: 0;
  animation-fill-mode: forwards;
  min-height: 380px;
  display: flex;
  flex-direction: column;
}

@keyframes cardSlideIn {
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

.environment-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, var(--icon-color, #3498db), transparent);
  transform: scaleX(0);
  transition: transform 0.3s;
}

.environment-card.hover-effect:hover {
  transform: translateY(-8px) scale(1.02);
  box-shadow: 0 15px 40px rgba(0, 0, 0, 0.12);
}

.environment-card.hover-effect:hover::before {
  transform: scaleX(1);
}

.environment-card.active {
  transform: translateY(-5px);
  box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
  border-color: var(--icon-color);
  background: #ffffff;
}

.environment-card.active::before {
  transform: scaleX(1);
}

.env-icon-container {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 1.5rem;
}

.env-icon {
  font-size: 4rem;
  text-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  position: relative;
  z-index: 2;
  animation: iconFloat 3s ease-in-out infinite;
}

.icon-glow {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 80px;
  height: 80px;
  background: radial-gradient(circle, var(--glow-color, #3498db) 0%, transparent 70%);
  border-radius: 50%;
  opacity: 0.3;
  transform: translate(-50%, -50%);
  animation: glowPulse 2s ease-in-out infinite;
}

/* Availability indicator styles */
.availability-indicator {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  margin-top: 0.5rem;
  padding: 0.25rem 0.6rem;
  border-radius: 12px;
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  transition: all 0.3s ease;
}

.availability-indicator.available {
  background: rgba(39, 174, 96, 0.15);
  color: #27ae60;
  border: 1px solid rgba(39, 174, 96, 0.3);
}

.availability-indicator.unavailable {
  background: rgba(231, 76, 60, 0.15);
  color: #e74c3c;
  border: 1px solid rgba(231, 76, 60, 0.3);
}

.availability-indicator.checking {
  background: rgba(241, 196, 15, 0.15);
  color: #f39c12;
  border: 1px solid rgba(241, 196, 15, 0.3);
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  display: inline-block;
}

.availability-indicator.available .status-dot {
  background: #27ae60;
  box-shadow: 0 0 6px rgba(39, 174, 96, 0.6);
  animation: statusPulse 2s ease-in-out infinite;
}

.availability-indicator.unavailable .status-dot {
  background: #e74c3c;
}

.availability-indicator.checking .status-dot {
  background: #f39c12;
  animation: statusBlink 1s ease-in-out infinite;
}

.status-text {
  line-height: 1;
}

@keyframes statusPulse {
  0%, 100% { opacity: 1; box-shadow: 0 0 6px rgba(39, 174, 96, 0.6); }
  50% { opacity: 0.7; box-shadow: 0 0 12px rgba(39, 174, 96, 0.8); }
}

@keyframes statusBlink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.difficulty-indicator {
  margin-top: 1rem;
  text-align: center;
  background: rgba(0, 0, 0, 0.05);
  padding: 0.6rem 0.8rem;
  border-radius: 15px;
  width: fit-content;
  margin-left: auto;
  margin-right: auto;
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.difficulty-stars {
  margin-bottom: 0.4rem;
}

.star {
  font-size: 0.9rem;
  opacity: 0.3;
  transition: opacity 0.3s;
}

.star.filled {
  opacity: 1;
}

.difficulty-label {
  font-size: 0.8rem;
  color: #555;
  font-weight: 600;
}

@keyframes iconFloat {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
}

@keyframes glowPulse {
  0%, 100% { opacity: 0.3; transform: translate(-50%, -50%) scale(1); }
  50% { opacity: 0.6; transform: translate(-50%, -50%) scale(1.1); }
}

.env-info {
  text-align: center;
  margin-bottom: 1.5rem;
  flex-grow: 1;
  display: flex;
  flex-direction: column;
}

.env-name {
  color: #2c3e50;
  font-size: 1.5rem;
  margin-bottom: 0.75rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
}

.new-badge {
  background: linear-gradient(45deg, #ff6b6b, #ff8e53);
  color: white;
  font-size: 0.6rem;
  padding: 0.2rem 0.5rem;
  border-radius: 10px;
  font-weight: 700;
  animation: bounce 1s infinite;
}

@keyframes bounce {
  0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
  40% { transform: translateY(-5px); }
  60% { transform: translateY(-3px); }
}

.env-description {
  color: #4a5568;
  margin-bottom: 1.2rem;
  line-height: 1.6;
  font-size: 1rem;
  flex-grow: 1;
}

.env-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  justify-content: center;
  margin-bottom: 1.2rem;
}

.env-tag {
  background: linear-gradient(135deg, var(--tag-color), rgba(0, 0, 0, 0.1));
  color: white;
  padding: 0.3rem 0.8rem;
  border-radius: 15px;
  font-size: 0.85rem;
  font-weight: 600;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  transition: transform 0.2s;
}

.env-tag:hover {
  transform: scale(1.05);
}

.env-details {
  margin-top: 1.2rem;
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.detail-item {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  font-size: 0.95rem;
  color: #444;
  background: rgba(0, 0, 0, 0.04);
  padding: 0.6rem;
  border-radius: 12px;
}

.detail-icon {
  font-size: 1.1rem;
}

.env-actions {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  text-align: center;
  margin-top: 1.5rem;
}

.btn {
  border: none;
  border-radius: 25px;
  cursor: pointer;
  transition: all 0.3s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  font-weight: 600;
  position: relative;
  overflow: hidden;
}

.btn-select {
  background: linear-gradient(135deg, var(--btn-color, #3498db), rgba(0, 0, 0, 0.1));
  color: white;
  padding: 0.8rem 1.8rem;
  font-size: 1rem;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.btn::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
  transition: left 0.5s;
}

.btn:hover::before {
  left: 100%;
}

.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
}

.active-indicator {
  position: absolute;
  top: 15px;
  right: 15px;
}

.pulse-ring {
  width: 20px;
  height: 20px;
  border: 2px solid var(--icon-color, #4CAF50);
  border-radius: 50%;
  position: absolute;
  animation: pulse-ring 1.5s ease-out infinite;
}

.pulse-dot {
  width: 8px;
  height: 8px;
  background: var(--icon-color, #4CAF50);
  border-radius: 50%;
  position: absolute;
  top: 6px;
  left: 6px;
}

@keyframes pulse-ring {
  0% {
    transform: scale(.33);
    opacity: 1;
  }
  80%, 100% {
    transform: scale(2);
    opacity: 0;
  }
}

.env-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.9);
  border-radius: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  visibility: hidden;
  transition: all 0.3s;
}

.env-overlay.visible {
  opacity: 1;
  visibility: visible;
}

.overlay-content {
  color: white;
  text-align: center;
  padding: 1rem;
}

.overlay-content h5 {
  margin-bottom: 1rem;
  color: #FFD700;
}

.fact-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.fact-list li {
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
}

.card-decoration {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
  overflow: hidden;
  border-radius: 24px;
}

.deco-circle {
  position: absolute;
  top: -20px;
  right: -20px;
  width: 80px;
  height: 80px;
  background: radial-gradient(circle, var(--deco-color, #3498db) 0%, transparent 70%);
  border-radius: 50%;
  opacity: 0.1;
}

.deco-triangle {
  position: absolute;
  bottom: -15px;
  left: -15px;
  width: 0;
  height: 0;
  border-left: 40px solid transparent;
  border-right: 40px solid transparent;
  border-bottom: 40px solid var(--deco-color, #3498db);
  opacity: 0.1;
  transform: rotate(45deg);
}

/* 预览模态框 */
.preview-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  animation: fadeIn 0.3s;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.preview-content {
  background: white;
  border-radius: 20px;
  padding: 2rem;
  max-width: 600px;
  width: 90%;
  max-height: 80vh;
  overflow-y: auto;
  animation: slideUp 0.3s ease-out;
}

@keyframes slideUp {
  from { transform: translateY(50px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

.preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
  border-bottom: 2px solid #eee;
  padding-bottom: 1rem;
}

.close-btn {
  background: #ff5722;
  color: white;
  border: none;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  cursor: pointer;
  font-size: 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.preview-body {
  margin-bottom: 1.5rem;
}

.mock-preview {
  background: linear-gradient(135deg, var(--env-color, #667eea), rgba(0, 0, 0, 0.1));
  height: 200px;
  border-radius: 15px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: white;
  margin-bottom: 1rem;
}

.preview-icon {
  font-size: 3rem;
  margin-bottom: 0.5rem;
}

.preview-text {
  font-size: 1.2rem;
  font-weight: 600;
}

.preview-info p {
  margin-bottom: 0.5rem;
}

.preview-actions {
  text-align: center;
}

.btn-primary {
  background: linear-gradient(135deg, #3498db, #2980b9);
  color: white;
  padding: 0.8rem 2rem;
  font-size: 1rem;
}

.selector-footer {
  margin-top: auto;
  text-align: center;
  padding: 0.5rem;
  color: rgba(52, 73, 94, 0.7);
  z-index: 2;
  position: relative;
}

.about-link {
  display: inline-block;
  margin-top: 0.5rem;
  color: #3498db;
  text-decoration: underline;
  transition: all 0.3s;
}

.about-link:hover {
  color: rgba(52, 152, 219, 0.8);
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .environment-grid {
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  }
}

@media (max-width: 768px) {
  .environment-selector {
    padding: 1rem;
    height: auto;
    min-height: 100vh;
    overflow-y: auto;
  }
  
  .environment-grid {
    grid-template-columns: minmax(280px, 1fr);
    gap: 1.5rem;
    height: auto;
    overflow-y: visible;
    padding-right: 0;
  }
  
  .environment-card {
    padding: 1.5rem;
    max-width: 500px;
    margin-left: auto;
    margin-right: auto;
    min-height: 350px;
  }
  
  .welcome-title {
    font-size: 1.8rem;
  }
  
  .env-icon {
    font-size: 3rem;
  }
}

@media (max-width: 480px) {
  .environment-selector {
    padding: 0.75rem;
  }
  
  .environment-card {
    padding: 1rem;
  }
  
  .env-icon {
    font-size: 2.5rem;
  }
  
  .env-name {
    font-size: 1.2rem;
  }

  .welcome-title {
    font-size: 1.6rem;
  }
  
  .selector-footer {
    padding: 1rem;
    font-size: 0.9rem;
  }
}
</style>