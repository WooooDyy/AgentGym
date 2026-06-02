import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

// Environment-aware proxy targets
// Docker: uses host.docker.internal to reach host machine ports
// Local: uses localhost directly
const isDocker = process.env.DOCKER_ENV === 'true';
const dockerHost = process.env.ENV_HOST || 'host.docker.internal';
const getTarget = (service, port) => isDocker ? `http://${dockerHost}:${port}` : `http://localhost:${port}`;

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    vueDevTools(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api/textcraft': {
        target: getTarget('textcraft', 36001),
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/textcraft/, '')
      },
      '/api/babyai': {
        target: getTarget('babyai', 36002),
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/babyai/, '')
      },
      '/api/sciworld': {
        target: getTarget('sciworld', 36003),
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/sciworld/, '')
      },
      '/api/webarena': {
        target: getTarget('webarena', 36004),
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/webarena/, '')
      },
      '/api/searchqa': {
        target: getTarget('searchqa', 36005),
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/searchqa/, '')
      }
    }
  }
})
