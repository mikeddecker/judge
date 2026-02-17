import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  define: {
    'process.env.PASSWORD_MIN_LENGTH': JSON.stringify(process.env.PASSWORD_MIN_LENGTH || '8'),
  },
  server: {
    host: "0.0.0.0",
    port: Number(process.env.WEB_DOCKER_PORT) || 5173,
    proxy: {
      "/api": {
        target: `http://api:${Number(process.env.API_DOCKER_PORT)}`,
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api/, ""),
      },
    },
  }
})

