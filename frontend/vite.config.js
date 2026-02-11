import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '0.0.0.0',
    port: 5000,
    allowedHosts: true,
    proxy: {
      '/generate': 'http://localhost:8000',
      '/audio': 'http://localhost:8000',
      '/meta': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
