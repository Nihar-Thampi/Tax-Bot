import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: true,
    // In Docker the source is a bind mount from Windows or macOS, which sends no file events
    // into the container, so the dev server polls for changes instead (VITE_POLL is set there).
    watch: process.env.VITE_POLL ? { usePolling: true, interval: 300 } : undefined,
    proxy: {
      '/api': {
        // API_URL points at the backend container in Docker; locally it's the Flask dev server.
        target: process.env.API_URL ?? 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
    },
  },
})
