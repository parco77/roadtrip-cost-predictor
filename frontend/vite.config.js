import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Dev:  vite serves :5173 and proxies /api -> the FastAPI backend on :8000, so a local
//       run needs no backend URL and no CORS at all - src/api.js calls relative "/api/...".
// Prod: `npm run build` writes dist/, which the static host uploads. The backend lives on
//       another origin there, so src/api.js prefixes VITE_API_URL - see the note in it.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
