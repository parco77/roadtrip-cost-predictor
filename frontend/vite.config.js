import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Dev:  vite serves :5173 and proxies /api -> the FastAPI backend on :8000.
//       That keeps CORS out of the picture and means no backend URL is ever
//       hardcoded in the app - src/api.js only ever calls relative "/api/...".
// Prod: `npm run build` writes into ../static, which app.py already mounts at "/",
//       so `uvicorn app:app` alone then serves both the API and the UI.
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
    outDir: '../static',
    emptyOutDir: true,
  },
})
