import { defineConfig } from 'vite';

// Static SPA. The fetch script writes artifacts into public/runs/<run_id>/,
// which Vite serves verbatim at /runs/<run_id>/. No server-side code, no
// credentials in the bundle.
export default defineConfig({
  server: {
    port: 5173,
    open: true,
    // Forward rollout requests to the FastAPI inference service.
    proxy: { '/api': 'http://localhost:8000' },
  },
  build: { target: 'es2020', outDir: 'dist' },
});
