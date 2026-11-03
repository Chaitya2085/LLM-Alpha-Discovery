import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the page on :5173 and passes /api to the Python backend on :8000.
export default defineConfig({
  plugins: [react()],
  build: { chunkSizeWarningLimit: 800 },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
