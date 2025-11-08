import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API_PROXY_TARGET = process.env.VITE_API_BASE_URL || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/cases": API_PROXY_TARGET,
      "/predict": API_PROXY_TARGET,
      "/classes": API_PROXY_TARGET,
      "/health": API_PROXY_TARGET
    }
  }
});
