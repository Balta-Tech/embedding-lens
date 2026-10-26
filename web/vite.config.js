import { defineConfig } from "vite";

export default defineConfig({
  build: { outDir: "dist", emptyOutDir: true },
  // In development the live features talk to the local Python backend through /api
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
