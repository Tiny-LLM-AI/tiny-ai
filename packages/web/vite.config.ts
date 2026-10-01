import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@math-llm/core": path.resolve(__dirname, "../core/src/index.ts"),
    },
  },
});
