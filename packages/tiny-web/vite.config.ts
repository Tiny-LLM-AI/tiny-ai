import path from "node:path";
import { defineConfig } from "vite";
import modelsApiPlugin from "./vite.models-plugin.js";

export default defineConfig({
  root: ".",
  server: { port: 5200, fs: { allow: ["../.."] } },
  define: {
    "import.meta.env.TINY_GPT_MODEL_ID": JSON.stringify(
      process.env.TINY_GPT_MODEL_ID ?? "",
    ),
  },
  plugins: [modelsApiPlugin()],
  resolve: {
    alias: {
      "@math-llm/tiny-llm": path.resolve(__dirname, "../tiny-llm/src/index.ts"),
      "@math-llm/tiny-llm/train-worker": path.resolve(__dirname, "../tiny-llm/src/train-worker.ts"),
    },
  },
  optimizeDeps: {
    include: [
      "@tensorflow/tfjs",
      "@tensorflow/tfjs-backend-webgl",
      "@tensorflow/tfjs-backend-wasm",
    ],
  },
  worker: {
    format: "es",
  },
  assetsInclude: ["**/*.wasm"],
});
