import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    // Web helpers import the workspace package. Tests must use current source,
    // not an absent or stale dist/ from a previous build.
    alias: { "@math-llm/tiny-llm": fileURLToPath(new URL("./src/index.ts", import.meta.url)) },
  },
  test: { include: ["test/**/*.test.ts"] },
});
