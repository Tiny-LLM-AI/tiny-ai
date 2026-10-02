#!/usr/bin/env node
import { formatCount } from "../config.js";
import {
  EXPORTS_DIR,
  listExports,
  repoRoot,
  resolveExportDir,
} from "../model-store.js";

const args = process.argv.slice(2);
const cmd = args[0] ?? "list";

if (cmd === "list" || cmd === "ls") {
  const base = repoRoot();
  const exports = listExports(base);
  if (exports.length === 0) {
    console.log(`No exports in ${EXPORTS_DIR}/ yet.`);
    console.log("Train in the browser (Stop / Export) or:");
    console.log("  npm run train:tiny -- --preset small --steps 1000 --out models/mini");
    process.exit(0);
  }
  console.log(`Exports in ${EXPORTS_DIR}/ (${exports.length}):\n`);
  for (const e of exports) {
    const loss =
      e.loss >= 0 ? `loss ${e.loss.toFixed(3)}` : "loss —";
    const step = e.step > 0 ? `step ${e.step}` : "step —";
    const label = e.label ? ` · ${e.label}` : "";
    console.log(
      `  ${e.id}\n    ${e.version} · ${step} · ${loss} · ${formatCount(e.parameterCount)} params${label}`,
    );
    console.log(`    npm run start:model -- ${e.version} --mode chat`);
    console.log("");
  }
  process.exit(0);
}

if (cmd === "path") {
  const id = args[1];
  if (!id) {
    console.error("Usage: npm run models -- path <v001|full-export-id>");
    process.exit(1);
  }
  console.log(resolveExportDir(id, repoRoot()));
  process.exit(0);
}

console.log(`Usage:
  npm run models:list
  npm run models -- path v001
`);
