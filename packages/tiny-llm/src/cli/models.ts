#!/usr/bin/env node
import { formatCount } from "../config.js";
import {
  EXPORTS_DIR,
  MODELS_DIR,
  listExports,
  listSavedModels,
  repoRoot,
  resolveModelDir,
} from "../model-store.js";

const args = process.argv.slice(2);
const cmd = args[0] ?? "list";

function printEntry(e: {
  id: string;
  version: string;
  step: number;
  loss: number;
  label?: string;
  parameterCount: number;
  path: string;
}): void {
  const loss = e.loss >= 0 ? `loss ${e.loss.toFixed(3)}` : "loss —";
  const step = e.step > 0 ? `step ${e.step}` : "step —";
  const label = e.label ? ` · ${e.label}` : "";
  console.log(
    `  ${e.id}\n    ${e.version} · ${step} · ${loss} · ${formatCount(e.parameterCount)} params${label}`,
  );
  console.log(`    path: ${e.path}`);
  console.log(`    npm run start:model -- ${e.id} --mode chat`);
  console.log("");
}

if (cmd === "list" || cmd === "ls") {
  const base = repoRoot();
  const saved = listSavedModels(base);
  const legacy = listExports(base);
  const seen = new Set(saved.map((e) => e.id));
  const all = [...saved, ...legacy.filter((e) => !seen.has(e.id))];

  if (all.length === 0) {
    console.log(`No saved models in ${MODELS_DIR}/ yet.`);
    console.log("Train in the browser (Stop / Save) or:");
    console.log("  npm run train:tiny -- --preset small --steps 1000 --out models/vi-small");
    process.exit(0);
  }

  if (saved.length > 0) {
    console.log(`Saved models in ${MODELS_DIR}/ (${saved.length}):\n`);
    for (const e of saved) printEntry(e);
  }
  const onlyLegacy = legacy.filter((e) => !saved.some((s) => s.id === e.id));
  if (onlyLegacy.length > 0) {
    console.log(`Legacy exports in ${EXPORTS_DIR}/ (${onlyLegacy.length}):\n`);
    for (const e of onlyLegacy) printEntry(e);
  }
  process.exit(0);
}

if (cmd === "path") {
  const id = args[1];
  if (!id) {
    console.error("Usage: npm run models -- path <folder-name|v001>");
    process.exit(1);
  }
  console.log(resolveModelDir(id, repoRoot()));
  process.exit(0);
}

console.log(`Usage:
  npm run models:list
  npm run models -- path vi-step1000-...
`);
