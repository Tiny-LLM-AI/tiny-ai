#!/usr/bin/env node
import { spawn } from "node:child_process";
import path from "node:path";
import { formatCount } from "../config.js";
import { countModelParams } from "../gpt.js";
import { readExportMeta, repoRoot, resolveExportDir } from "../model-store.js";
import { loadModelDir, parseArgs } from "./files.js";
import { initNodeBackend } from "./node-backend.js";

const HELP = `Usage: npm run start:model -- <export-id> [options]

  <export-id>     v001, v002, or full folder name under models/exports/
  --mode chat     Interactive chat (default)
  --mode train    Resume training (--steps required)
  --mode web      Open Tiny GPT UI with this model loaded
  --steps N       Training steps for --mode train (default 2000)
  --temperature T Chat temperature (default 0.2)

Examples:
  npm run models:list
  npm run start:model -- v001 --mode chat
  npm run start:model -- v002 --mode train --steps 5000
  npm run start:model -- v003-step120-2026-10-02-14-30-00 --mode web
`;

const args = parseArgs(process.argv.slice(2));
const positional = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const exportId = positional[0] ?? args.model ?? args.export;

if (args.help || !exportId) {
  console.log(HELP);
  process.exit(args.help ? 0 : 1);
}

const baseDir = repoRoot();
const modelDir = resolveExportDir(exportId, baseDir);
const mode = args.mode ?? "chat";
const meta = readExportMeta(modelDir);

console.log(`Model: ${path.relative(baseDir, modelDir)}`);
if (meta) {
  console.log(
    `  ${meta.version} · step ${meta.step} · loss ${meta.loss >= 0 ? meta.loss.toFixed(3) : "—"} · exported ${meta.exportedAt}`,
  );
}

if (mode === "web") {
  const exportKey = path.basename(modelDir);
  const child = spawn("npm", ["run", "dev:tiny"], {
    cwd: baseDir,
    stdio: "inherit",
    shell: process.platform === "win32",
    env: {
      ...process.env,
      TINY_GPT_MODEL_ID: exportKey,
    },
  });
  child.on("exit", (code) => process.exit(code ?? 0));
} else if (mode === "train") {
  const steps = args.steps ?? "2000";
  const child = spawn(
    "npm",
    ["run", "train:tiny", "--", "--resume", modelDir, "--steps", steps],
    { cwd: baseDir, stdio: "inherit", shell: process.platform === "win32" },
  );
  child.on("exit", (code) => process.exit(code ?? 0));
} else if (mode === "chat") {
  await initNodeBackend();
  const model = loadModelDir(modelDir);
  console.log(
    `Loaded ${formatCount(countModelParams(model))} params. Chat — empty line to quit.\n`,
  );
  const { default: readline } = await import("node:readline/promises");
  const { generate } = await import("../generate.js");
  const temperature = Number(args.temperature ?? 0.2);
  const maxNewTokens = Number(args["max-tokens"] ?? 80);
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: "> ",
  });
  rl.prompt();
  for await (const prompt of rl) {
    if (!prompt.trim()) break;
    const { text } = generate(model, prompt, {
      temperature,
      topK: 20,
      maxNewTokens,
      seed: Date.now(),
    });
    console.log(`${prompt}${text}`);
    rl.prompt();
  }
  rl.close();
} else {
  console.error(`Unknown mode: ${mode}. Use chat, train, or web.`);
  process.exit(1);
}
