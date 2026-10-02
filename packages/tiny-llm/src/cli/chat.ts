import path from "node:path";
import readline from "node:readline/promises";
import { formatCount } from "../config.js";
import { generate } from "../generate.js";
import { countModelParams } from "../gpt.js";
import { loadModelDir, parseArgs } from "./files.js";
import { initNodeBackend } from "./node-backend.js";

const args = parseArgs(process.argv.slice(2));
const baseDir = process.env.INIT_CWD ?? process.cwd();
const dir = path.resolve(baseDir, args.model ?? "models/mini");
const temperature = Number(args.temperature ?? 0);
const maxNewTokens = Number(args["max-tokens"] ?? 80);

await initNodeBackend();
const model = loadModelDir(dir);
console.log(`Loaded ${dir} (${formatCount(countModelParams(model))} params). Type a prompt, empty line to quit.`);

const rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: "> " });
rl.prompt();
for await (const prompt of rl) {
  if (!prompt.trim()) break;
  const { text } = generate(model, prompt, { temperature, topK: 20, maxNewTokens, seed: Date.now() });
  console.log(`${prompt}${text}`);
  rl.prompt();
}
rl.close();
