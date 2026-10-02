import os from "node:os";
import path from "node:path";
import {
  DEFAULT_CONFIG,
  formatBytes,
  formatCount,
  memoryEstimate,
  normalizeConfig,
  paramBreakdown,
  presetConfig,
  shapeForTargetParams,
  type GptConfig,
  type PresetId,
} from "../config.js";
import { encodeLines, generateArithmeticLines, splitCorpus } from "../corpus.js";
import { generate } from "../generate.js";
import { countModelParams, createModel, disposeModel } from "../gpt.js";
import { buildTokenizer } from "../tokenizer.js";
import { Trainer, lineAccuracy } from "../trainer.js";
import { loadModelDir, parseArgs, readCorpusFile, saveModelDir } from "./files.js";
import { initNodeBackend } from "./node-backend.js";
import { saveModelVersioned } from "../model-store.js";

const HELP = `Usage: npm run train:tiny -- [options]

  --preset mini|small|medium|large|xl   Model size preset (default mini)
  --target-params 50000000               Auto-fill layers/width for this many params
  --layers N --d-model N --heads N --ffn N --context N
  --lr 0.001 --batch 32
  --corpus path.txt|path.json            Training text (one line per document)
  --max-number 9                         Without --corpus: generate arithmetic 0..N
  --steps 2000                           Optimizer steps
  --out models/mini                      Output folder (model.json + weights.bin)
  --resume models/mini                   Continue training a saved model
  --force                                Train even if memory estimate exceeds RAM
`;

const args = parseArgs(process.argv.slice(2));
if (args.help) {
  console.log(HELP);
  process.exit(0);
}

const baseDir = process.env.INIT_CWD ?? process.cwd();
const resolvePath = (p: string) => (path.isAbsolute(p) ? p : path.resolve(baseDir, p));

const lines = args.corpus
  ? readCorpusFile(resolvePath(args.corpus))
  : generateArithmeticLines(Number(args["max-number"] ?? 9), 100_000, 42);
if (lines.length === 0) {
  console.error("Corpus is empty.");
  process.exit(1);
}
const { train, val } = splitCorpus(lines, 0.1, 42);

const backend = await initNodeBackend();
console.log(`Backend: ${backend}`);

let model;
if (args.resume) {
  model = loadModelDir(resolvePath(args.resume));
  console.log(`Resumed ${args.resume}`);
} else {
  const tokenizer = buildTokenizer(lines.join("\n"));
  const preset = (args.preset ?? "mini") as PresetId;
  let cfg: GptConfig = { ...DEFAULT_CONFIG, ...presetConfig(preset), vocabSize: tokenizer.vocab.length };
  if (args.context) cfg.contextLength = Number(args.context);
  if (args["target-params"]) cfg = { ...cfg, ...shapeForTargetParams(Number(args["target-params"]), cfg.vocabSize, cfg.contextLength) };
  if (args.layers) cfg.layers = Number(args.layers);
  if (args["d-model"]) cfg.dModel = Number(args["d-model"]);
  if (args.heads) cfg.heads = Number(args.heads);
  if (args.ffn) cfg.ffnSize = Number(args.ffn);
  if (args.lr) cfg.learningRate = Number(args.lr);
  if (args.batch) cfg.batchSize = Number(args.batch);
  cfg = normalizeConfig(cfg);

  const mem = memoryEstimate(cfg);
  console.log(`Params: ${paramBreakdown(cfg)}`);
  console.log(`Memory estimate: ${formatBytes(mem.totalBytes)} (system RAM ${formatBytes(os.totalmem())})`);
  if (mem.totalBytes > os.totalmem() * 0.9 && !args.force) {
    console.error("Estimated memory exceeds system RAM. Use a smaller size, smaller --batch, or --force.");
    process.exit(1);
  }
  model = createModel(cfg, tokenizer);
}

const cfg = model.config;
console.log(
  `Model: ${formatCount(countModelParams(model))} params · layers ${cfg.layers} · d_model ${cfg.dModel} · heads ${cfg.heads} · ffn ${cfg.ffnSize} · context ${cfg.contextLength} · vocab ${cfg.vocabSize}`,
);
console.log(`Corpus: ${train.length} train lines · ${val.length} val lines`);

const trainer = new Trainer(model, encodeLines(model.tokenizer, train), encodeLines(model.tokenizer, val));
const steps = Number(args.steps ?? 2000);
const evalEvery = Math.max(1, Math.floor(steps / 20));
const outDir = resolvePath(args.out ?? args.resume ?? `models/${args.preset ?? "mini"}`);
const start = performance.now();

let running = 0;
let lastTrainLoss = -1;
let lastValLoss = -1;
let lastValAcc = -1;
for (let s = 1; s <= steps; s += 1) {
  running += trainer.trainStep();
  if (s % evalEvery === 0 || s === steps) {
    const valLoss = trainer.evalLoss(2);
    const acc = lineAccuracy(model, val.length > 0 ? val : train, 30);
    lastTrainLoss = running / evalEvery;
    lastValLoss = valLoss;
    lastValAcc = acc;
    const secs = (performance.now() - start) / 1000;
    console.log(
      `step ${s}/${steps}  loss ${lastTrainLoss.toFixed(3)}  val ${valLoss.toFixed(3)}  val-acc ${(acc * 100).toFixed(0)}%  ${secs.toFixed(0)}s`,
    );
    running = 0;
  }
}

saveModelDir(model, outDir);
console.log(`Saved to ${outDir}`);

const versioned = saveModelVersioned(model, {
  step: steps,
  loss: lastTrainLoss,
  valLoss: lastValLoss,
  valAcc: lastValAcc,
  exportedAt: new Date().toISOString(),
  label: path.basename(outDir),
});
console.log(`Versioned export → models/exports/${versioned.id}`);
console.log(`  npm run start:model -- ${versioned.version} --mode chat`);
const sample = val[0] ?? train[0];
const prompt = sample.includes("=") ? sample.slice(0, sample.indexOf("=") + 1) : sample.slice(0, Math.ceil(sample.length / 2));
console.log(`Sample: "${prompt}" → "${generate(model, prompt, { maxNewTokens: 40 }).text}"`);
trainer.dispose();
disposeModel(model);
