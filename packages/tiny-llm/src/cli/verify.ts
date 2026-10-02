import { PRESETS, formatCount, paramCount, shapeForTargetParams } from "../config.js";
import { encodeLines, generateArithmeticLines, splitCorpus } from "../corpus.js";
import { generate } from "../generate.js";
import { countModelParams, createModel, disposeModel } from "../gpt.js";
import { buildTokenizer } from "../tokenizer.js";
import { Trainer, lineAccuracy } from "../trainer.js";
import { initNodeBackend } from "./node-backend.js";

console.log(`Backend: ${await initNodeBackend()}`);

for (const target of [1e4, 1e6, 1e7, 1e8, 1e9]) {
  const shape = shapeForTargetParams(target, 100, 256);
  const p = paramCount({ vocabSize: 100, contextLength: 256, ...shape });
  console.log(`target ${formatCount(target)} → ${formatCount(p)} (layers ${shape.layers}, d_model ${shape.dModel}, heads ${shape.heads})`);
}

const lines = generateArithmeticLines(3, 1000, 42);
const { train, val } = splitCorpus(lines, 0.15, 42);
const tokenizer = buildTokenizer(lines.join("\n"));
const model = createModel({ ...PRESETS[0].config, vocabSize: 0, contextLength: 16 }, tokenizer);
console.log(`Mini model: ${countModelParams(model)} params`);
const trainer = new Trainer(model, encodeLines(tokenizer, train), encodeLines(tokenizer, val));
const start = performance.now();
for (let s = 1; s <= 1500; s += 1) {
  const loss = trainer.trainStep();
  if (s % 300 === 0) {
    console.log(`step ${s} loss ${loss.toFixed(3)} train-acc ${(lineAccuracy(model, train, 40) * 100).toFixed(0)}% val-acc ${(lineAccuracy(model, val) * 100).toFixed(0)}%`);
  }
}
console.log(`${((performance.now() - start) / 1000).toFixed(1)}s`);
for (const prompt of ["2+2", "2+2=", "3*3=", "1-3="]) {
  console.log(`"${prompt}" → "${generate(model, prompt, { maxNewTokens: 6 }).text}"`);
}
trainer.dispose();
disposeModel(model);
