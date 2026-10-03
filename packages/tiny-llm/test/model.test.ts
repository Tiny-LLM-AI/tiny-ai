import * as tf from "@tensorflow/tfjs";
import { beforeAll, describe, expect, it } from "vitest";
import {
  BOS_ID,
  DEFAULT_CONFIG,
  PRESETS,
  Trainer,
  UNK_ID,
  buildTokenizer,
  countModelParams,
  createModel,
  decode,
  deserializeModel,
  disposeModel,
  encode,
  encodeLines,
  forward,
  generate,
  generateAsync,
  generateArithmeticLines,
  normalizeConfig,
  paramCount,
  parseCorpus,
  sampleBatch,
  serializeModel,
  shapeForTargetParams,
  splitCorpus,
  mulberry32,
  type GptConfig,
} from "../src/index.js";

const TINY: GptConfig = { ...DEFAULT_CONFIG, contextLength: 12, dModel: 16, layers: 2, heads: 2, ffnSize: 32, batchSize: 8 };

it("fixed-size asynchronous inference preserves causal predictions and releases tensors", async () => {
  const tokenizer = buildTokenizer("xin chào 123456789");
  const model = createModel({ ...TINY, vocabSize: tokenizer.vocab.length }, tokenizer);
  try {
    const count = tf.memory().numTensors;
    for (const prompt of ["x", "xin chào", "xin chào 123456789"]) {
      const options = { maxNewTokens: 15, temperature: 0 };
      const expected = generate(model, prompt, options);
      const actual = await generateAsync(model, prompt, options);
      expect(actual.text).toBe(expected.text);
      expect(actual.steps.map(s => s.chosenId)).toEqual(expected.steps.map(s => s.chosenId));
    }
    expect(tf.memory().numTensors).toBe(count);
  } finally { disposeModel(model); }
});

beforeAll(async () => {
  await tf.setBackend("cpu");
  await tf.ready();
});

describe("tokenizer", () => {
  it("round-trips Vietnamese text as one token per letter", () => {
    const text = "Xin chào, tôi là mô hình ngôn ngữ tiếng Việt.";
    const tok = buildTokenizer(text);
    const ids = encode(tok, text);
    expect(ids).not.toContain(UNK_ID);
    expect(ids.length).toBe([...text.normalize("NFC")].length);
    expect(decode(tok, ids)).toBe(text.normalize("NFC"));
  });

  it("maps unknown characters to <UNK> instead of throwing", () => {
    const tok = buildTokenizer("2+3=5");
    expect(encode(tok, "2+x")).toEqual([tok.index.get("2"), tok.index.get("+"), UNK_ID]);
  });
});

describe("corpus", () => {
  it("generates arithmetic lines without a special format", () => {
    const lines = generateArithmeticLines(3, 1000, 1);
    expect(lines).toContain("2+2=4");
    expect(lines).toContain("1-3=-2");
  });

  it("splits lines into disjoint train/val", () => {
    const lines = parseCorpus(generateArithmeticLines(9, 500, 1).join("\n"));
    const { train, val } = splitCorpus(lines, 0.2, 1);
    expect(train.length + val.length).toBe(lines.length);
    const valSet = new Set(val);
    expect(train.some((l) => valSet.has(l))).toBe(false);
  });

  it("builds causal batches where y is x shifted by one", () => {
    const tok = buildTokenizer("abc");
    const stream = encodeLines(tok, ["abc", "cab"]);
    const batch = sampleBatch(stream, 4, 3, mulberry32(1));
    for (let b = 0; b < 4; b += 1) {
      for (let t = 0; t < 2; t += 1) expect(batch.y[b * 3 + t]).toBe(batch.x[b * 3 + t + 1]);
    }
    expect(stream[0]).toBe(BOS_ID);
  });
});

describe("config", () => {
  it("paramCount matches the real TF variable sizes", () => {
    const tok = buildTokenizer("hello world 0123456789+-*/=");
    const model = createModel(TINY, tok);
    expect(countModelParams(model)).toBe(paramCount(model.config));
    disposeModel(model);
  });

  it.each([1e4, 1e6, 1e7, 1e8, 1e9])("shapeForTargetParams(%d) lands within 15%%", (target) => {
    const shape = shapeForTargetParams(target, 100, 256);
    const p = paramCount({ vocabSize: 100, contextLength: 256, ...shape });
    expect(Math.abs(p - target) / target).toBeLessThan(0.15);
    expect(shape.dModel % shape.heads).toBe(0);
  });

  it("presets scale from tens of thousands to about a billion params", () => {
    const counts = PRESETS.map((p) => paramCount({ ...p.config, vocabSize: 100 }));
    expect(counts[0]).toBeLessThan(50_000);
    expect(counts.at(-1)!).toBeGreaterThan(9e8);
    for (let i = 1; i < counts.length; i += 1) expect(counts[i]).toBeGreaterThan(counts[i - 1]);
  });

  it("normalizeConfig keeps heads dividing dModel", () => {
    const cfg = normalizeConfig({ ...TINY, dModel: 30, heads: 4 });
    expect(cfg.dModel % cfg.heads).toBe(0);
  });
});

describe("gpt", () => {
  it("is causal: changing a later token does not change earlier logits", () => {
    const tok = buildTokenizer("abcdef");
    const model = createModel(TINY, tok);
    const a = [1, 4, 5, 6, 7];
    const b = [1, 4, 5, 6, 9];
    const [la, lb] = [a, b].map((ids) =>
      tf.tidy(() => forward(model, tf.tensor2d([ids], [1, ids.length], "int32")).arraySync() as number[][][]),
    );
    for (let t = 0; t < 4; t += 1) {
      for (let v = 0; v < la[0][t].length; v += 1) expect(la[0][t][v]).toBeCloseTo(lb[0][t][v], 5);
    }
    expect(la[0][4]).not.toEqual(lb[0][4]);
    disposeModel(model);
  });

  it("save/load round-trip keeps outputs identical", () => {
    const tok = buildTokenizer("xin chào 123");
    const model = createModel(TINY, tok);
    const { manifest, weights } = serializeModel(model);
    const copy = deserializeModel(JSON.parse(JSON.stringify(manifest)), weights);
    const ids = [1, 4, 5, 6];
    const run = (m: typeof model) =>
      tf.tidy(() => Array.from(forward(m, tf.tensor2d([ids], [1, 4], "int32")).dataSync()));
    expect(run(copy)).toEqual(run(model));
    disposeModel(model);
    disposeModel(copy);
  });

  it("training lowers the loss", () => {
    const lines = generateArithmeticLines(2, 100, 1);
    const tok = buildTokenizer(lines.join("\n"));
    const model = createModel({ ...TINY, learningRate: 5e-3 }, tok);
    const { train, val } = splitCorpus(lines, 0.2, 1);
    const trainer = new Trainer(model, encodeLines(tok, train), encodeLines(tok, val));
    const first = trainer.evalLoss(2);
    for (let i = 0; i < 60; i += 1) trainer.trainStep();
    expect(trainer.evalLoss(2)).toBeLessThan(first * 0.8);
    trainer.dispose();
    disposeModel(model);
  }, 60_000);

  it("untrained val char accuracy is modest, not inflated", () => {
    const lines = generateArithmeticLines(5, 200, 1);
    const tok = buildTokenizer(lines.join("\n"));
    const { train, val } = splitCorpus(lines, 0.15, 1);
    const model = createModel(TINY, tok);
    const trainer = new Trainer(model, encodeLines(tok, train), encodeLines(tok, val));
    const acc = trainer.evalTokenAccuracy(8);
    expect(acc).toBeGreaterThan(0);
    expect(acc).toBeLessThan(0.35);
    trainer.dispose();
    disposeModel(model);
  });
});
