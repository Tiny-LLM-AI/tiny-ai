import * as tf from "@tensorflow/tfjs";
import { beforeAll, describe, it, expect } from "vitest";
import { DEFAULT_CONFIG, browserModelIssue, normalizeConfig } from "../src/config.js";
import { createModel, disposeModel, forward } from "../src/gpt.js";
import { exportWeights, importWeights, serializeModel, deserializeModel } from "../src/model-io.js";
import { encodeLines } from "../src/corpus.js";
import { Trainer } from "../src/trainer.js";
import { generateAsync } from "../src/generate.js";
import { viTrainingConfig, viTokenizer } from "../../tiny-web/src/vi-training.js";
import { cliCommand } from "../../tiny-web/src/settings.js";
import { restoreViCurriculum, newViCurriculum } from "../src/vi-curriculum.js";

beforeAll(async () => { await tf.setBackend("cpu"); await tf.ready(); });
const lines = ["Tôi đang học tiếng Việt.", "Chúng ta cùng đọc một cuốn sách."];

describe("dynamic VI architecture", () => {
  it.each([
    { layers: 1, dModel: 16, heads: 1, ffnSize: 24, contextLength: 8, batchSize: 1 },
    { layers: 2, dModel: 30, heads: 3, ffnSize: 55, contextLength: 17, batchSize: 2 },
    { layers: 3, dModel: 32, heads: 4, ffnSize: 64, contextLength: 24, batchSize: 3 },
    { layers: 1, dModel: 128, heads: 8, ffnSize: 256, contextLength: 32, batchSize: 2 },
  ])("preserves custom shape and trains/round-trips $layers layers, width $dModel", (custom) => {
    const before = tf.memory().numTensors;
    const cfg = viTrainingConfig({ ...DEFAULT_CONFIG, ...custom, learningRate: 0.001, dropout: 0 });
    expect(cfg).toMatchObject(custom);
    const tok = viTokenizer();
    const model = createModel(cfg, tok);
    const trainer = new Trainer(model, encodeLines(tok, lines), encodeLines(tok, lines));
    const weightsBefore = exportWeights(model);
    for (let i = 0; i < 3; i++) expect(Number.isFinite(trainer.trainStep())).toBe(true);
    expect(exportWeights(model)).not.toEqual(weightsBefore);
    const { manifest, weights } = serializeModel(model);
    const restored = deserializeModel(manifest, weights);
    expect(restored.config).toEqual(model.config);
    const logits = (m: typeof model) => tf.tidy(() => Array.from(forward(m, tf.tensor2d([[1,4,5]], [1,3], "int32")).dataSync()));
    expect(logits(restored)).toEqual(logits(model));
    trainer.dispose(); disposeModel(model); disposeModel(restored);
    expect(tf.memory().numTensors).toBe(before);
  });
  it("rejects invalid dimensions instead of looping or allocating unbounded tensors", () => {
    for (const dModel of [NaN, Infinity, 1e100, undefined]) {
      expect(() => normalizeConfig({ ...DEFAULT_CONFIG, dModel: dModel as number })).toThrow();
    }
    expect(browserModelIssue({ ...DEFAULT_CONFIG, vocabSize: 200, contextLength: 100000, batchSize: 1024 })).not.toBeNull();
  });
  it("failed weight import leaves the current model unchanged and failed load releases tensors", () => {
    const model = createModel({ ...DEFAULT_CONFIG, dModel: 8, heads: 1, layers: 1, ffnSize: 16 }, viTokenizer());
    const { manifest, weights } = serializeModel(model);
    const wrong = weights.slice(1);
    expect(() => importWeights(model, wrong)).toThrow();
    expect(exportWeights(model)).toEqual(weights);
    const bad = weights.slice(); bad[10] = NaN;
    expect(() => importWeights(model, bad)).toThrow();
    expect(exportWeights(model)).toEqual(weights);
    const before = tf.memory().numTensors;
    expect(() => deserializeModel(manifest, wrong)).toThrow();
    expect(tf.memory().numTensors).toBe(before);
    disposeModel(model);
  });
  it("cancels generation without accessing a disposed model on the next iteration", async () => {
    const model = createModel({ ...DEFAULT_CONFIG, dModel: 8, heads: 1, layers: 1, ffnSize: 16 }, viTokenizer());
    const controller = new AbortController();
    controller.abort(); disposeModel(model);
    const result = await generateAsync(model, "Xin chào", { signal: controller.signal });
    expect(result.steps).toHaveLength(0);
  });
  it("includes dropout and seed in the generated dynamic CLI command", () => {
    const cmd = cliCommand({ ...DEFAULT_CONFIG, layers: 3, dModel: 32, heads: 4, ffnSize: 64, dropout: 0.15, seed: 123 });
    expect(cmd).toContain("--layers 3 --d-model 32 --heads 4 --ffn 64");
    expect(cmd).toContain("--dropout 0.15 --seed 123");
  });
  it("restores malformed optional checkpoint fields safely", () => {
    const restored = restoreViCurriculum({ ...newViCurriculum(), wikiSeed: null });
    expect(typeof restored.wikiSeed).toBe("string");
  });
});
