import * as tf from "@tensorflow/tfjs";
import type { GptConfig } from "./config.js";
import { createModel, disposeModel, paramSpecs, type GptModel } from "./gpt.js";
import { tokenizerFromVocab } from "./tokenizer.js";

export const MODEL_FORMAT = "tiny-gpt/v1";

/** Stored as `model.json`; weights live next to it in `weights.bin` (float32, little-endian). */
export interface ModelManifest {
  format: typeof MODEL_FORMAT;
  config: GptConfig;
  vocab: string[];
  tensors: { name: string; shape: number[]; offset: number; length: number }[];
  parameterCount: number;
}

/** All weights in canonical order, concatenated. */
export function exportWeights(model: GptModel): Float32Array {
  let total = 0;
  for (const v of model.params.values()) total += v.size;
  const out = new Float32Array(total);
  let offset = 0;
  for (const v of model.params.values()) {
    out.set(v.dataSync() as Float32Array, offset);
    offset += v.size;
  }
  return out;
}

export function importWeights(model: GptModel, flat: Float32Array): void {
  const expected = [...model.params.values()].reduce((sum, v) => sum + v.size, 0);
  if (expected !== flat.length) throw new Error(`Weight size mismatch: expected ${expected}, got ${flat.length}`);
  if (flat.some((value) => !Number.isFinite(value))) throw new Error("Weights contain NaN or Infinity.");
  let offset = 0;
  for (const v of model.params.values()) {
    const slice = flat.subarray(offset, offset + v.size);
    tf.tidy(() => v.assign(tf.tensor(slice, v.shape, "float32")));
    offset += v.size;
  }
  if (offset !== flat.length) throw new Error(`Weight size mismatch: expected ${offset}, got ${flat.length}`);
}

export function buildManifest(model: GptModel): ModelManifest {
  let offset = 0;
  const tensors = paramSpecs(model.config).map((spec) => {
    const length = spec.shape.reduce((a, b) => a * b, 1);
    const entry = { name: spec.name, shape: spec.shape, offset, length };
    offset += length;
    return entry;
  });
  return { format: MODEL_FORMAT, config: model.config, vocab: model.tokenizer.vocab, tensors, parameterCount: offset };
}

export function serializeModel(model: GptModel): { manifest: ModelManifest; weights: Float32Array } {
  return { manifest: buildManifest(model), weights: exportWeights(model) };
}

export function deserializeModel(manifest: ModelManifest, weights: Float32Array): GptModel {
  if (manifest.format !== MODEL_FORMAT) throw new Error(`Unsupported model format ${manifest.format}`);
  const model = createModel(manifest.config, tokenizerFromVocab(manifest.vocab));
  try {
    importWeights(model, weights);
    return model;
  } catch (err) {
    disposeModel(model);
    throw err;
  }
}
