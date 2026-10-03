import * as tf from "@tensorflow/tfjs";
import { normalizeConfig, type GptConfig } from "./config.js";
import type { Tokenizer } from "./tokenizer.js";

export interface GptModel {
  config: GptConfig;
  tokenizer: Tokenizer;
  /** Insertion order is the canonical order used for serialization. */
  params: Map<string, tf.Variable>;
}

export interface ParamSpec {
  name: string;
  shape: number[];
  init: "normal" | "residual" | "zeros" | "ones";
}

export function paramSpecs(cfg: GptConfig): ParamSpec[] {
  const { vocabSize: v, contextLength: c, dModel: d, layers, ffnSize: f } = cfg;
  const specs: ParamSpec[] = [
    { name: "wte", shape: [v, d], init: "normal" },
    { name: "wpe", shape: [c, d], init: "normal" },
  ];
  for (let i = 0; i < layers; i += 1) {
    const p = `h${i}`;
    specs.push(
      { name: `${p}.ln1.g`, shape: [d], init: "ones" },
      { name: `${p}.ln1.b`, shape: [d], init: "zeros" },
      { name: `${p}.attn.wqkv`, shape: [d, 3 * d], init: "normal" },
      { name: `${p}.attn.bqkv`, shape: [3 * d], init: "zeros" },
      { name: `${p}.attn.wo`, shape: [d, d], init: "residual" },
      { name: `${p}.attn.bo`, shape: [d], init: "zeros" },
      { name: `${p}.ln2.g`, shape: [d], init: "ones" },
      { name: `${p}.ln2.b`, shape: [d], init: "zeros" },
      { name: `${p}.mlp.w1`, shape: [d, f], init: "normal" },
      { name: `${p}.mlp.b1`, shape: [f], init: "zeros" },
      { name: `${p}.mlp.w2`, shape: [f, d], init: "residual" },
      { name: `${p}.mlp.b2`, shape: [d], init: "zeros" },
    );
  }
  specs.push(
    { name: "lnf.g", shape: [d], init: "ones" },
    { name: "lnf.b", shape: [d], init: "zeros" },
  );
  return specs;
}

export function createModel(config: GptConfig, tokenizer: Tokenizer): GptModel {
  const cfg = normalizeConfig({ ...config, vocabSize: tokenizer.vocab.length });
  const params = new Map<string, tf.Variable>();
  const residualStd = 0.02 / Math.sqrt(2 * cfg.layers);
  let seed = cfg.seed;
  for (const spec of paramSpecs(cfg)) {
    seed += 1;
    const initial = tf.tidy(() => {
      switch (spec.init) {
        case "ones":
          return tf.ones(spec.shape);
        case "zeros":
          return tf.zeros(spec.shape);
        case "residual":
          return tf.randomNormal(spec.shape, 0, residualStd, "float32", seed);
        default:
          return tf.randomNormal(spec.shape, 0, 0.02, "float32", seed);
      }
    });
    params.set(spec.name, tf.variable(initial, true));
    initial.dispose();
  }
  return { config: cfg, tokenizer, params };
}

export function disposeModel(model: GptModel): void {
  for (const v of model.params.values()) v.dispose();
  model.params.clear();
}

export function countModelParams(model: GptModel): number {
  let n = 0;
  for (const v of model.params.values()) n += v.size;
  return n;
}

function p(model: GptModel, name: string): tf.Variable {
  const v = model.params.get(name);
  if (!v) throw new Error(`Missing parameter ${name}`);
  return v;
}

function layerNorm(x: tf.Tensor, g: tf.Tensor, b: tf.Tensor): tf.Tensor {
  const { mean, variance } = tf.moments(x, -1, true);
  return tf.add(
    tf.mul(tf.div(tf.sub(x, mean), tf.sqrt(tf.add(variance, 1e-5))), g),
    b,
  );
}

function gelu(x: tf.Tensor): tf.Tensor {
  const inner = tf.mul(
    Math.sqrt(2 / Math.PI),
    tf.add(x, tf.mul(0.044715, tf.pow(x, 3))),
  );
  return tf.mul(tf.mul(0.5, x), tf.add(1, tf.tanh(inner)));
}

/** [B, T, in] × [in, out] + bias → [B, T, out] */
function linear(x: tf.Tensor, w: tf.Tensor, bias: tf.Tensor): tf.Tensor {
  const [b, t, din] = x.shape;
  const out = w.shape[1]!;
  return tf.add(
    tf.reshape(tf.matMul(tf.reshape(x, [b * t, din]), w as tf.Tensor2D), [
      b,
      t,
      out,
    ]),
    bias,
  );
}

function causalMask(t: number): tf.Tensor {
  const lower = tf.linalg.bandPart(tf.ones([t, t]), -1, 0);
  return tf.mul(tf.sub(1, lower), -1e9);
}

export interface ForwardOptions {
  training?: boolean;
  /** When set, receives attention weights [B, heads, T, T] per layer (kept alive - caller disposes). */
  attention?: tf.Tensor[];
}

/** ids: [B, T] int32 with T ≤ contextLength. Returns logits [B, T, vocab]. */
export function forward(
  model: GptModel,
  ids: tf.Tensor2D,
  opts: ForwardOptions = {},
): tf.Tensor3D {
  const { dModel: d, heads, layers, dropout } = model.config;
  const [b, t] = ids.shape;
  const hd = d / heads;
  const drop = (x: tf.Tensor) =>
    opts.training && dropout > 0 ? tf.dropout(x, dropout) : x;

  const pos = tf.slice(p(model, "wpe"), [0, 0], [t, d]);
  let x: tf.Tensor = drop(tf.add(tf.gather(p(model, "wte"), ids), pos));
  const mask = causalMask(t);
  const scale = 1 / Math.sqrt(hd);

  for (let i = 0; i < layers; i += 1) {
    const pre = `h${i}`;
    const h = layerNorm(x, p(model, `${pre}.ln1.g`), p(model, `${pre}.ln1.b`));
    const qkv = linear(
      h,
      p(model, `${pre}.attn.wqkv`),
      p(model, `${pre}.attn.bqkv`),
    );
    const [q, k, v] = tf
      .split(qkv, 3, -1)
      .map((m) => tf.transpose(tf.reshape(m, [b, t, heads, hd]), [0, 2, 1, 3]));
    let att = tf.add(tf.mul(tf.matMul(q, k, false, true), scale), mask);
    att = tf.softmax(att, -1);
    if (opts.attention) opts.attention.push(tf.keep(att.clone()));
    const y = tf.reshape(tf.transpose(tf.matMul(drop(att), v), [0, 2, 1, 3]), [
      b,
      t,
      d,
    ]);
    x = tf.add(
      x,
      drop(linear(y, p(model, `${pre}.attn.wo`), p(model, `${pre}.attn.bo`))),
    );

    const h2 = layerNorm(x, p(model, `${pre}.ln2.g`), p(model, `${pre}.ln2.b`));
    const m = gelu(
      linear(h2, p(model, `${pre}.mlp.w1`), p(model, `${pre}.mlp.b1`)),
    );
    x = tf.add(
      x,
      drop(linear(m, p(model, `${pre}.mlp.w2`), p(model, `${pre}.mlp.b2`))),
    );
  }

  const xf = layerNorm(x, p(model, "lnf.g"), p(model, "lnf.b"));
  const logits = tf.matMul(
    tf.reshape(xf, [b * t, d]),
    p(model, "wte") as tf.Tensor2D,
    false,
    true,
  );
  return tf.reshape(logits, [b, t, model.config.vocabSize]);
}

/** Mean next-token cross-entropy over every position. */
export function lossFromLogits(
  logits: tf.Tensor3D,
  targets: tf.Tensor2D,
): tf.Scalar {
  const v = logits.shape[2];
  const flat = tf.reshape(logits, [-1, v]);
  const logp = tf.logSoftmax(flat);
  const picked = tf.sum(
    tf.mul(logp, tf.oneHot(tf.reshape(targets, [-1]), v)),
    -1,
  );
  return tf.neg(tf.mean(picked)) as tf.Scalar;
}

export interface ForwardTrace {
  tokenIds: number[];
  /** [layer][head][query][key] */
  attention: number[][][][];
  /** Softmax probabilities of the next token after the last position. */
  nextProbs: Float32Array;
  nextLogits: Float32Array;
}

/** Window of the last contextLength ids (inference). */
export function contextWindow(ids: number[], contextLength: number): number[] {
  return ids.length > contextLength
    ? ids.slice(ids.length - contextLength)
    : ids;
}

export interface ForwardTraceOptions {
  /** Default true. Set false for a faster forward pass when only next-token probs are needed. */
  captureAttention?: boolean;
}

export function forwardTrace(
  model: GptModel,
  ids: number[],
  opts: ForwardTraceOptions = {},
): ForwardTrace {
  const captureAttention = opts.captureAttention ?? true;
  const tokenIds = contextWindow(ids, model.config.contextLength);
  const attnTensors: tf.Tensor[] = [];
  const { logits, probs } = tf.tidy(() => {
    const input = tf.tensor2d([tokenIds], [1, tokenIds.length], "int32");
    const out = forward(
      model,
      input,
      captureAttention ? { attention: attnTensors } : {},
    );
    const last = tf.reshape(
      tf.slice(
        out,
        [0, tokenIds.length - 1, 0],
        [1, 1, model.config.vocabSize],
      ),
      [-1],
    );
    return { logits: tf.keep(last), probs: tf.keep(tf.softmax(last)) };
  });
  const attention = captureAttention
    ? attnTensors.map((a) => (a.arraySync() as number[][][][])[0])
    : [];
  attnTensors.forEach((a) => a.dispose());
  const nextLogits = logits.dataSync() as Float32Array;
  const nextProbs = probs.dataSync() as Float32Array;
  logits.dispose();
  probs.dispose();
  return {
    tokenIds,
    attention,
    nextProbs: Float32Array.from(nextProbs),
    nextLogits: Float32Array.from(nextLogits),
  };
}

/** One GPU read for several token embedding rows (faster than per-token slices in the UI). */
export function embeddingRows(
  model: GptModel,
  tokenIds: number[],
  maxDims = 8,
): number[][] {
  if (tokenIds.length === 0) return [];
  const dims = Math.min(maxDims, model.config.dModel);
  return tf.tidy(() => {
    const wte = p(model, "wte");
    const rows = tf.slice(
      tf.gather(wte, tf.tensor1d(tokenIds, "int32")),
      [0, 0],
      [tokenIds.length, dims],
    );
    return rows.arraySync() as number[][];
  });
}

/** Compile shaders / warm caches so the first chat message is not a cold start. */
export function warmupForward(model: GptModel): void {
  tf.tidy(() => {
    const input = tf.zeros([1, model.config.contextLength], "int32") as tf.Tensor2D;
    const logits = forward(model, input);
    // Complete shader compilation and weight upload before chat is enabled.
    tf.softmax(logits.slice([0, 0, 0], [1, 1, model.config.vocabSize]).reshape([-1])).dataSync();
  });
}

/** First `maxDims` values of a token's embedding vector, for display. */
export function embeddingRow(
  model: GptModel,
  tokenId: number,
  maxDims = 8,
): number[] {
  const wte = p(model, "wte");
  const dims = Math.min(maxDims, model.config.dModel);
  return tf.tidy(() =>
    Array.from(tf.slice(wte, [tokenId, 0], [1, dims]).dataSync()),
  );
}

export interface WteDisplaySlice {
  /** [embedding_dim][token_column] */
  values: number[][];
  tokenIds: number[];
  highlightCol: number;
}

/**
 * Token embedding matrix for the UI: columns = tokens (characters), rows = embedding dims.
 * Centers the token window on `focusTokenId` when set.
 */
export function sampleWteDisplay(
  model: GptModel,
  maxDims = 16,
  maxTokens = 12,
  focusTokenId = -1,
): WteDisplaySlice {
  const vocab = model.config.vocabSize;
  const dims = Math.min(maxDims, model.config.dModel);
  const nTok = Math.min(maxTokens, vocab);
  let start = 0;
  if (focusTokenId >= 0)
    start = Math.max(
      0,
      Math.min(focusTokenId - Math.floor(nTok / 2), vocab - nTok),
    );
  const tokenIds = Array.from({ length: nTok }, (_, i) => start + i);
  const rows = tf.tidy(() => {
    const block = tf
      .slice(p(model, "wte"), [start, 0], [nTok, dims])
      .arraySync() as number[][];
    return Array.from({ length: dims }, (_, d) =>
      block.map((tokRow) => tokRow[d]),
    );
  });
  const highlightCol =
    focusTokenId >= start && focusTokenId < start + nTok
      ? focusTokenId - start
      : -1;
  return { values: rows, tokenIds, highlightCol };
}

/** Up to maxRows × maxCols slice of a parameter, for display. */
export function sampleParam(
  model: GptModel,
  name: string,
  maxRows = 32,
  maxCols = 32,
): { rows: number; cols: number; shape: number[]; values: number[][] } {
  const v = p(model, name);
  const shape = v.shape;
  const values = tf.tidy(() => {
    const m = (
      shape.length === 1 ? tf.reshape(v, [1, shape[0]]) : v
    ) as tf.Tensor2D;
    const r = Math.min(maxRows, m.shape[0]);
    const c = Math.min(maxCols, m.shape[1]);
    return tf.slice(m, [0, 0], [r, c]).arraySync() as number[][];
  });
  return { rows: values.length, cols: values[0]?.length ?? 0, shape, values };
}
