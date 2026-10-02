import type { Rng } from "./random.js";

export interface TokenBatch {
  /** [batch × context] input ids, row-major. */
  x: Int32Array;
  /** Next-token targets for every position (causal LM). */
  y: Int32Array;
  batchSize: number;
  contextLength: number;
}

/** Repeats a short stream so a full context window always fits. */
export function ensureStreamLength(stream: Int32Array, contextLength: number): Int32Array {
  if (stream.length === 0) return stream;
  if (stream.length > contextLength + 1) return stream;
  const out: number[] = [];
  while (out.length <= contextLength + 1) out.push(...stream);
  return Int32Array.from(out);
}

/** Random windows of contextLength+1 tokens; loss is taken at every position. */
export function sampleBatch(stream: Int32Array, batchSize: number, contextLength: number, rng: Rng): TokenBatch {
  const data = ensureStreamLength(stream, contextLength);
  const x = new Int32Array(batchSize * contextLength);
  const y = new Int32Array(batchSize * contextLength);
  const maxStart = data.length - contextLength - 1;
  for (let b = 0; b < batchSize; b += 1) {
    const start = Math.floor(rng() * (maxStart + 1));
    for (let t = 0; t < contextLength; t += 1) {
      x[b * contextLength + t] = data[start + t];
      y[b * contextLength + t] = data[start + t + 1];
    }
  }
  return { x, y, batchSize, contextLength };
}
