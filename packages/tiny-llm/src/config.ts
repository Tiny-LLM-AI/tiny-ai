/** Shape + training hyper-parameters of a GPT-style decoder. Every field is free to change. */
export interface GptConfig {
  /** Filled from the tokenizer when the model is created. */
  vocabSize: number;
  contextLength: number;
  dModel: number;
  layers: number;
  heads: number;
  ffnSize: number;
  dropout: number;
  learningRate: number;
  batchSize: number;
  seed: number;
}

export type PresetId = "mini" | "small" | "medium" | "large" | "xl";

export interface Preset {
  id: PresetId;
  label: string;
  config: Omit<GptConfig, "vocabSize">;
}

export const PRESETS: Preset[] = [
  {
    id: "mini",
    label: "Mini (~30k)",
    config: { contextLength: 32, dModel: 32, layers: 2, heads: 2, ffnSize: 128, dropout: 0, learningRate: 3e-3, batchSize: 32, seed: 42 },
  },
  {
    id: "small",
    label: "Small (~1M)",
    config: { contextLength: 64, dModel: 128, layers: 4, heads: 4, ffnSize: 512, dropout: 0, learningRate: 1e-3, batchSize: 32, seed: 42 },
  },
  {
    id: "medium",
    label: "Medium (~10M)",
    config: { contextLength: 128, dModel: 384, layers: 6, heads: 6, ffnSize: 1536, dropout: 0.1, learningRate: 6e-4, batchSize: 32, seed: 42 },
  },
  {
    id: "large",
    label: "Large (~100M)",
    config: { contextLength: 256, dModel: 768, layers: 12, heads: 12, ffnSize: 3072, dropout: 0.1, learningRate: 3e-4, batchSize: 16, seed: 42 },
  },
  {
    id: "xl",
    label: "XL (~1B)",
    config: { contextLength: 512, dModel: 2048, layers: 20, heads: 16, ffnSize: 8192, dropout: 0.1, learningRate: 2e-4, batchSize: 8, seed: 42 },
  },
];

export const DEFAULT_CONFIG: GptConfig = { vocabSize: 0, ...PRESETS[0].config };

export function presetConfig(id: PresetId): Omit<GptConfig, "vocabSize"> {
  const preset = PRESETS.find((p) => p.id === id) ?? PRESETS[0];
  return { ...preset.config };
}

type Shape = Pick<GptConfig, "vocabSize" | "contextLength" | "dModel" | "layers" | "ffnSize">;

/** Exact number of trainable values (output head is tied to the token embedding). */
export function paramCount(cfg: Shape): number {
  const { vocabSize: v, contextLength: c, dModel: d, layers: l, ffnSize: f } = cfg;
  const perBlock = 4 * d * d + 2 * d * f + 9 * d + f;
  return v * d + c * d + l * perBlock + 2 * d;
}

export function paramBreakdown(cfg: Shape): string {
  const { vocabSize: v, contextLength: c, dModel: d, layers: l, ffnSize: f } = cfg;
  const perBlock = 4 * d * d + 2 * d * f + 9 * d + f;
  return `tokens ${v}×${d} + positions ${c}×${d} + ${l} blocks × ${perBlock.toLocaleString()} + final norm ${2 * d} = ${paramCount(cfg).toLocaleString()}`;
}

export interface MemoryEstimate {
  /** Weights + gradients + Adam m/v, float32. */
  trainingStateBytes: number;
  activationBytes: number;
  totalBytes: number;
}

export function memoryEstimate(cfg: GptConfig): MemoryEstimate {
  const params = paramCount(cfg);
  const trainingStateBytes = params * 4 * 4;
  const { batchSize: b, contextLength: t, dModel: d, layers: l, heads: h, ffnSize: f, vocabSize: v } = cfg;
  const perLayer = t * (8 * d + 2 * f) + h * t * t * 2;
  const activationBytes = 4 * b * (l * perLayer + t * v * 2);
  return { trainingStateBytes, activationBytes, totalBytes: trainingStateBytes + activationBytes };
}

export type Trainability = "browser" | "browser-slow" | "cli";

export const BROWSER_FAST_LIMIT = 5_000_000;
export const BROWSER_MAX_LIMIT = 50_000_000;

export function trainability(cfg: Shape): Trainability {
  const p = paramCount(cfg);
  if (p <= BROWSER_FAST_LIMIT) return "browser";
  if (p <= BROWSER_MAX_LIMIT) return "browser-slow";
  return "cli";
}

export function formatBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(value >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

export function formatCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

/** Largest head count with head size >= 16 (64 for wide models) that divides dModel. */
export function suggestedHeads(dModel: number): number {
  const headSize = dModel >= 256 ? 64 : 16;
  let h = Math.max(1, Math.floor(dModel / headSize));
  while (h > 1 && dModel % h !== 0) h -= 1;
  return h;
}

/**
 * Pick layers/dModel/heads/ffn whose param count is closest to `target`,
 * preferring GPT-like width/depth ratios (dModel / layers around 64).
 */
export function shapeForTargetParams(
  target: number,
  vocabSize: number,
  contextLength: number,
): Pick<GptConfig, "dModel" | "layers" | "heads" | "ffnSize"> {
  let best = { dModel: 16, layers: 1, heads: 1, ffnSize: 64 };
  let bestScore = Number.POSITIVE_INFINITY;
  for (let layers = 1; layers <= 96; layers += 1) {
    for (let dModel = 16; dModel <= 12288; dModel += dModel < 256 ? 16 : 64) {
      const ffnSize = 4 * dModel;
      const p = paramCount({ vocabSize, contextLength, dModel, layers, ffnSize });
      const sizeErr = Math.abs(Math.log(p / Math.max(1, target)));
      const ratioErr = Math.abs(Math.log(dModel / layers / 64));
      const score = sizeErr + 0.08 * ratioErr;
      if (score < bestScore) {
        bestScore = score;
        best = { dModel, layers, heads: suggestedHeads(dModel), ffnSize };
      }
    }
  }
  return best;
}

/** Fix values that would break the model (heads must divide dModel, everything >= 1). */
export function normalizeConfig(cfg: GptConfig): GptConfig {
  const dModel = Math.max(4, Math.floor(cfg.dModel));
  let heads = Math.max(1, Math.min(Math.floor(cfg.heads), dModel));
  while (dModel % heads !== 0) heads -= 1;
  return {
    ...cfg,
    contextLength: Math.max(2, Math.floor(cfg.contextLength)),
    dModel,
    layers: Math.max(1, Math.floor(cfg.layers)),
    heads,
    ffnSize: Math.max(1, Math.floor(cfg.ffnSize)),
    dropout: Math.min(0.9, Math.max(0, cfg.dropout)),
    learningRate: Math.max(1e-6, cfg.learningRate),
    batchSize: Math.max(1, Math.floor(cfg.batchSize)),
  };
}
