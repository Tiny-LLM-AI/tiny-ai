import * as tf from "@tensorflow/tfjs";
import { contextWindow, forward, type GptModel } from "./gpt.js";
import { mulberry32, type Rng } from "./random.js";
import {
  BOS_ID,
  EOS_ID,
  PAD_ID,
  encode,
  isSpecialId,
  tokenText,
} from "./tokenizer.js";

export interface GenerateOptions {
  maxNewTokens?: number;
  /** 0 = greedy. */
  temperature?: number;
  topK?: number;
  seed?: number;
}

export interface GenerateAsyncOptions extends GenerateOptions {
  /** Called after each token; use for streaming UI. */
  onStep?: (
    step: GenerateStep,
    partialText: string,
    stepIndex: number,
  ) => void | Promise<void>;
}

export interface GenerateStep {
  /** Token ids the model saw for this step (prompt + generated so far). */
  contextIds: number[];
  chosenId: number;
  chosenText: string;
  /** Top candidates, highest probability first. */
  top: { id: number; text: string; prob: number }[];
}

export interface GenerateResult {
  text: string;
  steps: GenerateStep[];
  promptIds: number[];
}

function pick(
  probs: Float32Array,
  temperature: number,
  topK: number,
  rng: Rng,
): number {
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < probs.length; i += 1)
      if (probs[i] > probs[best]) best = i;
    return best;
  }
  const scaled = Array.from(probs, (p, id) => ({
    id,
    w: Math.pow(Math.max(p, 1e-12), 1 / temperature),
  }));
  scaled.sort((a, b) => b.w - a.w);
  const pool = topK > 0 ? scaled.slice(0, topK) : scaled;
  const total = pool.reduce((s, c) => s + c.w, 0);
  let r = rng() * total;
  for (const c of pool) {
    r -= c.w;
    if (r <= 0) return c.id;
  }
  return pool[pool.length - 1].id;
}

/** Yield without waiting a full animation frame - keeps chat responsive but faster than rAF. */
const yieldToMain = () =>
  new Promise<void>((resolve) => setTimeout(resolve, 0));

function idsToText(model: GptModel, out: number[]): string {
  return out
    .filter((id) => !isSpecialId(id))
    .map((id) => model.tokenizer.vocab[id])
    .join("");
}

/** Continues any free-form prompt token by token until <EOS> or maxNewTokens. */
export function generate(
  model: GptModel,
  prompt: string,
  opts: GenerateOptions = {},
): GenerateResult {
  const { maxNewTokens = 32, temperature = 0, topK = 0, seed = 1 } = opts;
  const rng = mulberry32(seed);
  const promptIds = [BOS_ID, ...encode(model.tokenizer, prompt)];
  const ids = [...promptIds];
  const steps: GenerateStep[] = [];
  const out: number[] = [];

  for (let i = 0; i < maxNewTokens; i += 1) {
    const window = contextWindow(ids, model.config.contextLength);
    const probs = tf.tidy(() => {
      const logits = forward(
        model,
        tf.tensor2d([window], [1, window.length], "int32"),
      );
      const last = tf.reshape(
        tf.slice(
          logits,
          [0, window.length - 1, 0],
          [1, 1, model.config.vocabSize],
        ),
        [-1],
      );
      return tf.softmax(last).dataSync() as Float32Array;
    });
    const chosenId = pick(probs, temperature, topK, rng);
    const top = Array.from(probs, (prob, id) => ({
      id,
      text: tokenText(model.tokenizer, id),
      prob,
    }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 10);
    steps.push({
      contextIds: [...window],
      chosenId,
      chosenText: tokenText(model.tokenizer, chosenId),
      top,
    });
    if (chosenId === EOS_ID || chosenId === PAD_ID || chosenId === BOS_ID)
      break;
    ids.push(chosenId);
    out.push(chosenId);
  }

  return { text: idsToText(model, out), steps, promptIds };
}

/**
 * Same as `generate`, but yields to the browser between tokens so the UI can update.
 * Each step runs one forward pass; `onStep` receives the partial decoded text so far.
 */
export async function generateAsync(
  model: GptModel,
  prompt: string,
  opts: GenerateAsyncOptions = {},
): Promise<GenerateResult> {
  const {
    maxNewTokens = 32,
    temperature = 0,
    topK = 0,
    seed = 1,
    onStep,
  } = opts;
  const rng = mulberry32(seed);
  const promptIds = [BOS_ID, ...encode(model.tokenizer, prompt)];
  const ids = [...promptIds];
  const steps: GenerateStep[] = [];
  const out: number[] = [];

  for (let i = 0; i < maxNewTokens; i += 1) {
    await yieldToMain();
    const window = contextWindow(ids, model.config.contextLength);
    const probs = tf.tidy(() => {
      const logits = forward(
        model,
        tf.tensor2d([window], [1, window.length], "int32"),
      );
      const last = tf.reshape(
        tf.slice(
          logits,
          [0, window.length - 1, 0],
          [1, 1, model.config.vocabSize],
        ),
        [-1],
      );
      return tf.softmax(last).dataSync() as Float32Array;
    });
    const chosenId = pick(probs, temperature, topK, rng);
    const top = Array.from(probs, (prob, id) => ({
      id,
      text: tokenText(model.tokenizer, id),
      prob,
    }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 10);
    const step: GenerateStep = {
      contextIds: [...window],
      chosenId,
      chosenText: tokenText(model.tokenizer, chosenId),
      top,
    };
    steps.push(step);
    if (chosenId === EOS_ID || chosenId === PAD_ID || chosenId === BOS_ID) {
      await onStep?.(step, idsToText(model, out), steps.length - 1);
      await yieldToMain();
      break;
    }
    ids.push(chosenId);
    out.push(chosenId);
    await onStep?.(step, idsToText(model, out), steps.length - 1);
  }

  return { text: idsToText(model, out), steps, promptIds };
}
