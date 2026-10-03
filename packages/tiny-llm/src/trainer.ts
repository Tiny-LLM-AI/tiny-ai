import * as tf from "@tensorflow/tfjs";
import { sampleBatch } from "./batches.js";
import { encodeLines, promptAndTarget } from "./corpus.js";
import type { Tokenizer } from "./tokenizer.js";
import { generate } from "./generate.js";
import { forward, lossFromLogits, type GptModel } from "./gpt.js";
import { mulberry32, type Rng } from "./random.js";
import { isSpecialId } from "./tokenizer.js";

export interface TrainerOptions {
  weightDecay?: number;
  clipNorm?: number;
}

/** AdamW + global-norm clipping over a causal LM token stream. */
export class Trainer {
  readonly model: GptModel;
  private trainStream: Int32Array;
  private valStream: Int32Array;
  private optimizer: tf.Optimizer;
  private rng: Rng;
  private weightDecay: number;
  private clipNorm: number;
  private decayVars: tf.Variable[];
  step = 0;

  constructor(model: GptModel, trainStream: Int32Array, valStream: Int32Array, opts: TrainerOptions = {}) {
    this.model = model;
    this.trainStream = trainStream;
    this.valStream = valStream;
    this.optimizer = tf.train.adam(model.config.learningRate, 0.9, 0.95, 1e-8);
    this.rng = mulberry32(model.config.seed + 7);
    this.weightDecay = opts.weightDecay ?? 0.01;
    this.clipNorm = opts.clipNorm ?? 1.0;
    this.decayVars = [...model.params.entries()].filter(([, v]) => v.rank === 2).map(([, v]) => v);
  }

  /** One optimizer step; returns the batch loss. */
  trainStep(): number {
    const value = this.trainStepTensor();
    const loss = value.dataSync()[0];
    value.dispose();
    return loss;
  }

  /**
   * One optimizer step without reading anything back from the device, so WebGL can
   * queue several steps. Caller owns (and must dispose) the returned loss scalar.
   */
  trainStepTensor(): tf.Scalar {
    const { batchSize, contextLength } = this.model.config;
    const batch = sampleBatch(this.trainStream, batchSize, contextLength, this.rng);
    const x = tf.tensor2d(batch.x, [batchSize, contextLength], "int32");
    const y = tf.tensor2d(batch.y, [batchSize, contextLength], "int32");
    const vars = [...this.model.params.values()];

    const { value, grads } = tf.variableGrads(() => lossFromLogits(forward(this.model, x, { training: true }), y), vars);
    const clipped = tf.tidy(() => {
      const names = Object.keys(grads);
      const norm = tf.sqrt(tf.addN(names.map((n) => tf.sum(tf.square(grads[n])))));
      const scale = tf.minimum(1, tf.div(this.clipNorm, tf.add(norm, 1e-6)));
      const out: Record<string, tf.Tensor> = {};
      for (const n of names) out[n] = tf.mul(grads[n], scale);
      return out;
    });
    this.optimizer.applyGradients(clipped as tf.NamedTensorMap);

    if (this.weightDecay > 0) {
      const factor = 1 - this.model.config.learningRate * this.weightDecay;
      tf.tidy(() => {
        for (const v of this.decayVars) v.assign(tf.mul(v, factor));
      });
    }

    Object.values(grads).forEach((g) => g.dispose());
    Object.values(clipped).forEach((g) => g.dispose());
    x.dispose();
    y.dispose();
    this.step += 1;
    return value;
  }

  /** Replace the training token stream (e.g. when new wiki text is appended). */
  setTrainLines(tokenizer: Tokenizer, lines: string[]): void {
    this.trainStream = encodeLines(tokenizer, lines);
  }

  /** Average loss on held-out val windows. Returns -1 when there is no val set. */
  evalLoss(batches = 4): number {
    if (this.valStream.length === 0) return -1;
    const stream = this.valStream;
    const { batchSize, contextLength } = this.model.config;
    const rng = mulberry32(1234);
    let total = 0;
    for (let i = 0; i < batches; i += 1) {
      const batch = sampleBatch(stream, batchSize, contextLength, rng);
      total += tf.tidy(() => {
        const x = tf.tensor2d(batch.x, [batchSize, contextLength], "int32");
        const y = tf.tensor2d(batch.y, [batchSize, contextLength], "int32");
        return lossFromLogits(forward(this.model, x), y).dataSync()[0];
      });
    }
    return total / batches;
  }

  /**
   * Held-out accuracy: % of **content characters** (not BOS/EOS/PAD/UNK) predicted
   * correctly on random val windows. Returns -1 when there is no val set.
   */
  evalTokenAccuracy(batches = 6): number {
    if (this.valStream.length === 0) return -1;
    const stream = this.valStream;
    const { batchSize, contextLength } = this.model.config;
    const rng = mulberry32(5678);
    let correct = 0;
    let total = 0;
    for (let i = 0; i < batches; i += 1) {
      const batch = sampleBatch(stream, batchSize, contextLength, rng);
      tf.tidy(() => {
        const x = tf.tensor2d(batch.x, [batchSize, contextLength], "int32");
        const logits = forward(this.model, x);
        const preds = tf.argMax(logits, -1).dataSync();
        for (let j = 0; j < preds.length; j += 1) {
          const target = batch.y[j];
          if (isSpecialId(target)) continue;
          total += 1;
          if (preds[j] === target) correct += 1;
        }
      });
    }
    return total > 0 ? correct / total : -1;
  }

  dispose(): void {
    this.optimizer.dispose();
  }
}

/** Share of lines whose continuation is reproduced exactly (greedy) from its prompt part. */
export function lineAccuracy(model: GptModel, lines: string[], maxLines = 40): number {
  if (lines.length === 0) return 0;
  const sample = lines.slice(0, maxLines);
  let correct = 0;
  for (const line of sample) {
    const { prompt, target } = promptAndTarget(line);
    const out = generate(model, prompt, { maxNewTokens: [...target].length + 2, temperature: 0 });
    if (out.text === target) correct += 1;
  }
  return correct / sample.length;
}
