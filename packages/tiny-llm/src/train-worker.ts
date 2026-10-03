import * as tf from "@tensorflow/tfjs";
import { initTfBackend } from "./backend.js";
import type { GptConfig } from "./config.js";
import { encodeLines } from "./corpus.js";
import { countModelParams, createModel, disposeModel } from "./gpt.js";
import { exportWeights, importWeights } from "./model-io.js";
import { tokenizerFromVocab } from "./tokenizer.js";
import { Trainer } from "./trainer.js";

export interface TrainWorkerRequest {
  type: "TRAIN";
  config: GptConfig;
  vocab: string[];
  trainLines: string[];
  valLines: string[];
  /** Continue from these weights (canonical order); null = fresh init. */
  weights: Float32Array | null;
  startStep: number;
  /** Optional absolute stopping step, used for the foundation phase. */
  endStep?: number;
}

export interface TrainWorkerUpdateCorpus {
  type: "UPDATE_CORPUS";
  trainLines: string[];
}

export type TrainWorkerMessage =
  | TrainWorkerRequest
  | TrainWorkerUpdateCorpus
  | { type: "STOP" | "PAUSE" | "RESUME" };

export interface TrainWorkerProgress {
  type: "STARTED" | "PROGRESS" | "PAUSED" | "RESUMED" | "STOPPED" | "ERROR";
  step: number;
  loss: number;
  /** -1 when not measured this tick. */
  valLoss: number;
  valAccuracy: number;
  backend?: string;
  trainLines?: number;
  /** Full weights; only sent periodically for small models and always on STOPPED. */
  weights?: Float32Array | null;
  message?: string;
}

/** Above this size weights are only synced back to the UI when training stops. */
const LIVE_SYNC_PARAM_LIMIT = 2_000_000;
const TICK_MS = 150;
/** Steps queued on the device before one async loss read-back. */
const STEPS_PER_READ = 4;
const VAL_EVERY_MS = 5000;
const SYNC_EVERY_MS = 2000;

let stopRequested = false;
let pauseRequested = false;
let wakePaused: (() => void) | null = null;
let running = false;
let pendingTrainLines: string[] | null = null;

function post(msg: TrainWorkerProgress, transfer: Transferable[] = []): void {
  (self as unknown as Worker).postMessage(msg, transfer);
}

const yieldToEvents = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

function applyPendingTrain(
  trainer: Trainer,
  tokenizer: ReturnType<typeof tokenizerFromVocab>,
  currentCount: number,
): number {
  if (!pendingTrainLines) return currentCount;
  trainer.setTrainLines(tokenizer, pendingTrainLines);
  const n = pendingTrainLines.length;
  pendingTrainLines = null;
  return n;
}

async function run(req: TrainWorkerRequest): Promise<void> {
  let step = req.startStep;
  let model: ReturnType<typeof createModel> | null = null;
  let trainer: Trainer | null = null;
  let lastLoss = -1;
  try {
    const backend = await initTfBackend();
    const tokenizer = tokenizerFromVocab(req.vocab);
    model = createModel(req.config, tokenizer);
    if (req.weights) importWeights(model, req.weights);
    trainer = new Trainer(
      model,
      encodeLines(tokenizer, req.trainLines),
      encodeLines(tokenizer, req.valLines),
    );
    const liveSync = countModelParams(model) <= LIVE_SYNC_PARAM_LIMIT;
    post({
      type: "STARTED",
      step,
      loss: 0,
      valLoss: -1,
      valAccuracy: -1,
      backend,
      trainLines: req.trainLines.length,
    });

    let trainLineCount = req.trainLines.length;
    let lastValEval = performance.now();
    let lastSync = performance.now();
    while (!stopRequested && (req.endStep == null || step < req.endStep)) {
      if (pauseRequested) {
        const weights = exportWeights(model);
        post({ type: "PAUSED", step, loss: lastLoss, valLoss: -1, valAccuracy: -1, weights }, [weights.buffer]);
        await new Promise<void>((resolve) => { wakePaused = resolve; });
        wakePaused = null;
        if (stopRequested) break;
        post({ type: "RESUMED", step, loss: lastLoss, valLoss: -1, valAccuracy: -1 });
      }
      trainLineCount = applyPendingTrain(trainer, tokenizer, trainLineCount);
      const tickStart = performance.now();
      let loss = 0;
      let n = 0;
      do {
        trainLineCount = applyPendingTrain(trainer, tokenizer, trainLineCount);
        const pending: tf.Scalar[] = [];
        for (let i = 0; i < STEPS_PER_READ && !stopRequested && (req.endStep == null || step < req.endStep); i += 1) {
          pending.push(trainer.trainStepTensor());
          step += 1;
        }
        const sum = tf.addN(pending);
        loss += (await sum.data())[0]!;
        n += pending.length;
        sum.dispose();
        pending.forEach((t) => t.dispose());
      } while (performance.now() - tickStart < TICK_MS && !stopRequested && (req.endStep == null || step < req.endStep));

      lastLoss = loss / n;
      if (!Number.isFinite(lastLoss)) throw new Error("Training loss became non-finite. Reduce the learning rate.");
      const now = performance.now();
      let valLoss = -1;
      let valAccuracy = -1;
      if (now - lastValEval > VAL_EVERY_MS && req.valLines.length > 0) {
        valLoss = trainer.evalLoss(4);
        valAccuracy = trainer.evalTokenAccuracy(8);
        lastValEval = performance.now();
      }
      let weights: Float32Array | null = null;
      if (liveSync && now - lastSync > SYNC_EVERY_MS) {
        weights = exportWeights(model);
        lastSync = now;
      }
      post(
        {
          type: "PROGRESS",
          step,
          loss: loss / n,
          valLoss,
          valAccuracy,
          backend,
          trainLines: pendingTrainLines?.length ?? trainLineCount,
          weights,
        },
        weights ? [weights.buffer] : [],
      );
      await yieldToEvents();
    }

    const weights = exportWeights(model);
    post(
      {
        type: "STOPPED",
        step,
        loss: lastLoss,
        valLoss: trainer.evalLoss(1),
        valAccuracy: trainer.evalTokenAccuracy(2),
        backend,
        trainLines: trainLineCount,
        weights,
      },
      [weights.buffer],
    );
  } catch (err) {
    post({
      type: "ERROR",
      step,
      loss: 0,
      valLoss: -1,
      valAccuracy: -1,
      message: err instanceof Error ? err.message : String(err),
    });
  } finally {
    trainer?.dispose();
    if (model) disposeModel(model);
    running = false;
    wakePaused = null;
  }
}

self.onmessage = (event: MessageEvent<TrainWorkerMessage>) => {
  if (event.data.type === "STOP") {
    stopRequested = true;
    wakePaused?.();
    return;
  }
  if (event.data.type === "PAUSE") { pauseRequested = true; return; }
  if (event.data.type === "RESUME") { pauseRequested = false; wakePaused?.(); return; }
  if (event.data.type === "UPDATE_CORPUS") {
    pendingTrainLines = event.data.trainLines;
    return;
  }
  if (event.data.type !== "TRAIN" || running) return;
  running = true;
  pauseRequested = false;
  stopRequested = false;
  pendingTrainLines = null;
  void run(event.data);
};
