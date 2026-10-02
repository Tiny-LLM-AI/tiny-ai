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
}

export interface TrainWorkerProgress {
  type: "STARTED" | "PROGRESS" | "STOPPED" | "ERROR";
  step: number;
  loss: number;
  /** -1 when not measured this tick. */
  valLoss: number;
  valAccuracy: number;
  backend?: string;
  /** Full weights; only sent periodically for small models and always on STOPPED. */
  weights?: Float32Array | null;
  message?: string;
}

/** Above this size weights are only synced back to the UI when training stops. */
const LIVE_SYNC_PARAM_LIMIT = 2_000_000;
const TICK_MS = 150;
const VAL_EVERY_MS = 5000;
const SYNC_EVERY_MS = 2000;

let stopRequested = false;

function post(msg: TrainWorkerProgress, transfer: Transferable[] = []): void {
  (self as unknown as Worker).postMessage(msg, transfer);
}

const yieldToEvents = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

async function run(req: TrainWorkerRequest): Promise<void> {
  let step = req.startStep;
  try {
    const backend = await initTfBackend();
    const tokenizer = tokenizerFromVocab(req.vocab);
    const model = createModel(req.config, tokenizer);
    if (req.weights) importWeights(model, req.weights);
    const trainer = new Trainer(model, encodeLines(tokenizer, req.trainLines), encodeLines(tokenizer, req.valLines));
    const liveSync = countModelParams(model) <= LIVE_SYNC_PARAM_LIMIT;
    post({ type: "STARTED", step, loss: 0, valLoss: -1, valAccuracy: -1, backend });

    let lastValEval = performance.now();
    let lastSync = performance.now();
    while (!stopRequested) {
      const tickStart = performance.now();
      let loss = 0;
      let n = 0;
      do {
        loss += trainer.trainStep();
        n += 1;
        step += 1;
      } while (performance.now() - tickStart < TICK_MS && !stopRequested);

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
      post({ type: "PROGRESS", step, loss: loss / n, valLoss, valAccuracy, backend, weights }, weights ? [weights.buffer] : []);
      await yieldToEvents();
    }

    const weights = exportWeights(model);
    post({ type: "STOPPED", step, loss: 0, valLoss: -1, valAccuracy: -1, backend, weights }, [weights.buffer]);
    trainer.dispose();
    disposeModel(model);
  } catch (err) {
    post({ type: "ERROR", step, loss: 0, valLoss: -1, valAccuracy: -1, message: err instanceof Error ? err.message : String(err) });
  }
}

self.onmessage = (event: MessageEvent<TrainWorkerRequest | { type: "STOP" }>) => {
  if (event.data.type === "STOP") {
    stopRequested = true;
    return;
  }
  if (event.data.type !== "TRAIN") return;
  stopRequested = false;
  void run(event.data);
};
