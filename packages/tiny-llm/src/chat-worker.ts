import { initTfBackend } from "./backend.js";
import type { GptConfig } from "./config.js";
import { generateAsync, type GenerateResult } from "./generate.js";
import { createModel, disposeModel, type GptModel } from "./gpt.js";
import { importWeights } from "./model-io.js";
import { tokenizerFromVocab } from "./tokenizer.js";

export interface ChatWorkerInit {
  type: "INIT";
  config: GptConfig;
  vocab: string[];
  weights: Float32Array | null;
}

export interface ChatWorkerSync {
  type: "SYNC";
  weights: Float32Array;
}

export interface ChatWorkerGenerate {
  type: "GENERATE";
  id: number;
  prompt: string;
  temperature: number;
  topK: number;
  maxNewTokens: number;
  seed: number;
}

export type ChatWorkerRequest = ChatWorkerInit | ChatWorkerSync | ChatWorkerGenerate | { type: "STOP" };

export type ChatWorkerMessage =
  | { type: "READY"; backend: string }
  | { type: "STEP"; id: number; partial: string; stepIndex: number }
  | { type: "DONE"; id: number; result: GenerateResult }
  | { type: "ERROR"; id?: number; message: string };

let model: GptModel | null = null;
let generating = false;
let stopRequested = false;

const yieldToEvents = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

function post(msg: ChatWorkerMessage): void {
  (self as unknown as Worker).postMessage(msg);
}

async function init(req: ChatWorkerInit): Promise<void> {
  if (model) {
    disposeModel(model);
    model = null;
  }
  const backend = await initTfBackend();
  const tokenizer = tokenizerFromVocab(req.vocab);
  model = createModel(req.config, tokenizer);
  if (req.weights) importWeights(model, req.weights);
  post({ type: "READY", backend });
}

function sync(weights: Float32Array): void {
  if (!model) return;
  importWeights(model, weights);
}

async function runGenerate(req: ChatWorkerGenerate): Promise<void> {
  if (!model) {
    post({ type: "ERROR", id: req.id, message: "Chat worker not initialized" });
    return;
  }
  if (generating) {
    post({ type: "ERROR", id: req.id, message: "Chat worker is busy" });
    return;
  }
  generating = true;
  stopRequested = false;
  try {
    const result = await generateAsync(model, req.prompt, {
      temperature: req.temperature,
      topK: req.topK,
      maxNewTokens: req.maxNewTokens,
      seed: req.seed,
      onStep: async (_step, partial, stepIndex) => {
        if (stopRequested) throw new Error("stopped");
        post({ type: "STEP", id: req.id, partial, stepIndex });
        await yieldToEvents();
      },
    });
    if (!stopRequested) post({ type: "DONE", id: req.id, result });
  } catch (err) {
    if (!stopRequested) {
      post({ type: "ERROR", id: req.id, message: err instanceof Error ? err.message : String(err) });
    }
  } finally {
    generating = false;
    stopRequested = false;
  }
}

self.onmessage = (event: MessageEvent<ChatWorkerRequest>) => {
  const data = event.data;
  if (data.type === "STOP") {
    stopRequested = true;
    return;
  }
  if (data.type === "INIT") {
    void init(data);
    return;
  }
  if (data.type === "SYNC") {
    sync(data.weights);
    return;
  }
  if (data.type === "GENERATE") {
    void runGenerate(data);
  }
};
