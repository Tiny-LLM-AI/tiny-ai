import * as tf from "@tensorflow/tfjs";
import { afterAll, beforeAll, expect, it, vi } from "vitest";
import { DEFAULT_CONFIG } from "../src/config.js";
import { buildTokenizer } from "../src/tokenizer.js";
import type { TrainWorkerMessage, TrainWorkerProgress } from "../src/train-worker.js";

vi.mock("../src/backend.js", () => ({ initTfBackend: async () => { await tf.setBackend("cpu"); await tf.ready(); return "cpu"; } }));
let receive: (message: TrainWorkerProgress) => void = () => {};
const endpoint = {
  postMessage: (message: TrainWorkerProgress) => receive(message),
  onmessage: (_event: { data: TrainWorkerMessage }) => {},
};
beforeAll(async () => {
  vi.stubGlobal("self", endpoint);
  await import("../src/train-worker.js");
});
afterAll(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

async function run(pause: boolean, endStep = 12): Promise<{ weights: Float32Array; types: string[]; step: number }> {
  const lines = ["Tôi đang học tiếng Việt.", "Chúng ta đọc sách mỗi ngày."];
  const tokenizer = buildTokenizer(lines.join("\n"));
  const types: string[] = [];
  let requestedPause = false;
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("Worker did not finish")), 10000);
    receive = (message) => {
      types.push(message.type);
      if (message.type === "ERROR") { clearTimeout(timeout); reject(new Error(message.message)); }
      if (pause && message.type === "PROGRESS" && !requestedPause) {
        requestedPause = true;
        setTimeout(() => endpoint.onmessage({ data: { type: "PAUSE" } }), 0);
      }
      if (message.type === "PAUSED") {
        setTimeout(() => endpoint.onmessage({ data: { type: "RESUME" } }), 5);
      }
      if (message.type === "STOPPED") {
        clearTimeout(timeout);
        resolve({ weights: message.weights!, types, step: message.step });
      }
    };
    endpoint.onmessage({ data: {
      type: "TRAIN", config: { ...DEFAULT_CONFIG, vocabSize: tokenizer.vocab.length, layers: 1, dModel: 8, heads: 1, ffnSize: 16, contextLength: 8, batchSize: 2, dropout: 0 },
      vocab: tokenizer.vocab, trainLines: lines, valLines: [], weights: null, startStep: 0, endStep,
    } });
  });
}

it("Pause/Resume preserves Adam/RNG: final weights match uninterrupted training", async () => {
  let clock = 0;
  const time = vi.spyOn(performance, "now").mockImplementation(() => clock += 200);
  const before = tf.memory().numTensors;
  try {
    const uninterrupted = await run(false);
    const paused = await run(true);
    expect(paused.types).toContain("PAUSED");
    expect(paused.types).toContain("RESUMED");
    expect(paused.step).toBe(12);
    expect(paused.weights).toEqual(uninterrupted.weights);
    expect(tf.memory().numTensors).toBe(before);
    const odd = await run(false, 5);
    expect(odd.step).toBe(5);
  } finally { time.mockRestore(); }
});
