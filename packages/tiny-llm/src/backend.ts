import * as tf from "@tensorflow/tfjs";

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T | null> {
  return Promise.race([promise, new Promise<null>((resolve) => setTimeout(() => resolve(null), ms))]);
}

/**
 * Browser / worker backend selection. WebGL in a worker needs OffscreenCanvas and can hang,
 * so every attempt is time-boxed and CPU is the final fallback.
 */
export async function initTfBackend(): Promise<string> {
  const inWorker = typeof (globalThis as { document?: unknown }).document === "undefined";
  const canWebgl = !inWorker || typeof (globalThis as { OffscreenCanvas?: unknown }).OffscreenCanvas !== "undefined";
  const order: string[] = canWebgl ? ["webgl", "cpu"] : ["cpu"];
  for (const name of order) {
    try {
      if (name === "webgl") await import("@tensorflow/tfjs-backend-webgl");
      const ok = await withTimeout(tf.setBackend(name), 4000);
      if (ok) {
        await tf.ready();
        return tf.getBackend();
      }
    } catch {
      // try next backend
    }
  }
  await tf.setBackend("cpu");
  await tf.ready();
  return tf.getBackend();
}
