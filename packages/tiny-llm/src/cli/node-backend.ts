import * as tf from "@tensorflow/tfjs";
import util from "node:util";

/** tfjs-node 4.x calls util.isXxx helpers that Node 23+ removed. */
function polyfillLegacyUtil(): void {
  const u = util as unknown as Record<string, unknown>;
  const helpers: Record<string, (v: unknown) => boolean> = {
    isArray: Array.isArray,
    isNullOrUndefined: (v) => v === null || v === undefined,
    isNull: (v) => v === null,
    isUndefined: (v) => v === undefined,
    isNumber: (v) => typeof v === "number",
    isString: (v) => typeof v === "string",
    isBoolean: (v) => typeof v === "boolean",
    isFunction: (v) => typeof v === "function",
    isObject: (v) => v !== null && typeof v === "object",
  };
  for (const [name, fn] of Object.entries(helpers)) {
    if (typeof u[name] !== "function") u[name] = fn;
  }
}

/** Fastest available Node backend: CUDA → native CPU → WASM → pure JS. */
export async function initNodeBackend(): Promise<string> {
  polyfillLegacyUtil();
  for (const pkg of ["@tensorflow/tfjs-node-gpu", "@tensorflow/tfjs-node"]) {
    try {
      const moduleName = pkg;
      await import(moduleName);
      if (await tf.setBackend("tensorflow")) {
        await tf.ready();
        return `tensorflow (${pkg})`;
      }
    } catch {
      // not installed or native binding unavailable for this Node version
    }
  }
  try {
    await import("@tensorflow/tfjs-backend-wasm");
    if (await tf.setBackend("wasm")) {
      await tf.ready();
      return "wasm";
    }
  } catch {
    // fall through
  }
  await tf.setBackend("cpu");
  await tf.ready();
  return "cpu (slow - install @tensorflow/tfjs-node or tfjs-node-gpu)";
}
