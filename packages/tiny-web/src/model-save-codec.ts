import type { ViCurriculum } from "@math-llm/tiny-llm";
import type { ModelManifest } from "@math-llm/tiny-llm";

export interface SavePayloadMeta {
  step: number;
  loss: number;
  valLoss: number;
  valAcc: number;
  exportedAt: string;
  label?: string;
  viCurriculum?: ViCurriculum;
}

/** Bundled save body: u32 manifestLen, u32 metaLen, manifest JSON, meta JSON, float32 weights. */
export function encodeSavePayload(
  meta: SavePayloadMeta,
  manifest: ModelManifest,
  weights: Float32Array,
): Uint8Array {
  const enc = new TextEncoder();
  const manifestBytes = enc.encode(JSON.stringify(manifest));
  const metaBytes = enc.encode(JSON.stringify(meta));
  const total =
    8 + manifestBytes.length + metaBytes.length + weights.byteLength;
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  view.setUint32(0, manifestBytes.length, true);
  view.setUint32(4, metaBytes.length, true);
  let off = 8;
  out.set(manifestBytes, off);
  off += manifestBytes.length;
  out.set(metaBytes, off);
  off += metaBytes.length;
  out.set(
    new Uint8Array(weights.buffer, weights.byteOffset, weights.byteLength),
    off,
  );
  return out;
}

export function decodeSavePayload(buf: ArrayBuffer | Uint8Array): {
  meta: SavePayloadMeta;
  manifest: ModelManifest;
  weights: Float32Array;
} {
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  if (bytes.byteLength < 8) throw new Error("Save payload too short");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const manifestLen = view.getUint32(0, true);
  const metaLen = view.getUint32(4, true);
  let off = 8;
  const dec = new TextDecoder();
  const manifest = JSON.parse(
    dec.decode(bytes.subarray(off, off + manifestLen)),
  ) as ModelManifest;
  off += manifestLen;
  const meta = JSON.parse(
    dec.decode(bytes.subarray(off, off + metaLen)),
  ) as SavePayloadMeta;
  off += metaLen;
  const weightBytes = bytes.byteLength - off;
  if (weightBytes <= 0 || weightBytes % 4 !== 0) {
    throw new Error("Save payload missing weights");
  }
  const weightBuf = new ArrayBuffer(weightBytes);
  new Uint8Array(weightBuf).set(bytes.subarray(off));
  const weights = new Float32Array(weightBuf);
  return { meta, manifest, weights };
}
