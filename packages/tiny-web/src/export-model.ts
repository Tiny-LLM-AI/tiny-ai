import { serializeModel, type GptModel } from "@math-llm/tiny-llm";

export interface ModelExportMeta {
  step: number;
  loss: number;
  valLoss: number;
  valAcc: number;
  exportedAt: string;
  label?: string;
}

export interface SavedExportInfo {
  id: string;
  dir: string;
  version: string;
}

export interface ExportListEntry {
  id: string;
  version: string;
  step: number;
  loss: number;
  valLoss: number;
  valAcc: number;
  exportedAt: string;
  label?: string;
  parameterCount: number;
  path: string;
}

function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i += 1) {
    crc ^= data[i]!;
    for (let j = 0; j < 8; j += 1) {
      const mask = -(crc & 1);
      crc = (crc >>> 1) ^ (0xedb88320 & mask);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function zipStore(files: Record<string, Uint8Array>): Uint8Array {
  const enc = new TextEncoder();
  const locals: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;

  for (const [path, data] of Object.entries(files)) {
    const name = enc.encode(path);
    const checksum = crc32(data);
    const local = new Uint8Array(30 + name.length + data.length);
    const lv = new DataView(local.buffer);
    lv.setUint32(0, 0x04034b50, true);
    lv.setUint16(4, 20, true);
    lv.setUint16(8, 0, true);
    lv.setUint32(14, checksum, true);
    lv.setUint32(18, data.length, true);
    lv.setUint32(22, data.length, true);
    lv.setUint16(26, name.length, true);
    local.set(name, 30);
    local.set(data, 30 + name.length);
    locals.push(local);

    const cd = new Uint8Array(46 + name.length);
    const cv = new DataView(cd.buffer);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint16(4, 20, true);
    cv.setUint16(6, 20, true);
    cv.setUint16(8, 0, true);
    cv.setUint32(16, checksum, true);
    cv.setUint32(20, data.length, true);
    cv.setUint32(24, data.length, true);
    cv.setUint16(28, name.length, true);
    cv.setUint32(42, offset, true);
    cd.set(name, 46);
    central.push(cd);
    offset += local.length;
  }

  const centralSize = central.reduce((n, b) => n + b.length, 0);
  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, central.length, true);
  ev.setUint16(10, central.length, true);
  ev.setUint32(12, centralSize, true);
  ev.setUint32(16, offset, true);

  const total = offset + centralSize + eocd.length;
  const out = new Uint8Array(total);
  let pos = 0;
  for (const part of locals) {
    out.set(part, pos);
    pos += part.length;
  }
  for (const part of central) {
    out.set(part, pos);
    pos += part.length;
  }
  out.set(eocd, pos);
  return out;
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export async function fetchExportList(): Promise<ExportListEntry[]> {
  const res = await fetch("/api/models/list");
  if (!res.ok) return [];
  const data = (await res.json()) as { exports: ExportListEntry[] };
  return data.exports ?? [];
}

/** Save to models/exports/v00N-step…/ on dev server + optional zip backup. */
export async function saveModelToStore(
  model: GptModel,
  meta: ModelExportMeta,
): Promise<SavedExportInfo> {
  const { manifest, weights } = serializeModel(model);
  const body = new Uint8Array(
    weights.buffer,
    weights.byteOffset,
    weights.byteLength,
  );
  const res = await fetch("/api/models/export", {
    method: "POST",
    headers: {
      "Content-Type": "application/octet-stream",
      "X-Export-Meta": JSON.stringify(meta),
      "X-Export-Manifest": JSON.stringify(manifest),
    },
    body: body as BodyInit,
  });
  if (!res.ok) {
    const err = (await res.json().catch(() => ({}))) as { error?: string };
    throw new Error(err.error ?? `Export HTTP ${res.status}`);
  }
  return (await res.json()) as SavedExportInfo;
}

export function downloadModelZip(
  model: GptModel,
  meta: ModelExportMeta,
  dirName: string,
): void {
  const { manifest, weights } = serializeModel(model);
  const enc = new TextEncoder();
  const prefix = `${dirName}/`;
  const files: Record<string, Uint8Array> = {
    [`${prefix}model.json`]: enc.encode(JSON.stringify(manifest, null, 2)),
    [`${prefix}weights.bin`]: new Uint8Array(
      weights.buffer,
      weights.byteOffset,
      weights.byteLength,
    ),
    [`${prefix}meta.json`]: enc.encode(JSON.stringify(meta, null, 2)),
  };
  downloadBlob(new Blob([zipStore(files) as BlobPart], { type: "application/zip" }), `${dirName}.zip`);
}

export async function exportModel(
  model: GptModel,
  meta: ModelExportMeta,
): Promise<string> {
  try {
    const saved = await saveModelToStore(model, meta);
    return saved.id;
  } catch {
    const fallback = `export-step${meta.step}-${meta.exportedAt.slice(0, 19).replace(/[:T]/g, "-")}`;
    downloadModelZip(model, meta, fallback);
    return `${fallback}.zip (download only — run npm run dev:tiny to save to models/exports/)`;
  }
}

export async function loadExportById(
  exportId: string,
): Promise<{
  manifest: import("@math-llm/tiny-llm").ModelManifest;
  weights: Float32Array;
  meta: ModelExportMeta & { version?: string };
}> {
  const list = await fetchExportList();
  const entry =
    list.find(
      (e) =>
        e.id === exportId ||
        e.version === exportId ||
        e.id.startsWith(exportId),
    ) ?? null;
  const id = entry?.id ?? exportId;
  const base = `/models/exports/${encodeURIComponent(id)}`;
  const [manifest, meta, weightsRes] = await Promise.all([
    fetch(`${base}/model.json`).then((r) => {
      if (!r.ok) throw new Error(`model.json HTTP ${r.status}`);
      return r.json();
    }),
    fetch(`${base}/meta.json`)
      .then((r) => (r.ok ? r.json() : {}))
      .catch(() => ({})),
    fetch(`${base}/weights.bin`).then((r) => {
      if (!r.ok) throw new Error(`weights.bin HTTP ${r.status}`);
      return r.arrayBuffer();
    }),
  ]);
  return {
    manifest,
    meta: meta as ModelExportMeta & { version?: string },
    weights: new Float32Array(weightsRes),
  };
}
