import type { ViCurriculum } from "@math-llm/tiny-llm";
import { importWeights, serializeModel, type GptModel } from "@math-llm/tiny-llm";
import { encodeSavePayload } from "./model-save-codec.js";

export interface ModelExportMeta {
  step: number;
  loss: number;
  valLoss: number;
  valAcc: number;
  exportedAt: string;
  label?: string;
  viCurriculum?: ViCurriculum;
}

export interface SavedModelInfo {
  id: string;
  dir: string;
  path: string;
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
  viCurriculum?: ViCurriculum;
  parameterCount: number;
  path: string;
  vocabSize?: number;
  viCharset?: boolean;
}

export async function fetchExportList(): Promise<ExportListEntry[]> {
  const res = await fetch("/api/models/list");
  if (!res.ok) return [];
  const data = (await res.json()) as { exports: ExportListEntry[] };
  return data.exports ?? [];
}

/** Most recent saved VI run with the fixed charset (skip bad vi-step* arithmetic saves). */
export function findLatestViExport(exports: ExportListEntry[]): ExportListEntry | null {
  const vi = exports.filter(
    (e) =>
      (e.label === "vi" || e.id.startsWith("vi-")) &&
      (e.viCharset === true || (e.vocabSize != null && e.vocabSize > 100)),
  );
  if (vi.length === 0) return null;
  return [...vi].sort((a, b) => b.step - a.step || b.id.localeCompare(a.id))[0]!;
}

export const VI_LAST_MODEL_KEY = "tiny-vi-last-model";

/** Save to models/<name>/ on dev server (model.json + weights.bin + meta.json). */
export async function saveModelToStore(
  model: GptModel,
  meta: ModelExportMeta,
  folderName?: string,
): Promise<SavedModelInfo> {
  const { manifest, weights } = serializeModel(model);
  const payload = encodeSavePayload(meta, manifest, weights);
  const headers: Record<string, string> = {
    "Content-Type": "application/vnd.tiny-gpt-save.v1",
  };
  if (folderName?.trim()) headers["X-Save-Name"] = folderName.trim();

  const res = await fetch("/api/models/save", {
    method: "POST",
    headers,
    body: payload as BodyInit,
  });
  if (!res.ok) {
    const err = (await res.json().catch(() => ({}))) as { error?: string };
    throw new Error(
      err.error ??
        `Save failed (HTTP ${res.status}). Run npm run dev:tiny so the save API is available.`,
    );
  }
  return (await res.json()) as SavedModelInfo;
}

/** Save with explicit weights (e.g. synced from train worker on Stop). */
export async function saveModelSnapshot(
  model: GptModel,
  meta: ModelExportMeta,
  weights: Float32Array,
  folderName?: string,
): Promise<SavedModelInfo> {
  importWeights(model, weights);
  const { manifest } = serializeModel(model);
  const payload = encodeSavePayload(meta, manifest, weights);
  const headers: Record<string, string> = {
    "Content-Type": "application/vnd.tiny-gpt-save.v1",
  };
  if (folderName?.trim()) headers["X-Save-Name"] = folderName.trim();
  const res = await fetch("/api/models/save", {
    method: "POST",
    headers,
    body: payload as BodyInit,
  });
  if (!res.ok) {
    const err = (await res.json().catch(() => ({}))) as { error?: string };
    throw new Error(err.error ?? `Save failed (HTTP ${res.status})`);
  }
  return (await res.json()) as SavedModelInfo;
}

export async function saveModel(
  model: GptModel,
  meta: ModelExportMeta,
  folderName?: string,
): Promise<string> {
  const saved = await saveModelToStore(model, meta, folderName);
  return saved.id;
}

/** @deprecated use saveModel */
export async function exportModel(
  model: GptModel,
  meta: ModelExportMeta,
): Promise<string> {
  return saveModel(model, meta);
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
    list.find((e) => e.id === exportId) ??
    list.find((e) => e.version === exportId) ??
    list.find((e) => e.id.startsWith(exportId)) ??
    null;
  const id = entry?.id ?? exportId;
  const basePath = entry?.path
    ? `/${entry.path.replace(/\\/g, "/")}`
    : `/models/${encodeURIComponent(id)}`;
  const [manifest, meta, weightsRes] = await Promise.all([
    fetch(`${basePath}/model.json`).then((r) => {
      if (!r.ok) throw new Error(`model.json HTTP ${r.status}`);
      return r.json();
    }),
    fetch(`${basePath}/meta.json`)
      .then((r) => (r.ok ? r.json() : {}))
      .catch(() => ({})),
    fetch(`${basePath}/weights.bin`).then((r) => {
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
