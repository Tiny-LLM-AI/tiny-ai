import { randomUUID } from "node:crypto";
import type { ViCurriculum } from "./vi-curriculum.js";
import fs from "node:fs";
import path from "node:path";
import type { GptModel } from "./gpt.js";
import { serializeModel, type ModelManifest } from "./model-io.js";

export const EXPORTS_DIR = "models/exports";
export const MODELS_DIR = "models";

export interface ModelExportMeta {
  version: string;
  step: number;
  loss: number;
  valLoss: number;
  valAcc: number;
  exportedAt: string;
  label?: string;
  viCurriculum?: ViCurriculum;
}

export interface SavedModelMeta {
  step: number;
  loss: number;
  valLoss: number;
  valAcc: number;
  exportedAt: string;
  label?: string;
  viCurriculum?: ViCurriculum;
}

export interface ExportIndexEntry {
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

export interface ExportIndex {
  updatedAt: string;
  exports: ExportIndexEntry[];
}

function stamp(): string {
  return new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
}

export function repoRoot(from = process.cwd()): string {
  let dir = from;
  for (let i = 0; i < 8; i += 1) {
    if (fs.existsSync(path.join(dir, "package.json"))) {
      const pkg = JSON.parse(
        fs.readFileSync(path.join(dir, "package.json"), "utf8"),
      ) as { workspaces?: string[] };
      if (pkg.workspaces?.includes("packages/*")) return dir;
    }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return from;
}

export function exportsRoot(baseDir = repoRoot()): string {
  return path.join(baseDir, EXPORTS_DIR);
}

export function nextExportVersion(exportsDir: string): string {
  if (!fs.existsSync(exportsDir)) return "v001";
  const nums = fs
    .readdirSync(exportsDir, { withFileTypes: true })
    .filter((d) => d.isDirectory() && /^v\d{3}/.test(d.name))
    .map((d) => Number(d.name.slice(1, 4)))
    .filter((n) => Number.isFinite(n));
  const next = (nums.length ? Math.max(...nums) : 0) + 1;
  return `v${String(next).padStart(3, "0")}`;
}

export function buildExportId(
  version: string,
  step: number,
  label?: string,
): string {
  const safeLabel = label
    ? `-${label.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-|-$/g, "")}`
    : "";
  return `${version}-step${step}${safeLabel}-${stamp()}`;
}

export function indexPath(exportsDir: string): string {
  return path.join(exportsDir, "index.json");
}

export function readExportIndex(exportsDir: string): ExportIndex {
  const file = indexPath(exportsDir);
  if (!fs.existsSync(file)) return { updatedAt: "", exports: [] };
  return JSON.parse(fs.readFileSync(file, "utf8")) as ExportIndex;
}

export function writeExportIndex(exportsDir: string, index: ExportIndex): void {
  fs.mkdirSync(exportsDir, { recursive: true });
  index.updatedAt = new Date().toISOString();
  fs.writeFileSync(indexPath(exportsDir), JSON.stringify(index, null, 2));
}

export function upsertExportIndex(
  exportsDir: string,
  entry: ExportIndexEntry,
): void {
  const index = readExportIndex(exportsDir);
  index.exports = index.exports.filter((e) => e.id !== entry.id);
  index.exports.unshift(entry);
  writeExportIndex(exportsDir, index);
}

export interface SaveExportInput {
  manifest: ModelManifest;
  weights: Float32Array;
  meta: Omit<SavedModelMeta, never> & { label?: string };
  label?: string;
  viCurriculum?: ViCurriculum;
}

function writeModelFiles(
  dir: string,
  input: SaveExportInput,
  meta: SavedModelMeta & { version?: string },
): void {
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(
    path.join(dir, "model.json"),
    JSON.stringify(input.manifest, null, 2),
  );
  fs.writeFileSync(
    path.join(dir, "weights.bin"),
    Buffer.from(
      input.weights.buffer,
      input.weights.byteOffset,
      input.weights.byteLength,
    ),
  );
  fs.writeFileSync(path.join(dir, "meta.json"), JSON.stringify(meta, null, 2));
}

export function defaultSaveFolderName(step: number, label = "vi"): string {
  return `${label}-step${step}-${stamp()}-${randomUUID().slice(0, 8)}`;
}

/** Save like CLI `--out models/my-model`: model.json + weights.bin in models/<name>/ */
export function saveModelToFolder(
  input: SaveExportInput,
  folderName: string,
  baseDir = repoRoot(),
): { id: string; dir: string; path: string } {
  const dir = path.join(baseDir, MODELS_DIR, folderName);
  writeModelFiles(dir, input, {
    ...input.meta,
    label: input.label ?? input.meta.label,
  });
  return { id: folderName, dir, path: path.relative(baseDir, dir) };
}

export function saveVersionedExport(
  input: SaveExportInput,
  baseDir = repoRoot(),
): { id: string; dir: string; version: string } {
  const root = exportsRoot(baseDir);
  const version = nextExportVersion(root);
  const id = buildExportId(version, input.meta.step, input.label ?? input.meta.label);
  const dir = path.join(root, id);
  fs.mkdirSync(dir, { recursive: true });

  const meta: ModelExportMeta = {
    ...input.meta,
    version,
    label: input.label ?? input.meta.label,
  };

  writeModelFiles(dir, input, meta);
  fs.writeFileSync(
    path.join(dir, "README.txt"),
    [
      `Tiny GPT export ${id}`,
      "",
      "Test (chat):",
      `  npm run start:model -- ${id} --mode chat`,
      "",
      "Resume training:",
      `  npm run start:model -- ${id} --mode train --steps 5000`,
      "",
      "Open in browser UI:",
      `  npm run start:model -- ${id} --mode web`,
    ].join("\n"),
  );

  upsertExportIndex(root, {
    id,
    version,
    step: meta.step,
    loss: meta.loss,
    valLoss: meta.valLoss,
    valAcc: meta.valAcc,
    exportedAt: meta.exportedAt,
    label: meta.label,
    parameterCount: input.manifest.parameterCount,
    path: path.relative(baseDir, dir),
  });

  return { id, dir, version };
}

export function saveModelVersioned(
  model: GptModel,
  meta: Omit<ModelExportMeta, "version"> & { label?: string },
  baseDir = repoRoot(),
  label?: string,
): { id: string; dir: string; version: string } {
  const { manifest, weights } = serializeModel(model);
  return saveVersionedExport({ manifest, weights, meta, label }, baseDir);
}

export function listSavedModels(baseDir = repoRoot()): ExportIndexEntry[] {
  const root = path.join(baseDir, MODELS_DIR);
  if (!fs.existsSync(root)) return [];
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter(
      (d) =>
        d.isDirectory() &&
        d.name !== "exports" &&
        fs.existsSync(path.join(root, d.name, "model.json")),
    )
    .map((d) => {
      const dir = path.join(root, d.name);
      const manifest = JSON.parse(
        fs.readFileSync(path.join(dir, "model.json"), "utf8"),
      ) as ModelManifest;
      let meta: Partial<SavedModelMeta> = {};
      const metaFile = path.join(dir, "meta.json");
      if (fs.existsSync(metaFile)) {
        meta = JSON.parse(fs.readFileSync(metaFile, "utf8")) as SavedModelMeta;
      }
      const vocabSize = manifest.vocab.length;
      return {
        id: d.name,
        version: d.name,
        step: meta.step ?? 0,
        loss: meta.loss ?? -1,
        valLoss: meta.valLoss ?? -1,
        valAcc: meta.valAcc ?? -1,
        exportedAt: meta.exportedAt ?? "",
        label: meta.label,
        parameterCount: manifest.parameterCount,
        path: path.relative(baseDir, dir),
        vocabSize,
        viCharset: vocabSize > 100 && manifest.vocab.includes("đ"),
      } satisfies ExportIndexEntry;
    })
    .sort((a, b) => b.id.localeCompare(a.id));
}

export function resolveModelDir(idOrPrefix: string, baseDir = repoRoot()): string {
  const modelsRoot = path.join(baseDir, MODELS_DIR);
  const direct = path.join(modelsRoot, idOrPrefix);
  if (fs.existsSync(path.join(direct, "model.json"))) return direct;

  if (fs.existsSync(modelsRoot)) {
    const dirs = fs
      .readdirSync(modelsRoot, { withFileTypes: true })
      .filter((d) => d.isDirectory() && d.name !== "exports")
      .map((d) => d.name)
      .sort()
      .reverse();
    const match = dirs.find(
      (name) => name === idOrPrefix || name.startsWith(idOrPrefix),
    );
    if (match && fs.existsSync(path.join(modelsRoot, match, "model.json"))) {
      return path.join(modelsRoot, match);
    }
  }

  const exportRoot = exportsRoot(baseDir);
  const exportDirect = path.join(exportRoot, idOrPrefix);
  if (fs.existsSync(path.join(exportDirect, "model.json"))) return exportDirect;

  const index = readExportIndex(exportRoot);
  const exact = index.exports.find(
    (e) => e.id === idOrPrefix || e.version === idOrPrefix,
  );
  if (exact) return path.join(baseDir, exact.path);

  const prefix = index.exports.find((e) => e.id.startsWith(idOrPrefix));
  if (prefix) return path.join(baseDir, prefix.path);

  if (fs.existsSync(exportRoot)) {
    const dirs = fs
      .readdirSync(exportRoot, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .map((d) => d.name)
      .sort()
      .reverse();
    const match = dirs.find(
      (name) => name === idOrPrefix || name.startsWith(idOrPrefix),
    );
    if (match && fs.existsSync(path.join(exportRoot, match, "model.json"))) {
      return path.join(exportRoot, match);
    }
  }

  throw new Error(`Model not found: ${idOrPrefix}. Run: npm run models:list`);
}

/** @deprecated use resolveModelDir */
export function resolveExportDir(idOrPrefix: string, baseDir = repoRoot()): string {
  return resolveModelDir(idOrPrefix, baseDir);
}

export function listExports(baseDir = repoRoot()): ExportIndexEntry[] {
  const root = exportsRoot(baseDir);
  const fromIndex = readExportIndex(root).exports;
  if (fromIndex.length > 0) return fromIndex;

  if (!fs.existsSync(root)) return [];
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter(
      (d) =>
        d.isDirectory() && fs.existsSync(path.join(root, d.name, "model.json")),
    )
    .map((d) => {
      const dir = path.join(root, d.name);
      const manifest = JSON.parse(
        fs.readFileSync(path.join(dir, "model.json"), "utf8"),
      ) as ModelManifest;
      let meta: Partial<ModelExportMeta> = {};
      const metaFile = path.join(dir, "meta.json");
      if (fs.existsSync(metaFile)) {
        meta = JSON.parse(fs.readFileSync(metaFile, "utf8")) as ModelExportMeta;
      }
      return {
        id: d.name,
        version: meta.version ?? d.name.slice(0, 4),
        step: meta.step ?? 0,
        loss: meta.loss ?? -1,
        valLoss: meta.valLoss ?? -1,
        valAcc: meta.valAcc ?? -1,
        exportedAt: meta.exportedAt ?? "",
        label: meta.label,
        parameterCount: manifest.parameterCount,
        path: path.relative(baseDir, dir),
      } satisfies ExportIndexEntry;
    })
    .sort((a, b) => b.id.localeCompare(a.id));
}

export function readExportMeta(dir: string): ModelExportMeta | null {
  const file = path.join(dir, "meta.json");
  if (!fs.existsSync(file)) return null;
  return JSON.parse(fs.readFileSync(file, "utf8")) as ModelExportMeta;
}
