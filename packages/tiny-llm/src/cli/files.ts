import fs from "node:fs";
import path from "node:path";
import { parseCorpus } from "../corpus.js";
import { deserializeModel, serializeModel, type ModelManifest } from "../model-io.js";
import type { GptModel } from "../gpt.js";

/** `.txt` = one line per document; `.json` = arithmetic dataset `{ examples: [{question, answer}] }` or string[]. */
export function readCorpusFile(file: string): string[] {
  const text = fs.readFileSync(file, "utf8");
  if (!file.endsWith(".json")) return parseCorpus(text);
  const data = JSON.parse(text) as unknown;
  if (Array.isArray(data)) return data.map(String).filter(Boolean);
  const examples = (data as { examples?: { question: string; answer: string }[] }).examples ?? [];
  return examples.map((e) => `${e.question}${e.answer}`);
}

export function saveModelDir(model: GptModel, dir: string): void {
  fs.mkdirSync(dir, { recursive: true });
  const { manifest, weights } = serializeModel(model);
  fs.writeFileSync(path.join(dir, "model.json"), JSON.stringify(manifest, null, 2));
  fs.writeFileSync(path.join(dir, "weights.bin"), Buffer.from(weights.buffer, weights.byteOffset, weights.byteLength));
}

export function loadModelDir(dir: string): GptModel {
  const manifest = JSON.parse(fs.readFileSync(path.join(dir, "model.json"), "utf8")) as ModelManifest;
  const buf = fs.readFileSync(path.join(dir, "weights.bin"));
  const weights = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  return deserializeModel(manifest, weights);
}

export function parseArgs(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) out[key] = "true";
    else {
      out[key] = next;
      i += 1;
    }
  }
  return out;
}
