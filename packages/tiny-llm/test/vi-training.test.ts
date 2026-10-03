import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { newViCurriculum, restoreViCurriculum, foundationReady, VI_FOUNDATION_ID } from "../src/vi-curriculum.js";
import { initViCorpusHoldout, appendViTrainLines, ViWikiCrawler, viTokenizer } from "../../tiny-web/src/vi-training.js";
import { encodeSavePayload, decodeSavePayload } from "../../tiny-web/src/model-save-codec.js";
import { DEFAULT_CONFIG } from "../src/config.js";
import type { ModelManifest } from "../src/model-io.js";

const lines = Array.from({ length: 30 }, (_, i) => `Đây là câu tiếng Việt dùng để kiểm tra dữ liệu số ${i}.`);

describe("Vietnamese foundation curriculum", () => {
  it("does not count legacy Wikipedia steps as foundation progress", () => {
    expect(foundationReady(restoreViCurriculum(undefined))).toBe(false);
    expect(foundationReady(restoreViCurriculum({ step: 90000 }))).toBe(false);
    expect(foundationReady(restoreViCurriculum({ ...newViCurriculum(), corpusId: "old", foundationSteps: 90000 }))).toBe(false);
  });
  it("persists progress and waits for the configured number of steps", () => {
    const state = { ...newViCurriculum(10), foundationSteps: 9 };
    expect(foundationReady(restoreViCurriculum(state))).toBe(false);
    expect(foundationReady(restoreViCurriculum({ ...state, foundationSteps: 10 }))).toBe(true);
    expect(() => newViCurriculum(0)).toThrow();
  });
  it("deduplicates before splitting and keeps the same holdout during Wikipedia expansion", () => {
    const split = initViCorpusHoldout([...lines, ...lines]);
    expect(split.train.length + split.val.length).toBe(lines.length);
    expect(initViCorpusHoldout(lines)).toEqual(split);
    const newLine = "Một câu mới từ Wikipedia giúp mô hình học thêm ngữ cảnh.";
    const expanded = appendViTrainLines(split.train, split.val, [...lines, newLine, newLine]);
    expect(expanded).toHaveLength(split.train.length + 1);
    expect(expanded.some((line) => split.val.includes(line))).toBe(false);
    expect(split.train.every((line) => expanded.includes(line))).toBe(true);
  });
  it("rejects foreign or misleading Wikipedia hosts before fetching", () => {
    for (const url of ["https://en.wikipedia.org/wiki/Test", "https://vi.wikipedia.org.evil.test/wiki/Test", "http://vi.wikipedia.org/wiki/Test"]) {
      expect(() => new ViWikiCrawler(url)).toThrow();
    }
    expect(() => new ViWikiCrawler("https://vi.wikipedia.org/wiki/Tiếng_Việt")).not.toThrow();
  });
  it("round-trips phase metadata and Unicode vocab through the save codec", () => {
    const manifest: ModelManifest = { format: "tiny-gpt/v1", config: DEFAULT_CONFIG, vocab: viTokenizer().vocab, tensors: [], parameterCount: 2 };
    const meta = { step: 50, loss: 1, valLoss: 1, valAcc: 0.2, exportedAt: "test", viCurriculum: { ...newViCurriculum(40), foundationSteps: 40 } };
    const decoded = decodeSavePayload(encodeSavePayload(meta, manifest, Float32Array.of(1, 2)));
    expect(decoded.meta).toEqual(meta);
    expect(decoded.manifest.vocab).toEqual(manifest.vocab);
    expect(foundationReady(decoded.meta.viCurriculum!)).toBe(true);
  });
  it("downloaded foundation corpus matches its pinned fingerprint", () => {
    const text = readFileSync(new URL("../../../train-data/vi-foundation.txt", import.meta.url));
    expect(VI_FOUNDATION_ID.endsWith(":" + createHash("sha256").update(text).digest("hex"))).toBe(true);
  });
});
