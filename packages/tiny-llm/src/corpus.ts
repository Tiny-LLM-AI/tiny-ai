import { mulberry32, shuffleInPlace } from "./random.js";
import { BOS_ID, EOS_ID, encode, normalizeText, type Tokenizer } from "./tokenizer.js";

/** Every line of the corpus is one document. */
export function parseCorpus(text: string): string[] {
  return normalizeText(text)
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
}

export function totalArithmeticLines(maxNumber: number): number {
  let count = 0;
  for (let a = 0; a <= maxNumber; a += 1) {
    for (let b = 0; b <= maxNumber; b += 1) {
      count += 3;
      if (b !== 0 && a % b === 0) count += 1;
    }
  }
  return count;
}

/** Lines like `2+3=5`, `7-9=-2`, `3*4=12`, `8/2=4` for operands 0..maxNumber. */
export function generateArithmeticLines(maxNumber: number, maxLines: number, seed: number): string[] {
  const all: string[] = [];
  for (let a = 0; a <= maxNumber; a += 1) {
    for (let b = 0; b <= maxNumber; b += 1) {
      all.push(`${a}+${b}=${a + b}`);
      all.push(`${a}-${b}=${a - b}`);
      all.push(`${a}*${b}=${a * b}`);
      if (b !== 0 && a % b === 0) all.push(`${a}/${b}=${a / b}`);
    }
  }
  shuffleInPlace(all, mulberry32(seed));
  return all.slice(0, Math.max(1, maxLines));
}

export interface CorpusSplit {
  train: string[];
  val: string[];
}

export function splitCorpus(lines: string[], valRatio = 0.1, seed = 42): CorpusSplit {
  if (lines.length < 2 || valRatio <= 0) return { train: [...lines], val: [] };
  const order = lines.map((_, i) => i);
  shuffleInPlace(order, mulberry32(seed ^ 0x9e3779b9));
  const valCount = Math.max(1, Math.min(lines.length - 1, Math.floor(lines.length * valRatio)));
  const valSet = new Set(order.slice(0, valCount));
  const train: string[] = [];
  const val: string[] = [];
  lines.forEach((line, i) => (valSet.has(i) ? val : train).push(line));
  return { train, val };
}

/** One long token stream: `<BOS> line <EOS> <BOS> line <EOS> …`. */
export function encodeLines(tokenizer: Tokenizer, lines: string[]): Int32Array {
  const ids: number[] = [];
  for (const line of lines) {
    ids.push(BOS_ID, ...encode(tokenizer, line), EOS_ID);
  }
  return Int32Array.from(ids);
}

export function corpusTokenCount(lines: string[]): number {
  let n = 0;
  for (const l of lines) n += [...l].length + 2;
  return n;
}

/**
 * Splits a line into prompt + expected continuation for accuracy checks:
 * `2+3=5` → prompt `2+3=`, target `5`. Lines without `=` are split in half.
 */
export function promptAndTarget(line: string): { prompt: string; target: string } {
  const chars = [...line];
  const eq = chars.indexOf("=");
  const cut = eq >= 0 ? eq + 1 : Math.max(1, Math.floor(chars.length / 2));
  return { prompt: chars.slice(0, cut).join(""), target: chars.slice(cut).join("") };
}
