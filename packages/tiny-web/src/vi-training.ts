import {
  normalizeConfig,
  SPECIAL_TOKENS,
  splitCorpus,
  tokenizerFromVocab,
  type GptConfig,
  type Tokenizer,
} from "@math-llm/tiny-llm";
import {
  fetchWikipediaCorpus,
  parseWikipediaTitle,
} from "./wikipedia.js";

export const VI_DEFAULT_SEED = "https://vi.wikipedia.org/wiki/Tiếng_Việt";

/** Fixed val ratio + seed for reproducible hold-out across runs. */
export const VI_VAL_RATIO = 0.1;
export const VI_VAL_SEED = 42;

export type ViModelShape = Omit<GptConfig, "vocabSize">;

/** Vietnamese training changes the charset, never the selected architecture/hyperparameters. */
export function viTrainingConfig(shape: ViModelShape): GptConfig {
  return normalizeConfig({ ...shape, vocabSize: viVocabSize() });
}

/** Minimum sentence length after cleaning (ViWikiBench uses 150 for paragraphs; sentences are shorter). */
export const VI_MIN_LINE_LEN = 24;

const TONES = ["", "\u0300", "\u0301", "\u0303", "\u0309", "\u0323"];
const VOWELS = "aăâeêioôơuưy";
const PUNCT = " .,;:!?-–—()[]\"'“”‘’…/%&+=*#@";

/** Vietnamese letters with tone marks or đ — filters out pure-ASCII / foreign snippets. */
const VI_DIACRITIC_RE =
  /[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]/i;

function buildViCharset(): string[] {
  const chars = new Set<string>();
  for (const c of "abcdefghijklmnopqrstuvwxyzđ0123456789") chars.add(c);
  for (const v of VOWELS)
    for (const t of TONES) chars.add(`${v}${t}`.normalize("NFC"));
  for (const c of [...chars]) chars.add(c.toUpperCase());
  for (const c of PUNCT) chars.add(c);
  return [...chars].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

const VI_CHARS = buildViCharset();
const VI_CHAR_SET = new Set(VI_CHARS);

/** Fixed vocabulary: a–z, đ, 0–9, tones, punctuation — stable across all wiki articles. */
export function viTokenizer(): Tokenizer {
  return tokenizerFromVocab([...SPECIAL_TOKENS, ...VI_CHARS]);
}

export function viVocabSize(): number {
  return viTokenizer().vocab.length;
}

export function usesViVocab(tok: Tokenizer): boolean {
  const vocab = viTokenizer().vocab;
  return (
    tok.vocab.length === vocab.length && tok.vocab.every((c, i) => c === vocab[i])
  );
}

/** Saved manifest uses the fixed Vietnamese charset (not arithmetic / word vocab). */
export function isViManifest(manifest: { vocab: string[] }): boolean {
  const expected = viTokenizer().vocab;
  return (
    manifest.vocab.length === expected.length &&
    manifest.vocab.every((c, i) => c === expected[i])
  );
}

/** Drops characters outside the fixed Vietnamese charset. */
export function cleanViLine(line: string): string {
  return [...line.normalize("NFC")]
    .filter((c) => VI_CHAR_SET.has(c))
    .join("")
    .replace(/\s+/g, " ")
    .trim();
}

export function hasViDiacritic(line: string): boolean {
  return VI_DIACRITIC_RE.test(line);
}

/** Keep lines with valid Vietnamese orthography (NFC, charset, min length, has diacritics). */
export function filterViLines(lines: string[]): string[] {
  return lines
    .map(cleanViLine)
    .filter((l) => l.length >= VI_MIN_LINE_LEN && hasViDiacritic(l));
}

export interface ViCorpusSplit {
  train: string[];
  val: string[];
}

/**
 * One-time hold-out split for validation. New articles are appended only to `train`.
 * `seed` fixes which lines stay in val for the whole run.
 */
export function initViCorpusHoldout(
  lines: string[],
  valRatio = VI_VAL_RATIO,
  seed = VI_VAL_SEED,
): ViCorpusSplit {
  const filtered = [...new Set(filterViLines(lines))];
  if (filtered.length < 8) {
    return { train: filtered, val: [] };
  }
  const split = splitCorpus(filtered, valRatio, seed);
  if (split.val.length === 0 && filtered.length >= 2) {
    split.val.push(filtered[0]!);
    split.train = filtered.slice(1);
  }
  return split;
}

/** Append new wiki lines to train only; skip duplicates and val hold-out lines. */
export function appendViTrainLines(
  train: string[],
  val: string[],
  newLines: string[],
): string[] {
  const valSet = new Set(val);
  const seen = new Set(train);
  const out = [...train];
  for (const line of filterViLines(newLines)) {
    if (valSet.has(line) || seen.has(line)) continue;
    seen.add(line);
    out.push(line);
  }
  return out;
}

export interface ViArticle {
  title: string;
  lines: string[];
}

/** Walks vi.wikipedia by following links from the seed article. */
export class ViWikiCrawler {
  private queue: string[] = [];
  private seen = new Set<string>();
  private readonly lang: string;
  private readonly controller = new AbortController();

  constructor(seed: string) {
    if (/^https?:/i.test(seed.trim())) {
      const url = new URL(seed.trim());
      if (url.protocol !== "https:" || url.hostname !== "vi.wikipedia.org") {
        throw new Error("Chỉ nhận link https://vi.wikipedia.org/wiki/... bằng tiếng Việt.");
      }
    }
    this.lang = "vi";
    const title = parseWikipediaTitle(seed);
    this.queue.push(title);
    this.seen.add(title);
  }

  stop(): void { this.controller.abort(); }

  async next(onSkip?: (title: string, reason: string) => void): Promise<ViArticle> {
    let lastError = "No more Wikipedia articles to fetch.";
    while (this.queue.length > 0) {
      this.controller.signal.throwIfAborted();
      const title = this.queue.shift()!;
      try {
        const { corpus } = await fetchWikipediaCorpus(title, this.lang, this.controller.signal);
        const lines = filterViLines(corpus.split("\n"));
        await this.enqueueLinks(title);
        if (lines.length === 0) {
          onSkip?.(title, "no usable text");
          continue;
        }
        return { title, lines };
      } catch (err) {
        this.controller.signal.throwIfAborted();
        lastError = err instanceof Error ? err.message : String(err);
        onSkip?.(title, lastError);
      }
    }
    throw new Error(lastError);
  }

  private async enqueueLinks(title: string): Promise<void> {
    for (const link of await fetchWikipediaLinks(title, this.lang, 50, this.controller.signal)) {
      if (this.seen.has(link)) continue;
      this.seen.add(link);
      this.queue.push(link);
    }
  }

  get queued(): number {
    return this.queue.length;
  }
}

async function fetchWikipediaLinks(
  title: string,
  lang: string,
  limit: number,
  signal: AbortSignal,
): Promise<string[]> {
  const params = new URLSearchParams({
    action: "query",
    prop: "links",
    titles: title,
    plnamespace: "0",
    pllimit: String(limit),
    redirects: "1",
    format: "json",
    origin: "*",
  });
  try {
    const res = await fetch(`https://${lang}.wikipedia.org/w/api.php?${params}`, { signal: AbortSignal.any([signal, AbortSignal.timeout(15000)]) });
    if (!res.ok) return [];
    const data = (await res.json()) as {
      query?: { pages?: Record<string, { links?: { title: string }[] }> };
    };
    const page = Object.values(data.query?.pages ?? {})[0];
    return (page?.links ?? []).map((l) => l.title);
  } catch {
    return [];
  }
}
