import { SPECIAL_TOKENS, tokenizerFromVocab, type Tokenizer } from "@math-llm/tiny-llm";
import {
  fetchWikipediaCorpus,
  parseWikipediaTitle,
  wikipediaLangFromUrl,
} from "./wikipedia.js";

export const VI_DEFAULT_SEED = "https://vi.wikipedia.org/wiki/Tiếng_Việt";

const TONES = ["", "\u0300", "\u0301", "\u0303", "\u0309", "\u0323"];
const VOWELS = "aăâeêioôơuưy";
const PUNCT = " .,;:!?-–—()[]\"'“”‘’…/%&+=*#@";

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

/** Fixed vocabulary so weights stay valid across articles. */
export function viTokenizer(): Tokenizer {
  return tokenizerFromVocab([...SPECIAL_TOKENS, ...VI_CHARS]);
}

export function usesViVocab(tok: Tokenizer): boolean {
  const vocab = viTokenizer().vocab;
  return (
    tok.vocab.length === vocab.length && tok.vocab.every((c, i) => c === vocab[i])
  );
}

/** Drops characters outside the fixed vocabulary. */
export function cleanViLine(line: string): string {
  return [...line.normalize("NFC")]
    .filter((c) => VI_CHAR_SET.has(c))
    .join("")
    .replace(/\s+/g, " ")
    .trim();
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

  constructor(seed: string) {
    this.lang = seed.startsWith("http") ? wikipediaLangFromUrl(seed) : "vi";
    const title = parseWikipediaTitle(seed);
    this.queue.push(title);
    this.seen.add(title);
  }

  async next(onSkip?: (title: string, reason: string) => void): Promise<ViArticle> {
    while (this.queue.length > 0) {
      const title = this.queue.shift()!;
      try {
        const { corpus } = await fetchWikipediaCorpus(title, this.lang);
        const lines = corpus
          .split("\n")
          .map(cleanViLine)
          .filter((l) => l.length >= 8);
        await this.enqueueLinks(title);
        if (lines.length === 0) {
          onSkip?.(title, "no usable text");
          continue;
        }
        return { title, lines };
      } catch (err) {
        onSkip?.(title, err instanceof Error ? err.message : String(err));
      }
    }
    throw new Error("No more Wikipedia articles to fetch.");
  }

  private async enqueueLinks(title: string): Promise<void> {
    for (const link of await fetchWikipediaLinks(title, this.lang, 50)) {
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
    const res = await fetch(`https://${lang}.wikipedia.org/w/api.php?${params}`);
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
