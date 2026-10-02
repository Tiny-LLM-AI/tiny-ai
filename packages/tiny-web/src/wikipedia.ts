/** Parse a Wikipedia article URL or plain title into an article title. */
export function parseWikipediaTitle(urlOrTitle: string): string {
  const trimmed = urlOrTitle.trim();
  if (!trimmed) throw new Error("Enter a Wikipedia URL or article title.");
  if (!trimmed.startsWith("http")) return trimmed.replace(/_/g, " ");
  let u: URL;
  try {
    u = new URL(trimmed);
  } catch {
    throw new Error("Invalid URL.");
  }
  if (!u.hostname.includes("wikipedia.org")) throw new Error("URL must be from wikipedia.org");
  const lang = u.hostname.split(".")[0];
  if (!lang) throw new Error("Could not detect Wikipedia language from URL.");
  const m = u.pathname.match(/\/wiki\/(.+)/);
  if (!m) throw new Error("Paste a link to an article, e.g. https://vi.wikipedia.org/wiki/Hà_Nội");
  return decodeURIComponent(m[1]).replace(/_/g, " ");
}

export function wikipediaLangFromUrl(urlOrTitle: string): string {
  const trimmed = urlOrTitle.trim();
  if (!trimmed.startsWith("http")) return "vi";
  try {
    return new URL(trimmed).hostname.split(".")[0] || "vi";
  } catch {
    return "vi";
  }
}

/** Fetch plain text of one Wikipedia article (browser or Node). */
export async function fetchWikipediaArticle(titleOrUrl: string, lang = "vi"): Promise<string> {
  const title = parseWikipediaTitle(titleOrUrl);
  const wikiLang = titleOrUrl.trim().startsWith("http") ? wikipediaLangFromUrl(titleOrUrl) : lang;
  const url =
    `https://${wikiLang}.wikipedia.org/w/api.php?` +
    `action=query&prop=extracts&explaintext=1&redirects=1&titles=${encodeURIComponent(title)}&format=json&origin=*`;
  // Custom headers in the browser trigger a CORS preflight; Node needs a User-Agent or Wikipedia rejects it.
  const inBrowser = typeof window !== "undefined";
  const res = await fetch(url, inBrowser ? {} : { headers: { "User-Agent": "TinyGPT-LLM-mini/2.0 (educational)" } });
  if (!res.ok) throw new Error(`Wikipedia HTTP ${res.status}`);
  const data = (await res.json()) as {
    query?: { pages?: Record<string, { extract?: string; missing?: string; title?: string }> };
    error?: { info?: string };
  };
  if (data.error?.info) throw new Error(data.error.info);
  const pages = data.query?.pages;
  if (!pages) throw new Error("Empty response from Wikipedia.");
  const page = Object.values(pages)[0];
  if (!page || page.missing) throw new Error(`Article not found: ${title}`);
  const text = page.extract?.trim();
  if (!text) throw new Error(`No text in article: ${page.title ?? title}`);
  return text;
}

/** Turn Wikipedia plain text into one-line-per-sentence corpus lines. */
export function wikipediaToCorpus(text: string): string {
  const lines: string[] = [];
  for (const block of text.split(/\n+/)) {
    const p = block.trim();
    if (!p || p === "=" || /^=+ .+ =+$/.test(p)) continue;
    if (p.length <= 200) {
      lines.push(p);
      continue;
    }
    for (const s of p.split(/(?<=[.!?…])\s+/)) {
      const t = s.trim();
      if (t.length >= 8) lines.push(t);
    }
  }
  return lines.join("\n");
}

export async function fetchWikipediaCorpus(titleOrUrl: string, lang = "vi"): Promise<{ title: string; corpus: string; lineCount: number }> {
  const title = parseWikipediaTitle(titleOrUrl);
  const raw = await fetchWikipediaArticle(titleOrUrl, lang);
  const corpus = wikipediaToCorpus(raw);
  const lineCount = corpus.split("\n").filter(Boolean).length;
  if (lineCount === 0) throw new Error("Article had no usable sentences.");
  return { title, corpus, lineCount };
}
