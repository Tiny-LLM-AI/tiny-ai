#!/usr/bin/env node
/**
 * Crawl vi.wikipedia into train-data/vi-wikipedia.txt (one line per sentence).
 * Usage: npm run fetch:vi-corpus [--articles 30]
 */
import fs from "node:fs";
import path from "node:path";

const DEFAULT_SEED = "Tiếng Việt";
const DEFAULT_ARTICLES = 30;
const MIN_LINE = 24;
const VI_DIACRITIC =
  /[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]/i;

const args = process.argv.slice(2);
let maxArticles = DEFAULT_ARTICLES;
for (let i = 0; i < args.length; i += 1) {
  if (args[i] === "--articles" && args[i + 1]) {
    maxArticles = Math.max(1, Number(args[++i]) || DEFAULT_ARTICLES);
  }
}

function repoRoot() {
  let dir = process.cwd();
  for (let i = 0; i < 8; i += 1) {
    if (fs.existsSync(path.join(dir, "package.json"))) {
      const pkg = JSON.parse(fs.readFileSync(path.join(dir, "package.json"), "utf8"));
      if (pkg.workspaces?.includes("packages/*")) return dir;
    }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return process.cwd();
}

async function fetchArticle(title) {
  const url =
    "https://vi.wikipedia.org/w/api.php?" +
    new URLSearchParams({
      action: "query",
      prop: "extracts",
      explaintext: "1",
      redirects: "1",
      titles: title,
      format: "json",
    });
  const res = await fetch(url, {
    headers: { "User-Agent": "TinyGPT-LLM-mini/2.0 (educational corpus fetch)" },
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  const page = Object.values(data.query?.pages ?? {})[0];
  if (!page?.extract || page.missing) throw new Error(`Missing: ${title}`);
  return page.extract.trim();
}

async function fetchLinks(title, limit = 40) {
  const url =
    "https://vi.wikipedia.org/w/api.php?" +
    new URLSearchParams({
      action: "query",
      prop: "links",
      plnamespace: "0",
      pllimit: String(limit),
      redirects: "1",
      titles: title,
      format: "json",
    });
  const res = await fetch(url, {
    headers: { "User-Agent": "TinyGPT-LLM-mini/2.0 (educational corpus fetch)" },
  });
  if (!res.ok) return [];
  const data = await res.json();
  const page = Object.values(data.query?.pages ?? {})[0];
  return (page?.links ?? []).map((l) => l.title);
}

function toLines(text) {
  const out = [];
  for (const block of text.split(/\n+/)) {
    const p = block.trim();
    if (!p || /^=+ .+ =+$/.test(p)) continue;
    const parts = p.length <= 200 ? [p] : p.split(/(?<=[.!?…])\s+/);
    for (const s of parts) {
      const line = s.trim().normalize("NFC");
      if (line.length >= MIN_LINE && VI_DIACRITIC.test(line)) out.push(line);
    }
  }
  return out;
}

async function main() {
  const root = repoRoot();
  const outFile = path.join(root, "train-data", "vi-wikipedia.txt");
  const queue = [DEFAULT_SEED];
  const seen = new Set();
  const allLines = [];
  const lineSet = new Set();
  let fetched = 0;

  console.log(`Fetching up to ${maxArticles} vi.wikipedia articles…`);
  while (queue.length > 0 && fetched < maxArticles) {
    const title = queue.shift();
    if (!title || seen.has(title)) continue;
    seen.add(title);
    try {
      const raw = await fetchArticle(title);
      const lines = toLines(raw);
      for (const l of lines) {
        if (lineSet.has(l)) continue;
        lineSet.add(l);
        allLines.push(l);
      }
      fetched += 1;
      console.log(`  ${title}: +${lines.length} lines (total ${allLines.length})`);
      for (const link of await fetchLinks(title)) {
        if (!seen.has(link)) queue.push(link);
      }
    } catch (err) {
      console.warn(`  skip ${title}: ${err instanceof Error ? err.message : err}`);
    }
  }

  fs.mkdirSync(path.dirname(outFile), { recursive: true });
  fs.writeFileSync(outFile, `${allLines.join("\n")}\n`, "utf8");
  const kb = (Buffer.byteLength(allLines.join("\n")) / 1024).toFixed(0);
  console.log(`\nWrote ${allLines.length} lines (${kb} KB) → ${path.relative(root, outFile)}`);
  console.log(
    "Train CLI: npm run train:tiny -- --preset small --corpus train-data/vi-wikipedia.txt --steps 20000 --out models/vi-wiki",
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
