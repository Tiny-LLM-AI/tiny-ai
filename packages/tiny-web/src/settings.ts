import {
  PRESETS,
  buildTokenizer,
  corpusTokenCount,
  formatBytes,
  formatCount,
  generateArithmeticLines,
  memoryEstimate,
  normalizeConfig,
  paramBreakdown,
  paramCount,
  parseCorpus,
  shapeForTargetParams,
  totalArithmeticLines,
  trainability,
  type GptConfig,
  type PresetId,
  type Trainability,
} from "@math-llm/tiny-llm";
import { HINT, TOOLTIP, fieldBlock, tipIcon } from "./help.js";
import { fetchWikipediaCorpus, wikipediaLangFromUrl } from "./wikipedia.js";
export type ModelShape = Omit<GptConfig, "vocabSize">;

export interface SettingsState {
  shape: ModelShape;
  corpusText: string;
}

const SHAPE_FIELDS: {
  name: keyof ModelShape;
  label: string;
  step: string;
  min: string;
}[] = [
  { name: "layers", label: "Layers", step: "1", min: "1" },
  { name: "dModel", label: "d_model (width)", step: "1", min: "4" },
  { name: "heads", label: "Attention heads", step: "1", min: "1" },
  { name: "ffnSize", label: "FFN size", step: "1", min: "1" },
  { name: "contextLength", label: "Context length", step: "1", min: "2" },
  {
    name: "learningRate",
    label: "Learning rate",
    step: "any",
    min: "0.000001",
  },
  { name: "batchSize", label: "Batch size", step: "1", min: "1" },
  { name: "dropout", label: "Dropout", step: "0.05", min: "0" },
];

const BADGE: Record<Trainability, { cls: string; text: string }> = {
  browser: { cls: "badge-ok", text: "Browser OK" },
  "browser-slow": { cls: "badge-warn", text: "Browser slow" },
  cli: { cls: "badge-cli", text: "CLI only" },
};

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function cliCommand(shape: ModelShape): string {
  return [
    "npm run train:tiny --",
    "--corpus train-data/your-text.txt",
    `--layers ${shape.layers} --d-model ${shape.dModel} --heads ${shape.heads} --ffn ${shape.ffnSize}`,
    `--context ${shape.contextLength} --lr ${shape.learningRate} --batch ${shape.batchSize}`,
    "--steps 20000 --out models/custom",
  ].join(" ");
}

export function badgeHtml(t: Trainability): string {
  const b = BADGE[t];
  return `<span class="badge ${b.cls}">${b.text}</span>`;
}

function vocabChip(c: string): string {
  const label = c === " " ? "␣" : escapeHtml(c);
  return `<span class="vocab-chip">${label}</span>`;
}

export function renderSettingsPage(
  app: HTMLElement,
  initial: SettingsState,
  handlers: { onApply: (s: SettingsState) => void; onBack: () => void },
): void {
  const s = initial.shape;
  app.innerHTML = `
    <div class="settings-page">
      <header class="settings-head">
        <button id="back" class="sec">← Back</button>
        <strong>Settings</strong>
        <span class="muted">Applying rebuilds the tokenizer and model - training restarts from scratch</span>
      </header>
      <form id="settings-form" class="settings-form">
        <div class="settings-col settings-col-left">
          <h3 class="settings-col-title">Model size</h3>
          <div class="field">
            <span class="field-label-row"><span class="field-label">Preset</span>${tipIcon(TOOLTIP.preset)}</span>
            <div class="quick-row wrap">${PRESETS.map((p) => `<button type="button" class="chip" data-preset="${p.id}">${p.label}</button>`).join("")}</div>
            <span class="field-hint">${HINT.preset}</span>
          </div>
          ${fieldBlock(
            "Target parameters",
            HINT.targetParams,
            `<div class="inline-field-row"><input id="target-params" type="number" min="1000" step="1" placeholder="e.g. 1000000000" /><button type="button" id="auto-fill" class="sec">Auto-fill</button></div>`,
            TOOLTIP.targetParams,
          )}
          <div class="shape-grid">
            ${SHAPE_FIELDS.map((f) =>
              fieldBlock(
                f.label,
                HINT[f.name as keyof typeof HINT] ?? "",
                `<input name="${f.name}" type="number" min="${f.min}" step="${f.step}" value="${s[f.name]}" />`,
                TOOLTIP[f.name as keyof typeof TOOLTIP],
              ),
            ).join("")}
          </div>
          <div class="preview">
            <div class="preview-top"><b id="param-preview"></b> parameters <span id="badge"></span></div>
            <div id="param-formula" class="muted preview-line"></div>
            <div class="preview-line"><span id="mem-preview"></span> ${tipIcon(TOOLTIP.memory)}</div>
            <div id="cli-wrap" class="cli-wrap"></div>
          </div>
        </div>
        <div class="settings-col settings-col-right">
          <h3 class="settings-col-title">Training text (corpus)</h3>
          <div class="field vietnamese-field">
            <span class="field-label-row"><span class="field-label">Vietnamese / Wikipedia</span>${tipIcon(TOOLTIP.vietnamese)}</span>
            <span class="field-hint">${HINT.vietnamese}</span>
            <div class="dataset-toolbar wiki-toolbar">
              <button type="button" id="load-vi-sample" class="chip">Load VI sample</button>
              <label class="muted wiki-replace"><input type="checkbox" id="wiki-replace" /> Replace corpus</label>
            </div>
            <div class="wiki-fetch-row">
              <input id="wiki-url" type="url" class="wiki-url-input" placeholder="https://vi.wikipedia.org/wiki/Hà_Nội" spellcheck="false" />
              <select id="wiki-lang" class="wiki-lang-select" title="Language wiki">
                <option value="vi">vi</option>
                <option value="en">en</option>
              </select>
              <button type="button" id="wiki-fetch" class="chip">Fetch article</button>
            </div>
            <div id="wiki-status" class="suggest-line muted"></div>
          </div>
          <div class="field dataset-field">
            <span class="field-label-row"><span class="field-label">Corpus</span>${tipIcon(TOOLTIP.corpus)}</span>
            <div class="dataset-toolbar">
              <span class="muted">Arithmetic 0–</span>
              <input id="arith-max" type="number" min="0" step="1" value="9" class="small-input" />
              <button type="button" id="gen-arith" class="chip">Generate arithmetic</button>
              <label class="chip file-chip">Load .txt<input id="load-file" type="file" accept=".txt,.csv,.md,text/plain" hidden /></label>
              <button type="button" id="clear-corpus" class="chip">Clear</button>
            </div>
            <textarea id="corpus" class="dataset-json" spellcheck="false" placeholder="One line per example. Any language.">${escapeHtml(initial.corpusText)}</textarea>
            <div id="corpus-stats" class="suggest-line muted"></div>
            <div id="vocab-preview" class="vocab-preview"></div>
            <span class="field-hint">${HINT.corpus}</span>
          </div>
        </div>
        <div class="settings-footer">
          <p class="field-hint apply-note">${HINT.apply}</p>
          <button type="submit" class="apply-btn">Apply & rebuild model</button>
        </div>
      </form>
    </div>
  `;

  const form = app.querySelector<HTMLFormElement>("#settings-form")!;
  const corpusEl = form.querySelector<HTMLTextAreaElement>("#corpus")!;
  const input = (name: string) =>
    form.querySelector<HTMLInputElement>(`input[name="${name}"]`)!;

  const readShape = (): ModelShape => {
    const num = (name: keyof ModelShape) => Number(input(name).value);
    const cfg = normalizeConfig({
      vocabSize: 0,
      layers: num("layers"),
      dModel: num("dModel"),
      heads: num("heads"),
      ffnSize: num("ffnSize"),
      contextLength: num("contextLength"),
      learningRate: num("learningRate") || s.learningRate,
      batchSize: num("batchSize"),
      dropout: num("dropout"),
      seed: s.seed,
    });
    const { vocabSize: _v, ...shape } = cfg;
    return shape;
  };

  const writeShape = (shape: Partial<ModelShape>) => {
    for (const [k, v] of Object.entries(shape)) {
      const el = form.querySelector<HTMLInputElement>(`input[name="${k}"]`);
      if (el) el.value = String(v);
    }
  };

  const vocabSize = () => buildTokenizer(corpusEl.value).vocab.length;

  const updatePreview = () => {
    const shape = readShape();
    const cfg: GptConfig = { ...shape, vocabSize: vocabSize() };
    form.querySelector("#param-preview")!.textContent =
      `${formatCount(paramCount(cfg))} (${paramCount(cfg).toLocaleString()})`;
    form.querySelector("#param-formula")!.textContent = paramBreakdown(cfg);
    const mem = memoryEstimate(cfg);
    form.querySelector("#mem-preview")!.textContent =
      `Training memory ≈ ${formatBytes(mem.totalBytes)} (weights ${formatBytes(paramCount(cfg) * 4)})`;
    const t = trainability(cfg);
    form.querySelector("#badge")!.innerHTML = badgeHtml(t);
    form.querySelector("#cli-wrap")!.innerHTML =
      t === "browser"
        ? ""
        : `<div class="field-hint">${t === "cli" ? "Too large for the browser. Train it with the CLI:" : "Works in the browser but slowly. Faster with the CLI:"}</div><pre class="cli-cmd">${escapeHtml(cliCommand(shape))}</pre>`;
  };

  const updateCorpusStats = () => {
    const lines = parseCorpus(corpusEl.value);
    const tok = buildTokenizer(corpusEl.value);
    const chars = tok.vocab.slice(4);
    form.querySelector("#corpus-stats")!.textContent =
      `${lines.length.toLocaleString()} lines · ${corpusTokenCount(lines).toLocaleString()} tokens · vocab ${tok.vocab.length} (${chars.length} characters + 4 special)`;
    form.querySelector("#vocab-preview")!.innerHTML =
      chars.length === 0
        ? `<span class="muted">No characters yet — add corpus text.</span>`
        : `<div class="vocab-table-wrap"><table class="matrix vocab-table"><thead><tr><th>#</th><th>char</th><th>id</th></tr></thead><tbody>${chars
            .map(
              (c, i) =>
                `<tr><td>${i + 1}</td><td class="tok">${vocabChip(c)}</td><td>${i + 4}</td></tr>`,
            )
            .join("")}</tbody></table></div>`;
    updatePreview();
  };

  form.oninput = (e) => {
    if (e.target === corpusEl) updateCorpusStats();
    else updatePreview();
  };

  form.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach((btn) => {
    btn.onclick = () => {
      const preset = PRESETS.find(
        (p) => p.id === (btn.dataset.preset as PresetId),
      )!;
      writeShape(preset.config);
      updatePreview();
    };
  });

  form.querySelector<HTMLButtonElement>("#auto-fill")!.onclick = () => {
    const target = Number(
      form.querySelector<HTMLInputElement>("#target-params")!.value,
    );
    if (!target || target < 1000) return;
    writeShape(
      shapeForTargetParams(target, vocabSize(), readShape().contextLength),
    );
    updatePreview();
  };

  form.querySelector<HTMLButtonElement>("#gen-arith")!.onclick = () => {
    const max = Math.max(
      0,
      Math.floor(
        Number(form.querySelector<HTMLInputElement>("#arith-max")!.value) || 0,
      ),
    );
    const lines = generateArithmeticLines(
      max,
      Math.min(totalArithmeticLines(max), 20000),
      42,
    );
    corpusEl.value = lines.join("\n");
    updateCorpusStats();
  };

  form.querySelector<HTMLInputElement>("#load-file")!.onchange = async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    corpusEl.value = await file.text();
    updateCorpusStats();
  };

  form.querySelector<HTMLButtonElement>("#clear-corpus")!.onclick = () => {
    corpusEl.value = "";
    updateCorpusStats();
  };

  const wikiStatus = (msg: string) => {
    form.querySelector("#wiki-status")!.textContent = msg;
  };

  const appendCorpus = (text: string, replace: boolean) => {
    const block = text.trim();
    if (!block) return;
    corpusEl.value =
      replace || !corpusEl.value.trim()
        ? block
        : `${corpusEl.value.trim()}\n${block}`;
    updateCorpusStats();
  };

  form.querySelector<HTMLButtonElement>("#load-vi-sample")!.onclick =
    async () => {
      wikiStatus("Loading sample…");
      try {
        const res = await fetch("/corpus/vi-sample.txt");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        const replace =
          form.querySelector<HTMLInputElement>("#wiki-replace")!.checked;
        appendCorpus(text, replace);
        wikiStatus(
          `Loaded ${text.split("\n").filter(Boolean).length} sample lines.`,
        );
      } catch (err) {
        wikiStatus(err instanceof Error ? err.message : String(err));
      }
    };

  form.querySelector<HTMLButtonElement>("#wiki-fetch")!.onclick = async () => {
    const input = form.querySelector<HTMLInputElement>("#wiki-url")!;
    const url = input.value.trim();
    if (!url) {
      wikiStatus("Paste a Wikipedia article URL first.");
      return;
    }
    const langSel = form.querySelector<HTMLSelectElement>("#wiki-lang")!;
    if (url.startsWith("http")) langSel.value = wikipediaLangFromUrl(url);
    const btn = form.querySelector<HTMLButtonElement>("#wiki-fetch")!;
    btn.disabled = true;
    wikiStatus("Fetching from Wikipedia…");
    try {
      const { title, corpus, lineCount } = await fetchWikipediaCorpus(
        url,
        langSel.value,
      );
      const replace =
        form.querySelector<HTMLInputElement>("#wiki-replace")!.checked;
      appendCorpus(corpus, replace);
      wikiStatus(
        `Added “${title}”: ${lineCount.toLocaleString()} lines (${corpus.length.toLocaleString()} chars). Apply & rebuild, then Train.`,
      );
    } catch (err) {
      wikiStatus(err instanceof Error ? err.message : String(err));
    } finally {
      btn.disabled = false;
    }
  };

  form.onsubmit = (e) => {
    e.preventDefault();
    if (parseCorpus(corpusEl.value).length === 0) {
      form.querySelector("#corpus-stats")!.textContent =
        "Corpus is empty - add at least one line.";
      return;
    }
    handlers.onApply({ shape: readShape(), corpusText: corpusEl.value });
  };
  app.querySelector<HTMLButtonElement>("#back")!.onclick = handlers.onBack;

  updateCorpusStats();
}
