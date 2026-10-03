/** Short UI copy - hints under titles/fields; longer text lives in ? tooltips. */

export const HINT = {
  flow: "text → tokens → embeddings → N × (attention + MLP) → next-token probabilities → pick → repeat",
  chat: "Type anything and press Enter.",
  train:
    "Runs until Stop. When you Stop, model is saved to models/ (model.json + weights.bin). Train again continues from current weights.",
  exportModel:
    "Save → models/vi-stepN-…/ (model.json + weights.bin + meta.json). List: npm run models:list",
  stats: "",
  statsChart: "",
  preset: "Pick a size, or type a target parameter count and press Auto-fill.",
  targetParams: "e.g. 1000000 (1M), 50000000, 1000000000 (1B).",
  layers: "Number of transformer blocks stacked on top of each other.",
  dModel: "Width of every token vector.",
  heads: "Parallel attention heads per block (must divide d_model).",
  ffnSize: "Hidden width of each block's MLP (usually 4 × d_model).",
  contextLength: "How many tokens the model can look back at.",
  learningRate: "Step size of the Adam optimizer.",
  batchSize: "Text windows per training step.",
  dropout:
    "Randomly drops activations while training (0–0.3). Helps large models generalize.",
  corpus:
    "One line = one document. Any language: arithmetic, Vietnamese, English, code…",
  vietnamese:
    "Load sample or fetch one article into the corpus manually. Use the main page for foundation corpus training followed by Vietnamese Wikipedia.",
  viTrain:
    "Bước 1: corpus tiếng Việt có sẵn → tự dừng/lưu đủ số bước. Bước 2: nhập link vi.wikipedia.org để học thêm trên cùng weights. Load để tiếp tục checkpoint.",
  wikiUrl:
    "Paste a vi.wikipedia.org article link. Text is downloaded and split into lines (one sentence per line).",
  apply:
    "Rebuilds tokenizer and model from scratch. Training restarts at step 0.",
} as const;

export const TOOLTIP = {
  preset:
    "Presets are typical GPT shapes. Mini/Small train comfortably in the browser. Medium is slow in the browser. Large (~100M) and XL (~1B) need the Node CLI - a 1B model needs about 16 GB just for weights + optimizer state.",
  targetParams:
    "Type how many parameters you want (for example 1000000000 for 1B). Auto-fill picks layers, d_model, heads and FFN size with GPT-like proportions so the total lands close to your target. You can still edit every field afterwards.",
  layers:
    "A layer (block) = one attention step + one MLP step. Each block lets the model refine every token's meaning using the other tokens. More layers = deeper reasoning, more params, slower training. GPT-2 small has 12, GPT-3 has 96.",
  dModel:
    "d_model is the length of the vector that represents each token inside the model. Wider vectors can hold more information per token. Parameters grow roughly with layers × d_model².",
  heads:
    "Attention is split into several heads that look at the context in different ways (one head may track the operator, another the previous digit). d_model must be divisible by the number of heads; each head gets d_model / heads dimensions.",
  ffnSize:
    "Each block has a small 2-layer MLP (feed-forward network) applied to every token. FFN size is its hidden width - 4 × d_model is the standard choice. This is where most parameters live.",
  contextLength:
    "Maximum number of tokens the model sees at once. Longer context = it can use more of the conversation, but attention cost grows with context².",
  learningRate:
    "How far weights move each step. Small models like 1e-3 to 3e-3; big models need smaller values (3e-4 to 1e-4) or training becomes unstable.",
  batchSize:
    "How many random text windows are processed per step. Larger batches give smoother training but need more memory.",
  dropout:
    "During training, randomly zeroes a fraction of activations so the model cannot rely on memorizing exact patterns. 0 for tiny models, 0.1 for larger ones.",
  corpus:
    "Training text. Every line is one example; the model learns to predict the next character at every position of every line (causal language modelling - the same objective as GPT). The vocabulary is built automatically from all characters found, so Vietnamese works out of the box. To 'speak' a language well you need a lot of text (megabytes) and a large model.",
  vietnamese:
    "Use the Vietnamese section to load sample sentences or fetch a Wikipedia article. The browser downloads article text via the official Wikipedia API (read-only). Training itself still starts when you press Train on the main page - fetch only fills the corpus.",
  wikiUrl:
    "Example: https://vi.wikipedia.org/wiki/Hà_Nội or https://vi.wikipedia.org/wiki/Trí_tuệ_nhân_tạo. You can fetch several articles - each one is appended to the corpus. For very large training, save the corpus and use npm run train:tiny in the terminal.",
  memory:
    "Estimated memory to train: weights + gradients + Adam's two moment buffers (4 × params × 4 bytes) plus activations for one batch. Browser tabs usually cap at a few GB.",
  tooltipT1:
    "The prompt split into tokens (characters). Each token id selects one row of the embedding matrix - that row (first dimensions shown) is the vector the model actually works with. BOS marks the start of a line.",
  tooltipT2:
    "Attention weights: row = the token being processed, column = the earlier token it looks at. Each row sums to 100%. Upper-right is always 0 because the model may not look at the future (causal mask). Pick a layer and head to compare.",
  tooltipT3:
    "How the next token is chosen: final vector × embedding matrix → one score per vocabulary token → softmax → probabilities. Greedy decoding takes the top one. Step buttons walk through each generated token.",
  tooltipT4:
    "Weight values with heat colors (red = positive, blue = negative). For wte (token embeddings): each column is one token/character, each row is one embedding dimension (e0, e1, …). After chat, the column for the token picked in T3 is highlighted.",
} as const;

export const GUIDE_SECTIONS = [
  {
    title: "How a GPT works",
    body: "Text → tokens → embedding vectors → several transformer blocks (attention mixes information between tokens, MLP transforms each token) → scores for every possible next token → pick one → append → repeat.",
  },
  {
    title: "Training",
    body: "The model reads your corpus and, at every position, tries to predict the next character. Loss measures how wrong it was; Adam nudges all weights to lower it. val = lines held out from training, to check it generalizes instead of memorizing.",
  },
  {
    title: "Sizes",
    body: "Everything is configurable in Settings: layers, width, heads, context, vocabulary (from your text). Small models train in the browser; 100M–1B need the Node CLI (npm run train:tiny).",
  },
] as const;

export function boxHeader(title: string, hint?: string): string {
  return `<div class="box-title">${title}</div>${hint ? `<div class="box-hint">${hint}</div>` : ""}`;
}

export function tipIcon(tooltip: string): string {
  return `<span class="help-tip" tabindex="0" role="button" aria-label="Help">?<span class="help-tip-bubble">${tooltip}</span></span>`;
}

export function panelHeader(
  title: string,
  tooltip: string,
  extra = "",
): string {
  return `<div class="box-title-row"><span class="box-title-text">${title}</span>${tipIcon(tooltip)}${extra}</div>`;
}

export function fieldBlock(
  label: string,
  hint: string,
  inputHtml: string,
  tooltip?: string,
): string {
  return `<label class="field">
    <span class="field-label-row"><span class="field-label">${label}</span>${tooltip ? tipIcon(tooltip) : ""}</span>
    ${inputHtml}
    <span class="field-hint">${hint}</span>
  </label>`;
}
