# LLM-mini

Learn how language models work at the matrix level. Two independent packages, each with its own web UI:

| Package | Folder | UI | Port |
|---------|--------|-----|------|
| **Tiny LLM** | `packages/tiny-llm` + `packages/tiny-web` | Dynamic GPT (mini → 1B params) on TF.js, attention/weights panels, CLI for big models | `5200` |
| **Full Follow LLM** | `packages/full-follow-llm` + `packages/full-follow-llm-web` | GPT-style Transformer with attention, forward/backward traces | `5199` |

Everything is plain TypeScript - no PyTorch. Forward pass, backpropagation, and optimizers are written by hand.

## Quick start

Requires **Node.js 22+**.

```bash
npm install

npm run dev:tiny    # Tiny LLM UI → http://localhost:5200
npm run dev:full    # Full Follow LLM UI → http://localhost:5199
```

Build all packages:

```bash
npm run build
```

Tests:

```bash
npm run test:tiny
npm run test:full
```

CLI (Tiny LLM):

```bash
npm run train:tiny -- --preset small --steps 3000
npm run chat:tiny -- --model models/small
```

CLI (Full Follow LLM):

```bash
npm run train:full
npm run ask:full -- "2+3="
```

---

## Tiny LLM (`packages/tiny-llm` + `packages/tiny-web`)

A real, fully configurable GPT (decoder-only transformer) built on TensorFlow.js - from ~30k parameters up to ~1B.

```text
text → char tokens → embeddings + positions → N × (LayerNorm → causal multi-head attention → LayerNorm → GELU MLP) → tied output head → softmax → next token
```

- **Everything dynamic:** layers, d_model, heads, FFN size, context length, learning rate, batch, dropout. The vocabulary is built from whatever text you train on (NFC-normalized, so Vietnamese works).
- **Target params → auto-fill:** type e.g. `1000000000` in Settings and press Auto-fill; layers/width/heads are picked with GPT-like proportions. Every field stays editable.
- **Causal LM objective:** the model predicts the next character at every position, so any prompt gets a continuation (`2+2` → `=4`), not just prompts ending in `=`.
- **Browser** trains small models in a Web Worker (WebGL when available); **Node CLI** trains big ones (tfjs-node-gpu → tfjs-node → wasm → cpu).

### Presets

| Preset | layers | d_model | heads | FFN | ctx | ≈ params | Where |
|--------|-------:|--------:|------:|----:|----:|---------:|-------|
| mini   | 2  | 32   | 2  | 128  | 32   | ~30k  | browser |
| small  | 4  | 128  | 4  | 512  | 64   | ~0.8M | browser |
| medium | 6  | 384  | 6  | 1536 | 128  | ~11M  | browser (slow) / CLI |
| large  | 12 | 768  | 12 | 3072 | 256  | ~85M  | CLI |
| xl     | 20 | 2048 | 16 | 8192 | 512  | ~1.0B | CLI (big GPU / lots of RAM) |

Params: `vocab·d + ctx·d + layers·(4d² + 2d·ffn + 9d + ffn) + 2d`.
Training memory ≈ params × 16 bytes (weights + grads + Adam m/v) + activations - a 1B model needs ~16 GB+.
Settings shows the exact count, memory estimate, and a **Browser OK / Browser slow / CLI only** badge with a ready-to-copy CLI command.

### CLI

```bash
npm run train:tiny -- --preset small --corpus train-data/vi-sample.txt --steps 5000 --out models/vi-small
npm run train:tiny -- --target-params 1000000000 --corpus my-big-text.txt --out models/1b
npm run train:tiny -- --layers 8 --d-model 256 --heads 8 --ffn 1024 --context 128 --corpus data.txt
npm run train:tiny -- --resume models/vi-small --steps 5000      # continue training
npm run chat:tiny -- --model models/vi-small --temperature 0.8
npm run verify:tiny                                             # quick smoke test
```

Corpus: `.txt` (one line per example) or `.json` (`string[]` or `{ examples: [...] }`). Without `--corpus` it generates arithmetic (`--max-number N`).
Models are saved as `model.json` (config + vocab) + `weights.bin` (float32).

**Browser:** Stop / **Export version** → `models/exports/v001-stepN-…/` (auto version). **Load** dropdown in UI.

**CLI scripts:**

```bash
npm run models:list
npm run start:model -- v001 --mode chat
npm run start:model -- v002 --mode train --steps 5000
npm run start:model -- v001 --mode web
```

### Training Vietnamese

See **[docs/training-vietnamese.md](docs/training-vietnamese.md)** for the full guide (curriculum: sample → Wikipedia, UI button, CLI, references).

**Quick (UI):** Settings → **Start Vietnamese training** → auto-fetch vi.wikipedia + Train.

**Quick (CLI):**

```bash
npm run train:tiny -- --preset small --corpus train-data/vi-sample.txt --steps 3000 --out models/vi-phase1
npm run train:tiny -- --resume models/vi-phase1 --corpus train-data/vi-wiki.txt --steps 10000 --out models/vi-phase2
npm run chat:tiny -- --model models/vi-phase2
```

A character-level model learns spelling and diacritics first, then words, then grammar. Real fluency needs both a large model and a large corpus.

### Dev watch mode

```bash
npm run dev:tiny:watch   # Vite + tsc --watch (tiny-llm)
npm run dev:tiny         # Vite only (port 5200)
```

---

## Full Follow LLM (`packages/full-follow-llm`)

Full GPT-style decoder (~2,900 params default):

```text
"2+3=" → embeddings → attention (Q/K/V) → layer norm → MLP → logits → softmax
```

- **Library:** `packages/full-follow-llm` - model, training, inference, CLI
- **Web UI:** `packages/full-follow-llm-web` - React app with chat, forward trace, training panel, weight updates

```bash
npm run dev:full
```

---

## Project layout

```text
packages/
  tiny-llm/                  @math-llm/tiny-llm - dynamic GPT, trainer, CLI
  tiny-web/                  @math-llm/tiny-web - Vite UI (port 5200)
  full-follow-llm/           @math-llm/full-follow-llm - Transformer library + CLI
  full-follow-llm-web/       @math-llm/full-follow-llm-web - React UI (port 5199)
train-data/arithmetic.json   Shared arithmetic dataset (used by Full Follow LLM)
```
