import {
  BOS_ID,
  EOS_ID,
  PAD_ID,
  buildTokenizer,
  createModel,
  deserializeModel,
  disposeModel,
  embeddingRows,
  exportWeights,
  forwardTrace,
  warmupForward,
  formatCount,
  generateArithmeticLines,
  importWeights,
  initTfBackend,
  isSpecialId,
  paramCount,
  parseCorpus,
  presetConfig,
  sampleParam,
  sampleWteDisplay,
  splitCorpus,
  tokenText,
  trainability,
  type ForwardTrace,
  type GenerateResult,
  type GptConfig,
  type GptModel,
  type Tokenizer,
  type ChatWorkerMessage,
  type TrainWorkerProgress,
  type TrainWorkerRequest,
} from "@math-llm/tiny-llm";
import {
  GUIDE_SECTIONS,
  HINT,
  TOOLTIP,
  boxHeader,
  panelHeader,
} from "./help.js";
import { paintChartRow } from "./line-chart.js";
import {
  VI_DEFAULT_SEED,
  ViWikiCrawler,
  usesViVocab,
  viTokenizer,
  type ViArticle,
} from "./vi-training.js";
import {
  badgeHtml,
  cliCommand,
  escapeHtml,
  renderSettingsPage,
  type ModelShape,
} from "./settings.js";
import {
  exportModel,
  fetchExportList,
  loadExportById,
  type ExportListEntry,
} from "./export-model.js";

interface ChatTurn {
  role: "user" | "assistant";
  text: string;
  /** Assistant only: queued → thinking (before first token) → streaming → done. */
  status?: "queued" | "thinking" | "streaming" | "done";
  /** Assistant only: the user prompt this reply continues. */
  prompt?: string;
}

const MAX_CHART_POINTS = 300;
const MAX_TRACE_TOKENS = 24;
/** Fixed sampling - not exposed in UI. */
const CHAT_TEMPERATURE = 0.2;
const CHAT_TOP_K = 20;

let shape: ModelShape = { ...presetConfig("mini") };
let corpusText = generateArithmeticLines(9, 1000, 42).join("\n");

let tokenizer: Tokenizer = buildTokenizer(corpusText);
let trainLines: string[] = [];
let valLines: string[] = [];
let model: GptModel | null = null;
let backendName = "…";

let worker: Worker | null = null;
let training = false;
let stopping = false;
let step = 0;
let lastLoss = -1;
let lastValLoss = -1;
let lastValAcc = -1;
let lossHistory: number[] = [];
let accHistory: number[] = [];
let lossEma = -1;
const LOSS_EMA_ALPHA = 0.85;

let chatHistory: ChatTurn[] = [];
let chatGenerating = false;
let chatStreamStep = 0;
let chatWorker: Worker | null = null;
let chatWorkerReady = false;
let chatWorkerReadyResolve: (() => void) | null = null;
let chatGenerateId = 0;
let activeChatGenerateId = 0;
let activeChatTurn: ChatTurn | null = null;
let chatGenerateResolve: ((result: GenerateResult) => void) | null = null;
let chatGenerateReject: ((err: Error) => void) | null = null;
let lastGen: GenerateResult | null = null;
let pickStep = 0;
let traceTokenIds: number[] | null = null;
let attentionTrace: ForwardTrace | null = null;
let attentionJob = 0;
let attnLayer = 0;
let attnHead = 0;
let weightName = "wte";
let lastExportName = "";
let exportCatalog: ExportListEntry[] = [];
let loadedExportId = "";

declare global {
  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
  interface ImportMetaEnv {
    readonly TINY_GPT_MODEL_ID?: string;
  }
}

let showSettings = false;
let showGuide = false;

const app = document.querySelector<HTMLDivElement>("#app")!;

function fullConfig(): GptConfig {
  return { ...shape, vocabSize: tokenizer.vocab.length };
}

function rebuild(fixedTokenizer?: Tokenizer): void {
  stopWorkerNow();
  if (model) disposeModel(model);
  tokenizer = fixedTokenizer ?? buildTokenizer(corpusText);
  const split = splitCorpus(parseCorpus(corpusText), 0.1, shape.seed);
  trainLines = split.train;
  valLines = split.val;
  model =
    trainability(fullConfig()) === "cli"
      ? null
      : createModel(fullConfig(), tokenizer);
  if (model) warmupForward(model);
  initChatWorker();
  step = 0;
  lastLoss = lastValLoss = lastValAcc = -1;
  lossHistory = [];
  accHistory = [];
  lastGen = null;
  traceTokenIds = null;
  attentionTrace = null;
  pickStep = 0;
  attnLayer = attnHead = 0;
  weightName = "wte";
}

function displayToken(id: number): string {
  if (id === BOS_ID) return "BOS";
  if (id === EOS_ID) return "EOS";
  if (id === PAD_ID) return "PAD";
  if (isSpecialId(id)) return "UNK";
  const t = tokenText(tokenizer, id);
  return t === " " ? "␣" : escapeHtml(t);
}

/* ---------- training ---------- */

function exportMeta() {
  return {
    step,
    loss: lastLoss,
    valLoss: lastValLoss,
    valAcc: lastValAcc,
    exportedAt: new Date().toISOString(),
  };
}

async function saveModelExport(): Promise<void> {
  if (!model || step <= 0) {
    setStatus("Train at least one step before exporting.");
    return;
  }
  setStatus("Exporting to models/exports/…");
  try {
    lastExportName = await exportModel(model, exportMeta());
    exportCatalog = await fetchExportList();
    loadedExportId = lastExportName.includes(".zip") ? "" : lastExportName;
    setStatus(
      `Saved models/exports/${lastExportName} · npm run start:model -- ${lastExportName.slice(0, 4)} --mode chat`,
    );
  } catch (err) {
    setStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
  }
  paintTrainState();
  renderExportControls();
}

async function loadSelectedExport(id: string): Promise<void> {
  if (!id || training || viActive) return;
  stopWorkerNow();
  setStatus(`Loading ${id}…`);
  try {
    const { manifest, weights, meta } = await loadExportById(id);
    if (model) disposeModel(model);
    model = deserializeModel(manifest, weights);
    tokenizer = model.tokenizer;
    warmupForward(model);
    step = meta.step ?? 0;
    lastLoss = meta.loss ?? -1;
    lastValLoss = meta.valLoss ?? -1;
    lastValAcc = meta.valAcc ?? -1;
    loadedExportId = id;
    shape = {
      layers: model.config.layers,
      dModel: model.config.dModel,
      heads: model.config.heads,
      ffnSize: model.config.ffnSize,
      contextLength: model.config.contextLength,
      learningRate: model.config.learningRate,
      batchSize: model.config.batchSize,
      dropout: model.config.dropout,
      seed: model.config.seed,
    };
    const split = splitCorpus(parseCorpus(corpusText), 0.1, shape.seed);
    trainLines = split.train;
    valLines = split.val;
    initChatWorker();
    lastGen = null;
    traceTokenIds = null;
    attentionTrace = null;
    setStatus(`Loaded ${id} (step ${step}). Train to continue or chat to test.`);
    render();
  } catch (err) {
    setStatus(`Load failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}

function renderExportControls(): void {
  const sel = document.querySelector<HTMLSelectElement>("#export-load");
  if (!sel) return;
  sel.innerHTML =
    `<option value="">— load saved version —</option>` +
    exportCatalog
      .map(
        (e) =>
          `<option value="${escapeHtml(e.id)}" ${e.id === loadedExportId ? "selected" : ""}>${escapeHtml(e.version)} · step ${e.step}${e.loss >= 0 ? ` · loss ${e.loss.toFixed(2)}` : ""}</option>`,
      )
      .join("");
}

async function refreshExportCatalog(): Promise<void> {
  exportCatalog = await fetchExportList();
  renderExportControls();
}

function stopWorkerNow(): void {
  worker?.terminate();
  worker = null;
  training = false;
  stopping = false;
}

function startTraining(): void {
  if (!model || training) return;
  stopping = false;
  const w = new Worker(
    new URL("../../tiny-llm/src/train-worker.ts", import.meta.url),
    { type: "module" },
  );
  worker = w;
  training = true;
  const weights = exportWeights(model);
  const req: TrainWorkerRequest = {
    type: "TRAIN",
    config: model.config,
    vocab: tokenizer.vocab,
    trainLines,
    valLines,
    weights,
    startStep: step,
  };
  w.onmessage = (e: MessageEvent<TrainWorkerProgress>) => {
    if (w !== worker) return;
    onProgress(e.data);
  };
  w.onerror = (e) => {
    if (w !== worker) return;
    stopWorkerNow();
    setStatus(`Worker error: ${e.message}`);
    paintTrainState();
  };
  w.postMessage(req, [weights.buffer]);
  paintTrainState();
}

/* ---------- Vietnamese Wikipedia rounds ---------- */

const VI_REPLAY_MAX = 2000;
let viActive = false;
let viSwitching = false;
let viRound = 0;
let viRoundEnd = 0;
let viSeed = VI_DEFAULT_SEED;
let viStepsPerRound = 300;
let viCrawler: ViWikiCrawler | null = null;
let viNext: Promise<ViArticle> | null = null;
let viCurrent = "";
let viNextTitle = "";
let viMemory: string[] = [];

let pausing = false;
let paused = false;

function requestPause(): void {
  if (paused || pausing) return;
  if (!worker) {
    paused = true;
    setStatus(`Paused at step ${step}.`);
    paintTrainState();
    return;
  }
  pausing = true;
  worker.postMessage({ type: "STOP" });
  setStatus("Pausing…");
  paintTrainState();
}

function resumeTraining(): void {
  if (!paused || !model) return;
  paused = false;
  if (viActive && step >= viRoundEnd) void nextViRound();
  else startTraining();
}

function stopViTraining(): void {
  paused = false;
  pausing = false;
  viActive = false;
  viSwitching = false;
  viNext = null;
}

function prefetchViArticle(): void {
  if (!viCrawler) return;
  viNextTitle = "fetching…";
  const pending = viCrawler.next();
  viNext = pending;
  pending
    .then((a) => {
      if (viNext === pending) viNextTitle = a.title;
      paintViState();
    })
    .catch(() => {
      if (viNext === pending) viNextTitle = "none left";
      paintViState();
    });
}

function sampleReplay(count: number): string[] {
  const pool = [...viMemory];
  for (let i = pool.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [pool[i], pool[j]] = [pool[j]!, pool[i]!];
  }
  return pool.slice(0, count);
}

async function startViTraining(): Promise<void> {
  if (training || viActive) return;
  const seedInput = document.querySelector<HTMLInputElement>("#vi-seed");
  const stepsInput = document.querySelector<HTMLInputElement>("#vi-steps");
  viSeed = seedInput?.value.trim() || VI_DEFAULT_SEED;
  viStepsPerRound = Math.max(10, Math.floor(Number(stepsInput?.value) || 300));
  try {
    viCrawler = new ViWikiCrawler(viSeed);
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
    return;
  }
  if (!model || !usesViVocab(tokenizer)) {
    corpusText = "";
    rebuild(viTokenizer());
    chatHistory = [];
    render();
  }
  if (!model) {
    setStatus("Model too large for the browser.");
    return;
  }
  viActive = true;
  viRound = 0;
  viMemory = [];
  prefetchViArticle();
  await nextViRound();
}

async function nextViRound(): Promise<void> {
  if (!viActive || !viNext) return;
  setStatus("Fetching article…");
  paintTrainState();
  let article: ViArticle;
  try {
    article = await viNext;
  } catch (err) {
    stopViTraining();
    setStatus(`Vietnamese training ended: ${err instanceof Error ? err.message : String(err)}`);
    paintTrainState();
    return;
  }
  if (!viActive) return;
  prefetchViArticle();

  viRound += 1;
  viCurrent = article.title;
  const lines = [...article.lines, ...sampleReplay(article.lines.length)];
  const split = splitCorpus(lines, 0.1, shape.seed + viRound);
  trainLines = split.train;
  valLines = split.val.length > 0 ? split.val : article.lines.slice(0, 5);
  viMemory.push(...article.lines);
  if (viMemory.length > VI_REPLAY_MAX)
    viMemory.splice(0, viMemory.length - VI_REPLAY_MAX);

  viRoundEnd = step + viStepsPerRound;
  if (paused) {
    setStatus(`Paused. Round ${viRound} (“${article.title}”) is ready — press Resume.`);
    paintTrainState();
    return;
  }
  startTraining();
  setStatus(`Round ${viRound}: training on “${article.title}” (${article.lines.length} lines)`);
  paintTrainState();
}

function paintViState(): void {
  const el = document.querySelector<HTMLElement>("#vi-status");
  if (el) {
    if (viActive) {
      el.textContent = `${paused ? "Paused · " : ""}Round ${viRound} · now: ${viCurrent || "…"} · until step ${viRoundEnd} · next: ${viNextTitle || "…"} · queue ${viCrawler?.queued ?? 0} · replay pool ${viMemory.length} lines`;
    } else if (viRound > 0) {
      el.textContent = `Stopped after ${viRound} round(s). Last article: ${viCurrent}`;
    } else {
      el.textContent = HINT.viTrain;
    }
  }
  const startBtn = document.querySelector<HTMLButtonElement>("#vi-start");
  if (startBtn) startBtn.disabled = training || viActive || paused;
}

function requestStop(): void {
  pausing = false;
  if (paused && !worker) {
    paused = false;
    stopViTraining();
    setStatus("Stopped. Exporting model…");
    void saveModelExport();
    paintTrainState();
    return;
  }
  paused = false;
  if (viActive) {
    stopViTraining();
    if (!worker) {
      setStatus("Stopped. Exporting model…");
      void saveModelExport();
      paintTrainState();
      return;
    }
  }
  if (!worker) return;
  worker.postMessage({ type: "STOP" });
  stopping = true;
  setStatus("Stopping…");
  paintTrainState();
}

function onProgress(msg: TrainWorkerProgress): void {
  if (msg.backend) backendName = msg.backend;
  if (msg.weights && model) {
    importWeights(model, msg.weights);
    syncChatWorkerWeights(msg.weights);
  }
  step = msg.step;
  switch (msg.type) {
    case "STARTED":
      setStatus(`Training on ${msg.backend}…`);
      lossEma = -1;
      break;
    case "PROGRESS":
      lastLoss = msg.loss;
      lossEma =
        lossEma < 0
          ? msg.loss
          : LOSS_EMA_ALPHA * lossEma + (1 - LOSS_EMA_ALPHA) * msg.loss;
      pushCapped(lossHistory, lossEma);
      if (msg.valLoss >= 0) lastValLoss = msg.valLoss;
      if (msg.valAccuracy >= 0) {
        lastValAcc = msg.valAccuracy;
        pushCapped(accHistory, msg.valAccuracy);
      }
      if (viActive && !viSwitching && step >= viRoundEnd && worker) {
        viSwitching = true;
        worker.postMessage({ type: "STOP" });
      }
      break;
    case "STOPPED":
      stopWorkerNow();
      if (pausing) {
        pausing = false;
        paused = true;
        viSwitching = false;
        setStatus(`Paused at step ${step}. Weights kept — press Resume.`);
        break;
      }
      if (viSwitching) {
        viSwitching = false;
        void nextViRound();
        break;
      }
      setStatus("Stopped. Exporting model…");
      void saveModelExport().then(() => {
        setStatus(
          `Stopped step ${step}. Saved models/exports/${lastExportName}`,
        );
        paintTrainState();
      });
      break;
    case "ERROR":
      stopWorkerNow();
      stopViTraining();
      setStatus(`Error: ${msg.message ?? "unknown"}`);
      break;
  }
  paintTrainState();
  if (msg.weights) paintT4();
}

function pushCapped(arr: number[], v: number): void {
  arr.push(v);
  if (arr.length > MAX_CHART_POINTS)
    arr.splice(0, arr.length - MAX_CHART_POINTS);
}

function setStatus(text: string): void {
  const el = document.querySelector("#train-status");
  if (el) el.textContent = text;
}

function fmt(v: number, digits = 3): string {
  if (v < 0) return "-";
  if (!Number.isFinite(v)) return "-";
  return v.toFixed(digits);
}

function paintTrainState(): void {
  const set = (id: string, v: string) => {
    const el = document.querySelector(id);
    if (el) el.textContent = v;
  };
  set("#step", String(step));
  set("#loss", fmt(lastLoss));
  set("#val-loss", fmt(lastValLoss));
  set("#val-acc", lastValAcc < 0 ? "-" : `${(lastValAcc * 100).toFixed(0)}%`);
  set("#backend", backendName);
  const trainBtn = document.querySelector<HTMLButtonElement>("#train-btn");
  const stopBtn = document.querySelector<HTMLButtonElement>("#stop-btn");
  const resetBtn = document.querySelector<HTMLButtonElement>("#reset-btn");
  const exportBtn = document.querySelector<HTMLButtonElement>("#export-btn");
  const pauseBtn = document.querySelector<HTMLButtonElement>("#pause-btn");
  const busy = training || viActive || paused;
  if (trainBtn) trainBtn.disabled = busy || !model;
  if (stopBtn) stopBtn.disabled = !busy || stopping;
  if (pauseBtn) {
    pauseBtn.textContent = paused ? "Resume" : "Pause";
    pauseBtn.disabled = paused ? false : !(training || viActive) || stopping || pausing;
  }
  if (resetBtn) resetBtn.disabled = busy || !model;
  if (exportBtn) exportBtn.disabled = training || pausing || !model || step <= 0;
  const chart = document.querySelector<HTMLElement>("#chart-row");
  if (chart) paintChartRow(chart, lossHistory, accHistory);
  paintViState();
}

/* ---------- chat ---------- */

function disposeChatWorker(): void {
  chatWorker?.terminate();
  chatWorker = null;
  chatWorkerReady = false;
  chatWorkerReadyResolve = null;
  activeChatTurn = null;
  chatGenerateResolve = null;
  chatGenerateReject = null;
}

function initChatWorker(): void {
  disposeChatWorker();
  if (!model) return;
  chatWorker = new Worker(
    new URL("../../tiny-llm/src/chat-worker.ts", import.meta.url),
    { type: "module" },
  );
  chatWorkerReady = false;
  chatWorker.onmessage = (e: MessageEvent<ChatWorkerMessage>) =>
    onChatWorkerMessage(e.data);
  chatWorker.onerror = (e) => {
    const err = new Error(e.message || "Chat worker error");
    chatGenerateReject?.(err);
    chatGenerateReject = null;
    chatGenerateResolve = null;
    activeChatTurn = null;
  };
  const weights = exportWeights(model);
  chatWorker.postMessage(
    { type: "INIT", config: model.config, vocab: tokenizer.vocab, weights },
    [weights.buffer],
  );
}

function syncChatWorkerWeights(weights: Float32Array): void {
  if (!chatWorker) return;
  const copy = new Float32Array(weights);
  chatWorker.postMessage({ type: "SYNC", weights: copy }, [copy.buffer]);
}

function onChatWorkerMessage(msg: ChatWorkerMessage): void {
  if (msg.type === "READY") {
    chatWorkerReady = true;
    chatWorkerReadyResolve?.();
    chatWorkerReadyResolve = null;
    return;
  }
  if (msg.type === "STEP") {
    if (msg.id !== activeChatGenerateId || !activeChatTurn) return;
    chatStreamStep = msg.stepIndex + 1;
    activeChatTurn.text = msg.partial;
    if (msg.partial && activeChatTurn.status !== "streaming") {
      activeChatTurn.status = "streaming";
      paintChat();
    } else {
      updateStreamingBubble(msg.partial);
    }
    paintStreamStatus();
    return;
  }
  if (msg.type === "DONE") {
    if (msg.id !== activeChatGenerateId) return;
    chatGenerateResolve?.(msg.result);
    chatGenerateResolve = null;
    chatGenerateReject = null;
    return;
  }
  if (msg.type === "ERROR") {
    if (msg.id !== undefined && msg.id !== activeChatGenerateId) return;
    chatGenerateReject?.(new Error(msg.message));
    chatGenerateReject = null;
    chatGenerateResolve = null;
  }
}

async function waitChatWorkerReady(): Promise<void> {
  if (chatWorkerReady) return;
  if (!chatWorker) throw new Error("Chat worker not available");
  await new Promise<void>((resolve) => {
    if (chatWorkerReady) resolve();
    else chatWorkerReadyResolve = resolve;
  });
}

function runChatGeneration(prompt: string): Promise<GenerateResult> {
  return new Promise((resolve, reject) => {
    if (!chatWorker) {
      reject(new Error("Chat worker not available"));
      return;
    }
    const id = ++chatGenerateId;
    activeChatGenerateId = id;
    chatGenerateResolve = resolve;
    chatGenerateReject = reject;
    chatWorker.postMessage({
      type: "GENERATE",
      id,
      prompt,
      temperature: CHAT_TEMPERATURE,
      topK: CHAT_TOP_K,
      maxNewTokens: model!.config.contextLength,
      seed: Date.now() & 0xffff,
    });
  });
}

function scheduleTracePanels(): void {
  const run = () => {
    paintT1();
    paintT3();
    paintT4();
    scheduleAttentionTrace();
  };
  if (typeof requestIdleCallback === "function")
    requestIdleCallback(run, { timeout: 800 });
  else setTimeout(run, 0);
}

function sendChat(): void {
  const input = document.querySelector<HTMLTextAreaElement>("#chat-input");
  if (!input || !model) return;
  const text = input.value;
  if (!text.trim()) return;
  input.value = "";
  chatHistory.push({ role: "user", text });
  chatHistory.push({
    role: "assistant",
    text: "",
    status: chatGenerating ? "queued" : "thinking",
    prompt: text,
  });
  paintChat();
  paintStreamStatus();
  focusChatInput();
  void processChatQueue();
}

async function processChatQueue(): Promise<void> {
  if (chatGenerating) return;
  chatGenerating = true;
  try {
    if (chatWorker) await waitChatWorkerReady();
    for (;;) {
      const turn = chatHistory.find(
        (t) =>
          t.role === "assistant" &&
          (t.status === "queued" || t.status === "thinking"),
      );
      if (!turn || !model) break;
      turn.status = "thinking";
      activeChatTurn = turn;
      chatStreamStep = 0;
      paintChat();
      paintStreamStatus();

      let result: GenerateResult;
      try {
        result = await runChatGeneration(turn.prompt ?? "");
      } catch (err) {
        turn.text = err instanceof Error ? err.message : String(err);
        turn.status = "done";
        activeChatTurn = null;
        paintChat();
        continue;
      } finally {
        activeChatTurn = null;
      }

      turn.text =
        result.text ||
        "(no output - the model predicted end-of-line right away)";
      turn.status = "done";
      lastGen = result;
      pickStep = 0;
      syncTraceTokenIds();
      paintChat();
      scheduleTracePanels();
    }
  } finally {
    chatGenerating = false;
    chatStreamStep = 0;
    activeChatTurn = null;
    paintStreamStatus();
  }
}

function focusChatInput(): void {
  const input = document.querySelector<HTMLTextAreaElement>("#chat-input");
  if (input && !input.disabled) input.focus();
}

function paintStreamStatus(): void {
  const el = document.querySelector<HTMLElement>("#chat-status");
  if (!el) return;
  const queued = chatHistory.filter((t) => t.status === "queued").length;
  const parts: string[] = [];
  if (chatGenerating)
    parts.push(
      chatStreamStep > 0 ? `Generating… token ${chatStreamStep}` : "Thinking…",
    );
  if (queued > 0) parts.push(`${queued} in queue`);
  el.textContent = parts.join(" · ");
}

function syncTraceTokenIds(): void {
  if (!lastGen) {
    traceTokenIds = null;
    attentionTrace = null;
    return;
  }
  traceTokenIds = lastGen.steps[pickStep]?.contextIds ?? lastGen.promptIds;
  attentionTrace = null;
}

function scheduleAttentionTrace(): void {
  const job = ++attentionJob;
  const ids = traceTokenIds;
  if (!model || !ids?.length) return;
  paintT2();
  const run = () => {
    if (job !== attentionJob || !model || traceTokenIds !== ids) return;
    attentionTrace = forwardTrace(model, ids, { captureAttention: true });
    if (job === attentionJob) paintT2();
  };
  if (typeof requestIdleCallback === "function")
    requestIdleCallback(run, { timeout: 1500 });
  else setTimeout(run, 50);
}

function updateStreamingBubble(text: string): void {
  const bubble = document.querySelector<HTMLElement>(
    ".chat-msg-streaming .chat-bubble",
  );
  if (bubble) {
    bubble.innerHTML = `${escapeHtml(text)}<span class="chat-cursor" aria-hidden="true">▋</span>`;
    bubble
      .closest(".chat-log")
      ?.scrollTo(0, bubble.closest(".chat-log")!.scrollHeight);
    return;
  }
  paintChat();
}

function paintChat(): void {
  const log = document.querySelector<HTMLElement>("#chat-log");
  if (!log) return;
  if (chatHistory.length === 0) {
    log.innerHTML = `<div class="chat-empty">${HINT.chat}</div>`;
    return;
  }
  const dots =
    '<span class="typing-dots" aria-label="thinking"><span></span><span></span><span></span></span>';
  log.innerHTML = chatHistory
    .map((m) => {
      if (m.role === "user")
        return `<div class="chat-msg chat-msg-user"><div class="chat-bubble">${escapeHtml(m.text)}</div></div>`;
      if (m.status === "queued") {
        return `<div class="chat-msg chat-msg-assistant chat-msg-queued"><div class="chat-bubble"><span class="chat-queued">queued</span>${dots}</div></div>`;
      }
      if (m.status === "thinking") {
        return `<div class="chat-msg chat-msg-assistant chat-msg-streaming"><div class="chat-bubble">${dots}</div></div>`;
      }
      if (m.status === "streaming") {
        return `<div class="chat-msg chat-msg-assistant chat-msg-streaming"><div class="chat-bubble">${escapeHtml(m.text)}<span class="chat-cursor" aria-hidden="true">▋</span></div></div>`;
      }
      return `<div class="chat-msg chat-msg-assistant"><div class="chat-bubble">${escapeHtml(m.text)}</div></div>`;
    })
    .join("");
  log.scrollTop = log.scrollHeight;
}

/* ---------- panels ---------- */

function heat(v: number, max: number): string {
  const a = max > 0 ? Math.min(1, Math.abs(v) / max) : 0;
  return v >= 0
    ? `rgba(220,38,38,${(a * 0.75).toFixed(2)})`
    : `rgba(37,99,235,${(a * 0.75).toFixed(2)})`;
}

function emptyPanel(text: string): string {
  return `<div class="muted panel-empty">${text}</div>`;
}

function paintT1(): void {
  const el = document.querySelector<HTMLElement>("#t1");
  if (!el) return;
  if (!model || !traceTokenIds?.length) {
    el.innerHTML = emptyPanel(
      model
        ? "Send a chat message to see its tokens."
        : "Model too large for the browser.",
    );
    return;
  }
  const all = traceTokenIds;
  const ids = all.slice(-MAX_TRACE_TOKENS);
  const offset = all.length - ids.length;
  const dims = Math.min(8, model.config.dModel);
  const embs = embeddingRows(model, ids, dims);
  const rows = ids.map((id, i) => ({ id, pos: offset + i, emb: embs[i] }));
  const max = Math.max(1e-9, ...rows.flatMap((r) => r.emb.map(Math.abs)));
  el.innerHTML = `
    <table class="matrix">
      <thead><tr><th>pos</th><th>token</th><th>id</th>${Array.from({ length: dims }, (_, j) => `<th>e${j}</th>`).join("")}</tr></thead>
      <tbody>${rows
        .map(
          (r) =>
            `<tr><td>${r.pos}</td><td class="tok">${displayToken(r.id)}</td><td>${r.id}</td>${r.emb
              .map(
                (v) =>
                  `<td style="background:${heat(v, max)}">${v.toFixed(2)}</td>`,
              )
              .join("")}</tr>`,
        )
        .join("")}</tbody>
    </table>
    <div class="muted panel-note">vocab ${tokenizer.vocab.length} · d_model ${model.config.dModel} (first ${dims} dims shown)</div>`;
}

function paintT2(): void {
  const el = document.querySelector<HTMLElement>("#t2");
  if (!el) return;
  if (!model || !traceTokenIds?.length) {
    el.innerHTML = emptyPanel("Send a chat message to see attention.");
    return;
  }
  if (!attentionTrace?.attention.length) {
    el.innerHTML = emptyPanel("Computing attention…");
    return;
  }
  const { layers, heads } = model.config;
  attnLayer = Math.min(attnLayer, layers - 1);
  attnHead = Math.min(attnHead, heads - 1);
  const full = attentionTrace.attention[attnLayer][attnHead];
  const n = Math.min(full.length, MAX_TRACE_TOKENS);
  const start = full.length - n;
  const labels = attentionTrace.tokenIds.slice(start).map(displayToken);
  const opts = (count: number, sel: number) =>
    Array.from(
      { length: count },
      (_, i) =>
        `<option value="${i}" ${i === sel ? "selected" : ""}>${i}</option>`,
    ).join("");
  el.innerHTML = `
    <div class="panel-controls">
      layer <select id="attn-layer">${opts(layers, attnLayer)}</select>
      head <select id="attn-head">${opts(heads, attnHead)}</select>
    </div>
    <table class="matrix attn">
      <thead><tr><th></th>${labels.map((l) => `<th class="tok">${l}</th>`).join("")}</tr></thead>
      <tbody>${labels
        .map((l, qi) => {
          const row = full[start + qi].slice(start);
          return `<tr><th class="tok sticky-col">${l}</th>${row
            .map(
              (v) =>
                `<td style="background:rgba(22,163,74,${(v * 0.85).toFixed(2)})">${v >= 0.005 ? Math.round(v * 100) : ""}</td>`,
            )
            .join("")}</tr>`;
        })
        .join("")}</tbody>
    </table>`;
  el.querySelector<HTMLSelectElement>("#attn-layer")!.onchange = (e) => {
    attnLayer = Number((e.target as HTMLSelectElement).value);
    paintT2();
  };
  el.querySelector<HTMLSelectElement>("#attn-head")!.onchange = (e) => {
    attnHead = Number((e.target as HTMLSelectElement).value);
    paintT2();
  };
}

function paintT3(): void {
  const el = document.querySelector<HTMLElement>("#t3");
  if (!el) return;
  if (!model || !lastGen || lastGen.steps.length === 0) {
    el.innerHTML = emptyPanel(
      lastGen
        ? "The model stopped immediately (predicted end-of-line)."
        : "Send a chat message to see how each token is picked.",
    );
    return;
  }
  const steps = lastGen.steps;
  const cur = steps[pickStep];
  const tokenLabel = (t: string) =>
    t === " " ? "␣" : t.startsWith("<") ? t.slice(1, -1) : escapeHtml(t);
  el.innerHTML = `
    <div class="step-row">${steps
      .map(
        (s, i) =>
          `<button type="button" class="step-btn ${i === pickStep ? "on" : ""}" data-step="${i}">B${i + 1} ${tokenLabel(s.chosenText)}</button>`,
      )
      .join("")}</div>
    <table class="matrix pick">
      <thead><tr><th>#</th><th>token</th><th>prob</th><th></th></tr></thead>
      <tbody>${cur.top
        .map(
          (t, i) =>
            `<tr class="${t.id === cur.chosenId ? "chosen" : ""}"><td>${i + 1}</td><td class="tok">${tokenLabel(t.text)}</td><td>${(t.prob * 100).toFixed(1)}%</td><td><div class="bar" style="width:${Math.max(1, t.prob * 100).toFixed(0)}%"></div></td></tr>`,
        )
        .join("")}</tbody>
    </table>
    <div class="pick-formula">logits = LayerNorm(h_last) · Wteᵀ → softmax → sample (T=${CHAT_TEMPERATURE}) → <b>${tokenLabel(cur.chosenText)}</b></div>`;
  el.querySelectorAll<HTMLButtonElement>("[data-step]").forEach((b) => {
    b.onclick = () => {
      pickStep = Number(b.dataset.step);
      syncTraceTokenIds();
      paintT1();
      paintT3();
      paintT4();
      scheduleAttentionTrace();
    };
  });
}

function paintT4(): void {
  const el = document.querySelector<HTMLElement>("#t4");
  if (!el) return;
  if (!model) {
    el.innerHTML = emptyPanel("Model too large for the browser.");
    return;
  }
  const names = [...model.params.keys()];
  if (!names.includes(weightName)) weightName = names[0];
  const focusToken = lastGen ? (lastGen.steps[pickStep]?.chosenId ?? -1) : -1;

  if (weightName === "wte") {
    const wte = sampleWteDisplay(model, 16, 12, focusToken);
    const max = Math.max(1e-9, ...wte.values.flat().map(Math.abs));
    const colLabels = wte.tokenIds.map((id) => displayToken(id));
    el.innerHTML = `
      <div class="panel-controls">
        <select id="weight-name">${names.map((n) => `<option ${n === weightName ? "selected" : ""}>${n}</option>`).join("")}</select>
        <span class="muted">wte [vocab × d_model] · columns = tokens · rows = embedding dims${wte.highlightCol >= 0 ? ` · highlighted = B${pickStep + 1}` : ""}</span>
      </div>
      <table class="matrix weights-num">
        <thead><tr><th class="sticky-col"></th>${colLabels.map((l, ci) => `<th class="tok ${ci === wte.highlightCol ? "weight-col-hl" : ""}">${l}</th>`).join("")}</tr></thead>
        <tbody>${wte.values
          .map(
            (row, di) =>
              `<tr><th class="sticky-col">e${di}</th>${row
                .map(
                  (v, ci) =>
                    `<td class="${ci === wte.highlightCol ? "weight-col-hl" : ""}" style="background:${heat(v, max)}">${v.toFixed(3)}</td>`,
                )
                .join("")}</tr>`,
          )
          .join("")}</tbody>
      </table>`;
  } else {
    const sample = sampleParam(model, weightName, 16, 12);
    const max = Math.max(1e-9, ...sample.values.flat().map(Math.abs));
    const colHeaders =
      sample.cols > 0
        ? `<thead><tr><th class="sticky-col"></th>${Array.from({ length: sample.cols }, (_, j) => `<th>c${j}</th>`).join("")}</tr></thead>`
        : "";
    el.innerHTML = `
      <div class="panel-controls">
        <select id="weight-name">${names.map((n) => `<option ${n === weightName ? "selected" : ""}>${n}</option>`).join("")}</select>
        <span class="muted">shape [${sample.shape.join("×")}] · top-left ${sample.rows}×${sample.cols}</span>
      </div>
      <table class="matrix weights-num">
        ${colHeaders}
        <tbody>${sample.values
          .map(
            (row, ri) =>
              `<tr><th class="sticky-col">r${ri}</th>${row
                .map(
                  (v) =>
                    `<td style="background:${heat(v, max)}">${v.toFixed(3)}</td>`,
                )
                .join("")}</tr>`,
          )
          .join("")}</tbody>
      </table>`;
  }

  el.querySelector<HTMLSelectElement>("#weight-name")!.onchange = (e) => {
    weightName = (e.target as HTMLSelectElement).value;
    paintT4();
  };
}

function paintPanels(): void {
  paintT1();
  paintT2();
  paintT3();
  paintT4();
}

/* ---------- pages ---------- */

function render(): void {
  if (showSettings) {
    renderSettingsPage(
      app,
      { shape, corpusText },
      {
        onApply: (s) => {
          stopViTraining();
          shape = s.shape;
          corpusText = s.corpusText;
          rebuild();
          chatHistory = [];
          showSettings = false;
          render();
        },
        onBack: () => {
          showSettings = false;
          render();
        },
      },
    );
    return;
  }
  renderMain();
}

function renderMain(): void {
  const cfg = fullConfig();
  const t = trainability(cfg);
  app.innerHTML = `
    <header class="top">
      <strong>Tiny GPT</strong>
      <span class="muted">${formatCount(paramCount(cfg))} params · ${cfg.layers} layers · d ${cfg.dModel} · ${cfg.heads} heads · ctx ${cfg.contextLength}</span>
      ${badgeHtml(t)}
      <span class="flow-line muted">${HINT.flow}</span>
      <div class="guide-wrap">
        <button id="toggle-guide" class="link-btn ${showGuide ? "on" : ""}">How it works</button>
        ${showGuide ? `<div class="guide-panel">${GUIDE_SECTIONS.map((s) => `<section><h4>${s.title}</h4><p>${s.body}</p></section>`).join("")}</div>` : ""}
      </div>
      <button id="open-settings" class="sec top-btn">Settings</button>
    </header>
    <aside class="left">
      <div class="box chat-box">
        ${boxHeader("Chat")}
        <div id="chat-log" class="chat-log"></div>
        <div id="chat-status" class="chat-status muted"></div>
        <div class="chat-compose">
          <textarea id="chat-input" class="chat-input" rows="2" placeholder="Type anything… e.g. 2+3= or hello" ${model ? "" : "disabled"}></textarea>
          <div class="chat-actions">
            <button type="button" id="send-chat" class="send-btn" ${model ? "" : "disabled"}>Send</button>
            <button type="button" id="clear-chat" class="link-btn">Clear</button>
          </div>
        </div>
      </div>
      <div class="box train-box">
        ${boxHeader("Train", HINT.train)}
        ${
          t === "cli"
            ? `<div class="field-hint">This model is too large to train in the browser. Use the CLI:</div><pre class="cli-cmd">${escapeHtml(cliCommand(shape))}</pre>`
            : `<div class="btn-row">
                <button id="train-btn">Train</button>
                <button id="pause-btn" class="sec" disabled>Pause</button>
                <button id="stop-btn" class="sec" disabled>Stop</button>
                <button id="export-btn" class="sec" disabled>Export version</button>
                <button id="reset-btn" class="sec">Reset weights</button>
              </div>
              <div class="vi-train">
                <div class="vi-train-row">
                  <input id="vi-seed" type="url" class="wiki-url-input" value="${escapeHtml(viSeed)}" spellcheck="false" title="Start article (vi.wikipedia.org)" />
                  <input id="vi-steps" type="number" min="10" step="10" class="small-input" value="${viStepsPerRound}" title="Steps per article" />
                  <button type="button" id="vi-start">Start Vietnamese training</button>
                </div>
                <div id="vi-status" class="field-hint"></div>
              </div>
              <div class="export-load-row">
                <select id="export-load" class="export-load-select" title="Saved models in models/exports/"></select>
                <button type="button" id="export-load-btn" class="sec">Load</button>
                <button type="button" id="export-refresh-btn" class="link-btn">Refresh</button>
              </div>
              <div class="field-hint">${HINT.exportModel}${lastExportName ? ` · Last: <code>${escapeHtml(lastExportName)}</code>` : ""}${loadedExportId ? ` · Loaded: <code>${escapeHtml(loadedExportId)}</code>` : ""}</div>`
        }
        <div class="stats">step: <b id="step">${step}</b> loss: <b id="loss">0</b> val loss: <b id="val-loss">0</b> val char acc: <b id="val-acc">0</b></div>
        <div class="box-hint">${HINT.stats} ${HINT.statsChart}</div>
        <div class="stats muted">${trainLines.length} train / ${valLines.length} val lines · vocab ${tokenizer.vocab.length} · lr ${shape.learningRate} · batch ${shape.batchSize} · backend <span id="backend">${backendName}</span></div>
        <div id="train-status" class="muted"></div>
        <div id="chart-row" class="chart-wrap"></div>
      </div>
    </aside>
    <main class="right">
      <div class="box table-box">${panelHeader("T1 · Tokens → embeddings", TOOLTIP.tooltipT1)}<div id="t1" class="panel-scroll"></div></div>
      <div class="box table-box">${panelHeader("T2 · Attention", TOOLTIP.tooltipT2)}<div id="t2" class="panel-scroll"></div></div>
      <div class="box table-box pick-box">${panelHeader("T3 · Next-token pick", TOOLTIP.tooltipT3)}<div id="t3" class="panel-scroll"></div></div>
      <div class="box table-box bias-box">${panelHeader("T4 · Weight matrix", TOOLTIP.tooltipT4)}<div id="t4" class="panel-scroll"></div></div>
    </main>
  `;
  bindMain();
  paintChat();
  paintTrainState();
  paintPanels();
  void refreshExportCatalog();
}

function bindMain(): void {
  const q = <T extends HTMLElement>(sel: string) => app.querySelector<T>(sel);
  q<HTMLButtonElement>("#open-settings")!.onclick = () => {
    showSettings = true;
    render();
  };
  q<HTMLButtonElement>("#toggle-guide")!.onclick = () => {
    showGuide = !showGuide;
    render();
  };
  const chatInput = q<HTMLTextAreaElement>("#chat-input")!;
  chatInput.onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void sendChat();
    }
  };
  q<HTMLButtonElement>("#send-chat")!.onclick = () => void sendChat();
  q<HTMLButtonElement>("#clear-chat")!.onclick = () => {
    const pending = chatHistory.some(
      (t) => t.role === "assistant" && t.status !== "done",
    );
    if (pending) return;
    chatHistory = [];
    paintChat();
    paintStreamStatus();
  };
  const trainBtn = q<HTMLButtonElement>("#train-btn");
  if (trainBtn) trainBtn.onclick = startTraining;
  const stopBtn = q<HTMLButtonElement>("#stop-btn");
  if (stopBtn) stopBtn.onclick = requestStop;
  const pauseBtn = q<HTMLButtonElement>("#pause-btn");
  if (pauseBtn)
    pauseBtn.onclick = () => (paused ? resumeTraining() : requestPause());
  const viStartBtn = q<HTMLButtonElement>("#vi-start");
  if (viStartBtn) viStartBtn.onclick = () => void startViTraining();
  const exportBtn = q<HTMLButtonElement>("#export-btn");
  if (exportBtn) exportBtn.onclick = () => void saveModelExport();
  const exportLoadBtn = q<HTMLButtonElement>("#export-load-btn");
  if (exportLoadBtn)
    exportLoadBtn.onclick = () => {
      const id = q<HTMLSelectElement>("#export-load")!.value;
      void loadSelectedExport(id);
    };
  const exportRefreshBtn = q<HTMLButtonElement>("#export-refresh-btn");
  if (exportRefreshBtn) exportRefreshBtn.onclick = () => void refreshExportCatalog();
  const resetBtn = q<HTMLButtonElement>("#reset-btn");
  if (resetBtn)
    resetBtn.onclick = () => {
      if (training || viActive) return;
      rebuild();
      render();
    };
}

app.innerHTML = `<div class="muted panel-empty">Starting TensorFlow.js…</div>`;
void initTfBackend().then(async (name) => {
  backendName = `${name} (UI)`;
  rebuild();
  exportCatalog = await fetchExportList();
  const urlModel = new URLSearchParams(location.search).get("model");
  const envModel = import.meta.env.TINY_GPT_MODEL_ID;
  const autoLoad = urlModel || envModel || "";
  render();
  if (autoLoad) await loadSelectedExport(autoLoad);
});
