import {
  foundationReady,
  planViPhase,
  chatReady,
  newViCurriculum,
  restoreViCurriculum,
  VI_FOUNDATION_ID,
  BOS_ID,
  EOS_ID,
  PAD_ID,
  browserModelIssue,
  normalizeConfig,
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
  generateAsync,
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
  type TrainWorkerMessage,
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
  appendViTrainLines,
  initViCorpusHoldout,
  usesViVocab,
  viTokenizer,
  viTrainingConfig,
  viVocabSize,
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
  VI_LAST_MODEL_KEY,
  fetchExportList,
  loadExportById,
  saveModel,
  saveModelSnapshot,
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
let modelIo: "saving" | "loading" | null = null;
let syncedStep = 0;
let syncedFoundationSteps = 0;
let step = 0;
let lastLoss = -1;
let lastValLoss = -1;
let lastValAcc = -1;
let lossHistory: number[] = [];
let accHistory: number[] = [];
let lossEma = -1;
const LOSS_EMA_ALPHA = 0.85;

interface ChatSession { id: string; title: string; turns: ChatTurn[] }
const CHAT_STORE = "tiny-chat-sessions-v1";
let sessions: ChatSession[] = [];
let activeSession = "";
try {
  const saved = JSON.parse(localStorage.getItem(CHAT_STORE) ?? "null");
  if (saved && Array.isArray(saved.sessions)) {
    sessions = saved.sessions.filter((x: ChatSession) => typeof x.id === "string" && typeof x.title === "string" && Array.isArray(x.turns));
    for (const session of sessions) session.turns = session.turns.filter(t => (t.role === "user" || t.role === "assistant") && typeof t.text === "string").map(t => ({...t, status: "done"}));
    activeSession = saved.activeSession;
  }
} catch { /* Malformed or unavailable browser storage. */ }
if (!sessions.length) sessions.push({ id: crypto.randomUUID(), title: "New chat", turns: [] });
if (!sessions.some(s => s.id === activeSession)) activeSession = sessions[0]!.id;
let chatHistory: ChatTurn[] = sessions.find(s => s.id === activeSession)!.turns;
function persistChat(): void {
  const session = sessions.find(s => s.id === activeSession)!;
  session.turns = chatHistory;
  session.title = chatHistory.find(t => t.role === "user")?.text.slice(0, 50) || "New chat";
  try { localStorage.setItem(CHAT_STORE, JSON.stringify({ sessions, activeSession })); }
  catch { setStatus("Không lưu được history: bộ nhớ trình duyệt đầy hoặc bị chặn."); }
}
function newChat(): void {
  if (chatGenerating) return;
  persistChat();
  const session = { id: crypto.randomUUID(), title: "New chat", turns: [] };
  sessions.unshift(session); activeSession = session.id; chatHistory = session.turns;
  lastGen = null;
  render();
}
let chatGenerating = false;
let chatStreamStep = 0;
let chatGenerateId = 0;
let activeChatGenerateId = 0;
let activeChatTurn: ChatTurn | null = null;
let chatAbort: AbortController | null = null;
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

function chatTemperature(): number {
  return usesViVocab(tokenizer) ? 0.12 : CHAT_TEMPERATURE;
}

function rebuild(fixedTokenizer?: Tokenizer): void {
  stopViTraining();
  viCurriculum = newViCurriculum();
  viTrainLines = [];
  viValLines = [];
  viRound = 0;
  loadedExportId = lastExportName = "";
  syncedStep = syncedFoundationSteps = 0;
  stopWorkerNow();
  if (model) disposeModel(model);
  tokenizer = fixedTokenizer ?? buildTokenizer(corpusText);
  const split = splitCorpus(parseCorpus(corpusText), 0.1, shape.seed);
  trainLines = split.train;
  valLines = split.val;
  model =
    browserModelIssue(fullConfig()) !== null
      ? null
      : createModel(fullConfig(), tokenizer);
  if (model) warmupForward(model);
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
    viCurriculum: usesViVocab(tokenizer) ? { ...viCurriculum } : undefined,
    loss: lastLoss,
    valLoss: lastValLoss,
    valAcc: lastValAcc,
    exportedAt: new Date().toISOString(),
    label: usesViVocab(tokenizer) ? "vi" : "custom",
  };
}

async function saveModelExport(
  weightsFromWorker?: Float32Array,
): Promise<boolean> {
  if (modelIo || training || viPreparing || pausing) return false;
  if (!model || step <= 0) {
    setStatus("Train at least one step before saving.");
    return false;
  }
  if (weightsFromWorker) importWeights(model, weightsFromWorker);
  modelIo = "saving";
  paintTrainState();
  setStatus("Saving to models/…");
  try {
    const meta = exportMeta();
    if (weightsFromWorker) {
      const saved = await saveModelSnapshot(model, meta, weightsFromWorker);
      lastExportName = saved.id;
    } else {
      lastExportName = await saveModel(model, meta);
    }
    exportCatalog = await fetchExportList().catch(() => exportCatalog);
    loadedExportId = lastExportName;
    try {
      localStorage.setItem(VI_LAST_MODEL_KEY, lastExportName);
    } catch {
      /* private mode */
    }
    const nextHint =
      usesViVocab(tokenizer) &&
      foundationReady(viCurriculum) &&
      viCurriculum.phase === "foundation"
        ? " · Đã đủ mốc nền: bạn có thể học tiếp Wikipedia hoặc tăng số bước."
        : "";
    setStatus(`Saved models/${lastExportName}${nextHint}`);
    paintTrainState();
    paintStreamStatus();
    renderExportControls();
    return true;
  } catch (err) {
    setStatus(
      `Save failed: ${err instanceof Error ? err.message : String(err)}`,
    );
    paintTrainState();
    return false;
  } finally {
    modelIo = null;
    paintTrainState();
  }
}

async function loadSelectedExport(id: string): Promise<boolean> {
  if (!id) {
    setStatus("Pick a saved model in the list first.");
    return false;
  }
  if (
    training ||
    viPreparing ||
    viActive ||
    paused ||
    modelIo ||
    chatGenerating
  ) {
    setStatus("Stop training before loading a model.");
    return false;
  }
  stopWorkerNow();
  stopViTraining();
  paused = false;
  ++chatGenerateId;
  activeChatGenerateId = 0;
  chatGenerating = false;
  activeChatTurn = null;
  modelIo = "loading";
  paintTrainState();
  setStatus(`Loading ${id}…`);
  try {
    const { manifest, weights, meta } = await loadExportById(id);
    const cfg = normalizeConfig({
      ...manifest.config,
      vocabSize: manifest.vocab.length,
    });
    const issue = browserModelIssue(cfg);
    if (issue) throw new Error(issue);
    const nextModel = deserializeModel(manifest, weights);
    try {
      warmupForward(nextModel);
    } catch (err) {
      disposeModel(nextModel);
      throw err;
    }
    if (model) disposeModel(model);
    model = nextModel;
    tokenizer = model.tokenizer;
    warmupForward(model);
    step = meta.step ?? 0;
    syncedStep = step;
    viCurriculum = restoreViCurriculum(meta.viCurriculum);
    syncedFoundationSteps = viCurriculum.foundationSteps;
    lossHistory = [];
    accHistory = [];
    viSeed = viCurriculum.wikiSeed;
    viTrainLines = [];
    viValLines = [];
    viRound = 0;
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
    trainLines = usesViVocab(tokenizer) ? [] : split.train;
    valLines = usesViVocab(tokenizer) ? [] : split.val;
    // Keep persisted conversations when changing models.
    lastGen = null;
    traceTokenIds = null;
    attentionTrace = null;
    pickStep = 0;
    if (usesViVocab(tokenizer)) {
      try {
        localStorage.setItem(VI_LAST_MODEL_KEY, id);
      } catch {
        /* private mode */
      }
    }
    render();
    const viOk = usesViVocab(tokenizer);
    const vocabHint = viOk
      ? "VI charset OK"
      : `vocab ${tokenizer.vocab.length} (expected ${viVocabSize()} for VI)`;
    setStatus(
      `Loaded ${id} · step ${step} · ${vocabHint}. Chat uses this checkpoint.`,
    );
    return true;
  } catch (err) {
    setStatus(
      `Load failed: ${err instanceof Error ? err.message : String(err)}`,
    );
    return false;
  } finally {
    modelIo = null;
    paintTrainState();
  }
}

function renderExportControls(): void {
  const sel = document.querySelector<HTMLSelectElement>("#export-load");
  if (!sel) return;
  const previous = sel.value;
  const sorted = [...exportCatalog].sort((a, b) =>
    (Date.parse(b.exportedAt) || 0) - (Date.parse(a.exportedAt) || 0) || b.id.localeCompare(a.id));
  const selected = sorted.some(e => e.id === previous) ? previous : sorted[0]?.id ?? "";
  sel.innerHTML =
    `<option value="">— load saved version —</option>` +
    sorted
      .map((e) => {
        const tag = e.viCharset
          ? "VI"
          : e.vocabSize
            ? `⚠ vocab ${e.vocabSize} (not VI)`
            : "⚠ unknown vocab";
        const acc =
          e.valAcc >= 0 ? ` · val ${(e.valAcc * 100).toFixed(0)}%` : "";
        return `<option value="${escapeHtml(e.id)}" ${e.id === selected ? "selected" : ""}>${tag} · step ${e.step}${acc} · ${escapeHtml(e.version)}</option>`;
      })
      .join("");
}

async function refreshExportCatalog(): Promise<void> {
  try {
    exportCatalog = await fetchExportList();
    renderExportControls();
  } catch (err) {
    setStatus(
      `Không đọc được danh sách model: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

function stopWorkerNow(): void {
  worker?.terminate();
  worker = null;
  training = false;
  stopping = false;
}

function startTraining(): void {
  if (!model || training || viPreparing || modelIo || chatGenerating) return;
  if (usesViVocab(tokenizer) && !viActive) {
    void startViTraining();
    return;
  }
  if (trainLines.length === 0) {
    setStatus("No training corpus loaded.");
    return;
  }
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
    endStep: viActive ? viCurriculum.plannedEndStep : undefined,
  };
  w.onmessage = (e: MessageEvent<TrainWorkerProgress>) => {
    if (w !== worker) return;
    onProgress(e.data);
  };
  w.onerror = (e) => {
    if (w !== worker) return;
    stopWorkerNow();
    stopViTraining();
    step = syncedStep;
    viCurriculum.foundationSteps = syncedFoundationSteps;
    setStatus(`Worker error: ${e.message}`);
    paintTrainState();
  };
  w.postMessage(req, [weights.buffer]);
  paintTrainState();
}

/* ---------- Vietnamese Wikipedia (continuous corpus) ---------- */

let viCurriculum = newViCurriculum();
let viPreparing = false;
let viRun = 0;
let viActive = false;
let viRound = 0;
let viSeed = VI_DEFAULT_SEED;
let viCrawler: ViWikiCrawler | null = null;
let viNext: Promise<ViArticle> | null = null;
let viCurrent = "";
let viNextTitle = "";
/** Growing train set; val is fixed at start (hold-out orthography check). */
let viTrainLines: string[] = [];
let viValLines: string[] = [];
let viIngesting = false;

let pausing = false;
let paused = false;

function syncViCorpusToWorker(): void {
  if (!worker || !training) return;
  const msg: TrainWorkerMessage = {
    type: "UPDATE_CORPUS",
    trainLines: viTrainLines,
  };
  worker.postMessage(msg);
}

function requestPause(): void {
  if (paused || pausing) return;
  if (!worker) {
    paused = true;
    setStatus(`Paused at step ${step}.`);
    paintTrainState();
    return;
  }
  pausing = true;
  worker.postMessage({ type: "PAUSE" });
  setStatus("Pausing…");
  paintTrainState();
}

function resumeTraining(): void {
  if (!paused || !model || modelIo || chatGenerating) return;
  paused = false;
  training = true;
  worker?.postMessage({ type: "RESUME" });
  setStatus("Resuming…");
  paintTrainState();
}

function stopViTraining(): void {
  paused = false;
  pausing = false;
  viActive = false;
  viPreparing = false;
  viRun += 1;
  viIngesting = false;
  viNext = null;
  viCrawler?.stop();
  viCrawler = null;
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

async function ingestViArticles(): Promise<void> {
  if (viIngesting) return;
  viIngesting = true;
  const run = viRun;
  try {
    while (viActive && viCrawler && run === viRun) {
      if (paused || pausing) {
        await new Promise((resolve) => setTimeout(resolve, 200));
        continue;
      }
      if (!viNext) prefetchViArticle();
      if (!viNext) break;
      let article: ViArticle;
      try {
        article = await viNext;
      } catch (err) {
        if (viActive && run === viRun) {
          setStatus(
            `Corpus fetch ended: ${err instanceof Error ? err.message : String(err)}. Training continues on ${viTrainLines.length} lines.`,
          );
        }
        break;
      }
      if (!viActive || run !== viRun) break;
      prefetchViArticle();

      viRound += 1;
      viCurrent = article.title;
      viTrainLines = appendViTrainLines(
        viTrainLines,
        viValLines,
        article.lines,
      );
      trainLines = viTrainLines;
      syncViCorpusToWorker();
      setStatus(
        `VI corpus: ${viTrainLines.length} train lines (hold-out val ${viValLines.length}) · article ${viRound} “${article.title}”`,
      );
      paintTrainState();
    }
  } finally {
    if (run === viRun) viIngesting = false;
  }
}

/** Keep the explicitly loaded VI model; otherwise initialize the fixed VI charset. */
async function ensureViModel(): Promise<boolean> {
  const cfg = viTrainingConfig(shape);
  const issue = browserModelIssue(cfg);
  if (issue) throw new Error(issue);
  if (model && usesViVocab(tokenizer)) return true;
  const nextTokenizer = viTokenizer();
  const nextModel = createModel(cfg, nextTokenizer);
  try {
    warmupForward(nextModel);
  } catch (err) {
    disposeModel(nextModel);
    throw err;
  }
  stopWorkerNow();
  if (model) disposeModel(model);
  model = nextModel;
  tokenizer = nextTokenizer;
  const { vocabSize: _v, ...selectedShape } = cfg;
  shape = selectedShape;
  step = syncedStep = syncedFoundationSteps = 0;
  viCurriculum = newViCurriculum();
  viRound = 0;
  lastLoss = lastValLoss = lastValAcc = -1;
  lossHistory = [];
  accHistory = [];
  // Keep persisted conversations when changing models.
  lastGen = null;
  traceTokenIds = null;
  attentionTrace = null;
  loadedExportId = lastExportName = "";
  render();
  return true;
}

async function loadFoundationCorpus(): Promise<{
  train: string[];
  val: string[];
}> {
  const response = await fetch("/api/corpus/vi-foundation");
  if (!response.ok)
    throw new Error(
      "Thiếu corpus nền. Chạy npm run fetch:vi-foundation rồi npm run dev:tiny.",
    );
  const text = await response.text();
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(text),
  );
  const hash = [...new Uint8Array(digest)]
    .map((v) => v.toString(16).padStart(2, "0"))
    .join("");
  if (!VI_FOUNDATION_ID.endsWith(":" + hash))
    throw new Error(
      "Corpus nền khác phiên bản đã ghi nhận. Hãy chạy lại npm run fetch:vi-foundation.",
    );
  const split = initViCorpusHoldout(parseCorpus(text));
  if (!split.train.length || !split.val.length)
    throw new Error("Corpus nền không đủ dữ liệu train/validation.");
  // Assign only in the caller, so a cancelled request cannot alter a new run.
  return split;
}

/** Phase 1: local monolingual corpus only. Never fetch Wikipedia here. */
async function startViTraining(): Promise<void> {
  if (
    training ||
    viActive ||
    viPreparing ||
    paused ||
    modelIo ||
    chatGenerating
  )
    return;
  const requestedSteps = Number(
    document.querySelector<HTMLInputElement>("#vi-foundation-steps")?.value ??
      viCurriculum.targetSteps,
  );
  if (!Number.isSafeInteger(requestedSteps) || requestedSteps < 1) {
    setStatus("Số bước corpus nền phải là số nguyên dương.");
    return;
  }
  viPreparing = true;
  const run = ++viRun;
  paintTrainState();
  try {
    if (!(await ensureViModel()))
      throw new Error("Model quá lớn để train trong browser.");
    if (run !== viRun) return;
    viCurriculum.targetSteps = requestedSteps;
    lastValLoss = lastValAcc = -1;
    const stepsInput = document.querySelector<HTMLInputElement>(
      "#vi-foundation-steps",
    );
    if (stepsInput) stepsInput.value = String(requestedSteps);
    if (foundationReady(viCurriculum)) {
      setStatus(
        "Đã đủ mốc corpus nền. Tăng số bước để luyện thêm hoặc chuyển sang Wikipedia.",
      );
      return;
    }
    setStatus("Đang đọc corpus tiếng Việt trong train-data…");
    const foundation = await loadFoundationCorpus();
    if (run !== viRun) return;
    viTrainLines = foundation.train;
    viValLines = foundation.val;
    trainLines = viTrainLines;
    valLines = viValLines;
    planViPhase(viCurriculum, "foundation", step);
    viRound = 0;
    viActive = true;
    viPreparing = false;
    startTraining();
    setStatus("Giai đoạn 1: học corpus tiếng Việt. Chưa tải Wikipedia.");
  } catch (err) {
    if (run === viRun)
      setStatus(err instanceof Error ? err.message : String(err));
  } finally {
    if (run === viRun) {
      viPreparing = false;
      paintTrainState();
    }
  }
}

async function startChatTraining(): Promise<void> {
  if (training || viActive || viPreparing || paused || modelIo || chatGenerating || !model || !foundationReady(viCurriculum)) return;
  viPreparing = true;
  const run = ++viRun;
  try {
    const [foundation, response] = await Promise.all([loadFoundationCorpus(), fetch("/api/corpus/vi-chat-basic")]);
    if (!response.ok) throw new Error("Thiếu corpus hội thoại cơ bản.");
    const chat = parseCorpus(await response.text());
    if (run !== viRun) return;
    // Chat must dominate this phase: appending 33 examples to 82k news lines
    // made nearly every sampled window news. Keep news only as a separate
    // regression validation set; do not call it chat accuracy.
    viTrainLines = [...new Set(chat.map(line => line.trim()).filter(Boolean))];
    if (!viTrainLines.length) throw new Error("Corpus hội thoại trống.");
    viValLines = foundation.val;
    trainLines = viTrainLines; valLines = viValLines;
    planViPhase(viCurriculum, "chat", step); viActive = true; viPreparing = false;
    startTraining();
    setStatus("Giai đoạn 2: train hội thoại cơ bản tiếng Việt.");
  } catch (err) { setStatus(err instanceof Error ? err.message : String(err)); }
  finally { if (run === viRun) { viPreparing = false; paintTrainState(); } }
}

/** Phase 3 is explicit: reuse weights, retain foundation data, append VI Wikipedia. */
async function startWikiTraining(): Promise<void> {
  if (
    training ||
    viActive ||
    viPreparing ||
    paused ||
    modelIo ||
    chatGenerating
  )
    return;
  if (!model || !usesViVocab(tokenizer) || !chatReady(viCurriculum)) {
    setStatus("Hãy hoàn thành foundation và hội thoại cơ bản trước khi học Wikipedia.");
    return;
  }
  const seed =
    document.querySelector<HTMLInputElement>("#vi-wiki-seed")?.value.trim() ||
    VI_DEFAULT_SEED;
  let crawler: ViWikiCrawler;
  try {
    crawler = new ViWikiCrawler(seed);
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
    return;
  }
  viPreparing = true;
  viCrawler = crawler;
  const run = ++viRun;
  paintTrainState();
  try {
    setStatus("Đang chuẩn bị corpus nền và bài Wikipedia tiếng Việt…");
    const foundation = await loadFoundationCorpus();
    if (run !== viRun) return;
    const first = await crawler.next();
    if (run !== viRun) return;
    viValLines = foundation.val;
    viTrainLines = appendViTrainLines(
      foundation.train,
      viValLines,
      first.lines,
    );
    trainLines = viTrainLines;
    valLines = viValLines;
    viSeed = seed;
    planViPhase(viCurriculum, "wikipedia", step);
    viCurriculum.wikiSeed = seed;
    viCrawler = crawler;
    viRound = 1;
    viCurrent = first.title;
    viActive = true;
    viPreparing = false;
    startTraining();
    prefetchViArticle();
    void ingestViArticles();
    setStatus(
      "Giai đoạn 3: tiếp tục weights hiện tại, thêm Wikipedia và giữ corpus nền.",
    );
  } catch (err) {
    if (run === viRun)
      setStatus(err instanceof Error ? err.message : String(err));
  } finally {
    if (run === viRun) {
      viPreparing = false;
      paintTrainState();
    }
  }
}

function paintViState(): void {
  const progress = document.querySelector<HTMLProgressElement>("#vi-progress");
  const target = Math.max(step, viCurriculum.plannedEndStep ?? (step + Math.max(0, viCurriculum.targetSteps - viCurriculum.foundationSteps)));
  if (progress) {
    progress.max = Math.max(1, target);
    progress.value = step;
  }
  const total = document.querySelector<HTMLElement>("#vi-total-progress");
  if (total) total.textContent = `Tổng tiến độ: ${step}/${target} bước · ${target > 0 ? Math.floor(step / target * 100) : 0}% · ${viCurriculum.phase === "foundation" ? "Bước 1: corpus nền" : viCurriculum.phase === "chat" ? "Bước 2: hội thoại" : "Bước 3: Wikipedia"}`;
  const el = document.querySelector<HTMLElement>("#vi-status");
  if (el) {
    const accHint =
      lastValAcc >= 0 ? ` · val ký tự ${(lastValAcc * 100).toFixed(0)}%` : "";
    if (viPreparing) {
      el.textContent = "Đang chuẩn bị dữ liệu…";
    } else if (viCurriculum.phase === "foundation") {
      el.textContent = `Corpus nền: ${viCurriculum.foundationSteps}/${viCurriculum.targetSteps} bước · ${viTrainLines.length} train / ${viValLines.length} val. Mốc bước không chứng minh model đã hiểu tiếng Việt.`;
    } else if (viCurriculum.phase === "chat") {
      el.textContent = `Hội thoại: ${viCurriculum.chatSteps}/${viCurriculum.chatTargetSteps} bước${accHint} (validation tin tức, không phải độ đúng câu trả lời)`;
    } else if (viActive) {
      el.textContent = `${paused ? "Paused · " : ""}${formatCount(paramCount(fullConfig()))} params · ${viTrainLines.length} train · val ${viValLines.length}${accHint} · ${viRound} wiki article(s) · “${viCurrent || "…"}” · queue ${viCrawler?.queued ?? 0}`;
    } else if (viRound > 0) {
      el.textContent = `Stopped · ${viTrainLines.length} lines · ${viRound} article(s)${accHint}`;
    } else {
      el.textContent = HINT.viTrain;
    }
  }
  const startBtn = document.querySelector<HTMLButtonElement>("#vi-start");
  if (startBtn)
    startBtn.disabled =
      training ||
      viActive ||
      paused ||
      viPreparing ||
      !!modelIo ||
      chatGenerating;
  const wikiBtn = document.querySelector<HTMLButtonElement>("#vi-wiki-start");
  if (wikiBtn)
    wikiBtn.disabled =
      training ||
      viActive ||
      paused ||
      viPreparing ||
      !!modelIo ||
      chatGenerating ||
      !foundationReady(viCurriculum) || !chatReady(viCurriculum);
  const chatBtn = document.querySelector<HTMLButtonElement>("#vi-chat-start");
  if (chatBtn) {
    chatBtn.disabled = training || viActive || viPreparing || !!modelIo || chatGenerating || !foundationReady(viCurriculum);
    chatBtn.title = "Train riêng các mẫu hội thoại trong train-data/vi-chat-basic.txt; kiểm tra câu trả lời sau mỗi đợt.";
  }
  for (const id of ["#vi-foundation-steps", "#vi-wiki-seed"]) {
    const input = document.querySelector<HTMLInputElement>(id);
    if (input)
      input.disabled =
        training || viActive || paused || viPreparing || !!modelIo;
  }
}

function requestStop(): void {
  if (viPreparing) {
    stopViTraining();
    setStatus("Đã hủy chuẩn bị dữ liệu.");
    paintTrainState();
    return;
  }
  pausing = false;
  if (paused && !worker) {
    paused = false;
    stopViTraining();
    setStatus("Stopped. Saving model…");
    void saveModelExport();
    paintTrainState();
    return;
  }
  paused = false;
  if (viActive) {
    stopViTraining();
    if (!worker) {
      setStatus("Stopped. Saving model…");
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
  if (msg.type === "ERROR") {
    stopWorkerNow();
    stopViTraining();
    step = syncedStep;
    viCurriculum.foundationSteps = syncedFoundationSteps;
    setStatus(
      `Train lỗi: ${msg.message ?? "unknown"}. Giữ weights gần nhất ở step ${step}.`,
    );
    paintTrainState();
    return;
  }
  if (msg.backend) backendName = `webgl (chat) · ${msg.backend} (training)`;
  if (msg.weights && model) {
    importWeights(model, msg.weights);
  }
  if (viCurriculum.phase === "foundation" && usesViVocab(tokenizer)) {
    viCurriculum.foundationSteps += Math.max(0, msg.step - step);
  } else if (viCurriculum.phase === "chat" && usesViVocab(tokenizer)) {
    viCurriculum.chatSteps += Math.max(0, msg.step - step);
  }
  step = msg.step;
  if (msg.weights) {
    syncedStep = step;
    syncedFoundationSteps = viCurriculum.foundationSteps;
  }
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
      if (msg.trainLines != null && viActive) {
        paintViState();
      }
      break;
    case "PAUSED":
      training = false;
      pausing = false;
      paused = true;
      setStatus(
        `Paused at step ${step}. Weights và optimizer được giữ; Resume để tiếp tục.`,
      );
      break;
    case "RESUMED":
      training = true;
      paused = false;
      setStatus("Training resumed.");
      break;
    case "STOPPED":
      stopWorkerNow();
      if (msg.loss >= 0) lastLoss = msg.loss;
      if (msg.valLoss >= 0) lastValLoss = msg.valLoss;
      if (msg.valAccuracy >= 0) lastValAcc = msg.valAccuracy;
      stopViTraining();
      setStatus("Stopped. Saving model…");
      void saveModelExport(msg.weights ?? undefined);
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
  const busy = training || viActive || paused || viPreparing;
  const locked = !!modelIo || chatGenerating;
  if (trainBtn) trainBtn.disabled = busy || locked || !model;
  if (stopBtn) stopBtn.disabled = !busy || stopping;
  if (pauseBtn) {
    pauseBtn.textContent = paused ? "Resume" : "Pause";
    pauseBtn.disabled =
      locked ||
      viPreparing ||
      (paused ? false : !(training || viActive) || stopping || pausing);
  }
  if (resetBtn) resetBtn.disabled = busy || locked || !model;
  if (exportBtn)
    exportBtn.disabled =
      !!modelIo ||
      training ||
      viPreparing ||
      pausing ||
      stopping ||
      !model ||
      step <= 0;
  for (const id of ["#export-load", "#export-load-btn", "#open-settings"]) {
    const control = document.querySelector<HTMLButtonElement>(id);
    if (control) control.disabled = busy || locked;
  }
  const send = document.querySelector<HTMLButtonElement>("#send-chat");
  if (send)
    send.disabled =
      !model || !!modelIo || training || viPreparing || pausing || stopping;
  const stopChat = document.querySelector<HTMLButtonElement>("#stop-chat");
  if (stopChat) stopChat.disabled = !chatGenerating;
  const corpusStats = document.querySelector<HTMLElement>("#corpus-summary");
  if (corpusStats)
    corpusStats.textContent = `${trainLines.length.toLocaleString()} train / ${valLines.length.toLocaleString()} val · vocab ${tokenizer.vocab.length} · lr ${shape.learningRate} · batch ${shape.batchSize}`;
  const chart = document.querySelector<HTMLElement>("#chart-row");
  if (chart) paintChartRow(chart, lossHistory, accHistory);
  paintViState();
  paintStreamStatus();
}

/* ---------- chat (main-thread model — same weights as Load / Train) ---------- */

async function runChatGeneration(prompt: string): Promise<GenerateResult> {
  if (!model) throw new Error("No model loaded");
  const id = ++chatGenerateId;
  activeChatGenerateId = id;
  chatAbort = new AbortController();
  try {
    return await generateAsync(model, prompt, {
      signal: chatAbort.signal,
      temperature: chatTemperature(),
      topK: CHAT_TOP_K,
      maxNewTokens: model.config.contextLength,
      seed: Date.now() & 0xffff,
      onStep: async (_step, partial, stepIndex) => {
        if (id !== activeChatGenerateId || !activeChatTurn) return;
        chatStreamStep = stepIndex + 1;
        activeChatTurn.text = partial;
        if (partial && activeChatTurn.status !== "streaming") {
          activeChatTurn.status = "streaming";
          paintChat();
        } else {
          updateStreamingBubble(partial);
        }
        paintStreamStatus();
      },
    });
  } finally {
    if (activeChatGenerateId === id) {
      activeChatGenerateId = 0;
      chatAbort = null;
    }
  }
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
  if (
    !input ||
    !model ||
    modelIo ||
    training ||
    viPreparing ||
    pausing ||
    stopping
  )
    return;
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
  paintTrainState();
  try {
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
        const preceding = chatHistory.slice(0, chatHistory.indexOf(turn));
        const prompt = preceding.map(t => t.role === "user"
          ? `Người dùng: ${t.text} = Trợ lý:` : t.text).join(" ");
        result = await runChatGeneration(prompt);
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
    paintChat();
    paintTrainState();
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
  if (loadedExportId) parts.push(`model: ${loadedExportId}`);
  else if (step > 0) parts.push(`step ${step}`);
  if (training) parts.push("Pause để chat bằng weights mới nhất");
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
  persistChat();
  const history = document.querySelector<HTMLSelectElement>("#chat-history");
  if (history) {
    history.innerHTML = sessions.map(s => `<option value="${escapeHtml(s.id)}" ${s.id === activeSession ? "selected" : ""}>${escapeHtml(s.title)}</option>`).join("");
    history.disabled = chatGenerating;
  }
  const newButton = document.querySelector<HTMLButtonElement>("#clear-chat");
  if (newButton) newButton.disabled = chatGenerating;
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
    <div class="pick-formula">logits = LayerNorm(h_last) · Wteᵀ → softmax → sample (T=${chatTemperature()}) → <b>${tokenLabel(cur.chosenText)}</b></div>`;
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
          const keepVi = usesViVocab(tokenizer) && corpusText === s.corpusText;
          corpusText = s.corpusText;
          rebuild(keepVi ? viTokenizer() : undefined);
          // Keep persisted conversations when changing models.
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
  const t = browserModelIssue(cfg) ? "cli" : trainability(cfg);
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
        <label>History <select id="chat-history" aria-label="Chat history"></select></label>
        <div class="field-hint">Context: ${model?.config.contextLength ?? shape.contextLength} token ký tự, gồm prompt và phần trả lời đang sinh. Lịch sử dài giữ phần gần nhất; history đầy đủ lưu riêng trên trình duyệt.</div>
        <div id="chat-log" class="chat-log"></div>
        <div id="chat-status" class="chat-status muted"></div>
        <div class="chat-compose">
          <textarea id="chat-input" class="chat-input" rows="2" placeholder="Type anything… e.g. 2+3= or hello" ${model ? "" : "disabled"}></textarea>
          <div class="chat-actions">
            <button type="button" id="send-chat" class="send-btn" ${model ? "" : "disabled"}>Send</button>
            <button type="button" id="stop-chat" class="sec" disabled>Dừng sinh</button>
            <button type="button" id="clear-chat" class="link-btn">New chat</button>
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
                <button id="export-btn" class="sec" disabled>Save</button>
                <button id="reset-btn" class="sec">Reset</button>
              </div>
              <div class="vi-train">
                <div class="field-hint">Dùng cấu hình hiện tại: ${shape.layers} layers · d ${shape.dModel} · ${shape.heads} heads · context ${shape.contextLength}. Đổi trong Settings trước khi train.</div>
                <div class="field-hint">Corpus nền: 81.995 câu tiếng Việt (~13,2 MB). Mốc bước là kế hoạch luyện tập, không xác nhận khả năng hiểu.</div>
                <label for="vi-foundation-steps">Số bước corpus nền / số bước thêm mỗi đợt <input id="vi-foundation-steps" type="number" min="1" step="1" value="${viCurriculum.targetSteps}"></label>
                <button type="button" id="vi-start" class="vi-start-btn">1. Train corpus tiếng Việt</button>
                <button type="button" id="vi-chat-start" class="vi-start-btn">2. Train hội thoại cơ bản</button>
                <label for="vi-wiki-seed">Link Wikipedia tiếng Việt <input id="vi-wiki-seed" type="url" value="${escapeHtml(viSeed)}"></label>
                <button type="button" id="vi-wiki-start" class="sec">3. Train tiếp Wikipedia</button>
                <progress id="vi-progress" max="${viCurriculum.targetSteps}" value="${Math.min(viCurriculum.foundationSteps, viCurriculum.targetSteps)}" aria-label="Tổng tiến độ training"></progress>
                <div id="vi-total-progress" class="field-hint"></div>
                <div id="vi-status" class="field-hint"></div>
              </div>
              <div class="export-load-row">
                <select id="export-load" class="export-load-select" title="Saved models in models/"></select>
                <button type="button" id="export-load-btn" class="sec">Load</button>
                <button type="button" id="export-refresh-btn" class="link-btn">Refresh</button>
              </div>
              <div class="field-hint">${HINT.exportModel}${lastExportName ? ` · Last: <code>${escapeHtml(lastExportName)}</code>` : ""}${loadedExportId ? ` · Loaded: <code>${escapeHtml(loadedExportId)}</code>` : ""}</div>`
        }
        <div class="stats">step: <b id="step">${step}</b> loss: <b id="loss">0</b> val loss: <b id="val-loss">0</b> val char acc (hold-out): <b id="val-acc">0</b></div>
        <div class="box-hint">${HINT.stats} ${HINT.statsChart}</div>
        <div id="corpus-summary" class="stats muted"></div><div class="stats muted">backend <span id="backend">${backendName}</span></div>
        <div id="train-status" class="muted" role="status" aria-live="polite"></div>
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
    if (
      training ||
      viActive ||
      viPreparing ||
      paused ||
      modelIo ||
      chatGenerating
    ) {
      setStatus("Stop training before changing settings.");
      return;
    }
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
  q<HTMLButtonElement>("#stop-chat")!.onclick = () => {
    chatAbort?.abort();
    for (const turn of chatHistory) {
      if (turn.status === "queued") {
        turn.status = "done";
        turn.text = "(Đã hủy)";
      }
    }
    paintChat();
  };
  q<HTMLButtonElement>("#clear-chat")!.onclick = newChat;
  q<HTMLSelectElement>("#chat-history")!.onchange = (event) => {
    if (chatGenerating) return;
    persistChat();
    activeSession = (event.target as HTMLSelectElement).value;
    chatHistory = sessions.find(s => s.id === activeSession)!.turns;
    lastGen = null;
    paintChat(); paintStreamStatus();
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
  const chatTrainBtn = q<HTMLButtonElement>("#vi-chat-start");
  if (chatTrainBtn) chatTrainBtn.onclick = () => void startChatTraining();
  const wikiInput = q<HTMLInputElement>("#vi-wiki-seed");
  if (wikiInput)
    wikiInput.oninput = () => {
      viSeed = wikiInput.value;
    };
  const targetInput = q<HTMLInputElement>("#vi-foundation-steps");
  if (targetInput)
    targetInput.onchange = () => {
      const value = Number(targetInput.value);
      if (Number.isSafeInteger(value) && value > 0) {
        viCurriculum.targetSteps = value;
        paintViState();
      }
    };
  const wikiBtn = q<HTMLButtonElement>("#vi-wiki-start");
  if (wikiBtn) wikiBtn.onclick = () => void startWikiTraining();
  const exportBtn = q<HTMLButtonElement>("#export-btn");
  if (exportBtn) exportBtn.onclick = () => void saveModelExport();
  const exportLoadBtn = q<HTMLButtonElement>("#export-load-btn");
  if (exportLoadBtn)
    exportLoadBtn.onclick = () => {
      const id = q<HTMLSelectElement>("#export-load")!.value;
      void loadSelectedExport(id);
    };
  const exportRefreshBtn = q<HTMLButtonElement>("#export-refresh-btn");
  if (exportRefreshBtn)
    exportRefreshBtn.onclick = () => void refreshExportCatalog();
  const resetBtn = q<HTMLButtonElement>("#reset-btn");
  if (resetBtn)
    resetBtn.onclick = () => {
      if (
        training ||
        viActive ||
        paused ||
        viPreparing ||
        modelIo ||
        chatGenerating
      )
        return;
      rebuild(usesViVocab(tokenizer) ? viTokenizer() : undefined);
      render();
    };
}

app.innerHTML = `<div class="muted panel-empty">Starting TensorFlow.js…</div>`;
void initTfBackend().then(async (name) => {
  backendName = `${name} (UI)`;
  rebuild();
  exportCatalog = await fetchExportList().catch(() => []);
  render();
  setStatus("Chọn sẵn bản lưu gần nhất. Bấm Load để nạp model.");
}).catch((error) => {
  app.textContent = error instanceof Error ? error.message : String(error);
});
