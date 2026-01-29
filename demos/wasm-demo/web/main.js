const logEl = document.getElementById("log");

const jsonTextareaEl = document.getElementById("circuit-json");
const jsonGutterEl = document.getElementById("circuit-json-gutter");
const jsonErrorEl = document.getElementById("json-error");

const presetButtons = Array.from(document.querySelectorAll(".preset-btn"));

const circuitJsonToggleEl = document.getElementById("circuit-json-toggle");
const circuitJsonBodyEl = document.getElementById("circuit-json-body");

const fileDropEl = document.getElementById("file-drop");
const fileInputEl = document.getElementById("file-input");
const fileMetaEl = document.getElementById("file-meta");
const fileNameEl = document.getElementById("file-name");
const fileSizeEl = document.getElementById("file-size");
const fileClearEl = document.getElementById("file-clear");

const compressSpartanEl = document.getElementById("compress-spartan");
const downloadSpartanEl = document.getElementById("download-spartan");
const runBtnEl = document.getElementById("run");
const clearLogEl = document.getElementById("clear-log");
const runStatusEl = document.getElementById("run-status");

const jsonFormatEl = document.getElementById("json-format");
const jsonCopyEl = document.getElementById("json-copy");
const jsonDownloadEl = document.getElementById("json-download");
const jsonValidateEl = document.getElementById("json-validate");

const autoscrollEl = document.getElementById("autoscroll");
const copyLogsEl = document.getElementById("copy-logs");

const statusBundleEl = document.getElementById("status-bundle");
const statusCoiEl = document.getElementById("status-coi");
const statusThreadsEl = document.getElementById("status-threads");
const commitBtnEl = document.getElementById("commit-btn");
const buildLineEl = document.getElementById("build-line");

const threadsFixEl = document.getElementById("threads-fix");
const coiHeadersEl = document.getElementById("coi-headers");
const copyCoiHeadersEl = document.getElementById("copy-coi-headers");

const infoBtnEl = document.getElementById("info-btn");
const infoPanelEl = document.getElementById("info-panel");
const threadCheckCommandEl = document.getElementById("thread-check-command");
const copyThreadCheckEl = document.getElementById("copy-thread-check");
const copyStatusEl = document.getElementById("copy-status");

let logs = [];
let autoScroll = false;
let circuitJsonCollapsed = circuitJsonBodyEl?.hidden === true;

let lastSpartanProofBytes = null;
let lastSpartanProofFilename = null;
let lastJsonFilename = "circuit.json";
let proverWorker = null;
let activeWasmBundle = null; // "pkg" | "pkg_threads"
let activeWasmThreads = 0;
let runId = 0;
let runInProgress = false;

const urlParams = new URLSearchParams(window.location.search);
const threadsParam = urlParams.get("threads"); // "1" | "0" | null
const nthreadsParam = urlParams.get("nthreads"); // integer string | null
const threadsForcedOn = threadsParam === "1";
const threadsForcedOff = threadsParam === "0";
const nthreadsRequested = nthreadsParam ? Number.parseInt(nthreadsParam, 10) : null;

function isSafari() {
  const ua = navigator.userAgent ?? "";
  const vendor = navigator.vendor ?? "";
  const isApple = vendor.includes("Apple");
  const isSafariUA = ua.includes("Safari");
  const isOtherBrowser =
    ua.includes("Chrome") || ua.includes("CriOS") || ua.includes("Edg") || ua.includes("OPR");
  return isApple && isSafariUA && !isOtherBrowser;
}

const safariDisablesThreads = isSafari() && !threadsForcedOn;

// Minimal Wasm module that requires the threads proposal (shared memory + atomic instruction).
// Generated from:
// (module (memory 1 1 shared) (func i32.const 0 i32.atomic.load drop))
const WASM_THREADS_VALIDATE_BYTES = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 5, 4, 1, 3, 1, 1, 10,
  11, 1, 9, 0, 65, 0, 254, 16, 2, 0, 26, 11,
]);

function supportsWasmThreadsRuntime() {
  if (typeof WebAssembly !== "object" || typeof WebAssembly.Memory !== "function") return false;
  if (typeof WebAssembly.validate !== "function") return false;
  if (safariDisablesThreads) return false;
  if (self.crossOriginIsolated !== true) return false;
  if (typeof SharedArrayBuffer !== "function") return false;
  if (typeof Atomics !== "object") return false;

  // Firefox requires SABs to be transferable via MessageChannel for wasm threads.
  // Safari/Chrome tolerate this check (it just throws if unsupported).
  if (typeof MessageChannel !== "undefined") {
    try {
      new MessageChannel().port1.postMessage(new SharedArrayBuffer(1));
    } catch {
      return false;
    }
  }

  try {
    if (!WebAssembly.validate(WASM_THREADS_VALIDATE_BYTES)) return false;
  } catch {
    return false;
  }

  try {
    const mem = new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true });
    return mem.buffer instanceof SharedArrayBuffer;
  } catch {
    return false;
  }
}

const supportsThreadsRuntime = supportsWasmThreadsRuntime();

const preferThreads = !threadsForcedOff && supportsThreadsRuntime;
const threadsHint = threadsForcedOn ? "?threads=1" : threadsForcedOff ? "?threads=0" : "auto";

const WASM_SINGLE = "./pkg/neo_fold_demo.js";
const WASM_THREADS = "./pkg_threads/neo_fold_demo.js";

function defaultThreadCount() {
  const hc = navigator.hardwareConcurrency ?? 4;
  return Math.min(Math.max(1, hc), 4);
}

function setText(el, text) {
  if (!el) return;
  el.textContent = text;
}

function setChip(el, { text, tone = "neutral", icon = "●", title } = {}) {
  if (!el) return;
  el.textContent = text ?? "";
  el.dataset.icon = icon;
  el.classList.remove("chip-neutral", "chip-success", "chip-warning", "chip-danger");
  el.classList.add(
    tone === "success"
      ? "chip-success"
      : tone === "warning"
        ? "chip-warning"
        : tone === "danger"
          ? "chip-danger"
          : "chip-neutral",
  );
  if (typeof title === "string") el.title = title;
}

async function loadBuildInfo(bundle) {
  if (!commitBtnEl) return;
  setChip(commitBtnEl, { text: "Commit: loading…", tone: "warning", icon: "…" });
  commitBtnEl.disabled = true;

  try {
    const url =
      bundle === "pkg_threads"
        ? "./pkg_threads/build_info.json"
        : bundle === "pkg"
          ? "./pkg/build_info.json"
          : "./build_info.json";

    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) throw new Error(`${url}: HTTP ${resp.status}`);
    const info = await resp.json();

    const commit = typeof info?.git_commit === "string" ? info.git_commit : "";
    const commitShort =
      typeof info?.git_commit_short === "string" && info.git_commit_short.length > 0
        ? info.git_commit_short
        : commit
          ? commit.slice(0, 12)
          : "";
    const dirty = info?.git_dirty === true;
    const builtAt = typeof info?.build_time_utc === "string" ? info.build_time_utc : "";

    if (!commitShort) {
      setChip(commitBtnEl, { text: "Commit: unknown", tone: "warning", icon: "!" });
      return;
    }

    setChip(commitBtnEl, { text: `Commit: ${commitShort}${dirty ? "*" : ""}`, icon: "⧉" });
    commitBtnEl.disabled = false;
    commitBtnEl.dataset.commit = commit || commitShort;

    const tipParts = [];
    tipParts.push(commit || commitShort);
    if (dirty) tipParts.push("dirty working tree");
    if (builtAt) tipParts.push(`built: ${builtAt}`);
    commitBtnEl.title = tipParts.join("\n");

    if (buildLineEl) {
      const buildParts = [`Build: ${bundle}`];
      buildParts.push(commitShort + (dirty ? "*" : ""));
      if (builtAt) buildParts.push(`built ${builtAt}`);
      buildLineEl.textContent = buildParts.join(" · ");
    }
  } catch (e) {
    setChip(commitBtnEl, { text: "Commit: unavailable", tone: "warning", icon: "!" });
    if (buildLineEl) buildLineEl.textContent = "Build: unavailable";
  }
}

async function loadWasmModule() {
  if (!preferThreads) {
    return { wasm: await import(WASM_SINGLE), bundle: "pkg" };
  }

  try {
    return { wasm: await import(WASM_THREADS), bundle: "pkg_threads" };
  } catch (e) {
    // Threads bundle might not be built. Fall back to single-thread.
    logWarn("Threads supported, but failed to load threads bundle; falling back to single-thread.");
    logWarn("Build threads bundle with: ./demos/wasm-demo/build_wasm.sh");
    logWarn(`Load error: ${String(e)}`);
    return { wasm: await import(WASM_SINGLE), bundle: "pkg" };
  }
}

async function copyToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  // Fallback for older Safari: temporary textarea + execCommand.
  const tmp = document.createElement("textarea");
  tmp.value = text;
  tmp.setAttribute("readonly", "");
  tmp.style.position = "fixed";
  tmp.style.top = "-1000px";
  tmp.style.left = "-1000px";
  document.body.appendChild(tmp);
  tmp.focus();
  tmp.select();
  document.execCommand("copy");
  document.body.removeChild(tmp);
}

let clockFormatter = null;
try {
  clockFormatter = new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
} catch {
  clockFormatter = null;
}

function fmtClock(d) {
  try {
    if (clockFormatter) return clockFormatter.format(d);
    // Fallback: HH:MM:SS (local time-ish without i18n)
    const pad2 = (n) => String(n).padStart(2, "0");
    return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
  } catch {
    return "";
  }
}

function safeStringify(value) {
  return JSON.stringify(
    value,
    (_k, v) => (typeof v === "bigint" ? v.toString() : v),
    2,
  );
}

function fmtMs(ms) {
  if (typeof ms !== "number" || !Number.isFinite(ms)) return String(ms);
  return `${ms.toFixed(1)} ms`;
}

function fmtBytes(bytes) {
  if (typeof bytes !== "number" || !Number.isFinite(bytes)) return String(bytes);
  if (bytes < 1024) return `${bytes.toFixed(0)} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(2)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(2)} MB`;
}

function fmtList(values, maxItems = 32) {
  if (!Array.isArray(values)) return String(values);
  const shown = values.slice(0, maxItems).map((v) => String(v));
  const more = values.length > maxItems ? ` … (+${values.length - maxItems} more)` : "";
  return `[${shown.join(", ")}]${more}`;
}

function fmtMsList(values, maxItems = 32) {
  if (!Array.isArray(values)) return String(values);
  const shown = values.slice(0, maxItems).map((v) => fmtMs(v));
  const more = values.length > maxItems ? ` … (+${values.length - maxItems} more)` : "";
  return `[${shown.join(", ")}]${more}`;
}

function tryParseTestExport(json) {
  try {
    return JSON.parse(json);
  } catch {
    return null;
  }
}

function appendLogLine({ time, level, msg }) {
  if (!logEl) return;
  const lineEl = document.createElement("div");
  lineEl.className = `log-line log-${level}`;

  const tEl = document.createElement("span");
  tEl.className = "log-time";
  tEl.textContent = time;

  const lvlEl = document.createElement("span");
  lvlEl.className = "log-level";
  lvlEl.textContent = level;

  const msgEl = document.createElement("span");
  msgEl.className = "log-msg";
  msgEl.textContent = msg;

  lineEl.append(tEl, lvlEl, msgEl);
  logEl.appendChild(lineEl);

  if (autoScroll) {
    logEl.scrollTop = logEl.scrollHeight;
  }
}

function addLog(level, msg) {
  const entry = { time: fmtClock(new Date()), level, msg: String(msg) };
  logs.push(entry);
  appendLogLine(entry);
}

function logInfo(msg) {
  addLog("info", msg);
}

function logWarn(msg) {
  addLog("warn", msg);
}

function logError(msg) {
  addLog("error", msg);
}

function clearLogs() {
  logs = [];
  if (logEl) logEl.textContent = "";
}

function setCircuitJsonCollapsed(collapsed) {
  circuitJsonCollapsed = Boolean(collapsed);
  if (circuitJsonBodyEl) circuitJsonBodyEl.hidden = circuitJsonCollapsed;
  if (circuitJsonToggleEl) {
    circuitJsonToggleEl.setAttribute("aria-expanded", circuitJsonCollapsed ? "false" : "true");
    const label = circuitJsonCollapsed ? "Expand Circuit JSON" : "Collapse Circuit JSON";
    circuitJsonToggleEl.setAttribute("aria-label", label);
    circuitJsonToggleEl.title = label;
  }
}

function setButtonsEnabled(enabled) {
  for (const btn of presetButtons) btn.disabled = !enabled;
  if (fileInputEl) fileInputEl.disabled = !enabled;
  if (fileClearEl) fileClearEl.disabled = !enabled || fileMetaEl?.hidden !== false;
  if (runBtnEl) runBtnEl.disabled = !enabled;
  if (jsonTextareaEl) jsonTextareaEl.disabled = !enabled;
  if (compressSpartanEl) compressSpartanEl.disabled = !enabled;
  if (downloadSpartanEl) downloadSpartanEl.disabled = !enabled || !lastSpartanProofBytes;
  if (jsonFormatEl) jsonFormatEl.disabled = !enabled;
  if (jsonCopyEl) jsonCopyEl.disabled = !enabled;
  if (jsonDownloadEl) jsonDownloadEl.disabled = !enabled;
  if (jsonValidateEl) jsonValidateEl.disabled = !enabled;
}

function downloadBytes(filename, bytes) {
  const blob = new Blob([bytes], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function downloadText(filename, text) {
  const bytes = new TextEncoder().encode(text);
  downloadBytes(filename, bytes);
}

function ensureProverWorker() {
  if (proverWorker) return proverWorker;
  proverWorker = new Worker(new URL("./prover_worker.js", import.meta.url), { type: "module" });
  proverWorker.addEventListener("error", (e) => {
    logError(`Worker error: ${e?.message ?? String(e)}`);
    console.error(e);
  });
  return proverWorker;
}

const PRESETS = {
  toy_square: {
    label: "Toy circuit",
    url: "./examples/toy_square.json",
    filename: "toy_square.json",
  },
  toy_square_folding_8_steps: {
    label: "Toy folding circuit (8 steps)",
    url: "./examples/toy_square_folding_8_steps.json",
    filename: "toy_square_folding_8_steps.json",
  },
  poseidon2_ic_batch_1: {
    label: "Poseidon2 IC (batch 1)",
    url: "./examples/poseidon2_ic_batch_1.json",
    filename: "poseidon2_ic_batch_1.json",
  },
};

function countLines(text) {
  let lines = 1;
  for (let i = 0; i < text.length; i++) {
    if (text.charCodeAt(i) === 10) lines++;
  }
  return lines;
}

function updateEditorGutter() {
  if (!jsonGutterEl || !jsonTextareaEl) return;

  const text = jsonTextareaEl.value ?? "";
  const lineCount = countLines(text);
  const digits = String(lineCount).length;
  const minWidth = Math.max(52, digits * 8 + 24);
  jsonGutterEl.style.minWidth = `${minWidth}px`;

  let out = "";
  for (let i = 1; i <= lineCount; i++) out += `${i}\n`;
  jsonGutterEl.textContent = out;
  jsonGutterEl.scrollTop = jsonTextareaEl.scrollTop;
}

let gutterRaf = 0;
function scheduleEditorGutterUpdate() {
  if (gutterRaf) return;
  gutterRaf = requestAnimationFrame(() => {
    gutterRaf = 0;
    updateEditorGutter();
  });
}

function clearJsonError() {
  if (!jsonErrorEl) return;
  jsonErrorEl.hidden = true;
  jsonErrorEl.textContent = "";
}

function normalizeJsonErrorMessage(e) {
  const msg = String(e?.message ?? e);
  return msg.replace(/^SyntaxError:\s*/i, "");
}

function posFromLineCol(text, line, col) {
  const targetLine = Math.max(1, Number(line) || 1);
  const targetCol = Math.max(1, Number(col) || 1);
  let curLine = 1;
  let i = 0;
  while (i < text.length && curLine < targetLine) {
    if (text.charCodeAt(i) === 10) curLine++;
    i++;
  }
  return Math.min(text.length, i + (targetCol - 1));
}

function lineColFromPos(text, pos) {
  const p = Math.min(Math.max(0, Number(pos) || 0), text.length);
  let line = 1;
  let lastNewline = -1;
  for (let i = 0; i < p; i++) {
    if (text.charCodeAt(i) === 10) {
      line++;
      lastNewline = i;
    }
  }
  const col = p - lastNewline;
  return { line, col };
}

function parseJsonErrorLocation(message, text) {
  // V8: "... at position 123"
  const mPos = message.match(/\bposition\s+(\d+)\b/i);
  if (mPos) {
    const pos = Number.parseInt(mPos[1], 10);
    const { line, col } = lineColFromPos(text, pos);
    return { pos, line, col };
  }

  // Safari: "... at line 1 column 23"
  const mLineCol = message.match(/\bline\s+(\d+)\s+column\s+(\d+)\b/i);
  if (mLineCol) {
    const line = Number.parseInt(mLineCol[1], 10);
    const col = Number.parseInt(mLineCol[2], 10);
    const pos = posFromLineCol(text, line, col);
    return { pos, line, col };
  }

  return null;
}

function validateJson(text, { focusOnError = false } = {}) {
  const src = String(text ?? "");
  try {
    const value = JSON.parse(src);
    clearJsonError();
    return { ok: true, value };
  } catch (e) {
    const msg = normalizeJsonErrorMessage(e);
    const loc = parseJsonErrorLocation(msg, src);
    const locText = loc ? ` (line ${loc.line}, col ${loc.col})` : "";
    if (focusOnError) setCircuitJsonCollapsed(false);
    if (jsonErrorEl) {
      jsonErrorEl.textContent = `Invalid JSON: ${msg}${locText}`;
      jsonErrorEl.hidden = false;
    }
    if (focusOnError && jsonTextareaEl) {
      const pos = loc?.pos ?? Math.min(src.length, Math.max(0, src.length - 1));
      jsonTextareaEl.focus();
      jsonTextareaEl.setSelectionRange(pos, Math.min(src.length, pos + 1));
    }
    return { ok: false, error: msg, loc };
  }
}

let autoValidateTimer = 0;
function scheduleAutoValidate() {
  if (!jsonTextareaEl) return;
  const len = jsonTextareaEl.value?.length ?? 0;
  if (len > 1_000_000) return; // avoid expensive parse on huge inputs while typing
  if (autoValidateTimer) window.clearTimeout(autoValidateTimer);
  autoValidateTimer = window.setTimeout(() => validateJson(jsonTextareaEl.value), 800);
}

function setJsonText(text, { filename } = {}) {
  if (!jsonTextareaEl) return;
  jsonTextareaEl.value = text ?? "";
  clearJsonError();
  scheduleEditorGutterUpdate();
  if (typeof filename === "string" && filename.length > 0) lastJsonFilename = filename;
}

async function loadPreset(key) {
  const preset = PRESETS[key];
  if (!preset) throw new Error(`Unknown preset: ${key}`);
  const resp = await fetch(preset.url);
  if (!resp.ok) throw new Error(`Failed to load preset: ${resp.status}`);
  const txt = await resp.text();
  setJsonText(txt, { filename: preset.filename });
  setActivePreset(key);
  logInfo(`Loaded preset: ${preset.label} (${fmtBytes(txt.length)})`);
}

function setActivePreset(key) {
  for (const btn of presetButtons) {
    const isActive = btn.dataset.preset === key;
    btn.classList.toggle("is-active", isActive);
    btn.setAttribute("aria-pressed", isActive ? "true" : "false");
  }
}

function clearFileSelection() {
  if (fileInputEl) fileInputEl.value = "";
  if (fileMetaEl) fileMetaEl.hidden = true;
  if (fileNameEl) fileNameEl.textContent = "";
  if (fileSizeEl) fileSizeEl.textContent = "";
}

async function loadFile(file) {
  const txt = await file.text();
  setJsonText(txt, { filename: file.name });
  setActivePreset(null);
  if (fileMetaEl) fileMetaEl.hidden = false;
  if (fileNameEl) fileNameEl.textContent = file.name;
  if (fileSizeEl) fileSizeEl.textContent = `(${fmtBytes(file.size)})`;
  logInfo(`Loaded file: ${file.name} (${fmtBytes(file.size)})`);
}

function setRunUiState({ running, label, status } = {}) {
  if (runBtnEl) {
    runBtnEl.dataset.loading = running ? "true" : "false";
    const labelEl = runBtnEl.querySelector(".btn-label");
    if (labelEl && typeof label === "string") labelEl.textContent = label;
  }
  if (runStatusEl && typeof status === "string") runStatusEl.textContent = status;
}

async function run() {
  const json = jsonTextareaEl?.value ?? "";
  lastSpartanProofBytes = null;
  lastSpartanProofFilename = null;
  if (downloadSpartanEl) downloadSpartanEl.disabled = true;
  if (!json.trim()) {
    logWarn("No JSON provided.");
    return;
  }

  const parsed = validateJson(json, { focusOnError: true });
  if (!parsed.ok) return;

  if (runInProgress) {
    logWarn("Run already in progress.");
    return;
  }
  runInProgress = true;
  setCircuitJsonCollapsed(true);

  const doSpartan = Boolean(compressSpartanEl?.checked);

  setButtonsEnabled(false);
  setRunUiState({ running: true, label: "Running…", status: "Preparing…" });
  try {
    const worker = ensureProverWorker();
    const id = ++runId;
    logInfo(`— Run ${id} —`);
    const result = await new Promise((resolve, reject) => {
      const onMessage = (ev) => {
        const msg = ev.data;
        if (!msg || msg.id !== id) return;

        if (msg.type === "log") {
          const level = msg.level === "error" ? "error" : msg.level === "warn" ? "warn" : "info";
          addLog(level, msg.line);
          return;
        }
        if (msg.type === "phase") {
          const phaseText = typeof msg.phase === "string" ? msg.phase : "Running…";
          setRunUiState({ running: true, label: phaseText, status: phaseText });
          return;
        }
        if (msg.type === "done") {
          if (msg.spartan?.bytes && msg.spartan?.filename) {
            lastSpartanProofBytes = new Uint8Array(msg.spartan.bytes);
            lastSpartanProofFilename = msg.spartan.filename;
          }
          cleanup();
          resolve(msg);
          return;
        }
        if (msg.type === "error") {
          cleanup();
          reject(new Error(msg.error ?? "Unknown worker error"));
          return;
        }
      };

      const onError = (ev) => {
        cleanup();
        reject(new Error(ev?.message ?? "Worker error"));
      };

      const onMessageError = () => {
        cleanup();
        reject(new Error("Worker message error"));
      };

      const cleanup = () => {
        worker.removeEventListener("message", onMessage);
        worker.removeEventListener("error", onError);
        worker.removeEventListener("messageerror", onMessageError);
      };

      worker.addEventListener("message", onMessage);
      worker.addEventListener("error", onError);
      worker.addEventListener("messageerror", onMessageError);
      worker.postMessage({
        type: "run",
        id,
        json,
        doSpartan,
        bundle: activeWasmBundle ?? "pkg",
        threads: activeWasmThreads,
      });
    });

    if (result?.spartan?.bytes && downloadSpartanEl) {
      downloadSpartanEl.disabled = false;
    }
    setRunUiState({ running: false, label: "Prove + Verify", status: "Done." });
  } catch (e) {
    logError(`ERROR: ${e}`);
    console.error(e);
    setRunUiState({ running: false, label: "Prove + Verify", status: "Failed." });
  } finally {
    setButtonsEnabled(true);
    runInProgress = false;
    if (runBtnEl) runBtnEl.dataset.loading = "false";
  }
}

async function main() {
  setButtonsEnabled(false);
  setCircuitJsonCollapsed(circuitJsonCollapsed);
  if (circuitJsonToggleEl) {
    circuitJsonToggleEl.addEventListener("click", () => {
      setCircuitJsonCollapsed(!circuitJsonCollapsed);
    });
  }
  setRunUiState({ running: false, label: "Prove + Verify", status: "" });
  logInfo("Loading wasm…");
  try {
    setChip(statusBundleEl, {
      text: `Bundle: ${preferThreads ? "auto (prefers threads)" : "pkg"} (${threadsHint})`,
      tone: preferThreads ? "warning" : "neutral",
      icon: "⬤",
    });

    setChip(statusCoiEl, {
      text: `Cross-Origin Isolation: ${self.crossOriginIsolated === true ? "enabled" : "disabled"}`,
      tone: self.crossOriginIsolated === true ? "success" : "danger",
      icon: self.crossOriginIsolated === true ? "✓" : "×",
      title:
        self.crossOriginIsolated === true
          ? "SharedArrayBuffer allowed (COOP/COEP enabled)"
          : "Need COOP/COEP headers to enable SharedArrayBuffer / wasm threads",
    });

    if (threadsForcedOn && !supportsThreadsRuntime) {
      logWarn("Threads requested (?threads=1) but not supported in this context.");
      logWarn("Need: crossOriginIsolated + SharedArrayBuffer + wasm threads support.");
      setChip(statusThreadsEl, {
        text: "Threads: requested but unavailable",
        tone: "danger",
        icon: "×",
      });
    } else if (threadsForcedOff) {
      setChip(statusThreadsEl, {
        text: "Threads: disabled (?threads=0)",
        tone: "warning",
        icon: "!",
      });
    } else if (supportsThreadsRuntime) {
      setChip(statusThreadsEl, { text: "Threads: supported (auto)", tone: "warning", icon: "!" });
    } else {
      setChip(statusThreadsEl, {
        text: "Threads: unavailable (Fix)",
        tone: "warning",
        icon: "!",
        title: "Click for COOP/COEP header snippet",
      });
      if (threadsFixEl) threadsFixEl.hidden = false;
    }

    if (supportsThreadsRuntime) {
      logInfo("Threads supported (cross-origin isolated + SharedArrayBuffer).");
      logInfo(`hardwareConcurrency=${String(navigator.hardwareConcurrency ?? "?")}`);
    } else {
      logWarn("Threads not supported (missing COOP/COEP / SharedArrayBuffer).");
    }

    const { wasm, bundle } = await loadWasmModule();
    window.__neo_fold_wasm = wasm;
    await wasm.default();
    wasm.init_panic_hook();

    setChip(statusBundleEl, { text: `Bundle: ${bundle} (${threadsHint})`, tone: "neutral", icon: "⬤" });
    await loadBuildInfo(bundle);
    activeWasmBundle = bundle;

    if (bundle === "pkg_threads") {
      if (typeof wasm.init_thread_pool !== "function") {
        logError("ERROR: threads bundle loaded, but wasm-threads exports are missing.");
        setChip(statusThreadsEl, {
          text: "Threads: error (missing init_thread_pool)",
          tone: "danger",
          icon: "×",
        });
      } else if (!supportsThreadsRuntime) {
        logError("ERROR: threads bundle loaded, but wasm threads are not available.");
        logError("Need: crossOriginIsolated + SharedArrayBuffer + wasm threads support.");
        setChip(statusThreadsEl, { text: "Threads: disabled (no wasm threads)", tone: "danger", icon: "×" });
      } else {
        const defaultThreads = defaultThreadCount();
        const n =
          typeof nthreadsRequested === "number" && Number.isFinite(nthreadsRequested) && nthreadsRequested > 0
            ? nthreadsRequested
            : defaultThreads;
        activeWasmThreads = n;
        logInfo(`Threads available; prover worker will use ${n} threads.`);
        setChip(statusThreadsEl, { text: `Threads: enabled (${n})`, tone: "success", icon: "✓" });
      }
    } else {
      if (threadsForcedOff) {
        setChip(statusThreadsEl, { text: "Threads: disabled (?threads=0)", tone: "warning", icon: "!" });
      } else if (supportsThreadsRuntime) {
        setChip(statusThreadsEl, { text: "Threads: disabled (single-thread bundle)", tone: "warning", icon: "!" });
      } else {
        setChip(statusThreadsEl, {
          text: "Threads: unavailable (Fix)",
          tone: "warning",
          icon: "!",
          title: "Click for COOP/COEP header snippet",
        });
        if (threadsFixEl) threadsFixEl.hidden = false;
      }
    }
  } catch (e) {
    logError("Failed to load wasm bundle.");
    logError(`Did you run ./demos/wasm-demo/build_wasm.sh${preferThreads ? "" : " --no-threads"} ?`);
    logError(String(e));
    console.error(e);
    setChip(statusThreadsEl, { text: "Threads: error", tone: "danger", icon: "×" });
    return;
  }
  logInfo("Wasm loaded.");

  const threadCheckCommand =
    "window.__neo_fold_wasm.default().then(exp => exp.memory.buffer.constructor.name)";
  if (threadCheckCommandEl) threadCheckCommandEl.value = threadCheckCommand;

  if (copyThreadCheckEl) {
    copyThreadCheckEl.addEventListener("click", async () => {
      try {
        await copyToClipboard(threadCheckCommand);
        setText(copyStatusEl, "Copied to clipboard.");
        setTimeout(() => setText(copyStatusEl, ""), 1500);
      } catch (e) {
        setText(copyStatusEl, `Copy failed: ${String(e)}`);
      }
    });
  }

  if (commitBtnEl) {
    commitBtnEl.addEventListener("click", async () => {
      const commit = commitBtnEl.dataset.commit;
      if (!commit) return;
      try {
        await copyToClipboard(commit);
        logInfo(`Copied commit to clipboard: ${commit}`);
      } catch (e) {
        logError(`Copy failed: ${String(e)}`);
      }
    });
  }

  if (downloadSpartanEl) {
    downloadSpartanEl.addEventListener("click", () => {
      if (!lastSpartanProofBytes || !lastSpartanProofFilename) {
        logWarn("No Spartan SNARK available to download (run with Spartan enabled first).");
        return;
      }
      downloadBytes(lastSpartanProofFilename, lastSpartanProofBytes);
      logInfo(
        `Downloaded Spartan SNARK: ${lastSpartanProofFilename} (${fmtBytes(lastSpartanProofBytes.length)})`,
      );
    });
  }

  if (coiHeadersEl) {
    coiHeadersEl.textContent = [
      "Cross-Origin-Opener-Policy: same-origin",
      "Cross-Origin-Embedder-Policy: require-corp",
      "Cross-Origin-Resource-Policy: same-origin",
    ].join("\n");
  }

  if (copyCoiHeadersEl && coiHeadersEl) {
    copyCoiHeadersEl.addEventListener("click", async () => {
      try {
        await copyToClipboard(coiHeadersEl.textContent ?? "");
        logInfo("Copied COOP/COEP header snippet.");
      } catch (e) {
        logError(`Copy failed: ${String(e)}`);
      }
    });
  }

  if (statusThreadsEl && threadsFixEl) {
    statusThreadsEl.addEventListener("click", () => {
      threadsFixEl.hidden = !threadsFixEl.hidden;
    });
  }

  if (clearLogEl) {
    clearLogEl.addEventListener("click", () => clearLogs());
  }

  if (copyLogsEl) {
    copyLogsEl.addEventListener("click", async () => {
      try {
        const txt = logs.map((l) => `${l.time} ${l.level.toUpperCase()} ${l.msg}`).join("\n");
        await copyToClipboard(txt);
        logInfo("Copied logs to clipboard.");
      } catch (e) {
        logError(`Copy failed: ${String(e)}`);
      }
    });
  }

  if (autoscrollEl) {
    autoScroll = Boolean(autoscrollEl.checked);
    autoscrollEl.addEventListener("change", () => {
      autoScroll = Boolean(autoscrollEl.checked);
    });
  }

  if (jsonTextareaEl) {
    jsonTextareaEl.addEventListener("scroll", () => {
      if (jsonGutterEl) jsonGutterEl.scrollTop = jsonTextareaEl.scrollTop;
    });
    jsonTextareaEl.addEventListener("input", () => {
      scheduleEditorGutterUpdate();
      scheduleAutoValidate();
    });
  }

  if (fileDropEl && fileInputEl) {
    const setDrag = (on) => fileDropEl.classList.toggle("is-dragover", on);
    const prevent = (e) => {
      e.preventDefault();
      e.stopPropagation();
    };

    fileDropEl.addEventListener("dragenter", (e) => {
      prevent(e);
      setDrag(true);
    });
    fileDropEl.addEventListener("dragover", (e) => {
      prevent(e);
      setDrag(true);
    });
    fileDropEl.addEventListener("dragleave", (e) => {
      prevent(e);
      setDrag(false);
    });
    fileDropEl.addEventListener("drop", async (e) => {
      prevent(e);
      setDrag(false);
      const file = e.dataTransfer?.files?.[0];
      if (!file) return;
      await loadFile(file);
    });

    fileInputEl.addEventListener("change", async (ev) => {
      const file = ev.target.files?.[0];
      if (!file) return;
      await loadFile(file);
    });
  }

  if (fileClearEl) {
    fileClearEl.addEventListener("click", () => clearFileSelection());
  }

  for (const btn of presetButtons) {
    btn.addEventListener("click", async () => {
      const key = btn.dataset.preset;
      if (!key) return;
      try {
        clearFileSelection();
        await loadPreset(key);
      } catch (e) {
        logError(`ERROR: ${e}`);
        console.error(e);
      }
    });
  }

  if (runBtnEl) runBtnEl.addEventListener("click", run);

  if (jsonFormatEl) {
    jsonFormatEl.addEventListener("click", () => {
      const src = jsonTextareaEl?.value ?? "";
      const parsed = validateJson(src, { focusOnError: true });
      if (!parsed.ok) return;
      setJsonText(JSON.stringify(parsed.value, null, 2) + "\n");
      logInfo("Formatted JSON.");
    });
  }

  if (jsonCopyEl) {
    jsonCopyEl.addEventListener("click", async () => {
      try {
        await copyToClipboard(jsonTextareaEl?.value ?? "");
        logInfo("Copied JSON to clipboard.");
      } catch (e) {
        logError(`Copy failed: ${String(e)}`);
      }
    });
  }

  if (jsonDownloadEl) {
    jsonDownloadEl.addEventListener("click", () => {
      downloadText(lastJsonFilename || "circuit.json", jsonTextareaEl?.value ?? "");
      logInfo(`Downloaded JSON: ${lastJsonFilename || "circuit.json"}`);
    });
  }

  if (jsonValidateEl) {
    jsonValidateEl.addEventListener("click", () => {
      const ok = validateJson(jsonTextareaEl?.value ?? "", { focusOnError: true }).ok;
      if (ok) logInfo("JSON valid.");
    });
  }

  if (infoBtnEl && infoPanelEl) {
    const close = () => {
      infoPanelEl.hidden = true;
      document.body.style.overflow = "";
    };

    infoBtnEl.addEventListener("click", () => {
      infoPanelEl.hidden = false;
      document.body.style.overflow = "hidden";
      if (threadCheckCommandEl) {
        threadCheckCommandEl.focus();
        threadCheckCommandEl.select();
      }
    });

    infoPanelEl.addEventListener("click", (e) => {
      const target = e.target;
      if (target && target instanceof HTMLElement && target.dataset.close === "true") close();
    });

    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !infoPanelEl.hidden) close();
    });
  }

  await loadPreset("poseidon2_ic_batch_1");

  updateEditorGutter();
  setButtonsEnabled(true);
}

main().catch((e) => {
  logError(`Fatal error: ${e}`);
  console.error(e);
});
