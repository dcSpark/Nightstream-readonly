function postToHost(msg) {
  // iOS (WKWebView)
  try {
    const handler = window.webkit?.messageHandlers?.neofold;
    if (handler) {
      handler.postMessage(msg);
      return;
    }
  } catch {
    // ignore
  }

  // Android (WebView addJavascriptInterface)
  try {
    const bridge = window.neofoldAndroid;
    if (bridge && typeof bridge.postMessage === "function") {
      bridge.postMessage(JSON.stringify(msg));
    }
  } catch {
    // ignore
  }
}

function supportsWasmThreadsRuntime() {
  if (typeof WebAssembly !== "object" || typeof WebAssembly.Memory !== "function") return false;
  if (self.crossOriginIsolated !== true) return false;
  if (typeof SharedArrayBuffer !== "function") return false;
  if (typeof Atomics !== "object") return false;
  try {
    const mem = new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true });
    return mem.buffer instanceof SharedArrayBuffer;
  } catch {
    return false;
  }
}

async function exists(path) {
  try {
    const resp = await fetch(path, { method: "HEAD", cache: "no-store" });
    return resp.ok;
  } catch {
    return false;
  }
}

function arrayBufferToBase64(buf) {
  if (!(buf instanceof ArrayBuffer)) return null;
  const bytes = new Uint8Array(buf);
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

let proverWorker = null;
let nextRunId = 1;
const inflightRuns = new Map(); // id -> timeout handle
const DEFAULT_RUN_WATCHDOG_MS = 10 * 60 * 1000;

function isIOSUserAgent(ua) {
  return typeof ua === "string" && /\b(iPad|iPhone|iPod)\b/i.test(ua);
}

function clearRunWatchdog(id) {
  const t = inflightRuns.get(id);
  if (t) clearTimeout(t);
  inflightRuns.delete(id);
}

function failInflightRuns(error) {
  const ids = Array.from(inflightRuns.keys());
  for (const id of ids) {
    clearRunWatchdog(id);
    postToHost({ type: "error", id, error: String(error) });
  }
}

function resetProverWorker(error) {
  const message = error != null ? String(error) : "Prover worker reset.";

  failInflightRuns(message);

  if (proverWorker) {
    try {
      proverWorker.terminate();
    } catch {
      // ignore
    }
  }
  proverWorker = null;
}

function ensureProverWorker() {
  if (proverWorker) return proverWorker;

  proverWorker = new Worker(new URL("./prover_worker.js", import.meta.url), { type: "module" });
  proverWorker.addEventListener("error", (e) => {
    const message = `Prover worker error: ${String(e?.message ?? e)}`;
    resetProverWorker(message);
    postToHost({ type: "error", id: -1, error: message });
  });
  proverWorker.addEventListener("messageerror", (e) => {
    const message = `Prover worker message error: ${String(e?.message ?? e)}`;
    resetProverWorker(message);
    postToHost({ type: "error", id: -1, error: message });
  });

  proverWorker.addEventListener("message", (ev) => {
    const msg = ev.data;
    const id = msg?.id;
    if (typeof id !== "number") return;

    if (msg?.type === "log") {
      postToHost({ type: "log", id, line: String(msg?.line ?? "") });
      return;
    }

    if (msg?.type === "error") {
      clearRunWatchdog(id);
      postToHost({ type: "error", id, error: String(msg?.error ?? "Unknown error") });
      return;
    }

    if (msg?.type === "done") {
      clearRunWatchdog(id);
      let spartan = null;
      const sp = msg?.spartan;
      if (sp && typeof sp?.filename === "string" && sp?.bytes instanceof ArrayBuffer) {
        const b64 = arrayBufferToBase64(sp.bytes);
        if (b64) spartan = { filename: sp.filename, bytes_b64: b64 };
      }
      postToHost({ type: "done", id, raw: msg?.raw ?? null, spartan });
    }
  });

  return proverWorker;
}

async function getCapabilities() {
  const supportsThreadsRuntime = supportsWasmThreadsRuntime();
  const singleBundlePresent = await exists("./pkg/neo_fold_demo.js");
  const threadsBundlePresent = await exists("./pkg_threads/neo_fold_demo.js");

  return {
    userAgent: navigator.userAgent,
    crossOriginIsolated: self.crossOriginIsolated === true,
    supportsThreadsRuntime,
    singleBundlePresent,
    threadsBundlePresent,
    defaultThreads: navigator.hardwareConcurrency ?? 4,
  };
}

window.__swift_neofold_getCapabilities = getCapabilities;

window.__swift_neofold_startRun = async function startRun(json, doSpartan, opts) {
  ensureProverWorker();

  const requestedId = typeof opts?.id === "number" ? Math.floor(opts.id) : null;
  const id =
    requestedId != null && Number.isFinite(requestedId) && requestedId >= 0
      ? requestedId
      : nextRunId++;
  if (requestedId != null && Number.isFinite(requestedId) && requestedId >= 0) {
    nextRunId = Math.max(nextRunId, id + 1);
  }

  const cap = await getCapabilities();
  const forceThreads = opts?.forceThreads === true;
  const disableThreads = opts?.disableThreads === true;

  const canUseThreads = cap.supportsThreadsRuntime && cap.threadsBundlePresent;
  const wantThreads = forceThreads ? true : disableThreads ? false : canUseThreads;

  const bundle = wantThreads ? "pkg_threads" : "pkg";
  const recommendedThreads = isIOSUserAgent(cap.userAgent)
    ? Math.max(1, Math.min(2, cap.defaultThreads))
    : cap.defaultThreads;
  const threads =
    wantThreads && typeof opts?.threads === "number"
      ? Math.max(1, Math.floor(opts.threads))
      : wantThreads
        ? recommendedThreads
        : 0;

  const watchdogMs =
    typeof opts?.watchdogMs === "number" && Number.isFinite(opts.watchdogMs) && opts.watchdogMs > 0
      ? Math.max(1000, Math.floor(opts.watchdogMs))
      : DEFAULT_RUN_WATCHDOG_MS;

  clearRunWatchdog(id);
  inflightRuns.set(
    id,
    setTimeout(() => {
      const seconds = Math.round(watchdogMs / 1000);
      resetProverWorker(`Prover worker timed out after ${seconds}s; restarting.`);
    }, watchdogMs),
  );

  proverWorker.postMessage({
    type: "run",
    id,
    json: String(json ?? ""),
    doSpartan: doSpartan === true,
    bundle,
    threads,
  });

  return { id, bundle, threads, capabilities: cap };
};

(async function () {
  try {
    ensureProverWorker();
    const cap = await getCapabilities();
    postToHost({ type: "ready", capabilities: cap });
  } catch (e) {
    postToHost({ type: "error", id: -1, error: String(e) });
  }
})();

