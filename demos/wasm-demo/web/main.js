const logEl = document.getElementById("log");
const jsonEl = document.getElementById("circuit-json");
const statusBundleEl = document.getElementById("status-bundle");
const statusCoiEl = document.getElementById("status-coi");
const statusThreadsEl = document.getElementById("status-threads");
const commitBtnEl = document.getElementById("commit-btn");
const infoBtnEl = document.getElementById("info-btn");
const infoPanelEl = document.getElementById("info-panel");
const threadCheckCommandEl = document.getElementById("thread-check-command");
const copyThreadCheckEl = document.getElementById("copy-thread-check");
const copyStatusEl = document.getElementById("copy-status");
const compressSpartanEl = document.getElementById("compress-spartan");
const downloadSpartanEl = document.getElementById("download-spartan");

let lastSpartanProofBytes = null;
let lastSpartanProofFilename = null;
let proverWorker = null;
let activeWasmBundle = null; // "pkg" | "pkg_threads"
let activeWasmThreads = 0;
let runId = 0;
let runInProgress = false;

const urlParams = new URLSearchParams(window.location.search);
const threadsParam = urlParams.get("threads"); // "1" | "0" | null
const threadsForcedOn = threadsParam === "1";
const threadsForcedOff = threadsParam === "0";

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

const supportsThreadsRuntime = supportsWasmThreadsRuntime();

const preferThreads = !threadsForcedOff && supportsThreadsRuntime;
const threadsHint =
  threadsForcedOn ? "?threads=1" : threadsForcedOff ? "?threads=0" : "auto";

const WASM_SINGLE = "./pkg/neo_fold_demo.js";
const WASM_THREADS = "./pkg_threads/neo_fold_demo.js";

function setBadge(el, text, kind) {
  if (!el) return;
  el.textContent = text;
  el.classList.remove("ok", "warn", "bad");
  if (kind) el.classList.add(kind);
}

function setText(el, text) {
  if (!el) return;
  el.textContent = text;
}

async function loadBuildInfo(bundle) {
  if (!commitBtnEl) return;
  setBadge(commitBtnEl, "Commit: loading…", "warn");
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
      setBadge(commitBtnEl, "Commit: unknown", "warn");
      return;
    }

    setBadge(commitBtnEl, `Commit: ${commitShort}${dirty ? "*" : ""}`);
    commitBtnEl.disabled = false;
    commitBtnEl.dataset.commit = commit || commitShort;

    const tipParts = [];
    tipParts.push(commit || commitShort);
    if (dirty) tipParts.push("dirty working tree");
    if (builtAt) tipParts.push(`built: ${builtAt}`);
    commitBtnEl.title = tipParts.join("\n");
  } catch (e) {
    setBadge(commitBtnEl, "Commit: unavailable", "warn");
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
    log("Threads supported, but failed to load threads bundle; falling back to single-thread.");
    log(`Build threads bundle with: ./demos/wasm-demo/build_wasm.sh --threads`);
    log(`Load error: ${String(e)}`);
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

function log(line) {
  logEl.textContent += `${line}\n`;
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

function setButtonsEnabled(enabled) {
  document.getElementById("load-toy").disabled = !enabled;
  document.getElementById("load-toy-folding").disabled = !enabled;
  document.getElementById("load-poseidon2").disabled = !enabled;
  document.getElementById("run").disabled = !enabled;
  document.getElementById("file-input").disabled = !enabled;
  if (compressSpartanEl) compressSpartanEl.disabled = !enabled;
  if (downloadSpartanEl) downloadSpartanEl.disabled = !enabled || !lastSpartanProofBytes;
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

function ensureProverWorker() {
  if (proverWorker) return proverWorker;
  proverWorker = new Worker(new URL("./prover_worker.js", import.meta.url), { type: "module" });
  proverWorker.addEventListener("error", (e) => {
    log(`Worker error: ${e?.message ?? String(e)}`);
    console.error(e);
  });
  return proverWorker;
}

async function loadToy() {
  const resp = await fetch("./examples/toy_square.json");
  if (!resp.ok) throw new Error(`Failed to load toy example: ${resp.status}`);
  const txt = await resp.text();
  jsonEl.value = txt;
  log(`Loaded toy circuit (${txt.length} bytes).`);
}

async function loadToyFolding() {
  const resp = await fetch("./examples/toy_square_folding_8_steps.json");
  if (!resp.ok) throw new Error(`Failed to load toy folding example: ${resp.status}`);
  const txt = await resp.text();
  jsonEl.value = txt;
  log(`Loaded toy folding circuit (8 steps) (${txt.length} bytes).`);
}

async function loadPoseidon2Batch1() {
  const resp = await fetch("./examples/poseidon2_ic_batch_1.json");
  if (!resp.ok) throw new Error(`Failed to load Poseidon2 example: ${resp.status}`);
  const txt = await resp.text();
  jsonEl.value = txt;
  log(`Loaded Poseidon2 IC circuit batch 1 (${txt.length} bytes).`);
}

async function run() {
  const json = jsonEl.value;
  // Keep each run's output self-contained.
  logEl.textContent = "";
  lastSpartanProofBytes = null;
  lastSpartanProofFilename = null;
  if (downloadSpartanEl) downloadSpartanEl.disabled = true;
  if (!json.trim()) {
    log("No JSON provided.");
    return;
  }

  if (runInProgress) {
    log("Run already in progress.");
    return;
  }
  runInProgress = true;

  const doSpartan = Boolean(compressSpartanEl?.checked);

  setButtonsEnabled(false);
  try {
    const worker = ensureProverWorker();
    const id = ++runId;
    const result = await new Promise((resolve, reject) => {
      const onMessage = (ev) => {
        const msg = ev.data;
        if (!msg || msg.id !== id) return;

        if (msg.type === "log") {
          log(msg.line);
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
  } catch (e) {
    log(`ERROR: ${e}`);
    console.error(e);
  } finally {
    setButtonsEnabled(true);
    runInProgress = false;
  }
}

async function main() {
  setButtonsEnabled(false);
  log("Loading wasm...");
  try {
    setBadge(
      statusBundleEl,
      `Bundle: ${preferThreads ? "auto (prefers threads)" : "pkg"} (${threadsHint})`,
      preferThreads ? "warn" : undefined,
    );
    setBadge(
      statusCoiEl,
      `crossOriginIsolated: ${String(self.crossOriginIsolated === true)}`,
      self.crossOriginIsolated === true ? "ok" : "warn",
    );

    if (threadsForcedOn && !supportsThreadsRuntime) {
      log("Threads requested (?threads=1) but not supported in this context.");
      log("Need: crossOriginIsolated + SharedArrayBuffer (COOP/COEP headers).");
      setBadge(statusThreadsEl, "Threads: requested but unavailable", "bad");
    } else if (threadsForcedOff) {
      setBadge(statusThreadsEl, "Threads: disabled (?threads=0)", "warn");
    } else if (supportsThreadsRuntime) {
      setBadge(statusThreadsEl, "Threads: supported (auto)", "warn");
    } else {
      setBadge(statusThreadsEl, "Threads: unavailable (no COOP/COEP)", "warn");
    }

    if (supportsThreadsRuntime) {
      log("Threads supported (cross-origin isolated + SharedArrayBuffer).");
      log(`hardwareConcurrency=${String(navigator.hardwareConcurrency ?? "?")}`);
    } else {
      log("Threads not supported (missing COOP/COEP / SharedArrayBuffer).");
    }

    const { wasm, bundle } = await loadWasmModule();
    window.__neo_fold_wasm = wasm;
    await wasm.default();
    wasm.init_panic_hook();

    setBadge(statusBundleEl, `Bundle: ${bundle} (${threadsHint})`);
    await loadBuildInfo(bundle);
    activeWasmBundle = bundle;

    if (bundle === "pkg_threads") {
      if (typeof wasm.init_thread_pool !== "function") {
        log("ERROR: threads bundle loaded, but wasm-threads exports are missing.");
        setBadge(statusThreadsEl, "Threads: error (missing init_thread_pool)", "bad");
      } else if (!supportsThreadsRuntime) {
        log("ERROR: threads bundle loaded, but SharedArrayBuffer is not available.");
        setBadge(statusThreadsEl, "Threads: disabled (no SharedArrayBuffer)", "bad");
      } else {
        const n = Math.max(1, navigator.hardwareConcurrency ?? 4);
        activeWasmThreads = n;
        log(`Initializing wasm thread pool (${n} threads)...`);
        setBadge(statusThreadsEl, `Threads: initializing (${n})…`, "warn");
        await wasm.init_thread_pool(n);
        log("Wasm thread pool ready.");
        setBadge(statusThreadsEl, `Threads: enabled (${n})`, "ok");
      }
    } else {
      if (threadsForcedOff) {
        setBadge(statusThreadsEl, "Threads: disabled (?threads=0)", "warn");
      } else if (supportsThreadsRuntime) {
        setBadge(
          statusThreadsEl,
          "Threads: disabled (using single-thread bundle)",
          "warn",
        );
      } else {
        setBadge(statusThreadsEl, "Threads: unavailable (no COOP/COEP)", "warn");
      }
    }
  } catch (e) {
    log("Failed to load wasm bundle.");
    log(
      `Did you run ./demos/wasm-demo/build_wasm.sh${preferThreads ? " --threads" : ""} ?`,
    );
    log(String(e));
    console.error(e);
    setBadge(statusThreadsEl, "Threads: error", "bad");
    return;
  }
  log("Wasm loaded.");

  const threadCheckCommand =
    "window.__neo_fold_wasm.default().then(exp => exp.memory.buffer.constructor.name)";
  if (threadCheckCommandEl) threadCheckCommandEl.value = threadCheckCommand;

  if (infoBtnEl && infoPanelEl) {
    infoBtnEl.addEventListener("click", () => {
      infoPanelEl.hidden = !infoPanelEl.hidden;
      if (!infoPanelEl.hidden && threadCheckCommandEl) {
        threadCheckCommandEl.focus();
        threadCheckCommandEl.select();
      }
    });
  }

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
        log(`Copied commit to clipboard: ${commit}`);
      } catch (e) {
        log(`Copy failed: ${String(e)}`);
      }
    });
  }

  document.getElementById("clear-log").addEventListener("click", () => {
    logEl.textContent = "";
  });
  if (downloadSpartanEl) {
    downloadSpartanEl.addEventListener("click", () => {
      if (!lastSpartanProofBytes || !lastSpartanProofFilename) {
        log("No Spartan SNARK available to download (run with Spartan enabled first).");
        return;
      }
      downloadBytes(lastSpartanProofFilename, lastSpartanProofBytes);
      log(
        `Downloaded Spartan SNARK: ${lastSpartanProofFilename} (${fmtBytes(lastSpartanProofBytes.length)})`,
      );
    });
  }
  document.getElementById("load-toy").addEventListener("click", async () => {
    try {
      await loadToy();
    } catch (e) {
      log(`ERROR: ${e}`);
      console.error(e);
    }
  });
  document.getElementById("load-toy-folding").addEventListener("click", async () => {
    try {
      await loadToyFolding();
    } catch (e) {
      log(`ERROR: ${e}`);
      console.error(e);
    }
  });
  document.getElementById("load-poseidon2").addEventListener("click", async () => {
    try {
      await loadPoseidon2Batch1();
    } catch (e) {
      log(`ERROR: ${e}`);
      console.error(e);
    }
  });
  document.getElementById("run").addEventListener("click", run);
  document.getElementById("file-input").addEventListener("change", async (ev) => {
    const file = ev.target.files?.[0];
    if (!file) return;
    const txt = await file.text();
    jsonEl.value = txt;
    log(`Loaded file "${file.name}" (${txt.length} bytes).`);
  });

  await loadToy();
  setButtonsEnabled(true);
}

main().catch((e) => {
  log(`Fatal error: ${e}`);
  console.error(e);
});
