const logEl = document.getElementById("log");
const jsonEl = document.getElementById("circuit-json");
const statusBundleEl = document.getElementById("status-bundle");
const statusCoiEl = document.getElementById("status-coi");
const statusThreadsEl = document.getElementById("status-threads");
const infoBtnEl = document.getElementById("info-btn");
const infoPanelEl = document.getElementById("info-panel");
const threadCheckCommandEl = document.getElementById("thread-check-command");
const copyThreadCheckEl = document.getElementById("copy-thread-check");
const copyStatusEl = document.getElementById("copy-status");

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
  if (!json.trim()) {
    log("No JSON provided.");
    return;
  }

  setButtonsEnabled(false);
  try {
    log("Running prove+verify…");
    log(`Input JSON size: ${fmtBytes(json.length)}`);
    const parsed = tryParseTestExport(json);
    if (parsed) {
      log(
        `Input export: constraints=${parsed.num_constraints} variables=${parsed.num_variables} steps=${parsed.witness?.length ?? "?"}`,
      );
    }

    const start = performance.now();
    const result = window.__neo_fold_wasm.prove_verify_test_export_json(json);
    const ms = performance.now() - start;

    log(`OK: verify_ok=${result.verify_ok} steps=${result.steps} (total ${fmtMs(ms)})`);

    if (result.params) {
      const p = result.params;
      log(
        `Params: b=${p.b} d=${p.d} kappa=${p.kappa} k_rho=${p.k_rho} T=${p.T} s=${p.s} lambda=${p.lambda}`,
      );
    }

    if (result.circuit) {
      const c = result.circuit;
      log(
        `Circuit (R1CS): constraints=${c.r1cs_constraints} variables=${c.r1cs_variables} padded_n=${c.r1cs_padded_n} A_nnz=${c.r1cs_a_nnz} B_nnz=${c.r1cs_b_nnz} C_nnz=${c.r1cs_c_nnz}`,
      );
      log(
        `Witness: steps=${c.witness_steps} fields_total=${c.witness_fields_total} fields_min=${c.witness_fields_min} fields_max=${c.witness_fields_max} nonzero=${c.witness_nonzero_fields_total} (${(c.witness_nonzero_ratio * 100).toFixed(2)}%)`,
      );
      log(
        `Circuit (CCS): n=${c.ccs_n} m=${c.ccs_m} t=${c.ccs_t} max_degree=${c.ccs_max_degree} poly_terms=${c.ccs_poly_terms} nnz_total=${c.ccs_matrix_nnz_total}`,
      );
      if (Array.isArray(c.ccs_matrix_nnz) && c.ccs_matrix_nnz.length > 0) {
        log(`CCS matrices nnz: [${c.ccs_matrix_nnz.join(", ")}]`);
      }
    }

    if (result.timings_ms) {
      const t = result.timings_ms;
      log(
        `Timings: ajtai=${fmtMs(t.ajtai_setup)} build_ccs=${fmtMs(t.build_ccs)} prepare_witness=${fmtMs(t.prepare_witness)} session_init=${fmtMs(t.session_init)}`,
      );
      log(
        `Timings: add_steps_total=${fmtMs(t.add_steps_total)} (avg=${fmtMs(t.add_step_avg)} min=${fmtMs(t.add_step_min)} max=${fmtMs(t.add_step_max)}) prove=${fmtMs(t.fold_and_prove)} verify=${fmtMs(t.verify)} total=${fmtMs(t.total)}`,
      );
      if (Array.isArray(t.fold_steps) && t.fold_steps.length > 0) {
        log(`Folding prove per-step: ${fmtMsList(t.fold_steps)}`);
        log(
          `Folding prove per-step stats: avg=${fmtMs(t.fold_step_avg)} min=${fmtMs(t.fold_step_min)} max=${fmtMs(t.fold_step_max)}`,
        );
      }
    }

    if (result.proof_estimate) {
      const pe = result.proof_estimate;
      log(
        `Proof estimate: proof_steps=${pe.proof_steps} final_acc_len=${pe.final_accumulator_len}`,
      );
      log(
        `Proof estimate: commitments fold_lane=${pe.fold_lane_commitments} mem_cpu_val=${pe.mem_cpu_val_claim_commitments} val_lane=${pe.val_lane_commitments} total=${pe.total_commitments}`,
      );
      log(
        `Proof estimate: commitment_bytes=${pe.commitment_bytes} (d=${pe.commitment_d} kappa=${pe.commitment_kappa}) estimated_commitment_bytes=${fmtBytes(pe.estimated_commitment_bytes)}`,
      );
    }

    if (result.folding) {
      const f = result.folding;
      log(`Folding k_in per step: ${fmtList(f.k_in)}`);
      log(`Folding accumulator len after step: ${fmtList(f.acc_len_after)}`);
    }

    log("Raw result:");
    log(safeStringify(result));
  } catch (e) {
    log(`ERROR: ${e}`);
    console.error(e);
  } finally {
    setButtonsEnabled(true);
  }
}

async function main() {
  setButtonsEnabled(false);
  log("Loading wasm...");
  try {
    setBadge(statusBundleEl, `Bundle: ${preferThreads ? "auto (prefers threads)" : "pkg"} (${threadsHint})`, preferThreads ? "warn" : undefined);
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

    if (bundle === "pkg_threads") {
      if (typeof wasm.init_thread_pool !== "function") {
        log("ERROR: threads bundle loaded, but wasm-threads exports are missing.");
        setBadge(statusThreadsEl, "Threads: error (missing init_thread_pool)", "bad");
      } else if (!supportsThreadsRuntime) {
        log("ERROR: threads bundle loaded, but SharedArrayBuffer is not available.");
        setBadge(statusThreadsEl, "Threads: disabled (no SharedArrayBuffer)", "bad");
      } else {
        const n = Math.max(1, navigator.hardwareConcurrency ?? 4);
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

  document.getElementById("clear-log").addEventListener("click", () => {
    logEl.textContent = "";
  });
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
