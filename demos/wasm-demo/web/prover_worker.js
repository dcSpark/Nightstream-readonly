let wasm = null;
let wasmBundle = null; // "pkg" | "pkg_threads"
let wasmThreads = 0;
let threadsDisabled = false;
let threadsDisabledNotified = false;
let threadsDisableReason = null;

// Minimal Wasm module that requires the threads proposal (shared memory + atomic instruction).
// Generated from:
// (module (memory 1 1 shared) (func i32.const 0 i32.atomic.load drop))
const WASM_THREADS_VALIDATE_BYTES = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 5, 4, 1, 3,
  1, 1, 10, 11, 1, 9, 0, 65, 0, 254, 16, 2, 0, 26, 11,
]);

function supportsWasmThreadsRuntime() {
  if (typeof WebAssembly !== "object" || typeof WebAssembly.Memory !== "function") return false;
  if (typeof WebAssembly.validate !== "function") return false;
  if (self.crossOriginIsolated !== true) return false;
  if (typeof SharedArrayBuffer !== "function") return false;
  if (typeof Atomics !== "object") return false;

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

function chooseThreadCount(requested) {
  const n =
    typeof requested === "number" && Number.isFinite(requested) && requested > 0
      ? requested
      : navigator.hardwareConcurrency ?? 4;
  return Math.min(Math.max(1, n), 8);
}

function withTimeout(promise, timeoutMs, label) {
  const ms = Math.max(0, timeoutMs ?? 0);
  if (ms === 0) return promise;
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      setTimeout(() => reject(new Error(`${label} timed out after ${ms} ms`)), ms);
    }),
  ]);
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

function safeStringify(value) {
  return JSON.stringify(
    value,
    (_k, v) => (typeof v === "bigint" ? v.toString() : v),
    2,
  );
}

function tryParseTestExport(json) {
  try {
    return JSON.parse(json);
  } catch {
    return null;
  }
}

function emit(id, msg) {
  self.postMessage({ id, ...msg });
}

function phase(id, phase) {
  emit(id, { type: "phase", phase: String(phase) });
}

function notifyThreadsDisabled(id, reason) {
  if (threadsDisabledNotified) return;
  threadsDisabledNotified = true;
  emit(id, {
    type: "worker_state",
    bundle: "pkg",
    threads: 0,
    threads_disabled: true,
    threads_disabled_reason: String(reason ?? ""),
    restart_worker: true,
  });
}

function log(id, line, level = "info") {
  emit(id, { type: "log", level, line: String(line) });
}

async function ensureWasm({ id, bundle, threads }) {
  if (wasm && wasmBundle === bundle && wasmThreads === threads) return;

  phase(id, "Loading wasm…");

  const supportsThreads = supportsWasmThreadsRuntime();

  let selectedBundle = bundle === "pkg_threads" ? "pkg_threads" : "pkg";
  let selectedThreads = threads ?? 0;

  if (threadsDisabled && selectedBundle === "pkg_threads") {
    log(id, "Threads previously failed; using single-thread bundle.");
    selectedBundle = "pkg";
    selectedThreads = 0;
  }

  if (selectedBundle === "pkg_threads" && !supportsThreads) {
    log(id, "Threads requested, but wasm threads are not available in this worker.", "warn");
    log(id, "Falling back to single-thread bundle.", "warn");
    threadsDisabled = true;
    threadsDisableReason = "Wasm threads not available in this worker";
    notifyThreadsDisabled(id, threadsDisableReason);
    selectedBundle = "pkg";
    selectedThreads = 0;
  }

  if (selectedBundle === "pkg_threads") {
    selectedThreads = chooseThreadCount(selectedThreads);
  } else {
    selectedThreads = 0;
  }

  if (wasm && wasmBundle === selectedBundle && wasmThreads === selectedThreads) return;

  async function loadBundleOrThrow(which) {
    const entry =
      which === "pkg_threads" ? "./pkg_threads/neo_fold_demo.js" : "./pkg/neo_fold_demo.js";
    const mod = await import(entry);
    await mod.default();
    mod.init_panic_hook();
    return mod;
  }

  try {
    wasm = await loadBundleOrThrow(selectedBundle);
  } catch (e) {
    if (selectedBundle === "pkg_threads") {
      log(id, "Failed to load threads bundle; falling back to single-thread.", "warn");
      log(id, `Load error: ${String(e)}`, "warn");
      selectedBundle = "pkg";
      selectedThreads = 0;
      wasm = await loadBundleOrThrow("pkg");
    } else {
      throw e;
    }
  }

  if (selectedBundle === "pkg_threads" && typeof wasm.init_thread_pool === "function") {
    const n = selectedThreads;
    phase(id, "Initializing threads…");
    log(id, `Initializing wasm thread pool (${n} threads)...`);
    try {
      await withTimeout(wasm.init_thread_pool(n), 8000, "init_thread_pool");
      log(id, "Wasm thread pool ready.");
    } catch (e) {
      log(id, `Threads init failed; falling back to single-thread. Error: ${String(e)}`, "warn");
      threadsDisabled = true;
      threadsDisableReason = String(e);
      notifyThreadsDisabled(id, threadsDisableReason);
      selectedBundle = "pkg";
      selectedThreads = 0;
      wasm = await loadBundleOrThrow("pkg");
    }
  }

  wasmBundle = selectedBundle;
  wasmThreads = selectedThreads;

  phase(id, "Ready");
}

async function runProveVerify({ id, json, doSpartan, bundle, threads }) {
  await ensureWasm({ id, bundle, threads });

  phase(id, "Preparing…");
  let session = null;
  let foldProof = null;
  let spartan = null;

  try {
    log(id, "Running prove+verify…");
    log(id, `Input JSON size: ${fmtBytes(json.length)}`);
    const parsed = tryParseTestExport(json);
    if (parsed) {
      log(
        id,
        `Input export: constraints=${parsed.num_constraints} variables=${parsed.num_variables} steps=${parsed.witness?.length ?? "?"}`,
      );
    }

    const totalStart = performance.now();

    const createStart = performance.now();
    session = new wasm.NeoFoldSession(json);
    const createMs = performance.now() - createStart;

    const setup = session.setup_timings_ms();
    const params = session.params_summary();
    const circuit = session.circuit_summary();

    log(id, `Session ready (${fmtMs(createMs)})`);

    if (params) {
      log(
        id,
        `Params: b=${params.b} d=${params.d} kappa=${params.kappa} k_rho=${params.k_rho} T=${params.T} s=${params.s} lambda=${params.lambda}`,
      );
    }

    if (circuit) {
      log(
        id,
        `Circuit (R1CS): constraints=${circuit.r1cs_constraints} variables=${circuit.r1cs_variables} padded_n=${circuit.r1cs_padded_n} A_nnz=${circuit.r1cs_a_nnz} B_nnz=${circuit.r1cs_b_nnz} C_nnz=${circuit.r1cs_c_nnz}`,
      );
      log(
        id,
        `Witness: steps=${circuit.witness_steps} fields_total=${circuit.witness_fields_total} fields_min=${circuit.witness_fields_min} fields_max=${circuit.witness_fields_max} nonzero=${circuit.witness_nonzero_fields_total} (${(circuit.witness_nonzero_ratio * 100).toFixed(2)}%)`,
      );
      log(
        id,
        `Circuit (CCS): n=${circuit.ccs_n} m=${circuit.ccs_m} t=${circuit.ccs_t} max_degree=${circuit.ccs_max_degree} poly_terms=${circuit.ccs_poly_terms} nnz_total=${circuit.ccs_matrix_nnz_total}`,
      );
      if (Array.isArray(circuit.ccs_matrix_nnz) && circuit.ccs_matrix_nnz.length > 0) {
        log(id, `CCS matrices nnz: [${circuit.ccs_matrix_nnz.join(", ")}]`);
      }
    }

    if (setup) {
      log(
        id,
        `Timings: ajtai=${fmtMs(setup.ajtai_setup)} build_ccs=${fmtMs(setup.build_ccs)} session_init=${fmtMs(setup.session_init)}`,
      );
    }

    log(id, "Adding witness steps…");
    const addStart = performance.now();
    session.add_steps_from_test_export_json(json);
    const addMs = performance.now() - addStart;
    log(id, `Timings: add_steps_total=${fmtMs(addMs)}`);

    log(id, "Folding + proving…");
    phase(id, "Proving…");
    const proveStart = performance.now();
    foldProof = session.fold_and_prove();
    const proveMs = performance.now() - proveStart;

    const foldSteps = foldProof.fold_step_ms();
    log(id, `Timings: prove=${fmtMs(proveMs)}`);
    if (Array.isArray(foldSteps) && foldSteps.length > 0) {
      log(id, `Folding prove per-step: ${fmtMsList(foldSteps)}`);
      const sum = foldSteps.reduce((acc, v) => acc + v, 0);
      const avg = sum / foldSteps.length;
      const min = foldSteps.reduce((acc, v) => Math.min(acc, v), Infinity);
      const max = foldSteps.reduce((acc, v) => Math.max(acc, v), 0);
      log(
        id,
        `Folding prove per-step stats: avg=${fmtMs(avg)} min=${fmtMs(min)} max=${fmtMs(max)}`,
      );
    }

    log(id, "Verifying folding proof…");
    phase(id, "Verifying…");
    const verifyStart = performance.now();
    const verifyOk = session.verify(foldProof);
    const verifyMs = performance.now() - verifyStart;
    log(id, `Timings: verify=${fmtMs(verifyMs)}`);

    const totalMs = performance.now() - totalStart;
    log(id, `OK: verify_ok=${verifyOk} steps=${foldProof.step_count()} (total ${fmtMs(totalMs)})`);

    const proofEstimate = foldProof.proof_estimate();
    if (proofEstimate) {
      log(
        id,
        `Proof estimate: proof_steps=${proofEstimate.proof_steps} final_acc_len=${proofEstimate.final_accumulator_len}`,
      );
      log(
        id,
        `Proof estimate: commitments fold_lane=${proofEstimate.fold_lane_commitments} mem_cpu_val=${proofEstimate.mem_cpu_val_claim_commitments} val_lane=${proofEstimate.val_lane_commitments} total=${proofEstimate.total_commitments}`,
      );
      log(
        id,
        `Proof estimate: commitment_bytes=${proofEstimate.commitment_bytes} (d=${proofEstimate.commitment_d} kappa=${proofEstimate.commitment_kappa}) estimated_commitment_bytes=${fmtBytes(proofEstimate.estimated_commitment_bytes)}`,
      );
    }

    const folding = foldProof.folding_summary();
    if (folding) {
      log(id, `Folding k_in per step: ${fmtList(folding.k_in)}`);
      log(id, `Folding accumulator len after step: ${fmtList(folding.acc_len_after)}`);
    }

    let spartanSnarkBuf = null;
    let spartanFilename = null;

    let spartanProveMs = undefined;
    let spartanVerifyMs = undefined;
    let spartanVerifyOk = undefined;
    let spartanSnarkBytesLen = undefined;
    let spartanPackedBytesLen = undefined;
    let spartanVkBytesLen = undefined;

    if (doSpartan) {
      log(id, "Compressing with Spartan2…");
      phase(id, "Compressing…");
      const spStart = performance.now();
      spartan = session.spartan_prove(foldProof);
      spartanProveMs = performance.now() - spStart;

      const snarkBytes = spartan.bytes();
      spartanSnarkBytesLen = snarkBytes.length;

      spartanPackedBytesLen =
        typeof spartan.vk_and_snark_bytes_len === "function"
          ? spartan.vk_and_snark_bytes_len()
          : undefined;
      spartanVkBytesLen =
        typeof spartanPackedBytesLen === "number"
          ? Math.max(0, spartanPackedBytesLen - spartanSnarkBytesLen)
          : undefined;

      const sizeParts = [`snark=${fmtBytes(spartanSnarkBytesLen)}`];
      if (typeof spartanVkBytesLen === "number") sizeParts.push(`vk=${fmtBytes(spartanVkBytesLen)}`);
      if (typeof spartanPackedBytesLen === "number") {
        sizeParts.push(`total(vk+snark)=${fmtBytes(spartanPackedBytesLen)}`);
      }
      log(id, `Spartan2: prove=${fmtMs(spartanProveMs)} ${sizeParts.join(" ")}`);

      phase(id, "Verifying Spartan2…");
      const spVerifyStart = performance.now();
      spartanVerifyOk = session.spartan_verify(spartan);
      spartanVerifyMs = performance.now() - spVerifyStart;
      log(id, `Spartan2: verify=${fmtMs(spartanVerifyMs)} ok=${String(spartanVerifyOk)}`);

      // Transfer SNARK bytes back to the UI thread for downloading.
      spartanSnarkBuf = snarkBytes.buffer.slice(
        snarkBytes.byteOffset,
        snarkBytes.byteOffset + snarkBytes.byteLength,
      );
      spartanFilename = `neo_fold_spartan_snark_${Date.now()}.bin`;
    }

    const raw = {
      steps: foldProof.step_count(),
      verify_ok: verifyOk,
      circuit,
      params,
      timings_ms: {
        ajtai_setup: setup?.ajtai_setup,
        build_ccs: setup?.build_ccs,
        session_init: setup?.session_init,
        add_steps_total: addMs,
        fold_and_prove: proveMs,
        fold_steps: foldSteps,
        verify: verifyMs,
        total: totalMs,
      },
      proof_estimate: proofEstimate,
      folding,
      spartan: doSpartan
        ? {
            prove_ms: spartanProveMs,
            verify_ms: spartanVerifyMs,
            verify_ok: spartanVerifyOk,
            snark_bytes: spartanSnarkBytesLen,
            vk_bytes: spartanVkBytesLen,
            vk_and_snark_bytes: spartanPackedBytesLen,
          }
        : undefined,
    };

    log(id, "\n\nRaw result:");
    log(id, safeStringify(raw));

    phase(id, "Done.");
    return { spartanSnarkBuf, spartanFilename };
  } finally {
    try {
      spartan?.free?.();
    } catch {}
    try {
      foldProof?.free?.();
    } catch {}
    try {
      session?.free?.();
    } catch {}
  }
}

async function runRv32Fibonacci({ id, asm, riscv, doSpartan, bundle, threads }) {
  await ensureWasm({ id, bundle, threads });

  phase(id, "Preparing…");
  const src = String(asm ?? "");
  const n = typeof riscv?.n === "number" ? riscv.n : null;
  const ramBytes = typeof riscv?.ram_bytes === "number" ? riscv.ram_bytes : null;
  const chunkSize = typeof riscv?.chunk_size === "number" ? riscv.chunk_size : null;
  const maxSteps = typeof riscv?.max_steps === "number" ? riscv.max_steps : 0;

  if (!src.trim()) throw new Error("No RISC-V program provided.");
  if (!Number.isFinite(n) || n < 0) throw new Error("Invalid riscv.n");
  if (!Number.isFinite(ramBytes) || ramBytes <= 0) throw new Error("Invalid riscv.ram_bytes");
  if (!Number.isFinite(chunkSize) || chunkSize <= 0) throw new Error("Invalid riscv.chunk_size");
  if (!Number.isFinite(maxSteps) || maxSteps < 0) throw new Error("Invalid riscv.max_steps");

  log(id, "Running RV32 Fibonacci prove+verify…");
  log(id, `Input text size: ${fmtBytes(src.length)}`);
  log(id, `Config: n=${n} ram_bytes=${ramBytes} chunk_size=${chunkSize} max_steps=${maxSteps}`);

  async function runOnce() {
    phase(id, "Proving…");
    const totalStart = performance.now();
    const result = wasm.prove_verify_rv32_b1_fibonacci_asm(
      src,
      n,
      ramBytes,
      chunkSize,
      maxSteps,
      Boolean(doSpartan),
    );
    const totalMs = performance.now() - totalStart;
    return { result, totalMs };
  }

  let result;
  let totalMs;
  try {
    ({ result, totalMs } = await runOnce());
  } catch (e) {
    const isThreadsBundle = wasmBundle === "pkg_threads";
    const msg = String(e);
    const isTrap =
      msg.includes("RuntimeError: unreachable") ||
      msg.includes("unreachable") ||
      msg.includes("memory access out of bounds");

    // If the threads bundle panics/traps at runtime, retry once in the single-thread bundle.
    // This avoids bricking the demo when the threads build is stale or incompatible with the
    // current browser/runtime.
    if (isThreadsBundle && isTrap && !threadsDisabled) {
      log(id, `Threads run crashed (${msg}). Falling back to single-thread bundle…`, "warn");
      threadsDisabled = true;
      threadsDisableReason = msg;
      notifyThreadsDisabled(id, threadsDisableReason);

      await ensureWasm({ id, bundle: "pkg", threads: 0 });
      log(id, "Retrying RV32 Fibonacci prove+verify in single-thread mode…", "warn");
      ({ result, totalMs } = await runOnce());
    } else {
      throw e;
    }
  }

  if (result) {
    log(
      id,
      `OK: verify_ok=${String(result.verify_ok)} expected=fib(${result.n})=${result.expected} folds=${result.folds} (total ${fmtMs(totalMs)})`,
    );
    log(id, `Timings: prove=${fmtMs(result.prove_ms)} verify=${fmtMs(result.verify_ms)}`);
    if (typeof result.trace_len === "number") log(id, `Trace length: ${result.trace_len} instructions`);
    log(
      id,
      `Circuit (CCS): constraints=${result.ccs_constraints} variables=${result.ccs_variables} shout_lookups=${String(result.shout_lookups ?? "?")}`,
    );
  }

  let spartanSnarkBuf = null;
  let spartanFilename = null;

  if (result?.spartan?.snark) {
    let snarkBytes = result.spartan.snark;
    if (Array.isArray(snarkBytes)) snarkBytes = new Uint8Array(snarkBytes);
    if (!(snarkBytes instanceof Uint8Array)) snarkBytes = new Uint8Array(snarkBytes);
    spartanSnarkBuf = snarkBytes.buffer.slice(snarkBytes.byteOffset, snarkBytes.byteOffset + snarkBytes.byteLength);
    spartanFilename = `neo_fold_spartan_snark_${Date.now()}.bin`;

    log(
      id,
      `Spartan2: prove=${fmtMs(result.spartan.prove_ms)} verify=${fmtMs(result.spartan.verify_ms)} ok=${String(result.spartan.verify_ok)} snark=${fmtBytes(result.spartan.snark_bytes)}`,
    );
  }

  const raw = {
    ...result,
    spartan: result?.spartan
      ? {
          ...result.spartan,
          snark: undefined, // avoid dumping large byte arrays
        }
      : undefined,
  };

  log(id, "\n\nRaw result:");
  log(id, safeStringify(raw));

  phase(id, "Done.");
  return { spartanSnarkBuf, spartanFilename };
}

self.addEventListener("message", async (ev) => {
  const msg = ev.data;
  const id = msg?.id;
  if (typeof id !== "number") return;

  if (msg?.type !== "run") return;

  try {
    const mode = typeof msg?.mode === "string" ? msg.mode : "test_export";
    const { spartanSnarkBuf, spartanFilename } =
      mode === "rv32_fibonacci" ? await runRv32Fibonacci(msg) : await runProveVerify(msg);
    if (spartanSnarkBuf && spartanFilename) {
      self.postMessage(
        { type: "done", id, spartan: { filename: spartanFilename, bytes: spartanSnarkBuf } },
        [spartanSnarkBuf],
      );
    } else {
      emit(id, { type: "done" });
    }
  } catch (e) {
    emit(id, { type: "error", error: String(e) });
  }
});
