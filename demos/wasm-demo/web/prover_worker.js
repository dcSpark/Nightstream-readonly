let wasm = null;
let wasmBundle = null; // "pkg" | "pkg_threads"
let wasmThreads = 0;

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

function log(id, line) {
  emit(id, { type: "log", line: String(line) });
}

async function ensureWasm({ bundle, threads }) {
  if (wasm && wasmBundle === bundle && wasmThreads === threads) return;

  wasmBundle = bundle;
  wasmThreads = threads ?? 0;

  const entry =
    bundle === "pkg_threads" ? "./pkg_threads/neo_fold_demo.js" : "./pkg/neo_fold_demo.js";

  wasm = await import(entry);
  await wasm.default();
  wasm.init_panic_hook();

  if (bundle === "pkg_threads" && typeof wasm.init_thread_pool === "function") {
    const n = Math.max(1, wasmThreads || (navigator.hardwareConcurrency ?? 4));
    await wasm.init_thread_pool(n);
  }
}

async function runProveVerify({ id, json, doSpartan, bundle, threads }) {
  await ensureWasm({ bundle, threads });

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
  const session = new wasm.NeoFoldSession(json);
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
  const proveStart = performance.now();
  const foldProof = session.fold_and_prove();
  const proveMs = performance.now() - proveStart;

  const foldSteps = foldProof.fold_step_ms();
  log(id, `Timings: prove=${fmtMs(proveMs)}`);
  if (Array.isArray(foldSteps) && foldSteps.length > 0) {
    log(id, `Folding prove per-step: ${fmtMsList(foldSteps)}`);
    const sum = foldSteps.reduce((acc, v) => acc + v, 0);
    const avg = sum / foldSteps.length;
    const min = foldSteps.reduce((acc, v) => Math.min(acc, v), Infinity);
    const max = foldSteps.reduce((acc, v) => Math.max(acc, v), 0);
    log(id, `Folding prove per-step stats: avg=${fmtMs(avg)} min=${fmtMs(min)} max=${fmtMs(max)}`);
  }

  log(id, "Verifying folding proof…");
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
    const spStart = performance.now();
    const spartan = session.spartan_prove(foldProof);
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

  return { spartanSnarkBuf, spartanFilename };
}

self.addEventListener("message", async (ev) => {
  const msg = ev.data;
  const id = msg?.id;
  if (typeof id !== "number") return;

  if (msg?.type !== "run") return;

  try {
    const { spartanSnarkBuf, spartanFilename } = await runProveVerify(msg);
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

