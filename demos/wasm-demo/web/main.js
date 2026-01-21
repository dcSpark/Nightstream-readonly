const logEl = document.getElementById("log");
const jsonEl = document.getElementById("circuit-json");

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
    const wasm = await import("./pkg/neo_fold_demo.js");
    window.__neo_fold_wasm = wasm;
    await wasm.default();
    wasm.init_panic_hook();
  } catch (e) {
    log("Failed to load wasm bundle.");
    log("Did you run ./demos/wasm-demo/build_wasm.sh ?");
    log(String(e));
    console.error(e);
    return;
  }
  log("Wasm loaded.");

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
