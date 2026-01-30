use wasm_bindgen::prelude::*;

mod riscv_asm;

use js_sys::Date;
use neo_fold::test_export::{
    estimate_proof, folding_summary, parse_test_export_json, run_test_export, TestExportSession,
};
use neo_math::F;
use neo_spartan_bridge::circuit::FoldRunWitness;
use p3_field::PrimeCharacteristicRing;

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[cfg(feature = "wasm-threads")]
#[wasm_bindgen]
pub fn init_thread_pool(num_threads: usize) -> js_sys::Promise {
    wasm_bindgen_rayon::init_thread_pool(num_threads)
}

/// Parse a `TestExport` JSON (same schema as `crates/neo-fold/poseidon2-tests/*.json`),
/// then run prove+verify and return a small result object.
#[wasm_bindgen]
pub fn prove_verify_test_export_json(json: &str) -> Result<JsValue, JsValue> {
    let export = parse_test_export_json(json)
        .map_err(|e| JsValue::from_str(&format!("parse error: {e}")))?;

    let result =
        run_test_export(&export).map_err(|e| JsValue::from_str(&format!("run error: {e}")))?;

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("serialize error: {e}")))
}

#[derive(serde::Serialize)]
struct Rv32FibRunResult {
    n: u32,
    expected: u32,
    verify_ok: bool,
    prove_ms: f64,
    verify_ms: f64,
    trace_len: Option<usize>,
    folds: usize,
    ccs_constraints: usize,
    ccs_variables: usize,
    shout_lookups: Option<usize>,
    spartan: Option<Rv32FibSpartanResult>,
}

#[derive(serde::Serialize)]
struct Rv32FibSpartanResult {
    prove_ms: f64,
    verify_ms: f64,
    verify_ok: bool,
    snark_bytes: usize,
    snark: Vec<u8>,
}

fn fib_u32(n: u32) -> u32 {
    let mut n = n;
    let mut a = 0u32;
    let mut b = 1u32;
    while n > 0 {
        let next = a.wrapping_add(b);
        a = b;
        b = next;
        n -= 1;
    }
    a
}

/// Prove+verify the RV32 Fibonacci program under the B1 shared-bus step circuit.
///
/// Expected guest semantics:
/// - reads `n` from RAM[0x104] (u32)
/// - writes `fib(n)` to RAM[0x100] (u32)
/// - halts via `ecall` (treated as `Halt` in this VM)
#[wasm_bindgen]
pub fn prove_verify_rv32_b1_fibonacci_asm(
    asm: &str,
    n: u32,
    ram_bytes: usize,
    chunk_size: usize,
    max_steps: usize,
    do_spartan: bool,
) -> Result<JsValue, JsValue> {
    let program_bytes = riscv_asm::assemble_rv32_mini_asm(asm).map_err(|e| JsValue::from_str(&e))?;
    if program_bytes.is_empty() {
        return Err(JsValue::from_str("assembled program is empty"));
    }

    let expected = fib_u32(n);
    let expected_f = F::from_u64(expected as u64);

    let mut run = {
        let mut b = neo_fold::riscv_shard::Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
            .xlen(32)
            .ram_bytes(ram_bytes)
            .ram_init_u32(/*addr=*/ 0x104, n)
            .chunk_size(chunk_size)
            .shout_auto_minimal()
            .output(/*output_addr=*/ 0x100, /*expected_output=*/ expected_f);
        if max_steps > 0 {
            b = b.max_steps(max_steps);
        }
        b.prove().map_err(|e| JsValue::from_str(&format!("prove error: {e}")))?
    };

    let prove_ms = run.prove_duration().as_secs_f64() * 1000.0;

    run.verify().map_err(|e| JsValue::from_str(&format!("verify error: {e}")))?;
    let verify_ok = true;
    let verify_ms = run
        .verify_duration()
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0);

    let trace_len = run.riscv_trace_len().ok();
    let folds = run.fold_count();
    let ccs_constraints = run.ccs_num_constraints();
    let ccs_variables = run.ccs_num_variables();
    let shout_lookups = run.shout_lookup_count().ok();

    let spartan = if do_spartan {
        let acc_init = &[];
        let witness = fold_run_witness_placeholder(run.proof());
        let prove_start = Date::now();
        let keypair = neo_spartan_bridge::setup_fold_run(run.params(), run.ccs(), acc_init, run.proof(), witness.clone())
            .map_err(|e| JsValue::from_str(&format!("spartan setup error: {e}")))?;
        let spartan =
            neo_spartan_bridge::prove_fold_run(&keypair.pk, run.params(), run.ccs(), acc_init, run.proof(), witness)
                .map_err(|e| JsValue::from_str(&format!("spartan prove error: {e}")))?;
        let prove_ms = Date::now() - prove_start;

        let verify_start = Date::now();
        let verify_ok = neo_spartan_bridge::verify_fold_run(&keypair.vk, run.params(), run.ccs(), &spartan)
            .map_err(|e| JsValue::from_str(&format!("spartan verify error: {e}")))?;
        let verify_ms = Date::now() - verify_start;

        let snark = spartan.snark_data.clone();
        let snark_bytes = snark.len();

        Some(Rv32FibSpartanResult {
            prove_ms,
            verify_ms,
            verify_ok,
            snark_bytes,
            snark,
        })
    } else {
        None
    };

    let result = Rv32FibRunResult {
        n,
        expected,
        verify_ok,
        prove_ms,
        verify_ms,
        trace_len,
        folds,
        ccs_constraints,
        ccs_variables,
        shout_lookups,
        spartan,
    };

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&format!("serialize error: {e}")))
}

/// Stateful JS-facing session wrapper.
///
/// Construct once from a circuit JSON, then:
/// - `add_step_*` incrementally
/// - `fold_and_prove()` to obtain an opaque proof handle
/// - `verify(proof)` to check it
#[wasm_bindgen]
pub struct NeoFoldSession {
    inner: TestExportSession,
}

fn fold_run_witness_placeholder(run: &neo_fold::shard::ShardProof) -> FoldRunWitness {
    // NOTE: The Spartan bridge circuit currently does not require these witness matrices.
    // Keep them as correctly-shaped placeholders (one entry per step) so we can wire them up later.
    let per_step_empty = (0..run.steps.len()).map(|_| Vec::new()).collect::<Vec<_>>();
    let rlc_rhos = run.steps.iter().map(|s| s.fold.rlc_rhos.clone()).collect::<Vec<_>>();
    FoldRunWitness::from_fold_run(run.clone(), per_step_empty.clone(), rlc_rhos, per_step_empty)
}

#[wasm_bindgen]
impl NeoFoldSession {
    /// Create a new session from a circuit JSON (same fields as `TestExport` but without `witness`).
    ///
    /// Note: serde ignores unknown fields, so passing a full `TestExport` JSON is also accepted.
    #[wasm_bindgen(constructor)]
    pub fn new(circuit_json: &str) -> Result<NeoFoldSession, JsValue> {
        let inner = TestExportSession::new_from_circuit_json(circuit_json)
            .map_err(|e| JsValue::from_str(&e))?;
        Ok(NeoFoldSession { inner })
    }

    pub fn step_count(&self) -> usize {
        self.inner.step_count()
    }

    pub fn setup_timings_ms(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(self.inner.setup_timings_ms())
            .map_err(|e| JsValue::from_str(&format!("serialize error: {e}")))
    }

    pub fn params_summary(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.inner.params_summary())
            .map_err(|e| JsValue::from_str(&format!("serialize error: {e}")))
    }

    pub fn circuit_summary(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.inner.circuit_summary())
            .map_err(|e| JsValue::from_str(&format!("serialize error: {e}")))
    }

    /// Add one step from a witness vector `z` encoded as JSON array of u64s.
    pub fn add_step_witness_json(&mut self, witness_json: &str) -> Result<(), JsValue> {
        self.inner
            .add_step_witness_json(witness_json)
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Add one step from `(x, w)` encoded as JSON arrays of u64s.
    pub fn add_step_io_json(&mut self, x_json: &str, w_json: &str) -> Result<(), JsValue> {
        self.inner
            .add_step_io_json(x_json, w_json)
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Add all witness steps from a full `TestExport` JSON (uses only `witness`).
    pub fn add_steps_from_test_export_json(&mut self, json: &str) -> Result<(), JsValue> {
        self.inner
            .add_steps_from_test_export_json(json)
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Set verifier-side step-linking equality pairs from JSON.
    ///
    /// Format: `[[prev_idx, next_idx], ...]` (must be non-empty for multi-step verification).
    pub fn set_step_linking_pairs_json(&mut self, json: &str) -> Result<(), JsValue> {
        self.inner
            .set_step_linking_pairs_json(json)
            .map_err(|e| JsValue::from_str(&e))
    }

    pub fn fold_and_prove(&mut self) -> Result<NeoFoldProof, JsValue> {
        let (proof, fold_step_ms) = self
            .inner
            .fold_and_prove_with_step_timings()
            .map_err(|e| JsValue::from_str(&e))?;
        Ok(NeoFoldProof { proof, fold_step_ms })
    }

    pub fn verify(&self, proof: &NeoFoldProof) -> Result<bool, JsValue> {
        self.inner.verify(&proof.proof).map_err(|e| JsValue::from_str(&e))
    }

    /// Compress a folding proof into a Spartan2 proof (Merkle-MLE engine).
    pub fn spartan_prove(&self, proof: &NeoFoldProof) -> Result<SpartanCompressedProof, JsValue> {
        let acc_init = self
            .inner
            .initial_accumulator()
            .map(|acc| acc.me.as_slice())
            .unwrap_or(&[]);

        let witness = fold_run_witness_placeholder(&proof.proof);
        let keypair = neo_spartan_bridge::setup_fold_run(self.inner.params(), self.inner.ccs(), acc_init, &proof.proof, witness.clone())
            .map_err(|e| JsValue::from_str(&format!("spartan setup error: {e}")))?;
        let spartan = neo_spartan_bridge::prove_fold_run(
            &keypair.pk,
            self.inner.params(),
            self.inner.ccs(),
            acc_init,
            &proof.proof,
            witness,
        )
        .map_err(|e| JsValue::from_str(&format!("spartan prove error: {e}")))?;

        Ok(SpartanCompressedProof {
            inner: spartan,
            vk: keypair.vk,
        })
    }

    pub fn spartan_verify(&self, proof: &SpartanCompressedProof) -> Result<bool, JsValue> {
        neo_spartan_bridge::verify_fold_run(&proof.vk, self.inner.params(), self.inner.ccs(), &proof.inner)
            .map_err(|e| JsValue::from_str(&format!("spartan verify error: {e}")))
    }
}

/// Opaque proof handle returned by `NeoFoldSession::fold_and_prove()`.
#[wasm_bindgen]
pub struct NeoFoldProof {
    proof: neo_fold::shard::ShardProof,
    fold_step_ms: Vec<f64>,
}

#[wasm_bindgen]
impl NeoFoldProof {
    pub fn step_count(&self) -> usize {
        self.proof.steps.len()
    }

    pub fn fold_step_ms(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.fold_step_ms)
            .map_err(|e| JsValue::from_str(&format!("serialize error: {e}")))
    }

    pub fn proof_estimate(&self) -> Result<JsValue, JsValue> {
        let est = estimate_proof(&self.proof);
        serde_wasm_bindgen::to_value(&est)
            .map_err(|e| JsValue::from_str(&format!("serialize error: {e}")))
    }

    pub fn folding_summary(&self) -> Result<JsValue, JsValue> {
        let summary = folding_summary(&self.proof);
        serde_wasm_bindgen::to_value(&summary)
            .map_err(|e| JsValue::from_str(&format!("serialize error: {e}")))
    }
}

/// Opaque Spartan proof handle returned by `NeoFoldSession::spartan_prove()`.
#[wasm_bindgen]
pub struct SpartanCompressedProof {
    inner: neo_spartan_bridge::api::SpartanProof,
    vk: neo_spartan_bridge::SpartanVerifierKey,
}

#[wasm_bindgen]
impl SpartanCompressedProof {
    /// Size of the downloadable artifact (SNARK proof only; excludes `vk`).
    pub fn bytes_len(&self) -> usize {
        self.inner.snark_bytes_len()
    }

    /// Downloadable Spartan proof bytes (SNARK proof only; excludes `vk`).
    pub fn bytes(&self) -> Vec<u8> {
        self.inner.snark_data.clone()
    }

    /// Size of the combined artifact (vk + snark).
    ///
    /// This is optional in the UI; when present it can be used to estimate vk size as
    /// `(vk+snark) - snark`.
    pub fn vk_and_snark_bytes_len(&self) -> usize {
        // NOTE: neo-spartan-bridge no longer carries vk bytes in the proof object; keep this
        // helper for the demo UI by counting the serialized vk plus the snark bytes.
        let vk_len = bincode::serialize(&self.vk).map(|b| b.len()).unwrap_or(0);
        vk_len + self.inner.snark_data.len()
    }
}
