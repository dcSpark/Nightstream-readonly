use wasm_bindgen::prelude::*;

use neo_fold::test_export::{
    estimate_proof, folding_summary, parse_test_export_json, run_test_export, TestExportSession,
};

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
