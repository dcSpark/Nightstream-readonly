use wasm_bindgen::prelude::*;

use neo_fold::test_export::{parse_test_export_json, run_test_export};

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
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
