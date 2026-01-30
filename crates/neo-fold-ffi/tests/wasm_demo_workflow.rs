use neo_fold_ffi::{neo_fold_free_bytes, neo_fold_run_wasm_demo_workflow_json};

fn call_workflow(json: &str) -> Result<serde_json::Value, String> {
    let mut out_ptr: *mut u8 = core::ptr::null_mut();
    let mut out_len: usize = 0;
    let mut err_ptr: *mut u8 = core::ptr::null_mut();
    let mut err_len: usize = 0;

    let rc = neo_fold_run_wasm_demo_workflow_json(
        json.as_ptr(),
        json.len(),
        0,
        &mut out_ptr,
        &mut out_len,
        &mut err_ptr,
        &mut err_len,
    );

    if rc != 0 {
        let err = if !err_ptr.is_null() && err_len > 0 {
            let bytes = unsafe { core::slice::from_raw_parts(err_ptr, err_len) };
            String::from_utf8_lossy(bytes).to_string()
        } else {
            format!("neo_fold_run_wasm_demo_workflow_json failed with rc={rc} (no error buffer)")
        };
        neo_fold_free_bytes(err_ptr, err_len);
        return Err(err);
    }

    assert!(err_ptr.is_null());
    assert_eq!(err_len, 0);

    let bytes = unsafe { core::slice::from_raw_parts(out_ptr, out_len) }.to_vec();
    neo_fold_free_bytes(out_ptr, out_len);

    serde_json::from_slice(&bytes).map_err(|e| format!("output JSON parse error: {e}"))
}

#[test]
fn workflow_returns_expected_shape() {
    let json = include_str!("../../../demos/wasm-demo/web/examples/toy_square.json");
    let v = call_workflow(json).expect("workflow should succeed");

    assert_eq!(v["steps"].as_u64(), Some(1));
    assert_eq!(v["verify_ok"].as_bool(), Some(true));
    assert!(v["params"].is_object());
    assert!(v["circuit"].is_object());
    assert!(v["timings_ms"].is_object());
    assert!(v["proof_estimate"].is_object());
    assert!(v["folding"].is_object());
}

#[test]
fn workflow_reports_parse_error() {
    let err = call_workflow("{ not valid json").expect_err("expected parse failure");
    assert!(err.contains("session init error") || err.contains("parse") || err.contains("JSON"));
}
