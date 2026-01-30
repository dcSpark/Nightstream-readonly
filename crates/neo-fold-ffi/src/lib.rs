use neo_fold::test_export::{
    estimate_proof, folding_summary, parse_test_export_json, run_test_export, TestExportSession,
};

#[cfg(feature = "spartan")]
use neo_spartan_bridge::circuit::FoldRunWitness;

use std::time::Instant;

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(feature = "spartan")]
fn fold_run_witness_placeholder(run: &neo_fold::shard::ShardProof) -> FoldRunWitness {
    let pi_ccs_proofs = run.steps.iter().map(|s| s.fold.ccs_proof.clone()).collect();
    let rlc_rhos = run.steps.iter().map(|s| s.fold.rlc_rhos.clone()).collect();

    // NOTE: The current Spartan bridge circuit does not yet use these matrices
    // (they are kept as correctly-shaped placeholders so we can wire them up
    // later without changing the FFI API).
    let per_step_empty = (0..run.steps.len()).map(|_| Vec::new()).collect::<Vec<_>>();

    FoldRunWitness::from_fold_run(run.clone(), pi_ccs_proofs, per_step_empty.clone(), rlc_rhos, per_step_empty)
}

fn write_null(out_ptr: *mut *mut u8, out_len: *mut usize) {
    if !out_ptr.is_null() {
        // SAFETY: caller promised `out_ptr` is valid when non-null.
        unsafe { *out_ptr = core::ptr::null_mut() };
    }
    if !out_len.is_null() {
        // SAFETY: caller promised `out_len` is valid when non-null.
        unsafe { *out_len = 0 };
    }
}

fn write_allocated_bytes(out_ptr: *mut *mut u8, out_len: *mut usize, bytes: Vec<u8>) -> i32 {
    if out_ptr.is_null() || out_len.is_null() {
        return 1;
    }

    let boxed = bytes.into_boxed_slice();
    let len = boxed.len();
    let ptr = Box::into_raw(boxed) as *mut u8;

    // SAFETY: caller provided valid pointers for outputs.
    unsafe {
        *out_ptr = ptr;
        *out_len = len;
    }
    0
}

/// Runs prove+verify for a `TestExport` JSON (UTF-8 bytes).
///
/// On success returns 0 and sets `(*out_ptr, *out_len)` to an allocated UTF-8 JSON buffer
/// containing the `TestExportResult`.
///
/// On failure returns non-zero and sets `(*err_ptr, *err_len)` to an allocated UTF-8 error
/// message. Buffers must be freed with `neo_fold_free_bytes`.
#[no_mangle]
pub extern "C" fn neo_fold_prove_verify_test_export_json(
    json_ptr: *const u8,
    json_len: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
    err_ptr: *mut *mut u8,
    err_len: *mut usize,
) -> i32 {
    if out_ptr.is_null() || out_len.is_null() || err_ptr.is_null() || err_len.is_null() {
        return 1;
    }

    write_null(out_ptr, out_len);
    write_null(err_ptr, err_len);

    if json_ptr.is_null() {
        let _ = write_allocated_bytes(
            err_ptr,
            err_len,
            b"null json_ptr passed to neo_fold_prove_verify_test_export_json".to_vec(),
        );
        return 2;
    }

    let json_bytes = unsafe { core::slice::from_raw_parts(json_ptr, json_len) };
    let json = match core::str::from_utf8(json_bytes) {
        Ok(s) => s,
        Err(e) => {
            let _ = write_allocated_bytes(err_ptr, err_len, format!("input is not valid UTF-8: {e}").into_bytes());
            return 3;
        }
    };

    let export = match parse_test_export_json(json) {
        Ok(e) => e,
        Err(e) => {
            let _ = write_allocated_bytes(err_ptr, err_len, format!("parse error: {e}").into_bytes());
            return 4;
        }
    };

    let result = match run_test_export(&export) {
        Ok(r) => r,
        Err(e) => {
            let _ = write_allocated_bytes(err_ptr, err_len, format!("run error: {e}").into_bytes());
            return 5;
        }
    };

    let out_json = match serde_json::to_vec(&result) {
        Ok(v) => v,
        Err(e) => {
            let _ = write_allocated_bytes(err_ptr, err_len, format!("serialize error: {e}").into_bytes());
            return 6;
        }
    };

    write_allocated_bytes(out_ptr, out_len, out_json)
}

/// Runs the same high-level workflow as `demos/wasm-demo/web/prover_worker.js` and returns a
/// JSON result shaped similarly to its “Raw result” object.
///
/// This is intended for side-by-side comparisons between wasm and native (iOS).
///
/// If `do_spartan != 0`, this also runs Spartan2 compression (requires building this crate with
/// `--features spartan`).
///
/// Buffers must be freed with `neo_fold_free_bytes`.
#[no_mangle]
pub extern "C" fn neo_fold_run_wasm_demo_workflow_json(
    json_ptr: *const u8,
    json_len: usize,
    do_spartan: i32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
    err_ptr: *mut *mut u8,
    err_len: *mut usize,
) -> i32 {
    if out_ptr.is_null() || out_len.is_null() || err_ptr.is_null() || err_len.is_null() {
        return 1;
    }

    write_null(out_ptr, out_len);
    write_null(err_ptr, err_len);

    if json_ptr.is_null() {
        let _ = write_allocated_bytes(
            err_ptr,
            err_len,
            b"null json_ptr passed to neo_fold_run_wasm_demo_workflow_json".to_vec(),
        );
        return 2;
    }

    let json_bytes = unsafe { core::slice::from_raw_parts(json_ptr, json_len) };
    let json = match core::str::from_utf8(json_bytes) {
        Ok(s) => s,
        Err(e) => {
            let _ = write_allocated_bytes(err_ptr, err_len, format!("input is not valid UTF-8: {e}").into_bytes());
            return 3;
        }
    };

    let total_start = Instant::now();

    let create_start = Instant::now();
    let mut session = match TestExportSession::new_from_circuit_json(json) {
        Ok(s) => s,
        Err(e) => {
            let _ = write_allocated_bytes(err_ptr, err_len, format!("session init error: {e}").into_bytes());
            return 4;
        }
    };
    let _create_ms = elapsed_ms(create_start);

    let setup = session.setup_timings_ms().clone();
    let params = session.params_summary();
    let circuit = session.circuit_summary();

    let add_start = Instant::now();
    if let Err(e) = session.add_steps_from_test_export_json(json) {
        let _ = write_allocated_bytes(
            err_ptr,
            err_len,
            format!("add_steps_from_test_export_json error: {e}").into_bytes(),
        );
        return 5;
    }
    let add_ms = elapsed_ms(add_start);

    let prove_start = Instant::now();
    let (fold_run, fold_steps) = match session.fold_and_prove_with_step_timings() {
        Ok(v) => v,
        Err(e) => {
            let _ = write_allocated_bytes(err_ptr, err_len, format!("fold_and_prove error: {e}").into_bytes());
            return 6;
        }
    };
    let prove_ms = elapsed_ms(prove_start);

    let verify_start = Instant::now();
    let verify_ok = match session.verify(&fold_run) {
        Ok(ok) => ok,
        Err(e) => {
            let _ = write_allocated_bytes(err_ptr, err_len, format!("verify error: {e}").into_bytes());
            return 7;
        }
    };
    let verify_ms = elapsed_ms(verify_start);

    let proof_estimate = estimate_proof(&fold_run);
    let folding = folding_summary(&fold_run);

    let total_ms = elapsed_ms(total_start);

    let mut raw_obj = serde_json::Map::new();
    raw_obj.insert("steps".to_string(), serde_json::json!(fold_run.steps.len()));
    raw_obj.insert("verify_ok".to_string(), serde_json::json!(verify_ok));
    raw_obj.insert("circuit".to_string(), serde_json::json!(circuit));
    raw_obj.insert("params".to_string(), serde_json::json!(params));
    raw_obj.insert(
        "timings_ms".to_string(),
        serde_json::json!({
            "ajtai_setup": setup.ajtai_setup,
            "build_ccs": setup.build_ccs,
            "session_init": setup.session_init,
            "add_steps_total": add_ms,
            "fold_and_prove": prove_ms,
            "fold_steps": fold_steps,
            "verify": verify_ms,
            "total": total_ms,
        }),
    );
    raw_obj.insert("proof_estimate".to_string(), serde_json::json!(proof_estimate));
    raw_obj.insert("folding".to_string(), serde_json::json!(folding));

    if do_spartan != 0 {
        #[cfg(not(feature = "spartan"))]
        {
            let _ = write_allocated_bytes(
                err_ptr,
                err_len,
                b"spartan requested but neo-fold-ffi was built without --features spartan".to_vec(),
            );
            return 8;
        }

        #[cfg(feature = "spartan")]
        {
            let acc_init = session
                .initial_accumulator()
                .map(|acc| acc.me.as_slice())
                .unwrap_or(&[]);

            let witness = fold_run_witness_placeholder(&fold_run);

            let sp_prove_start = Instant::now();
            let spartan =
                match neo_spartan_bridge::prove_fold_run(session.params(), session.ccs(), acc_init, &fold_run, witness)
                {
                    Ok(p) => p,
                    Err(e) => {
                        let _ =
                            write_allocated_bytes(err_ptr, err_len, format!("spartan prove error: {e}").into_bytes());
                        return 9;
                    }
                };
            let sp_prove_ms = elapsed_ms(sp_prove_start);

            let snark_bytes = match spartan.snark_bytes_len() {
                Ok(n) => n,
                Err(e) => {
                    let _ = write_allocated_bytes(
                        err_ptr,
                        err_len,
                        format!("spartan snark_bytes_len error: {e}").into_bytes(),
                    );
                    return 10;
                }
            };
            let vk_bytes = match spartan.vk_bytes_len() {
                Ok(n) => n,
                Err(e) => {
                    let _ = write_allocated_bytes(
                        err_ptr,
                        err_len,
                        format!("spartan vk_bytes_len error: {e}").into_bytes(),
                    );
                    return 11;
                }
            };
            let vk_and_snark_bytes = spartan.proof_data.len();

            let sp_verify_start = Instant::now();
            let sp_verify_ok = match neo_spartan_bridge::verify_fold_run(session.params(), session.ccs(), &spartan) {
                Ok(ok) => ok,
                Err(e) => {
                    let _ = write_allocated_bytes(err_ptr, err_len, format!("spartan verify error: {e}").into_bytes());
                    return 12;
                }
            };
            let sp_verify_ms = elapsed_ms(sp_verify_start);

            raw_obj.insert(
                "spartan".to_string(),
                serde_json::json!({
                    "prove_ms": sp_prove_ms,
                    "verify_ms": sp_verify_ms,
                    "verify_ok": sp_verify_ok,
                    "snark_bytes": snark_bytes,
                    "vk_bytes": vk_bytes,
                    "vk_and_snark_bytes": vk_and_snark_bytes,
                }),
            );
        }
    }

    let out_json = match serde_json::to_vec(&serde_json::Value::Object(raw_obj)) {
        Ok(v) => v,
        Err(e) => {
            let _ = write_allocated_bytes(err_ptr, err_len, format!("serialize error: {e}").into_bytes());
            return 13;
        }
    };

    write_allocated_bytes(out_ptr, out_len, out_json)
}

/// Free a buffer previously allocated by this crate.
#[no_mangle]
pub extern "C" fn neo_fold_free_bytes(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }

    // SAFETY:
    // - The buffer was allocated by Rust (via `Box<[u8]>`) and leaked with `Box::into_raw`.
    // - `len` must be exactly the original length.
    unsafe {
        let slice = core::ptr::slice_from_raw_parts_mut(ptr, len);
        drop(Box::from_raw(slice));
    }
}
