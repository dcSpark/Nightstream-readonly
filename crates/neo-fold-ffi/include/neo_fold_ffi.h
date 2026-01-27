#ifndef NEO_FOLD_FFI_H
#define NEO_FOLD_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Runs prove+verify for a TestExport JSON (UTF-8 bytes).
//
// On success returns 0 and sets (*out_ptr, *out_len) to an allocated UTF-8 JSON buffer
// containing the TestExportResult.
//
// On failure returns non-zero and sets (*err_ptr, *err_len) to an allocated UTF-8 error message.
//
// Buffers must be freed with neo_fold_free_bytes.
//
// Return codes:
// - 0: success (out_* set)
// - 1: invalid output pointers (out_* or err_* were NULL)
// - 2: json_ptr was NULL
// - 3: input was not valid UTF-8
// - 4: JSON parse error
// - 5: neo-fold run error
// - 6: JSON serialize error
int32_t neo_fold_prove_verify_test_export_json(
    const uint8_t* json_ptr,
    size_t json_len,
    uint8_t** out_ptr,
    size_t* out_len,
    uint8_t** err_ptr,
    size_t* err_len);

// Runs a workflow equivalent to `demos/wasm-demo/web/prover_worker.js` and returns a JSON
// object shaped like its "Raw result".
//
// This is intended for side-by-side comparisons between wasm and native (iOS).
//
// If do_spartan != 0, also runs Spartan2 compression (requires building neo-fold-ffi with
// `--features spartan`).
int32_t neo_fold_run_wasm_demo_workflow_json(
    const uint8_t* json_ptr,
    size_t json_len,
    int32_t do_spartan,
    uint8_t** out_ptr,
    size_t* out_len,
    uint8_t** err_ptr,
    size_t* err_len);

void neo_fold_free_bytes(uint8_t* ptr, size_t len);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NEO_FOLD_FFI_H
