//! IVC negative test: unsatisfiable step witness should be rejected by verify_ivc_step.
//! This test intentionally does NOT call `check_ccs_rowwise_zero` beforehand.

use neo::{F, NeoParams};
use neo::ivc::{
    Accumulator, LastNExtractor, StepBindingSpec,
    prove_ivc_step_with_extractor, verify_ivc_step,
};
use neo_ccs::{Mat, r1cs::r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;

/// Tiny R1CS-encoded CCS for constraint: (z0 - z1) * z0 = 0
fn tiny_r1cs_to_ccs() -> neo_ccs::CcsStructure<F> {
    // One row, two vars: A = [1, -1], B = [1, 0], C = [0, 0]
    // Row-wise, (A z)[i] * (B z)[i] - (C z)[i] = 0  =>  (z0 - z1) * z0 = 0
    let a = Mat::from_row_major(1, 2, vec![F::ONE, -F::ONE]);
    let b = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);
    let c = Mat::from_row_major(1, 2, vec![F::ZERO, F::ZERO]);
    r1cs_to_ccs(a, b, c)
}

#[test]
fn ivc_unsat_step_witness_should_fail_verify() {
    // Step CCS and params
    let step_ccs = tiny_r1cs_to_ccs();
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);

    // Base accumulator: no prior commitment, no y (y_len = 0)
    let prev_acc = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![], step: 0 };

    // Binding spec: no y, no app-input binding; const-1 witness at index 0
    // We ensure witness[0] == 1 below to satisfy the const-1 convention.
    let binding = StepBindingSpec {
        y_step_offsets: vec![],
        x_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    // UNSAT step witness for (z0 - z1) * z0 = 0: choose z = [1, 5]
    // (1 - 5) * 1 = -4 != 0, so the step CCS is violated.
    // Note: witness[0] == 1 satisfies the const-1 convention used by the augmented CCS.
    let step_witness = vec![F::ONE, F::from_u64(5)];

    // No app public inputs; y_len == 0, so extractor returns empty y_step
    let extractor = LastNExtractor { n: 0 };

    // Prove one IVC step (prover can always produce a transcript; soundness is in verify)
    let step_res = prove_ivc_step_with_extractor(
        &params,
        &step_ccs,
        &step_witness,
        &prev_acc,
        prev_acc.step,
        None,                 // no app public inputs
        &extractor,
        &binding,
    ).expect("IVC step proving should not error");

    // Verify should REJECT because the step witness violates the step CCS
    let ok = verify_ivc_step(
        &step_ccs,
        &step_res.proof,
        &prev_acc,
        &binding,
        &params,
        None, // prev_augmented_x
    ).expect("verify_ivc_step should not error");

    // Expect rejection; if this assertion fails, it demonstrates the bug the user reported.
    assert!(!ok, "IVC verification accepted an unsatisfiable step witness");
}

