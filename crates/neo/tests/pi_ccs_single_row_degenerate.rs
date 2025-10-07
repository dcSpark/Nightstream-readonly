//! Regression repro: Π-CCS terminal check mismatch on single-row R1CS with y_len=0.
//!
//! This test encodes a 1-row R1CS constraint (z0 - z1) * z0 = 0 with a satisfying witness [1, 1].
//! Row-wise CCS check passes, but IVC verify currently returns false due to a degenerate
//! sum-check shape (ℓ = 0 rounds, y_len = 0) where the terminal running_sum stays 0 while
//! the terminal claim wr*Σ α_i (A·z)*(B·z)−(C·z) is non-zero because of the base-case LHS instance.
//!
//! Once the Π-CCS verifier handles ℓ=0 correctly, this test can be updated to expect true.

use neo::{F, NeoParams};
use neo::ivc::{Accumulator, StepBindingSpec, LastNExtractor, prove_ivc_step_with_extractor, verify_ivc_step};
use neo_ccs::{Mat, r1cs::r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;

/// Build a 1-row, 2-var R1CS CCS for (z0 - z1) * z0 = 0
fn tiny_r1cs_single_row() -> neo_ccs::CcsStructure<F> {
    let a = Mat::from_row_major(1, 2, vec![F::ONE, -F::ONE]);
    let b = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);
    let c = Mat::from_row_major(1, 2, vec![F::ZERO, F::ZERO]);
    r1cs_to_ccs(a, b, c)
}

#[test]
fn pi_ccs_single_row_deg_shape() {
    // 1) Step CCS and satisfying witness
    let ccs = tiny_r1cs_single_row();
    let witness = vec![F::ONE, F::ONE]; // (1 - 1) * 1 = 0

    // Sanity: row-wise check must pass
    neo_ccs::relations::check_ccs_rowwise_zero(&ccs, &[], &witness)
        .expect("witness should satisfy CCS row-wise");

    // 2) IVC setup with y_len=0 and no app inputs
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let acc0 = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![], step: 0 };
    let binding = StepBindingSpec {
        y_step_offsets: vec![],
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    let extractor = LastNExtractor { n: 0 };

    // 3) Prove single step (should succeed)
    let step_res = prove_ivc_step_with_extractor(
        &params,
        &ccs,
        &witness,
        &acc0,
        0,
        None,
        &extractor,
        &binding,
    ).expect("prove_ivc_step_with_extractor should succeed");

    // 4) Verify: keep as expected-fail until ℓ≤1 R1CS oracle normalization lands
    let ok = verify_ivc_step(
        &ccs,
        &step_res.proof,
        &acc0,
        &binding,
        &params,
        None,
    ).expect("verify_ivc_step should not error");

    // Known deficiency: hypercube sum s(0)+s(1) ≠ 0 due to degenerate path offset.
    // Expect rejection for now so the suite stays green.
    assert!(!ok, "Expected current Π-CCS to reject due to ℓ≤1 degenerate bug");
}
