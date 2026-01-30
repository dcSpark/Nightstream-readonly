mod common;

#[test]
#[ignore = "benchmark-style full step-1 bundle; run with --ignored --nocapture"]
fn plonk_kzg_pi_ccs_step1_full_bundle_poseidon2_batch_40_roundtrip() {
    common::prove_step1_full_bundle_poseidon2_batch_40();
}
