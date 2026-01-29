mod common;

#[test]
#[ignore = "benchmark-style multi-proof bundle; run with --ignored --nocapture"]
fn plonk_kzg_pi_ccs_terminal_nc_chunked_poseidon2_batch_40_roundtrip() {
    common::prove_step1_nc_bundle_poseidon2_batch_40();
}
