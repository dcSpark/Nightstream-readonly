//! Red-team API/binding tests that exercise neo::{prove, verify}.
//!
//! Targets: 
//! - #26 (wrong CCS/public_input) - Verify with wrong `(CCS, public_input)` binding check
//! - #29 (DoS length guard) - Unbounded bincode deserialization (DoS) size guard in `neo::verify`

#![cfg(feature = "redteam")]

use neo::{prove, verify, ProveInput, CcsStructure, NeoParams, F};
use p3_field::PrimeCharacteristicRing;
use neo_ccs::{Mat, SparsePoly, Term};

fn dummy_ccs_structure() -> CcsStructure<F> {
    // Single 4x3 zero matrix and a trivial arity-1 polynomial.
    // With zero witness, the row-wise CCS check is satisfied.
    let matrices = vec![Mat::zero(4, 3, F::ZERO)];
    let terms = vec![Term { coeff: F::ONE, exps: vec![1] }];
    let f = SparsePoly::new(1, terms);
    CcsStructure::new(matrices, f).expect("valid CCS")
}

#[test]
fn rt26_verify_wrong_public_input_must_fail() {
    // Arrange: small, satisfiable CCS + zero witness
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let ccs = dummy_ccs_structure();
    let witness = vec![F::ZERO; ccs.m]; // length matches m
    let public_input: Vec<F> = vec![];  // empty is fine here

    // Prove for (ccs, public_input)
    let proof = prove(ProveInput {
        params: &params, ccs: &ccs, public_input: &public_input, witness: &witness
    }).expect("prove ok");

    // Tamper: change the public_input at verification time
    let bad_public_input = vec![F::from_u64(1)];
    let ok = verify(&ccs, &bad_public_input, &proof).expect("verify runs");
    assert!(!ok, "verification must return false when public_input differs");
}

#[test]
fn rt26_verify_wrong_ccs_must_fail() {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let mut ccs = dummy_ccs_structure();
    let witness = vec![F::ZERO; ccs.m];
    let public_input: Vec<F> = vec![];

    let proof = prove(ProveInput {
        params: &params, ccs: &ccs, public_input: &public_input, witness: &witness
    }).expect("prove ok");

    // Tamper CCS deterministically: flip one entry in the first matrix
    ccs.matrices[0][(0, 0)] = F::from_u64(1);

    let ok = verify(&ccs, &public_input, &proof).expect("verify runs");
    assert!(!ok, "verification must return false when CCS differs");
}

#[test]
fn rt29_proof_bundle_too_large_rejected_fast() {
    // Construct a dummy Proof with an oversized size header (no need for a real proof)
    const MAX: usize = 64 * 1024 * 1024; // must match neo::verify guard
    let too_big = (MAX as u32) + 1;

    let proof = neo::Proof {
        v: 1,
        public_io: Vec::new(),        // ok: we fail before checking digest presence
        bundle: too_big.to_le_bytes() // first 4 bytes = declared uncompressed length
            .to_vec(),
    };

    let res = verify(&dummy_ccs_structure(), &[], &proof);
    assert!(res.is_err(), "oversized bundle must error before decompression");
    let msg = format!("{:?}", res.err().unwrap());
    assert!(msg.contains("proof bundle too large"), "guard should trigger; got: {msg}");
}
