#![allow(non_snake_case)]

use neo_spartan_bridge::circuit::FoldRunWitness;
use neo_spartan_bridge::circuit::fold_circuit::CircuitPolyTerm;
use neo_spartan_bridge::circuit::FoldRunCircuit;
use neo_spartan_bridge::{prove_fold_run, verify_fold_run};
use neo_spartan_bridge::CircuitF;
use p3_field::PrimeField64;
use spartan2::traits::snark::R1CSSNARKTrait;
use std::fs;
use std::path::PathBuf;

fn fmt_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let kb = bytes as f64 / 1024.0;
    if kb < 1024.0 {
        return format!("{kb:.2} KB");
    }
    let mb = kb / 1024.0;
    format!("{mb:.2} MB")
}

#[test]
fn test_poseidon2_ic_batch_1_spartan_proof_size() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join("../neo-fold/poseidon2-tests/poseidon2_ic_circuit_batch_1.json");
    let json = fs::read_to_string(&json_path).expect("read poseidon2 batch-1 json");
    let export = neo_fold::test_export::parse_test_export_json(&json).expect("parse test-export json");

    assert_eq!(
        export.witness.len(),
        1,
        "expected batch-1 export to have exactly 1 witness step"
    );

    let mut session =
        neo_fold::test_export::TestExportSession::new_from_export(&export).expect("session init");
    session
        .add_steps_from_test_export_json(&json)
        .expect("add witness steps");

    let (fold_run, _fold_step_ms) = session
        .fold_and_prove_with_step_timings()
        .expect("fold_and_prove");

    let ok = session.verify(&fold_run).expect("verify folding proof");
    assert!(ok, "folding proof should verify");

    let acc_init = session
        .initial_accumulator()
        .map(|acc| acc.me.clone())
        .unwrap_or_default();

    let pi_ccs_proofs = fold_run
        .steps
        .iter()
        .map(|s| s.fold.ccs_proof.clone())
        .collect();
    let rlc_rhos = fold_run.steps.iter().map(|s| s.fold.rlc_rhos.clone()).collect();
    let per_step_empty = (0..fold_run.steps.len()).map(|_| Vec::new()).collect::<Vec<_>>();
    let witness = FoldRunWitness::from_fold_run(
        fold_run.clone(),
        pi_ccs_proofs,
        per_step_empty.clone(),
        rlc_rhos,
        per_step_empty,
    );

    let witness_for_setup = witness.clone();

    let proof = prove_fold_run(
        session.params(),
        session.ccs(),
        acc_init.as_slice(),
        &fold_run,
        witness,
    )
    .expect("spartan prove");

    println!(
        "Spartan proof bytes: {} ({})",
        proof.proof_data.len(),
        fmt_bytes(proof.proof_data.len())
    );

    type E = spartan2::provider::GoldilocksP3MerkleMleEngine;
    type SNARK = spartan2::spartan::R1CSSNARK<E>;
    type VK = spartan2::spartan::SpartanVerifierKey<E>;

    let (vk, snark): (VK, SNARK) =
        bincode::deserialize(&proof.proof_data).expect("deserialize (vk, snark)");
    let vk_bytes = bincode::serialize(&vk).expect("serialize vk").len();
    let snark_bytes = bincode::serialize(&snark).expect("serialize snark").len();

    println!(
        "Spartan proof breakdown: vk={} ({}) snark={} ({})",
        vk_bytes,
        fmt_bytes(vk_bytes),
        snark_bytes,
        fmt_bytes(snark_bytes)
    );
    assert_eq!(
        vk_bytes + snark_bytes,
        proof.proof_data.len(),
        "expected bincode tuple to equal vk+snark sizes"
    );

    // Recompute vk deterministically from the circuit definition and compare.
    let poly_f: Vec<CircuitPolyTerm> = session
        .ccs()
        .f
        .terms()
        .iter()
        .map(|term| CircuitPolyTerm {
            coeff: CircuitF::from(term.coeff.as_canonical_u64()),
            coeff_native: term.coeff,
            exps: term.exps.iter().map(|e| *e as u32).collect(),
        })
        .collect();

    let delta = CircuitF::from(7u64);
    let circuit = FoldRunCircuit::new(
        proof.instance.clone(),
        Some(witness_for_setup),
        delta,
        session.params().b,
        poly_f,
    );

    let (_pk_recomputed, vk_recomputed) = SNARK::setup(circuit).expect("recompute (pk, vk)");

    let vk_ser = bincode::serialize(&vk).expect("serialize proof vk");
    let vk_recomputed_ser = bincode::serialize(&vk_recomputed).expect("serialize recomputed vk");

    assert_eq!(
        vk_recomputed_ser,
        vk_ser,
        "recomputed vk should match the vk included in proof bytes"
    );

    let ok = verify_fold_run(session.params(), session.ccs(), &proof).expect("spartan verify");
    assert!(ok, "spartan proof should verify");
}
