#![allow(non_snake_case)]

use neo_spartan_bridge::circuit::fold_circuit::CircuitPolyTerm;
use neo_spartan_bridge::circuit::FoldRunCircuit;
use neo_spartan_bridge::circuit::FoldRunWitness;
use neo_spartan_bridge::CircuitF;
use neo_spartan_bridge::{prove_fold_run, setup_fold_run, verify_fold_run};
use p3_field::PrimeField64;
use spartan2::traits::snark::R1CSSNARKTrait;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn fmt_ms(ms: f64) -> String {
    format!("{ms:.1} ms")
}

fn fmt_ms_list(values: &[f64], max_items: usize) -> String {
    let shown = values
        .iter()
        .take(max_items)
        .map(|v| fmt_ms(*v))
        .collect::<Vec<_>>();
    let more = values
        .len()
        .checked_sub(max_items)
        .filter(|&n| n > 0)
        .map(|n| format!(" … (+{n} more)"))
        .unwrap_or_default();
    format!("[{}]{more}", shown.join(", "))
}

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

    println!("Running prove+verify…");
    println!("Input JSON size: {}", fmt_bytes(json.len()));
    println!(
        "Input export: constraints={} variables={} steps={}",
        export.num_constraints,
        export.num_variables,
        export.witness.len()
    );

    let total_start = Instant::now();

    let create_start = Instant::now();
    let mut session = neo_fold::test_export::TestExportSession::new_from_circuit_json(&json).expect("session init");
    let create_ms = create_start.elapsed().as_secs_f64() * 1000.0;

    let setup = session.setup_timings_ms().clone();
    let params = session.params_summary();
    let circuit = session.circuit_summary();

    println!("Session ready ({})", fmt_ms(create_ms));
    println!(
        "Params: b={} d={} kappa={} k_rho={} T={} s={} lambda={}",
        params.b, params.d, params.kappa, params.k_rho, params.T, params.s, params.lambda
    );
    println!(
        "Circuit (R1CS): constraints={} variables={} padded_n={} A_nnz={} B_nnz={} C_nnz={}",
        circuit.r1cs_constraints,
        circuit.r1cs_variables,
        circuit.r1cs_padded_n,
        circuit.r1cs_a_nnz,
        circuit.r1cs_b_nnz,
        circuit.r1cs_c_nnz
    );
    println!(
        "Witness: steps={} fields_total={} fields_min={} fields_max={} nonzero={} ({:.2}%)",
        circuit.witness_steps,
        circuit.witness_fields_total,
        circuit.witness_fields_min,
        circuit.witness_fields_max,
        circuit.witness_nonzero_fields_total,
        circuit.witness_nonzero_ratio * 100.0
    );
    println!(
        "Circuit (CCS): n={} m={} t={} max_degree={} poly_terms={} nnz_total={}",
        circuit.ccs_n,
        circuit.ccs_m,
        circuit.ccs_t,
        circuit.ccs_max_degree,
        circuit.ccs_poly_terms,
        circuit.ccs_matrix_nnz_total
    );
    println!("CCS matrices nnz: {:?}", circuit.ccs_matrix_nnz);
    println!(
        "Timings: ajtai={} build_ccs={} session_init={}",
        fmt_ms(setup.ajtai_setup),
        fmt_ms(setup.build_ccs),
        fmt_ms(setup.session_init)
    );

    println!("Adding witness steps…");
    let add_start = Instant::now();
    session
        .add_steps_from_test_export_json(&json)
        .expect("add witness steps");
    let add_ms = add_start.elapsed().as_secs_f64() * 1000.0;
    println!("Timings: add_steps_total={}", fmt_ms(add_ms));

    println!("Folding + proving…");
    let prove_start = Instant::now();
    let (fold_run, fold_step_ms) = session
        .fold_and_prove_with_step_timings()
        .expect("fold_and_prove");
    let prove_ms = prove_start.elapsed().as_secs_f64() * 1000.0;
    println!("Timings: prove={}", fmt_ms(prove_ms));
    if !fold_step_ms.is_empty() {
        println!("Folding prove per-step: {}", fmt_ms_list(&fold_step_ms, 32));
        let sum: f64 = fold_step_ms.iter().sum();
        let avg = sum / fold_step_ms.len() as f64;
        let min = fold_step_ms.iter().copied().fold(f64::INFINITY, f64::min);
        let max = fold_step_ms.iter().copied().fold(0.0, f64::max);
        println!(
            "Folding prove per-step stats: avg={} min={} max={}",
            fmt_ms(avg),
            fmt_ms(min),
            fmt_ms(max)
        );
    }

    println!("Verifying folding proof…");
    let verify_start = Instant::now();
    let ok = session.verify(&fold_run).expect("verify folding proof");
    let verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
    assert!(ok, "folding proof should verify");
    println!(
        "OK: verify_ok=true steps={} (total {})",
        fold_run.steps.len(),
        fmt_ms(total_start.elapsed().as_secs_f64() * 1000.0)
    );
    println!("Timings: verify={}", fmt_ms(verify_ms));

    let acc_init = session
        .initial_accumulator()
        .map(|acc| acc.me.clone())
        .unwrap_or_default();

    let rlc_rhos = fold_run
        .steps
        .iter()
        .map(|s| s.fold.rlc_rhos.clone())
        .collect();
    let per_step_empty = (0..fold_run.steps.len())
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    let witness = FoldRunWitness::from_fold_run(fold_run.clone(), per_step_empty.clone(), rlc_rhos, per_step_empty);

    let witness_for_setup = witness.clone();

    println!("Compressing with Spartan2…");
    let keypair = setup_fold_run(
        session.params(),
        session.ccs(),
        acc_init.as_slice(),
        &fold_run,
        witness_for_setup.clone(),
    )
    .expect("spartan setup");

    let spartan_prove_start = Instant::now();
    let proof = prove_fold_run(
        &keypair.pk,
        session.params(),
        session.ccs(),
        acc_init.as_slice(),
        &fold_run,
        witness,
    )
    .expect("spartan prove");
    let spartan_prove_ms = spartan_prove_start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "Spartan SNARK bytes: {} ({})",
        proof.snark_data.len(),
        fmt_bytes(proof.snark_data.len())
    );

    type E = spartan2::provider::GoldilocksP3MerkleMleEngine;
    type SNARK = spartan2::spartan::R1CSSNARK<E>;
    let vk_bytes = bincode::serialize(&keypair.vk).expect("serialize vk").len();

    println!(
        "Spartan proof breakdown: vk={} ({}) snark={} ({})",
        vk_bytes,
        fmt_bytes(vk_bytes),
        proof.snark_data.len(),
        fmt_bytes(proof.snark_data.len())
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

    let vk_recomputed_ser = bincode::serialize(&vk_recomputed).expect("serialize recomputed vk");
    let vk_ser = bincode::serialize(&keypair.vk).expect("serialize proof vk");

    assert_eq!(vk_recomputed_ser, vk_ser, "recomputed vk should match the setup vk");

    let spartan_verify_start = Instant::now();
    let ok = verify_fold_run(&keypair.vk, session.params(), session.ccs(), &proof).expect("spartan verify");
    let spartan_verify_ms = spartan_verify_start.elapsed().as_secs_f64() * 1000.0;
    assert!(ok, "spartan proof should verify");

    println!(
        "Spartan2 timings: prove={} verify={}",
        fmt_ms(spartan_prove_ms),
        fmt_ms(spartan_verify_ms)
    );
}
