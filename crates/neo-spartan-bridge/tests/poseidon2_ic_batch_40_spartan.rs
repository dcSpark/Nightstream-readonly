#![allow(non_snake_case)]

//! C0 sizing benchmark for Spartan2 (Hash-MLE PCS) on a larger Poseidon2 IC circuit.
//!
//! This is meant to be a *repeatable* measurement for C0 in
//! `docs/proof-compression-plan.md`: Hash‑MLE openings scale with
//! `m = log2(num_vars_padded)`, and the proof grows beyond the `< 50KB` budget.
//!
//! **How to run**
//! - `cargo test -p neo-spartan-bridge --release --test poseidon2_ic_batch_40_spartan -- --ignored --nocapture`
//!
//! **Profiling**
//! - `./scripts/profile_for_ai.sh neo-spartan-bridge poseidon2_ic_batch_40_spartan test_poseidon2_ic_batch_40_spartan_proof_size --ignored`
//!
//! **Notes**
//! - `poseidon2_ic_circuit_batch_40.json` currently contains `witness.len() == 1`. To force multiple
//!   fold steps (which increases `m` and makes the size blow-up more obvious), this test repeats the
//!   same witness for `target_folding_steps`.
//! - First run may include a `neo-fold` cache miss; run twice for steadier timings.
//!
//! **Example output (sizes are stable; timings vary)**
//! - `Spartan SNARK bytes: 848558 (828.67 KB)`
//! - `Spartan proof breakdown: vk=14159309 (13.50 MB) snark=848558 (828.67 KB)`
//! - `Spartan shape: num_cons=131072 num_vars_padded=262144 m=log2(num_vars_padded)=18 public=12 challenges=0`

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

fn print_summary_table(title: &str, rows: &[(&str, String)]) {
    let header_left = "Metric";
    let header_right = "Value";

    let mut left_width = header_left.len();
    let mut right_width = header_right.len();
    for (k, v) in rows {
        left_width = left_width.max(k.len());
        right_width = right_width.max(v.len());
    }

    let hline = format!("+-{}-+-{}-+", "-".repeat(left_width), "-".repeat(right_width));

    println!();
    println!("{title}");
    println!("{hline}");
    println!(
        "| {:<left_width$} | {:<right_width$} |",
        header_left,
        header_right,
        left_width = left_width,
        right_width = right_width
    );
    println!("{hline}");
    for (k, v) in rows {
        println!(
            "| {:<left_width$} | {:<right_width$} |",
            k,
            v,
            left_width = left_width,
            right_width = right_width
        );
    }
    println!("{hline}");
    println!();
}

#[test]
#[ignore = "C0 sizing benchmark; run manually with --ignored --nocapture"]
fn test_poseidon2_ic_batch_40_spartan_proof_size() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join("../neo-fold/poseidon2-tests/poseidon2_ic_circuit_batch_40.json");
    let json = fs::read_to_string(&json_path).expect("read poseidon2 batch-40 json");
    let export = neo_fold::test_export::parse_test_export_json(&json).expect("parse test-export json");

    let target_folding_steps: usize = 2;

    assert!(
        !export.witness.is_empty(),
        "expected poseidon2 export to include at least one witness step"
    );
    println!("Export witness steps: {}", export.witness.len());
    println!(
        "Target folding steps: {}{}",
        target_folding_steps,
        if export.witness.len() < target_folding_steps {
            " (repeating first witness)"
        } else {
            ""
        }
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
    let circuit_summary = session.circuit_summary();

    let n_pow2 = circuit_summary.r1cs_padded_n.next_power_of_two();
    let m_log2 = n_pow2.trailing_zeros();

    println!("Session ready ({})", fmt_ms(create_ms));
    println!(
        "Params: b={} d={} kappa={} k_rho={} T={} s={} lambda={}",
        params.b, params.d, params.kappa, params.k_rho, params.T, params.s, params.lambda
    );
    println!(
        "Circuit (R1CS): constraints={} variables={} padded_n={} next_pow2_n={} m=log2(next_pow2_n)={}",
        circuit_summary.r1cs_constraints, circuit_summary.r1cs_variables, circuit_summary.r1cs_padded_n, n_pow2, m_log2
    );
    println!(
        "Circuit (R1CS): A_nnz={} B_nnz={} C_nnz={}",
        circuit_summary.r1cs_a_nnz, circuit_summary.r1cs_b_nnz, circuit_summary.r1cs_c_nnz
    );
    println!(
        "Witness: steps={} fields_total={} fields_min={} fields_max={} nonzero={} ({:.2}%)",
        circuit_summary.witness_steps,
        circuit_summary.witness_fields_total,
        circuit_summary.witness_fields_min,
        circuit_summary.witness_fields_max,
        circuit_summary.witness_nonzero_fields_total,
        circuit_summary.witness_nonzero_ratio * 100.0
    );
    println!(
        "Circuit (CCS): n={} m={} t={} max_degree={} poly_terms={} nnz_total={}",
        circuit_summary.ccs_n,
        circuit_summary.ccs_m,
        circuit_summary.ccs_t,
        circuit_summary.ccs_max_degree,
        circuit_summary.ccs_poly_terms,
        circuit_summary.ccs_matrix_nnz_total
    );
    println!("CCS matrices nnz: {:?}", circuit_summary.ccs_matrix_nnz);
    println!(
        "Timings: ajtai={} build_ccs={} session_init={}",
        fmt_ms(setup.ajtai_setup),
        fmt_ms(setup.build_ccs),
        fmt_ms(setup.session_init)
    );

    println!("Adding witness steps…");
    let add_start = Instant::now();
    for i in 0..target_folding_steps {
        let z = &export.witness[i % export.witness.len()];
        session.add_step_witness_u64(z).expect("add witness step");
    }
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

    let (pk_recomputed, vk_recomputed) = SNARK::setup(circuit).expect("recompute (pk, vk)");

    let sizes = pk_recomputed.sizes();
    let num_vars_padded = sizes[5] + sizes[6] + sizes[7];
    let m_vars = num_vars_padded.trailing_zeros();
    println!(
        "Spartan shape: num_cons={} num_vars_padded={} m=log2(num_vars_padded)={} public={} challenges={}",
        sizes[4], num_vars_padded, m_vars, sizes[8], sizes[9]
    );
    println!(
        "C0_SUMMARY: batch=40 target_steps={} padded_n={} spartan_m={} snark_bytes={} vk_bytes={} total_bytes={}",
        target_folding_steps,
        circuit_summary.r1cs_padded_n,
        m_vars,
        proof.snark_data.len(),
        vk_bytes,
        vk_bytes + proof.snark_data.len()
    );

    let vk_ser = bincode::serialize(&keypair.vk).expect("serialize proof vk");
    let vk_recomputed_ser = bincode::serialize(&vk_recomputed).expect("serialize recomputed vk");

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

    let total_prove_ms = prove_ms + spartan_prove_ms;
    let total_verify_ms = verify_ms + spartan_verify_ms;
    let fold_step_count = fold_run.steps.len();
    let fold_steps_str = if fold_step_ms.is_empty() {
        "<none>".to_string()
    } else {
        fmt_ms_list(&fold_step_ms, 256)
    };

    print_summary_table(
        "RUN SUMMARY",
        &[
            ("Proving time (fold)", fmt_ms(prove_ms)),
            ("Proving time (spartan)", fmt_ms(spartan_prove_ms)),
            ("Total proving time", fmt_ms(total_prove_ms)),
            ("Verification time (fold)", fmt_ms(verify_ms)),
            ("Verification time (spartan)", fmt_ms(spartan_verify_ms)),
            ("Total verification time", fmt_ms(total_verify_ms)),
            (
                "Proof size (Spartan SNARK)",
                format!("{} ({})", proof.snark_data.len(), fmt_bytes(proof.snark_data.len())),
            ),
            ("Fold steps", fold_step_count.to_string()),
            ("Fold step times", fold_steps_str),
        ],
    );
}
