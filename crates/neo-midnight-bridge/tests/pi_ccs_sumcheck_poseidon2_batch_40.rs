use blake2b_simd::State as TranscriptHash;
use midnight_curves::Bls12;
use midnight_proofs::dev::cost_model::circuit_model;
use midnight_proofs::poly::kzg::params::ParamsKZG;
use neo_math::KExtensions;
use neo_midnight_bridge::k_field::{host_sumcheck_round_claim, KRepr};
use neo_midnight_bridge::relations::{PiCcsSumcheckInstance, PiCcsSumcheckRelation, PiCcsSumcheckWitness};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::fs;
use std::path::PathBuf;

fn k_to_repr(k: &neo_math::K) -> KRepr {
    let (c0, c1) = k.to_limbs_u64();
    KRepr { c0, c1 }
}

#[test]
fn plonk_kzg_pi_ccs_sumcheck_poseidon2_batch_40_roundtrip() {
    // Build a real Pi-CCS proof from the Poseidon2 batch_40 export, then
    // prove its sumcheck algebra inside Midnight's PLONK/KZG verifier stack.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join("../neo-fold/poseidon2-tests/poseidon2_ic_circuit_batch_40.json");
    let json = fs::read_to_string(&json_path).expect("read poseidon2 batch-40 json");
    let export = neo_fold::test_export::parse_test_export_json(&json).expect("parse test-export json");

    // The export currently contains a single witness vector. We repeat it to force multi-step
    // folding (so we can see the `m` growth on the Spartan side, and exercise more of the
    // sumcheck logic on the Pi-CCS side).
    let target_folding_steps: usize = 2;

    let mut session = neo_fold::test_export::TestExportSession::new_from_circuit_json(&json).expect("session init");
    for i in 0..target_folding_steps {
        let z = &export.witness[i % export.witness.len()];
        session.add_step_witness_u64(z).expect("add witness step");
    }

    let (fold_run, _step_ms) = session
        .fold_and_prove_with_step_timings()
        .expect("fold_and_prove");
    assert_eq!(fold_run.steps.len(), target_folding_steps);
    assert!(session.verify(&fold_run).expect("verify"));

    // Take the first step's Pi-CCS proof.
    let pi = &fold_run.steps[0].fold.ccs_proof;

    let n_rounds = pi.sumcheck_rounds.len();
    assert!(n_rounds > 0, "expected at least one sumcheck round");
    let poly_len = pi.sumcheck_rounds[0].len();
    assert!(poly_len > 0, "expected non-empty round polynomial");
    println!("Pi-CCS sumcheck shape: rounds={n_rounds} poly_len={poly_len}");
    assert!(
        pi.sumcheck_rounds.iter().all(|r| r.len() == poly_len),
        "expected uniform polynomial length across rounds"
    );
    assert_eq!(
        pi.sumcheck_challenges.len(),
        n_rounds,
        "expected one challenge per round"
    );

    // Public instance = (initial_sum, final_sum).
    let rounds_repr: Vec<Vec<KRepr>> = pi
        .sumcheck_rounds
        .iter()
        .map(|round| round.iter().map(|k| k_to_repr(k)).collect())
        .collect();
    let initial_sum = pi
        .sc_initial_sum
        .as_ref()
        .map(k_to_repr)
        .unwrap_or_else(|| host_sumcheck_round_claim(&rounds_repr[0]));
    let final_sum = k_to_repr(&pi.sumcheck_final);
    let challenges: Vec<KRepr> = pi
        .sumcheck_challenges
        .iter()
        .map(|k| k_to_repr(k))
        .collect();
    let instance = PiCcsSumcheckInstance {
        bundle_digest: [0u128; 2],
        initial_sum,
        final_sum,
        challenges,
    };
    let witness = PiCcsSumcheckWitness { rounds: rounds_repr };

    let rel = PiCcsSumcheckRelation { n_rounds, poly_len };

    // Use test-only KZG params. Production must use Midnight's SRS.
    let circuit = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel);
    println!("MidnightCircuit: {circuit:?}");
    let model = circuit_model::<_, 48, 32>(&circuit);
    let k: u32 = model.k;
    println!("Midnight min_k() for PiCcsSumcheckRelation: {k}");
    println!(
        "CircuitModel: rows={} table_rows={} unusable_rows={} max_deg={} advice_cols={} fixed_cols={} lookups={}",
        model.rows,
        model.table_rows,
        model.nb_unusable_rows,
        model.max_deg,
        model.advice_columns,
        model.fixed_columns,
        model.lookups
    );
    let rng = ChaCha20Rng::from_seed([21u8; 32]);
    let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(k, rng);

    let vk = midnight_zk_stdlib::setup_vk(&params, &rel);
    let pk = midnight_zk_stdlib::setup_pk(&rel, &vk);
    let proof = midnight_zk_stdlib::prove::<_, TranscriptHash>(
        &params,
        &pk,
        &rel,
        &instance,
        witness,
        ChaCha20Rng::from_seed([22u8; 32]),
    )
    .expect("prove");

    println!("Midnight proof bytes: {}", proof.len());
    assert!(proof.len() < 50 * 1024, "expected < 50KB proof");

    let params_v = params.verifier_params();
    midnight_zk_stdlib::verify::<PiCcsSumcheckRelation, TranscriptHash>(&params_v, &vk, &instance, None, &proof)
        .expect("verify");
}
