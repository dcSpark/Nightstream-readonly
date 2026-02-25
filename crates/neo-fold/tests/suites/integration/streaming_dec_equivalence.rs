#![allow(non_snake_case)]

use neo_ajtai::{
    set_global_pp, set_global_pp_seeded, setup_par, try_get_loaded_global_pp_for_dims, AjtaiSModule, Commitment as Cmt,
};
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{CcsClaim, CcsStructure, CcsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_prove_with_witnesses, fold_shard_verify, CommitMixers};
use neo_math::{D, F, K};
use neo_memory::ajtai::{commit_cols_for_ccs_m, encode_vector_for_ccs_m};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn create_identity_ccs(n: usize) -> CcsStructure<F> {
    let mat = Mat::identity(n);
    let f = SparsePoly::new(1, vec![]);
    CcsStructure::new(vec![mat], f).expect("CCS")
}

fn mixers() -> CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt> {
    crate::common_setup::default_mixers()
}

fn build_single_step_bundle_with_salt(
    params: &NeoParams,
    l: &AjtaiSModule,
    m: usize,
    salt: u64,
) -> StepWitnessBundle<Cmt, F, K> {
    let m_in = 0usize;
    // SuperNeo packed NC enforces bounded balanced decomposition; use small synthetic coefficients.
    let z: Vec<F> = (0..m)
        .map(|i| match ((i as u64).wrapping_add(salt)) % 3 {
            0 => -F::ONE,
            1 => F::ZERO,
            _ => F::ONE,
        })
        .collect();
    let Z = encode_vector_for_ccs_m(params, m, &z).expect("encode witness for CCS width");
    let c = l.commit(&Z);
    let mcs_inst = CcsClaim { c, x: vec![], m_in };
    let mcs_wit = CcsWitness { w: z, Z };
    StepWitnessBundle::from((mcs_inst, mcs_wit))
}

fn build_single_step_bundle(params: &NeoParams, l: &AjtaiSModule, m: usize) -> StepWitnessBundle<Cmt, F, K> {
    build_single_step_bundle_with_salt(params, l, m, 0)
}

fn build_single_step_bundle_small_coeffs(
    params: &NeoParams,
    l: &AjtaiSModule,
    m: usize,
    salt: u64,
) -> StepWitnessBundle<Cmt, F, K> {
    let m_in = 0usize;
    let z: Vec<F> = (0..m)
        .map(|i| match ((i as u64).wrapping_add(salt)) % 3 {
            0 => -F::ONE,
            1 => F::ZERO,
            _ => F::ONE,
        })
        .collect();
    let Z = encode_vector_for_ccs_m(params, m, &z).expect("encode witness for CCS width");
    let c = l.commit(&Z);
    let mcs_inst = CcsClaim { c, x: vec![], m_in };
    let mcs_wit = CcsWitness { w: z, Z };
    StepWitnessBundle::from((mcs_inst, mcs_wit))
}

fn assert_step_fold_eq(a: &neo_fold::shard::FoldStep, b: &neo_fold::shard::FoldStep) {
    assert_eq!(a.ccs_out, b.ccs_out, "ccs_out mismatch");
    assert_eq!(a.rlc_rhos, b.rlc_rhos, "rlc_rhos mismatch");
    assert_eq!(a.rlc_parent, b.rlc_parent, "rlc_parent mismatch");
    assert_eq!(a.dec_children, b.dec_children, "dec_children mismatch");
}

#[test]
fn streaming_dec_matches_materialized_dec_with_loaded_pp() {
    let mut selected: Option<(usize, NeoParams, CcsStructure<F>)> = None;
    for n in 32usize..256usize {
        let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
        let ccs = create_identity_ccs(n);
        let m_commit = commit_cols_for_ccs_m(ccs.m);
        if try_get_loaded_global_pp_for_dims(D, m_commit).is_some() {
            selected = Some((n, params, ccs));
            break;
        }

        let mut rng = ChaCha8Rng::seed_from_u64(7 ^ (n as u64));
        let pp = setup_par(&mut rng, D, params.kappa as usize, m_commit).expect("setup_par");
        match set_global_pp(pp) {
            Ok(()) => {
                selected = Some((n, params, ccs));
                break;
            }
            Err(e) if e.to_string().contains("seed is already registered") => {
                // This dimension has a seeded-only registry entry in this test process; try another n.
                continue;
            }
            Err(e) => panic!("set_global_pp: {e}"),
        }
    }
    let (n, mut params, ccs) = selected.expect("failed to find a dimension with loaded global PP");
    params.k_rho = 8; // must satisfy count·T·(b−1) < b^k_rho even for count=1

    let m_commit = commit_cols_for_ccs_m(ccs.m);
    assert!(
        try_get_loaded_global_pp_for_dims(D, m_commit).is_some(),
        "expected loaded PP for n={n}, m_commit={m_commit}",
    );
    let l = AjtaiSModule::from_global_for_dims(D, m_commit).expect("from_global_for_dims");

    let step = build_single_step_bundle(&params, &l, ccs.m);
    let steps_witness = vec![step];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mode = FoldingMode::Optimized;
    let mixers = mixers();

    let mut tr_stream = Poseidon2Transcript::new(b"streaming-dec/loaded");
    let proof_stream = fold_shard_prove(
        mode.clone(),
        &mut tr_stream,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("streaming prove");

    let mut tr_mat = Poseidon2Transcript::new(b"streaming-dec/loaded");
    let (proof_mat, _outputs, _wits) = fold_shard_prove_with_witnesses(
        mode.clone(),
        &mut tr_mat,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("materialized prove");

    assert_eq!(proof_stream.steps.len(), 1);
    assert_eq!(proof_mat.steps.len(), 1);
    assert_step_fold_eq(&proof_stream.steps[0].fold, &proof_mat.steps[0].fold);

    let mut tr_v1 = Poseidon2Transcript::new(b"streaming-dec/loaded");
    let _verify_stream = fold_shard_verify(
        mode.clone(),
        &mut tr_v1,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof_stream,
        mixers,
    )
    .expect("verify streaming proof");

    let mut tr_v2 = Poseidon2Transcript::new(b"streaming-dec/loaded");
    let _verify_mat = fold_shard_verify(
        mode,
        &mut tr_v2,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof_mat,
        mixers,
    )
    .expect("verify materialized proof");
}

#[test]
fn streaming_dec_matches_materialized_dec_with_loaded_pp_superneo_packed() {
    let n = 108usize; // multiple of D=54 => SuperNeo packed embedding
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 8; // must satisfy count·T·(b−1) < b^k_rho even for count=1

    let m_commit = commit_cols_for_ccs_m(ccs.m);
    assert_eq!(
        m_commit * D,
        ccs.m,
        "expected packed commit width for SuperNeo-compatible m"
    );
    if try_get_loaded_global_pp_for_dims(D, m_commit).is_none() {
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        let pp = setup_par(&mut rng, D, params.kappa as usize, m_commit).expect("setup_par");
        if let Err(e) = set_global_pp(pp) {
            let msg = e.to_string();
            if !msg.contains("already loaded") && !msg.contains("already registered") {
                panic!("set_global_pp: {e}");
            }
        }
    }
    let l = AjtaiSModule::from_global_for_dims(D, m_commit).expect("from_global_for_dims");
    // In packed embedding mode the NC relation currently enforces small coefficients directly.
    // Use range-safe witness values here so this test isolates streaming-vs-materialized DEC parity.
    let step = build_single_step_bundle_small_coeffs(&params, &l, ccs.m, 17);
    assert_eq!(step.mcs.1.Z.cols(), m_commit, "packed witness columns mismatch");
    let steps_witness = vec![step];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mode = FoldingMode::Optimized;
    let mixers = mixers();

    let mut tr_stream = Poseidon2Transcript::new(b"streaming-dec/loaded/superneo-packed");
    let proof_stream = fold_shard_prove(
        mode.clone(),
        &mut tr_stream,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("streaming prove");

    let mut tr_mat = Poseidon2Transcript::new(b"streaming-dec/loaded/superneo-packed");
    let (proof_mat, _outputs, _wits) = fold_shard_prove_with_witnesses(
        mode.clone(),
        &mut tr_mat,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("materialized prove");

    assert_eq!(proof_stream.steps.len(), 1);
    assert_eq!(proof_mat.steps.len(), 1);
    assert_step_fold_eq(&proof_stream.steps[0].fold, &proof_mat.steps[0].fold);

    let mut tr_v1 = Poseidon2Transcript::new(b"streaming-dec/loaded/superneo-packed");
    let _verify_stream = fold_shard_verify(
        mode.clone(),
        &mut tr_v1,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof_stream,
        mixers,
    )
    .expect("verify streaming proof");

    let mut tr_v2 = Poseidon2Transcript::new(b"streaming-dec/loaded/superneo-packed");
    let _verify_mat = fold_shard_verify(
        mode,
        &mut tr_v2,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof_mat,
        mixers,
    )
    .expect("verify materialized proof");
}

#[test]
fn streaming_dec_matches_materialized_dec_with_seeded_pp() {
    let mut selected: Option<(usize, NeoParams, CcsStructure<F>, [u8; 32])> = None;
    for n in 17usize..256usize {
        let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
        let ccs = create_identity_ccs(n);
        let m_commit = commit_cols_for_ccs_m(ccs.m);
        if try_get_loaded_global_pp_for_dims(D, m_commit).is_some() {
            continue;
        }
        let mut seed = [7u8; 32];
        seed[0] = (n & 0xff) as u8;
        match set_global_pp_seeded(D, params.kappa as usize, m_commit, seed) {
            Ok(()) => {
                selected = Some((n, params, ccs, seed));
                break;
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("already loaded") || msg.contains("already registered") {
                    continue;
                }
                panic!("set_global_pp_seeded: {e}");
            }
        }
    }
    let (n, mut params, ccs, _seed) = selected.expect("failed to find a dimension for seeded global PP");
    params.k_rho = 8; // must satisfy count·T·(b−1) < b^k_rho even for count=1
    let m_commit = commit_cols_for_ccs_m(ccs.m);
    assert!(
        try_get_loaded_global_pp_for_dims(D, m_commit).is_none(),
        "expected PP to remain unloaded for seeded entry (n={n}, m_commit={m_commit})",
    );
    let l = AjtaiSModule::from_global_for_dims(D, m_commit).expect("from_global_for_dims");

    let step = build_single_step_bundle(&params, &l, ccs.m);
    let steps_witness = vec![step];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mode = FoldingMode::Optimized;
    let mixers = mixers();

    let mut tr_stream = Poseidon2Transcript::new(b"streaming-dec/seeded");
    let proof_stream = fold_shard_prove(
        mode.clone(),
        &mut tr_stream,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("streaming prove");

    let mut tr_mat = Poseidon2Transcript::new(b"streaming-dec/seeded");
    let (proof_mat, _outputs, _wits) = fold_shard_prove_with_witnesses(
        mode.clone(),
        &mut tr_mat,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("materialized prove");

    assert_eq!(proof_stream.steps.len(), 1);
    assert_eq!(proof_mat.steps.len(), 1);
    assert_step_fold_eq(&proof_stream.steps[0].fold, &proof_mat.steps[0].fold);

    let mut tr_v1 = Poseidon2Transcript::new(b"streaming-dec/seeded");
    let _verify_stream = fold_shard_verify(
        mode.clone(),
        &mut tr_v1,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof_stream,
        mixers,
    )
    .expect("verify streaming proof");

    let mut tr_v2 = Poseidon2Transcript::new(b"streaming-dec/seeded");
    let _verify_mat = fold_shard_verify(
        mode,
        &mut tr_v2,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof_mat,
        mixers,
    )
    .expect("verify materialized proof");
}

#[test]
fn streaming_dec_matches_materialized_dec_multi_step_ccs_only() {
    let n = 16usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 10;

    let m_commit = commit_cols_for_ccs_m(ccs.m);
    if try_get_loaded_global_pp_for_dims(D, m_commit).is_none() {
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let pp = setup_par(&mut rng, D, params.kappa as usize, m_commit).expect("setup_par");
        if let Err(e) = set_global_pp(pp) {
            let msg = e.to_string();
            if !msg.contains("already loaded") && !msg.contains("already registered") {
                panic!("set_global_pp: {e}");
            }
        }
    }
    let l = AjtaiSModule::from_global_for_dims(D, m_commit).expect("from_global_for_dims");

    let step0 = build_single_step_bundle_with_salt(&params, &l, ccs.m, 1);
    let step1 = build_single_step_bundle_with_salt(&params, &l, ccs.m, 2);
    let steps_witness = vec![step0, step1];
    let steps_instance: Vec<StepInstanceBundle<Cmt, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mode = FoldingMode::Optimized;
    let mixers = mixers();

    let mut tr_stream = Poseidon2Transcript::new(b"streaming-dec/multi-step");
    let proof_stream = fold_shard_prove(
        mode.clone(),
        &mut tr_stream,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("streaming prove");

    let mut tr_mat = Poseidon2Transcript::new(b"streaming-dec/multi-step");
    let (proof_mat, _outputs, _wits) = fold_shard_prove_with_witnesses(
        mode.clone(),
        &mut tr_mat,
        &params,
        &ccs,
        &steps_witness,
        &[],
        &[],
        &l,
        mixers,
    )
    .expect("materialized prove");

    assert_eq!(proof_stream.steps.len(), proof_mat.steps.len());
    for (s, m) in proof_stream.steps.iter().zip(proof_mat.steps.iter()) {
        assert_step_fold_eq(&s.fold, &m.fold);
    }

    let mut tr_v1 = Poseidon2Transcript::new(b"streaming-dec/multi-step");
    let _verify_stream = fold_shard_verify(
        mode.clone(),
        &mut tr_v1,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof_stream,
        mixers,
    )
    .expect("verify streaming proof");

    let mut tr_v2 = Poseidon2Transcript::new(b"streaming-dec/multi-step");
    let _verify_mat = fold_shard_verify(
        mode,
        &mut tr_v2,
        &params,
        &ccs,
        &steps_instance,
        &[],
        &proof_mat,
        mixers,
    )
    .expect("verify materialized proof");
}
