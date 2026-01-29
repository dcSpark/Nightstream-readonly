use blake2b_simd::State as TranscriptHash;
use midnight_curves::Bls12;
use midnight_proofs::dev::cost_model::circuit_model;
use midnight_proofs::poly::kzg::params::ParamsKZG;
use neo_midnight_bridge::bundle_verifier::{
    derive_step_fs_values, verify_bundle_digest_v2, verify_step_fs_values, verify_sumcheck_challenges_from_rounds,
    StepBundleStatementV2,
};
use neo_midnight_bridge::fs::FsChannel;
use neo_midnight_bridge::goldilocks::{host_sub_mod, GOLDILOCKS_P_U64};
use neo_midnight_bridge::k_field::{host_k_add, host_k_eval_horner, KRepr, K_DELTA_U64};
use neo_midnight_bridge::relations::{PiCcsSumcheckPublicRoundsInstance, PiCcsSumcheckPublicRoundsRelation};
use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn reduce_to_gl(u: u64) -> u64 {
    if u >= GOLDILOCKS_P_U64 {
        u - GOLDILOCKS_P_U64
    } else {
        u
    }
}

fn rand_krepr(rng: &mut impl RngCore) -> KRepr {
    KRepr {
        c0: reduce_to_gl(rng.next_u64()),
        c1: reduce_to_gl(rng.next_u64()),
    }
}

fn k_double(a: KRepr) -> KRepr {
    host_k_add(a, a)
}

fn k_sub(a: KRepr, b: KRepr) -> KRepr {
    KRepr {
        c0: host_sub_mod(a.c0, b.c0),
        c1: host_sub_mod(a.c1, b.c1),
    }
}

fn generate_fs_bound_sumcheck(
    bundle_digest32: [u8; 32],
    channel: FsChannel,
    n_rounds: usize,
    poly_len: usize,
    initial_sum: KRepr,
    rng: &mut impl RngCore,
) -> (Vec<Vec<KRepr>>, Vec<KRepr>, KRepr) {
    assert!(n_rounds > 0);
    assert_eq!(
        poly_len, 2,
        "this helper currently only supports poly_len=2 (linear polys)"
    );

    let mut rounds: Vec<Vec<KRepr>> = Vec::with_capacity(n_rounds);
    let mut challenges: Vec<KRepr> = Vec::with_capacity(n_rounds);

    let mut running_sum = initial_sum;
    for round_idx in 0..n_rounds {
        // Choose random a, then solve b s.t. p(0)+p(1)=running_sum for p(x)=a + b x.
        //
        // p(0)=a, p(1)=a+b => p(0)+p(1)=2a+b.
        let a = rand_krepr(rng);
        let b = k_sub(running_sum, k_double(a));
        let coeffs = vec![a, b];

        let ch = neo_midnight_bridge::fs::derive_sumcheck_round_challenge(
            bundle_digest32,
            channel,
            round_idx as u32,
            &coeffs,
        );
        challenges.push(ch);

        running_sum = host_k_eval_horner(&coeffs, ch, K_DELTA_U64);
        rounds.push(coeffs);
    }

    (rounds, challenges, running_sum)
}

#[test]
fn bundle_verifier_recomputes_fs_from_public_rounds() {
    // Small dimensions matching the "split" layout: n_rounds = ell_* + ell_d.
    let ell_d: usize = 2;
    let ell_n: usize = 1;
    let ell_m: usize = 1;
    let n_rounds_fe = ell_n + ell_d;
    let n_rounds_nc = ell_m + ell_d;
    let poly_len = 2;

    // Arbitrary statement digests (in production these come from params/ccs/accumulators).
    let mut rng = ChaCha20Rng::from_seed([9u8; 32]);
    let mut params_digest32 = [0u8; 32];
    let mut ccs_digest32 = [0u8; 32];
    let mut initial_acc_digest32 = [0u8; 32];
    let mut final_acc_digest32 = [0u8; 32];
    rng.fill_bytes(&mut params_digest32);
    rng.fill_bytes(&mut ccs_digest32);
    rng.fill_bytes(&mut initial_acc_digest32);
    rng.fill_bytes(&mut final_acc_digest32);

    let statement = StepBundleStatementV2 {
        step_idx: 7,
        params_digest32,
        ccs_digest32,
        initial_acc_digest32,
        final_acc_digest32,
    };
    let bundle_digest32 =
        verify_bundle_digest_v2(&statement, statement.bundle_digest_u128_limbs_le()).expect("bundle digest self-check");

    // Generate self-consistent sumcheck transcripts where each challenge is FS-derived from
    // the round polynomial coefficients (linear, poly_len=2).
    let initial_sum_fe = rand_krepr(&mut rng);
    let (fe_rounds, fe_chals, final_sum_fe) = generate_fs_bound_sumcheck(
        bundle_digest32,
        FsChannel::Fe,
        n_rounds_fe,
        poly_len,
        initial_sum_fe,
        &mut rng,
    );
    let initial_sum_nc = rand_krepr(&mut rng);
    let (nc_rounds, nc_chals, final_sum_nc) = generate_fs_bound_sumcheck(
        bundle_digest32,
        FsChannel::Nc,
        n_rounds_nc,
        poly_len,
        initial_sum_nc,
        &mut rng,
    );

    // Prove the FS anchor relation (sumcheck + public rounds) for FE and NC.
    let rel = PiCcsSumcheckPublicRoundsRelation {
        n_rounds: n_rounds_fe,
        poly_len,
    };
    let circuit = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel);
    let model = circuit_model::<_, 48, 32>(&circuit);
    let k: u32 = model.k;
    let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(k, ChaCha20Rng::from_seed([10u8; 32]));
    let params_v = params.verifier_params();

    let fe_inst = PiCcsSumcheckPublicRoundsInstance {
        bundle_digest: statement.bundle_digest_u128_limbs_le(),
        initial_sum: initial_sum_fe,
        final_sum: final_sum_fe,
        challenges: fe_chals.clone(),
        rounds: fe_rounds.clone(),
    };
    let vk = midnight_zk_stdlib::setup_vk(&params, &rel);
    let pk = midnight_zk_stdlib::setup_pk(&rel, &vk);
    let fe_proof = midnight_zk_stdlib::prove::<_, TranscriptHash>(
        &params,
        &pk,
        &rel,
        &fe_inst,
        (),
        ChaCha20Rng::from_seed([11u8; 32]),
    )
    .expect("prove fe fs-anchor");
    midnight_zk_stdlib::verify::<PiCcsSumcheckPublicRoundsRelation, TranscriptHash>(
        &params_v, &vk, &fe_inst, None, &fe_proof,
    )
    .expect("verify fe fs-anchor");

    let nc_inst = PiCcsSumcheckPublicRoundsInstance {
        bundle_digest: statement.bundle_digest_u128_limbs_le(),
        initial_sum: initial_sum_nc,
        final_sum: final_sum_nc,
        challenges: nc_chals.clone(),
        rounds: nc_rounds.clone(),
    };
    let nc_proof = midnight_zk_stdlib::prove::<_, TranscriptHash>(
        &params,
        &pk,
        &rel,
        &nc_inst,
        (),
        ChaCha20Rng::from_seed([12u8; 32]),
    )
    .expect("prove nc fs-anchor");
    midnight_zk_stdlib::verify::<PiCcsSumcheckPublicRoundsRelation, TranscriptHash>(
        &params_v, &vk, &nc_inst, None, &nc_proof,
    )
    .expect("verify nc fs-anchor");

    // Verifier-side: recompute challenges from public rounds and compare.
    verify_sumcheck_challenges_from_rounds(bundle_digest32, FsChannel::Fe, &fe_rounds, &fe_chals)
        .expect("fe fs replay");
    verify_sumcheck_challenges_from_rounds(bundle_digest32, FsChannel::Nc, &nc_rounds, &nc_chals)
        .expect("nc fs replay");

    // Derive the remaining transcript-scoped values (gamma, alpha, beta_*) and check equality.
    let derived = derive_step_fs_values(bundle_digest32, &fe_rounds, &nc_rounds, ell_d, ell_n, ell_m);
    verify_step_fs_values(
        &derived,
        derived.gamma,
        &derived.alpha,
        &derived.beta_a,
        &derived.beta_r,
        &derived.beta_m,
    )
    .expect("derived values self-check");

    // Negative checks: a mismatch is detected.
    let mut bad_fe_chals = fe_chals.clone();
    bad_fe_chals[0] = rand_krepr(&mut rng);
    assert!(
        verify_sumcheck_challenges_from_rounds(bundle_digest32, FsChannel::Fe, &fe_rounds, &bad_fe_chals).is_err(),
        "expected mismatch to be caught"
    );

    let bad_gamma = rand_krepr(&mut rng);
    assert!(
        verify_step_fs_values(
            &derived,
            bad_gamma,
            &derived.alpha,
            &derived.beta_a,
            &derived.beta_r,
            &derived.beta_m
        )
        .is_err(),
        "expected gamma mismatch to be caught"
    );
}
