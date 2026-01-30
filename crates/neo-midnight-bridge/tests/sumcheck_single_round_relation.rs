use blake2b_simd::State as TranscriptHash;
use midnight_curves::Bls12;
use midnight_proofs::poly::kzg::params::ParamsKZG;
use neo_midnight_bridge::goldilocks::GOLDILOCKS_P_U64;
use neo_midnight_bridge::k_field::{host_k_add, host_k_eval_horner, host_sumcheck_round_claim, KRepr, K_DELTA_U64};
use neo_midnight_bridge::relations::{SumcheckSingleRoundRelation, SumcheckSingleRoundWitness};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn rand_gl(rng: &mut impl RngCore) -> u64 {
    let x = rng.next_u64();
    x % GOLDILOCKS_P_U64
}

fn rand_k(rng: &mut impl RngCore) -> KRepr {
    KRepr {
        c0: rand_gl(rng),
        c1: rand_gl(rng),
    }
}

#[test]
fn plonk_kzg_sumcheck_single_round_roundtrip() {
    let rel = SumcheckSingleRoundRelation { n_coeffs: 4 };

    let k: u32 = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel).min_k();
    let rng = ChaCha20Rng::from_seed([10u8; 32]);
    let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(k, rng);

    let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
    let coeffs: Vec<KRepr> = (0..rel.n_coeffs).map(|_| rand_k(&mut rng)).collect();
    let challenge = rand_k(&mut rng);
    let claimed_sum = host_sumcheck_round_claim(&coeffs);
    let next_sum = host_k_eval_horner(&coeffs, challenge, K_DELTA_U64);

    let witness = SumcheckSingleRoundWitness {
        coeffs,
        challenge,
        claimed_sum,
        next_sum,
    };

    let vk = midnight_zk_stdlib::setup_vk(&params, &rel);
    let pk = midnight_zk_stdlib::setup_pk(&rel, &vk);
    let proof = midnight_zk_stdlib::prove::<_, TranscriptHash>(
        &params,
        &pk,
        &rel,
        &(),
        witness.clone(),
        ChaCha20Rng::from_seed([12u8; 32]),
    )
    .expect("prove");

    let params_v = params.verifier_params();
    midnight_zk_stdlib::verify::<SumcheckSingleRoundRelation, TranscriptHash>(&params_v, &vk, &(), None, &proof)
        .expect("verify");

    // A wrong `next_sum` must not verify.
    let mut bad_witness = witness;
    bad_witness.next_sum = host_k_add(bad_witness.next_sum, KRepr { c0: 1, c1: 0 });

    let bad_proof = midnight_zk_stdlib::prove::<_, TranscriptHash>(
        &params,
        &pk,
        &rel,
        &(),
        bad_witness,
        ChaCha20Rng::from_seed([13u8; 32]),
    );

    match bad_proof {
        Err(_) => {}
        Ok(proof) => {
            let res = midnight_zk_stdlib::verify::<SumcheckSingleRoundRelation, TranscriptHash>(
                &params_v,
                &vk,
                &(),
                None,
                &proof,
            );
            assert!(res.is_err(), "bad proof unexpectedly verified");
        }
    }
}
