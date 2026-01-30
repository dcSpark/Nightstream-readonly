use blake2b_simd::State as TranscriptHash;
use midnight_curves::Bls12;
use midnight_proofs::poly::kzg::params::ParamsKZG;
use neo_midnight_bridge::goldilocks::{host_mul_quotient_and_remainder, GOLDILOCKS_P_U64};
use neo_midnight_bridge::relations::{GoldilocksMulInstance, GoldilocksMulRelation};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[test]
fn plonk_kzg_goldilocks_mul_mod_roundtrip() {
    // Build relation + instance.
    let rel = GoldilocksMulRelation;

    // Generate KZG params (test-only). Production should use Midnight's SRS.
    let k: u32 = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel).min_k();
    let rng = ChaCha20Rng::from_seed([42u8; 32]);
    let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(k, rng);

    // Sample x,y < p.
    let x = 123456789u64 % GOLDILOCKS_P_U64;
    let y = 987654321u64 % GOLDILOCKS_P_U64;
    let (_kq, r) = host_mul_quotient_and_remainder(x, y);

    let instance = GoldilocksMulInstance { x, y, z: r };

    // Keygen + prove.
    let vk = midnight_zk_stdlib::setup_vk(&params, &rel);
    let pk = midnight_zk_stdlib::setup_pk(&rel, &vk);
    let proof = midnight_zk_stdlib::prove::<_, TranscriptHash>(
        &params,
        &pk,
        &rel,
        &instance,
        (),
        ChaCha20Rng::from_seed([43u8; 32]),
    )
    .expect("prove");

    let params_v = params.verifier_params();
    midnight_zk_stdlib::verify::<GoldilocksMulRelation, TranscriptHash>(&params_v, &vk, &instance, None, &proof)
        .expect("verify");
}
