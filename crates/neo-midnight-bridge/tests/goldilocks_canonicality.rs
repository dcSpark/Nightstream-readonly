mod common;

use blake2b_simd::State as TranscriptHash;
use common::goldilocks_canonicality_relations::{
    gl_modulus_u64, GlAddAmbiguousCarryInstance, GlAddAmbiguousCarryRelation, GlAddAmbiguousCarryWitness,
    GlAllocPublicInstance, GlAllocPublicRelation,
};
use midnight_curves::Bls12;
use midnight_proofs::poly::kzg::params::ParamsKZG;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[test]
fn plonk_kzg_alloc_gl_public_rejects_noncanonical() {
    let rel = GlAllocPublicRelation;

    let k: u32 = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel).min_k();
    let rng = ChaCha20Rng::from_seed([10u8; 32]);
    let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(k, rng);

    let vk = midnight_zk_stdlib::setup_vk(&params, &rel);
    let pk = midnight_zk_stdlib::setup_pk(&rel, &vk);

    // x == p is a non-canonical Goldilocks representative; verification must fail.
    let instance = GlAllocPublicInstance { x: gl_modulus_u64() };
    let proof_res = midnight_zk_stdlib::prove::<_, TranscriptHash>(
        &params,
        &pk,
        &rel,
        &instance,
        (),
        ChaCha20Rng::from_seed([11u8; 32]),
    );
    match proof_res {
        Err(_) => {}
        Ok(proof) => {
            let params_v = params.verifier_params();
            assert!(
                midnight_zk_stdlib::verify::<GlAllocPublicRelation, TranscriptHash>(
                    &params_v, &vk, &instance, None, &proof
                )
                .is_err(),
                "expected verification to fail for x == p"
            );
        }
    }
}

#[test]
fn plonk_kzg_gl_add_accepts_canonical_output() {
    let rel = GlAddAmbiguousCarryRelation;

    let k: u32 = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel).min_k();
    let rng = ChaCha20Rng::from_seed([20u8; 32]);
    let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(k, rng);

    let vk = midnight_zk_stdlib::setup_vk(&params, &rel);
    let pk = midnight_zk_stdlib::setup_pk(&rel, &vk);

    // 1 + (p-1) == 0 (mod p), canonical.
    let instance = GlAddAmbiguousCarryInstance {
        x: 1,
        y: gl_modulus_u64() - 1,
        z: 0,
    };
    let witness = GlAddAmbiguousCarryWitness {
        x_val: instance.x,
        y_val: instance.y,
    };

    let proof = midnight_zk_stdlib::prove::<_, TranscriptHash>(
        &params,
        &pk,
        &rel,
        &instance,
        witness,
        ChaCha20Rng::from_seed([21u8; 32]),
    )
    .expect("prove");

    let params_v = params.verifier_params();
    midnight_zk_stdlib::verify::<GlAddAmbiguousCarryRelation, TranscriptHash>(&params_v, &vk, &instance, None, &proof)
        .expect("verify");
}

#[test]
fn plonk_kzg_gl_add_rejects_noncanonical_output() {
    let rel = GlAddAmbiguousCarryRelation;

    let k: u32 = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel).min_k();
    let rng = ChaCha20Rng::from_seed([30u8; 32]);
    let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(k, rng);

    let vk = midnight_zk_stdlib::setup_vk(&params, &rel);
    let pk = midnight_zk_stdlib::setup_pk(&rel, &vk);

    // Non-canonical output z == p should be impossible after in-circuit canonicality checks.
    let instance = GlAddAmbiguousCarryInstance {
        x: 1,
        y: gl_modulus_u64() - 1,
        z: gl_modulus_u64(),
    };
    // Force `carry = 0` even though x+y == p by setting unconstrained metadata to (0,0).
    let witness = GlAddAmbiguousCarryWitness { x_val: 0, y_val: 0 };

    let proof_res = midnight_zk_stdlib::prove::<_, TranscriptHash>(
        &params,
        &pk,
        &rel,
        &instance,
        witness,
        ChaCha20Rng::from_seed([31u8; 32]),
    );
    match proof_res {
        Err(_) => {}
        Ok(proof) => {
            let params_v = params.verifier_params();
            assert!(
                midnight_zk_stdlib::verify::<GlAddAmbiguousCarryRelation, TranscriptHash>(
                    &params_v, &vk, &instance, None, &proof
                )
                .is_err(),
                "expected verification to fail for z == p"
            );
        }
    }
}
