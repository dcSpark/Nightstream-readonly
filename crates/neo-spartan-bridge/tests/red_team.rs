// crates/neo-spartan-bridge/tests/red_team.rs
use neo_spartan_bridge::hash_mle::{prove_hash_mle, verify_hash_mle, F};
use rand::{Rng, SeedableRng};
use ff::Field;

fn gen_poly_point(m: usize, seed: u64) -> (Vec<F>, Vec<F>) {
    let n = 1usize << m;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let poly = (0..n).map(|_| F::from(rng.random::<u64>())).collect::<Vec<_>>();
    let point = (0..m).map(|_| F::from(rng.random::<u64>())).collect::<Vec<_>>();
    (poly, point)
}

#[test]
fn mle_rejects_eval_tamper() {
    let (poly, point) = gen_poly_point(4, 123);
    let prf = prove_hash_mle(&poly, &point).unwrap();

    // tamper eval
    let mut t = prf.clone();
    t.eval = t.eval + F::ONE;
    assert!(verify_hash_mle(&t).is_err(), "tampered eval must be rejected");
}

#[test]
fn mle_rejects_point_tamper() {
    let (poly, point) = gen_poly_point(3, 999);
    let prf = prove_hash_mle(&poly, &point).unwrap();

    // tamper point
    let mut t = prf.clone();
    t.point[0] = t.point[0] + F::ONE;
    assert!(verify_hash_mle(&t).is_err(), "tampered point must be rejected");
}

#[test]
fn mle_rejects_mix_and_match_commitment() {
    let (poly1, point1) = gen_poly_point(3, 1);
    let (poly2, point2) = gen_poly_point(3, 2);
    let prf1 = prove_hash_mle(&poly1, &point1).unwrap();
    let prf2 = prove_hash_mle(&poly2, &point2).unwrap();

    // mix prf2 payload with prf1 commitment
    let mut forged = prf2.clone();
    forged.commitment = prf1.commitment.clone();
    assert!(verify_hash_mle(&forged).is_err(), "mix-and-match commitment must be rejected");
}
