use neo_math::{Rq, Fq, D, SAction, cf, cf_inv};
use p3_field::PrimeCharacteristicRing;

#[test]
fn rot_matches_ring_multiplication_on_vectors() {
    // random-looking but deterministic vectors
    let mut v = [Fq::ZERO; D];
    for i in 0..D { v[i] = Fq::from_u64((i as u64).wrapping_mul(7919)); }
    let mut a_coeffs = [Fq::ZERO; D];
    for i in 0..D { a_coeffs[i] = Fq::from_u64((i as u64).wrapping_mul(104729)); }
    let a = Rq(a_coeffs);

    let rot = SAction::from_ring(a);
    let lhs = rot.apply_vec(&v);

    let rhs = cf(a.mul(&cf_inv(v)));
    assert_eq!(lhs, rhs);
}

#[test]
fn s_action_identity() {
    let id = SAction::from_ring(Rq::one());
    let v = [Fq::ONE; D];
    assert_eq!(id.apply_vec(&v), v);
}
