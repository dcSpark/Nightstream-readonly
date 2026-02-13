use neo_ccs::{r1cs_to_ccs, Mat};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn square_r1cs_uses_three_matrix_embedding() {
    let n = 4usize;
    let m = 4usize;

    let mut a = Mat::zero(n, m, F::ZERO);
    let mut b = Mat::zero(n, m, F::ZERO);
    let mut c = Mat::zero(n, m, F::ZERO);
    a[(0, 0)] = F::ONE;
    b[(0, 1)] = F::ONE;
    c[(0, 2)] = F::ONE;

    let ccs = r1cs_to_ccs(a, b, c);
    assert_eq!(ccs.t(), 3, "square R1CS must not auto-insert identity matrix");
    assert!(!ccs.matrices[0].is_identity(), "M0 should be A, not identity");
}

#[test]
fn rectangular_r1cs_uses_three_matrix_embedding() {
    let n = 2usize;
    let m = 5usize;

    let mut a = Mat::zero(n, m, F::ZERO);
    let mut b = Mat::zero(n, m, F::ZERO);
    let mut c = Mat::zero(n, m, F::ZERO);
    a[(0, 0)] = F::ONE;
    b[(0, 1)] = F::ONE;
    c[(0, 2)] = F::ONE;

    let ccs = r1cs_to_ccs(a, b, c);
    assert_eq!(ccs.t(), 3);
    assert_eq!(ccs.n, n);
    assert_eq!(ccs.m, m);
}
