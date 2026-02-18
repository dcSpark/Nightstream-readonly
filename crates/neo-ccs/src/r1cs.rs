use p3_field::Field;

use crate::{
    matrix::Mat,
    poly::{SparsePoly, Term},
    relations::CcsStructure,
};

/// Minimal **R1CS → CCS** helper: given A, B, C ∈ F^{n×m}, produce CCS with
/// M_0=A, M_1=B, M_2=C and f(X0,X1,X2) = X0·X1 − X2 (elementwise).
///
/// This is the standard embedding: row-wise, `A z ∘ B z = C z`, i.e., `f=0`.
pub fn r1cs_to_ccs<F: Field>(a: Mat<F>, b: Mat<F>, c: Mat<F>) -> CcsStructure<F> {
    assert_eq!(a.rows(), b.rows());
    assert_eq!(a.rows(), c.rows());
    assert_eq!(a.cols(), b.cols());
    assert_eq!(a.cols(), c.cols());

    // Base polynomial f(X0,X1,X2) = X0 * X1 - X2
    let base_terms = vec![
        Term {
            coeff: F::ONE,
            exps: vec![1, 1, 0],
        }, // X1 * X2
        Term {
            coeff: -F::ONE,
            exps: vec![0, 0, 1],
        }, // -X3
    ];
    let f_base = SparsePoly::new(3, base_terms);

    CcsStructure::new(vec![a, b, c], f_base).expect("valid R1CS→CCS structure")
}
