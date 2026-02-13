use neo_fold::session::CcsBuilder;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn ccs_builder_square_does_not_insert_identity_matrix() {
    let mut cs = CcsBuilder::<F>::new(1, 0).expect("CcsBuilder::new");
    cs.r1cs_terms([(0, F::from_u64(2))], [(0, F::ONE)], [(0, F::ONE)]);

    // n == m == 1 (square), but builder should keep 3-matrix embedding.
    let ccs = cs.build_rect(1, 0).expect("build_rect");
    assert_eq!(ccs.t(), 3, "square build_rect must not auto-insert identity matrix");
    assert!(!ccs.matrices[0].is_identity(), "M0 should be A, not identity");
}
