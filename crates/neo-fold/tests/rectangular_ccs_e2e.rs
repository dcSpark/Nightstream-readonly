#![allow(non_snake_case)]

use neo_ajtai::AjtaiSModule;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{CcsBuilder, FoldingSession};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_rectangular_ccs_single_step_prove_verify() {
    // One R1CS constraint: a * b = c, with an extra unused witness column to ensure m != n.
    let mut cs = CcsBuilder::<F>::new(1, 0).expect("CcsBuilder::new");
    cs.r1cs_terms([(1, F::ONE)], [(2, F::ONE)], [(3, F::ONE)]);

    // m = 5 variables: [1, a, b, c, dummy], n = 1 constraint row.
    let ccs = cs.build_rect(5, 0).expect("build_rect");

    let public_input = vec![F::ONE];
    let witness = vec![F::from_u64(2), F::from_u64(3), F::from_u64(6), F::ZERO];

    let mut session = FoldingSession::<AjtaiSModule>::new_ajtai_seeded(FoldingMode::Optimized, &ccs, [7u8; 32])
        .expect("new_ajtai_seeded");
    session
        .add_step_io(&ccs, &public_input, &witness)
        .expect("add_step_io");

    let run = session
        .prove_and_verify_collected(&ccs)
        .expect("prove_and_verify_collected");

    let dims = neo_reductions::engines::utils::build_dims_and_policy(session.params(), &ccs).expect("build_dims");
    assert_ne!(dims.ell_m, dims.ell_n, "test must exercise n != m");

    assert_eq!(run.steps.len(), 1);
    let step0 = &run.steps[0];

    assert_eq!(
        step0.fold.ccs_proof.variant,
        neo_fold::optimized_engine::PiCcsProofVariant::SplitNcV1
    );
    assert_eq!(step0.fold.ccs_proof.sumcheck_rounds.len(), dims.ell);
    assert_eq!(step0.fold.ccs_proof.sumcheck_rounds_nc.len(), dims.ell_nc);

    assert_eq!(step0.fold.ccs_out[0].s_col.len(), dims.ell_m);
    assert_eq!(step0.fold.ccs_out[0].y_zcol.len(), 1usize << dims.ell_d);
}

