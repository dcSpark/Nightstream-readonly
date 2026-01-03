#![allow(non_snake_case)]

use neo_ajtai::{set_global_pp, setup as ajtai_setup, AjtaiSModule};
use neo_ccs::{CcsStructure, Mat, SparsePoly, Term};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{FoldingSession, ProveInput};
use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    set_global_pp(pp).expect("set_global_pp");
}

fn r1cs_f_base() -> SparsePoly<F> {
    SparsePoly::new(
        3,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![1, 1, 0],
            }, // X1 * X2
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 1],
            }, // -X3
        ],
    )
}

#[test]
fn test_regression_optimized_all_public_inputs() {
    // Regression: optimized mode should handle m_in == m (empty witness).
    let n_constraints = 3usize;
    let n_vars = 3usize;

    let mut A = Mat::zero(n_constraints, n_vars, F::ZERO);
    let mut B = Mat::zero(n_constraints, n_vars, F::ZERO);
    let mut C = Mat::zero(n_constraints, n_vars, F::ZERO);

    // Row 0: (x0 + x1) * x2 = x0
    A[(0, 0)] = F::ONE;
    A[(0, 1)] = F::ONE;
    B[(0, 2)] = F::ONE;
    C[(0, 0)] = F::ONE;

    // Row 1: x1 * x1 = x1
    A[(1, 1)] = F::ONE;
    B[(1, 1)] = F::ONE;
    C[(1, 1)] = F::ONE;

    // Row 2: x2 * x2 = x2
    A[(2, 2)] = F::ONE;
    B[(2, 2)] = F::ONE;
    C[(2, 2)] = F::ONE;

    let ccs = neo_ccs::r1cs_to_ccs(A, B, C);

    let params =
        NeoParams::goldilocks_auto_r1cs_ccs(n_constraints).expect("goldilocks_auto_r1cs_ccs should find valid params");

    setup_ajtai_for_dims(n_vars);
    let l = AjtaiSModule::from_global_for_dims(D, n_vars).expect("AjtaiSModule init");

    // Valid witness: z = [x0, x1, x2] = [1, 0, 1]
    let public_input = vec![F::ONE, F::ZERO, F::ONE];
    let witness: Vec<F> = vec![];

    let mut session = FoldingSession::new(FoldingMode::Optimized, params, l.clone());
    let input = ProveInput {
        ccs: &ccs,
        public_input: &public_input,
        witness: &witness,
        output_claims: &[],
    };

    session
        .add_step_from_io(&input)
        .expect("add_step should succeed");
    let run = session.fold_and_prove(&ccs).expect("fold_and_prove");

    let public_mcss = session.mcss_public();
    let ok = session.verify(&ccs, &public_mcss, &run).expect("verify");
    assert!(ok, "verification should pass with empty witness");
}

#[test]
fn test_regression_optimized_normalizes_identity_first() {
    // Regression: session should accept a square CCS that is not identity-first
    // by calling `ensure_identity_first()` internally.
    let n_constraints = 3usize;
    let n_vars = 3usize;

    let mut A = Mat::zero(n_constraints, n_vars, F::ZERO);
    let mut B = Mat::zero(n_constraints, n_vars, F::ZERO);
    let mut C = Mat::zero(n_constraints, n_vars, F::ZERO);

    // Row 0: x0 * x1 = w0  (w0 is at index 2)
    A[(0, 0)] = F::ONE;
    B[(0, 1)] = F::ONE;
    C[(0, 2)] = F::ONE;

    // Row 1: w0 * w0 = w0
    A[(1, 2)] = F::ONE;
    B[(1, 2)] = F::ONE;
    C[(1, 2)] = F::ONE;

    // Row 2: x0 * x0 = x0
    A[(2, 0)] = F::ONE;
    B[(2, 0)] = F::ONE;
    C[(2, 0)] = F::ONE;

    // Intentionally build a 3-matrix CCS even though it's square, so M0 is not identity.
    let ccs = CcsStructure::new(vec![A, B, C], r1cs_f_base()).expect("valid R1CS→CCS structure");

    let params =
        NeoParams::goldilocks_auto_r1cs_ccs(n_constraints).expect("goldilocks_auto_r1cs_ccs should find valid params");

    setup_ajtai_for_dims(n_vars);
    let l = AjtaiSModule::from_global_for_dims(D, n_vars).expect("AjtaiSModule init");

    let public_input = vec![F::ONE, F::ONE];
    let witness = vec![F::ONE];

    let mut session = FoldingSession::new(FoldingMode::Optimized, params, l.clone());
    let input = ProveInput {
        ccs: &ccs,
        public_input: &public_input,
        witness: &witness,
        output_claims: &[],
    };

    session
        .add_step_from_io(&input)
        .expect("add_step should succeed");
    let run = session.fold_and_prove(&ccs).expect("fold_and_prove");

    let public_mcss = session.mcss_public();
    let ok = session.verify(&ccs, &public_mcss, &run).expect("verify");
    assert!(ok, "verification should pass after identity-first normalization");
}
