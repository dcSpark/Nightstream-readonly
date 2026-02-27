#![allow(non_snake_case)]

use neo_ajtai::{set_global_pp, setup as ajtai_setup, AjtaiSModule};
use neo_ccs::{r1cs_to_ccs, Mat};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{me_from_z_balanced, Accumulator, FoldingSession};
use neo_fold::shard::StepLinkingConfig;
use neo_math::{D, F, K};
use neo_memory::ajtai::commit_cols_for_ccs_m;
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;

fn setup_ajtai_for_dims(m: usize) {
    let m_commit = commit_cols_for_ccs_m(m);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let pp = ajtai_setup(&mut rng, D, 4, m_commit).expect("Ajtai setup should succeed");
    set_global_pp(pp).expect("set_global_pp");
}

#[test]
fn test_session_multifold_k3_three_steps_r1cs_optimized() {
    let n = 4usize;
    let A = Mat::identity(n);
    let B = Mat::identity(n);
    let C = Mat::identity(n);
    let ccs = r1cs_to_ccs(A, B, C);

    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("goldilocks_auto_r1cs_ccs should find valid params");

    setup_ajtai_for_dims(n);
    let l = AjtaiSModule::from_global_for_dims(D, commit_cols_for_ccs_m(n)).expect("AjtaiSModule init");

    let dims = neo_reductions::engines::utils::build_dims_and_policy(&params, &ccs).expect("dims");
    let ell_n = dims.ell_n;

    let r: Vec<K> = vec![K::from(F::from_u64(3)); ell_n];

    let m_in = 2;
    let z_seed_1: Vec<F> = vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO];
    let z_seed_2: Vec<F> = vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO];

    let (me1, Z1) = me_from_z_balanced(&params, &ccs, &l, &z_seed_1, &r, m_in).expect("seed1 ME ok");
    let (me2, Z2) = me_from_z_balanced(&params, &ccs, &l, &z_seed_2, &r, m_in).expect("seed2 ME ok");

    // k=3: 2 seed MEs in accumulator + 1 new MCS per step
    let acc = Accumulator {
        me: vec![me1, me2],
        witnesses: vec![Z1, Z2],
    };

    let mut session = FoldingSession::new(FoldingMode::Optimized, params, l.clone())
        .with_initial_accumulator(acc, &ccs)
        .expect("with_initial_accumulator");

    {
        let x: Vec<F> = vec![F::ZERO, F::ZERO];
        let w: Vec<F> = vec![F::ZERO, F::ZERO];
        session.add_step_io(&ccs, &x, &w).expect("add_step 1");
    }

    {
        let x: Vec<F> = vec![F::ZERO, F::ZERO];
        let w: Vec<F> = vec![F::ZERO, F::ZERO];
        session.add_step_io(&ccs, &x, &w).expect("add_step 2");
    }

    {
        let x: Vec<F> = vec![F::ZERO, F::ZERO];
        let w: Vec<F> = vec![F::ZERO, F::ZERO];
        session.add_step_io(&ccs, &x, &w).expect("add_step 3");
    }

    let run = session
        .fold_and_prove(&ccs)
        .expect("fold_and_prove should produce a FoldRun");

    // CCS-only main flow auto-batches consecutive steps.
    assert_eq!(
        run.steps.len(),
        1,
        "ccs-only auto batching should collapse three steps into one fold step"
    );
    for (i, step) in run.steps.iter().enumerate() {
        assert_eq!(
            step.fold.dec_children.len(),
            params.k_rho as usize,
            "step {} should have k_rho={} DEC children",
            i,
            params.k_rho
        );
    }
    // Final outputs are the dec_children of the last step
    let final_outputs = &run.steps.last().unwrap().fold.dec_children;
    assert_eq!(
        final_outputs.len(),
        params.k_rho as usize,
        "final outputs should have k_rho={}",
        params.k_rho
    );

    let mcss_public = session.mcss_public();
    session.set_step_linking(StepLinkingConfig::new(vec![(0, 0)]));
    let ok = session
        .verify(&ccs, &mcss_public, &run)
        .expect("verify should run");
    assert!(ok, "optimized verification should pass");
}

#[test]
fn test_session_rejects_out_of_range_packed_step_witness() {
    let n = D; // SuperNeo-packed compatible width
    let A = Mat::identity(n);
    let B = Mat::identity(n);
    let C = Mat::identity(n);
    let ccs = r1cs_to_ccs(A, B, C);

    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("goldilocks_auto_r1cs_ccs should find valid params");

    setup_ajtai_for_dims(n);
    let l = AjtaiSModule::from_global_for_dims(D, commit_cols_for_ccs_m(n)).expect("AjtaiSModule init");

    let mut session = FoldingSession::new(FoldingMode::Optimized, params, l);

    // For b=2 and D=54, values above ~2^54 are not representable in D balanced digits.
    let x: Vec<F> = vec![F::from_u64(1u64 << 60), F::from_u64((1u64 << 60) + 1)];
    let w: Vec<F> = (0..(n - x.len()))
        .map(|i| F::from_u64((1u64 << 60) + 2 + i as u64))
        .collect();

    let err = session
        .add_step_io(&ccs, &x, &w)
        .expect_err("out-of-range packed witness must be rejected");
    assert!(err.to_string().contains("not representable"), "unexpected error: {err}");
}
