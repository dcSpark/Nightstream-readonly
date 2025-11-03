#![allow(non_snake_case)]

use neo_fold::session::{FoldingSession, NeoStep, StepArtifacts, StepSpec, Accumulator, me_from_z_balanced};
use neo_fold::pi_ccs::FoldingMode;
use neo_ccs::{CcsStructure, Mat, SparsePoly, Term};
use neo_ajtai::{setup as ajtai_setup, set_global_pp, AjtaiSModule};
use rand_chacha::rand_core::SeedableRng;
use neo_params::NeoParams;
use neo_math::{F, K, D};
use p3_field::PrimeCharacteristicRing;

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

fn tiny_ccs_id(n: usize, m: usize) -> CcsStructure<F> {
    assert_eq!(n, m, "use square tiny ccs");
    let m0 = Mat::identity(n);
    let f = SparsePoly::new(1, vec![Term { coeff: F::ONE, exps: vec![1] }]);
    CcsStructure::new(vec![m0], f).unwrap()
}


#[derive(Clone)]
struct NoInputs;

struct OneShotStep {
    pub ccs: CcsStructure<F>,
    pub spec: StepSpec,
    pub z: Vec<F>,
}

impl OneShotStep {
    fn new(n: usize) -> Self {
        let ccs = tiny_ccs_id(n, n);
        let spec = StepSpec {
            y_len: 0,
            const1_index: 0,
            y_step_indices: vec![],
            app_input_indices: None,
            m_in: 0,
        };
        let z = vec![F::ONE; n];
        Self { ccs, spec, z }
    }
}

impl NeoStep for OneShotStep {
    type ExternalInputs = NoInputs;

    fn state_len(&self) -> usize { 0 }
    fn step_spec(&self) -> StepSpec { self.spec.clone() }

    fn synthesize_step(
        &mut self,
        _i: usize,
        _y_prev: &[F],
        _inputs: &Self::ExternalInputs,
    ) -> StepArtifacts {
        StepArtifacts {
            ccs: self.ccs.clone(),
            witness: self.z.clone(),
            public_app_inputs: vec![],
            spec: self.spec.clone(),
        }
    }
}

#[test]
fn test_session_single_fold_with_optimized() {
    let n = 2usize; // small square CCS
    
    // Use auto R1CS→CCS params builder with min_lambda=96, safety_margin=2
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n)
        .expect("goldilocks_auto_r1cs_ccs should find valid params");
    setup_ajtai_for_dims(n);
    let l = AjtaiSModule::from_global_for_dims(D, n).unwrap();

    // Build CCS structure
    let ccs = tiny_ccs_id(n, n);

    // Use small witness values so the expanded multi-extension witness Z stays within digit capacity
    let witness_z = vec![F::from_u64(1); n];
    
    // Compute ell_n for r length
    let dims = neo_reductions::optimized_engine::context::build_dims_and_policy(&params, &ccs).unwrap();
    let ell_n = dims.ell_n;
    let r = vec![K::from(F::from_u64(3)); ell_n];
    
    // Build the ME instance from the witness (me_from_z_balanced expands witness_z into Z)
    let (me_inst, Z) = me_from_z_balanced(&params, &ccs, &l, &witness_z, &r, 0)
        .expect("me_from_z_balanced should succeed");
    
    let acc = Accumulator {
        me: vec![me_inst],
        witnesses: vec![Z],
    };

    // Session with Accumulator (k=2: 1 accumulated ME + 1 new MCS)
    let mut session = FoldingSession::new(
        FoldingMode::Optimized,
        params,
        l.clone(),
    )
    .with_initial_accumulator(acc, &ccs)
    .expect("with_initial_accumulator should succeed");

    let mut step = OneShotStep::new(n);
    session
        .prove_step(&mut step, &NoInputs)
        .expect("prove_step should succeed");

    let run = session
        .finalize(&step.ccs)
        .expect("finalize should produce a FoldRun");

    // Verify using the session verifier shim
    let public_mcss = session.mcss_public();
    let ok = session
        .verify(&step.ccs, &public_mcss, &run)
        .expect("verify should run");
    assert!(ok, "session verification should pass");
}

