#![allow(non_snake_case)]

//! Runner for the `TestExport` JSON schema (R1CS A/B/C + per-step witness).
//!
//! This module is intentionally "circuit-agnostic": the JSON encodes a concrete R1CS instance,
//! so the runner does not synthesize Poseidon2/SHA256/etc. Instead, it:
//! 1) Converts the exported R1CS to CCS
//! 2) Feeds each witness step into a `FoldingSession`
//! 3) Runs fold+prove+verify
//! 4) Returns rich metrics (sizes + timings) for demos/benchmarks

use crate::pi_ccs::FoldingMode;
use crate::session::{FoldingSession, NeoStep, StepArtifacts, StepSpec};
use crate::shard::StepLinkingConfig;
use neo_ajtai::{set_global_pp, setup as ajtai_setup, AjtaiSModule};
use neo_ccs::{r1cs_to_ccs, CcsMatrix, CcsStructure, Mat};
use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use js_sys::Date;

#[derive(Serialize, Deserialize, Clone)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(usize, usize, u64)>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TestExport {
    pub num_constraints: usize,
    pub num_variables: usize,
    pub matrix_a: SparseMatrix,
    pub matrix_b: SparseMatrix,
    pub matrix_c: SparseMatrix,
    // one witness per step
    pub witness: Vec<Vec<u64>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TestExportResult {
    pub steps: usize,
    pub verify_ok: bool,
    pub circuit: TestExportCircuitSummary,
    pub params: TestExportParamsSummary,
    pub timings_ms: TestExportTimingsMs,
    pub proof_estimate: TestExportProofEstimate,
    pub folding: TestExportFoldingSummary,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TestExportParamsSummary {
    pub b: u32,
    pub d: u32,
    pub kappa: u32,
    pub k_rho: u32,
    pub T: u32,
    pub s: u32,
    pub lambda: u32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TestExportCircuitSummary {
    // Input (export) shape.
    pub r1cs_constraints: usize,
    pub r1cs_variables: usize,
    pub r1cs_padded_n: usize,
    pub r1cs_a_nnz: usize,
    pub r1cs_b_nnz: usize,
    pub r1cs_c_nnz: usize,

    // Witness stats (from export).
    pub witness_steps: usize,
    pub witness_fields_total: usize,
    pub witness_fields_min: usize,
    pub witness_fields_max: usize,
    pub witness_nonzero_fields_total: usize,
    pub witness_nonzero_ratio: f64,

    // Derived CCS stats (after embedding).
    pub ccs_n: usize,
    pub ccs_m: usize,
    pub ccs_t: usize,
    pub ccs_max_degree: u32,
    pub ccs_poly_terms: usize,
    pub ccs_matrix_nnz: Vec<usize>,
    pub ccs_matrix_nnz_total: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TestExportTimingsMs {
    pub ajtai_setup: f64,
    pub build_ccs: f64,
    pub prepare_witness: f64,
    pub session_init: f64,

    pub add_steps_total: f64,
    pub add_step_avg: f64,
    pub add_step_min: f64,
    pub add_step_max: f64,

    pub fold_and_prove: f64,
    pub fold_steps: Vec<f64>,
    pub fold_step_avg: f64,
    pub fold_step_min: f64,
    pub fold_step_max: f64,
    pub verify: f64,
    pub total: f64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TestExportFoldingSummary {
    /// Per-step `k_in = acc_len_before + 1` (number of ME(b,L) instances folded that step).
    pub k_in: Vec<usize>,
    /// Per-step accumulator length after folding (`dec_children.len()`).
    pub acc_len_after: Vec<usize>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TestExportProofEstimate {
    pub proof_steps: usize,
    pub final_accumulator_len: usize,

    pub fold_lane_commitments: usize,
    pub mem_cpu_val_claim_commitments: usize,
    pub val_lane_commitments: usize,
    pub total_commitments: usize,

    pub commitment_d: usize,
    pub commitment_kappa: usize,
    pub commitment_bytes: usize,
    pub estimated_commitment_bytes: f64,
}

pub fn parse_test_export_json(json: &str) -> Result<TestExport, serde_json::Error> {
    serde_json::from_str(json)
}

#[cfg(target_arch = "wasm32")]
type TimePoint = f64;
#[cfg(not(target_arch = "wasm32"))]
type TimePoint = Instant;

#[inline]
fn time_now() -> TimePoint {
    #[cfg(target_arch = "wasm32")]
    {
        Date::now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        Instant::now()
    }
}

#[inline]
fn elapsed_ms(start: TimePoint) -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        Date::now() - start
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        start.elapsed().as_secs_f64() * 1_000.0
    }
}

fn sparse_to_dense_mat(sparse: &SparseMatrix, rows: usize, cols: usize) -> Mat<F> {
    let mut data = vec![F::ZERO; rows * cols];
    for &(row, col, val) in &sparse.entries {
        data[row * cols + col] = F::from_u64(val);
    }
    Mat::from_row_major(rows, cols, data)
}

fn build_step_ccs(r1cs: &TestExport) -> CcsStructure<F> {
    let n = r1cs.num_constraints;
    let m = r1cs.num_variables;

    let n = n.max(m);

    let a = sparse_to_dense_mat(&r1cs.matrix_a, n, n);
    let b = sparse_to_dense_mat(&r1cs.matrix_b, n, n);
    let c = sparse_to_dense_mat(&r1cs.matrix_c, n, n);
    let s0 = r1cs_to_ccs(a, b, c);

    // ensure_identity_first_owned will now work since n == m_padded
    s0.ensure_identity_first_owned()
        .expect("ensure_identity_first_owned should succeed")
}

fn pad_witness_to_m(mut z: Vec<F>, m_target: usize) -> Vec<F> {
    z.resize(m_target, F::ZERO);
    z
}

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let pp = ajtai_setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

#[derive(Clone)]
struct NoInputs;

struct StepCircuit {
    steps: Vec<Vec<F>>,
    step_spec: StepSpec,
    step_ccs: Arc<CcsStructure<F>>,
}

impl NeoStep for StepCircuit {
    type ExternalInputs = NoInputs;

    fn state_len(&self) -> usize {
        0
    }

    fn step_spec(&self) -> StepSpec {
        self.step_spec.clone()
    }

    fn synthesize_step(
        &mut self,
        step_idx: usize,
        _y_prev: &[F],
        _inputs: &Self::ExternalInputs,
    ) -> StepArtifacts {
        let z = self.steps[step_idx].clone();
        let z_padded = pad_witness_to_m(z, self.step_ccs.m);
        StepArtifacts {
            ccs: self.step_ccs.clone(),
            witness: z_padded,
            public_app_inputs: vec![],
            spec: self.step_spec.clone(),
        }
    }
}

pub fn run_test_export(export: &TestExport) -> Result<TestExportResult, String> {
    let total_start = time_now();

    let r1cs_constraints = export.num_constraints;
    let r1cs_variables = export.num_variables;
    let r1cs_padded_n = r1cs_constraints.max(r1cs_variables);

    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(r1cs_padded_n)
        .map_err(|e| format!("goldilocks_auto_r1cs_ccs failed: {e}"))?;

    // Needed for now for the decomposition to be bijective in pi_ccs.
    params.b = 3;

    let params_summary = TestExportParamsSummary {
        b: params.b,
        d: params.d,
        kappa: params.kappa,
        k_rho: params.k_rho,
        T: params.T,
        s: params.s,
        lambda: params.lambda,
    };

    let ajtai_start = time_now();
    setup_ajtai_for_dims(r1cs_padded_n);
    let l = AjtaiSModule::from_global_for_dims(D, r1cs_padded_n)
        .map_err(|e| format!("Ajtai init: {e}"))?;
    let ajtai_setup_ms = elapsed_ms(ajtai_start);

    let step_spec = StepSpec {
        y_len: 0,
        const1_index: 0,
        y_step_indices: vec![],
        app_input_indices: Some(vec![]),
        m_in: 1,
    };

    let build_ccs_start = time_now();
    let step_ccs = Arc::new(build_step_ccs(export));
    let build_ccs_ms = elapsed_ms(build_ccs_start);

    let ccs_matrix_nnz: Vec<usize> = step_ccs
        .matrices
        .iter()
        .map(|m| match m {
            CcsMatrix::Identity { n } => *n,
            CcsMatrix::Csc(csc) => csc.vals.len(),
        })
        .collect();
    let ccs_matrix_nnz_total: usize = ccs_matrix_nnz.iter().sum();

    let witness_steps = export.witness.len();
    let mut witness_fields_total: usize = 0;
    let mut witness_fields_min: usize = usize::MAX;
    let mut witness_fields_max: usize = 0;
    let mut witness_nonzero_fields_total: usize = 0;
    for step in &export.witness {
        witness_fields_total = witness_fields_total.saturating_add(step.len());
        witness_fields_min = witness_fields_min.min(step.len());
        witness_fields_max = witness_fields_max.max(step.len());
        witness_nonzero_fields_total = witness_nonzero_fields_total
            .saturating_add(step.iter().filter(|&&v| v != 0).count());
    }
    if witness_steps == 0 {
        witness_fields_min = 0;
    }
    let witness_nonzero_ratio = if witness_fields_total == 0 {
        0.0
    } else {
        witness_nonzero_fields_total as f64 / witness_fields_total as f64
    };

    let prepare_witness_start = time_now();
    let mut circuit = StepCircuit {
        steps: export
            .witness
            .iter()
            .map(|step_witness| step_witness.iter().map(|f| F::from_u64(*f)).collect())
            .collect(),
        step_spec: step_spec.clone(),
        step_ccs: step_ccs.clone(),
    };
    let prepare_witness_ms = elapsed_ms(prepare_witness_start);

    let session_init_start = time_now();
    let mut session = FoldingSession::new(FoldingMode::Optimized, params, l.clone());
    let session_init_ms = elapsed_ms(session_init_start);
    // For multi-step runs, require a minimal linking constraint so verification is well-defined.
    // We link `x[0]` across adjacent steps; for typical R1CS exports this is the constant-1 slot.
    session.set_step_linking(StepLinkingConfig::new(vec![(0, 0)]));

    let add_steps_start = time_now();
    let mut add_step_sum: f64 = 0.0;
    let mut add_step_min: f64 = f64::INFINITY;
    let mut add_step_max: f64 = 0.0;
    for _ in 0..export.witness.len() {
        let step_start = time_now();
        session
            .add_step(&mut circuit, &NoInputs)
            .map_err(|e| format!("add_step failed: {e}"))?;
        let ms = elapsed_ms(step_start);
        add_step_sum += ms;
        add_step_min = add_step_min.min(ms);
        add_step_max = add_step_max.max(ms);
    }
    let add_steps_total_ms = elapsed_ms(add_steps_start);
    if witness_steps == 0 {
        add_step_min = 0.0;
    }
    let add_step_avg = if witness_steps == 0 {
        0.0
    } else {
        add_step_sum / witness_steps as f64
    };

    let fold_and_prove_start = time_now();
    let (run, fold_steps) = session
        .fold_and_prove_with_step_timings(step_ccs.as_ref())
        .map_err(|e| format!("fold_and_prove failed: {e}"))?;
    let fold_and_prove_ms = elapsed_ms(fold_and_prove_start);

    let mcss_public = session.mcss_public();
    let verify_start = time_now();
    let ok = session
        .verify(step_ccs.as_ref(), &mcss_public, &run)
        .map_err(|e| format!("verify failed: {e}"))?;
    let verify_ms = elapsed_ms(verify_start);
    if !ok {
        return Err("optimized verification failed".into());
    }

    let mut fold_lane_commitments: usize = 0;
    let mut mem_cpu_val_claim_commitments: usize = 0;
    let mut val_lane_commitments: usize = 0;
    for step in &run.steps {
        fold_lane_commitments = fold_lane_commitments
            .saturating_add(step.fold.ccs_out.len() + step.fold.dec_children.len() + 1);
        mem_cpu_val_claim_commitments = mem_cpu_val_claim_commitments
            .saturating_add(step.mem.cpu_me_claims_val.len());
        if let Some(val) = &step.val_fold {
            val_lane_commitments = val_lane_commitments.saturating_add(val.dec_children.len() + 1);
        }
    }
    let total_commitments: usize = fold_lane_commitments
        .saturating_add(mem_cpu_val_claim_commitments)
        .saturating_add(val_lane_commitments);

    let (commitment_d, commitment_kappa) = run
        .steps
        .first()
        .map(|s| (s.fold.rlc_parent.c.d, s.fold.rlc_parent.c.kappa))
        .unwrap_or((0, 0));
    let commitment_bytes: usize = commitment_d.saturating_mul(commitment_kappa).saturating_mul(8);
    let estimated_commitment_bytes: f64 = total_commitments as f64 * commitment_bytes as f64;

    let final_accumulator_len = run
        .steps
        .last()
        .map(|s| s.fold.dec_children.len())
        .unwrap_or(0);

    let circuit_summary = TestExportCircuitSummary {
        r1cs_constraints,
        r1cs_variables,
        r1cs_padded_n,
        r1cs_a_nnz: export.matrix_a.entries.len(),
        r1cs_b_nnz: export.matrix_b.entries.len(),
        r1cs_c_nnz: export.matrix_c.entries.len(),

        witness_steps,
        witness_fields_total,
        witness_fields_min,
        witness_fields_max,
        witness_nonzero_fields_total,
        witness_nonzero_ratio,

        ccs_n: step_ccs.n,
        ccs_m: step_ccs.m,
        ccs_t: step_ccs.t(),
        ccs_max_degree: step_ccs.max_degree(),
        ccs_poly_terms: step_ccs.f.terms().len(),
        ccs_matrix_nnz,
        ccs_matrix_nnz_total,
    };

    let fold_step_avg = if fold_steps.is_empty() {
        0.0
    } else {
        fold_steps.iter().sum::<f64>() / fold_steps.len() as f64
    };
    let fold_step_min = fold_steps.iter().copied().fold(f64::INFINITY, f64::min);
    let fold_step_min = if fold_steps.is_empty() || !fold_step_min.is_finite() {
        0.0
    } else {
        fold_step_min
    };
    let fold_step_max = fold_steps.iter().copied().fold(0.0, f64::max);

    let timings_ms = TestExportTimingsMs {
        ajtai_setup: ajtai_setup_ms,
        build_ccs: build_ccs_ms,
        prepare_witness: prepare_witness_ms,
        session_init: session_init_ms,

        add_steps_total: add_steps_total_ms,
        add_step_avg,
        add_step_min,
        add_step_max,

        fold_and_prove: fold_and_prove_ms,
        fold_steps,
        fold_step_avg,
        fold_step_min,
        fold_step_max,
        verify: verify_ms,
        total: elapsed_ms(total_start),
    };

    let mut k_in = Vec::with_capacity(run.steps.len());
    let mut acc_len = 0usize;
    for step in &run.steps {
        k_in.push(acc_len + 1);
        acc_len = step.fold.dec_children.len();
    }
    let acc_len_after = run.steps.iter().map(|s| s.fold.dec_children.len()).collect();
    let folding = TestExportFoldingSummary { k_in, acc_len_after };

    let proof_estimate = TestExportProofEstimate {
        proof_steps: run.steps.len(),
        final_accumulator_len,

        fold_lane_commitments,
        mem_cpu_val_claim_commitments,
        val_lane_commitments,
        total_commitments,

        commitment_d,
        commitment_kappa,
        commitment_bytes,
        estimated_commitment_bytes,
    };

    Ok(TestExportResult {
        steps: run.steps.len(),
        verify_ok: ok,
        circuit: circuit_summary,
        params: params_summary,
        timings_ms,
        proof_estimate,
        folding,
    })
}

