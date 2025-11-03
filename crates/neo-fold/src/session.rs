//! Thin, ergonomic session layer for Π-CCS folding.
//!
//! This module provides:
//!   1) A small adapter trait (`NeoStep`) and `prove_step` for Nova/Sonobe-style step synthesis.
//!   2) A direct IO path via `ProveInput` + `prove_step_from_io` (callers hand us (x, w)).
//!   3) A session driver (`FoldingSession`) that **hides** commitment mixers.
//!
//! Concepts (paper-aligned):
//! - Accumulator: the k ME(b, L) claims you carry between steps (the inputs to the next Π_CCS).
//! - k is never defaulted: if you supply Accumulator with k>0, we multi-fold; if not, k=1 (simple case).
//!
//! Notes:
//! - We default to Ajtai mixers internally (no frontend mixers required).
//! - If you want k>1, pass an explicit Accumulator via `with_initial_accumulator(...)`.
//!
//! This file contains only ergonomics. The formal pipeline Π_CCS → Π_RLC → Π_DEC
//! lives in `fold_many_prove/verify` (engine-agnostic).

#![allow(non_snake_case)]

use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance, Mat};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ajtai::{Commitment as Cmt, s_lincomb, s_mul, decomp_b, DecompStyle};
use neo_params::NeoParams;
use neo_math::{F, D, K};
use neo_math::ring::Rq as RqEl;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::folding::{self, CommitMixers, FoldRun};
use crate::pi_ccs::FoldingMode;
use crate::PiCcsError;

/// Optional application-level "output claim".
/// (Not consumed by Π-CCS core yet; kept for API parity / future use.)
#[derive(Clone, Debug)]
pub struct OutputClaim<Ff> {
    pub tag: &'static [u8],
    pub expected: Ff,
}

/// Direct inputs for a single step when you don't want to implement `NeoStep`.
/// We'll compute the commitment and split (x | w) for you.
#[derive(Clone, Debug)]
pub struct ProveInput<'a> {
    pub ccs: &'a CcsStructure<F>,   // the circuit (must match witness length)
    pub public_input: &'a [F],      // x
    pub witness: &'a [F],           // w
    pub output_claims: &'a [OutputClaim<F>], // optional; recorded but not enforced here
}

/// Where special coordinates live inside the step witness `z`.
#[derive(Clone, Debug)]
pub struct StepSpec {
    /// Number of public state elements carried between steps (y).
    pub y_len: usize,
    /// Index of constant 1 inside z (optional informational).
    pub const1_index: usize,
    /// Indices in z carried out as next-step public state (y).
    pub y_step_indices: Vec<usize>,
    /// Optional: indices in z corresponding to app-level public inputs.
    pub app_input_indices: Option<Vec<usize>>,
    /// Number of public inputs `m_in` (prefix length in z).
    pub m_in: usize,
}

/// What a step must return for the session to run Π-CCS.
#[derive(Clone, Debug)]
pub struct StepArtifacts {
    pub ccs: CcsStructure<F>,
    /// Concrete witness vector for this step (length = m).
    pub witness: Vec<F>,
    /// App inputs you want logged/exposed (optional, informational).
    pub public_app_inputs: Vec<F>,
    pub spec: StepSpec,
}

/// User implements this for their program.
pub trait NeoStep {
    type ExternalInputs: Clone;

    /// How many public state elements are threaded between steps?
    fn state_len(&self) -> usize;

    /// Static metadata about where special elements live inside `z`.
    fn step_spec(&self) -> StepSpec;

    /// Produce the CCS structure and a concrete witness for this step.
    fn synthesize_step(
        &mut self,
        step_idx: usize,
        y_prev: &[F],
        inputs: &Self::ExternalInputs,
    ) -> StepArtifacts;
}

/// Decompose z ∈ F^m into base-b digits Z ∈ F^{D×m} (unbalanced, current MCS path).
fn decompose_z_to_Z(params: &NeoParams, z: &[F]) -> Mat<F> {
    let d = D;
    let m = z.len();
    let b = params.b as u64;
    let mut Z = Mat::zero(d, m, F::ZERO);
    for c in 0..m {
        let mut v = z[c].as_canonical_u64();
        for rho in 0..d {
            Z[(rho, c)] = F::from_u64(v % b);
            v /= b;
        }
    }
    Z
}

/// Convert a diagonal (scalar) matrix to an Ajtai ring element for S-action.
/// Assumes ρ = s·I; takes the (0,0) entry as s.
fn diag_mat_to_rq(mat: &Mat<F>) -> RqEl {
    debug_assert_eq!(mat.rows(), D);
    debug_assert_eq!(mat.cols(), D);
    let s = mat[(0, 0)];
    RqEl::from_field_scalar(s)
}

/// Default Ajtai mixers (hidden internally).
fn default_mixers() -> CommitMixers<
    fn(&[Mat<F>], &[Cmt]) -> Cmt,
    fn(&[Cmt], u32) -> Cmt
> {
    fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Cmt]) -> Cmt {
        let rq_els: Vec<RqEl> = rhos.iter().map(diag_mat_to_rq).collect();
        s_lincomb(&rq_els, cs).expect("s_lincomb should succeed")
    }
    fn combine_b_pows(cs: &[Cmt], b: u32) -> Cmt {
        let mut acc = cs[0].clone();
        let mut pow = F::from_u64(b as u64);
        for i in 1..cs.len() {
            let rq_pow = RqEl::from_field_scalar(pow);
            let term = s_mul(&rq_pow, &cs[i]);
            acc.add_inplace(&term);
            pow *= F::from_u64(b as u64);
        }
        acc
    }
    CommitMixers { mix_rhos_commits, combine_b_pows }
}

/// An *Accumulator* is the k ME(b, L) claims carried between steps.
/// This is exactly the paper's ME(b, L)^k input vector for the next Π_CCS.
#[derive(Clone, Debug)]
pub struct Accumulator {
    pub me: Vec<MeInstance<Cmt, F, K>>,
    pub witnesses: Vec<Mat<F>>, // Z_i for each me[i]
}

impl Accumulator {
    /// Sanity checks: dimensions, common r, consistent m_in, and witness shape.
    pub fn check(&self, params: &NeoParams, s: &CcsStructure<F>) -> Result<(), PiCcsError> {
        if self.me.len() != self.witnesses.len() {
            return Err(PiCcsError::InvalidInput("Accumulator: me.len() != witnesses.len()".into()));
        }
        if self.me.is_empty() {
            return Ok(());
        }
        // Dims for r length
        use crate::optimized_engine::context;
        let dims = context::build_dims_and_policy(params, s)?;
        let ell_n = dims.ell_n;

        // Common r and m_in
        let r0 = &self.me[0].r;
        if r0.len() != ell_n {
            return Err(PiCcsError::InvalidInput(format!(
                "Accumulator: r length mismatch (expected ell_n={}, got {})", ell_n, r0.len()
            )));
        }
        let m_in0 = self.me[0].m_in;
        for (i, (m, Z)) in self.me.iter().zip(self.witnesses.iter()).enumerate() {
            if m.r.len() != ell_n {
                return Err(PiCcsError::InvalidInput(format!(
                    "Accumulator[{}]: r length mismatch (expected ell_n={}, got {})", i, ell_n, m.r.len()
                )));
            }
            if m.r != *r0 {
                return Err(PiCcsError::InvalidInput("Accumulator: all ME inputs must share the same r".into()));
            }
            if m.m_in != m_in0 {
                return Err(PiCcsError::InvalidInput("Accumulator: all ME inputs must share the same m_in".into()));
            }
            if Z.rows() != D || Z.cols() != s.m {
                return Err(PiCcsError::InvalidInput(format!(
                    "Accumulator[{}]: Z has shape {}x{}, expected {}x{}", i, Z.rows(), Z.cols(), D, s.m
                )));
            }
            if m.X.rows() != D || m.X.cols() != m.m_in {
                return Err(PiCcsError::InvalidInput("Accumulator: X dimension mismatch with m_in".into()));
            }
        }
        Ok(())
    }
}

/// Ergonomic helper: build an ME(b, L) instance from a raw witness z with **balanced** digits.
/// This is handy for constructing an explicit Accumulator.
///
/// - `z` is the full vector (x || w), length must equal `s.m`.
/// - `r` must have length `ell_n` (from dims).
/// - `m_in` is how many columns of Z to project into X (first m_in).
pub fn me_from_z_balanced<Lm: SModuleHomomorphism<F, Cmt>>(
    params: &NeoParams,
    s: &CcsStructure<F>,  // should be identity-first
    l: &Lm,
    z: &[F],
    r: &[K],
    m_in: usize,
) -> Result<(MeInstance<Cmt, F, K>, Mat<F>), PiCcsError> {
    if z.len() != s.m {
        return Err(PiCcsError::InvalidInput(format!(
            "me_from_z_balanced: z length {} != CCS.m {}", z.len(), s.m
        )));
    }
    if m_in > s.m {
        return Err(PiCcsError::InvalidInput("me_from_z_balanced: m_in exceeds s.m".into()));
    }
    use crate::optimized_engine::context;
    let dims = context::build_dims_and_policy(params, s)?;
    if r.len() != dims.ell_n {
        return Err(PiCcsError::InvalidInput(format!(
            "me_from_z_balanced: r length {} != ell_n {}", r.len(), dims.ell_n
        )));
    }

    // Balanced Ajtai decomposition (row-major D×m), matching existing tests/tools.
    let d = D;
    let m = z.len();
    let z_digits = decomp_b(z, params.b, d, DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for col in 0..m {
        for row in 0..d {
            row_major[row * m + col] = z_digits[col * d + row];
        }
    }
    let Z = Mat::from_row_major(d, m, row_major);
    let c = l.commit(&Z);

    // X := first m_in columns of Z
    let mut X = Mat::zero(d, m_in, F::ZERO);
    for rr in 0..d {
        for cc in 0..m_in {
            X[(rr, cc)] = Z[(rr, cc)];
        }
    }

    // y, y_scalars: sized correctly, zeros (they're recomputed after RLC in the orchestrator)
    let t = s.t();
    let y = vec![vec![K::ZERO; d]; t];
    let y_scalars = vec![K::ZERO; t];

    let me = MeInstance::<Cmt, F, K> {
        c_step_coords: vec![], u_offset: 0, u_len: 0,
        c,
        X,
        r: r.to_vec(),
        y,
        y_scalars,
        m_in,
        fold_digest: [0u8; 32],
    };

    Ok((me, Z))
}

/// Minimal session that provides an ergonomic per-step API.
/// Mixers are hidden and default to Ajtai; the initial Accumulator is optional.
/// If absent, we run the *simple* k=1 case (no ME inputs).
pub struct FoldingSession<L>
where
    L: SModuleHomomorphism<F, Cmt> + Clone,
{
    mode: FoldingMode,
    params: NeoParams,
    l: L,
    mixers: CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>,

    // Collected MCS steps (instance + witness), in order
    mcss: Vec<(McsInstance<Cmt, F>, McsWitness<F>)>,

    // Optional initial accumulated ME(b, L)^k inputs (k = me.len()).
    acc0: Option<Accumulator>,

    // Optional: app-level claims recorded per step (not enforced here yet)
    step_claims: Vec<Vec<OutputClaim<F>>>,
}

impl<L> FoldingSession<L>
where
    L: SModuleHomomorphism<F, Cmt> + Clone,
{
    /// Create a new session with default Ajtai mixers and no initial accumulator (k=1 simple flow).
    pub fn new(
        mode: FoldingMode,
        params: NeoParams,
        l: L,
    ) -> Self {
        Self {
            mode,
            params,
            l,
            mixers: default_mixers(),
            mcss: vec![],
            acc0: None,
            step_claims: vec![],
        }
    }

    /// Inject an explicit initial Accumulator (k = acc.me.len()). This enables k>1 multi-folding.
    pub fn with_initial_accumulator(mut self, acc: Accumulator, s: &CcsStructure<F>) -> Result<Self, PiCcsError> {
        let s_norm = s
            .ensure_identity_first()
            .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;
        acc.check(&self.params, &s_norm)?;
        self.acc0 = Some(acc);
        Ok(self)
    }

    /// Access the accumulated public MCS instances (for verification APIs).
    pub fn mcss_public(&self) -> Vec<McsInstance<Cmt, F>> {
        self.mcss.iter().map(|(i, _)| i.clone()).collect()
    }

    /// Push one step using the `NeoStep` synthesis adapter.
    pub fn prove_step<S: NeoStep>(
        &mut self,
        stepper: &mut S,
        inputs: &S::ExternalInputs,
    ) -> Result<(), PiCcsError> {
        let y_prev = vec![F::ZERO; stepper.state_len()];
        let StepArtifacts { ccs, witness: z, spec, public_app_inputs: _ } =
            stepper.synthesize_step(self.mcss.len(), &y_prev, inputs);

        // Identity-first normalization
        let s_norm = ccs
            .ensure_identity_first()
            .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;

        // z must match CCS dimension
        if z.len() != s_norm.m {
            return Err(PiCcsError::InvalidInput(format!(
                "step witness length {} != CCS.m {}",
                z.len(), s_norm.m
            )));
        }
        if spec.m_in > z.len() {
            return Err(PiCcsError::InvalidInput("m_in exceeds witness length".into()));
        }

        // Build MCS instance + witness
        let Z = decompose_z_to_Z(&self.params, &z);
        let c = self.l.commit(&Z);
        let m_in = spec.m_in;
        let x = z[..m_in].to_vec();
        let w = z[m_in..].to_vec();

        let mcs_inst = McsInstance { c, x, m_in };
        let mcs_wit = McsWitness { w, Z };

        self.mcss.push((mcs_inst, mcs_wit));
        self.step_claims.push(vec![]);

        Ok(())
    }

    /// Push one step directly from (x, w) without implementing `NeoStep`.
    /// We compute the commitment and split (x | w) for you.
    pub fn prove_step_from_io(
        &mut self,
        input: &ProveInput<'_>,
    ) -> Result<(), PiCcsError> {
        // Normalize CCS to identity-first
        let s_norm = input
            .ccs
            .ensure_identity_first()
            .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;

        let m_in = input.public_input.len();
        if m_in + input.witness.len() != s_norm.m {
            return Err(PiCcsError::InvalidInput(format!(
                "len(x) + len(w) = {} but CCS.m = {}",
                m_in + input.witness.len(),
                s_norm.m
            )));
        }

        // Build z = [x | w], compute Z and commitment c
        let mut z = Vec::with_capacity(s_norm.m);
        z.extend_from_slice(input.public_input);
        z.extend_from_slice(input.witness);

        let Z = decompose_z_to_Z(&self.params, &z);
        let c = self.l.commit(&Z);

        // Produce MCS instance + witness
        let mcs_inst = McsInstance { c, x: input.public_input.to_vec(), m_in };
        let mcs_wit = McsWitness { w: input.witness.to_vec(), Z };

        self.mcss.push((mcs_inst, mcs_wit));
        self.step_claims.push(input.output_claims.to_vec());

        Ok(())
    }

    /// Finalize: run folding over all collected steps and return a `FoldRun`.
    /// This method manages the transcript internally for ease of use.
    pub fn finalize(
        &mut self,
        s: &CcsStructure<F>,
    ) -> Result<FoldRun, PiCcsError> {
        let mut tr = Poseidon2Transcript::new(b"neo.fold/session");
        self.finalize_with_transcript(&mut tr, s)
    }

    /// Finalize with a caller-provided transcript (advanced users).
    pub fn finalize_with_transcript(
        &mut self,
        tr: &mut Poseidon2Transcript,
        s: &CcsStructure<F>,
    ) -> Result<FoldRun, PiCcsError> {
        // Normalize CCS
        let s_norm = s
            .ensure_identity_first()
            .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;

        // Determine canonical m_in from steps and ensure they all match (needed for RLC).
        let m_in_steps = self
            .mcss
            .first()
            .map(|(inst, _)| inst.m_in)
            .unwrap_or(0);
        if !self.mcss.iter().all(|(inst, _)| inst.m_in == m_in_steps) {
            return Err(PiCcsError::InvalidInput(
                "all steps must share the same m_in".into(),
            ));
        }

        // Validate or default the accumulator: None → k=1 simple case (no ME inputs).
        let (seed_me, seed_me_wit) = match &self.acc0 {
            Some(acc) => {
                acc.check(&self.params, &s_norm)?;
                // Also ensure accumulator m_in matches steps' m_in to avoid X-mixing shape issues.
                let acc_m_in = acc.me.first().map(|m| m.m_in).unwrap_or(m_in_steps);
                if acc_m_in != m_in_steps {
                    return Err(PiCcsError::InvalidInput(
                        "initial Accumulator.m_in must match steps' m_in".into(),
                    ));
                }
                (acc.me.clone(), acc.witnesses.clone())
            }
            None => (vec![], vec![]), // k=1
        };

        let mcss: Vec<(McsInstance<Cmt, F>, McsWitness<F>)> = self.mcss.clone();

        folding::fold_many_prove(
            self.mode.clone(),
            tr,
            &self.params,
            &s_norm,
            &mcss,
            &seed_me,
            &seed_me_wit,
            &self.l,
            self.mixers,
        )
    }

    /// Verify a finished run against the public MCS list.
    /// This method manages the transcript internally for ease of use.
    pub fn verify(
        &self,
        s: &CcsStructure<F>,
        mcss_public: &[neo_ccs::McsInstance<Cmt, F>],
        run: &FoldRun,
    ) -> Result<bool, PiCcsError> {
        let mut tr = Poseidon2Transcript::new(b"neo.fold/session");
        self.verify_with_transcript(&mut tr, s, mcss_public, run)
    }

    /// Verify with a caller-provided transcript (advanced users).
    pub fn verify_with_transcript(
        &self,
        tr: &mut Poseidon2Transcript,
        s: &CcsStructure<F>,
        mcss_public: &[neo_ccs::McsInstance<Cmt, F>],
        run: &FoldRun,
    ) -> Result<bool, PiCcsError> {
        // Normalize CCS
        let s_norm = s
            .ensure_identity_first()
            .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;

        // m_in consistency across public MCS
        let m_in_steps = mcss_public.first().map(|inst| inst.m_in).unwrap_or(0);
        if !mcss_public.iter().all(|inst| inst.m_in == m_in_steps) {
            return Err(PiCcsError::InvalidInput(
                "all steps must share the same m_in".into(),
            ));
        }

        // Validate (or empty) initial accumulator to mirror finalize()
        let (seed_me, _seed_me_wit) = match &self.acc0 {
            Some(acc) => {
                acc.check(&self.params, &s_norm)?;
                let acc_m_in = acc.me.first().map(|m| m.m_in).unwrap_or(m_in_steps);
                if acc_m_in != m_in_steps {
                    return Err(PiCcsError::InvalidInput(
                        "initial Accumulator.m_in must match steps' m_in".into(),
                    ));
                }
                (acc.me.clone(), acc.witnesses.clone())
            }
            None => (vec![], vec![]), // k=1
        };

        folding::fold_many_verify(
            self.mode.clone(),
            tr,
            &self.params,
            &s_norm,
            mcss_public,
            &seed_me,
            run,
            self.mixers,
        )
    }
}
