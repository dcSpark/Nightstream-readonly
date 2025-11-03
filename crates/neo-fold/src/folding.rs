//! Coordinator for repeated folding: for each MCS, run
//!   Π_CCS (via pi_ccs facade) → RLC (public) → DEC (public).
//!
//! Engine-agnostic: CCS goes through `pi_ccs::prove/verify` with FoldingMode.
//! RLC/DEC are performed publicly here, and commitment checks are injected
//! through user-supplied S-action mixers.

#![allow(non_snake_case)]

use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_transcript::labels as tr_labels;

use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance, Mat};
use neo_ajtai::Commitment as Cmt;

use neo_params::NeoParams;
use neo_math::{F, K, D};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::PiCcsError;
use crate::optimized_engine::context;

// Pull Π_CCS and RLC/DEC via the public facade (engine-agnostic)
use crate::pi_ccs::{self as ccs, FoldingMode, PiCcsProof};

/// One fold step’s artifacts.
#[derive(Clone, Debug)]
pub struct FoldStep {
    /// Π_CCS outputs (k ME(b,L) instances)
    pub ccs_out: Vec<MeInstance<Cmt, F, K>>,
    /// Π_CCS proof (engine-agnostic re-export)
    pub ccs_proof: PiCcsProof,

    /// RLC mixing matrices ρ_i ∈ S ⊆ F^{D×D}
    pub rlc_rhos: Vec<Mat<F>>,
    /// The combined parent after RLC: ME(B,L) with B=b^k
    pub rlc_parent: MeInstance<Cmt, F, K>,

    /// DEC children: k ME(b,L) after decomposition of the parent
    pub dec_children: Vec<MeInstance<Cmt, F, K>>,
}

/// Entire multi-fold run.
#[derive(Clone, Debug)]
pub struct FoldRun {
    pub steps: Vec<FoldStep>,
    /// Alias to the last step's children (final outputs)
    pub final_outputs: Vec<MeInstance<Cmt, F, K>>,
}

/// Commitment mixers so the coordinator stays scheme-agnostic.
/// - `mix_rhos_commits(ρ, cs)` returns Σ ρ_i · c_i  (S-action).
/// - `combine_b_pows(cs, b)` returns Σ \bar b^{i-1} c_i  (DEC check).
#[derive(Clone, Copy)]
pub struct CommitMixers<MR, MB>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
    MB: Fn(&[Cmt], u32) -> Cmt,
{
    pub mix_rhos_commits: MR,
    pub combine_b_pows: MB,
}

/// Plan: number of folds = number of MCS instances.
#[inline]
pub fn plan_num_folds(mcs_count: usize) -> usize { mcs_count }

// ---------------------------------------------------------------------------
// Prover orchestration
// ---------------------------------------------------------------------------

/// Run all folds (one per MCS). The fan-in `k` is inferred: `k = acc_init.len() + 1`.
///
/// Inputs:
/// - `mode`: which Π_CCS engine to use (Optimized / PaperExact / Crosscheck).
/// - `mcss`: Vec of (MCS instance, witness) to be folded, in order.
/// - `acc_init`: initial accumulator (k-1 ME(b,L) instances).
/// - `acc_wit_init`: witnesses Z for the initial accumulator (same order/length as `acc_init`).
/// - `l`: S-module homomorphism (commitment map) to open children in DEC.
/// - `mixers`: commitment mixers for RLC and DEC checks.
pub fn fold_many_prove<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_in: &CcsStructure<F>,
    mcss: &[(McsInstance<Cmt, F>, McsWitness<F>)],
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<FoldRun, PiCcsError>
where
    F: PrimeField64 + PrimeCharacteristicRing + Copy,
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    tr.append_message(tr_labels::PI_CCS, b"");
    let s = s_in
        .ensure_identity_first()
        .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;
    let context::Dims { ell_d, .. } = context::build_dims_and_policy(params, &s)?;

    // Infer k from initial accumulator
    if acc_init.len() != acc_wit_init.len() {
        return Err(PiCcsError::InvalidInput("initial accumulator size mismatch".into()));
    }
    let k = acc_init.len() + 1;

    // Current accumulator (carried inputs)
    let mut accumulator = acc_init.to_vec();
    let mut accumulator_wit = acc_wit_init.to_vec();

    let mut steps = Vec::with_capacity(mcss.len());

    for (idx, (mcs_i, wit_i)) in mcss.iter().enumerate() {
        // --- Π_CCS via facade (chooses engine by `mode`)
        let (ccs_out, ccs_proof) = ccs::prove(
            mode.clone(),
            tr, params, &s,
            core::slice::from_ref(mcs_i),
            core::slice::from_ref(wit_i),
            &accumulator,
            &accumulator_wit,
            l,
        )?;

        if ccs_out.len() != k {
            return Err(PiCcsError::ProtocolError(format!(
                "Π_CCS returned {} outputs; expected k={k}", ccs_out.len()
            )));
        }

        // Witnesses for CCS outputs: [Z_mcs, Z_seed...]
        let mut outs_Z: Vec<Mat<F>> = Vec::with_capacity(k);
        outs_Z.push(wit_i.Z.clone());
        outs_Z.extend(accumulator_wit.iter().cloned());

        // --- RLC (via facade): pick ρ, combine k → parent
        let rhos = ccs::sample_diag_rhos(tr, k, params)?;
        // Compute parent via RLC and combined witness Z_mix = Σ ρ_i·Z_i
        let (mut parent_pub, Z_mix) = ccs::rlc_with_commit(
            mode.clone(),
            &s, params, &rhos, &ccs_out, &outs_Z, ell_d, mixers.mix_rhos_commits,
        );

        // Recompute parent's y from Z_mix to ensure DEC y-equality holds in all modes.
        // y_j := Z_mix · (M_j^T · r^b) with padding to 2^{ell_d}; y_scalars are base-b recomposition.
        let (y_new, y_scalars) = ccs::compute_y_from_Z_and_r(&s, &Z_mix, &parent_pub.r, ell_d, params.b);
        parent_pub.y = y_new;
        parent_pub.y_scalars = y_scalars;

        // --- DEC (public): split Z_mix into k children; compute (c_i, X_i, y_(i,j))
        let Z_split = ccs::split_b_matrix_k(&Z_mix, k, params.b)?;
        // Commitments for children via L(Z_i)
        let child_cs: Vec<Cmt> = Z_split.iter().map(|Zi| l.commit(Zi)).collect();
        let (children, ok_y, ok_X, ok_c) = ccs::dec_children_with_commit(
            mode.clone(),
            &s, params, &parent_pub, &Z_split, ell_d, &child_cs, mixers.combine_b_pows,
        );
        // Enforce y/X/commitment equality in all modes by default.
        if !(ok_y && ok_X && ok_c) {
            return Err(PiCcsError::ProtocolError(format!(
                "DEC public check failed (y={}, X={}, c={})", ok_y, ok_X, ok_c
            )));
        }

        // Advance accumulator
        accumulator = children.clone();
        accumulator_wit = Z_split;

        steps.push(FoldStep {
            ccs_out,
            ccs_proof,
            rlc_rhos: rhos,
            rlc_parent: parent_pub,
            dec_children: children,
        });

        tr.append_message(b"fold/step_done", &(idx as u64).to_le_bytes());
    }

    let final_outputs = accumulator.clone();
    Ok(FoldRun { steps, final_outputs })
}

// ---------------------------------------------------------------------------
// Verifier orchestration
// ---------------------------------------------------------------------------

/// Verify a multi-fold run end-to-end.
/// Fan-in `k` is inferred from the initial accumulator: `k = acc_init.len() + 1`.
pub fn fold_many_verify<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_in: &CcsStructure<F>,
    mcss: &[McsInstance<Cmt, F>],
    acc_init: &[MeInstance<Cmt, F, K>],
    run: &FoldRun,
    mixers: CommitMixers<MR, MB>,
) -> Result<bool, PiCcsError>
where
    F: PrimeField64 + PrimeCharacteristicRing + Copy,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    tr.append_message(tr_labels::PI_CCS, b"");
    let s = s_in
        .ensure_identity_first()
        .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;
    let context::Dims { ell_d, .. } = context::build_dims_and_policy(params, &s)?;

    if mcss.len() != run.steps.len() {
        return Err(PiCcsError::InvalidInput("mcss.len() must equal run.steps.len()".into()));
    }

    let mut accumulator = acc_init.to_vec(); // public initial accumulator (k-1)

    for (i, step) in run.steps.iter().enumerate() {
        // 1) Verify Π_CCS via facade (engine-agnostic)
        let ok_ccs = ccs::verify(
            mode.clone(),
            tr, params, &s,
            core::slice::from_ref(&mcss[i]),
            &accumulator,
            &step.ccs_out,
            &step.ccs_proof,
        )?;
        if !ok_ccs {
            return Err(PiCcsError::SumcheckError("Π_CCS verification failed".into()));
        }

        // 2) Verify RLC publicly: recompute parent from (rhos, ccs_out)
        let parent_pub = ccs::rlc_public(&s, params, &step.rlc_rhos, &step.ccs_out, mixers.mix_rhos_commits, ell_d);
        // Check X, y, c, r
        if parent_pub.X.as_slice() != step.rlc_parent.X.as_slice() {
            return Err(PiCcsError::ProtocolError("RLC X mismatch".into()));
        }
        if parent_pub.y.len() != step.rlc_parent.y.len()
            || parent_pub.y.iter().zip(&step.rlc_parent.y).any(|(a, b)| a != b)
        {
            return Err(PiCcsError::ProtocolError("RLC y mismatch".into()));
        }
        if parent_pub.c != step.rlc_parent.c {
            return Err(PiCcsError::ProtocolError("RLC commitment mismatch".into()));
        }
        if parent_pub.r != step.rlc_parent.r {
            return Err(PiCcsError::ProtocolError("RLC r mismatch".into()));
        }

        // 3) Verify DEC publicly (X, y, c)
        if !ccs::verify_dec_public(&s, params, &step.rlc_parent, &step.dec_children, mixers.combine_b_pows, ell_d) {
            return Err(PiCcsError::ProtocolError("DEC public check failed".into()));
        }

        // Advance accumulator
        accumulator = step.dec_children.clone();
        tr.append_message(b"fold/verify_step_done", &(i as u64).to_le_bytes());
    }

    Ok(true)
}
