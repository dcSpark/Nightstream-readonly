use crate::PiCcsProof;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{matrix::Mat, MeInstance};
use neo_math::{F, K};
use neo_memory::output_check::OutputBindingProof;

pub type TwistProofK = neo_memory::twist::TwistProof<K>;
pub type ShoutProofK = neo_memory::shout::ShoutProof<K>;

/// Route A Shout address pre-time proof metadata, grouped by `ell_addr`.
///
/// Shout addr-pre is an address-domain sumcheck, and the number of rounds equals
/// `ell_addr = d * ell` (the number of address-bit columns per lane under bit-addressing).
///
/// For performance, we batch multiple Shout lanes together using shared challenges
/// (batched sumcheck). The batched sumcheck requires *all claims in the batch to have the same
/// number of rounds*, so we group lanes by `ell_addr` and run one batched sumcheck per group.
///
/// Within each group, when a Shout lane is provably inactive for a step (no lookups), we can
/// skip its address-domain sumcheck entirely. We still bind all `claimed_sums` to the transcript,
/// but we include sumcheck rounds only for the active subset.
#[derive(Clone, Debug)]
pub struct ShoutAddrPreProof<KK> {
    /// Claimed sums per Shout lane.
    ///
    /// Lanes are flattened in `(lut_idx, lane_idx)` order, where `lut_idx` is the
    /// index in `step.lut_instances`, and `lane_idx` ranges over `inst.lanes.max(1)`.
    pub claimed_sums: Vec<KK>,
    /// Per-`ell_addr` batched sumcheck proofs.
    ///
    /// Groups must be sorted by `ell_addr` and contain at most one entry per `ell_addr`.
    pub groups: Vec<ShoutAddrPreGroupProof<KK>>,
}

#[derive(Clone, Debug)]
pub struct ShoutAddrPreGroupProof<KK> {
    /// Address-bit width (sumcheck round count) for this group.
    pub ell_addr: u32,
    /// Indices of active lanes (into `claimed_sums`) that include address sumcheck rounds.
    ///
    /// This list must be strictly increasing.
    pub active_lanes: Vec<u32>,
    /// Sumcheck rounds for active lanes only, in `active_lanes` order.
    ///
    /// `round_polys[active_idx][round] = coeffs`, and each inner `round` vector has length `ell_addr`.
    pub round_polys: Vec<Vec<Vec<KK>>>,
    /// Shared terminal address point for this group (length = `ell_addr`).
    pub r_addr: Vec<KK>,
}

impl<KK> Default for ShoutAddrPreProof<KK> {
    fn default() -> Self {
        Self {
            claimed_sums: Vec::new(),
            groups: Vec::new(),
        }
    }
}

/// One fold step’s artifacts (Π_CCS → Π_RLC → Π_DEC).
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

#[derive(Clone, Debug)]
#[must_use]
pub struct ShardObligations<C, FF, KK> {
    pub main: Vec<MeInstance<C, FF, KK>>,
    pub val: Vec<MeInstance<C, FF, KK>>,
}

impl<C, FF, KK> ShardObligations<C, FF, KK> {
    pub fn all_len(&self) -> usize {
        self.main.len() + self.val.len()
    }

    pub fn iter_all(&self) -> impl Iterator<Item = &MeInstance<C, FF, KK>> {
        self.main.iter().chain(self.val.iter())
    }

    pub fn require_all_finalized(
        &self,
        did_finalize_main: bool,
        did_finalize_val: bool,
    ) -> Result<(), crate::PiCcsError> {
        if !self.main.is_empty() && !did_finalize_main {
            return Err(crate::PiCcsError::ProtocolError(
                "finalizer did not process main obligations".into(),
            ));
        }
        if !self.val.is_empty() && !did_finalize_val {
            return Err(crate::PiCcsError::ProtocolError(
                "finalizer did not process val-lane obligations".into(),
            ));
        }
        Ok(())
    }

    pub fn split(self) -> (Vec<MeInstance<C, FF, KK>>, Vec<MeInstance<C, FF, KK>>) {
        (self.main, self.val)
    }
}

#[derive(Clone, Debug)]
#[must_use]
pub struct ShardFoldOutputs<C, FF, KK> {
    pub obligations: ShardObligations<C, FF, KK>,
}

#[derive(Clone, Debug)]
pub struct ShardFoldWitnesses<FF> {
    /// Witnesses for `ShardFoldOutputs::obligations.main` (one per ME instance).
    pub final_main_wits: Vec<Mat<FF>>,
    /// Witnesses for `ShardFoldOutputs::obligations.val` (one per ME instance).
    pub val_lane_wits: Vec<Mat<FF>>,
}

#[derive(Clone, Debug)]
pub enum MemOrLutProof {
    Twist(TwistProofK),
    Shout(ShoutProofK),
}

#[derive(Clone, Debug)]
pub struct MemSidecarProof<C, FF, KK> {
    /// ME claims evaluated at `r_val` (Twist val-eval terminal point).
    ///
    /// Shared-bus mode only: these are CPU ME openings at `r_val` that include appended bus openings.
    pub val_me_claims: Vec<MeInstance<C, FF, KK>>,
    /// CPU ME openings at `r_time` used to bind WB booleanity terminals to committed trace columns.
    pub wb_me_claims: Vec<MeInstance<C, FF, KK>>,
    /// CPU ME openings at `r_time` used to bind WP quiescence terminals to committed trace columns.
    pub wp_me_claims: Vec<MeInstance<C, FF, KK>>,
    /// Route A Shout address pre-time proofs batched across all Shout instances in the step.
    pub shout_addr_pre: ShoutAddrPreProof<KK>,
    pub proofs: Vec<MemOrLutProof>,
}

/// Proof for the Route A shared-challenge batched sum-check (time/row rounds).
///
/// This batches CCS (row/time rounds) with Twist/Shout time-domain oracles so all
/// protocols share the same transcript-derived `r` (enabling Π_RLC folding).
#[derive(Clone, Debug)]
pub struct BatchedTimeProof {
    /// Claimed sums per participating oracle (in the same order as `round_polys`).
    pub claimed_sums: Vec<K>,
    /// Degree bounds per participating oracle.
    pub degree_bounds: Vec<usize>,
    /// Domain-separation labels per participating oracle.
    pub labels: Vec<&'static [u8]>,
    /// Per-claim sum-check messages: `round_polys[claim][round] = coeffs`.
    pub round_polys: Vec<Vec<Vec<K>>>,
}

/// Proof data for a standalone Π_RLC → Π_DEC lane (no Π_CCS).
#[derive(Clone, Debug)]
pub struct RlcDecProof {
    /// RLC mixing matrices ρ_i ∈ S ⊆ F^{D×D}
    pub rlc_rhos: Vec<Mat<F>>,
    /// The combined parent after RLC: ME(B,L) with B=b^k
    pub rlc_parent: MeInstance<Cmt, F, K>,
    /// DEC children: k ME(b,L) after decomposition of the parent
    pub dec_children: Vec<MeInstance<Cmt, F, K>>,
}

#[derive(Clone, Debug)]
pub struct StepProof {
    pub fold: FoldStep,
    pub mem: MemSidecarProof<Cmt, F, K>,
    pub batched_time: BatchedTimeProof,
    /// Optional folding lane(s) for ME claims evaluated at `r_val`.
    ///
    /// Each proof is an independent Π_RLC→Π_DEC lane (k=1 in current usage).
    pub val_fold: Vec<RlcDecProof>,
    /// Reserved WB folding lane(s) for staged booleanity claims.
    pub wb_fold: Vec<RlcDecProof>,
    /// Reserved WP folding lane(s) for staged quiescence claims.
    pub wp_fold: Vec<RlcDecProof>,
}

#[derive(Clone, Debug)]
pub struct ShardProof {
    pub steps: Vec<StepProof>,
    /// Optional output binding proof (proves final memory matches claimed outputs).
    /// Twist linkage is proven as an extra Route-A batched-time claim on the final step.
    pub output_proof: Option<OutputBindingProof>,
}

impl ShardProof {
    pub fn compute_final_obligations(&self, acc_init: &[MeInstance<Cmt, F, K>]) -> ShardObligations<Cmt, F, K> {
        self.compute_fold_outputs(acc_init).obligations
    }

    /// Returns the final main accumulator only (does not include Twist `r_val` obligations).
    pub fn compute_final_main_children(&self, acc_init: &[MeInstance<Cmt, F, K>]) -> Vec<MeInstance<Cmt, F, K>> {
        self.compute_fold_outputs(acc_init).obligations.main
    }

    /// Legacy alias for CCS-only codepaths: compute the final main accumulator.
    pub fn compute_final_outputs(&self, acc_init: &[MeInstance<Cmt, F, K>]) -> Vec<MeInstance<Cmt, F, K>> {
        self.compute_final_main_children(acc_init)
    }

    pub fn compute_fold_outputs(&self, acc_init: &[MeInstance<Cmt, F, K>]) -> ShardFoldOutputs<Cmt, F, K> {
        let main = if self.steps.is_empty() {
            acc_init.to_vec()
        } else {
            self.steps
                .last()
                .expect("non-empty")
                .fold
                .dec_children
                .clone()
        };

        let mut val = Vec::new();
        for step in &self.steps {
            for p in &step.val_fold {
                val.extend_from_slice(&p.dec_children);
            }
            for p in &step.wb_fold {
                val.extend_from_slice(&p.dec_children);
            }
            for p in &step.wp_fold {
                val.extend_from_slice(&p.dec_children);
            }
        }

        ShardFoldOutputs {
            obligations: ShardObligations { main, val },
        }
    }
}
