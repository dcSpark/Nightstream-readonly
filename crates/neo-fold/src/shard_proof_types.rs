use crate::folding::FoldStep;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{matrix::Mat, MeInstance};
use neo_math::{F, K};

pub type TwistProofK = neo_memory::twist::TwistProof<K>;
pub type ShoutProofK = neo_memory::shout::ShoutProof<K>;

#[derive(Clone, Debug)]
pub struct ShardFoldOutputs<C, FF, KK> {
    /// The shard's final main accumulator (the normal folding lane).
    pub final_main_acc: Vec<MeInstance<C, FF, KK>>,
    /// Additional ME instances that must be satisfied for Twist val-eval (the `r_val` lane).
    ///
    /// These are the Π_DEC children produced by each step's `val_fold` proof, concatenated
    /// in step order. They cannot be merged into the main accumulator because `r_val` differs
    /// from the shared `r_time`.
    pub val_lane_obligations: Vec<MeInstance<C, FF, KK>>,
}

#[derive(Clone, Debug)]
pub struct ShardFoldWitnesses<FF> {
    /// Witnesses for `ShardFoldOutputs::final_main_acc` (one per ME instance).
    pub final_main_wits: Vec<Mat<FF>>,
    /// Witnesses for `ShardFoldOutputs::val_lane_obligations` (one per ME instance).
    pub val_lane_wits: Vec<Mat<FF>>,
}

#[derive(Clone, Debug)]
pub enum MemOrLutProof {
    Twist(TwistProofK),
    Shout(ShoutProofK),
}

#[derive(Clone, Debug)]
pub struct MemSidecarProof<C, FF, KK> {
    /// Memory/LUT ME claims evaluated at the shared `r_time` point.
    pub me_claims_time: Vec<MeInstance<C, FF, KK>>,
    /// Additional ME claims evaluated at `r_val` (Twist val-eval terminal point).
    pub me_claims_val: Vec<MeInstance<C, FF, KK>>,
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
    /// Optional second folding lane for Twist val-eval ME claims at `r_val`.
    pub val_fold: Option<RlcDecProof>,
}

#[derive(Clone, Debug)]
pub struct ShardProof {
    pub steps: Vec<StepProof>,
}

impl ShardProof {
    /// Returns the final main accumulator only.
    ///
    /// If Twist val-eval is enabled, this does **not** include the additional `r_val`-lane
    /// ME obligations; use `compute_fold_outputs` to obtain the full set of obligations.
    pub fn compute_final_children(&self, acc_init: &[MeInstance<Cmt, F, K>]) -> Vec<MeInstance<Cmt, F, K>> {
        self.compute_fold_outputs(acc_init).final_main_acc
    }

    pub fn compute_fold_outputs(&self, acc_init: &[MeInstance<Cmt, F, K>]) -> ShardFoldOutputs<Cmt, F, K> {
        let final_main_acc = if self.steps.is_empty() {
            acc_init.to_vec()
        } else {
            self.steps
                .last()
                .expect("non-empty")
                .fold
                .dec_children
                .clone()
        };

        let mut val_lane_obligations = Vec::new();
        for step in &self.steps {
            if let Some(val_fold) = &step.val_fold {
                val_lane_obligations.extend_from_slice(&val_fold.dec_children);
            }
        }

        ShardFoldOutputs {
            final_main_acc,
            val_lane_obligations,
        }
    }
}
