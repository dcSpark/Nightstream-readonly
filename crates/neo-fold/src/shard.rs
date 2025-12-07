//! Shard-level folding: CPU (CCS) + Memory sidecar (Twist/Shout).
//!
//! Architecture (per integration-summary.md):
//!
//! At each folding step, aggregate:
//! 1. (k) running ME instances
//! 2. fresh ME instances from Π_CCS
//! 3. fresh ME instances from Π_Twist and Π_Shout
//! 4. fresh ME instances from Index→OneHot adapter
//!
//! Then: (all ME claims) → Π_RLC → Π_DEC → k_rho children
//!
//! CURRENT IMPLEMENTATION (simplified):
//! - CPU folding runs first (Π_CCS → Π_RLC → Π_DEC per step)
//! - Memory sidecar proved after CPU folding
//! - Final merge combines CPU output + memory ME via Π_RLC → Π_DEC
//!
//! TODO: True per-step integration requires per-step memory instances.

#![allow(non_snake_case)]

use crate::folding::{CommitMixers, FoldStep};
use crate::pi_ccs::{self as ccs, FoldingMode};
use crate::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat, MeInstance};
use neo_math::{from_complex, D, F, K};
use neo_memory::witness::StepWitnessBundle;
use neo_memory::{shout, twist};
use neo_params::NeoParams;
use neo_reductions::engines::utils;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

// Type aliases
type TwistProofK = neo_memory::twist::TwistProof<K>;
type ShoutProofK = neo_memory::shout::ShoutProof<K>;

// ============================================================================
// Proof Structures
// ============================================================================

#[derive(Clone, Debug)]
pub enum MemOrLutProof {
    Twist(TwistProofK),
    Shout(ShoutProofK),
}

#[derive(Clone, Debug)]
pub struct MemSidecarProof<C, FF, KK> {
    pub me_claims: Vec<MeInstance<C, FF, KK>>,
    pub proofs: Vec<MemOrLutProof>,
}

#[derive(Clone, Debug)]
pub struct StepProof {
    pub fold: FoldStep,
    pub mem: MemSidecarProof<Cmt, F, K>,
}

#[derive(Clone, Debug)]
pub struct ShardProof {
    pub steps: Vec<StepProof>,
}

impl ShardProof {
    pub fn compute_final_children(&self, acc_init: &[MeInstance<Cmt, F, K>]) -> Vec<MeInstance<Cmt, F, K>> {
        if self.steps.is_empty() {
            return acc_init.to_vec();
        }
        self.steps.last().unwrap().fold.dec_children.clone()
    }
}

// ============================================================================
// Utilities
// ============================================================================

pub fn pad_witness_matrix(mat: &Mat<F>, target_cols: usize) -> Result<Mat<F>, PiCcsError> {
    if mat.cols() > target_cols {
        return Err(PiCcsError::InvalidInput(format!(
            "Witness matrix has {} cols, exceeds target {}",
            mat.cols(), target_cols
        )));
    }
    if mat.cols() == target_cols {
        return Ok(mat.clone());
    }
    let mut out = Mat::zero(mat.rows(), target_cols, F::ZERO);
    for r in 0..mat.rows() {
        for c in 0..mat.cols() {
            out[(r, c)] = mat[(r, c)];
        }
    }
    Ok(out)
}

pub fn absorb_step_memory_commitments(
    tr: &mut Poseidon2Transcript,
    step: &StepWitnessBundle<Cmt, F, K>,
) {
    tr.append_message(b"step/absorb_memory_start", &[]);
    tr.append_message(b"step/lut_count", &(step.lut_instances.len() as u64).to_le_bytes());
    for (inst, _) in &step.lut_instances {
        shout::absorb_commitments(tr, inst);
    }
    tr.append_message(b"step/mem_count", &(step.mem_instances.len() as u64).to_le_bytes());
    for (inst, _) in &step.mem_instances {
        twist::absorb_commitments(tr, inst);
    }
    tr.append_message(b"step/absorb_memory_done", &[]);
}

fn total_addr_bits_from_public(
    lut_instances: &[neo_memory::witness::LutInstance<Cmt, F>],
    mem_instances: &[neo_memory::witness::MemInstance<Cmt, F>],
) -> Result<Option<usize>, PiCcsError> {
    let first = lut_instances
        .first()
        .map(|inst| inst.d * inst.ell)
        .or_else(|| mem_instances.first().map(|inst| inst.d * inst.ell));

    if let Some(expected) = first {
        for inst in lut_instances {
            let bits = inst.d * inst.ell;
            if bits != expected {
                return Err(PiCcsError::InvalidInput(format!(
                    "Inconsistent total_addr_bits across public lookups (saw {}, expected {})",
                    bits, expected
                )));
            }
        }
        for inst in mem_instances {
            let bits = inst.d * inst.ell;
            if bits != expected {
                return Err(PiCcsError::InvalidInput(format!(
                    "Inconsistent total_addr_bits across public memories (saw {}, expected {})",
                    bits, expected
                )));
            }
        }
    }

    Ok(first)
}

fn sample_shared_addr(tr: &mut Poseidon2Transcript, len: usize) -> Vec<K> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        tr.append_message(b"shard/r_addr_shared_idx", &(i as u64).to_le_bytes());
        let c0 = tr.challenge_field(b"shard/r_addr_shared/0");
        let c1 = tr.challenge_field(b"shard/r_addr_shared/1");
        out.push(from_complex(c0, c1));
    }
    out
}

pub fn normalize_me_claims(
    me_claims: &mut [MeInstance<Cmt, F, K>],
    ell_n: usize,
    ell_d: usize,
    t: usize,
) -> Result<(), PiCcsError> {
    let y_pad = 1usize << ell_d;
    for (i, me) in me_claims.iter_mut().enumerate() {
        if me.r.len() != ell_n {
            return Err(PiCcsError::InvalidInput(format!(
                "ME[{}] r.len()={}, expected ell_n={}", i, me.r.len(), ell_n
            )));
        }
        if me.y.len() > t {
            return Err(PiCcsError::InvalidInput(format!(
                "ME[{}] y.len()={}, expected <= t={}", i, me.y.len(), t
            )));
        }
        for (j, row) in me.y.iter_mut().enumerate() {
            if row.len() > y_pad {
                return Err(PiCcsError::InvalidInput(format!(
                    "ME[{}] y[{}].len()={}, expected <= {}", i, j, row.len(), y_pad
                )));
            }
            row.resize(y_pad, K::ZERO);
        }
        me.y.resize_with(t, || vec![K::ZERO; y_pad]);
        if me.y_scalars.len() > t {
            return Err(PiCcsError::InvalidInput(format!(
                "ME[{}] y_scalars.len()={}, expected <= t={}", i, me.y_scalars.len(), t
            )));
        }
        me.y_scalars.resize(t, K::ZERO);
    }
    Ok(())
}

fn validate_me_batch_invariants(
    batch: &[MeInstance<Cmt, F, K>],
    context: &str,
) -> Result<(), PiCcsError> {
    if batch.is_empty() {
        return Ok(());
    }
    let me0 = &batch[0];
    let r0 = &me0.r;
    let m_in0 = me0.m_in;
    let y_len0 = me0.y.len();
    let y_row_len0 = me0.y.first().map(|r| r.len()).unwrap_or(0);
    let y_scalars_len0 = me0.y_scalars.len();

    if me0.X.rows() != D {
        return Err(PiCcsError::ProtocolError(format!(
            "{}: ME claim 0 has X.rows()={}, expected D={}", context, me0.X.rows(), D
        )));
    }
    if me0.X.cols() != m_in0 {
        return Err(PiCcsError::ProtocolError(format!(
            "{}: ME claim 0 has X.cols()={}, expected m_in={}", context, me0.X.cols(), m_in0
        )));
    }

    for (i, me) in batch.iter().enumerate().skip(1) {
        if me.r != *r0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has different r than claim 0 (r-alignment required for RLC)", context, i
            )));
        }
        if me.m_in != m_in0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has m_in={}, expected {}", context, i, me.m_in, m_in0
            )));
        }
        if me.X.rows() != D || me.X.cols() != m_in0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has X shape {}x{}, expected {}x{}",
                context, i, me.X.rows(), me.X.cols(), D, m_in0
            )));
        }
        if me.y.len() != y_len0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has y.len()={}, expected {}", context, i, me.y.len(), y_len0
            )));
        }
        for (j, row) in me.y.iter().enumerate() {
            if row.len() != y_row_len0 {
                return Err(PiCcsError::ProtocolError(format!(
                    "{}: ME claim {} has y[{}].len()={}, expected {}",
                    context, i, j, row.len(), y_row_len0
                )));
            }
        }
        if me.y_scalars.len() != y_scalars_len0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has y_scalars.len()={}, expected {}",
                context, i, me.y_scalars.len(), y_scalars_len0
            )));
        }
    }
    Ok(())
}

// ============================================================================
// Shard Proving
// ============================================================================

pub fn fold_shard_prove<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<ShardProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt>,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let s = s_me
        .ensure_identity_first()
        .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;
    let utils::Dims { ell_d, ell_n, .. } = utils::build_dims_and_policy(params, &s)?;
    let k_dec = params.k_rho as usize;
    let ring = ccs::RotRing::goldilocks();

    // Initialize accumulator
    let mut accumulator = acc_init.to_vec();
    let mut accumulator_wit = acc_wit_init.to_vec();

    let mut step_proofs = Vec::with_capacity(steps.len());

    for (idx, step) in steps.iter().enumerate() {
        absorb_step_memory_commitments(tr, step);

        let (mcs_inst, mcs_wit) = &step.mcs;

        // k = accumulator.len() + 1
        let k = accumulator.len() + 1;

        // Π_CCS
        let (ccs_out, ccs_proof) = ccs::prove(
            mode.clone(),
            tr,
            params,
            &s,
            core::slice::from_ref(mcs_inst),
            core::slice::from_ref(mcs_wit),
            &accumulator,
            &accumulator_wit,
            l,
        )?;

        if ccs_out.len() != k {
            return Err(PiCcsError::ProtocolError(format!(
                "Π_CCS returned {} outputs; expected k={k}",
                ccs_out.len()
            )));
        }

        // Witnesses for CCS outputs: [Z_mcs, Z_seed...]
        let mut outs_Z: Vec<Mat<F>> = Vec::with_capacity(k);
        outs_Z.push(mcs_wit.Z.clone());
        outs_Z.extend(accumulator_wit.iter().cloned());

        // Memory sidecar for this step, r-aligned to CCS
        let canonical_r_cycle = ccs_out.first().map(|me| me.r.clone());
        let canonical_r_ref = canonical_r_cycle.as_deref();

        let total_addr_bits = total_addr_bits_from_public(
            &step.lut_instances.iter().map(|(i, _)| i.clone()).collect::<Vec<_>>(),
            &step.mem_instances.iter().map(|(i, _)| i.clone()).collect::<Vec<_>>(),
        )?;
        let shared_r_addr: Option<Vec<K>> = total_addr_bits.map(|len| sample_shared_addr(tr, len));
        let shared_r_addr_ref: Option<&[K]> = shared_r_addr.as_deref();

        let mut me_claims = Vec::new();
        let mut me_wits = Vec::new();
        let mut proofs = Vec::new();

        for (lut_inst, lut_wit) in &step.lut_instances {
            let (me, w, proof) = shout::prove(
                mode.clone(),
                tr,
                params,
                lut_inst,
                lut_wit,
                l,
                ell_n,
                mcs_inst.m_in,
                canonical_r_ref,
                shared_r_addr_ref,
            )?;
            me_claims.extend(me);
            me_wits.extend(w);
            proofs.push(MemOrLutProof::Shout(proof));
        }

        for (mem_inst, mem_wit) in &step.mem_instances {
            let (me, w, proof) = twist::prove(
                mode.clone(),
                tr,
                params,
                mem_inst,
                mem_wit,
                l,
                ell_n,
                mcs_inst.m_in,
                canonical_r_ref,
                shared_r_addr_ref,
            )?;
            me_claims.extend(me);
            me_wits.extend(w);
            proofs.push(MemOrLutProof::Twist(proof));
        }

        normalize_me_claims(&mut me_claims, ell_n, ell_d, s.t())?;

        // Build RLC inputs: CCS outputs + memory ME
        let mut rlc_inputs = ccs_out.clone();
        rlc_inputs.extend(me_claims.clone());

        let mut rlc_wits = outs_Z.clone();
        for w in me_wits.iter() {
            rlc_wits.push(pad_witness_matrix(w, s.m)?);
        }

        // RLC
        let rhos = ccs::sample_rot_rhos_n(tr, params, &ring, rlc_inputs.len())?;
        let (parent_pub, Z_mix) = ccs::rlc_with_commit(
            mode.clone(),
            &s,
            params,
            &rhos,
            &rlc_inputs,
            &rlc_wits,
            ell_d,
            mixers.mix_rhos_commits,
        );

        // DEC
        let Z_split = ccs::split_b_matrix_k(&Z_mix, k_dec, params.b)?;
        let child_cs: Vec<Cmt> = Z_split.iter().map(|Zi| l.commit(Zi)).collect();
        let (children, ok_y, ok_X, ok_c) = ccs::dec_children_with_commit(
            mode.clone(),
            &s,
            params,
            &parent_pub,
            &Z_split,
            ell_d,
            &child_cs,
            mixers.combine_b_pows,
        );
        if !(ok_y && ok_X && ok_c) {
            return Err(PiCcsError::ProtocolError(format!(
                "DEC public check failed at step {} (y={}, X={}, c={})",
                idx, ok_y, ok_X, ok_c
            )));
        }

        accumulator = children.clone();
        accumulator_wit = Z_split;

        step_proofs.push(StepProof {
            fold: FoldStep {
                ccs_out,
                ccs_proof,
                rlc_rhos: rhos,
                rlc_parent: parent_pub,
                dec_children: children,
            },
            mem: MemSidecarProof {
                me_claims: me_claims.clone(),
                proofs,
            },
        });

        tr.append_message(b"fold/step_done", &(idx as u64).to_le_bytes());
    }

    Ok(ShardProof { steps: step_proofs })
}

// ============================================================================
// Shard Verification
// ============================================================================

pub fn fold_shard_verify<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<(), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt>,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let s = s_me
        .ensure_identity_first()
        .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;
    let utils::Dims { ell_d, ell_n, .. } = utils::build_dims_and_policy(params, &s)?;
    let ring = ccs::RotRing::goldilocks();

    if steps.len() != proof.steps.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "step count mismatch: public {} vs proof {}",
            steps.len(),
            proof.steps.len()
        )));
    }

    let mut accumulator = acc_init.to_vec();

    for (idx, (step, step_proof)) in steps.iter().zip(proof.steps.iter()).enumerate() {
        absorb_step_memory_commitments(tr, step);

        // Verify CCS
        let ok_ccs = ccs::verify(
            mode.clone(),
            tr,
            params,
            &s,
            core::slice::from_ref(&step.mcs.0),
            &accumulator,
            &step_proof.fold.ccs_out,
            &step_proof.fold.ccs_proof,
        )?;
        if !ok_ccs {
            return Err(PiCcsError::SumcheckError("Π_CCS verification failed".into()));
        }

        let observed_digest = tr.digest32();
        if observed_digest != step_proof.fold.ccs_proof.header_digest.as_slice() {
            return Err(PiCcsError::ProtocolError("Π_CCS header digest mismatch".into()));
        }

        let total_addr_bits = total_addr_bits_from_public(
            &step.lut_instances.iter().map(|(i, _)| i.clone()).collect::<Vec<_>>(),
            &step.mem_instances.iter().map(|(i, _)| i.clone()).collect::<Vec<_>>(),
        )?;
        let shared_r_addr: Option<Vec<K>> = total_addr_bits.map(|len| sample_shared_addr(tr, len));
        let shared_r_addr_ref: Option<&[K]> = shared_r_addr.as_deref();

        let canonical_r_cycle = step_proof.fold.ccs_out.first().map(|me| me.r.clone());
        let canonical_r_ref: Option<&[K]> = canonical_r_cycle.as_deref();

        // Verify mem proofs and collect ME claims
        let proofs_mem = &step_proof.mem.proofs;
        let mut me_claim_offset = 0;
        let mut collected_me = Vec::new();

        let expected_proofs = step.lut_instances.len() + step.mem_instances.len();
        if proofs_mem.len() != expected_proofs {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: mem proof count mismatch (expected {}, got {})",
                idx, expected_proofs, proofs_mem.len()
            )));
        }

        for (proof_idx, (inst, _)) in step.lut_instances.iter().enumerate() {
            let p = &proofs_mem[proof_idx];
            let shout_proof = match p {
                MemOrLutProof::Shout(proof) => proof,
                _ => return Err(PiCcsError::InvalidInput("expected Shout proof".into())),
            };
            let total_addr_bits = inst.d * inst.ell;
            let shout_me_count = total_addr_bits + 2;
            let me_slice =
                step_proof.mem.me_claims.get(me_claim_offset..me_claim_offset + shout_me_count).ok_or_else(|| {
                    PiCcsError::InvalidInput("Not enough ME claims for Shout".into())
                })?;
            me_claim_offset += shout_me_count;
            shout::verify(
                mode.clone(),
                tr,
                params,
                inst,
                shout_proof,
                me_slice,
                l,
                ell_n,
                canonical_r_ref,
                shared_r_addr_ref,
            )?;
            collected_me.extend_from_slice(me_slice);
        }

        let proof_mem_offset = step.lut_instances.len();
        for (i_mem, (inst, _)) in step.mem_instances.iter().enumerate() {
            let p = &proofs_mem[proof_mem_offset + i_mem];
            let twist_proof = match p {
                MemOrLutProof::Twist(proof) => proof,
                _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
            };
            let total_addr_bits = inst.d * inst.ell;
            let twist_me_count = 2 * total_addr_bits + 5;
            let me_slice =
                step_proof.mem.me_claims.get(me_claim_offset..me_claim_offset + twist_me_count).ok_or_else(|| {
                    PiCcsError::InvalidInput("Not enough ME claims for Twist".into())
                })?;
            me_claim_offset += twist_me_count;
            twist::verify(
                mode.clone(),
                tr,
                params,
                inst,
                twist_proof,
                me_slice,
                l,
                ell_n,
                canonical_r_ref,
                shared_r_addr_ref,
            )?;
            collected_me.extend_from_slice(me_slice);
        }

        if me_claim_offset != step_proof.mem.me_claims.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "Unused ME claims in step {}: consumed {}, proof has {}",
                idx,
                me_claim_offset,
                step_proof.mem.me_claims.len()
            )));
        }

        validate_me_batch_invariants(&collected_me, "verify step mem+ccs combined")?;

        // Sample rhos and check (after memory proofs, matching prover order)
        let k_rhos = step_proof.fold.rlc_rhos.len();
        let rhos_from_tr = ccs::sample_rot_rhos_n(tr, params, &ring, k_rhos)?;
        for (j, (sampled, stored)) in rhos_from_tr.iter().zip(step_proof.fold.rlc_rhos.iter()).enumerate() {
            if sampled.as_slice() != stored.as_slice() {
                return Err(PiCcsError::ProtocolError(format!(
                    "RLC ρ #{} mismatch: transcript vs proof",
                    j
                )));
            }
        }

        // Recompute RLC parent
        let mut rlc_inputs = step_proof.fold.ccs_out.clone();
        rlc_inputs.extend_from_slice(&collected_me);

        let parent_pub =
            ccs::rlc_public(&s, params, &step_proof.fold.rlc_rhos, &rlc_inputs, mixers.mix_rhos_commits, ell_d);

        if parent_pub.X.as_slice() != step_proof.fold.rlc_parent.X.as_slice() {
            return Err(PiCcsError::ProtocolError("RLC X mismatch".into()));
        }
        if parent_pub.c != step_proof.fold.rlc_parent.c {
            return Err(PiCcsError::ProtocolError("RLC commitment mismatch".into()));
        }
        if parent_pub.r != step_proof.fold.rlc_parent.r {
            return Err(PiCcsError::ProtocolError("RLC r mismatch".into()));
        }
        if parent_pub.y != step_proof.fold.rlc_parent.y {
            return Err(PiCcsError::ProtocolError("RLC y mismatch".into()));
        }
        if parent_pub.y_scalars != step_proof.fold.rlc_parent.y_scalars {
            return Err(PiCcsError::ProtocolError("RLC y_scalars mismatch".into()));
        }

        if !ccs::verify_dec_public(
            &s,
            params,
            &step_proof.fold.rlc_parent,
            &step_proof.fold.dec_children,
            mixers.combine_b_pows,
            ell_d,
        ) {
            return Err(PiCcsError::ProtocolError("DEC public check failed".into()));
        }

        accumulator = step_proof.fold.dec_children.clone();

        tr.append_message(b"fold/step_done", &(idx as u64).to_le_bytes());
    }

    Ok(())
}
