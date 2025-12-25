//! CPU-linked memory/lookup integration (Jolt-style shared polynomial namespace).
//!
//! This module provides a helper that:
//! - Appends deterministic "copy-out" matrices to the CCS structure so `Π_CCS` emits
//!   openings for Twist/Shout columns in `ccs_out[0].y_scalars`.
//! - Packs the per-step Twist/Shout column values into unused coordinates of the CPU witness `z`
//!   and recomputes the CPU commitment.
//! - Marks each `MemInstance`/`LutInstance` with `cpu_opening_base` so Route A verification uses
//!   CPU openings instead of separate memory commitments.

use crate::PiCcsError;
use crate::memory_sidecar::cpu_bus::compute_bus_col_offsets_for_instances;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat};
use neo_math::{F, K};
use neo_memory::ajtai::decode_vector as ajtai_decode_vector;
use neo_memory::encode::ajtai_encode_vector;
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

fn extend_ccs_with_cpu_bus_copyouts(
    mut ccs: CcsStructure<F>,
    m_in: usize,
    chunk_size: usize,
    bus_cols_total: usize,
) -> Result<CcsStructure<F>, PiCcsError> {
    if bus_cols_total == 0 {
        return Ok(ccs);
    }
    if chunk_size == 0 {
        return Err(PiCcsError::InvalidInput(
            "cpu-linked mode requires chunk_size > 0".into(),
        ));
    }
    if m_in
        .checked_add(chunk_size)
        .ok_or_else(|| PiCcsError::InvalidInput("m_in + chunk_size overflow".into()))?
        > ccs.n
    {
        return Err(PiCcsError::InvalidInput(format!(
            "cpu-linked mode requires m_in + chunk_size <= ccs.n, got m_in={} chunk_size={} ccs.n={}",
            m_in, chunk_size, ccs.n
        )));
    }

    let slots_total = bus_cols_total
        .checked_mul(chunk_size)
        .ok_or_else(|| PiCcsError::InvalidInput("slots_total overflow".into()))?;
    if slots_total > ccs.m {
        return Err(PiCcsError::InvalidInput(format!(
            "cpu-linked mode needs {} witness slots but CCS.m is {}",
            slots_total, ccs.m
        )));
    }
    let bus_base = ccs.m - slots_total;
    if bus_base < m_in {
        return Err(PiCcsError::InvalidInput(format!(
            "cpu-linked mode needs bus_base={} >= m_in={} (insufficient slack in witness vector)",
            bus_base, m_in
        )));
    }

    // Soundness guard: CPU-linked mode only eliminates CPU↔memory forking if the *CPU CCS constraints*
    // actually consume/constrain the bus region witness coordinates. Otherwise the bus becomes a
    // "dead witness" slice: Twist/Shout can be satisfied using one set of access rows while the CPU
    // step constraints are satisfied using an unrelated set of values under the same commitment.
    //
    // Require that at least one matrix referenced by the CCS polynomial touches witness columns
    // in the bus region `[bus_base .. ccs.m)`.
    {
        let old_t = ccs.t();
        let mut poly_uses = vec![false; old_t];
        for term in ccs.f.terms() {
            if term.exps.len() != old_t {
                return Err(PiCcsError::InvalidInput(
                    "CCS polynomial exponent vector length drift".into(),
                ));
            }
            for (i, &exp) in term.exps.iter().enumerate() {
                if exp != 0 {
                    poly_uses[i] = true;
                }
            }
        }
        if !poly_uses.iter().any(|&b| b) {
            return Err(PiCcsError::InvalidInput(
                "cpu-linked mode requires a non-vacuous CPU CCS polynomial (f has no terms)".into(),
            ));
        }

        let mut cpu_consumes_bus = false;
        for (i, used) in poly_uses.iter().enumerate() {
            if !used {
                continue;
            }
            let mat = ccs.matrices.get(i).ok_or_else(|| {
                PiCcsError::ProtocolError("cpu-link: matrix index out of bounds (internal error)".into())
            })?;
            if bus_base >= mat.cols() {
                continue;
            }
            for r in 0..mat.rows() {
                if mat.row(r)[bus_base..].iter().any(|v| *v != F::ZERO) {
                    cpu_consumes_bus = true;
                    break;
                }
            }
            if cpu_consumes_bus {
                break;
            }
        }
        if !cpu_consumes_bus {
            return Err(PiCcsError::InvalidInput(format!(
                "cpu-linked mode requires CPU constraints to consume the CPU bus region (witness cols {}..{}); \
none of the matrices referenced by the CCS polynomial touch that region. \
See docs/neo-with-twist-and-shout/mem-cpu-linkage.md.",
                bus_base, ccs.m
            )));
        }
    }

    let old_t = ccs.t();
    let mut copyout_mats: Vec<Mat<F>> = Vec::with_capacity(bus_cols_total);
    for col_id in 0..bus_cols_total {
        let start = bus_base + col_id * chunk_size;
        let mut m = Mat::zero(ccs.n, ccs.m, F::ZERO);
        for t in 0..chunk_size {
            let row = m_in + t;
            let col = start + t;
            m[(row, col)] = F::ONE;
        }
        copyout_mats.push(m);
    }
    debug_assert_eq!(copyout_mats.len(), bus_cols_total);

    ccs.matrices.extend(copyout_mats);

    // Extend the CCS polynomial arity by dummy vars at the end (all new exponents are 0).
    let new_t = ccs.t();
    let mut new_terms = ccs.f.terms().to_vec();
    for term in &mut new_terms {
        if term.exps.len() != old_t {
            return Err(PiCcsError::InvalidInput(
                "CCS polynomial exponent vector length drift".into(),
            ));
        }
        term.exps.resize(new_t, 0);
    }
    ccs.f = neo_ccs::poly::SparsePoly::new(new_t, new_terms);

    // Re-validate CCS structure after mutation.
    CcsStructure::new(ccs.matrices, ccs.f)
        .map_err(|e| PiCcsError::InvalidInput(format!("invalid CCS after cpu bus extension: {e:?}")))
}

/// Convert a set of steps to CPU-linked Twist/Shout mode by:
/// - packing all Twist/Shout columns into the tail of the CPU witness `z`,
/// - adding copy-out matrices to the CCS to expose those columns as `y_scalars`,
/// - setting `cpu_opening_base` for each instance and clearing per-column commitments.
///
/// Deterministic packing rule (no choices):
/// - Let `max_steps = max(inst.steps)` across all Twist/Shout instances.
/// - Let `cols_total = Σ(inst.twist_layout().expected_len()) + Σ(inst.shout_layout().expected_len())`
///   using the *first* step as the template.
/// - Reserve the tail region `z[pack_start .. ccs.m)` where `pack_start = ccs.m - cols_total*max_steps`.
/// - For each column (LUT instances first, then MEM instances; each in their existing `*_layout()` order),
///   store its per-step values in `z[start .. start+max_steps)`, and copy them into output rows
///   `[m_in .. m_in+max_steps)` via a dedicated copy-out CCS matrix.
///
/// Returns the augmented CCS structure (with copy-out matrices appended). Mutates `steps` in place.
pub fn make_steps_cpu_linked<L>(
    params: &NeoParams,
    l: &L,
    mut ccs: CcsStructure<F>,
    steps: &mut [StepWitnessBundle<Cmt, F, K>],
) -> Result<CcsStructure<F>, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt>,
{
    if steps.is_empty() {
        return Ok(ccs);
    }

    let m_in = steps[0].mcs.0.m_in;
    let template_lut_insts: Vec<_> = steps[0]
        .lut_instances
        .iter()
        .map(|(inst, _)| inst.clone())
        .collect();
    let template_mem_insts: Vec<_> = steps[0]
        .mem_instances
        .iter()
        .map(|(inst, _)| inst.clone())
        .collect();
    let lut_count = template_lut_insts.len();
    let mem_count = template_mem_insts.len();
    for (i, step) in steps.iter().enumerate() {
        if step.mcs.0.m_in != m_in {
            return Err(PiCcsError::InvalidInput(format!(
                "step {i}: m_in mismatch (got {}, expected {})",
                step.mcs.0.m_in, m_in
            )));
        }
        if step.lut_instances.len() != lut_count {
            return Err(PiCcsError::InvalidInput(format!(
                "step {i}: lut instance count mismatch (got {}, expected {})",
                step.lut_instances.len(),
                lut_count
            )));
        }
        if step.mem_instances.len() != mem_count {
            return Err(PiCcsError::InvalidInput(format!(
                "step {i}: mem instance count mismatch (got {}, expected {})",
                step.mem_instances.len(),
                mem_count
            )));
        }
    }

    let max_steps = steps
        .iter()
        .flat_map(|s| {
            s.lut_instances
                .iter()
                .map(|(inst, _)| inst.steps)
                .chain(s.mem_instances.iter().map(|(inst, _)| inst.steps))
        })
        .max()
        .unwrap_or(0);

    if max_steps == 0 {
        // No Twist/Shout instances.
        return Ok(ccs);
    }
    if m_in + max_steps > ccs.n {
        return Err(PiCcsError::InvalidInput(format!(
            "cpu-linked mode requires m_in + max_steps <= ccs.n, got m_in={} max_steps={} ccs.n={}",
            m_in, max_steps, ccs.n
        )));
    }

    let (lut_offsets, mem_offsets, cols_total) =
        compute_bus_col_offsets_for_instances(template_lut_insts.iter(), template_mem_insts.iter())?;

    let slots_total = cols_total
        .checked_mul(max_steps)
        .ok_or_else(|| PiCcsError::InvalidInput("slots_total overflow".into()))?;
    if slots_total > ccs.m {
        return Err(PiCcsError::InvalidInput(format!(
            "cpu-linked mode needs {} witness slots but CCS.m is {}",
            slots_total, ccs.m
        )));
    }
    let pack_start = ccs.m - slots_total;
    if pack_start < m_in {
        return Err(PiCcsError::InvalidInput(format!(
            "cpu-linked mode needs pack_start={} >= m_in={} (insufficient slack in witness vector)",
            pack_start, m_in
        )));
    }

    // Build copy-out matrices deterministically, and stamp cpu_opening_base for each instance.
    let old_t = ccs.t();

    // Copy-out matrices are laid out as: LUT instances then MEM instances, in `*_layout()` order.
    for (i_lut, _) in template_lut_insts.iter().enumerate() {
        let inst_off = *lut_offsets.get(i_lut).ok_or_else(|| {
            PiCcsError::ProtocolError("cpu-link: missing lut offset (internal error)".into())
        })?;
        let base = old_t
            .checked_add(inst_off)
            .ok_or_else(|| PiCcsError::InvalidInput("cpu_opening_base overflow".into()))?;
        // Stamp base into every step's corresponding instance.
        for step in steps.iter_mut() {
            step.lut_instances[i_lut].0.cpu_opening_base = Some(base);
            step.lut_instances[i_lut].0.comms.clear();
            // Shared-bus mode uses a fixed stride (`max_steps`) for packing/copy-outs.
            step.lut_instances[i_lut].0.steps = max_steps;
        }
    }
    for (i_mem, _) in template_mem_insts.iter().enumerate() {
        let inst_off = *mem_offsets.get(i_mem).ok_or_else(|| {
            PiCcsError::ProtocolError("cpu-link: missing mem offset (internal error)".into())
        })?;
        let base = old_t
            .checked_add(inst_off)
            .ok_or_else(|| PiCcsError::InvalidInput("cpu_opening_base overflow".into()))?;
        for step in steps.iter_mut() {
            step.mem_instances[i_mem].0.cpu_opening_base = Some(base);
            step.mem_instances[i_mem].0.comms.clear();
            step.mem_instances[i_mem].0.steps = max_steps;
        }
    }

    // Augment the CCS structure: append bus copy-out matrices and extend the polynomial arity.
    ccs = extend_ccs_with_cpu_bus_copyouts(ccs, m_in, max_steps, cols_total)?;

    // Pack witness values and recompute CPU commitments per step.
    for (step_idx, step) in steps.iter_mut().enumerate() {
        let mut col_i = 0usize;
        // LUT first
        for (lut_inst, lut_wit) in step.lut_instances.iter() {
            let cols = lut_inst.shout_layout().expected_len();
            if lut_wit.mats.len() != cols {
                return Err(PiCcsError::InvalidInput(format!(
                    "step {step_idx}: Shout mats.len()={} != expected {cols}",
                    lut_wit.mats.len()
                )));
            }
            for mat in lut_wit.mats.iter() {
                let decoded = ajtai_decode_vector(params, mat);
                for t in 0..max_steps {
                    let src_idx = m_in + t;
                    let val = decoded.get(src_idx).copied().unwrap_or(F::ZERO);
                    let dst = pack_start + col_i * max_steps + t;
                    let w_idx = dst - m_in;
                    let cur = step.mcs.1.w[w_idx];
                    if cur != F::ZERO && cur != val {
                        return Err(PiCcsError::InvalidInput(format!(
                            "step {step_idx}: CPU bus slot z[{dst}] already set (got {cur:?}, want {val:?})",
                        )));
                    }
                    step.mcs.1.w[w_idx] = val;
                }
                col_i += 1;
            }
        }
        // MEM
        for (mem_inst, mem_wit) in step.mem_instances.iter() {
            let cols = mem_inst.twist_layout().expected_len();
            if mem_wit.mats.len() != cols {
                return Err(PiCcsError::InvalidInput(format!(
                    "step {step_idx}: Twist mats.len()={} != expected {cols}",
                    mem_wit.mats.len()
                )));
            }
            for mat in mem_wit.mats.iter() {
                let decoded = ajtai_decode_vector(params, mat);
                for t in 0..max_steps {
                    let src_idx = m_in + t;
                    let val = decoded.get(src_idx).copied().unwrap_or(F::ZERO);
                    let dst = pack_start + col_i * max_steps + t;
                    let w_idx = dst - m_in;
                    let cur = step.mcs.1.w[w_idx];
                    if cur != F::ZERO && cur != val {
                        return Err(PiCcsError::InvalidInput(format!(
                            "step {step_idx}: CPU bus slot z[{dst}] already set (got {cur:?}, want {val:?})",
                        )));
                    }
                    step.mcs.1.w[w_idx] = val;
                }
                col_i += 1;
            }
        }
        if col_i != cols_total {
            return Err(PiCcsError::InvalidInput(format!(
                "step {step_idx}: packed columns drift (packed {col_i}, expected {cols_total})"
            )));
        }

        // Shared-bus invariant: after packing into the CPU witness, delete the independent
        // Twist/Shout witness namespace to prevent accidental future use.
        for (_, wit) in step.lut_instances.iter_mut() {
            wit.mats.clear();
        }
        for (_, wit) in step.mem_instances.iter_mut() {
            wit.mats.clear();
        }

        // Recompute Z and commitment.
        let mut z_full: Vec<F> = Vec::with_capacity(ccs.m);
        z_full.extend_from_slice(&step.mcs.0.x);
        z_full.extend_from_slice(&step.mcs.1.w);
        if z_full.len() != ccs.m {
            return Err(PiCcsError::InvalidInput(format!(
                "step {step_idx}: z length {} != CCS.m {}",
                z_full.len(),
                ccs.m
            )));
        }
        let Z = ajtai_encode_vector(params, &z_full);
        let c = l.commit(&Z);
        step.mcs.0.c = c;
        step.mcs.1.Z = Z;
    }

    // Re-validate CCS structure after mutation.
    CcsStructure::new(ccs.matrices, ccs.f).map_err(|e| PiCcsError::InvalidInput(format!("invalid CCS after cpu-link: {e:?}")))
}

/// Ensure the CCS structure passed to verification contains the deterministic bus copy-out matrices
/// required by CPU-linked (`cpu_opening_base`) memory/lookup instances.
///
/// This is needed when a verifier is handed the *base* CCS (without bus copy-outs) alongside
/// CPU-linked step instances. The `cpu_opening_base` indices implicitly commit to an `old_t`
/// (the base CCS matrix count) and to the canonical bus column ordering.
pub fn ensure_ccs_has_cpu_bus_copyouts_for_cpu_linked_steps<C, Kf>(
    ccs: &CcsStructure<F>,
    steps: &[StepInstanceBundle<C, F, Kf>],
) -> Result<CcsStructure<F>, PiCcsError>
where
    C: Clone,
    Kf: Clone,
{
    // Find a step that actually has Twist/Shout instances (CCS-only sessions don't need bus logic).
    let step0 = steps
        .iter()
        .find(|s| !s.lut_insts.is_empty() || !s.mem_insts.is_empty());
    let Some(step0) = step0 else {
        return Ok(ccs.clone());
    };

    let step0_has_some = step0
        .lut_insts
        .iter()
        .any(|inst| inst.cpu_opening_base.is_some())
        || step0.mem_insts.iter().any(|inst| inst.cpu_opening_base.is_some());
    let step0_has_none = step0
        .lut_insts
        .iter()
        .any(|inst| inst.cpu_opening_base.is_none())
        || step0.mem_insts.iter().any(|inst| inst.cpu_opening_base.is_none());
    if step0_has_some && step0_has_none {
        return Err(PiCcsError::InvalidInput(
            "mixed cpu_opening_base mode within step (some instances set it, others do not)".into(),
        ));
    }
    if !step0_has_some {
        // Legacy mode: no CPU-linking expected, so no CCS extension required.
        return Ok(ccs.clone());
    }

    // CPU-linked mode: require all steps with instances to be CPU-linked (no mixed-mode session).
    for (i, step) in steps.iter().enumerate() {
        if step.lut_insts.is_empty() && step.mem_insts.is_empty() {
            continue;
        }
        let has_some = step.lut_insts.iter().any(|inst| inst.cpu_opening_base.is_some())
            || step.mem_insts.iter().any(|inst| inst.cpu_opening_base.is_some());
        let has_none = step.lut_insts.iter().any(|inst| inst.cpu_opening_base.is_none())
            || step.mem_insts.iter().any(|inst| inst.cpu_opening_base.is_none());
        if has_some && has_none {
            return Err(PiCcsError::InvalidInput(format!(
                "step {i}: mixed cpu_opening_base mode within step"
            )));
        }
        if !has_some {
            return Err(PiCcsError::InvalidInput(format!(
                "step {i}: mixed cpu_opening_base mode across steps (expected CPU-linked)"
            )));
        }
    }

    // Derive canonical bus geometry from the template step instances.
    let (lut_offsets, mem_offsets, bus_cols_total) =
        compute_bus_col_offsets_for_instances(step0.lut_insts.iter(), step0.mem_insts.iter())?;
    let chunk_size = step0
        .lut_insts
        .iter()
        .map(|inst| inst.steps)
        .chain(step0.mem_insts.iter().map(|inst| inst.steps))
        .max()
        .unwrap_or(0);

    // Recover old_t from cpu_opening_base = old_t + inst_off and validate consistency.
    let mut old_t_opt: Option<usize> = None;
    for (i, inst) in step0.lut_insts.iter().enumerate() {
        let base = inst.cpu_opening_base.ok_or_else(|| {
            PiCcsError::InvalidInput("CPU-linked mode: missing cpu_opening_base for Shout".into())
        })?;
        let inst_off = *lut_offsets.get(i).ok_or_else(|| {
            PiCcsError::ProtocolError("CPU-linked mode: missing lut offset (internal error)".into())
        })?;
        let cand = base
            .checked_sub(inst_off)
            .ok_or_else(|| PiCcsError::InvalidInput("cpu_opening_base underflow".into()))?;
        match old_t_opt {
            None => old_t_opt = Some(cand),
            Some(prev) if prev == cand => {}
            Some(prev) => {
                return Err(PiCcsError::InvalidInput(format!(
                    "CPU-linked mode: inconsistent old_t inferred from cpu_opening_base (got {cand}, expected {prev})"
                )))
            }
        }
    }
    for (i, inst) in step0.mem_insts.iter().enumerate() {
        let base = inst.cpu_opening_base.ok_or_else(|| {
            PiCcsError::InvalidInput("CPU-linked mode: missing cpu_opening_base for Twist".into())
        })?;
        let inst_off = *mem_offsets.get(i).ok_or_else(|| {
            PiCcsError::ProtocolError("CPU-linked mode: missing mem offset (internal error)".into())
        })?;
        let cand = base
            .checked_sub(inst_off)
            .ok_or_else(|| PiCcsError::InvalidInput("cpu_opening_base underflow".into()))?;
        match old_t_opt {
            None => old_t_opt = Some(cand),
            Some(prev) if prev == cand => {}
            Some(prev) => {
                return Err(PiCcsError::InvalidInput(format!(
                    "CPU-linked mode: inconsistent old_t inferred from cpu_opening_base (got {cand}, expected {prev})"
                )))
            }
        }
    }
    let old_t = old_t_opt.ok_or_else(|| {
        PiCcsError::InvalidInput("CPU-linked mode: no instances available to infer old_t".into())
    })?;

    let want_extended_t = old_t
        .checked_add(bus_cols_total)
        .ok_or_else(|| PiCcsError::InvalidInput("old_t + bus_cols_total overflow".into()))?;

    if ccs.t() == want_extended_t {
        return Ok(ccs.clone());
    }
    if ccs.t() != old_t {
        return Err(PiCcsError::InvalidInput(format!(
            "CPU-linked mode: CCS.t={} does not match expected base old_t={} or extended t={}",
            ccs.t(),
            old_t,
            want_extended_t
        )));
    }

    let m_in = step0.mcs_inst.m_in;
    extend_ccs_with_cpu_bus_copyouts(ccs.clone(), m_in, chunk_size, bus_cols_total)
}
