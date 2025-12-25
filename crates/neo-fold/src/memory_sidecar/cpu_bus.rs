//! Shared CPU-bus geometry + ordering helpers (single source of truth).
//!
//! In CPU-linked / shared-bus mode, Twist/Shout access-row columns live inside the *CPU witness*
//! commitment namespace. We reserve a canonical "bus region" in the CPU witness `z` and expose
//! its per-cycle values to the memory sidecar via deterministic CCS copy-out matrices.
//!
//! Canonical bus column ordering (no decisions):
//! 1) All Shout instances in `step.lut_instances` order, each in `shout_layout()` order.
//! 2) All Twist instances in `step.mem_instances` order, each in `twist_layout()` order.
//!
//! Storage rule inside the CPU witness `z` (length = `m`):
//! - Let `chunk_size` be the stride used for packing (typically the maximum supported steps).
//! - Let `bus_cols_total` be the total number of bus columns across all instances.
//! - The bus region is the tail:
//!     `bus_base = m - bus_cols_total * chunk_size`
//! - For bus column `col_id` and step-in-chunk `t`:
//!     `z[bus_base + col_id*chunk_size + t]`
//!
//! Copy-out matrices then map those witness coordinates into the time rows `[m_in + t]`.

use crate::PiCcsError;
use neo_math::F;
use neo_memory::witness::{LutInstance, MemInstance, StepWitnessBundle};

#[derive(Clone, Debug)]
pub struct CpuBusLayout {
    pub m_in: usize,
    pub m: usize,
    pub chunk_size: usize,
    pub bus_cols_total: usize,
    pub bus_base: usize,
    /// Starting bus-column id per Shout instance (same order as `step.lut_instances`).
    pub lut_offsets: Vec<usize>,
    /// Starting bus-column id per Twist instance (same order as `step.mem_instances`).
    pub mem_offsets: Vec<usize>,
}

impl CpuBusLayout {
    #[inline]
    pub fn bus_cell_index(&self, col_id: usize, step_in_chunk: usize) -> usize {
        self.bus_base + col_id * self.chunk_size + step_in_chunk
    }

    #[inline]
    pub fn time_row_index(&self, step_in_chunk: usize) -> usize {
        self.m_in + step_in_chunk
    }
}

pub fn compute_bus_col_offsets_for_instances<'a, Cmt: 'a, Ff: 'a>(
    lut_insts: impl IntoIterator<Item = &'a LutInstance<Cmt, Ff>>,
    mem_insts: impl IntoIterator<Item = &'a MemInstance<Cmt, Ff>>,
) -> Result<(Vec<usize>, Vec<usize>, usize), PiCcsError> {
    let lut_insts: Vec<&'a LutInstance<Cmt, Ff>> = lut_insts.into_iter().collect();
    let mem_insts: Vec<&'a MemInstance<Cmt, Ff>> = mem_insts.into_iter().collect();

    let mut lut_offsets = Vec::with_capacity(lut_insts.len());
    let mut mem_offsets = Vec::with_capacity(mem_insts.len());
    let mut cursor = 0usize;

    for inst in lut_insts {
        lut_offsets.push(cursor);
        cursor = cursor
            .checked_add(inst.shout_layout().expected_len())
            .ok_or_else(|| PiCcsError::InvalidInput("bus_cols_total overflow".into()))?;
    }
    for inst in mem_insts {
        mem_offsets.push(cursor);
        cursor = cursor
            .checked_add(inst.twist_layout().expected_len())
            .ok_or_else(|| PiCcsError::InvalidInput("bus_cols_total overflow".into()))?;
    }

    Ok((lut_offsets, mem_offsets, cursor))
}

pub fn infer_cpu_bus_layout_for_step<Cmt, K>(
    step: &StepWitnessBundle<Cmt, F, K>,
) -> Result<Option<CpuBusLayout>, PiCcsError>
where
    Cmt: Clone,
    K: Clone,
{
    let m_in = step.mcs.0.m_in;
    if step.mcs.0.x.len() != m_in {
        return Err(PiCcsError::InvalidInput(format!(
            "CPU bus: step.mcs.x.len()={} != m_in={}",
            step.mcs.0.x.len(),
            m_in
        )));
    }
    let m = m_in
        .checked_add(step.mcs.1.w.len())
        .ok_or_else(|| PiCcsError::InvalidInput("CPU bus: witness length overflow".into()))?;

    if step.lut_instances.is_empty() && step.mem_instances.is_empty() {
        return Ok(None);
    }

    let chunk_size = step
        .lut_instances
        .iter()
        .map(|(inst, _)| inst.steps)
        .chain(step.mem_instances.iter().map(|(inst, _)| inst.steps))
        .max()
        .unwrap_or(0);

    if chunk_size == 0 {
        return Err(PiCcsError::InvalidInput(
            "CPU bus: instances present but chunk_size computed as 0".into(),
        ));
    }

    let (lut_offsets, mem_offsets, bus_cols_total) = compute_bus_col_offsets_for_instances(
        step.lut_instances.iter().map(|(inst, _)| inst),
        step.mem_instances.iter().map(|(inst, _)| inst),
    )?;

    let slots_total = bus_cols_total
        .checked_mul(chunk_size)
        .ok_or_else(|| PiCcsError::InvalidInput("CPU bus: slots_total overflow".into()))?;
    if slots_total > m {
        return Err(PiCcsError::InvalidInput(format!(
            "CPU bus: need {} bus slots but witness length m is {}",
            slots_total, m
        )));
    }
    let bus_base = m - slots_total;
    if bus_base < m_in {
        return Err(PiCcsError::InvalidInput(format!(
            "CPU bus: bus_base={} < m_in={} (bus region overlaps public inputs)",
            bus_base, m_in
        )));
    }

    Ok(Some(CpuBusLayout {
        m_in,
        m,
        chunk_size,
        bus_cols_total,
        bus_base,
        lut_offsets,
        mem_offsets,
    }))
}

pub fn read_z_at<Cmt, K>(step: &StepWitnessBundle<Cmt, F, K>, z_idx: usize) -> Result<F, PiCcsError> {
    let m_in = step.mcs.0.m_in;
    if z_idx < m_in {
        return step
            .mcs
            .0
            .x
            .get(z_idx)
            .copied()
            .ok_or_else(|| PiCcsError::InvalidInput("CPU bus: z index out of bounds in x".into()));
    }
    let w_idx = z_idx
        .checked_sub(m_in)
        .ok_or_else(|| PiCcsError::InvalidInput("CPU bus: z_idx < m_in underflow".into()))?;
    step.mcs
        .1
        .w
        .get(w_idx)
        .copied()
        .ok_or_else(|| PiCcsError::InvalidInput("CPU bus: z index out of bounds in w".into()))
}
