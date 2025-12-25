use crate::PiCcsError;
use neo_ccs::poly::SparsePoly;
use neo_ccs::{CcsStructure, Mat};
use neo_math::{F, K};
use neo_memory::ajtai::decode_vector as ajtai_decode_vector;
use neo_memory::witness::{LutInstance, MemInstance, StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

#[derive(Clone, Debug)]
pub(crate) struct CpuBusSpec {
    pub chunk_size: usize,
    pub m_in: usize,
    pub bus_cols: usize,
    pub bus_base: usize,
}

impl CpuBusSpec {
    pub fn bus_cell_index(&self, col_id: usize, step_idx: usize) -> usize {
        self.bus_base + col_id * self.chunk_size + step_idx
    }

    pub fn time_row_index(&self, step_idx: usize) -> usize {
        self.m_in + step_idx
    }
}

pub(crate) fn shout_bus_cols<Cmt: Clone>(inst: &LutInstance<Cmt, F>) -> usize {
    (inst.d * inst.ell) + 2
}

pub(crate) fn twist_bus_cols<Cmt: Clone>(inst: &MemInstance<Cmt, F>) -> usize {
    (2 * inst.d * inst.ell) + 5
}

pub(crate) fn bus_cols_for_step_instance<Cmt: Clone, KK: Clone>(step: &StepInstanceBundle<Cmt, F, KK>) -> usize {
    let mut total = 0usize;
    for inst in &step.lut_insts {
        total += shout_bus_cols(inst);
    }
    for inst in &step.mem_insts {
        total += twist_bus_cols(inst);
    }
    total
}

pub(crate) fn bus_cols_for_step_witness<Cmt: Clone, KK: Clone>(step: &StepWitnessBundle<Cmt, F, KK>) -> usize {
    let mut total = 0usize;
    for (inst, _) in &step.lut_instances {
        total += shout_bus_cols(inst);
    }
    for (inst, _) in &step.mem_instances {
        total += twist_bus_cols(inst);
    }
    total
}

pub(crate) fn infer_chunk_size_from_witness_steps<Cmt: Clone, KK: Clone>(
    steps: &[StepWitnessBundle<Cmt, F, KK>],
) -> Result<usize, PiCcsError> {
    let mut max_steps = 0usize;
    for step in steps {
        for (inst, _) in &step.lut_instances {
            max_steps = max_steps.max(inst.steps);
        }
        for (inst, _) in &step.mem_instances {
            max_steps = max_steps.max(inst.steps);
        }
    }
    if max_steps == 0 {
        return Err(PiCcsError::InvalidInput(
            "cannot infer chunk_size (no mem/lut instances present)".into(),
        ));
    }
    Ok(max_steps)
}

pub(crate) fn infer_chunk_size_from_instance_steps<Cmt: Clone, KK: Clone>(
    steps: &[StepInstanceBundle<Cmt, F, KK>],
) -> Result<usize, PiCcsError> {
    let mut max_steps = 0usize;
    for step in steps {
        for inst in &step.lut_insts {
            max_steps = max_steps.max(inst.steps);
        }
        for inst in &step.mem_insts {
            max_steps = max_steps.max(inst.steps);
        }
    }
    if max_steps == 0 {
        return Err(PiCcsError::InvalidInput(
            "cannot infer chunk_size (no mem/lut instances present)".into(),
        ));
    }
    Ok(max_steps)
}

pub(crate) fn infer_cpu_bus_spec_for_witness_steps<Cmt: Clone, KK: Clone>(
    s: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, KK>],
) -> Result<CpuBusSpec, PiCcsError> {
    if steps.is_empty() {
        return Err(PiCcsError::InvalidInput("no steps".into()));
    }
    let m_in = steps[0].mcs.0.m_in;
    for (i, step) in steps.iter().enumerate() {
        if step.mcs.0.m_in != m_in {
            return Err(PiCcsError::InvalidInput(format!(
                "m_in mismatch across steps (step 0 has {m_in}, step {i} has {})",
                step.mcs.0.m_in
            )));
        }
    }

    let bus_cols = bus_cols_for_step_witness(&steps[0]);
    for (i, step) in steps.iter().enumerate().skip(1) {
        let cur = bus_cols_for_step_witness(step);
        if cur != bus_cols {
            return Err(PiCcsError::InvalidInput(format!(
                "bus column count mismatch across steps (step 0 has {bus_cols}, step {i} has {cur})"
            )));
        }
    }

    if bus_cols == 0 {
        return Ok(CpuBusSpec {
            chunk_size: 0,
            m_in,
            bus_cols,
            bus_base: s.m,
        });
    }

    let chunk_size = infer_chunk_size_from_witness_steps(steps)?;
    let bus_region_len = bus_cols
        .checked_mul(chunk_size)
        .ok_or_else(|| PiCcsError::InvalidInput("bus region length overflow".into()))?;
    if bus_region_len > s.m {
        return Err(PiCcsError::InvalidInput(format!(
            "bus region too large: bus_cols({bus_cols}) * chunk_size({chunk_size}) = {bus_region_len} > m({})",
            s.m
        )));
    }
    if m_in
        .checked_add(chunk_size)
        .ok_or_else(|| PiCcsError::InvalidInput("m_in + chunk_size overflow".into()))?
        > s.n
    {
        return Err(PiCcsError::InvalidInput(format!(
            "bus time rows out of range: m_in({m_in}) + chunk_size({chunk_size}) > n({})",
            s.n
        )));
    }

    Ok(CpuBusSpec {
        chunk_size,
        m_in,
        bus_cols,
        bus_base: s.m - bus_region_len,
    })
}

pub(crate) fn infer_cpu_bus_spec_for_instance_steps<Cmt: Clone, KK: Clone>(
    s: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, KK>],
) -> Result<CpuBusSpec, PiCcsError> {
    if steps.is_empty() {
        return Err(PiCcsError::InvalidInput("no steps".into()));
    }
    let m_in = steps[0].mcs_inst.m_in;
    for (i, step) in steps.iter().enumerate() {
        if step.mcs_inst.m_in != m_in {
            return Err(PiCcsError::InvalidInput(format!(
                "m_in mismatch across steps (step 0 has {m_in}, step {i} has {})",
                step.mcs_inst.m_in
            )));
        }
    }

    let bus_cols = bus_cols_for_step_instance(&steps[0]);
    for (i, step) in steps.iter().enumerate().skip(1) {
        let cur = bus_cols_for_step_instance(step);
        if cur != bus_cols {
            return Err(PiCcsError::InvalidInput(format!(
                "bus column count mismatch across steps (step 0 has {bus_cols}, step {i} has {cur})"
            )));
        }
    }

    if bus_cols == 0 {
        return Ok(CpuBusSpec {
            chunk_size: 0,
            m_in,
            bus_cols,
            bus_base: s.m,
        });
    }

    let chunk_size = infer_chunk_size_from_instance_steps(steps)?;
    let bus_region_len = bus_cols
        .checked_mul(chunk_size)
        .ok_or_else(|| PiCcsError::InvalidInput("bus region length overflow".into()))?;
    if bus_region_len > s.m {
        return Err(PiCcsError::InvalidInput(format!(
            "bus region too large: bus_cols({bus_cols}) * chunk_size({chunk_size}) = {bus_region_len} > m({})",
            s.m
        )));
    }
    if m_in
        .checked_add(chunk_size)
        .ok_or_else(|| PiCcsError::InvalidInput("m_in + chunk_size overflow".into()))?
        > s.n
    {
        return Err(PiCcsError::InvalidInput(format!(
            "bus time rows out of range: m_in({m_in}) + chunk_size({chunk_size}) > n({})",
            s.n
        )));
    }

    Ok(CpuBusSpec {
        chunk_size,
        m_in,
        bus_cols,
        bus_base: s.m - bus_region_len,
    })
}

pub(crate) fn extend_ccs_with_cpu_bus_copyouts(s: &CcsStructure<F>, bus: &CpuBusSpec) -> Result<CcsStructure<F>, PiCcsError> {
    if bus.bus_cols == 0 {
        return Ok(s.clone());
    }
    if s.n != s.m {
        return Err(PiCcsError::InvalidInput(format!(
            "shared-bus requires square CCS (n==m) for identity-first ME semantics, got {}×{}",
            s.n, s.m
        )));
    }

    let mut matrices = s.matrices.clone();
    let f: SparsePoly<F> = s.f.append_zero_vars(bus.bus_cols);

    for col_id in 0..bus.bus_cols {
        let mut mat = Mat::zero(s.n, s.m, F::ZERO);
        for j in 0..bus.chunk_size {
            let row = bus.time_row_index(j);
            let col = bus.bus_cell_index(col_id, j);
            if row >= s.n || col >= s.m {
                return Err(PiCcsError::InvalidInput(format!(
                    "bus copy-out index out of range (row={row}, col={col}, n={}, m={})",
                    s.n, s.m
                )));
            }
            mat.set(row, col, F::ONE);
        }
        matrices.push(mat);
    }

    CcsStructure::new(matrices, f).map_err(|e| PiCcsError::InvalidInput(format!("invalid CCS after bus extension: {e:?}")))
}

pub(crate) fn decode_cpu_z_to_k(params: &NeoParams, Z: &Mat<F>) -> Vec<K> {
    ajtai_decode_vector(params, Z).into_iter().map(Into::into).collect()
}

pub(crate) fn build_time_vec_from_bus_col(
    z: &[K],
    bus: &CpuBusSpec,
    col_id: usize,
    steps_len: usize,
    pow2_cycle: usize,
) -> Result<Vec<K>, PiCcsError> {
    if col_id >= bus.bus_cols {
        return Err(PiCcsError::InvalidInput(format!(
            "bus col_id out of range: {col_id} >= {}",
            bus.bus_cols
        )));
    }
    if steps_len > bus.chunk_size {
        return Err(PiCcsError::InvalidInput(format!(
            "steps_len({steps_len}) > bus.chunk_size({})",
            bus.chunk_size
        )));
    }
    let mut out = vec![K::ZERO; pow2_cycle];
    for j in 0..bus.chunk_size {
        let t = bus.time_row_index(j);
        if t >= out.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "bus time index out of range: t={t} >= pow2_cycle={pow2_cycle}"
            )));
        }
        if j >= steps_len {
            continue;
        }
        let idx = bus.bus_cell_index(col_id, j);
        let v = z
            .get(idx)
            .copied()
            .ok_or_else(|| PiCcsError::InvalidInput(format!("CPU witness too short for bus idx={idx}")))?;
        out[t] = v;
    }
    Ok(out)
}
