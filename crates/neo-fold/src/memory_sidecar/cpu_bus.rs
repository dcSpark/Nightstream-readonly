use crate::PiCcsError;
use neo_ccs::{CcsMatrix, CcsStructure, Mat, MeInstance};
use neo_math::{F, K};
use neo_memory::ajtai::decode_vector as ajtai_decode_vector;
use neo_memory::cpu::{build_bus_layout_for_instances_with_shout_and_twist_lanes, BusLayout};
use neo_memory::sparse_time::SparseIdxVec;
use neo_memory::witness::{LutInstance, MemInstance, StepInstanceBundle, StepWitnessBundle};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use std::collections::{HashMap, HashSet};

pub(crate) trait BusStepView<Cmt> {
    fn m_in(&self) -> usize;
    fn public_x(&self) -> &[F];
    fn lut_insts_len(&self) -> usize;
    fn mem_insts_len(&self) -> usize;
    fn lut_inst(&self, idx: usize) -> &LutInstance<Cmt, F>;
    fn mem_inst(&self, idx: usize) -> &MemInstance<Cmt, F>;
}

impl<Cmt, KK> BusStepView<Cmt> for StepWitnessBundle<Cmt, F, KK> {
    fn m_in(&self) -> usize {
        self.mcs.0.m_in
    }

    fn public_x(&self) -> &[F] {
        &self.mcs.0.x
    }

    fn lut_insts_len(&self) -> usize {
        self.lut_instances.len()
    }

    fn mem_insts_len(&self) -> usize {
        self.mem_instances.len()
    }

    fn lut_inst(&self, idx: usize) -> &LutInstance<Cmt, F> {
        &self.lut_instances[idx].0
    }

    fn mem_inst(&self, idx: usize) -> &MemInstance<Cmt, F> {
        &self.mem_instances[idx].0
    }
}

impl<Cmt, KK> BusStepView<Cmt> for StepInstanceBundle<Cmt, F, KK> {
    fn m_in(&self) -> usize {
        self.mcs_inst.m_in
    }

    fn public_x(&self) -> &[F] {
        &self.mcs_inst.x
    }

    fn lut_insts_len(&self) -> usize {
        self.lut_insts.len()
    }

    fn mem_insts_len(&self) -> usize {
        self.mem_insts.len()
    }

    fn lut_inst(&self, idx: usize) -> &LutInstance<Cmt, F> {
        &self.lut_insts[idx]
    }

    fn mem_inst(&self, idx: usize) -> &MemInstance<Cmt, F> {
        &self.mem_insts[idx]
    }
}

fn infer_chunk_size_from_steps<Cmt, S: BusStepView<Cmt>>(steps: &[S]) -> Result<usize, PiCcsError> {
    let mut max_steps = 0usize;
    for step in steps {
        for i in 0..step.lut_insts_len() {
            max_steps = max_steps.max(step.lut_inst(i).steps);
        }
        for i in 0..step.mem_insts_len() {
            max_steps = max_steps.max(step.mem_inst(i).steps);
        }
    }
    if max_steps == 0 {
        // No instances => no bus; chunk_size is irrelevant but must be >=1 for layout helpers.
        return Ok(1);
    }
    Ok(max_steps)
}

fn infer_bus_layout_for_steps<Cmt, S: BusStepView<Cmt>>(
    s: &CcsStructure<F>,
    steps: &[S],
) -> Result<BusLayout, PiCcsError> {
    if steps.is_empty() {
        return Err(PiCcsError::InvalidInput("no steps".into()));
    }
    let m_in = steps[0].m_in();
    for (i, step) in steps.iter().enumerate() {
        let cur_m_in = step.m_in();
        if cur_m_in != m_in {
            return Err(PiCcsError::InvalidInput(format!(
                "m_in mismatch across steps (step 0 has {m_in}, step {i} has {cur_m_in})"
            )));
        }
    }

    let chunk_size = infer_chunk_size_from_steps(steps)?;

    let base_shout_ell_addrs: Vec<usize> = (0..steps[0].lut_insts_len())
        .map(|i| {
            let inst = steps[0].lut_inst(i);
            inst.d * inst.ell
        })
        .collect();
    let base_shout_lanes: Vec<usize> = (0..steps[0].lut_insts_len())
        .map(|i| {
            let inst = steps[0].lut_inst(i);
            inst.lanes
        })
        .collect();
    let base_twist_ell_addrs: Vec<usize> = (0..steps[0].mem_insts_len())
        .map(|i| {
            let inst = steps[0].mem_inst(i);
            inst.d * inst.ell
        })
        .collect();
    let base_twist_lanes: Vec<usize> = (0..steps[0].mem_insts_len())
        .map(|i| {
            let inst = steps[0].mem_inst(i);
            inst.lanes
        })
        .collect();

    for (i, step) in steps.iter().enumerate().skip(1) {
        let cur_shout: Vec<usize> = (0..step.lut_insts_len())
            .map(|j| {
                let inst = step.lut_inst(j);
                inst.d * inst.ell
            })
            .collect();
        let cur_shout_lanes: Vec<usize> = (0..step.lut_insts_len())
            .map(|j| {
                let inst = step.lut_inst(j);
                inst.lanes
            })
            .collect();
        let cur_twist: Vec<usize> = (0..step.mem_insts_len())
            .map(|j| {
                let inst = step.mem_inst(j);
                inst.d * inst.ell
            })
            .collect();
        let cur_twist_lanes: Vec<usize> = (0..step.mem_insts_len())
            .map(|j| {
                let inst = step.mem_inst(j);
                inst.lanes
            })
            .collect();
        if cur_shout != base_shout_ell_addrs
            || cur_shout_lanes != base_shout_lanes
            || cur_twist != base_twist_ell_addrs
            || cur_twist_lanes != base_twist_lanes
        {
            return Err(PiCcsError::InvalidInput(format!(
                "shared CPU bus layout mismatch across steps (step 0 vs step {i})"
            )));
        }
    }

    let shout_ell_addrs_and_lanes = base_shout_ell_addrs
        .iter()
        .copied()
        .zip(base_shout_lanes.iter().copied())
        .map(|(ell_addr, lanes)| (ell_addr, lanes));
    let twist_ell_addrs_and_lanes = base_twist_ell_addrs
        .iter()
        .copied()
        .zip(base_twist_lanes.iter().copied())
        .map(|(ell_addr, lanes)| (ell_addr, lanes));
    let layout = build_bus_layout_for_instances_with_shout_and_twist_lanes(
        s.m,
        m_in,
        chunk_size,
        shout_ell_addrs_and_lanes,
        twist_ell_addrs_and_lanes,
    )
    .map_err(PiCcsError::InvalidInput)?;

    // If there are no bus columns (no Twist/Shout instances), Route A doesn't use the bus time rows.
    // Allow small CCS instances (including m_in == n) in this case.
    if layout.bus_cols == 0 {
        return Ok(layout);
    }

    if m_in
        .checked_add(layout.chunk_size)
        .ok_or_else(|| PiCcsError::InvalidInput("m_in + chunk_size overflow".into()))?
        > s.n
    {
        return Err(PiCcsError::InvalidInput(format!(
            "bus time rows out of range: m_in({m_in}) + chunk_size({}) > n({})",
            layout.chunk_size, s.n
        )));
    }

    Ok(layout)
}

pub(crate) fn prepare_ccs_for_shared_cpu_bus_steps<'a, Cmt, S: BusStepView<Cmt>>(
    s0: &'a CcsStructure<F>,
    steps: &[S],
) -> Result<(&'a CcsStructure<F>, BusLayout), PiCcsError> {
    let bus = infer_bus_layout_for_steps(s0, steps)?;
    let padding_rows = ensure_ccs_has_shared_bus_padding_for_steps(s0, &bus, steps)?;
    ensure_ccs_binds_shared_bus_for_steps(s0, &bus, &padding_rows)?;
    // Performance: do NOT materialize bus copyout matrices into the CCS. Instead, we append the
    // corresponding ME openings directly from the witness (see `append_bus_openings_to_me_*`).
    Ok((s0, bus))
}

#[inline]
fn chi_for_row_index(r: &[K], idx: usize) -> K {
    // Multilinear basis polynomial χ_r(idx) where idx is interpreted in little-endian bits:
    // χ_r(idx) = ∏_i (idx_i ? r_i : (1 - r_i)).
    let mut acc = K::ONE;
    for (bit, &ri) in r.iter().enumerate() {
        let is_one = ((idx >> bit) & 1) == 1;
        acc *= if is_one { ri } else { K::ONE - ri };
    }
    acc
}

#[inline]
fn precompute_contiguous_time_weights(r: &[K], start_row: usize, len: usize, n_pad: usize) -> Vec<K> {
    if len == 0 {
        return Vec::new();
    }

    // For large contiguous windows, build χ_r over the full boolean domain once and slice.
    // This avoids repeated per-index basis recomputation in hot Route-A paths.
    const FULL_CHI_MAX_PAD: usize = 1 << 20;
    let naive_ops = len.saturating_mul(r.len().max(1));
    let use_full_table = len >= 1024 && n_pad <= FULL_CHI_MAX_PAD && naive_ops >= n_pad;

    if use_full_table {
        let mut chi = Vec::with_capacity(n_pad);
        chi.push(K::ONE);
        for &ri in r {
            let one_minus_ri = K::ONE - ri;
            let cur_len = chi.len();
            chi.reserve(cur_len);
            for i in 0..cur_len {
                let v = chi[i];
                chi[i] = v * one_minus_ri;
                chi.push(v * ri);
            }
        }
        return chi[start_row..start_row + len].to_vec();
    }

    let mut out = Vec::with_capacity(len);
    for off in 0..len {
        out.push(chi_for_row_index(r, start_row + off));
    }
    out
}

pub(crate) fn append_bus_openings_to_me_instance<Cmt>(
    params: &NeoParams,
    bus: &BusLayout,
    core_t: usize,
    Z: &Mat<F>,
    me: &mut MeInstance<Cmt, F, K>,
) -> Result<(), PiCcsError>
where
    Cmt: Clone,
{
    if bus.bus_cols == 0 {
        return Ok(());
    }

    let y_pad = (params.d as usize).next_power_of_two();
    let d = neo_math::D;
    if y_pad < d {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings require y_pad >= D (y_pad={y_pad}, D={d})"
        )));
    }
    if Z.rows() != d {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings require Z.rows()==D (got {}, want {})",
            Z.rows(),
            d
        )));
    }
    if Z.cols() != bus.m {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings require Z.cols()==bus.m (got {}, want {})",
            Z.cols(),
            bus.m
        )));
    }
    if me.m_in != bus.m_in {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings require ME.m_in==bus.m_in (got {}, want {})",
            me.m_in, bus.m_in
        )));
    }
    if me.r.len() == 0 {
        return Err(PiCcsError::InvalidInput("bus openings require non-empty ME.r".into()));
    }
    let n_pad = 1usize
        .checked_shl(me.r.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("2^ell_n overflow".into()))?;
    for j in 0..bus.chunk_size {
        let row = bus.time_index(j);
        if row >= n_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "bus time_index({j})={row} out of range for ell_n={} (n_pad={})",
                me.r.len(),
                n_pad
            )));
        }
    }

    // Idempotent append: allow callers to call this once; reject unexpected shapes.
    let want_len = core_t
        .checked_add(bus.bus_cols)
        .ok_or_else(|| PiCcsError::InvalidInput("core_t + bus_cols overflow".into()))?;
    if me.y.len() == want_len && me.y_scalars.len() == want_len {
        return Ok(());
    }
    if me.y.len() != core_t || me.y_scalars.len() != core_t {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings expect ME y/y_scalars to start at core_t (y.len()={}, y_scalars.len()={}, core_t={})",
            me.y.len(),
            me.y_scalars.len(),
            core_t
        )));
    }
    for (j, row) in me.y.iter().enumerate() {
        if row.len() != y_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "bus openings require ME.y[{j}].len()==y_pad (got {}, want {})",
                row.len(),
                y_pad
            )));
        }
    }

    // Precompute χ_r(time_index(j)) weights for the bus time rows.
    let time_weights = precompute_contiguous_time_weights(&me.r, bus.time_index(0), bus.chunk_size, n_pad);
    let weighted_rows: Vec<(usize, K)> = time_weights
        .into_iter()
        .enumerate()
        .filter_map(|(j, w)| (w != K::ZERO).then_some((j, w)))
        .collect();

    // Base-b powers for recomposition.
    let bK = K::from(F::from_u64(params.b as u64));
    let mut pow_b = Vec::with_capacity(d);
    let mut cur = K::ONE;
    for _ in 0..d {
        pow_b.push(cur);
        cur *= bK;
    }

    // Append bus openings in canonical col_id order so `bus_y_base = y_scalars.len() - bus_cols`
    // remains valid.
    for col_id in 0..bus.bus_cols {
        let z_indices: Vec<usize> = weighted_rows
            .iter()
            .map(|(j, _)| bus.bus_cell(col_id, *j))
            .collect();
        let mut y_row = vec![K::ZERO; y_pad];
        let mut y_scalar = K::ZERO;
        for rho in 0..d {
            let mut acc = K::ZERO;
            for ((_, w), &z_idx) in weighted_rows.iter().zip(z_indices.iter()) {
                acc += *w * K::from(Z[(rho, z_idx)]);
            }
            y_row[rho] = acc;
            y_scalar += acc * pow_b[rho];
        }

        me.y.push(y_row);
        me.y_scalars.push(y_scalar);
    }

    Ok(())
}

pub(crate) fn append_bus_openings_to_me_instance_at_js<Cmt>(
    params: &NeoParams,
    bus: &BusLayout,
    core_t: usize,
    Z: &Mat<F>,
    me: &mut MeInstance<Cmt, F, K>,
    js: &[usize],
) -> Result<(), PiCcsError>
where
    Cmt: Clone,
{
    if bus.bus_cols == 0 {
        return Ok(());
    }

    let y_pad = (params.d as usize).next_power_of_two();
    let d = neo_math::D;
    if y_pad < d {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings require y_pad >= D (y_pad={y_pad}, D={d})"
        )));
    }
    if Z.rows() != d {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings require Z.rows()==D (got {}, want {})",
            Z.rows(),
            d
        )));
    }
    if Z.cols() != bus.m {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings require Z.cols()==bus.m (got {}, want {})",
            Z.cols(),
            bus.m
        )));
    }
    if me.m_in != bus.m_in {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings require ME.m_in==bus.m_in (got {}, want {})",
            me.m_in, bus.m_in
        )));
    }
    if me.r.is_empty() {
        return Err(PiCcsError::InvalidInput("bus openings require non-empty ME.r".into()));
    }

    let n_pad = 1usize
        .checked_shl(me.r.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("2^ell_n overflow".into()))?;
    for &j in js {
        if j >= bus.chunk_size {
            return Err(PiCcsError::InvalidInput(format!(
                "bus j out of range: j={j} >= bus.chunk_size={}",
                bus.chunk_size
            )));
        }
        let row = bus.time_index(j);
        if row >= n_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "bus time_index({j})={row} out of range for ell_n={} (n_pad={})",
                me.r.len(),
                n_pad
            )));
        }
    }

    // Idempotent append: allow callers to call this once; reject unexpected shapes.
    let want_len = core_t
        .checked_add(bus.bus_cols)
        .ok_or_else(|| PiCcsError::InvalidInput("core_t + bus_cols overflow".into()))?;
    if me.y.len() == want_len && me.y_scalars.len() == want_len {
        return Ok(());
    }
    if me.y.len() != core_t || me.y_scalars.len() != core_t {
        return Err(PiCcsError::InvalidInput(format!(
            "bus openings expect ME y/y_scalars to start at core_t (y.len()={}, y_scalars.len()={}, core_t={})",
            me.y.len(),
            me.y_scalars.len(),
            core_t
        )));
    }
    for (j, row) in me.y.iter().enumerate() {
        if row.len() != y_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "bus openings require ME.y[{j}].len()==y_pad (got {}, want {})",
                row.len(),
                y_pad
            )));
        }
    }

    // Precompute χ_r(time_index(j)) weights for the selected bus rows.
    let dense_selection = js.len().saturating_mul(3) >= bus.chunk_size;
    let time_weights: Vec<K> = if dense_selection {
        let all = precompute_contiguous_time_weights(&me.r, bus.time_index(0), bus.chunk_size, n_pad);
        let mut out = Vec::with_capacity(js.len());
        for &j in js {
            out.push(all[j]);
        }
        out
    } else {
        let mut out = Vec::with_capacity(js.len());
        for &j in js {
            out.push(chi_for_row_index(&me.r, bus.time_index(j)));
        }
        out
    };
    let weighted_rows: Vec<(usize, K)> = js
        .iter()
        .copied()
        .zip(time_weights.iter().copied())
        .filter_map(|(j, w)| (w != K::ZERO).then_some((j, w)))
        .collect();

    // Base-b powers for recomposition.
    let bK = K::from(F::from_u64(params.b as u64));
    let mut pow_b = Vec::with_capacity(d);
    let mut cur = K::ONE;
    for _ in 0..d {
        pow_b.push(cur);
        cur *= bK;
    }

    // Append bus openings in canonical col_id order so `bus_y_base = y_scalars.len() - bus_cols`
    // remains valid.
    for col_id in 0..bus.bus_cols {
        let z_indices: Vec<usize> = weighted_rows
            .iter()
            .map(|(j, _)| bus.bus_cell(col_id, *j))
            .collect();
        let mut y_row = vec![K::ZERO; y_pad];
        let mut y_scalar = K::ZERO;
        for rho in 0..d {
            let mut acc = K::ZERO;
            for ((_, w), &z_idx) in weighted_rows.iter().zip(z_indices.iter()) {
                acc += *w * K::from(Z[(rho, z_idx)]);
            }
            y_row[rho] = acc;
            y_scalar += acc * pow_b[rho];
        }

        me.y.push(y_row);
        me.y_scalars.push(y_scalar);
    }

    Ok(())
}

/// Append time-indexed openings for a column-major region of the CPU witness.
///
/// This is a "no shared CPU bus tail" bridge: instead of materializing copyout matrices for
/// per-row columns (e.g. an execution trace), we compute their Route-A time-combined openings
/// directly from the committed witness matrix `Z` and append them to the ME instance.
///
/// Semantics: for each `col_id` in `cols`, append an opening for the vector
/// `{ z[col_base + col_id * t_len + j] }_{j=0..t_len-1}` combined with weights
/// `χ_r(m_in + j)` where `r = me.r`.
pub(crate) fn append_col_major_time_openings_to_me_instance<Cmt>(
    params: &NeoParams,
    m_in: usize,
    t_len: usize,
    col_base: usize,
    cols: &[usize],
    core_t: usize,
    Z: &Mat<F>,
    me: &mut MeInstance<Cmt, F, K>,
) -> Result<(), PiCcsError>
where
    Cmt: Clone,
{
    if cols.is_empty() {
        return Ok(());
    }
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput("trace openings require t_len >= 1".into()));
    }

    let y_pad = (params.d as usize).next_power_of_two();
    let d = neo_math::D;
    if y_pad < d {
        return Err(PiCcsError::InvalidInput(format!(
            "trace openings require y_pad >= D (y_pad={y_pad}, D={d})"
        )));
    }
    if Z.rows() != d {
        return Err(PiCcsError::InvalidInput(format!(
            "trace openings require Z.rows()==D (got {}, want {})",
            Z.rows(),
            d
        )));
    }
    if me.m_in != m_in {
        return Err(PiCcsError::InvalidInput(format!(
            "trace openings require ME.m_in==m_in (got {}, want {})",
            me.m_in, m_in
        )));
    }
    if me.r.is_empty() {
        return Err(PiCcsError::InvalidInput("trace openings require non-empty ME.r".into()));
    }
    if col_base >= Z.cols() {
        return Err(PiCcsError::InvalidInput(format!(
            "trace openings require col_base < m (col_base={}, m={})",
            col_base,
            Z.cols()
        )));
    }

    let n_pad = 1usize
        .checked_shl(me.r.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("2^ell_n overflow".into()))?;
    for j in 0..t_len {
        let row = m_in
            .checked_add(j)
            .ok_or_else(|| PiCcsError::InvalidInput("m_in + t overflow".into()))?;
        if row >= n_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "trace time row index out of range: (m_in + j)={} out of range for ell_n={} (n_pad={})",
                row,
                me.r.len(),
                n_pad
            )));
        }
    }

    // Idempotent append: allow callers to call this once; reject unexpected shapes.
    let want_len = core_t
        .checked_add(cols.len())
        .ok_or_else(|| PiCcsError::InvalidInput("core_t + cols.len overflow".into()))?;
    if me.y.len() == want_len && me.y_scalars.len() == want_len {
        return Ok(());
    }
    if me.y.len() != core_t || me.y_scalars.len() != core_t {
        return Err(PiCcsError::InvalidInput(format!(
            "trace openings expect ME y/y_scalars to start at core_t (y.len()={}, y_scalars.len()={}, core_t={})",
            me.y.len(),
            me.y_scalars.len(),
            core_t
        )));
    }
    for (j, row) in me.y.iter().enumerate() {
        if row.len() != y_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "trace openings require ME.y[{j}].len()==y_pad (got {}, want {})",
                row.len(),
                y_pad
            )));
        }
    }

    // Precompute χ_r(m_in + j) weights for the time rows.
    let time_weights = precompute_contiguous_time_weights(&me.r, m_in, t_len, n_pad);
    let weighted_rows: Vec<(usize, K)> = time_weights
        .into_iter()
        .enumerate()
        .filter_map(|(j, w)| (w != K::ZERO).then_some((j, w)))
        .collect();

    // Base-b powers for recomposition.
    let bK = K::from(F::from_u64(params.b as u64));
    let mut pow_b = Vec::with_capacity(d);
    let mut cur = K::ONE;
    for _ in 0..d {
        pow_b.push(cur);
        cur *= bK;
    }

    for &col_id in cols {
        let col_offset = col_id
            .checked_mul(t_len)
            .ok_or_else(|| PiCcsError::InvalidInput("trace col_id * t_len overflow".into()))?;
        let col_start = col_base
            .checked_add(col_offset)
            .ok_or_else(|| PiCcsError::InvalidInput("trace col_base + col_offset overflow".into()))?;
        let col_end = col_start
            .checked_add(t_len - 1)
            .ok_or_else(|| PiCcsError::InvalidInput("trace col_end overflow".into()))?;
        if col_end >= Z.cols() {
            return Err(PiCcsError::InvalidInput(format!(
                "trace openings: column span out of range (col_start={col_start}, col_end={col_end}, m={})",
                Z.cols()
            )));
        }

        let mut y_row = vec![K::ZERO; y_pad];
        let mut y_scalar = K::ZERO;
        for rho in 0..d {
            let mut acc = K::ZERO;
            for (j, w) in weighted_rows.iter() {
                acc += *w * K::from(Z[(rho, col_start + *j)]);
            }
            y_row[rho] = acc;
            y_scalar += acc * pow_b[rho];
        }

        me.y.push(y_row);
        me.y_scalars.push(y_scalar);
    }

    Ok(())
}

/// Append time-indexed openings for a column-major region of the CPU witness, using only the
/// selected time rows `js`.
///
/// This is valid when the caller knows that for each opened column `col_id`, all omitted rows
/// (`j` not in `js`) are zero in the witness; then the opening can be computed by summing only
/// over `js`.
pub(crate) fn append_col_major_time_openings_to_me_instance_at_js<Cmt>(
    params: &NeoParams,
    m_in: usize,
    t_len: usize,
    col_base: usize,
    cols: &[usize],
    core_t: usize,
    Z: &Mat<F>,
    me: &mut MeInstance<Cmt, F, K>,
    js: &[usize],
) -> Result<(), PiCcsError>
where
    Cmt: Clone,
{
    if cols.is_empty() {
        return Ok(());
    }
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput("trace openings require t_len >= 1".into()));
    }

    let y_pad = (params.d as usize).next_power_of_two();
    let d = neo_math::D;
    if y_pad < d {
        return Err(PiCcsError::InvalidInput(format!(
            "trace openings require y_pad >= D (y_pad={y_pad}, D={d})"
        )));
    }
    if Z.rows() != d {
        return Err(PiCcsError::InvalidInput(format!(
            "trace openings require Z.rows()==D (got {}, want {})",
            Z.rows(),
            d
        )));
    }
    if me.m_in != m_in {
        return Err(PiCcsError::InvalidInput(format!(
            "trace openings require ME.m_in==m_in (got {}, want {})",
            me.m_in, m_in
        )));
    }
    if me.r.is_empty() {
        return Err(PiCcsError::InvalidInput("trace openings require non-empty ME.r".into()));
    }
    if col_base >= Z.cols() {
        return Err(PiCcsError::InvalidInput(format!(
            "trace openings require col_base < m (col_base={}, m={})",
            col_base,
            Z.cols()
        )));
    }

    let n_pad = 1usize
        .checked_shl(me.r.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("2^ell_n overflow".into()))?;
    for &j in js {
        if j >= t_len {
            return Err(PiCcsError::InvalidInput(format!(
                "trace js out of range: j={j} >= t_len={t_len}"
            )));
        }
        let row = m_in
            .checked_add(j)
            .ok_or_else(|| PiCcsError::InvalidInput("m_in + j overflow".into()))?;
        if row >= n_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "trace time row index out of range: (m_in + j)={row} out of range for ell_n={} (n_pad={})",
                me.r.len(),
                n_pad
            )));
        }
    }

    // Idempotent append: allow callers to call this once; reject unexpected shapes.
    let want_len = core_t
        .checked_add(cols.len())
        .ok_or_else(|| PiCcsError::InvalidInput("core_t + cols.len overflow".into()))?;
    if me.y.len() == want_len && me.y_scalars.len() == want_len {
        return Ok(());
    }
    if me.y.len() != core_t || me.y_scalars.len() != core_t {
        return Err(PiCcsError::InvalidInput(format!(
            "trace openings expect ME y/y_scalars to start at core_t (y.len()={}, y_scalars.len()={}, core_t={})",
            me.y.len(),
            me.y_scalars.len(),
            core_t
        )));
    }
    for (j, row) in me.y.iter().enumerate() {
        if row.len() != y_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "trace openings require ME.y[{j}].len()==y_pad (got {}, want {})",
                row.len(),
                y_pad
            )));
        }
    }

    // Precompute χ_r(m_in + j) weights for the selected time rows.
    let dense_selection = js.len().saturating_mul(3) >= t_len;
    let time_weights: Vec<(usize, K)> = if dense_selection {
        let all = precompute_contiguous_time_weights(&me.r, m_in, t_len, n_pad);
        let mut out = Vec::with_capacity(js.len());
        for &j in js {
            out.push((j, all[j]));
        }
        out
    } else {
        let mut out = Vec::with_capacity(js.len());
        for &j in js {
            out.push((j, chi_for_row_index(&me.r, m_in + j)));
        }
        out
    };
    let weighted_rows: Vec<(usize, K)> = time_weights
        .into_iter()
        .filter_map(|(j, w)| (w != K::ZERO).then_some((j, w)))
        .collect();

    // Base-b powers for recomposition.
    let bK = K::from(F::from_u64(params.b as u64));
    let mut pow_b = Vec::with_capacity(d);
    let mut cur = K::ONE;
    for _ in 0..d {
        pow_b.push(cur);
        cur *= bK;
    }

    for &col_id in cols {
        let col_offset = col_id
            .checked_mul(t_len)
            .ok_or_else(|| PiCcsError::InvalidInput("trace col_id * t_len overflow".into()))?;
        let col_start = col_base
            .checked_add(col_offset)
            .ok_or_else(|| PiCcsError::InvalidInput("trace col_base + col_offset overflow".into()))?;
        let col_end = col_start
            .checked_add(t_len - 1)
            .ok_or_else(|| PiCcsError::InvalidInput("trace col_end overflow".into()))?;
        if col_end >= Z.cols() {
            return Err(PiCcsError::InvalidInput(format!(
                "trace openings: column span out of range (col_start={col_start}, col_end={col_end}, m={})",
                Z.cols()
            )));
        }

        let mut y_row = vec![K::ZERO; y_pad];
        let mut y_scalar = K::ZERO;
        for rho in 0..d {
            let mut acc = K::ZERO;
            for (j, w) in weighted_rows.iter() {
                acc += *w * K::from(Z[(rho, col_start + *j)]);
            }
            y_row[rho] = acc;
            y_scalar += acc * pow_b[rho];
        }

        me.y.push(y_row);
        me.y_scalars.push(y_scalar);
    }

    Ok(())
}

fn active_matrix_indices(s: &CcsStructure<F>) -> Vec<usize> {
    let t = s.matrices.len();
    let mut active = vec![false; t];
    for term in s.f.terms() {
        for (j, &exp) in term.exps.iter().enumerate() {
            if exp != 0 {
                active[j] = true;
            }
        }
    }
    active
        .iter()
        .enumerate()
        .filter_map(|(j, &is_active)| is_active.then_some(j))
        .collect()
}

fn ccs_col_has_any_nonzero(mat: &CcsMatrix<F>, col: usize) -> bool {
    match mat {
        CcsMatrix::Identity { n } => col < *n,
        CcsMatrix::Csc(csc) => csc.col_ptr[col] < csc.col_ptr[col + 1],
    }
}

fn ccs_col_has_nonzero_outside_padding_rows(mat: &CcsMatrix<F>, col: usize, is_padding_row: &[bool]) -> bool {
    match mat {
        CcsMatrix::Identity { n } => {
            if col >= *n {
                return false;
            }
            let row = col;
            row < is_padding_row.len() && !is_padding_row[row]
        }
        CcsMatrix::Csc(csc) => {
            let s0 = csc.col_ptr[col];
            let e0 = csc.col_ptr[col + 1];
            for k in s0..e0 {
                let row = csc.row_idx[k];
                if row < is_padding_row.len() && !is_padding_row[row] {
                    return true;
                }
            }
            false
        }
    }
}

struct BusColLabel {
    col_id: usize,
    label: String,
}

fn required_bus_cols_for_layout(layout: &BusLayout) -> Vec<BusColLabel> {
    let mut out = Vec::<BusColLabel>::new();
    for (lut_idx, inst) in layout.shout_cols.iter().enumerate() {
        for (lane_idx, shout) in inst.lanes.iter().enumerate() {
            for (b, col_id) in shout.addr_bits.clone().enumerate() {
                out.push(BusColLabel {
                    col_id,
                    label: format!("shout[{lut_idx}].lane[{lane_idx}].addr_bits[{b}]"),
                });
            }
            out.push(BusColLabel {
                col_id: shout.has_lookup,
                label: format!("shout[{lut_idx}].lane[{lane_idx}].has_lookup"),
            });
            out.push(BusColLabel {
                col_id: shout.val,
                label: format!("shout[{lut_idx}].lane[{lane_idx}].val"),
            });
        }
    }

    for (mem_idx, inst) in layout.twist_cols.iter().enumerate() {
        for (lane_idx, twist) in inst.lanes.iter().enumerate() {
            for (b, col_id) in twist.ra_bits.clone().enumerate() {
                out.push(BusColLabel {
                    col_id,
                    label: format!("twist[{mem_idx}].lane[{lane_idx}].ra_bits[{b}]"),
                });
            }
            for (b, col_id) in twist.wa_bits.clone().enumerate() {
                out.push(BusColLabel {
                    col_id,
                    label: format!("twist[{mem_idx}].lane[{lane_idx}].wa_bits[{b}]"),
                });
            }
            out.push(BusColLabel {
                col_id: twist.has_read,
                label: format!("twist[{mem_idx}].lane[{lane_idx}].has_read"),
            });
            out.push(BusColLabel {
                col_id: twist.has_write,
                label: format!("twist[{mem_idx}].lane[{lane_idx}].has_write"),
            });
            out.push(BusColLabel {
                col_id: twist.wv,
                label: format!("twist[{mem_idx}].lane[{lane_idx}].wv"),
            });
            out.push(BusColLabel {
                col_id: twist.rv,
                label: format!("twist[{mem_idx}].lane[{lane_idx}].rv"),
            });
            out.push(BusColLabel {
                col_id: twist.inc,
                label: format!("twist[{mem_idx}].lane[{lane_idx}].inc_at_write_addr"),
            });
        }
    }

    out
}

fn required_bus_binding_cols_for_layout(layout: &BusLayout) -> Vec<BusColLabel> {
    // Note: `inc_at_write_addr` is a Twist-internal witness field derived from the sparse
    // memory state. Many CPU CCSes do not (and should not) constrain it outside padding rows;
    // it is constrained by the Twist Route-A checks instead. We still require the canonical
    // padding constraint `(1 - has_write) * inc_at_write_addr = 0`.
    let inc_cols: HashSet<usize> = layout
        .twist_cols
        .iter()
        .flat_map(|inst| inst.lanes.iter().map(|t| t.inc))
        .collect();

    // Shout key `addr_bits` are often constrained outside the *main* CPU CCS:
    // - by a decode/semantics sidecar CCS, and/or
    // - by VM-specific constraints that live outside the shared-bus binding gadget.
    //
    // The Route-A Shout argument already constrains `(addr_bits, val)` internally via:
    // - per-lane Shout value/adaptor terminal checks, and
    // - trace linkage checks (`verify_route_a_memory_step_no_shared_cpu_bus`) that bind the
    //   CPU trace's `(shout_has_lookup, shout_val, shout_lhs, shout_rhs)` to the sidecar openings.
    //
    // So the critical CPU→bus requirement here is that the CPU CCS binds `has_lookup` and `val`
    // outside padding rows; requiring `addr_bits` outside padding rows would force CPUs to
    // materialize a packed 64-bit key scalar, which can violate Neo's Ajtai encoding bounds
    // (d=54 with balanced base-b digits).
    let shout_addr_cols: HashSet<usize> = layout
        .shout_cols
        .iter()
        .flat_map(|inst| inst.lanes.iter().flat_map(|s| s.addr_bits.clone()))
        .collect();
    required_bus_cols_for_layout(layout)
        .into_iter()
        .filter(|c| !inc_cols.contains(&c.col_id))
        .filter(|c| !shout_addr_cols.contains(&c.col_id))
        .collect()
}

fn ensure_ccs_references_bus_cols(
    s: &CcsStructure<F>,
    bus: &BusLayout,
    required_cols: &[BusColLabel],
) -> Result<(), PiCcsError> {
    if bus.bus_cols == 0 {
        return Ok(());
    }

    let active = active_matrix_indices(s);
    if active.is_empty() {
        // If the CCS polynomial does not depend on any matrix (e.g. f == 0), the CCS imposes
        // no constraints at all. In that case this check is not meaningful, so we skip it.
        return Ok(());
    }

    let mut missing: Vec<&BusColLabel> = Vec::new();
    for col in required_cols {
        if col.col_id >= bus.bus_cols {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus internal error: required col_id {} out of range (bus_cols={})",
                col.col_id, bus.bus_cols
            )));
        }
        let mut all_js_present = true;
        for j in 0..bus.chunk_size {
            let z_idx = bus.bus_cell(col.col_id, j);
            if z_idx >= s.m {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus internal error: bus z index {} out of range (m={})",
                    z_idx, s.m
                )));
            }

            let mut found = false;
            'active_mats: for &mj in &active {
                let mat = &s.matrices[mj];
                if ccs_col_has_any_nonzero(mat, z_idx) {
                    found = true;
                    break 'active_mats;
                }
            }
            if !found {
                all_js_present = false;
                break;
            }
        }
        if !all_js_present {
            missing.push(col);
        }
    }

    if missing.is_empty() {
        return Ok(());
    }

    let mut examples: Vec<String> = missing
        .iter()
        .take(8)
        .map(|c| format!("col_id {} ({})", c.col_id, c.label))
        .collect();
    if missing.len() > examples.len() {
        examples.push(format!("... ({} more)", missing.len() - examples.len()));
    }

    Err(PiCcsError::InvalidInput(format!(
        "shared_cpu_bus=true but CPU CCS does not reference required bus columns in any active constraint matrix.\n\
         This makes the bus a dead witness: CPU semantics can fork from Twist/Shout semantics.\n\
         Fix: make CPU semantics use the bus coordinates directly, or add equality constraints tying any shadow columns to the bus.\n\
         Missing examples: {}",
        examples.join(", ")
    )))
}

fn ensure_ccs_references_bus_cols_outside_padding_rows(
    s: &CcsStructure<F>,
    bus: &BusLayout,
    padding_rows: &HashSet<usize>,
    required_cols: &[BusColLabel],
) -> Result<(), PiCcsError> {
    if bus.bus_cols == 0 {
        return Ok(());
    }

    let active = active_matrix_indices(s);
    if active.is_empty() {
        // If the CCS polynomial does not depend on any matrix (e.g. f == 0), the CCS imposes
        // no constraints at all. In that case this check is not meaningful, so we skip it.
        return Ok(());
    }

    let mut is_padding_row = vec![false; s.n];
    for &r in padding_rows {
        if r < s.n {
            is_padding_row[r] = true;
        }
    }

    let mut missing: Vec<&BusColLabel> = Vec::new();
    for col in required_cols {
        if col.col_id >= bus.bus_cols {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus internal error: required col_id {} out of range (bus_cols={})",
                col.col_id, bus.bus_cols
            )));
        }
        let mut all_js_present = true;
        for j in 0..bus.chunk_size {
            let z_idx = bus.bus_cell(col.col_id, j);
            if z_idx >= s.m {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus internal error: bus z index {} out of range (m={})",
                    z_idx, s.m
                )));
            }

            let mut found = false;
            'active_mats: for &mj in &active {
                let mat = &s.matrices[mj];
                if ccs_col_has_nonzero_outside_padding_rows(mat, z_idx, &is_padding_row) {
                    found = true;
                    break 'active_mats;
                }
            }
            if !found {
                all_js_present = false;
                break;
            }
        }
        if !all_js_present {
            missing.push(col);
        }
    }

    if missing.is_empty() {
        return Ok(());
    }

    let mut examples: Vec<String> = missing
        .iter()
        .take(8)
        .map(|c| format!("col_id {} ({})", c.col_id, c.label))
        .collect();
    if missing.len() > examples.len() {
        examples.push(format!("... ({} more)", missing.len() - examples.len()));
    }

    Err(PiCcsError::InvalidInput(format!(
        "shared_cpu_bus=true but CPU CCS never references core bus columns outside the canonical padding rows.\n\
         This is a common linkage footgun: padding constraints (1-has_*)*field=0 only restrict inactive fields, \
         and do not bind active bus semantics to CPU semantics.\n\
         Fix: inject binding constraints tying CPU semantics columns to the shared bus (recommended: \
         `neo_memory::cpu::constraints::extend_ccs_with_shared_cpu_bus_constraints` / `CpuConstraintBuilder`).\n\
         Missing examples: {}",
        examples.join(", ")
    )))
}

#[derive(Clone, Debug)]
struct BusPaddingLabel {
    flag_z_idx: usize,
    field_z_idx: usize,
    label: String,
}

fn required_bus_padding_for_layout(bus: &BusLayout) -> Vec<BusPaddingLabel> {
    let mut out = Vec::<BusPaddingLabel>::new();

    for (lut_idx, inst) in bus.shout_cols.iter().enumerate() {
        for (lane_idx, shout) in inst.lanes.iter().enumerate() {
            for j in 0..bus.chunk_size {
                let has_lookup_z = bus.bus_cell(shout.has_lookup, j);
                let val_z = bus.bus_cell(shout.val, j);

                // (1 - has_lookup) * val = 0
                out.push(BusPaddingLabel {
                    flag_z_idx: has_lookup_z,
                    field_z_idx: val_z,
                    label: format!("shout[{lut_idx}].lane[{lane_idx}][j={j}]: (1-has_lookup)*val"),
                });

                // (1 - has_lookup) * addr_bits[b] = 0
                for (b, col_id) in shout.addr_bits.clone().enumerate() {
                    let bit_z = bus.bus_cell(col_id, j);
                    out.push(BusPaddingLabel {
                        flag_z_idx: has_lookup_z,
                        field_z_idx: bit_z,
                        label: format!("shout[{lut_idx}].lane[{lane_idx}][j={j}]: (1-has_lookup)*addr_bits[{b}]"),
                    });
                }
            }
        }
    }

    for (mem_idx, inst) in bus.twist_cols.iter().enumerate() {
        for (lane_idx, twist) in inst.lanes.iter().enumerate() {
            for j in 0..bus.chunk_size {
                let has_read_z = bus.bus_cell(twist.has_read, j);
                let has_write_z = bus.bus_cell(twist.has_write, j);

                let wv_z = bus.bus_cell(twist.wv, j);
                let rv_z = bus.bus_cell(twist.rv, j);
                let inc_z = bus.bus_cell(twist.inc, j);

                // (1 - has_read) * rv = 0
                out.push(BusPaddingLabel {
                    flag_z_idx: has_read_z,
                    field_z_idx: rv_z,
                    label: format!("twist[{mem_idx}].lane[{lane_idx}][j={j}]: (1-has_read)*rv"),
                });

                // (1 - has_read) * ra_bits[b] = 0
                for (b, col_id) in twist.ra_bits.clone().enumerate() {
                    let bit_z = bus.bus_cell(col_id, j);
                    out.push(BusPaddingLabel {
                        flag_z_idx: has_read_z,
                        field_z_idx: bit_z,
                        label: format!("twist[{mem_idx}].lane[{lane_idx}][j={j}]: (1-has_read)*ra_bits[{b}]"),
                    });
                }

                // (1 - has_write) * wv = 0
                out.push(BusPaddingLabel {
                    flag_z_idx: has_write_z,
                    field_z_idx: wv_z,
                    label: format!("twist[{mem_idx}].lane[{lane_idx}][j={j}]: (1-has_write)*wv"),
                });

                // (1 - has_write) * inc_at_write_addr = 0
                out.push(BusPaddingLabel {
                    flag_z_idx: has_write_z,
                    field_z_idx: inc_z,
                    label: format!("twist[{mem_idx}].lane[{lane_idx}][j={j}]: (1-has_write)*inc_at_write_addr"),
                });

                // (1 - has_write) * wa_bits[b] = 0
                for (b, col_id) in twist.wa_bits.clone().enumerate() {
                    let bit_z = bus.bus_cell(col_id, j);
                    out.push(BusPaddingLabel {
                        flag_z_idx: has_write_z,
                        field_z_idx: bit_z,
                        label: format!("twist[{mem_idx}].lane[{lane_idx}][j={j}]: (1-has_write)*wa_bits[{b}]"),
                    });
                }
            }
        }
    }

    out
}

fn infer_public_constant_one_cols_from_steps<Cmt, S: BusStepView<Cmt>>(steps: &[S]) -> Vec<usize> {
    if steps.is_empty() {
        return Vec::new();
    }
    let m_in = steps[0].m_in();
    if m_in == 0 {
        return Vec::new();
    }

    let mut is_const_one = vec![true; m_in];
    for step in steps {
        let x = step.public_x();
        if x.len() != m_in {
            // Treat as no const-one. Other layers validate x length.
            return Vec::new();
        }
        for c in 0..m_in {
            if x[c] != F::ONE {
                is_const_one[c] = false;
            }
        }
    }
    is_const_one
        .iter()
        .enumerate()
        .filter_map(|(c, &ok)| ok.then_some(c))
        .collect()
}

fn ensure_ccs_has_bus_padding_constraints(
    s: &CcsStructure<F>,
    bus: &BusLayout,
    const_one_cols: &[usize],
    required: &[BusPaddingLabel],
) -> Result<HashSet<usize>, PiCcsError> {
    if bus.bus_cols == 0 {
        return Ok(HashSet::new());
    }

    let active = active_matrix_indices(s);
    if active.is_empty() {
        // CCS has no active constraints (e.g., f == 0); skip guardrail to preserve plumbing tests.
        return Ok(HashSet::new());
    }

    if const_one_cols.is_empty() {
        return Err(PiCcsError::InvalidInput(
            "shared_cpu_bus=true but no public constant-one column found to validate required padding constraints.\n\
             Fix: include a public input column that is always 1 (e.g., x[0]=1) and build padding constraints using it \
             (recommended: use `neo_memory::cpu::constraints::extend_ccs_with_shared_cpu_bus_constraints` or \
              `neo_memory::cpu::constraints::CpuConstraintBuilder`)."
                .into(),
        ));
    }

    // This validation is intentionally strict and recognizes two safe patterns in the canonical
    // R1CS embedding:
    //
    // 1) Explicit padding constraints:
    //      (1 - flag) * field = 0
    //
    // 2) For *bit* fields only (addr bits), padding implied by a gated-bit constraint:
    //      bit * (bit - flag) = 0
    //    combined with a boolean constraint on `flag`. When `flag ∈ {0,1}`, this implies:
    //      - flag=0 => bit=0
    //      - flag=1 => bit ∈ {0,1}
    //
    // We accept (2) to allow circuits to drop redundant per-bit padding rows while staying safe.
    let (a_idx, b_idx, c_idx) = if s.matrices.len() >= 4 && s.matrices[0].is_identity() {
        (1usize, 2usize, 3usize)
    } else if s.matrices.len() >= 3 {
        (0usize, 1usize, 2usize)
    } else {
        return Err(PiCcsError::InvalidInput(
            "shared_cpu_bus=true but CPU CCS has fewer than 3 matrices; cannot validate padding constraints".into(),
        ));
    };

    let n = s.n;
    let const_one_cols: HashSet<usize> = const_one_cols.iter().copied().collect();

    let mut c_has_nonzero = vec![false; n];
    let mut b_count = vec![0u8; n];
    let mut b_col1 = vec![0usize; n];
    let mut b_val1 = vec![F::ZERO; n];
    let mut b_col2 = vec![0usize; n];
    let mut b_val2 = vec![F::ZERO; n];

    let mut a_count = vec![0u8; n];
    let mut a_col1 = vec![0usize; n];
    let mut a_val1 = vec![F::ZERO; n];
    let mut a_col2 = vec![0usize; n];
    let mut a_val2 = vec![F::ZERO; n];

    let scan_c = |mat: &CcsMatrix<F>, c_has_nonzero: &mut [bool]| match mat {
        CcsMatrix::Identity { n } => {
            let cap = core::cmp::min(*n, c_has_nonzero.len());
            for row in 0..cap {
                c_has_nonzero[row] = true;
            }
        }
        CcsMatrix::Csc(csc) => {
            for &row in &csc.row_idx {
                if row < c_has_nonzero.len() {
                    c_has_nonzero[row] = true;
                }
            }
        }
    };

    let scan_b = |mat: &CcsMatrix<F>,
                  b_count: &mut [u8],
                  b_col1: &mut [usize],
                  b_val1: &mut [F],
                  b_col2: &mut [usize],
                  b_val2: &mut [F]| {
        match mat {
            CcsMatrix::Identity { n } => {
                let cap = core::cmp::min(*n, b_count.len());
                for row in 0..cap {
                    b_count[row] = 1;
                    b_col1[row] = row;
                    b_val1[row] = F::ONE;
                }
            }
            CcsMatrix::Csc(csc) => {
                for col in 0..csc.ncols {
                    let s0 = csc.col_ptr[col];
                    let e0 = csc.col_ptr[col + 1];
                    for k in s0..e0 {
                        let row = csc.row_idx[k];
                        if row >= b_count.len() {
                            continue;
                        }
                        match b_count[row] {
                            0 => {
                                b_count[row] = 1;
                                b_col1[row] = col;
                                b_val1[row] = csc.vals[k];
                            }
                            1 => {
                                b_count[row] = 2;
                                b_col2[row] = col;
                                b_val2[row] = csc.vals[k];
                            }
                            _ => {
                                b_count[row] = 3;
                            }
                        }
                    }
                }
            }
        }
    };

    let scan_a = |mat: &CcsMatrix<F>,
                  a_count: &mut [u8],
                  a_col1: &mut [usize],
                  a_val1: &mut [F],
                  a_col2: &mut [usize],
                  a_val2: &mut [F]| {
        match mat {
            CcsMatrix::Identity { n } => {
                let cap = core::cmp::min(*n, a_count.len());
                for row in 0..cap {
                    a_count[row] = 1;
                    a_col1[row] = row;
                    a_val1[row] = F::ONE;
                }
            }
            CcsMatrix::Csc(csc) => {
                for col in 0..csc.ncols {
                    let s0 = csc.col_ptr[col];
                    let e0 = csc.col_ptr[col + 1];
                    for k in s0..e0 {
                        let row = csc.row_idx[k];
                        if row >= a_count.len() {
                            continue;
                        }
                        match a_count[row] {
                            0 => {
                                a_count[row] = 1;
                                a_col1[row] = col;
                                a_val1[row] = csc.vals[k];
                            }
                            1 => {
                                a_count[row] = 2;
                                a_col2[row] = col;
                                a_val2[row] = csc.vals[k];
                            }
                            _ => {
                                a_count[row] = 3;
                            }
                        }
                    }
                }
            }
        }
    };

    scan_c(&s.matrices[c_idx], &mut c_has_nonzero);
    scan_b(
        &s.matrices[b_idx],
        &mut b_count,
        &mut b_col1,
        &mut b_val1,
        &mut b_col2,
        &mut b_val2,
    );
    scan_a(
        &s.matrices[a_idx],
        &mut a_count,
        &mut a_col1,
        &mut a_val1,
        &mut a_col2,
        &mut a_val2,
    );

    // Find boolean constraints for flag columns: flag * (flag - 1) = 0.
    let mut flag_is_boolean: HashSet<usize> = HashSet::new();
    let mut flag_boolean_row: HashMap<usize, usize> = HashMap::new();
    let mut selector_rows: HashSet<usize> = HashSet::new();
    for row in 0..n {
        if c_has_nonzero[row] {
            continue;
        }
        if a_count[row] != 1 || a_val1[row] != F::ONE {
            continue;
        }
        let flag_col = a_col1[row];
        if b_count[row] != 2 {
            continue;
        }

        let (c1, v1) = (b_col1[row], b_val1[row]);
        let (c2, v2) = (b_col2[row], b_val2[row]);

        let ok = (c1 == flag_col && v1 == F::ONE && const_one_cols.contains(&c2) && v2 == -F::ONE)
            || (c2 == flag_col && v2 == F::ONE && const_one_cols.contains(&c1) && v1 == -F::ONE);
        if !ok {
            continue;
        }

        flag_is_boolean.insert(flag_col);
        flag_boolean_row.entry(flag_col).or_insert(row);
        selector_rows.insert(row);
    }

    // Explicit padding constraints present in CCS: (1 - flag) * field = 0.
    let mut present: HashSet<(usize, usize)> = HashSet::new();
    let mut padding_rows: HashSet<usize> = HashSet::new();
    for row in 0..n {
        if c_has_nonzero[row] {
            continue;
        }
        if b_count[row] != 1 || b_val1[row] != F::ONE {
            continue;
        }
        let field_col = b_col1[row];
        if a_count[row] != 2 {
            continue;
        }
        let (c1, v1) = (a_col1[row], a_val1[row]);
        let (c2, v2) = (a_col2[row], a_val2[row]);
        if v1 != -v2 {
            continue;
        }

        let flag_col = if const_one_cols.contains(&c1) {
            Some(c2)
        } else if const_one_cols.contains(&c2) {
            Some(c1)
        } else {
            None
        };
        let Some(flag_col) = flag_col else {
            continue;
        };

        present.insert((flag_col, field_col));
        padding_rows.insert(row);
    }

    // Bitness constraints that imply address-bit padding: bit * (bit - flag) = 0.
    //
    // We only treat these as satisfying padding when `flag` is also boolean.
    let mut implied: HashMap<(usize, usize), usize> = HashMap::new();
    for row in 0..n {
        if c_has_nonzero[row] {
            continue;
        }
        if a_count[row] != 1 || a_val1[row] != F::ONE {
            continue;
        }
        let bit_col = a_col1[row];
        if b_count[row] != 2 {
            continue;
        }

        let (c1, v1) = (b_col1[row], b_val1[row]);
        let (c2, v2) = (b_col2[row], b_val2[row]);

        let flag_col = if c1 == bit_col && v1 == F::ONE && v2 == -F::ONE {
            Some(c2)
        } else if c2 == bit_col && v2 == F::ONE && v1 == -F::ONE {
            Some(c1)
        } else {
            None
        };
        let Some(flag_col) = flag_col else {
            continue;
        };
        if !flag_is_boolean.contains(&flag_col) {
            continue;
        }

        implied.insert((flag_col, bit_col), row);
    }

    let mut missing: Vec<&BusPaddingLabel> = Vec::new();
    for req in required {
        if present.contains(&(req.flag_z_idx, req.field_z_idx)) {
            continue;
        }
        if implied.contains_key(&(req.flag_z_idx, req.field_z_idx)) {
            continue;
        }
        if !present.contains(&(req.flag_z_idx, req.field_z_idx)) {
            missing.push(req);
        }
    }

    if missing.is_empty() {
        // Treat selector boolean and bitness constraints as padding-only rows so they don't
        // accidentally satisfy "binding" presence checks.
        for row in selector_rows {
            padding_rows.insert(row);
        }
        for &row in implied.values() {
            padding_rows.insert(row);
        }
        for &row in flag_boolean_row.values() {
            padding_rows.insert(row);
        }
        return Ok(padding_rows);
    }

    let mut examples: Vec<String> = missing
        .iter()
        .take(8)
        .map(|c| format!("{}", c.label))
        .collect();
    if missing.len() > examples.len() {
        examples.push(format!("... ({} more)", missing.len() - examples.len()));
    }

    Err(PiCcsError::InvalidInput(format!(
        "shared_cpu_bus=true but CPU CCS is missing required padding constraints that force inactive bus fields to zero.\n\
         This is a common footgun: Twist/Shout gate checks by has_* flags, so unconstrained bus fields become arbitrary degrees of freedom.\n\
         Fix: inject the canonical shared-bus constraints (binding + padding) using `neo_memory::cpu::constraints::extend_ccs_with_shared_cpu_bus_constraints`, \
         or add constraints of the form (1 - has_*) * field = 0 for each gated bus field (recommended: use `neo_memory::cpu::constraints::CpuConstraintBuilder`).\n\
         Missing examples: {}",
        examples.join(", ")
    )))
}

fn ensure_ccs_binds_shared_bus_for_steps(
    s: &CcsStructure<F>,
    bus: &BusLayout,
    padding_rows: &HashSet<usize>,
) -> Result<(), PiCcsError> {
    if bus.bus_cols == 0 {
        return Ok(());
    }
    let required = required_bus_cols_for_layout(bus);
    ensure_ccs_references_bus_cols(s, bus, &required)?;

    let binding_required = required_bus_binding_cols_for_layout(bus);
    ensure_ccs_references_bus_cols_outside_padding_rows(s, bus, padding_rows, &binding_required)
}

fn ensure_ccs_has_shared_bus_padding_for_steps<Cmt, S: BusStepView<Cmt>>(
    s: &CcsStructure<F>,
    bus: &BusLayout,
    steps: &[S],
) -> Result<HashSet<usize>, PiCcsError> {
    if steps.is_empty() || bus.bus_cols == 0 {
        return Ok(HashSet::new());
    }
    let const_one_cols = infer_public_constant_one_cols_from_steps(steps);
    let required = required_bus_padding_for_layout(bus);
    ensure_ccs_has_bus_padding_constraints(s, bus, &const_one_cols, &required)
}

pub(crate) fn decode_cpu_z_to_k(params: &NeoParams, Z: &Mat<F>) -> Vec<K> {
    ajtai_decode_vector(params, Z)
        .into_iter()
        .map(Into::into)
        .collect()
}

pub(crate) fn build_time_sparse_from_bus_col(
    z: &[K],
    bus: &BusLayout,
    col_id: usize,
    steps_len: usize,
    pow2_cycle: usize,
) -> Result<SparseIdxVec<K>, PiCcsError> {
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
    let mut entries: Vec<(usize, K)> = Vec::new();
    for j in 0..bus.chunk_size {
        let t = bus.time_index(j);
        if t >= pow2_cycle {
            return Err(PiCcsError::InvalidInput(format!(
                "bus time index out of range: t={t} >= pow2_cycle={pow2_cycle}"
            )));
        }
        if j >= steps_len {
            continue;
        }
        let idx = bus.bus_cell(col_id, j);
        let v = z
            .get(idx)
            .copied()
            .ok_or_else(|| PiCcsError::InvalidInput(format!("CPU witness too short for bus idx={idx}")))?;
        if v != K::ZERO {
            entries.push((t, v));
        }
    }
    Ok(SparseIdxVec::from_entries(pow2_cycle, entries))
}

pub(crate) fn build_time_sparse_from_bus_col_at_js(
    z: &[K],
    bus: &BusLayout,
    col_id: usize,
    js: &[usize],
    pow2_cycle: usize,
) -> Result<SparseIdxVec<K>, PiCcsError> {
    if col_id >= bus.bus_cols {
        return Err(PiCcsError::InvalidInput(format!(
            "bus col_id out of range: {col_id} >= {}",
            bus.bus_cols
        )));
    }
    let mut entries: Vec<(usize, K)> = Vec::new();
    for &j in js {
        if j >= bus.chunk_size {
            return Err(PiCcsError::InvalidInput(format!(
                "bus j out of range: j={j} >= bus.chunk_size={}",
                bus.chunk_size
            )));
        }
        let t = bus.time_index(j);
        if t >= pow2_cycle {
            return Err(PiCcsError::InvalidInput(format!(
                "bus time index out of range: t={t} >= pow2_cycle={pow2_cycle}"
            )));
        }
        let idx = bus.bus_cell(col_id, j);
        let v = z
            .get(idx)
            .copied()
            .ok_or_else(|| PiCcsError::InvalidInput(format!("CPU witness too short for bus idx={idx}")))?;
        if v != K::ZERO {
            entries.push((t, v));
        }
    }
    Ok(SparseIdxVec::from_entries(pow2_cycle, entries))
}
