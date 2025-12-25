use crate::builder::CpuArithmetization;
use crate::plain::LutTable;
use crate::plain::PlainMemLayout;
use neo_ajtai::{decomp_b, DecompStyle};
use neo_ccs::matrix::Mat;
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_params::NeoParams;
use neo_vm_trace::{StepTrace, VmTrace};
use p3_field::{PrimeCharacteristicRing, PrimeField, PrimeField64};
use p3_goldilocks::Goldilocks;
use std::collections::HashMap;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct SharedCpuBusConfig<F> {
    /// Geometry for each Twist instance (twist_id -> layout).
    pub mem_layouts: HashMap<u32, PlainMemLayout>,
    /// Sparse initial memory values: (twist_id, addr) -> value.
    pub initial_mem: HashMap<(u32, u64), F>,
}

/// Adapter that implements CpuArithmetization for a generic R1CS-based CPU.
///
/// It assumes the user provides:
/// 1. The CCS structure (wrapped R1CS matrices).
/// 2. A witness builder function that maps a `StepTrace` to the R1CS witness vector `w`.
/// 3. Shout tables to verify trace correctness.
pub struct R1csCpu<F, Cmt, L>
where
    F: PrimeField + PrimeField64,
    L: SModuleHomomorphism<F, Cmt>,
{
    pub ccs: CcsStructure<F>,
    pub params: NeoParams,
    pub committer: L,

    /// Number of public inputs m_in (prefix of z treated as x).
    /// The witness z is split as z = (x[0..m_in], w[m_in..]).
    pub m_in: usize,

    /// Cached Shout tables for verification: shout_id -> (key -> val)
    /// Used to ensure trace Shout lookups are valid against the table.
    pub shout_cache: HashMap<u32, HashMap<u64, F>>,

    /// Shout table metadata needed to write the shared CPU bus (d, n_side).
    pub shout_meta: HashMap<u32, (usize, usize)>,

    /// Optional shared CPU-bus configuration.
    /// When present, we overwrite a reserved tail segment of `z_vec` with Twist/Shout access rows
    /// (in deterministic id order) before Ajtai decomposition + commitment.
    pub shared_cpu_bus: Option<SharedCpuBusConfig<F>>,

    /// Function to map a step trace to the full witness z = (x, w).
    /// The witness MUST satisfy the CCS relation.
    pub step_to_witness: Box<dyn Fn(&StepTrace<u64, u64>) -> Vec<F> + Send + Sync>,

    _phantom: PhantomData<Cmt>,
}

impl<F, Cmt, L> R1csCpu<F, Cmt, L>
where
    F: PrimeField + PrimeField64,
    L: SModuleHomomorphism<F, Cmt>,
{
    pub fn new(
        ccs: CcsStructure<F>,
        params: NeoParams,
        committer: L,
        m_in: usize,
        tables: &HashMap<u32, LutTable<F>>,
        step_to_witness: Box<dyn Fn(&StepTrace<u64, u64>) -> Vec<F> + Send + Sync>,
    ) -> Self {
        // Build fast Shout cache from tables
        let mut shout_cache = HashMap::new();
        let mut shout_meta = HashMap::new();
        for (id, table) in tables {
            // Assuming table content is dense (val = content[addr])
            // LutTable currently has `content: Vec<F>`.
            // We'll map key -> val.
            let mut map = HashMap::new();
            for (key, val) in table.content.iter().enumerate() {
                map.insert(key as u64, *val);
            }
            shout_cache.insert(*id, map);
            shout_meta.insert(*id, (table.d, table.n_side));
        }

        Self {
            ccs,
            params,
            committer,
            m_in,
            shout_cache,
            shout_meta,
            shared_cpu_bus: None,
            step_to_witness,
            _phantom: PhantomData,
        }
    }

    pub fn with_shared_cpu_bus(mut self, cfg: SharedCpuBusConfig<F>) -> Self {
        self.shared_cpu_bus = Some(cfg);
        self
    }
}

// R1csCpu implementation specifically for Goldilocks field because neo_ajtai::decomp_b uses Goldilocks
// The trait impl needs to be generic F, but decomp_b expects &[Fq] (Goldilocks).
// We need to cast F to Goldilocks if possible, or restrict F to Goldilocks.
// Given neo-ajtai is hardcoded to Goldilocks (Fq), we should restrict here.

impl<Cmt, L> CpuArithmetization<Goldilocks, Cmt> for R1csCpu<Goldilocks, Cmt, L>
where
    L: SModuleHomomorphism<Goldilocks, Cmt>,
{
    fn build_ccs_chunks(
        &self,
        trace: &VmTrace<u64, u64>,
        chunk_size: usize,
    ) -> Result<Vec<(McsInstance<Cmt, Goldilocks>, McsWitness<Goldilocks>)>, Self::Error> {
        if chunk_size != 1 {
            return Err(format!(
                "R1csCpu does not support chunk_size={} (expected 1)",
                chunk_size
            ));
        }
        self.build_ccs_steps(trace)
    }

    fn build_ccs_steps(
        &self,
        trace: &VmTrace<u64, u64>,
    ) -> Result<Vec<(McsInstance<Cmt, Goldilocks>, McsWitness<Goldilocks>)>, Self::Error> {
        let mut mcss = Vec::with_capacity(trace.steps.len());

        // Shared CPU-bus bookkeeping (optional).
        let (table_ids, mem_ids, mut mem_state, bus_base) = if let Some(bus) = &self.shared_cpu_bus {
            let mut table_ids: Vec<u32> = self.shout_cache.keys().copied().collect();
            table_ids.sort_unstable();
            let mut mem_ids: Vec<u32> = bus.mem_layouts.keys().copied().collect();
            mem_ids.sort_unstable();

            // Seed per-memory sparse state with initial values.
            let mut mem_state: HashMap<u32, HashMap<u64, Goldilocks>> = HashMap::new();
            for (&mem_id, layout) in &bus.mem_layouts {
                let mut st = HashMap::<u64, Goldilocks>::new();
                for ((init_mem_id, addr), &val) in bus.initial_mem.iter() {
                    if *init_mem_id == mem_id && val != Goldilocks::ZERO {
                        if (*addr as usize) >= layout.k {
                            return Err(format!(
                                "shared_cpu_bus: initial_mem out of range for mem_id={mem_id}: addr={addr} >= k={}",
                                layout.k
                            ));
                        }
                        if st.insert(*addr, val).is_some() {
                            return Err(format!(
                                "shared_cpu_bus: initial_mem contains duplicate addr={addr} for mem_id={mem_id}"
                            ));
                        }
                    }
                }
                mem_state.insert(mem_id, st);
            }

            // Compute total bus column count in canonical order: Shout then Twist.
            let mut bus_cols_total = 0usize;
            for table_id in &table_ids {
                let (d, n_side) = self
                    .shout_meta
                    .get(table_id)
                    .copied()
                    .ok_or_else(|| format!("missing shout_meta for table_id={table_id}"))?;
                if n_side == 0 || !n_side.is_power_of_two() {
                    return Err(format!("shared_cpu_bus: shout n_side must be power-of-two, got {n_side}"));
                }
                let ell = n_side.trailing_zeros() as usize;
                bus_cols_total = bus_cols_total
                    .checked_add(d * ell + 2)
                    .ok_or_else(|| "shared_cpu_bus: bus_cols overflow".to_string())?;
            }
            for mem_id in &mem_ids {
                let layout = bus
                    .mem_layouts
                    .get(mem_id)
                    .ok_or_else(|| format!("missing mem_layout for mem_id={mem_id}"))?;
                if layout.n_side == 0 || !layout.n_side.is_power_of_two() {
                    return Err(format!(
                        "shared_cpu_bus: twist n_side must be power-of-two, got {}",
                        layout.n_side
                    ));
                }
                let ell = layout.n_side.trailing_zeros() as usize;
                bus_cols_total = bus_cols_total
                    .checked_add(2 * layout.d * ell + 5)
                    .ok_or_else(|| "shared_cpu_bus: bus_cols overflow".to_string())?;
            }

            if bus_cols_total > self.ccs.m {
                return Err(format!(
                    "shared_cpu_bus: bus region too large: bus_cols({bus_cols_total}) > ccs.m({})",
                    self.ccs.m
                ));
            }
            let bus_base = self.ccs.m - bus_cols_total;
            if bus_base < self.m_in {
                return Err(format!(
                    "shared_cpu_bus: bus_base({bus_base}) overlaps public input region m_in({})",
                    self.m_in
                ));
            }

            (Some(table_ids), Some(mem_ids), Some(mem_state), Some(bus_base))
        } else {
            (None, None, None, None)
        };

        for step in &trace.steps {
            // 1. Verify Shout lookups in this step against the table cache
            for shout in &step.shout_events {
                if let Some(table) = self.shout_cache.get(&shout.shout_id.0) {
                    // Check val == table[key]
                    if let Some(&expected) = table.get(&shout.key) {
                        let trace_val = Goldilocks::from_u64(shout.value);
                        if trace_val != expected {
                            return Err(format!(
                                "Shout mismatch at step pc={:?}: shout {} key {} has {}, trace has {}",
                                step.pc_before,
                                shout.shout_id.0,
                                shout.key,
                                expected.as_canonical_u64(),
                                trace_val.as_canonical_u64()
                            ));
                        }
                    } else {
                        return Err(format!(
                            "Shout key {} not found in shout {}",
                            shout.key, shout.shout_id.0
                        ));
                    }
                } else {
                    return Err(format!("Missing Shout table for shout_id {}", shout.shout_id.0));
                }
            }

            // 2. Build witness z
            let mut z_vec = (self.step_to_witness)(step);

            // Allow witness builders to omit trailing dummy variables (including the shared-bus tail).
            if z_vec.len() < self.m_in {
                return Err(format!(
                    "Witness length {} is shorter than m_in={}",
                    z_vec.len(),
                    self.m_in
                ));
            }
            if z_vec.len() > self.ccs.m {
                return Err(format!(
                    "Witness length {} exceeds CCS width {}",
                    z_vec.len(),
                    self.ccs.m
                ));
            }
            if z_vec.len() != self.ccs.m {
                z_vec.resize(self.ccs.m, Goldilocks::ZERO);
            }

            // 2.1 Overwrite the shared CPU-bus tail (if enabled).
            if let (Some(table_ids), Some(mem_ids), Some(mem_state), Some(bus_base), Some(bus)) = (
                table_ids.as_ref(),
                mem_ids.as_ref(),
                mem_state.as_mut(),
                bus_base,
                self.shared_cpu_bus.as_ref(),
            ) {
                // Index in the bus tail (col-major, chunk_size=1 => contiguous).
                let mut off = 0usize;
                let expected_bus_len = self.ccs.m - bus_base;

                // Zero the bus tail so that inactive instances/ports are guaranteed zero.
                for i in 0..expected_bus_len {
                    z_vec[bus_base + i] = Goldilocks::ZERO;
                }

                // Build per-step maps for fast lookup.
                let mut shout_by_id: HashMap<u32, (u64, Goldilocks)> = HashMap::new();
                for ev in &step.shout_events {
                    let id = ev.shout_id.0;
                    if shout_by_id
                        .insert(id, (ev.key, Goldilocks::from_u64(ev.value)))
                        .is_some()
                    {
                        return Err(format!("multiple shout events for shout_id={id} in one step"));
                    }
                }

                let mut read_by_id: HashMap<u32, (u64, Goldilocks)> = HashMap::new();
                let mut write_by_id: HashMap<u32, (u64, Goldilocks)> = HashMap::new();
                for ev in &step.twist_events {
                    let id = ev.twist_id.0;
                    match ev.kind {
                        neo_vm_trace::TwistOpKind::Read => {
                            if read_by_id
                                .insert(id, (ev.addr, Goldilocks::from_u64(ev.value)))
                                .is_some()
                            {
                                return Err(format!("multiple twist reads for twist_id={id} in one step"));
                            }
                        }
                        neo_vm_trace::TwistOpKind::Write => {
                            if write_by_id
                                .insert(id, (ev.addr, Goldilocks::from_u64(ev.value)))
                                .is_some()
                            {
                                return Err(format!("multiple twist writes for twist_id={id} in one step"));
                            }
                        }
                    }
                }

                // Helper: write dim-major, bit-minor, little-endian bits for `addr`.
                fn write_addr_bits_dim_major_le(
                    z: &mut [Goldilocks],
                    bus_base: usize,
                    off: &mut usize,
                    addr: u64,
                    d: usize,
                    n_side: usize,
                    ell: usize,
                ) {
                    let mut tmp = addr;
                    for _dim in 0..d {
                        let comp = (tmp % (n_side as u64)) as u64;
                        tmp /= n_side as u64;
                        for bit in 0..ell {
                            z[bus_base + *off] = if (comp >> bit) & 1 == 1 {
                                Goldilocks::ONE
                            } else {
                                Goldilocks::ZERO
                            };
                            *off += 1;
                        }
                    }
                }

                // Shout: addr_bits, has_lookup, val.
                for table_id in table_ids {
                    let (d, n_side) = self
                        .shout_meta
                        .get(table_id)
                        .copied()
                        .ok_or_else(|| format!("missing shout_meta for table_id={table_id}"))?;
                    let ell = n_side.trailing_zeros() as usize;

                    if let Some((key, val)) = shout_by_id.get(table_id).copied() {
                        write_addr_bits_dim_major_le(&mut z_vec, bus_base, &mut off, key, d, n_side, ell);
                        z_vec[bus_base + off] = Goldilocks::ONE;
                        off += 1;
                        z_vec[bus_base + off] = val;
                        off += 1;
                    } else {
                        // addr bits are 0
                        off += d * ell;
                        z_vec[bus_base + off] = Goldilocks::ZERO; // has_lookup
                        off += 1;
                        z_vec[bus_base + off] = Goldilocks::ZERO; // val
                        off += 1;
                    }
                }

                // Twist: ra_bits, wa_bits, has_read, has_write, wv, rv, inc_at_write_addr.
                for mem_id in mem_ids {
                    let layout = bus
                        .mem_layouts
                        .get(mem_id)
                        .ok_or_else(|| format!("missing mem_layout for mem_id={mem_id}"))?;
                    let ell = layout.n_side.trailing_zeros() as usize;

                    // Read port.
                    let (has_read, ra, rv) = if let Some((addr, val)) = read_by_id.get(mem_id).copied() {
                        (Goldilocks::ONE, addr, val)
                    } else {
                        (Goldilocks::ZERO, 0u64, Goldilocks::ZERO)
                    };
                    if has_read == Goldilocks::ONE {
                        write_addr_bits_dim_major_le(&mut z_vec, bus_base, &mut off, ra, layout.d, layout.n_side, ell);
                    } else {
                        off += layout.d * ell;
                    }

                    // Write port.
                    let (has_write, wa, wv) = if let Some((addr, val)) = write_by_id.get(mem_id).copied() {
                        (Goldilocks::ONE, addr, val)
                    } else {
                        (Goldilocks::ZERO, 0u64, Goldilocks::ZERO)
                    };
                    if has_write == Goldilocks::ONE {
                        write_addr_bits_dim_major_le(&mut z_vec, bus_base, &mut off, wa, layout.d, layout.n_side, ell);
                    } else {
                        off += layout.d * ell;
                    }

                    // Flags + vals + inc.
                    z_vec[bus_base + off] = has_read;
                    off += 1;
                    z_vec[bus_base + off] = has_write;
                    off += 1;
                    z_vec[bus_base + off] = wv;
                    off += 1;
                    z_vec[bus_base + off] = rv;
                    off += 1;

                    let mut inc = Goldilocks::ZERO;
                    if has_write == Goldilocks::ONE && (wa as usize) < layout.k {
                        let st = mem_state.entry(*mem_id).or_default();
                        let old = st.get(&wa).copied().unwrap_or(Goldilocks::ZERO);
                        inc = wv - old;
                        if wv == Goldilocks::ZERO {
                            st.remove(&wa);
                        } else {
                            st.insert(wa, wv);
                        }
                    }
                    z_vec[bus_base + off] = inc;
                    off += 1;
                }

                // Sanity: we must fill exactly the reserved bus tail.
                if off != expected_bus_len {
                    return Err(format!(
                        "shared_cpu_bus internal error: wrote {} bus entries, expected {}",
                        off, expected_bus_len
                    ));
                }
            }

            // 3. Decompose z -> Z matrix
            let d = self.params.d as usize;
            let m = z_vec.len(); // == ccs.m after padding

            // Validate m_in
            let m_in = self.m_in;
            if m_in > m {
                return Err(format!("m_in={} exceeds witness length m={}", m_in, m));
            }

            // Decompose: Z is d x m
            let z_digits = decomp_b(&z_vec, self.params.b, d, DecompStyle::Balanced);

            // Convert to Mat (row-major d x m)
            // decomp_b returns digits "per element", i.e. column-major for the (d × m) matrix:
            // z_digits[c*d + r] = digit r (row) of value c (column).
            let mut mat_data = vec![Goldilocks::ZERO; d * m];
            for c in 0..m {
                for r in 0..d {
                    mat_data[r * m + c] = z_digits[c * d + r];
                }
            }
            let z_mat = Mat::from_row_major(d, m, mat_data);

            // 4. Commit to Z
            let c = self.committer.commit(&z_mat);

            // 5. Build Instance/Witness
            // Split z into public inputs x and private witness w
            let x = z_vec[..m_in].to_vec();
            let w = z_vec[m_in..].to_vec();

            mcss.push((McsInstance { c, x, m_in }, McsWitness { w, Z: z_mat }));
        }

        Ok(mcss)
    }
    type Error = String;
}
