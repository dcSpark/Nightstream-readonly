use super::*;

pub(crate) fn sparse_trace_col_from_values(
    m_in: usize,
    ell_n: usize,
    values: &[K],
) -> Result<SparseIdxVec<K>, PiCcsError> {
    let pow2_cycle = 1usize
        .checked_shl(ell_n as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("WB/WP: 2^ell_n overflow".into()))?;
    let t_len = values.len();
    if m_in
        .checked_add(t_len)
        .ok_or_else(|| PiCcsError::InvalidInput("WB/WP: m_in + t_len overflow".into()))?
        > pow2_cycle
    {
        return Err(PiCcsError::InvalidInput(format!(
            "WB/WP: trace rows out of range (m_in={m_in}, t_len={t_len}, 2^ell_n={pow2_cycle})"
        )));
    }
    let mut entries = Vec::new();
    for (j, &v) in values.iter().enumerate() {
        if v != K::ZERO {
            entries.push((m_in + j, v));
        }
    }
    Ok(SparseIdxVec::from_entries(pow2_cycle, entries))
}

#[inline]
pub(crate) fn decode_k_to_u32(v: K, ctx: &str) -> Result<u32, PiCcsError> {
    let coeffs = v.as_coeffs();
    if coeffs.iter().skip(1).any(|&c| c != F::ZERO) {
        return Err(PiCcsError::ProtocolError(format!(
            "{ctx}: expected base-field value while decoding shared decode columns"
        )));
    }
    let lo = coeffs
        .first()
        .copied()
        .ok_or_else(|| PiCcsError::ProtocolError(format!("{ctx}: missing base coefficient")))?
        .as_canonical_u64();
    if lo > u32::MAX as u64 {
        return Err(PiCcsError::ProtocolError(format!(
            "{ctx}: value {lo} exceeds u32 range while decoding shared decode columns"
        )));
    }
    Ok(lo as u32)
}

pub(crate) fn resolve_shared_decode_lookup_lut_indices(
    step: &StepWitnessBundle<Cmt, F, K>,
    decode_layout: &Rv32DecodeSidecarLayout,
) -> Result<(Vec<usize>, Vec<usize>), PiCcsError> {
    let decode_open_cols = rv32_decode_lookup_backed_cols(decode_layout);
    let mut decode_lut_indices = Vec::with_capacity(decode_open_cols.len());
    for &col_id in decode_open_cols.iter() {
        let table_id = rv32_decode_lookup_table_id_for_col(col_id);
        let idx = step
            .lut_instances
            .iter()
            .position(|(inst, _)| inst.table_id == table_id)
            .ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "W2(shared): missing decode lookup table_id={table_id} for col_id={col_id}"
                ))
            })?;
        decode_lut_indices.push(idx);
    }

    Ok((decode_open_cols, decode_lut_indices))
}

pub(crate) struct WeightedMaskOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    active: SparseIdxVec<K>,
    cols: Vec<SparseIdxVec<K>>,
    weights: Vec<K>,
}

impl WeightedMaskOracleSparseTime {
    pub(crate) fn new(active: SparseIdxVec<K>, cols: Vec<SparseIdxVec<K>>, weights: Vec<K>, r_cycle: &[K]) -> Self {
        debug_assert_eq!(cols.len(), weights.len());
        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            active,
            cols,
            weights,
        }
    }
}

impl RoundOracle for WeightedMaskOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.cols.is_empty() {
            return vec![K::ZERO; points.len()];
        }

        if self.active.len() == 1 {
            let gate = K::ONE - self.active.singleton_value();
            let mut acc = K::ZERO;
            for (col, w) in self.cols.iter().zip(self.weights.iter()) {
                acc += *w * col.singleton_value();
            }
            return vec![self.prefix_eq * gate * acc; points.len()];
        }

        let mut pairs = gather_pairs_from_sparse(self.active.entries());
        for col in self.cols.iter() {
            pairs.extend(gather_pairs_from_sparse(col.entries()));
        }
        pairs.sort_unstable();
        pairs.dedup();
        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = K::ONE - self.active.get(child0);
            let gate1 = K::ONE - self.active.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);
            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }
                let mut sum_x = K::ZERO;
                for (col, w) in self.cols.iter().zip(self.weights.iter()) {
                    let c0 = col.get(child0);
                    let c1 = col.get(child1);
                    if c0 == K::ZERO && c1 == K::ZERO {
                        continue;
                    }
                    sum_x += *w * interp(c0, c1, x);
                }
                ys[i] += chi_x * gate_x * sum_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single_k(r, self.r_cycle[self.bit_idx]);
        self.active.fold_round_in_place(r);
        for col in self.cols.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.bit_idx += 1;
    }
}

pub(crate) struct FormulaOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    cols: Vec<SparseIdxVec<K>>,
    degree_bound: usize,
    eval_fn: Box<dyn Fn(&[K]) -> K>,
}

impl FormulaOracleSparseTime {
    pub(crate) fn new(
        cols: Vec<SparseIdxVec<K>>,
        degree_bound: usize,
        r_cycle: &[K],
        eval_fn: Box<dyn Fn(&[K]) -> K>,
    ) -> Self {
        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            cols,
            degree_bound,
            eval_fn,
        }
    }
}

impl RoundOracle for FormulaOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.cols.is_empty() {
            return vec![K::ZERO; points.len()];
        }

        let mut pairs = Vec::new();
        for col in self.cols.iter() {
            pairs.extend(gather_pairs_from_sparse(col.entries()));
        }
        pairs.sort_unstable();
        pairs.dedup();

        let mut ys = vec![K::ZERO; points.len()];
        let mut vals = vec![K::ZERO; self.cols.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;
            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);
            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                for (j, col) in self.cols.iter().enumerate() {
                    vals[j] = interp(col.get(child0), col.get(child1), x);
                }
                let f_x = (self.eval_fn)(&vals);
                if f_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * f_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single_k(r, self.r_cycle[self.bit_idx]);
        for col in self.cols.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.bit_idx += 1;
    }
}

#[inline]
pub(crate) fn unpack_interleaved_halves_lsb(addr_bits: &[K]) -> Result<(K, K), PiCcsError> {
    if !addr_bits.len().is_multiple_of(2) {
        return Err(PiCcsError::InvalidInput(format!(
            "shout linkage expects even ell_addr, got {}",
            addr_bits.len()
        )));
    }
    let half_len = addr_bits.len() / 2;
    let two = K::from(F::from_u64(2));
    let mut pow = K::ONE;
    let mut lhs = K::ZERO;
    let mut rhs = K::ZERO;
    for k in 0..half_len {
        lhs += pow * addr_bits[2 * k];
        rhs += pow * addr_bits[2 * k + 1];
        pow *= two;
    }
    Ok((lhs, rhs))
}

pub(crate) fn extract_trace_cpu_link_openings(
    m: usize,
    core_t: usize,
    y_prefix_cols: usize,
    step: &StepInstanceBundle<Cmt, F, K>,
    ccs_out0: &MeInstance<Cmt, F, K>,
) -> Result<Option<TraceCpuLinkOpenings>, PiCcsError> {
    if step.mem_insts.is_empty() && step.lut_insts.is_empty() {
        return Ok(None);
    }

    // RV32 trace linkage: the prover appends time-combined openings for selected CPU trace columns
    // to the CCS ME output at r_time. We use those to bind Twist instances (PROG/REG/RAM) to the
    // same trace, without embedding a shared CPU bus tail.
    let trace = Rv32TraceLayout::new();
    let trace_cols_to_open: Vec<usize> = vec![
        trace.active,
        trace.cycle,
        trace.pc_before,
        trace.instr_word,
        trace.rs1_addr,
        trace.rs1_val,
        trace.rs2_addr,
        trace.rs2_val,
        trace.rd_addr,
        trace.rd_val,
        trace.ram_addr,
        trace.ram_rv,
        trace.ram_wv,
        trace.shout_has_lookup,
        trace.shout_val,
        trace.shout_lhs,
        trace.shout_rhs,
    ];

    let m_in = step.mcs_inst.m_in;
    let t_len = step
        .mem_insts
        .first()
        .map(|inst| inst.steps)
        .or_else(|| {
            // Shout event-table instances may have `steps != t_len`; prefer a non-event-table
            // instance if present, otherwise fall back to inferring from the trace layout.
            step.lut_insts
                .iter()
                .find(|inst| !matches!(inst.table_spec, Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. })))
                .map(|inst| inst.steps)
        })
        .or_else(|| {
            // Trace CCS layout inference: z = [x (m_in) | trace_cols * t_len]
            let w = m.checked_sub(m_in)?;
            if trace.cols == 0 || w % trace.cols != 0 {
                return None;
            }
            Some(w / trace.cols)
        })
        .ok_or_else(|| PiCcsError::InvalidInput("missing mem/lut instances".into()))?;
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput("trace linkage requires steps>=1".into()));
    }
    for (i, inst) in step.mem_insts.iter().enumerate() {
        if inst.steps != t_len {
            return Err(PiCcsError::InvalidInput(format!(
                "trace linkage requires stable steps across mem instances (mem_idx={i} has steps={}, expected {t_len})",
                inst.steps
            )));
        }
    }
    let trace_len = trace
        .cols
        .checked_mul(t_len)
        .ok_or_else(|| PiCcsError::InvalidInput("trace cols * t_len overflow".into()))?;
    let expected_m = m_in
        .checked_add(trace_len)
        .ok_or_else(|| PiCcsError::InvalidInput("m_in + trace_len overflow".into()))?;
    if m < expected_m {
        return Err(PiCcsError::InvalidInput(format!(
            "trace linkage expects m >= m_in + trace.cols*t_len (m={}; min_m={expected_m} for t_len={t_len}, trace_cols={})",
            m, trace.cols
        )));
    }
    let expected_y_len = core_t
        .checked_add(y_prefix_cols)
        .and_then(|v| v.checked_add(trace_cols_to_open.len()))
        .ok_or_else(|| PiCcsError::InvalidInput("core_t + y_prefix_cols + trace_openings overflow".into()))?;
    if ccs_out0.y_scalars.len() != expected_y_len {
        return Err(PiCcsError::InvalidInput(format!(
            "trace linkage expects CPU ME output to contain exactly core_t + y_prefix_cols + trace_openings y_scalars (have {}, expected {expected_y_len})",
            ccs_out0.y_scalars.len(),
        )));
    }
    let cpu_open = |idx: usize| -> Result<K, PiCcsError> {
        ccs_out0
            .y_scalars
            .get(core_t + y_prefix_cols + idx)
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError("missing CPU trace linkage opening".into()))
    };

    Ok(Some(TraceCpuLinkOpenings {
        shout_has_lookup: cpu_open(13)?,
        shout_val: cpu_open(14)?,
        shout_lhs: cpu_open(15)?,
        shout_rhs: cpu_open(16)?,
    }))
}

pub(crate) fn expected_trace_shout_table_id_from_openings(
    core_t: usize,
    step: &StepInstanceBundle<Cmt, F, K>,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    r_time: &[K],
) -> Result<K, PiCcsError> {
    if !decode_stage_required_for_step_instance(step) {
        return Ok(K::ZERO);
    }

    if mem_proof.wp_me_claims.len() != 1 {
        return Err(PiCcsError::ProtocolError(
            "decode-linked Shout table_id check requires one WP ME claim".into(),
        ));
    }
    let wp_me = &mem_proof.wp_me_claims[0];
    if wp_me.r.as_slice() != r_time {
        return Err(PiCcsError::ProtocolError(
            "decode-linked Shout table_id check: WP ME r mismatch".into(),
        ));
    }
    if wp_me.c != step.mcs_inst.c {
        return Err(PiCcsError::ProtocolError(
            "decode-linked Shout table_id check: WP ME commitment mismatch".into(),
        ));
    }
    if wp_me.m_in != step.mcs_inst.m_in {
        return Err(PiCcsError::ProtocolError(
            "decode-linked Shout table_id check: WP ME m_in mismatch".into(),
        ));
    }

    let trace = Rv32TraceLayout::new();
    let decode_layout = Rv32DecodeSidecarLayout::new();
    let wp_cols = rv32_trace_wp_opening_columns(&trace);
    let control_extra_cols = if control_stage_required_for_step_instance(step) {
        rv32_trace_control_extra_opening_columns(&trace)
    } else {
        Vec::new()
    };
    let decode_open_cols = rv32_decode_lookup_backed_cols(&decode_layout);

    let decode_open_start = core_t
        .checked_add(wp_cols.len())
        .and_then(|v| v.checked_add(control_extra_cols.len()))
        .ok_or_else(|| {
            PiCcsError::InvalidInput("decode-linked Shout table_id check: decode_open_start overflow".into())
        })?;
    let decode_open_end = decode_open_start
        .checked_add(decode_open_cols.len())
        .ok_or_else(|| {
            PiCcsError::InvalidInput("decode-linked Shout table_id check: decode_open_end overflow".into())
        })?;
    if wp_me.y_scalars.len() < decode_open_end {
        return Err(PiCcsError::ProtocolError(format!(
            "decode-linked Shout table_id check: missing decode openings (got {}, need at least {decode_open_end})",
            wp_me.y_scalars.len()
        )));
    }

    let decode_open = &wp_me.y_scalars[decode_open_start..decode_open_end];
    let decode_open_col = |col_id: usize| -> Result<K, PiCcsError> {
        let idx = decode_open_cols
            .iter()
            .position(|&c| c == col_id)
            .ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "decode-linked Shout table_id check: missing decode opening col {col_id}"
                ))
            })?;
        Ok(decode_open[idx])
    };

    Ok(decode_open_col(decode_layout.shout_table_id)?)
}

pub(crate) fn prove_twist_addr_pre_time(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    cpu_bus: &BusLayout,
    ell_n: usize,
    r_cycle: &[K],
) -> Result<Vec<TwistAddrPreProverData>, PiCcsError> {
    if step.mem_instances.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(step.mem_instances.len());

    let cpu_z_k = crate::memory_sidecar::cpu_bus::decode_cpu_z_to_k(params, &step.mcs.1.Z);
    if cpu_bus.shout_cols.len() != step.lut_instances.len() || cpu_bus.twist_cols.len() != step.mem_instances.len() {
        return Err(PiCcsError::InvalidInput(
            "shared_cpu_bus layout mismatch for step (instance counts)".into(),
        ));
    }

    for (idx, (mem_inst, _mem_wit)) in step.mem_instances.iter().enumerate() {
        neo_memory::addr::validate_twist_bit_addressing(mem_inst)?;
        let pow2_cycle = 1usize << ell_n;
        if mem_inst.steps > pow2_cycle {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist(Route A): steps={} exceeds 2^ell_cycle={pow2_cycle}",
                mem_inst.steps
            )));
        }

        let bus = cpu_bus.clone();
        let z = cpu_z_k.clone();

        let ell_addr = mem_inst.d * mem_inst.ell;
        let expected_lanes = mem_inst.lanes.max(1);
        let twist_inst_cols = bus.twist_cols.get(idx).ok_or_else(|| {
            PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch: missing twist_cols for mem_idx={idx}"
            ))
        })?;
        if twist_inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at mem_idx={idx}: expected lanes={expected_lanes}, got {}",
                twist_inst_cols.lanes.len()
            )));
        }

        let mut lanes: Vec<TwistLaneSparseCols> = Vec::with_capacity(twist_inst_cols.lanes.len());
        for (lane_idx, twist_cols) in twist_inst_cols.lanes.iter().enumerate() {
            if twist_cols.ra_bits.end - twist_cols.ra_bits.start != ell_addr
                || twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at mem_idx={idx}, lane={lane_idx}: expected ell_addr={ell_addr}"
                )));
            }

            let mut ra_bits = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.ra_bits.clone() {
                ra_bits.push(crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                    &z,
                    &bus,
                    col_id,
                    mem_inst.steps,
                    pow2_cycle,
                )?);
            }

            let mut wa_bits = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.wa_bits.clone() {
                wa_bits.push(crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                    &z,
                    &bus,
                    col_id,
                    mem_inst.steps,
                    pow2_cycle,
                )?);
            }

            let has_read = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                &z,
                &bus,
                twist_cols.has_read,
                mem_inst.steps,
                pow2_cycle,
            )?;
            let has_write = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                &z,
                &bus,
                twist_cols.has_write,
                mem_inst.steps,
                pow2_cycle,
            )?;
            let wv = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                &z,
                &bus,
                twist_cols.wv,
                mem_inst.steps,
                pow2_cycle,
            )?;
            let rv = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                &z,
                &bus,
                twist_cols.rv,
                mem_inst.steps,
                pow2_cycle,
            )?;
            let inc_at_write_addr = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                &z,
                &bus,
                twist_cols.inc,
                mem_inst.steps,
                pow2_cycle,
            )?;

            lanes.push(TwistLaneSparseCols {
                ra_bits,
                wa_bits,
                has_read,
                has_write,
                wv,
                rv,
                inc_at_write_addr,
            });
        }

        let decoded = TwistDecodedColsSparse { lanes };

        let init_sparse: Vec<(usize, K)> = match &mem_inst.init {
            MemInit::Zero => Vec::new(),
            MemInit::Sparse(pairs) => pairs
                .iter()
                .map(|(addr, val)| {
                    let addr_usize = usize::try_from(*addr).map_err(|_| {
                        PiCcsError::InvalidInput(format!("Twist: init address doesn't fit usize: addr={addr}"))
                    })?;
                    if addr_usize >= mem_inst.k {
                        return Err(PiCcsError::InvalidInput(format!(
                            "Twist: init address out of range: addr={addr} >= k={}",
                            mem_inst.k
                        )));
                    }
                    Ok((addr_usize, (*val).into()))
                })
                .collect::<Result<_, _>>()?,
        };

        let mut read_addr_oracle =
            TwistReadCheckAddrOracleSparseTimeMultiLane::new(init_sparse.clone(), r_cycle, &decoded.lanes);
        let mut write_addr_oracle =
            TwistWriteCheckAddrOracleSparseTimeMultiLane::new(init_sparse, r_cycle, &decoded.lanes);

        let labels: [&[u8]; 2] = [b"twist/read_addr_pre".as_slice(), b"twist/write_addr_pre".as_slice()];
        let claimed_sums = vec![K::ZERO, K::ZERO];
        tr.append_message(b"twist/addr_pre_time/claim_idx", &(idx as u64).to_le_bytes());
        bind_batched_claim_sums(tr, b"twist/addr_pre_time/claimed_sums", &claimed_sums, &labels);

        let mut claims = [
            BatchedClaim {
                oracle: &mut read_addr_oracle,
                claimed_sum: K::ZERO,
                label: labels[0],
            },
            BatchedClaim {
                oracle: &mut write_addr_oracle,
                claimed_sum: K::ZERO,
                label: labels[1],
            },
        ];

        let (r_addr, per_claim_results) = run_batched_sumcheck_prover_ds(tr, b"twist/addr_pre_time", idx, &mut claims)?;
        if per_claim_results.len() != 2 {
            return Err(PiCcsError::ProtocolError(format!(
                "twist addr-pre per-claim results len()={}, expected 2",
                per_claim_results.len()
            )));
        }

        out.push(TwistAddrPreProverData {
            addr_pre: BatchedAddrProof {
                claimed_sums,
                round_polys: vec![
                    per_claim_results[0].round_polys.clone(),
                    per_claim_results[1].round_polys.clone(),
                ],
                r_addr: r_addr.clone(),
            },
            decoded,
            read_check_claim_sum: per_claim_results[0].final_value,
            write_check_claim_sum: per_claim_results[1].final_value,
        });
    }

    Ok(out)
}
