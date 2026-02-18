use super::*;

pub struct RouteAShoutTimeClaimsGuard<'a> {
    pub lane_ranges: Vec<core::ops::Range<usize>>,
    pub lanes: Vec<RouteAShoutTimeLaneClaims<'a>>,
    pub gamma_groups: Vec<RouteAShoutTimeGammaGroupClaims<'a>>,
    pub bitness: Vec<Vec<Box<dyn RoundOracle>>>,
}

pub struct RouteAShoutTimeLaneClaims<'a> {
    pub value_prefix: RoundOraclePrefix<'a>,
    pub adapter_prefix: RoundOraclePrefix<'a>,
    pub event_table_hash_prefix: Option<RoundOraclePrefix<'a>>,
    pub value_claim: K,
    pub adapter_claim: K,
    pub event_table_hash_claim: Option<K>,
    pub gamma_group: Option<usize>,
}

pub struct RouteAShoutTimeGammaGroupClaims<'a> {
    pub value_prefix: RoundOraclePrefix<'a>,
    pub adapter_prefix: RoundOraclePrefix<'a>,
    pub value_claim: K,
    pub adapter_claim: K,
}

pub fn build_route_a_shout_time_claims_guard<'a>(
    shout_oracles: &'a mut [RouteAShoutTimeOracles],
    shout_gamma_groups: &'a mut [RouteAShoutGammaGroupOracles],
    ell_n: usize,
) -> RouteAShoutTimeClaimsGuard<'a> {
    let mut lane_ranges: Vec<core::ops::Range<usize>> = Vec::with_capacity(shout_oracles.len());
    let mut lanes: Vec<RouteAShoutTimeLaneClaims<'a>> = Vec::new();
    let mut gamma_groups: Vec<RouteAShoutTimeGammaGroupClaims<'a>> = Vec::with_capacity(shout_gamma_groups.len());
    let mut bitness: Vec<Vec<Box<dyn RoundOracle>>> = Vec::with_capacity(shout_oracles.len());

    for o in shout_oracles.iter_mut() {
        bitness.push(core::mem::take(&mut o.bitness));
        let start = lanes.len();
        for lane in o.lanes.iter_mut() {
            lanes.push(RouteAShoutTimeLaneClaims {
                value_prefix: RoundOraclePrefix::new(lane.value.as_mut(), ell_n),
                adapter_prefix: RoundOraclePrefix::new(lane.adapter.as_mut(), ell_n),
                event_table_hash_prefix: lane
                    .event_table_hash
                    .as_deref_mut()
                    .map(|o| RoundOraclePrefix::new(o, ell_n)),
                value_claim: lane.value_claim,
                adapter_claim: lane.adapter_claim,
                event_table_hash_claim: lane.event_table_hash_claim,
                gamma_group: lane.gamma_group,
            });
        }
        let end = lanes.len();
        lane_ranges.push(start..end);
    }

    for g in shout_gamma_groups.iter_mut() {
        gamma_groups.push(RouteAShoutTimeGammaGroupClaims {
            value_prefix: RoundOraclePrefix::new(g.value.as_mut(), ell_n),
            adapter_prefix: RoundOraclePrefix::new(g.adapter.as_mut(), ell_n),
            value_claim: g.value_claim,
            adapter_claim: g.adapter_claim,
        });
    }

    RouteAShoutTimeClaimsGuard {
        lane_ranges,
        lanes,
        gamma_groups,
        bitness,
    }
}

pub struct ShoutRouteAProtocol<'a> {
    guard: RouteAShoutTimeClaimsGuard<'a>,
}

impl<'a> ShoutRouteAProtocol<'a> {
    pub fn new(
        shout_oracles: &'a mut [RouteAShoutTimeOracles],
        shout_gamma_groups: &'a mut [RouteAShoutGammaGroupOracles],
        ell_n: usize,
    ) -> Self {
        Self {
            guard: build_route_a_shout_time_claims_guard(shout_oracles, shout_gamma_groups, ell_n),
        }
    }
}

impl<'o> TimeBatchedClaims for ShoutRouteAProtocol<'o> {
    fn append_time_claims<'a>(
        &'a mut self,
        _ell_n: usize,
        claimed_sums: &mut Vec<K>,
        degree_bounds: &mut Vec<usize>,
        labels: &mut Vec<&'static [u8]>,
        claim_is_dynamic: &mut Vec<bool>,
        claims: &mut Vec<BatchedClaim<'a>>,
    ) {
        append_route_a_shout_time_claims(
            &mut self.guard,
            claimed_sums,
            degree_bounds,
            labels,
            claim_is_dynamic,
            claims,
        );
    }
}

pub fn append_route_a_shout_time_claims<'a>(
    guard: &'a mut RouteAShoutTimeClaimsGuard<'_>,
    claimed_sums: &mut Vec<K>,
    degree_bounds: &mut Vec<usize>,
    labels: &mut Vec<&'static [u8]>,
    claim_is_dynamic: &mut Vec<bool>,
    claims: &mut Vec<BatchedClaim<'a>>,
) {
    if guard.lane_ranges.is_empty() {
        return;
    }
    if guard.bitness.len() != guard.lane_ranges.len() {
        panic!("shout bitness count mismatch");
    }

    let mut lane_ranges_iter = guard.lane_ranges.iter();
    let mut next_end = lane_ranges_iter.next().expect("non-empty").end;
    let mut bitness_iter = guard.bitness.iter_mut();

    for (lane_idx, lane) in guard.lanes.iter_mut().enumerate() {
        if lane.gamma_group.is_none() {
            claimed_sums.push(lane.value_claim);
            degree_bounds.push(lane.value_prefix.degree_bound());
            labels.push(b"shout/value");
            claim_is_dynamic.push(true);
            claims.push(BatchedClaim {
                oracle: &mut lane.value_prefix,
                claimed_sum: lane.value_claim,
                label: b"shout/value",
            });

            claimed_sums.push(lane.adapter_claim);
            degree_bounds.push(lane.adapter_prefix.degree_bound());
            labels.push(b"shout/adapter");
            claim_is_dynamic.push(true);
            claims.push(BatchedClaim {
                oracle: &mut lane.adapter_prefix,
                claimed_sum: lane.adapter_claim,
                label: b"shout/adapter",
            });
        }

        if let Some(prefix) = lane.event_table_hash_prefix.as_mut() {
            let claim = lane
                .event_table_hash_claim
                .expect("event_table_hash_claim missing");
            claimed_sums.push(claim);
            degree_bounds.push(prefix.degree_bound());
            labels.push(b"shout/event_table_hash");
            claim_is_dynamic.push(true);
            claims.push(BatchedClaim {
                oracle: prefix,
                claimed_sum: claim,
                label: b"shout/event_table_hash",
            });
        }

        if lane_idx + 1 == next_end {
            let bitness_vec = bitness_iter.next().expect("shout bitness idx drift");
            for bit_oracle in bitness_vec.iter_mut() {
                claimed_sums.push(K::ZERO);
                degree_bounds.push(bit_oracle.degree_bound());
                labels.push(b"shout/bitness");
                claim_is_dynamic.push(false);
                claims.push(BatchedClaim {
                    oracle: bit_oracle.as_mut(),
                    claimed_sum: K::ZERO,
                    label: b"shout/bitness",
                });
            }

            next_end = lane_ranges_iter.next().map(|r| r.end).unwrap_or(usize::MAX);
        }
    }

    for group in guard.gamma_groups.iter_mut() {
        claimed_sums.push(group.value_claim);
        degree_bounds.push(group.value_prefix.degree_bound());
        labels.push(b"shout/value");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: &mut group.value_prefix,
            claimed_sum: group.value_claim,
            label: b"shout/value",
        });

        claimed_sums.push(group.adapter_claim);
        degree_bounds.push(group.adapter_prefix.degree_bound());
        labels.push(b"shout/adapter");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: &mut group.adapter_prefix,
            claimed_sum: group.adapter_claim,
            label: b"shout/adapter",
        });
    }

    if bitness_iter.next().is_some() {
        panic!("shout bitness not fully consumed");
    }
}

pub struct RouteATwistTimeClaimsGuard<'a> {
    pub read_check_prefixes: Vec<RoundOraclePrefix<'a>>,
    pub write_check_prefixes: Vec<RoundOraclePrefix<'a>>,
    pub read_check_claims: Vec<K>,
    pub write_check_claims: Vec<K>,
    pub bitness: Vec<Vec<Box<dyn RoundOracle>>>,
}

pub fn build_route_a_twist_time_claims_guard<'a>(
    twist_oracles: &'a mut [RouteATwistTimeOracles],
    ell_n: usize,
    read_check_claims: Vec<K>,
    write_check_claims: Vec<K>,
) -> RouteATwistTimeClaimsGuard<'a> {
    let mut read_check_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(twist_oracles.len());
    let mut write_check_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(twist_oracles.len());
    let mut bitness: Vec<Vec<Box<dyn RoundOracle>>> = Vec::with_capacity(twist_oracles.len());

    if read_check_claims.len() != twist_oracles.len() {
        panic!(
            "twist read-check claim count mismatch (claims={}, oracles={})",
            read_check_claims.len(),
            twist_oracles.len()
        );
    }
    if write_check_claims.len() != twist_oracles.len() {
        panic!(
            "twist write-check claim count mismatch (claims={}, oracles={})",
            write_check_claims.len(),
            twist_oracles.len()
        );
    }

    for o in twist_oracles.iter_mut() {
        bitness.push(core::mem::take(&mut o.bitness));
        read_check_prefixes.push(RoundOraclePrefix::new(o.read_check.as_mut(), ell_n));
        write_check_prefixes.push(RoundOraclePrefix::new(o.write_check.as_mut(), ell_n));
    }

    RouteATwistTimeClaimsGuard {
        read_check_prefixes,
        write_check_prefixes,
        read_check_claims,
        write_check_claims,
        bitness,
    }
}

pub fn append_route_a_twist_time_claims<'a>(
    guard: &'a mut RouteATwistTimeClaimsGuard<'_>,
    claimed_sums: &mut Vec<K>,
    degree_bounds: &mut Vec<usize>,
    labels: &mut Vec<&'static [u8]>,
    claim_is_dynamic: &mut Vec<bool>,
    claims: &mut Vec<BatchedClaim<'a>>,
) {
    for (((read_check_time, write_check_time), bitness_vec), (read_claim, write_claim)) in guard
        .read_check_prefixes
        .iter_mut()
        .zip(guard.write_check_prefixes.iter_mut())
        .zip(guard.bitness.iter_mut())
        .zip(
            guard
                .read_check_claims
                .iter()
                .zip(guard.write_check_claims.iter()),
        )
    {
        claimed_sums.push(*read_claim);
        degree_bounds.push(read_check_time.degree_bound());
        labels.push(b"twist/read_check");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: read_check_time,
            claimed_sum: *read_claim,
            label: b"twist/read_check",
        });

        claimed_sums.push(*write_claim);
        degree_bounds.push(write_check_time.degree_bound());
        labels.push(b"twist/write_check");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: write_check_time,
            claimed_sum: *write_claim,
            label: b"twist/write_check",
        });

        for bit_oracle in bitness_vec.iter_mut() {
            claimed_sums.push(K::ZERO);
            degree_bounds.push(bit_oracle.degree_bound());
            labels.push(b"twist/bitness");
            claim_is_dynamic.push(false);
            claims.push(BatchedClaim {
                oracle: bit_oracle.as_mut(),
                claimed_sum: K::ZERO,
                label: b"twist/bitness",
            });
        }
    }
}

pub struct TwistRouteAProtocol<'a> {
    guard: RouteATwistTimeClaimsGuard<'a>,
}

impl<'a> TwistRouteAProtocol<'a> {
    pub fn new(
        twist_oracles: &'a mut [RouteATwistTimeOracles],
        ell_n: usize,
        read_check_claims: Vec<K>,
        write_check_claims: Vec<K>,
    ) -> Self {
        Self {
            guard: build_route_a_twist_time_claims_guard(twist_oracles, ell_n, read_check_claims, write_check_claims),
        }
    }
}

impl<'o> TimeBatchedClaims for TwistRouteAProtocol<'o> {
    fn append_time_claims<'a>(
        &'a mut self,
        _ell_n: usize,
        claimed_sums: &mut Vec<K>,
        degree_bounds: &mut Vec<usize>,
        labels: &mut Vec<&'static [u8]>,
        claim_is_dynamic: &mut Vec<bool>,
        claims: &mut Vec<BatchedClaim<'a>>,
    ) {
        append_route_a_twist_time_claims(
            &mut self.guard,
            claimed_sums,
            degree_bounds,
            labels,
            claim_is_dynamic,
            claims,
        );
    }
}

#[inline]
pub(crate) fn has_trace_lookup_families_instance(step: &StepInstanceBundle<Cmt, F, K>) -> bool {
    step.lut_insts
        .iter()
        .any(|inst| rv32_is_decode_lookup_table_id(inst.table_id) || rv32_is_width_lookup_table_id(inst.table_id))
}

#[inline]
pub(crate) fn has_trace_lookup_families_witness(step: &StepWitnessBundle<Cmt, F, K>) -> bool {
    step.lut_instances
        .iter()
        .any(|(inst, _)| rv32_is_decode_lookup_table_id(inst.table_id) || rv32_is_width_lookup_table_id(inst.table_id))
}

#[inline]
pub(crate) fn wb_wp_required_for_step_instance(step: &StepInstanceBundle<Cmt, F, K>) -> bool {
    // Stage gating is keyed by lookup-family presence instead of fixed `m_in`/`mem_id`
    // assumptions so adapter-side routing can evolve without hardcoding RV32 shapes here.
    has_trace_lookup_families_instance(step)
}

#[inline]
pub(crate) fn wb_wp_required_for_step_witness(step: &StepWitnessBundle<Cmt, F, K>) -> bool {
    has_trace_lookup_families_witness(step)
}

pub(crate) fn build_bus_layout_for_step_witness(
    step: &StepWitnessBundle<Cmt, F, K>,
    t_len: usize,
) -> Result<BusLayout, PiCcsError> {
    let m = step.mcs.1.Z.cols();
    let m_in = step.mcs.0.m_in;
    let shout_shapes: Vec<ShoutInstanceShape> = step
        .lut_instances
        .iter()
        .map(|(inst, _)| ShoutInstanceShape {
            ell_addr: inst.d * inst.ell,
            lanes: inst.lanes.max(1),
            n_vals: 1usize,
            addr_group: inst.addr_group,
            selector_group: inst.selector_group,
        })
        .collect();
    let grouped_shout_instances = shout_shapes
        .iter()
        .filter(|shape| shape.addr_group.is_some())
        .count();
    let twist = step
        .mem_instances
        .iter()
        .map(|(inst, _)| (inst.d * inst.ell, inst.lanes.max(1)));
    build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes(m, m_in, t_len, shout_shapes, twist).map_err(
        |e| {
            PiCcsError::InvalidInput(format!(
                "step bus layout failed: m={m}, m_in={m_in}, t_len={t_len}, lut_insts={}, grouped_lut_insts={grouped_shout_instances}: {e}",
                step.lut_instances.len()
            ))
        },
    )
}

#[inline]
pub(crate) fn decode_stage_required_for_step_instance(step: &StepInstanceBundle<Cmt, F, K>) -> bool {
    wb_wp_required_for_step_instance(step)
        && step
            .lut_insts
            .iter()
            .any(|inst| rv32_is_decode_lookup_table_id(inst.table_id))
}

#[inline]
pub(crate) fn decode_stage_required_for_step_witness(step: &StepWitnessBundle<Cmt, F, K>) -> bool {
    wb_wp_required_for_step_witness(step)
        && step
            .lut_instances
            .iter()
            .any(|(inst, _)| rv32_is_decode_lookup_table_id(inst.table_id))
}

#[inline]
pub(crate) fn width_stage_required_for_step_instance(step: &StepInstanceBundle<Cmt, F, K>) -> bool {
    wb_wp_required_for_step_instance(step)
        && step
            .lut_insts
            .iter()
            .any(|inst| rv32_is_width_lookup_table_id(inst.table_id))
}

#[inline]
pub(crate) fn width_stage_required_for_step_witness(step: &StepWitnessBundle<Cmt, F, K>) -> bool {
    wb_wp_required_for_step_witness(step)
        && step
            .lut_instances
            .iter()
            .any(|(inst, _)| rv32_is_width_lookup_table_id(inst.table_id))
}

#[inline]
pub(crate) fn control_stage_required_for_step_instance(step: &StepInstanceBundle<Cmt, F, K>) -> bool {
    decode_stage_required_for_step_instance(step)
}

#[inline]
pub(crate) fn control_stage_required_for_step_witness(step: &StepWitnessBundle<Cmt, F, K>) -> bool {
    decode_stage_required_for_step_witness(step)
}

pub(crate) fn build_route_a_wb_wp_time_claims(
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    r_cycle: &[K],
) -> Result<(Option<(Box<dyn RoundOracle>, K)>, Option<(Box<dyn RoundOracle>, K)>), PiCcsError> {
    if !wb_wp_required_for_step_witness(step) {
        return Ok((None, None));
    }

    let trace = Rv32TraceLayout::new();
    let t_len = infer_rv32_trace_t_len_for_wb_wp(step, &trace)?;
    let m_in = step.mcs.0.m_in;
    let ell_n = r_cycle.len();
    let wb_bool_cols = rv32_trace_wb_columns(&trace);
    let wp_cols = rv32_trace_wp_columns(&trace);

    let mut decode_cols = Vec::with_capacity(1 + wb_bool_cols.len() + wp_cols.len());
    decode_cols.push(trace.active);
    decode_cols.extend(wb_bool_cols.iter().copied());
    decode_cols.extend(wp_cols.iter().copied());
    let decoded = decode_trace_col_values_batch(params, step, t_len, &decode_cols)?;

    let wb_weights = wb_weight_vector(r_cycle, wb_bool_cols.len());
    let mut wb_bool_sparse_cols: Vec<SparseIdxVec<K>> = Vec::with_capacity(wb_bool_cols.len());
    for &col_id in wb_bool_cols.iter() {
        let vals = decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("WB: missing decoded bool column {col_id}")))?;
        wb_bool_sparse_cols.push(sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }

    let wb_oracle = LazyWeightedBitnessOracleSparseTime::new_with_cycle(r_cycle, wb_bool_sparse_cols, wb_weights);

    let wp_cols = rv32_trace_wp_columns(&trace);
    let weights = wp_weight_vector(r_cycle, wp_cols.len());
    let active_vals = decoded
        .get(&trace.active)
        .ok_or_else(|| PiCcsError::ProtocolError(format!("WP: missing decoded active column {}", trace.active)))?;
    let active = sparse_trace_col_from_values(m_in, ell_n, &active_vals)?;

    let mut sparse_cols: Vec<SparseIdxVec<K>> = Vec::with_capacity(wp_cols.len());
    for &col_id in wp_cols.iter() {
        let vals = decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("WP: missing decoded column {col_id}")))?;
        sparse_cols.push(sparse_trace_col_from_values(m_in, ell_n, &vals)?);
    }

    let oracle = WeightedMaskOracleSparseTime::new(active, sparse_cols, weights, r_cycle);
    Ok((Some((Box::new(wb_oracle), K::ZERO)), Some((Box::new(oracle), K::ZERO))))
}

pub(crate) fn build_route_a_decode_time_claims(
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    r_cycle: &[K],
) -> Result<(Option<(Box<dyn RoundOracle>, K)>, Option<(Box<dyn RoundOracle>, K)>), PiCcsError> {
    if !decode_stage_required_for_step_witness(step) {
        return Ok((None, None));
    }

    let trace = Rv32TraceLayout::new();
    let decode = Rv32DecodeSidecarLayout::new();
    let t_len = infer_rv32_trace_t_len_for_wb_wp(step, &trace)?;
    let m_in = step.mcs.0.m_in;
    let ell_n = r_cycle.len();

    let cpu_cols = vec![
        trace.active,
        trace.halted,
        trace.instr_word,
        trace.rs1_val,
        trace.rs2_val,
        trace.rd_val,
        trace.ram_addr,
        trace.shout_has_lookup,
        trace.shout_val,
        trace.shout_lhs,
        trace.shout_rhs,
    ];
    let cpu_decoded = decode_trace_col_values_batch(params, step, t_len, &cpu_cols)?;

    let decode_decoded = {
        let instr_vals = cpu_decoded
            .get(&trace.instr_word)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing instr_word decode column".into()))?;
        let active_vals = cpu_decoded
            .get(&trace.active)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing active decode column".into()))?;
        if instr_vals.len() != t_len || active_vals.len() != t_len {
            return Err(PiCcsError::ProtocolError(format!(
                "W2(shared): decoded CPU column lengths drift (instr={}, active={}, t_len={t_len})",
                instr_vals.len(),
                active_vals.len()
            )));
        }
        let mut decoded = BTreeMap::<usize, Vec<K>>::new();
        for col_id in 0..decode.cols {
            decoded.insert(col_id, Vec::with_capacity(t_len));
        }
        for j in 0..t_len {
            let instr_word = decode_k_to_u32(instr_vals[j], "W2(shared)/instr_word")?;
            let active = active_vals[j] != K::ZERO;
            let mut row = rv32_decode_lookup_backed_row_from_instr_word(&decode, instr_word, active);
            if !active {
                row.fill(F::ZERO);
            }
            for (col_id, value) in row.into_iter().enumerate() {
                decoded
                    .get_mut(&col_id)
                    .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): decode map build failed".into()))?
                    .push(K::from(value));
            }
        }

        // In shared lookup-backed mode, overwrite lookup-backed decode columns with the values
        // actually committed on the shared Shout bus so prover oracles and verifier terminals
        // are sourced from identical openings.
        let (decode_open_cols, decode_lut_indices) = resolve_shared_decode_lookup_lut_indices(step, &decode)?;
        let bus = build_bus_layout_for_step_witness(step, t_len)?;
        if bus.shout_cols.len() != step.lut_instances.len() {
            return Err(PiCcsError::ProtocolError(
                "W2(shared): bus layout shout lane count drift".into(),
            ));
        }
        let mut bus_val_cols = Vec::with_capacity(decode_open_cols.len());
        for &lut_idx in decode_lut_indices.iter() {
            let inst_cols = bus.shout_cols.get(lut_idx).ok_or_else(|| {
                PiCcsError::ProtocolError("W2(shared): missing shout cols for decode lookup table".into())
            })?;
            let lane0 = inst_cols.lanes.get(0).ok_or_else(|| {
                PiCcsError::ProtocolError("W2(shared): expected one shout lane for decode lookup table".into())
            })?;
            bus_val_cols.push(lane0.primary_val());
        }
        let lookup_vals = decode_lookup_backed_col_values_batch(
            params,
            bus.bus_base,
            t_len,
            &step.mcs.1.Z,
            bus.bus_cols,
            &bus_val_cols,
        )?;
        for (open_idx, &decode_col_id) in decode_open_cols.iter().enumerate() {
            let bus_col_id = bus_val_cols[open_idx];
            let values = lookup_vals.get(&bus_col_id).ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "W2(shared): missing decoded lookup values for bus_col={bus_col_id}"
                ))
            })?;
            decoded.insert(decode_col_id, values.clone());
        }

        // Recompute derived decode helper columns from opened lookup-backed decode columns.
        let rd_is_zero_vals = decoded
            .get(&decode.rd_is_zero)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing rd_is_zero decode column".into()))?;
        let funct7_b0_vals = decoded
            .get(&decode.funct7_bit[0])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct7_bit[0] decode column".into()))?;
        let funct7_b1_vals = decoded
            .get(&decode.funct7_bit[1])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct7_bit[1] decode column".into()))?;
        let funct7_b2_vals = decoded
            .get(&decode.funct7_bit[2])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct7_bit[2] decode column".into()))?;
        let funct7_b3_vals = decoded
            .get(&decode.funct7_bit[3])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct7_bit[3] decode column".into()))?;
        let funct7_b4_vals = decoded
            .get(&decode.funct7_bit[4])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct7_bit[4] decode column".into()))?;
        let funct7_b5_vals = decoded
            .get(&decode.funct7_bit[5])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct7_bit[5] decode column".into()))?;
        let funct7_b6_vals = decoded
            .get(&decode.funct7_bit[6])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct7_bit[6] decode column".into()))?;
        let op_lui_vals = decoded
            .get(&decode.op_lui)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing op_lui decode column".into()))?;
        let op_auipc_vals = decoded
            .get(&decode.op_auipc)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing op_auipc decode column".into()))?;
        let op_jal_vals = decoded
            .get(&decode.op_jal)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing op_jal decode column".into()))?;
        let op_jalr_vals = decoded
            .get(&decode.op_jalr)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing op_jalr decode column".into()))?;
        let op_alu_imm_vals = decoded
            .get(&decode.op_alu_imm)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing op_alu_imm decode column".into()))?;
        let op_alu_reg_vals = decoded
            .get(&decode.op_alu_reg)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing op_alu_reg decode column".into()))?;
        let funct3_is0_vals = decoded
            .get(&decode.funct3_is[0])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct3_is[0] decode column".into()))?;
        let funct3_is1_vals = decoded
            .get(&decode.funct3_is[1])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct3_is[1] decode column".into()))?;
        let funct3_is2_vals = decoded
            .get(&decode.funct3_is[2])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct3_is[2] decode column".into()))?;
        let funct3_is3_vals = decoded
            .get(&decode.funct3_is[3])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct3_is[3] decode column".into()))?;
        let funct3_is4_vals = decoded
            .get(&decode.funct3_is[4])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct3_is[4] decode column".into()))?;
        let funct3_is5_vals = decoded
            .get(&decode.funct3_is[5])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct3_is[5] decode column".into()))?;
        let funct3_is6_vals = decoded
            .get(&decode.funct3_is[6])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct3_is[6] decode column".into()))?;
        let funct3_is7_vals = decoded
            .get(&decode.funct3_is[7])
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing funct3_is[7] decode column".into()))?;
        let rs2_vals = decoded
            .get(&decode.rs2)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing rs2 decode column".into()))?;
        let imm_i_vals = decoded
            .get(&decode.imm_i)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): missing imm_i decode column".into()))?;

        let mut op_lui_write = Vec::with_capacity(t_len);
        let mut op_auipc_write = Vec::with_capacity(t_len);
        let mut op_jal_write = Vec::with_capacity(t_len);
        let mut op_jalr_write = Vec::with_capacity(t_len);
        let mut op_alu_imm_write = Vec::with_capacity(t_len);
        let mut op_alu_reg_write = Vec::with_capacity(t_len);
        let mut alu_reg_delta = Vec::with_capacity(t_len);
        let mut alu_imm_delta = Vec::with_capacity(t_len);
        let mut alu_imm_shift_rhs_delta = Vec::with_capacity(t_len);
        for j in 0..t_len {
            let rd_keep = K::ONE - rd_is_zero_vals[j];
            op_lui_write.push(op_lui_vals[j] * rd_keep);
            op_auipc_write.push(op_auipc_vals[j] * rd_keep);
            op_jal_write.push(op_jal_vals[j] * rd_keep);
            op_jalr_write.push(op_jalr_vals[j] * rd_keep);
            op_alu_imm_write.push(op_alu_imm_vals[j] * rd_keep);
            op_alu_reg_write.push(op_alu_reg_vals[j] * rd_keep);
            let funct7_bits = [
                funct7_b0_vals[j],
                funct7_b1_vals[j],
                funct7_b2_vals[j],
                funct7_b3_vals[j],
                funct7_b4_vals[j],
                funct7_b5_vals[j],
                funct7_b6_vals[j],
            ];
            let funct3_is = [
                funct3_is0_vals[j],
                funct3_is1_vals[j],
                funct3_is2_vals[j],
                funct3_is3_vals[j],
                funct3_is4_vals[j],
                funct3_is5_vals[j],
                funct3_is6_vals[j],
                funct3_is7_vals[j],
            ];
            alu_reg_delta.push(w2_alu_reg_table_delta_from_bits(funct7_bits, funct3_is));
            alu_imm_delta.push(funct7_bits[5] * funct3_is[5]);
            alu_imm_shift_rhs_delta.push((funct3_is1_vals[j] + funct3_is5_vals[j]) * (rs2_vals[j] - imm_i_vals[j]));
        }
        decoded.insert(decode.op_lui_write, op_lui_write);
        decoded.insert(decode.op_auipc_write, op_auipc_write);
        decoded.insert(decode.op_jal_write, op_jal_write);
        decoded.insert(decode.op_jalr_write, op_jalr_write);
        decoded.insert(decode.op_alu_imm_write, op_alu_imm_write);
        decoded.insert(decode.op_alu_reg_write, op_alu_reg_write);
        decoded.insert(decode.alu_reg_table_delta, alu_reg_delta);
        decoded.insert(decode.alu_imm_table_delta, alu_imm_delta);
        decoded.insert(decode.alu_imm_shift_rhs_delta, alu_imm_shift_rhs_delta);

        decoded
    };

    let cpu_value_at = |col_id: usize, row: usize| -> Result<K, PiCcsError> {
        cpu_decoded
            .get(&col_id)
            .and_then(|v| v.get(row))
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {col_id}")))
    };
    let decode_value_at = |col_id: usize, row: usize| -> Result<K, PiCcsError> {
        decode_decoded
            .get(&col_id)
            .and_then(|v| v.get(row))
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {col_id}")))
    };

    let mut imm_residual_vals: Vec<Vec<K>> = (0..W2_IMM_RESIDUAL_COUNT)
        .map(|_| Vec::with_capacity(t_len))
        .collect();
    for j in 0..t_len {
        let active = cpu_value_at(trace.active, j)?;
        let halted = cpu_value_at(trace.halted, j)?;
        let decode_opcode = decode_value_at(decode.opcode, j)?;
        let rd_has_write = decode_value_at(decode.rd_has_write, j)?;
        let rd_is_zero = decode_value_at(decode.rd_is_zero, j)?;
        let rs1_val = cpu_value_at(trace.rs1_val, j)?;
        let rs2_val = cpu_value_at(trace.rs2_val, j)?;
        let rd_val = cpu_value_at(trace.rd_val, j)?;
        let ram_has_read = decode_value_at(decode.ram_has_read, j)?;
        let ram_has_write = decode_value_at(decode.ram_has_write, j)?;
        let ram_addr = cpu_value_at(trace.ram_addr, j)?;
        let shout_has_lookup = cpu_value_at(trace.shout_has_lookup, j)?;
        let shout_val = cpu_value_at(trace.shout_val, j)?;
        let shout_lhs = cpu_value_at(trace.shout_lhs, j)?;
        let shout_rhs = cpu_value_at(trace.shout_rhs, j)?;
        let opcode_flags = [
            decode_value_at(decode.op_lui, j)?,
            decode_value_at(decode.op_auipc, j)?,
            decode_value_at(decode.op_jal, j)?,
            decode_value_at(decode.op_jalr, j)?,
            decode_value_at(decode.op_branch, j)?,
            decode_value_at(decode.op_load, j)?,
            decode_value_at(decode.op_store, j)?,
            decode_value_at(decode.op_alu_imm, j)?,
            decode_value_at(decode.op_alu_reg, j)?,
            decode_value_at(decode.op_misc_mem, j)?,
            decode_value_at(decode.op_system, j)?,
            decode_value_at(decode.op_amo, j)?,
        ];
        let funct3_is = [
            decode_value_at(decode.funct3_is[0], j)?,
            decode_value_at(decode.funct3_is[1], j)?,
            decode_value_at(decode.funct3_is[2], j)?,
            decode_value_at(decode.funct3_is[3], j)?,
            decode_value_at(decode.funct3_is[4], j)?,
            decode_value_at(decode.funct3_is[5], j)?,
            decode_value_at(decode.funct3_is[6], j)?,
            decode_value_at(decode.funct3_is[7], j)?,
        ];
        let rs2_decode = decode_value_at(decode.rs2, j)?;
        let imm_i = decode_value_at(decode.imm_i, j)?;
        let imm_s = decode_value_at(decode.imm_s, j)?;

        let funct3_bits = [
            decode_value_at(decode.funct3_bit[0], j)?,
            decode_value_at(decode.funct3_bit[1], j)?,
            decode_value_at(decode.funct3_bit[2], j)?,
        ];
        let funct7_bits = [
            decode_value_at(decode.funct7_bit[0], j)?,
            decode_value_at(decode.funct7_bit[1], j)?,
            decode_value_at(decode.funct7_bit[2], j)?,
            decode_value_at(decode.funct7_bit[3], j)?,
            decode_value_at(decode.funct7_bit[4], j)?,
            decode_value_at(decode.funct7_bit[5], j)?,
            decode_value_at(decode.funct7_bit[6], j)?,
        ];
        let imm = w2_decode_immediate_residuals(
            decode_value_at(decode.imm_i, j)?,
            decode_value_at(decode.imm_s, j)?,
            decode_value_at(decode.imm_b, j)?,
            decode_value_at(decode.imm_j, j)?,
            [
                decode_value_at(decode.rd_bit[0], j)?,
                decode_value_at(decode.rd_bit[1], j)?,
                decode_value_at(decode.rd_bit[2], j)?,
                decode_value_at(decode.rd_bit[3], j)?,
                decode_value_at(decode.rd_bit[4], j)?,
            ],
            funct3_bits,
            [
                decode_value_at(decode.rs1_bit[0], j)?,
                decode_value_at(decode.rs1_bit[1], j)?,
                decode_value_at(decode.rs1_bit[2], j)?,
                decode_value_at(decode.rs1_bit[3], j)?,
                decode_value_at(decode.rs1_bit[4], j)?,
            ],
            [
                decode_value_at(decode.rs2_bit[0], j)?,
                decode_value_at(decode.rs2_bit[1], j)?,
                decode_value_at(decode.rs2_bit[2], j)?,
                decode_value_at(decode.rs2_bit[3], j)?,
                decode_value_at(decode.rs2_bit[4], j)?,
            ],
            funct7_bits,
        );

        let op_write_flags = [
            opcode_flags[0] * (K::ONE - rd_is_zero),
            opcode_flags[1] * (K::ONE - rd_is_zero),
            opcode_flags[2] * (K::ONE - rd_is_zero),
            opcode_flags[3] * (K::ONE - rd_is_zero),
            opcode_flags[7] * (K::ONE - rd_is_zero),
            opcode_flags[8] * (K::ONE - rd_is_zero),
        ];
        let shout_table_id = decode_value_at(decode.shout_table_id, j)?;
        let alu_reg_table_delta = w2_alu_reg_table_delta_from_bits(funct7_bits, funct3_is);
        let alu_imm_table_delta = funct7_bits[5] * funct3_is[5];
        let alu_imm_shift_rhs_delta = (funct3_is[1] + funct3_is[5]) * (rs2_decode - imm_i);
        let selector_residuals = w2_decode_selector_residuals(
            active,
            decode_opcode,
            opcode_flags,
            funct3_is,
            funct3_bits,
            opcode_flags[11],
        );
        let bitness_residuals = w2_decode_bitness_residuals(opcode_flags, funct3_is);
        let alu_branch_residuals = w2_alu_branch_lookup_residuals(
            active,
            halted,
            shout_has_lookup,
            shout_lhs,
            shout_rhs,
            shout_table_id,
            rs1_val,
            rs2_val,
            rd_has_write,
            rd_is_zero,
            rd_val,
            ram_has_read,
            ram_has_write,
            ram_addr,
            shout_val,
            funct3_bits,
            funct7_bits,
            opcode_flags,
            op_write_flags,
            funct3_is,
            alu_reg_table_delta,
            alu_imm_table_delta,
            alu_imm_shift_rhs_delta,
            rs2_decode,
            imm_i,
            imm_s,
        );
        if let Some((idx, _)) = selector_residuals
            .iter()
            .enumerate()
            .find(|(_, r)| **r != K::ZERO)
        {
            return Err(PiCcsError::ProtocolError(format!(
                "decode/fields selector residual non-zero at row={j}, idx={idx}"
            )));
        }
        if let Some((idx, _)) = bitness_residuals
            .iter()
            .enumerate()
            .find(|(_, r)| **r != K::ZERO)
        {
            return Err(PiCcsError::ProtocolError(format!(
                "decode/fields bitness residual non-zero at row={j}, idx={idx}"
            )));
        }
        if let Some((idx, _)) = alu_branch_residuals
            .iter()
            .enumerate()
            .find(|(_, r)| **r != K::ZERO)
        {
            return Err(PiCcsError::ProtocolError(format!(
                "decode/fields alu_branch residual non-zero at row={j}, idx={idx}"
            )));
        }

        for (k, r) in imm.iter().enumerate() {
            imm_residual_vals[k].push(*r);
        }
    }

    let main_field_cols = vec![
        trace.active,
        trace.halted,
        trace.rs1_val,
        trace.rs2_val,
        trace.rd_val,
        trace.ram_addr,
        trace.shout_has_lookup,
        trace.shout_val,
        trace.shout_lhs,
        trace.shout_rhs,
    ];
    let decode_field_cols = vec![
        decode.opcode,
        decode.rd_is_zero,
        decode.rd_has_write,
        decode.ram_has_read,
        decode.ram_has_write,
        decode.shout_table_id,
        decode.op_lui,
        decode.op_auipc,
        decode.op_jal,
        decode.op_jalr,
        decode.op_branch,
        decode.op_load,
        decode.op_store,
        decode.op_alu_imm,
        decode.op_alu_reg,
        decode.op_misc_mem,
        decode.op_system,
        decode.op_amo,
        decode.funct3_is[0],
        decode.funct3_is[1],
        decode.funct3_is[2],
        decode.funct3_is[3],
        decode.funct3_is[4],
        decode.funct3_is[5],
        decode.funct3_is[6],
        decode.funct3_is[7],
        decode.funct3_bit[0],
        decode.funct3_bit[1],
        decode.funct3_bit[2],
        decode.funct7_bit[0],
        decode.funct7_bit[1],
        decode.funct7_bit[2],
        decode.funct7_bit[3],
        decode.funct7_bit[4],
        decode.funct7_bit[5],
        decode.funct7_bit[6],
        decode.rs2,
        decode.imm_i,
        decode.imm_s,
    ];
    let mut main_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for &col_id in main_field_cols.iter() {
        let vals = cpu_decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing CPU decoded column {col_id}")))?;
        main_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }
    let mut decode_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for &col_id in decode_field_cols.iter() {
        let vals = decode_decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing decode lookup-backed column {col_id}")))?;
        decode_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }
    let main_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        main_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing main sparse column {col_id}")))
    };
    let decode_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        decode_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing decode sparse column {col_id}")))
    };

    let mut fields_sparse_cols = Vec::with_capacity(main_field_cols.len() + decode_field_cols.len());
    for &col_id in main_field_cols.iter() {
        fields_sparse_cols.push(main_col(col_id)?);
    }
    for &col_id in decode_field_cols.iter() {
        fields_sparse_cols.push(decode_col(col_id)?);
    }

    let mut imm_sparse_cols = Vec::with_capacity(imm_residual_vals.len());
    for vals in imm_residual_vals.iter() {
        imm_sparse_cols.push(sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }

    let pow2_cycle = 1usize
        .checked_shl(ell_n as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("W2: 2^ell_n overflow".into()))?;
    let active_zero = SparseIdxVec::from_entries(pow2_cycle, Vec::new());
    let fields_weights = w2_decode_pack_weight_vector(r_cycle, W2_FIELDS_RESIDUAL_COUNT);
    let fields_oracle = FormulaOracleSparseTime::new(
        fields_sparse_cols,
        5,
        r_cycle,
        Box::new(move |vals: &[K]| {
            let mut idx = 0usize;
            let active = vals[idx];
            idx += 1;
            let halted = vals[idx];
            idx += 1;
            let rs1_val = vals[idx];
            idx += 1;
            let rs2_val = vals[idx];
            idx += 1;
            let rd_val = vals[idx];
            idx += 1;
            let ram_addr = vals[idx];
            idx += 1;
            let shout_has_lookup = vals[idx];
            idx += 1;
            let shout_val = vals[idx];
            idx += 1;
            let shout_lhs = vals[idx];
            idx += 1;
            let shout_rhs = vals[idx];
            idx += 1;
            let decode_opcode = vals[idx];
            idx += 1;
            let rd_is_zero = vals[idx];
            idx += 1;
            let rd_has_write = vals[idx];
            idx += 1;
            let ram_has_read = vals[idx];
            idx += 1;
            let ram_has_write = vals[idx];
            idx += 1;
            let shout_table_id = vals[idx];
            idx += 1;
            let opcode_flags = [
                vals[idx],
                vals[idx + 1],
                vals[idx + 2],
                vals[idx + 3],
                vals[idx + 4],
                vals[idx + 5],
                vals[idx + 6],
                vals[idx + 7],
                vals[idx + 8],
                vals[idx + 9],
                vals[idx + 10],
                vals[idx + 11],
            ];
            idx += 12;
            let funct3_is = [
                vals[idx],
                vals[idx + 1],
                vals[idx + 2],
                vals[idx + 3],
                vals[idx + 4],
                vals[idx + 5],
                vals[idx + 6],
                vals[idx + 7],
            ];
            idx += 8;
            let funct3_bits = [vals[idx], vals[idx + 1], vals[idx + 2]];
            idx += 3;
            let funct7_bits = [
                vals[idx],
                vals[idx + 1],
                vals[idx + 2],
                vals[idx + 3],
                vals[idx + 4],
                vals[idx + 5],
                vals[idx + 6],
            ];
            idx += 7;
            let rs2_decode = vals[idx];
            idx += 1;
            let imm_i = vals[idx];
            idx += 1;
            let imm_s = vals[idx];
            let rd_keep = K::ONE - rd_is_zero;
            let op_write_flags = [
                opcode_flags[0] * rd_keep,
                opcode_flags[1] * rd_keep,
                opcode_flags[2] * rd_keep,
                opcode_flags[3] * rd_keep,
                opcode_flags[7] * rd_keep,
                opcode_flags[8] * rd_keep,
            ];
            let alu_reg_table_delta = w2_alu_reg_table_delta_from_bits(funct7_bits, funct3_is);
            let alu_imm_table_delta = funct7_bits[5] * funct3_is[5];
            let alu_imm_shift_rhs_delta = (funct3_is[1] + funct3_is[5]) * (rs2_decode - imm_i);
            let selector_residuals = w2_decode_selector_residuals(
                active,
                decode_opcode,
                opcode_flags,
                funct3_is,
                funct3_bits,
                opcode_flags[11],
            );
            let bitness_residuals = w2_decode_bitness_residuals(opcode_flags, funct3_is);
            let alu_branch_residuals = w2_alu_branch_lookup_residuals(
                active,
                halted,
                shout_has_lookup,
                shout_lhs,
                shout_rhs,
                shout_table_id,
                rs1_val,
                rs2_val,
                rd_has_write,
                rd_is_zero,
                rd_val,
                ram_has_read,
                ram_has_write,
                ram_addr,
                shout_val,
                funct3_bits,
                funct7_bits,
                opcode_flags,
                op_write_flags,
                funct3_is,
                alu_reg_table_delta,
                alu_imm_table_delta,
                alu_imm_shift_rhs_delta,
                rs2_decode,
                imm_i,
                imm_s,
            );
            let mut weighted = K::ZERO;
            let mut w_idx = 0usize;
            for r in selector_residuals {
                weighted += fields_weights[w_idx] * r;
                w_idx += 1;
            }
            for r in bitness_residuals {
                weighted += fields_weights[w_idx] * r;
                w_idx += 1;
            }
            for r in alu_branch_residuals {
                weighted += fields_weights[w_idx] * r;
                w_idx += 1;
            }
            debug_assert_eq!(w_idx, fields_weights.len());
            debug_assert_eq!(idx + 1, vals.len());
            weighted
        }),
    );
    let imm_oracle = WeightedMaskOracleSparseTime::new(
        active_zero,
        imm_sparse_cols,
        w2_decode_imm_weight_vector(r_cycle, 4),
        r_cycle,
    );

    Ok((
        Some((Box::new(fields_oracle), K::ZERO)),
        Some((Box::new(imm_oracle), K::ZERO)),
    ))
}

pub(crate) type W3TimeClaims = (
    Option<(Box<dyn RoundOracle>, K)>,
    Option<(Box<dyn RoundOracle>, K)>,
    Option<(Box<dyn RoundOracle>, K)>,
    Option<(Box<dyn RoundOracle>, K)>,
    Option<(Box<dyn RoundOracle>, K)>,
);
