use super::*;

pub fn verify_route_a_memory_step(
    tr: &mut Poseidon2Transcript,
    cpu_bus: &BusLayout,
    m: usize,
    core_t: usize,
    step: &StepInstanceBundle<Cmt, F, K>,
    prev_step: Option<&StepInstanceBundle<Cmt, F, K>>,
    ccs_out0: &MeInstance<Cmt, F, K>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    batched_claimed_sums: &[K],
    claim_idx_start: usize,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    shout_pre: &[ShoutAddrPreVerifyData],
    twist_pre: &[TwistAddrPreVerifyData],
    step_idx: usize,
) -> Result<RouteAMemoryVerifyOutput, PiCcsError> {
    let chi_cycle_at_r_time = eq_points(r_time, r_cycle);
    if ccs_out0.r.as_slice() != r_time {
        return Err(PiCcsError::ProtocolError(
            "CPU ME output r mismatch (expected shared r_time)".into(),
        ));
    }
    let trace_mode = wb_wp_required_for_step_instance(step);
    let cpu_link = if trace_mode {
        extract_trace_cpu_link_openings(m, core_t, cpu_bus.bus_cols, step, ccs_out0)?
    } else {
        None
    };
    let enforce_trace_shout_linkage = trace_mode && !step.lut_insts.is_empty();
    if enforce_trace_shout_linkage && cpu_link.is_none() {
        return Err(PiCcsError::ProtocolError(
            "missing CPU trace linkage openings in shared-bus mode".into(),
        ));
    }
    let has_prev = prev_step.is_some();
    if let Some(prev) = prev_step {
        if prev.mem_insts.len() != step.mem_insts.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist rollover requires stable mem instance count: prev has {}, current has {}",
                prev.mem_insts.len(),
                step.mem_insts.len()
            )));
        }
        for (idx, (prev_inst, inst)) in prev.mem_insts.iter().zip(step.mem_insts.iter()).enumerate() {
            if prev_inst.d != inst.d
                || prev_inst.ell != inst.ell
                || prev_inst.k != inst.k
                || prev_inst.lanes != inst.lanes
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "Twist rollover requires stable geometry at mem_idx={}: prev (k={}, d={}, ell={}, lanes={}) vs cur (k={}, d={}, ell={}, lanes={})",
                    idx,
                    prev_inst.k,
                    prev_inst.d,
                    prev_inst.ell,
                    prev_inst.lanes,
                    inst.k,
                    inst.d,
                    inst.ell,
                    inst.lanes
                )));
            }
        }
    }

    for (idx, inst) in step.lut_insts.iter().enumerate() {
        if !inst.comms.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "shared CPU bus requires metadata-only Shout instances (comms must be empty, lut_idx={idx})"
            )));
        }
    }
    for (idx, inst) in step.mem_insts.iter().enumerate() {
        if !inst.comms.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "shared CPU bus requires metadata-only Twist instances (comms must be empty, mem_idx={idx})"
            )));
        }
    }
    if let Some(prev) = prev_step {
        for (idx, inst) in prev.lut_insts.iter().enumerate() {
            if !inst.comms.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared CPU bus requires metadata-only Shout instances (comms must be empty, prev lut_idx={idx})"
                )));
            }
        }
        for (idx, inst) in prev.mem_insts.iter().enumerate() {
            if !inst.comms.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared CPU bus requires metadata-only Twist instances (comms must be empty, prev mem_idx={idx})"
                )));
            }
        }
    }

    let proofs_mem = &mem_proof.proofs;

    if cpu_bus.shout_cols.len() != step.lut_insts.len() || cpu_bus.twist_cols.len() != step.mem_insts.len() {
        return Err(PiCcsError::InvalidInput(
            "shared_cpu_bus layout mismatch for step (instance counts)".into(),
        ));
    }

    let bus_y_base_time = if cpu_bus.bus_cols > 0 {
        let min_len = core_t
            .checked_add(cpu_bus.bus_cols)
            .ok_or_else(|| PiCcsError::InvalidInput("core_t + bus_cols overflow".into()))?;
        if ccs_out0.y_scalars.len() < min_len {
            return Err(PiCcsError::InvalidInput(
                "CPU y_scalars too short for shared-bus openings".into(),
            ));
        }
        core_t
    } else {
        0usize
    };
    let wb_enabled = wb_wp_required_for_step_instance(step);
    let wp_enabled = wb_wp_required_for_step_instance(step);
    let w2_enabled = decode_stage_required_for_step_instance(step);
    let w3_enabled = width_stage_required_for_step_instance(step);
    let control_enabled = control_stage_required_for_step_instance(step);
    let claim_plan = RouteATimeClaimPlan::build(
        step,
        claim_idx_start,
        wb_enabled,
        wp_enabled,
        w2_enabled,
        w3_enabled,
        control_enabled,
    )?;
    if claim_plan.claim_idx_end > batched_final_values.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "batched_final_values too short (need at least {}, have {})",
            claim_plan.claim_idx_end,
            batched_final_values.len()
        )));
    }
    if claim_plan.claim_idx_end > batched_claimed_sums.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "batched_claimed_sums too short (need at least {}, have {})",
            claim_plan.claim_idx_end,
            batched_claimed_sums.len()
        )));
    }

    let expected_proofs = step.lut_insts.len() + step.mem_insts.len();
    if proofs_mem.len() != expected_proofs {
        return Err(PiCcsError::InvalidInput(format!(
            "mem proof count mismatch (expected {}, got {})",
            expected_proofs,
            proofs_mem.len()
        )));
    }
    let total_shout_lanes: usize = step.lut_insts.iter().map(|inst| inst.lanes.max(1)).sum();
    if shout_pre.len() != total_shout_lanes {
        return Err(PiCcsError::InvalidInput(format!(
            "shout pre-time count mismatch (expected total_lanes={}, got {})",
            total_shout_lanes,
            shout_pre.len()
        )));
    }
    if twist_pre.len() != step.mem_insts.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "twist pre-time count mismatch (expected {}, got {})",
            step.mem_insts.len(),
            twist_pre.len()
        )));
    }

    let mut twist_time_openings: Vec<TwistTimeLaneOpenings> = Vec::with_capacity(step.mem_insts.len());

    // Shout instances first.
    let mut shout_lane_base: usize = 0;
    let mut shout_trace_sums = ShoutTraceLinkSums::default();
    #[derive(Clone)]
    struct ShoutGammaLaneVerifyData {
        has_lookup: K,
        val: K,
        addr_bits: Vec<K>,
        pre: ShoutAddrPreVerifyData,
    }
    let mut shout_addr_range_counts = std::collections::HashMap::<(usize, usize), usize>::new();
    for inst_cols in cpu_bus.shout_cols.iter() {
        for lane_cols in inst_cols.lanes.iter() {
            let key = (lane_cols.addr_bits.start, lane_cols.addr_bits.end);
            *shout_addr_range_counts.entry(key).or_insert(0) += 1;
        }
    }
    let mut shout_gamma_lane_data: Vec<Option<ShoutGammaLaneVerifyData>> = vec![None; total_shout_lanes];
    for (proof_idx, inst) in step.lut_insts.iter().enumerate() {
        match &proofs_mem[proof_idx] {
            MemOrLutProof::Shout(_proof) => {}
            _ => return Err(PiCcsError::InvalidInput("expected Shout proof".into())),
        }
        if matches!(
            inst.table_spec,
            Some(LutTableSpec::RiscvOpcodePacked { .. } | LutTableSpec::RiscvOpcodeEventTablePacked { .. })
        ) {
            return Err(PiCcsError::InvalidInput(
                "packed RISC-V Shout table specs are not supported on the shared CPU bus".into(),
            ));
        }

        let ell_addr = inst.d * inst.ell;
        let expected_lanes = inst.lanes.max(1);
        let lane_table_id = if enforce_trace_shout_linkage {
            rv32_trace_link_table_id_from_spec(&inst.table_spec)?.map(|table_id| K::from(F::from_u64(table_id as u64)))
        } else {
            None
        };

        let inst_cols = cpu_bus
            .shout_cols
            .get(proof_idx)
            .ok_or_else(|| PiCcsError::InvalidInput("shared_cpu_bus layout mismatch (shout)".into()))?;
        if inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at lut_idx={proof_idx}: bus shout lanes={} but instance expects {expected_lanes}",
                inst_cols.lanes.len()
            )));
        }

        struct ShoutLaneOpen {
            addr_bits: Vec<K>,
            has_lookup: K,
            val: K,
            shared_addr_group: bool,
            shared_addr_group_size: usize,
        }
        let mut lane_opens: Vec<ShoutLaneOpen> = Vec::with_capacity(expected_lanes);
        for (lane_idx, shout_cols) in inst_cols.lanes.iter().enumerate() {
            if shout_cols.addr_bits.end - shout_cols.addr_bits.start != ell_addr {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at lut_idx={proof_idx}, lane_idx={lane_idx}: expected ell_addr={ell_addr}"
                )));
            }

            let mut addr_bits_open = Vec::with_capacity(ell_addr);
            for (_j, col_id) in shout_cols.addr_bits.clone().enumerate() {
                addr_bits_open.push(
                    ccs_out0
                        .y_scalars
                        .get(cpu_bus.y_scalar_index(bus_y_base_time, col_id))
                        .copied()
                        .ok_or_else(|| {
                            PiCcsError::ProtocolError("CPU y_scalars missing Shout addr_bits opening".into())
                        })?,
                );
            }
            let has_lookup_open = ccs_out0
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_time, shout_cols.has_lookup))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Shout has_lookup opening".into()))?;
            let val_open = ccs_out0
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_time, shout_cols.primary_val()))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Shout val opening".into()))?;
            let key = (shout_cols.addr_bits.start, shout_cols.addr_bits.end);
            let shared_addr_group_size = shout_addr_range_counts.get(&key).copied().unwrap_or(0);
            let shared_addr_group = shared_addr_group_size > 1;

            lane_opens.push(ShoutLaneOpen {
                addr_bits: addr_bits_open,
                has_lookup: has_lookup_open,
                val: val_open,
                shared_addr_group,
                shared_addr_group_size,
            });
        }

        let shout_claims = claim_plan
            .shout
            .get(proof_idx)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("missing Shout claim schedule at index {}", proof_idx)))?;
        if shout_claims.lanes.len() != expected_lanes {
            return Err(PiCcsError::ProtocolError(format!(
                "Shout claim schedule lane count mismatch at lut_idx={proof_idx}: expected {expected_lanes}, got {}",
                shout_claims.lanes.len()
            )));
        }
        if shout_lane_base
            .checked_add(expected_lanes)
            .ok_or_else(|| PiCcsError::ProtocolError("shout lane index overflow".into()))?
            > shout_pre.len()
        {
            return Err(PiCcsError::ProtocolError("Shout pre-time lane indexing drift".into()));
        }

        // Route A Shout ordering in batched_time:
        // - value (time rounds only) per lane
        // - adapter (time rounds only) per lane
        // - aggregated bitness for (addr_bits, has_lookup)
        {
            let mut opens: Vec<K> = Vec::with_capacity(expected_lanes * (ell_addr + 1));
            for lane in lane_opens.iter() {
                opens.extend_from_slice(&lane.addr_bits);
                opens.push(lane.has_lookup);
            }
            let weights = bitness_weights(r_cycle, opens.len(), 0x5348_4F55_54u64 + proof_idx as u64);
            let mut acc = K::ZERO;
            for (w, b) in weights.iter().zip(opens.iter()) {
                acc += *w * *b * (*b - K::ONE);
            }
            let expected = chi_cycle_at_r_time * acc;
            if expected != batched_final_values[shout_claims.bitness] {
                return Err(PiCcsError::ProtocolError(
                    "shout/bitness terminal value mismatch".into(),
                ));
            }
        }

        for (lane_idx, lane) in lane_opens.iter().enumerate() {
            if let Some(lane_table_id) = lane_table_id {
                shout_trace_sums.has_lookup += lane.has_lookup;
                shout_trace_sums.val += lane.val;
                shout_trace_sums.table_id += lane.has_lookup * lane_table_id;
                let (lhs, rhs) = unpack_interleaved_halves_lsb(&lane.addr_bits)?;
                if lane.shared_addr_group {
                    let inv_count = K::from_u64(lane.shared_addr_group_size as u64).inverse();
                    shout_trace_sums.lhs += lhs * inv_count;
                    shout_trace_sums.rhs += rhs * inv_count;
                } else {
                    shout_trace_sums.lhs += lhs;
                    shout_trace_sums.rhs += rhs;
                }
            }

            let pre = shout_pre.get(shout_lane_base + lane_idx).ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "missing pre-time Shout lane data at index {}",
                    shout_lane_base + lane_idx
                ))
            })?;
            let lane_claims = shout_claims
                .lanes
                .get(lane_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("shout claim schedule lane idx drift".into()))?;

            if lane_claims.gamma_group.is_some() {
                if !pre.is_active {
                    if pre.addr_claim_sum != K::ZERO || pre.addr_final != K::ZERO || lane.has_lookup != K::ZERO {
                        return Err(PiCcsError::ProtocolError(
                            "shout gamma lane inactive-row invariants violated".into(),
                        ));
                    }
                }
                shout_gamma_lane_data[shout_lane_base + lane_idx] = Some(ShoutGammaLaneVerifyData {
                    has_lookup: lane.has_lookup,
                    val: lane.val,
                    addr_bits: lane.addr_bits.clone(),
                    pre: pre.clone(),
                });
            } else {
                let value_idx = lane_claims
                    .value
                    .ok_or_else(|| PiCcsError::ProtocolError("missing shout value claim idx".into()))?;
                let adapter_idx = lane_claims
                    .adapter
                    .ok_or_else(|| PiCcsError::ProtocolError("missing shout adapter claim idx".into()))?;
                let value_claim = batched_claimed_sums[value_idx];
                let value_final = batched_final_values[value_idx];
                let adapter_claim = batched_claimed_sums[adapter_idx];
                let adapter_final = batched_final_values[adapter_idx];

                let expected_value_final = chi_cycle_at_r_time * lane.has_lookup * lane.val;
                if expected_value_final != value_final {
                    return Err(PiCcsError::ProtocolError("shout value terminal value mismatch".into()));
                }

                let eq_addr = eq_bits_prod(&lane.addr_bits, &pre.r_addr)?;
                let expected_adapter_final = chi_cycle_at_r_time * lane.has_lookup * eq_addr;
                if expected_adapter_final != adapter_final {
                    return Err(PiCcsError::ProtocolError(
                        "shout adapter terminal value mismatch".into(),
                    ));
                }

                if value_claim != pre.addr_claim_sum {
                    return Err(PiCcsError::ProtocolError(
                        "shout value claimed sum != addr claimed sum".into(),
                    ));
                }

                if pre.is_active {
                    let expected_addr_final = pre.table_eval_at_r_addr * adapter_claim;
                    if expected_addr_final != pre.addr_final {
                        return Err(PiCcsError::ProtocolError("shout addr terminal value mismatch".into()));
                    }
                } else {
                    // If we skipped the addr-pre sumcheck, the only sound case is "no lookups".
                    // Enforce this by requiring the addr claim + adapter claim to be zero.
                    if pre.addr_claim_sum != K::ZERO {
                        return Err(PiCcsError::ProtocolError(
                            "shout addr-pre skipped but addr claim is nonzero".into(),
                        ));
                    }
                    if adapter_claim != K::ZERO {
                        return Err(PiCcsError::ProtocolError(
                            "shout addr-pre skipped but adapter claim is nonzero".into(),
                        ));
                    }
                    if pre.addr_final != K::ZERO {
                        return Err(PiCcsError::ProtocolError(
                            "shout addr-pre skipped but addr_final is nonzero".into(),
                        ));
                    }
                }
            }
        }

        shout_lane_base += expected_lanes;
    }
    if shout_lane_base != shout_pre.len() {
        return Err(PiCcsError::ProtocolError(
            "shout pre-time lanes not fully consumed".into(),
        ));
    }
    if !step.lut_insts.is_empty() && enforce_trace_shout_linkage {
        let cpu = cpu_link
            .ok_or_else(|| PiCcsError::ProtocolError("missing CPU trace linkage openings in shared-bus mode".into()))?;
        let expected_table_id = if decode_stage_required_for_step_instance(step) {
            Some(expected_trace_shout_table_id_from_openings(
                core_t, step, mem_proof, r_time,
            )?)
        } else {
            None
        };
        verify_non_event_trace_shout_linkage(cpu, shout_trace_sums, expected_table_id)?;
    }

    for group in claim_plan.shout_gamma_groups.iter() {
        let weights = bitness_weights(r_cycle, group.lanes.len(), 0x5348_5F47_414D_4Du64 ^ group.key);
        let value_claim = batched_claimed_sums[group.value];
        let value_final = batched_final_values[group.value];
        let adapter_claim = batched_claimed_sums[group.adapter];
        let adapter_final = batched_final_values[group.adapter];

        let mut expected_value_claim = K::ZERO;
        let mut expected_value_final = K::ZERO;
        let mut expected_adapter_claim = K::ZERO;
        let mut expected_adapter_final = K::ZERO;
        for (slot, lane_ref) in group.lanes.iter().enumerate() {
            let lane = shout_gamma_lane_data
                .get(lane_ref.flat_lane_idx)
                .and_then(|x| x.as_ref())
                .ok_or_else(|| PiCcsError::ProtocolError("missing shout gamma lane verify data".into()))?;
            let w = weights[slot];
            let eq_addr = eq_bits_prod(&lane.addr_bits, &lane.pre.r_addr)?;
            expected_value_claim += w * lane.pre.addr_claim_sum;
            expected_value_final += w * lane.has_lookup * lane.val;
            expected_adapter_claim += w * lane.pre.addr_final;
            expected_adapter_final += w * lane.pre.table_eval_at_r_addr * lane.has_lookup * eq_addr;
        }
        expected_value_final *= chi_cycle_at_r_time;
        expected_adapter_final *= chi_cycle_at_r_time;

        if value_claim != expected_value_claim {
            return Err(PiCcsError::ProtocolError(
                "shout gamma value claimed sum mismatch".into(),
            ));
        }
        if value_final != expected_value_final {
            return Err(PiCcsError::ProtocolError(
                "shout gamma value terminal mismatch".into(),
            ));
        }
        if adapter_claim != expected_adapter_claim {
            return Err(PiCcsError::ProtocolError(
                "shout gamma adapter claimed sum mismatch".into(),
            ));
        }
        if adapter_final != expected_adapter_final {
            return Err(PiCcsError::ProtocolError(
                "shout gamma adapter terminal mismatch".into(),
            ));
        }
    }

    // Twist instances next.
    let proof_mem_offset = step.lut_insts.len();

    // --------------------------------------------------------------------
    // Twist time checks at addr-pre `r_addr`.
    // --------------------------------------------------------------------
    for (i_mem, inst) in step.mem_insts.iter().enumerate() {
        let twist_proof = match &proofs_mem[proof_mem_offset + i_mem] {
            MemOrLutProof::Twist(proof) => proof,
            _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
        };
        let layout = inst.twist_layout();
        let ell_addr = layout
            .lanes
            .get(0)
            .ok_or_else(|| PiCcsError::InvalidInput("TwistWitnessLayout has no lanes".into()))?
            .ell_addr;

        let twist_inst_cols = cpu_bus
            .twist_cols
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput("shared_cpu_bus layout mismatch (twist)".into()))?;
        let expected_lanes = inst.lanes.max(1);
        if twist_inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at mem_idx={i_mem}: expected lanes={expected_lanes}, got {}",
                twist_inst_cols.lanes.len()
            )));
        }

        struct TwistLaneTimeOpen {
            ra_bits: Vec<K>,
            wa_bits: Vec<K>,
            has_read: K,
            has_write: K,
            wv: K,
            rv: K,
            inc: K,
        }

        let mut lane_opens: Vec<TwistLaneTimeOpen> = Vec::with_capacity(twist_inst_cols.lanes.len());
        for (lane_idx, twist_cols) in twist_inst_cols.lanes.iter().enumerate() {
            if twist_cols.ra_bits.end - twist_cols.ra_bits.start != ell_addr
                || twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at mem_idx={i_mem}, lane={lane_idx}: expected ell_addr={ell_addr}"
                )));
            }

            let mut ra_bits_open = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.ra_bits.clone() {
                ra_bits_open.push(
                    ccs_out0
                        .y_scalars
                        .get(cpu_bus.y_scalar_index(bus_y_base_time, col_id))
                        .copied()
                        .ok_or_else(|| {
                            PiCcsError::ProtocolError("CPU y_scalars missing Twist ra_bits opening".into())
                        })?,
                );
            }
            let mut wa_bits_open = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.wa_bits.clone() {
                wa_bits_open.push(
                    ccs_out0
                        .y_scalars
                        .get(cpu_bus.y_scalar_index(bus_y_base_time, col_id))
                        .copied()
                        .ok_or_else(|| {
                            PiCcsError::ProtocolError("CPU y_scalars missing Twist wa_bits opening".into())
                        })?,
                );
            }

            let has_read_open = ccs_out0
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_time, twist_cols.has_read))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Twist has_read opening".into()))?;
            let has_write_open = ccs_out0
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_time, twist_cols.has_write))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Twist has_write opening".into()))?;
            let wv_open = ccs_out0
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_time, twist_cols.wv))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Twist wv opening".into()))?;
            let rv_open = ccs_out0
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_time, twist_cols.rv))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Twist rv opening".into()))?;
            let inc_write_open = ccs_out0
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_time, twist_cols.inc))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Twist inc opening".into()))?;

            lane_opens.push(TwistLaneTimeOpen {
                ra_bits: ra_bits_open,
                wa_bits: wa_bits_open,
                has_read: has_read_open,
                has_write: has_write_open,
                wv: wv_open,
                rv: rv_open,
                inc: inc_write_open,
            });
        }

        let pre = twist_pre
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing Twist pre-time data at index {}", i_mem)))?;
        let r_addr = &pre.r_addr;
        if r_addr.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist r_addr.len()={}, expected ell_addr={}",
                r_addr.len(),
                ell_addr
            )));
        }

        let twist_claims = claim_plan
            .twist
            .get(i_mem)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("missing Twist claim schedule at index {}", i_mem)))?;

        // Route A Twist ordering in batched_time:
        // - read_check (time rounds only)
        // - write_check (time rounds only)
        // - bitness for ra_bits then wa_bits then has_read then has_write (time-only)
        let read_check_claim = batched_claimed_sums[twist_claims.read_check];
        let read_check_final = batched_final_values[twist_claims.read_check];
        let write_check_claim = batched_claimed_sums[twist_claims.write_check];
        let write_check_final = batched_final_values[twist_claims.write_check];

        if read_check_claim != pre.read_check_claim_sum {
            return Err(PiCcsError::ProtocolError(
                "twist read_check claimed sum != addr-pre final".into(),
            ));
        }
        if write_check_claim != pre.write_check_claim_sum {
            return Err(PiCcsError::ProtocolError(
                "twist write_check claimed sum != addr-pre final".into(),
            ));
        }

        // Aggregated bitness terminal check (ra_bits, wa_bits, has_read, has_write).
        {
            let mut opens: Vec<K> = Vec::with_capacity(expected_lanes * (2 * ell_addr + 2));
            for lane in lane_opens.iter() {
                opens.extend_from_slice(&lane.ra_bits);
                opens.extend_from_slice(&lane.wa_bits);
                opens.push(lane.has_read);
                opens.push(lane.has_write);
            }
            let weights = bitness_weights(r_cycle, opens.len(), 0x5457_4953_54u64 + i_mem as u64);
            let mut acc = K::ZERO;
            for (w, b) in weights.iter().zip(opens.iter()) {
                acc += *w * *b * (*b - K::ONE);
            }
            let expected = chi_cycle_at_r_time * acc;
            if expected != batched_final_values[twist_claims.bitness] {
                return Err(PiCcsError::ProtocolError(
                    "twist/bitness terminal value mismatch".into(),
                ));
            }
        }

        let val_eval = twist_proof
            .val_eval
            .as_ref()
            .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;

        let init_at_r_addr = eval_init_at_r_addr(&inst.init, inst.k, r_addr)?;
        let claimed_val = init_at_r_addr + val_eval.claimed_inc_sum_lt;

        // Terminal checks for read_check / write_check at (r_time, r_addr).
        let mut expected_read_check_final = K::ZERO;
        let mut expected_write_check_final = K::ZERO;
        for lane in lane_opens.iter() {
            let read_eq_addr = eq_bits_prod(&lane.ra_bits, r_addr)?;
            expected_read_check_final += chi_cycle_at_r_time * lane.has_read * (claimed_val - lane.rv) * read_eq_addr;

            let write_eq_addr = eq_bits_prod(&lane.wa_bits, r_addr)?;
            expected_write_check_final +=
                chi_cycle_at_r_time * lane.has_write * (lane.wv - claimed_val - lane.inc) * write_eq_addr;
        }
        if expected_read_check_final != read_check_final {
            return Err(PiCcsError::ProtocolError(
                "twist/read_check terminal value mismatch".into(),
            ));
        }

        if expected_write_check_final != write_check_final {
            return Err(PiCcsError::ProtocolError(
                "twist/write_check terminal value mismatch".into(),
            ));
        }

        twist_time_openings.push(TwistTimeLaneOpenings {
            lanes: lane_opens
                .into_iter()
                .map(|lane| TwistTimeLaneOpeningsLane {
                    wa_bits: lane.wa_bits,
                    has_write: lane.has_write,
                    inc_at_write_addr: lane.inc,
                })
                .collect(),
        });
    }

    // --------------------------------------------------------------------
    // Phase 2: Verify batched Twist val-eval sum-check, deriving shared r_val.
    // --------------------------------------------------------------------
    let mut r_val: Vec<K> = Vec::new();
    let mut val_eval_finals: Vec<K> = Vec::new();
    if !step.mem_insts.is_empty() {
        let plan = crate::memory_sidecar::claim_plan::TwistValEvalClaimPlan::build(step.mem_insts.iter(), has_prev);
        let claim_count = plan.claim_count;

        let mut per_claim_rounds: Vec<Vec<Vec<K>>> = Vec::with_capacity(claim_count);
        let mut per_claim_sums: Vec<K> = Vec::with_capacity(claim_count);
        let mut bind_claims: Vec<(u8, K)> = Vec::with_capacity(claim_count);
        let mut claim_idx = 0usize;

        for (i_mem, _inst) in step.mem_insts.iter().enumerate() {
            let twist_proof = match &proofs_mem[proof_mem_offset + i_mem] {
                MemOrLutProof::Twist(proof) => proof,
                _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
            };
            let val = twist_proof
                .val_eval
                .as_ref()
                .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;

            per_claim_rounds.push(val.rounds_lt.clone());
            per_claim_sums.push(val.claimed_inc_sum_lt);
            bind_claims.push((plan.bind_tags[claim_idx], val.claimed_inc_sum_lt));
            claim_idx += 1;

            per_claim_rounds.push(val.rounds_total.clone());
            per_claim_sums.push(val.claimed_inc_sum_total);
            bind_claims.push((plan.bind_tags[claim_idx], val.claimed_inc_sum_total));
            claim_idx += 1;

            if has_prev {
                let prev_total = val.claimed_prev_inc_sum_total.ok_or_else(|| {
                    PiCcsError::InvalidInput("Twist(Route A): missing claimed_prev_inc_sum_total".into())
                })?;
                let prev_rounds = val
                    .rounds_prev_total
                    .clone()
                    .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing rounds_prev_total".into()))?;
                per_claim_rounds.push(prev_rounds);
                per_claim_sums.push(prev_total);
                bind_claims.push((plan.bind_tags[claim_idx], prev_total));
                claim_idx += 1;
            } else if val.claimed_prev_inc_sum_total.is_some() || val.rounds_prev_total.is_some() {
                return Err(PiCcsError::InvalidInput(
                    "Twist(Route A): rollover fields present but prev_step is None".into(),
                ));
            }
        }

        tr.append_message(
            b"twist/val_eval/batch_start",
            &(step.mem_insts.len() as u64).to_le_bytes(),
        );
        tr.append_message(b"twist/val_eval/step_idx", &(step_idx as u64).to_le_bytes());
        bind_twist_val_eval_claim_sums(tr, &bind_claims);

        let (r_val_out, finals_out, ok) = verify_batched_sumcheck_rounds_ds(
            tr,
            b"twist/val_eval_batch",
            step_idx,
            &per_claim_rounds,
            &per_claim_sums,
            &plan.labels,
            &plan.degree_bounds,
        );
        if !ok {
            return Err(PiCcsError::SumcheckError(
                "twist val-eval batched sumcheck invalid".into(),
            ));
        }
        if r_val_out.len() != r_time.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval r_val.len()={}, expected ell_n={}",
                r_val_out.len(),
                r_time.len()
            )));
        }
        if finals_out.len() != claim_count {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval finals.len()={}, expected {}",
                finals_out.len(),
                claim_count
            )));
        }
        r_val = r_val_out;
        val_eval_finals = finals_out;

        tr.append_message(b"twist/val_eval/batch_done", &[]);
    }

    // Verify val-eval terminal identity against CPU ME openings at r_val.
    let lt = if step.mem_insts.is_empty() {
        if !r_val.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "twist val-eval produced r_val but no mem instances are present".into(),
            ));
        }
        K::ZERO
    } else {
        if r_val.len() != r_time.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval r_val.len()={}, expected ell_n={}",
                r_val.len(),
                r_time.len()
            )));
        }
        lt_eval(&r_val, r_time)
    };

    let (cpu_me_val_cur, cpu_me_val_prev, bus_y_base_val) = if step.mem_insts.is_empty() {
        if !mem_proof.val_me_claims.is_empty() {
            return Err(PiCcsError::InvalidInput(
                "proof contains val-lane CPU ME claims with no Twist instances".into(),
            ));
        }
        (None, None, 0usize)
    } else {
        let expected = 1usize + usize::from(has_prev);
        if mem_proof.val_me_claims.len() != expected {
            return Err(PiCcsError::InvalidInput(format!(
                "shared bus expects {} CPU ME claim(s) at r_val, got {}",
                expected,
                mem_proof.val_me_claims.len()
            )));
        }

        let cpu_me_cur = mem_proof
            .val_me_claims
            .get(0)
            .ok_or_else(|| PiCcsError::ProtocolError("missing CPU ME claim at r_val".into()))?;
        if cpu_me_cur.r.as_slice() != r_val {
            return Err(PiCcsError::ProtocolError(
                "CPU ME(val) r mismatch (expected r_val)".into(),
            ));
        }
        if cpu_me_cur.c != step.mcs_inst.c {
            return Err(PiCcsError::ProtocolError(
                "CPU ME(val) commitment mismatch (current step)".into(),
            ));
        }
        let cpu_me_prev = if has_prev {
            let prev_inst =
                prev_step.ok_or_else(|| PiCcsError::ProtocolError("prev_step missing with has_prev=true".into()))?;
            let cpu_me_prev = mem_proof
                .val_me_claims
                .get(1)
                .ok_or_else(|| PiCcsError::ProtocolError("missing prev CPU ME claim at r_val".into()))?;
            if cpu_me_prev.r.as_slice() != r_val {
                return Err(PiCcsError::ProtocolError(
                    "CPU ME(val/prev) r mismatch (expected r_val)".into(),
                ));
            }
            if cpu_me_prev.c != prev_inst.mcs_inst.c {
                return Err(PiCcsError::ProtocolError("CPU ME(val/prev) commitment mismatch".into()));
            }
            Some(cpu_me_prev)
        } else {
            None
        };

        let bus_y_base_val = cpu_me_cur
            .y_scalars
            .len()
            .checked_sub(cpu_bus.bus_cols)
            .ok_or_else(|| PiCcsError::InvalidInput("CPU y_scalars too short for bus openings".into()))?;

        (Some(cpu_me_cur), cpu_me_prev, bus_y_base_val)
    };

    for (i_mem, inst) in step.mem_insts.iter().enumerate() {
        let twist_proof = match &proofs_mem[proof_mem_offset + i_mem] {
            MemOrLutProof::Twist(proof) => proof,
            _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
        };
        let val_eval = twist_proof
            .val_eval
            .as_ref()
            .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;
        let layout = inst.twist_layout();
        let ell_addr = layout
            .lanes
            .get(0)
            .ok_or_else(|| PiCcsError::InvalidInput("TwistWitnessLayout has no lanes".into()))?
            .ell_addr;

        let cpu_me_cur =
            cpu_me_val_cur.ok_or_else(|| PiCcsError::ProtocolError("missing CPU ME claim at r_val".into()))?;

        let twist_inst_cols = cpu_bus
            .twist_cols
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput("shared_cpu_bus layout mismatch (twist)".into()))?;
        let expected_lanes = inst.lanes.max(1);
        if twist_inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at mem_idx={i_mem}: expected lanes={expected_lanes}, got {}",
                twist_inst_cols.lanes.len()
            )));
        }

        let r_addr = twist_pre
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing Twist pre-time data at index {}", i_mem)))?
            .r_addr
            .as_slice();

        let mut inc_at_r_addr_val = K::ZERO;
        for (lane_idx, twist_cols) in twist_inst_cols.lanes.iter().enumerate() {
            if twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at mem_idx={i_mem}, lane={lane_idx}: expected ell_addr={ell_addr}"
                )));
            }

            let mut wa_bits_val_open = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.wa_bits.clone() {
                wa_bits_val_open.push(
                    cpu_me_cur
                        .y_scalars
                        .get(cpu_bus.y_scalar_index(bus_y_base_val, col_id))
                        .copied()
                        .ok_or_else(|| {
                            PiCcsError::ProtocolError("CPU y_scalars missing wa_bits(val) opening".into())
                        })?,
                );
            }
            let has_write_val_open = cpu_me_cur
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_val, twist_cols.has_write))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing has_write(val) opening".into()))?;
            let inc_at_write_addr_val_open = cpu_me_cur
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_val, twist_cols.inc))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing inc(val) opening".into()))?;

            let eq_wa_val = eq_bits_prod(&wa_bits_val_open, r_addr)?;
            inc_at_r_addr_val += has_write_val_open * inc_at_write_addr_val_open * eq_wa_val;
        }

        let expected_lt_final = inc_at_r_addr_val * lt;
        let claims_per_mem = if has_prev { 3 } else { 2 };
        let base = claims_per_mem * i_mem;
        if expected_lt_final != val_eval_finals[base] {
            return Err(PiCcsError::ProtocolError(
                "twist/val_eval_lt terminal value mismatch".into(),
            ));
        }
        let expected_total_final = inc_at_r_addr_val;
        if expected_total_final != val_eval_finals[base + 1] {
            return Err(PiCcsError::ProtocolError(
                "twist/val_eval_total terminal value mismatch".into(),
            ));
        }

        if has_prev {
            let prev =
                prev_step.ok_or_else(|| PiCcsError::ProtocolError("prev_step missing with has_prev=true".into()))?;
            let prev_inst = prev
                .mem_insts
                .get(i_mem)
                .ok_or_else(|| PiCcsError::ProtocolError("missing prev mem instance".into()))?;
            let cpu_me_prev = cpu_me_val_prev
                .ok_or_else(|| PiCcsError::ProtocolError("missing prev CPU ME claim at r_val".into()))?;

            // Terminal check for prev-total: uses previous-step openings at current r_val.
            let mut inc_at_r_addr_prev = K::ZERO;
            for (lane_idx, twist_cols) in twist_inst_cols.lanes.iter().enumerate() {
                if twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr {
                    return Err(PiCcsError::InvalidInput(format!(
                        "shared_cpu_bus layout mismatch at mem_idx={i_mem}, lane={lane_idx}: expected ell_addr={ell_addr}"
                    )));
                }

                let mut wa_bits_prev_open = Vec::with_capacity(ell_addr);
                for col_id in twist_cols.wa_bits.clone() {
                    wa_bits_prev_open.push(
                        cpu_me_prev
                            .y_scalars
                            .get(cpu_bus.y_scalar_index(bus_y_base_val, col_id))
                            .copied()
                            .ok_or_else(|| {
                                PiCcsError::ProtocolError("CPU y_scalars missing wa_bits(prev) opening".into())
                            })?,
                    );
                }
                let has_write_prev_open = cpu_me_prev
                    .y_scalars
                    .get(cpu_bus.y_scalar_index(bus_y_base_val, twist_cols.has_write))
                    .copied()
                    .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing has_write(prev) opening".into()))?;
                let inc_prev_open = cpu_me_prev
                    .y_scalars
                    .get(cpu_bus.y_scalar_index(bus_y_base_val, twist_cols.inc))
                    .copied()
                    .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing inc(prev) opening".into()))?;

                let eq_wa_prev = eq_bits_prod(&wa_bits_prev_open, r_addr)?;
                inc_at_r_addr_prev += has_write_prev_open * inc_prev_open * eq_wa_prev;
            }
            if inc_at_r_addr_prev != val_eval_finals[base + 2] {
                return Err(PiCcsError::ProtocolError(
                    "twist/rollover_prev_total terminal value mismatch".into(),
                ));
            }

            // Enforce rollover equation: Init_i(r_addr) == Init_{i-1}(r_addr) + PrevTotal(i).
            let claimed_prev_total = val_eval
                .claimed_prev_inc_sum_total
                .ok_or_else(|| PiCcsError::ProtocolError("twist rollover missing claimed_prev_inc_sum_total".into()))?;
            let init_prev_at_r_addr = eval_init_at_r_addr(&prev_inst.init, prev_inst.k, r_addr)?;
            let init_cur_at_r_addr = eval_init_at_r_addr(&inst.init, inst.k, r_addr)?;
            if init_cur_at_r_addr != init_prev_at_r_addr + claimed_prev_total {
                return Err(PiCcsError::ProtocolError("twist rollover init check failed".into()));
            }
        }
    }

    verify_route_a_wb_wp_terminals(
        core_t,
        step,
        r_time,
        r_cycle,
        batched_final_values,
        &claim_plan,
        mem_proof,
    )?;
    verify_route_a_decode_terminals(
        core_t,
        step,
        r_time,
        r_cycle,
        batched_final_values,
        &claim_plan,
        mem_proof,
    )?;
    verify_route_a_width_terminals(
        core_t,
        step,
        r_time,
        r_cycle,
        batched_final_values,
        &claim_plan,
        mem_proof,
    )?;
    verify_route_a_control_terminals(
        core_t,
        step,
        r_time,
        r_cycle,
        batched_final_values,
        &claim_plan,
        mem_proof,
    )?;

    Ok(RouteAMemoryVerifyOutput {
        claim_idx_end: claim_plan.claim_idx_end,
        twist_time_openings,
    })
}
