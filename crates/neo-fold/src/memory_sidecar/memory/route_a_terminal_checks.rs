use super::*;

pub(crate) fn verify_route_a_decode_terminals(
    core_t: usize,
    step: &StepInstanceBundle<Cmt, F, K>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    claim_plan: &RouteATimeClaimPlan,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
) -> Result<(), PiCcsError> {
    if claim_plan.decode_fields.is_none() && claim_plan.decode_immediates.is_none() {
        return Ok(());
    }

    if mem_proof.wb_me_claims.len() != 1 {
        return Err(PiCcsError::ProtocolError(
            "W2 requires WB ME openings for shared active/bit terminals".into(),
        ));
    }

    let decode_layout = Rv32DecodeSidecarLayout::new();
    let decode_open_cols = rv32_decode_lookup_backed_cols(&decode_layout);
    if mem_proof.wp_me_claims.len() != 1 {
        return Err(PiCcsError::ProtocolError(
            "W2 requires WP ME openings for shared main-trace/decode terminals".into(),
        ));
    }
    let wp_me = &mem_proof.wp_me_claims[0];
    if wp_me.r.as_slice() != r_time {
        return Err(PiCcsError::ProtocolError(
            "W2 WP ME claim r mismatch (expected r_time)".into(),
        ));
    }
    if wp_me.c != step.mcs_inst.c {
        return Err(PiCcsError::ProtocolError("W2 WP ME claim commitment mismatch".into()));
    }
    if wp_me.m_in != step.mcs_inst.m_in {
        return Err(PiCcsError::ProtocolError("W2 WP ME claim m_in mismatch".into()));
    }
    let trace = Rv32TraceLayout::new();
    let wp_cols = rv32_trace_wp_opening_columns(&trace);
    let control_extra_cols = if control_stage_required_for_step_instance(step) {
        rv32_trace_control_extra_opening_columns(&trace)
    } else {
        Vec::new()
    };
    let decode_open_start = core_t
        .checked_add(wp_cols.len())
        .and_then(|v| v.checked_add(control_extra_cols.len()))
        .ok_or_else(|| PiCcsError::InvalidInput("W2 decode opening start overflow".into()))?;
    let decode_open_end = decode_open_start
        .checked_add(decode_open_cols.len())
        .ok_or_else(|| PiCcsError::InvalidInput("W2 decode opening end overflow".into()))?;
    if wp_me.y_scalars.len() < decode_open_end {
        return Err(PiCcsError::ProtocolError(format!(
            "W2 decode openings missing on WP ME claim (got {}, need at least {decode_open_end})",
            wp_me.y_scalars.len()
        )));
    }
    let decode_open = &wp_me.y_scalars[decode_open_start..decode_open_end];
    let decode_open_map: BTreeMap<usize, K> = decode_open_cols
        .iter()
        .copied()
        .zip(decode_open.iter().copied())
        .collect();
    let decode_open_col = |col_id: usize| -> Result<K, PiCcsError> {
        decode_open_map
            .get(&col_id)
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W2(shared) missing decode opening col_id={col_id}")))
    };
    let wb_me = &mem_proof.wb_me_claims[0];
    let wb_cols = rv32_trace_wb_columns(&trace);
    let need_wb = core_t
        .checked_add(wb_cols.len())
        .ok_or_else(|| PiCcsError::InvalidInput("W2 WB opening count overflow".into()))?;
    if wb_me.y_scalars.len() != need_wb {
        return Err(PiCcsError::ProtocolError(format!(
            "W2 WB opening length mismatch (got {}, expected {need_wb})",
            wb_me.y_scalars.len()
        )));
    }
    let wb_open = &wb_me.y_scalars[core_t..];
    let wb_open_col = |col_id: usize| -> Result<K, PiCcsError> {
        let idx = wb_cols
            .iter()
            .position(|&c| c == col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing WB opening column {col_id}")))?;
        Ok(wb_open[idx])
    };

    let wp_cols = rv32_trace_wp_opening_columns(&trace);
    let need_wp = core_t
        .checked_add(wp_cols.len())
        .ok_or_else(|| PiCcsError::InvalidInput("W2 WP opening count overflow".into()))?;
    if wp_me.y_scalars.len() < need_wp {
        return Err(PiCcsError::ProtocolError(format!(
            "W2 WP opening length mismatch (got {}, expected at least {need_wp})",
            wp_me.y_scalars.len()
        )));
    }
    let wp_open = &wp_me.y_scalars[core_t..need_wp];
    let wp_open_col = |col_id: usize| -> Result<K, PiCcsError> {
        let idx = wp_cols
            .iter()
            .position(|&c| c == col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W2 missing WP opening column {col_id}")))?;
        Ok(wp_open[idx])
    };

    if let Some(claim_idx) = claim_plan.decode_fields {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "w2/decode_fields claim index out of range".into(),
            ));
        }
        let opcode_flags = [
            decode_open_col(decode_layout.op_lui)?,
            decode_open_col(decode_layout.op_auipc)?,
            decode_open_col(decode_layout.op_jal)?,
            decode_open_col(decode_layout.op_jalr)?,
            decode_open_col(decode_layout.op_branch)?,
            decode_open_col(decode_layout.op_load)?,
            decode_open_col(decode_layout.op_store)?,
            decode_open_col(decode_layout.op_alu_imm)?,
            decode_open_col(decode_layout.op_alu_reg)?,
            decode_open_col(decode_layout.op_misc_mem)?,
            decode_open_col(decode_layout.op_system)?,
            decode_open_col(decode_layout.op_amo)?,
        ];
        let funct3_is = [
            decode_open_col(decode_layout.funct3_is[0])?,
            decode_open_col(decode_layout.funct3_is[1])?,
            decode_open_col(decode_layout.funct3_is[2])?,
            decode_open_col(decode_layout.funct3_is[3])?,
            decode_open_col(decode_layout.funct3_is[4])?,
            decode_open_col(decode_layout.funct3_is[5])?,
            decode_open_col(decode_layout.funct3_is[6])?,
            decode_open_col(decode_layout.funct3_is[7])?,
        ];
        let funct3_bits = [
            decode_open_col(decode_layout.funct3_bit[0])?,
            decode_open_col(decode_layout.funct3_bit[1])?,
            decode_open_col(decode_layout.funct3_bit[2])?,
        ];
        let funct7_bits = [
            decode_open_col(decode_layout.funct7_bit[0])?,
            decode_open_col(decode_layout.funct7_bit[1])?,
            decode_open_col(decode_layout.funct7_bit[2])?,
            decode_open_col(decode_layout.funct7_bit[3])?,
            decode_open_col(decode_layout.funct7_bit[4])?,
            decode_open_col(decode_layout.funct7_bit[5])?,
            decode_open_col(decode_layout.funct7_bit[6])?,
        ];
        let rd_is_zero = decode_open_col(decode_layout.rd_is_zero)?;
        let op_write_flags = [
            opcode_flags[0] * (K::ONE - rd_is_zero),
            opcode_flags[1] * (K::ONE - rd_is_zero),
            opcode_flags[2] * (K::ONE - rd_is_zero),
            opcode_flags[3] * (K::ONE - rd_is_zero),
            opcode_flags[7] * (K::ONE - rd_is_zero),
            opcode_flags[8] * (K::ONE - rd_is_zero),
        ];
        let alu_reg_table_delta = funct7_bits[5] * (funct3_is[0] + funct3_is[5]);
        let alu_imm_table_delta = funct7_bits[5] * funct3_is[5];
        let rs2_decode = decode_open_col(decode_layout.rs2)?;
        let imm_i = decode_open_col(decode_layout.imm_i)?;
        let alu_imm_shift_rhs_delta = (funct3_is[1] + funct3_is[5]) * (rs2_decode - imm_i);
        let shout_has_lookup = wp_open_col(trace.shout_has_lookup)?;
        let rs1_val = wp_open_col(trace.rs1_val)?;
        let shout_lhs = wp_open_col(trace.shout_lhs)?;
        let shout_table_id = decode_open_col(decode_layout.shout_table_id)?;

        let selector_residuals = w2_decode_selector_residuals(
            wp_open_col(trace.active)?,
            decode_open_col(decode_layout.opcode)?,
            opcode_flags,
            funct3_is,
            funct3_bits,
            decode_open_col(decode_layout.op_amo)?,
        );
        let bitness_residuals = w2_decode_bitness_residuals(opcode_flags, funct3_is);
        let alu_branch_residuals = w2_alu_branch_lookup_residuals(
            wp_open_col(trace.active)?,
            wb_open_col(trace.halted)?,
            shout_has_lookup,
            shout_lhs,
            wp_open_col(trace.shout_rhs)?,
            shout_table_id,
            rs1_val,
            wp_open_col(trace.rs2_val)?,
            decode_open_col(decode_layout.rd_has_write)?,
            rd_is_zero,
            wp_open_col(trace.rd_val)?,
            decode_open_col(decode_layout.ram_has_read)?,
            decode_open_col(decode_layout.ram_has_write)?,
            wp_open_col(trace.ram_addr)?,
            wp_open_col(trace.shout_val)?,
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
            decode_open_col(decode_layout.imm_s)?,
        );

        let mut residuals = Vec::with_capacity(W2_FIELDS_RESIDUAL_COUNT);
        residuals.extend_from_slice(&selector_residuals);
        residuals.extend_from_slice(&bitness_residuals);
        residuals.extend_from_slice(&alu_branch_residuals);
        let mut weighted = K::ZERO;
        let weights = w2_decode_pack_weight_vector(r_cycle, residuals.len());
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "w2/decode_fields terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.decode_immediates {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "w2/decode_immediates claim index out of range".into(),
            ));
        }
        let residuals = w2_decode_immediate_residuals(
            decode_open_col(decode_layout.imm_i)?,
            decode_open_col(decode_layout.imm_s)?,
            decode_open_col(decode_layout.imm_b)?,
            decode_open_col(decode_layout.imm_j)?,
            [
                decode_open_col(decode_layout.rd_bit[0])?,
                decode_open_col(decode_layout.rd_bit[1])?,
                decode_open_col(decode_layout.rd_bit[2])?,
                decode_open_col(decode_layout.rd_bit[3])?,
                decode_open_col(decode_layout.rd_bit[4])?,
            ],
            [
                decode_open_col(decode_layout.funct3_bit[0])?,
                decode_open_col(decode_layout.funct3_bit[1])?,
                decode_open_col(decode_layout.funct3_bit[2])?,
            ],
            [
                decode_open_col(decode_layout.rs1_bit[0])?,
                decode_open_col(decode_layout.rs1_bit[1])?,
                decode_open_col(decode_layout.rs1_bit[2])?,
                decode_open_col(decode_layout.rs1_bit[3])?,
                decode_open_col(decode_layout.rs1_bit[4])?,
            ],
            [
                decode_open_col(decode_layout.rs2_bit[0])?,
                decode_open_col(decode_layout.rs2_bit[1])?,
                decode_open_col(decode_layout.rs2_bit[2])?,
                decode_open_col(decode_layout.rs2_bit[3])?,
                decode_open_col(decode_layout.rs2_bit[4])?,
            ],
            [
                decode_open_col(decode_layout.funct7_bit[0])?,
                decode_open_col(decode_layout.funct7_bit[1])?,
                decode_open_col(decode_layout.funct7_bit[2])?,
                decode_open_col(decode_layout.funct7_bit[3])?,
                decode_open_col(decode_layout.funct7_bit[4])?,
                decode_open_col(decode_layout.funct7_bit[5])?,
                decode_open_col(decode_layout.funct7_bit[6])?,
            ],
        );
        let mut weighted = K::ZERO;
        let weights = w2_decode_imm_weight_vector(r_cycle, residuals.len());
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "w2/decode_immediates terminal value mismatch".into(),
            ));
        }
    }

    Ok(())
}

pub(crate) fn verify_route_a_width_terminals(
    core_t: usize,
    step: &StepInstanceBundle<Cmt, F, K>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    claim_plan: &RouteATimeClaimPlan,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
) -> Result<(), PiCcsError> {
    let any_w3_claim = claim_plan.width_bitness.is_some()
        || claim_plan.width_quiescence.is_some()
        || claim_plan.width_selector_linkage.is_some()
        || claim_plan.width_load_semantics.is_some()
        || claim_plan.width_store_semantics.is_some();
    if !any_w3_claim {
        return Ok(());
    }

    if mem_proof.wp_me_claims.len() != 1 {
        return Err(PiCcsError::ProtocolError(
            "W3 requires WP ME openings for shared main-trace terminals".into(),
        ));
    }

    let trace = Rv32TraceLayout::new();
    let width = Rv32WidthSidecarLayout::new();
    let decode = Rv32DecodeSidecarLayout::new();

    let wp_me = &mem_proof.wp_me_claims[0];
    if wp_me.r.as_slice() != r_time {
        return Err(PiCcsError::ProtocolError(
            "W3 WP ME claim r mismatch (expected r_time)".into(),
        ));
    }
    if wp_me.c != step.mcs_inst.c {
        return Err(PiCcsError::ProtocolError("W3 WP ME claim commitment mismatch".into()));
    }
    if wp_me.m_in != step.mcs_inst.m_in {
        return Err(PiCcsError::ProtocolError("W3 WP ME claim m_in mismatch".into()));
    }
    let wp_cols = rv32_trace_wp_opening_columns(&trace);
    let need_wp = core_t
        .checked_add(wp_cols.len())
        .ok_or_else(|| PiCcsError::InvalidInput("W3 WP opening count overflow".into()))?;
    if wp_me.y_scalars.len() < need_wp {
        return Err(PiCcsError::ProtocolError(format!(
            "W3 WP ME opening length mismatch (got {}, expected at least {need_wp})",
            wp_me.y_scalars.len()
        )));
    }
    let wp_open = &wp_me.y_scalars[core_t..need_wp];
    let wp_open_col = |col_id: usize| -> Result<K, PiCcsError> {
        let idx = wp_cols
            .iter()
            .position(|&c| c == col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3 missing WP opening column {col_id}")))?;
        Ok(wp_open[idx])
    };

    let decode_open_cols = rv32_decode_lookup_backed_cols(&decode);
    let control_extra_cols = if control_stage_required_for_step_instance(step) {
        rv32_trace_control_extra_opening_columns(&trace)
    } else {
        Vec::new()
    };
    let decode_open_start = core_t
        .checked_add(wp_cols.len())
        .and_then(|v| v.checked_add(control_extra_cols.len()))
        .ok_or_else(|| PiCcsError::InvalidInput("W3 decode opening start overflow".into()))?;
    let decode_open_end = decode_open_start
        .checked_add(decode_open_cols.len())
        .ok_or_else(|| PiCcsError::InvalidInput("W3 decode opening end overflow".into()))?;
    if wp_me.y_scalars.len() < decode_open_end {
        return Err(PiCcsError::ProtocolError(format!(
            "W3 decode openings missing on WP ME claim (got {}, need at least {decode_open_end})",
            wp_me.y_scalars.len()
        )));
    }
    let decode_open = &wp_me.y_scalars[decode_open_start..decode_open_end];
    let decode_open_map: BTreeMap<usize, K> = decode_open_cols
        .iter()
        .copied()
        .zip(decode_open.iter().copied())
        .collect();
    let decode_open_col = |col_id: usize| -> Result<K, PiCcsError> {
        decode_open_map
            .get(&col_id)
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3(shared) missing decode opening col_id={col_id}")))
    };
    let width_open_cols = rv32_width_lookup_backed_cols(&width);
    let width_open_start = decode_open_end;
    let width_open_end = width_open_start
        .checked_add(width_open_cols.len())
        .ok_or_else(|| PiCcsError::InvalidInput("W3 width opening end overflow".into()))?;
    if wp_me.y_scalars.len() < width_open_end {
        return Err(PiCcsError::ProtocolError(format!(
            "W3 width openings missing on WP ME claim (got {}, need at least {width_open_end})",
            wp_me.y_scalars.len()
        )));
    }
    let width_open_map: BTreeMap<usize, K> = wp_me.y_scalars[width_open_start..width_open_end]
        .iter()
        .copied()
        .zip(width_open_cols.iter().copied())
        .map(|(v, col_id)| (col_id, v))
        .collect();
    let width_open_col = |col_id: usize| -> Result<K, PiCcsError> {
        width_open_map
            .get(&col_id)
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3 missing width opening col_id={col_id}")))
    };

    let active = wp_open_col(trace.active)?;
    let rd_has_write = decode_open_col(decode.rd_has_write)?;
    let rd_val = wp_open_col(trace.rd_val)?;
    let ram_has_read = decode_open_col(decode.ram_has_read)?;
    let ram_has_write = decode_open_col(decode.ram_has_write)?;
    let ram_rv = wp_open_col(trace.ram_rv)?;
    let ram_wv = wp_open_col(trace.ram_wv)?;
    let rs2_val = wp_open_col(trace.rs2_val)?;

    let mut ram_rv_low_bits = [K::ZERO; 16];
    let mut rs2_low_bits = [K::ZERO; 16];
    for k in 0..16 {
        ram_rv_low_bits[k] = width_open_col(width.ram_rv_low_bit[k])?;
        rs2_low_bits[k] = width_open_col(width.rs2_low_bit[k])?;
    }
    let ram_rv_q16 = width_open_col(width.ram_rv_q16)?;
    let rs2_q16 = width_open_col(width.rs2_q16)?;
    let funct3_is = [
        decode_open_col(decode.funct3_is[0])?,
        decode_open_col(decode.funct3_is[1])?,
        decode_open_col(decode.funct3_is[2])?,
        decode_open_col(decode.funct3_is[3])?,
        decode_open_col(decode.funct3_is[4])?,
        decode_open_col(decode.funct3_is[5])?,
        decode_open_col(decode.funct3_is[6])?,
        decode_open_col(decode.funct3_is[7])?,
    ];
    let op_load = decode_open_col(decode.op_load)?;
    let op_store = decode_open_col(decode.op_store)?;
    let load_flags = [
        op_load * funct3_is[0],
        op_load * funct3_is[4],
        op_load * funct3_is[1],
        op_load * funct3_is[5],
        op_load * funct3_is[2],
    ];
    let store_flags = [
        op_store * funct3_is[0],
        op_store * funct3_is[1],
        op_store * funct3_is[2],
    ];

    if let Some(claim_idx) = claim_plan.width_bitness {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError("w3/bitness claim index out of range".into()));
        }
        let mut bitness_open = Vec::with_capacity(32);
        bitness_open.extend_from_slice(&ram_rv_low_bits);
        bitness_open.extend_from_slice(&rs2_low_bits);
        let weights = w3_bitness_weight_vector(r_cycle, bitness_open.len());
        let mut weighted = K::ZERO;
        for (b, w) in bitness_open.iter().zip(weights.iter()) {
            weighted += *w * *b * (*b - K::ONE);
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError("w3/bitness terminal value mismatch".into()));
        }
    }

    if let Some(claim_idx) = claim_plan.width_quiescence {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "w3/quiescence claim index out of range".into(),
            ));
        }
        let mut quiescence_open = vec![ram_rv_q16, rs2_q16];
        quiescence_open.extend_from_slice(&ram_rv_low_bits);
        quiescence_open.extend_from_slice(&rs2_low_bits);
        let weights = w3_quiescence_weight_vector(r_cycle, quiescence_open.len());
        let mut weighted = K::ZERO;
        for (v, w) in quiescence_open.iter().zip(weights.iter()) {
            weighted += *w * *v;
        }
        let expected = eq_points(r_time, r_cycle) * (K::ONE - active) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "w3/quiescence terminal value mismatch".into(),
            ));
        }
    }

    if claim_plan.width_selector_linkage.is_some() {
        return Err(PiCcsError::ProtocolError(
            "w3/selector_linkage must be disabled in reduced width-sidecar mode".into(),
        ));
    }

    if let Some(claim_idx) = claim_plan.width_load_semantics {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "w3/load_semantics claim index out of range".into(),
            ));
        }
        let residuals = w3_load_semantics_residuals(
            rd_val,
            ram_rv,
            rd_has_write,
            ram_has_read,
            load_flags,
            ram_rv_q16,
            ram_rv_low_bits,
        );
        let weights = w3_load_weight_vector(r_cycle, residuals.len());
        let mut weighted = K::ZERO;
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "w3/load_semantics terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.width_store_semantics {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "w3/store_semantics claim index out of range".into(),
            ));
        }
        let residuals = w3_store_semantics_residuals(
            ram_wv,
            ram_rv,
            rs2_val,
            rd_has_write,
            ram_has_read,
            ram_has_write,
            store_flags,
            rs2_q16,
            ram_rv_low_bits,
            rs2_low_bits,
        );
        let weights = w3_store_weight_vector(r_cycle, residuals.len());
        let mut weighted = K::ZERO;
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "w3/store_semantics terminal value mismatch".into(),
            ));
        }
    }

    Ok(())
}

pub(crate) fn verify_route_a_control_terminals(
    core_t: usize,
    step: &StepInstanceBundle<Cmt, F, K>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    claim_plan: &RouteATimeClaimPlan,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
) -> Result<(), PiCcsError> {
    let any_control_claim = claim_plan.control_next_pc_linear.is_some()
        || claim_plan.control_next_pc_control.is_some()
        || claim_plan.control_branch_semantics.is_some()
        || claim_plan.control_writeback.is_some();
    if !any_control_claim {
        return Ok(());
    }

    if mem_proof.wp_me_claims.len() != 1 {
        return Err(PiCcsError::ProtocolError(
            "control stage requires WP ME openings for main-trace terminals".into(),
        ));
    }
    let trace = Rv32TraceLayout::new();
    let decode = Rv32DecodeSidecarLayout::new();

    let wp_me = &mem_proof.wp_me_claims[0];
    if wp_me.r.as_slice() != r_time {
        return Err(PiCcsError::ProtocolError(
            "control stage WP ME claim r mismatch (expected r_time)".into(),
        ));
    }
    if wp_me.c != step.mcs_inst.c {
        return Err(PiCcsError::ProtocolError(
            "control stage WP ME claim commitment mismatch".into(),
        ));
    }
    if wp_me.m_in != step.mcs_inst.m_in {
        return Err(PiCcsError::ProtocolError(
            "control stage WP ME claim m_in mismatch".into(),
        ));
    }
    let wp_base_cols = rv32_trace_wp_opening_columns(&trace);
    let control_extra_cols = rv32_trace_control_extra_opening_columns(&trace);
    let need_wp_min = core_t
        .checked_add(wp_base_cols.len())
        .ok_or_else(|| PiCcsError::InvalidInput("control stage WP opening count overflow".into()))?;
    if wp_me.y_scalars.len() < need_wp_min {
        return Err(PiCcsError::ProtocolError(format!(
            "control stage WP ME opening length mismatch (got {}, expected at least {need_wp_min})",
            wp_me.y_scalars.len()
        )));
    }
    let need_control_min = need_wp_min
        .checked_add(control_extra_cols.len())
        .ok_or_else(|| PiCcsError::InvalidInput("control stage WP+extra opening count overflow".into()))?;
    if wp_me.y_scalars.len() < need_control_min {
        return Err(PiCcsError::ProtocolError(format!(
            "control stage requires control extra WP openings (got {}, expected at least {need_control_min})",
            wp_me.y_scalars.len()
        )));
    }
    let wp_open = &wp_me.y_scalars[core_t..];
    let wp_open_col = |col_id: usize| -> Result<K, PiCcsError> {
        if let Some(idx) = wp_base_cols.iter().position(|&c| c == col_id) {
            return Ok(wp_open[idx]);
        }
        if let Some(extra_idx) = control_extra_cols.iter().position(|&c| c == col_id) {
            let idx = wp_base_cols
                .len()
                .checked_add(extra_idx)
                .ok_or_else(|| PiCcsError::InvalidInput("control stage WP extra index overflow".into()))?;
            return wp_open.get(idx).copied().ok_or_else(|| {
                PiCcsError::ProtocolError(format!("control stage missing WP extra opening column {col_id}"))
            });
        }
        Err(PiCcsError::ProtocolError(format!(
            "control stage missing WP opening column {col_id}"
        )))
    };
    let decode_open_cols = rv32_decode_lookup_backed_cols(&decode);
    let decode_open_start = need_control_min;
    let decode_open_end = decode_open_start
        .checked_add(decode_open_cols.len())
        .ok_or_else(|| PiCcsError::InvalidInput("control stage decode opening end overflow".into()))?;
    if wp_me.y_scalars.len() < decode_open_end {
        return Err(PiCcsError::ProtocolError(format!(
            "control stage decode openings missing on WP ME claim (got {}, need at least {decode_open_end})",
            wp_me.y_scalars.len()
        )));
    }
    let decode_open = &wp_me.y_scalars[decode_open_start..decode_open_end];
    let decode_open_map: BTreeMap<usize, K> = decode_open_cols
        .iter()
        .copied()
        .zip(decode_open.iter().copied())
        .collect();
    let decode_open_col = |col_id: usize| -> Result<K, PiCcsError> {
        decode_open_map
            .get(&col_id)
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("control(shared) missing decode opening col_id={col_id}")))
    };

    let active = wp_open_col(trace.active)?;
    let pc_before = wp_open_col(trace.pc_before)?;
    let pc_after = wp_open_col(trace.pc_after)?;
    let rs1_val = wp_open_col(trace.rs1_val)?;
    let rd_val = wp_open_col(trace.rd_val)?;
    let jalr_drop_bit = wp_open_col(trace.jalr_drop_bit)?;
    let pc_carry = wp_open_col(trace.pc_carry)?;
    let shout_val = wp_open_col(trace.shout_val)?;
    let funct3_bits = [
        decode_open_col(decode.funct3_bit[0])?,
        decode_open_col(decode.funct3_bit[1])?,
        decode_open_col(decode.funct3_bit[2])?,
    ];
    let rs1_bits = [
        decode_open_col(decode.rs1_bit[0])?,
        decode_open_col(decode.rs1_bit[1])?,
        decode_open_col(decode.rs1_bit[2])?,
        decode_open_col(decode.rs1_bit[3])?,
        decode_open_col(decode.rs1_bit[4])?,
    ];
    let rs2_bits = [
        decode_open_col(decode.rs2_bit[0])?,
        decode_open_col(decode.rs2_bit[1])?,
        decode_open_col(decode.rs2_bit[2])?,
        decode_open_col(decode.rs2_bit[3])?,
        decode_open_col(decode.rs2_bit[4])?,
    ];
    let funct7_bits = [
        decode_open_col(decode.funct7_bit[0])?,
        decode_open_col(decode.funct7_bit[1])?,
        decode_open_col(decode.funct7_bit[2])?,
        decode_open_col(decode.funct7_bit[3])?,
        decode_open_col(decode.funct7_bit[4])?,
        decode_open_col(decode.funct7_bit[5])?,
        decode_open_col(decode.funct7_bit[6])?,
    ];

    let op_lui = decode_open_col(decode.op_lui)?;
    let op_auipc = decode_open_col(decode.op_auipc)?;
    let op_jal = decode_open_col(decode.op_jal)?;
    let op_jalr = decode_open_col(decode.op_jalr)?;
    let op_branch = decode_open_col(decode.op_branch)?;
    let op_load = decode_open_col(decode.op_load)?;
    let op_store = decode_open_col(decode.op_store)?;
    let op_alu_imm = decode_open_col(decode.op_alu_imm)?;
    let op_alu_reg = decode_open_col(decode.op_alu_reg)?;
    let op_misc_mem = decode_open_col(decode.op_misc_mem)?;
    let op_system = decode_open_col(decode.op_system)?;
    let op_amo = decode_open_col(decode.op_amo)?;
    let rd_is_zero = decode_open_col(decode.rd_is_zero)?;
    let op_lui_write = op_lui * (K::ONE - rd_is_zero);
    let op_auipc_write = op_auipc * (K::ONE - rd_is_zero);
    let op_jal_write = op_jal * (K::ONE - rd_is_zero);
    let op_jalr_write = op_jalr * (K::ONE - rd_is_zero);
    let imm_i = decode_open_col(decode.imm_i)?;
    let imm_b = decode_open_col(decode.imm_b)?;
    let imm_j = decode_open_col(decode.imm_j)?;
    let funct3_is6 = decode_open_col(decode.funct3_is[6])?;
    let funct3_is7 = decode_open_col(decode.funct3_is[7])?;

    if let Some(claim_idx) = claim_plan.control_next_pc_linear {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "control/next_pc_linear claim index out of range".into(),
            ));
        }
        let residual = control_next_pc_linear_residual(
            pc_before,
            pc_after,
            op_lui,
            op_auipc,
            op_load,
            op_store,
            op_alu_imm,
            op_alu_reg,
            op_misc_mem,
            op_system,
            op_amo,
        );
        let weights = control_next_pc_linear_weight_vector(r_cycle, 1);
        let expected = eq_points(r_time, r_cycle) * weights[0] * residual;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "control/next_pc_linear terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.control_next_pc_control {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "control/next_pc_control claim index out of range".into(),
            ));
        }
        let residuals = control_next_pc_control_residuals(
            active,
            pc_before,
            pc_after,
            rs1_val,
            jalr_drop_bit,
            pc_carry,
            imm_i,
            imm_b,
            imm_j,
            op_jal,
            op_jalr,
            op_branch,
            shout_val,
            funct3_bits[0],
        );
        let weights = control_next_pc_control_weight_vector(r_cycle, residuals.len());
        let mut weighted = K::ZERO;
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "control/next_pc_control terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.control_branch_semantics {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "control/branch_semantics claim index out of range".into(),
            ));
        }
        let residuals = control_branch_semantics_residuals(
            op_branch,
            shout_val,
            funct3_bits[0],
            funct3_bits[1],
            funct3_bits[2],
            funct3_is6,
            funct3_is7,
        );
        let weights = control_branch_semantics_weight_vector(r_cycle, residuals.len());
        let mut weighted = K::ZERO;
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "control/branch_semantics terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.control_writeback {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "control/writeback claim index out of range".into(),
            ));
        }
        let imm_u = control_imm_u_from_bits(funct3_bits, rs1_bits, rs2_bits, funct7_bits);
        let residuals = control_writeback_residuals(
            rd_val,
            pc_before,
            imm_u,
            op_lui_write,
            op_auipc_write,
            op_jal_write,
            op_jalr_write,
        );
        let weights = control_writeback_weight_vector(r_cycle, residuals.len());
        let mut weighted = K::ZERO;
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "control/writeback terminal value mismatch".into(),
            ));
        }
    }

    Ok(())
}
