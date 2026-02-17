use super::*;

pub(crate) fn width_lookup_bus_val_cols_witness(
    step: &StepWitnessBundle<Cmt, F, K>,
    t_len: usize,
) -> Result<Vec<usize>, PiCcsError> {
    let width = Rv32WidthSidecarLayout::new();
    let width_cols = rv32_width_lookup_backed_cols(&width);
    let mut width_bus_col_by_col: BTreeMap<usize, usize> = BTreeMap::new();
    let m_in = step.mcs.0.m_in;
    let bus = build_bus_layout_for_step_witness(step, t_len)?;
    if bus.shout_cols.len() != step.lut_instances.len() {
        return Err(PiCcsError::ProtocolError(
            "W3(shared): bus shout lane count drift while resolving width lookup columns".into(),
        ));
    }
    let bus_base_delta = bus
        .bus_base
        .checked_sub(m_in)
        .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): bus_base underflow".into()))?;
    if bus_base_delta % t_len != 0 {
        return Err(PiCcsError::ProtocolError(format!(
            "W3(shared): bus_base alignment mismatch (bus_base_delta={bus_base_delta}, t_len={t_len})"
        )));
    }
    let bus_col_offset = bus_base_delta / t_len;
    for (lut_idx, (inst, _)) in step.lut_instances.iter().enumerate() {
        if !rv32_is_width_lookup_table_id(inst.table_id) {
            continue;
        }
        let width_col_id = width_cols
            .iter()
            .copied()
            .find(|&col_id| rv32_width_lookup_table_id_for_col(col_id) == inst.table_id)
            .ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "W3(shared): width lookup table_id={} does not map to a known width column",
                    inst.table_id
                ))
            })?;
        let inst_cols = bus
            .shout_cols
            .get(lut_idx)
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): missing shout cols for width lookup table".into()))?;
        let lane0 = inst_cols.lanes.get(0).ok_or_else(|| {
            PiCcsError::ProtocolError("W3(shared): expected one shout lane for width lookup table".into())
        })?;
        width_bus_col_by_col.insert(width_col_id, bus_col_offset + lane0.primary_val());
    }
    let mut out = Vec::with_capacity(width_cols.len());
    for &col_id in width_cols.iter() {
        let bus_col = width_bus_col_by_col.get(&col_id).copied().ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "W3(shared): missing width lookup bus val column for width col_id={col_id}"
            ))
        })?;
        out.push(bus_col);
    }
    Ok(out)
}

pub(crate) fn build_route_a_width_time_claims(
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    r_cycle: &[K],
) -> Result<W3TimeClaims, PiCcsError> {
    if !width_stage_required_for_step_witness(step) {
        return Ok((None, None, None, None, None));
    }
    let trace = Rv32TraceLayout::new();
    let width = Rv32WidthSidecarLayout::new();
    let decode = Rv32DecodeSidecarLayout::new();
    let m_in = step.mcs.0.m_in;
    let ell_n = r_cycle.len();
    let t_len = infer_rv32_trace_t_len_for_wb_wp(step, &trace)?;
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput("W3: t_len must be >= 1".into()));
    }

    let main_col_ids = [
        trace.active,
        trace.instr_word,
        trace.rd_val,
        trace.ram_rv,
        trace.ram_wv,
        trace.rs2_val,
    ];
    let main_decoded = decode_trace_col_values_batch(params, step, t_len, &main_col_ids)?;
    let width_col_ids = rv32_width_lookup_backed_cols(&width);
    let width_decoded: BTreeMap<usize, Vec<K>> = {
        let width_bus_abs_cols = width_lookup_bus_val_cols_witness(step, t_len)?;
        let bus = build_bus_layout_for_step_witness(step, t_len)?;
        let bus_base_delta = bus
            .bus_base
            .checked_sub(m_in)
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): bus_base underflow".into()))?;
        if bus_base_delta % t_len != 0 {
            return Err(PiCcsError::ProtocolError(format!(
                "W3(shared): bus_base alignment mismatch (bus_base_delta={bus_base_delta}, t_len={t_len})"
            )));
        }
        let bus_col_offset = bus_base_delta / t_len;
        let mut width_bus_val_cols = Vec::with_capacity(width_bus_abs_cols.len());
        for abs_col in width_bus_abs_cols.iter().copied() {
            let local_col = abs_col.checked_sub(bus_col_offset).ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "W3(shared): width lookup bus column underflow (abs_col={abs_col}, bus_col_offset={bus_col_offset})"
                ))
            })?;
            if local_col >= bus.bus_cols {
                return Err(PiCcsError::ProtocolError(format!(
                    "W3(shared): width lookup bus column out of range (local_col={local_col}, bus_cols={})",
                    bus.bus_cols
                )));
            }
            width_bus_val_cols.push(local_col);
        }
        let lookup_vals = decode_lookup_backed_col_values_batch(
            params,
            bus.bus_base,
            t_len,
            &step.mcs.1.Z,
            bus.bus_cols,
            &width_bus_val_cols,
        )?;
        let mut by_col = BTreeMap::<usize, Vec<K>>::new();
        for (idx, &col_id) in width_col_ids.iter().enumerate() {
            let bus_col_id = width_bus_val_cols[idx];
            let vals = lookup_vals.get(&bus_col_id).ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "W3(shared): missing decoded lookup values for bus_col={bus_col_id}"
                ))
            })?;
            by_col.insert(col_id, vals.clone());
        }
        by_col
    };
    let decode_col_ids: Vec<usize> = core::iter::once(decode.op_load)
        .chain(core::iter::once(decode.op_store))
        .chain(core::iter::once(decode.rd_has_write))
        .chain(core::iter::once(decode.ram_has_read))
        .chain(core::iter::once(decode.ram_has_write))
        .chain(decode.funct3_is.iter().copied())
        .collect();
    let decode_decoded = {
        let instr_vals = main_decoded
            .get(&trace.instr_word)
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): missing instr_word decode column".into()))?;
        let active_vals = main_decoded
            .get(&trace.active)
            .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): missing active decode column".into()))?;
        if instr_vals.len() != t_len || active_vals.len() != t_len {
            return Err(PiCcsError::ProtocolError(format!(
                "W3(shared): decoded CPU column lengths drift (instr={}, active={}, t_len={t_len})",
                instr_vals.len(),
                active_vals.len()
            )));
        }
        let mut decoded = BTreeMap::<usize, Vec<K>>::new();
        for &col_id in decode_col_ids.iter() {
            decoded.insert(col_id, Vec::with_capacity(t_len));
        }
        for j in 0..t_len {
            let instr_word = decode_k_to_u32(instr_vals[j], "W3(shared)/instr_word")?;
            let active = active_vals[j] != K::ZERO;
            let mut row = rv32_decode_lookup_backed_row_from_instr_word(&decode, instr_word, active);
            if !active {
                row.fill(F::ZERO);
            }
            for &col_id in decode_col_ids.iter() {
                decoded
                    .get_mut(&col_id)
                    .ok_or_else(|| PiCcsError::ProtocolError("W3(shared): decode map build failed".into()))?
                    .push(K::from(row[col_id]));
            }
        }
        decoded
    };

    let mut main_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for &col_id in main_col_ids.iter() {
        let vals = main_decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3 missing main decoded column {col_id}")))?;
        main_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }
    let mut width_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for &col_id in width_col_ids.iter() {
        let vals = width_decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3 missing width decoded column {col_id}")))?;
        width_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }
    let mut decode_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for &col_id in decode_col_ids.iter() {
        let vals = decode_decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3 missing decode decoded column {col_id}")))?;
        decode_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }

    let main_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        main_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3 missing main sparse column {col_id}")))
    };
    let width_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        width_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3 missing width sparse column {col_id}")))
    };
    let decode_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        decode_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("W3 missing decode sparse column {col_id}")))
    };

    let bitness_cols: Vec<usize> = width
        .ram_rv_low_bit
        .iter()
        .chain(width.rs2_low_bit.iter())
        .copied()
        .collect();
    let mut bitness_sparse = Vec::with_capacity(bitness_cols.len());
    for &col_id in bitness_cols.iter() {
        bitness_sparse.push(width_col(col_id)?);
    }
    let bitness_weights = w3_bitness_weight_vector(r_cycle, bitness_cols.len());
    let bitness_oracle = FormulaOracleSparseTime::new(
        bitness_sparse,
        3,
        r_cycle,
        Box::new(move |vals: &[K]| {
            let mut weighted = K::ZERO;
            for (b, w) in vals.iter().zip(bitness_weights.iter()) {
                weighted += *w * *b * (*b - K::ONE);
            }
            weighted
        }),
    );

    let mut quiescence_sparse = Vec::with_capacity(1 + width.cols);
    quiescence_sparse.push(main_col(trace.active)?);
    for &col_id in width_col_ids.iter() {
        quiescence_sparse.push(width_col(col_id)?);
    }
    let quiescence_weights = w3_quiescence_weight_vector(r_cycle, width.cols);
    let quiescence_oracle = FormulaOracleSparseTime::new(
        quiescence_sparse,
        3,
        r_cycle,
        Box::new(move |vals: &[K]| {
            let active = vals[0];
            let mut weighted = K::ZERO;
            for (i, w) in quiescence_weights.iter().enumerate() {
                weighted += *w * vals[1 + i];
            }
            (K::ONE - active) * weighted
        }),
    );

    let mut load_sparse = Vec::with_capacity(31);
    load_sparse.push(main_col(trace.rd_val)?);
    load_sparse.push(main_col(trace.ram_rv)?);
    load_sparse.push(decode_col(decode.rd_has_write)?);
    load_sparse.push(decode_col(decode.ram_has_read)?);
    load_sparse.push(decode_col(decode.op_load)?);
    load_sparse.push(decode_col(decode.funct3_is[0])?);
    load_sparse.push(decode_col(decode.funct3_is[1])?);
    load_sparse.push(decode_col(decode.funct3_is[2])?);
    load_sparse.push(decode_col(decode.funct3_is[4])?);
    load_sparse.push(decode_col(decode.funct3_is[5])?);
    load_sparse.push(width_col(width.ram_rv_q16)?);
    for &col_id in width.ram_rv_low_bit.iter() {
        load_sparse.push(width_col(col_id)?);
    }
    let load_weights = w3_load_weight_vector(r_cycle, 16);
    let load_oracle = FormulaOracleSparseTime::new(
        load_sparse,
        4,
        r_cycle,
        Box::new(move |vals: &[K]| {
            let rd_val = vals[0];
            let ram_rv = vals[1];
            let rd_has_write = vals[2];
            let ram_has_read = vals[3];
            let op_load = vals[4];
            let funct3_is_0 = vals[5];
            let funct3_is_1 = vals[6];
            let funct3_is_2 = vals[7];
            let funct3_is_4 = vals[8];
            let funct3_is_5 = vals[9];
            let ram_rv_q16 = vals[10];
            let load_flags = [
                op_load * funct3_is_0,
                op_load * funct3_is_4,
                op_load * funct3_is_1,
                op_load * funct3_is_5,
                op_load * funct3_is_2,
            ];
            let mut ram_rv_low_bits = [K::ZERO; 16];
            ram_rv_low_bits.copy_from_slice(&vals[11..27]);
            let residuals = w3_load_semantics_residuals(
                rd_val,
                ram_rv,
                rd_has_write,
                ram_has_read,
                load_flags,
                ram_rv_q16,
                ram_rv_low_bits,
            );
            let mut weighted = K::ZERO;
            for (r, w) in residuals.iter().zip(load_weights.iter()) {
                weighted += *w * *r;
            }
            weighted
        }),
    );

    let mut store_sparse = Vec::with_capacity(45);
    store_sparse.push(main_col(trace.ram_wv)?);
    store_sparse.push(main_col(trace.ram_rv)?);
    store_sparse.push(main_col(trace.rs2_val)?);
    store_sparse.push(decode_col(decode.rd_has_write)?);
    store_sparse.push(decode_col(decode.ram_has_read)?);
    store_sparse.push(decode_col(decode.ram_has_write)?);
    store_sparse.push(decode_col(decode.op_store)?);
    store_sparse.push(decode_col(decode.funct3_is[0])?);
    store_sparse.push(decode_col(decode.funct3_is[1])?);
    store_sparse.push(decode_col(decode.funct3_is[2])?);
    store_sparse.push(width_col(width.rs2_q16)?);
    for &col_id in width.ram_rv_low_bit.iter() {
        store_sparse.push(width_col(col_id)?);
    }
    for &col_id in width.rs2_low_bit.iter() {
        store_sparse.push(width_col(col_id)?);
    }
    let store_weights = w3_store_weight_vector(r_cycle, 12);
    let store_oracle = FormulaOracleSparseTime::new(
        store_sparse,
        4,
        r_cycle,
        Box::new(move |vals: &[K]| {
            let ram_wv = vals[0];
            let ram_rv = vals[1];
            let rs2_val = vals[2];
            let rd_has_write = vals[3];
            let ram_has_read = vals[4];
            let ram_has_write = vals[5];
            let op_store = vals[6];
            let funct3_is_0 = vals[7];
            let funct3_is_1 = vals[8];
            let funct3_is_2 = vals[9];
            let rs2_q16 = vals[10];
            let store_flags = [op_store * funct3_is_0, op_store * funct3_is_1, op_store * funct3_is_2];
            let mut ram_rv_low_bits = [K::ZERO; 16];
            ram_rv_low_bits.copy_from_slice(&vals[11..27]);
            let mut rs2_low_bits = [K::ZERO; 16];
            rs2_low_bits.copy_from_slice(&vals[27..43]);
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
            let mut weighted = K::ZERO;
            for (r, w) in residuals.iter().zip(store_weights.iter()) {
                weighted += *w * *r;
            }
            weighted
        }),
    );

    Ok((
        Some((Box::new(bitness_oracle), K::ZERO)),
        Some((Box::new(quiescence_oracle), K::ZERO)),
        None,
        Some((Box::new(load_oracle), K::ZERO)),
        Some((Box::new(store_oracle), K::ZERO)),
    ))
}

type ControlTimeClaims = (
    Option<(Box<dyn RoundOracle>, K)>,
    Option<(Box<dyn RoundOracle>, K)>,
    Option<(Box<dyn RoundOracle>, K)>,
    Option<(Box<dyn RoundOracle>, K)>,
);

pub(crate) fn build_route_a_control_time_claims(
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    r_cycle: &[K],
) -> Result<ControlTimeClaims, PiCcsError> {
    if !control_stage_required_for_step_witness(step) {
        return Ok((None, None, None, None));
    }
    let trace = Rv32TraceLayout::new();
    let decode = Rv32DecodeSidecarLayout::new();
    let m_in = step.mcs.0.m_in;
    let ell_n = r_cycle.len();
    let t_len = infer_rv32_trace_t_len_for_wb_wp(step, &trace)?;
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput("control stage: t_len must be >= 1".into()));
    }

    let main_col_ids = vec![
        trace.active,
        trace.instr_word,
        trace.pc_before,
        trace.pc_after,
        trace.rs1_val,
        trace.rd_val,
        trace.shout_val,
        trace.jalr_drop_bit,
    ];
    let decode_col_ids = vec![
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
        decode.op_lui_write,
        decode.op_auipc_write,
        decode.op_jal_write,
        decode.op_jalr_write,
        decode.rd_is_zero,
        decode.imm_i,
        decode.imm_b,
        decode.imm_j,
        decode.funct3_is[6],
        decode.funct3_is[7],
        decode.funct3_bit[0],
        decode.funct3_bit[1],
        decode.funct3_bit[2],
        decode.rs1_bit[0],
        decode.rs1_bit[1],
        decode.rs1_bit[2],
        decode.rs1_bit[3],
        decode.rs1_bit[4],
        decode.rs2_bit[0],
        decode.rs2_bit[1],
        decode.rs2_bit[2],
        decode.rs2_bit[3],
        decode.rs2_bit[4],
        decode.funct7_bit[0],
        decode.funct7_bit[1],
        decode.funct7_bit[2],
        decode.funct7_bit[3],
        decode.funct7_bit[4],
        decode.funct7_bit[5],
        decode.funct7_bit[6],
    ];

    let main_decoded = decode_trace_col_values_batch(params, step, t_len, &main_col_ids)?;
    let decode_decoded = {
        let instr_vals = main_decoded
            .get(&trace.instr_word)
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing instr_word decode column".into()))?;
        let active_vals = main_decoded
            .get(&trace.active)
            .ok_or_else(|| PiCcsError::ProtocolError("control(shared): missing active decode column".into()))?;
        if instr_vals.len() != t_len || active_vals.len() != t_len {
            return Err(PiCcsError::ProtocolError(format!(
                "control(shared): decoded CPU column lengths drift (instr={}, active={}, t_len={t_len})",
                instr_vals.len(),
                active_vals.len()
            )));
        }
        let mut decoded = BTreeMap::<usize, Vec<K>>::new();
        for &col_id in decode_col_ids.iter() {
            decoded.insert(col_id, Vec::with_capacity(t_len));
        }
        for j in 0..t_len {
            let instr_word = decode_k_to_u32(instr_vals[j], "control(shared)/instr_word")?;
            let active = active_vals[j] != K::ZERO;
            let mut row = rv32_decode_lookup_backed_row_from_instr_word(&decode, instr_word, active);
            if !active {
                row.fill(F::ZERO);
            }
            let rd_has_write = if active {
                K::ONE - K::from(row[decode.rd_is_zero])
            } else {
                K::ZERO
            };
            let op_lui = K::from(row[decode.op_lui]);
            let op_auipc = K::from(row[decode.op_auipc]);
            let op_jal = K::from(row[decode.op_jal]);
            let op_jalr = K::from(row[decode.op_jalr]);
            for &col_id in decode_col_ids.iter() {
                let val = match col_id {
                    c if c == decode.op_lui_write => op_lui * rd_has_write,
                    c if c == decode.op_auipc_write => op_auipc * rd_has_write,
                    c if c == decode.op_jal_write => op_jal * rd_has_write,
                    c if c == decode.op_jalr_write => op_jalr * rd_has_write,
                    _ => K::from(row[col_id]),
                };
                decoded
                    .get_mut(&col_id)
                    .ok_or_else(|| PiCcsError::ProtocolError("control(shared): decode map build failed".into()))?
                    .push(val);
            }
        }
        decoded
    };

    let mut main_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for &col_id in main_col_ids.iter() {
        let vals = main_decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("control stage missing main decoded column {col_id}")))?;
        main_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }
    let mut decode_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for &col_id in decode_col_ids.iter() {
        let vals = decode_decoded.get(&col_id).ok_or_else(|| {
            PiCcsError::ProtocolError(format!("control stage missing decode decoded column {col_id}"))
        })?;
        decode_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }

    let main_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        main_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("control stage missing main sparse col {col_id}")))
    };
    let decode_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        decode_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("control stage missing decode sparse col {col_id}")))
    };

    let linear_sparse = vec![
        main_col(trace.pc_before)?,
        main_col(trace.pc_after)?,
        decode_col(decode.op_lui)?,
        decode_col(decode.op_auipc)?,
        decode_col(decode.op_load)?,
        decode_col(decode.op_store)?,
        decode_col(decode.op_alu_imm)?,
        decode_col(decode.op_alu_reg)?,
        decode_col(decode.op_misc_mem)?,
        decode_col(decode.op_system)?,
        decode_col(decode.op_amo)?,
    ];
    let linear_weights = control_next_pc_linear_weight_vector(r_cycle, 1);
    let linear_oracle = FormulaOracleSparseTime::new(
        linear_sparse,
        3,
        r_cycle,
        Box::new(move |vals: &[K]| {
            let residual = control_next_pc_linear_residual(
                vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], vals[8], vals[9], vals[10],
            );
            linear_weights[0] * residual
        }),
    );

    let control_sparse = vec![
        main_col(trace.active)?,
        main_col(trace.pc_before)?,
        main_col(trace.pc_after)?,
        main_col(trace.rs1_val)?,
        main_col(trace.jalr_drop_bit)?,
        main_col(trace.shout_val)?,
        decode_col(decode.funct3_bit[0])?,
        decode_col(decode.op_jal)?,
        decode_col(decode.op_jalr)?,
        decode_col(decode.op_branch)?,
        decode_col(decode.imm_i)?,
        decode_col(decode.imm_b)?,
        decode_col(decode.imm_j)?,
    ];
    let control_weights = control_next_pc_control_weight_vector(r_cycle, 5);
    let control_oracle = FormulaOracleSparseTime::new(
        control_sparse,
        5,
        r_cycle,
        Box::new(move |vals: &[K]| {
            let residuals = control_next_pc_control_residuals(
                vals[0],
                vals[1],
                vals[2],
                vals[3],
                vals[4],
                vals[10],
                vals[11],
                vals[12],
                vals[7],
                vals[8],
                vals[9],
                vals[5],
                vals[6],
            );
            let mut weighted = K::ZERO;
            for (r, w) in residuals.iter().zip(control_weights.iter()) {
                weighted += *w * *r;
            }
            weighted
        }),
    );

    let branch_sparse = vec![
        decode_col(decode.op_branch)?,
        main_col(trace.shout_val)?,
        decode_col(decode.funct3_bit[0])?,
        decode_col(decode.funct3_bit[1])?,
        decode_col(decode.funct3_bit[2])?,
        decode_col(decode.funct3_is[6])?,
        decode_col(decode.funct3_is[7])?,
    ];
    let branch_weights = control_branch_semantics_weight_vector(r_cycle, 3);
    let branch_oracle = FormulaOracleSparseTime::new(
        branch_sparse,
        4,
        r_cycle,
        Box::new(move |vals: &[K]| {
            let residuals =
                control_branch_semantics_residuals(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6]);
            let mut weighted = K::ZERO;
            for (r, w) in residuals.iter().zip(branch_weights.iter()) {
                weighted += *w * *r;
            }
            weighted
        }),
    );

    let mut write_sparse = vec![
        main_col(trace.rd_val)?,
        main_col(trace.pc_before)?,
        decode_col(decode.op_lui)?,
        decode_col(decode.op_auipc)?,
        decode_col(decode.op_jal)?,
        decode_col(decode.op_jalr)?,
        decode_col(decode.rd_is_zero)?,
        decode_col(decode.funct3_bit[0])?,
        decode_col(decode.funct3_bit[1])?,
        decode_col(decode.funct3_bit[2])?,
    ];
    for &col_id in decode.rs1_bit.iter() {
        write_sparse.push(decode_col(col_id)?);
    }
    for &col_id in decode.rs2_bit.iter() {
        write_sparse.push(decode_col(col_id)?);
    }
    for &col_id in decode.funct7_bit.iter() {
        write_sparse.push(decode_col(col_id)?);
    }
    let write_weights = control_writeback_weight_vector(r_cycle, 4);
    let write_oracle = FormulaOracleSparseTime::new(
        write_sparse,
        4,
        r_cycle,
        Box::new(move |vals: &[K]| {
            let rd_val = vals[0];
            let pc_before = vals[1];
            let op_lui = vals[2];
            let op_auipc = vals[3];
            let op_jal = vals[4];
            let op_jalr = vals[5];
            let rd_is_zero = vals[6];
            let op_lui_write = op_lui * (K::ONE - rd_is_zero);
            let op_auipc_write = op_auipc * (K::ONE - rd_is_zero);
            let op_jal_write = op_jal * (K::ONE - rd_is_zero);
            let op_jalr_write = op_jalr * (K::ONE - rd_is_zero);
            let funct3_bits = [vals[7], vals[8], vals[9]];
            let rs1_bits = [vals[10], vals[11], vals[12], vals[13], vals[14]];
            let rs2_bits = [vals[15], vals[16], vals[17], vals[18], vals[19]];
            let funct7_bits = [vals[20], vals[21], vals[22], vals[23], vals[24], vals[25], vals[26]];
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
            let mut weighted = K::ZERO;
            for (r, w) in residuals.iter().zip(write_weights.iter()) {
                weighted += *w * *r;
            }
            weighted
        }),
    );

    Ok((
        Some((Box::new(linear_oracle), K::ZERO)),
        Some((Box::new(control_oracle), K::ZERO)),
        Some((Box::new(branch_oracle), K::ZERO)),
        Some((Box::new(write_oracle), K::ZERO)),
    ))
}

pub(crate) fn emit_route_a_wb_wp_me_claims(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    step: &StepWitnessBundle<Cmt, F, K>,
    r_time: &[K],
) -> Result<(Vec<MeInstance<Cmt, F, K>>, Vec<MeInstance<Cmt, F, K>>), PiCcsError> {
    if !wb_wp_required_for_step_witness(step) {
        return Ok((Vec::new(), Vec::new()));
    }

    let trace = Rv32TraceLayout::new();
    let t_len = infer_rv32_trace_t_len_for_wb_wp(step, &trace)?;
    let m_in = step.mcs.0.m_in;
    let core_t = s.t();
    let (mcs_inst, mcs_wit) = &step.mcs;

    let wb_cols = rv32_trace_wb_columns(&trace);
    let mut wb_claims = ts::emit_me_claims_for_mats(
        tr,
        b"cpu/me_digest_wb_time",
        params,
        s,
        core::slice::from_ref(&mcs_inst.c),
        core::slice::from_ref(&mcs_wit.Z),
        r_time,
        m_in,
    )?;
    if wb_claims.len() != 1 {
        return Err(PiCcsError::ProtocolError(format!(
            "WB expects exactly one CPU ME claim at r_time, got {}",
            wb_claims.len()
        )));
    }
    crate::memory_sidecar::cpu_bus::append_col_major_time_openings_to_me_instance(
        params,
        m_in,
        t_len,
        m_in,
        &wb_cols,
        core_t,
        &mcs_wit.Z,
        &mut wb_claims[0],
    )?;

    let mut wp_cols = rv32_trace_wp_opening_columns(&trace);
    if control_stage_required_for_step_witness(step) {
        wp_cols.extend(rv32_trace_control_extra_opening_columns(&trace));
    }
    if decode_stage_required_for_step_witness(step) {
        let decode_layout = Rv32DecodeSidecarLayout::new();
        let (_decode_open_cols, decode_lut_indices) = resolve_shared_decode_lookup_lut_indices(step, &decode_layout)?;
        let bus = build_bus_layout_for_step_witness(step, t_len)?;
        if bus.shout_cols.len() != step.lut_instances.len() {
            return Err(PiCcsError::ProtocolError(
                "W2(shared): bus layout shout lane count drift".into(),
            ));
        }
        let bus_base_delta = bus
            .bus_base
            .checked_sub(m_in)
            .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): bus_base underflow".into()))?;
        if bus_base_delta % t_len != 0 {
            return Err(PiCcsError::ProtocolError(format!(
                "W2(shared): bus_base alignment mismatch (bus_base_delta={}, t_len={t_len})",
                bus_base_delta
            )));
        }
        let bus_col_offset = bus_base_delta / t_len;
        for &lut_idx in decode_lut_indices.iter() {
            let inst_cols = bus.shout_cols.get(lut_idx).ok_or_else(|| {
                PiCcsError::ProtocolError("W2(shared): missing shout cols for decode lookup table".into())
            })?;
            let lane0 = inst_cols.lanes.get(0).ok_or_else(|| {
                PiCcsError::ProtocolError("W2(shared): expected one shout lane for decode lookup table".into())
            })?;
            wp_cols.push(bus_col_offset + lane0.primary_val());
        }
    }
    if width_stage_required_for_step_witness(step) {
        wp_cols.extend(width_lookup_bus_val_cols_witness(step, t_len)?);
    }
    let mut wp_claims = ts::emit_me_claims_for_mats(
        tr,
        b"cpu/me_digest_wp_time",
        params,
        s,
        core::slice::from_ref(&mcs_inst.c),
        core::slice::from_ref(&mcs_wit.Z),
        r_time,
        m_in,
    )?;
    if wp_claims.len() != 1 {
        return Err(PiCcsError::ProtocolError(format!(
            "WP expects exactly one CPU ME claim at r_time, got {}",
            wp_claims.len()
        )));
    }
    crate::memory_sidecar::cpu_bus::append_col_major_time_openings_to_me_instance(
        params,
        m_in,
        t_len,
        m_in,
        &wp_cols,
        core_t,
        &mcs_wit.Z,
        &mut wp_claims[0],
    )?;
    Ok((wb_claims, wp_claims))
}

pub(crate) fn verify_route_a_wb_wp_terminals(
    core_t: usize,
    step: &StepInstanceBundle<Cmt, F, K>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    claim_plan: &RouteATimeClaimPlan,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
) -> Result<(), PiCcsError> {
    let trace = Rv32TraceLayout::new();

    if let Some(claim_idx) = claim_plan.wb_bool {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "wb/booleanity claim index out of range".into(),
            ));
        }
        if mem_proof.wb_me_claims.len() != 1 {
            return Err(PiCcsError::ProtocolError(format!(
                "WB expects exactly one ME claim at r_time (got {})",
                mem_proof.wb_me_claims.len()
            )));
        }
        let me = &mem_proof.wb_me_claims[0];
        if me.r.as_slice() != r_time {
            return Err(PiCcsError::ProtocolError(
                "WB ME claim r mismatch (expected r_time)".into(),
            ));
        }
        if me.c != step.mcs_inst.c {
            return Err(PiCcsError::ProtocolError("WB ME claim commitment mismatch".into()));
        }
        if me.m_in != step.mcs_inst.m_in {
            return Err(PiCcsError::ProtocolError("WB ME claim m_in mismatch".into()));
        }

        let wb_bool_cols = rv32_trace_wb_columns(&trace);
        let need = core_t
            .checked_add(wb_bool_cols.len())
            .ok_or_else(|| PiCcsError::InvalidInput("WB opening count overflow".into()))?;
        if me.y_scalars.len() != need {
            return Err(PiCcsError::ProtocolError(format!(
                "WB ME opening length mismatch (got {}, expected {need})",
                me.y_scalars.len()
            )));
        }

        let wb_bool_open = &me.y_scalars[core_t..];
        let wb_weights = wb_weight_vector(r_cycle, wb_bool_cols.len());
        let mut wb_weighted_bitness = K::ZERO;
        for (&b, &w) in wb_bool_open.iter().zip(wb_weights.iter()) {
            wb_weighted_bitness += w * b * (b - K::ONE);
        }

        let expected_terminal = eq_points(r_time, r_cycle) * wb_weighted_bitness;
        let observed_terminal = batched_final_values[claim_idx];
        if observed_terminal != expected_terminal {
            return Err(PiCcsError::ProtocolError(
                "wb/booleanity terminal value mismatch".into(),
            ));
        }
    } else if !mem_proof.wb_me_claims.is_empty() {
        return Err(PiCcsError::ProtocolError(
            "unexpected WB ME claims: wb/booleanity stage is not enabled".into(),
        ));
    }

    if let Some(claim_idx) = claim_plan.wp_quiescence {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "wp/quiescence claim index out of range".into(),
            ));
        }
        if mem_proof.wp_me_claims.len() != 1 {
            return Err(PiCcsError::ProtocolError(format!(
                "WP expects exactly one ME claim at r_time (got {})",
                mem_proof.wp_me_claims.len()
            )));
        }
        let me = &mem_proof.wp_me_claims[0];
        if me.r.as_slice() != r_time {
            return Err(PiCcsError::ProtocolError(
                "WP ME claim r mismatch (expected r_time)".into(),
            ));
        }
        if me.c != step.mcs_inst.c {
            return Err(PiCcsError::ProtocolError("WP ME claim commitment mismatch".into()));
        }
        if me.m_in != step.mcs_inst.m_in {
            return Err(PiCcsError::ProtocolError("WP ME claim m_in mismatch".into()));
        }

        let wp_open_cols = rv32_trace_wp_opening_columns(&trace);
        let need_min = core_t
            .checked_add(wp_open_cols.len())
            .ok_or_else(|| PiCcsError::InvalidInput("WP opening count overflow".into()))?;
        if me.y_scalars.len() < need_min {
            return Err(PiCcsError::ProtocolError(format!(
                "WP ME opening length mismatch (got {}, expected at least {need_min})",
                me.y_scalars.len()
            )));
        }

        let active_open = me
            .y_scalars
            .get(core_t)
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError("WP missing active opening".into()))?;
        let wp_open_end = core_t
            .checked_add(wp_open_cols.len())
            .ok_or_else(|| PiCcsError::InvalidInput("WP opening end overflow".into()))?;
        let wp_open = &me.y_scalars[(core_t + 1)..wp_open_end];
        let wp_weights = wp_weight_vector(r_cycle, wp_open.len());
        let mut wp_weighted_sum = K::ZERO;
        for (&v, &w) in wp_open.iter().zip(wp_weights.iter()) {
            wp_weighted_sum += w * v;
        }
        let expected_terminal = eq_points(r_time, r_cycle) * (K::ONE - active_open) * wp_weighted_sum;
        let observed_terminal = batched_final_values[claim_idx];
        if observed_terminal != expected_terminal {
            return Err(PiCcsError::ProtocolError(
                "wp/quiescence terminal value mismatch".into(),
            ));
        }
    } else if !mem_proof.wp_me_claims.is_empty() {
        return Err(PiCcsError::ProtocolError(
            "unexpected WP ME claims: wp/quiescence stage is not enabled".into(),
        ));
    }

    Ok(())
}

