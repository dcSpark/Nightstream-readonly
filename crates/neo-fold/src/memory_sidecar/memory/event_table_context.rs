use super::*;

pub(crate) fn build_event_table_shout_context(
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    ell_n: usize,
    r_cycle: &[K],
) -> Result<(K, K, K, Option<RouteAShoutEventTraceHashOracle>), PiCcsError> {
    let any_event_table_shout = step
        .lut_instances
        .iter()
        .any(|(inst, _wit)| matches!(inst.table_spec, Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. })));
    if any_event_table_shout {
        for (idx, (inst, _wit)) in step.lut_instances.iter().enumerate() {
            if !matches!(inst.table_spec, Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. })) {
                return Err(PiCcsError::InvalidInput(format!(
                    "event-table Shout mode requires all Shout instances to use RiscvOpcodeEventTablePacked (lut_idx={idx})"
                )));
            }
        }
    }

    let (event_alpha, event_beta, event_gamma) = if any_event_table_shout {
        if r_cycle.len() < 3 {
            return Err(PiCcsError::InvalidInput("event-table Shout requires ell_n >= 3".into()));
        }
        (r_cycle[0], r_cycle[1], r_cycle[2])
    } else {
        (K::ZERO, K::ZERO, K::ZERO)
    };

    let shout_event_trace_hash: Option<RouteAShoutEventTraceHashOracle> = if any_event_table_shout {
        let m_in = step.mcs.0.m_in;
        if m_in != 5 {
            return Err(PiCcsError::InvalidInput(format!(
                "event-table Shout trace linkage expects m_in=5 (got {m_in})"
            )));
        }
        let trace = Rv32TraceLayout::new();
        let m = step.mcs.1.Z.cols();
        let t_len = step
            .mem_instances
            .first()
            .map(|(inst, _wit)| inst.steps)
            .or_else(|| {
                let w = m.checked_sub(m_in)?;
                if trace.cols == 0 || w % trace.cols != 0 {
                    return None;
                }
                Some(w / trace.cols)
            })
            .ok_or_else(|| PiCcsError::InvalidInput("event-table Shout trace linkage missing t_len".into()))?;
        if t_len == 0 {
            return Err(PiCcsError::InvalidInput(
                "event-table Shout trace linkage requires t_len >= 1".into(),
            ));
        }
        let pow2_cycle = 1usize
            .checked_shl(ell_n as u32)
            .ok_or_else(|| PiCcsError::InvalidInput("event-table Shout: 2^ell_n overflow".into()))?;
        if m_in
            .checked_add(t_len)
            .ok_or_else(|| PiCcsError::InvalidInput("event-table Shout: m_in + t_len overflow".into()))?
            > pow2_cycle
        {
            return Err(PiCcsError::InvalidInput(format!(
                "event-table Shout: trace time rows out of range: m_in({m_in}) + t_len({t_len}) > 2^ell_n({pow2_cycle})"
            )));
        }

        let d = neo_math::D;
        let z = &step.mcs.1.Z;
        if z.rows() != d {
            return Err(PiCcsError::InvalidInput(format!(
                "event-table Shout: CPU witness Z.rows()={} != D={d}",
                z.rows()
            )));
        }
        if z.cols() != m {
            return Err(PiCcsError::ProtocolError(
                "event-table Shout: CPU witness width drift".into(),
            ));
        }

        let b_k = K::from(F::from_u64(params.b as u64));
        let mut pow_b = Vec::with_capacity(d);
        let mut cur = K::ONE;
        for _ in 0..d {
            pow_b.push(cur);
            cur *= b_k;
        }
        let decode_idx = |idx: usize| -> Result<K, PiCcsError> {
            if idx >= m {
                return Err(PiCcsError::InvalidInput(format!(
                    "event-table Shout: z idx out of range (idx={idx}, m={m})"
                )));
            }
            let mut acc = K::ZERO;
            for rho in 0..d {
                acc += pow_b[rho] * K::from(z[(rho, idx)]);
            }
            Ok(acc)
        };

        let trace_base = m_in;
        let shout_col = |col_id: usize, j: usize| -> Result<K, PiCcsError> {
            let col_offset = col_id
                .checked_mul(t_len)
                .ok_or_else(|| PiCcsError::InvalidInput("trace col_id * t_len overflow".into()))?;
            let idx = trace_base
                .checked_add(col_offset)
                .and_then(|x| x.checked_add(j))
                .ok_or_else(|| PiCcsError::InvalidInput("trace z idx overflow".into()))?;
            decode_idx(idx)
        };

        let mut gate_entries: Vec<(usize, K)> = Vec::new();
        let mut hash_entries: Vec<(usize, K)> = Vec::new();
        for j in 0..t_len {
            let t = m_in + j;
            let gate = shout_col(trace.shout_has_lookup, j)?;
            if gate == K::ZERO {
                continue;
            }
            gate_entries.push((t, gate));

            let val = shout_col(trace.shout_val, j)?;
            let lhs = shout_col(trace.shout_lhs, j)?;
            let rhs = shout_col(trace.shout_rhs, j)?;
            let hash = K::ONE + event_alpha * val + event_beta * lhs + event_gamma * rhs;
            if hash != K::ZERO {
                hash_entries.push((t, hash));
            }
        }

        let gate = SparseIdxVec::from_entries(pow2_cycle, gate_entries);
        let hash = SparseIdxVec::from_entries(pow2_cycle, hash_entries);
        let (oracle, claim) = ShoutValueOracleSparse::new(r_cycle, gate, hash);
        Some(RouteAShoutEventTraceHashOracle {
            oracle: Box::new(oracle),
            claim,
        })
    } else {
        None
    };

    Ok((event_alpha, event_beta, event_gamma, shout_event_trace_hash))
}
