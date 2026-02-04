use neo_ccs::relations::CcsStructure;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;
use std::collections::HashMap;

use crate::cpu::{build_bus_layout_for_instances_with_shout_and_twist_lanes, BusLayout};
use crate::riscv::exec_table::Rv32ExecTable;
use crate::riscv::trace::{
    extract_shout_lanes_over_time, extract_twist_lanes_over_time, Rv32TraceLayout, Rv32TraceWitness,
};

use super::constraint_builder::{build_r1cs_ccs, Constraint};

/// Fixed-width, time-in-rows trace CCS layout.
///
/// This is an MVP "wiring invariants" CCS for Tier 2.1:
/// - fixed columns over time (`t` rows),
/// - small AIR-like invariants compiled into a CCS,
/// - no ISA semantics (ALU/mem correctness) yet.
///
/// Witness layout (column-major trace region):
/// `cell(trace_col, row) = trace_base + trace_col * t + row`.
#[derive(Clone, Debug)]
pub struct Rv32TraceCcsLayout {
    pub t: usize,
    pub m_in: usize,
    pub m: usize,

    // Public scalars.
    pub const_one: usize,
    pub pc0: usize,
    pub pc_final: usize,
    pub halted_in: usize,
    pub halted_out: usize,

    pub trace_base: usize,
    pub trace: Rv32TraceLayout,
}

impl Rv32TraceCcsLayout {
    pub fn new(t: usize) -> Result<Self, String> {
        if t == 0 {
            return Err("Rv32TraceCcsLayout: t must be >= 1".into());
        }

        let const_one: usize = 0;
        let pc0: usize = 1;
        let pc_final: usize = 2;
        let halted_in: usize = 3;
        let halted_out: usize = 4;
        let m_in: usize = 5;

        let trace = Rv32TraceLayout::new();
        let trace_base = m_in;
        let trace_len = trace
            .cols
            .checked_mul(t)
            .ok_or_else(|| "Rv32TraceCcsLayout: trace_len overflow".to_string())?;
        let m = trace_base
            .checked_add(trace_len)
            .ok_or_else(|| "Rv32TraceCcsLayout: m overflow".to_string())?;

        Ok(Self {
            t,
            m_in,
            m,
            const_one,
            pc0,
            pc_final,
            halted_in,
            halted_out,
            trace_base,
            trace,
        })
    }

    /// Full witness index for a trace cell.
    #[inline]
    pub fn cell(&self, trace_col: usize, row: usize) -> usize {
        debug_assert!(trace_col < self.trace.cols);
        debug_assert!(row < self.t);
        self.trace_base + trace_col * self.t + row
    }
}

/// Build the public inputs `x` and witness `w` for the trace CCS from an exec table.
pub fn rv32_trace_ccs_witness_from_exec_table(
    layout: &Rv32TraceCcsLayout,
    exec: &Rv32ExecTable,
) -> Result<(Vec<F>, Vec<F>), String> {
    let wit = Rv32TraceWitness::from_exec_table(&layout.trace, exec)?;
    rv32_trace_ccs_witness_from_trace_witness(layout, &wit)
}

/// Build the public inputs `x` and witness `w` for the trace CCS from a trace witness.
pub fn rv32_trace_ccs_witness_from_trace_witness(
    layout: &Rv32TraceCcsLayout,
    wit: &Rv32TraceWitness,
) -> Result<(Vec<F>, Vec<F>), String> {
    if wit.t != layout.t {
        return Err(format!(
            "trace CCS witness: t mismatch (wit.t={} layout.t={})",
            wit.t, layout.t
        ));
    }
    if wit.cols.len() != layout.trace.cols {
        return Err(format!(
            "trace CCS witness: width mismatch (wit.cols={} trace.cols={})",
            wit.cols.len(),
            layout.trace.cols
        ));
    }

    let mut x = vec![F::ZERO; layout.m_in];
    x[layout.const_one] = F::ONE;
    x[layout.pc0] = wit.cols[layout.trace.pc_before][0];
    x[layout.pc_final] = wit.cols[layout.trace.pc_after][layout.t - 1];
    x[layout.halted_in] = wit.cols[layout.trace.halted][0];
    x[layout.halted_out] = wit.cols[layout.trace.halted][layout.t - 1];

    let mut w = vec![F::ZERO; layout.m - layout.m_in];
    for trace_col in 0..layout.trace.cols {
        let col = &wit.cols[trace_col];
        for row in 0..layout.t {
            let idx = layout.cell(trace_col, row);
            w[idx - layout.m_in] = col[row];
        }
    }

    Ok((x, w))
}

/// Build an MVP trace CCS that enforces only wiring invariants (AIR-like constraints),
/// not full ISA semantics.
pub fn build_rv32_trace_wiring_ccs(layout: &Rv32TraceCcsLayout) -> Result<CcsStructure<F>, String> {
    let one = layout.const_one;
    let t = layout.t;
    let tr = |c: usize, i: usize| -> usize { layout.cell(c, i) };
    let l = &layout.trace;

    let bool01 = |x: usize| -> Constraint<F> {
        // x * (x - 1) = 0
        Constraint::terms(x, false, vec![(x, F::ONE), (one, -F::ONE)])
    };

    let mut cons: Vec<Constraint<F>> = Vec::new();

    // Public bindings.
    cons.push(Constraint::terms(
        one,
        false,
        vec![(layout.pc0, F::ONE), (tr(l.pc_before, 0), -F::ONE)],
    ));
    cons.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.pc_final, F::ONE),
            (tr(l.pc_after, t - 1), -F::ONE),
        ],
    ));
    cons.push(Constraint::terms(
        one,
        false,
        vec![(layout.halted_in, F::ONE), (tr(l.halted, 0), -F::ONE)],
    ));
    cons.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.halted_out, F::ONE),
            (tr(l.halted, t - 1), -F::ONE),
        ],
    ));

    for i in 0..t {
        let active = tr(l.active, i);
        let halted = tr(l.halted, i);
        let rd_has_write = tr(l.rd_has_write, i);
        let ram_has_read = tr(l.ram_has_read, i);
        let ram_has_write = tr(l.ram_has_write, i);
        let shout_has_lookup = tr(l.shout_has_lookup, i);

        // Booleans.
        cons.push(bool01(active));
        cons.push(bool01(halted));
        cons.push(bool01(rd_has_write));
        cons.push(bool01(ram_has_read));
        cons.push(bool01(ram_has_write));
        cons.push(bool01(shout_has_lookup));
        for &b in &l.rd_bit {
            cons.push(bool01(tr(b, i)));
        }

        // Inactive padding invariants: (1 - active) * col = 0.
        for &c in &[
            l.instr_word,
            l.opcode,
            l.funct3,
            l.funct7,
            l.rd,
            l.rs1,
            l.rs2,
            l.prog_addr,
            l.prog_value,
            l.rs1_addr,
            l.rs1_val,
            l.rs2_addr,
            l.rs2_val,
            l.rd_has_write,
            l.rd_addr,
            l.rd_val,
            l.ram_has_read,
            l.ram_has_write,
            l.ram_addr,
            l.ram_rv,
            l.ram_wv,
            l.shout_has_lookup,
            l.shout_val,
            l.shout_lhs,
            l.shout_rhs,
        ] {
            cons.push(Constraint::terms(active, true, vec![(tr(c, i), F::ONE)]));
        }

        // rd packing: rd == Σ 2^k * rd_bit[k].
        cons.push(Constraint::terms(
            one,
            false,
            vec![
                (tr(l.rd, i), F::ONE),
                (tr(l.rd_bit[0], i), -F::ONE),
                (tr(l.rd_bit[1], i), -F::from_u64(2)),
                (tr(l.rd_bit[2], i), -F::from_u64(4)),
                (tr(l.rd_bit[3], i), -F::from_u64(8)),
                (tr(l.rd_bit[4], i), -F::from_u64(16)),
            ],
        ));

        // rd_is_zero prefix products.
        //
        // z01 = (1-b0)*(1-b1)
        cons.push(Constraint {
            condition_col: tr(l.rd_bit[0], i),
            negate_condition: true,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (tr(l.rd_bit[1], i), -F::ONE)],
            c_terms: vec![(tr(l.rd_is_zero_01, i), F::ONE)],
        });
        // z012 = z01*(1-b2)
        cons.push(Constraint {
            condition_col: tr(l.rd_is_zero_01, i),
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (tr(l.rd_bit[2], i), -F::ONE)],
            c_terms: vec![(tr(l.rd_is_zero_012, i), F::ONE)],
        });
        // z0123 = z012*(1-b3)
        cons.push(Constraint {
            condition_col: tr(l.rd_is_zero_012, i),
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (tr(l.rd_bit[3], i), -F::ONE)],
            c_terms: vec![(tr(l.rd_is_zero_0123, i), F::ONE)],
        });
        // z = z0123*(1-b4)
        cons.push(Constraint {
            condition_col: tr(l.rd_is_zero_0123, i),
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (tr(l.rd_bit[4], i), -F::ONE)],
            c_terms: vec![(tr(l.rd_is_zero, i), F::ONE)],
        });

        // Sound x0 invariant: rd_has_write * rd_is_zero = 0.
        cons.push(Constraint::terms(
            rd_has_write,
            false,
            vec![(tr(l.rd_is_zero, i), F::ONE)],
        ));

        // If rd_has_write==0, rd_addr and rd_val must be 0.
        cons.push(Constraint::terms(rd_has_write, true, vec![(tr(l.rd_addr, i), F::ONE)]));
        cons.push(Constraint::terms(rd_has_write, true, vec![(tr(l.rd_val, i), F::ONE)]));

        // RAM bus padding: (1 - flag) * value == 0.
        cons.push(Constraint::terms(ram_has_read, true, vec![(tr(l.ram_rv, i), F::ONE)]));
        cons.push(Constraint::terms(
            ram_has_write,
            true,
            vec![(tr(l.ram_wv, i), F::ONE)],
        ));

        // Shout padding: (1 - has_lookup) * val == 0.
        cons.push(Constraint::terms(
            shout_has_lookup,
            true,
            vec![(tr(l.shout_val, i), F::ONE)],
        ));
        cons.push(Constraint::terms(
            shout_has_lookup,
            true,
            vec![(tr(l.shout_lhs, i), F::ONE)],
        ));
        cons.push(Constraint::terms(
            shout_has_lookup,
            true,
            vec![(tr(l.shout_rhs, i), F::ONE)],
        ));

        // Active → PROG binding.
        cons.push(Constraint::terms(
            active,
            false,
            vec![(tr(l.prog_addr, i), F::ONE), (tr(l.pc_before, i), -F::ONE)],
        ));
        cons.push(Constraint::terms(
            active,
            false,
            vec![
                (tr(l.prog_value, i), F::ONE),
                (tr(l.instr_word, i), -F::ONE),
            ],
        ));

        // Active → REG addr bindings; rd_has_write → rd_addr binding.
        cons.push(Constraint::terms(
            active,
            false,
            vec![(tr(l.rs1_addr, i), F::ONE), (tr(l.rs1, i), -F::ONE)],
        ));
        cons.push(Constraint::terms(
            active,
            false,
            vec![(tr(l.rs2_addr, i), F::ONE), (tr(l.rs2, i), -F::ONE)],
        ));
        cons.push(Constraint::terms(
            rd_has_write,
            false,
            vec![(tr(l.rd_addr, i), F::ONE), (tr(l.rd, i), -F::ONE)],
        ));
    }

    for i in 0..t.saturating_sub(1) {
        // pc_after[i] == pc_before[i+1]
        cons.push(Constraint::terms(
            one,
            false,
            vec![
                (tr(l.pc_after, i), F::ONE),
                (tr(l.pc_before, i + 1), -F::ONE),
            ],
        ));

        // cycle[i+1] == cycle[i] + 1
        cons.push(Constraint::terms(
            one,
            false,
            vec![
                (tr(l.cycle, i + 1), F::ONE),
                (tr(l.cycle, i), -F::ONE),
                (one, -F::ONE),
            ],
        ));

        // Once inactive, remain inactive: active[i+1] * (1 - active[i]) == 0
        cons.push(Constraint::terms(
            tr(l.active, i + 1),
            false,
            vec![(one, F::ONE), (tr(l.active, i), -F::ONE)],
        ));

        // Once halted, remain halted: halted[i] * (1 - halted[i+1]) == 0
        cons.push(Constraint::terms(
            tr(l.halted, i),
            false,
            vec![(one, F::ONE), (tr(l.halted, i + 1), -F::ONE)],
        ));
    }

    build_r1cs_ccs(&cons, cons.len(), layout.m, layout.const_one)
}

/// Trace wiring CCS layout extended with a **PROG + REG + RAM Twist bus region**.
///
/// This is a Tier 2.1 "Phase 3 bridge" used to prove that PROG and REG accesses
/// are consistent with the trace, using the existing Route-A Twist subprotocols.
///
/// Concretely, we append a shared-bus tail to the trace witness `z` (column-major over time):
/// - Twist instance 0: `PROG_ID` (lanes=1, ell_addr=prog_d)
/// - Twist instance 1: `REG_ID`  (lanes=2, ell_addr=5)
///
/// The bus region is laid out exactly like `cpu::BusLayout`, so Neo-Fold can reuse the
/// existing shared-bus Route-A pipeline to prove/verify the Twist sidecars.
#[derive(Clone, Debug)]
pub struct Rv32TraceTwistCcsLayout {
    pub t: usize,
    pub m_in: usize,
    pub m: usize,

    // Public scalars.
    pub const_one: usize,
    pub pc0: usize,
    pub pc_final: usize,
    pub halted_in: usize,
    pub halted_out: usize,

    pub trace_base: usize,
    pub trace: Rv32TraceLayout,

    /// Canonical Shout table ids (in the same order as `bus.shout_cols`).
    pub shout_table_ids: Vec<u32>,

    /// Shared-bus tail for Shout + PROG + REG + RAM instances.
    pub bus: BusLayout,
}

impl Rv32TraceTwistCcsLayout {
    pub const PROG_MEM_IDX: usize = 0;
    pub const REG_MEM_IDX: usize = 1;
    pub const RAM_MEM_IDX: usize = 2;

    pub fn new(t: usize, prog_d: usize, ram_d: usize, shout_table_ids: &[u32]) -> Result<Self, String> {
        if t == 0 {
            return Err("Rv32TraceTwistCcsLayout: t must be >= 1".into());
        }
        if prog_d == 0 {
            return Err("Rv32TraceTwistCcsLayout: prog_d must be >= 1".into());
        }
        if ram_d == 0 {
            return Err("Rv32TraceTwistCcsLayout: ram_d must be >= 1".into());
        }

        // Canonicalize Shout table ids (no duplicates, stable order).
        let mut shout_table_ids: Vec<u32> = shout_table_ids.to_vec();
        shout_table_ids.sort_unstable();
        shout_table_ids.dedup();

        let const_one: usize = 0;
        let pc0: usize = 1;
        let pc_final: usize = 2;
        let halted_in: usize = 3;
        let halted_out: usize = 4;
        let m_in: usize = 5;

        let trace = Rv32TraceLayout::new();
        let trace_base = m_in;

        // Bus columns: Shout + PROG (1 lane) + REG (2 lanes) + RAM (1 lane).
        // For Twist: per-lane columns are `[ra_bits, wa_bits, has_read, has_write, wv, rv, inc]`
        // so `bus_cols = Σ lanes * (2*ell_addr + 5)`.
        //
        // For Shout (RISC-V implicit tables): each instance has `ell_addr = 2*xlen = 64` bits and
        // per-lane columns `[addr_bits, has_lookup, val]` so `lane_len = ell_addr + 2`.
        let shout_ell_addr = 64usize;
        let shout_lane_len = shout_ell_addr + 2;
        let shout_bus_cols = shout_table_ids
            .len()
            .checked_mul(shout_lane_len)
            .ok_or("Rv32TraceTwistCcsLayout: shout bus overflow")?;

        let prog_bus_cols = 1usize
            .checked_mul(2usize.checked_mul(prog_d).ok_or("Rv32TraceTwistCcsLayout: prog bus overflow")? + 5)
            .ok_or("Rv32TraceTwistCcsLayout: prog bus overflow")?;
        let reg_bus_cols = 2usize
            .checked_mul(2usize.checked_mul(5).ok_or("Rv32TraceTwistCcsLayout: reg bus overflow")? + 5)
            .ok_or("Rv32TraceTwistCcsLayout: reg bus overflow")?;
        let ram_bus_cols = 1usize
            .checked_mul(2usize.checked_mul(ram_d).ok_or("Rv32TraceTwistCcsLayout: ram bus overflow")? + 5)
            .ok_or("Rv32TraceTwistCcsLayout: ram bus overflow")?;
        let bus_cols = shout_bus_cols
            .checked_add(prog_bus_cols)
            .and_then(|c| c.checked_add(reg_bus_cols))
            .and_then(|c| c.checked_add(ram_bus_cols))
            .ok_or("Rv32TraceTwistCcsLayout: bus_cols overflow")?;

        let trace_len = trace
            .cols
            .checked_mul(t)
            .ok_or_else(|| "Rv32TraceTwistCcsLayout: trace_len overflow".to_string())?;
        let bus_len = bus_cols
            .checked_mul(t)
            .ok_or_else(|| "Rv32TraceTwistCcsLayout: bus_len overflow".to_string())?;
        let m = trace_base
            .checked_add(trace_len)
            .and_then(|m| m.checked_add(bus_len))
            .ok_or_else(|| "Rv32TraceTwistCcsLayout: m overflow".to_string())?;

        // Build a canonical BusLayout for Shout + PROG + REG + RAM.
        let shout_instances: Vec<(usize, usize)> = (0..shout_table_ids.len())
            .map(|_| (shout_ell_addr, 1usize))
            .collect();
        let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
            m,
            m_in,
            /*chunk_size=*/ t,
            shout_instances,
            [(prog_d, 1usize), (5usize, 2usize), (ram_d, 1usize)],
        )
        .map_err(|e| format!("Rv32TraceTwistCcsLayout: bus layout: {e}"))?;
        if bus.twist_cols.len() != 3 {
            return Err("Rv32TraceTwistCcsLayout: expected 3 Twist instances (PROG, REG, RAM)".into());
        }
        if bus.shout_cols.len() != shout_table_ids.len() {
            return Err("Rv32TraceTwistCcsLayout: shout instance count mismatch".into());
        }

        Ok(Self {
            t,
            m_in,
            m,
            const_one,
            pc0,
            pc_final,
            halted_in,
            halted_out,
            trace_base,
            trace,
            shout_table_ids,
            bus,
        })
    }

    /// Trace-region witness index for a trace cell.
    #[inline]
    pub fn trace_cell(&self, trace_col: usize, row: usize) -> usize {
        debug_assert!(trace_col < self.trace.cols);
        debug_assert!(row < self.t);
        self.trace_base + trace_col * self.t + row
    }
}

/// Build the public inputs `x` and witness `w` for the trace+Twist CCS.
///
/// `init_regs` provides the public initial REG state (addresses 0..32). This is used to compute
/// the Twist `inc_at_write_addr` bus column for reg writes.
///
/// `init_ram` provides the public initial RAM state (sparse). This is used to compute the Twist
/// `inc_at_write_addr` bus column for RAM writes.
pub fn rv32_trace_twist_ccs_witness_from_exec_table(
    layout: &Rv32TraceTwistCcsLayout,
    exec: &Rv32ExecTable,
    init_regs: &HashMap<u64, u64>,
    init_ram: &HashMap<u64, u64>,
) -> Result<(Vec<F>, Vec<F>), String> {
    if exec.rows.len() != layout.t {
        return Err(format!(
            "trace+Twist CCS witness: t mismatch (exec.rows.len()={} layout.t={})",
            exec.rows.len(),
            layout.t
        ));
    }

    // Fill the core trace witness first.
    let wit = Rv32TraceWitness::from_exec_table(&layout.trace, exec)?;

    let mut x = vec![F::ZERO; layout.m_in];
    x[layout.const_one] = F::ONE;
    x[layout.pc0] = wit.cols[layout.trace.pc_before][0];
    x[layout.pc_final] = wit.cols[layout.trace.pc_after][layout.t - 1];
    x[layout.halted_in] = wit.cols[layout.trace.halted][0];
    x[layout.halted_out] = wit.cols[layout.trace.halted][layout.t - 1];

    let mut w = vec![F::ZERO; layout.m - layout.m_in];

    // Core trace region.
    for trace_col in 0..layout.trace.cols {
        let col = &wit.cols[trace_col];
        for row in 0..layout.t {
            let idx = layout.trace_cell(trace_col, row);
            w[idx - layout.m_in] = col[row];
        }
    }

    // Extract fixed-lane sidecar time-series and compute `inc_at_write_addr` from public init state.
    let ram_lane = &layout.bus.twist_cols[Rv32TraceTwistCcsLayout::RAM_MEM_IDX].lanes[0];
    let ram_ell_addr = ram_lane.ra_bits.end - ram_lane.ra_bits.start;
    let twist = extract_twist_lanes_over_time(exec, init_regs, init_ram, ram_ell_addr)?;
    let shout = extract_shout_lanes_over_time(exec, &layout.shout_table_ids)?;

    // Fill PROG + REG bus tail (laid out by `layout.bus`).
    //
    // IMPORTANT: bus time indices are `t = m_in + j` in Route A, but the witness stores per-step
    // values in `j` order. `bus_cell(col_id, j)` uses `j`, and Route A handles the `m_in` offset.
    if layout.bus.shout_cols.len() != shout.len() {
        return Err("trace+Twist witness: shout instance count mismatch".into());
    }

    let prog_lane = &layout.bus.twist_cols[Rv32TraceTwistCcsLayout::PROG_MEM_IDX].lanes[0];
    let reg_lanes = &layout.bus.twist_cols[Rv32TraceTwistCcsLayout::REG_MEM_IDX].lanes;
    if reg_lanes.len() != 2 {
        return Err("trace+Twist witness: REG Twist instance must have 2 lanes".into());
    }
    let reg_lane0 = &reg_lanes[0];
    let reg_lane1 = &reg_lanes[1];

    let write_bits = |w: &mut [F], addr: u64, bit_cols: std::ops::Range<usize>, j: usize| {
        let mut a = addr;
        for col_id in bit_cols {
            let idx = layout.bus.bus_cell(col_id, j) - layout.m_in;
            w[idx] = if (a & 1) == 1 { F::ONE } else { F::ZERO };
            a >>= 1;
        }
    };

    for j in 0..layout.t {
        // Shout instances (1 lane per table in this MVP).
        for (inst_idx, inst) in layout.bus.shout_cols.iter().enumerate() {
            if inst.lanes.len() != 1 {
                return Err("trace+Twist witness: Shout lanes != 1 is not supported in this MVP".into());
            }
            let lane = &inst.lanes[0];
            if shout[inst_idx].has_lookup[j] {
                w[layout.bus.bus_cell(lane.has_lookup, j) - layout.m_in] = F::ONE;
                w[layout.bus.bus_cell(lane.val, j) - layout.m_in] = F::from_u64(shout[inst_idx].value[j]);
                write_bits(&mut w, shout[inst_idx].key[j], lane.addr_bits.clone(), j);
            }
        }

        // PROG instance (1 lane, read-only).
        if twist.prog.has_read[j] {
            w[layout.bus.bus_cell(prog_lane.has_read, j) - layout.m_in] = F::ONE;
            w[layout.bus.bus_cell(prog_lane.rv, j) - layout.m_in] = F::from_u64(twist.prog.rv[j]);
            write_bits(&mut w, twist.prog.ra[j], prog_lane.ra_bits.clone(), j);
        }

        // REG lane0: read rs1; optional write rd.
        w[layout.bus.bus_cell(reg_lane0.has_read, j) - layout.m_in] = if twist.reg_lane0.has_read[j] {
            F::ONE
        } else {
            F::ZERO
        };
        w[layout.bus.bus_cell(reg_lane0.rv, j) - layout.m_in] = F::from_u64(twist.reg_lane0.rv[j]);
        write_bits(&mut w, twist.reg_lane0.ra[j], reg_lane0.ra_bits.clone(), j);

        if twist.reg_lane0.has_write[j] {
            w[layout.bus.bus_cell(reg_lane0.has_write, j) - layout.m_in] = F::ONE;
            w[layout.bus.bus_cell(reg_lane0.wv, j) - layout.m_in] = F::from_u64(twist.reg_lane0.wv[j]);
            w[layout.bus.bus_cell(reg_lane0.inc, j) - layout.m_in] = twist.reg_lane0.inc_at_write_addr[j];
            write_bits(&mut w, twist.reg_lane0.wa[j], reg_lane0.wa_bits.clone(), j);
        }

        // REG lane1: read rs2.
        w[layout.bus.bus_cell(reg_lane1.has_read, j) - layout.m_in] = if twist.reg_lane1.has_read[j] {
            F::ONE
        } else {
            F::ZERO
        };
        w[layout.bus.bus_cell(reg_lane1.rv, j) - layout.m_in] = F::from_u64(twist.reg_lane1.rv[j]);
        write_bits(&mut w, twist.reg_lane1.ra[j], reg_lane1.ra_bits.clone(), j);

        // RAM instance (1 lane, fixed-lane MVP: at most 1 read + 1 write per row).
        w[layout.bus.bus_cell(ram_lane.has_read, j) - layout.m_in] = if twist.ram.has_read[j] {
            F::ONE
        } else {
            F::ZERO
        };
        w[layout.bus.bus_cell(ram_lane.has_write, j) - layout.m_in] = if twist.ram.has_write[j] {
            F::ONE
        } else {
            F::ZERO
        };

        if twist.ram.has_read[j] {
            w[layout.bus.bus_cell(ram_lane.rv, j) - layout.m_in] = F::from_u64(twist.ram.rv[j]);
            write_bits(&mut w, twist.ram.ra[j], ram_lane.ra_bits.clone(), j);
        }
        if twist.ram.has_write[j] {
            w[layout.bus.bus_cell(ram_lane.wv, j) - layout.m_in] = F::from_u64(twist.ram.wv[j]);
            w[layout.bus.bus_cell(ram_lane.inc, j) - layout.m_in] = twist.ram.inc_at_write_addr[j];
            write_bits(&mut w, twist.ram.wa[j], ram_lane.wa_bits.clone(), j);
        }
    }

    Ok((x, w))
}

/// Build a trace wiring CCS with a shared-bus tail that exposes PROG+REG+RAM Twist lanes.
///
/// This CCS enforces:
/// - the base trace wiring invariants (same as `build_rv32_trace_wiring_ccs`), and
/// - **bus bindings** tying the PROG/REG/RAM Twist lanes to the trace columns, plus
/// - canonical bus padding constraints `(1 - has_*) * field = 0` for all gated bus fields.
pub fn build_rv32_trace_wiring_ccs_with_prog_reg_ram_twist(
    layout: &Rv32TraceTwistCcsLayout,
) -> Result<CcsStructure<F>, String> {
    let one = layout.const_one;
    let t = layout.t;
    let tr = |c: usize, i: usize| -> usize { layout.trace_cell(c, i) };
    let l = &layout.trace;

    let bool01 = |x: usize| -> Constraint<F> {
        // x * (x - 1) = 0
        Constraint::terms(x, false, vec![(x, F::ONE), (one, -F::ONE)])
    };

    let lin_eq = |a: usize, b: usize| -> Constraint<F> {
        Constraint::terms(one, false, vec![(a, F::ONE), (b, -F::ONE)])
    };

    let lin_zero = |a: usize| -> Constraint<F> { Constraint::terms(one, false, vec![(a, F::ONE)]) };

    let mut cons: Vec<Constraint<F>> = Vec::new();

    // Public bindings.
    cons.push(lin_eq(layout.pc0, tr(l.pc_before, 0)));
    cons.push(lin_eq(layout.pc_final, tr(l.pc_after, t - 1)));
    cons.push(lin_eq(layout.halted_in, tr(l.halted, 0)));
    cons.push(lin_eq(layout.halted_out, tr(l.halted, t - 1)));

    // Resolve PROG/REG bus lane descriptors once; we bind per-row via `bus_cell`.
    if layout.bus.shout_cols.len() != layout.shout_table_ids.len() {
        return Err("trace+Twist CCS: shout instance count mismatch".into());
    }
    let prog_lane = &layout.bus.twist_cols[Rv32TraceTwistCcsLayout::PROG_MEM_IDX].lanes[0];
    let reg_lanes = &layout.bus.twist_cols[Rv32TraceTwistCcsLayout::REG_MEM_IDX].lanes;
    if reg_lanes.len() != 2 {
        return Err("trace+Twist CCS: REG Twist instance must have 2 lanes".into());
    }
    let reg_lane0 = &reg_lanes[0];
    let reg_lane1 = &reg_lanes[1];
    let ram_lane = &layout.bus.twist_cols[Rv32TraceTwistCcsLayout::RAM_MEM_IDX].lanes[0];

    for i in 0..t {
        let active = tr(l.active, i);
        let halted = tr(l.halted, i);
        let rd_has_write = tr(l.rd_has_write, i);
        let ram_has_read = tr(l.ram_has_read, i);
        let ram_has_write = tr(l.ram_has_write, i);
        let shout_has_lookup = tr(l.shout_has_lookup, i);

        // Core booleans.
        cons.push(bool01(active));
        cons.push(bool01(halted));
        cons.push(bool01(rd_has_write));
        cons.push(bool01(ram_has_read));
        cons.push(bool01(ram_has_write));
        cons.push(bool01(shout_has_lookup));
        for &b in &l.rd_bit {
            cons.push(bool01(tr(b, i)));
        }

        // Shout lane booleans + canonical padding.
        for inst in &layout.bus.shout_cols {
            for lane in &inst.lanes {
                let has_lookup = layout.bus.bus_cell(lane.has_lookup, i);
                let val = layout.bus.bus_cell(lane.val, i);

                cons.push(bool01(has_lookup));
                // (1 - has_lookup) * val = 0
                cons.push(Constraint::terms(has_lookup, true, vec![(val, F::ONE)]));
                // (1 - has_lookup) * addr_bits[b] = 0
                for col_id in lane.addr_bits.clone() {
                    let bit = layout.bus.bus_cell(col_id, i);
                    cons.push(Constraint::terms(has_lookup, true, vec![(bit, F::ONE)]));
                }
            }
        }

        // Trace ↔ Shout linkage (fixed-lane policy): sum lanes must match the trace view.
        {
            let mut has_terms = vec![(shout_has_lookup, F::ONE)];
            let mut val_terms = vec![(tr(l.shout_val, i), F::ONE)];
            for inst in &layout.bus.shout_cols {
                for lane in &inst.lanes {
                    let has_lookup = layout.bus.bus_cell(lane.has_lookup, i);
                    let val = layout.bus.bus_cell(lane.val, i);
                    has_terms.push((has_lookup, -F::ONE));
                    val_terms.push((val, -F::ONE));
                }
            }
            cons.push(Constraint::terms(one, false, has_terms));
            cons.push(Constraint::terms(one, false, val_terms));
        }

        // Inactive padding invariants: (1 - active) * col = 0.
        for &c in &[
            l.instr_word,
            l.opcode,
            l.funct3,
            l.funct7,
            l.rd,
            l.rs1,
            l.rs2,
            l.prog_addr,
            l.prog_value,
            l.rs1_addr,
            l.rs1_val,
            l.rs2_addr,
            l.rs2_val,
            l.rd_has_write,
            l.rd_addr,
            l.rd_val,
            l.ram_has_read,
            l.ram_has_write,
            l.ram_addr,
            l.ram_rv,
            l.ram_wv,
            l.shout_has_lookup,
            l.shout_val,
            l.shout_lhs,
            l.shout_rhs,
        ] {
            cons.push(Constraint::terms(active, true, vec![(tr(c, i), F::ONE)]));
        }

        // rd packing: rd == Σ 2^k * rd_bit[k].
        cons.push(Constraint::terms(
            one,
            false,
            vec![
                (tr(l.rd, i), F::ONE),
                (tr(l.rd_bit[0], i), -F::ONE),
                (tr(l.rd_bit[1], i), -F::from_u64(2)),
                (tr(l.rd_bit[2], i), -F::from_u64(4)),
                (tr(l.rd_bit[3], i), -F::from_u64(8)),
                (tr(l.rd_bit[4], i), -F::from_u64(16)),
            ],
        ));

        // rd_is_zero prefix products.
        cons.push(Constraint {
            condition_col: tr(l.rd_bit[0], i),
            negate_condition: true,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (tr(l.rd_bit[1], i), -F::ONE)],
            c_terms: vec![(tr(l.rd_is_zero_01, i), F::ONE)],
        });
        cons.push(Constraint {
            condition_col: tr(l.rd_is_zero_01, i),
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (tr(l.rd_bit[2], i), -F::ONE)],
            c_terms: vec![(tr(l.rd_is_zero_012, i), F::ONE)],
        });
        cons.push(Constraint {
            condition_col: tr(l.rd_is_zero_012, i),
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (tr(l.rd_bit[3], i), -F::ONE)],
            c_terms: vec![(tr(l.rd_is_zero_0123, i), F::ONE)],
        });
        cons.push(Constraint {
            condition_col: tr(l.rd_is_zero_0123, i),
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (tr(l.rd_bit[4], i), -F::ONE)],
            c_terms: vec![(tr(l.rd_is_zero, i), F::ONE)],
        });

        // Sound x0 invariant: rd_has_write * rd_is_zero = 0.
        cons.push(Constraint::terms(
            rd_has_write,
            false,
            vec![(tr(l.rd_is_zero, i), F::ONE)],
        ));

        // If rd_has_write==0, rd_addr and rd_val must be 0.
        cons.push(Constraint::terms(rd_has_write, true, vec![(tr(l.rd_addr, i), F::ONE)]));
        cons.push(Constraint::terms(rd_has_write, true, vec![(tr(l.rd_val, i), F::ONE)]));

        // RAM bus padding: (1 - flag) * value == 0.
        cons.push(Constraint::terms(ram_has_read, true, vec![(tr(l.ram_rv, i), F::ONE)]));
        cons.push(Constraint::terms(
            ram_has_write,
            true,
            vec![(tr(l.ram_wv, i), F::ONE)],
        ));

        // Shout padding: (1 - has_lookup) * val == 0.
        cons.push(Constraint::terms(
            shout_has_lookup,
            true,
            vec![(tr(l.shout_val, i), F::ONE)],
        ));
        cons.push(Constraint::terms(
            shout_has_lookup,
            true,
            vec![(tr(l.shout_lhs, i), F::ONE)],
        ));
        cons.push(Constraint::terms(
            shout_has_lookup,
            true,
            vec![(tr(l.shout_rhs, i), F::ONE)],
        ));

        // Active → PROG binding.
        cons.push(Constraint::terms(
            active,
            false,
            vec![(tr(l.prog_addr, i), F::ONE), (tr(l.pc_before, i), -F::ONE)],
        ));
        cons.push(Constraint::terms(
            active,
            false,
            vec![
                (tr(l.prog_value, i), F::ONE),
                (tr(l.instr_word, i), -F::ONE),
            ],
        ));

        // Active → REG addr bindings; rd_has_write → rd_addr binding.
        cons.push(Constraint::terms(
            active,
            false,
            vec![(tr(l.rs1_addr, i), F::ONE), (tr(l.rs1, i), -F::ONE)],
        ));
        cons.push(Constraint::terms(
            active,
            false,
            vec![(tr(l.rs2_addr, i), F::ONE), (tr(l.rs2, i), -F::ONE)],
        ));
        cons.push(Constraint::terms(
            rd_has_write,
            false,
            vec![(tr(l.rd_addr, i), F::ONE), (tr(l.rd, i), -F::ONE)],
        ));

        // ====================================================================
        // PROG + REG Twist bus bindings (trace-linked)
        // ====================================================================

        // PROG: has_read == active, has_write == 0, rv == prog_value, and addr bits pack to prog_addr.
        {
            let has_read = layout.bus.bus_cell(prog_lane.has_read, i);
            let has_write = layout.bus.bus_cell(prog_lane.has_write, i);
            let rv = layout.bus.bus_cell(prog_lane.rv, i);
            let wv = layout.bus.bus_cell(prog_lane.wv, i);
            let inc = layout.bus.bus_cell(prog_lane.inc, i);

            cons.push(lin_eq(has_read, active));
            cons.push(lin_zero(has_write));
            cons.push(lin_eq(rv, tr(l.prog_value, i)));
            // Bind write-lane cells outside padding rows (PROG is read-only).
            cons.push(lin_zero(wv));
            for col_id in prog_lane.wa_bits.clone() {
                let bit = layout.bus.bus_cell(col_id, i);
                cons.push(lin_zero(bit));
            }

            // Canonical padding: (1-has_read)*rv = 0 and (1-has_read)*ra_bits[b] = 0.
            cons.push(Constraint::terms(has_read, true, vec![(rv, F::ONE)]));
            for col_id in prog_lane.ra_bits.clone() {
                let bit = layout.bus.bus_cell(col_id, i);
                cons.push(Constraint::terms(has_read, true, vec![(bit, F::ONE)]));
            }

            // Canonical padding for unused write lane (has_write==0 forces all to 0).
            cons.push(Constraint::terms(has_write, true, vec![(wv, F::ONE)]));
            cons.push(Constraint::terms(has_write, true, vec![(inc, F::ONE)]));
            for col_id in prog_lane.wa_bits.clone() {
                let bit = layout.bus.bus_cell(col_id, i);
                cons.push(Constraint::terms(has_write, true, vec![(bit, F::ONE)]));
            }

            // Pack prog_addr from ra_bits.
            let mut terms = Vec::with_capacity(prog_lane.ra_bits.end - prog_lane.ra_bits.start + 1);
            terms.push((tr(l.prog_addr, i), F::ONE));
            let mut pow = F::ONE;
            for col_id in prog_lane.ra_bits.clone() {
                let bit = layout.bus.bus_cell(col_id, i);
                terms.push((bit, -pow));
                pow *= F::from_u64(2);
            }
            cons.push(Constraint::terms(one, false, terms));
        }

        // REG lane0: read rs1; optional write rd.
        {
            let has_read = layout.bus.bus_cell(reg_lane0.has_read, i);
            let has_write = layout.bus.bus_cell(reg_lane0.has_write, i);
            let rv = layout.bus.bus_cell(reg_lane0.rv, i);
            let wv = layout.bus.bus_cell(reg_lane0.wv, i);
            let inc = layout.bus.bus_cell(reg_lane0.inc, i);

            cons.push(lin_eq(has_read, active));
            cons.push(lin_eq(has_write, rd_has_write));
            cons.push(lin_eq(rv, tr(l.rs1_val, i)));
            cons.push(lin_eq(wv, tr(l.rd_val, i)));

            // Canonical padding.
            cons.push(Constraint::terms(has_read, true, vec![(rv, F::ONE)]));
            for col_id in reg_lane0.ra_bits.clone() {
                let bit = layout.bus.bus_cell(col_id, i);
                cons.push(Constraint::terms(has_read, true, vec![(bit, F::ONE)]));
            }
            cons.push(Constraint::terms(has_write, true, vec![(wv, F::ONE)]));
            cons.push(Constraint::terms(has_write, true, vec![(inc, F::ONE)]));
            for col_id in reg_lane0.wa_bits.clone() {
                let bit = layout.bus.bus_cell(col_id, i);
                cons.push(Constraint::terms(has_write, true, vec![(bit, F::ONE)]));
            }

            // Pack rs1_addr from ra_bits.
            let mut terms = Vec::with_capacity(reg_lane0.ra_bits.end - reg_lane0.ra_bits.start + 1);
            terms.push((tr(l.rs1_addr, i), F::ONE));
            let mut pow = F::ONE;
            for col_id in reg_lane0.ra_bits.clone() {
                let bit = layout.bus.bus_cell(col_id, i);
                terms.push((bit, -pow));
                pow *= F::from_u64(2);
            }
            cons.push(Constraint::terms(one, false, terms));

            // Pack rd_addr from wa_bits (rd_addr is already 0 when rd_has_write==0).
            let mut terms = Vec::with_capacity(reg_lane0.wa_bits.end - reg_lane0.wa_bits.start + 1);
            terms.push((tr(l.rd_addr, i), F::ONE));
            let mut pow = F::ONE;
            for col_id in reg_lane0.wa_bits.clone() {
                let bit = layout.bus.bus_cell(col_id, i);
                terms.push((bit, -pow));
                pow *= F::from_u64(2);
            }
            cons.push(Constraint::terms(one, false, terms));
        }

        // REG lane1: read rs2; no writes.
        {
            let has_read = layout.bus.bus_cell(reg_lane1.has_read, i);
            let has_write = layout.bus.bus_cell(reg_lane1.has_write, i);
            let rv = layout.bus.bus_cell(reg_lane1.rv, i);
            let wv = layout.bus.bus_cell(reg_lane1.wv, i);
            let inc = layout.bus.bus_cell(reg_lane1.inc, i);

            cons.push(lin_eq(has_read, active));
            cons.push(lin_zero(has_write));
            cons.push(lin_eq(rv, tr(l.rs2_val, i)));
            // Bind write-lane cells outside padding rows (lane1 is read-only by convention).
            cons.push(lin_zero(wv));
            for col_id in reg_lane1.wa_bits.clone() {
                let bit = layout.bus.bus_cell(col_id, i);
                cons.push(lin_zero(bit));
            }

            // Canonical padding.
            cons.push(Constraint::terms(has_read, true, vec![(rv, F::ONE)]));
            for col_id in reg_lane1.ra_bits.clone() {
                let bit = layout.bus.bus_cell(col_id, i);
                cons.push(Constraint::terms(has_read, true, vec![(bit, F::ONE)]));
            }
            cons.push(Constraint::terms(has_write, true, vec![(wv, F::ONE)]));
            cons.push(Constraint::terms(has_write, true, vec![(inc, F::ONE)]));
            for col_id in reg_lane1.wa_bits.clone() {
                let bit = layout.bus.bus_cell(col_id, i);
                cons.push(Constraint::terms(has_write, true, vec![(bit, F::ONE)]));
            }

            // Pack rs2_addr from ra_bits.
            let mut terms = Vec::with_capacity(reg_lane1.ra_bits.end - reg_lane1.ra_bits.start + 1);
            terms.push((tr(l.rs2_addr, i), F::ONE));
            let mut pow = F::ONE;
            for col_id in reg_lane1.ra_bits.clone() {
                let bit = layout.bus.bus_cell(col_id, i);
                terms.push((bit, -pow));
                pow *= F::from_u64(2);
            }
            cons.push(Constraint::terms(one, false, terms));
        }

        // RAM lane0: fixed-lane MVP (at most 1 read + 1 write per row).
        {
            let has_read = layout.bus.bus_cell(ram_lane.has_read, i);
            let has_write = layout.bus.bus_cell(ram_lane.has_write, i);
            let rv = layout.bus.bus_cell(ram_lane.rv, i);
            let wv = layout.bus.bus_cell(ram_lane.wv, i);
            let inc = layout.bus.bus_cell(ram_lane.inc, i);

            // Bind selectors and values to the trace columns.
            cons.push(lin_eq(has_read, tr(l.ram_has_read, i)));
            cons.push(lin_eq(has_write, tr(l.ram_has_write, i)));
            cons.push(lin_eq(rv, tr(l.ram_rv, i)));
            cons.push(lin_eq(wv, tr(l.ram_wv, i)));

            // Canonical padding.
            cons.push(Constraint::terms(has_read, true, vec![(rv, F::ONE)]));
            for col_id in ram_lane.ra_bits.clone() {
                let bit = layout.bus.bus_cell(col_id, i);
                cons.push(Constraint::terms(has_read, true, vec![(bit, F::ONE)]));
            }
            cons.push(Constraint::terms(has_write, true, vec![(wv, F::ONE)]));
            cons.push(Constraint::terms(has_write, true, vec![(inc, F::ONE)]));
            for col_id in ram_lane.wa_bits.clone() {
                let bit = layout.bus.bus_cell(col_id, i);
                cons.push(Constraint::terms(has_write, true, vec![(bit, F::ONE)]));
            }

            // If has_read, pack ram_addr from ra_bits.
            let mut terms = Vec::with_capacity(ram_lane.ra_bits.end - ram_lane.ra_bits.start + 1);
            terms.push((tr(l.ram_addr, i), F::ONE));
            let mut pow = F::ONE;
            for col_id in ram_lane.ra_bits.clone() {
                let bit = layout.bus.bus_cell(col_id, i);
                terms.push((bit, -pow));
                pow *= F::from_u64(2);
            }
            cons.push(Constraint::terms(has_read, false, terms));

            // If has_write, pack ram_addr from wa_bits.
            let mut terms = Vec::with_capacity(ram_lane.wa_bits.end - ram_lane.wa_bits.start + 1);
            terms.push((tr(l.ram_addr, i), F::ONE));
            let mut pow = F::ONE;
            for col_id in ram_lane.wa_bits.clone() {
                let bit = layout.bus.bus_cell(col_id, i);
                terms.push((bit, -pow));
                pow *= F::from_u64(2);
            }
            cons.push(Constraint::terms(has_write, false, terms));
        }
    }

    for i in 0..t.saturating_sub(1) {
        // pc_after[i] == pc_before[i+1]
        cons.push(lin_eq(tr(l.pc_after, i), tr(l.pc_before, i + 1)));

        // cycle[i+1] == cycle[i] + 1
        cons.push(Constraint::terms(
            one,
            false,
            vec![(tr(l.cycle, i + 1), F::ONE), (tr(l.cycle, i), -F::ONE), (one, -F::ONE)],
        ));

        // Once inactive, remain inactive: active[i+1] * (1 - active[i]) == 0
        cons.push(Constraint::terms(
            tr(l.active, i + 1),
            false,
            vec![(one, F::ONE), (tr(l.active, i), -F::ONE)],
        ));

        // Once halted, remain halted: halted[i] * (1 - halted[i+1]) == 0
        cons.push(Constraint::terms(
            tr(l.halted, i),
            false,
            vec![(one, F::ONE), (tr(l.halted, i + 1), -F::ONE)],
        ));
    }

    build_r1cs_ccs(&cons, cons.len(), layout.m, layout.const_one)
}
