use neo_ccs::relations::CcsStructure;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use crate::riscv::exec_table::Rv32ExecTable;
use crate::riscv::trace::{Rv32TraceLayout, Rv32TraceWitness};

use super::constraint_builder::{build_r1cs_ccs, Constraint};

/// Fixed-width, time-in-rows trace CCS layout.
///
/// This is a Tier 2.1 trace CCS with fixed columns over time (`t` rows),
/// AIR-like wiring invariants, and a compact subset of ISA semantics guards.
/// It is not yet full RV32 B1 semantics parity.
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

fn push_tier21_value_semantics(
    cons: &mut Vec<Constraint<F>>,
    one: usize,
    tr: &impl Fn(usize, usize) -> usize,
    l: &Rv32TraceLayout,
    i: usize,
    active: usize,
    rd_has_write: usize,
    ram_has_read: usize,
    shout_has_lookup: usize,
) {
    let pow2 = |k: usize| F::from_u64(1u64 << k);
    let two16 = F::from_u64(1u64 << 16);
    let lb_sign_coeff = F::from_u64((1u64 << 32) - (1u64 << 7));
    let lh_sign_coeff = F::from_u64((1u64 << 32) - (1u64 << 15));
    let f3 = |k: usize| tr(l.funct3_is[k], i);

    // funct3 one-hot helpers: active -> exactly one; always pack to funct3.
    cons.push(Constraint::terms(
        active,
        false,
        vec![
            (f3(0), F::ONE),
            (f3(1), F::ONE),
            (f3(2), F::ONE),
            (f3(3), F::ONE),
            (f3(4), F::ONE),
            (f3(5), F::ONE),
            (f3(6), F::ONE),
            (f3(7), F::ONE),
            (one, -F::ONE),
        ],
    ));
    cons.push(Constraint::terms(
        one,
        false,
        vec![
            (tr(l.funct3, i), F::ONE),
            (f3(1), -F::from_u64(1)),
            (f3(2), -F::from_u64(2)),
            (f3(3), -F::from_u64(3)),
            (f3(4), -F::from_u64(4)),
            (f3(5), -F::from_u64(5)),
            (f3(6), -F::from_u64(6)),
            (f3(7), -F::from_u64(7)),
        ],
    ));

    // Low-bit decompositions used for subword load/store semantics.
    {
        let mut terms = vec![(tr(l.rs2_val, i), F::ONE), (tr(l.rs2_q16, i), -two16)];
        for (k, &bit_col) in l.rs2_low_bit.iter().enumerate() {
            terms.push((tr(bit_col, i), -pow2(k)));
        }
        cons.push(Constraint::terms(active, false, terms));
    }
    {
        let mut terms = vec![(tr(l.ram_rv, i), F::ONE), (tr(l.ram_rv_q16, i), -two16)];
        for (k, &bit_col) in l.ram_rv_low_bit.iter().enumerate() {
            terms.push((tr(bit_col, i), -pow2(k)));
        }
        cons.push(Constraint::terms(ram_has_read, false, terms));
    }
    cons.push(Constraint::terms(
        ram_has_read,
        true,
        vec![(tr(l.ram_rv_q16, i), F::ONE)],
    ));
    for &bit_col in &l.ram_rv_low_bit {
        cons.push(Constraint::terms(ram_has_read, true, vec![(tr(bit_col, i), F::ONE)]));
    }

    // Load/store sub-op decode.
    for &flag in &[l.is_lb, l.is_lbu, l.is_lh, l.is_lhu, l.is_lw] {
        cons.push(Constraint::terms(
            tr(flag, i),
            false,
            vec![(tr(flag, i), F::ONE), (tr(l.op_load, i), -F::ONE)],
        ));
    }
    cons.push(Constraint::terms(
        one,
        false,
        vec![
            (tr(l.is_lb, i), F::ONE),
            (tr(l.is_lbu, i), F::ONE),
            (tr(l.is_lh, i), F::ONE),
            (tr(l.is_lhu, i), F::ONE),
            (tr(l.is_lw, i), F::ONE),
            (tr(l.op_load, i), -F::ONE),
        ],
    ));
    cons.push(Constraint::terms(
        tr(l.op_load, i),
        false,
        vec![
            (tr(l.funct3, i), F::ONE),
            (tr(l.is_lbu, i), -F::from_u64(4)),
            (tr(l.is_lh, i), -F::from_u64(1)),
            (tr(l.is_lhu, i), -F::from_u64(5)),
            (tr(l.is_lw, i), -F::from_u64(2)),
        ],
    ));

    for &flag in &[l.is_sb, l.is_sh, l.is_sw] {
        cons.push(Constraint::terms(
            tr(flag, i),
            false,
            vec![(tr(flag, i), F::ONE), (tr(l.op_store, i), -F::ONE)],
        ));
    }
    cons.push(Constraint::terms(
        one,
        false,
        vec![
            (tr(l.is_sb, i), F::ONE),
            (tr(l.is_sh, i), F::ONE),
            (tr(l.is_sw, i), F::ONE),
            (tr(l.op_store, i), -F::ONE),
        ],
    ));
    cons.push(Constraint::terms(
        tr(l.op_store, i),
        false,
        vec![
            (tr(l.funct3, i), F::ONE),
            (tr(l.is_sh, i), -F::from_u64(1)),
            (tr(l.is_sw, i), -F::from_u64(2)),
        ],
    ));

    // Write gates for value-binding rules.
    cons.push(Constraint::mul(
        tr(l.op_alu_imm, i),
        rd_has_write,
        tr(l.op_alu_imm_write, i),
    ));
    cons.push(Constraint::mul(
        tr(l.op_alu_reg, i),
        rd_has_write,
        tr(l.op_alu_reg_write, i),
    ));
    cons.push(Constraint::mul(tr(l.is_lb, i), rd_has_write, tr(l.is_lb_write, i)));
    cons.push(Constraint::mul(tr(l.is_lbu, i), rd_has_write, tr(l.is_lbu_write, i)));
    cons.push(Constraint::mul(tr(l.is_lh, i), rd_has_write, tr(l.is_lh_write, i)));
    cons.push(Constraint::mul(tr(l.is_lhu, i), rd_has_write, tr(l.is_lhu_write, i)));
    cons.push(Constraint::mul(tr(l.is_lw, i), rd_has_write, tr(l.is_lw_write, i)));

    // ALU table-id deltas from funct7 bit5.
    cons.push(Constraint::terms(
        f3(0),
        false,
        vec![
            (tr(l.alu_reg_table_delta, i), F::ONE),
            (tr(l.funct7_bit[5], i), -F::ONE),
        ],
    ));
    cons.push(Constraint::terms(
        f3(5),
        false,
        vec![
            (tr(l.alu_reg_table_delta, i), F::ONE),
            (tr(l.funct7_bit[5], i), -F::ONE),
        ],
    ));
    for &k in &[1usize, 2, 3, 4, 6, 7] {
        cons.push(Constraint::terms(
            f3(k),
            false,
            vec![(tr(l.alu_reg_table_delta, i), F::ONE)],
        ));
    }
    cons.push(Constraint::terms(
        f3(5),
        false,
        vec![
            (tr(l.alu_imm_table_delta, i), F::ONE),
            (tr(l.funct7_bit[5], i), -F::ONE),
        ],
    ));
    for &k in &[0usize, 1, 2, 3, 4, 6, 7] {
        cons.push(Constraint::terms(
            f3(k),
            false,
            vec![(tr(l.alu_imm_table_delta, i), F::ONE)],
        ));
    }

    // Tier 2.1 scope lock: RV32I only in trace mode.
    cons.push(Constraint::terms(one, false, vec![(tr(l.op_amo, i), F::ONE)]));
    cons.push(Constraint::terms(
        tr(l.op_alu_reg, i),
        false,
        vec![(tr(l.funct7_bit[0], i), F::ONE)],
    ));

    // Shout lookup policy: required for ALU/BRANCH; forbidden elsewhere.
    cons.push(Constraint::terms(
        tr(l.op_alu_imm, i),
        false,
        vec![(shout_has_lookup, F::ONE), (one, -F::ONE)],
    ));
    cons.push(Constraint::terms(
        tr(l.op_alu_reg, i),
        false,
        vec![(shout_has_lookup, F::ONE), (one, -F::ONE)],
    ));
    cons.push(Constraint::terms(
        shout_has_lookup,
        true,
        vec![(tr(l.shout_table_id, i), F::ONE)],
    ));

    // ALU lookup binding.
    cons.push(Constraint::terms_or(
        &[tr(l.op_alu_imm, i), tr(l.op_alu_reg, i)],
        false,
        vec![(tr(l.shout_lhs, i), F::ONE), (tr(l.rs1_val, i), -F::ONE)],
    ));
    cons.push(Constraint::terms(
        tr(l.op_alu_imm, i),
        false,
        vec![(tr(l.shout_rhs, i), F::ONE), (tr(l.imm_i, i), -F::ONE)],
    ));
    cons.push(Constraint::terms(
        tr(l.op_alu_reg, i),
        false,
        vec![(tr(l.shout_rhs, i), F::ONE), (tr(l.rs2_val, i), -F::ONE)],
    ));
    cons.push(Constraint::terms(
        tr(l.op_alu_imm_write, i),
        false,
        vec![(tr(l.rd_val, i), F::ONE), (tr(l.shout_val, i), -F::ONE)],
    ));
    cons.push(Constraint::terms(
        tr(l.op_alu_reg_write, i),
        false,
        vec![(tr(l.rd_val, i), F::ONE), (tr(l.shout_val, i), -F::ONE)],
    ));

    // ALU table-id mapping.
    cons.push(Constraint::terms(
        tr(l.op_alu_reg, i),
        false,
        vec![
            (tr(l.shout_table_id, i), F::ONE),
            (f3(0), -F::from_u64(3)),
            (f3(1), -F::from_u64(7)),
            (f3(2), -F::from_u64(5)),
            (f3(3), -F::from_u64(6)),
            (f3(4), -F::from_u64(1)),
            (f3(5), -F::from_u64(8)),
            (f3(6), -F::from_u64(2)),
            (tr(l.alu_reg_table_delta, i), -F::ONE),
        ],
    ));
    cons.push(Constraint::terms(
        tr(l.op_alu_imm, i),
        false,
        vec![
            (tr(l.shout_table_id, i), F::ONE),
            (f3(0), -F::from_u64(3)),
            (f3(1), -F::from_u64(7)),
            (f3(2), -F::from_u64(5)),
            (f3(3), -F::from_u64(6)),
            (f3(4), -F::from_u64(1)),
            (f3(5), -F::from_u64(8)),
            (f3(6), -F::from_u64(2)),
            (tr(l.alu_imm_table_delta, i), -F::ONE),
        ],
    ));

    // Branch table-id mapping:
    // EQ=10 for BEQ/BNE, SLT=5 for BLT/BGE, SLTU=6 for BLTU/BGEU.
    cons.push(Constraint::terms(
        tr(l.op_branch, i),
        false,
        vec![
            (tr(l.shout_table_id, i), F::ONE),
            (tr(l.funct3_bit[2], i), F::from_u64(5)),
            (tr(l.branch_f3b1_op, i), -F::ONE),
            (one, -F::from_u64(10)),
        ],
    ));

    // Load value binding.
    cons.push(Constraint::terms(
        tr(l.is_lw_write, i),
        false,
        vec![(tr(l.rd_val, i), F::ONE), (tr(l.ram_rv, i), -F::ONE)],
    ));
    {
        let mut terms = vec![(tr(l.rd_val, i), F::ONE)];
        for (k, &bit_col) in l.ram_rv_low_bit.iter().enumerate().take(8) {
            let coeff = if k == 7 { lb_sign_coeff } else { pow2(k) };
            terms.push((tr(bit_col, i), -coeff));
        }
        cons.push(Constraint::terms(tr(l.is_lb_write, i), false, terms));
    }
    {
        let mut terms = vec![(tr(l.rd_val, i), F::ONE)];
        for (k, &bit_col) in l.ram_rv_low_bit.iter().enumerate().take(8) {
            terms.push((tr(bit_col, i), -pow2(k)));
        }
        cons.push(Constraint::terms(tr(l.is_lbu_write, i), false, terms));
    }
    {
        let mut terms = vec![(tr(l.rd_val, i), F::ONE)];
        for (k, &bit_col) in l.ram_rv_low_bit.iter().enumerate().take(16) {
            let coeff = if k == 15 { lh_sign_coeff } else { pow2(k) };
            terms.push((tr(bit_col, i), -coeff));
        }
        cons.push(Constraint::terms(tr(l.is_lh_write, i), false, terms));
    }
    {
        let mut terms = vec![(tr(l.rd_val, i), F::ONE)];
        for (k, &bit_col) in l.ram_rv_low_bit.iter().enumerate().take(16) {
            terms.push((tr(bit_col, i), -pow2(k)));
        }
        cons.push(Constraint::terms(tr(l.is_lhu_write, i), false, terms));
    }

    // Store value binding.
    cons.push(Constraint::terms(
        tr(l.is_sw, i),
        false,
        vec![(tr(l.ram_wv, i), F::ONE), (tr(l.rs2_val, i), -F::ONE)],
    ));
    {
        let mut terms = vec![(tr(l.ram_wv, i), F::ONE), (tr(l.ram_rv, i), -F::ONE)];
        for k in 0..8 {
            let coeff = pow2(k);
            terms.push((tr(l.ram_rv_low_bit[k], i), coeff));
            terms.push((tr(l.rs2_low_bit[k], i), -coeff));
        }
        cons.push(Constraint::terms(tr(l.is_sb, i), false, terms));
    }
    {
        let mut terms = vec![(tr(l.ram_wv, i), F::ONE), (tr(l.ram_rv, i), -F::ONE)];
        for k in 0..16 {
            let coeff = pow2(k);
            terms.push((tr(l.ram_rv_low_bit[k], i), coeff));
            terms.push((tr(l.rs2_low_bit[k], i), -coeff));
        }
        cons.push(Constraint::terms(tr(l.is_sh, i), false, terms));
    }
    cons.push(Constraint::terms(
        tr(l.is_sb, i),
        false,
        vec![(ram_has_read, F::ONE), (one, -F::ONE)],
    ));
    cons.push(Constraint::terms(
        tr(l.is_sh, i),
        false,
        vec![(ram_has_read, F::ONE), (one, -F::ONE)],
    ));
}

/// Build the base trace CCS (wiring invariants + partial ISA semantics guards).
pub fn build_rv32_trace_wiring_ccs(layout: &Rv32TraceCcsLayout) -> Result<CcsStructure<F>, String> {
    let one = layout.const_one;
    let t = layout.t;
    let tr = |c: usize, i: usize| -> usize { layout.cell(c, i) };
    let l = &layout.trace;
    let opcode_flags = [
        l.op_lui,
        l.op_auipc,
        l.op_jal,
        l.op_jalr,
        l.op_branch,
        l.op_load,
        l.op_store,
        l.op_alu_imm,
        l.op_alu_reg,
        l.op_misc_mem,
        l.op_system,
        l.op_amo,
    ];

    let bool01 = |x: usize| -> Constraint<F> {
        // x * (x - 1) = 0
        Constraint::terms(x, false, vec![(x, F::ONE), (one, -F::ONE)])
    };

    let signext_imm12 = F::from_u64((1u64 << 32) - (1u64 << 11));
    let signext_imm13 = F::from_u64((1u64 << 32) - (1u64 << 12));
    let signext_imm21 = F::from_u64((1u64 << 32) - (1u64 << 20));

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
        vec![(layout.pc_final, F::ONE), (tr(l.pc_after, t - 1), -F::ONE)],
    ));
    cons.push(Constraint::terms(
        one,
        false,
        vec![(layout.halted_in, F::ONE), (tr(l.halted, 0), -F::ONE)],
    ));
    cons.push(Constraint::terms(
        one,
        false,
        vec![(layout.halted_out, F::ONE), (tr(l.halted, t - 1), -F::ONE)],
    ));
    // Execution anchor: the first trace row must be active.
    cons.push(Constraint::terms(
        one,
        false,
        vec![(tr(l.active, 0), F::ONE), (one, -F::ONE)],
    ));

    for i in 0..t {
        let active = tr(l.active, i);
        let halted = tr(l.halted, i);
        let rd_has_write = tr(l.rd_has_write, i);
        let ram_has_read = tr(l.ram_has_read, i);
        let ram_has_write = tr(l.ram_has_write, i);
        let shout_has_lookup = tr(l.shout_has_lookup, i);

        // Canonical AIR-style one-column.
        cons.push(Constraint::terms(
            one,
            false,
            vec![(tr(l.one, i), F::ONE), (one, -F::ONE)],
        ));

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
        for &b in &l.funct3_bit {
            cons.push(bool01(tr(b, i)));
        }
        for &b in &l.rs1_bit {
            cons.push(bool01(tr(b, i)));
        }
        for &b in &l.rs2_bit {
            cons.push(bool01(tr(b, i)));
        }
        for &b in &l.funct7_bit {
            cons.push(bool01(tr(b, i)));
        }
        cons.push(bool01(tr(l.branch_taken, i)));
        cons.push(bool01(tr(l.branch_invert_shout, i)));
        cons.push(bool01(tr(l.branch_f3b1_op, i)));
        cons.push(bool01(tr(l.branch_invert_shout_prod, i)));
        cons.push(bool01(tr(l.jalr_drop_bit[0], i)));
        cons.push(bool01(tr(l.jalr_drop_bit[1], i)));
        for &f in &opcode_flags {
            cons.push(bool01(tr(f, i)));
        }
        for &f in &[
            l.is_lb,
            l.is_lbu,
            l.is_lh,
            l.is_lhu,
            l.is_lw,
            l.is_sb,
            l.is_sh,
            l.is_sw,
            l.op_lui_write,
            l.op_alu_imm_write,
            l.op_alu_reg_write,
            l.is_lb_write,
            l.is_lbu_write,
            l.is_lh_write,
            l.is_lhu_write,
            l.is_lw_write,
        ] {
            cons.push(bool01(tr(f, i)));
        }
        for &f in &l.funct3_is {
            cons.push(bool01(tr(f, i)));
        }
        for &b in &l.ram_rv_low_bit {
            cons.push(bool01(tr(b, i)));
        }
        for &b in &l.rs2_low_bit {
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
            l.op_lui,
            l.op_auipc,
            l.op_jal,
            l.op_jalr,
            l.op_branch,
            l.op_load,
            l.op_store,
            l.op_alu_imm,
            l.op_alu_reg,
            l.op_misc_mem,
            l.op_system,
            l.op_amo,
            l.op_lui_write,
            l.op_auipc_write,
            l.op_jal_write,
            l.op_jalr_write,
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
            l.shout_table_id,
            l.is_lb,
            l.is_lbu,
            l.is_lh,
            l.is_lhu,
            l.is_lw,
            l.is_sb,
            l.is_sh,
            l.is_sw,
            l.op_alu_imm_write,
            l.op_alu_reg_write,
            l.is_lb_write,
            l.is_lbu_write,
            l.is_lh_write,
            l.is_lhu_write,
            l.is_lw_write,
            l.funct3_is[0],
            l.funct3_is[1],
            l.funct3_is[2],
            l.funct3_is[3],
            l.funct3_is[4],
            l.funct3_is[5],
            l.funct3_is[6],
            l.funct3_is[7],
            l.alu_reg_table_delta,
            l.alu_imm_table_delta,
            l.ram_rv_q16,
            l.rs2_q16,
            l.ram_rv_low_bit[0],
            l.ram_rv_low_bit[1],
            l.ram_rv_low_bit[2],
            l.ram_rv_low_bit[3],
            l.ram_rv_low_bit[4],
            l.ram_rv_low_bit[5],
            l.ram_rv_low_bit[6],
            l.ram_rv_low_bit[7],
            l.ram_rv_low_bit[8],
            l.ram_rv_low_bit[9],
            l.ram_rv_low_bit[10],
            l.ram_rv_low_bit[11],
            l.ram_rv_low_bit[12],
            l.ram_rv_low_bit[13],
            l.ram_rv_low_bit[14],
            l.ram_rv_low_bit[15],
            l.rs2_low_bit[0],
            l.rs2_low_bit[1],
            l.rs2_low_bit[2],
            l.rs2_low_bit[3],
            l.rs2_low_bit[4],
            l.rs2_low_bit[5],
            l.rs2_low_bit[6],
            l.rs2_low_bit[7],
            l.rs2_low_bit[8],
            l.rs2_low_bit[9],
            l.rs2_low_bit[10],
            l.rs2_low_bit[11],
            l.rs2_low_bit[12],
            l.rs2_low_bit[13],
            l.rs2_low_bit[14],
            l.rs2_low_bit[15],
            l.rd_bit[0],
            l.rd_bit[1],
            l.rd_bit[2],
            l.rd_bit[3],
            l.rd_bit[4],
            l.funct3_bit[0],
            l.funct3_bit[1],
            l.funct3_bit[2],
            l.rs1_bit[0],
            l.rs1_bit[1],
            l.rs1_bit[2],
            l.rs1_bit[3],
            l.rs1_bit[4],
            l.rs2_bit[0],
            l.rs2_bit[1],
            l.rs2_bit[2],
            l.rs2_bit[3],
            l.rs2_bit[4],
            l.funct7_bit[0],
            l.funct7_bit[1],
            l.funct7_bit[2],
            l.funct7_bit[3],
            l.funct7_bit[4],
            l.funct7_bit[5],
            l.funct7_bit[6],
            l.imm_i,
            l.imm_s,
            l.imm_b,
            l.imm_j,
            l.branch_taken,
            l.branch_invert_shout,
            l.branch_taken_imm,
            l.branch_f3b1_op,
            l.branch_invert_shout_prod,
            l.jalr_drop_bit[0],
            l.jalr_drop_bit[1],
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

        // Field bit-packings.
        cons.push(Constraint::terms(
            one,
            false,
            vec![
                (tr(l.funct3, i), F::ONE),
                (tr(l.funct3_bit[0], i), -F::ONE),
                (tr(l.funct3_bit[1], i), -F::from_u64(2)),
                (tr(l.funct3_bit[2], i), -F::from_u64(4)),
            ],
        ));
        cons.push(Constraint::terms(
            one,
            false,
            vec![
                (tr(l.rs1, i), F::ONE),
                (tr(l.rs1_bit[0], i), -F::ONE),
                (tr(l.rs1_bit[1], i), -F::from_u64(2)),
                (tr(l.rs1_bit[2], i), -F::from_u64(4)),
                (tr(l.rs1_bit[3], i), -F::from_u64(8)),
                (tr(l.rs1_bit[4], i), -F::from_u64(16)),
            ],
        ));
        cons.push(Constraint::terms(
            one,
            false,
            vec![
                (tr(l.rs2, i), F::ONE),
                (tr(l.rs2_bit[0], i), -F::ONE),
                (tr(l.rs2_bit[1], i), -F::from_u64(2)),
                (tr(l.rs2_bit[2], i), -F::from_u64(4)),
                (tr(l.rs2_bit[3], i), -F::from_u64(8)),
                (tr(l.rs2_bit[4], i), -F::from_u64(16)),
            ],
        ));
        cons.push(Constraint::terms(
            one,
            false,
            vec![
                (tr(l.funct7, i), F::ONE),
                (tr(l.funct7_bit[0], i), -F::ONE),
                (tr(l.funct7_bit[1], i), -F::from_u64(2)),
                (tr(l.funct7_bit[2], i), -F::from_u64(4)),
                (tr(l.funct7_bit[3], i), -F::from_u64(8)),
                (tr(l.funct7_bit[4], i), -F::from_u64(16)),
                (tr(l.funct7_bit[5], i), -F::from_u64(32)),
                (tr(l.funct7_bit[6], i), -F::from_u64(64)),
            ],
        ));

        // Opcode-class one-hot on active rows.
        {
            let mut terms = vec![(active, -F::ONE)];
            for &f in &opcode_flags {
                terms.push((tr(f, i), F::ONE));
            }
            cons.push(Constraint::terms(one, false, terms));
        }

        // opcode must match opcode-class one-hot.
        cons.push(Constraint::terms(
            one,
            false,
            vec![
                (tr(l.opcode, i), F::ONE),
                (tr(l.op_lui, i), -F::from_u64(0x37)),
                (tr(l.op_auipc, i), -F::from_u64(0x17)),
                (tr(l.op_jal, i), -F::from_u64(0x6F)),
                (tr(l.op_jalr, i), -F::from_u64(0x67)),
                (tr(l.op_branch, i), -F::from_u64(0x63)),
                (tr(l.op_load, i), -F::from_u64(0x03)),
                (tr(l.op_store, i), -F::from_u64(0x23)),
                (tr(l.op_alu_imm, i), -F::from_u64(0x13)),
                (tr(l.op_alu_reg, i), -F::from_u64(0x33)),
                (tr(l.op_misc_mem, i), -F::from_u64(0x0F)),
                (tr(l.op_system, i), -F::from_u64(0x73)),
                (tr(l.op_amo, i), -F::from_u64(0x2F)),
            ],
        ));

        // Compact field packing back into instr_word.
        cons.push(Constraint::terms(
            one,
            false,
            vec![
                (tr(l.instr_word, i), F::ONE),
                (tr(l.opcode, i), -F::ONE),
                (tr(l.rd, i), -F::from_u64(1u64 << 7)),
                (tr(l.funct3, i), -F::from_u64(1u64 << 12)),
                (tr(l.rs1, i), -F::from_u64(1u64 << 15)),
                (tr(l.rs2, i), -F::from_u64(1u64 << 20)),
                (tr(l.funct7, i), -F::from_u64(1u64 << 25)),
            ],
        ));

        // Signed immediate reconstruction helpers from decoded instruction bits.
        //
        // imm_i[11:0] = instr[31:20], sign-extended to 32 bits.
        cons.push(Constraint::terms(
            one,
            false,
            vec![
                (tr(l.imm_i, i), F::ONE),
                (tr(l.rs2_bit[0], i), -F::ONE),
                (tr(l.rs2_bit[1], i), -F::from_u64(2)),
                (tr(l.rs2_bit[2], i), -F::from_u64(4)),
                (tr(l.rs2_bit[3], i), -F::from_u64(8)),
                (tr(l.rs2_bit[4], i), -F::from_u64(16)),
                (tr(l.funct7_bit[0], i), -F::from_u64(32)),
                (tr(l.funct7_bit[1], i), -F::from_u64(64)),
                (tr(l.funct7_bit[2], i), -F::from_u64(128)),
                (tr(l.funct7_bit[3], i), -F::from_u64(256)),
                (tr(l.funct7_bit[4], i), -F::from_u64(512)),
                (tr(l.funct7_bit[5], i), -F::from_u64(1024)),
                (tr(l.funct7_bit[6], i), -signext_imm12),
            ],
        ));

        // imm_s = {instr[31:25], instr[11:7]}, sign-extended.
        cons.push(Constraint::terms(
            one,
            false,
            vec![
                (tr(l.imm_s, i), F::ONE),
                (tr(l.rd_bit[0], i), -F::ONE),
                (tr(l.rd_bit[1], i), -F::from_u64(2)),
                (tr(l.rd_bit[2], i), -F::from_u64(4)),
                (tr(l.rd_bit[3], i), -F::from_u64(8)),
                (tr(l.rd_bit[4], i), -F::from_u64(16)),
                (tr(l.funct7_bit[0], i), -F::from_u64(32)),
                (tr(l.funct7_bit[1], i), -F::from_u64(64)),
                (tr(l.funct7_bit[2], i), -F::from_u64(128)),
                (tr(l.funct7_bit[3], i), -F::from_u64(256)),
                (tr(l.funct7_bit[4], i), -F::from_u64(512)),
                (tr(l.funct7_bit[5], i), -F::from_u64(1024)),
                (tr(l.funct7_bit[6], i), -signext_imm12),
            ],
        ));

        // imm_b = {instr[31], instr[7], instr[30:25], instr[11:8], 0}, sign-extended.
        cons.push(Constraint::terms(
            one,
            false,
            vec![
                (tr(l.imm_b, i), F::ONE),
                (tr(l.rd_bit[1], i), -F::from_u64(2)),
                (tr(l.rd_bit[2], i), -F::from_u64(4)),
                (tr(l.rd_bit[3], i), -F::from_u64(8)),
                (tr(l.rd_bit[4], i), -F::from_u64(16)),
                (tr(l.funct7_bit[0], i), -F::from_u64(32)),
                (tr(l.funct7_bit[1], i), -F::from_u64(64)),
                (tr(l.funct7_bit[2], i), -F::from_u64(128)),
                (tr(l.funct7_bit[3], i), -F::from_u64(256)),
                (tr(l.funct7_bit[4], i), -F::from_u64(512)),
                (tr(l.funct7_bit[5], i), -F::from_u64(1024)),
                (tr(l.rd_bit[0], i), -F::from_u64(2048)),
                (tr(l.funct7_bit[6], i), -signext_imm13),
            ],
        ));

        // imm_j = {instr[31], instr[19:12], instr[20], instr[30:21], 0}, sign-extended.
        cons.push(Constraint::terms(
            one,
            false,
            vec![
                (tr(l.imm_j, i), F::ONE),
                (tr(l.rs2_bit[1], i), -F::from_u64(2)),
                (tr(l.rs2_bit[2], i), -F::from_u64(4)),
                (tr(l.rs2_bit[3], i), -F::from_u64(8)),
                (tr(l.rs2_bit[4], i), -F::from_u64(16)),
                (tr(l.funct7_bit[0], i), -F::from_u64(32)),
                (tr(l.funct7_bit[1], i), -F::from_u64(64)),
                (tr(l.funct7_bit[2], i), -F::from_u64(128)),
                (tr(l.funct7_bit[3], i), -F::from_u64(256)),
                (tr(l.funct7_bit[4], i), -F::from_u64(512)),
                (tr(l.funct7_bit[5], i), -F::from_u64(1024)),
                (tr(l.rs2_bit[0], i), -F::from_u64(2048)),
                (tr(l.funct3_bit[0], i), -F::from_u64(4096)),
                (tr(l.funct3_bit[1], i), -F::from_u64(8192)),
                (tr(l.funct3_bit[2], i), -F::from_u64(16384)),
                (tr(l.rs1_bit[0], i), -F::from_u64(32768)),
                (tr(l.rs1_bit[1], i), -F::from_u64(65536)),
                (tr(l.rs1_bit[2], i), -F::from_u64(131072)),
                (tr(l.rs1_bit[3], i), -F::from_u64(262144)),
                (tr(l.rs1_bit[4], i), -F::from_u64(524288)),
                (tr(l.funct7_bit[6], i), -signext_imm21),
            ],
        ));

        // Branch helper products.
        cons.push(Constraint::mul(
            tr(l.funct3_bit[1], i),
            tr(l.funct3_bit[2], i),
            tr(l.branch_f3b1_op, i),
        ));
        cons.push(Constraint::mul(
            tr(l.branch_invert_shout, i),
            tr(l.shout_val, i),
            tr(l.branch_invert_shout_prod, i),
        ));
        cons.push(Constraint::mul(
            tr(l.branch_taken, i),
            tr(l.imm_b, i),
            tr(l.branch_taken_imm, i),
        ));

        // LUI semantics: rd_val = imm_u (imm_u occupies bits [31:12]) when rd_has_write=1.
        cons.push(Constraint::terms(
            tr(l.op_lui_write, i),
            false,
            vec![
                (tr(l.rd_val, i), F::ONE),
                (tr(l.funct3, i), -F::from_u64(1u64 << 12)),
                (tr(l.rs1, i), -F::from_u64(1u64 << 15)),
                (tr(l.rs2, i), -F::from_u64(1u64 << 20)),
                (tr(l.funct7, i), -F::from_u64(1u64 << 25)),
            ],
        ));

        // Straight-line PC rule for non-control rows: pc_after = pc_before + 4.
        // Control rows (JAL/JALR/BRANCH) are excluded.
        cons.push(Constraint::terms_or(
            &[
                tr(l.op_lui, i),
                tr(l.op_auipc, i),
                tr(l.op_load, i),
                tr(l.op_store, i),
                tr(l.op_alu_imm, i),
                tr(l.op_alu_reg, i),
                tr(l.op_misc_mem, i),
                tr(l.op_system, i),
                tr(l.op_amo, i),
            ],
            false,
            vec![
                (tr(l.pc_after, i), F::ONE),
                (tr(l.pc_before, i), -F::ONE),
                (one, -F::from_u64(4)),
            ],
        ));

        // JAL/JALR/BRANCH control-flow targets.
        cons.push(Constraint::terms(
            tr(l.op_jal, i),
            false,
            vec![
                (tr(l.pc_after, i), F::ONE),
                (tr(l.pc_before, i), -F::ONE),
                (tr(l.imm_j, i), -F::ONE),
            ],
        ));
        // JALR target uses 4-byte alignment in this VM profile:
        // pc_after + drop_bit0 + 2*drop_bit1 == rs1_val + imm_i
        //
        // Tier 2.1 trace policy lock: only already-4-byte-aligned JALR sums are
        // accepted in trace mode, so drop bits must be zero.
        cons.push(Constraint::terms(
            tr(l.op_jalr, i),
            false,
            vec![
                (tr(l.pc_after, i), F::ONE),
                (tr(l.jalr_drop_bit[0], i), F::ONE),
                (tr(l.jalr_drop_bit[1], i), F::from_u64(2)),
                (tr(l.rs1_val, i), -F::ONE),
                (tr(l.imm_i, i), -F::ONE),
            ],
        ));
        cons.push(Constraint::terms(
            tr(l.op_jalr, i),
            false,
            vec![(tr(l.jalr_drop_bit[0], i), F::ONE)],
        ));
        cons.push(Constraint::terms(
            tr(l.op_jalr, i),
            false,
            vec![(tr(l.jalr_drop_bit[1], i), F::ONE)],
        ));

        // Branch compare/taken semantics from funct3 and shout compare output.
        cons.push(Constraint::terms(
            tr(l.op_branch, i),
            false,
            vec![
                (tr(l.branch_invert_shout, i), F::ONE),
                (tr(l.funct3_bit[0], i), -F::ONE),
            ],
        ));
        // Valid branch funct3 set: disallow 010/011 via b1 <= b2.
        cons.push(Constraint::terms(
            tr(l.op_branch, i),
            false,
            vec![(tr(l.funct3_bit[1], i), F::ONE), (tr(l.branch_f3b1_op, i), -F::ONE)],
        ));
        cons.push(Constraint::terms(
            tr(l.op_branch, i),
            false,
            vec![(shout_has_lookup, F::ONE), (one, -F::ONE)],
        ));
        cons.push(Constraint::terms(
            tr(l.op_branch, i),
            false,
            vec![(tr(l.shout_lhs, i), F::ONE), (tr(l.rs1_val, i), -F::ONE)],
        ));
        cons.push(Constraint::terms(
            tr(l.op_branch, i),
            false,
            vec![(tr(l.shout_rhs, i), F::ONE), (tr(l.rs2_val, i), -F::ONE)],
        ));
        // taken = shout_val XOR branch_invert_shout.
        cons.push(Constraint::terms(
            tr(l.op_branch, i),
            false,
            vec![
                (tr(l.branch_taken, i), F::ONE),
                (tr(l.shout_val, i), -F::ONE),
                (tr(l.branch_invert_shout, i), -F::ONE),
                (tr(l.branch_invert_shout_prod, i), F::from_u64(2)),
            ],
        ));
        // pc_after = pc_before + 4 + branch_taken * (imm_b - 4).
        cons.push(Constraint::terms(
            tr(l.op_branch, i),
            false,
            vec![
                (tr(l.pc_after, i), F::ONE),
                (tr(l.pc_before, i), -F::ONE),
                (one, -F::from_u64(4)),
                (tr(l.branch_taken_imm, i), -F::ONE),
                (tr(l.branch_taken, i), F::from_u64(4)),
            ],
        ));

        // Non-branch rows must keep branch helper columns at 0.
        cons.push(Constraint::terms_or(
            &[
                tr(l.op_lui, i),
                tr(l.op_auipc, i),
                tr(l.op_jal, i),
                tr(l.op_jalr, i),
                tr(l.op_load, i),
                tr(l.op_store, i),
                tr(l.op_alu_imm, i),
                tr(l.op_alu_reg, i),
                tr(l.op_misc_mem, i),
                tr(l.op_system, i),
                tr(l.op_amo, i),
            ],
            false,
            vec![(tr(l.branch_taken, i), F::ONE)],
        ));
        cons.push(Constraint::terms_or(
            &[
                tr(l.op_lui, i),
                tr(l.op_auipc, i),
                tr(l.op_jal, i),
                tr(l.op_jalr, i),
                tr(l.op_load, i),
                tr(l.op_store, i),
                tr(l.op_alu_imm, i),
                tr(l.op_alu_reg, i),
                tr(l.op_misc_mem, i),
                tr(l.op_system, i),
                tr(l.op_amo, i),
            ],
            false,
            vec![(tr(l.branch_invert_shout, i), F::ONE)],
        ));
        cons.push(Constraint::terms_or(
            &[
                tr(l.op_lui, i),
                tr(l.op_auipc, i),
                tr(l.op_jal, i),
                tr(l.op_jalr, i),
                tr(l.op_load, i),
                tr(l.op_store, i),
                tr(l.op_alu_imm, i),
                tr(l.op_alu_reg, i),
                tr(l.op_misc_mem, i),
                tr(l.op_system, i),
                tr(l.op_amo, i),
            ],
            false,
            vec![(tr(l.branch_taken_imm, i), F::ONE)],
        ));
        cons.push(Constraint::terms_or(
            &[
                tr(l.op_lui, i),
                tr(l.op_auipc, i),
                tr(l.op_jal, i),
                tr(l.op_jalr, i),
                tr(l.op_load, i),
                tr(l.op_store, i),
                tr(l.op_alu_imm, i),
                tr(l.op_alu_reg, i),
                tr(l.op_misc_mem, i),
                tr(l.op_system, i),
                tr(l.op_amo, i),
            ],
            false,
            vec![(tr(l.branch_f3b1_op, i), F::ONE)],
        ));
        cons.push(Constraint::terms_or(
            &[
                tr(l.op_lui, i),
                tr(l.op_auipc, i),
                tr(l.op_jal, i),
                tr(l.op_jalr, i),
                tr(l.op_load, i),
                tr(l.op_store, i),
                tr(l.op_alu_imm, i),
                tr(l.op_alu_reg, i),
                tr(l.op_misc_mem, i),
                tr(l.op_system, i),
                tr(l.op_amo, i),
            ],
            false,
            vec![(tr(l.branch_invert_shout_prod, i), F::ONE)],
        ));
        cons.push(Constraint::terms_or(
            &[
                tr(l.op_lui, i),
                tr(l.op_auipc, i),
                tr(l.op_jal, i),
                tr(l.op_branch, i),
                tr(l.op_load, i),
                tr(l.op_store, i),
                tr(l.op_alu_imm, i),
                tr(l.op_alu_reg, i),
                tr(l.op_misc_mem, i),
                tr(l.op_system, i),
                tr(l.op_amo, i),
            ],
            false,
            vec![(tr(l.jalr_drop_bit[0], i), F::ONE)],
        ));
        cons.push(Constraint::terms_or(
            &[
                tr(l.op_lui, i),
                tr(l.op_auipc, i),
                tr(l.op_jal, i),
                tr(l.op_branch, i),
                tr(l.op_load, i),
                tr(l.op_store, i),
                tr(l.op_alu_imm, i),
                tr(l.op_alu_reg, i),
                tr(l.op_misc_mem, i),
                tr(l.op_system, i),
                tr(l.op_amo, i),
            ],
            false,
            vec![(tr(l.jalr_drop_bit[1], i), F::ONE)],
        ));

        // LOAD/STORE effective address semantics.
        cons.push(Constraint::terms(
            tr(l.op_load, i),
            false,
            vec![
                (tr(l.ram_addr, i), F::ONE),
                (tr(l.rs1_val, i), -F::ONE),
                (tr(l.imm_i, i), -F::ONE),
            ],
        ));
        cons.push(Constraint::terms(
            tr(l.op_store, i),
            false,
            vec![
                (tr(l.ram_addr, i), F::ONE),
                (tr(l.rs1_val, i), -F::ONE),
                (tr(l.imm_s, i), -F::ONE),
            ],
        ));

        // RAM class policy.
        // LOAD rows must read RAM; STORE rows must write RAM.
        cons.push(Constraint::terms(
            tr(l.op_load, i),
            false,
            vec![(ram_has_read, F::ONE), (one, -F::ONE)],
        ));
        cons.push(Constraint::terms(
            tr(l.op_store, i),
            false,
            vec![(ram_has_write, F::ONE), (one, -F::ONE)],
        ));
        // Non-memory rows must not touch RAM lanes.
        cons.push(Constraint::terms_or(
            &[
                tr(l.op_lui, i),
                tr(l.op_auipc, i),
                tr(l.op_jal, i),
                tr(l.op_jalr, i),
                tr(l.op_branch, i),
                tr(l.op_alu_imm, i),
                tr(l.op_alu_reg, i),
                tr(l.op_misc_mem, i),
                tr(l.op_system, i),
            ],
            false,
            vec![(ram_has_read, F::ONE)],
        ));
        cons.push(Constraint::terms_or(
            &[
                tr(l.op_lui, i),
                tr(l.op_auipc, i),
                tr(l.op_jal, i),
                tr(l.op_jalr, i),
                tr(l.op_branch, i),
                tr(l.op_alu_imm, i),
                tr(l.op_alu_reg, i),
                tr(l.op_misc_mem, i),
                tr(l.op_system, i),
            ],
            false,
            vec![(ram_has_write, F::ONE)],
        ));
        cons.push(Constraint::terms_or(
            &[
                tr(l.op_lui, i),
                tr(l.op_auipc, i),
                tr(l.op_jal, i),
                tr(l.op_jalr, i),
                tr(l.op_branch, i),
                tr(l.op_alu_imm, i),
                tr(l.op_alu_reg, i),
                tr(l.op_misc_mem, i),
                tr(l.op_system, i),
            ],
            false,
            vec![(tr(l.ram_addr, i), F::ONE)],
        ));

        // Non-writeback classes must not assert rd_has_write.
        cons.push(Constraint::terms_or(
            &[
                tr(l.op_branch, i),
                tr(l.op_store, i),
                tr(l.op_misc_mem, i),
                tr(l.op_system, i),
            ],
            false,
            vec![(rd_has_write, F::ONE)],
        ));

        push_tier21_value_semantics(
            &mut cons,
            one,
            &tr,
            l,
            i,
            active,
            rd_has_write,
            ram_has_read,
            shout_has_lookup,
        );

        // Bind class+write helper flags.
        cons.push(Constraint::mul(tr(l.op_lui, i), rd_has_write, tr(l.op_lui_write, i)));
        cons.push(Constraint::mul(
            tr(l.op_auipc, i),
            rd_has_write,
            tr(l.op_auipc_write, i),
        ));
        cons.push(Constraint::mul(tr(l.op_jal, i), rd_has_write, tr(l.op_jal_write, i)));
        cons.push(Constraint::mul(tr(l.op_jalr, i), rd_has_write, tr(l.op_jalr_write, i)));

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

        // On active rows, `halted` is exactly the SYSTEM opcode class bit.
        cons.push(Constraint::terms(
            active,
            false,
            vec![(halted, F::ONE), (tr(l.op_system, i), -F::ONE)],
        ));

        // Writeback-class policy:
        // for classes that produce an rd result, rd_has_write must be asserted unless rd==0.
        for &op_flag in &[
            l.op_lui,
            l.op_auipc,
            l.op_jal,
            l.op_jalr,
            l.op_load,
            l.op_alu_imm,
            l.op_alu_reg,
            l.op_amo,
        ] {
            cons.push(Constraint::terms(
                tr(op_flag, i),
                false,
                vec![(rd_has_write, F::ONE), (tr(l.rd_is_zero, i), F::ONE), (one, -F::ONE)],
            ));
        }

        // Class-specific writeback semantics (only when the row both belongs to the class
        // and actually writes a destination register).
        // AUIPC: rd = pc_before + imm_u.
        cons.push(Constraint::terms(
            tr(l.op_auipc_write, i),
            false,
            vec![
                (tr(l.rd_val, i), F::ONE),
                (tr(l.pc_before, i), -F::ONE),
                (tr(l.funct3, i), -F::from_u64(1u64 << 12)),
                (tr(l.rs1, i), -F::from_u64(1u64 << 15)),
                (tr(l.rs2, i), -F::from_u64(1u64 << 20)),
                (tr(l.funct7, i), -F::from_u64(1u64 << 25)),
            ],
        ));
        // JAL/JALR: rd = pc_before + 4 (link value).
        cons.push(Constraint::terms(
            tr(l.op_jal_write, i),
            false,
            vec![
                (tr(l.rd_val, i), F::ONE),
                (tr(l.pc_before, i), -F::ONE),
                (one, -F::from_u64(4)),
            ],
        ));
        cons.push(Constraint::terms(
            tr(l.op_jalr_write, i),
            false,
            vec![
                (tr(l.rd_val, i), F::ONE),
                (tr(l.pc_before, i), -F::ONE),
                (one, -F::from_u64(4)),
            ],
        ));

        // If rd_has_write==0, rd_addr and rd_val must be 0.
        cons.push(Constraint::terms(rd_has_write, true, vec![(tr(l.rd_addr, i), F::ONE)]));
        cons.push(Constraint::terms(rd_has_write, true, vec![(tr(l.rd_val, i), F::ONE)]));

        // RAM bus padding: (1 - flag) * value == 0.
        cons.push(Constraint::terms(ram_has_read, true, vec![(tr(l.ram_rv, i), F::ONE)]));
        cons.push(Constraint::terms(ram_has_write, true, vec![(tr(l.ram_wv, i), F::ONE)]));

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
            vec![(tr(l.prog_value, i), F::ONE), (tr(l.instr_word, i), -F::ONE)],
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
            vec![(tr(l.pc_after, i), F::ONE), (tr(l.pc_before, i + 1), -F::ONE)],
        ));

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

        // Halted tail quiescence:
        // once halted, the next row must be inactive and keep the same pc_after.
        cons.push(Constraint::terms(
            tr(l.halted, i),
            false,
            vec![(tr(l.active, i + 1), F::ONE)],
        ));
        cons.push(Constraint::terms(
            tr(l.halted, i),
            false,
            vec![(tr(l.pc_after, i), F::ONE), (tr(l.pc_after, i + 1), -F::ONE)],
        ));
    }

    build_r1cs_ccs(&cons, cons.len(), layout.m, layout.const_one)
}
