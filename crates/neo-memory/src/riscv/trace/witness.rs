use neo_vm_trace::TwistOpKind;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use crate::riscv::exec_table::Rv32ExecTable;
use crate::riscv::lookups::{uninterleave_bits, RiscvOpcode, RiscvShoutTables};

use super::layout::Rv32TraceLayout;

#[inline]
fn sign_extend_to_u32(value: u32, bits: u32) -> u32 {
    debug_assert!(bits > 0 && bits <= 32);
    let shift = 32 - bits;
    (((value << shift) as i32) >> shift) as u32
}

#[inline]
fn imm_i_from_word(instr_word: u32) -> u32 {
    sign_extend_to_u32((instr_word >> 20) & 0x0fff, 12)
}

#[inline]
fn imm_s_from_word(instr_word: u32) -> u32 {
    let imm = ((instr_word >> 7) & 0x1f) | (((instr_word >> 25) & 0x7f) << 5);
    sign_extend_to_u32(imm, 12)
}

#[inline]
fn imm_b_from_word(instr_word: u32) -> u32 {
    let imm = (((instr_word >> 31) & 0x1) << 12)
        | (((instr_word >> 7) & 0x1) << 11)
        | (((instr_word >> 25) & 0x3f) << 5)
        | (((instr_word >> 8) & 0xf) << 1);
    sign_extend_to_u32(imm, 13)
}

#[inline]
fn imm_j_from_word(instr_word: u32) -> u32 {
    let imm = (((instr_word >> 31) & 0x1) << 20)
        | (((instr_word >> 12) & 0xff) << 12)
        | (((instr_word >> 20) & 0x1) << 11)
        | (((instr_word >> 21) & 0x3ff) << 1);
    sign_extend_to_u32(imm, 21)
}

#[derive(Clone, Debug)]
pub struct Rv32TraceWitness {
    pub t: usize,
    /// Column-major: `cols[col][row]`.
    pub cols: Vec<Vec<F>>,
}

impl Rv32TraceWitness {
    pub fn new_zero(layout: &Rv32TraceLayout, t: usize) -> Self {
        Self {
            t,
            cols: vec![vec![F::ZERO; t]; layout.cols],
        }
    }

    pub fn from_exec_table(layout: &Rv32TraceLayout, exec: &Rv32ExecTable) -> Result<Self, String> {
        let cols = exec.to_columns();
        let t = cols.len();
        let mut wit = Self::new_zero(layout, t);

        for i in 0..t {
            wit.cols[layout.one][i] = F::ONE;

            // Control / fetch
            wit.cols[layout.active][i] = if cols.active[i] { F::ONE } else { F::ZERO };
            wit.cols[layout.halted][i] = if cols.halted[i] { F::ONE } else { F::ZERO };
            wit.cols[layout.cycle][i] = F::from_u64(cols.cycle[i]);
            wit.cols[layout.pc_before][i] = F::from_u64(cols.pc_before[i]);
            wit.cols[layout.pc_after][i] = F::from_u64(cols.pc_after[i]);
            wit.cols[layout.instr_word][i] = F::from_u64(cols.instr_word[i] as u64);

            // Decoded fields
            wit.cols[layout.opcode][i] = F::from_u64(cols.opcode[i] as u64);
            wit.cols[layout.funct3][i] = F::from_u64(cols.funct3[i] as u64);
            wit.cols[layout.funct7][i] = F::from_u64(cols.funct7[i] as u64);
            wit.cols[layout.rd][i] = F::from_u64(cols.rd[i] as u64);
            wit.cols[layout.rs1][i] = F::from_u64(cols.rs1[i] as u64);
            wit.cols[layout.rs2][i] = F::from_u64(cols.rs2[i] as u64);

            let instr_word = cols.instr_word[i];
            wit.cols[layout.imm_i][i] = F::from_u64(imm_i_from_word(instr_word) as u64);
            wit.cols[layout.imm_s][i] = F::from_u64(imm_s_from_word(instr_word) as u64);
            wit.cols[layout.imm_b][i] = F::from_u64(imm_b_from_word(instr_word) as u64);
            wit.cols[layout.imm_j][i] = F::from_u64(imm_j_from_word(instr_word) as u64);

            // Compact opcode-class one-hot.
            let opcode_u64 = cols.opcode[i] as u64;
            let is = |op: u64| if opcode_u64 == op { F::ONE } else { F::ZERO };
            wit.cols[layout.op_lui][i] = is(0x37);
            wit.cols[layout.op_auipc][i] = is(0x17);
            wit.cols[layout.op_jal][i] = is(0x6F);
            wit.cols[layout.op_jalr][i] = is(0x67);
            wit.cols[layout.op_branch][i] = is(0x63);
            wit.cols[layout.op_load][i] = is(0x03);
            wit.cols[layout.op_store][i] = is(0x23);
            wit.cols[layout.op_alu_imm][i] = is(0x13);
            wit.cols[layout.op_alu_reg][i] = is(0x33);
            wit.cols[layout.op_misc_mem][i] = is(0x0F);
            wit.cols[layout.op_system][i] = is(0x73);
            wit.cols[layout.op_amo][i] = is(0x2F);

            // PROG view
            wit.cols[layout.prog_addr][i] = F::from_u64(cols.prog_addr[i]);
            wit.cols[layout.prog_value][i] = F::from_u64(cols.prog_value[i]);

            // REG view
            wit.cols[layout.rs1_addr][i] = F::from_u64(cols.rs1_addr[i]);
            wit.cols[layout.rs1_val][i] = F::from_u64(cols.rs1_val[i]);
            wit.cols[layout.rs2_addr][i] = F::from_u64(cols.rs2_addr[i]);
            wit.cols[layout.rs2_val][i] = F::from_u64(cols.rs2_val[i]);
            wit.cols[layout.rd_has_write][i] = if cols.rd_has_write[i] { F::ONE } else { F::ZERO };
            wit.cols[layout.rd_addr][i] = F::from_u64(cols.rd_addr[i]);
            wit.cols[layout.rd_val][i] = F::from_u64(cols.rd_val[i]);

            // Class+write helper flags (for class-specific writeback semantics).
            let rd_has_write = wit.cols[layout.rd_has_write][i];
            wit.cols[layout.op_lui_write][i] = wit.cols[layout.op_lui][i] * rd_has_write;
            wit.cols[layout.op_auipc_write][i] = wit.cols[layout.op_auipc][i] * rd_has_write;
            wit.cols[layout.op_jal_write][i] = wit.cols[layout.op_jal][i] * rd_has_write;
            wit.cols[layout.op_jalr_write][i] = wit.cols[layout.op_jalr][i] * rd_has_write;
            wit.cols[layout.op_alu_imm_write][i] = wit.cols[layout.op_alu_imm][i] * rd_has_write;
            wit.cols[layout.op_alu_reg_write][i] = wit.cols[layout.op_alu_reg][i] * rd_has_write;

            // Load/store sub-op selectors from opcode+funct3.
            let funct3 = cols.funct3[i] as u64;
            let is_load = cols.opcode[i] as u64 == 0x03;
            let is_store = cols.opcode[i] as u64 == 0x23;
            let flag = |on: bool| if on { F::ONE } else { F::ZERO };
            let is_lb = is_load && funct3 == 0b000;
            let is_lh = is_load && funct3 == 0b001;
            let is_lw = is_load && funct3 == 0b010;
            let is_lbu = is_load && funct3 == 0b100;
            let is_lhu = is_load && funct3 == 0b101;
            let is_sb = is_store && funct3 == 0b000;
            let is_sh = is_store && funct3 == 0b001;
            let is_sw = is_store && funct3 == 0b010;
            wit.cols[layout.is_lb][i] = flag(is_lb);
            wit.cols[layout.is_lbu][i] = flag(is_lbu);
            wit.cols[layout.is_lh][i] = flag(is_lh);
            wit.cols[layout.is_lhu][i] = flag(is_lhu);
            wit.cols[layout.is_lw][i] = flag(is_lw);
            wit.cols[layout.is_sb][i] = flag(is_sb);
            wit.cols[layout.is_sh][i] = flag(is_sh);
            wit.cols[layout.is_sw][i] = flag(is_sw);
            wit.cols[layout.is_lb_write][i] = wit.cols[layout.is_lb][i] * rd_has_write;
            wit.cols[layout.is_lbu_write][i] = wit.cols[layout.is_lbu][i] * rd_has_write;
            wit.cols[layout.is_lh_write][i] = wit.cols[layout.is_lh][i] * rd_has_write;
            wit.cols[layout.is_lhu_write][i] = wit.cols[layout.is_lhu][i] * rd_has_write;
            wit.cols[layout.is_lw_write][i] = wit.cols[layout.is_lw][i] * rd_has_write;

            // rd bit plumbing
            let rd_u64 = cols.rd[i] as u64;
            let rd_b0 = ((rd_u64 >> 0) & 1) as u64;
            let rd_b1 = ((rd_u64 >> 1) & 1) as u64;
            let rd_b2 = ((rd_u64 >> 2) & 1) as u64;
            let rd_b3 = ((rd_u64 >> 3) & 1) as u64;
            let rd_b4 = ((rd_u64 >> 4) & 1) as u64;
            wit.cols[layout.rd_bit[0]][i] = F::from_u64(rd_b0);
            wit.cols[layout.rd_bit[1]][i] = F::from_u64(rd_b1);
            wit.cols[layout.rd_bit[2]][i] = F::from_u64(rd_b2);
            wit.cols[layout.rd_bit[3]][i] = F::from_u64(rd_b3);
            wit.cols[layout.rd_bit[4]][i] = F::from_u64(rd_b4);

            let funct3_u64 = cols.funct3[i] as u64;
            for (k, &bit_col) in layout.funct3_bit.iter().enumerate() {
                wit.cols[bit_col][i] = F::from_u64((funct3_u64 >> k) & 1);
            }
            let is_active = cols.active[i];
            for (k, &f3_col) in layout.funct3_is.iter().enumerate() {
                wit.cols[f3_col][i] = if is_active && funct3_u64 == k as u64 {
                    F::ONE
                } else {
                    F::ZERO
                };
            }

            let rs1_u64 = cols.rs1[i] as u64;
            for (k, &bit_col) in layout.rs1_bit.iter().enumerate() {
                wit.cols[bit_col][i] = F::from_u64((rs1_u64 >> k) & 1);
            }

            let rs2_u64 = cols.rs2[i] as u64;
            for (k, &bit_col) in layout.rs2_bit.iter().enumerate() {
                wit.cols[bit_col][i] = F::from_u64((rs2_u64 >> k) & 1);
            }

            let rs2_val_u64 = cols.rs2_val[i];
            wit.cols[layout.rs2_q16][i] = F::from_u64(rs2_val_u64 >> 16);
            for (k, &bit_col) in layout.rs2_low_bit.iter().enumerate() {
                wit.cols[bit_col][i] = F::from_u64((rs2_val_u64 >> k) & 1);
            }

            let funct7_u64 = cols.funct7[i] as u64;
            for (k, &bit_col) in layout.funct7_bit.iter().enumerate() {
                wit.cols[bit_col][i] = F::from_u64((funct7_u64 >> k) & 1);
            }
            let funct7_b5 = (funct7_u64 >> 5) & 1;
            let f3_is_0 = if is_active && funct3_u64 == 0 { 1 } else { 0 };
            let f3_is_5 = if is_active && funct3_u64 == 5 { 1 } else { 0 };
            wit.cols[layout.alu_reg_table_delta][i] = F::from_u64(funct7_b5 * (f3_is_0 + f3_is_5));
            wit.cols[layout.alu_imm_table_delta][i] = F::from_u64(funct7_b5 * f3_is_5);
            let shift_f3_sel = wit.cols[layout.funct3_is[1]][i] + wit.cols[layout.funct3_is[5]][i];
            wit.cols[layout.alu_imm_shift_rhs_delta][i] =
                shift_f3_sel * (wit.cols[layout.rs2][i] - wit.cols[layout.imm_i][i]);

            let one_minus_b0 = F::ONE - wit.cols[layout.rd_bit[0]][i];
            let one_minus_b1 = F::ONE - wit.cols[layout.rd_bit[1]][i];
            let one_minus_b2 = F::ONE - wit.cols[layout.rd_bit[2]][i];
            let one_minus_b3 = F::ONE - wit.cols[layout.rd_bit[3]][i];
            let one_minus_b4 = F::ONE - wit.cols[layout.rd_bit[4]][i];

            let rd_is_zero_01 = one_minus_b0 * one_minus_b1;
            let rd_is_zero_012 = rd_is_zero_01 * one_minus_b2;
            let rd_is_zero_0123 = rd_is_zero_012 * one_minus_b3;
            let rd_is_zero = rd_is_zero_0123 * one_minus_b4;

            wit.cols[layout.rd_is_zero_01][i] = rd_is_zero_01;
            wit.cols[layout.rd_is_zero_012][i] = rd_is_zero_012;
            wit.cols[layout.rd_is_zero_0123][i] = rd_is_zero_0123;
            wit.cols[layout.rd_is_zero][i] = rd_is_zero;

            // Helper columns default to zero; set class-specific values below.
            wit.cols[layout.branch_taken][i] = F::ZERO;
            wit.cols[layout.branch_invert_shout][i] = F::ZERO;
            wit.cols[layout.branch_taken_imm][i] = F::ZERO;
            wit.cols[layout.branch_f3b1_op][i] = F::ZERO;
            wit.cols[layout.branch_invert_shout_prod][i] = F::ZERO;
            wit.cols[layout.jalr_drop_bit[0]][i] = F::ZERO;
            wit.cols[layout.jalr_drop_bit[1]][i] = F::ZERO;
        }

        // Normalize RAM events per row: at most one read + one write.
        for (i, r) in exec.rows.iter().enumerate() {
            if !r.active {
                continue;
            }

            let mut read: Option<(u64, u64)> = None;
            let mut write: Option<(u64, u64)> = None;
            for e in &r.ram_events {
                match e.kind {
                    TwistOpKind::Read => {
                        if read.is_some() {
                            return Err(format!("multiple RAM reads in one cycle={}", r.cycle));
                        }
                        read = Some((e.addr, e.value));
                    }
                    TwistOpKind::Write => {
                        if write.is_some() {
                            return Err(format!("multiple RAM writes in one cycle={}", r.cycle));
                        }
                        write = Some((e.addr, e.value));
                    }
                }
            }

            wit.cols[layout.ram_has_read][i] = if read.is_some() { F::ONE } else { F::ZERO };
            wit.cols[layout.ram_has_write][i] = if write.is_some() { F::ONE } else { F::ZERO };

            match (read, write) {
                (Some((ra, rv)), Some((wa, wv))) => {
                    if ra != wa {
                        return Err(format!(
                            "RAM read/write addr mismatch in one cycle {}: ra={:#x} wa={:#x}",
                            r.cycle, ra, wa
                        ));
                    }
                    wit.cols[layout.ram_addr][i] = F::from_u64(ra);
                    wit.cols[layout.ram_rv][i] = F::from_u64(rv);
                    wit.cols[layout.ram_wv][i] = F::from_u64(wv);
                    wit.cols[layout.ram_rv_q16][i] = F::from_u64(rv >> 16);
                    for (k, &bit_col) in layout.ram_rv_low_bit.iter().enumerate() {
                        wit.cols[bit_col][i] = F::from_u64((rv >> k) & 1);
                    }
                }
                (Some((ra, rv)), None) => {
                    wit.cols[layout.ram_addr][i] = F::from_u64(ra);
                    wit.cols[layout.ram_rv][i] = F::from_u64(rv);
                    wit.cols[layout.ram_rv_q16][i] = F::from_u64(rv >> 16);
                    for (k, &bit_col) in layout.ram_rv_low_bit.iter().enumerate() {
                        wit.cols[bit_col][i] = F::from_u64((rv >> k) & 1);
                    }
                }
                (None, Some((wa, wv))) => {
                    wit.cols[layout.ram_addr][i] = F::from_u64(wa);
                    wit.cols[layout.ram_wv][i] = F::from_u64(wv);
                }
                (None, None) => {}
            }
        }

        // Normalize Shout events per row: at most one lookup event.
        for (i, r) in exec.rows.iter().enumerate() {
            if !r.active {
                continue;
            }
            match r.shout_events.as_slice() {
                [] => {}
                [ev] => {
                    wit.cols[layout.shout_has_lookup][i] = F::ONE;
                    wit.cols[layout.shout_val][i] = F::from_u64(ev.value);
                    wit.cols[layout.shout_table_id][i] = F::from_u64(ev.shout_id.0 as u64);
                    let table_idx = usize::try_from(ev.shout_id.0).map_err(|_| {
                        format!(
                            "Shout table id does not fit usize in one-lane trace view at cycle={}: table_id={}",
                            r.cycle, ev.shout_id.0
                        )
                    })?;
                    if table_idx >= layout.shout_table_has_lookup.len() {
                        return Err(format!(
                            "unsupported Shout table id in one-lane trace view at cycle={}: table_id={} (supported: 0..{})",
                            r.cycle,
                            ev.shout_id.0,
                            layout.shout_table_has_lookup.len() - 1
                        ));
                    }
                    wit.cols[layout.shout_table_has_lookup[table_idx]][i] = F::ONE;
                    let (lhs, rhs) = uninterleave_bits(ev.key as u128);
                    wit.cols[layout.shout_lhs][i] = F::from_u64(lhs);
                    // Canonicalize shift keys: RISC-V shifts use only the low 5 bits of `rhs`.
                    let rhs = if let Some(op) = RiscvShoutTables::new(/*xlen=*/ 32).id_to_opcode(ev.shout_id) {
                        if matches!(op, RiscvOpcode::Sll | RiscvOpcode::Srl | RiscvOpcode::Sra) {
                            rhs & 0x1F
                        } else {
                            rhs
                        }
                    } else {
                        rhs
                    };
                    wit.cols[layout.shout_rhs][i] = F::from_u64(rhs);
                }
                _ => {
                    return Err(format!(
                        "multiple Shout events in one cycle={} (fixed-lane trace view only supports 1)",
                        r.cycle
                    ));
                }
            }
        }

        // Branch/JALR semantic helpers.
        for i in 0..t {
            let opcode = cols.opcode[i] as u64;
            let funct3 = cols.funct3[i] as u64;
            let f3_b1 = (funct3 >> 1) & 1;
            let f3_b2 = (funct3 >> 2) & 1;
            wit.cols[layout.branch_f3b1_op][i] = F::from_u64(f3_b1 * f3_b2);

            if opcode == 0x63 {
                let invert = funct3 & 1;
                let shout_val = match exec.rows[i].shout_events.as_slice() {
                    [ev] => ev.value & 1,
                    _ => 0,
                };
                let taken = if invert == 1 { 1 - shout_val } else { shout_val };
                let imm_b = imm_b_from_word(cols.instr_word[i]) as u64;

                wit.cols[layout.branch_invert_shout][i] = F::from_u64(invert);
                wit.cols[layout.branch_taken][i] = F::from_u64(taken);
                wit.cols[layout.branch_taken_imm][i] = F::from_u64(if taken == 1 { imm_b } else { 0 });
                wit.cols[layout.branch_invert_shout_prod][i] = F::from_u64(invert * shout_val);
            }

            if opcode == 0x67 {
                let imm_i = imm_i_from_word(cols.instr_word[i]);
                let rs1 = cols.rs1_val[i] as u32;
                let sum = rs1.wrapping_add(imm_i);
                wit.cols[layout.jalr_drop_bit[0]][i] = F::from_u64((sum & 1) as u64);
                wit.cols[layout.jalr_drop_bit[1]][i] = F::from_u64(((sum >> 1) & 1) as u64);
            }
        }

        Ok(wit)
    }
}
