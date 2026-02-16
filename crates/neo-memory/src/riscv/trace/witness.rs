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
fn imm_b_from_word(instr_word: u32) -> u32 {
    let imm = (((instr_word >> 31) & 0x1) << 12)
        | (((instr_word >> 7) & 0x1) << 11)
        | (((instr_word >> 25) & 0x3f) << 5)
        | (((instr_word >> 8) & 0xf) << 1);
    sign_extend_to_u32(imm, 13)
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
            if !cols.active[i] {
                // Inactive rows stay quiescent; WB/WP sidecars enforce these zeros.
                wit.cols[layout.rd_is_zero_01][i] = F::ONE;
                wit.cols[layout.rd_is_zero_012][i] = F::ONE;
                wit.cols[layout.rd_is_zero_0123][i] = F::ONE;
                wit.cols[layout.rd_is_zero][i] = F::ONE;
                continue;
            }

            // Retained decode fields.
            wit.cols[layout.opcode][i] = F::from_u64(cols.opcode[i] as u64);
            wit.cols[layout.funct3][i] = F::from_u64(cols.funct3[i] as u64);

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

            let rs1_u64 = cols.rs1[i] as u64;
            for (k, &bit_col) in layout.rs1_bit.iter().enumerate() {
                wit.cols[bit_col][i] = F::from_u64((rs1_u64 >> k) & 1);
            }

            let rs2_u64 = cols.rs2[i] as u64;
            for (k, &bit_col) in layout.rs2_bit.iter().enumerate() {
                wit.cols[bit_col][i] = F::from_u64((rs2_u64 >> k) & 1);
            }

            let funct7_u64 = cols.funct7[i] as u64;
            for (k, &bit_col) in layout.funct7_bit.iter().enumerate() {
                wit.cols[bit_col][i] = F::from_u64((funct7_u64 >> k) & 1);
            }

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
                }
                (Some((ra, rv)), None) => {
                    wit.cols[layout.ram_addr][i] = F::from_u64(ra);
                    wit.cols[layout.ram_rv][i] = F::from_u64(rv);
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
            if !cols.active[i] {
                continue;
            }
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
