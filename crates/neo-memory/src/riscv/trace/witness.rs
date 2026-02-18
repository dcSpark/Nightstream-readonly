use neo_vm_trace::TwistOpKind;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use crate::riscv::exec_table::Rv32ExecTable;
use crate::riscv::lookups::{uninterleave_bits, RiscvOpcode, RiscvShoutTables};

use super::layout::Rv32TraceLayout;

#[inline]
fn sign_extend_to_u32(value: u32, bits: u32) -> u32 {
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
            if !cols.active[i] {
                // Inactive rows stay quiescent; WB/WP sidecars enforce these zeros.
                continue;
            }

            // REG view
            wit.cols[layout.rs1_addr][i] = F::from_u64(cols.rs1_addr[i]);
            wit.cols[layout.rs1_val][i] = F::from_u64(cols.rs1_val[i]);
            wit.cols[layout.rs2_addr][i] = F::from_u64(cols.rs2_addr[i]);
            wit.cols[layout.rs2_val][i] = F::from_u64(cols.rs2_val[i]);
            // Keep rd_addr aligned with decoded instruction field.
            // REG write enable is carried by decode lookup selectors, so on non-write rows
            // this address is don't-care for bus semantics.
            wit.cols[layout.rd_addr][i] = F::from_u64(cols.rd[i] as u64);
            wit.cols[layout.rd_val][i] = F::from_u64(cols.rd_val[i]);
            if cols.opcode[i] == 0x67 {
                let rs1 = cols.rs1_val[i] as u32;
                let imm_i = imm_i_from_word(cols.instr_word[i]);
                let drop = rs1.wrapping_add(imm_i) & 1;
                wit.cols[layout.jalr_drop_bit][i] = F::from_u64(drop as u64);
            } else if cols.opcode[i] == 0x6F {
                let pc = cols.pc_before[i] as u32;
                let imm_j = imm_j_from_word(cols.instr_word[i]);
                let sum = (pc as u64) + (imm_j as u64);
                let carry = if sum >= (1u64 << 32) { 1u64 } else { 0u64 };
                wit.cols[layout.jalr_drop_bit][i] = F::from_u64(carry);
            } else if cols.opcode[i] == 0x63 {
                let pc_before = cols.pc_before[i] as u32;
                let pc_after = cols.pc_after[i] as u32;
                let imm_b = imm_b_from_word(cols.instr_word[i]);
                let taken = pc_after != pc_before.wrapping_add(4);
                if taken {
                    let sum = (pc_before as u64) + (imm_b as u64);
                    let carry = if sum >= (1u64 << 32) { 1u64 } else { 0u64 };
                    wit.cols[layout.jalr_drop_bit][i] = F::from_u64(carry);
                }
            }
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

        // Normalize fixed-lane Shout view for the main trace.
        //
        // Shared-bus mode may carry auxiliary lookup families in addition to
        // opcode-backed Shout events. The fixed-lane CPU shout glue must only
        // bind to canonical RV32 opcode tables.
        let shout_tables = RiscvShoutTables::new(/*xlen=*/ 32);
        for (i, r) in exec.rows.iter().enumerate() {
            if !r.active {
                continue;
            }
            let primary = r
                .shout_events
                .iter()
                .find(|ev| shout_tables.id_to_opcode(ev.shout_id).is_some());

            if let Some(ev) = primary {
                wit.cols[layout.shout_has_lookup][i] = F::ONE;
                wit.cols[layout.shout_val][i] = F::from_u64(ev.value);
                let (lhs, rhs) = uninterleave_bits(ev.key as u128);
                wit.cols[layout.shout_lhs][i] = F::from_u64(lhs);
                // Canonicalize shift keys: RISC-V shifts use only the low 5 bits of `rhs`.
                let rhs = if let Some(op) = shout_tables.id_to_opcode(ev.shout_id) {
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
        }
        Ok(wit)
    }
}
