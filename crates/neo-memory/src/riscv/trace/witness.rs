use neo_vm_trace::TwistOpKind;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use crate::riscv::exec_table::Rv32ExecTable;
use crate::riscv::lookups::{uninterleave_bits, RiscvOpcode, RiscvShoutTables};

use super::layout::Rv32TraceLayout;

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

        Ok(wit)
    }
}
