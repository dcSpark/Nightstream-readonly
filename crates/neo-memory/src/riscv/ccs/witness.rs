use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use neo_vm_trace::{StepTrace, TwistOpKind};

use crate::riscv::lookups::{RAM_ID, PROG_ID};

use super::constants::NEQ_TABLE_ID;
use super::Rv32B1Layout;

/// Build a CPU witness vector `z` (CPU region only; the bus tail is written by `R1csCpu`).
pub fn rv32_b1_chunk_to_witness(
    layout: Rv32B1Layout,
) -> Box<dyn Fn(&[StepTrace<u64, u64>]) -> Vec<F> + Send + Sync> {
    Box::new(move |chunk: &[StepTrace<u64, u64>]| {
        rv32_b1_chunk_to_witness_checked(&layout, chunk).unwrap_or_else(|e| {
            panic!("RV32 B1 witness build failed: {e}");
        })
    })
}

pub fn rv32_b1_chunk_to_witness_checked(layout: &Rv32B1Layout, chunk: &[StepTrace<u64, u64>]) -> Result<Vec<F>, String> {
    let mut z = vec![F::ZERO; layout.bus.bus_base];

    z[layout.const_one] = F::ONE;

    if let Some(first) = chunk.first() {
        z[layout.pc0] = F::from_u64(first.pc_before);
        for r in 0..32 {
            z[layout.regs0_start + r] = F::from_u64(first.regs_before[r]);
        }
        z[layout.regs0_start] = F::ZERO;
    }

    // Carry the final architectural state forward into padding rows so the final state is
    // unambiguous under L1-style fixed-row execution.
    let mut carried_pc = 0u64;
    let mut carried_regs = [0u64; 32];

    for j in 0..layout.chunk_size {
        if j >= chunk.len() {
            z[layout.is_active(j)] = F::ZERO;

            z[layout.pc_in(j)] = F::from_u64(carried_pc);
            z[layout.pc_out(j)] = F::from_u64(carried_pc);
            for r in 0..32 {
                z[layout.reg_in(r, j)] = F::from_u64(carried_regs[r]);
                z[layout.reg_out(r, j)] = F::from_u64(carried_regs[r]);
            }
            continue;
        }
        let step = &chunk[j];

        z[layout.is_active(j)] = F::ONE;
        z[layout.pc_in(j)] = F::from_u64(step.pc_before);
        z[layout.pc_out(j)] = F::from_u64(step.pc_after);

        // Registers.
        for r in 0..32 {
            z[layout.reg_in(r, j)] = F::from_u64(step.regs_before[r]);
            z[layout.reg_out(r, j)] = F::from_u64(step.regs_after[r]);
        }

        carried_pc = step.pc_after;
        carried_regs.copy_from_slice(&step.regs_after);
        carried_regs[0] = 0;

        // Instruction word: read from PROG_ID Twist event (commitment-bound source).
        let instr_word_u32 = step
            .twist_events
            .iter()
            .find(|ev| ev.twist_id == PROG_ID && ev.kind == TwistOpKind::Read)
            .ok_or_else(|| {
                format!(
                    "RV32 B1: missing PROG_ID read event at pc={:#x} (chunk j={j})",
                    step.pc_before
                )
            })
            .and_then(|ev| {
                u32::try_from(ev.value).map_err(|_| {
                    format!(
                        "RV32 B1: PROG_ID read value does not fit in u32 at pc={:#x}: value={:#x}",
                        step.pc_before, ev.value
                    )
                })
            })?;
        z[layout.instr_word(j)] = F::from_u64(instr_word_u32 as u64);

        // Bits.
        for i in 0..32 {
            z[layout.instr_bit(i, j)] = if ((instr_word_u32 >> i) & 1) == 1 { F::ONE } else { F::ZERO };
        }

        // Decode fields.
        let opcode = instr_word_u32 & 0x7f;
        let rd = (instr_word_u32 >> 7) & 0x1f;
        let funct3 = (instr_word_u32 >> 12) & 0x7;
        let rs1 = (instr_word_u32 >> 15) & 0x1f;
        let rs2 = (instr_word_u32 >> 20) & 0x1f;
        let funct7 = (instr_word_u32 >> 25) & 0x7f;

        z[layout.opcode(j)] = F::from_u64(opcode as u64);
        z[layout.funct3(j)] = F::from_u64(funct3 as u64);
        z[layout.funct7(j)] = F::from_u64(funct7 as u64);
        z[layout.rd_field(j)] = F::from_u64(rd as u64);
        z[layout.rs1_field(j)] = F::from_u64(rs1 as u64);
        z[layout.rs2_field(j)] = F::from_u64(rs2 as u64);

        // Helpers for immediate representations:
        // - `sx_u32` matches the CCS u32-style encoding used for imm_i / imm_s.
        // - `from_i32` matches the CCS signed encoding used for imm_b / imm_j.
        let sx_u32 = |x: i32| x as u32 as u64;
        let from_i32 = |v: i32| -> F {
            if v >= 0 {
                F::from_u64(v as u64)
            } else {
                -F::from_u64((-v) as u64)
            }
        };

        // Immediate raw fields.
        let imm12_raw = ((instr_word_u32 >> 20) & 0xfff) as u32;
        z[layout.imm12_raw(j)] = F::from_u64(imm12_raw as u64);

        // I-type immediate (sign-extended 12-bit).
        let imm_i = ((imm12_raw as i32) << 20) >> 20;
        z[layout.imm_i(j)] = F::from_u64(sx_u32(imm_i));

        // S-type immediate (sign-extended 12-bit).
        let imm_s_raw = (((instr_word_u32 >> 7) & 0x1f) | (((instr_word_u32 >> 25) & 0x7f) << 5)) as u32;
        let imm_s = ((imm_s_raw as i32) << 20) >> 20;
        z[layout.imm_s(j)] = F::from_u64(sx_u32(imm_s));

        // U-type immediate (upper 20 bits).
        let imm_u = (instr_word_u32 & 0xfffff000) as u64;
        z[layout.imm_u(j)] = F::from_u64(imm_u);

        // B-type immediate raw bits (before sign extension).
        let imm_b_raw = (((instr_word_u32 >> 7) & 0x1) << 11)
            | (((instr_word_u32 >> 8) & 0xf) << 1)
            | (((instr_word_u32 >> 25) & 0x3f) << 5)
            | (((instr_word_u32 >> 31) & 0x1) << 12);
        z[layout.imm_b_raw(j)] = F::from_u64(imm_b_raw as u64);
        let imm_b = ((imm_b_raw as i32) << 19) >> 19;
        z[layout.imm_b(j)] = from_i32(imm_b);

        // J-type immediate raw bits (before sign extension).
        let imm_j_raw = (((instr_word_u32 >> 21) & 0x3ff) << 1)
            | (((instr_word_u32 >> 20) & 0x1) << 11)
            | (((instr_word_u32 >> 12) & 0xff) << 12)
            | (((instr_word_u32 >> 31) & 0x1) << 20);
        z[layout.imm_j_raw(j)] = F::from_u64(imm_j_raw as u64);
        let imm_j = ((imm_j_raw as i32) << 11) >> 11;
        z[layout.imm_j(j)] = from_i32(imm_j);

        // Shift amount (always safe to fill).
        let shamt = ((instr_word_u32 >> 20) & 0x1f) as u64;
        z[layout.shamt(j)] = F::from_u64(shamt);

        // One-hot flags by opcode/funct fields.
        let is_add = opcode == 0x33 && funct3 == 0 && funct7 == 0;
        let is_sub = opcode == 0x33 && funct3 == 0 && funct7 == 0x20;
        let is_sll = opcode == 0x33 && funct3 == 1 && funct7 == 0;
        let is_slt = opcode == 0x33 && funct3 == 2 && funct7 == 0;
        let is_sltu = opcode == 0x33 && funct3 == 3 && funct7 == 0;
        let is_xor = opcode == 0x33 && funct3 == 4 && funct7 == 0;
        let is_srl = opcode == 0x33 && funct3 == 5 && funct7 == 0;
        let is_sra = opcode == 0x33 && funct3 == 5 && funct7 == 0x20;
        let is_or = opcode == 0x33 && funct3 == 6 && funct7 == 0;
        let is_and = opcode == 0x33 && funct3 == 7 && funct7 == 0;

        let is_addi = opcode == 0x13 && funct3 == 0;
        let is_slti = opcode == 0x13 && funct3 == 2;
        let is_sltiu = opcode == 0x13 && funct3 == 3;
        let is_xori = opcode == 0x13 && funct3 == 4;
        let is_ori = opcode == 0x13 && funct3 == 6;
        let is_andi = opcode == 0x13 && funct3 == 7;
        let is_slli = opcode == 0x13 && funct3 == 1 && funct7 == 0;
        let is_srli = opcode == 0x13 && funct3 == 5 && funct7 == 0;
        let is_srai = opcode == 0x13 && funct3 == 5 && funct7 == 0x20;

        let is_lw = opcode == 0x03 && funct3 == 2;
        let is_sw = opcode == 0x23 && funct3 == 2;
        let is_lui = opcode == 0x37;
        let is_auipc = opcode == 0x17;
        let is_beq = opcode == 0x63 && funct3 == 0;
        let is_bne = opcode == 0x63 && funct3 == 1;
        let is_blt = opcode == 0x63 && funct3 == 4;
        let is_bge = opcode == 0x63 && funct3 == 5;
        let is_bltu = opcode == 0x63 && funct3 == 6;
        let is_bgeu = opcode == 0x63 && funct3 == 7;
        let is_jal = opcode == 0x6f;
        let is_jalr = opcode == 0x67 && funct3 == 0;
        let is_halt = opcode == 0x73 && imm12_raw == 0;

        // Reject unsupported instructions.
        if !(is_add
            || is_sub
            || is_sll
            || is_slt
            || is_sltu
            || is_xor
            || is_srl
            || is_sra
            || is_or
            || is_and
            || is_addi
            || is_slti
            || is_sltiu
            || is_xori
            || is_ori
            || is_andi
            || is_slli
            || is_srli
            || is_srai
            || is_lw
            || is_sw
            || is_lui
            || is_auipc
            || is_beq
            || is_bne
            || is_blt
            || is_bge
            || is_bltu
            || is_bgeu
            || is_jal
            || is_jalr
            || is_halt)
        {
            return Err(format!(
                "RV32 B1: unsupported instruction at pc={:#x}: word={:#x}",
                step.pc_before, instr_word_u32
            ));
        }

        z[layout.is_add(j)] = if is_add { F::ONE } else { F::ZERO };
        z[layout.is_sub(j)] = if is_sub { F::ONE } else { F::ZERO };
        z[layout.is_sll(j)] = if is_sll { F::ONE } else { F::ZERO };
        z[layout.is_slt(j)] = if is_slt { F::ONE } else { F::ZERO };
        z[layout.is_sltu(j)] = if is_sltu { F::ONE } else { F::ZERO };
        z[layout.is_xor(j)] = if is_xor { F::ONE } else { F::ZERO };
        z[layout.is_srl(j)] = if is_srl { F::ONE } else { F::ZERO };
        z[layout.is_sra(j)] = if is_sra { F::ONE } else { F::ZERO };
        z[layout.is_or(j)] = if is_or { F::ONE } else { F::ZERO };
        z[layout.is_and(j)] = if is_and { F::ONE } else { F::ZERO };
        z[layout.is_addi(j)] = if is_addi { F::ONE } else { F::ZERO };
        z[layout.is_slti(j)] = if is_slti { F::ONE } else { F::ZERO };
        z[layout.is_sltiu(j)] = if is_sltiu { F::ONE } else { F::ZERO };
        z[layout.is_xori(j)] = if is_xori { F::ONE } else { F::ZERO };
        z[layout.is_ori(j)] = if is_ori { F::ONE } else { F::ZERO };
        z[layout.is_andi(j)] = if is_andi { F::ONE } else { F::ZERO };
        z[layout.is_slli(j)] = if is_slli { F::ONE } else { F::ZERO };
        z[layout.is_srli(j)] = if is_srli { F::ONE } else { F::ZERO };
        z[layout.is_srai(j)] = if is_srai { F::ONE } else { F::ZERO };
        z[layout.is_lw(j)] = if is_lw { F::ONE } else { F::ZERO };
        z[layout.is_sw(j)] = if is_sw { F::ONE } else { F::ZERO };
        z[layout.is_lui(j)] = if is_lui { F::ONE } else { F::ZERO };
        z[layout.is_auipc(j)] = if is_auipc { F::ONE } else { F::ZERO };
        z[layout.is_beq(j)] = if is_beq { F::ONE } else { F::ZERO };
        z[layout.is_bne(j)] = if is_bne { F::ONE } else { F::ZERO };
        z[layout.is_blt(j)] = if is_blt { F::ONE } else { F::ZERO };
        z[layout.is_bge(j)] = if is_bge { F::ONE } else { F::ZERO };
        z[layout.is_bltu(j)] = if is_bltu { F::ONE } else { F::ZERO };
        z[layout.is_bgeu(j)] = if is_bgeu { F::ONE } else { F::ZERO };
        z[layout.is_jal(j)] = if is_jal { F::ONE } else { F::ZERO };
        z[layout.is_jalr(j)] = if is_jalr { F::ONE } else { F::ZERO };
        z[layout.is_halt(j)] = if is_halt { F::ONE } else { F::ZERO };

        // One-hot register selectors.
        let rs1_idx = rs1 as usize;
        let rs2_idx = rs2 as usize;
        let rd_idx = rd as usize;
        z[layout.rs1_sel(rs1_idx, j)] = F::ONE;
        z[layout.rs2_sel(rs2_idx, j)] = F::ONE;

        // rd_sel: writes set rd_sel[rd] = 1, non-writes set rd_sel[0] = 1 (x0).
        let writes_rd = is_add
            || is_sub
            || is_sll
            || is_slt
            || is_sltu
            || is_xor
            || is_srl
            || is_sra
            || is_or
            || is_and
            || is_addi
            || is_slti
            || is_sltiu
            || is_xori
            || is_ori
            || is_andi
            || is_slli
            || is_srli
            || is_srai
            || is_lw
            || is_lui
            || is_auipc
            || is_jal
            || is_jalr;
        if writes_rd {
            z[layout.rd_sel(rd_idx, j)] = F::ONE;
        } else {
            z[layout.rd_sel(0, j)] = F::ONE;
        }

        // Selected operand values.
        z[layout.rs1_val(j)] = F::from_u64(step.regs_before[rs1_idx]);
        z[layout.rs2_val(j)] = F::from_u64(step.regs_before[rs2_idx]);

        // Shared-bus bound values: lookup_key / alu_out / mem_rv / eff_addr.
        z[layout.add_has_lookup(j)] = if is_add || is_addi || is_lw || is_sw || is_auipc || is_jalr {
            F::ONE
        } else {
            F::ZERO
        };
        z[layout.and_has_lookup(j)] = if is_and || is_andi { F::ONE } else { F::ZERO };
        z[layout.xor_has_lookup(j)] = if is_xor || is_xori { F::ONE } else { F::ZERO };
        z[layout.or_has_lookup(j)] = if is_or || is_ori { F::ONE } else { F::ZERO };
        z[layout.sll_has_lookup(j)] = if is_sll || is_slli { F::ONE } else { F::ZERO };
        z[layout.srl_has_lookup(j)] = if is_srl || is_srli { F::ONE } else { F::ZERO };
        z[layout.sra_has_lookup(j)] = if is_sra || is_srai { F::ONE } else { F::ZERO };
        z[layout.slt_has_lookup(j)] = if is_slt || is_slti || is_blt || is_bge { F::ONE } else { F::ZERO };
        z[layout.sltu_has_lookup(j)] = if is_sltu || is_sltiu || is_bltu || is_bgeu { F::ONE } else { F::ZERO };

        // Default zeros.
        z[layout.lookup_key(j)] = F::ZERO;
        z[layout.alu_out(j)] = F::ZERO;
        z[layout.add_a0b0(j)] = F::ZERO;
        z[layout.mem_rv(j)] = F::ZERO;
        z[layout.eff_addr(j)] = F::ZERO;
        z[layout.rd_write_val(j)] = F::ZERO;
        z[layout.br_taken(j)] = F::ZERO;
        z[layout.br_not_taken(j)] = F::ZERO;

        // Shout event (at most one per step in this circuit).
        let mut shout_event: Option<(u32, u64, u64)> = None;
        for ev in &step.shout_events {
            let id = ev.shout_id.0;
            if id <= NEQ_TABLE_ID {
                if shout_event.is_some() {
                    return Err(format!(
                        "RV32 B1: multiple shout events in one step (pc={:#x}, chunk j={j})",
                        step.pc_before
                    ));
                }
                shout_event = Some((id, ev.key, ev.value));
            } else {
                return Err(format!(
                    "RV32 B1: unsupported shout table id={id} at pc={:#x} (chunk j={j})",
                    step.pc_before
                ));
            }
        }
        if let Some((_id, key, value)) = shout_event {
            z[layout.lookup_key(j)] = F::from_u64(key);
            z[layout.alu_out(j)] = F::from_u64(value);
            if is_add || is_addi || is_lw || is_sw || is_auipc || is_jalr {
                let a0 = key & 1;
                let b0 = (key >> 1) & 1;
                z[layout.add_a0b0(j)] = F::from_u64(a0 & b0);
            }
        }

        // RAM read/write events if present.
        for ev in &step.twist_events {
            if ev.twist_id == RAM_ID {
                match ev.kind {
                    TwistOpKind::Read => {
                        z[layout.eff_addr(j)] = F::from_u64(ev.addr);
                        z[layout.mem_rv(j)] = F::from_u64(ev.value);
                    }
                    TwistOpKind::Write => {
                        z[layout.eff_addr(j)] = F::from_u64(ev.addr);
                    }
                }
            }
        }

        // Writeback value.
        if is_add
            || is_sub
            || is_sll
            || is_slt
            || is_sltu
            || is_xor
            || is_srl
            || is_sra
            || is_or
            || is_and
            || is_addi
            || is_slti
            || is_sltiu
            || is_xori
            || is_ori
            || is_andi
            || is_slli
            || is_srli
            || is_srai
            || is_auipc
        {
            z[layout.rd_write_val(j)] = z[layout.alu_out(j)];
        }
        if is_lw {
            z[layout.rd_write_val(j)] = z[layout.mem_rv(j)];
        }
        if is_lui {
            z[layout.rd_write_val(j)] = z[layout.imm_u(j)];
        }
        if is_jal || is_jalr {
            z[layout.rd_write_val(j)] = F::from_u64(step.pc_before.wrapping_add(4));
        }
        if is_beq || is_bne || is_blt || is_bltu {
            z[layout.br_taken(j)] = z[layout.alu_out(j)];
            z[layout.br_not_taken(j)] = F::ONE - z[layout.br_taken(j)];
        }
        if is_bge || is_bgeu {
            z[layout.br_taken(j)] = F::ONE - z[layout.alu_out(j)];
            z[layout.br_not_taken(j)] = F::ONE - z[layout.br_taken(j)];
        }
    }

    z[layout.pc_final] = F::from_u64(carried_pc);
    for r in 0..32 {
        z[layout.regs_final_start + r] = F::from_u64(carried_regs[r]);
    }
    z[layout.regs_final_start] = F::ZERO;

    Ok(z)
}
