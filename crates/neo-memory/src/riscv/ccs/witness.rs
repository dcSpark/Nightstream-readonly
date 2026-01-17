use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;

use neo_vm_trace::{StepTrace, TwistOpKind};

use crate::riscv::lookups::{
    decode_instruction, BranchCondition, RiscvInstruction, RiscvMemOp, RiscvOpcode, JOLT_CYCLE_TRACK_ECALL_NUM,
    JOLT_PRINT_ECALL_NUM, RAM_ID, PROG_ID,
};

use super::constants::{
    ADD_TABLE_ID, AND_TABLE_ID, EQ_TABLE_ID, NEQ_TABLE_ID, OR_TABLE_ID, SLL_TABLE_ID, SLT_TABLE_ID, SLTU_TABLE_ID,
    SRA_TABLE_ID, SRL_TABLE_ID, SUB_TABLE_ID, XOR_TABLE_ID,
};
use super::Rv32B1Layout;

#[inline]
fn set_bus_cell<Ff: PrimeCharacteristicRing>(z: &mut [Ff], layout: &Rv32B1Layout, bus_col: usize, j: usize, val: Ff) {
    let col = layout.bus.bus_cell(bus_col, j);
    z[col] = val;
}

#[inline]
fn write_bus_u64_bits<Ff: PrimeCharacteristicRing>(
    z: &mut [Ff],
    layout: &Rv32B1Layout,
    start_bus_col: usize,
    len: usize,
    j: usize,
    mut value: u64,
) {
    assert!(
        len <= 64,
        "RV32 B1 witness: bus bit range too large for u64 writer (len={len})"
    );
    for k in 0..len {
        let bit = (value & 1) as u64;
        value >>= 1;
        set_bus_cell(
            z,
            layout,
            start_bus_col + k,
            j,
            if bit == 1 { Ff::ONE } else { Ff::ZERO },
        );
    }
}

fn set_ecall_helpers(
    z: &mut [F],
    layout: &Rv32B1Layout,
    j: usize,
    a0_u64: u64,
    is_halt: bool,
) -> Result<(), String> {
    let a0_u32 = u32::try_from(a0_u64).map_err(|_| format!("RV32 B1: a0 value does not fit in u32: {a0_u64}"))?;

    for bit in 0..32 {
        z[layout.ecall_a0_bit(bit, j)] = if ((a0_u32 >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
    }

    let cycle_const = JOLT_CYCLE_TRACK_ECALL_NUM;
    let print_const = JOLT_PRINT_ECALL_NUM;

    let mut cycle_prefix = if ((a0_u32 ^ cycle_const) & 1) == 0 { 1u32 } else { 0u32 };
    z[layout.ecall_cycle_prefix(0, j)] = if cycle_prefix == 1 { F::ONE } else { F::ZERO };
    for k in 1..31 {
        let bit_match = ((a0_u32 >> k) ^ (cycle_const >> k)) & 1;
        cycle_prefix &= 1u32 ^ bit_match;
        z[layout.ecall_cycle_prefix(k, j)] = if cycle_prefix == 1 { F::ONE } else { F::ZERO };
    }
    let cycle_match31 = (((a0_u32 >> 31) ^ (cycle_const >> 31)) & 1) == 0;
    let is_cycle = cycle_prefix == 1 && cycle_match31;
    z[layout.ecall_is_cycle(j)] = if is_cycle { F::ONE } else { F::ZERO };

    let mut print_prefix = if ((a0_u32 ^ print_const) & 1) == 0 { 1u32 } else { 0u32 };
    z[layout.ecall_print_prefix(0, j)] = if print_prefix == 1 { F::ONE } else { F::ZERO };
    for k in 1..31 {
        let bit_match = ((a0_u32 >> k) ^ (print_const >> k)) & 1;
        print_prefix &= 1u32 ^ bit_match;
        z[layout.ecall_print_prefix(k, j)] = if print_prefix == 1 { F::ONE } else { F::ZERO };
    }
    let print_match31 = (((a0_u32 >> 31) ^ (print_const >> 31)) & 1) == 0;
    let is_print = print_prefix == 1 && print_match31;
    z[layout.ecall_is_print(j)] = if is_print { F::ONE } else { F::ZERO };

    let ecall_halts = !(is_cycle || is_print);
    z[layout.ecall_halts(j)] = if ecall_halts { F::ONE } else { F::ZERO };
    z[layout.halt_effective(j)] = if is_halt && ecall_halts { F::ONE } else { F::ZERO };

    Ok(())
}

/// Build a CPU witness vector `z` for shared-bus mode.
///
/// In shared-bus mode, `R1csCpu` overwrites the reserved bus tail from `StepTrace` events, so this
/// witness builder leaves the bus region at its zero default and only populates CPU columns.
pub fn rv32_b1_chunk_to_witness(
    layout: Rv32B1Layout,
) -> Box<dyn Fn(&[StepTrace<u64, u64>]) -> Vec<F> + Send + Sync> {
    Box::new(move |chunk: &[StepTrace<u64, u64>]| {
        rv32_b1_chunk_to_witness_checked(&layout, chunk).unwrap_or_else(|e| {
            panic!("RV32 B1 witness build failed: {e}");
        })
    })
}

/// Build a full witness vector `z`, including the bus tail (standalone/debug/test use).
pub fn rv32_b1_chunk_to_full_witness(
    layout: Rv32B1Layout,
) -> Box<dyn Fn(&[StepTrace<u64, u64>]) -> Vec<F> + Send + Sync> {
    Box::new(move |chunk: &[StepTrace<u64, u64>]| {
        rv32_b1_chunk_to_full_witness_checked(&layout, chunk).unwrap_or_else(|e| {
            panic!("RV32 B1 full witness build failed: {e}");
        })
    })
}

pub fn rv32_b1_chunk_to_witness_checked(layout: &Rv32B1Layout, chunk: &[StepTrace<u64, u64>]) -> Result<Vec<F>, String> {
    rv32_b1_chunk_to_witness_internal(layout, chunk, /*fill_bus=*/ false)
}

pub fn rv32_b1_chunk_to_full_witness_checked(layout: &Rv32B1Layout, chunk: &[StepTrace<u64, u64>]) -> Result<Vec<F>, String> {
    rv32_b1_chunk_to_witness_internal(layout, chunk, /*fill_bus=*/ true)
}

fn rv32_b1_chunk_to_witness_internal(
    layout: &Rv32B1Layout,
    chunk: &[StepTrace<u64, u64>],
    fill_bus: bool,
) -> Result<Vec<F>, String> {
    let mut z = vec![F::ZERO; layout.m];

    z[layout.const_one] = F::ONE;

    let add_shout_idx = layout
        .shout_idx(ADD_TABLE_ID)
        .map_err(|e| format!("RV32 B1: {e}"))?;
    let add_lane = &layout.bus.shout_cols[add_shout_idx].lanes[0];
    let prog_lane = &layout.bus.twist_cols[layout.prog_twist_idx].lanes[0];
    let ram_lane = &layout.bus.twist_cols[layout.ram_twist_idx].lanes[0];

    // Carry the architectural state forward through padding rows.
    // Initialize from the chunk's start state so fully-inactive chunks are well-defined.
    let mut carried_pc = 0u64;
    let mut carried_regs = [0u64; 32];

    if let Some(first) = chunk.first() {
        z[layout.pc0] = F::from_u64(first.pc_before);
        for r in 0..32 {
            z[layout.regs0_start + r] = F::from_u64(first.regs_before[r]);
            carried_regs[r] = first.regs_before[r];
        }
        z[layout.regs0_start] = F::ZERO;
        carried_regs[0] = 0;
        carried_pc = first.pc_before;
    }

    for j in 0..layout.chunk_size {
        if j >= chunk.len() {
            z[layout.is_active(j)] = F::ZERO;

            z[layout.pc_in(j)] = F::from_u64(carried_pc);
            z[layout.pc_out(j)] = F::from_u64(carried_pc);
            for r in 0..32 {
                z[layout.reg_in(r, j)] = F::from_u64(carried_regs[r]);
                z[layout.reg_out(r, j)] = F::from_u64(carried_regs[r]);
            }
            // Columns constrained independently of `is_active` must be set consistently on padding rows.
            for bit in 0..32 {
                z[layout.rd_write_bit(bit, j)] = F::ZERO;
                z[layout.mem_rv_bit(bit, j)] = F::ZERO;
                z[layout.mul_lo_bit(bit, j)] = F::ZERO;
                z[layout.mul_hi_bit(bit, j)] = F::ZERO;
                z[layout.div_quot_bit(bit, j)] = F::ZERO;
                z[layout.div_rem_bit(bit, j)] = F::ZERO;
                z[layout.rs1_bit(bit, j)] = F::ZERO;
                z[layout.rs2_bit(bit, j)] = F::ZERO;
            }
            for bit in 0..2 {
                z[layout.mul_carry_bit(bit, j)] = F::ZERO;
            }
            for k in 0..31 {
                z[layout.rs2_zero_prefix(k, j)] = F::ONE;
            }
            for k in 0..31 {
                z[layout.mul_hi_prefix(k, j)] = F::ZERO;
            }
            z[layout.rs2_is_zero(j)] = F::ONE;
            z[layout.rs2_nonzero(j)] = F::ZERO;
            set_ecall_helpers(&mut z, layout, j, carried_regs[10], false)?;
            continue;
        }
        let step = &chunk[j];

        // A row is active iff it contains exactly one PROG_ID read (B1 instruction fetch).
        // Padding rows contain no Twist/Shout events and are treated as inactive.
        let mut prog_read: Option<(u64, u64)> = None;
        let mut ram_read: Option<(u64, u64)> = None;
        let mut ram_write: Option<(u64, u64)> = None;
        for ev in &step.twist_events {
            if ev.twist_id == PROG_ID {
                match ev.kind {
                    TwistOpKind::Read => {
                        if prog_read.replace((ev.addr, ev.value)).is_some() {
                            return Err(format!(
                                "RV32 B1: multiple PROG_ID reads in one step at pc={:#x} (chunk j={j})",
                                step.pc_before
                            ));
                        }
                    }
                    TwistOpKind::Write => {
                        return Err(format!(
                            "RV32 B1: unexpected PROG_ID write at pc={:#x} (chunk j={j})",
                            step.pc_before
                        ));
                    }
                }
            } else if ev.twist_id == RAM_ID {
                match ev.kind {
                    TwistOpKind::Read => {
                        if ram_read.replace((ev.addr, ev.value)).is_some() {
                            return Err(format!(
                                "RV32 B1: multiple RAM reads in one step at pc={:#x} (chunk j={j})",
                                step.pc_before
                            ));
                        }
                    }
                    TwistOpKind::Write => {
                        if ram_write.replace((ev.addr, ev.value)).is_some() {
                            return Err(format!(
                                "RV32 B1: multiple RAM writes in one step at pc={:#x} (chunk j={j})",
                                step.pc_before
                            ));
                        }
                    }
                }
            } else {
                return Err(format!(
                    "RV32 B1: unexpected twist_id={} at pc={:#x} (chunk j={j})",
                    ev.twist_id.0, step.pc_before
                ));
            }
        }

        if prog_read.is_none() {
            if !step.twist_events.is_empty() || !step.shout_events.is_empty() {
                return Err(format!(
                    "RV32 B1: non-empty events in inactive row at step cycle={} (chunk j={j})",
                    step.cycle
                ));
            }

            z[layout.is_active(j)] = F::ZERO;
            z[layout.pc_in(j)] = F::from_u64(carried_pc);
            z[layout.pc_out(j)] = F::from_u64(carried_pc);
            for r in 0..32 {
                z[layout.reg_in(r, j)] = F::from_u64(carried_regs[r]);
                z[layout.reg_out(r, j)] = F::from_u64(carried_regs[r]);
            }
            // Columns constrained independently of `is_active` must be set consistently on padding rows.
            for bit in 0..32 {
                z[layout.rd_write_bit(bit, j)] = F::ZERO;
                z[layout.mem_rv_bit(bit, j)] = F::ZERO;
                z[layout.mul_lo_bit(bit, j)] = F::ZERO;
                z[layout.mul_hi_bit(bit, j)] = F::ZERO;
                z[layout.div_quot_bit(bit, j)] = F::ZERO;
                z[layout.div_rem_bit(bit, j)] = F::ZERO;
                z[layout.rs1_bit(bit, j)] = F::ZERO;
                z[layout.rs2_bit(bit, j)] = F::ZERO;
            }
            for bit in 0..2 {
                z[layout.mul_carry_bit(bit, j)] = F::ZERO;
            }
            for k in 0..31 {
                z[layout.rs2_zero_prefix(k, j)] = F::ONE;
            }
            for k in 0..31 {
                z[layout.mul_hi_prefix(k, j)] = F::ZERO;
            }
            z[layout.rs2_is_zero(j)] = F::ONE;
            z[layout.rs2_nonzero(j)] = F::ZERO;
            set_ecall_helpers(&mut z, layout, j, carried_regs[10], false)?;
            continue;
        }

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
        let (prog_addr, prog_value) = prog_read.expect("checked prog_read is present");
        if prog_addr != step.pc_before {
            return Err(format!(
                "RV32 B1: PROG_ID read addr mismatch at pc={:#x} (chunk j={j}): read_addr={:#x}",
                step.pc_before, prog_addr
            ));
        }
        let instr_word_u32 = u32::try_from(prog_value).map_err(|_| {
            format!(
                "RV32 B1: PROG_ID read value does not fit in u32 at pc={:#x}: value={:#x}",
                step.pc_before, prog_value
            )
        })?;
        z[layout.instr_word(j)] = F::from_u64(instr_word_u32 as u64);
        if fill_bus {
            set_bus_cell(&mut z, layout, prog_lane.has_read, j, F::ONE);
            set_bus_cell(&mut z, layout, prog_lane.has_write, j, F::ZERO);
            write_bus_u64_bits(
                &mut z,
                layout,
                prog_lane.ra_bits.start,
                prog_lane.ra_bits.end - prog_lane.ra_bits.start,
                j,
                prog_addr,
            );
            set_bus_cell(&mut z, layout, prog_lane.rv, j, F::from_u64(prog_value));
            set_bus_cell(&mut z, layout, prog_lane.wv, j, F::ZERO);
            set_bus_cell(&mut z, layout, prog_lane.inc, j, F::ZERO);
        }

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

        // One-hot flags: use the shared decoder as the single source of truth.
        let decoded = decode_instruction(instr_word_u32)
            .map_err(|e| format!("RV32 B1: decode failed at pc={:#x}: {e}", step.pc_before))?;

        let mut is_add = false;
        let mut is_sub = false;
        let mut is_sll = false;
        let mut is_slt = false;
        let mut is_sltu = false;
        let mut is_xor = false;
        let mut is_srl = false;
        let mut is_sra = false;
        let mut is_or = false;
        let mut is_and = false;
        let mut is_mul = false;
        let mut is_mulh = false;
        let mut is_mulhu = false;
        let mut is_mulhsu = false;
        let mut is_div = false;
        let mut is_divu = false;
        let mut is_rem = false;
        let mut is_remu = false;

        let mut is_addi = false;
        let mut is_slti = false;
        let mut is_sltiu = false;
        let mut is_xori = false;
        let mut is_ori = false;
        let mut is_andi = false;
        let mut is_slli = false;
        let mut is_srli = false;
        let mut is_srai = false;

        let mut is_lb = false;
        let mut is_lbu = false;
        let mut is_lh = false;
        let mut is_lhu = false;
        let mut is_lw = false;
        let mut is_sb = false;
        let mut is_sh = false;
        let mut is_sw = false;
        let mut is_amoswap_w = false;
        let mut is_amoadd_w = false;
        let mut is_amoxor_w = false;
        let mut is_amoor_w = false;
        let mut is_amoand_w = false;
        let mut is_lui = false;
        let mut is_auipc = false;
        let mut is_beq = false;
        let mut is_bne = false;
        let mut is_blt = false;
        let mut is_bge = false;
        let mut is_bltu = false;
        let mut is_bgeu = false;
        let mut is_jal = false;
        let mut is_jalr = false;
        let mut is_fence = false;
        let mut is_halt = false;

        match decoded {
            RiscvInstruction::RAlu { op, .. } => match op {
                RiscvOpcode::Add => is_add = true,
                RiscvOpcode::Sub => is_sub = true,
                RiscvOpcode::Sll => is_sll = true,
                RiscvOpcode::Slt => is_slt = true,
                RiscvOpcode::Sltu => is_sltu = true,
                RiscvOpcode::Xor => is_xor = true,
                RiscvOpcode::Srl => is_srl = true,
                RiscvOpcode::Sra => is_sra = true,
                RiscvOpcode::Or => is_or = true,
                RiscvOpcode::And => is_and = true,
                RiscvOpcode::Mul => is_mul = true,
                RiscvOpcode::Mulh => is_mulh = true,
                RiscvOpcode::Mulhu => is_mulhu = true,
                RiscvOpcode::Mulhsu => is_mulhsu = true,
                RiscvOpcode::Div => is_div = true,
                RiscvOpcode::Divu => is_divu = true,
                RiscvOpcode::Rem => is_rem = true,
                RiscvOpcode::Remu => is_remu = true,
                _ => {}
            },
            RiscvInstruction::IAlu { op, .. } => match op {
                RiscvOpcode::Add => is_addi = true,
                RiscvOpcode::Slt => is_slti = true,
                RiscvOpcode::Sltu => is_sltiu = true,
                RiscvOpcode::Xor => is_xori = true,
                RiscvOpcode::Or => is_ori = true,
                RiscvOpcode::And => is_andi = true,
                RiscvOpcode::Sll => is_slli = true,
                RiscvOpcode::Srl => is_srli = true,
                RiscvOpcode::Sra => is_srai = true,
                _ => {}
            },
            RiscvInstruction::Load { op, .. } => match op {
                RiscvMemOp::Lb => is_lb = true,
                RiscvMemOp::Lbu => is_lbu = true,
                RiscvMemOp::Lh => is_lh = true,
                RiscvMemOp::Lhu => is_lhu = true,
                RiscvMemOp::Lw => is_lw = true,
                _ => {}
            },
            RiscvInstruction::Store { op, .. } => match op {
                RiscvMemOp::Sb => is_sb = true,
                RiscvMemOp::Sh => is_sh = true,
                RiscvMemOp::Sw => is_sw = true,
                _ => {}
            },
            RiscvInstruction::Amo { op, .. } => match op {
                RiscvMemOp::AmoswapW => is_amoswap_w = true,
                RiscvMemOp::AmoaddW => is_amoadd_w = true,
                RiscvMemOp::AmoxorW => is_amoxor_w = true,
                RiscvMemOp::AmoorW => is_amoor_w = true,
                RiscvMemOp::AmoandW => is_amoand_w = true,
                _ => {}
            },
            RiscvInstruction::Lui { .. } => is_lui = true,
            RiscvInstruction::Auipc { .. } => is_auipc = true,
            RiscvInstruction::Branch { cond, .. } => match cond {
                BranchCondition::Eq => is_beq = true,
                BranchCondition::Ne => is_bne = true,
                BranchCondition::Lt => is_blt = true,
                BranchCondition::Ge => is_bge = true,
                BranchCondition::Ltu => is_bltu = true,
                BranchCondition::Geu => is_bgeu = true,
            },
            RiscvInstruction::Jal { .. } => is_jal = true,
            RiscvInstruction::Jalr { .. } => is_jalr = true,
            RiscvInstruction::Fence { .. } => is_fence = true,
            RiscvInstruction::Halt => is_halt = true,
            _ => {}
        }

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
            || is_mul
            || is_mulh
            || is_mulhu
            || is_mulhsu
            || is_div
            || is_divu
            || is_rem
            || is_remu
            || is_addi
            || is_slti
            || is_sltiu
            || is_xori
            || is_ori
            || is_andi
            || is_slli
            || is_srli
            || is_srai
            || is_lb
            || is_lbu
            || is_lh
            || is_lhu
            || is_lw
            || is_sb
            || is_sh
            || is_sw
            || is_amoswap_w
            || is_amoadd_w
            || is_amoxor_w
            || is_amoor_w
            || is_amoand_w
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
            || is_fence
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
        z[layout.is_mul(j)] = if is_mul { F::ONE } else { F::ZERO };
        z[layout.is_mulh(j)] = if is_mulh { F::ONE } else { F::ZERO };
        z[layout.is_mulhu(j)] = if is_mulhu { F::ONE } else { F::ZERO };
        z[layout.is_mulhsu(j)] = if is_mulhsu { F::ONE } else { F::ZERO };
        z[layout.is_div(j)] = if is_div { F::ONE } else { F::ZERO };
        z[layout.is_divu(j)] = if is_divu { F::ONE } else { F::ZERO };
        z[layout.is_rem(j)] = if is_rem { F::ONE } else { F::ZERO };
        z[layout.is_remu(j)] = if is_remu { F::ONE } else { F::ZERO };
        z[layout.is_addi(j)] = if is_addi { F::ONE } else { F::ZERO };
        z[layout.is_slti(j)] = if is_slti { F::ONE } else { F::ZERO };
        z[layout.is_sltiu(j)] = if is_sltiu { F::ONE } else { F::ZERO };
        z[layout.is_xori(j)] = if is_xori { F::ONE } else { F::ZERO };
        z[layout.is_ori(j)] = if is_ori { F::ONE } else { F::ZERO };
        z[layout.is_andi(j)] = if is_andi { F::ONE } else { F::ZERO };
        z[layout.is_slli(j)] = if is_slli { F::ONE } else { F::ZERO };
        z[layout.is_srli(j)] = if is_srli { F::ONE } else { F::ZERO };
        z[layout.is_srai(j)] = if is_srai { F::ONE } else { F::ZERO };
        z[layout.is_lb(j)] = if is_lb { F::ONE } else { F::ZERO };
        z[layout.is_lbu(j)] = if is_lbu { F::ONE } else { F::ZERO };
        z[layout.is_lh(j)] = if is_lh { F::ONE } else { F::ZERO };
        z[layout.is_lhu(j)] = if is_lhu { F::ONE } else { F::ZERO };
        z[layout.is_lw(j)] = if is_lw { F::ONE } else { F::ZERO };
        z[layout.is_sb(j)] = if is_sb { F::ONE } else { F::ZERO };
        z[layout.is_sh(j)] = if is_sh { F::ONE } else { F::ZERO };
        z[layout.is_sw(j)] = if is_sw { F::ONE } else { F::ZERO };
        z[layout.is_amoswap_w(j)] = if is_amoswap_w { F::ONE } else { F::ZERO };
        z[layout.is_amoadd_w(j)] = if is_amoadd_w { F::ONE } else { F::ZERO };
        z[layout.is_amoxor_w(j)] = if is_amoxor_w { F::ONE } else { F::ZERO };
        z[layout.is_amoor_w(j)] = if is_amoor_w { F::ONE } else { F::ZERO };
        z[layout.is_amoand_w(j)] = if is_amoand_w { F::ONE } else { F::ZERO };
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
        z[layout.is_fence(j)] = if is_fence { F::ONE } else { F::ZERO };
        z[layout.is_halt(j)] = if is_halt { F::ONE } else { F::ZERO };
        set_ecall_helpers(&mut z, layout, j, step.regs_before[10], is_halt)?;

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
            || is_mul
            || is_mulh
            || is_mulhu
            || is_mulhsu
            || is_div
            || is_divu
            || is_rem
            || is_remu
            || is_addi
            || is_slti
            || is_sltiu
            || is_xori
            || is_ori
            || is_andi
            || is_slli
            || is_srli
            || is_srai
            || is_lb
            || is_lbu
            || is_lh
            || is_lhu
            || is_lw
            || is_amoswap_w
            || is_amoadd_w
            || is_amoxor_w
            || is_amoor_w
            || is_amoand_w
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
        let rs1_u32 = u32::try_from(step.regs_before[rs1_idx])
            .map_err(|_| format!("RV32 B1: rs1 value does not fit in u32 at pc={:#x}", step.pc_before))?;
        let rs2_u32 = u32::try_from(step.regs_before[rs2_idx])
            .map_err(|_| format!("RV32 B1: rs2 value does not fit in u32 at pc={:#x}", step.pc_before))?;
        let rs1_u64 = rs1_u32 as u64;
        let rs2_u64 = rs2_u32 as u64;
        z[layout.rs1_val(j)] = F::from_u64(rs1_u64);
        z[layout.rs2_val(j)] = F::from_u64(rs2_u64);

        // Helpers used by in-circuit RV32M constraints.
        let rs1_sign = (rs1_u32 >> 31) & 1;
        let rs2_sign = (rs2_u32 >> 31) & 1;
        let rs1_abs = if rs1_sign == 0 { rs1_u64 } else { (1u64 << 32) - rs1_u64 };
        let rs2_abs = if rs2_sign == 0 { rs2_u64 } else { (1u64 << 32) - rs2_u64 };
        z[layout.rs1_abs(j)] = F::from_u64(rs1_abs);
        z[layout.rs2_abs(j)] = F::from_u64(rs2_abs);
        z[layout.rs1_rs2_sign_and(j)] = F::from_u64((rs1_sign & rs2_sign) as u64);
        z[layout.rs1_sign_rs2_val(j)] = F::from_u64((rs1_sign as u64) * rs2_u64);
        z[layout.rs2_sign_rs1_val(j)] = F::from_u64((rs2_sign as u64) * rs1_u64);

        // MUL product (always computed; constraints use it even when `is_mul=0`).
        let mul_prod = (rs1_u64 as u128) * (rs2_u64 as u128);
        let mul_lo = (mul_prod & 0xffff_ffff) as u64;
        let mul_hi = ((mul_prod >> 32) & 0xffff_ffff) as u64;
        z[layout.mul_lo(j)] = F::from_u64(mul_lo);
        z[layout.mul_hi(j)] = F::from_u64(mul_hi);

        // Bit decomposition for rs1/rs2 (used for sign, abs, and zero checks).
        for bit in 0..32 {
            z[layout.rs1_bit(bit, j)] = if ((rs1_u32 >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
            z[layout.rs2_bit(bit, j)] = if ((rs2_u32 >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
        }
        let mut prefix = F::ONE - z[layout.rs2_bit(0, j)];
        z[layout.rs2_zero_prefix(0, j)] = prefix;
        for k in 1..31 {
            prefix *= F::ONE - z[layout.rs2_bit(k, j)];
            z[layout.rs2_zero_prefix(k, j)] = prefix;
        }
        z[layout.rs2_is_zero(j)] = prefix * (F::ONE - z[layout.rs2_bit(31, j)]);
        z[layout.rs2_nonzero(j)] = F::ONE - z[layout.rs2_is_zero(j)];

        // DIV/REM helpers (unsigned + signed): quotient/remainder decomposition and remainder < divisor check.
        let is_divu_or_remu = is_divu || is_remu;
        let is_div_or_rem = is_div || is_rem;
        let rs2_is_zero = rs2_u32 == 0;
        z[layout.is_divu_or_remu(j)] = if is_divu_or_remu { F::ONE } else { F::ZERO };
        z[layout.is_div_or_rem(j)] = if is_div_or_rem { F::ONE } else { F::ZERO };

        let do_rem_check = is_divu_or_remu && !rs2_is_zero;
        let do_rem_check_signed = is_div_or_rem && !rs2_is_zero;
        z[layout.div_rem_check(j)] = if do_rem_check { F::ONE } else { F::ZERO };
        z[layout.div_rem_check_signed(j)] = if do_rem_check_signed { F::ONE } else { F::ZERO };
        z[layout.divu_by_zero(j)] = if is_divu && rs2_is_zero { F::ONE } else { F::ZERO };
        z[layout.div_by_zero(j)] = if is_div && rs2_is_zero { F::ONE } else { F::ZERO };
        z[layout.div_nonzero(j)] = if is_div && !rs2_is_zero { F::ONE } else { F::ZERO };
        z[layout.rem_by_zero(j)] = if is_rem && rs2_is_zero { F::ONE } else { F::ZERO };
        z[layout.rem_nonzero(j)] = if is_rem && !rs2_is_zero { F::ONE } else { F::ZERO };

        let (div_quot, div_rem, div_divisor) = if is_divu_or_remu {
            if rs2_is_zero {
                (u32::MAX as u64, rs1_u64, rs2_u64)
            } else {
                (rs1_u64 / rs2_u64, rs1_u64 % rs2_u64, rs2_u64)
            }
        } else if is_div_or_rem {
            if rs2_is_zero {
                (0u64, rs1_abs, rs2_abs)
            } else {
                (rs1_abs / rs2_abs, rs1_abs % rs2_abs, rs2_abs)
            }
        } else {
            (0u64, 0u64, 0u64)
        };
        z[layout.div_quot(j)] = F::from_u64(div_quot);
        z[layout.div_rem(j)] = F::from_u64(div_rem);
        z[layout.div_divisor(j)] = F::from_u64(div_divisor);
        z[layout.div_prod(j)] = F::from_u64(((div_divisor as u128) * (div_quot as u128)) as u64);

        let div_sign = (rs1_sign ^ rs2_sign) as u64;
        z[layout.div_sign(j)] = F::from_u64(div_sign);
        let (div_quot_signed, div_quot_carry) = if div_sign == 0 {
            (div_quot, 0u64)
        } else if div_quot == 0 {
            (0u64, 1u64)
        } else {
            ((1u64 << 32) - div_quot, 0u64)
        };
        let (div_rem_signed, div_rem_carry) = if rs1_sign == 0 {
            (div_rem, 0u64)
        } else if div_rem == 0 {
            (0u64, 1u64)
        } else {
            ((1u64 << 32) - div_rem, 0u64)
        };
        z[layout.div_quot_signed(j)] = F::from_u64(div_quot_signed);
        z[layout.div_rem_signed(j)] = F::from_u64(div_rem_signed);
        z[layout.div_quot_carry(j)] = F::from_u64(div_quot_carry);
        z[layout.div_rem_carry(j)] = F::from_u64(div_rem_carry);

        // Shared-bus bound values: lookup_key / alu_out / mem_rv / eff_addr.
        let add_has_lookup = is_add
            || is_addi
            || is_lb
            || is_lbu
            || is_lh
            || is_lhu
            || is_lw
            || is_sb
            || is_sh
            || is_sw
            || is_amoadd_w
            || is_auipc
            || is_jalr;
        z[layout.add_has_lookup(j)] = if add_has_lookup { F::ONE } else { F::ZERO };
        let and_has_lookup = is_and || is_andi || is_amoand_w;
        z[layout.and_has_lookup(j)] = if and_has_lookup { F::ONE } else { F::ZERO };
        let xor_has_lookup = is_xor || is_xori || is_amoxor_w;
        z[layout.xor_has_lookup(j)] = if xor_has_lookup { F::ONE } else { F::ZERO };
        let or_has_lookup = is_or || is_ori || is_amoor_w;
        z[layout.or_has_lookup(j)] = if or_has_lookup { F::ONE } else { F::ZERO };
        let sll_has_lookup = is_sll || is_slli;
        z[layout.sll_has_lookup(j)] = if sll_has_lookup { F::ONE } else { F::ZERO };
        let srl_has_lookup = is_srl || is_srli;
        z[layout.srl_has_lookup(j)] = if srl_has_lookup { F::ONE } else { F::ZERO };
        let sra_has_lookup = is_sra || is_srai;
        z[layout.sra_has_lookup(j)] = if sra_has_lookup { F::ONE } else { F::ZERO };
        let slt_has_lookup = is_slt || is_slti || is_blt || is_bge;
        z[layout.slt_has_lookup(j)] = if slt_has_lookup { F::ONE } else { F::ZERO };
        let sltu_has_lookup = is_sltu || is_sltiu || is_bltu || is_bgeu || do_rem_check || do_rem_check_signed;
        z[layout.sltu_has_lookup(j)] = if sltu_has_lookup { F::ONE } else { F::ZERO };

        let is_amo = is_amoswap_w || is_amoadd_w || is_amoxor_w || is_amoor_w || is_amoand_w;
        let ram_has_read = is_lb || is_lbu || is_lh || is_lhu || is_lw || is_sb || is_sh || is_amo;
        let ram_has_write = is_sb || is_sh || is_sw || is_amo;
        z[layout.ram_has_read(j)] = if ram_has_read { F::ONE } else { F::ZERO };
        z[layout.ram_has_write(j)] = if ram_has_write { F::ONE } else { F::ZERO };

        // Default zeros.
        z[layout.lookup_key(j)] = F::ZERO;
        z[layout.alu_out(j)] = F::ZERO;
        z[layout.add_a0b0(j)] = F::ZERO;
        z[layout.mem_rv(j)] = F::ZERO;
        z[layout.eff_addr(j)] = F::ZERO;
        z[layout.ram_wv(j)] = F::ZERO;
        z[layout.rd_write_val(j)] = F::ZERO;
        z[layout.br_taken(j)] = F::ZERO;
        z[layout.br_not_taken(j)] = F::ZERO;

        // RAM events: validate shape and fill the RAM twist lane + CPU mirrors.
        let is_load = is_lb || is_lbu || is_lh || is_lhu || is_lw;
        let is_store_rmw = is_sb || is_sh;
        if is_load {
            if ram_read.is_none() || ram_write.is_some() {
                return Err(format!(
                    "RV32 B1: load expects one RAM read and no RAM write at pc={:#x} (chunk j={j})",
                    step.pc_before
                ));
            }
        } else if is_store_rmw {
            if ram_read.is_none() || ram_write.is_none() {
                return Err(format!(
                    "RV32 B1: byte/half store expects one RAM read and one RAM write at pc={:#x} (chunk j={j})",
                    step.pc_before
                ));
            }
        } else if is_sw {
            if ram_read.is_some() || ram_write.is_none() {
                return Err(format!(
                    "RV32 B1: SW expects one RAM write and no RAM read at pc={:#x} (chunk j={j})",
                    step.pc_before
                ));
            }
        } else if is_amo {
            if ram_read.is_none() || ram_write.is_none() {
                return Err(format!(
                    "RV32 B1: AMO expects one RAM read and one RAM write at pc={:#x} (chunk j={j})",
                    step.pc_before
                ));
            }
        } else if ram_read.is_some() || ram_write.is_some() {
            return Err(format!(
                "RV32 B1: unexpected RAM event(s) at pc={:#x} (chunk j={j})",
                step.pc_before
            ));
        }

        if let (Some((read_addr, _)), Some((write_addr, _))) = (ram_read, ram_write) {
            if read_addr != write_addr {
                return Err(format!(
                    "RV32 B1: RAM read/write addr mismatch at pc={:#x} (chunk j={j}): read_addr={:#x}, write_addr={:#x}",
                    step.pc_before, read_addr, write_addr
                ));
            }
        }

        if let Some((addr, value)) = ram_read {
            z[layout.eff_addr(j)] = F::from_u64(addr);
            z[layout.mem_rv(j)] = F::from_u64(value);
        }
        if let Some((addr, value)) = ram_write {
            z[layout.eff_addr(j)] = F::from_u64(addr);
            z[layout.ram_wv(j)] = F::from_u64(value);
        }

        if fill_bus {
            let ram_bus_has_read = ram_read.is_some();
            let ram_bus_has_write = ram_write.is_some();
            set_bus_cell(
                &mut z,
                layout,
                ram_lane.has_read,
                j,
                if ram_bus_has_read { F::ONE } else { F::ZERO },
            );
            set_bus_cell(
                &mut z,
                layout,
                ram_lane.has_write,
                j,
                if ram_bus_has_write { F::ONE } else { F::ZERO },
            );
            if let Some((addr, value)) = ram_read {
                write_bus_u64_bits(
                    &mut z,
                    layout,
                    ram_lane.ra_bits.start,
                    ram_lane.ra_bits.end - ram_lane.ra_bits.start,
                    j,
                    addr,
                );
                set_bus_cell(&mut z, layout, ram_lane.rv, j, F::from_u64(value));
            }
            if let Some((addr, value)) = ram_write {
                write_bus_u64_bits(
                    &mut z,
                    layout,
                    ram_lane.wa_bits.start,
                    ram_lane.wa_bits.end - ram_lane.wa_bits.start,
                    j,
                    addr,
                );
                set_bus_cell(&mut z, layout, ram_lane.wv, j, F::from_u64(value));
            }
            set_bus_cell(&mut z, layout, ram_lane.inc, j, F::ZERO);
        }

        // Shout events: expect at most one lookup and bind it to a single lane.
        let shout_ev = match step.shout_events.as_slice() {
            [] => None,
            [one] => Some(one),
            _ => {
                let ids: Vec<u32> = step.shout_events.iter().map(|ev| ev.shout_id.0).collect();
                return Err(format!(
                    "RV32 B1: multiple shout events in one step (pc={:#x}, chunk j={j}): shout_ids={ids:?}; this circuit has 1 Shout port (no lanes) so you must either provision multiple Shout lanes in the shared CPU bus or split into micro-steps",
                    step.pc_before
                ));
            }
        };

        let mut expected_table_id: Option<u32> = None;
        let mut expect_table = |flag: bool, table_id: u32, name: &str| -> Result<(), String> {
            if !flag {
                return Ok(());
            }
            if expected_table_id.replace(table_id).is_some() {
                return Err(format!(
                    "RV32 B1: multiple Shout lookups expected at pc={:#x} (chunk j={j})",
                    step.pc_before
                ));
            }
            if layout.shout_idx(table_id).is_err() {
                return Err(format!(
                    "RV32 B1: missing Shout table {name} (id={table_id}) at pc={:#x} (chunk j={j})",
                    step.pc_before
                ));
            }
            Ok(())
        };

        expect_table(add_has_lookup, ADD_TABLE_ID, "ADD")?;
        expect_table(and_has_lookup, AND_TABLE_ID, "AND")?;
        expect_table(xor_has_lookup, XOR_TABLE_ID, "XOR")?;
        expect_table(or_has_lookup, OR_TABLE_ID, "OR")?;
        expect_table(sll_has_lookup, SLL_TABLE_ID, "SLL")?;
        expect_table(srl_has_lookup, SRL_TABLE_ID, "SRL")?;
        expect_table(sra_has_lookup, SRA_TABLE_ID, "SRA")?;
        expect_table(slt_has_lookup, SLT_TABLE_ID, "SLT")?;
        expect_table(sltu_has_lookup, SLTU_TABLE_ID, "SLTU")?;
        expect_table(is_sub, SUB_TABLE_ID, "SUB")?;
        expect_table(is_beq, EQ_TABLE_ID, "EQ")?;
        expect_table(is_bne, NEQ_TABLE_ID, "NEQ")?;

        match (expected_table_id, shout_ev) {
            (None, None) => {}
            (None, Some(ev)) => {
                return Err(format!(
                    "RV32 B1: unexpected shout event id={} at pc={:#x} (chunk j={j})",
                    ev.shout_id.0, step.pc_before
                ));
            }
            (Some(expected), None) => {
                return Err(format!(
                    "RV32 B1: missing shout event for table_id={expected} at pc={:#x} (chunk j={j})",
                    step.pc_before
                ));
            }
            (Some(expected), Some(ev)) => {
                let got = ev.shout_id.0;
                if got != expected {
                    return Err(format!(
                        "RV32 B1: shout table id mismatch at pc={:#x} (chunk j={j}): expected={expected}, got={got}",
                        step.pc_before
                    ));
                }
                let table_idx = layout
                    .shout_idx(expected)
                    .map_err(|e| format!("RV32 B1: {e} at pc={:#x} (chunk j={j})", step.pc_before))?;
                let lane = &layout.bus.shout_cols[table_idx].lanes[0];
                if fill_bus {
                    set_bus_cell(&mut z, layout, lane.has_lookup, j, F::ONE);
                    write_bus_u64_bits(
                        &mut z,
                        layout,
                        lane.addr_bits.start,
                        lane.addr_bits.end - lane.addr_bits.start,
                        j,
                        ev.key,
                    );
                    set_bus_cell(&mut z, layout, lane.val, j, F::from_u64(ev.value));
                }
                z[layout.alu_out(j)] = F::from_u64(ev.value);
            }
        }

        if fill_bus {
            let add_a0 = z[layout.bus.bus_cell(add_lane.addr_bits.start + 0, j)];
            let add_b0 = z[layout.bus.bus_cell(add_lane.addr_bits.start + 1, j)];
            z[layout.add_a0b0(j)] = add_a0 * add_b0;
        } else if let Some(ev) = shout_ev {
            if ev.shout_id.0 == ADD_TABLE_ID {
                let a0 = if (ev.key & 1) == 1 { F::ONE } else { F::ZERO };
                let b0 = if ((ev.key >> 1) & 1) == 1 { F::ONE } else { F::ZERO };
                z[layout.add_a0b0(j)] = a0 * b0;
            }
        }

        let rs1_i32 = rs1_u32 as i32;
        let rs2_i32 = rs2_u32 as i32;
        let mulh_u32 = if is_mulh {
            ((rs1_i32 as i64 * rs2_i32 as i64) >> 32) as i32 as u32
        } else {
            0u32
        };
        let mulhsu_u32 = if is_mulhsu {
            ((rs1_i32 as i64 * rs2_u32 as i64) >> 32) as i32 as u32
        } else {
            0u32
        };

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
        if is_mul {
            z[layout.rd_write_val(j)] = F::from_u64(mul_lo);
        }
        if is_mulhu {
            z[layout.rd_write_val(j)] = F::from_u64(mul_hi);
        }
        if is_mulh {
            z[layout.rd_write_val(j)] = F::from_u64(mulh_u32 as u64);
        }
        if is_mulhsu {
            z[layout.rd_write_val(j)] = F::from_u64(mulhsu_u32 as u64);
        }
        if is_divu {
            z[layout.rd_write_val(j)] = z[layout.div_quot(j)];
        }
        if is_remu {
            z[layout.rd_write_val(j)] = z[layout.div_rem(j)];
        }
        if is_div {
            if rs2_is_zero {
                z[layout.rd_write_val(j)] = F::from_u64(u32::MAX as u64);
            } else {
                z[layout.rd_write_val(j)] = z[layout.div_quot_signed(j)];
            }
        }
        if is_rem {
            z[layout.rd_write_val(j)] = z[layout.div_rem_signed(j)];
        }
        if is_lw || is_amoswap_w || is_amoadd_w || is_amoxor_w || is_amoor_w || is_amoand_w {
            z[layout.rd_write_val(j)] = z[layout.mem_rv(j)];
        }
        if is_lb || is_lbu || is_lh || is_lhu {
            let mem_rv_u64 = z[layout.mem_rv(j)].as_canonical_u64();
            let mem_rv_u32 = u32::try_from(mem_rv_u64).map_err(|_| {
                format!(
                    "RV32 B1: mem_rv does not fit in u32 at pc={:#x}: {mem_rv_u64}",
                    step.pc_before
                )
            })?;
            if is_lb {
                let byte = (mem_rv_u32 & 0xff) as u8;
                z[layout.rd_write_val(j)] = F::from_u64((byte as i8 as i32 as u32) as u64);
            }
            if is_lbu {
                z[layout.rd_write_val(j)] = F::from_u64((mem_rv_u32 & 0xff) as u64);
            }
            if is_lh {
                let half = (mem_rv_u32 & 0xffff) as u16;
                z[layout.rd_write_val(j)] = F::from_u64((half as i16 as i32 as u32) as u64);
            }
            if is_lhu {
                z[layout.rd_write_val(j)] = F::from_u64((mem_rv_u32 & 0xffff) as u64);
            }
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

        let mul_carry = if is_mulh {
            let rhs = (mul_hi as i128)
                - (rs1_sign as i128) * (rs2_u64 as i128)
                - (rs2_sign as i128) * (rs1_u64 as i128)
                + (rs1_sign as i128) * (rs2_sign as i128) * (1i128 << 32)
                + (1i128 << 32);
            let diff = rhs - (mulh_u32 as i128);
            if diff < 0 || diff % (1i128 << 32) != 0 {
                return Err(format!(
                    "RV32 B1: MULH carry mismatch at pc={:#x} (chunk j={j})",
                    step.pc_before
                ));
            }
            (diff >> 32) as u64
        } else if is_mulhsu {
            let rhs = (mul_hi as i128) - (rs1_sign as i128) * (rs2_u64 as i128) + (1i128 << 32);
            let diff = rhs - (mulhsu_u32 as i128);
            if diff < 0 || diff % (1i128 << 32) != 0 {
                return Err(format!(
                    "RV32 B1: MULHSU carry mismatch at pc={:#x} (chunk j={j})",
                    step.pc_before
                ));
            }
            (diff >> 32) as u64
        } else {
            0u64
        };
        z[layout.mul_carry(j)] = F::from_u64(mul_carry);

        let rd_write_u64 = z[layout.rd_write_val(j)].as_canonical_u64();
        let rd_write_u32 = u32::try_from(rd_write_u64)
            .map_err(|_| format!("RV32 B1: rd_write_val does not fit in u32: {rd_write_u64}"))?;
        let mem_rv_u64 = z[layout.mem_rv(j)].as_canonical_u64();
        let mem_rv_u32 = u32::try_from(mem_rv_u64)
            .map_err(|_| format!("RV32 B1: mem_rv does not fit in u32: {mem_rv_u64}"))?;
        let div_quot_u32 = u32::try_from(div_quot).map_err(|_| "RV32 B1: div_quot overflow".to_string())?;
        let div_rem_u32 = u32::try_from(div_rem).map_err(|_| "RV32 B1: div_rem overflow".to_string())?;

        for bit in 0..32 {
            z[layout.rd_write_bit(bit, j)] = if ((rd_write_u32 >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
            z[layout.mem_rv_bit(bit, j)] = if ((mem_rv_u32 >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
            z[layout.mul_lo_bit(bit, j)] = if ((mul_lo as u32 >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
            z[layout.mul_hi_bit(bit, j)] = if ((mul_hi as u32 >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
            z[layout.div_quot_bit(bit, j)] = if ((div_quot_u32 >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
            z[layout.div_rem_bit(bit, j)] = if ((div_rem_u32 >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
        }
        for bit in 0..2 {
            z[layout.mul_carry_bit(bit, j)] = if ((mul_carry >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
        }

        let mut prefix = if (mul_hi as u32 & 1) == 1 { F::ONE } else { F::ZERO };
        z[layout.mul_hi_prefix(0, j)] = prefix;
        for k in 1..31 {
            let bit = ((mul_hi as u32 >> k) & 1) == 1;
            prefix *= if bit { F::ONE } else { F::ZERO };
            z[layout.mul_hi_prefix(k, j)] = prefix;
        }
    }

    z[layout.pc_final] = F::from_u64(carried_pc);
    for r in 0..32 {
        z[layout.regs_final_start + r] = F::from_u64(carried_regs[r]);
    }
    z[layout.regs_final_start] = F::ZERO;

    // Chunk-level halting state used for cross-chunk padding semantics.
    z[layout.halted_in] = F::ONE - z[layout.is_active(0)];
    let j_last = layout.chunk_size - 1;
    z[layout.halted_out] = F::ONE - z[layout.is_active(j_last)] + z[layout.halt_effective(j_last)];

    Ok(z)
}
