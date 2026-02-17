use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;

use neo_vm_trace::{StepTrace, TwistOpKind};

use crate::riscv::lookups::{
    decode_instruction, BranchCondition, RiscvInstruction, RiscvMemOp, RiscvOpcode, PROG_ID, RAM_ID, REG_ID,
};

use super::constants::{
    ADD_TABLE_ID, AND_TABLE_ID, EQ_TABLE_ID, OR_TABLE_ID, SLL_TABLE_ID, SLTU_TABLE_ID, SLT_TABLE_ID, SRA_TABLE_ID,
    SRL_TABLE_ID, SUB_TABLE_ID, XOR_TABLE_ID,
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

/// Build a CPU witness vector `z` for shared-bus mode.
///
/// In shared-bus mode, `R1csCpu` overwrites the reserved bus tail from `StepTrace` events, so this
/// witness builder leaves the bus region at its zero default and only populates CPU columns.
pub fn rv32_b1_chunk_to_witness(layout: Rv32B1Layout) -> Box<dyn Fn(&[StepTrace<u64, u64>]) -> Vec<F> + Send + Sync> {
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

pub fn rv32_b1_chunk_to_witness_checked(
    layout: &Rv32B1Layout,
    chunk: &[StepTrace<u64, u64>],
) -> Result<Vec<F>, String> {
    rv32_b1_chunk_to_witness_internal(layout, chunk, /*fill_bus=*/ false)
}

pub fn rv32_b1_chunk_to_full_witness_checked(
    layout: &Rv32B1Layout,
    chunk: &[StepTrace<u64, u64>],
) -> Result<Vec<F>, String> {
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
    let reg_inst = &layout.bus.twist_cols[layout.reg_twist_idx];
    if reg_inst.lanes.len() < 2 {
        return Err(format!(
            "RV32 B1 witness: REG_ID twist instance must have >=2 lanes, got {}",
            reg_inst.lanes.len()
        ));
    }
    let reg_lane0 = &reg_inst.lanes[0];
    let reg_lane1 = &reg_inst.lanes[1];

    // Carry the architectural state forward through padding rows.
    // Initialize from the chunk's start state so fully-inactive chunks are well-defined.
    let mut carried_pc = 0u64;

    if let Some(first) = chunk.first() {
        z[layout.pc0] = F::from_u64(first.pc_before);
        carried_pc = first.pc_before;
    }

    let mut rv32m_count = 0u64;
    for j in 0..layout.chunk_size {
        if j >= chunk.len() {
            z[layout.is_active(j)] = F::ZERO;

            z[layout.pc_in(j)] = F::from_u64(carried_pc);
            z[layout.pc_out(j)] = F::from_u64(carried_pc);
            z[layout.halt_effective(j)] = F::ZERO;
            z[layout.reg_has_write(j)] = F::ZERO;
            z[layout.rd_is_zero_01(j)] = F::ONE;
            z[layout.rd_is_zero_012(j)] = F::ONE;
            z[layout.rd_is_zero_0123(j)] = F::ONE;
            z[layout.rd_is_zero(j)] = F::ONE;
            // Columns constrained independently of `is_active` must be set consistently on padding rows.
            for bit in 0..32 {
                z[layout.mem_rv_bit(bit, j)] = F::ZERO;
                z[layout.mul_lo_bit(bit, j)] = F::ZERO;
                z[layout.mul_hi_bit(bit, j)] = F::ZERO;
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
            continue;
        }
        let step = &chunk[j];

        // A row is active iff it contains exactly one PROG_ID read (B1 instruction fetch).
        // Padding rows contain no Twist/Shout events and are treated as inactive.
        let mut prog_read: Option<(u64, u64)> = None;
        let mut ram_read: Option<(u64, u64)> = None;
        let mut ram_write: Option<(u64, u64)> = None;
        let mut reg_lane0_read: Option<(u64, u64)> = None;
        let mut reg_lane0_write: Option<(u64, u64)> = None;
        let mut reg_lane1_read: Option<(u64, u64)> = None;
        let mut reg_lane1_write: Option<(u64, u64)> = None;
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
            } else if ev.twist_id == REG_ID {
                let lane = ev
                    .lane
                    .ok_or_else(|| format!("RV32 B1: missing lane for REG_ID event at pc={:#x}", step.pc_before))?;
                match (lane, ev.kind) {
                    (0, TwistOpKind::Read) => {
                        if reg_lane0_read.replace((ev.addr, ev.value)).is_some() {
                            return Err(format!(
                                "RV32 B1: multiple REG_ID lane0 reads in one step at pc={:#x} (chunk j={j})",
                                step.pc_before
                            ));
                        }
                    }
                    (0, TwistOpKind::Write) => {
                        if reg_lane0_write.replace((ev.addr, ev.value)).is_some() {
                            return Err(format!(
                                "RV32 B1: multiple REG_ID lane0 writes in one step at pc={:#x} (chunk j={j})",
                                step.pc_before
                            ));
                        }
                    }
                    (1, TwistOpKind::Read) => {
                        if reg_lane1_read.replace((ev.addr, ev.value)).is_some() {
                            return Err(format!(
                                "RV32 B1: multiple REG_ID lane1 reads in one step at pc={:#x} (chunk j={j})",
                                step.pc_before
                            ));
                        }
                    }
                    (1, TwistOpKind::Write) => {
                        if reg_lane1_write.replace((ev.addr, ev.value)).is_some() {
                            return Err(format!(
                                "RV32 B1: multiple REG_ID lane1 writes in one step at pc={:#x} (chunk j={j})",
                                step.pc_before
                            ));
                        }
                    }
                    (lane, _) => {
                        return Err(format!(
                            "RV32 B1: unexpected REG_ID lane={lane} at pc={:#x} (chunk j={j}); expected lane 0 or 1",
                            step.pc_before
                        ));
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
            z[layout.halt_effective(j)] = F::ZERO;
            z[layout.reg_has_write(j)] = F::ZERO;
            z[layout.rd_is_zero_01(j)] = F::ONE;
            z[layout.rd_is_zero_012(j)] = F::ONE;
            z[layout.rd_is_zero_0123(j)] = F::ONE;
            z[layout.rd_is_zero(j)] = F::ONE;
            // Columns constrained independently of `is_active` must be set consistently on padding rows.
            for bit in 0..32 {
                z[layout.mem_rv_bit(bit, j)] = F::ZERO;
                z[layout.mul_lo_bit(bit, j)] = F::ZERO;
                z[layout.mul_hi_bit(bit, j)] = F::ZERO;
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
            continue;
        }

        z[layout.is_active(j)] = F::ONE;
        z[layout.pc_in(j)] = F::from_u64(step.pc_before);
        z[layout.pc_out(j)] = F::from_u64(step.pc_after);

        carried_pc = step.pc_after;

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

        // Minimal decode bit plumbing (matches `push_rv32_b1_decode_constraints`).
        for bit in 0..5 {
            z[layout.rd_bit(bit, j)] = if ((rd >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
        }
        for bit in 0..7 {
            z[layout.funct7_bit(bit, j)] = if ((funct7 >> bit) & 1) == 1 { F::ONE } else { F::ZERO };
        }

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
        let imm_b = ((imm_b_raw as i32) << 19) >> 19;
        z[layout.imm_b(j)] = from_i32(imm_b);

        // J-type immediate raw bits (before sign extension).
        let imm_j_raw = (((instr_word_u32 >> 21) & 0x3ff) << 1)
            | (((instr_word_u32 >> 20) & 0x1) << 11)
            | (((instr_word_u32 >> 12) & 0xff) << 12)
            | (((instr_word_u32 >> 31) & 0x1) << 20);
        let imm_j = ((imm_j_raw as i32) << 11) >> 11;
        z[layout.imm_j(j)] = from_i32(imm_j);

        // Decode into a compact representation:
        // - opcode-class one-hot flags
        // - a few control signals for branches and ALU op selection
        let decoded = decode_instruction(instr_word_u32)
            .map_err(|e| format!("RV32 B1: decode failed at pc={:#x}: {e}", step.pc_before))?;

        let mut is_mul = false;
        let mut is_mulh = false;
        let mut is_mulhu = false;
        let mut is_mulhsu = false;
        let mut is_div = false;
        let mut is_divu = false;
        let mut is_rem = false;
        let mut is_remu = false;

        // Opcode-class flags.
        let mut is_alu_reg = false;
        let mut is_alu_imm = false;
        let mut is_load = false;
        let mut is_store = false;
        let mut is_amo = false;
        let mut is_branch = false;
        let mut is_lui = false;
        let mut is_auipc = false;
        let mut is_jal = false;
        let mut is_jalr = false;
        let mut is_fence = false;
        let mut is_halt = false;

        // Branch control.
        let mut br_cmp_eq = false;
        let mut br_cmp_lt = false;
        let mut br_cmp_ltu = false;
        let mut br_invert = false;

        // Shout selector helpers.
        let mut add_alu = false;
        let mut and_alu = false;
        let mut xor_alu = false;
        let mut or_alu = false;
        let mut slt_alu = false;
        let mut sltu_alu = false;
        let mut sub_has_lookup = false;
        let mut eq_has_lookup = false;
        let mut sll_has_lookup = false;
        let mut srl_has_lookup = false;
        let mut sra_has_lookup = false;
        let mut slt_has_lookup = false;
        let mut sltu_has_lookup_base = false;

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

        match decoded {
            RiscvInstruction::RAlu { op, .. } => match op {
                // RV32I ALU (R-type).
                RiscvOpcode::Add => {
                    is_alu_reg = true;
                    add_alu = true;
                }
                RiscvOpcode::Sub => {
                    is_alu_reg = true;
                    sub_has_lookup = true;
                }
                RiscvOpcode::Sll => {
                    is_alu_reg = true;
                    sll_has_lookup = true;
                }
                RiscvOpcode::Slt => {
                    is_alu_reg = true;
                    slt_alu = true;
                    slt_has_lookup = true;
                }
                RiscvOpcode::Sltu => {
                    is_alu_reg = true;
                    sltu_alu = true;
                    sltu_has_lookup_base = true;
                }
                RiscvOpcode::Xor => {
                    is_alu_reg = true;
                    xor_alu = true;
                }
                RiscvOpcode::Srl => {
                    is_alu_reg = true;
                    srl_has_lookup = true;
                }
                RiscvOpcode::Sra => {
                    is_alu_reg = true;
                    sra_has_lookup = true;
                }
                RiscvOpcode::Or => {
                    is_alu_reg = true;
                    or_alu = true;
                }
                RiscvOpcode::And => {
                    is_alu_reg = true;
                    and_alu = true;
                }
                // RV32M (R-type, funct7=0b0000001).
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
                RiscvOpcode::Add => {
                    is_alu_imm = true;
                    add_alu = true;
                }
                RiscvOpcode::Slt => {
                    is_alu_imm = true;
                    slt_alu = true;
                    slt_has_lookup = true;
                }
                RiscvOpcode::Sltu => {
                    is_alu_imm = true;
                    sltu_alu = true;
                    sltu_has_lookup_base = true;
                }
                RiscvOpcode::Xor => {
                    is_alu_imm = true;
                    xor_alu = true;
                }
                RiscvOpcode::Or => {
                    is_alu_imm = true;
                    or_alu = true;
                }
                RiscvOpcode::And => {
                    is_alu_imm = true;
                    and_alu = true;
                }
                RiscvOpcode::Sll => {
                    is_alu_imm = true;
                    sll_has_lookup = true;
                }
                RiscvOpcode::Srl => {
                    is_alu_imm = true;
                    srl_has_lookup = true;
                }
                RiscvOpcode::Sra => {
                    is_alu_imm = true;
                    sra_has_lookup = true;
                }
                _ => {}
            },
            RiscvInstruction::Load { op, .. } => {
                is_load = true;
                match op {
                    RiscvMemOp::Lb => is_lb = true,
                    RiscvMemOp::Lbu => is_lbu = true,
                    RiscvMemOp::Lh => is_lh = true,
                    RiscvMemOp::Lhu => is_lhu = true,
                    RiscvMemOp::Lw => is_lw = true,
                    _ => {}
                }
            }
            RiscvInstruction::Store { op, .. } => {
                is_store = true;
                match op {
                    RiscvMemOp::Sb => is_sb = true,
                    RiscvMemOp::Sh => is_sh = true,
                    RiscvMemOp::Sw => is_sw = true,
                    _ => {}
                }
            }
            RiscvInstruction::Amo { op, .. } => {
                is_amo = true;
                match op {
                    RiscvMemOp::AmoswapW => is_amoswap_w = true,
                    RiscvMemOp::AmoaddW => is_amoadd_w = true,
                    RiscvMemOp::AmoxorW => is_amoxor_w = true,
                    RiscvMemOp::AmoorW => is_amoor_w = true,
                    RiscvMemOp::AmoandW => is_amoand_w = true,
                    _ => {}
                }
            }
            RiscvInstruction::Lui { .. } => is_lui = true,
            RiscvInstruction::Auipc { .. } => is_auipc = true,
            RiscvInstruction::Branch { cond, .. } => {
                is_branch = true;
                match cond {
                    BranchCondition::Eq => {
                        br_cmp_eq = true;
                        br_invert = false;
                        eq_has_lookup = true;
                    }
                    BranchCondition::Ne => {
                        // Represent BNE as EQ + invert.
                        br_cmp_eq = true;
                        br_invert = true;
                        eq_has_lookup = true;
                    }
                    BranchCondition::Lt => {
                        br_cmp_lt = true;
                        br_invert = false;
                        slt_has_lookup = true;
                    }
                    BranchCondition::Ge => {
                        br_cmp_lt = true;
                        br_invert = true;
                        slt_has_lookup = true;
                    }
                    BranchCondition::Ltu => {
                        br_cmp_ltu = true;
                        br_invert = false;
                        sltu_has_lookup_base = true;
                    }
                    BranchCondition::Geu => {
                        br_cmp_ltu = true;
                        br_invert = true;
                        sltu_has_lookup_base = true;
                    }
                }
            }
            RiscvInstruction::Jal { .. } => is_jal = true,
            RiscvInstruction::Jalr { .. } => is_jalr = true,
            RiscvInstruction::Fence { .. } => is_fence = true,
            RiscvInstruction::Halt => is_halt = true,
            _ => {}
        }

        if is_mul || is_mulh || is_mulhu || is_mulhsu || is_div || is_divu || is_rem || is_remu {
            is_alu_reg = true;
        }

        // Reject unsupported instructions.
        if !(is_alu_reg
            || is_alu_imm
            || is_load
            || is_store
            || is_amo
            || is_branch
            || is_lui
            || is_auipc
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

        z[layout.is_alu_reg(j)] = if is_alu_reg { F::ONE } else { F::ZERO };
        z[layout.is_alu_imm(j)] = if is_alu_imm { F::ONE } else { F::ZERO };
        z[layout.is_load(j)] = if is_load { F::ONE } else { F::ZERO };
        z[layout.is_store(j)] = if is_store { F::ONE } else { F::ZERO };
        z[layout.is_amo(j)] = if is_amo { F::ONE } else { F::ZERO };
        z[layout.is_branch(j)] = if is_branch { F::ONE } else { F::ZERO };
        z[layout.is_lui(j)] = if is_lui { F::ONE } else { F::ZERO };
        z[layout.is_auipc(j)] = if is_auipc { F::ONE } else { F::ZERO };
        z[layout.is_jal(j)] = if is_jal { F::ONE } else { F::ZERO };
        z[layout.is_jalr(j)] = if is_jalr { F::ONE } else { F::ZERO };
        z[layout.is_fence(j)] = if is_fence { F::ONE } else { F::ZERO };
        z[layout.is_halt(j)] = if is_halt { F::ONE } else { F::ZERO };

        z[layout.br_cmp_eq(j)] = if br_cmp_eq { F::ONE } else { F::ZERO };
        z[layout.br_cmp_lt(j)] = if br_cmp_lt { F::ONE } else { F::ZERO };
        z[layout.br_cmp_ltu(j)] = if br_cmp_ltu { F::ONE } else { F::ZERO };
        z[layout.br_invert(j)] = if br_invert { F::ONE } else { F::ZERO };

        z[layout.add_alu(j)] = if add_alu { F::ONE } else { F::ZERO };
        z[layout.and_alu(j)] = if and_alu { F::ONE } else { F::ZERO };
        z[layout.xor_alu(j)] = if xor_alu { F::ONE } else { F::ZERO };
        z[layout.or_alu(j)] = if or_alu { F::ONE } else { F::ZERO };
        z[layout.slt_alu(j)] = if slt_alu { F::ONE } else { F::ZERO };
        z[layout.sltu_alu(j)] = if sltu_alu { F::ONE } else { F::ZERO };
        z[layout.sub_has_lookup(j)] = if sub_has_lookup { F::ONE } else { F::ZERO };
        z[layout.eq_has_lookup(j)] = if eq_has_lookup { F::ONE } else { F::ZERO };

        z[layout.is_mul(j)] = if is_mul { F::ONE } else { F::ZERO };
        z[layout.is_mulh(j)] = if is_mulh { F::ONE } else { F::ZERO };
        z[layout.is_mulhu(j)] = if is_mulhu { F::ONE } else { F::ZERO };
        z[layout.is_mulhsu(j)] = if is_mulhsu { F::ONE } else { F::ZERO };
        z[layout.is_div(j)] = if is_div { F::ONE } else { F::ZERO };
        z[layout.is_divu(j)] = if is_divu { F::ONE } else { F::ZERO };
        z[layout.is_rem(j)] = if is_rem { F::ONE } else { F::ZERO };
        z[layout.is_remu(j)] = if is_remu { F::ONE } else { F::ZERO };
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

        let rs1_idx = rs1 as usize;
        let rs2_idx = rs2 as usize;
        let rd_idx = rd as usize;

        // Derived group/control signals.
        let writes_rd = is_alu_reg || is_alu_imm || is_load || is_amo || is_lui || is_auipc || is_jal || is_jalr;
        z[layout.writes_rd(j)] = if writes_rd { F::ONE } else { F::ZERO };

        // pc_plus4 is true for all non-branch/non-jump active rows.
        let pc_plus4 = !is_branch && !is_jal && !is_jalr;
        z[layout.pc_plus4(j)] = if pc_plus4 { F::ONE } else { F::ZERO };

        // wb_from_alu selects the ALU/shout-backed writeback path.
        let is_rv32m = is_mul || is_mulh || is_mulhu || is_mulhsu || is_div || is_divu || is_rem || is_remu;
        if is_rv32m {
            rv32m_count = rv32m_count
                .checked_add(1)
                .ok_or_else(|| "RV32 B1: rv32m_count overflow".to_string())?;
        }
        let wb_from_alu = is_alu_imm || (is_alu_reg && !is_rv32m) || is_auipc;
        z[layout.wb_from_alu(j)] = if wb_from_alu { F::ONE } else { F::ZERO };

        let reg_has_write = writes_rd && rd_idx != 0;
        z[layout.reg_has_write(j)] = if reg_has_write { F::ONE } else { F::ZERO };

        z[layout.halt_effective(j)] = if is_halt { F::ONE } else { F::ZERO };

        // rd_is_zero_* chain from rd bits.
        let rd_b7 = (rd as u64) & 1;
        let rd_b8 = ((rd as u64) >> 1) & 1;
        let rd_b9 = ((rd as u64) >> 2) & 1;
        let rd_b10 = ((rd as u64) >> 3) & 1;
        let rd_b11 = ((rd as u64) >> 4) & 1;
        let rd_is_zero_01 = (1 - rd_b7) * (1 - rd_b8);
        let rd_is_zero_012 = rd_is_zero_01 * (1 - rd_b9);
        let rd_is_zero_0123 = rd_is_zero_012 * (1 - rd_b10);
        let rd_is_zero = rd_is_zero_0123 * (1 - rd_b11);
        z[layout.rd_is_zero_01(j)] = if rd_is_zero_01 == 1 { F::ONE } else { F::ZERO };
        z[layout.rd_is_zero_012(j)] = if rd_is_zero_012 == 1 { F::ONE } else { F::ZERO };
        z[layout.rd_is_zero_0123(j)] = if rd_is_zero_0123 == 1 { F::ONE } else { F::ZERO };
        z[layout.rd_is_zero(j)] = if rd_is_zero == 1 { F::ONE } else { F::ZERO };

        // Selected operand values.
        let rs1_u32 = u32::try_from(step.regs_before[rs1_idx])
            .map_err(|_| format!("RV32 B1: rs1 value does not fit in u32 at pc={:#x}", step.pc_before))?;
        let rs2_u32 = u32::try_from(step.regs_before[rs2_idx])
            .map_err(|_| format!("RV32 B1: rs2 value does not fit in u32 at pc={:#x}", step.pc_before))?;
        let rs1_u64 = rs1_u32 as u64;
        let rs2_u64 = rs2_u32 as u64;
        z[layout.rs1_val(j)] = F::from_u64(rs1_u64);
        z[layout.rs2_val(j)] = F::from_u64(rs2_u64);
        if is_rv32m {
            z[layout.rv32m_rs1_val(j)] = z[layout.rs1_val(j)];
            z[layout.rv32m_rs2_val(j)] = z[layout.rs2_val(j)];
        }

        // Shift rhs helper (see semantics sidecar): select rs2_val for reg shifts and rs2_field for imm shifts.
        // This value is only used when a shift Shout table is active, but we set it unconditionally.
        z[layout.shift_rhs(j)] = if is_alu_imm {
            F::from_u64(rs2 as u64)
        } else {
            F::from_u64(rs2_u64)
        };

        // Regfile Twist events (REG_ID): validate and optionally write bus lanes.
        if reg_lane1_write.is_some() {
            return Err(format!(
                "RV32 B1: unexpected REG_ID lane1 write at pc={:#x} (chunk j={j})",
                step.pc_before
            ));
        }
        let (rf0_ra, rf0_rv) = reg_lane0_read.ok_or_else(|| {
            format!(
                "RV32 B1: missing REG_ID lane0 read at pc={:#x} (chunk j={j})",
                step.pc_before
            )
        })?;
        let (rf1_ra, rf1_rv) = reg_lane1_read.ok_or_else(|| {
            format!(
                "RV32 B1: missing REG_ID lane1 read at pc={:#x} (chunk j={j})",
                step.pc_before
            )
        })?;

        if rf0_ra != rs1_idx as u64 {
            return Err(format!(
                "RV32 B1: REG_ID lane0 read addr mismatch at pc={:#x} (chunk j={j}): expected rs1_addr={:#x}, got {rf0_ra:#x}",
                step.pc_before,
                rs1_idx as u64
            ));
        }
        if rf0_rv != rs1_u64 {
            return Err(format!(
                "RV32 B1: REG_ID lane0 read value mismatch at pc={:#x} (chunk j={j}): expected rs1_val={:#x}, got {rf0_rv:#x}",
                step.pc_before, rs1_u64
            ));
        }

        if rf1_ra != rs2_idx as u64 {
            return Err(format!(
                "RV32 B1: REG_ID lane1 read addr mismatch at pc={:#x} (chunk j={j}): expected rs2_addr={:#x}, got {rf1_ra:#x}",
                step.pc_before,
                rs2_idx as u64
            ));
        }
        if rf1_rv != rs2_u64 {
            return Err(format!(
                "RV32 B1: REG_ID lane1 read value mismatch at pc={:#x} (chunk j={j}): expected rs2_val={rs2_u64:#x}, got {rf1_rv:#x}",
                step.pc_before
            ));
        }

        let rf0_write = reg_lane0_write;
        if reg_has_write != rf0_write.is_some() {
            return Err(format!(
                "RV32 B1: REG_ID lane0 write presence mismatch at pc={:#x} (chunk j={j}): reg_has_write={reg_has_write}, has_write_event={}",
                step.pc_before,
                rf0_write.is_some()
            ));
        }

        if fill_bus {
            // Lane 0 (rs1 read + optional rd write).
            set_bus_cell(&mut z, layout, reg_lane0.has_read, j, F::ONE);
            write_bus_u64_bits(
                &mut z,
                layout,
                reg_lane0.ra_bits.start,
                reg_lane0.ra_bits.end - reg_lane0.ra_bits.start,
                j,
                rf0_ra,
            );
            set_bus_cell(&mut z, layout, reg_lane0.rv, j, F::from_u64(rf0_rv));

            set_bus_cell(
                &mut z,
                layout,
                reg_lane0.has_write,
                j,
                if rf0_write.is_some() { F::ONE } else { F::ZERO },
            );
            if let Some((wa, wv)) = rf0_write {
                write_bus_u64_bits(
                    &mut z,
                    layout,
                    reg_lane0.wa_bits.start,
                    reg_lane0.wa_bits.end - reg_lane0.wa_bits.start,
                    j,
                    wa,
                );
                set_bus_cell(&mut z, layout, reg_lane0.wv, j, F::from_u64(wv));
            }
            set_bus_cell(&mut z, layout, reg_lane0.inc, j, F::ZERO);

            // Lane 1 (rs2/a0 read).
            set_bus_cell(&mut z, layout, reg_lane1.has_read, j, F::ONE);
            set_bus_cell(&mut z, layout, reg_lane1.has_write, j, F::ZERO);
            write_bus_u64_bits(
                &mut z,
                layout,
                reg_lane1.ra_bits.start,
                reg_lane1.ra_bits.end - reg_lane1.ra_bits.start,
                j,
                rf1_ra,
            );
            set_bus_cell(&mut z, layout, reg_lane1.rv, j, F::from_u64(rf1_rv));
            set_bus_cell(&mut z, layout, reg_lane1.inc, j, F::ZERO);
        }

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

        // Shared-bus bound values: Shout selectors + Twist mirrors.
        let imm_i_u64 = sx_u32(imm_i);
        let imm_s_u64 = sx_u32(imm_s);

        let alu_rhs_u64 = if is_alu_imm { imm_i_u64 } else { rs2_u64 };
        z[layout.alu_rhs(j)] = F::from_u64(alu_rhs_u64);

        let add_has_lookup = add_alu || is_load || is_store || is_amoadd_w || is_auipc || is_jalr;
        z[layout.add_has_lookup(j)] = if add_has_lookup { F::ONE } else { F::ZERO };
        let and_has_lookup = and_alu || is_amoand_w;
        z[layout.and_has_lookup(j)] = if and_has_lookup { F::ONE } else { F::ZERO };
        let xor_has_lookup = xor_alu || is_amoxor_w;
        z[layout.xor_has_lookup(j)] = if xor_has_lookup { F::ONE } else { F::ZERO };
        let or_has_lookup = or_alu || is_amoor_w;
        z[layout.or_has_lookup(j)] = if or_has_lookup { F::ONE } else { F::ZERO };
        z[layout.sll_has_lookup(j)] = if sll_has_lookup { F::ONE } else { F::ZERO };
        z[layout.srl_has_lookup(j)] = if srl_has_lookup { F::ONE } else { F::ZERO };
        z[layout.sra_has_lookup(j)] = if sra_has_lookup { F::ONE } else { F::ZERO };
        z[layout.slt_has_lookup(j)] = if slt_has_lookup { F::ONE } else { F::ZERO };
        let sltu_has_lookup = sltu_has_lookup_base || do_rem_check || do_rem_check_signed;
        z[layout.sltu_has_lookup(j)] = if sltu_has_lookup { F::ONE } else { F::ZERO };

        let ram_has_read = is_load || is_sb || is_sh || is_amo;
        let ram_has_write = is_store || is_amo;
        z[layout.ram_has_read(j)] = if ram_has_read { F::ONE } else { F::ZERO };
        z[layout.ram_has_write(j)] = if ram_has_write { F::ONE } else { F::ZERO };

        // Default zeros.
        z[layout.alu_out(j)] = F::ZERO;
        z[layout.br_invert_alu(j)] = F::ZERO;
        z[layout.add_a0b0(j)] = F::ZERO;
        z[layout.add_lhs(j)] = F::ZERO;
        z[layout.add_rhs(j)] = F::ZERO;
        z[layout.mem_rv(j)] = F::ZERO;
        z[layout.eff_addr(j)] = F::ZERO;
        z[layout.ram_wv(j)] = F::ZERO;
        z[layout.rd_write_val(j)] = F::ZERO;
        z[layout.br_taken(j)] = F::ZERO;
        z[layout.br_not_taken(j)] = F::ZERO;

        // RAM events: validate shape and fill the RAM twist lane + CPU mirrors.
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

        // ADD-table operand selection (for semantics sidecar key wiring).
        //
        // NOTE: For AMOADD.W, the ADD Shout lookup is used for the *memory update* (mem_rv + rs2),
        // not for the effective address (which is rs1).
        if add_has_lookup {
            let (lhs, rhs) = if add_alu {
                if is_alu_imm {
                    (rs1_u64, imm_i_u64)
                } else {
                    (rs1_u64, rs2_u64)
                }
            } else if is_load {
                (rs1_u64, imm_i_u64)
            } else if is_store {
                (rs1_u64, imm_s_u64)
            } else if is_auipc {
                (step.pc_before, imm_u)
            } else if is_jalr {
                (rs1_u64, imm_i_u64)
            } else if is_amoadd_w {
                let mem_rv_u64 = z[layout.mem_rv(j)].as_canonical_u64();
                (mem_rv_u64, rs2_u64)
            } else {
                (0u64, 0u64)
            };
            z[layout.add_lhs(j)] = F::from_u64(lhs);
            z[layout.add_rhs(j)] = F::from_u64(rhs);
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
        expect_table(sub_has_lookup, SUB_TABLE_ID, "SUB")?;
        expect_table(eq_has_lookup, EQ_TABLE_ID, "EQ")?;

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
                    set_bus_cell(&mut z, layout, lane.primary_val(), j, F::from_u64(ev.value));
                }
                z[layout.alu_out(j)] = F::from_u64(ev.value);
            }
        }

        // Branch decision helper product (used by the semantics CCS): br_invert_alu = br_invert * alu_out.
        z[layout.br_invert_alu(j)] = z[layout.br_invert(j)] * z[layout.alu_out(j)];

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
        if wb_from_alu {
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
        if is_branch {
            let taken = if br_invert {
                F::ONE - z[layout.alu_out(j)]
            } else {
                z[layout.alu_out(j)]
            };
            z[layout.br_taken(j)] = taken;
            z[layout.br_not_taken(j)] = F::ONE - taken;
        }

        // Poseidon2 read ECALL: set rd_write_val = ecall_rd_val so the existing
        // enforce_u32_bits range check covers it.
        if is_poseidon2_read_ecall {
            z[layout.rd_write_val(j)] = z[layout.ecall_rd_val(j)];
        }

        let mul_carry = if is_mulh {
            let rhs =
                (mul_hi as i128) - (rs1_sign as i128) * (rs2_u64 as i128) - (rs2_sign as i128) * (rs1_u64 as i128)
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
        let _ = u32::try_from(rd_write_u64)
            .map_err(|_| format!("RV32 B1: rd_write_val does not fit in u32: {rd_write_u64}"))?;
        let mem_rv_u64 = z[layout.mem_rv(j)].as_canonical_u64();
        let mem_rv_u32 =
            u32::try_from(mem_rv_u64).map_err(|_| format!("RV32 B1: mem_rv does not fit in u32: {mem_rv_u64}"))?;

        for bit in 0..32 {
            z[layout.mem_rv_bit(bit, j)] = if ((mem_rv_u32 >> bit) & 1) == 1 {
                F::ONE
            } else {
                F::ZERO
            };
            let mul_lo_or_div_quot = if is_div || is_divu || is_rem || is_remu {
                div_quot as u32
            } else {
                mul_lo as u32
            };
            z[layout.mul_lo_bit(bit, j)] = if ((mul_lo_or_div_quot >> bit) & 1) == 1 {
                F::ONE
            } else {
                F::ZERO
            };
            z[layout.mul_hi_bit(bit, j)] = if ((mul_hi as u32 >> bit) & 1) == 1 {
                F::ONE
            } else {
                F::ZERO
            };
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

        if is_rv32m {
            z[layout.rv32m_rd_write_val(j)] = z[layout.rd_write_val(j)];
        }
    }

    z[layout.pc_final] = F::from_u64(carried_pc);
    z[layout.rv32m_count] = F::from_u64(rv32m_count);

    // Chunk-level halting state used for cross-chunk padding semantics.
    z[layout.halted_in] = F::ONE - z[layout.is_active(0)];
    let j_last = layout.chunk_size - 1;
    z[layout.halted_out] = F::ONE - z[layout.is_active(j_last)] + z[layout.halt_effective(j_last)];

    Ok(z)
}
