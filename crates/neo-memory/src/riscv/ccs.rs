//! RV32 "B1" RISC-V step CCS (shared-bus compatible).
//!
//! This module provides a **sound, shared-bus-compatible** step circuit for a small,
//! MVP RV32 subset. The circuit is expressed as an R1CS→CCS:
//! - `A(z) * B(z) = C(z)` with `C = 0` for almost all rows
//! - CCS uses the rectangular-friendly 3-matrix embedding (`M_0=A, M_1=B, M_2=C`)
//!
//! The witness `z` includes a **reserved bus tail** whose column schema matches
//! `cpu::bus_layout::BusLayout`. The bus tail itself is written from `StepTrace`
//! events by `R1csCpu` (shared-bus mode), and is verified by the Twist/Shout sidecars.
//!
//! ## Execution model (Phase 1)
//! - Each lane `j` in a chunk is one architectural step.
//! - `is_active[j] ∈ {0,1}` gates padding; inactive lanes keep `(pc, regs)` constant and perform
//!   no bus activity (enforced via shared-bus padding constraints).
//! - Intra-chunk continuity is enforced: `pc_in[j+1] = pc_out[j]` and `regs_in[j+1] = regs_out[j]`.
//! - The chunk exposes public boundary state (`pc0/regs0` at lane 0 and `pc_final/regs_final` at the
//!   last lane). Multi-chunk executions must chain these boundary values across chunks at a higher layer.
//!
//! The CCS here constrains the **CPU glue**:
//! - ROM fetch binding (`PROG_ID`) via shared-bus bindings
//! - instruction decode from a committed 32-bit instruction word
//! - register-file update pattern
//! - RAM load/store binding (`RAM_ID`) via shared-bus bindings
//! - Shout key wiring for `ADD` lookups (table id 3)
//!
//! Supported RV32IMA subset (RV32, word-only memory, no compressed):
//! - ALU (R-type): `ADD`, `SUB`, `SLL`, `SLT`, `SLTU`, `XOR`, `SRL`, `SRA`, `OR`, `AND`
//! - M (R-type, in-circuit): `MUL`, `MULH`, `MULHU`, `MULHSU`, `DIV`, `DIVU`, `REM`, `REMU`
//! - ALU (I-type): `ADDI`, `SLTI`, `SLTIU`, `XORI`, `ORI`, `ANDI`, `SLLI`, `SRLI`, `SRAI`
//! - Memory (byte/half/word): `LB`, `LBU`, `LH`, `LHU`, `LW`, `SB`, `SH`, `SW`
//! - Atomics (word): `AMOADD.W`, `AMOAND.W`, `AMOOR.W`, `AMOXOR.W`, `AMOSWAP.W`
//! - Branch: `BEQ`, `BNE`, `BLT`, `BGE`, `BLTU`, `BGEU`
//! - Jump: `JAL`, `JALR`
//! - U-type: `LUI`, `AUIPC`
//! - System: `FENCE`, `ECALL(imm=0)` (halts)

use std::collections::HashMap;

use neo_ccs::relations::CcsStructure;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use crate::plain::PlainMemLayout;
use crate::riscv::lookups::{PROG_ID, RAM_ID, REG_ID};

mod bus_bindings;
mod config;
mod constants;
mod constraint_builder;
mod layout;
mod trace;
mod witness;

pub use bus_bindings::{
    rv32_b1_shared_cpu_bus_config, rv32_trace_shared_bus_requirements, rv32_trace_shared_bus_requirements_with_specs,
    rv32_trace_shared_cpu_bus_config, rv32_trace_shared_cpu_bus_config_with_specs, TraceShoutBusSpec,
};
pub use layout::Rv32B1Layout;
pub use trace::{
    build_rv32_trace_wiring_ccs, build_rv32_trace_wiring_ccs_with_reserved_rows, rv32_trace_ccs_witness_from_exec_table,
    rv32_trace_ccs_witness_from_trace_witness, Rv32TraceCcsLayout,
};
pub use witness::{
    rv32_b1_chunk_to_full_witness, rv32_b1_chunk_to_full_witness_checked, rv32_b1_chunk_to_witness,
    rv32_b1_chunk_to_witness_checked,
};

/// Verifier-side step-linking pairs for chaining multi-chunk executions.
///
/// For each adjacent pair of shard steps (chunks) `i` and `i+1`, require:
/// - `pc_final[i] == pc0[i+1]`
/// - `regs_final[i][r] == regs0[i+1][r]` for `r ∈ [0..32)`
/// - `halted_out[i] == halted_in[i+1]`
///
/// This is the minimal glue needed to make `chunk_size` a semantic no-op: the CPU state must form
/// one contiguous execution across chunks.
pub fn rv32_b1_step_linking_pairs(layout: &Rv32B1Layout) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(2);
    pairs.push((layout.pc_final, layout.pc0));
    pairs.push((layout.halted_out, layout.halted_in));
    pairs
}

/// Minimal Shout table set intended for small RV32 programs that only need:
/// - `ADD` (address/ALU wiring), and
/// - `EQ`/`NEQ` (BEQ/BNE).
pub const RV32_B1_SHOUT_PROFILE_MIN3: &[u32] = &[ADD_TABLE_ID, EQ_TABLE_ID, NEQ_TABLE_ID];

/// Full RV32I Shout table set (ids 0..=11).
pub const RV32_B1_SHOUT_PROFILE_FULL12: &[u32] = &[
    AND_TABLE_ID,
    XOR_TABLE_ID,
    OR_TABLE_ID,
    ADD_TABLE_ID,
    SUB_TABLE_ID,
    SLT_TABLE_ID,
    SLTU_TABLE_ID,
    SLL_TABLE_ID,
    SRL_TABLE_ID,
    SRA_TABLE_ID,
    EQ_TABLE_ID,
    NEQ_TABLE_ID,
];

/// Full RV32I Shout table set for trace-wiring mode (ids 0..=11).
pub const RV32_TRACE_SHOUT_PROFILE_FULL12: &[u32] = &[
    AND_TABLE_ID,
    XOR_TABLE_ID,
    OR_TABLE_ID,
    ADD_TABLE_ID,
    SUB_TABLE_ID,
    SLT_TABLE_ID,
    SLTU_TABLE_ID,
    SLL_TABLE_ID,
    SRL_TABLE_ID,
    SRA_TABLE_ID,
    EQ_TABLE_ID,
    NEQ_TABLE_ID,
];

/// Full RV32IM Shout table set (ids 0..=19).
/// M tables are optional; RV32 B1 proves M ops in-circuit and ignores their Shout lanes.
pub const RV32_B1_SHOUT_PROFILE_FULL20: &[u32] = &[
    AND_TABLE_ID,
    XOR_TABLE_ID,
    OR_TABLE_ID,
    ADD_TABLE_ID,
    SUB_TABLE_ID,
    SLT_TABLE_ID,
    SLTU_TABLE_ID,
    SLL_TABLE_ID,
    SRL_TABLE_ID,
    SRA_TABLE_ID,
    EQ_TABLE_ID,
    NEQ_TABLE_ID,
    MUL_TABLE_ID,
    MULH_TABLE_ID,
    MULHU_TABLE_ID,
    MULHSU_TABLE_ID,
    DIV_TABLE_ID,
    DIVU_TABLE_ID,
    REM_TABLE_ID,
    REMU_TABLE_ID,
];

use bus_bindings::injected_bus_constraints_len;
use config::{derive_mem_ids_and_ell_addrs, derive_shout_ids_and_ell_addrs};
use constraint_builder::{build_r1cs_ccs, Constraint};
use layout::build_layout_with_m;

use constants::{
    ADD_TABLE_ID, AND_TABLE_ID, DIVU_TABLE_ID, DIV_TABLE_ID, EQ_TABLE_ID, MULHSU_TABLE_ID, MULHU_TABLE_ID,
    MULH_TABLE_ID, MUL_TABLE_ID, NEQ_TABLE_ID, OR_TABLE_ID, REMU_TABLE_ID, REM_TABLE_ID, RV32_XLEN, SLL_TABLE_ID,
    SLTU_TABLE_ID, SLT_TABLE_ID, SRA_TABLE_ID, SRL_TABLE_ID, SUB_TABLE_ID, XOR_TABLE_ID,
};

fn pow2_u64(i: usize) -> u64 {
    1u64 << i
}

fn enforce_u32_bits(
    constraints: &mut Vec<Constraint<F>>,
    one: usize,
    value_col: usize,
    bits_start: usize,
    chunk_size: usize,
    j: usize,
) {
    // bit_i * (1 - bit_i) = 0 for each bit.
    for bit in 0..32 {
        let b = bits_start + bit * chunk_size + j;
        constraints.push(Constraint {
            condition_col: b,
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (b, -F::ONE)],
            c_terms: Vec::new(),
        });
    }

    // value = sum_i 2^i * bit_i
    let mut terms = vec![(value_col, F::ONE)];
    for bit in 0..32 {
        let b = bits_start + bit * chunk_size + j;
        terms.push((b, -F::from_u64(pow2_u64(bit))));
    }
    constraints.push(Constraint::terms(one, false, terms));
}

fn push_rv32m_sidecar_constraints(
    constraints: &mut Vec<Constraint<F>>,
    layout: &Rv32B1Layout,
    j: usize,
    sltu_enabled: bool,
) {
    let one = layout.const_one;

    // mul_lo bits are used as scratch u32 bits:
    // - on MUL* rows, they decompose mul_lo,
    // - on DIV*/REM* rows, they decompose div_quot.
    //
    // The bits are always boolean, but the reconstruction constraint is gated by the opcode family.
    for bit in 0..32 {
        let b = layout.mul_lo_bit(bit, j);
        constraints.push(Constraint {
            condition_col: b,
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (b, -F::ONE)],
            c_terms: Vec::new(),
        });
    }

    // On MUL* rows: mul_lo = Σ 2^i * mul_lo_bit[i]
    {
        let mut terms = vec![(layout.mul_lo(j), F::ONE)];
        for bit in 0..32 {
            terms.push((layout.mul_lo_bit(bit, j), -F::from_u64(pow2_u64(bit))));
        }
        constraints.push(Constraint::terms_or(
            &[
                layout.is_mul(j),
                layout.is_mulh(j),
                layout.is_mulhu(j),
                layout.is_mulhsu(j),
            ],
            false,
            terms,
        ));
    }

    enforce_u32_bits(
        constraints,
        one,
        layout.mul_hi(j),
        layout.mul_hi_bits_start,
        layout.chunk_size,
        j,
    );

    // Disambiguate the MUL decomposition in Goldilocks by ruling out `mul_hi == 0xffff_ffff`.
    //
    // For 32-bit operands, the true 64-bit product has `mul_hi <= 0xffff_fffe`. Without this,
    // the field equation `rs1*rs2 = mul_lo + 2^32*mul_hi (mod p)` also admits the solution
    // `mul_lo + 2^32*mul_hi = rs1*rs2 + p` when `rs1*rs2 <= 2^32-2`, where `p = 2^64-2^32+1`.
    //
    // We enforce `mul_hi != 0xffff_ffff` by constraining `∏_{i=0..31} mul_hi_bit[i] = 0`.
    constraints.push(Constraint::terms(
        one,
        false,
        vec![(layout.mul_hi_prefix(0, j), F::ONE), (layout.mul_hi_bit(0, j), -F::ONE)],
    ));
    for k in 1..31 {
        constraints.push(Constraint::mul(
            layout.mul_hi_prefix(k - 1, j),
            layout.mul_hi_bit(k, j),
            layout.mul_hi_prefix(k, j),
        ));
    }
    constraints.push(Constraint {
        condition_col: layout.mul_hi_prefix(30, j),
        negate_condition: false,
        additional_condition_cols: Vec::new(),
        b_terms: vec![(layout.mul_hi_bit(31, j), F::ONE)],
        c_terms: Vec::new(),
    });

    // mul_carry bits (0..3, but only 0..2 will satisfy the MULH equations).
    for bit in 0..2 {
        let b = layout.mul_carry_bit(bit, j);
        constraints.push(Constraint {
            condition_col: b,
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (b, -F::ONE)], // 1 - b
            c_terms: Vec::new(),
        });
    }
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.mul_carry(j), F::ONE),
            (layout.mul_carry_bit(0, j), -F::ONE),
            (layout.mul_carry_bit(1, j), -F::from_u64(2)),
        ],
    ));

    // MUL decomposition (always enforced): rs1_val * rs2_val = mul_lo + 2^32 * mul_hi.
    constraints.push(Constraint {
        condition_col: layout.rs1_val(j),
        negate_condition: false,
        additional_condition_cols: Vec::new(),
        b_terms: vec![(layout.rs2_val(j), F::ONE)],
        c_terms: vec![
            (layout.mul_lo(j), F::ONE),
            (layout.mul_hi(j), F::from_u64(pow2_u64(32))),
        ],
    });

    // MUL/MULHU writeback.
    constraints.push(Constraint::terms(
        layout.is_mul(j),
        false,
        vec![(layout.rd_write_val(j), F::ONE), (layout.mul_lo(j), -F::ONE)],
    ));
    constraints.push(Constraint::terms(
        layout.is_mulhu(j),
        false,
        vec![(layout.rd_write_val(j), F::ONE), (layout.mul_hi(j), -F::ONE)],
    ));

    // rs1_bit[i] ∈ {0,1}
    for bit in 0..32 {
        let b = layout.rs1_bit(bit, j);
        constraints.push(Constraint {
            condition_col: b,
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (b, -F::ONE)], // 1 - b
            c_terms: Vec::new(),
        });
    }

    // rs1_val = Σ 2^i * rs1_bit[i]
    {
        let mut terms = vec![(layout.rs1_val(j), F::ONE)];
        for bit in 0..32 {
            terms.push((layout.rs1_bit(bit, j), -F::from_u64(pow2_u64(bit))));
        }
        constraints.push(Constraint::terms(one, false, terms));
    }

    let rs1_sign = layout.rs1_bit(31, j);
    let rs2_sign = layout.rs2_bit(31, j);

    // rs1_abs / rs2_abs from two's-complement sign bits.
    constraints.push(Constraint::terms(
        rs1_sign,
        true,
        vec![(layout.rs1_abs(j), F::ONE), (layout.rs1_val(j), -F::ONE)],
    ));
    constraints.push(Constraint::terms(
        rs1_sign,
        false,
        vec![
            (layout.rs1_abs(j), F::ONE),
            (layout.rs1_val(j), F::ONE),
            (one, -F::from_u64(pow2_u64(32))),
        ],
    ));
    constraints.push(Constraint::terms(
        rs2_sign,
        true,
        vec![(layout.rs2_abs(j), F::ONE), (layout.rs2_val(j), -F::ONE)],
    ));
    constraints.push(Constraint::terms(
        rs2_sign,
        false,
        vec![
            (layout.rs2_abs(j), F::ONE),
            (layout.rs2_val(j), F::ONE),
            (one, -F::from_u64(pow2_u64(32))),
        ],
    ));

    // Sign helpers.
    constraints.push(Constraint::mul(rs1_sign, rs2_sign, layout.rs1_rs2_sign_and(j)));
    constraints.push(Constraint::mul(rs1_sign, layout.rs2_val(j), layout.rs1_sign_rs2_val(j)));
    constraints.push(Constraint::mul(rs2_sign, layout.rs1_val(j), layout.rs2_sign_rs1_val(j)));

    // MULH/MULHSU writeback with signed correction.
    constraints.push(Constraint::terms(
        layout.is_mulh(j),
        false,
        vec![
            (layout.rd_write_val(j), F::ONE),
            (layout.mul_carry(j), F::from_u64(pow2_u64(32))),
            (layout.mul_hi(j), -F::ONE),
            (layout.rs1_sign_rs2_val(j), F::ONE),
            (layout.rs2_sign_rs1_val(j), F::ONE),
            (layout.rs1_rs2_sign_and(j), -F::from_u64(pow2_u64(32))),
            (one, -F::from_u64(pow2_u64(32))),
        ],
    ));
    constraints.push(Constraint::terms(
        layout.is_mulhsu(j),
        false,
        vec![
            (layout.rd_write_val(j), F::ONE),
            (layout.mul_carry(j), F::from_u64(pow2_u64(32))),
            (layout.mul_hi(j), -F::ONE),
            (layout.rs1_sign_rs2_val(j), F::ONE),
            (one, -F::from_u64(pow2_u64(32))),
        ],
    ));

    if !sltu_enabled {
        return;
    }

    // On DIV*/REM* rows: div_quot = Σ 2^i * mul_lo_bit[i].
    //
    // This prevents mod-p wraparound witnesses in the DIV/REM equation.
    {
        let mut terms = vec![(layout.div_quot(j), F::ONE)];
        for bit in 0..32 {
            terms.push((layout.mul_lo_bit(bit, j), -F::from_u64(pow2_u64(bit))));
        }
        constraints.push(Constraint::terms_or(
            &[layout.is_div(j), layout.is_divu(j), layout.is_rem(j), layout.is_remu(j)],
            false,
            terms,
        ));
    }

    // Prefix product chain for Π_{i=0..31} (1 - rs2_bit[i]).
    // prefix[0] = (1 - b0)
    constraints.push(Constraint {
        condition_col: one,
        negate_condition: false,
        additional_condition_cols: Vec::new(),
        b_terms: vec![(one, F::ONE), (layout.rs2_bit(0, j), -F::ONE)],
        c_terms: vec![(layout.rs2_zero_prefix(0, j), F::ONE)],
    });
    // prefix[k] = prefix[k-1] * (1 - b_k) for k=1..30
    for k in 1..31 {
        constraints.push(Constraint {
            condition_col: layout.rs2_zero_prefix(k - 1, j),
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (layout.rs2_bit(k, j), -F::ONE)],
            c_terms: vec![(layout.rs2_zero_prefix(k, j), F::ONE)],
        });
    }
    // rs2_is_zero = prefix[30] * (1 - b_31)
    constraints.push(Constraint {
        condition_col: layout.rs2_zero_prefix(30, j),
        negate_condition: false,
        additional_condition_cols: Vec::new(),
        b_terms: vec![(one, F::ONE), (layout.rs2_bit(31, j), -F::ONE)],
        c_terms: vec![(layout.rs2_is_zero(j), F::ONE)],
    });

    // rs2_nonzero = 1 - rs2_is_zero.
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.rs2_nonzero(j), F::ONE),
            (layout.rs2_is_zero(j), F::ONE),
            (one, -F::ONE),
        ],
    ));

    // is_divu_or_remu = is_divu + is_remu.
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.is_divu_or_remu(j), F::ONE),
            (layout.is_divu(j), -F::ONE),
            (layout.is_remu(j), -F::ONE),
        ],
    ));

    // is_div_or_rem = is_div + is_rem.
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.is_div_or_rem(j), F::ONE),
            (layout.is_div(j), -F::ONE),
            (layout.is_rem(j), -F::ONE),
        ],
    ));

    // div_rem_check (unsigned) = is_divu_or_remu * rs2_nonzero.
    constraints.push(Constraint::mul(
        layout.is_divu_or_remu(j),
        layout.rs2_nonzero(j),
        layout.div_rem_check(j),
    ));

    // div_rem_check_signed = is_div_or_rem * rs2_nonzero.
    constraints.push(Constraint::mul(
        layout.is_div_or_rem(j),
        layout.rs2_nonzero(j),
        layout.div_rem_check_signed(j),
    ));

    // divu_by_zero = is_divu * rs2_is_zero.
    constraints.push(Constraint::mul(
        layout.is_divu(j),
        layout.rs2_is_zero(j),
        layout.divu_by_zero(j),
    ));

    // div_by_zero / div_nonzero for signed DIV.
    constraints.push(Constraint::mul(
        layout.is_div(j),
        layout.rs2_is_zero(j),
        layout.div_by_zero(j),
    ));
    constraints.push(Constraint::mul(
        layout.is_div(j),
        layout.rs2_nonzero(j),
        layout.div_nonzero(j),
    ));

    // rem_nonzero / rem_by_zero for signed REM.
    constraints.push(Constraint::mul(
        layout.is_rem(j),
        layout.rs2_nonzero(j),
        layout.rem_nonzero(j),
    ));
    constraints.push(Constraint::mul(
        layout.is_rem(j),
        layout.rs2_is_zero(j),
        layout.rem_by_zero(j),
    ));

    // DIVU by zero: quotient must be all 1s.
    constraints.push(Constraint::terms(
        layout.divu_by_zero(j),
        false,
        vec![(layout.div_quot(j), F::ONE), (one, -F::from_u64(u32::MAX as u64))],
    ));

    // div_divisor selects rs2_val (unsigned) or rs2_abs (signed).
    constraints.push(Constraint::terms(
        layout.is_divu_or_remu(j),
        false,
        vec![(layout.div_divisor(j), F::ONE), (layout.rs2_val(j), -F::ONE)],
    ));
    constraints.push(Constraint::terms(
        layout.is_div_or_rem(j),
        false,
        vec![(layout.div_divisor(j), F::ONE), (layout.rs2_abs(j), -F::ONE)],
    ));

    // div_prod = div_divisor * div_quot (always computed).
    constraints.push(Constraint::mul(
        layout.div_divisor(j),
        layout.div_quot(j),
        layout.div_prod(j),
    ));

    // Unsigned: dividend = divisor * quotient + remainder.
    constraints.push(Constraint::terms(
        layout.is_divu_or_remu(j),
        false,
        vec![
            (layout.rs1_val(j), F::ONE),
            (layout.div_prod(j), -F::ONE),
            (layout.div_rem(j), -F::ONE),
        ],
    ));

    // Signed: |dividend| = |divisor| * quotient + remainder (divisor != 0).
    constraints.push(Constraint::terms(
        layout.div_rem_check_signed(j),
        false,
        vec![
            (layout.rs1_abs(j), F::ONE),
            (layout.div_prod(j), -F::ONE),
            (layout.div_rem(j), -F::ONE),
        ],
    ));

    // div_sign = rs1_sign XOR rs2_sign.
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.div_sign(j), F::ONE),
            (rs1_sign, -F::ONE),
            (rs2_sign, -F::ONE),
            (layout.rs1_rs2_sign_and(j), F::from_u64(2)),
        ],
    ));
    // div_sign boolean.
    constraints.push(Constraint::terms(
        layout.div_sign(j),
        false,
        vec![(layout.div_sign(j), F::ONE), (one, -F::ONE)],
    ));

    // div_quot_carry / div_rem_carry bits (used to normalize negative zero).
    for &carry in &[layout.div_quot_carry(j), layout.div_rem_carry(j)] {
        constraints.push(Constraint {
            condition_col: carry,
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (carry, -F::ONE)], // 1 - carry
            c_terms: Vec::new(),
        });
    }
    // If sign=0, carry must be 0.
    constraints.push(Constraint::terms(
        layout.div_sign(j),
        true,
        vec![(layout.div_quot_carry(j), F::ONE)],
    ));
    constraints.push(Constraint::terms(
        rs1_sign,
        true,
        vec![(layout.div_rem_carry(j), F::ONE)],
    ));

    // Signed quotient / remainder (two's complement, with carry to allow -0 -> 0).
    constraints.push(Constraint::terms(
        layout.div_sign(j),
        true,
        vec![(layout.div_quot_signed(j), F::ONE), (layout.div_quot(j), -F::ONE)],
    ));
    constraints.push(Constraint::terms(
        layout.div_sign(j),
        false,
        vec![
            (layout.div_quot_signed(j), F::ONE),
            (layout.div_quot_carry(j), F::from_u64(pow2_u64(32))),
            (layout.div_quot(j), F::ONE),
            (one, -F::from_u64(pow2_u64(32))),
        ],
    ));
    constraints.push(Constraint::terms(
        rs1_sign,
        true,
        vec![(layout.div_rem_signed(j), F::ONE), (layout.div_rem(j), -F::ONE)],
    ));
    constraints.push(Constraint::terms(
        rs1_sign,
        false,
        vec![
            (layout.div_rem_signed(j), F::ONE),
            (layout.div_rem_carry(j), F::from_u64(pow2_u64(32))),
            (layout.div_rem(j), F::ONE),
            (one, -F::from_u64(pow2_u64(32))),
        ],
    ));

    // Writeback for DIVU/REMU.
    constraints.push(Constraint::terms(
        layout.is_divu(j),
        false,
        vec![(layout.rd_write_val(j), F::ONE), (layout.div_quot(j), -F::ONE)],
    ));
    constraints.push(Constraint::terms(
        layout.is_remu(j),
        false,
        vec![(layout.rd_write_val(j), F::ONE), (layout.div_rem(j), -F::ONE)],
    ));

    // Writeback for DIV (signed): divisor != 0 uses signed quotient, divisor == 0 yields -1.
    constraints.push(Constraint::terms(
        layout.div_nonzero(j),
        false,
        vec![(layout.rd_write_val(j), F::ONE), (layout.div_quot_signed(j), -F::ONE)],
    ));
    constraints.push(Constraint::terms(
        layout.div_by_zero(j),
        false,
        vec![(layout.rd_write_val(j), F::ONE), (one, -F::from_u64(u32::MAX as u64))],
    ));

    // Writeback for REM (signed): signed remainder (dividend sign).
    constraints.push(Constraint::terms(
        layout.is_rem(j),
        false,
        vec![(layout.rd_write_val(j), F::ONE), (layout.div_rem_signed(j), -F::ONE)],
    ));
    constraints.push(Constraint::terms(
        layout.rem_by_zero(j),
        false,
        vec![(layout.rd_write_val(j), F::ONE), (layout.rs1_val(j), -F::ONE)],
    ));

    // For divisor != 0, require remainder < divisor via a SLTU Shout lookup.
    constraints.push(Constraint::terms(
        layout.div_rem_check(j),
        false,
        vec![(layout.alu_out(j), F::ONE), (one, -F::ONE)],
    ));
    constraints.push(Constraint::terms(
        layout.div_rem_check_signed(j),
        false,
        vec![(layout.alu_out(j), F::ONE), (one, -F::ONE)],
    ));
}

/// Build an RV32M “sidecar” CCS for RV32 B1 chunks.
///
/// This CCS intentionally contains only the MUL/DIV/REM helper constraints that we no longer
/// include in the main RV32 B1 step CCS. It is meant to be proven/verified as an **additional**
/// argument whenever the guest program uses RV32M ops.
pub fn build_rv32_b1_rv32m_sidecar_ccs(layout: &Rv32B1Layout) -> Result<CcsStructure<F>, String> {
    let mut constraints: Vec<Constraint<F>> = Vec::new();
    let sltu_enabled = layout.table_ids.binary_search(&SLTU_TABLE_ID).is_ok();

    for j in 0..layout.chunk_size {
        push_rv32m_sidecar_constraints(&mut constraints, layout, j, sltu_enabled);
    }

    let n = constraints.len();
    build_r1cs_ccs(&constraints, n, layout.m, layout.const_one)
}

/// Build an RV32M “event” sidecar CCS for a subset of lanes in an RV32 B1 chunk.
///
/// This is a **sparse-over-time** variant of `build_rv32_b1_rv32m_sidecar_ccs` intended for
/// `chunk_size > 1`, where paying the full RV32M helper gadget on every lane of a chunk is
/// wasteful when RV32M instructions are rare.
///
/// The CCS includes RV32M helper constraints only for the selected lanes, plus a per-selected-lane
/// guard constraint requiring that exactly one RV32M opcode flag is set on that lane. This makes it
/// sound for the verifier to accept a proof that only checks the selected subset:
/// - the guard forces every selected lane to actually be an RV32M instruction, and
/// - the decode plumbing sidecar proves the public `rv32m_count`, so selecting exactly `rv32m_count`
///   lanes implies all RV32M lanes are covered.
pub fn build_rv32_b1_rv32m_event_sidecar_ccs(
    layout: &Rv32B1Layout,
    selected_lanes: &[usize],
) -> Result<CcsStructure<F>, String> {
    if selected_lanes.is_empty() {
        return Err("RV32M event sidecar: selected_lanes must be non-empty".into());
    }

    let mut lanes: Vec<usize> = selected_lanes.to_vec();
    lanes.sort_unstable();
    lanes.dedup();
    if lanes.len() != selected_lanes.len() {
        return Err("RV32M event sidecar: selected_lanes must be unique".into());
    }
    if let Some(&max_lane) = lanes.last() {
        if max_lane >= layout.chunk_size {
            return Err(format!(
                "RV32M event sidecar: lane index out of range: lane={max_lane} (chunk_size={})",
                layout.chunk_size
            ));
        }
    }

    let one = layout.const_one;
    let sltu_enabled = layout.table_ids.binary_search(&SLTU_TABLE_ID).is_ok();

    let mut constraints: Vec<Constraint<F>> = Vec::new();
    for &j in &lanes {
        // Guard: selected lanes must be RV32M (exactly one of the 8 RV32M op flags is set).
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.is_mul(j), F::ONE),
                (layout.is_mulh(j), F::ONE),
                (layout.is_mulhu(j), F::ONE),
                (layout.is_mulhsu(j), F::ONE),
                (layout.is_div(j), F::ONE),
                (layout.is_divu(j), F::ONE),
                (layout.is_rem(j), F::ONE),
                (layout.is_remu(j), F::ONE),
                (one, -F::ONE),
            ],
        ));

        push_rv32m_sidecar_constraints(&mut constraints, layout, j, sltu_enabled);
    }

    let n = constraints.len();
    build_r1cs_ccs(&constraints, n, layout.m, layout.const_one)
}

fn rv32_b1_semantic_constraints_impl(
    layout: &Rv32B1Layout,
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    include_decode: bool,
) -> Result<Vec<Constraint<F>>, String> {
    let one = layout.const_one;

    let mut constraints = Vec::<Constraint<F>>::new();

    let shout_cols = |table_id: u32| {
        layout
            .table_ids
            .binary_search(&table_id)
            .ok()
            .map(|idx| &layout.bus.shout_cols[idx].lanes[0])
    };

    // The ADD table is required because this circuit uses it for address/ALU wiring (LW/SW/AUIPC/JALR).
    let add_shout_idx = layout.shout_idx(ADD_TABLE_ID)?;
    let add_cols = &layout.bus.shout_cols[add_shout_idx].lanes[0];

    let and_cols = shout_cols(AND_TABLE_ID);
    let xor_cols = shout_cols(XOR_TABLE_ID);
    let or_cols = shout_cols(OR_TABLE_ID);
    let sub_cols = shout_cols(SUB_TABLE_ID);
    let slt_cols = shout_cols(SLT_TABLE_ID);
    let sltu_cols = shout_cols(SLTU_TABLE_ID);
    let sll_cols = shout_cols(SLL_TABLE_ID);
    let srl_cols = shout_cols(SRL_TABLE_ID);
    let sra_cols = shout_cols(SRA_TABLE_ID);
    let eq_cols = shout_cols(EQ_TABLE_ID);
    let mul_cols = shout_cols(MUL_TABLE_ID);
    let mulh_cols = shout_cols(MULH_TABLE_ID);
    let mulhu_cols = shout_cols(MULHU_TABLE_ID);
    let mulhsu_cols = shout_cols(MULHSU_TABLE_ID);
    let div_cols = shout_cols(DIV_TABLE_ID);
    let divu_cols = shout_cols(DIVU_TABLE_ID);
    let rem_cols = shout_cols(REM_TABLE_ID);
    let remu_cols = shout_cols(REMU_TABLE_ID);

    let ell_addr = 2 * RV32_XLEN;
    for (name, cols_opt) in [
        ("ADD", Some(add_cols)),
        ("AND", and_cols),
        ("XOR", xor_cols),
        ("OR", or_cols),
        ("SUB", sub_cols),
        ("SLT", slt_cols),
        ("SLTU", sltu_cols),
        ("SLL", sll_cols),
        ("SRL", srl_cols),
        ("SRA", sra_cols),
        ("EQ", eq_cols),
        ("MUL", mul_cols),
        ("MULH", mulh_cols),
        ("MULHU", mulhu_cols),
        ("MULHSU", mulhsu_cols),
        ("DIV", div_cols),
        ("DIVU", divu_cols),
        ("REM", rem_cols),
        ("REMU", remu_cols),
    ] {
        if let Some(cols) = cols_opt {
            if cols.addr_bits.end - cols.addr_bits.start != ell_addr {
                return Err(format!(
                    "{name} shout bus layout mismatch: expected ell_addr={ell_addr}, got {}",
                    cols.addr_bits.end - cols.addr_bits.start
                ));
            }
        }
    }

    // If a Shout table isn't included, forbid the corresponding instruction variants.
    //
    // These are sound as a single linear constraint per step: all flags are boolean, so
    // `sum(forbidden_flags)=0` implies each forbidden flag is 0 (the sum range is tiny vs field size).
    let forbid_and = and_cols.is_none();
    let forbid_or = or_cols.is_none();
    let forbid_xor = xor_cols.is_none();
    let forbid_sub = sub_cols.is_none();
    let forbid_sll = sll_cols.is_none();
    let forbid_srl = srl_cols.is_none();
    let forbid_sra = sra_cols.is_none();
    let forbid_slt = slt_cols.is_none();
    let forbid_sltu = sltu_cols.is_none();
    let forbid_eq = eq_cols.is_none();
    for j in 0..layout.chunk_size {
        let mut forbidden = Vec::new();
        if forbid_and {
            forbidden.push((layout.and_has_lookup(j), F::ONE));
        }
        if forbid_or {
            forbidden.push((layout.or_has_lookup(j), F::ONE));
        }
        if forbid_xor {
            forbidden.push((layout.xor_has_lookup(j), F::ONE));
        }
        if forbid_sub {
            forbidden.push((layout.sub_has_lookup(j), F::ONE));
        }
        if forbid_sll {
            forbidden.push((layout.sll_has_lookup(j), F::ONE));
        }
        if forbid_srl {
            forbidden.push((layout.srl_has_lookup(j), F::ONE));
        }
        if forbid_sra {
            forbidden.push((layout.sra_has_lookup(j), F::ONE));
        }
        if forbid_slt {
            forbidden.push((layout.slt_has_lookup(j), F::ONE));
        }
        if forbid_sltu {
            forbidden.push((layout.sltu_has_lookup(j), F::ONE));
            // DIVU/REMU need SLTU to prove `rem < divisor` when divisor != 0.
            forbidden.push((layout.is_divu(j), F::ONE));
            forbidden.push((layout.is_remu(j), F::ONE));
            // DIV/REM need SLTU for the signed remainder bound check.
            forbidden.push((layout.is_div(j), F::ONE));
            forbidden.push((layout.is_rem(j), F::ONE));
        }
        if forbid_eq {
            forbidden.push((layout.eq_has_lookup(j), F::ONE));
        }
        if !forbidden.is_empty() {
            constraints.push(Constraint::terms(one, false, forbidden));
        }
    }
    let _ = (
        mul_cols,
        mulh_cols,
        mulhu_cols,
        mulhsu_cols,
        div_cols,
        divu_cols,
        rem_cols,
        remu_cols,
    );

    // Alignment constraints require bit-addressed memories (n_side=2).
    let prog_id = PROG_ID.0;
    let prog_layout = mem_layouts
        .get(&prog_id)
        .ok_or_else(|| format!("mem_layouts missing PROG_ID={prog_id}"))?;
    if prog_layout.n_side != 2 || prog_layout.d < 2 {
        return Err("RV32 B1: PROG_ID must use n_side=2 and d>=2 (bit addressing)".into());
    }
    let ram_id = RAM_ID.0;
    let ram_layout = mem_layouts
        .get(&ram_id)
        .ok_or_else(|| format!("mem_layouts missing RAM_ID={ram_id}"))?;
    if ram_layout.n_side != 2 || ram_layout.d < 2 {
        return Err("RV32 B1: RAM_ID must use n_side=2 and d>=2 (bit addressing)".into());
    }

    let prog = &layout.bus.twist_cols[layout.prog_twist_idx].lanes[0];
    let ram = &layout.bus.twist_cols[layout.ram_twist_idx].lanes[0];

    let pack_interleaved_operand =
        |addr_bits_start: usize, j: usize, parity: usize, value_col: usize| -> Vec<(usize, F)> {
            debug_assert!(parity == 0 || parity == 1, "parity must be 0 (even) or 1 (odd)");
            let mut terms = vec![(value_col, F::ONE)];
            for i in 0..RV32_XLEN {
                let bit_col_id = addr_bits_start + 2 * i + parity;
                let bit = layout.bus.bus_cell(bit_col_id, j);
                terms.push((bit, -F::from_u64(pow2_u64(i))));
            }
            terms
        };

    // --- Public I/O binding (initial + final PC) ---
    // Initial PC binds to lane 0.
    let j0 = 0usize;
    constraints.push(Constraint::terms(
        one,
        false,
        vec![(layout.pc_in(j0), F::ONE), (layout.pc0, -F::ONE)],
    ));

    // Final PC binds to the last lane.
    let j_last = layout.chunk_size - 1;
    constraints.push(Constraint::terms(
        one,
        false,
        vec![(layout.pc_out(j_last), F::ONE), (layout.pc_final, -F::ONE)],
    ));

    // --- Cross-chunk halting / padding semantics (L1-style) ---
    // halted_in/out are booleans.
    constraints.push(Constraint::terms(
        layout.halted_in,
        false,
        vec![(layout.halted_in, F::ONE), (one, -F::ONE)],
    ));
    constraints.push(Constraint::terms(
        layout.halted_out,
        false,
        vec![(layout.halted_out, F::ONE), (one, -F::ONE)],
    ));

    // halted_in + is_active[0] = 1 (chunk starts active iff not halted).
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.halted_in, F::ONE),
            (layout.is_active(j0), F::ONE),
            (one, -F::ONE),
        ],
    ));

    // halted_out = 1 - is_active[last] + halt_effective[last].
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.halted_out, F::ONE),
            (layout.is_active(j_last), F::ONE),
            (layout.halt_effective(j_last), -F::ONE),
            (one, -F::ONE),
        ],
    ));

    for j in 0..layout.chunk_size {
        let is_active = layout.is_active(j);
        let pc_in = layout.pc_in(j);
        let pc_out = layout.pc_out(j);
        let add_a0 = layout.bus.bus_cell(add_cols.addr_bits.start + 0, j);
        let add_b0 = layout.bus.bus_cell(add_cols.addr_bits.start + 1, j);

        // Dedicated zero column.
        constraints.push(Constraint::zero(one, layout.zero(j)));

        // is_active is boolean.
        constraints.push(Constraint::terms(
            is_active,
            false,
            vec![(is_active, F::ONE), (one, -F::ONE)],
        ));

        // Inactive rows keep PC constant: (1 - is_active) * (pc_out - pc_in) = 0.
        constraints.push(Constraint::terms(
            is_active,
            true,
            vec![(pc_out, F::ONE), (pc_in, -F::ONE)],
        ));

        if include_decode {
            push_rv32_b1_decode_constraints(&mut constraints, layout, j)?;
        }

        // --------------------------------------------------------------------
        // Regfile-as-Twist glue
        // --------------------------------------------------------------------

        // rd_is_zero = 1 iff the decoded rd field is 0.
        //
        // Since `rd_field` is a 5-bit value (instr bits [11:7]), we can compute:
        // rd_is_zero_01   = (1-b0) * (1-b1)
        // rd_is_zero_012  = rd_is_zero_01  * (1-b2)
        // rd_is_zero_0123 = rd_is_zero_012 * (1-b3)
        // rd_is_zero      = rd_is_zero_0123 * (1-b4)
        let rd_b0 = layout.rd_bit(0, j);
        let rd_b1 = layout.rd_bit(1, j);
        let rd_b2 = layout.rd_bit(2, j);
        let rd_b3 = layout.rd_bit(3, j);
        let rd_b4 = layout.rd_bit(4, j);
        constraints.push(Constraint {
            condition_col: rd_b0,
            negate_condition: true,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (rd_b1, -F::ONE)],
            c_terms: vec![(layout.rd_is_zero_01(j), F::ONE)],
        });
        constraints.push(Constraint {
            condition_col: layout.rd_is_zero_01(j),
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (rd_b2, -F::ONE)],
            c_terms: vec![(layout.rd_is_zero_012(j), F::ONE)],
        });
        constraints.push(Constraint {
            condition_col: layout.rd_is_zero_012(j),
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (rd_b3, -F::ONE)],
            c_terms: vec![(layout.rd_is_zero_0123(j), F::ONE)],
        });
        constraints.push(Constraint {
            condition_col: layout.rd_is_zero_0123(j),
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (rd_b4, -F::ONE)],
            c_terms: vec![(layout.rd_is_zero(j), F::ONE)],
        });

        // reg_has_write = writes_rd * (1 - rd_is_zero)
        //
        // This:
        // - disables writes to x0 (rd==0) soundly without inverse gadgets, and
        // - keeps rd_write_val semantics unchanged (it can be "junk" when rd==0).
        //
        // Note: `writes_rd` is a boolean group signal proven by the decode plumbing sidecar.
        constraints.push(Constraint {
            condition_col: layout.writes_rd(j),
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(one, F::ONE), (layout.rd_is_zero(j), -F::ONE)],
            c_terms: vec![(layout.reg_has_write(j), F::ONE)],
        });

        // ECALL always halts in RV32 B1: halt_effective = is_halt.
        constraints.push(Constraint::terms(
            one,
            false,
            vec![(layout.halt_effective(j), F::ONE), (layout.is_halt(j), -F::ONE)],
        ));

        // --------------------------------------------------------------------
        // RV32M sparse event columns (for M-event arguments)
        // --------------------------------------------------------------------

        // rv32m_{rs1,rs2,rd_write}_val must be:
        // - 0 on non-RV32M rows, and
        // - equal to the corresponding full column on RV32M rows.
        //
        // Since RV32M op flags are one-hot, their sum is a 0/1 gate.
        let rv32m_flags = [
            layout.is_mul(j),
            layout.is_mulh(j),
            layout.is_mulhu(j),
            layout.is_mulhsu(j),
            layout.is_div(j),
            layout.is_divu(j),
            layout.is_rem(j),
            layout.is_remu(j),
        ];
        constraints.push(Constraint {
            condition_col: rv32m_flags[0],
            negate_condition: false,
            additional_condition_cols: rv32m_flags[1..].to_vec(),
            b_terms: vec![(layout.rs1_val(j), F::ONE)],
            c_terms: vec![(layout.rv32m_rs1_val(j), F::ONE)],
        });
        constraints.push(Constraint {
            condition_col: rv32m_flags[0],
            negate_condition: false,
            additional_condition_cols: rv32m_flags[1..].to_vec(),
            b_terms: vec![(layout.rs2_val(j), F::ONE)],
            c_terms: vec![(layout.rv32m_rs2_val(j), F::ONE)],
        });
        constraints.push(Constraint {
            condition_col: rv32m_flags[0],
            negate_condition: false,
            additional_condition_cols: rv32m_flags[1..].to_vec(),
            b_terms: vec![(layout.rd_write_val(j), F::ONE)],
            c_terms: vec![(layout.rv32m_rd_write_val(j), F::ONE)],
        });

        // --------------------------------------------------------------------
        // Always-on memory/store safety plumbing
        // --------------------------------------------------------------------

        // Range-check mem_rv to 32 bits so byte/half extraction is sound.
        enforce_u32_bits(
            &mut constraints,
            one,
            layout.mem_rv(j),
            layout.mem_rv_bits_start,
            layout.chunk_size,
            j,
        );

        // rs2_bit[i] ∈ {0,1}
        for bit in 0..32 {
            let b = layout.rs2_bit(bit, j);
            constraints.push(Constraint {
                condition_col: b,
                negate_condition: false,
                additional_condition_cols: Vec::new(),
                b_terms: vec![(one, F::ONE), (b, -F::ONE)], // 1 - b
                c_terms: Vec::new(),
            });
        }

        // rs2_val = Σ 2^i * rs2_bit[i]
        {
            let mut terms = vec![(layout.rs2_val(j), F::ONE)];
            for bit in 0..32 {
                terms.push((layout.rs2_bit(bit, j), -F::from_u64(pow2_u64(bit))));
            }
            constraints.push(Constraint::terms(one, false, terms));
        }

        // ALU right operand helper:
        // - ALU reg: rhs = rs2_val
        // - ALU imm: rhs = imm_i
        //
        // This is used for Shout key wiring in the semantics sidecar (e.g. AND/OR/XOR/ADD/SLT/SLTU).
        constraints.push(Constraint::terms(
            layout.is_alu_reg(j),
            false,
            vec![(layout.alu_rhs(j), F::ONE), (layout.rs2_val(j), -F::ONE)],
        ));
        constraints.push(Constraint::terms(
            layout.is_alu_imm(j),
            false,
            vec![(layout.alu_rhs(j), F::ONE), (layout.imm_i(j), -F::ONE)],
        ));

        // Shift rhs helper:
        // - shift reg ops use `rs2_val`,
        // - shift imm ops use the 5-bit shamt field (instr[24:20]) which lives in `rs2_field`.
        //
        // We define a single scalar `shift_rhs` that selects the correct operand based on `is_alu_imm`.
        // It is safe for non-shift rows because `shift_rhs` is only used when a shift Shout table is active.
        constraints.push(Constraint {
            condition_col: layout.is_alu_imm(j),
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(layout.rs2_field(j), F::ONE), (layout.rs2_val(j), -F::ONE)],
            c_terms: vec![(layout.shift_rhs(j), F::ONE), (layout.rs2_val(j), -F::ONE)],
        });

        // RAM effective address is computed via the ADD Shout lookup (mod 2^32 semantics).
        constraints.push(Constraint::terms_or(
            &[
                layout.is_lb(j),
                layout.is_lbu(j),
                layout.is_lh(j),
                layout.is_lhu(j),
                layout.is_lw(j),
            ],
            false,
            vec![(layout.eff_addr(j), F::ONE), (layout.alu_out(j), -F::ONE)],
        ));
        constraints.push(Constraint::terms_or(
            &[layout.is_sb(j), layout.is_sh(j), layout.is_sw(j)],
            false,
            vec![(layout.eff_addr(j), F::ONE), (layout.alu_out(j), -F::ONE)],
        ));

        // Atomics use rs1 as the effective address (no immediate).
        constraints.push(Constraint::terms_or(
            &[
                layout.is_amoswap_w(j),
                layout.is_amoadd_w(j),
                layout.is_amoxor_w(j),
                layout.is_amoor_w(j),
                layout.is_amoand_w(j),
            ],
            false,
            vec![(layout.eff_addr(j), F::ONE), (layout.rs1_val(j), -F::ONE)],
        ));

        // RAM bus selectors must be derived from instruction flags to avoid bypassing Twist.
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.ram_has_read(j), F::ONE),
                (layout.is_lb(j), -F::ONE),
                (layout.is_lbu(j), -F::ONE),
                (layout.is_lh(j), -F::ONE),
                (layout.is_lhu(j), -F::ONE),
                (layout.is_lw(j), -F::ONE),
                (layout.is_sb(j), -F::ONE),
                (layout.is_sh(j), -F::ONE),
                (layout.is_amoswap_w(j), -F::ONE),
                (layout.is_amoadd_w(j), -F::ONE),
                (layout.is_amoxor_w(j), -F::ONE),
                (layout.is_amoor_w(j), -F::ONE),
                (layout.is_amoand_w(j), -F::ONE),
            ],
        ));
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.ram_has_write(j), F::ONE),
                (layout.is_sb(j), -F::ONE),
                (layout.is_sh(j), -F::ONE),
                (layout.is_sw(j), -F::ONE),
                (layout.is_amoswap_w(j), -F::ONE),
                (layout.is_amoadd_w(j), -F::ONE),
                (layout.is_amoxor_w(j), -F::ONE),
                (layout.is_amoor_w(j), -F::ONE),
                (layout.is_amoand_w(j), -F::ONE),
            ],
        ));

        // RAM write value (WV): SW and AMOSWAP use rs2, other AMOs use a Shout output.
        constraints.push(Constraint::terms_or(
            &[layout.is_sw(j), layout.is_amoswap_w(j)],
            false,
            vec![(layout.ram_wv(j), F::ONE), (layout.rs2_val(j), -F::ONE)],
        ));
        // SB/SH write: merge low byte/halfword into the existing word.
        {
            let mut terms = vec![(layout.ram_wv(j), F::ONE), (layout.mem_rv(j), -F::ONE)];
            for bit in 0..8 {
                let coeff = F::from_u64(pow2_u64(bit));
                terms.push((layout.mem_rv_bit(bit, j), coeff));
                terms.push((layout.rs2_bit(bit, j), -coeff));
            }
            constraints.push(Constraint::terms(layout.is_sb(j), false, terms));
        }
        {
            let mut terms = vec![(layout.ram_wv(j), F::ONE), (layout.mem_rv(j), -F::ONE)];
            for bit in 0..16 {
                let coeff = F::from_u64(pow2_u64(bit));
                terms.push((layout.mem_rv_bit(bit, j), coeff));
                terms.push((layout.rs2_bit(bit, j), -coeff));
            }
            constraints.push(Constraint::terms(layout.is_sh(j), false, terms));
        }
        for &f in &[
            layout.is_amoadd_w(j),
            layout.is_amoxor_w(j),
            layout.is_amoor_w(j),
            layout.is_amoand_w(j),
        ] {
            constraints.push(Constraint::terms(
                f,
                false,
                vec![(layout.ram_wv(j), F::ONE), (layout.alu_out(j), -F::ONE)],
            ));
        }

        // Shout selectors.
        //
        // These selectors are part of the shared-bus binding surface: if they are wrong, the prover
        // can bypass Shout by setting `has_lookup=0`. So even in the “decode/semantics sidecar”
        // architecture, we must constrain them somewhere. Here we keep the definitions in the
        // semantics CCS (they are cheap and tie directly to ISA semantics like remainder checks).

        // ADD table: used for:
        // - ADD/ADDI (add_alu)
        // - load/store address compute (is_load/is_store)
        // - AMOADD.W (mem_rv + rs2)
        // - AUIPC (pc + imm_u)
        // - JALR target (rs1 + imm_i)
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.add_has_lookup(j), F::ONE),
                (layout.add_alu(j), -F::ONE),
                (layout.is_load(j), -F::ONE),
                (layout.is_store(j), -F::ONE),
                (layout.is_amoadd_w(j), -F::ONE),
                (layout.is_auipc(j), -F::ONE),
                (layout.is_jalr(j), -F::ONE),
            ],
        ));

        // AND/XOR/OR tables: ALU (reg/imm) + AMO word ops.
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.and_has_lookup(j), F::ONE),
                (layout.and_alu(j), -F::ONE),
                (layout.is_amoand_w(j), -F::ONE),
            ],
        ));
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.xor_has_lookup(j), F::ONE),
                (layout.xor_alu(j), -F::ONE),
                (layout.is_amoxor_w(j), -F::ONE),
            ],
        ));
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.or_has_lookup(j), F::ONE),
                (layout.or_alu(j), -F::ONE),
                (layout.is_amoor_w(j), -F::ONE),
            ],
        ));

        // SLT/SLTU Shout activation:
        // - ALU SLT/SLTU use slt_alu/sltu_alu,
        // - branches use br_cmp_lt/br_cmp_ltu,
        // - DIV*/REM* remainder bounds use div_rem_check(_signed).
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.slt_has_lookup(j), F::ONE),
                (layout.slt_alu(j), -F::ONE),
                (layout.br_cmp_lt(j), -F::ONE),
            ],
        ));
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.sltu_has_lookup(j), F::ONE),
                (layout.sltu_alu(j), -F::ONE),
                (layout.br_cmp_ltu(j), -F::ONE),
                (layout.div_rem_check(j), -F::ONE),
                (layout.div_rem_check_signed(j), -F::ONE),
            ],
        ));

        // Twist RAM model (RV32 B1 / MVP):
        // - RAM is byte-addressed (the Twist bus address is the architectural byte address).
        // - Each Twist read/write is a full 32-bit word value at that byte address: the little-endian
        //   4-byte window starting at `eff_addr`.
        // - LB/LBU/LH/LHU derive rd_write_val from the low byte/halfword of `mem_rv`.
        // - SB/SH are proven as read-modify-write over that same word window: `ram_wv` equals `mem_rv`
        //   with the low byte/halfword replaced by rs2's low bits.
        //
        // Alignment is enforced later via the low address bits on the RAM Twist lane.

        // Instruction-specific writeback:
        // - Shout-backed ALU ops + AUIPC: rd_write_val = alu_out
        // - Loads/AMO: rd_write_val derived from mem_rv
        // - LUI: rd_write_val = imm_u
        // - JAL/JALR: rd_write_val = pc_in + 4
        //
        // `wb_from_alu` is proven in the decode plumbing sidecar, so the semantics CCS can stay
        // compact here.
        constraints.push(Constraint::terms(
            layout.wb_from_alu(j),
            false,
            vec![(layout.rd_write_val(j), F::ONE), (layout.alu_out(j), -F::ONE)],
        ));
        constraints.push(Constraint::terms_or(
            &[
                layout.is_lw(j),
                layout.is_amoswap_w(j),
                layout.is_amoadd_w(j),
                layout.is_amoxor_w(j),
                layout.is_amoor_w(j),
                layout.is_amoand_w(j),
            ],
            false,
            vec![(layout.rd_write_val(j), F::ONE), (layout.mem_rv(j), -F::ONE)],
        ));
        // LB: sign-extend low byte.
        {
            let mut terms = vec![(layout.rd_write_val(j), F::ONE)];
            for bit in 0..8 {
                let coeff = if bit == 7 {
                    F::from_u64(pow2_u64(32) - pow2_u64(7))
                } else {
                    F::from_u64(pow2_u64(bit))
                };
                terms.push((layout.mem_rv_bit(bit, j), -coeff));
            }
            constraints.push(Constraint::terms(layout.is_lb(j), false, terms));
        }
        // LBU: zero-extend low byte.
        {
            let mut terms = vec![(layout.rd_write_val(j), F::ONE)];
            for bit in 0..8 {
                terms.push((layout.mem_rv_bit(bit, j), -F::from_u64(pow2_u64(bit))));
            }
            constraints.push(Constraint::terms(layout.is_lbu(j), false, terms));
        }
        // LH: sign-extend low halfword.
        {
            let mut terms = vec![(layout.rd_write_val(j), F::ONE)];
            for bit in 0..16 {
                let coeff = if bit == 15 {
                    F::from_u64(pow2_u64(32) - pow2_u64(15))
                } else {
                    F::from_u64(pow2_u64(bit))
                };
                terms.push((layout.mem_rv_bit(bit, j), -coeff));
            }
            constraints.push(Constraint::terms(layout.is_lh(j), false, terms));
        }
        // LHU: zero-extend low halfword.
        {
            let mut terms = vec![(layout.rd_write_val(j), F::ONE)];
            for bit in 0..16 {
                terms.push((layout.mem_rv_bit(bit, j), -F::from_u64(pow2_u64(bit))));
            }
            constraints.push(Constraint::terms(layout.is_lhu(j), false, terms));
        }
        constraints.push(Constraint::terms(
            layout.is_lui(j),
            false,
            vec![(layout.rd_write_val(j), F::ONE), (layout.imm_u(j), -F::ONE)],
        ));

        // JAL/JALR writeback: rd_write_val = pc_in + 4.
        constraints.push(Constraint::terms_or(
            &[layout.is_jal(j), layout.is_jalr(j)],
            false,
            vec![
                (layout.rd_write_val(j), F::ONE),
                (pc_in, -F::ONE),
                (one, -F::from_u64(4)),
            ],
        ));

        // PC update:
        // - Straight-line (non-branch/non-jump) instructions: pc_out = pc_in + 4.
        constraints.push(Constraint::terms(
            layout.pc_plus4(j),
            false,
            vec![(pc_out, F::ONE), (pc_in, -F::ONE), (one, -F::from_u64(4))],
        ));

        // - JAL: pc_out = pc_in + imm_j.
        constraints.push(Constraint::terms(
            layout.is_jal(j),
            false,
            vec![(pc_out, F::ONE), (pc_in, -F::ONE), (layout.imm_j(j), -F::ONE)],
        ));

        // - JALR: pc_out = (rs1 + imm_i) & !1.
        //
        // The ADD-table Shout output `alu_out` is (rs1 + imm_i) mod 2^32.
        // Let a0,b0 be the operand LSBs (from the interleaved ADD key bits), and let a0b0 = a0*b0.
        // Then lsb(alu_out) = a0 XOR b0 = a0 + b0 - 2*a0b0, and pc_out = alu_out - lsb.
        constraints.push(Constraint::terms(
            layout.is_jalr(j),
            false,
            vec![
                (pc_out, F::ONE),
                (layout.alu_out(j), -F::ONE),
                (add_a0, F::ONE),
                (add_b0, F::ONE),
                (layout.add_a0b0(j), -F::from_u64(2)),
            ],
        ));

        // Branch control: br_taken/br_not_taken are only set on branch rows.
        constraints.push(Constraint::terms(
            layout.is_branch(j),
            true, // (1 - is_branch) * br_taken = 0
            vec![(layout.br_taken(j), F::ONE)],
        ));
        constraints.push(Constraint::terms(
            layout.is_branch(j),
            true, // (1 - is_branch) * br_not_taken = 0
            vec![(layout.br_not_taken(j), F::ONE)],
        ));

        // Exactly one branch outcome on branch rows: br_taken + br_not_taken = is_branch.
        constraints.push(Constraint::terms(
            one,
            false,
            vec![
                (layout.br_taken(j), F::ONE),
                (layout.br_not_taken(j), F::ONE),
                (layout.is_branch(j), -F::ONE),
            ],
        ));

        // Branch decision: br_taken = alu_out XOR br_invert (only on branch rows).
        //
        // Let p = alu_out * br_invert. Then:
        //   alu_out XOR br_invert = alu_out + br_invert - 2*p
        constraints.push(Constraint::mul(
            layout.alu_out(j),
            layout.br_invert(j),
            layout.br_invert_alu(j),
        ));
        constraints.push(Constraint::terms(
            layout.is_branch(j),
            false,
            vec![
                (layout.br_taken(j), F::ONE),
                (layout.alu_out(j), -F::ONE),
                (layout.br_invert(j), -F::ONE),
                (layout.br_invert_alu(j), F::from_u64(2)),
            ],
        ));

        // Branch PC update:
        // - Taken: pc_out = pc_in + imm_b.
        // - Not taken: pc_out = pc_in + 4.
        constraints.push(Constraint::terms(
            layout.br_taken(j),
            false,
            vec![(pc_out, F::ONE), (pc_in, -F::ONE), (layout.imm_b(j), -F::ONE)],
        ));
        constraints.push(Constraint::terms(
            layout.br_not_taken(j),
            false,
            vec![(pc_out, F::ONE), (pc_in, -F::ONE), (one, -F::from_u64(4))],
        ));

        // Helper: bind the product of the ADD-table operand LSBs (used for JALR mask in Phase 2).
        constraints.push(Constraint::mul(add_a0, add_b0, layout.add_a0b0(j)));

        // --- Shout key correctness (ADD table bus addr bits interleaving) ---
        let mut even_terms = vec![(layout.rs1_val(j), F::ONE)];
        for i in 0..RV32_XLEN {
            let bit_col_id = add_cols.addr_bits.start + 2 * i;
            let bit = layout.bus.bus_cell(bit_col_id, j);
            even_terms.push((bit, -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms_or(
            &[
                layout.add_alu(j),
                layout.is_load(j),
                layout.is_store(j),
                layout.is_jalr(j),
            ],
            false,
            even_terms,
        ));

        // AMOADD.W uses mem_rv as the left ADD operand (old memory value).
        let mut even_terms_amoadd = vec![(layout.mem_rv(j), F::ONE)];
        for i in 0..RV32_XLEN {
            let bit_col_id = add_cols.addr_bits.start + 2 * i;
            let bit = layout.bus.bus_cell(bit_col_id, j);
            even_terms_amoadd.push((bit, -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms(layout.is_amoadd_w(j), false, even_terms_amoadd));

        let mut even_terms_auipc = vec![(pc_in, F::ONE)];
        for i in 0..RV32_XLEN {
            let bit_col_id = add_cols.addr_bits.start + 2 * i;
            let bit = layout.bus.bus_cell(bit_col_id, j);
            even_terms_auipc.push((bit, -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms(layout.is_auipc(j), false, even_terms_auipc));

        let mut odd_terms_add = vec![(layout.rs2_val(j), F::ONE)];
        for i in 0..RV32_XLEN {
            let bit_col_id = add_cols.addr_bits.start + 2 * i + 1;
            let bit = layout.bus.bus_cell(bit_col_id, j);
            odd_terms_add.push((bit, -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms_or(&[layout.is_amoadd_w(j)], false, odd_terms_add));

        let mut odd_terms_add_alu = vec![(layout.alu_rhs(j), F::ONE)];
        for i in 0..RV32_XLEN {
            let bit_col_id = add_cols.addr_bits.start + 2 * i + 1;
            let bit = layout.bus.bus_cell(bit_col_id, j);
            odd_terms_add_alu.push((bit, -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms(layout.add_alu(j), false, odd_terms_add_alu));

        let mut odd_terms_addi = vec![(layout.imm_i(j), F::ONE)];
        for i in 0..RV32_XLEN {
            let bit_col_id = add_cols.addr_bits.start + 2 * i + 1;
            let bit = layout.bus.bus_cell(bit_col_id, j);
            odd_terms_addi.push((bit, -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms_or(
            &[layout.is_load(j), layout.is_jalr(j)],
            false,
            odd_terms_addi,
        ));

        let mut odd_terms_sw = vec![(layout.imm_s(j), F::ONE)];
        for i in 0..RV32_XLEN {
            let bit_col_id = add_cols.addr_bits.start + 2 * i + 1;
            let bit = layout.bus.bus_cell(bit_col_id, j);
            odd_terms_sw.push((bit, -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms_or(&[layout.is_store(j)], false, odd_terms_sw));

        let mut odd_terms_auipc = vec![(layout.imm_u(j), F::ONE)];
        for i in 0..RV32_XLEN {
            let bit_col_id = add_cols.addr_bits.start + 2 * i + 1;
            let bit = layout.bus.bus_cell(bit_col_id, j);
            odd_terms_auipc.push((bit, -F::from_u64(pow2_u64(i))));
        }
        constraints.push(Constraint::terms(layout.is_auipc(j), false, odd_terms_auipc));

        // --- Shout key correctness (EQ/NEQ table bus addr bits interleaving) ---
        if let Some(eq_cols) = eq_cols {
            let flag = layout.eq_has_lookup(j);
            let mut even = vec![(layout.rs1_val(j), F::ONE)];
            for i in 0..RV32_XLEN {
                let bit_col_id = eq_cols.addr_bits.start + 2 * i;
                let bit = layout.bus.bus_cell(bit_col_id, j);
                even.push((bit, -F::from_u64(pow2_u64(i))));
            }
            constraints.push(Constraint::terms(flag, false, even));

            let mut odd = vec![(layout.rs2_val(j), F::ONE)];
            for i in 0..RV32_XLEN {
                let bit_col_id = eq_cols.addr_bits.start + 2 * i + 1;
                let bit = layout.bus.bus_cell(bit_col_id, j);
                odd.push((bit, -F::from_u64(pow2_u64(i))));
            }
            constraints.push(Constraint::terms(flag, false, odd));
        }

        // --- Shout key correctness (other opcode tables) ---
        // AND / OR / XOR (R-type uses rs2, I-type uses imm_i).
        if let Some(and_cols) = and_cols {
            constraints.push(Constraint::terms(
                layout.and_alu(j),
                false,
                pack_interleaved_operand(and_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_amoand_w(j),
                false,
                pack_interleaved_operand(and_cols.addr_bits.start, j, 0, layout.mem_rv(j)),
            ));
            constraints.push(Constraint::terms(
                layout.and_alu(j),
                false,
                pack_interleaved_operand(and_cols.addr_bits.start, j, 1, layout.alu_rhs(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_amoand_w(j),
                false,
                pack_interleaved_operand(and_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
        }

        if let Some(or_cols) = or_cols {
            constraints.push(Constraint::terms(
                layout.or_alu(j),
                false,
                pack_interleaved_operand(or_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_amoor_w(j),
                false,
                pack_interleaved_operand(or_cols.addr_bits.start, j, 0, layout.mem_rv(j)),
            ));
            constraints.push(Constraint::terms(
                layout.or_alu(j),
                false,
                pack_interleaved_operand(or_cols.addr_bits.start, j, 1, layout.alu_rhs(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_amoor_w(j),
                false,
                pack_interleaved_operand(or_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
        }

        if let Some(xor_cols) = xor_cols {
            constraints.push(Constraint::terms(
                layout.xor_alu(j),
                false,
                pack_interleaved_operand(xor_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_amoxor_w(j),
                false,
                pack_interleaved_operand(xor_cols.addr_bits.start, j, 0, layout.mem_rv(j)),
            ));
            constraints.push(Constraint::terms(
                layout.xor_alu(j),
                false,
                pack_interleaved_operand(xor_cols.addr_bits.start, j, 1, layout.alu_rhs(j)),
            ));
            constraints.push(Constraint::terms(
                layout.is_amoxor_w(j),
                false,
                pack_interleaved_operand(xor_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
        }

        // SUB (R-type only).
        if let Some(sub_cols) = sub_cols {
            constraints.push(Constraint::terms(
                layout.sub_has_lookup(j),
                false,
                pack_interleaved_operand(sub_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.sub_has_lookup(j),
                false,
                pack_interleaved_operand(sub_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
        }

        // Shifts (R-type uses rs2, I-type uses shamt).
        if let Some(sll_cols) = sll_cols {
            constraints.push(Constraint::terms(
                layout.sll_has_lookup(j),
                false,
                pack_interleaved_operand(sll_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.sll_has_lookup(j),
                false,
                pack_interleaved_operand(sll_cols.addr_bits.start, j, 1, layout.shift_rhs(j)),
            ));
        }

        if let Some(srl_cols) = srl_cols {
            constraints.push(Constraint::terms(
                layout.srl_has_lookup(j),
                false,
                pack_interleaved_operand(srl_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.srl_has_lookup(j),
                false,
                pack_interleaved_operand(srl_cols.addr_bits.start, j, 1, layout.shift_rhs(j)),
            ));
        }

        if let Some(sra_cols) = sra_cols {
            constraints.push(Constraint::terms(
                layout.sra_has_lookup(j),
                false,
                pack_interleaved_operand(sra_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.sra_has_lookup(j),
                false,
                pack_interleaved_operand(sra_cols.addr_bits.start, j, 1, layout.shift_rhs(j)),
            ));
        }

        // SLT/SLTU (ALU + branch comparisons).
        if let Some(slt_cols) = slt_cols {
            constraints.push(Constraint::terms(
                layout.slt_has_lookup(j),
                false,
                pack_interleaved_operand(slt_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.slt_alu(j),
                false,
                pack_interleaved_operand(slt_cols.addr_bits.start, j, 1, layout.alu_rhs(j)),
            ));
            constraints.push(Constraint::terms(
                layout.br_cmp_lt(j),
                false,
                pack_interleaved_operand(slt_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
        }

        if let Some(sltu_cols) = sltu_cols {
            constraints.push(Constraint::terms_or(
                &[layout.sltu_alu(j), layout.br_cmp_ltu(j)],
                false,
                pack_interleaved_operand(sltu_cols.addr_bits.start, j, 0, layout.rs1_val(j)),
            ));
            // DIVU/REMU remainder validity check uses SLTU(rem, divisor).
            constraints.push(Constraint::terms(
                layout.div_rem_check(j),
                false,
                pack_interleaved_operand(sltu_cols.addr_bits.start, j, 0, layout.div_rem(j)),
            ));
            // DIV/REM remainder validity check uses SLTU(|rem|, |divisor|).
            constraints.push(Constraint::terms(
                layout.div_rem_check_signed(j),
                false,
                pack_interleaved_operand(sltu_cols.addr_bits.start, j, 0, layout.div_rem(j)),
            ));
            constraints.push(Constraint::terms(
                layout.sltu_alu(j),
                false,
                pack_interleaved_operand(sltu_cols.addr_bits.start, j, 1, layout.alu_rhs(j)),
            ));
            constraints.push(Constraint::terms(
                layout.br_cmp_ltu(j),
                false,
                pack_interleaved_operand(sltu_cols.addr_bits.start, j, 1, layout.rs2_val(j)),
            ));
            constraints.push(Constraint::terms(
                layout.div_rem_check(j),
                false,
                pack_interleaved_operand(sltu_cols.addr_bits.start, j, 1, layout.div_divisor(j)),
            ));
            constraints.push(Constraint::terms(
                layout.div_rem_check_signed(j),
                false,
                pack_interleaved_operand(sltu_cols.addr_bits.start, j, 1, layout.div_divisor(j)),
            ));
        }

        // --- Alignment constraints (MVP) ---
        // ROM fetch is always 32-bit, so enforce pc_in % 4 == 0 via PROG read address bits.
        let prog_bit0 = layout.bus.bus_cell(prog.ra_bits.start + 0, j);
        let prog_bit1 = layout.bus.bus_cell(prog.ra_bits.start + 1, j);
        constraints.push(Constraint::zero(one, prog_bit0));
        constraints.push(Constraint::zero(one, prog_bit1));

        // Enforce alignment for half/word accesses via RAM bus addr bits.
        let ra0 = layout.bus.bus_cell(ram.ra_bits.start + 0, j);
        let ra1 = layout.bus.bus_cell(ram.ra_bits.start + 1, j);
        let wa0 = layout.bus.bus_cell(ram.wa_bits.start + 0, j);
        let wa1 = layout.bus.bus_cell(ram.wa_bits.start + 1, j);
        let amo_flags = [
            layout.is_amoswap_w(j),
            layout.is_amoadd_w(j),
            layout.is_amoxor_w(j),
            layout.is_amoor_w(j),
            layout.is_amoand_w(j),
        ];
        constraints.push(Constraint::terms_or(
            &[
                layout.is_lh(j),
                layout.is_lhu(j),
                layout.is_lw(j),
                layout.is_sh(j),
                amo_flags[0],
                amo_flags[1],
                amo_flags[2],
                amo_flags[3],
                amo_flags[4],
            ],
            false,
            vec![(ra0, F::ONE)],
        ));
        constraints.push(Constraint::terms_or(
            &[
                layout.is_lw(j),
                amo_flags[0],
                amo_flags[1],
                amo_flags[2],
                amo_flags[3],
                amo_flags[4],
            ],
            false,
            vec![(ra1, F::ONE)],
        ));
        constraints.push(Constraint::terms_or(
            &[
                layout.is_sh(j),
                layout.is_sw(j),
                amo_flags[0],
                amo_flags[1],
                amo_flags[2],
                amo_flags[3],
                amo_flags[4],
            ],
            false,
            vec![(wa0, F::ONE)],
        ));
        constraints.push(Constraint::terms_or(
            &[
                layout.is_sw(j),
                amo_flags[0],
                amo_flags[1],
                amo_flags[2],
                amo_flags[3],
                amo_flags[4],
            ],
            false,
            vec![(wa1, F::ONE)],
        ));
    }

    // --- Intra-chunk composition / padding semantics ---
    // Enforce monotone activity and state continuity:
    // - is_active[j+1] => is_active[j]
    // - pc_in[j+1] == pc_out[j] for all j
    //
    // The unconditional continuity ensures padding rows (is_active=0) *carry* the final
    // architectural state forward, making the final state unambiguous in an L1-style layout.
    for j in 0..layout.chunk_size.saturating_sub(1) {
        let a = layout.is_active(j);
        let b = layout.is_active(j + 1);

        // b * (1 - a) = 0
        constraints.push(Constraint::terms(b, false, vec![(one, F::ONE), (a, -F::ONE)]));

        // HALT terminates execution within a chunk: halt_effective[j] => is_active[j+1] == 0.
        constraints.push(Constraint::terms(
            layout.halt_effective(j),
            false,
            vec![(layout.is_active(j + 1), F::ONE)],
        ));

        // pc_in[j+1] - pc_out[j] = 0
        constraints.push(Constraint::terms(
            one,
            false,
            vec![(layout.pc_in(j + 1), F::ONE), (layout.pc_out(j), -F::ONE)],
        ));
    }

    Ok(constraints)
}

/// Build the RV32 B1 semantics constraint set **excluding** instruction decode plumbing.
///
/// This assumes a separate decode-plumbing sidecar CCS proves instruction bits/fields/immediates and one-hot flags.
fn semantic_constraints_without_decode(
    layout: &Rv32B1Layout,
    mem_layouts: &HashMap<u32, PlainMemLayout>,
) -> Result<Vec<Constraint<F>>, String> {
    rv32_b1_semantic_constraints_impl(layout, mem_layouts, false)
}

fn push_rv32_b1_decode_constraints(
    constraints: &mut Vec<Constraint<F>>,
    layout: &Rv32B1Layout,
    j: usize,
) -> Result<(), String> {
    let one = layout.const_one;
    let is_active = layout.is_active(j);
    let instr_word = layout.instr_word(j);

    // --------------------------------------------------------------------
    // Minimal bit plumbing (no 32-wide instr bits)
    // --------------------------------------------------------------------

    // rd bits (instr[11:7]) and funct7 bits (instr[31:25]) are the only explicit
    // decompositions we keep in-circuit.
    for bit in 0..5 {
        let b = layout.rd_bit(bit, j);
        // b*(b - is_active) = 0  => inactive: b=0 ; active: b∈{0,1}
        constraints.push(Constraint::terms(b, false, vec![(b, F::ONE), (is_active, -F::ONE)]));
    }
    for bit in 0..7 {
        let b = layout.funct7_bit(bit, j);
        constraints.push(Constraint::terms(b, false, vec![(b, F::ONE), (is_active, -F::ONE)]));
    }

    // rd_field = Σ 2^i * rd_bit[i]
    {
        let mut terms = vec![(layout.rd_field(j), F::ONE)];
        for bit in 0..5 {
            terms.push((layout.rd_bit(bit, j), -F::from_u64(pow2_u64(bit))));
        }
        constraints.push(Constraint::terms(one, false, terms));
    }

    // funct7 = Σ 2^i * funct7_bit[i]
    {
        let mut terms = vec![(layout.funct7(j), F::ONE)];
        for bit in 0..7 {
            terms.push((layout.funct7_bit(bit, j), -F::from_u64(pow2_u64(bit))));
        }
        constraints.push(Constraint::terms(one, false, terms));
    }

    // Force some compact scalar fields to 0 on padding rows (keeps witness bounded).
    for &x in &[layout.funct3(j), layout.rs1_field(j), layout.rs2_field(j)] {
        // (1 - is_active) * x = 0
        constraints.push(Constraint::terms(is_active, true, vec![(x, F::ONE)]));
    }

    // Compact field packing:
    // instr_word = opcode
    //           + (rd_field  << 7)
    //           + (funct3    << 12)
    //           + (rs1_field << 15)
    //           + (rs2_field << 20)
    //           + (funct7    << 25)
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (instr_word, F::ONE),
            (layout.opcode(j), -F::ONE),
            (layout.rd_field(j), -F::from_u64(pow2_u64(7))),
            (layout.funct3(j), -F::from_u64(pow2_u64(12))),
            (layout.rs1_field(j), -F::from_u64(pow2_u64(15))),
            (layout.rs2_field(j), -F::from_u64(pow2_u64(20))),
            (layout.funct7(j), -F::from_u64(pow2_u64(25))),
        ],
    ));

    // --------------------------------------------------------------------
    // Immediates (match witness.rs encoding)
    // --------------------------------------------------------------------

    // I-type: imm_i = sx_u32(bits[31:20]) where bits[31:20] = funct7<<5 | rs2_field.
    {
        let sign = layout.funct7_bit(6, j);
        let mut terms = vec![(layout.imm_i(j), F::ONE)];
        terms.push((layout.rs2_field(j), -F::ONE));
        terms.push((layout.funct7(j), -F::from_u64(pow2_u64(5))));
        terms.push((sign, -F::from_u64(pow2_u64(32) - pow2_u64(12))));
        constraints.push(Constraint::terms(one, false, terms));
    }

    // S-type: imm_s = sx_u32(funct7<<5 | rd_field).
    {
        let sign = layout.funct7_bit(6, j);
        let mut terms = vec![(layout.imm_s(j), F::ONE)];
        terms.push((layout.rd_field(j), -F::ONE));
        terms.push((layout.funct7(j), -F::from_u64(pow2_u64(5))));
        terms.push((sign, -F::from_u64(pow2_u64(32) - pow2_u64(12))));
        constraints.push(Constraint::terms(one, false, terms));
    }

    // U-type: imm_u = bits[31:12] << 12.
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.imm_u(j), F::ONE),
            (layout.funct3(j), -F::from_u64(pow2_u64(12))),
            (layout.rs1_field(j), -F::from_u64(pow2_u64(15))),
            (layout.rs2_field(j), -F::from_u64(pow2_u64(20))),
            (layout.funct7(j), -F::from_u64(pow2_u64(25))),
        ],
    ));

    // B-type: imm_b signed (from_i32), with net sign coefficient -2^12 on instr[31].
    {
        let mut terms = vec![(layout.imm_b(j), F::ONE)];
        // instr[7] -> imm[11]
        terms.push((layout.rd_bit(0, j), -F::from_u64(pow2_u64(11))));
        // instr[11:8] -> imm[4:1]
        for i in 0..4 {
            terms.push((layout.rd_bit(1 + i, j), -F::from_u64(pow2_u64(1 + i))));
        }
        // instr[30:25] -> imm[10:5]
        for i in 0..6 {
            terms.push((layout.funct7_bit(i, j), -F::from_u64(pow2_u64(5 + i))));
        }
        // instr[31] sign: net coefficient -2^12 => +2^12 on LHS.
        terms.push((layout.funct7_bit(6, j), F::from_u64(pow2_u64(12))));
        constraints.push(Constraint::terms(one, false, terms));
    }

    // J-type: imm_j signed (from_i32), derived from compact fields + REG lane1 addr bits.
    {
        let reg = &layout.bus.twist_cols[layout.reg_twist_idx];
        if reg.lanes.len() < 2 {
            return Err("RV32 B1 decode: REG_ID requires 2 lanes".into());
        }
        let rs2_bits = &reg.lanes[1].ra_bits;
        if rs2_bits.end - rs2_bits.start < 5 {
            return Err("RV32 B1 decode: REG lane1 ra_bits must have len>=5".into());
        }
        let rs2_b0 = layout.bus.bus_cell(rs2_bits.start + 0, j);
        let rs2_b1 = layout.bus.bus_cell(rs2_bits.start + 1, j);
        let rs2_b2 = layout.bus.bus_cell(rs2_bits.start + 2, j);
        let rs2_b3 = layout.bus.bus_cell(rs2_bits.start + 3, j);
        let rs2_b4 = layout.bus.bus_cell(rs2_bits.start + 4, j);

        let mut terms = vec![(layout.imm_j(j), F::ONE)];
        // instr[19:12] -> imm[19:12] (8 bits)
        terms.push((layout.funct3(j), -F::from_u64(pow2_u64(12))));
        terms.push((layout.rs1_field(j), -F::from_u64(pow2_u64(15))));
        // instr[20] -> imm[11]
        terms.push((rs2_b0, -F::from_u64(pow2_u64(11))));
        // instr[24:21] -> imm[4:1]
        terms.push((rs2_b1, -F::from_u64(pow2_u64(1))));
        terms.push((rs2_b2, -F::from_u64(pow2_u64(2))));
        terms.push((rs2_b3, -F::from_u64(pow2_u64(3))));
        terms.push((rs2_b4, -F::from_u64(pow2_u64(4))));
        // instr[30:25] -> imm[10:5]
        for i in 0..6 {
            terms.push((layout.funct7_bit(i, j), -F::from_u64(pow2_u64(5 + i))));
        }
        // instr[31] sign: net coefficient -2^20 => +2^20 on LHS.
        terms.push((layout.funct7_bit(6, j), F::from_u64(pow2_u64(20))));
        constraints.push(Constraint::terms(one, false, terms));
    }

    // --------------------------------------------------------------------
    // Compact opcode-class decode (one-hot) + control flags
    // --------------------------------------------------------------------

    let class_flags = [
        layout.is_alu_reg(j),
        layout.is_alu_imm(j),
        layout.is_load(j),
        layout.is_store(j),
        layout.is_amo(j),
        layout.is_branch(j),
        layout.is_lui(j),
        layout.is_auipc(j),
        layout.is_jal(j),
        layout.is_jalr(j),
        layout.is_fence(j),
        layout.is_halt(j),
    ];

    // Each class flag is 0 on inactive rows and boolean on active rows: f*(f-is_active)=0.
    for &f in &class_flags {
        constraints.push(Constraint::terms(f, false, vec![(f, F::ONE), (is_active, -F::ONE)]));
    }

    // One-hot: sum(class_flags) = is_active.
    {
        let mut terms = Vec::with_capacity(class_flags.len() + 1);
        for &f in &class_flags {
            terms.push((f, F::ONE));
        }
        terms.push((is_active, -F::ONE));
        constraints.push(Constraint::terms(one, false, terms));
    }

    // opcode = Σ class_flag * opcode_const
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.opcode(j), F::ONE),
            (layout.is_alu_reg(j), -F::from_u64(0x33)),
            (layout.is_alu_imm(j), -F::from_u64(0x13)),
            (layout.is_load(j), -F::from_u64(0x03)),
            (layout.is_store(j), -F::from_u64(0x23)),
            (layout.is_amo(j), -F::from_u64(0x2f)),
            (layout.is_branch(j), -F::from_u64(0x63)),
            (layout.is_lui(j), -F::from_u64(0x37)),
            (layout.is_auipc(j), -F::from_u64(0x17)),
            (layout.is_jal(j), -F::from_u64(0x6f)),
            (layout.is_jalr(j), -F::from_u64(0x67)),
            (layout.is_fence(j), -F::from_u64(0x0f)),
            (layout.is_halt(j), -F::from_u64(0x73)),
        ],
    ));

    // --------------------------------------------------------------------
    // Branch control (BNE represented as EQ + invert)
    // --------------------------------------------------------------------

    // br_cmp_* and br_invert are 0 unless is_branch, and boolean when is_branch.
    for &f in &[
        layout.br_cmp_eq(j),
        layout.br_cmp_lt(j),
        layout.br_cmp_ltu(j),
        layout.br_invert(j),
    ] {
        constraints.push(Constraint::terms(
            f,
            false,
            vec![(f, F::ONE), (layout.is_branch(j), -F::ONE)],
        ));
    }

    // Exactly one compare mode on branch rows: br_cmp_eq + br_cmp_lt + br_cmp_ltu = is_branch.
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.br_cmp_eq(j), F::ONE),
            (layout.br_cmp_lt(j), F::ONE),
            (layout.br_cmp_ltu(j), F::ONE),
            (layout.is_branch(j), -F::ONE),
        ],
    ));

    // Branch funct3 mapping:
    // funct3 = br_invert + 4*br_cmp_lt + 6*br_cmp_ltu   (only when is_branch=1)
    constraints.push(Constraint::terms(
        layout.is_branch(j),
        false,
        vec![
            (layout.funct3(j), F::ONE),
            (layout.br_invert(j), -F::ONE),
            (layout.br_cmp_lt(j), -F::from_u64(4)),
            (layout.br_cmp_ltu(j), -F::from_u64(6)),
        ],
    ));

    // EQ table selector helper: eq_has_lookup == br_cmp_eq.
    constraints.push(Constraint::terms(
        one,
        false,
        vec![(layout.eq_has_lookup(j), F::ONE), (layout.br_cmp_eq(j), -F::ONE)],
    ));

    // --------------------------------------------------------------------
    // Load/store subflags + funct3 mapping
    // --------------------------------------------------------------------

    for &f in &[
        layout.is_lb(j),
        layout.is_lbu(j),
        layout.is_lh(j),
        layout.is_lhu(j),
        layout.is_lw(j),
    ] {
        constraints.push(Constraint::terms(
            f,
            false,
            vec![(f, F::ONE), (layout.is_load(j), -F::ONE)],
        ));
    }
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.is_lb(j), F::ONE),
            (layout.is_lbu(j), F::ONE),
            (layout.is_lh(j), F::ONE),
            (layout.is_lhu(j), F::ONE),
            (layout.is_lw(j), F::ONE),
            (layout.is_load(j), -F::ONE),
        ],
    ));
    // funct3 = 4*lbu + 1*lh + 5*lhu + 2*lw (lb is 0)
    constraints.push(Constraint::terms(
        layout.is_load(j),
        false,
        vec![
            (layout.funct3(j), F::ONE),
            (layout.is_lbu(j), -F::from_u64(4)),
            (layout.is_lh(j), -F::from_u64(1)),
            (layout.is_lhu(j), -F::from_u64(5)),
            (layout.is_lw(j), -F::from_u64(2)),
        ],
    ));

    for &f in &[layout.is_sb(j), layout.is_sh(j), layout.is_sw(j)] {
        constraints.push(Constraint::terms(
            f,
            false,
            vec![(f, F::ONE), (layout.is_store(j), -F::ONE)],
        ));
    }
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.is_sb(j), F::ONE),
            (layout.is_sh(j), F::ONE),
            (layout.is_sw(j), F::ONE),
            (layout.is_store(j), -F::ONE),
        ],
    ));
    // funct3 = 1*sh + 2*sw (sb is 0)
    constraints.push(Constraint::terms(
        layout.is_store(j),
        false,
        vec![
            (layout.funct3(j), F::ONE),
            (layout.is_sh(j), -F::from_u64(1)),
            (layout.is_sw(j), -F::from_u64(2)),
        ],
    ));

    // --------------------------------------------------------------------
    // RV32A (AMO word ops only)
    // --------------------------------------------------------------------

    for &f in &[
        layout.is_amoswap_w(j),
        layout.is_amoadd_w(j),
        layout.is_amoxor_w(j),
        layout.is_amoor_w(j),
        layout.is_amoand_w(j),
    ] {
        constraints.push(Constraint::terms(
            f,
            false,
            vec![(f, F::ONE), (layout.is_amo(j), -F::ONE)],
        ));
    }
    constraints.push(Constraint::terms(
        one,
        false,
        vec![
            (layout.is_amoswap_w(j), F::ONE),
            (layout.is_amoadd_w(j), F::ONE),
            (layout.is_amoxor_w(j), F::ONE),
            (layout.is_amoor_w(j), F::ONE),
            (layout.is_amoand_w(j), F::ONE),
            (layout.is_amo(j), -F::ONE),
        ],
    ));
    constraints.push(Constraint::eq_const(layout.is_amo(j), one, layout.funct3(j), 0b010));
    // funct5 (instr[31:27]) = 1*AMOSWAP + 4*AMOXOR + 8*AMOOR + 12*AMOAND (AMOADD is 0)
    constraints.push(Constraint::terms(
        layout.is_amo(j),
        false,
        vec![
            (layout.funct7_bit(2, j), F::from_u64(1)),  // 2^0
            (layout.funct7_bit(3, j), F::from_u64(2)),  // 2^1
            (layout.funct7_bit(4, j), F::from_u64(4)),  // 2^2
            (layout.funct7_bit(5, j), F::from_u64(8)),  // 2^3
            (layout.funct7_bit(6, j), F::from_u64(16)), // 2^4
            (layout.is_amoswap_w(j), -F::from_u64(1)),
            (layout.is_amoxor_w(j), -F::from_u64(4)),
            (layout.is_amoor_w(j), -F::from_u64(8)),
            (layout.is_amoand_w(j), -F::from_u64(12)),
        ],
    ));

    // --------------------------------------------------------------------
    // RV32I ALU decode (compact op selectors) + RV32M flags
    // --------------------------------------------------------------------

    // Base ALU selectors (valid for either ALU class): f*(f - is_alu_reg - is_alu_imm)=0.
    for &f in &[
        layout.add_alu(j),
        layout.and_alu(j),
        layout.xor_alu(j),
        layout.or_alu(j),
        layout.slt_alu(j),
        layout.sltu_alu(j),
        layout.sll_has_lookup(j),
        layout.srl_has_lookup(j),
        layout.sra_has_lookup(j),
    ] {
        constraints.push(Constraint::terms(
            f,
            false,
            vec![
                (f, F::ONE),
                (layout.is_alu_reg(j), -F::ONE),
                (layout.is_alu_imm(j), -F::ONE),
            ],
        ));
    }

    // SUB is R-type only.
    constraints.push(Constraint::terms(
        layout.sub_has_lookup(j),
        false,
        vec![(layout.sub_has_lookup(j), F::ONE), (layout.is_alu_reg(j), -F::ONE)],
    ));

    // RV32M flags are R-type only.
    for &f in &[
        layout.is_mul(j),
        layout.is_mulh(j),
        layout.is_mulhu(j),
        layout.is_mulhsu(j),
        layout.is_div(j),
        layout.is_divu(j),
        layout.is_rem(j),
        layout.is_remu(j),
    ] {
        constraints.push(Constraint::terms(
            f,
            false,
            vec![(f, F::ONE), (layout.is_alu_reg(j), -F::ONE)],
        ));
    }

    // Exactly one ALU op selector on each ALU row.
    constraints.push(Constraint::terms(
        layout.is_alu_reg(j),
        false,
        vec![
            (layout.add_alu(j), F::ONE),
            (layout.sub_has_lookup(j), F::ONE),
            (layout.sll_has_lookup(j), F::ONE),
            (layout.slt_alu(j), F::ONE),
            (layout.sltu_alu(j), F::ONE),
            (layout.xor_alu(j), F::ONE),
            (layout.srl_has_lookup(j), F::ONE),
            (layout.sra_has_lookup(j), F::ONE),
            (layout.or_alu(j), F::ONE),
            (layout.and_alu(j), F::ONE),
            (layout.is_mul(j), F::ONE),
            (layout.is_mulh(j), F::ONE),
            (layout.is_mulhu(j), F::ONE),
            (layout.is_mulhsu(j), F::ONE),
            (layout.is_div(j), F::ONE),
            (layout.is_divu(j), F::ONE),
            (layout.is_rem(j), F::ONE),
            (layout.is_remu(j), F::ONE),
            (one, -F::ONE),
        ],
    ));
    constraints.push(Constraint::terms(
        layout.is_alu_imm(j),
        false,
        vec![
            (layout.add_alu(j), F::ONE),
            (layout.sll_has_lookup(j), F::ONE),
            (layout.slt_alu(j), F::ONE),
            (layout.sltu_alu(j), F::ONE),
            (layout.xor_alu(j), F::ONE),
            (layout.srl_has_lookup(j), F::ONE),
            (layout.sra_has_lookup(j), F::ONE),
            (layout.or_alu(j), F::ONE),
            (layout.and_alu(j), F::ONE),
            (one, -F::ONE),
        ],
    ));

    // ALU funct3 mapping (reg/imm).
    constraints.push(Constraint::terms(
        layout.is_alu_reg(j),
        false,
        vec![
            (layout.funct3(j), F::ONE),
            (layout.sll_has_lookup(j), -F::from_u64(1)),
            (layout.slt_alu(j), -F::from_u64(2)),
            (layout.sltu_alu(j), -F::from_u64(3)),
            (layout.xor_alu(j), -F::from_u64(4)),
            (layout.srl_has_lookup(j), -F::from_u64(5)),
            (layout.sra_has_lookup(j), -F::from_u64(5)),
            (layout.or_alu(j), -F::from_u64(6)),
            (layout.and_alu(j), -F::from_u64(7)),
            (layout.is_mulh(j), -F::from_u64(1)),
            (layout.is_mulhsu(j), -F::from_u64(2)),
            (layout.is_mulhu(j), -F::from_u64(3)),
            (layout.is_div(j), -F::from_u64(4)),
            (layout.is_divu(j), -F::from_u64(5)),
            (layout.is_rem(j), -F::from_u64(6)),
            (layout.is_remu(j), -F::from_u64(7)),
        ],
    ));
    constraints.push(Constraint::terms(
        layout.is_alu_imm(j),
        false,
        vec![
            (layout.funct3(j), F::ONE),
            (layout.sll_has_lookup(j), -F::from_u64(1)),
            (layout.slt_alu(j), -F::from_u64(2)),
            (layout.sltu_alu(j), -F::from_u64(3)),
            (layout.xor_alu(j), -F::from_u64(4)),
            (layout.srl_has_lookup(j), -F::from_u64(5)),
            (layout.sra_has_lookup(j), -F::from_u64(5)),
            (layout.or_alu(j), -F::from_u64(6)),
            (layout.and_alu(j), -F::from_u64(7)),
        ],
    ));

    // funct7 constraints:
    // - R-type ALU: funct7 is determined by SUB/SRA (0x20) or RV32M (0x01), else 0.
    constraints.push(Constraint::terms(
        layout.is_alu_reg(j),
        false,
        vec![
            (layout.funct7(j), F::ONE),
            (layout.sub_has_lookup(j), -F::from_u64(0x20)),
            (layout.sra_has_lookup(j), -F::from_u64(0x20)),
            (layout.is_mul(j), -F::from_u64(0x01)),
            (layout.is_mulh(j), -F::from_u64(0x01)),
            (layout.is_mulhu(j), -F::from_u64(0x01)),
            (layout.is_mulhsu(j), -F::from_u64(0x01)),
            (layout.is_div(j), -F::from_u64(0x01)),
            (layout.is_divu(j), -F::from_u64(0x01)),
            (layout.is_rem(j), -F::from_u64(0x01)),
            (layout.is_remu(j), -F::from_u64(0x01)),
        ],
    ));

    // Shift immediate encodings:
    constraints.push(Constraint::zero(layout.sll_has_lookup(j), layout.funct7(j)));
    constraints.push(Constraint::zero(layout.srl_has_lookup(j), layout.funct7(j)));
    constraints.push(Constraint::eq_const(
        layout.sra_has_lookup(j),
        one,
        layout.funct7(j),
        0x20,
    ));

    // --------------------------------------------------------------------
    // Small ISA-specific restrictions (disallow unsupported encodings)
    // --------------------------------------------------------------------

    constraints.push(Constraint::zero(layout.is_jalr(j), layout.funct3(j))); // JALR requires funct3=0.
    constraints.push(Constraint::zero(layout.is_fence(j), layout.funct3(j))); // FENCE requires funct3=0.

    // ECALL (HALT) is exactly 0x0000_0073: all other fields must be 0.
    constraints.push(Constraint::zero(layout.is_halt(j), layout.funct3(j)));
    constraints.push(Constraint::zero(layout.is_halt(j), layout.funct7(j)));
    constraints.push(Constraint::zero(layout.is_halt(j), layout.rd_field(j)));
    constraints.push(Constraint::zero(layout.is_halt(j), layout.rs1_field(j)));
    constraints.push(Constraint::zero(layout.is_halt(j), layout.rs2_field(j)));

    Ok(())
}

/// Build the RV32 B1 **main** step constraint set.
///
/// The main step CCS is intentionally minimal: it exists primarily to host the injected shared-bus
/// constraints. Full RV32 B1 instruction semantics are proven in a separate sidecar CCS built from
/// [`full_semantic_constraints`].
fn semantic_constraints(
    _layout: &Rv32B1Layout,
    _mem_layouts: &HashMap<u32, PlainMemLayout>,
) -> Result<Vec<Constraint<F>>, String> {
    Ok(Vec::new())
}

/// Build an RV32 B1 “decode” sidecar CCS.
///
/// This CCS contains only the instruction decode plumbing (instruction bits, field packing,
/// immediate derivations, and one-hot instruction flags), plus a small set of derived group signals
/// used by downstream code.
///
/// It is intended to be proven/verified as an additional argument alongside:
/// - the main step CCS (shared-bus injection), and
/// - the semantics sidecar CCS (which assumes these decoded signals are sound).
pub fn build_rv32_b1_decode_plumbing_sidecar_ccs(layout: &Rv32B1Layout) -> Result<CcsStructure<F>, String> {
    let mut constraints: Vec<Constraint<F>> = Vec::new();

    for j in 0..layout.chunk_size {
        push_rv32_b1_decode_constraints(&mut constraints, layout, j)?;

        // Derived group/control signals (kept sound even if the main CCS is thin).
        //
        // writes_rd = OR over op-classes that write rd (one-hot => sum).
        constraints.push(Constraint::terms(
            layout.const_one,
            false,
            vec![
                (layout.writes_rd(j), F::ONE),
                (layout.is_alu_reg(j), -F::ONE),
                (layout.is_alu_imm(j), -F::ONE),
                (layout.is_load(j), -F::ONE),
                (layout.is_amo(j), -F::ONE),
                (layout.is_lui(j), -F::ONE),
                (layout.is_auipc(j), -F::ONE),
                (layout.is_jal(j), -F::ONE),
                (layout.is_jalr(j), -F::ONE),
            ],
        ));

        // pc_plus4 + is_branch + is_jal + is_jalr = is_active
        constraints.push(Constraint::terms(
            layout.const_one,
            false,
            vec![
                (layout.pc_plus4(j), F::ONE),
                (layout.is_branch(j), F::ONE),
                (layout.is_jal(j), F::ONE),
                (layout.is_jalr(j), F::ONE),
                (layout.is_active(j), -F::ONE),
            ],
        ));

        // wb_from_alu selects the Shout-backed writeback path:
        // wb_from_alu = is_alu_imm + is_alu_reg - is_rv32m + is_auipc
        constraints.push(Constraint::terms(
            layout.const_one,
            false,
            vec![
                (layout.wb_from_alu(j), F::ONE),
                (layout.is_alu_imm(j), -F::ONE),
                (layout.is_alu_reg(j), -F::ONE),
                (layout.is_mul(j), F::ONE),
                (layout.is_mulh(j), F::ONE),
                (layout.is_mulhu(j), F::ONE),
                (layout.is_mulhsu(j), F::ONE),
                (layout.is_div(j), F::ONE),
                (layout.is_divu(j), F::ONE),
                (layout.is_rem(j), F::ONE),
                (layout.is_remu(j), F::ONE),
                (layout.is_auipc(j), -F::ONE),
            ],
        ));
    }

    // Public RV32M activity: number of RV32M ops in this chunk (sum over one-hot flags).
    {
        let mut terms = vec![(layout.rv32m_count, F::ONE)];
        for j in 0..layout.chunk_size {
            terms.push((layout.is_mul(j), -F::ONE));
            terms.push((layout.is_mulh(j), -F::ONE));
            terms.push((layout.is_mulhu(j), -F::ONE));
            terms.push((layout.is_mulhsu(j), -F::ONE));
            terms.push((layout.is_div(j), -F::ONE));
            terms.push((layout.is_divu(j), -F::ONE));
            terms.push((layout.is_rem(j), -F::ONE));
            terms.push((layout.is_remu(j), -F::ONE));
        }
        constraints.push(Constraint::terms(layout.const_one, false, terms));
    }

    let n = constraints.len();
    build_r1cs_ccs(&constraints, n, layout.m, layout.const_one)
}

/// Build an RV32 B1 “semantics” sidecar CCS (decode excluded).
///
/// This CCS contains the full RV32 B1 step semantics, but assumes instruction decode plumbing is
/// proven separately via [`build_rv32_b1_decode_plumbing_sidecar_ccs`].
pub fn build_rv32_b1_semantics_sidecar_ccs(
    layout: &Rv32B1Layout,
    mem_layouts: &HashMap<u32, PlainMemLayout>,
) -> Result<CcsStructure<F>, String> {
    let constraints = semantic_constraints_without_decode(layout, mem_layouts)?;
    let n = constraints.len();
    build_r1cs_ccs(&constraints, n, layout.m, layout.const_one)
}

/// Build the RV32 B1 step CCS and its witness layout.
///
/// Requirements:
/// - `mem_layouts` must include `RAM_ID`, `PROG_ID`, and `REG_ID`.
/// - `mem_layouts[PROG_ID]` is byte-addressed (`n_side=2`, `ell=1`).
///
/// `shout_table_ids` must be non-empty and include the RV32 `ADD` table id (3). Any subset of the
/// base RV32I opcode tables (ids 0..=11) is allowed (unused tables can be bound to fixed-zero CPU
/// columns in the shared-bus config).
pub fn build_rv32_b1_step_ccs(
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    shout_table_ids: &[u32],
    chunk_size: usize,
) -> Result<(CcsStructure<F>, Rv32B1Layout), String> {
    let (layout, injected) = build_rv32_b1_layout_and_injected(mem_layouts, shout_table_ids, chunk_size)?;
    let constraints = semantic_constraints(&layout, mem_layouts)?;
    let n = constraints
        .len()
        .checked_add(injected)
        .ok_or_else(|| "RV32 B1: n overflow".to_string())?;
    let ccs = build_r1cs_ccs(&constraints, n, layout.m, layout.const_one)?;
    Ok((ccs, layout))
}

#[derive(Clone, Copy, Debug)]
pub struct Rv32B1StepCcsCounts {
    pub n: usize,
    pub m: usize,
    pub semantic: usize,
    pub injected: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Rv32B1AllCcsCounts {
    pub step: Rv32B1StepCcsCounts,
    pub decode_plumbing_n: usize,
    pub semantics_n: usize,
}

/// Estimate the RV32 B1 step CCS shape without materializing the CCS matrices.
///
/// This still constructs the semantic constraint vector in order to count it, but it avoids the
/// additional work done by `build_r1cs_ccs`.
pub fn estimate_rv32_b1_step_ccs_counts(
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    shout_table_ids: &[u32],
    chunk_size: usize,
) -> Result<Rv32B1StepCcsCounts, String> {
    let (layout, injected) = build_rv32_b1_layout_and_injected(mem_layouts, shout_table_ids, chunk_size)?;
    let semantic = semantic_constraints(&layout, mem_layouts)?.len();
    let n = semantic
        .checked_add(injected)
        .ok_or_else(|| "RV32 B1: n overflow".to_string())?;
    Ok(Rv32B1StepCcsCounts {
        n,
        m: layout.m,
        semantic,
        injected,
    })
}

/// Estimate the RV32 B1 step + sidecar CCS shapes without materializing CCS matrices.
///
/// This is intended for frontend heuristics (e.g. `chunk_size_auto`) that should consider the
/// *full proving workload*:
/// - the main step CCS (shared-bus host), plus
/// - the decode plumbing sidecar CCS, plus
/// - the semantics sidecar CCS.
pub fn estimate_rv32_b1_all_ccs_counts(
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    shout_table_ids: &[u32],
    chunk_size: usize,
) -> Result<Rv32B1AllCcsCounts, String> {
    let (layout, injected) = build_rv32_b1_layout_and_injected(mem_layouts, shout_table_ids, chunk_size)?;

    let semantic = semantic_constraints(&layout, mem_layouts)?.len();
    let n = semantic
        .checked_add(injected)
        .ok_or_else(|| "RV32 B1: n overflow".to_string())?;
    let step = Rv32B1StepCcsCounts {
        n,
        m: layout.m,
        semantic,
        injected,
    };

    // Decode plumbing sidecar count (same constraints as `build_rv32_b1_decode_plumbing_sidecar_ccs`,
    // but without building CCS matrices).
    let decode_plumbing_n = {
        let one = layout.const_one;
        let mut constraints: Vec<Constraint<F>> = Vec::new();

        for j in 0..layout.chunk_size {
            push_rv32_b1_decode_constraints(&mut constraints, &layout, j)?;

            // Derived group/control signals (kept sound even if the main CCS is thin).
            //
            // writes_rd = OR over op-classes that write rd (one-hot => sum).
            constraints.push(Constraint::terms(
                one,
                false,
                vec![
                    (layout.writes_rd(j), F::ONE),
                    (layout.is_alu_reg(j), -F::ONE),
                    (layout.is_alu_imm(j), -F::ONE),
                    (layout.is_load(j), -F::ONE),
                    (layout.is_amo(j), -F::ONE),
                    (layout.is_lui(j), -F::ONE),
                    (layout.is_auipc(j), -F::ONE),
                    (layout.is_jal(j), -F::ONE),
                    (layout.is_jalr(j), -F::ONE),
                ],
            ));

            // pc_plus4 + is_branch + is_jal + is_jalr = is_active
            constraints.push(Constraint::terms(
                one,
                false,
                vec![
                    (layout.pc_plus4(j), F::ONE),
                    (layout.is_branch(j), F::ONE),
                    (layout.is_jal(j), F::ONE),
                    (layout.is_jalr(j), F::ONE),
                    (layout.is_active(j), -F::ONE),
                ],
            ));

            // wb_from_alu selects the Shout-backed writeback path:
            // wb_from_alu = is_alu_imm + is_alu_reg - is_rv32m + is_auipc
            constraints.push(Constraint::terms(
                one,
                false,
                vec![
                    (layout.wb_from_alu(j), F::ONE),
                    (layout.is_alu_imm(j), -F::ONE),
                    (layout.is_alu_reg(j), -F::ONE),
                    (layout.is_mul(j), F::ONE),
                    (layout.is_mulh(j), F::ONE),
                    (layout.is_mulhu(j), F::ONE),
                    (layout.is_mulhsu(j), F::ONE),
                    (layout.is_div(j), F::ONE),
                    (layout.is_divu(j), F::ONE),
                    (layout.is_rem(j), F::ONE),
                    (layout.is_remu(j), F::ONE),
                    (layout.is_auipc(j), -F::ONE),
                ],
            ));
        }

        // Public RV32M activity: number of RV32M ops in this chunk (sum over one-hot flags).
        let mut terms = vec![(layout.rv32m_count, F::ONE)];
        for j in 0..layout.chunk_size {
            terms.push((layout.is_mul(j), -F::ONE));
            terms.push((layout.is_mulh(j), -F::ONE));
            terms.push((layout.is_mulhu(j), -F::ONE));
            terms.push((layout.is_mulhsu(j), -F::ONE));
            terms.push((layout.is_div(j), -F::ONE));
            terms.push((layout.is_divu(j), -F::ONE));
            terms.push((layout.is_rem(j), -F::ONE));
            terms.push((layout.is_remu(j), -F::ONE));
        }
        constraints.push(Constraint::terms(one, false, terms));

        constraints.len()
    };

    // Semantics sidecar count (decode excluded).
    let semantics_n = semantic_constraints_without_decode(&layout, mem_layouts)?.len();

    Ok(Rv32B1AllCcsCounts {
        step,
        decode_plumbing_n,
        semantics_n,
    })
}

fn build_rv32_b1_layout_and_injected(
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    shout_table_ids: &[u32],
    chunk_size: usize,
) -> Result<(Rv32B1Layout, usize), String> {
    if chunk_size == 0 {
        return Err("RV32 B1: chunk_size must be >= 1".into());
    }
    let ram_id = RAM_ID.0;
    let prog_id = PROG_ID.0;
    let reg_id = REG_ID.0;
    if !mem_layouts.contains_key(&ram_id) {
        return Err(format!("RV32 B1: mem_layouts missing RAM_ID={ram_id}"));
    }
    if !mem_layouts.contains_key(&prog_id) {
        return Err(format!("RV32 B1: mem_layouts missing PROG_ID={prog_id}"));
    }
    if !mem_layouts.contains_key(&reg_id) {
        return Err(format!("RV32 B1: mem_layouts missing REG_ID={reg_id}"));
    }

    // B1 circuit currently assumes only RISC-V opcode Shout tables (ell_addr = 2*xlen = 64).
    let (table_ids, shout_ell_addrs) = derive_shout_ids_and_ell_addrs(shout_table_ids)?;

    let (mem_ids, twist_ell_addrs) = derive_mem_ids_and_ell_addrs(mem_layouts)?;
    if mem_ids.len() != twist_ell_addrs.len() {
        return Err("RV32 B1: internal error (twist ell addrs mismatch)".into());
    }
    let shout_cols_per_step: usize = shout_ell_addrs.iter().sum::<usize>() + 2 * shout_ell_addrs.len();
    let twist_cols_per_step: usize = mem_ids
        .iter()
        .zip(twist_ell_addrs.iter())
        .map(|(mem_id, &ell_addr)| {
            let lanes = mem_layouts.get(mem_id).map(|l| l.lanes.max(1)).unwrap_or(1);
            lanes * (2 * ell_addr + 5)
        })
        .sum::<usize>();
    let bus_cols_per_step = shout_cols_per_step + twist_cols_per_step;
    let bus_region_len = bus_cols_per_step
        .checked_mul(chunk_size)
        .ok_or_else(|| "RV32 B1: bus_region_len overflow".to_string())?;

    // Probe layout to learn the CPU column footprint and count injected constraints.
    // We rebuild once with the minimal `m` after computing exact sizes.
    let mut probe_m = bus_region_len
        .checked_add(1)
        .ok_or_else(|| "RV32 B1: probe_m overflow".to_string())?;
    let probe = loop {
        match build_layout_with_m(probe_m, mem_layouts, &table_ids, chunk_size) {
            Ok(layout) => break layout,
            Err(e)
                if e.contains("need more padding columns before bus tail") || e.contains("overlaps public inputs") =>
            {
                probe_m = probe_m
                    .checked_mul(2)
                    .ok_or_else(|| "RV32 B1: probe_m overflow".to_string())?;
            }
            Err(e) => return Err(e),
        }
    };
    let cpu_cols_used = probe.halt_effective + chunk_size;
    let injected = injected_bus_constraints_len(&probe, &table_ids, &mem_ids);

    let m_cols_min = cpu_cols_used + bus_region_len;
    let mut m = m_cols_min;
    let layout = loop {
        match build_layout_with_m(m, mem_layouts, &table_ids, chunk_size) {
            Ok(layout) => break layout,
            Err(e)
                if e.contains("need more padding columns before bus tail") || e.contains("overlaps public inputs") =>
            {
                m = m
                    .checked_mul(2)
                    .ok_or_else(|| "RV32 B1: m overflow".to_string())?;
            }
            Err(e) => return Err(e),
        }
    };

    Ok((layout, injected))
}
