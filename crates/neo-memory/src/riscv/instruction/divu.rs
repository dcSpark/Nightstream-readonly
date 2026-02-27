use crate::riscv::instruction::{DecomposedOp, InstructionDescriptor, OperandMode, VirtualRegisterAllocator};
use crate::riscv::lookups::{compute_op, RiscvOpcode};

pub struct Divu;

impl InstructionDescriptor for Divu {
    fn opcode() -> Option<RiscvOpcode> {
        Some(RiscvOpcode::Divu)
    }

    fn operand_mode() -> OperandMode {
        OperandMode::MultiplyOperands
    }
}

pub fn eval(rs1: u64, rs2: u64, xlen: usize) -> u64 {
    compute_op(RiscvOpcode::Divu, rs1, rs2, xlen)
}

/// Jolt-style DIVU decomposition sequence.
///
/// ```text
/// ADVICE               -> v_q
/// ASSERT_VALID_DIV0    (rs2, v_q)
/// ASSERT_MULU_NO_OVF   (v_q, rs2)
/// MUL(v_q, rs2)        -> v_prod
/// ASSERT_LTE(v_prod, rs1)
/// SUB(rs1, v_prod)     -> v_rem
/// ASSERT_VALID_UREM(v_rem, rs2, rs1)
/// MOVE(v_q)            -> v_q
/// MOVE(v_q)            -> rd
/// ```
pub fn decomposition_sequence(rd: u8, rs1: u8, rs2: u8, alloc: &mut VirtualRegisterAllocator) -> Vec<DecomposedOp> {
    let rd = rd as u64;
    let rs1 = rs1 as u64;
    let rs2 = rs2 as u64;
    let v_q = alloc.allocate();
    let v_prod = alloc.allocate();
    let v_rem = alloc.allocate();

    vec![
        DecomposedOp::AdviceQuotient {
            dst: v_q,
            op: RiscvOpcode::Divu,
            lhs: rs1,
            rhs: rs2,
        },
        DecomposedOp::AssertValidDiv0 {
            divisor: rs2,
            quotient: v_q,
        },
        DecomposedOp::AssertMulUNoOverflow { lhs: v_q, rhs: rs2 },
        DecomposedOp::Mul {
            dst: v_prod,
            lhs: v_q,
            rhs: rs2,
        },
        DecomposedOp::AssertLte { lhs: v_prod, rhs: rs1 },
        DecomposedOp::Sub {
            dst: v_rem,
            lhs: rs1,
            rhs: v_prod,
        },
        DecomposedOp::AssertValidUnsignedRemainder {
            remainder: v_rem,
            divisor: rs2,
        },
        // Ensure the last virtual row has a write so commit-value linkage is active.
        DecomposedOp::Move { dst: v_q, src: v_q },
        DecomposedOp::Move { dst: rd, src: v_q },
    ]
}
