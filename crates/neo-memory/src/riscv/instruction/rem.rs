use crate::riscv::instruction::{DecomposedOp, InstructionDescriptor, OperandMode, VirtualRegisterAllocator};
use crate::riscv::lookups::{compute_op, RiscvOpcode};

pub struct Rem;

impl InstructionDescriptor for Rem {
    fn opcode() -> Option<RiscvOpcode> {
        Some(RiscvOpcode::Rem)
    }

    fn operand_mode() -> OperandMode {
        OperandMode::MultiplyOperands
    }
}

pub fn eval(rs1: u64, rs2: u64, xlen: usize) -> u64 {
    compute_op(RiscvOpcode::Rem, rs1, rs2, xlen)
}

/// Jolt-style signed REM decomposition (with virtual self-move rows for commit linkage).
pub fn decomposition_sequence(rd: u8, rs1: u8, rs2: u8, alloc: &mut VirtualRegisterAllocator) -> Vec<DecomposedOp> {
    let rd = rd as u64;
    let rs1 = rs1 as u64;
    let rs2 = rs2 as u64;

    let v_q = alloc.allocate();
    let v_rabs = alloc.allocate();
    let v_adj_div = alloc.allocate();
    let v_hi = alloc.allocate();
    let v_prod = alloc.allocate();
    let v_tmp = alloc.allocate();
    let v_r = alloc.allocate();
    let v_abs_div = alloc.allocate();

    vec![
        DecomposedOp::AdviceQuotient {
            dst: v_q,
            op: RiscvOpcode::Div,
            lhs: rs1,
            rhs: rs2,
        },
        DecomposedOp::AdviceRemainderAbs {
            dst: v_rabs,
            dividend: rs1,
            divisor: rs2,
        },
        DecomposedOp::AssertValidDiv0 {
            divisor: rs2,
            quotient: v_q,
        },
        DecomposedOp::ChangeDivisor {
            dst: v_adj_div,
            dividend: rs1,
            divisor: rs2,
        },
        DecomposedOp::AdviceQuotient {
            dst: v_hi,
            op: RiscvOpcode::Mulh,
            lhs: v_q,
            rhs: v_adj_div,
        },
        DecomposedOp::Mul {
            dst: v_prod,
            lhs: v_q,
            rhs: v_adj_div,
        },
        DecomposedOp::MovSign {
            dst: v_tmp,
            src: v_prod,
        },
        DecomposedOp::AssertEq { lhs: v_hi, rhs: v_tmp },
        DecomposedOp::MovSign { dst: v_tmp, src: rs1 },
        DecomposedOp::Xor {
            dst: v_r,
            lhs: v_rabs,
            rhs: v_tmp,
        },
        DecomposedOp::Sub {
            dst: v_r,
            lhs: v_r,
            rhs: v_tmp,
        },
        DecomposedOp::Add {
            dst: v_prod,
            lhs: v_prod,
            rhs: v_r,
        },
        DecomposedOp::AssertEq { lhs: v_prod, rhs: rs1 },
        DecomposedOp::MovSign {
            dst: v_tmp,
            src: v_adj_div,
        },
        DecomposedOp::Xor {
            dst: v_abs_div,
            lhs: v_adj_div,
            rhs: v_tmp,
        },
        DecomposedOp::Sub {
            dst: v_abs_div,
            lhs: v_abs_div,
            rhs: v_tmp,
        },
        DecomposedOp::AssertValidUnsignedRemainder {
            remainder: v_rabs,
            divisor: v_abs_div,
        },
        // Keep final virtual rows write-active for commit-value linkage.
        DecomposedOp::Move { dst: v_r, src: v_r },
        DecomposedOp::Move { dst: v_r, src: v_r },
        DecomposedOp::Move { dst: rd, src: v_r },
    ]
}
