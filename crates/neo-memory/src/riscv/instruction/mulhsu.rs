use crate::riscv::instruction::{DecomposedOp, InstructionDescriptor, OperandMode, VirtualRegisterAllocator};
use crate::riscv::lookups::{compute_op, RiscvOpcode};

pub struct Mulhsu;

impl InstructionDescriptor for Mulhsu {
    fn opcode() -> Option<RiscvOpcode> {
        Some(RiscvOpcode::Mulhsu)
    }

    fn operand_mode() -> OperandMode {
        OperandMode::MultiplyOperands
    }
}

pub fn eval(rs1: u64, rs2: u64, xlen: usize) -> u64 {
    compute_op(RiscvOpcode::Mulhsu, rs1, rs2, xlen)
}

/// Jolt-style MULHSU decomposition sequence.
///
/// ```text
/// MOVSIGN(rs1)      -> v_sign
/// SUB(x0, v_sign)   -> v_one
/// XOR(rs1, v_sign)  -> v_absx
/// ADD(v_absx, v_one)-> v_absx
/// MULHU(v_absx,rs2) -> v_hi
/// MUL(v_absx,rs2)   -> v_absx
/// XOR(v_hi, v_sign) -> v_hi
/// XOR(v_absx,v_sign)-> v_absx
/// ADD(v_absx,v_one) -> v_sum
/// SLTU(v_sum,v_absx)-> v_carry
/// ADD(v_hi,v_carry) -> v_hi
/// MOVE(v_hi)        -> rd (canonical non-virtual commit row)
/// ```
pub fn decomposition_sequence(rd: u8, rs1: u8, rs2: u8, alloc: &mut VirtualRegisterAllocator) -> Vec<DecomposedOp> {
    let rd = rd as u64;
    let rs1 = rs1 as u64;
    let rs2 = rs2 as u64;
    let v_sign = alloc.allocate();
    let v_one = alloc.allocate();
    let v_absx = alloc.allocate();
    let v_hi = alloc.allocate();

    vec![
        DecomposedOp::MovSign { dst: v_sign, src: rs1 },
        DecomposedOp::Sub {
            dst: v_one,
            lhs: 0,
            rhs: v_sign,
        },
        DecomposedOp::Xor {
            dst: v_absx,
            lhs: rs1,
            rhs: v_sign,
        },
        DecomposedOp::Add {
            dst: v_absx,
            lhs: v_absx,
            rhs: v_one,
        },
        DecomposedOp::Mulhu {
            dst: v_hi,
            lhs: v_absx,
            rhs: rs2,
        },
        DecomposedOp::Mul {
            dst: v_absx,
            lhs: v_absx,
            rhs: rs2,
        },
        DecomposedOp::Xor {
            dst: v_hi,
            lhs: v_hi,
            rhs: v_sign,
        },
        DecomposedOp::Xor {
            dst: v_absx,
            lhs: v_absx,
            rhs: v_sign,
        },
        DecomposedOp::Add {
            dst: v_sign,
            lhs: v_absx,
            rhs: v_one,
        },
        DecomposedOp::AdviceQuotient {
            dst: v_sign,
            op: RiscvOpcode::Sltu,
            lhs: v_sign,
            rhs: v_absx,
        },
        DecomposedOp::Add {
            dst: v_hi,
            lhs: v_hi,
            rhs: v_sign,
        },
        DecomposedOp::Move { dst: rd, src: v_hi },
    ]
}
