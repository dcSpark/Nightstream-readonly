use crate::riscv::instruction::{DecomposedOp, InstructionDescriptor, OperandMode, VirtualRegisterAllocator};
use crate::riscv::lookups::{compute_op, RiscvOpcode};

pub struct Mulh;

impl InstructionDescriptor for Mulh {
    fn opcode() -> Option<RiscvOpcode> {
        Some(RiscvOpcode::Mulh)
    }

    fn operand_mode() -> OperandMode {
        OperandMode::MultiplyOperands
    }
}

pub fn eval(rs1: u64, rs2: u64, xlen: usize) -> u64 {
    compute_op(RiscvOpcode::Mulh, rs1, rs2, xlen)
}

/// Jolt-style MULH decomposition sequence (8 rows).
///
/// ```text
/// MOVSIGN(rs1)      -> v_sx
/// MOVSIGN(rs2)      -> v_sy
/// MUL(v_sx, rs2)    -> v_sx
/// MUL(v_sy, rs1)    -> v_sy
/// MULHU(rs1, rs2)   -> v_acc
/// ADD(v_acc, v_sx)  -> v_acc
/// ADD(v_acc, v_sy)  -> v_acc
/// MOVE(v_acc)       -> rd (canonical non-virtual commit row)
/// ```
pub fn decomposition_sequence(rd: u8, rs1: u8, rs2: u8, alloc: &mut VirtualRegisterAllocator) -> Vec<DecomposedOp> {
    let rd = rd as u64;
    let rs1 = rs1 as u64;
    let rs2 = rs2 as u64;
    let v_sx = alloc.allocate();
    let v_sy = alloc.allocate();
    let v_acc = alloc.allocate();

    vec![
        DecomposedOp::MovSign { dst: v_sx, src: rs1 },
        DecomposedOp::MovSign { dst: v_sy, src: rs2 },
        DecomposedOp::Mul {
            dst: v_sx,
            lhs: v_sx,
            rhs: rs2,
        },
        DecomposedOp::Mul {
            dst: v_sy,
            lhs: v_sy,
            rhs: rs1,
        },
        DecomposedOp::Mulhu {
            dst: v_acc,
            lhs: rs1,
            rhs: rs2,
        },
        DecomposedOp::Add {
            dst: v_acc,
            lhs: v_acc,
            rhs: v_sx,
        },
        DecomposedOp::Add {
            dst: v_acc,
            lhs: v_acc,
            rhs: v_sy,
        },
        DecomposedOp::Move { dst: rd, src: v_acc },
    ]
}
