use crate::riscv::instruction::{DecomposedOp, InstructionDescriptor, OperandMode, VirtualRegisterAllocator};
use crate::riscv::lookups::{compute_op, RiscvOpcode};

pub struct Mulhu;

impl InstructionDescriptor for Mulhu {
    fn opcode() -> Option<RiscvOpcode> {
        Some(RiscvOpcode::Mulhu)
    }

    fn operand_mode() -> OperandMode {
        OperandMode::MultiplyOperands
    }
}

pub fn eval(rs1: u64, rs2: u64, xlen: usize) -> u64 {
    compute_op(RiscvOpcode::Mulhu, rs1, rs2, xlen)
}

/// Jolt-style MULHU decomposition sequence.
///
/// The first row computes the high-word product into a virtual accumulator.
/// The final non-virtual row commits `rd <- v_out`.
pub fn decomposition_sequence(rd: u8, rs1: u8, rs2: u8, alloc: &mut VirtualRegisterAllocator) -> Vec<DecomposedOp> {
    let rd = rd as u64;
    let rs1 = rs1 as u64;
    let rs2 = rs2 as u64;
    let v_out = alloc.allocate();

    vec![
        DecomposedOp::AdviceQuotient {
            dst: v_out,
            op: RiscvOpcode::Mulhu,
            lhs: rs1,
            rhs: rs2,
        },
        DecomposedOp::Move { dst: rd, src: v_out },
    ]
}
