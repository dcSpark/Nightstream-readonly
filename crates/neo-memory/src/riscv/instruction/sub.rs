use crate::riscv::instruction::{InstructionDescriptor, OperandMode};
use crate::riscv::lookups::{compute_op, RiscvOpcode};

pub struct Sub;

impl InstructionDescriptor for Sub {
    fn opcode() -> Option<RiscvOpcode> {
        Some(RiscvOpcode::Sub)
    }

    fn operand_mode() -> OperandMode {
        OperandMode::SubtractOperands
    }
}

pub fn eval(rs1: u64, rs2: u64, xlen: usize) -> u64 {
    compute_op(RiscvOpcode::Sub, rs1, rs2, xlen)
}
