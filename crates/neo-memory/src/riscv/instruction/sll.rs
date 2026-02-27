use crate::riscv::instruction::{InstructionDescriptor, OperandMode};
use crate::riscv::lookups::{compute_op, RiscvOpcode};

pub struct Sll;

impl InstructionDescriptor for Sll {
    fn opcode() -> Option<RiscvOpcode> {
        Some(RiscvOpcode::Sll)
    }

    fn operand_mode() -> OperandMode {
        OperandMode::Interleaved
    }
}

pub fn eval(rs1: u64, rs2: u64, xlen: usize) -> u64 {
    compute_op(RiscvOpcode::Sll, rs1, rs2, xlen)
}
