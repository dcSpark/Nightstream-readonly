use crate::riscv::instruction::{InstructionDescriptor, OperandMode};
use crate::riscv::lookups::{compute_op, RiscvOpcode};

pub struct Xor;

impl InstructionDescriptor for Xor {
    fn opcode() -> Option<RiscvOpcode> {
        Some(RiscvOpcode::Xor)
    }

    fn operand_mode() -> OperandMode {
        OperandMode::Interleaved
    }
}

pub fn eval(rs1: u64, rs2: u64, xlen: usize) -> u64 {
    compute_op(RiscvOpcode::Xor, rs1, rs2, xlen)
}
