use crate::riscv::instruction::{InstructionDescriptor, OperandMode};
use crate::riscv::lookups::{compute_op, RiscvOpcode};

pub struct Slt;

impl InstructionDescriptor for Slt {
    fn opcode() -> Option<RiscvOpcode> {
        Some(RiscvOpcode::Slt)
    }

    fn operand_mode() -> OperandMode {
        OperandMode::Interleaved
    }
}

pub fn eval(rs1: u64, rs2: u64, xlen: usize) -> u64 {
    compute_op(RiscvOpcode::Slt, rs1, rs2, xlen)
}
