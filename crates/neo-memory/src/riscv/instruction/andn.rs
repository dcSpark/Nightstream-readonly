use crate::riscv::instruction::{InstructionDescriptor, OperandMode};
use crate::riscv::lookups::{compute_op, RiscvOpcode};

pub struct Andn;

impl InstructionDescriptor for Andn {
    fn opcode() -> Option<RiscvOpcode> {
        Some(RiscvOpcode::Andn)
    }

    fn operand_mode() -> OperandMode {
        OperandMode::Interleaved
    }
}

pub fn eval(rs1: u64, rs2: u64, xlen: usize) -> u64 {
    compute_op(RiscvOpcode::Andn, rs1, rs2, xlen)
}
