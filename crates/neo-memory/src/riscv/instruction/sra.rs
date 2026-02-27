use crate::riscv::instruction::{InstructionDescriptor, OperandMode};
use crate::riscv::lookups::{compute_op, RiscvOpcode};

pub struct Sra;

impl InstructionDescriptor for Sra {
    fn opcode() -> Option<RiscvOpcode> {
        Some(RiscvOpcode::Sra)
    }

    fn operand_mode() -> OperandMode {
        OperandMode::Interleaved
    }
}

pub fn eval(rs1: u64, rs2: u64, xlen: usize) -> u64 {
    compute_op(RiscvOpcode::Sra, rs1, rs2, xlen)
}
