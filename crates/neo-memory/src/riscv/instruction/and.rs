use crate::riscv::instruction::{InstructionDescriptor, OperandMode};
use crate::riscv::lookups::{compute_op, RiscvOpcode};

pub struct And;

impl InstructionDescriptor for And {
    fn opcode() -> Option<RiscvOpcode> {
        Some(RiscvOpcode::And)
    }

    fn operand_mode() -> OperandMode {
        OperandMode::Interleaved
    }
}

pub fn eval(rs1: u64, rs2: u64, xlen: usize) -> u64 {
    compute_op(RiscvOpcode::And, rs1, rs2, xlen)
}
