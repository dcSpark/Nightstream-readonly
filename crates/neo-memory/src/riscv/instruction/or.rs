use crate::riscv::instruction::{InstructionDescriptor, OperandMode};
use crate::riscv::lookups::{compute_op, RiscvOpcode};

pub struct Or;

impl InstructionDescriptor for Or {
    fn opcode() -> Option<RiscvOpcode> {
        Some(RiscvOpcode::Or)
    }

    fn operand_mode() -> OperandMode {
        OperandMode::Interleaved
    }
}

pub fn eval(rs1: u64, rs2: u64, xlen: usize) -> u64 {
    compute_op(RiscvOpcode::Or, rs1, rs2, xlen)
}
