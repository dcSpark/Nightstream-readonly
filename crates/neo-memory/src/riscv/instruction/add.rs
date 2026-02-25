use crate::riscv::instruction::{InstructionDescriptor, OperandMode};
use crate::riscv::lookups::{compute_op, RiscvOpcode};

pub struct Add;

impl InstructionDescriptor for Add {
    fn opcode() -> Option<RiscvOpcode> {
        Some(RiscvOpcode::Add)
    }

    fn operand_mode() -> OperandMode {
        OperandMode::AddOperands
    }
}

pub fn eval(rs1: u64, rs2: u64, xlen: usize) -> u64 {
    compute_op(RiscvOpcode::Add, rs1, rs2, xlen)
}
