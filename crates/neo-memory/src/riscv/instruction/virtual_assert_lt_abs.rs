use crate::riscv::instruction::{InstructionDescriptor, OperandMode};

pub struct VirtualAssertLtAbs;

impl InstructionDescriptor for VirtualAssertLtAbs {
    fn opcode() -> Option<crate::riscv::lookups::RiscvOpcode> {
        None
    }

    fn operand_mode() -> OperandMode {
        OperandMode::Interleaved
    }
}
