use crate::riscv::instruction::{InstructionDescriptor, OperandMode};

pub struct VirtualAssertLtu;

impl InstructionDescriptor for VirtualAssertLtu {
    fn opcode() -> Option<crate::riscv::lookups::RiscvOpcode> {
        None
    }

    fn operand_mode() -> OperandMode {
        OperandMode::Interleaved
    }
}
