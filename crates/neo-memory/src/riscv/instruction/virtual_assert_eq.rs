use crate::riscv::instruction::{InstructionDescriptor, OperandMode};

pub struct VirtualAssertEq;

impl InstructionDescriptor for VirtualAssertEq {
    fn opcode() -> Option<crate::riscv::lookups::RiscvOpcode> {
        None
    }

    fn operand_mode() -> OperandMode {
        OperandMode::Interleaved
    }
}
