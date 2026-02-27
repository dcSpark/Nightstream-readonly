use crate::riscv::instruction::{InstructionDescriptor, OperandMode};

pub struct VirtualAdvice;

impl InstructionDescriptor for VirtualAdvice {
    fn opcode() -> Option<crate::riscv::lookups::RiscvOpcode> {
        None
    }

    fn operand_mode() -> OperandMode {
        OperandMode::Advice
    }
}
