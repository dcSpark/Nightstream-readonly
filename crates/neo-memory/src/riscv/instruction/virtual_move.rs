use crate::riscv::instruction::{InstructionDescriptor, OperandMode};

pub struct VirtualMove;

impl InstructionDescriptor for VirtualMove {
    fn opcode() -> Option<crate::riscv::lookups::RiscvOpcode> {
        None
    }

    fn operand_mode() -> OperandMode {
        OperandMode::Interleaved
    }
}
