use crate::riscv::instruction::{InstructionDescriptor, OperandMode};

pub struct VirtualMovsign;

impl InstructionDescriptor for VirtualMovsign {
    fn opcode() -> Option<crate::riscv::lookups::RiscvOpcode> {
        None
    }

    fn operand_mode() -> OperandMode {
        OperandMode::Interleaved
    }
}
