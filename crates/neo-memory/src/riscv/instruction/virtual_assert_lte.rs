use crate::riscv::instruction::{InstructionDescriptor, OperandMode};

pub struct VirtualAssertLte;

impl InstructionDescriptor for VirtualAssertLte {
    fn opcode() -> Option<crate::riscv::lookups::RiscvOpcode> {
        None
    }

    fn operand_mode() -> OperandMode {
        OperandMode::Interleaved
    }
}
