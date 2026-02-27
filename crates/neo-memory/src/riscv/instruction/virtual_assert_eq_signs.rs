use crate::riscv::instruction::{InstructionDescriptor, OperandMode};

pub struct VirtualAssertEqSigns;

impl InstructionDescriptor for VirtualAssertEqSigns {
    fn opcode() -> Option<crate::riscv::lookups::RiscvOpcode> {
        None
    }

    fn operand_mode() -> OperandMode {
        OperandMode::Interleaved
    }
}
