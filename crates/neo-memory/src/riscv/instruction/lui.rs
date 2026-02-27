use crate::riscv::lookups::RiscvInstruction;

pub fn is_lui(instr: &RiscvInstruction) -> bool {
    matches!(instr, RiscvInstruction::Lui { .. })
}
