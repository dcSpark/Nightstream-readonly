use crate::riscv::lookups::RiscvInstruction;

pub fn is_jal(instr: &RiscvInstruction) -> bool {
    matches!(instr, RiscvInstruction::Jal { .. })
}
