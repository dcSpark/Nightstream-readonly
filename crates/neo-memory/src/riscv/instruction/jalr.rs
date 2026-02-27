use crate::riscv::lookups::RiscvInstruction;

pub fn is_jalr(instr: &RiscvInstruction) -> bool {
    matches!(instr, RiscvInstruction::Jalr { .. })
}
