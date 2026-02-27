use crate::riscv::lookups::RiscvInstruction;

pub fn is_auipc(instr: &RiscvInstruction) -> bool {
    matches!(instr, RiscvInstruction::Auipc { .. })
}
