use crate::riscv::lookups::RiscvMemOp;

pub fn width_bytes(op: RiscvMemOp) -> usize {
    op.width_bytes()
}
