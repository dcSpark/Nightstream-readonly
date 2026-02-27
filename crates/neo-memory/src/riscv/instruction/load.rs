use crate::riscv::lookups::RiscvMemOp;

pub fn width_bytes(op: RiscvMemOp) -> usize {
    op.width_bytes()
}

pub fn is_sign_extend(op: RiscvMemOp) -> bool {
    op.is_sign_extend()
}
