use crate::riscv::lookups::BranchCondition;

pub fn evaluate(cond: BranchCondition, rs1: u64, rs2: u64, xlen: usize) -> bool {
    cond.evaluate(rs1, rs2, xlen)
}
