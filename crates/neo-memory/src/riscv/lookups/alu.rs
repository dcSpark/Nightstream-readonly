use super::bits::uninterleave_bits;
use super::isa::RiscvOpcode;

/// Compatibility shim for legacy callsites.
///
/// Canonical opcode semantics live in `riscv::instruction::compute_op`.
#[inline]
pub fn compute_op(op: RiscvOpcode, x: u64, y: u64, xlen: usize) -> u64 {
    crate::riscv::instruction::compute_op(op, x, y, xlen)
}

/// Compatibility shim for interleaved-index helpers used by small-xlen tests and
/// naive MLE evaluation paths.
#[inline]
pub fn lookup_entry(op: RiscvOpcode, index: u128, xlen: usize) -> u64 {
    let (x, y) = uninterleave_bits(index);
    compute_op(op, x, y, xlen)
}

/// Compatibility shim for signed branch comparisons.
#[inline]
pub(super) fn sign_extend(x: u64, xlen: usize) -> i64 {
    crate::riscv::instruction::sign_extend(x, xlen)
}
