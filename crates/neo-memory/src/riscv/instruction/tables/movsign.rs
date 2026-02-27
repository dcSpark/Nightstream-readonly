use p3_field::Field;

use crate::riscv::lookups::{RiscvLookupTable, RiscvOpcode};

// Placeholder mapping for Step 1 structure; semantics are introduced in Step 2+.
pub fn build<F: Field>(xlen: usize) -> RiscvLookupTable<F> {
    RiscvLookupTable::new(RiscvOpcode::Sra, xlen)
}
