use p3_field::Field;

use crate::riscv::lookups::{RiscvLookupTable, RiscvOpcode};

pub fn build<F: Field>(xlen: usize) -> RiscvLookupTable<F> {
    RiscvLookupTable::new(RiscvOpcode::Xor, xlen)
}
