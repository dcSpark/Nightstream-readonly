//! Instruction lookup-table wrappers.
//!
//! These wrappers provide a table-per-file organization while delegating to the
//! current `riscv::lookups` implementations.

pub mod and;
pub mod andn;
pub mod eq;
pub mod movsign;
pub mod or;
pub mod range_check;
pub mod sll;
pub mod slt;
pub mod sltu;
pub mod sra;
pub mod srl;
pub mod upper_word;
pub mod xor;
