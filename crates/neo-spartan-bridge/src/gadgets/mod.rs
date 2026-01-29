//! Gadgets for circuit synthesis
//!
//! This module provides bellpepper gadgets for:
//! - K-field arithmetic (2-limb representation over F)
//! - Π-CCS specific operations (eq, range, MLE evaluation)

pub mod common;
pub mod k_field;
pub mod pi_ccs;

pub use k_field::{alloc_k, k_add, k_lift_from_f, k_mul, KNum, KNumVar};

#[cfg(feature = "unsafe-gadgets")]
pub use common::{eq_gadget, mle_eval_gadget, range_product_gadget};
