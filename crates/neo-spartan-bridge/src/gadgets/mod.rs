//! Gadgets for circuit synthesis
//!
//! This module provides bellpepper gadgets for:
//! - K-field arithmetic (2-limb representation over F)
//! - Π-CCS specific operations (eq, range, MLE evaluation)

pub mod k_field;
pub mod pi_ccs;
pub mod common;

pub use k_field::{KNum, KNumVar, k_add, k_mul, k_lift_from_f, alloc_k};

// Re-export range_product_gadget (safe to use)
pub use common::range_product_gadget;

// eq_gadget and mle_eval_gadget are gated behind unsafe-gadgets feature
// as they use cs.get() which doesn't work in production
#[cfg(feature = "unsafe-gadgets")]
pub use common::{eq_gadget, mle_eval_gadget};

