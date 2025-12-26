//! CPU integration module for Neo.
//!
//! This module provides CPU-related functionality for integrating with Neo's
//! shared memory/lookup bus architecture.
//!
//! ## Submodules
//!
//! - `constraints`: CPU-to-bus binding constraints (adapted from Jolt zkVM)
//! - `r1cs_adapter`: R1CS-based CPU adapter for the shared bus
//!
//! ## Credits
//!
//! The constraint logic in `constraints` is ported from the Jolt zkVM project:
//! - Repository: https://github.com/a16z/jolt
//! - Original file: `jolt-core/src/zkvm/r1cs/constraints.rs`
//! - License: Apache-2.0 / MIT

pub mod constraints;
pub mod r1cs_adapter;

// Re-export commonly used types
pub use constraints::*;
pub use r1cs_adapter::*;

