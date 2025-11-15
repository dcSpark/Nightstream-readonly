//! Neo-Fold: High-level API for CCS folding
//!
//! This crate provides a convenient API for the CCS reduction protocol,
//! re-exporting functionality from the neo-reductions crate.

#![allow(non_snake_case)]

// Re-export everything from neo-reductions
pub use neo_reductions::{
    optimized_engine,
    paper_exact_engine,
    error,
    pi_ccs,
    pi_ccs_paper_exact,
    PiCcsProof,
    Challenges,
    pi_ccs_prove,
    pi_ccs_prove_simple,
    pi_ccs_verify,
    GenericCcsOracle,
    PiCcsError,
    sumcheck,
};

// Public folding coordinator (engine-agnostic orchestrator)
pub mod folding;

// Ergonomic per-step session API layered on top of the coordinator
pub mod session;
