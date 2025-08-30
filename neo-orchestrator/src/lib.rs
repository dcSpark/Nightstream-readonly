//! Neo Orchestrator: High-level SNARK interface for Neo Protocol
//!
//! This crate provides a simplified interface for the complete Neo protocol pipeline,
//! orchestrating the interaction between CCS constraints, Ajtai commitments, folding,
//! and Spartan2 compression.

use anyhow::Result;
use std::time::Instant;

// Re-export key types from neo-ccs and neo-fold
pub use neo_ccs::CcsStructure;
pub use neo_fold::{fold_to_single_me, spartan_compression};

// Type aliases for concrete types used in Neo protocol
type ConcreteMcsInstance = neo_ccs::McsInstance<Vec<u8>, neo_math::F>;
type ConcreteMcsWitness = neo_ccs::McsWitness<neo_math::F>;

/// Performance metrics for proof generation and verification
#[derive(Debug, Clone)]
pub struct Metrics {
    pub prove_ms: f64,
    pub proof_bytes: usize,
}

/// Errors that can occur during orchestration
#[derive(thiserror::Error, Debug)]
pub enum OrchestratorError {
    #[error("Folding failed: {0}")]
    FoldingError(String),
    
    #[error("Spartan compression failed: {0}")]
    CompressionError(String),
    
    #[error("Verification failed: {0}")]
    VerificationError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),
}

/// Generate a complete Neo SNARK proof for the given CCS instance and witness
///
/// This orchestrates the full pipeline:
/// 1. Folding: Reduce CCS instances to a single ME claim
/// 2. Compression: Convert ME claim to Spartan2 SNARK proof
///
/// Returns the proof bytes and performance metrics.
pub fn prove(
    ccs: &CcsStructure<neo_math::F>,
    instances: &[ConcreteMcsInstance],
    _witness: &[ConcreteMcsWitness],
) -> Result<(Vec<u8>, Metrics), OrchestratorError> {
    let prove_start = Instant::now();
    
    // Step 1: Execute folding pipeline
    let params = neo_params::NeoParams::goldilocks_127();
    let (me_instance, me_witness, _proofs) = fold_to_single_me(ccs, instances, &params)
        .map_err(|e| OrchestratorError::FoldingError(format!("{}", e)))?;
    
    // Step 2: Compress to Spartan2 SNARK
    let proof_bytes = spartan_compression::compress_me_to_spartan(&me_instance, &me_witness)
        .map_err(|e| OrchestratorError::CompressionError(format!("{}", e)))?;
    
    let prove_time = prove_start.elapsed();
    let metrics = Metrics {
        prove_ms: prove_time.as_secs_f64() * 1000.0,
        proof_bytes: proof_bytes.len(),
    };
    
    Ok((proof_bytes, metrics))
}

/// Verify a Neo SNARK proof against the given CCS instance
///
/// This verifies the complete proof pipeline by checking the Spartan2 proof
/// against the public inputs from the CCS instance.
pub fn verify(
    _ccs: &CcsStructure<neo_math::F>,
    instances: &[ConcreteMcsInstance], 
    proof: &[u8]
) -> bool {
    // Extract public inputs from the first instance (assuming single instance for now)
    let public_inputs = if let Some(instance) = instances.first() {
        &instance.x
    } else {
        &vec![]
    };
    
    // Verify the Spartan2 proof
    match spartan_compression::verify_spartan_me_proof(proof, public_inputs) {
        Ok(result) => result,
        Err(_) => false,
    }
}

/// Simplified prove interface for single instance/witness pairs
pub fn prove_single(
    ccs: &CcsStructure<neo_math::F>,
    instance: &ConcreteMcsInstance,
    witness: &ConcreteMcsWitness,
) -> Result<(Vec<u8>, Metrics), OrchestratorError> {
    prove(ccs, &[instance.clone()], &[witness.clone()])
}

/// Simplified verify interface for single instance
pub fn verify_single(
    ccs: &CcsStructure<neo_math::F>,
    instance: &ConcreteMcsInstance,
    proof: &[u8]
) -> bool {
    verify(ccs, &[instance.clone()], proof)
}

// Backward compatibility exports
pub mod neutronnova_integration {
    //! Placeholder module for backward compatibility
    //! This was used in the legacy integration but is no longer needed
    //! with the modern Neo architecture.
}

pub mod spartan2 {
    //! Direct integration with Spartan2 backend
    //! All functionality is now handled through the spartan_compression module
    //! in neo-fold for better separation of concerns.
}