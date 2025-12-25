//! Output Binding Integration for Shard Proofs.
//!
//! This module provides helpers to add output binding proofs to shard proofs,
//! ensuring that claimed program outputs are cryptographically bound to the
//! proven execution trace.
//!
//! ## Usage
//!
//! After generating a shard proof with Twist memory traces, use these helpers
//! to add output binding:
//!
//! ```ignore
//! // 1. Generate shard proof normally
//! let mut proof = fold_shard_prove(...)?;
//!
//! // 2. Add output binding
//! add_output_binding(&mut proof, &mut tr, &output_config)?;
//!
//! // 3. Verification includes output binding check
//! verify_with_output_binding(&proof, &mut tr, &output_config)?;
//! ```

use crate::PiCcsError;
use neo_math::{F, K};
use neo_memory::output_check::{
    verify_output_binding_proof, OutputBindingWitness, ProgramIO,
};
use neo_transcript::{Poseidon2Transcript, Transcript};

/// Configuration for output binding.
#[derive(Clone, Debug)]
pub struct OutputBindingConfig {
    /// Number of address bits for memory.
    pub num_bits: usize,
    /// The claimed program I/O (inputs and outputs with their addresses).
    pub program_io: ProgramIO<F>,
    /// Initial memory values at the challenge point (for verification).
    /// This should be derived from the public MemInit polynomial.
    pub val_init_at_r_prime: Option<K>,
    /// Terminal check value from finalized ME claims (for verification).
    /// This comes from the Twist inc_total sumcheck terminal.
    pub inc_terminal_value: Option<K>,
}

impl OutputBindingConfig {
    /// Create a new output binding config with just the I/O claims.
    pub fn new(num_bits: usize, program_io: ProgramIO<F>) -> Self {
        Self {
            num_bits,
            program_io,
            val_init_at_r_prime: None,
            inc_terminal_value: None,
        }
    }

    /// Set the initial value at the challenge point.
    pub fn with_val_init(mut self, val: K) -> Self {
        self.val_init_at_r_prime = Some(val);
        self
    }

    /// Set the terminal check value.
    pub fn with_inc_terminal(mut self, val: K) -> Self {
        self.inc_terminal_value = Some(val);
        self
    }
}

/// Data needed from Twist witness for output binding.
#[derive(Clone, Debug)]
pub struct TwistOutputData {
    /// Final memory state (values at all addresses after execution).
    pub final_memory_state: Vec<F>,
    /// Twist witness for output binding proof.
    pub witness: OutputBindingWitness,
}

/// Generate an output binding proof and attach it to a shard proof.
///
/// This should be called after `fold_shard_prove` with the Twist witness data.
///
/// # Arguments
/// * `proof` - The shard proof to augment with output binding.
/// * `tr` - The transcript (must be in the same state as after `fold_shard_prove`).
/// * `config` - Output binding configuration.
/// * `twist_data` - The Twist witness data needed for proof generation.
///
/// # Returns
/// The modified proof with output binding attached.
pub fn add_output_binding(
    proof: &mut crate::shard_proof_types::ShardProof,
    tr: &mut Poseidon2Transcript,
    config: &OutputBindingConfig,
    twist_data: &TwistOutputData,
) -> Result<(), PiCcsError> {
    // Domain separation
    tr.append_message(b"shard/output_binding_start", &[]);

    let output_proof = neo_memory::output_check::generate_output_binding_proof(
        tr,
        config.num_bits,
        config.program_io.clone(),
        &twist_data.final_memory_state,
        &twist_data.witness,
    )
    .map_err(|e| PiCcsError::ProtocolError(format!("output binding proof failed: {:?}", e)))?;

    proof.output_proof = Some(output_proof);
    Ok(())
}

/// Verify the output binding proof attached to a shard proof.
///
/// This should be called after `fold_shard_verify` with the same transcript.
///
/// # Arguments
/// * `proof` - The shard proof containing the output binding.
/// * `tr` - The transcript (must be in the same state as after `fold_shard_verify`).
/// * `config` - Output binding configuration (must include `val_init_at_r_prime` and `inc_terminal_value`).
///
/// # Returns
/// `Ok(())` if the output binding is valid, error otherwise.
pub fn verify_output_binding(
    proof: &crate::shard_proof_types::ShardProof,
    tr: &mut Poseidon2Transcript,
    config: &OutputBindingConfig,
) -> Result<(), PiCcsError> {
    let output_proof = proof.output_proof.as_ref().ok_or_else(|| {
        PiCcsError::InvalidInput("no output binding proof attached".into())
    })?;

    let val_init = config.val_init_at_r_prime.ok_or_else(|| {
        PiCcsError::InvalidInput("val_init_at_r_prime required for verification".into())
    })?;

    let inc_terminal = config.inc_terminal_value.ok_or_else(|| {
        PiCcsError::InvalidInput("inc_terminal_value required for verification".into())
    })?;

    // Domain separation (must match prover)
    tr.append_message(b"shard/output_binding_start", &[]);

    verify_output_binding_proof(
        tr,
        config.num_bits,
        config.program_io.clone(),
        val_init,
        output_proof,
        inc_terminal,
    )
    .map_err(|e| PiCcsError::ProtocolError(format!("output binding verification failed: {:?}", e)))
}

/// Check if a shard proof has output binding attached.
pub fn has_output_binding(proof: &crate::shard_proof_types::ShardProof) -> bool {
    proof.output_proof.is_some()
}

/// Create a simple output binding config for testing.
///
/// This creates a ProgramIO with a single output at the specified address.
pub fn simple_output_config(num_bits: usize, output_addr: u64, expected_output: F) -> OutputBindingConfig {
    let program_io = ProgramIO::new().with_output(output_addr, expected_output);
    OutputBindingConfig::new(num_bits, program_io)
}
