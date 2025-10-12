//! Shared types used by both IVC and NIVC
//!
//! These types form the contract between IVC and NIVC, ensuring they can work together
//! without duplication.

use crate::F;
use neo_ccs::CcsStructure;

/// IVC Accumulator - the running state that gets folded at each step
#[derive(Clone, Debug)]
pub struct Accumulator {
    /// Digest of the running commitment coordinates (binding for ρ derivation).
    pub c_z_digest: [u8; 32],
    /// **NEW**: Full commitment coordinates (public in the IVC step CCS).
    /// These are the actual Ajtai commitment coordinates that get folded.
    pub c_coords: Vec<F>,
    /// Compact y-outputs (the "protocol-internal" y's exposed by folding).
    /// These are the Y_j(r) scalars produced by the folding pipeline.
    pub y_compact: Vec<F>,
    /// Step counter bound into the transcript (prevents replay/mixing).
    pub step: u64,
}

impl Default for Accumulator {
    fn default() -> Self {
        Self {
            c_z_digest: [0u8; 32],
            c_coords: vec![],
            y_compact: vec![],
            step: 0,
        }
    }
}

/// Commitment structure for full commitment binding (replaces digest-only binding)
#[derive(Clone, Debug)]
pub struct Commitment {
    /// The exact serialized commitment bytes
    pub bytes: Vec<u8>,
    /// Domain for separation (e.g., "CCS.witness", "RLC.fold")  
    pub domain: &'static str,
}

impl Commitment {
    pub fn new(bytes: Vec<u8>, domain: &'static str) -> Self {
        Self { bytes, domain }
    }
    
    /// Create from digest (compatibility with existing code)
    pub fn from_digest(digest: [u8; 32], domain: &'static str) -> Self {
        Self { bytes: digest.to_vec(), domain }
    }
}

/// Trusted specification for step circuit binding.
/// **CRITICAL**: These offsets must come from a trusted source (circuit specification),
/// NOT from the prover/proof. Trusting prover-supplied offsets defeats security.
#[derive(Clone, Debug)]
pub struct StepBindingSpec {
    /// Positions of y_step values in the step witness (for linked witness binding)
    pub y_step_offsets: Vec<usize>,
    /// Positions of step program-supplied public inputs in the step witness (binds the tail of step_x)
    pub step_program_input_witness_indices: Vec<usize>,
    /// Positions of y_prev (state input) in the step witness - must be length y_len
    pub y_prev_witness_indices: Vec<usize>,
    /// Index of the constant-1 column in the step witness (for stitching constraints)
    pub const1_witness_index: usize,
}

/// IVC-specific proof for a single step
#[derive(Clone)]
pub struct IvcProof {
    /// The cryptographic proof for this step
    pub step_proof: crate::Proof,
    /// The accumulator after this step
    pub next_accumulator: Accumulator,
    /// Step number in the IVC chain
    pub step: u64,
    /// Optional step-specific metadata
    pub metadata: Option<Vec<u8>>,
    /// The step relation's public input x (so the verifier can rebuild the global public input)
    pub step_public_input: Vec<F>,
    /// Full augmented public input [step_x || ρ || y_prev || y_next] used for folding
    pub step_augmented_public_input: Vec<F>,
    /// Augmented input for the previous accumulator (LHS instance) used during folding
    pub prev_step_augmented_public_input: Vec<F>,
    /// ρ derived from transcript for this step (public)
    pub step_rho: F,
    /// y_prev used for this step (public)
    pub step_y_prev: Vec<F>,
    /// y_next produced by folding for this step (public)
    pub step_y_next: Vec<F>,
    /// **NEW**: The per-step commitment coordinates used in opening/lincomb
    pub c_step_coords: Vec<F>,
    /// **ARCHITECTURE**: ME instances from folding (for Stage 5 Final SNARK Layer)
    pub me_instances: Option<Vec<neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>>>,
    /// **ARCHITECTURE**: Digit witnesses from folding (for Stage 5 Final SNARK Layer)  
    pub digit_witnesses: Option<Vec<neo_ccs::MeWitness<F>>>,
    /// **ARCHITECTURE**: Folding proof data (for Stage 5 Final SNARK Layer)
    pub folding_proof: Option<neo_fold::FoldingProof>,
    // 🔒 REMOVED: Binding metadata no longer in proof (security vulnerability!)
    // Verifier must get these from a trusted StepBindingSpec instead
}

/// Input for a single IVC step
#[derive(Clone, Debug)]
pub struct IvcStepInput<'a> {
    /// Neo parameters for proving
    pub params: &'a crate::NeoParams,
    /// Base step CCS (the computation to be proven)
    pub step_ccs: &'a CcsStructure<F>,
    /// Witness for the step computation
    pub step_witness: &'a [F],
    /// Previous accumulator state
    pub prev_accumulator: &'a Accumulator,
    /// Current step number
    pub step: u64,
    /// Optional public input for the step
    pub public_input: Option<&'a [F]>,
    /// **REAL per-step contribution used by Nova EV**: y_next = y_prev + ρ * y_step
    /// This is the actual step output that gets folded (NOT a placeholder)
    pub y_step: &'a [F],
    /// **SECURITY**: Trusted binding specification (NOT from prover!)
    pub binding_spec: &'a StepBindingSpec,
    /// If true, app inputs in step_x are transcript-only and are NOT read from the witness.
    /// In this mode, `step_program_input_witness_indices` may be empty even when step_x has app inputs (NIVC).
    pub transcript_only_app_inputs: bool,
    /// Optional: the previous step's augmented public input [step_x || ρ || y_prev || y_next].
    /// If provided, it is propagated into the proof's `prev_step_augmented_public_input` to
    /// ensure exact chaining linkage.
    pub prev_augmented_x: Option<&'a [F]>,
}

/// Result of executing an IVC step
#[derive(Clone)]
pub struct IvcStepResult {
    /// The proof for this step
    pub proof: IvcProof,
    /// Updated computation state (for continuing the chain)
    pub next_state: Vec<F>,
}

/// IVC chain proof containing multiple steps
#[derive(Clone)]
pub struct IvcChainProof {
    /// Individual step proofs
    pub steps: Vec<IvcProof>,
    /// Final accumulator state
    pub final_accumulator: Accumulator,
    /// Total number of steps in the chain
    pub chain_length: u64,
}

/// Optional metadata for structural commitment binding
#[derive(Clone, Default)]
pub struct BindingMetadata<'a> {
    pub kv_pairs: &'a [(&'a str, u128)],
}

/// Input for a single step in an IVC chain
#[derive(Clone, Debug)]
pub struct IvcChainStepInput {
    pub witness: Vec<F>,
    pub public_input: Option<Vec<F>>,
    pub step: u64,
}

/// Configuration for building the complete Nova augmentation CCS
#[derive(Debug, Clone)]
pub struct AugmentConfig {
    /// Length of hash inputs for in-circuit ρ derivation
    pub hash_input_len: usize,
    /// Length of compact y vector (accumulator state)
    pub y_len: usize,
    /// Ajtai public parameters (kappa, m, d)
    pub ajtai_pp: (usize, usize, usize),
    /// Number of commitment limbs/elements (typically d * kappa)
    pub commit_len: usize,
}

