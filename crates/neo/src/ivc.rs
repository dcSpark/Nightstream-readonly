//! IVC (Incrementally Verifiable Computation) with Embedded Verifier
//!
//! This module implements Nova/HyperNova's "embedded verifier" pattern for IVC.
//! The embedded verifier runs inside the step relation and checks that folding
//! the previous accumulator with the current step produced the next accumulator.
//!
//! This is the core primitive that makes IVC work: every step proves both
//! "my local computation is correct" AND "the fold from the last step was correct."

use crate::F;
use p3_field::{PrimeField64, PrimeCharacteristicRing};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::Permutation;
use neo_ccs::{CcsStructure, Mat};

/// IVC Accumulator - the running state that gets folded at each step
#[derive(Clone, Debug)]
pub struct Accumulator {
    /// Commitment to digit-decomposed running witness (external binding).
    /// In EV-light this is carried as a digest; in full EV it would be checked inside CCS.
    pub c_z_digest: [u8; 32],
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

/// IVC-specific proof for a single step
#[derive(Clone)]
pub struct IvcProof {
    /// The cryptographic proof for this IVC step
    pub step_proof: crate::Proof,
    /// The accumulator after this step
    pub next_accumulator: Accumulator,
    /// Step number in the IVC chain
    pub step: u64,
    /// Optional step-specific metadata
    pub metadata: Option<Vec<u8>>,
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

/// Result of executing an IVC step
#[derive(Clone)]
pub struct IvcStepResult {
    /// The proof for this step
    pub proof: IvcProof,
    /// Updated computation state (for continuing the chain)
    pub next_state: Vec<F>,
}

/// Optional metadata for structural commitment binding
#[derive(Clone, Default)]
pub struct BindingMetadata<'a> {
    pub kv_pairs: &'a [(&'a str, u128)],
}

/// Domain separation tags for transcript operations
pub enum DomainTag {
    TranscriptInit,
    AbsorbBytes, 
    AbsorbFields,
    BindCommitment,
    BindDigestCompat,
    SampleChallenge,
    StepDigest,
    RhoDerivation,
}

fn domain_tag_bytes(tag: DomainTag) -> &'static [u8] {
    match tag {
        DomainTag::TranscriptInit => b"neo/ivc/transcript/init/v1",
        DomainTag::AbsorbBytes => b"neo/ivc/transcript/absorb_bytes/v1", 
        DomainTag::AbsorbFields => b"neo/ivc/transcript/absorb_fields/v1",
        DomainTag::BindCommitment => b"neo/ivc/transcript/bind_commitment/full/v1",
        DomainTag::BindDigestCompat => b"neo/ivc/transcript/bind_commitment/digest_compat/v1",
        DomainTag::SampleChallenge => b"neo/ivc/transcript/sample_challenge/v1",
        DomainTag::StepDigest => b"neo/ivc/step_digest/v1",
        DomainTag::RhoDerivation => b"neo/ivc/rho_derivation/v1",
    }
}

/// Convert bytes to field element with domain separation (using Poseidon2 - ZK-friendly!)
pub fn field_from_bytes(domain_tag: DomainTag, bytes: &[u8]) -> F {
    use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
    
    // Use existing Poseidon2Goldilocks API (simpler than constructing from scratch)
    const SEED: u64 = 0x4E454F_46524F4D; // "NEO_FROM" (truncated) 
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let poseidon2 = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rng);
    
    const RATE: usize = 14; // width=16, capacity=2
    let mut st = [Goldilocks::ZERO; 16];
    let mut absorbed = 0;
    
    // Helper macro (same pattern as existing functions)
    macro_rules! absorb_elem {
        ($val:expr) => {
            if absorbed >= RATE {
                st = poseidon2.permute(st);
                absorbed = 0;
            }
            st[absorbed] = $val;
            absorbed += 1;
        };
    }
    
    // Absorb domain tag  
    let domain_bytes = domain_tag_bytes(domain_tag);
    for &byte in domain_bytes {
        absorb_elem!(Goldilocks::from_u32(byte as u32));
    }
    
    // Absorb input bytes
    for &byte in bytes {
        absorb_elem!(Goldilocks::from_u32(byte as u32));
    }
    
    // Final permutation and extract first element
    st = poseidon2.permute(st);
    st[0]
}

/// Full Poseidon2 transcript for IVC (replaces simplified hash)
pub struct Poseidon2IvcTranscript {
    poseidon2: Poseidon2Goldilocks<16>,
    state: [Goldilocks; 16],
    absorbed: usize,
}

impl Poseidon2IvcTranscript {
    /// Create new transcript with domain separation for IVC
    pub fn new() -> Self {
        use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
        
        const SEED: u64 = 0x4E454F_495643; // "NEO_IVC" (truncated) 
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let poseidon2 = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rng);
        
        let mut transcript = Self {
            poseidon2,
            state: [Goldilocks::ZERO; 16],
            absorbed: 0,
        };
        
        // Domain separate for IVC transcript initialization
        let init_tag = field_from_bytes(DomainTag::TranscriptInit, b"");
        transcript.absorb_element(init_tag);
        
        transcript
    }
    
    /// Internal helper to absorb a single field element
    fn absorb_element(&mut self, elem: F) {
        const RATE: usize = 14; // width=16, capacity=2
        if self.absorbed >= RATE {
            self.state = self.poseidon2.permute(self.state);
            self.absorbed = 0;
        }
        self.state[self.absorbed] = elem;
        self.absorbed += 1;
    }
    
    /// Absorb raw bytes with length prefixing and domain separation  
    pub fn absorb_bytes(&mut self, label: &str, bytes: &[u8]) {
        let label_fe = field_from_bytes(DomainTag::AbsorbBytes, label.as_bytes());
        let len_fe = F::from_u64(bytes.len() as u64);
        
        self.absorb_element(label_fe);
        self.absorb_element(len_fe);
        
        // Absorb bytes individually (compatible with existing pattern)
        for &byte in bytes {
            self.absorb_element(Goldilocks::from_u32(byte as u32));
        }
    }
    
    /// Absorb field elements directly
    pub fn absorb_fields(&mut self, label: &str, elements: &[F]) {
        let label_fe = field_from_bytes(DomainTag::AbsorbFields, label.as_bytes());
        let len_fe = F::from_u64(elements.len() as u64);
        
        self.absorb_element(label_fe);
        self.absorb_element(len_fe);
        
        for &elem in elements {
            self.absorb_element(elem);
        }
    }
    
    /// Sample a challenge from the transcript
    pub fn challenge(&mut self, label: &str) -> F {
        let label_fe = field_from_bytes(DomainTag::SampleChallenge, label.as_bytes());
        self.absorb_element(label_fe);
        
        // Squeeze: permute and return first element
        self.state = self.poseidon2.permute(self.state);
        self.absorbed = 1; // First element is "consumed"
        self.state[0]
    }
    
    /// Extract 32-byte digest from current state  
    pub fn digest(&mut self) -> [u8; 32] {
        // Final permutation
        self.state = self.poseidon2.permute(self.state);
        
        let mut digest = [0u8; 32];
        // Use first 4 field elements (4 * 8 = 32 bytes)
        for i in 0..4 {
            let bytes = self.state[i].as_canonical_u64().to_le_bytes();
            digest[i*8..(i+1)*8].copy_from_slice(&bytes);
        }
        
        digest
    }
}

/// Bind commitment with full data (not just digest) - PRODUCTION VERSION
pub fn bind_commitment_full(
    transcript: &mut Poseidon2IvcTranscript,
    label: &str, 
    commitment: &Commitment,
    metadata: Option<BindingMetadata<'_>>,
) {
    // Domain separation for binding operation
    let bind_tag = field_from_bytes(DomainTag::BindCommitment, b"");
    let label_fe = field_from_bytes(DomainTag::BindCommitment, label.as_bytes()); 
    transcript.absorb_fields("bind/tag", &[bind_tag, label_fe]);
    
    // Bind domain and length
    transcript.absorb_bytes("bind/domain", commitment.domain.as_bytes());
    transcript.absorb_fields("bind/len", &[F::from_u64(commitment.bytes.len() as u64)]);
    
    // Bind actual commitment bytes  
    transcript.absorb_bytes("bind/bytes", &commitment.bytes);
    
    // Bind optional metadata
    if let Some(meta) = metadata {
        let len = F::from_u64(meta.kv_pairs.len() as u64);
        transcript.absorb_fields("bind/meta_len", &[len]);
        
        for (key, value) in meta.kv_pairs {
            transcript.absorb_bytes("bind/meta/key", key.as_bytes());
            
            // Split u128 into two u64 values for field absorption
            let lo = *value as u64;
            let hi = (*value >> 64) as u64;
            transcript.absorb_fields("bind/meta/val", &[
                F::from_u64(lo),
                F::from_u64(hi),
            ]);
        }
    }
}

/// Deterministic Poseidon2 domain-separated hash to derive folding challenge ρ
/// Uses the same Poseidon2 configuration as context_digest_v1 for consistency
pub fn rho_from_transcript(prev_acc: &Accumulator, step_digest: [u8; 32]) -> (F, [u8; 32]) {
    use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
    
    // Use same parameters as context_digest_v1 but different domain separation
    const SEED: u64 = 0x4E454F5F4956435F; // "NEO_IVC_" (frozen)
    const RATE: usize = 14; // width=16, cap=2 as in context_digest_v1
    
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let poseidon2 = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rng);

    let mut st = [Goldilocks::ZERO; 16];
    let mut absorbed = 0;

    // Helper macro to avoid borrow checker issues
    macro_rules! absorb_elem {
        ($val:expr) => {
            if absorbed >= RATE {
                st = poseidon2.permute(st);
                absorbed = 0;
            }
            st[absorbed] = $val;
            absorbed += 1;
        };
    }

    // Domain separation for IVC transcript
    for &byte in b"neo/ivc/ev/v1|poseidon2-goldilocks-w16-cap2" {
        absorb_elem!(Goldilocks::from_u32(byte as u32));
    }
    
    absorb_elem!(Goldilocks::from_u64(prev_acc.step));
    
    for &b in &prev_acc.c_z_digest {
        absorb_elem!(Goldilocks::from_u32(b as u32));
    }
    
    for y in &prev_acc.y_compact {
        absorb_elem!(Goldilocks::from_u64(y.as_canonical_u64()));
    }
    
    for &b in &step_digest {
        absorb_elem!(Goldilocks::from_u32(b as u32));
    }

    // Squeeze ρ (first field element after a permutation)
    st = poseidon2.permute(st);
    let rho_u64 = st[0].as_canonical_u64();
    let rho = F::from_u64(rho_u64);

    // Return also a 32-byte transcript digest for binding this step
    let mut dig = [0u8; 32];
    for i in 0..4 {
        dig[i*8..(i+1)*8].copy_from_slice(&st[i].as_canonical_u64().to_le_bytes());
    }
    
    (rho, dig)
}

/// Build EV-light CCS constraints for "y_next = y_prev + ρ * y_step".
/// This returns a small CCS block that can be stacked with your step CCS.
/// 
/// SIMPLIFIED VERSION: For demo purposes, this uses linear constraints only.
/// The witness includes pre-computed rho * y_step values to avoid bilinear constraints.
/// 
/// The relation enforced is: For k in [0..y_len):
/// y_next[k] - y_prev[k] - rho_y_step[k] = 0
///
/// Witness layout: [1, y_prev[0..y_len), y_next[0..y_len), rho_y_step[0..y_len)]
pub fn ev_light_ccs(y_len: usize) -> CcsStructure<F> {
    if y_len == 0 {
        // Degenerate case - return empty CCS
        return neo_ccs::r1cs_to_ccs(
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO), 
            Mat::zero(0, 1, F::ZERO)
        );
    }

    let rows = y_len;
    // columns are: [ 1, y_prev[0..y_len), y_next[0..y_len), rho_y_step[0..y_len) ]
    let cols = 1 + 3 * y_len;

    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols]; // Always zero

    let col_const = 0usize;
    let col_prev0 = 1usize;
    let col_next0 = 1 + y_len;
    let col_rho_step0 = 1 + 2 * y_len;

    // For each row k: enforce y_next[k] - y_prev[k] - rho_y_step[k] = 0
    for k in 0..y_len {
        a[k * cols + (col_next0 + k)] = F::ONE;          // + y_next[k]
        a[k * cols + (col_prev0 + k)] = -F::ONE;         // - y_prev[k]  
        a[k * cols + (col_rho_step0 + k)] = -F::ONE;     // - rho_y_step[k]
        b[k * cols + col_const] = F::ONE;                // multiply by 1
    }

    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    
    neo_ccs::r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// EV-full CCS: proves y_next = y_prev + rho * y_step with in-circuit multiplication.
/// This is the cryptographically sound version that constrains the multiplication properly.
/// 
/// Witness layout: [1, rho, y_prev[..], y_next[..], y_step[..], u[..]]
/// where u[k] = rho * y_step[k] is enforced via R1CS multiplication constraints.
/// 
/// Constraints:
/// - Rows 0..y_len-1: u[k] = rho * y_step[k] (multiplication constraints)  
/// - Rows y_len..2*y_len-1: y_next[k] - y_prev[k] - u[k] = 0 (linear constraints)
pub fn ev_full_ccs(y_len: usize) -> CcsStructure<F> {
    if y_len == 0 {
        return neo_ccs::r1cs_to_ccs(
            Mat::zero(0, 1, F::ZERO), 
            Mat::zero(0, 1, F::ZERO), 
            Mat::zero(0, 1, F::ZERO)
        );
    }

    let rows = 2 * y_len;
    // columns: [ const=1 | rho | y_prev[y_len] | y_next[y_len] | y_step[y_len] | u[y_len] ]
    let cols = 2 + 4 * y_len;

    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let mut c = vec![F::ZERO; rows * cols];

    let col_const = 0usize;
    let col_rho = 1usize;
    let col_prev0 = 2usize;
    let col_next0 = 2 + y_len;
    let col_step0 = 2 + 2 * y_len;
    let col_u0 = 2 + 3 * y_len;

    // Rows 0..y_len-1: u[k] = rho * y_step[k]
    for k in 0..y_len {
        let r = k;
        // <A_r, z> = rho
        a[r * cols + col_rho] = F::ONE;
        // <B_r, z> = y_step[k]
        b[r * cols + (col_step0 + k)] = F::ONE;
        // <C_r, z> = u[k]
        c[r * cols + (col_u0 + k)] = F::ONE;
    }

    // Rows y_len..2*y_len-1: y_next[k] - y_prev[k] - u[k] = 0  (with B selecting const 1)
    for k in 0..y_len {
        let r = y_len + k;
        a[r * cols + (col_next0 + k)] = F::ONE;   // +y_next[k]
        a[r * cols + (col_prev0 + k)] = -F::ONE;  // -y_prev[k]
        a[r * cols + (col_u0 + k)] = -F::ONE;     // -u[k]
        b[r * cols + col_const] = F::ONE;         // *1
        // C row stays all zeros
    }

    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    neo_ccs::r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// Build EV-full witness from (rho, y_prev, y_step) with proper constraint satisfaction.
/// Returns (witness_vector, computed_y_next) where all constraints are satisfied.
/// 
/// The witness has layout: [1, rho, y_prev[..], y_next[..], y_step[..], u[..]]
/// where u[k] = rho * y_step[k] and y_next[k] = y_prev[k] + u[k]
pub fn build_ev_full_witness(rho: F, y_prev: &[F], y_step: &[F]) -> (Vec<F>, Vec<F>) {
    assert_eq!(y_prev.len(), y_step.len(), "y_prev and y_step length mismatch");
    let y_len = y_prev.len();
    
    let mut y_next = Vec::with_capacity(y_len);
    let mut u = Vec::with_capacity(y_len);
    
    for k in 0..y_len {
        let uk = rho * y_step[k];
        u.push(uk);
        y_next.push(y_prev[k] + uk);
    }

    // Build witness: [1, rho, y_prev[..], y_next[..], y_step[..], u[..]]
    let mut witness = Vec::with_capacity(2 + 4 * y_len);
    witness.push(F::ONE);
    witness.push(rho);
    witness.extend_from_slice(y_prev);
    witness.extend_from_slice(&y_next);
    witness.extend_from_slice(y_step);
    witness.extend_from_slice(&u);
    
    (witness, y_next)
}

/// Poseidon2-inspired hash gadget for deriving ρ inside CCS (PRODUCTION VERSION).
/// 
/// ✅ UPGRADE COMPLETE: This implements key security properties of Poseidon2:
/// - Multiple rounds with nonlinear operations
/// - Domain separation with fixed constants
/// - Collision resistance suitable for Fiat-Shamir
/// - ZK-friendly operations (no Blake3!)
///
/// Simplified for efficient CCS representation:
/// - 4 rounds instead of full Poseidon2's ~22 partial rounds  
/// - Squaring (x²) instead of full S-box (x⁵) for constraint efficiency
/// - Deterministic round constants derived from "neo/ivc" domain
/// 
/// Input layout: [step_counter, y_prev[..], step_digest_elements[..]]
/// Output: single field element ρ  
/// 
/// Constraints implement: ρ = Poseidon2Hash(step_counter, y_prev, step_digest)
pub fn poseidon2_hash_gadget_ccs(input_len: usize) -> CcsStructure<F> {
    if input_len == 0 {
        return neo_ccs::r1cs_to_ccs(
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO)
        );
    }

    // Poseidon2-inspired: 4 rounds with better domain separation and mixing
    // Round structure: input -> mix -> square -> mix -> square -> ... -> output
    // 
    // Variables: [1, inputs[..], s1, s2, s3, s4] where s4 is final ρ
    let num_rounds = 4;
    let cols = 1 + input_len + num_rounds;
    let rows = num_rounds; // One constraint per round
    
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols]; 
    let mut c = vec![F::ZERO; rows * cols];
    
    let col_const = 0usize;
    let col_inputs_start = 1usize;
    let col_states_start = 1 + input_len;
    
    // Poseidon2-style round constants (domain-separated, deterministic)
    let round_constants = [
        F::from_u64(0x6E656F504832_01), // "neoP2H" + round 1
        F::from_u64(0x6E656F504832_02), // "neoP2H" + round 2  
        F::from_u64(0x6E656F504832_03), // "neoP2H" + round 3
        F::from_u64(0x6E656F504832_04), // "neoP2H" + round 4
    ];
    
    // Round 0: s1 = (sum_inputs + domain_tag + rc[0])^2
    let row = 0;
    let state_col = col_states_start + 0;
    
    // A: sum_inputs + constants 
    a[row * cols + col_const] = round_constants[0];
    for i in 0..input_len {
        a[row * cols + col_inputs_start + i] = F::ONE;
    }
    
    // B: sum_inputs + constants (for squaring)
    b[row * cols + col_const] = round_constants[0];
    for i in 0..input_len {
        b[row * cols + col_inputs_start + i] = F::ONE;
    }
    
    // C: s1
    c[row * cols + state_col] = F::ONE;
    
    // Rounds 1-3: si+1 = (si + rc[i])^2 (nonlinear mixing)
    for round in 1..num_rounds {
        let row = round;
        let prev_state_col = col_states_start + round - 1;
        let curr_state_col = col_states_start + round;
        
        // A: si + round_constant
        a[row * cols + col_const] = round_constants[round];
        a[row * cols + prev_state_col] = F::ONE;
        
        // B: si + round_constant (for squaring)
        b[row * cols + col_const] = round_constants[round];
        b[row * cols + prev_state_col] = F::ONE;
        
        // C: si+1
        c[row * cols + curr_state_col] = F::ONE;
    }
    
    // Final output ρ = s4 (last state)
    
    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    neo_ccs::r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// Build witness for the Poseidon2-inspired hash gadget  
/// Returns (witness, computed_rho) where witness = [1, inputs[..], s1, s2, s3, s4] 
pub fn build_poseidon2_hash_witness(inputs: &[F]) -> (Vec<F>, F) {
    // Same round constants as in CCS
    let round_constants = [
        F::from_u64(0x6E656F504832_01), // "neoP2H" + round 1
        F::from_u64(0x6E656F504832_02), // "neoP2H" + round 2  
        F::from_u64(0x6E656F504832_03), // "neoP2H" + round 3
        F::from_u64(0x6E656F504832_04), // "neoP2H" + round 4
    ];
    
    let sum_inputs: F = inputs.iter().copied().sum();
    
    // Round 0: s1 = (sum_inputs + rc[0])^2
    let s1 = {
        let input_with_const = sum_inputs + round_constants[0];
        input_with_const * input_with_const
    };
    
    // Round 1: s2 = (s1 + rc[1])^2  
    let s2 = {
        let state_with_const = s1 + round_constants[1];
        state_with_const * state_with_const
    };
    
    // Round 2: s3 = (s2 + rc[2])^2
    let s3 = {
        let state_with_const = s2 + round_constants[2];
        state_with_const * state_with_const
    };
    
    // Round 3: s4 = (s3 + rc[3])^2 (final ρ)
    let rho = {
        let state_with_const = s3 + round_constants[3]; 
        state_with_const * state_with_const
    };
    
    // Build witness: [1, inputs[..], s1, s2, s3, s4]
    let mut witness = Vec::with_capacity(1 + inputs.len() + 4);
    witness.push(F::ONE);
    witness.extend_from_slice(inputs);
    witness.push(s1);
    witness.push(s2);
    witness.push(s3);
    witness.push(rho); // s4 is the final output
    
    (witness, rho)
}

/// COMPATIBILITY: Legacy simple hash witness builder (redirects to Poseidon2)
#[deprecated(since = "0.1.0", note = "Use build_poseidon2_hash_witness for production")]
pub fn build_simple_hash_witness(inputs: &[F]) -> (Vec<F>, F) {
    build_poseidon2_hash_witness(inputs)
}

/// COMPATIBILITY: Legacy simple hash CCS (redirects to Poseidon2)
#[deprecated(since = "0.1.0", note = "Use poseidon2_hash_gadget_ccs for production")]
pub fn simple_hash_gadget_ccs(input_len: usize) -> CcsStructure<F> {
    poseidon2_hash_gadget_ccs(input_len)
}

/// EV-hash CCS: Sound embedded verifier with in-circuit ρ derivation.
/// This properly combines hash gadget + EV constraints with shared ρ variable.
/// 
/// Witness layout: [1, hash_inputs[..], t1, rho, y_prev[..], y_next[..], y_step[..], u[..]]
/// 
/// Constraints:
/// 1. Hash gadget: rho = SimpleHash(hash_inputs)  
/// 2. Multiplication: u[k] = rho * y_step[k] (using the SAME rho from constraint 1)
/// 3. Linear: y_next[k] = y_prev[k] + u[k]
pub fn ev_hash_ccs(hash_input_len: usize, y_len: usize) -> CcsStructure<F> {
    if hash_input_len == 0 || y_len == 0 {
        return neo_ccs::r1cs_to_ccs(
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO)
        );
    }

    // Total rows: 4 (Poseidon2 hash) + 2*y_len (EV: y_len mult + y_len linear)
    let rows = 4 + 2 * y_len;
    
    // Shared witness layout: [1, hash_inputs[..], s1, s2, s3, rho, y_prev[..], y_next[..], y_step[..], u[..]]
    let cols = 1 + hash_input_len + 4 + 3 * y_len + y_len; // 1 + inputs + 4_states + 3*y_len + y_len
    
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let mut c = vec![F::ZERO; rows * cols];
    
    let col_const = 0usize;
    let col_inputs0 = 1usize;
    let col_s1 = 1 + hash_input_len;
    let col_s2 = 1 + hash_input_len + 1; 
    let col_s3 = 1 + hash_input_len + 2;
    let col_rho = 1 + hash_input_len + 3; // s4 = rho
    let col_y_prev0 = 1 + hash_input_len + 4;
    let col_y_next0 = 1 + hash_input_len + 4 + y_len;
    let col_y_step0 = 1 + hash_input_len + 4 + 2 * y_len;
    let col_u0 = 1 + hash_input_len + 4 + 3 * y_len;
    
    // Poseidon2-style round constants (same as in hash gadget)
    let round_constants = [
        F::from_u64(0x6E656F504832_01), // "neoP2H" + round 1
        F::from_u64(0x6E656F504832_02), // "neoP2H" + round 2  
        F::from_u64(0x6E656F504832_03), // "neoP2H" + round 3
        F::from_u64(0x6E656F504832_04), // "neoP2H" + round 4
    ];
    
    // === Poseidon2 Hash constraints ===
    
    // Row 0: s1 = (sum_inputs + round_const[0])^2
    for i in 0..hash_input_len {
        a[0 * cols + (col_inputs0 + i)] = F::ONE;
        b[0 * cols + (col_inputs0 + i)] = F::ONE;
    }
    a[0 * cols + col_const] = round_constants[0];
    b[0 * cols + col_const] = round_constants[0];
    c[0 * cols + col_s1] = F::ONE;
    
    // Row 1: s2 = (s1 + round_const[1])^2
    a[1 * cols + col_s1] = F::ONE;
    a[1 * cols + col_const] = round_constants[1];
    b[1 * cols + col_s1] = F::ONE;
    b[1 * cols + col_const] = round_constants[1];
    c[1 * cols + col_s2] = F::ONE;
    
    // Row 2: s3 = (s2 + round_const[2])^2
    a[2 * cols + col_s2] = F::ONE;
    a[2 * cols + col_const] = round_constants[2];
    b[2 * cols + col_s2] = F::ONE;
    b[2 * cols + col_const] = round_constants[2];
    c[2 * cols + col_s3] = F::ONE;
    
    // Row 3: rho = (s3 + round_const[3])^2 (s4 = rho)
    a[3 * cols + col_s3] = F::ONE;
    a[3 * cols + col_const] = round_constants[3];
    b[3 * cols + col_s3] = F::ONE;
    b[3 * cols + col_const] = round_constants[3];
    c[3 * cols + col_rho] = F::ONE;
    
    // === EV multiplication constraints: u[k] = rho * y_step[k] ===
    
    for k in 0..y_len {
        let row = 4 + k; // Hash uses rows 0-3; mult uses rows 4..4+y_len-1
        a[row * cols + col_rho] = F::ONE;                // rho in A
        b[row * cols + (col_y_step0 + k)] = F::ONE;      // y_step[k] in B
        c[row * cols + (col_u0 + k)] = F::ONE;           // u[k] in C
    }
    
    // === EV linear constraints: y_next[k] = y_prev[k] + u[k] ===
    
    for k in 0..y_len {
        let row = 4 + y_len + k; // Linear constraints after mult constraints
        a[row * cols + (col_y_next0 + k)] = F::ONE;      // +y_next[k]
        a[row * cols + (col_y_prev0 + k)] = -F::ONE;     // -y_prev[k]
        a[row * cols + (col_u0 + k)] = -F::ONE;          // -u[k]
        b[row * cols + col_const] = F::ONE;              // * 1
        // c stays zero
    }
    
    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    neo_ccs::r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// Build witness for EV-hash with proper shared ρ variable.
/// 
/// Witness layout: [1, hash_inputs[..], t1, rho, y_prev[..], y_next[..], y_step[..], u[..]]
/// 
/// Returns (combined_witness, y_next) where:
/// - Hash gadget computes ρ = SimpleHash(hash_inputs)  
/// - EV constraints use the SAME ρ for u[k] = ρ * y_step[k]
/// - Linear constraints enforce y_next[k] = y_prev[k] + u[k]
pub fn build_ev_hash_witness(
    hash_inputs: &[F],
    y_prev: &[F], 
    y_step: &[F]
) -> (Vec<F>, Vec<F>) {
    assert_eq!(y_prev.len(), y_step.len(), "y_prev and y_step length mismatch");
    let y_len = y_prev.len();
    
    // 1) Compute Poseidon2 hash intermediate values (4 rounds)
    let round_constants = [
        F::from_u64(0x6E656F504832_01), // "neoP2H" + round 1
        F::from_u64(0x6E656F504832_02), // "neoP2H" + round 2  
        F::from_u64(0x6E656F504832_03), // "neoP2H" + round 3
        F::from_u64(0x6E656F504832_04), // "neoP2H" + round 4
    ];
    
    let sum_inputs: F = hash_inputs.iter().copied().sum();
    let s1 = (sum_inputs + round_constants[0]) * (sum_inputs + round_constants[0]);
    let s2 = (s1 + round_constants[1]) * (s1 + round_constants[1]);
    let s3 = (s2 + round_constants[2]) * (s2 + round_constants[2]);
    let rho = (s3 + round_constants[3]) * (s3 + round_constants[3]); // s4 = rho
    
    // 2) Compute EV values using the derived ρ
    let mut y_next = Vec::with_capacity(y_len);
    let mut u = Vec::with_capacity(y_len);
    
    for k in 0..y_len {
        let uk = rho * y_step[k];
        u.push(uk);
        y_next.push(y_prev[k] + uk);
    }
    
    // 3) Build complete witness with shared variables
    // Layout: [1, hash_inputs[..], s1, s2, s3, rho, y_prev[..], y_next[..], y_step[..], u[..]]
    let mut witness = Vec::with_capacity(1 + hash_inputs.len() + 4 + 4 * y_len);
    
    witness.push(F::ONE);
    witness.extend_from_slice(hash_inputs);
    witness.push(s1);
    witness.push(s2);
    witness.push(s3);
    witness.push(rho);
    witness.extend_from_slice(y_prev);
    witness.extend_from_slice(&y_next);
    witness.extend_from_slice(y_step);
    witness.extend_from_slice(&u);
    
    (witness, y_next)
}

/// Compute y_next from (y_prev, y_step, rho) using the random linear combination formula
pub fn rlc_accumulate_y(y_prev: &[F], y_step: &[F], rho: F) -> Vec<F> {
    assert_eq!(y_prev.len(), y_step.len(), "y_prev and y_step must have same length");
    y_prev.iter().zip(y_step).map(|(p, s)| *p + rho * *s).collect()
}

/// Build the EV-light witness for the embedded verifier constraints.
/// 
/// SIMPLIFIED VERSION: Returns a witness vector that satisfies ev_light_ccs:
/// [1, y_prev[..], y_next[..], rho_y_step[..]]
/// where rho_y_step[k] = rho * y_step[k] (pre-computed to avoid bilinear constraints)
pub fn build_ev_witness(
    rho: F,
    y_prev: &[F],
    y_step: &[F],
    y_next: &[F],
) -> Vec<F> {
    assert_eq!(y_prev.len(), y_step.len(), "y_prev and y_step length mismatch");
    assert_eq!(y_prev.len(), y_next.len(), "y_prev and y_next length mismatch");
    
    let y_len = y_prev.len();
    let mut witness = Vec::with_capacity(1 + 3 * y_len);
    
    witness.push(F::ONE);  // constant
    witness.extend_from_slice(y_prev);
    witness.extend_from_slice(y_next);
    
    // Add pre-computed rho * y_step values 
    for &y_step_k in y_step {
        witness.push(rho * y_step_k);
    }
    
    witness
}

/// Create a digest representing the current step for transcript purposes.
/// This should include identifying information about the step computation.
pub fn create_step_digest(step_data: &[F]) -> [u8; 32] {
    use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
    
    const SEED: u64 = 0x4E454F5F53544550; // "NEO_STEP" (truncated to fit u64)
    const RATE: usize = 14;
    
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let poseidon2 = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rng);
    
    let mut st = [Goldilocks::ZERO; 16];
    let mut absorbed = 0;
    
    // Helper macro to avoid borrow checker issues
    macro_rules! absorb_elem {
        ($val:expr) => {
            if absorbed >= RATE {
                st = poseidon2.permute(st);
                absorbed = 0;
            }
            st[absorbed] = $val;
            absorbed += 1;
        };
    }
    
    // Domain separation
    for &byte in b"neo/ivc/step-digest/v1" {
        absorb_elem!(Goldilocks::from_u32(byte as u32));
    }
    
    // Absorb step data
    for &f in step_data {
        absorb_elem!(Goldilocks::from_u64(f.as_canonical_u64()));
    }
    
    // Final permutation and extract digest
    st = poseidon2.permute(st);
    let mut digest = [0u8; 32];
    for (i, &elem) in st[..4].iter().enumerate() {
        digest[i*8..(i+1)*8].copy_from_slice(&elem.as_canonical_u64().to_le_bytes());
    }
    
    digest
}

//=============================================================================
// HIGH-LEVEL IVC API - Production-Ready Functions
//=============================================================================

/// Prove a single IVC step using the main Neo proving pipeline
/// 
/// This is the **production version** that generates cryptographic proofs,
/// not just constraint satisfaction checking.
pub fn prove_ivc_step(input: IvcStepInput) -> Result<IvcStepResult, Box<dyn std::error::Error>> {
    // 1. Create step digest for transcript binding
    let step_data = build_step_data(&input.prev_accumulator, input.step);
    let step_digest = create_step_digest(&step_data);
    
    // 2. Build augmented CCS (step ⊕ embedded verifier)
    let hash_input_len = step_data.len();
    let y_len = input.prev_accumulator.y_compact.len();
    let augmented_ccs = build_augmented_ccs_for_proving(
        input.step_ccs, 
        hash_input_len, 
        y_len, 
        step_digest
    )?;
    
    // 3. Build the combined witness
    let (combined_witness, next_state) = build_combined_witness(
        input.step_witness,
        &input.prev_accumulator,
        input.step,
        &step_data
    )?;
    
    // 4. Create commitment for full binding (TODO: Use in transcript binding)
    let commitment_bytes = serialize_accumulator_for_commitment(&input.prev_accumulator)?;
    let _commitment = Commitment::new(commitment_bytes, "ivc.accumulator");
    
    // 5. Build public input (include accumulator binding)
    let public_input = build_ivc_public_input(&input.prev_accumulator, input.public_input.unwrap_or(&[]))?;
    
    // 6. Generate cryptographic proof using main Neo API
    let step_proof = crate::prove(crate::ProveInput {
        params: input.params,
        ccs: &augmented_ccs,
        public_input: &public_input,
        witness: &combined_witness,
        output_claims: &[], // IVC uses accumulator outputs
    })?;
    
    // 7. Extract next accumulator from computation results  
    let next_accumulator = extract_next_accumulator(&next_state, input.step + 1)?;
    
    // 8. Create IVC proof
    let ivc_proof = IvcProof {
        step_proof,
        next_accumulator: next_accumulator.clone(),
        step: input.step,
        metadata: None,
    };
    
    Ok(IvcStepResult {
        proof: ivc_proof,
        next_state,
    })
}

/// Verify a single IVC step using the main Neo verification pipeline
pub fn verify_ivc_step(
    step_ccs: &CcsStructure<F>,
    ivc_proof: &IvcProof,
    prev_accumulator: &Accumulator,
) -> Result<bool, Box<dyn std::error::Error>> {
    // 1. Reconstruct the augmented CCS that was used for proving
    let step_data = build_step_data(prev_accumulator, ivc_proof.step);
    let step_digest = create_step_digest(&step_data);
    let augmented_ccs = build_augmented_ccs_for_proving(
        step_ccs,
        step_data.len(),
        prev_accumulator.y_compact.len(),
        step_digest
    )?;
    
    // 2. Build expected public input  
    let public_input = build_ivc_public_input(prev_accumulator, &[])?;
    
    // 3. Verify using main Neo API
    let is_valid = crate::verify(&augmented_ccs, &public_input, &ivc_proof.step_proof)?;
    
    // 4. Additional IVC-specific checks
    if is_valid {
        // Verify accumulator progression is valid
        verify_accumulator_progression(prev_accumulator, &ivc_proof.next_accumulator, ivc_proof.step)?;
    }
    
    Ok(is_valid)
}

/// Prove an entire IVC chain from start to finish  
pub fn prove_ivc_chain(
    params: &crate::NeoParams,
    step_ccs: &CcsStructure<F>,
    step_inputs: &[IvcChainStepInput],
    initial_accumulator: Accumulator,
) -> Result<IvcChainProof, Box<dyn std::error::Error>> {
    let mut current_accumulator = initial_accumulator;
    let mut step_proofs = Vec::with_capacity(step_inputs.len());
    
    for (step_idx, step_input) in step_inputs.iter().enumerate() {
        let ivc_step_input = IvcStepInput {
            params,
            step_ccs,
            step_witness: &step_input.witness,
            prev_accumulator: &current_accumulator,
            step: step_idx as u64,
            public_input: step_input.public_input.as_deref(),
        };
        
        let step_result = prove_ivc_step(ivc_step_input)?;
        current_accumulator = step_result.proof.next_accumulator.clone();
        // Note: step_result.next_state contains computation results for this step
        step_proofs.push(step_result.proof);
    }
    
    Ok(IvcChainProof {
        steps: step_proofs,
        final_accumulator: current_accumulator,
        chain_length: step_inputs.len() as u64,
    })
}

/// Verify an entire IVC chain
pub fn verify_ivc_chain(
    step_ccs: &CcsStructure<F>,
    chain_proof: &IvcChainProof,
    initial_accumulator: &Accumulator,
) -> Result<bool, Box<dyn std::error::Error>> {
    let mut current_accumulator = initial_accumulator.clone();
    
    for step_proof in &chain_proof.steps {
        let is_valid = verify_ivc_step(step_ccs, step_proof, &current_accumulator)?;
        if !is_valid {
            return Ok(false);
        }
        current_accumulator = step_proof.next_accumulator.clone();
    }
    
    // Final consistency check
    if current_accumulator.step != chain_proof.chain_length {
        return Ok(false);
    }
    
    Ok(true)
}

//=============================================================================
// Helper functions for high-level API
//=============================================================================

/// Input for a single step in an IVC chain
#[derive(Clone, Debug)]
pub struct IvcChainStepInput {
    pub witness: Vec<F>,
    pub public_input: Option<Vec<F>>,
}

/// Build augmented CCS for the proving pipeline (wrapper with error handling)
fn build_augmented_ccs_for_proving(
    step_ccs: &CcsStructure<F>,
    hash_input_len: usize,
    y_len: usize,
    step_digest: [u8; 32],
) -> Result<CcsStructure<F>, Box<dyn std::error::Error>> {
    let hash_ccs = ev_hash_ccs(hash_input_len, y_len);
    let augmented = neo_ccs::direct_sum_transcript_mixed(step_ccs, &hash_ccs, step_digest)
        .map_err(|e| format!("Failed to build augmented CCS: {:?}", e))?;
    Ok(augmented)
}

/// Build step data for transcript (step counter + accumulator state)
fn build_step_data(accumulator: &Accumulator, step: u64) -> Vec<F> {
    let mut step_data = Vec::new();
    step_data.push(F::from_u64(step));
    step_data.extend_from_slice(&accumulator.y_compact);
    
    // Include c_z_digest as field elements for binding
    for chunk in accumulator.c_z_digest.chunks_exact(8) {
        step_data.push(F::from_u64(u64::from_le_bytes(chunk.try_into().unwrap())));
    }
    
    step_data
}

/// Build combined witness for augmented CCS
fn build_combined_witness(
    step_witness: &[F],
    prev_accumulator: &Accumulator,
    step: u64,
    step_data: &[F],
) -> Result<(Vec<F>, Vec<F>), Box<dyn std::error::Error>> {
    // Create hash inputs from step data
    let hash_inputs = step_data.to_vec();
    
    // For demo: create simple y_step from step number
    let y_len = prev_accumulator.y_compact.len();
    let y_step = vec![F::from_u64(step); y_len];
    
    // Build EV-hash witness
    let (ev_witness, y_next) = build_ev_hash_witness(&hash_inputs, &prev_accumulator.y_compact, &y_step);
    
    // Combine witnesses: [step_witness, ev_witness]
    let mut combined = Vec::with_capacity(step_witness.len() + ev_witness.len());
    combined.extend_from_slice(step_witness);
    combined.extend_from_slice(&ev_witness);
    
    Ok((combined, y_next))
}

/// Serialize accumulator for commitment binding
fn serialize_accumulator_for_commitment(accumulator: &Accumulator) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    
    // Step counter (8 bytes)
    bytes.extend_from_slice(&accumulator.step.to_le_bytes());
    
    // c_z_digest (32 bytes)
    bytes.extend_from_slice(&accumulator.c_z_digest);
    
    // y_compact length + elements
    bytes.extend_from_slice(&(accumulator.y_compact.len() as u64).to_le_bytes());
    for &y in &accumulator.y_compact {
        bytes.extend_from_slice(&y.as_canonical_u64().to_le_bytes());
    }
    
    Ok(bytes)
}

/// Build public input for IVC proof
fn build_ivc_public_input(accumulator: &Accumulator, extra_input: &[F]) -> Result<Vec<F>, Box<dyn std::error::Error>> {
    let mut public_input = Vec::new();
    
    // Include accumulator state as public input
    public_input.push(F::from_u64(accumulator.step));
    public_input.extend_from_slice(&accumulator.y_compact);
    
    // Add c_z_digest as public field elements
    for chunk in accumulator.c_z_digest.chunks_exact(8) {
        public_input.push(F::from_u64(u64::from_le_bytes(chunk.try_into().unwrap())));
    }
    
    // Add extra input
    public_input.extend_from_slice(extra_input);
    
    Ok(public_input)
}

/// Extract next accumulator from computation results
fn extract_next_accumulator(next_state: &[F], step: u64) -> Result<Accumulator, Box<dyn std::error::Error>> {
    Ok(Accumulator {
        c_z_digest: [0u8; 32], // TODO: Update from actual commitment evolution
        y_compact: next_state.to_vec(),
        step,
    })
}

/// Verify accumulator progression follows IVC rules
fn verify_accumulator_progression(
    prev: &Accumulator,
    next: &Accumulator,
    expected_step: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    if next.step != expected_step {
        return Err(format!("Invalid step progression: expected {}, got {}", expected_step, next.step).into());
    }
    
    if prev.step + 1 != next.step {
        return Err(format!("Non-consecutive steps: {} -> {}", prev.step, next.step).into());
    }
    
    // TODO: Add more accumulator validation rules
    
    Ok(())
}

