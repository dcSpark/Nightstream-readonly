//! Output Sumcheck for binding program outputs to proofs.
//!
//! This module provides a sumcheck-based argument that proves the final memory
//! state matches the claimed I/O values. Similar to Jolt's output_check.rs.
//!
//! ## The Problem
//!
//! When proving RISC-V program execution, we need to ensure that:
//! 1. The execution trace is valid (handled by Twist + CCS)
//! 2. The claimed outputs match the actual final memory state (this module)
//!
//! Without this, a malicious prover could generate a valid execution proof
//! but lie about the program's outputs.
//!
//! ## The Solution: Output Sumcheck
//!
//! We prove a zero-check over the claimed I/O addresses:
//!
//! ```text
//! Σ_k eq(r_addr, k) · io_mask(k) · (Val_final(k) − Val_io(k)) = 0
//! ```
//!
//! Where:
//! - `r_addr`: Random address challenge derived from transcript (AFTER absorbing claims)
//! - `io_mask(k)`: 1 if address k is a claimed I/O address, 0 otherwise
//! - `Val_final(k)`: The actual final memory value at address k (committed by Twist)
//! - `Val_io(k)`: The publicly claimed I/O value at address k (provided by verifier)
//!
//! ## Security Properties
//!
//! 1. **Verifier enforces claim == 0**: The verifier pins the initial claim to 0,
//!    not trusting the prover's `claimed_sum`.
//! 2. **Transcript binding**: `r_addr` is sampled AFTER absorbing the claimed outputs,
//!    preventing the prover from choosing outputs after seeing challenges.
//! 3. **Claimed-address mask**: Only explicitly claimed addresses are constrained,
//!    avoiding silent zero-enforcement on unclaimed addresses.
//!
//! If the prover lies about outputs, `Val_final(k) ≠ Val_io(k)` for some output
//! addresses, and the sumcheck will fail (with overwhelming probability).

use crate::mle::build_chi_table;
use neo_math::{from_complex, KExtensions, K};
use neo_reductions::sumcheck::RoundOracle;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{Field, PrimeCharacteristicRing};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ============================================================================
// Transcript helpers for K (extension field)
// ============================================================================

/// Sample a K challenge from the transcript with proper domain separation.
fn sample_k_challenge(tr: &mut Poseidon2Transcript) -> K {
    let c = tr.challenge_field(b"output_check/chal/re");
    let d = tr.challenge_field(b"output_check/chal/im");
    from_complex(c, d)
}

/// Absorb a K value into the transcript.
fn absorb_k(tr: &mut Poseidon2Transcript, label: &'static [u8], k: K) {
    tr.append_fields(label, &k.as_coeffs());
}

/// Maximum supported num_bits to prevent overflow.
const MAX_NUM_BITS: usize = 30;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during output sumcheck operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputCheckError {
    /// A claimed address is outside the valid domain [0, 2^num_bits).
    AddressOutOfDomain { addr: u64, max_addr: u64 },
    /// Duplicate address in claims.
    DuplicateAddress { addr: u64 },
    /// The claimed sum is not zero (outputs don't match).
    NonZeroClaim,
    /// Verification failed at a specific round.
    RoundCheckFailed { round: usize, message: String },
    /// Final claim mismatch.
    FinalClaimMismatch { got: K, expected: K },
    /// Wrong number of rounds or challenges.
    DimensionMismatch { expected: usize, got: usize },
    /// num_bits is too large (would overflow).
    NumBitsTooLarge { num_bits: usize, max: usize },
    /// Round polynomial has wrong degree.
    WrongDegree { round: usize, expected: usize, got: usize },
}

impl std::fmt::Display for OutputCheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AddressOutOfDomain { addr, max_addr } => {
                write!(f, "Address {} out of domain [0, {})", addr, max_addr)
            }
            Self::DuplicateAddress { addr } => {
                write!(f, "Duplicate address {} in claims", addr)
            }
            Self::NonZeroClaim => {
                write!(f, "Output sumcheck: claimed_sum must be 0 for valid outputs")
            }
            Self::RoundCheckFailed { round, message } => {
                write!(f, "Round {} check failed: {}", round, message)
            }
            Self::FinalClaimMismatch { got, expected } => {
                write!(f, "Final claim mismatch: got {:?}, expected {:?}", got, expected)
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::NumBitsTooLarge { num_bits, max } => {
                write!(f, "num_bits {} exceeds maximum {}", num_bits, max)
            }
            Self::WrongDegree { round, expected, got } => {
                write!(f, "Round {} polynomial has degree {}, expected {}", round, got, expected)
            }
        }
    }
}

impl std::error::Error for OutputCheckError {}

// ============================================================================
// Program I/O Definition
// ============================================================================

/// Defines the I/O claims for a program.
///
/// This specifies which memory addresses/registers have claimed values,
/// allowing the verifier to know what values should be publicly checkable.
///
/// ## Important
///
/// The mask is defined by the **set of claimed addresses**, not a range.
/// Only addresses with explicit claims are constrained by the sumcheck.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProgramIO<F> {
    /// Claimed values at specified addresses, stored as a map for uniqueness.
    /// Key: address, Value: claimed value
    claims: BTreeMap<u64, F>,
}

impl<F: Field> Default for ProgramIO<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> ProgramIO<F> {
    /// Create a new empty ProgramIO.
    pub fn new() -> Self {
        Self {
            claims: BTreeMap::new(),
        }
    }

    /// Add a claimed value at an address.
    ///
    /// If the address already has a claim, it will be overwritten.
    /// Use `try_with_claim` if you want to detect duplicates.
    pub fn with_claim(mut self, addr: u64, value: F) -> Self {
        self.claims.insert(addr, value);
        self
    }

    /// Add an output claim (alias for `with_claim`).
    pub fn with_output(self, addr: u64, value: F) -> Self {
        self.with_claim(addr, value)
    }

    /// Add an input claim (alias for `with_claim`).
    pub fn with_input(self, addr: u64, value: F) -> Self {
        self.with_claim(addr, value)
    }

    /// Try to add a claim, returning error if address is duplicate.
    pub fn try_with_claim(mut self, addr: u64, value: F) -> Result<Self, OutputCheckError> {
        if self.claims.contains_key(&addr) {
            return Err(OutputCheckError::DuplicateAddress { addr });
        }
        self.claims.insert(addr, value);
        Ok(self)
    }

    /// Get all claimed addresses.
    pub fn claimed_addresses(&self) -> impl Iterator<Item = u64> + '_ {
        self.claims.keys().copied()
    }

    /// Get the claimed value at an address, if any.
    pub fn get_claim(&self, addr: u64) -> Option<F> {
        self.claims.get(&addr).copied()
    }

    /// Number of claims.
    pub fn num_claims(&self) -> usize {
        self.claims.len()
    }

    /// Check if there are any claims.
    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    /// Get all claims as (address, value) pairs.
    pub fn claims(&self) -> impl Iterator<Item = (u64, F)> + '_ {
        self.claims.iter().map(|(&a, &v)| (a, v))
    }

    /// Validate that all addresses are within the domain [0, 2^num_bits).
    pub fn validate(&self, num_bits: usize) -> Result<(), OutputCheckError> {
        if num_bits > MAX_NUM_BITS {
            return Err(OutputCheckError::NumBitsTooLarge {
                num_bits,
                max: MAX_NUM_BITS,
            });
        }
        let max_addr = 1u64 << num_bits;
        for &addr in self.claims.keys() {
            if addr >= max_addr {
                return Err(OutputCheckError::AddressOutOfDomain { addr, max_addr });
            }
        }
        Ok(())
    }

    /// Absorb claims into a transcript for Fiat-Shamir binding.
    ///
    /// This must be called BEFORE sampling `r_addr` to ensure the prover
    /// cannot choose outputs after seeing the challenge.
    pub fn absorb_into_transcript(&self, tr: &mut Poseidon2Transcript)
    where
        F: Into<neo_math::F> + Copy,
    {
        // Absorb number of claims as bytes (safer than field conversion)
        tr.append_message(b"output_check/num_claims", &(self.claims.len() as u64).to_le_bytes());

        // Absorb each (address, value) pair in sorted order (BTreeMap guarantees this)
        for (&addr, &value) in &self.claims {
            // Absorb address as bytes (not field element)
            tr.append_message(b"output_check/addr", &addr.to_le_bytes());
            // Absorb value as field element
            tr.append_fields(b"output_check/value", &[value.into()]);
        }
    }
}

// ============================================================================
// I/O Mask Polynomial (Claimed Addresses Only)
// ============================================================================

/// Evaluates the I/O mask polynomial at a given point.
///
/// The I/O mask is 1 for explicitly claimed addresses and 0 otherwise.
/// This uses sparse evaluation: Σ_{addr in claimed} χ_r(addr).
pub struct IOMaskPolynomial {
    /// The set of claimed addresses
    claimed_addresses: Vec<u64>,
    /// Number of address bits
    num_bits: usize,
}

impl IOMaskPolynomial {
    /// Create a new mask from claimed addresses.
    pub fn from_claims<F: Field>(program_io: &ProgramIO<F>, num_bits: usize) -> Self {
        Self {
            claimed_addresses: program_io.claimed_addresses().collect(),
            num_bits,
        }
    }

    /// Evaluate the I/O mask at a point r in K^{num_bits}.
    ///
    /// Returns: Σ_{addr in claimed} eq(r, addr)
    pub fn evaluate(&self, r: &[K]) -> K {
        assert_eq!(r.len(), self.num_bits, "Point dimension mismatch");

        let mut sum = K::ZERO;
        for &addr in &self.claimed_addresses {
            sum += chi_at_point(r, addr, self.num_bits);
        }
        sum
    }

    /// Build the mask table over the entire address space.
    ///
    /// Returns a vector of length 2^num_bits where entry k is 1 if k is a
    /// claimed address and 0 otherwise.
    pub fn build_table(&self) -> Vec<K> {
        let n = 1usize << self.num_bits;
        let mut table = vec![K::ZERO; n];
        for &addr in &self.claimed_addresses {
            if (addr as usize) < n {
                table[addr as usize] = K::ONE;
            }
        }
        table
    }
}

/// Compute χ_r(k) for a specific integer k.
///
/// χ_r(k) = Π_{i=0}^{ℓ-1} (r[i] if k_i else (1-r[i]))
fn chi_at_point(r: &[K], k: u64, num_bits: usize) -> K {
    let mut acc = K::ONE;
    for i in 0..num_bits {
        let bit = ((k >> i) & 1) == 1;
        if bit {
            acc *= r[i];
        } else {
            acc *= K::ONE - r[i];
        }
    }
    acc
}

// ============================================================================
// Claimed I/O Polynomial
// ============================================================================

/// Represents the publicly claimed I/O values as a sparse polynomial.
///
/// Val_io(k) = claimed value at address k if k is a claimed address, 0 otherwise.
pub struct ClaimedIOPolynomial<F> {
    /// Claims stored as (address, value) pairs
    claims: Vec<(u64, F)>,
    /// Number of address bits
    num_bits: usize,
}

impl<F: Field + Into<K>> ClaimedIOPolynomial<F> {
    pub fn new(program_io: &ProgramIO<F>, num_bits: usize) -> Self {
        Self {
            claims: program_io.claims().collect(),
            num_bits,
        }
    }

    /// Evaluate the claimed I/O polynomial at point r.
    ///
    /// Val_io(r) = Σ_k claimed[k] · χ_r(k)
    pub fn evaluate(&self, r: &[K]) -> K {
        assert_eq!(r.len(), self.num_bits, "Point dimension mismatch");

        let mut sum = K::ZERO;
        for (addr, val) in &self.claims {
            let chi_k = chi_at_point(r, *addr, self.num_bits);
            let val_k: K = (*val).into();
            sum += val_k * chi_k;
        }
        sum
    }

    /// Build the full table representation.
    pub fn build_table(&self) -> Vec<K> {
        let n = 1usize << self.num_bits;
        let mut table = vec![K::ZERO; n];
        for (addr, val) in &self.claims {
            if (*addr as usize) < n {
                table[*addr as usize] = (*val).into();
            }
        }
        table
    }
}

// ============================================================================
// Output Sumcheck Parameters
// ============================================================================

/// Parameters for the Output Sumcheck.
#[derive(Clone, Debug)]
pub struct OutputSumcheckParams<F> {
    /// Size of address space (K = 2^num_bits)
    pub k: usize,
    /// Number of address bits
    pub num_bits: usize,
    /// Random address challenge (derived from transcript)
    pub r_addr: Vec<K>,
    /// Program I/O specification
    pub program_io: ProgramIO<F>,
}

impl<F> OutputSumcheckParams<F> {
    /// Number of sumcheck rounds (= num_bits).
    pub fn num_rounds(&self) -> usize {
        self.num_bits
    }

    /// Degree bound for the sumcheck (product of 3 MLEs).
    pub fn degree_bound(&self) -> usize {
        3 // eq * io_mask * (val_final - val_io)
    }
}

impl<F: Field + Into<neo_math::F> + Copy> OutputSumcheckParams<F> {
    /// Sample parameters from a transcript.
    ///
    /// This is the recommended way to create parameters as it ensures proper
    /// Fiat-Shamir binding: the claimed outputs are absorbed BEFORE sampling r_addr.
    pub fn sample_from_transcript(
        tr: &mut Poseidon2Transcript,
        num_bits: usize,
        program_io: ProgramIO<F>,
    ) -> Result<Self, OutputCheckError> {
        // Validate claims are in domain (also checks num_bits is reasonable)
        program_io.validate(num_bits)?;

        // Absorb claims into transcript BEFORE sampling challenges
        program_io.absorb_into_transcript(tr);

        // Sample r_addr from transcript with per-index domain separation
        let mut r_addr = Vec::with_capacity(num_bits);
        for i in 0..num_bits {
            tr.append_message(b"output_check/r_addr/idx", &(i as u64).to_le_bytes());
            r_addr.push(sample_k_challenge(tr));
        }

        Ok(Self {
            k: 1 << num_bits,
            num_bits,
            r_addr,
            program_io,
        })
    }

    /// Create parameters with explicit r_addr (for testing only).
    ///
    /// **WARNING**: In production, use `sample_from_transcript` to ensure
    /// proper Fiat-Shamir binding.
    pub fn new_for_testing(
        num_bits: usize,
        r_addr: Vec<K>,
        program_io: ProgramIO<F>,
    ) -> Result<Self, OutputCheckError> {
        program_io.validate(num_bits)?;
        Ok(Self {
            k: 1 << num_bits,
            num_bits,
            r_addr,
            program_io,
        })
    }
}

// ============================================================================
// Output Sumcheck Prover
// ============================================================================

/// Prover for the Output Sumcheck.
///
/// Proves: Σ_k eq(r_addr, k) · io_mask(k) · (Val_final(k) − Val_io(k)) = 0
pub struct OutputSumcheckProver<F> {
    /// eq(r_addr, ·) table, folded during sumcheck
    eq_table: Vec<K>,
    /// io_mask(·) table, folded during sumcheck
    io_mask_table: Vec<K>,
    /// Val_final(·) table, folded during sumcheck
    val_final_table: Vec<K>,
    /// Val_io(·) table, folded during sumcheck
    val_io_table: Vec<K>,
    /// Remaining rounds
    rounds_remaining: usize,
    /// Challenges collected so far
    challenges: Vec<K>,
    /// Phantom for F
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Field + Into<K> + Clone> OutputSumcheckProver<F> {
    /// Initialize the prover with the final memory state.
    ///
    /// # Arguments
    /// * `params` - Sumcheck parameters including r_addr and program_io
    /// * `final_memory_state` - The actual final memory values (from Twist)
    ///
    /// # Errors
    /// Returns error if final_memory_state length doesn't match 2^num_bits.
    pub fn new(
        params: OutputSumcheckParams<F>,
        final_memory_state: &[F],
    ) -> Result<Self, OutputCheckError> {
        let num_bits = params.num_bits;
        let k = params.k;

        if final_memory_state.len() != k {
            return Err(OutputCheckError::DimensionMismatch {
                expected: k,
                got: final_memory_state.len(),
            });
        }

        // Build eq(r_addr, ·) table
        let eq_table = build_chi_table(&params.r_addr);

        // Build io_mask table from claimed addresses
        let io_mask = IOMaskPolynomial::from_claims(&params.program_io, num_bits);
        let io_mask_table = io_mask.build_table();

        // Build Val_final table
        let val_final_table: Vec<K> = final_memory_state.iter().map(|v| (*v).into()).collect();

        // Build Val_io table
        let claimed_io = ClaimedIOPolynomial::new(&params.program_io, num_bits);
        let val_io_table = claimed_io.build_table();

        Ok(Self {
            eq_table,
            io_mask_table,
            val_final_table,
            val_io_table,
            rounds_remaining: num_bits,
            challenges: Vec::with_capacity(num_bits),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Compute the sum over the hypercube (should be 0 if outputs match).
    pub fn compute_claim(&self) -> K {
        let n = self.eq_table.len();
        let mut sum = K::ZERO;
        for i in 0..n {
            let term = self.eq_table[i]
                * self.io_mask_table[i]
                * (self.val_final_table[i] - self.val_io_table[i]);
            sum += term;
        }
        sum
    }

    /// Get the challenges collected so far.
    pub fn challenges(&self) -> &[K] {
        &self.challenges
    }
}

impl<F: Field + Into<K> + Clone> RoundOracle for OutputSumcheckProver<F> {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let n = self.eq_table.len();

        // Handle the case where we're done (len == 1)
        if n == 1 {
            let term = self.eq_table[0]
                * self.io_mask_table[0]
                * (self.val_final_table[0] - self.val_io_table[0]);
            return vec![term; points.len()];
        }

        let half = n / 2;

        points
            .iter()
            .map(|&x| {
                let mut sum = K::ZERO;
                for i in 0..half {
                    // Interpolate each factor at x
                    let eq_0 = self.eq_table[i];
                    let eq_1 = self.eq_table[half + i];
                    let eq_x = eq_0 + (eq_1 - eq_0) * x;

                    let mask_0 = self.io_mask_table[i];
                    let mask_1 = self.io_mask_table[half + i];
                    let mask_x = mask_0 + (mask_1 - mask_0) * x;

                    let vf_0 = self.val_final_table[i];
                    let vf_1 = self.val_final_table[half + i];
                    let vf_x = vf_0 + (vf_1 - vf_0) * x;

                    let vio_0 = self.val_io_table[i];
                    let vio_1 = self.val_io_table[half + i];
                    let vio_x = vio_0 + (vio_1 - vio_0) * x;

                    sum += eq_x * mask_x * (vf_x - vio_x);
                }
                sum
            })
            .collect()
    }

    fn num_rounds(&self) -> usize {
        self.rounds_remaining
    }

    fn degree_bound(&self) -> usize {
        3 // Product of 3 linear terms
    }

    fn fold(&mut self, r: K) {
        // Guard against being called after completion
        if self.eq_table.len() <= 1 {
            return;
        }

        let n = self.eq_table.len();
        let half = n / 2;

        // Fold each table
        for i in 0..half {
            self.eq_table[i] = self.eq_table[i] + (self.eq_table[half + i] - self.eq_table[i]) * r;
            self.io_mask_table[i] =
                self.io_mask_table[i] + (self.io_mask_table[half + i] - self.io_mask_table[i]) * r;
            self.val_final_table[i] = self.val_final_table[i]
                + (self.val_final_table[half + i] - self.val_final_table[i]) * r;
            self.val_io_table[i] =
                self.val_io_table[i] + (self.val_io_table[half + i] - self.val_io_table[i]) * r;
        }

        self.eq_table.truncate(half);
        self.io_mask_table.truncate(half);
        self.val_final_table.truncate(half);
        self.val_io_table.truncate(half);

        self.challenges.push(r);
        self.rounds_remaining = self.rounds_remaining.saturating_sub(1);
    }
}

// ============================================================================
// Output Sumcheck Verifier
// ============================================================================

/// Verifier for the Output Sumcheck.
///
/// **Critical**: The verifier enforces that the initial claim is 0.
/// This ensures that outputs must match; a non-zero mismatch cannot be hidden.
#[derive(Clone, Debug)]
pub struct OutputSumcheckVerifier<F> {
    params: OutputSumcheckParams<F>,
}

impl<F: Field + Into<K> + Clone> OutputSumcheckVerifier<F> {
    pub fn new(params: OutputSumcheckParams<F>) -> Self {
        Self { params }
    }

    /// Compute the expected final claim at the challenge point.
    ///
    /// # Arguments
    /// * `r_prime` - The challenge point from the sumcheck (length = num_bits)
    /// * `val_final_at_r_prime` - Val_final(r') provided by the prover's opening
    ///
    /// # Returns
    /// The expected claim: eq(r_addr, r') · io_mask(r') · (val_final(r') - val_io(r'))
    pub fn expected_claim(&self, r_prime: &[K], val_final_at_r_prime: K) -> K {
        let num_bits = self.params.num_bits;
        assert_eq!(r_prime.len(), num_bits, "Challenge point dimension mismatch");

        // Compute eq(r_addr, r')
        let eq_eval = eq_points(&self.params.r_addr, r_prime);

        // Compute io_mask(r') using claimed addresses
        let io_mask = IOMaskPolynomial::from_claims(&self.params.program_io, num_bits);
        let io_mask_eval = io_mask.evaluate(r_prime);

        // Compute val_io(r')
        let claimed_io = ClaimedIOPolynomial::new(&self.params.program_io, num_bits);
        let val_io_eval = claimed_io.evaluate(r_prime);

        // Final claim
        eq_eval * io_mask_eval * (val_final_at_r_prime - val_io_eval)
    }

    /// Verify a complete output sumcheck proof.
    ///
    /// **Critical**: This function enforces that the initial claim is 0.
    /// It does NOT trust `proof.claimed_sum`.
    ///
    /// # Arguments
    /// * `proof` - The sumcheck proof (round polynomials)
    /// * `val_final_at_r_prime` - Opening of Val_final at the challenge point
    /// * `transcript_challenges` - Challenges derived during verification
    ///
    /// # Returns
    /// Ok(()) if verification passes, Err with description otherwise.
    pub fn verify(
        &self,
        proof: &OutputSumcheckProof,
        val_final_at_r_prime: K,
        transcript_challenges: &[K],
    ) -> Result<(), OutputCheckError> {
        let num_rounds = self.params.num_rounds();

        if proof.round_polys.len() != num_rounds {
            return Err(OutputCheckError::DimensionMismatch {
                expected: num_rounds,
                got: proof.round_polys.len(),
            });
        }

        if transcript_challenges.len() != num_rounds {
            return Err(OutputCheckError::DimensionMismatch {
                expected: num_rounds,
                got: transcript_challenges.len(),
            });
        }

        // **CRITICAL**: Enforce that the initial claim is 0.
        // We do NOT trust proof.claimed_sum.
        // For valid outputs, the sum must be zero.
        let mut current_claim = K::ZERO;
        let expected_degree = self.params.degree_bound() + 1; // coefficients = degree + 1

        // Verify each round
        for (round, (coeffs, &r)) in proof
            .round_polys
            .iter()
            .zip(transcript_challenges.iter())
            .enumerate()
        {
            // **CRITICAL**: Enforce degree bound
            if coeffs.len() != expected_degree {
                return Err(OutputCheckError::WrongDegree {
                    round,
                    expected: expected_degree,
                    got: coeffs.len(),
                });
            }

            // Check sum property: p(0) + p(1) = current_claim
            let p_0 = eval_poly(coeffs, K::ZERO);
            let p_1 = eval_poly(coeffs, K::ONE);
            if p_0 + p_1 != current_claim {
                return Err(OutputCheckError::RoundCheckFailed {
                    round,
                    message: format!(
                        "p(0) + p(1) = {:?}, expected {:?}",
                        p_0 + p_1,
                        current_claim
                    ),
                });
            }

            // Update claim for next round
            current_claim = eval_poly(coeffs, r);
        }

        // Check final claim matches expected
        let expected = self.expected_claim(transcript_challenges, val_final_at_r_prime);
        if current_claim != expected {
            return Err(OutputCheckError::FinalClaimMismatch {
                got: current_claim,
                expected,
            });
        }

        Ok(())
    }

    /// Verify using a transcript to derive challenges.
    ///
    /// This is the recommended verification method as it ensures proper
    /// Fiat-Shamir derivation of challenges.
    pub fn verify_with_transcript(
        &self,
        tr: &mut Poseidon2Transcript,
        proof: &OutputSumcheckProof,
        val_final_at_r_prime: K,
    ) -> Result<(), OutputCheckError> {
        let num_rounds = self.params.num_rounds();

        if proof.round_polys.len() != num_rounds {
            return Err(OutputCheckError::DimensionMismatch {
                expected: num_rounds,
                got: proof.round_polys.len(),
            });
        }

        // **CRITICAL**: Enforce that the initial claim is 0.
        let mut current_claim = K::ZERO;
        let mut challenges = Vec::with_capacity(num_rounds);
        let expected_degree = self.params.degree_bound() + 1;

        // Verify each round, deriving challenges from transcript
        for (round, coeffs) in proof.round_polys.iter().enumerate() {
            // **CRITICAL**: Enforce degree bound
            if coeffs.len() != expected_degree {
                return Err(OutputCheckError::WrongDegree {
                    round,
                    expected: expected_degree,
                    got: coeffs.len(),
                });
            }

            // Check sum property: p(0) + p(1) = current_claim
            let p_0 = eval_poly(coeffs, K::ZERO);
            let p_1 = eval_poly(coeffs, K::ONE);
            if p_0 + p_1 != current_claim {
                return Err(OutputCheckError::RoundCheckFailed {
                    round,
                    message: format!(
                        "p(0) + p(1) = {:?}, expected {:?}",
                        p_0 + p_1,
                        current_claim
                    ),
                });
            }

            // Absorb round polynomial into transcript
            for &c in coeffs {
                absorb_k(tr, b"output_check/round_coeff", c);
            }

            // Sample challenge from transcript
            let r = sample_k_challenge(tr);
            challenges.push(r);

            // Update claim for next round
            current_claim = eval_poly(coeffs, r);
        }

        // Check final claim matches expected
        let expected = self.expected_claim(&challenges, val_final_at_r_prime);
        if current_claim != expected {
            return Err(OutputCheckError::FinalClaimMismatch {
                got: current_claim,
                expected,
            });
        }

        Ok(())
    }
}

/// Helper to evaluate a polynomial given as coefficients.
fn eval_poly(coeffs: &[K], x: K) -> K {
    if coeffs.is_empty() {
        return K::ZERO;
    }
    let mut result = coeffs[coeffs.len() - 1];
    for &c in coeffs.iter().rev().skip(1) {
        result = result * x + c;
    }
    result
}

/// Compute eq(a, b) = Π_i (a_i · b_i + (1-a_i)(1-b_i))
fn eq_points(a: &[K], b: &[K]) -> K {
    assert_eq!(a.len(), b.len());
    let mut acc = K::ONE;
    for (ai, bi) in a.iter().zip(b.iter()) {
        acc *= *ai * *bi + (K::ONE - *ai) * (K::ONE - *bi);
    }
    acc
}

// ============================================================================
// Output Sumcheck Proof
// ============================================================================

/// Proof for the Output Sumcheck (standalone, requires external val_final opening).
///
/// The proof contains only the round polynomials. The verifier:
/// - Derives challenges from transcript (does not trust any r_prime in proof)
/// - Enforces initial claim = 0 (does not trust claimed_sum)
///
/// **Note**: This is the low-level proof. For full cryptographic binding, use
/// `OutputBindingProof` which includes authenticated Twist opening.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutputSumcheckProof {
    /// Round polynomials (coefficients for each round).
    /// Each polynomial must have exactly degree_bound + 1 = 4 coefficients.
    pub round_polys: Vec<Vec<K>>,
}


// ============================================================================
// Full Output Binding Proof (with authenticated Twist opening)
// ============================================================================

/// Complete output binding proof with authenticated Val_final opening.
///
/// This proof combines:
/// 1. The output sumcheck proving Σ eq·mask·(Val_final - Val_io) = 0
/// 2. A Twist-based sumcheck proving inc_total(r') = Σ_t has_write·inc·eq(wa, r')
/// 3. The claimed inc_total value (verifier computes Val_final = Val_init + inc_total)
///
/// **Security**: The verifier derives `Val_final(r')` internally from authenticated
/// Twist columns, rather than trusting an externally-provided value.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutputBindingProof {
    /// The output sumcheck proof (round polynomials only).
    pub output_sc: OutputSumcheckProof,
    /// The claimed total increment at r' (= Σ_t Inc(r', t)).
    pub inc_total_claim: K,
    /// Twist inc_total sumcheck round polynomials.
    /// Proves: inc_total_claim = Σ_t has_write(t) · inc(t) · eq(wa_bits(t), r')
    pub inc_total_rounds: Vec<Vec<K>>,
}


// ============================================================================
// Proof Generation Helpers
// ============================================================================

/// Generate an output sumcheck proof using a transcript.
///
/// This is the recommended way to generate proofs as it ensures proper
/// Fiat-Shamir binding.
pub fn generate_output_proof_with_transcript<F: Field + Into<K> + Into<neo_math::F> + Clone + Copy>(
    tr: &mut Poseidon2Transcript,
    num_bits: usize,
    program_io: ProgramIO<F>,
    final_memory_state: &[F],
) -> Result<OutputSumcheckProof, OutputCheckError> {
    let params = OutputSumcheckParams::sample_from_transcript(tr, num_bits, program_io)?;
    let mut prover = OutputSumcheckProver::new(params, final_memory_state)?;

    let num_rounds = prover.num_rounds();
    let degree_bound = prover.degree_bound();

    let mut round_polys = Vec::with_capacity(num_rounds);
    let eval_points: Vec<K> = (0..=degree_bound).map(|i| K::from_u64(i as u64)).collect();

    for _ in 0..num_rounds {
        // Get evaluations at multiple points
        let evals = prover.evals_at(&eval_points);

        // Interpolate to get coefficients
        let coeffs = interpolate(&eval_points, &evals);

        // Absorb round polynomial into transcript
        for &c in &coeffs {
            absorb_k(tr, b"output_check/round_coeff", c);
        }

        round_polys.push(coeffs);

        // Sample challenge from transcript and fold
        let r = sample_k_challenge(tr);
        prover.fold(r);
    }

    Ok(OutputSumcheckProof { round_polys })
}

/// Generate an output sumcheck proof with explicit challenge sampler (for testing).
pub fn generate_output_proof_for_testing<F: Field + Into<K> + Clone>(
    params: OutputSumcheckParams<F>,
    final_memory_state: &[F],
    mut sample_challenge: impl FnMut(&[K]) -> K,
) -> Result<OutputSumcheckProof, OutputCheckError> {
    let mut prover = OutputSumcheckProver::new(params, final_memory_state)?;

    let num_rounds = prover.num_rounds();
    let degree_bound = prover.degree_bound();

    let mut round_polys = Vec::with_capacity(num_rounds);
    let eval_points: Vec<K> = (0..=degree_bound).map(|i| K::from_u64(i as u64)).collect();

    for _ in 0..num_rounds {
        let evals = prover.evals_at(&eval_points);
        let coeffs = interpolate(&eval_points, &evals);
        round_polys.push(coeffs.clone());

        let r = sample_challenge(&coeffs);
        prover.fold(r);
    }

    Ok(OutputSumcheckProof { round_polys })
}

// ============================================================================
// Full Output Binding Prover/Verifier
// ============================================================================

/// Witness data needed for output binding proof generation.
///
/// This contains the Twist columns needed to prove `inc_total(r')`.
#[derive(Clone, Debug)]
pub struct OutputBindingWitness {
    /// Write address bits: `wa_bits[bit][time]`
    pub wa_bits: Vec<Vec<K>>,
    /// Has-write flag per time step
    pub has_write: Vec<K>,
    /// Increment value at write address per time step
    pub inc_at_write_addr: Vec<K>,
}

/// Generate a complete output binding proof.
///
/// This generates:
/// 1. The output sumcheck proof
/// 2. The Twist inc_total sumcheck proof
///
/// The verifier can then derive `Val_final(r')` internally without trusting
/// any prover-provided value.
pub fn generate_output_binding_proof<F: Field + Into<K> + Into<neo_math::F> + Clone + Copy>(
    tr: &mut Poseidon2Transcript,
    num_bits: usize,
    program_io: ProgramIO<F>,
    final_memory_state: &[F],
    twist_witness: &OutputBindingWitness,
) -> Result<OutputBindingProof, OutputCheckError> {
    use crate::twist_oracle::TwistTotalIncOracleSparse;
    use neo_reductions::sumcheck::RoundOracle;

    // Step 1: Generate output sumcheck parameters (absorbs claims, samples r_addr)
    let params = OutputSumcheckParams::sample_from_transcript(tr, num_bits, program_io)?;

    // Step 2: Generate output sumcheck proof
    let mut prover = OutputSumcheckProver::new(params, final_memory_state)?;
    let output_num_rounds = prover.num_rounds();
    let output_degree = prover.degree_bound();

    let mut output_round_polys = Vec::with_capacity(output_num_rounds);
    let output_eval_points: Vec<K> = (0..=output_degree).map(|i| K::from_u64(i as u64)).collect();

    for _ in 0..output_num_rounds {
        let evals = prover.evals_at(&output_eval_points);
        let coeffs = interpolate(&output_eval_points, &evals);

        for &c in &coeffs {
            absorb_k(tr, b"output_check/round_coeff", c);
        }
        output_round_polys.push(coeffs);

        let r = sample_k_challenge(tr);
        prover.fold(r);
    }

    let output_sc = OutputSumcheckProof {
        round_polys: output_round_polys,
    };

    // Get r' (the challenge point from output sumcheck)
    let r_prime = prover.challenges().to_vec();

    // Step 3: Generate Twist inc_total sumcheck
    // This proves: inc_total_claim = Σ_t has_write(t) · inc(t) · eq(wa_bits(t), r')
    tr.append_message(b"output_binding/inc_total_start", &[]);

    let (mut inc_oracle, inc_total_claim) = TwistTotalIncOracleSparse::new(
        &twist_witness.wa_bits,
        twist_witness.has_write.clone(),
        twist_witness.inc_at_write_addr.clone(),
        &r_prime,
    );

    // Absorb the claimed sum into transcript
    absorb_k(tr, b"output_binding/inc_total_claim", inc_total_claim);

    let inc_num_rounds = inc_oracle.num_rounds();
    let inc_degree = inc_oracle.degree_bound();

    let mut inc_total_rounds = Vec::with_capacity(inc_num_rounds);
    let inc_eval_points: Vec<K> = (0..=inc_degree).map(|i| K::from_u64(i as u64)).collect();

    for _ in 0..inc_num_rounds {
        let evals = inc_oracle.evals_at(&inc_eval_points);
        let coeffs = interpolate(&inc_eval_points, &evals);

        for &c in &coeffs {
            absorb_k(tr, b"output_binding/inc_round_coeff", c);
        }
        inc_total_rounds.push(coeffs);

        let r = sample_k_challenge(tr);
        inc_oracle.fold(r);
    }

    Ok(OutputBindingProof {
        output_sc,
        inc_total_claim,
        inc_total_rounds,
    })
}

/// Verify a complete output binding proof.
///
/// This verifies:
/// 1. The Twist inc_total sumcheck (derives inc_total at the terminal check)
/// 2. Computes Val_final(r') = Val_init(r') + inc_total(r')
/// 3. The output sumcheck (using the derived Val_final)
///
/// **Critical**: Val_final is computed internally, not accepted as a parameter.
pub fn verify_output_binding_proof<F: Field + Into<K> + Into<neo_math::F> + Clone + Copy>(
    tr: &mut Poseidon2Transcript,
    num_bits: usize,
    program_io: ProgramIO<F>,
    val_init_at_r_prime: K, // Derived from MemInit (verifier can recompute)
    proof: &OutputBindingProof,
    // Terminal check values (from finalized ME claims)
    inc_terminal_value: K,
) -> Result<(), OutputCheckError> {
    // Step 1: Derive output sumcheck parameters (absorbs claims, samples r_addr)
    let params = OutputSumcheckParams::sample_from_transcript(tr, num_bits, program_io)?;

    // Step 2: Verify output sumcheck and derive r' from transcript
    let output_verifier = OutputSumcheckVerifier::new(params);
    let output_num_rounds = output_verifier.params.num_rounds();
    let output_degree_bound = output_verifier.params.degree_bound();
    let expected_output_degree = output_degree_bound + 1;

    if proof.output_sc.round_polys.len() != output_num_rounds {
        return Err(OutputCheckError::DimensionMismatch {
            expected: output_num_rounds,
            got: proof.output_sc.round_polys.len(),
        });
    }

    let mut output_current_claim = K::ZERO; // CRITICAL: Enforce initial claim = 0
    let mut output_challenges = Vec::with_capacity(output_num_rounds);

    for (round, coeffs) in proof.output_sc.round_polys.iter().enumerate() {
        if coeffs.len() != expected_output_degree {
            return Err(OutputCheckError::WrongDegree {
                round,
                expected: expected_output_degree,
                got: coeffs.len(),
            });
        }

        let p_0 = eval_poly(coeffs, K::ZERO);
        let p_1 = eval_poly(coeffs, K::ONE);
        if p_0 + p_1 != output_current_claim {
            return Err(OutputCheckError::RoundCheckFailed {
                round,
                message: format!(
                    "output sumcheck: p(0) + p(1) = {:?}, expected {:?}",
                    p_0 + p_1,
                    output_current_claim
                ),
            });
        }

        for &c in coeffs {
            absorb_k(tr, b"output_check/round_coeff", c);
        }

        let r = sample_k_challenge(tr);
        output_challenges.push(r);
        output_current_claim = eval_poly(coeffs, r);
    }

    let r_prime = &output_challenges;

    // Step 3: Verify Twist inc_total sumcheck
    tr.append_message(b"output_binding/inc_total_start", &[]);
    absorb_k(tr, b"output_binding/inc_total_claim", proof.inc_total_claim);

    // Determine expected number of rounds (same as time domain rounds)
    let inc_num_rounds = proof.inc_total_rounds.len();
    if inc_num_rounds == 0 {
        return Err(OutputCheckError::DimensionMismatch {
            expected: 1, // At least one round expected
            got: 0,
        });
    }

    let mut inc_current_claim = proof.inc_total_claim;
    let mut _inc_challenges = Vec::with_capacity(inc_num_rounds);

    for (round, coeffs) in proof.inc_total_rounds.iter().enumerate() {
        // The degree bound for TwistTotalIncOracleSparse is 2 + num_bits (has_write * inc * prod of bit eq factors)
        // But each round reduces one variable, so degree is the number of factors
        // For now, just check p(0) + p(1) = claim

        let p_0 = eval_poly(coeffs, K::ZERO);
        let p_1 = eval_poly(coeffs, K::ONE);
        if p_0 + p_1 != inc_current_claim {
            return Err(OutputCheckError::RoundCheckFailed {
                round,
                message: format!(
                    "inc_total sumcheck: p(0) + p(1) = {:?}, expected {:?}",
                    p_0 + p_1,
                    inc_current_claim
                ),
            });
        }

        for &c in coeffs {
            absorb_k(tr, b"output_binding/inc_round_coeff", c);
        }

        let r = sample_k_challenge(tr);
        _inc_challenges.push(r);
        inc_current_claim = eval_poly(coeffs, r);
    }

    // Step 4: Terminal check for inc_total
    // The verifier receives inc_terminal_value from finalized ME claims
    if inc_current_claim != inc_terminal_value {
        return Err(OutputCheckError::FinalClaimMismatch {
            got: inc_current_claim,
            expected: inc_terminal_value,
        });
    }

    // Step 5: Compute authenticated Val_final(r')
    let val_final_at_r_prime = val_init_at_r_prime + proof.inc_total_claim;

    // Step 6: Complete output sumcheck final check
    let expected_output_claim = output_verifier.expected_claim(r_prime, val_final_at_r_prime);
    if output_current_claim != expected_output_claim {
        return Err(OutputCheckError::FinalClaimMismatch {
            got: output_current_claim,
            expected: expected_output_claim,
        });
    }

    Ok(())
}

/// Lagrange interpolation to get polynomial coefficients.
fn interpolate(xs: &[K], ys: &[K]) -> Vec<K> {
    assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    let mut coeffs = vec![K::ZERO; n];

    for i in 0..n {
        // Build Lagrange basis polynomial ℓ_i(x)
        let mut numer = vec![K::ZERO; n];
        numer[0] = K::ONE;
        let mut cur_deg = 0;

        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = xs[j];
            let mut next = vec![K::ZERO; n];
            for d in 0..=cur_deg {
                next[d + 1] += numer[d];
                next[d] -= xj * numer[d];
            }
            numer = next;
            cur_deg += 1;
        }

        // Denominator
        let mut denom = K::ONE;
        for j in 0..n {
            if i != j {
                denom *= xs[i] - xs[j];
            }
        }

        let scale = ys[i] * denom.inv();
        for d in 0..n {
            coeffs[d] += scale * numer[d];
        }
    }

    coeffs
}
