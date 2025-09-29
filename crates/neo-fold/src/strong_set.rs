//! Strong sampling set C ⊆ S (rotation-matrix ring) and Fiat–Shamir sampling.
//! Matches Neo §3.4 definition and usage in Π_RLC. See also Theorem 3 (expansion factor).
//! (Backend defines the element type `S` and how it acts on commitments/matrices/vectors.)
//!
//! Security note: caller must choose C that satisfies Neo's invertibility/expansion requirements.
//!
//! Domain separation tags used here are public and stable.
//!
//! Reference: Neo paper §§3.4, 4.5. (file-cited externally)

use neo_transcript::Transcript;
// use p3_field::PrimeField64; // no longer needed

/// Public, stable labels for Fiat–Shamir.
pub mod ds {
    pub const RLC: &[u8] = b"neo.rlc.v1/verify_linear";
    pub const DEC: &[u8] = b"neo.dec.v1/verify_decomposition";
    pub const CCS: &[u8] = b"neo.ccs.v1/eval_check";
}

/// A strong sampling set for challenges (rotation-operators in S).
#[derive(Clone, Debug)]
pub struct StrongSamplingSet<S> {
    elems: Vec<S>,
}

impl<S: Clone> StrongSamplingSet<S> {
    pub fn new(elems: Vec<S>) -> Self {
        assert!(!elems.is_empty(), "C must be non-empty");
        Self { elems }
    }

    #[inline]
    pub fn len(&self) -> usize { 
        self.elems.len() 
    }
    
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elems.is_empty()
    }

    /// Sample exactly k challenges ρ_i ∈ C by FS with bias-free rejection sampling.
    /// Uses the standard "threshold trick" on 64-bit output to avoid modulo bias.
    pub fn sample_k<T: Transcript>(&self, t: &mut T, label: &[u8], k: usize) -> Vec<S> {
        let n = self.elems.len() as u64;
        assert!(n > 0, "Strong set cannot be empty");
        let mut out = Vec::with_capacity(k);
        // Unbiased rejection sampling using the "zone" method:
        // zone = largest multiple of n that fits in u64
        let zone: u64 = n * (u64::MAX / n);

        for i in 0..k {
            t.append_message(b"neo/rlc/label", label);
            t.append_message(b"rho/index", &(i as u64).to_le_bytes());
            loop {
                let mut buf = [0u8; 8];
                t.challenge_bytes(b"chal/u64", &mut buf);
                let v = u64::from_le_bytes(buf);
                if v < zone {
                    let idx = (v % n) as usize;
                    out.push(self.elems[idx].clone());
                    break;
                }
                t.append_message(b"rejection", b"");
            }
        }
        out
    }
}

/// Canonical verification errors across reductions.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum VerificationError {
    #[error("linear relation on commitments failed at term {0}")]
    LinearCommit(usize),
    #[error("linear relation on X failed at row {0}, col {1}")]
    LinearX(usize, usize),
    #[error("linear relation on y[{0}] failed at coordinate {1}")]
    LinearY(usize, usize),
    #[error("decomposition identity failed for commitments (b^i * c_i sum != c)")]
    DecompCommit,
    #[error("decomposition identity failed for X (b^i * X_i rows sum != X)")]
    DecompX,
    #[error("decomposition identity failed for y[{0}] (b^i * y_i sum != y)")]
    DecompY(usize),
    #[error("CCS evaluation check failed (v != Q(alpha', r'))")]
    CcsEvalMismatch,
}
