// Temporarily allow unsafe for field construction - will be removed when p3 provides safe API
// #![forbid(unsafe_code)]
//! neo-spartan-bridge
//!
//! Last-mile compression: translate a final ME(b, L) claim into a p3-FRI proof.
//!
//! This provides post-quantum security via pure hash-based polynomial commitments,
//! staying native to small fields (Goldilocks) and avoiding elliptic curve cryptography.
//!
//! Architecture:
//! - NeoPcs trait: Clean p3-native PCS interface
//! - P3FriPCS: Real hash-based FRI implementation using p3-fri + Poseidon2  
//! - SpartanBridge: High-level interface for ME(b,L) compression
//! - No ff::PrimeField wrappers needed - everything stays in p3 ecosystem

mod p3fri_pcs;

pub use p3fri_pcs::{P3FriPCS, NeoPcs, FriConfig, Commitments, OpeningRequest, Proof};

// Import from our local p3fri_pcs module
// use p3_fri::FriConfig;
use p3_goldilocks::Goldilocks as F;

/// High-level bridge for compressing ME(b,L) claims to Spartan2(+FRI) proofs
pub struct SpartanBridge {
    pcs: P3FriPCS,
}

impl SpartanBridge {
    pub fn new(fri_cfg: FriConfig) -> Self {
        Self { pcs: P3FriPCS::new(fri_cfg) }
    }

    pub fn with_default_config() -> Self {
        Self { pcs: P3FriPCS::with_default_config() }
    }

    /// Compress a linearized ME(b,L) claim to a Spartan2(+FRI) proof.
    /// `polys_over_f` are the multilinear eval tables Spartan expects as polynomials over F.
    pub fn compress_me(
        &self,
        polys_over_f: &[Vec<F>],
        open_points: &[F],
    ) -> anyhow::Result<(Commitments, Proof)> {
        let (com, pd) = self.pcs.commit(polys_over_f);
        let req = OpeningRequest::<F> { points: open_points.to_vec(), evals_hint: None };
        let proof = self.pcs.open(&pd, &req);
        Ok((com, proof))
    }

    pub fn verify_me(
        &self,
        commitments: &Commitments,
        open_points: &[F],
        proof: &Proof,
    ) -> anyhow::Result<()> {
        let req = OpeningRequest::<F> { points: open_points.to_vec(), evals_hint: None };
        self.pcs.verify(commitments, &req, proof)
    }
}

/// Legacy compression artifact for compatibility
#[derive(Clone, Debug)]
pub struct SpartanCompressionArtifact {
    pub commitment_bytes: Vec<u8>,
    pub proof_bytes: Vec<u8>,
    /// Timing breakdown 
    pub timings: CompressionTimings,
}

#[derive(Clone, Debug, Default)]
pub struct CompressionTimings {
    pub commit_ms: u128,
    pub prove_ms: u128,  
    pub verify_ms: u128,
    pub total_bytes: usize,
}