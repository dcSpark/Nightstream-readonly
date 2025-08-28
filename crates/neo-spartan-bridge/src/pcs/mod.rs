pub mod p3fri;

use p3_field::Field;

/// Minimal PCS interface your bridge and neo-fold can rely on,
/// independent of Spartan2/bellpepper/ff.
/// 
/// This stays purely in p3 ecosystem - no ff::PrimeField conversions needed.
pub trait NeoPcs<F: Field> {
    type Commitment: Clone;
    type ProverData;
    type Proof;

    /// Commit to a set of oracles (each oracle is a polynomial given in some encoding).
    fn commit(&self, oracles: &[Vec<F>]) -> (Self::Commitment, Self::ProverData);

    /// Open all committed oracles at the set of points; return a batched opening proof.
    fn open(&self, pd: &Self::ProverData, points: &[F]) -> Self::Proof;

    /// Verify a batched opening proof for the commitment at the given points.
    fn verify(&self, cm: &Self::Commitment, points: &[F], proof: &Self::Proof) -> bool;
}

/// Backing types we standardize across PCS implementations
/// These are just byte vectors - clean boundary with external systems
pub type NeoPcsCommitment = Vec<u8>;
pub type NeoPcsProof = Vec<u8>;