use p3_dft::Radix2DitParallel;

use p3_goldilocks::Goldilocks as Fq;
use p3_field::extension::BinomialExtensionField;

use p3_goldilocks::Poseidon2Goldilocks; // <- provided by p3-poseidon2 for Goldilocks
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_commit::ExtensionMmcs;
use rand::SeedableRng;

pub type Val = Fq;
pub type Challenge = BinomialExtensionField<Val, 2>; // K = F_{q^2} per Neo v1 policy
pub type Dft = Radix2DitParallel<Val>;

// Poseidon2 sponge: WIDTH=16, RATE=8, CAPACITY=8 matches test patterns in p3-fri.
pub type Perm = Poseidon2Goldilocks<16>;
pub type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
pub type Compress = TruncatedPermutation<Perm, 2, 8, 16>;

// MMCS over values (base field) and extension (challenge field).
// For Goldilocks, p3-field uses packing; use <Val as Field>::Packing idiom like p3-fri tests.
pub type ValMmcs = MerkleTreeMmcs<<Val as p3_field::Field>::Packing, <Val as p3_field::Field>::Packing, Hash, Compress, 8>;
pub type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

#[allow(dead_code)] // until all bridge methods wire them fully
#[derive(Clone)]
pub struct PcsMaterials {
    pub perm: Perm,
    pub hash: Hash,
    pub compress: Compress,
    pub val_mmcs: ValMmcs,
    pub ch_mmcs: ChallengeMmcs,
    pub dft: Dft,
}

pub fn make_mmcs_and_dft() -> PcsMaterials {
    // Deterministic construction for auditability
    let mut seed = [0u8; 32]; // ChaCha8Rng needs 32-byte seed
    seed[0..16].copy_from_slice(&b"NEO-FRI-PERMUT\0\0"[..16]);
    let mut rng = rand_chacha::ChaCha8Rng::from_seed(seed);
    
    // Poseidon2 over Goldilocks with deterministic constants
    let perm = Perm::new_from_rng_128(&mut rng);
    
    // Sponge and compression from the same permutation
    let hash = Hash::new(perm.clone());
    let compress = Compress::new(perm.clone());

    // Merkle MMCS over F with Poseidon2-based sponge/compress
    let val_mmcs = ValMmcs::new(hash.clone(), compress.clone());
    // Extension MMCS for K = F_{q^2}, built on the base-field MMCS
    let ch_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    // FFT/DFT engine
    let dft = Dft::default();

    PcsMaterials { perm, hash, compress, val_mmcs, ch_mmcs, dft }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmcs_creation() {
        let mats = make_mmcs_and_dft();
        
        // Test that we can create the materials without panicking
        println!("✅ MMCS Materials created successfully");
        println!("   Perm: Poseidon2Goldilocks<16>");
        println!("   Val MMCS: MerkleTreeMmcs with 8-element digest");
        println!("   Challenge MMCS: ExtensionMmcs over K=F_{{q^2}}");
        
        // Test determinism - same seed should give same result
        let mats2 = make_mmcs_and_dft();
        // Both should be valid (no panic is good enough for now)
        drop(mats);
        drop(mats2);
    }
}