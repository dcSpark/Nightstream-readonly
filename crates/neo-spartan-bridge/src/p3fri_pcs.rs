use rand::{SeedableRng, rngs::SmallRng};
use serde::{Serialize, Deserialize};
use anyhow::{Result, bail};

use p3_challenger::DuplexChallenger;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64, integers::QuotientMap};
use p3_goldilocks::{Goldilocks as F, Poseidon2Goldilocks};

/// Extension field: K = F_{q^2} for Neo's single sum-check (128-bit soundness over Goldilocks)
#[allow(dead_code)]
pub type K = BinomialExtensionField<F, 2>;

// Real p3-fri types for correct structure
type Perm = Poseidon2Goldilocks<16>;
pub type Challenger = DuplexChallenger<F, Perm, 16, 8>;

#[derive(Clone, Debug)]
pub struct FriConfig {
    pub log_blowup: usize,
    pub log_final_poly_len: usize,
    pub num_queries: usize,
    pub proof_of_work_bits: u32,
}

impl Default for FriConfig {
    fn default() -> Self {
        Self {
            log_blowup: 1,           // 2^1 blowup
            log_final_poly_len: 0,   // stop when constant
            num_queries: 30,         // reasonable for development
            proof_of_work_bits: 8,   // small anti-grinding
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Commitments {
    pub inner: Vec<u8>, // Serialized commitment
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProverData {
    pub inner: Vec<u8>, // Serialized prover data
}

#[derive(Clone, Debug)]
pub struct OpeningRequest<FX> {
    pub points: Vec<FX>,
    pub evals_hint: Option<Vec<Vec<FX>>>, // Optional hint for expected evaluations
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proof {
    pub bytes: Vec<u8>,
}

/// Neo's clean PCS interface - stays native to p3 ecosystem without ff::PrimeField wrappers
pub trait NeoPcs {
    type FX: Field;
    
    fn commit(&self, polys: &[Vec<Self::FX>]) -> (Commitments, ProverData);
    fn open(&self, pd: &ProverData, req: &OpeningRequest<Self::FX>) -> Proof;
    fn verify(&self, com: &Commitments, req: &OpeningRequest<Self::FX>, proof: &Proof) -> Result<()>;
}

pub struct P3FriPCS {
    #[allow(dead_code)]
    config: FriConfig,
    #[allow(dead_code)]
    challenger_template: Challenger,
}

impl P3FriPCS {
    pub fn new(config: FriConfig) -> Self {
        let challenger = make_goldilocks_challenger();
        Self {
            config,
            challenger_template: challenger,
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(FriConfig::default())
    }
}

impl NeoPcs for P3FriPCS {
    type FX = F;

    fn commit(&self, polys: &[Vec<F>]) -> (Commitments, ProverData) {
        if polys.is_empty() {
            return (
                Commitments { inner: vec![] },
                ProverData { inner: vec![] },
            );
        }

        // TODO: Replace with real p3-FRI commit implementation
        // This is where we would:
        // 1. Create domains with self.pcs.natural_domain_for_degree(degree)
        // 2. Convert Vec<Vec<F>> to RowMajorMatrix<F>  
        // 3. Call self.pcs.commit([(domain, evals)])
        // 4. Serialize the real commitment and prover_data
        
        println!("P3FRI commit: {} polynomials (degree {})", polys.len(), polys[0].len());
        
        // Placeholder implementation using deterministic "commitments"
        let mut commitment_data = Vec::new();
        for (i, poly) in polys.iter().enumerate() {
            let poly_hash = poly.iter().fold(F::ZERO, |acc, &x| acc + x) 
                           + F::from_canonical_checked(i as u64).unwrap();
            commitment_data.extend_from_slice(&poly_hash.as_canonical_u64().to_le_bytes());
        }
        
        let prover_data = format!("p3fri_prover_data_{}_{}", polys.len(), commitment_data.len());
        
        (
            Commitments { inner: commitment_data },
            ProverData { inner: prover_data.into_bytes() },
        )
    }

    fn open(&self, pd: &ProverData, req: &OpeningRequest<F>) -> Proof {
        if pd.inner.is_empty() || req.points.is_empty() {
            return Proof { bytes: vec![] };
        }

        // TODO: Replace with real p3-FRI open implementation  
        // This is where we would:
        // 1. Deserialize prover_data
        // 2. Clone challenger and convert F points to K points
        // 3. Call self.pcs.open(vec![(&prover_data[0], points_per_matrix)], &mut challenger)
        // 4. Serialize the real (opened_values, proof)
        
        println!("P3FRI open: {} points", req.points.len());
        
        // Placeholder proof generation
        let mut proof_data = Vec::new();
        proof_data.extend_from_slice(&(req.points.len() as u32).to_le_bytes());
        proof_data.extend_from_slice(&pd.inner);
        
        for &point in &req.points {
            proof_data.extend_from_slice(&point.as_canonical_u64().to_le_bytes());
        }
        proof_data.extend_from_slice(b"p3fri_proof_placeholder");
        
        Proof { bytes: proof_data }
    }

    fn verify(&self, com: &Commitments, req: &OpeningRequest<F>, proof: &Proof) -> Result<()> {
        if com.inner.is_empty() || req.points.is_empty() || proof.bytes.is_empty() {
            bail!("Empty commitment, points, or proof");
        }

        // TODO: Replace with real p3-FRI verify implementation
        // This is where we would:
        // 1. Deserialize commitment and proof 
        // 2. Clone challenger and observe commitment[0]
        // 3. Convert F points to K points
        // 4. Build claims: vec![(commit, vec![(domain, values_at_points)])]  
        // 5. Call self.pcs.verify(claims, &fri_proof, &mut challenger)
        
        println!("P3FRI verify: {} points, {} commitment bytes, {} proof bytes", 
                 req.points.len(), com.inner.len(), proof.bytes.len());
        
        // Placeholder verification - basic sanity checks
        if proof.bytes.len() < 8 {
            bail!("Proof too short");
        }
        
        let expected_points = u32::from_le_bytes([
            proof.bytes[0], proof.bytes[1], proof.bytes[2], proof.bytes[3]
        ]) as usize;
        
        if expected_points != req.points.len() {
            bail!("Point count mismatch: expected {}, got {}", expected_points, req.points.len());
        }

        // For now, always succeed if basic structure is valid
        println!("P3FRI verify: PASS (placeholder implementation)");
        Ok(())
    }
}

pub fn make_goldilocks_challenger() -> Challenger {
    // TODO: Replace with real Poseidon2 setup
    // This is where we would:
    // 1. Create Poseidon2Goldilocks<16> with new_from_rng_128(&mut rng)
    // 2. Create DuplexChallenger::new(perm)
    
    let mut rng = SmallRng::seed_from_u64(0);
    let perm = Perm::new_from_rng_128(&mut rng);
    Challenger::new(perm)
}

// TODO: Implement make_goldilocks_pcs with real p3-FRI setup  
// This is where we would wire up the real:
// - Hash = PaddingFreeSponge<Perm, 16, 8, 8>
// - Compress = TruncatedPermutation<Perm, 2, 8, 16> 
// - ValMmcs = MerkleTreeMmcs<F::Packing, F::Packing, Hash, Compress, 8>
// - ChallengeMmcs = ExtensionMmcs<F, K, ValMmcs>
// - FriPcs = TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs>
// - FriParameters with config.log_blowup, config.num_queries, etc.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p3fri_pcs_creation() {
        let _pcs = P3FriPCS::with_default_config();
        println!("Created P3FriPCS with challenger-backed placeholder");
    }

    #[test]
    fn test_p3fri_commit_open_verify() {
        let pcs = P3FriPCS::with_default_config();
        
        // Create a simple polynomial: P(x) = 7 for all x (must be power of two)
        let degree = 64; // Power of two
        let poly = vec![F::from_canonical_checked(7).unwrap(); degree];
        let polys = vec![poly];
        
        // Commit to the polynomial
        let (commitment, prover_data) = pcs.commit(&polys);
        assert!(!commitment.inner.is_empty());
        
        // Open at a specific point
        let point = F::ZERO;
        let req = OpeningRequest {
            points: vec![point],
            evals_hint: None,
        };
        let proof = pcs.open(&prover_data, &req);
        
        // Verify the opening
        assert!(pcs.verify(&commitment, &req, &proof).is_ok());
        
        println!("✅ P3-FRI PCS test: PASS");
        println!("   Commitment: {} bytes", commitment.inner.len()); 
        println!("   Proof: {} bytes", proof.bytes.len());
        println!("   (Ready for real p3-fri implementation)");
    }

    #[test]
    fn test_challenger_creation() {
        let challenger = make_goldilocks_challenger();
        let _clone = challenger.clone();
        println!("✅ Goldilocks challenger creation: PASS");
        println!("   (Real Poseidon2Goldilocks<16> + DuplexChallenger)");
    }
}