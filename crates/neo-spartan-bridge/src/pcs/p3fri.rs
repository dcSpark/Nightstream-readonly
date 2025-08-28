use anyhow::Result;

use crate::pcs::mmcs::{make_mmcs_and_dft, PcsMaterials};

/// Public parameters you'll pass down from the bridge.
#[derive(Clone)]
pub struct P3FriParams {
    pub log_blowup: usize,       // e.g. 1–2 in prod, 2 in tests
    pub log_final_poly_len: usize,
    pub num_queries: usize,      // choose for your error budget
    pub proof_of_work_bits: usize, // DoS/grinding
}

impl Default for P3FriParams {
    fn default() -> Self {
        Self {
            log_blowup: 2,
            log_final_poly_len: 0,
            num_queries: 60,       // tune to your soundness target
            proof_of_work_bits: 8, // prevents trivial grinding
        }
    }
}

pub struct P3FriPCS {
    #[allow(dead_code)]
    mats: PcsMaterials,
    params: P3FriParams,
}

impl P3FriPCS {
    pub fn new(params: P3FriParams) -> Self {
        let mats = make_mmcs_and_dft();
        // Touch one field so `mats` isn't entirely dead on strict lints.
        let _ = &mats.hash;
        // TODO: Once p3-FRI API is stabilized, create real TwoAdicFriPcs here:
        // let fri_params = FriParameters {
        //     log_blowup: params.log_blowup,
        //     log_final_poly_len: params.log_final_poly_len,
        //     num_queries: params.num_queries,
        //     proof_of_work_bits: params.proof_of_work_bits,
        //     mmcs: mats.ch_mmcs.clone(),
        // };
        // let pcs = TwoAdicFriPcs::new(mats.dft.clone(), mats.val_mmcs.clone(), fri_params);
        Self { mats, params }
    }

    /// TODO: Implement with real p3-FRI once API issues are resolved
    /// This will be the natural two-adic domain that p3-fri expects.
    pub fn domain_for_degree(&self, degree: usize) -> usize {
        // Placeholder - return the degree
        // Real implementation would be:
        // self.pcs.natural_domain_for_degree(degree)
        degree
    }

    /// TODO: Real commit implementation
    /// This will commit a collection of matrices using real p3-FRI
    pub fn commit_placeholder(
        &self,
        num_polys: usize,
        degree: usize,
    ) -> Vec<u8> {
        // Placeholder commitment
        format!("p3fri_commit_{}_polys_degree_{}_blowup_{}", 
                num_polys, degree, self.params.log_blowup).into_bytes()
    }

    /// TODO: Real open implementation  
    /// This will open at points using real p3-FRI
    pub fn open_placeholder(
        &self,
        _commit: &[u8],
        num_points: usize,
        io_preimage: &[u8],
    ) -> Vec<u8> {
        // Placeholder proof
        format!("p3fri_proof_{}_points_io_{}_queries_{}", 
                num_points, io_preimage.len(), self.params.num_queries).into_bytes()
    }

    /// TODO: Real verify implementation
    /// This will verify using real p3-FRI  
    pub fn verify_placeholder(
        &self,
        commit: &[u8],
        proof: &[u8],
        _num_points: usize,
        io_preimage: &[u8],
    ) -> Result<()> {
        // Placeholder verification
        if commit.is_empty() || proof.is_empty() || io_preimage.is_empty() {
            anyhow::bail!("Empty inputs to verify");
        }
        
        // For now, always succeed for valid-looking inputs
        Ok(())
    }
}

// TODO: Once p3-FRI API is stabilized, implement the Spartan2 PCS adapter trait here:
// 
// This is where we would implement `spartan2::PCSEngineTrait<E>` for `P3FriPCSAdapter<E>`
// The adapter would:
// 1. Hold a P3FriPCS instance
// 2. Forward setup() -> P3FriPCS::new()
// 3. Forward commit() -> commit_round() with proper domain/matrix conversion  
// 4. Forward prove() -> open_round() with challenger seeded from transcript
// 5. Forward verify() -> verify_round() with same challenger seeding
//
// Once this adapter exists, we can replace HyraxPCS<E> with P3FriPCSAdapter<E> in the bridge

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p3fri_pcs_creation() {
        let params = P3FriParams::default();
        let pcs = P3FriPCS::new(params);
        
        println!("✅ P3FriPCS created with real MMCS materials");
        println!("   log_blowup: {}", pcs.params.log_blowup);
        println!("   num_queries: {}", pcs.params.num_queries);
        println!("   MMCS: Poseidon2 + MerkleTreeMmcs + ExtensionMmcs");
        println!("   Ready for real p3-FRI integration");
    }

    #[test] 
    fn test_domain_creation() {
        let pcs = P3FriPCS::new(P3FriParams::default());
        
        let degree = 64;
        let domain = pcs.domain_for_degree(degree);
        
        assert_eq!(domain, degree); // placeholder behavior
        
        println!("✅ Domain placeholder works (degree {})", degree);
    }

    #[test]
    fn test_commit_open_verify_flow() {
        let pcs = P3FriPCS::new(P3FriParams::default());
        
        let num_polys = 3;
        let degree = 64;
        let num_points = 2;
        let io_bytes = b"test_transcript_io";
        
        // Full placeholder flow
        let commit = pcs.commit_placeholder(num_polys, degree);
        let proof = pcs.open_placeholder(&commit, num_points, io_bytes);
        let result = pcs.verify_placeholder(&commit, &proof, num_points, io_bytes);
        
        assert!(result.is_ok());
        assert!(!commit.is_empty());
        assert!(!proof.is_empty());
        
        println!("✅ Commit/Open/Verify placeholder flow works");
        println!("   Commit: {} bytes", commit.len());
        println!("   Proof: {} bytes", proof.len());
        println!("   Ready for real p3-FRI calls");
    }
}