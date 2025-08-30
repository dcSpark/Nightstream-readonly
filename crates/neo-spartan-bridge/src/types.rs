use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ProofBundle {
    /// Spartan2 proof bytes (includes the Hash‑MLE PCS proof inside Spartan2's structure).
    pub proof: Vec<u8>,
    /// Verifier key (serialized)
    pub vk: Vec<u8>,
    /// Public IO you expect verifiers to re-encode identically (bridge header + public inputs).
    pub public_io_bytes: Vec<u8>,
}

impl ProofBundle {
    pub fn new_with_vk(proof: Vec<u8>, vk: Vec<u8>, public_io_bytes: Vec<u8>) -> Self {
        Self { proof, vk, public_io_bytes }
    }

    /// Legacy constructor for compatibility - FRI params are ignored (always 0)
    #[deprecated(note = "FRI is no longer used. Use new_with_vk instead.")]
    pub fn new(
        proof: Vec<u8>,
        public_io_bytes: Vec<u8>, 
        _fri_num_queries: usize,  // Deprecated: retained for compatibility; always 0 (FRI is not used)
        _fri_log_blowup: usize,   // Deprecated: retained for compatibility; always 0 (FRI is not used)
    ) -> Self {
        Self {
            proof,
            vk: Vec::new(), // Empty VK for legacy compatibility
            public_io_bytes,
        }
    }

    pub fn total_size(&self) -> usize {
        self.proof.len() + self.vk.len() + self.public_io_bytes.len()
    }

    // Legacy accessors for compatibility
    #[deprecated(note = "FRI is no longer used, returns 0")]
    pub fn fri_num_queries(&self) -> usize { 0 }
    
    #[deprecated(note = "FRI is no longer used, returns 0")]
    pub fn fri_log_blowup(&self) -> usize { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_sizes() {
        let b = ProofBundle::new_with_vk(vec![1,2], vec![3], vec![4,5,6]);
        assert_eq!(b.total_size(), 2+1+3);
    }

    #[test]
    #[allow(deprecated)]
    fn legacy_constructor() {
        let b = ProofBundle::new(vec![1,2], vec![4,5,6], 100, 2);
        assert_eq!(b.proof.len(), 2);
        assert_eq!(b.public_io_bytes.len(), 3);
        assert_eq!(b.fri_num_queries(), 0);
        assert_eq!(b.fri_log_blowup(), 0);
    }
}