use serde::{Serialize, Deserialize};
use std::fmt;

#[derive(Serialize, Deserialize)]
pub struct ProofBundle {
    /// Spartan2 proof bytes (includes the Hash‑MLE PCS proof inside Spartan2's structure).
    pub proof: Vec<u8>,
    /// Verifier key (serialized)
    pub vk: Vec<u8>,
    /// Public IO you expect verifiers to re-encode identically (bridge header + public inputs).
    pub public_io_bytes: Vec<u8>,
}

// Custom Debug implementation to avoid dumping massive binary data
impl fmt::Debug for ProofBundle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProofBundle")
            .field("proof", &format!("[{} bytes]", self.proof.len()))
            .field("vk", &format!("[{} bytes]", self.vk.len()))
            .field("public_io_bytes", &format!("[{} bytes]", self.public_io_bytes.len()))
            .finish()
    }
}

impl ProofBundle {
    pub fn new_with_vk(proof: Vec<u8>, vk: Vec<u8>, public_io_bytes: Vec<u8>) -> Self {
        Self { proof, vk, public_io_bytes }
    }

    pub fn total_size(&self) -> usize {
        self.proof.len() + self.vk.len() + self.public_io_bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_sizes() {
        let b = ProofBundle::new_with_vk(vec![1,2], vec![3], vec![4,5,6]);
        assert_eq!(b.total_size(), 2+1+3);
    }
}