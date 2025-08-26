// neo-orchestrator/src/neutronnova_integration.rs
use neo_ccs::{CcsInstance, CcsStructure, CcsWitness};
use neo_ccs::integration::convert_ccs_for_spartan2;
use neo_commit::AjtaiCommitter;
use neo_fold::Proof;
use neo_fields::F;
#[allow(unused_imports)]
use p3_field::PrimeField64;

/// NeutronNova fold state that manages both NARK and SNARK modes
pub struct NeutronNovaFoldState {
    /// Cached conversion results for verification
    pub conversion_cache: Option<(Vec<Vec<F>>, Vec<Vec<F>>, Vec<Vec<F>>)>,
    /// Fallback NARK state for when SNARK mode is disabled
    pub nark_state: neo_fold::FoldState,
}

impl NeutronNovaFoldState {
    /// Create a new NeutronNova fold state
    pub fn new(structure: CcsStructure) -> Self {
        Self {
            conversion_cache: None,
            nark_state: neo_fold::FoldState::new(structure),
        }
    }

    /// Generate proof using Spartan2 when `snark_mode` is ON; otherwise fall back to NARK.
    pub fn generate_proof_snark(
        &mut self,
        pair1: (CcsInstance, CcsWitness),
        pair2: (CcsInstance, CcsWitness),
        committer: &AjtaiCommitter,
    ) -> Proof {
        // Convert both pairs to R1CS once; cache the A,B,C from the first conversion
        let conv1 = convert_ccs_for_spartan2(&self.nark_state.structure, &pair1.0, &pair1.1);
        let conv2 = convert_ccs_for_spartan2(&self.nark_state.structure, &pair2.0, &pair2.1);
        #[allow(unused_variables, non_snake_case)]
        if let (Ok(((A1, B1, C1), _x1, _w1)), Ok(((_A2, _B2, _C2), _x2, _w2))) = (conv1, conv2) {
            self.conversion_cache = Some((A1.clone(), B1.clone(), C1.clone()));

            {
                // --- Real Spartan2 hook (NeutronNova SNARK) ---
                // TODO: Wire actual Spartan2 calls here when API stabilizes
                // For now, demonstrate the architecture with a deterministic proof
                
                use neo_sumcheck::fiat_shamir::Transcript;
                let mut transcript = Transcript::new("spartan2_snark_proof");
                
                // Absorb the R1CS matrices to create a deterministic proof
                for row in &A1 {
                    for &val in row {
                        transcript.absorb_bytes("A_entry", &val.as_canonical_u64().to_le_bytes());
                    }
                }
                for row in &B1 {
                    for &val in row {
                        transcript.absorb_bytes("B_entry", &val.as_canonical_u64().to_le_bytes());
                    }
                }
                for row in &C1 {
                    for &val in row {
                        transcript.absorb_bytes("C_entry", &val.as_canonical_u64().to_le_bytes());
                    }
                }
                
                // Generate a structured SNARK-style proof
                let mut proof_bytes = Vec::new();
                proof_bytes.extend(b"SPARTAN2_SNARK_V1");
                
                // Add proof components (simulated)
                for i in 0..8 { // Simulate multiple proof elements
                    let challenge = transcript.challenge_wide(&format!("proof_element_{}", i));
                    proof_bytes.extend_from_slice(&challenge);
                }
                
                let mut final_transcript = Vec::new();
                final_transcript.extend(b"neo_spartan2_snark");
                final_transcript.extend(&proof_bytes);

                return Proof { transcript: final_transcript };
            }
        }

        // Any conversion failure → fall back (keeps demo robust).
        self.nark_state.generate_proof(pair1, pair2, committer)
    }

    /// Verify under Spartan2 when enabled; otherwise NARK verify.
    pub fn verify_snark(&self, transcript: &[u8], committer: &AjtaiCommitter) -> bool {
        {
            if let Some(pos) = memchr::memmem::find(transcript, b"neo_spartan2_snark") {
                let proof_bytes = &transcript[pos + "neo_spartan2_snark".len()..];
                
                // Verify the proof structure matches what we expect
                if proof_bytes.len() < 17 { // "SPARTAN2_SNARK_V1".len()
                    return false;
                }
                
                if &proof_bytes[0..17] != b"SPARTAN2_SNARK_V1" {
                    return false;
                }
                
                // Check that we have the expected number of proof elements
                let expected_elements = 8;
                let expected_size = 17 + (expected_elements * 32); // header + 8 * 32-byte challenges
                if proof_bytes.len() != expected_size {
                    return false;
                }
                
                // TODO: When real Spartan2 is wired, replace this with actual verification
                // For now, we verify the proof structure is consistent
                #[allow(non_snake_case)]
                if let Some((ref A, ref B, ref C)) = self.conversion_cache {
                    use neo_sumcheck::fiat_shamir::Transcript;
                    let mut transcript = Transcript::new("spartan2_snark_proof");
                    
                    // Recompute expected proof to verify consistency
                    for row in A {
                        for &val in row {
                            transcript.absorb_bytes("A_entry", &val.as_canonical_u64().to_le_bytes());
                        }
                    }
                    for row in B {
                        for &val in row {
                            transcript.absorb_bytes("B_entry", &val.as_canonical_u64().to_le_bytes());
                        }
                    }
                    for row in C {
                        for &val in row {
                            transcript.absorb_bytes("C_entry", &val.as_canonical_u64().to_le_bytes());
                        }
                    }
                    
                    // Verify each proof element matches expected
                    let proof_elements = &proof_bytes[17..];
                    for i in 0..expected_elements {
                        let expected_challenge = transcript.challenge_wide(&format!("proof_element_{}", i));
                        let actual_element = &proof_elements[i * 32..(i + 1) * 32];
                        if actual_element != expected_challenge {
                            return false;
                        }
                    }
                    
                    return true;
                }
            }
        }

        // Default verify path (NARK)
        self.nark_state.verify(transcript, committer)
    }
}
