#[cfg(test)]
mod tests {
    use neo_ccs::{check_satisfiability, verifier_ccs, CcsInstance, CcsWitness};
    use neo_commit::{AjtaiCommitter, TOY_PARAMS};
    use neo_fields::{embed_base_to_ext, ExtF, F};
    use neo_fold::{extractor, FoldState, Proof};
    use neo_sumcheck::{FriOracle, PolyOracle};
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_verifier_ccs_satisfiability() {
        let structure = verifier_ccs();
        let instance = CcsInstance {
            commitment: vec![],
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };
        // For the verifier CCS with 4-element witness [a, b, a*b, a+b]:
        // Valid witness: a=2, b=3, a*b=6, a+b=5
        let witness = CcsWitness {
            z: vec![
                embed_base_to_ext(F::from_u64(2)),
                embed_base_to_ext(F::from_u64(3)),
                embed_base_to_ext(F::from_u64(6)),  // 2*3
                embed_base_to_ext(F::from_u64(5)),  // 2+3
            ],
        };
        assert!(check_satisfiability(&structure, &instance, &witness));

        // Invalid witness: a=2, b=3, a*b=7 (wrong!), a+b=5 (mul_check=6-7=-1≠0)
        let bad_witness = CcsWitness {
            z: vec![
                embed_base_to_ext(F::from_u64(2)),
                embed_base_to_ext(F::from_u64(3)),
                embed_base_to_ext(F::from_u64(7)),  // Wrong: should be 6
                embed_base_to_ext(F::from_u64(5)),  // Correct: 2+3=5
            ],
        };
        assert!(!check_satisfiability(&structure, &instance, &bad_witness));
    }

    #[test]
    fn test_extractor_valid_witness() {
        let proof = Proof { transcript: vec![1u8; 100] }; // Demo transcript
        let wit = extractor(&proof);
        assert_eq!(wit.z.len(), 4); // Matches demo verifier CCS inputs
        assert!(wit.z.iter().all(|&e| e != ExtF::ZERO)); // Non-trivial

        // Invalid transcript (e.g., too short) should produce default witness
        let bad_proof = Proof { transcript: vec![0u8; 10] };
        let bad_wit = extractor(&bad_proof);
        assert_eq!(bad_wit.z.len(), 4); // Still produces 4 elements
    }

    #[test]
    fn test_fri_compression_roundtrip() {
        let state = FoldState::new(verifier_ccs()); // Use verifier CCS for demo
        let transcript = vec![1u8; 16]; // Smaller transcript for tractable polynomial
        let (commit, proof) = state.compress_proof(&transcript);

        // Create the same polynomial as compress_proof does
        use neo_fields::from_base;
        use neo_poly::Polynomial;
        let poly = Polynomial::new(transcript.iter().map(|&b| from_base(F::from_u64(b as u64))).collect::<Vec<_>>());
        let point = vec![ExtF::ONE]; // Same point as compress_proof
        
        // Get the actual evaluation from the polynomial 
        let actual_eval = poly.eval(point[0]);
        
        let comm_vec = vec![commit];
        let proof_vec = vec![proof];
        let eval_vec = vec![actual_eval];
        let verifier = FriOracle::new_for_verifier(transcript.len().next_power_of_two() * 4);
        assert!(verifier.verify_openings(&comm_vec, &point, &eval_vec, &proof_vec));

        // Tamper proof and reject
        let mut bad_proof = proof_vec[0].clone();
        if !bad_proof.is_empty() { bad_proof[0] ^= 1; }
        let bad_proof_vec = vec![bad_proof];
        assert!(!verifier.verify_openings(&comm_vec, &point, &eval_vec, &bad_proof_vec));
    }

    #[test]
    fn test_recursive_ivc_chaining() {
        let mut state = FoldState::new(verifier_ccs());
        let committer = AjtaiCommitter::setup(TOY_PARAMS); // Use toy for speed
        let depth = 2;

        // Set initial CCS (demo) - use 4-element witness
        state.ccs_instance = Some((
            CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE },
            CcsWitness { z: vec![
                embed_base_to_ext(F::from_u64(2)),
                embed_base_to_ext(F::from_u64(3)),
                embed_base_to_ext(F::from_u64(6)),
                embed_base_to_ext(F::from_u64(5)),
            ] },
        ));

        assert!(state.recursive_ivc(depth, &committer));
        assert!(state.verify_state()); // Final state valid

        // Force failure (e.g., invalid initial witness)
        let mut bad_state = FoldState::new(verifier_ccs());
        bad_state.ccs_instance = Some((
            CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE },
            CcsWitness { z: vec![
                embed_base_to_ext(F::from_u64(2)),
                embed_base_to_ext(F::from_u64(3)),
                embed_base_to_ext(F::from_u64(7)), // Invalid: should be 6
                embed_base_to_ext(F::from_u64(5)),
            ] }, // Invalid witness
        ));
        assert!(!bad_state.recursive_ivc(depth, &committer));
    }

    #[test]
    fn test_verifier_ccs_structure() {
        let structure = verifier_ccs();
        assert_eq!(structure.mats.len(), 4); // 4 matrices as expected
        assert_eq!(structure.num_constraints, 2); // 2 constraints
        assert_eq!(structure.witness_size, 4); // 4 witness elements [a, b, a*b, a+b]
        assert_eq!(structure.max_deg, 2); // Degree 2 polynomial
    }

    #[test]
    fn test_compress_proof_deterministic() {
        let state = FoldState::new(verifier_ccs());
        let transcript = vec![42u8; 50];
        
        // Same transcript should produce same compression
        let (commit1, proof1) = state.compress_proof(&transcript);
        let (commit2, proof2) = state.compress_proof(&transcript);
        
        assert_eq!(commit1, commit2);
        assert_eq!(proof1, proof2);
        
        // Different transcripts should produce different compressions
        let different_transcript = vec![43u8; 50];
        let (commit3, _proof3) = state.compress_proof(&different_transcript);
        assert_ne!(commit1, commit3);
    }

    #[test]
    fn test_recursive_ivc_base_case() {
        let mut state = FoldState::new(verifier_ccs());
        let committer = AjtaiCommitter::setup(TOY_PARAMS);
        
        // Test depth 0 (base case)
        assert!(state.recursive_ivc(0, &committer));
        
        // With eval instances, should return true
        state.eval_instances.push(neo_fold::EvalInstance {
            commitment: vec![],
            r: vec![ExtF::ONE],
            ys: vec![ExtF::ZERO],
            u: ExtF::ZERO,
            e_eval: ExtF::ONE,
            norm_bound: 100,
        });
        assert!(state.recursive_ivc(0, &committer));
    }
}
