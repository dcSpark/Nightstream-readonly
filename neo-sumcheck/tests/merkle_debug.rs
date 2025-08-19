use neo_fields::{ExtF, F};
use neo_poly::Polynomial;
use neo_sumcheck::{FriOracle, PolyOracle};
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_merkle_roundtrip_debug() {
    // Test the simplest possible case to isolate the Merkle issue
    let poly = Polynomial::new(vec![ExtF::ZERO]); // Simple constant polynomial
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![poly.clone()], &mut transcript);
    
    // Get commitment
    let commits = oracle.commit();
    eprintln!("Commits: {:?}", commits[0]);
    
    // Open at point
    let point = vec![ExtF::ONE];
    let (evals, proofs) = oracle.open_at_point(&point);
    eprintln!("Eval: {:?}", evals[0]);
    eprintln!("Proof length: {}", proofs[0].len());
    
    // Create verifier with SAME domain size
    let domain_size = 4; // Minimum size for constant poly
    let verifier = FriOracle::new_for_verifier(domain_size);
    
    // Try to verify
    let result = verifier.verify_openings(&commits, &point, &evals, &proofs);
    eprintln!("Verification result: {}", result);
    
    if !result {
        eprintln!("DEBUGGING THE FAILURE:");
        eprintln!("Basic verification failed - this suggests transcript/domain mismatch");
    }
    
    assert!(result, "Basic Merkle roundtrip should pass");
}

#[test]
fn test_extension_field_hashing() {
    // Test that extension field hashing is consistent
    use neo_sumcheck::oracle::hash_extf;
    
    // Create ExtF with value [1, 4] like in the failing test
    let val = ExtF::new_complex(F::from_u64(1), F::from_u64(4));
    let hash1 = hash_extf(val);
    let hash2 = hash_extf(val);
    
    eprintln!("ExtF value: {:?}", val);
    eprintln!("Hash 1: {:?}", hash1);
    eprintln!("Hash 2: {:?}", hash2);
    
    assert_eq!(hash1, hash2, "Hashing should be deterministic");
}
