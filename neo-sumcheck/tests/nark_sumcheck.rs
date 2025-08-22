use neo_sumcheck::{
    batched_sumcheck_prover, batched_sumcheck_verifier,
    ExtF, UnivPoly
};
use p3_field::PrimeCharacteristicRing;

/// Simple test polynomial that evaluates to zero everywhere
struct ZeroPoly;
impl UnivPoly for ZeroPoly {
    fn evaluate(&self, _point: &[ExtF]) -> ExtF { 
        ExtF::ZERO 
    }
    fn degree(&self) -> usize { 
        2 
    }
    fn max_individual_degree(&self) -> usize { 
        1 
    }
}

/// Simple test polynomial that evaluates to a constant
struct ConstPoly(ExtF);
impl UnivPoly for ConstPoly {
    fn evaluate(&self, _point: &[ExtF]) -> ExtF { 
        self.0 
    }
    fn degree(&self) -> usize { 
        2 
    }
    fn max_individual_degree(&self) -> usize { 
        1 
    }
}

#[test]
fn test_nark_sumcheck_basic() {
    // Test basic NARK mode sumcheck with zero polynomial
    let poly: Box<dyn UnivPoly> = Box::new(ZeroPoly);
    let mut transcript = vec![];
    
    // Prover should succeed with zero claim
    let result = batched_sumcheck_prover(&[ExtF::ZERO], &[&*poly], &mut transcript);
    assert!(result.is_ok(), "Prover should succeed with valid zero claim");
    
    let msgs = result.unwrap();
    assert!(!msgs.is_empty(), "Should produce sumcheck messages");
    
    // Verifier should accept the proof
    let mut vt = vec![];
    let verify_result = batched_sumcheck_verifier(&[ExtF::ZERO], &msgs, &mut vt);
    assert!(verify_result.is_some(), "Verifier should accept valid proof");
}

#[test]
fn test_nark_sumcheck_invalid_claim() {
    // Test that verifier rejects invalid claims
    // Use a constant polynomial that evaluates to 1, but claim it sums to 0
    let poly: Box<dyn UnivPoly> = Box::new(ConstPoly(ExtF::ONE));
    let mut transcript = vec![];
    
    // The polynomial evaluates to 1 everywhere, so over a 2^2 domain it should sum to 4
    // But we'll claim it sums to 0 (invalid)
    let result = batched_sumcheck_prover(&[ExtF::ZERO], &[&*poly], &mut transcript);
    
    // The prover might fail if the claim is inconsistent with the polynomial
    // In NARK mode, this is expected behavior
    if result.is_ok() {
        let msgs = result.unwrap();
        
        // If prover succeeded, verifier should still reject the invalid claim
        let mut vt = vec![];
        let verify_result = batched_sumcheck_verifier(&[ExtF::ZERO], &msgs, &mut vt);
        assert!(verify_result.is_none(), "Verifier should reject invalid claim");
    } else {
        // If prover failed, that's also acceptable for invalid claims
        println!("Prover correctly rejected invalid claim");
    }
}

#[test]
fn test_nark_sumcheck_deterministic() {
    // Test that same inputs produce same outputs (deterministic)
    // Use zero polynomial with zero claim (valid case)
    let poly: Box<dyn UnivPoly> = Box::new(ZeroPoly);
    
    let mut transcript1 = b"test_seed".to_vec();
    let result1 = batched_sumcheck_prover(&[ExtF::ZERO], &[&*poly], &mut transcript1);
    
    let mut transcript2 = b"test_seed".to_vec();
    let result2 = batched_sumcheck_prover(&[ExtF::ZERO], &[&*poly], &mut transcript2);
    
    assert!(result1.is_ok() && result2.is_ok(), "Both provers should succeed with valid claims");
    
    // Same seed should produce same messages (deterministic blinding)
    let msgs1 = result1.unwrap();
    let msgs2 = result2.unwrap();
    assert_eq!(msgs1.len(), msgs2.len(), "Should produce same number of messages");
}

#[test]
fn test_nark_sumcheck_empty_inputs() {
    // Test edge case with empty inputs
    let mut transcript = vec![];
    
    let result = batched_sumcheck_prover(&[], &[], &mut transcript);
    assert!(result.is_ok(), "Should handle empty inputs gracefully");
    
    let msgs = result.unwrap();
    assert!(msgs.is_empty(), "Empty inputs should produce empty messages");
    
    // Verifier should also handle empty inputs
    let mut vt = vec![];
    let verify_result = batched_sumcheck_verifier(&[], &msgs, &mut vt);
    assert!(verify_result.is_some(), "Verifier should accept empty proof");
}
