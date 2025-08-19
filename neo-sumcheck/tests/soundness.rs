use neo_fields::ExtF;
use neo_poly::Polynomial;
use neo_sumcheck::{batched_sumcheck_prover, batched_sumcheck_verifier, oracle::FriOracle};
use neo_sumcheck::PolyOracle;
use quickcheck_macros::quickcheck;
use rand::Rng;
use p3_field::PrimeCharacteristicRing;

/// Test that verifier rejects random invalid claims with high probability.
/// This validates soundness (knowledge error <1/|K| per round, paper Lemma 9).
/// Property-based test for statistical security.
#[quickcheck]
fn prop_reject_invalid_claim(bad_offset: u64) -> bool {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![poly.clone()], &mut transcript);
    
    // Create an invalid claim by adding a random offset
    let actual_sum = poly.eval(ExtF::ONE) + poly.eval(ExtF::from_u64(2));
    let invalid_claim = actual_sum + ExtF::from_u64(bad_offset.wrapping_add(1)); // Ensure non-zero offset
    
    // Try to prove the invalid claim
    let result = batched_sumcheck_prover(&[invalid_claim], &[&poly], &mut oracle, &mut transcript);
    
    match result {
        Ok((msgs, comms)) => {
            // If prover succeeded, verifier should still reject
            let mut verifier_transcript = vec![];
            let verifier_oracle = FriOracle::new_for_verifier(8);
            batched_sumcheck_verifier(&[invalid_claim], &msgs, &comms, &verifier_oracle, &mut verifier_transcript).is_none()
        }
        Err(_) => true // Prover correctly failed
    }
}

/// Test that verifier rejects forged sumcheck messages.
/// Prevents adversarial prover from faking reductions (Theorem 1 soundness).
#[test]
fn test_reject_forged_msgs() {
    let poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![poly.clone()], &mut transcript);
    
    let claim = ExtF::ONE + ExtF::ONE;
    let result = batched_sumcheck_prover(&[claim], &[&poly], &mut oracle, &mut transcript);
    
    if let Ok((mut msgs, comms)) = result {
        // Forge the first message
        if !msgs.is_empty() {
            msgs[0] = (Polynomial::new(vec![ExtF::ONE]), ExtF::ONE); // Invalid message
        }
        
        let mut verifier_transcript = vec![];
        let verifier_oracle = FriOracle::new_for_verifier(8);
        assert!(batched_sumcheck_verifier(&[claim], &msgs, &comms, &verifier_oracle, &mut verifier_transcript).is_none());
    }
}

/// Test that FRI verifier rejects tampered proofs with high probability.
/// Validates proof system soundness against malicious provers.
#[test]
fn test_fri_soundness_reject_tampered() {
    let poly = Polynomial::new(vec![ExtF::ONE, ExtF::from_u64(2)]);
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![poly.clone()], &mut transcript);
    let comms = oracle.commit();
    let point = vec![ExtF::ONE];
    let (evals, mut proofs) = oracle.open_at_point(&point);
    
    let mut reject_count = 0;
    let trials = 100;
    
    for trial in 0..trials {
        if let Some(proof) = proofs.get_mut(0) {
            if !proof.is_empty() {
                // Tamper one byte based on trial number
                let tamper_pos = trial % proof.len();
                let original = proof[tamper_pos];
                proof[tamper_pos] ^= 1;
                
                let domain_size = (poly.degree() + 1).next_power_of_two() * 4;
                let verifier = FriOracle::new_for_verifier(domain_size);
                
                if !verifier.verify_openings(&comms, &point, &evals, &proofs) {
                    reject_count += 1;
                }
                
                // Restore original
                proof[tamper_pos] = original;
            }
        }
    }
    
    // Should reject most tampering attempts (>95%)
    assert!(reject_count > 95, "Only rejected {}/{} tampered proofs", reject_count, trials);
}

/// Test soundness for zero polynomial edge case.
/// Ensures verifier doesn't accept invalid proofs for dummy polynomials.
#[test]
fn test_zero_poly_soundness() {
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![], &mut transcript); // Empty -> dummy
    let comms = oracle.commit();
    let point = vec![ExtF::ONE];
    let (mut evals, proofs) = oracle.open_at_point(&point);
    
    // Tamper with evaluation
    if !evals.is_empty() {
        evals[0] += ExtF::ONE;
    }
    
    let verifier = FriOracle::new_for_verifier(4);
    assert!(!verifier.verify_openings(&comms, &point, &evals, &proofs),
            "Verifier should reject tampered zero poly evaluation");
}
