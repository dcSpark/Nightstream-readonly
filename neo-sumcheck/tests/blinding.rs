#[cfg(test)]
mod tests {
    use neo_fields::ExtF;
    use neo_poly::Polynomial;
    use neo_sumcheck::oracle::FriOracle;
    use neo_sumcheck::PolyOracle;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_blinding_consistency() {
        let poly = Polynomial::new(vec![ExtF::ONE, ExtF::from_u64(2)]);
        let mut transcript = vec![];
        let mut oracle = FriOracle::new(vec![poly.clone()], &mut transcript);

        let point = vec![ExtF::ONE];
        let (evals, proofs) = oracle.open_at_point(&point);
        let commit = oracle.commit();

        // Check eval includes blind
        let expected_blinded = poly.eval(point[0]) + oracle.blinds[0];
        assert_eq!(evals[0], expected_blinded, "Eval should include blind");

        // Verify using the blinded evaluation (as the proof was generated for the blinded value)
        // Match prover domain size: (degree + 1).next_power_of_two() * 4
        let domain_size = (poly.degree() + 1).next_power_of_two() * 4;
        let verifier = FriOracle::new_for_verifier(domain_size);
        
        // Use blinded eval for verification (proof is for blinded)
        let blinded_eval = evals[0];
        let verify_result = verifier.verify_openings(&commit, &point, &[blinded_eval], &proofs);
        assert!(verify_result, "Verification should pass with blinded eval");

        // Separate check for unblinded value
        let unblinded = blinded_eval - oracle.blinds[0];
        let expected = poly.eval(point[0]);
        assert_eq!(unblinded, expected, "Unblinded eval should match poly.eval");

        // Mismatch case: tamper the blinded evaluation
        let bad_blinded = evals[0] + ExtF::ONE;
        let bad_verify = verifier.verify_openings(&commit, &point, &[bad_blinded], &proofs);
        assert!(!bad_verify, "Verification should fail on tampered evaluation");
    }

    #[test]
    fn test_blinding_in_dummy_case() {
        let mut transcript = vec![];
        let mut oracle = FriOracle::new(vec![], &mut transcript); // Empty -> dummy

        let point = vec![ExtF::ONE];
        let (evals, proofs) = oracle.open_at_point(&point);
        let commit = oracle.commit();

        // Dummy should use non-zero poly + blind
        // Note: codewords field is private, but we can test that the oracle works
        assert!(!commit.is_empty(), "Dummy commit should be generated");
        
        // For dummy case, polynomial is [ExtF::ONE] with degree 0
        let dummy_poly = Polynomial::new(vec![ExtF::ONE]); // Match your dummy
        let expected_blinded = dummy_poly.eval(point[0]) + oracle.blinds[0];
        assert_eq!(evals[0], expected_blinded, "Dummy eval should include blind");

        // Verify using the blinded evaluation (as the proof was generated for the blinded value)
        let domain_size = (dummy_poly.degree() + 1).next_power_of_two() * 4;
        let verifier = FriOracle::new_for_verifier(domain_size);
        
        // Use blinded eval for verification (proof is for blinded)
        let blinded_eval = evals[0];
        let verify_result = verifier.verify_openings(&commit, &point, &[blinded_eval], &proofs);
        assert!(verify_result, "Dummy verification should pass with blinded eval");

        // Separate check for unblinded
        let unblinded = blinded_eval - oracle.blinds[0];
        let expected = dummy_poly.eval(point[0]);
        assert_eq!(unblinded, expected, "Unblinded should match dummy eval");
    }
}