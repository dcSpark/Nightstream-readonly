#![cfg(feature = "real_fri")]
use neo_commit::fri_pcs::{RealFriPCSWrapper, RealFriParams};
use neo_fields::{F, ExtF};
use p3_field::PrimeCharacteristicRing;

#[test]
fn fri_commit_open_verify_goldilocks() {
    // f(x) = 1 + 2x + 3x^2 over Goldilocks
    let coeffs = [F::from_u64(1), F::from_u64(2), F::from_u64(3)];

    // evaluate on a two-adic domain: size 8 (log_domain=3)
    let log_domain = 3usize;
    let n = 1usize << log_domain;
    let mut evals = vec![F::ZERO; n];
    for i in 0..n {
        let x = F::from_u64(i as u64);
        let mut pow = F::ONE;
        let mut acc = F::ZERO;
        for &c in &coeffs {
            acc += c * pow;
            pow *= x;
        }
        evals[i] = acc;
    }

    // Real PCS
    let pcs = RealFriPCSWrapper::with_params(RealFriParams { log_blowup: 2, num_queries: 64 });
    let (com, pd) = pcs.commit(&[evals], log_domain, None).expect("commit");

    // open at x = 5 in extension (real part only)
    let x = ExtF::new_real(F::from_u64(5));
    let y = {
        let mut pow = F::ONE; let mut acc = F::ZERO;
        for &c in &coeffs { acc += c * pow; pow *= F::from_u64(5); }
        ExtF::new_real(acc)
    };

    let pr = pcs.open(&com, &pd, 0, x).expect("open");
    assert_eq!(y, pr.evaluation);

    let ok = pcs.verify(&com, 0, x, y, &pr).expect("verify");
    assert!(ok, "FRI opening should verify");
}

#[test]
fn fri_simulated_fallback() {
    // Test that simulated version works when real_fri feature is disabled
    // This test will only run when real_fri is enabled, but tests the interface
    let pcs = RealFriPCSWrapper::new();
    
    // Simple evaluation vector for domain size 4 (log_domain=2)
    let log_domain = 2usize;
    let evals = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3), F::from_u64(4)];
    
    let (com, pd) = pcs.commit(&[evals], log_domain, None).expect("commit should work");
    
    // Test opening at a point
    let x = ExtF::new_real(F::from_u64(1));
    let pr = pcs.open(&com, &pd, 0, x).expect("open should work");
    
    // Verify the opening
    let ok = pcs.verify(&com, 0, x, pr.evaluation, &pr).expect("verify should work");
    assert!(ok, "Opening should verify");
    
    // Test size estimates
    assert!(pcs.proof_size_estimate(log_domain) > 0);
    assert!(pcs.commitment_size() > 0);
}
