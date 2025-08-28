//! Smoke tests for the complete p3-FRI bridge
//!
//! Tests the full pipeline: SpartanBridge with p3-FRI PCS placeholder

use neo_spartan_bridge::{SpartanBridge, FriConfig};
use p3_goldilocks::Goldilocks as F;
use p3_field::{integers::QuotientMap, PrimeCharacteristicRing};

#[test]
fn p3fri_poseidon2_smoke() {
    // Minimal FRI config (tune these params later for security/size/speed).
    let cfg = FriConfig {
        log_blowup: 1,
        log_final_poly_len: 0,
        num_queries: 30,
        proof_of_work_bits: 8,
    };
    let bridge = SpartanBridge::new(cfg);

    // A single polynomial eval table over F, length must be two-adic.
    // Here: P(x) = 7 (constant), so evaluations are constant.
    let n = 1 << 8;
    let poly = vec![F::from_canonical_checked(7).unwrap(); n];

    // One opening point from the canonical coset (e.g., x = 0).
    let point = F::ZERO;
    
    println!("Testing p3-FRI bridge with polynomial of length {}", n);
    let (com, prf) = bridge.compress_me(&[poly], &[point]).expect("compress_me should work");
    
    println!("Generated commitment with {} bytes", com.inner.len());
    println!("Generated proof of size {} bytes", prf.bytes.len());
    
    bridge.verify_me(&com, &[point], &prf).expect("verify_me should work");
    
    println!("✅ P3-FRI Poseidon2 smoke test: PASS");
}

#[test]
fn bridge_default_config() {
    let bridge = SpartanBridge::with_default_config();
    
    // Tiny test polynomial
    let n = 1 << 4; // 16 elements
    let poly = vec![F::from_canonical_checked(42).unwrap(); n];
    
    let points = vec![F::ZERO, F::ONE];
    
    println!("Testing bridge with default config...");
    let (commitments, proof) = bridge.compress_me(&[poly], &points).expect("compress with default config");
    
    bridge.verify_me(&commitments, &points, &proof).expect("verify with default config");
    
    println!("✅ Default config test: PASS");
    println!("   Proof size: {} bytes", proof.bytes.len());
}

#[test]
fn bridge_multiple_polynomials() {
    let bridge = SpartanBridge::with_default_config();
    
    let n = 1 << 3; // 8 elements each
    
    // Multiple test polynomials with different patterns
    let polys = vec![
        vec![F::from_canonical_checked(1).unwrap(); n], // constant 1
        vec![F::from_canonical_checked(2).unwrap(); n], // constant 2  
        (0..n).map(|i| F::from_canonical_checked(i as u64).unwrap()).collect(), // linear
    ];
    
    let points = vec![F::ZERO];
    
    println!("Testing bridge with {} polynomials...", polys.len());
    let (commitments, proof) = bridge.compress_me(&polys, &points).expect("multiple poly compress");
    
    bridge.verify_me(&commitments, &points, &proof).expect("multiple poly verify");
    
    println!("✅ Multiple polynomials test: PASS"); 
    println!("   Committed to {} polynomials", polys.len());
    println!("   Proof size: {} bytes", proof.bytes.len());
}
