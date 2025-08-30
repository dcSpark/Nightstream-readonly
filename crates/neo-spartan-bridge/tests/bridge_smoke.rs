#![allow(deprecated)] // Tests use legacy MEInstance/MEWitness for backward compatibility

//! Smoke tests for the neo-spartan-bridge with Hash-MLE PCS (Poseidon2-only)
//!
//! These tests verify the complete architectural foundation and API structure
//! for ME(b,L) -> Spartan2 + Hash-MLE compression pipeline.

use neo_spartan_bridge::compress_me_to_spartan;
use neo_ccs::{MEInstance, MEWitness};
use p3_goldilocks::Goldilocks as F;
use p3_field::{PrimeCharacteristicRing, integers::QuotientMap};

/// Create a minimal ME instance for testing using the actual neo-ccs structure
fn tiny_me_instance() -> (MEInstance, MEWitness) {
    // Create a minimal ME instance using the actual field structure
    let me = MEInstance {
        c_coords: vec![F::from_canonical_checked(1).unwrap(); 4], // Ajtai commitment coordinates
        y_outputs: vec![F::from_canonical_checked(2).unwrap(); 4], // public outputs  
        r_point: vec![F::from_canonical_checked(3).unwrap(); 2], // challenge point
        base_b: 4, // Base dimension
        header_digest: [0u8; 32], // Header hash
    };
    
    // Witness with matching structure
    let wit = MEWitness {
        z_digits: vec![1i64, 2i64, 3i64, 0i64, -1i64, 1i64, 0i64, 2i64], // witness digits (base-b)
        weight_vectors: vec![vec![F::ONE; 4], vec![F::ZERO; 4]], // weight matrices
        ajtai_rows: Some(vec![vec![F::from_canonical_checked(7).unwrap(); 4]; 2]), // Ajtai matrix rows
    };
    
    (me, wit)
}

#[test]
fn bridge_smoke_me_hash_mle() {
    println!("🧪 Testing complete ME(b,L) -> Spartan2 + Hash-MLE pipeline");
    
    let (me, wit) = tiny_me_instance();
    
    println!("   ME coordinates: {}", me.c_coords.len());
    println!("   Output values: {}", me.y_outputs.len());
    println!("   Challenge point dimension: {}", me.r_point.len());
    
    // Compress using Hash-MLE PCS (no FRI parameters needed)
    let proof = match compress_me_to_spartan(&me, &wit) {
        Ok(proof) => proof,
        Err(e) => {
            println!("🚨 Spartan2 API diagnostic:");
            println!("{:?}", e);
            println!("Full error chain:");
            let mut current_error: &dyn std::error::Error = &*e;
            loop {
                println!("  {}", current_error);
                match current_error.source() {
                    Some(source) => current_error = source,
                    None => break,
                }
            }
            panic!("compress_me_to_spartan failed - see diagnostic above");
        }
    };

    println!("   Total proof size: {} bytes", proof.total_size());
    
    // Verify proof structure
    assert!(!proof.proof.is_empty(), "Proof bytes should not be empty");
    assert!(!proof.vk.is_empty(), "Verifier key should not be empty");
    assert!(!proof.public_io_bytes.is_empty(), "Public IO should not be empty");
    
    // Real verification
    let ok = neo_spartan_bridge::verify_me_spartan(&proof).expect("verify should run");
    assert!(ok);
    
    println!("✅ ME(b,L) -> Spartan2 compression succeeded");
}

#[test]
fn determinism_check() {
    println!("🧪 Testing real SNARK proof generation");
    
    let (me, wit) = tiny_me_instance();
    
    // Generate two proofs with identical inputs – proofs are randomized
    let proof1 = compress_me_to_spartan(&me, &wit).expect("first proof");
    let proof2 = compress_me_to_spartan(&me, &wit).expect("second proof");
    
    // Real SNARKs are randomized: proofs usually differ, VKs match, IO matches
    assert_ne!(proof1.proof, proof2.proof, "Proofs need not be bit-equal");
    assert_eq!(proof1.vk, proof2.vk, "VK should be deterministic");
    assert_eq!(proof1.public_io_bytes, proof2.public_io_bytes, "Public IO should be stable");
    assert!(neo_spartan_bridge::verify_me_spartan(&proof1).unwrap());
    assert!(neo_spartan_bridge::verify_me_spartan(&proof2).unwrap());
    
    println!("✅ Real SNARK proof generation and verification succeeded");
}

#[test]
fn header_binding_consistency() {
    println!("🧪 Testing header digest binding");
    
    let (mut me, wit) = tiny_me_instance();
    
    // Generate proof with original header
    let proof1 = compress_me_to_spartan(&me, &wit).expect("original proof");
    
    // Change header digest
    me.header_digest[0] ^= 1; // flip one bit
    let proof2 = compress_me_to_spartan(&me, &wit).expect("modified proof");
    
    // Header digest change should affect public IO
    assert_ne!(
        proof1.public_io_bytes, 
        proof2.public_io_bytes,
        "Header digest must bind transcript"
    );
    
    println!("✅ Header digest properly binds to transcript");
}

#[test]
fn proof_bundle_structure() {
    println!("🧪 Testing ProofBundle structure");
    
    let (me, wit) = tiny_me_instance();
    let bundle = compress_me_to_spartan(&me, &wit).expect("proof bundle");
    
    // Verify bundle contains expected components
    assert!(bundle.proof.len() > 0, "Proof component should have content");
    assert!(bundle.vk.len() > 0, "Verifier key should have content");
    assert!(bundle.public_io_bytes.len() > 0, "Public IO should have content");
    
    // Verify size calculation
    let expected_size = bundle.proof.len() + bundle.vk.len() + bundle.public_io_bytes.len();
    assert_eq!(bundle.total_size(), expected_size, "Size calculation should be correct");
    
    println!("   Proof size: {} bytes", bundle.proof.len());
    println!("   VK size: {} bytes", bundle.vk.len());
    println!("   Public IO size: {} bytes", bundle.public_io_bytes.len());
    println!("✅ ProofBundle structure is valid");
}