use neo_fold::transcript::{FoldTranscript, Domain};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

#[test]
fn test_transcript_basic() {
    let mut tr = FoldTranscript::new(b"test");
    
    // Test domain separation
    tr.domain(Domain::CCS);
    
    // Test absorbing different types
    tr.absorb_u64(&[1, 2, 3]);
    tr.absorb_f(&[F::from_u32(42)]);
    tr.absorb_bytes(b"test_data");
    
    // Test challenge generation
    let f_challenges = tr.challenges_f(3);
    assert_eq!(f_challenges.len(), 3);
    
    let k_challenge = tr.challenge_k();
    println!("Extension challenge: {:?}", k_challenge);
    
    println!("Transcript test completed successfully");
}

#[test]
fn test_domain_separation() {
    let mut tr1 = FoldTranscript::new(b"test");
    let mut tr2 = FoldTranscript::new(b"test");
    
    // Same input, different domains
    tr1.domain(Domain::CCS);
    tr2.domain(Domain::Rlc);
    
    tr1.absorb_u64(&[1, 2, 3]);
    tr2.absorb_u64(&[1, 2, 3]);
    
    let c1 = tr1.challenge_f();
    let c2 = tr2.challenge_f();
    
    // Should be different due to domain separation
    assert_ne!(c1, c2);
    println!("Domain separation working: {} != {}", c1.as_canonical_u64(), c2.as_canonical_u64());
}
