// crates/neo-challenge/tests/red_team.rs
use neo_challenge::{StrongSetConfig, sample_kplus1_invertible};
use p3_challenger::{FieldChallenger, CanObserve, CanSample, CanSampleBits};
use p3_goldilocks::Goldilocks as Fq;
use p3_field::PrimeCharacteristicRing;

/// Minimal mock challenger that always returns 0 (for testing degenerate cases)
struct MockChallenger;

impl FieldChallenger<Fq> for MockChallenger {}

impl CanObserve<u8> for MockChallenger {
    fn observe(&mut self, _value: u8) {}
    
    fn observe_slice(&mut self, _values: &[u8]) {}
}

impl CanObserve<Fq> for MockChallenger {
    fn observe(&mut self, _value: Fq) {}
    
    fn observe_slice(&mut self, _values: &[Fq]) {}
}

impl CanSample<Fq> for MockChallenger {
    fn sample(&mut self) -> Fq {
        Fq::ZERO // Always return 0
    }
}

impl CanSampleBits<usize> for MockChallenger {
    fn sample_bits(&mut self, _bits: usize) -> usize {
        0 // Always return 0 to force degenerate coefficients
    }
}

#[test]
fn sampler_rejects_non_invertible_batch() {
    let cfg = StrongSetConfig {
        eta: 81, d: 54,
        coeff_bound: 0, // forces all coeffs to 0 → equal ρ
        domain_sep: b"neo.challenge.redteam",
        max_resamples: 3,
    };
    let mut ch = MockChallenger;
    let err = sample_kplus1_invertible(&mut ch, &cfg, 2).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("NonInvertible"), "must fail when pairwise differences are not invertible");
}