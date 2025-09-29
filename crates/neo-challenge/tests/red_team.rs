// crates/neo-challenge/tests/red_team.rs
use neo_challenge::{StrongSetConfig, sample_kplus1_invertible};
use neo_transcript::Transcript;
use p3_field::PrimeCharacteristicRing;
use neo_math::F;

/// Minimal mock transcript that always yields zeros.
#[derive(Clone, Default)]
struct MockTranscript;

impl Transcript for MockTranscript {
    fn new(_app_label: &'static [u8]) -> Self { Self }
    fn append_message(&mut self, _label: &'static [u8], _msg: &[u8]) {}
    fn append_fields(&mut self, _label: &'static [u8], _fs: &[F]) {}
    fn challenge_bytes(&mut self, _label: &'static [u8], out: &mut [u8]) { for b in out { *b = 0; } }
    fn challenge_field(&mut self, _label: &'static [u8]) -> F { F::ZERO }
    fn fork(&self, _scope: &'static [u8]) -> Self { Self }
    fn digest32(&mut self) -> [u8; 32] { [0u8; 32] }
}

#[test]
fn sampler_rejects_non_invertible_batch() {
    let cfg = StrongSetConfig {
        eta: 81, d: 54,
        coeff_bound: 0, // forces all coeffs to 0 → equal ρ
        domain_sep: b"neo.challenge.redteam",
        max_resamples: 3,
    };
    let mut tr = MockTranscript::new(b"redteam");
    let err = sample_kplus1_invertible(&mut tr, &cfg, 2).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("NonInvertible"), "must fail when pairwise differences are not invertible");
}
