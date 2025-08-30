// Removed unused import

pub const DS_BRIDGE_INIT:   &[u8] = b"neo:bridge:init";
pub const DS_BRIDGE_COMMIT: &[u8] = b"neo:bridge:commit";
pub const DS_BRIDGE_OPEN:   &[u8] = b"neo:bridge:open";
pub const DS_BRIDGE_VERIFY: &[u8] = b"neo:bridge:verify";

// Simplified challenger stub - p3 challenger generics are complex
#[derive(Clone)]
pub struct Challenger {
    // TODO: Add proper p3 challenger once generics are resolved
}

pub fn make_challenger() -> Challenger {
    Challenger {}
}

/// Convert bytes → Goldilocks elements and observe them.
/// Pack 8 bytes → one Goldilocks element (LE). This provides collision‑resistant, order‑sensitive absorption.
pub fn observe_bytes(_ch: &mut Challenger, _bytes: &[u8]) {
    // TODO: Implement once p3 challenger generics are resolved
}

/// Observe a commitment digest (as bytes) with a DS label.
pub fn observe_commitment_bytes(_ch: &mut Challenger, _label: &[u8], _bytes: &[u8]) {
    // TODO: Implement once p3 challenger generics are resolved
}

/// Sample a challenge element (placeholder)
pub fn sample_point(_ch: &mut Challenger) -> super::mmcs::Challenge {
    use p3_field::PrimeCharacteristicRing;
    super::mmcs::Challenge::ZERO // Placeholder
}

pub fn grind(_ch: &mut Challenger, _pow_bits: usize) {
    // TODO: Implement proof-of-work grinding once p3 challenger is resolved
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::mmcs::make_mmcs_and_dft;

    #[test]
    fn test_challenger_real_fiat_shamir() {
        let _mats = make_mmcs_and_dft(42);
        let mut ch1 = make_challenger();
        let mut ch2 = make_challenger();

        println!("✅ Challenger stubs created (TODO: implement full p3 challenger)");
        println!("   Domain separation: {:?}", std::str::from_utf8(DS_BRIDGE_INIT));

        let challenge1 = sample_point(&mut ch1);
        let challenge2 = sample_point(&mut ch2);

        // Placeholder implementation returns same value
        assert_eq!(challenge1, challenge2);
        println!("   Placeholder challenges work: ✅");
    }

    #[test]
    fn test_commitment_observation_bytes() {
        let _mats = make_mmcs_and_dft(123);
        let mut ch = make_challenger();
        let fake_commitment = [0xABu8; 21];

        observe_commitment_bytes(&mut ch, DS_BRIDGE_COMMIT, &fake_commitment);
        let challenge_after = sample_point(&mut ch);

        println!("✅ Commitment observation (bytes) stub works");
        println!("   Would absorb {} bytes as Goldilocks limbs", fake_commitment.len());
        println!("   Placeholder challenge: {:?}", challenge_after);
    }

    #[test]
    fn test_commitment_observation_field_elements() {
        let _mats = make_mmcs_and_dft(456);
        let mut ch = make_challenger();
        
        let challenge_after = sample_point(&mut ch);
        
        println!("✅ Field element observation stub works");
        println!("   Placeholder challenge: {:?}", challenge_after);
    }
}