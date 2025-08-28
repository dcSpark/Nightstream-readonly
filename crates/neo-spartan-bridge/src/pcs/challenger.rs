use crate::pcs::mmcs::Perm;

/// Domain separation labels for the bridge → PCS handshake.
pub(crate) const DS_BRIDGE_INIT: &[u8] = b"neo:bridge:init";
#[allow(dead_code)] // Will be used when real commitment observation is implemented
pub(crate) const DS_BRIDGE_COMMIT: &[u8] = b"neo:bridge:commit";

/// TODO: Implement real challenger once p3-challenger API is stable
/// Build a challenger from a Poseidon2 permutation, then "absorb" the bridge transcript preimage.
pub fn challenger_from_io_placeholder(_perm: &Perm, io_preimage: &[u8]) -> Vec<u8> {
    // Minimal domain-separated placeholder: DS || io_preimage.
    // This keeps the constants used and avoids moving the permutation.
    let mut out = Vec::with_capacity(DS_BRIDGE_INIT.len() + io_preimage.len());
    out.extend_from_slice(DS_BRIDGE_INIT);
    out.extend_from_slice(io_preimage);
    out
}

/// TODO: Implement real commitment observation once p3-challenger API is stable
#[allow(dead_code)]
pub fn observe_commitment_placeholder<C>(_: &C) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::mmcs::make_mmcs_and_dft;

    #[test]
    fn test_challenger_placeholder() {
        let mats = make_mmcs_and_dft();
        
        let test_io = b"test_io_data_for_transcript";
        let challenger_state = challenger_from_io_placeholder(&mats.perm, test_io);
        
        println!("✅ Challenger placeholder created from transcript IO");
        println!("   Domain separation: {:?}", std::str::from_utf8(DS_BRIDGE_INIT));
        println!("   IO data: {} bytes", test_io.len());
        println!("   Challenger state: {} bytes", challenger_state.len());
        
        // Test determinism - same input should give same result
        let challenger_state2 = challenger_from_io_placeholder(&mats.perm, test_io);
        assert_eq!(challenger_state, challenger_state2);
    }
}