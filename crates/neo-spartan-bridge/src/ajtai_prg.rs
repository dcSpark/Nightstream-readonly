use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

/// Delegate to the canonical Ajtai PRG expander in neo-ajtai.
pub fn expand_row_from_seed(seed32: [u8; 32], row_idx: u32, row_len: usize) -> Vec<Goldilocks> {
    // Use PRG v2 (includes row length in domain) for stronger domain separation
    let v = neo_ajtai::prg::expand_row_v2(&seed32, row_idx as u64, row_len);
    v.into_iter().map(|f| Goldilocks::from_u64(f.as_canonical_u64())).collect()
}
