use blake2b_simd::Params as Blake2bParams;

/// Versioned domain separator for step-bundle binding.
pub const BUNDLE_DIGEST_V1_DOMAIN: &[u8] = b"neo/midnight-bridge/bundle-digest/v1";
pub const BUNDLE_DIGEST_V2_DOMAIN: &[u8] = b"neo/midnight-bridge/bundle-digest/v2";

/// Convert a 32-byte digest into 4 little-endian `u64` limbs.
pub fn digest32_to_u64_limbs_le(digest: [u8; 32]) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    for (i, chunk) in digest.chunks_exact(8).enumerate() {
        limbs[i] = u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
    }
    limbs
}

/// Convert a 32-byte digest into 2 little-endian `u128` limbs.
///
/// This is a compact encoding for Midnight public inputs: a BLS12-381 scalar element can
/// represent any 128-bit integer injectively, so we can carry 32 bytes using 2 field elements
/// instead of 4 `u64` limbs.
pub fn digest32_to_u128_limbs_le(digest: [u8; 32]) -> [u128; 2] {
    let mut limbs = [0u128; 2];
    let (lo, hi) = digest.split_at(16);
    limbs[0] = u128::from_le_bytes(lo.try_into().expect("16 bytes"));
    limbs[1] = u128::from_le_bytes(hi.try_into().expect("16 bytes"));
    limbs
}

/// Blake2b-256 helper with a fixed domain separator.
pub fn blake2b_256_domain(domain: &[u8], msg: &[u8]) -> [u8; 32] {
    // NOTE: blake2b "personalization" is exactly 16 bytes, so we domain-separate
    // by prefixing the message instead of using `.personal(...)`.
    let mut full = Vec::with_capacity(domain.len() + msg.len());
    full.extend_from_slice(domain);
    full.extend_from_slice(msg);
    let hash = Blake2bParams::new().hash_length(32).hash(&full);
    hash.as_bytes().try_into().expect("32 bytes")
}

/// Compute a step-bundle binding digest.
///
/// This is meant as a *public identifier* shared across all proofs in a bundle, so
/// they cannot be mixed-and-matched across steps/runs when verified together.
///
/// Current convention (v1):
/// `H = Blake2b-256( BUNDLE_DIGEST_V1_DOMAIN || step_idx_le || header_digest32 )`
pub fn compute_step_bundle_digest_v1(step_idx: u32, header_digest: [u8; 32]) -> [u8; 32] {
    let mut msg = Vec::with_capacity(4 + 32);
    msg.extend_from_slice(&step_idx.to_le_bytes());
    msg.extend_from_slice(&header_digest);
    blake2b_256_domain(BUNDLE_DIGEST_V1_DOMAIN, &msg)
}

/// Compute a step-bundle binding digest from explicit statement digests.
///
/// This is the intended production shape for binding a bundle to:
/// - a fixed Neo parameter/config digest,
/// - a fixed CCS digest,
/// - an initial accumulator digest,
/// - a final accumulator digest,
/// - and the step index.
///
/// `H = Blake2b-256( BUNDLE_DIGEST_V2_DOMAIN || step_idx_le || params_digest || ccs_digest || initial_acc_digest || final_acc_digest )`
pub fn compute_step_bundle_digest_v2(
    step_idx: u32,
    params_digest: [u8; 32],
    ccs_digest: [u8; 32],
    initial_acc_digest: [u8; 32],
    final_acc_digest: [u8; 32],
) -> [u8; 32] {
    let mut msg = Vec::with_capacity(4 + 32 * 4);
    msg.extend_from_slice(&step_idx.to_le_bytes());
    msg.extend_from_slice(&params_digest);
    msg.extend_from_slice(&ccs_digest);
    msg.extend_from_slice(&initial_acc_digest);
    msg.extend_from_slice(&final_acc_digest);
    blake2b_256_domain(BUNDLE_DIGEST_V2_DOMAIN, &msg)
}
