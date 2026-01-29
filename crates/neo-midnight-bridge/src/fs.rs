use crate::goldilocks::GOLDILOCKS_P_U64;
use crate::k_field::KRepr;
use crate::statement::blake2b_256_domain;

pub const FS_DOMAIN: &[u8] = b"neo/midnight-bridge/fs/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FsChannel {
    Fe = 0,
    Nc = 1,
}

fn reduce_to_gl(u: u64) -> u64 {
    // Since p is close to 2^64, this is either u or u - p.
    if u >= GOLDILOCKS_P_U64 {
        u - GOLDILOCKS_P_U64
    } else {
        u
    }
}

fn hash_to_k(digest32: [u8; 32]) -> KRepr {
    let c0 = reduce_to_gl(u64::from_le_bytes(digest32[0..8].try_into().expect("8 bytes")));
    let c1 = reduce_to_gl(u64::from_le_bytes(digest32[8..16].try_into().expect("8 bytes")));
    KRepr { c0, c1 }
}

/// Encode `KRepr` as 16 little-endian bytes.
fn push_krepr_le(buf: &mut Vec<u8>, x: KRepr) {
    buf.extend_from_slice(&x.c0.to_le_bytes());
    buf.extend_from_slice(&x.c1.to_le_bytes());
}

fn push_krepr_vec_le(buf: &mut Vec<u8>, xs: &[KRepr]) {
    // Length framing avoids ambiguity in concatenations.
    let len_u32: u32 = xs
        .len()
        .try_into()
        .expect("vector too large for u32 length");
    buf.extend_from_slice(&len_u32.to_le_bytes());
    for &x in xs {
        push_krepr_le(buf, x);
    }
}

fn derive_seed_channel(
    bundle_digest32: [u8; 32],
    channel: FsChannel,
    label: &[u8],
    sumcheck_challenges: &[KRepr],
) -> [u8; 32] {
    let mut msg = Vec::with_capacity(32 + 1 + label.len() + 4 + 16 * sumcheck_challenges.len());
    msg.extend_from_slice(&bundle_digest32);
    msg.push(channel as u8);
    msg.extend_from_slice(label);
    push_krepr_vec_le(&mut msg, sumcheck_challenges);
    blake2b_256_domain(FS_DOMAIN, &msg)
}

fn derive_seed_shared(
    bundle_digest32: [u8; 32],
    label: &[u8],
    fe_sumcheck_challenges: &[KRepr],
    nc_sumcheck_challenges: &[KRepr],
) -> [u8; 32] {
    let mut msg = Vec::with_capacity(
        32 + label.len() + 4 + 16 * (fe_sumcheck_challenges.len() + nc_sumcheck_challenges.len()) + 8,
    );
    msg.extend_from_slice(&bundle_digest32);
    msg.extend_from_slice(label);
    push_krepr_vec_le(&mut msg, fe_sumcheck_challenges);
    push_krepr_vec_le(&mut msg, nc_sumcheck_challenges);
    blake2b_256_domain(FS_DOMAIN, &msg)
}

fn derive_k_from_seed(seed32: [u8; 32], label: &[u8], index: u32) -> KRepr {
    let mut msg = Vec::with_capacity(32 + label.len() + 4);
    msg.extend_from_slice(&seed32);
    msg.extend_from_slice(label);
    msg.extend_from_slice(&index.to_le_bytes());
    hash_to_k(blake2b_256_domain(FS_DOMAIN, &msg))
}

/// Derive the i-th sumcheck challenge from:
/// - `bundle_digest32`
/// - `channel`
/// - `round_index`
/// - `round_poly_coeffs`
pub fn derive_sumcheck_round_challenge(
    bundle_digest32: [u8; 32],
    channel: FsChannel,
    round_index: u32,
    coeffs: &[KRepr],
) -> KRepr {
    let mut msg = Vec::with_capacity(32 + 1 + 32 + 4 + 16 * coeffs.len());
    msg.extend_from_slice(&bundle_digest32);
    msg.push(channel as u8);
    msg.extend_from_slice(b"sumcheck_round");
    msg.extend_from_slice(&round_index.to_le_bytes());
    for &c in coeffs {
        push_krepr_le(&mut msg, c);
    }
    hash_to_k(blake2b_256_domain(FS_DOMAIN, &msg))
}

pub fn derive_all_sumcheck_challenges(
    bundle_digest32: [u8; 32],
    channel: FsChannel,
    rounds: &[Vec<KRepr>],
) -> Vec<KRepr> {
    rounds
        .iter()
        .enumerate()
        .map(|(i, coeffs)| derive_sumcheck_round_challenge(bundle_digest32, channel, i as u32, coeffs))
        .collect()
}

/// Derive a channel-scoped `gamma` from `(bundle_digest, channel, sumcheck_challenges)`.
pub fn derive_gamma_channel(bundle_digest32: [u8; 32], channel: FsChannel, sumcheck_challenges: &[KRepr]) -> KRepr {
    let seed = derive_seed_channel(bundle_digest32, channel, b"gamma_seed", sumcheck_challenges);
    derive_k_from_seed(seed, b"gamma", 0)
}

/// Derive a *shared* `gamma` for the step-bundle, bound to both sumcheck channels.
pub fn derive_gamma_shared(
    bundle_digest32: [u8; 32],
    fe_sumcheck_challenges: &[KRepr],
    nc_sumcheck_challenges: &[KRepr],
) -> KRepr {
    let seed = derive_seed_shared(
        bundle_digest32,
        b"gamma_shared_seed",
        fe_sumcheck_challenges,
        nc_sumcheck_challenges,
    );
    derive_k_from_seed(seed, b"gamma_shared", 0)
}

/// Derive `alpha ∈ K^{ell_d}` (shared) from the bundle digest and both sumcheck channels.
pub fn derive_alpha_shared(
    bundle_digest32: [u8; 32],
    fe_sumcheck_challenges: &[KRepr],
    nc_sumcheck_challenges: &[KRepr],
    ell_d: usize,
) -> Vec<KRepr> {
    let seed = derive_seed_shared(
        bundle_digest32,
        b"alpha_shared_seed",
        fe_sumcheck_challenges,
        nc_sumcheck_challenges,
    );
    (0..ell_d)
        .map(|i| derive_k_from_seed(seed, b"alpha_shared", i as u32))
        .collect()
}

/// Derive `beta_a ∈ K^{ell_d}` (shared) from the bundle digest and both sumcheck channels.
pub fn derive_beta_a_shared(
    bundle_digest32: [u8; 32],
    fe_sumcheck_challenges: &[KRepr],
    nc_sumcheck_challenges: &[KRepr],
    ell_d: usize,
) -> Vec<KRepr> {
    let seed = derive_seed_shared(
        bundle_digest32,
        b"beta_a_shared_seed",
        fe_sumcheck_challenges,
        nc_sumcheck_challenges,
    );
    (0..ell_d)
        .map(|i| derive_k_from_seed(seed, b"beta_a_shared", i as u32))
        .collect()
}

/// Derive `beta_r ∈ K^{ell_n}` (FE-only) from the bundle digest and FE sumcheck challenges.
pub fn derive_beta_r_fe(bundle_digest32: [u8; 32], fe_sumcheck_challenges: &[KRepr], ell_n: usize) -> Vec<KRepr> {
    let seed = derive_seed_channel(
        bundle_digest32,
        FsChannel::Fe,
        b"beta_r_fe_seed",
        fe_sumcheck_challenges,
    );
    (0..ell_n)
        .map(|i| derive_k_from_seed(seed, b"beta_r_fe", i as u32))
        .collect()
}

/// Derive `beta_m ∈ K^{ell_m}` (NC-only) from the bundle digest and NC sumcheck challenges.
pub fn derive_beta_m_nc(bundle_digest32: [u8; 32], nc_sumcheck_challenges: &[KRepr], ell_m: usize) -> Vec<KRepr> {
    let seed = derive_seed_channel(
        bundle_digest32,
        FsChannel::Nc,
        b"beta_m_nc_seed",
        nc_sumcheck_challenges,
    );
    (0..ell_m)
        .map(|i| derive_k_from_seed(seed, b"beta_m_nc", i as u32))
        .collect()
}
