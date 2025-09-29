// SPDX-License-Identifier: Apache-2.0
// crates/neo-spartan-bridge/src/rlc_guard.rs

//! Lightweight RLC guard helpers matching Poseidon2 transcript framing.
//!
//! This module derives per-coordinate coefficients from public data using
//! Poseidon2 (Goldilocks) and aggregates Ajtai PRG rows accordingly to create
//! a single inner-product guard: <G, z> = sum_i coeff[i] * c_step_coords[i].
//!
//! Inputs used:
//! - seed32: 32-byte public seed (use header_digest)
//! - c_step_coords: published step commitment coordinates
//!
//! Domain tag matches host intent: b"NEO_RLC_V1". While the host may include
//! additional transcript items (e.g., step digest, accumulator), this variant
//! uses the header digest and the c_step coordinates to keep the circuit light.

use p3_field::{PrimeField64, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks as F;
use neo_ccs::crypto::poseidon2_goldilocks as p2;

const RLC_TAG: &[u8] = b"NEO_RLC_V1";

/// Derive `n` coefficients from (seed, c_step_coords) using Poseidon2 packed hashing.
/// Matches the style of neo::ivc::generate_rlc_coefficients with simplified inputs.
pub fn derive_rlc_coefficients(seed32: [u8; 32], c_step_coords: &[neo_math::F], n: usize) -> Vec<F> {
    // Build packed bytes: tag || seed32 limbs || c_step limbs
    let mut limbs: Vec<u64> = Vec::with_capacity((RLC_TAG.len() + 32) / 8 + c_step_coords.len());
    // tag as u64 limbs LE
    for chunk in RLC_TAG.chunks(8) {
        let mut b = [0u8; 8];
        b[..chunk.len()].copy_from_slice(chunk);
        limbs.push(u64::from_le_bytes(b));
    }
    // seed bytes as u64 limbs LE
    for chunk in seed32.chunks(8) {
        let mut b = [0u8; 8];
        b[..chunk.len()].copy_from_slice(chunk);
        limbs.push(u64::from_le_bytes(b));
    }
    // c_step_coords as field limbs
    for &c in c_step_coords { limbs.push(c.as_canonical_u64()); }

    // Hash to F^4 seed using Poseidon2 packed-bytes helper
    let packed: Vec<u8> = limbs.iter().flat_map(|&x| x.to_le_bytes()).collect();
    let mut state = p2::poseidon2_hash_packed_bytes(&packed);

    // Iteratively expand n coefficients
    let mut coeffs = Vec::with_capacity(n);
    for i in 0..n {
        let mut input_vec = state.to_vec();
        input_vec.push(F::from_u64(i as u64));
        let packed_in: Vec<u8> = input_vec.iter().flat_map(|x| x.as_canonical_u64().to_le_bytes()).collect();
        state = p2::poseidon2_hash_packed_bytes(&packed_in);
        coeffs.push(F::from_u64(state[0].as_canonical_u64()));
    }
    coeffs
}
