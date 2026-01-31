//! Implicit Shout tables with closed-form MLE evaluation.
//!
//! This module contains helpers for "virtual" / implicit lookup tables that are not materialized
//! as dense vectors (e.g. `table[x] = x` on a `2^32` domain).

pub mod shout_oracle;

use neo_math::K;
use p3_field::PrimeCharacteristicRing;

/// Evaluate the multilinear extension of the identity table `table[a] = a` (little-endian bits).
///
/// For `r_addr = (r_0, ..., r_{n-1})`, this returns:
/// `Σ_i 2^i · r_i`.
pub fn eval_identity_mle_le(r_addr: &[K]) -> K {
    let mut acc = K::ZERO;
    let mut coeff = K::ONE; // 2^0
    for &ri in r_addr {
        acc += coeff * ri;
        coeff = coeff + coeff; // *= 2
    }
    acc
}
