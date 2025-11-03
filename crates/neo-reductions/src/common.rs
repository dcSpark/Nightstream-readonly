//! Common utilities and helper functions shared across engines.
//!
//! This module contains:
//! - Balanced base-b digit splitting for DEC operations
//! - RLC sampling (diagonal ρ matrices)
//! - ME relation helpers (compute y from Z and r)
//! - Matrix arithmetic helpers
//! - Extension field formatting utilities

#![allow(non_snake_case)]

use neo_ccs::{CcsStructure, Mat};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_math::{F, K, KExtensions, D};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::error::PiCcsError;

// ---------------------------------------------------------------------------
// Balanced Base-b Digit Splitting
// ---------------------------------------------------------------------------

/// Helper: returns (r, q) with r in [-floor(b/2), ..., ceil(b/2)-1].
/// This is proper balanced Euclidean division.
fn balanced_divrem(v: i128, b: i128) -> (i128, i128) {
    // Euclidean remainder in [0, b-1]
    let r0 = ((v % b) + b) % b;
    // Shift to balanced range around zero
    let half = b / 2; // floor(b/2)
    let mut r = r0;
    let mut q = (v - r0) / b;
    if r > half {
        r -= b;
        q += 1;
    }
    (r, q)
}

/// Split Z into **balanced base-b digits** Z = Σ_{i=0}^{k-1} b^i · Z_i, entrywise.
/// Each digit lies in [-floor(b/2), ..., ceil(b/2)-1]. Returns an error if an entry cannot be represented
/// within k digits (i.e., if |value| ≥ b^k) — this indicates a bad RLC sample.
pub fn split_b_matrix_k(Z: &Mat<F>, k: usize, b: u32) -> Result<Vec<Mat<F>>, PiCcsError> {
    let mut outs = (0..k)
        .map(|_| Mat::zero(Z.rows(), Z.cols(), F::ZERO))
        .collect::<Vec<_>>();

    let b_i = b as i128;
    let mut B: i128 = 1;
    for _ in 0..k { B = B.saturating_mul(b_i); } // b^k

    // Helpers to interpret field element as a small signed integer in (-(B-1), B-1)
    let p: u128 = F::ORDER_U64 as u128; // Goldilocks prime fits in u64
    let B_u: u128 = B as u128;

    for r in 0..Z.rows() {
        for c in 0..Z.cols() {
            let u = Z[(r, c)].as_canonical_u64() as u128;
            // Map to a small signed integer if within the DEC budget.
            let val_opt: Option<i128> = if u < B_u {
                Some(u as i128)
            } else if p.checked_sub(u).map(|w| w < B_u).unwrap_or(false) {
                // negative representative
                Some(-((p - u) as i128))
            } else {
                None
            };

            let mut v = match val_opt {
                Some(v) => v,
                None => return Err(PiCcsError::ProtocolError(format!(
                    "DEC split: entry at ({},{}) out of range for k={}, b={}",
                    r, c, k, b
                ))),
            };

            // Balanced digit extraction: r_i ∈ [-floor(b/2), ..., ceil(b/2)-1], v ← q
            for i in 0..k {
                let (r_i, q) = balanced_divrem(v, b_i);
                let digit_f = if r_i >= 0 {
                    F::from_u64(r_i as u64)
                } else {
                    let abs = (-r_i) as u64;
                    F::ZERO - F::from_u64(abs)
                };
                outs[i][(r, c)] = digit_f;
                v = q;
            }
            // Must consume exactly in k digits
            if v != 0 {
                return Err(PiCcsError::ProtocolError(format!(
                    "DEC split: value at ({},{}) needs more than k={} digits", r, c, k
                )));
            }
        }
    }
    Ok(outs)
}

// ---------------------------------------------------------------------------
// RLC Sampling
// ---------------------------------------------------------------------------

/// Sample `k` diagonal ρ_i ∈ S with independent Rademacher signs per Ajtai digit:
///   ρ_i = diag(s_0, ..., s_{D-1}), each s_r ∈ {+1, -1}.
///
/// Properties:
/// - Commutative (all diagonal) subring S.
/// - Expansion factor T = 1 (same as before), so the DEC bound
///   `(k+1) * T * (b-1) < B = b^k`
///   remains exactly the same check as the previous ±I sampler.
/// - The strong sampling set size inflates from 2 to 2^D, aligning much
///   better with the "negligible 1/|C|" requirement in §4.3 of the paper.
///
/// Fiat–Shamir usage:
/// We derive enough random bits from the transcript to assign an
/// independent sign to each diagonal entry (and for each ρ_i).
pub fn sample_diag_rhos(
    tr: &mut Poseidon2Transcript,
    k: usize,
    params: &NeoParams,
) -> Result<Vec<Mat<F>>, PiCcsError> {
    // Enforce (k+1) * T * (b-1) < B = b^k with T = 1 (unchanged).
    let b = params.b as u128;
    let k_u = k as u128;
    let lhs = (k_u + 1) * (b - 1);
    let mut pow = 1u128;
    for _ in 0..k { pow = pow.saturating_mul(b); } // b^k
    let rhs = pow; // B
    if lhs >= rhs {
        return Err(PiCcsError::InvalidInput(format!(
            "RLC bound violated: require (k+1)(b-1) < b^k; got {} < {} is false", lhs, rhs
        )));
    }
    // Helper to expand transcript randomness into `need` sign bits.
    #[inline]
    fn draw_sign_bits(
        tr: &mut Poseidon2Transcript,
        label_prefix: &'static [u8],
        need: usize,
        counter_seed: u64,
    ) -> Vec<F> {
        let mut out = Vec::with_capacity(need);
        let mut remaining = need;
        let mut ctr = counter_seed;
        while remaining > 0 {
            // Domain separate each chunk to get fresh pseudorandomness.
            tr.append_message(label_prefix, &ctr.to_le_bytes());
            let dig = tr.digest32(); // 32 fresh bytes
            for &byte in &dig {
                if remaining == 0 { break; }
                let s = if (byte & 1) == 0 { F::ONE } else { F::ZERO - F::ONE };
                out.push(s);
                remaining -= 1;
            }
            ctr = ctr.wrapping_add(1);
        }
        out
    }

    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        // Tag this ρ_i and derive D independent sign bits.
        tr.append_message(b"rlc/rho/index", &(i as u64).to_le_bytes());
        let signs = draw_sign_bits(tr, b"rlc/rho/chunk", D, i as u64);
        // Build ρ_i = diag(signs[0..D-1]).
        let mut rho = Mat::zero(D, D, F::ZERO);
        for r in 0..D {
            rho[(r, r)] = signs[r];
        }
        out.push(rho);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// ME Relation Helpers
// ---------------------------------------------------------------------------

/// Compute y from Z and r according to the ME relation: y_j := Z · (M_j^T · r^b).
///
/// Returns (y, y_scalars) where:
/// - y[j] is padded to 2^{ell_d} and contains the first D digits
/// - y_scalars[j] = Σ_{d=0}^{D-1} b^d · y[j][d] (base-b recomposition)
pub fn compute_y_from_Z_and_r(
    s: &CcsStructure<F>,
    Z: &Mat<F>,
    r: &[K],
    ell_d: usize,
    b: u32,
) -> (Vec<Vec<K>>, Vec<K>) {
    use neo_ccs::utils::mat_vec_mul_fk;
    let d_pad = 1usize << ell_d;
    let mut y_new: Vec<Vec<K>> = Vec::with_capacity(s.t());
    // Build r^b over rows
    let rb = neo_ccs::utils::tensor_point::<K>(r);
    // v_j = M_j^T · r^b ∈ K^m
    let mut vjs: Vec<Vec<K>> = Vec::with_capacity(s.t());
    for j in 0..s.t() {
        let mut vj = vec![K::ZERO; s.m];
        for row in 0..core::cmp::min(s.n, rb.len()) {
            let wr = rb[row];
            if wr == K::ZERO { continue; }
            let row_m = s.matrices[j].row(row);
            for c in 0..s.m { vj[c] += K::from(row_m[c]) * wr; }
        }
        vjs.push(vj);
    }
    // y_j = Z · v_j
    for j in 0..s.t() {
        let yj_digits = mat_vec_mul_fk::<F, K>(Z.as_slice(), Z.rows(), Z.cols(), &vjs[j]);
        let mut yj_pad = yj_digits; 
        let cur = yj_pad.len(); 
        if d_pad > cur { 
            yj_pad.resize(d_pad, K::ZERO); 
        }
        y_new.push(yj_pad);
    }
    // y_scalars: base-b recomposition from digits
    let bK = K::from(F::from_u64(b as u64));
    let mut y_scalars = Vec::with_capacity(s.t());
    for j in 0..s.t() {
        let mut acc = K::ZERO; 
        let mut pw = K::ONE;
        for d in 0..D { 
            acc += pw * y_new[j][d]; 
            pw *= bK; 
        }
        y_scalars.push(acc);
    }
    (y_new, y_scalars)
}

// ---------------------------------------------------------------------------
// Matrix Arithmetic
// ---------------------------------------------------------------------------

/// Left-multiply accumulator by rho: `acc += rho * a`.
pub fn left_mul_acc(acc: &mut Mat<F>, rho: &Mat<F>, a: &Mat<F>) {
    debug_assert_eq!(rho.rows(), rho.cols());
    debug_assert_eq!(rho.rows(), acc.rows());
    debug_assert_eq!(a.rows(), acc.rows());
    debug_assert_eq!(a.cols(), acc.cols());
    let d = acc.rows();
    let m = acc.cols();
    for r in 0..d {
        for c in 0..m {
            let mut sum = F::ZERO;
            for k in 0..d {
                sum += rho[(r, k)] * a[(k, c)];
            }
            acc[(r, c)] += sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Formatting Utilities
// ---------------------------------------------------------------------------

/// Helper formatting for extension field elements used in debug logs.
pub fn format_ext(x: K) -> String {
    let coeffs = x.as_coeffs();
    format!("({}, {})", coeffs[0].as_canonical_u64(), coeffs[1].as_canonical_u64())
}

