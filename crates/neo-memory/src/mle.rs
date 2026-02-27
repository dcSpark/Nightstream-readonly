//! Multilinear-extension helpers for Twist/Shout oracles.
use p3_field::Field;

/// Evaluate the less-than multilinear extension:
/// LT(j', j) = 1 if int(j') < int(j) else 0, with bit-vectors interpreted
/// little-endian. Valid over any field since it is a multilinear polynomial.
pub fn lt_eval<Kf: Field>(j_prime: &[Kf], j: &[Kf]) -> Kf {
    assert_eq!(j_prime.len(), j.len(), "lt_eval: length mismatch");
    let ell = j.len();

    // suffix[i] = Π_{k≥i} eq(j'_k, j_k)
    let mut suffix = vec![Kf::ONE; ell + 1];
    for i in (0..ell).rev() {
        let eq = eq_single(j_prime[i], j[i]);
        suffix[i] = suffix[i + 1] * eq;
    }

    let mut acc = Kf::ZERO;
    for i in 0..ell {
        let tail = suffix[i + 1];
        acc += (Kf::ONE - j_prime[i]) * j[i] * tail;
    }
    acc
}

/// Build the χ table for a point `r ∈ K^ℓ`, returning length `2^ℓ`.
///
/// χ_r[i] = Π_bit (r_bit if i_bit else 1-r_bit), little-endian bits.
pub fn build_chi_table<Kf: Field>(r: &[Kf]) -> Vec<Kf> {
    let ell = r.len();
    let n = 1usize << ell;
    let mut out = vec![Kf::ONE; n];

    // Gray-code style expansion: at step i, split every block into low/high halves.
    for (i, &ri) in r.iter().enumerate() {
        let stride = 1usize << i;
        let block = 1usize << (ell - i - 1);
        let one_minus = Kf::ONE - ri;

        let mut idx = 0usize;
        for _ in 0..block {
            for j in 0..stride {
                let a = out[idx + j];
                out[idx + j] = a * one_minus;
            }
            for j in 0..stride {
                let a = out[idx + stride + j];
                out[idx + stride + j] = a * ri;
            }
            idx += 2 * stride;
        }
    }

    out
}

/// Evaluate the multilinear extension of a vector `v` at point `r`.
///
/// `v` is interpreted over the Boolean hypercube of dimension `r.len()`.
pub fn mle_eval<F: Field, Kf: Field + From<F>>(v: &[F], r: &[Kf]) -> Kf {
    let chi = build_chi_table(r);
    assert_eq!(v.len(), chi.len(), "mle_eval: dimension mismatch");
    let mut acc = Kf::ZERO;
    for (val, weight) in v.iter().zip(chi.iter()) {
        acc += Kf::from(*val) * *weight;
    }
    acc
}

#[inline]
pub(crate) fn eq_single<Kf: Field>(a: Kf, b: Kf) -> Kf {
    (Kf::ONE - a) * (Kf::ONE - b) + a * b
}

/// Re-export the eq polynomial for convenience.
pub use neo_reductions::engines::paper_exact_engine::eq_points;

/// Compute χ_r[idx] for a single index without building the full table.
///
/// This is O(ℓ) instead of O(2^ℓ) for computing a single chi value.
/// Use this when you only need chi values for a sparse set of indices.
///
/// Formula: χ_r[idx] = Π_{bit} (r[bit] if (idx >> bit) & 1 else (1 - r[bit]))
#[inline]
pub fn chi_at_index<Kf: Field>(r: &[Kf], idx: usize) -> Kf {
    let mut acc = Kf::ONE;
    for (bit, &r_bit) in r.iter().enumerate() {
        if (idx >> bit) & 1 == 1 {
            acc *= r_bit;
        } else {
            acc *= Kf::ONE - r_bit;
        }
    }
    acc
}

// ============================================================================
// ME Instance Computation Helpers
// ============================================================================

use neo_ccs::matrix::Mat;
use neo_ccs::CcsStructure;
use neo_math::{F as BaseField, K as KElem};
use neo_params::NeoParams;
// Note: p3_field traits are needed for ONE, ZERO, from_u64 on concrete types
use p3_field::PrimeCharacteristicRing;

/// Compute y_j for all j ∈ [t] using CCS matrices.
///
/// For Neo's ME relation to be satisfied, we need:
///   y_j = Z · M_j^T · χ_r  for all j ∈ [t]
///
/// Where:
/// - Z is the d×m witness matrix (Ajtai encoded)
/// - M_j is the j-th CCS constraint matrix (n×m)
/// - χ_r is the Lagrange basis vector at evaluation point r (length n)
///
/// Returns: (y_vecs, y_scalars) where
/// - y_vecs[j][row] = (Z · M_j^T · χ_r)[row] for row in 0..d
/// - y_scalars[j] follows SuperNeo semantics: constant term of the ring row
pub fn compute_me_y_for_ccs(
    params: &NeoParams,
    s: &CcsStructure<BaseField>,
    z_padded: &Mat<BaseField>,
    r: &[KElem],
) -> (Vec<Vec<KElem>>, Vec<KElem>) {
    let d = z_padded.rows();

    // Validate r length against CCS row-domain dimensions.
    let n_pad = s.n.next_power_of_two();
    let ell = n_pad.trailing_zeros() as usize;
    debug_assert_eq!(r.len(), ell, "r length ({}) must match ell_n ({})", r.len(), ell);

    let ell_d = d.next_power_of_two().trailing_zeros() as usize;
    let (mut y_vecs, y_scalars) = neo_reductions::common::compute_y_from_Z_and_r(s, z_padded, r, ell_d, params.b);
    for yj in &mut y_vecs {
        yj.truncate(d);
    }
    (y_vecs, y_scalars)
}

/// Compute X = L_x(Z) - the projection of Z onto the first m_in columns.
///
/// For Neo's ME relation, X should be the first m_in columns of Z.
/// For memory witnesses where the first m_in columns are zero, X will be zero.
pub fn compute_me_x<F>(z_padded: &Mat<F>, m_in: usize) -> Mat<F>
where
    F: PrimeCharacteristicRing + Copy,
{
    let d = z_padded.rows();
    let mut x_mat = Mat::zero(d, m_in, F::ZERO);

    for row in 0..d {
        let z_row = z_padded.row(row);
        for c in 0..m_in.min(z_padded.cols()) {
            x_mat.set(row, c, z_row[c]);
        }
    }

    x_mat
}
