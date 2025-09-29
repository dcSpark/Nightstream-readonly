#![allow(dead_code)]
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

/// CSR-style triplets for one CCS matrix M_j
#[derive(Clone, Debug)]
pub struct CcsCsr {
    pub rows: usize,
    pub cols: usize,
    /// (row, col, value) with 0-based indices. Row-major tiebreak preferred.
    pub entries: Vec<(u32, u32, F)>,
}

/// Bundle of all matrices needed for the Pi-CCS terminal check.
#[derive(Clone, Debug)]
pub struct PiCcsEmbed {
    pub matrices: Vec<CcsCsr>,
}

#[inline]
pub fn pow_table(base_b: F, d: usize) -> Vec<F> {
    let mut out = vec![F::ONE; d];
    for i in 1..d { out[i] = out[i - 1] * base_b; }
    out
}

/// Compute χ_r over {0,1}^ell up to length n, where ell = r_bits.len() and n ≤ 2^ell.
/// χ_r[i] = ∏_t (bit_t(i) ? r_t : (1 - r_t)).
pub fn tensor_point_from_bits(r_bits: &[F], n: usize) -> Vec<F> {
    let ell = r_bits.len();
    assert!(n <= (1usize << ell), "n={} exceeds 2^ell (ell={})", n, ell);
    let mut out = vec![F::ONE; n];
    for i in 0..n {
        let mut acc = F::ONE;
        let mut mask = i;
        for t in 0..ell {
            let rt = r_bits[t];
            let term = if (mask & 1) == 1 { rt } else { F::ONE - rt };
            acc *= term;
            mask >>= 1;
        }
        out[i] = acc;
    }
    out
}

/// v = M^T * χ_r (sparse via triplets)
pub fn spmv_transpose(csr: &CcsCsr, chi: &[F]) -> Vec<F> {
    assert_eq!(csr.rows, chi.len(), "χ_r length {} != rows {}", chi.len(), csr.rows);
    let mut v = vec![F::ZERO; csr.cols];
    for &(r, c, a) in &csr.entries {
        v[c as usize] += a * chi[r as usize];
    }
    v
}

/// Expand per-digit weights from v by multiplying with base powers per digit (layout: c*d + r)
pub fn expand_weights_from_v(v: &[F], base_b: F, d: usize) -> Vec<F> {
    let pow_b = pow_table(base_b, d);
    let m = v.len();
    let mut w = vec![F::ZERO; d * m];
    for c in 0..m {
        for r in 0..d { w[c * d + r] = v[c] * pow_b[r]; }
    }
    w
}

/// Reference expansion for all matrices (host-side parity helper)
pub fn expand_weight_vectors_from_matrices(
    mats: &[CcsCsr],
    r_bits: &[F],
    base_b: F,
    d: usize,
    n_rows: usize,
) -> Vec<Vec<F>> {
    let chi = tensor_point_from_bits(r_bits, n_rows);
    mats.iter()
        .map(|mj| expand_weights_from_v(&spmv_transpose(mj, &chi), base_b, d))
        .collect()
}
