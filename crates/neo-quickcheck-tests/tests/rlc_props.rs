#![cfg(feature = "quickcheck")]
//! QuickCheck: combine-then-evaluate == evaluate-then-combine for partial evaluations.
//! This checks the algebraic heart of ΠRLC with scalar weights (constants in S).

#![allow(deprecated)]
#![allow(non_snake_case)] // Allow mathematical notation for matrices

use quickcheck_macros::quickcheck;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

/// Tensorizes r \in K^ell into r_b \in K^{2^ell}.
fn tensor_rb(r: &[K]) -> Vec<K> {
    let ell = r.len();
    let n = 1usize << ell;
    let mut rb = vec![K::ONE; n];
    for j in 0..n {
        let mut w = K::ONE;
        for i in 0..ell {
            let bit = (j >> i) & 1;
            w *= if bit == 1 { r[i] } else { K::ONE - r[i] };
        }
        rb[j] = w;
    }
    rb
}

/// y = Z * (M^T * r_b) over K; Z \in F^{d x m}, M \in F^{n x m}
fn partial_eval(Z: &[Vec<F>], M: &[Vec<F>], rb: &[K]) -> Vec<K> {
    let d = Z.len();
    let m = if d == 0 { 0 } else { Z[0].len() };
    let n = rb.len();

    // tmp = M^T * r_b \in K^m
    let mut tmp = vec![K::ZERO; m];
    for row in 0..n {
        let rb_row = rb[row];
        let Mi = &M[row];
        for col in 0..m {
            tmp[col] += K::from(Mi[col]) * rb_row;
        }
    }

    // y = Z * tmp  (Z is over F, embed into K on the fly)
    let mut y = vec![K::ZERO; d];
    for i in 0..d {
        let Zi = &Z[i];
        for col in 0..m {
            y[i] += K::from(Zi[col]) * tmp[col];
        }
    }
    y
}

fn clamp(x: u8, lo: usize, hi: usize) -> usize {
    let span = hi - lo + 1;
    lo + (x as usize % span)
}

fn take_u64(it: &mut impl Iterator<Item = u64>, default: u64) -> u64 {
    it.next().unwrap_or(default)
}

#[quickcheck]
fn rlc_linearity_holds(
    k_raw: u8, d_raw: u8, m_raw: u8, ell_raw: u8, t_raw: u8,
    mut data: Vec<u64>
) -> bool {
    // Small, CI-friendly sizes
    let k   = clamp(k_raw, 1, 3);
    let d   = clamp(d_raw, 1, 4);
    let m   = clamp(m_raw, 1, 4);
    let ell = clamp(ell_raw, 1, 3);
    let n   = 1usize << ell;
    let _t   = clamp(t_raw, 1, 2); // number of M_j; we'll just use j=0

    // Simple RNG from the input vector
    let mut it = data.drain(..);

    // Build r \in K^ell, then r_b \in K^n
    let mut r = Vec::with_capacity(ell);
    for _ in 0..ell {
        let u = take_u64(&mut it, 1);
        r.push(K::from(F::from_u64(u))); // arbitrary K via F embedding
    }
    let rb = tensor_rb(&r);

    // Build k matrices Z_i \in F^{d x m}
    let mut Zs = Vec::with_capacity(k);
    for _ in 0..k {
        let mut Z = Vec::with_capacity(d);
        for _ in 0..d {
            let mut row = Vec::with_capacity(m);
            for _ in 0..m {
                row.push(F::from_u64(take_u64(&mut it, 0)));
            }
            Z.push(row);
        }
        Zs.push(Z);
    }

    // Build one M \in F^{n x m} (t kept; we just use one)
    let mut M = Vec::with_capacity(n);
    for _ in 0..n {
        let mut row = Vec::with_capacity(m);
        for _ in 0..m {
            row.push(F::from_u64(take_u64(&mut it, 0)));
        }
        M.push(row);
    }

    // Scalar "rotation matrices": choose alpha_i \in F (constants in S)
    let mut alphas = Vec::with_capacity(k);
    for _ in 0..k {
        alphas.push(F::from_u64(take_u64(&mut it, 1)));
    }

    // evaluate-then-combine
    let mut lhs = vec![K::ZERO; d];
    for i in 0..k {
        let yi = partial_eval(&Zs[i], &M, &rb);
        for j in 0..d {
            lhs[j] += K::from(alphas[i]) * yi[j];
        }
    }

    // combine-then-evaluate
    let mut Z_sum = vec![vec![F::ZERO; m]; d];
    for i in 0..k {
        for r_ in 0..d {
            for c in 0..m {
                Z_sum[r_][c] += alphas[i] * Zs[i][r_][c];
            }
        }
    }
    let rhs = partial_eval(&Z_sum, &M, &rb);

    lhs == rhs
}
