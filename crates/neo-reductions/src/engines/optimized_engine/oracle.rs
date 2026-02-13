//! Optimized RoundOracle for Q(X) evaluation in Π_CCS.
//!
//! This oracle uses factored algebra, precomputed terms, and cached sparse formats
//! to efficiently evaluate the Q polynomial during sumcheck rounds. Mathematically
//! equivalent to paper-exact but ~10x faster.
//!
//! Variable order (rounds): first the `ell_n` row bits, then the `ell_d` Ajtai bits.

#![allow(non_snake_case)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat, McsInstance, McsWitness, MeInstance};
use neo_math::{Fq, KExtensions, D, K};
use p3_field::{Field, PrimeCharacteristicRing};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;
use std::sync::Arc;

use crate::sumcheck::RoundOracle;

use super::common::Challenges;
pub use super::sparse::SparseCache;

/// NC-only oracle for the split-NC Π_CCS variant.
///
/// Variable order (rounds): first the `ell_m` column bits, then the `ell_d` Ajtai bits.
///
/// This oracle evaluates the NC polynomial:
///   Q_nc(s, α) = eq(s, β_m) * eq(α, β_a) * Σ_i γ^{i+1} · N_i(Ẑ_i(α, s))
/// where `N_i(·)` is the digit-range (norm-check) range polynomial.
pub struct NcOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<F>,
{
    pub s: &'a CcsStructure<F>,
    pub params: &'a neo_params::NeoParams,
    pub mcs_witnesses: &'a [McsWitness<F>],
    pub me_witnesses: &'a [Mat<F>],
    pub ch: Challenges,

    pub ell_d: usize,
    pub ell_m: usize,
    pub d_sc: usize,

    pub round_idx: usize,
    pub col_chals: Vec<K>,
    pub ajtai_chals: Vec<K>,

    // Streaming tables over the remaining column bits.
    cur_len: usize,
    eq_beta_m_tbl: Vec<K>,
    // digits_tables[i][col_mask][rho] = Z_i[rho, col_mask] (with zero-padding).
    digits_tables: Vec<Vec<[K; D]>>,
    // weights[i][rho] = γ^{i+1} * χ_{β_a}(rho)
    weights: Vec<[K; D]>,
    // Cached t^2 values for the symmetric range polynomial.
    range_t_sq: Vec<K>,
}

impl<'a, F> NcOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<F>,
{
    pub fn new(
        s: &'a CcsStructure<F>,
        params: &'a neo_params::NeoParams,
        mcs_witnesses: &'a [McsWitness<F>],
        me_witnesses: &'a [Mat<F>],
        ch: Challenges,
        ell_d: usize,
        ell_m: usize,
        d_sc: usize,
    ) -> Self {
        assert!(!mcs_witnesses.is_empty(), "need at least one witness for NC");
        assert!(
            ch.beta_m.len() == ell_m,
            "NcOracle: beta_m length mismatch (expected {}, got {})",
            ell_m,
            ch.beta_m.len()
        );
        assert!(
            ch.beta_a.len() == ell_d,
            "NcOracle: beta_a length mismatch (expected {}, got {})",
            ell_d,
            ch.beta_a.len()
        );

        let m_pad = 1usize << ell_m;

        // Column-domain χ_{β_m} table.
        let eq_beta_m_tbl = chi_tail_weights(&ch.beta_m);
        debug_assert_eq!(eq_beta_m_tbl.len(), m_pad, "chi(beta_m) length mismatch");

        // Gather all Z witnesses in order: MCS first, then ME.
        let mut all_witnesses: Vec<&Mat<F>> = Vec::with_capacity(mcs_witnesses.len() + me_witnesses.len());
        for w in mcs_witnesses {
            all_witnesses.push(&w.Z);
        }
        for z in me_witnesses {
            all_witnesses.push(z);
        }

        // Precompute χ_{β_a}(rho) for rho=0..D-1.
        let mut w_beta_a = [K::ZERO; D];
        for rho in 0..D {
            w_beta_a[rho] = eq_points_bool_mask(rho, &ch.beta_a);
        }

        // weights[i][rho] = γ^{i+1} * χ_{β_a}(rho)
        let mut weights: Vec<[K; D]> = Vec::with_capacity(all_witnesses.len());
        let mut g = ch.gamma; // γ^1
        for _ in 0..all_witnesses.len() {
            let mut wi = [K::ZERO; D];
            for rho in 0..D {
                wi[rho] = g * w_beta_a[rho];
            }
            weights.push(wi);
            g *= ch.gamma;
        }

        // Column-domain digit tables.
        let mut digits_tables: Vec<Vec<[K; D]>> = Vec::with_capacity(all_witnesses.len());
        for Zi in all_witnesses {
            if Zi.rows() != D || Zi.cols() != s.m {
                panic!(
                    "Z shape mismatch: expected {}×{}, got {}×{}",
                    D,
                    s.m,
                    Zi.rows(),
                    Zi.cols()
                );
            }
            let mut tbl = vec![[K::ZERO; D]; m_pad];
            let cap = core::cmp::min(s.m, m_pad);
            for rho in 0..D {
                let z_row = Zi.row(rho);
                for col in 0..cap {
                    tbl[col][rho] = K::from(z_row[col]);
                }
            }
            digits_tables.push(tbl);
        }

        // Symmetric range polynomial cache.
        let mut range_t_sq = Vec::new();
        if params.b > 1 {
            range_t_sq.reserve((params.b - 1) as usize);
            for t in 1..(params.b as i64) {
                let tt = F::from_i64(t);
                range_t_sq.push(K::from(tt * tt));
            }
        }

        Self {
            s,
            params,
            mcs_witnesses,
            me_witnesses,
            ch,
            ell_d,
            ell_m,
            d_sc,
            round_idx: 0,
            col_chals: Vec::with_capacity(ell_m),
            ajtai_chals: Vec::with_capacity(ell_d),
            cur_len: m_pad,
            eq_beta_m_tbl,
            digits_tables,
            weights,
            range_t_sq,
        }
    }

    #[inline]
    fn num_rounds_total(&self) -> usize {
        self.ell_m + self.ell_d
    }

    #[inline]
    fn fold_table_inplace(table: &mut Vec<K>, r: K) {
        debug_assert!(table.len() >= 2 && table.len() % 2 == 0);
        let half = table.len() / 2;
        for i in 0..half {
            let lo = table[2 * i];
            let hi = table[2 * i + 1];
            table[i] = lo + (hi - lo) * r;
        }
        table.truncate(half);
    }

    #[inline]
    fn fold_digits_table_inplace(table: &mut Vec<[K; D]>, r: K) {
        debug_assert!(table.len() >= 2 && table.len() % 2 == 0);
        let half = table.len() / 2;
        for i in 0..half {
            let base = 2 * i;
            for rho in 0..D {
                let lo = table[base][rho];
                let hi = table[base + 1][rho];
                table[i][rho] = lo + (hi - lo) * r;
            }
        }
        table.truncate(half);
    }

    fn evals_col_phase_generic(&self, xs: &[K]) -> Vec<K> {
        debug_assert!(self.cur_len >= 2 && self.cur_len % 2 == 0);
        let tail_len = self.cur_len / 2;
        let xs_len = xs.len();
        if xs_len == 0 {
            return Vec::new();
        }

        // `tail_len` starts at m_pad/2 and halves each column round; parallelize only when big enough.
        const PAR_THRESHOLD: usize = 1 << 13;
        let evals_col_phase_seq = |tail_len: usize, xs: &[K]| -> Vec<K> {
            let xs_len = xs.len();
            let mut out = vec![K::ZERO; xs_len];
            let mut nc_sum_by_x = vec![K::ZERO; xs_len];
            let mut eq_beta_m_x = vec![K::ZERO; xs_len];

            for t in 0..tail_len {
                nc_sum_by_x.fill(K::ZERO);

                let idx = 2 * t;
                let e0 = self.eq_beta_m_tbl[idx];
                let e1 = self.eq_beta_m_tbl[idx + 1] - e0;

                for (x_idx, &x) in xs.iter().enumerate() {
                    eq_beta_m_x[x_idx] = e0 + e1 * x;
                }

                for (wit_idx, tbl) in self.digits_tables.iter().enumerate() {
                    let lo = &tbl[idx];
                    let hi = &tbl[idx + 1];
                    let weights = &self.weights[wit_idx];

                    for rho in 0..D {
                        let y0 = lo[rho];
                        let dy = hi[rho] - y0;
                        let w = weights[rho];
                        for (x_idx, &x) in xs.iter().enumerate() {
                            let y = y0 + dy * x;
                            nc_sum_by_x[x_idx] += w * range_product_cached(y, &self.range_t_sq);
                        }
                    }
                }

                for x_idx in 0..xs_len {
                    out[x_idx] += eq_beta_m_x[x_idx] * nc_sum_by_x[x_idx];
                }
            }

            out
        };

        if tail_len >= PAR_THRESHOLD {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                let (out, _scratch_nc, _scratch_eq) = (0..tail_len)
                    .into_par_iter()
                    .fold(
                        || (vec![K::ZERO; xs_len], vec![K::ZERO; xs_len], vec![K::ZERO; xs_len]),
                        |(mut out, mut nc_sum_by_x, mut eq_beta_m_x), t| {
                            nc_sum_by_x.fill(K::ZERO);

                            let idx = 2 * t;
                            let e0 = self.eq_beta_m_tbl[idx];
                            let e1 = self.eq_beta_m_tbl[idx + 1] - e0;

                            for (x_idx, &x) in xs.iter().enumerate() {
                                eq_beta_m_x[x_idx] = e0 + e1 * x;
                            }

                            for (wit_idx, tbl) in self.digits_tables.iter().enumerate() {
                                let lo = &tbl[idx];
                                let hi = &tbl[idx + 1];
                                let weights = &self.weights[wit_idx];

                                for rho in 0..D {
                                    let y0 = lo[rho];
                                    let dy = hi[rho] - y0;
                                    let w = weights[rho];
                                    for (x_idx, &x) in xs.iter().enumerate() {
                                        let y = y0 + dy * x;
                                        nc_sum_by_x[x_idx] += w * range_product_cached(y, &self.range_t_sq);
                                    }
                                }
                            }

                            for x_idx in 0..xs_len {
                                out[x_idx] += eq_beta_m_x[x_idx] * nc_sum_by_x[x_idx];
                            }
                            (out, nc_sum_by_x, eq_beta_m_x)
                        },
                    )
                    .reduce(
                        || (vec![K::ZERO; xs_len], vec![K::ZERO; xs_len], vec![K::ZERO; xs_len]),
                        |(mut out_a, nc_a, eq_a), (out_b, _nc_b, _eq_b)| {
                            for i in 0..xs_len {
                                out_a[i] += out_b[i];
                            }
                            (out_a, nc_a, eq_a)
                        },
                    );
                out
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                evals_col_phase_seq(tail_len, xs)
            }
        } else {
            evals_col_phase_seq(tail_len, xs)
        }
    }

    fn evals_col_phase_b2(&self, xs: &[K]) -> Vec<K> {
        debug_assert!(self.cur_len >= 2 && self.cur_len % 2 == 0);
        let tail_len = self.cur_len / 2;
        if xs.is_empty() {
            return Vec::new();
        }

        const PAR_THRESHOLD: usize = 1 << 13;
        let three = K::from(F::from_u64(3));

        let coeffs_seq = |tail_len: usize| -> [K; 5] {
            let mut coeffs = [K::ZERO; 5];
            for t in 0..tail_len {
                let idx = 2 * t;
                let e0 = self.eq_beta_m_tbl[idx];
                let e1 = self.eq_beta_m_tbl[idx + 1] - e0;

                let mut inner = [K::ZERO; 4];
                for (wit_idx, tbl) in self.digits_tables.iter().enumerate() {
                    let lo = &tbl[idx];
                    let hi = &tbl[idx + 1];
                    let weights = &self.weights[wit_idx];

                    for rho in 0..D {
                        let w = weights[rho];
                        let a = lo[rho];
                        let b = hi[rho] - a;

                        let a2 = a * a;
                        let a3 = a2 * a;
                        let b2 = b * b;
                        let b3 = b2 * b;

                        // N(a+bX) = (a+bX)^3 - (a+bX)
                        let t0 = a3 - a;
                        let t1 = (a2 * b).scale_base_k(three) - b;
                        let t2 = (a * b2).scale_base_k(three);
                        let t3 = b3;

                        inner[0] += w * t0;
                        inner[1] += w * t1;
                        inner[2] += w * t2;
                        inner[3] += w * t3;
                    }
                }

                // (e0 + e1 X) * (inner0 + inner1 X + inner2 X^2 + inner3 X^3)
                coeffs[0] += e0 * inner[0];
                coeffs[1] += e0 * inner[1] + e1 * inner[0];
                coeffs[2] += e0 * inner[2] + e1 * inner[1];
                coeffs[3] += e0 * inner[3] + e1 * inner[2];
                coeffs[4] += e1 * inner[3];
            }
            coeffs
        };

        let coeffs = if tail_len >= PAR_THRESHOLD {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                (0..tail_len)
                    .into_par_iter()
                    .fold(
                        || [K::ZERO; 5],
                        |mut coeffs, t| {
                            let idx = 2 * t;
                            let e0 = self.eq_beta_m_tbl[idx];
                            let e1 = self.eq_beta_m_tbl[idx + 1] - e0;

                            let mut inner = [K::ZERO; 4];
                            for (wit_idx, tbl) in self.digits_tables.iter().enumerate() {
                                let lo = &tbl[idx];
                                let hi = &tbl[idx + 1];
                                let weights = &self.weights[wit_idx];

                                for rho in 0..D {
                                    let w = weights[rho];
                                    let a = lo[rho];
                                    let b = hi[rho] - a;

                                    let a2 = a * a;
                                    let a3 = a2 * a;
                                    let b2 = b * b;
                                    let b3 = b2 * b;

                                    let t0 = a3 - a;
                                    let t1 = (a2 * b).scale_base_k(three) - b;
                                    let t2 = (a * b2).scale_base_k(three);
                                    let t3 = b3;

                                    inner[0] += w * t0;
                                    inner[1] += w * t1;
                                    inner[2] += w * t2;
                                    inner[3] += w * t3;
                                }
                            }

                            coeffs[0] += e0 * inner[0];
                            coeffs[1] += e0 * inner[1] + e1 * inner[0];
                            coeffs[2] += e0 * inner[2] + e1 * inner[1];
                            coeffs[3] += e0 * inner[3] + e1 * inner[2];
                            coeffs[4] += e1 * inner[3];
                            coeffs
                        },
                    )
                    .reduce(
                        || [K::ZERO; 5],
                        |mut a, b| {
                            for i in 0..5 {
                                a[i] += b[i];
                            }
                            a
                        },
                    )
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                coeffs_seq(tail_len)
            }
        } else {
            coeffs_seq(tail_len)
        };

        let xs_are_base = xs.iter().all(|&x| x.imag() == Fq::ZERO);
        if xs_are_base {
            xs.iter()
                .map(|&x| crate::sumcheck::poly_eval_k_base(&coeffs, x.real()))
                .collect()
        } else {
            xs.iter()
                .map(|&x| crate::sumcheck::poly_eval_k(&coeffs, x))
                .collect()
        }
    }

    fn evals_col_phase_b3(&self, xs: &[K]) -> Vec<K> {
        debug_assert!(self.cur_len >= 2 && self.cur_len % 2 == 0);
        let tail_len = self.cur_len / 2;
        if xs.is_empty() {
            return Vec::new();
        }

        const PAR_THRESHOLD: usize = 1 << 13;
        let four = K::from(F::from_u64(4));
        let five = K::from(F::from_u64(5));
        let ten = K::from(F::from_u64(10));
        let fifteen = K::from(F::from_u64(15));

        let coeffs_seq = |tail_len: usize| -> [K; 7] {
            let mut coeffs = [K::ZERO; 7];
            for t in 0..tail_len {
                let idx = 2 * t;
                let e0 = self.eq_beta_m_tbl[idx];
                let e1 = self.eq_beta_m_tbl[idx + 1] - e0;

                let mut inner = [K::ZERO; 6];
                for (wit_idx, tbl) in self.digits_tables.iter().enumerate() {
                    let lo = &tbl[idx];
                    let hi = &tbl[idx + 1];
                    let weights = &self.weights[wit_idx];

                    for rho in 0..D {
                        let w = weights[rho];
                        let a = lo[rho];
                        let b = hi[rho] - a;

                        let a2 = a * a;
                        let a3 = a2 * a;
                        let a4 = a2 * a2;
                        let a5 = a4 * a;

                        let b2 = b * b;
                        let b3 = b2 * b;
                        let b4 = b2 * b2;
                        let b5 = b4 * b;

                        // N(a+bX) = (a+bX)^5 - 5(a+bX)^3 + 4(a+bX)
                        let t0 = a5 - a3.scale_base_k(five) + a.scale_base_k(four);
                        let t1 = b * (a4.scale_base_k(five) - a2.scale_base_k(fifteen) + four);
                        let t2 = b2 * (a3.scale_base_k(ten) - a.scale_base_k(fifteen));
                        let t3 = b3 * (a2.scale_base_k(ten) - five);
                        let t4 = b4 * a.scale_base_k(five);
                        let t5 = b5;

                        inner[0] += w * t0;
                        inner[1] += w * t1;
                        inner[2] += w * t2;
                        inner[3] += w * t3;
                        inner[4] += w * t4;
                        inner[5] += w * t5;
                    }
                }

                // (e0 + e1 X) * Σ_{k=0..5} inner[k] X^k
                coeffs[0] += e0 * inner[0];
                coeffs[1] += e0 * inner[1] + e1 * inner[0];
                coeffs[2] += e0 * inner[2] + e1 * inner[1];
                coeffs[3] += e0 * inner[3] + e1 * inner[2];
                coeffs[4] += e0 * inner[4] + e1 * inner[3];
                coeffs[5] += e0 * inner[5] + e1 * inner[4];
                coeffs[6] += e1 * inner[5];
            }
            coeffs
        };

        let coeffs = if tail_len >= PAR_THRESHOLD {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                (0..tail_len)
                    .into_par_iter()
                    .fold(
                        || [K::ZERO; 7],
                        |mut coeffs, t| {
                            let idx = 2 * t;
                            let e0 = self.eq_beta_m_tbl[idx];
                            let e1 = self.eq_beta_m_tbl[idx + 1] - e0;

                            let mut inner = [K::ZERO; 6];
                            for (wit_idx, tbl) in self.digits_tables.iter().enumerate() {
                                let lo = &tbl[idx];
                                let hi = &tbl[idx + 1];
                                let weights = &self.weights[wit_idx];

                                for rho in 0..D {
                                    let w = weights[rho];
                                    let a = lo[rho];
                                    let b = hi[rho] - a;

                                    let a2 = a * a;
                                    let a3 = a2 * a;
                                    let a4 = a2 * a2;
                                    let a5 = a4 * a;

                                    let b2 = b * b;
                                    let b3 = b2 * b;
                                    let b4 = b2 * b2;
                                    let b5 = b4 * b;

                                    let t0 = a5 - a3.scale_base_k(five) + a.scale_base_k(four);
                                    let t1 = b * (a4.scale_base_k(five) - a2.scale_base_k(fifteen) + four);
                                    let t2 = b2 * (a3.scale_base_k(ten) - a.scale_base_k(fifteen));
                                    let t3 = b3 * (a2.scale_base_k(ten) - five);
                                    let t4 = b4 * a.scale_base_k(five);
                                    let t5 = b5;

                                    inner[0] += w * t0;
                                    inner[1] += w * t1;
                                    inner[2] += w * t2;
                                    inner[3] += w * t3;
                                    inner[4] += w * t4;
                                    inner[5] += w * t5;
                                }
                            }

                            coeffs[0] += e0 * inner[0];
                            coeffs[1] += e0 * inner[1] + e1 * inner[0];
                            coeffs[2] += e0 * inner[2] + e1 * inner[1];
                            coeffs[3] += e0 * inner[3] + e1 * inner[2];
                            coeffs[4] += e0 * inner[4] + e1 * inner[3];
                            coeffs[5] += e0 * inner[5] + e1 * inner[4];
                            coeffs[6] += e1 * inner[5];
                            coeffs
                        },
                    )
                    .reduce(
                        || [K::ZERO; 7],
                        |mut a, b| {
                            for i in 0..7 {
                                a[i] += b[i];
                            }
                            a
                        },
                    )
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                coeffs_seq(tail_len)
            }
        } else {
            coeffs_seq(tail_len)
        };

        let xs_are_base = xs.iter().all(|&x| x.imag() == Fq::ZERO);
        if xs_are_base {
            xs.iter()
                .map(|&x| crate::sumcheck::poly_eval_k_base(&coeffs, x.real()))
                .collect()
        } else {
            xs.iter()
                .map(|&x| crate::sumcheck::poly_eval_k(&coeffs, x))
                .collect()
        }
    }

    fn evals_col_phase(&self, xs: &[K]) -> Vec<K> {
        match self.params.b {
            2 => self.evals_col_phase_b2(xs),
            3 => self.evals_col_phase_b3(xs),
            _ => self.evals_col_phase_generic(xs),
        }
    }

    #[doc(hidden)]
    pub fn __test_col_phase_fast_vs_generic(&self, xs: &[K]) -> Option<(Vec<K>, Vec<K>)> {
        if self.round_idx >= self.ell_m {
            return None;
        }
        match self.params.b {
            2 => Some((self.evals_col_phase_b2(xs), self.evals_col_phase_generic(xs))),
            3 => Some((self.evals_col_phase_b3(xs), self.evals_col_phase_generic(xs))),
            _ => None,
        }
    }

    fn evals_ajtai_phase(&self, xs: &[K]) -> Vec<K> {
        let j = self.round_idx - self.ell_m;
        debug_assert!(j < self.ell_d, "NC Ajtai phase after all Ajtai bits");
        debug_assert!(
            self.cur_len == 1,
            "NC Ajtai phase requires finalized column point (cur_len={})",
            self.cur_len
        );

        let free_a = self.ell_d - j - 1;
        let w_beta_tail = chi_tail_weights(&self.ch.beta_a[j + 1..self.ell_d]);
        let head_stride = 1usize << (j + 1);
        debug_assert_eq!(w_beta_tail.len(), 1usize << free_a);

        // Prefix factor for eq(α, β_a).
        let mut eq_beta_pref = K::ONE;
        for i in 0..j {
            eq_beta_pref *= eq_lin(self.ajtai_chals[i], self.ch.beta_a[i]);
        }
        let beta_j = self.ch.beta_a[j];

        // eq(s', β_m) is the (single) entry after folding all column bits.
        let eq_beta_m = self.eq_beta_m_tbl[0];

        // Prefold digits by Ajtai prefix bits once per round.
        let mut digits_pref: Vec<[K; D]> = Vec::with_capacity(self.digits_tables.len());
        for tbl in self.digits_tables.iter() {
            let mut d = tbl[0];
            for b in 0..j {
                fold_bit_inplace(&mut d, b, self.ajtai_chals[b]);
            }
            digits_pref.push(d);
        }

        let mut out = vec![K::ZERO; xs.len()];
        for (x_idx, &x) in xs.iter().enumerate() {
            let eq_beta = eq_beta_m * (eq_beta_pref * eq_lin(x, beta_j));

            // Apply γ^{i+1} factors (witness order) explicitly.
            let mut g = self.ch.gamma;
            let mut weighted_sum = K::ZERO;
            for digits in digits_pref.iter() {
                let acc =
                    ajtai_tail_weighted_range_prefolded(digits, x, j, head_stride, &w_beta_tail, &self.range_t_sq);
                weighted_sum += g * acc;
                g *= self.ch.gamma;
            }

            out[x_idx] = eq_beta * weighted_sum;
        }

        out
    }
}

impl<'a, F> RoundOracle for NcOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<F>,
{
    fn num_rounds(&self) -> usize {
        self.num_rounds_total()
    }

    fn degree_bound(&self) -> usize {
        self.d_sc
    }

    fn evals_at(&mut self, xs: &[K]) -> Vec<K> {
        if self.round_idx < self.ell_m {
            self.evals_col_phase(xs)
        } else {
            self.evals_ajtai_phase(xs)
        }
    }

    fn fold(&mut self, r_i: K) {
        if self.round_idx < self.ell_m {
            self.col_chals.push(r_i);
            Self::fold_table_inplace(&mut self.eq_beta_m_tbl, r_i);
            for tbl in self.digits_tables.iter_mut() {
                Self::fold_digits_table_inplace(tbl, r_i);
            }
            self.cur_len /= 2;
        } else {
            self.ajtai_chals.push(r_i);
        }
        self.round_idx += 1;
    }
}

#[derive(Clone, Debug)]
struct CompiledPolyTerm {
    coeff: K,
    /// (var_pos, exponent), where `var_pos` indexes `RowStreamState::f_var_tables`.
    vars: Vec<(usize, u32)>,
}

/// Row-phase streaming state (over the row/time hypercube).
///
/// This replaces the old `evals_row_phase` strategy of enumerating row tails and repeatedly
/// running `precompute_for_r`. Instead, we materialize row-domain tables once and fold them
/// in-place as row challenges arrive.
struct RowStreamState {
    /// Current table length = 2^(remaining row bits).
    cur_len: usize,

    /// χ_{β_r}(row) table over the padded row domain (len = cur_len).
    eq_beta_r_tbl: Vec<K>,
    /// Optional χ_{r_inputs}(row) table (len = cur_len) for Eval gating.
    eq_r_inputs_tbl: Option<Vec<K>>,

    /// Recomposition of the first MCS witness `Z₁` into a row vector:
    /// `z1[c] = Σ_{ρ=0..D-1} b^ρ · Z₁[ρ,c]`.
    z1: Vec<K>,

    /// Tables for the variables used by the CCS polynomial `f`.
    /// Each entry is a row-domain table of `m_j(row) = (M_j · z1)[row]` at boolean row points.
    f_var_tables: Vec<Vec<K>>,
    /// Compiled sparse polynomial terms for `f` using `f_var_tables` indices.
    f_terms: Vec<CompiledPolyTerm>,

    /// Combined Eval block table over rows (already summed over α' and (i,j) coefficients).
    /// When present, Eval contribution is: `eq_r_inputs(r') * gamma_to_k * eval_tbl(r')`.
    eval_tbl: Option<Vec<K>>,
    gamma_to_k: K,

    b: u32,
    /// True if all streamed tables are still in the base-field embedding (imag=0).
    ///
    /// When this holds and evaluation points are also base-field, we can evaluate the hot
    /// row-phase logic entirely in `Fq` for a large speedup.
    all_base: bool,
}

impl RowStreamState {
    fn build<Ff>(
        s: &CcsStructure<Ff>,
        b: u32,
        ch: &Challenges,
        ell_d: usize,
        ell_n: usize,
        mcs_witnesses: &[McsWitness<Ff>],
        me_witnesses: &[Mat<Ff>],
        r_inputs: Option<&[K]>,
        sparse: &SparseCache<Ff>,
    ) -> Self
    where
        Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
        K: From<Ff>,
    {
        let n_pad = 1usize << ell_n;
        let n_eff = s.n;
        let t_mats = s.t();

        // Row-domain χ tables.
        let eq_beta_r_tbl = chi_tail_weights(&ch.beta_r);
        debug_assert_eq!(
            eq_beta_r_tbl.len(),
            n_pad,
            "chi(beta_r) length mismatch (ell_n={ell_n})"
        );

        let eq_r_inputs_tbl = r_inputs.map(|r| {
            let tbl = chi_tail_weights(r);
            debug_assert_eq!(tbl.len(), n_pad, "chi(r_inputs) length mismatch");
            tbl
        });

        let all_base = ch.gamma.imag() == Fq::ZERO
            && ch.alpha.iter().all(|x| x.imag() == Fq::ZERO)
            && ch.beta_a.iter().all(|x| x.imag() == Fq::ZERO)
            && ch.beta_r.iter().all(|x| x.imag() == Fq::ZERO)
            && r_inputs
                .map(|r| r.iter().all(|x| x.imag() == Fq::ZERO))
                .unwrap_or(true);

        // Compile CCS polynomial f to avoid scanning t variables per evaluation.
        if s.f.arity() != t_mats {
            panic!(
                "CCS polynomial arity mismatch: f.arity()={}, but s.t()={}",
                s.f.arity(),
                t_mats
            );
        }
        let mut used_vars = vec![false; t_mats];
        for term in s.f.terms() {
            if term.exps.len() != t_mats {
                panic!(
                    "CCS polynomial exponent vector length mismatch: got {}, expected {}",
                    term.exps.len(),
                    t_mats
                );
            }
            for (j, &exp) in term.exps.iter().enumerate() {
                if exp != 0 {
                    used_vars[j] = true;
                }
            }
        }
        let f_var_indices: Vec<usize> = used_vars
            .iter()
            .enumerate()
            .filter_map(|(j, &u)| u.then_some(j))
            .collect();

        let mut pos_by_j = vec![usize::MAX; t_mats];
        for (pos, &j) in f_var_indices.iter().enumerate() {
            pos_by_j[j] = pos;
        }

        let f_terms: Vec<CompiledPolyTerm> =
            s.f.terms()
                .iter()
                .map(|term| {
                    let mut vars = Vec::new();
                    for (j, &exp) in term.exps.iter().enumerate() {
                        if exp != 0 {
                            let pos = pos_by_j[j];
                            debug_assert_ne!(pos, usize::MAX, "missing f var mapping");
                            vars.push((pos, exp));
                        }
                    }
                    CompiledPolyTerm {
                        coeff: K::from(term.coeff),
                        vars,
                    }
                })
                .collect();

        // Gather witnesses in oracle order: all MCS first, then ME.
        let all_witnesses: Vec<&Mat<Ff>> = mcs_witnesses
            .iter()
            .map(|w| &w.Z)
            .chain(me_witnesses.iter())
            .collect();
        let k_total = all_witnesses.len();

        // Sanity: challenge vectors for Ajtai rounds must match ell_d.
        if ch.beta_a.len() != ell_d || ch.alpha.len() != ell_d {
            panic!(
                "Challenge length mismatch: alpha.len()={}, beta_a.len()={}, ell_d={ell_d}",
                ch.alpha.len(),
                ch.beta_a.len()
            );
        }
        // Build z1 = recomposition of Z_1 (first MCS witness).
        let mut z1: Vec<K> = vec![K::ZERO; s.m];
        {
            // Mat is row-major; compute by streaming rows contiguously.
            let base = Ff::from_u64(b as u64);
            let mut pow_b = [Ff::ZERO; D];
            pow_b[0] = Ff::ONE;
            for rho in 1..D {
                pow_b[rho] = pow_b[rho - 1] * base;
            }

            let Z1 = all_witnesses
                .first()
                .copied()
                .expect("need at least one witness for z1");
            if Z1.rows() != D || Z1.cols() != s.m {
                panic!(
                    "Z1 shape mismatch: expected {}×{}, got {}×{}",
                    D,
                    s.m,
                    Z1.rows(),
                    Z1.cols()
                );
            }
            for rho in 0..D {
                let w = pow_b[rho];
                let row = Z1.row(rho);
                for c in 0..s.m {
                    z1[c] += K::from(row[c] * w);
                }
            }
        }

        // f-var tables: m_j(row) = (M_j * z1)[row] for each variable used by f.
        if sparse.len() != t_mats {
            panic!(
                "sparse cache matrix count mismatch: got {}, expected {}",
                sparse.len(),
                t_mats
            );
        }
        let mut f_var_tables: Vec<Vec<K>> = Vec::with_capacity(f_var_indices.len());
        for &j in &f_var_indices {
            let mut out = vec![K::ZERO; n_pad];
            if let Some(csc) = sparse.csc(j) {
                if csc.ncols != z1.len() {
                    panic!(
                        "matrix-vector dim mismatch for j={j}: csc.ncols={} != z1.len()={}",
                        csc.ncols,
                        z1.len()
                    );
                }
                for c in 0..csc.ncols {
                    let s0 = csc.col_ptr[c];
                    let e0 = csc.col_ptr[c + 1];
                    if s0 == e0 {
                        continue;
                    }
                    let x_c = z1[c];
                    if x_c == K::ZERO {
                        continue;
                    }
                    for k in s0..e0 {
                        let r = csc.row_idx[k];
                        if r < n_eff {
                            out[r] += x_c.scale_base_k(K::from(csc.vals[k]));
                        }
                    }
                }
            } else {
                // Identity sentinel: (I · z1)[row] = z1[row]
                let cap = core::cmp::min(n_eff, z1.len());
                out[..cap].copy_from_slice(&z1[..cap]);
            }
            f_var_tables.push(out);
        }

        // Eval table (optional): only when both (a) there are carried witnesses, and (b) r_inputs exist.
        let mut gamma_to_k = K::ONE;
        for _ in 0..k_total {
            gamma_to_k *= ch.gamma;
        }

        let eval_tbl = if k_total >= 2 && eq_r_inputs_tbl.is_some() {
            let w_alpha: Vec<K> = (0..D)
                .map(|rho| eq_points_bool_mask(rho, &ch.alpha))
                .collect();

            let mut gamma_pow_i = vec![K::ONE; k_total];
            for i in 1..k_total {
                gamma_pow_i[i] = gamma_pow_i[i - 1] * ch.gamma;
            }
            let mut gamma_k_pow_j = vec![K::ONE; t_mats];
            for j in 1..t_mats {
                gamma_k_pow_j[j] = gamma_k_pow_j[j - 1] * gamma_to_k;
            }

            let mut eval_tbl = vec![K::ZERO; n_pad];
            for i_abs in 1..k_total {
                let Zi = all_witnesses[i_abs];
                let coeff_i = gamma_pow_i[i_abs];
                if coeff_i == K::ZERO {
                    continue;
                }

                // S_i(α) = Σ_ρ χ_α[ρ] · Z_i[ρ,·]
                let mut s_alpha = vec![K::ZERO; s.m];
                for rho in 0..D {
                    let w = w_alpha[rho];
                    if w == K::ZERO {
                        continue;
                    }
                    for c in 0..s.m {
                        s_alpha[c] += w.scale_base_k(K::from(Zi[(rho, c)]));
                    }
                }

                for j in 0..t_mats {
                    let coeff = coeff_i * gamma_k_pow_j[j];
                    if coeff == K::ZERO {
                        continue;
                    }

                    if sparse.csc(j).is_none() {
                        // Identity sentinel: (I · s_alpha)[row] = s_alpha[row]
                        let cap = core::cmp::min(n_eff, s_alpha.len());
                        for r in 0..cap {
                            eval_tbl[r] += coeff * s_alpha[r];
                        }
                        continue;
                    }

                    let csc = sparse
                        .csc(j)
                        .unwrap_or_else(|| panic!("missing CSC for matrix j={j}"));
                    if csc.ncols != s_alpha.len() {
                        panic!(
                            "matrix-vector dim mismatch for eval j={j}: csc.ncols={} != s_alpha.len()={}",
                            csc.ncols,
                            s_alpha.len()
                        );
                    }
                    for c in 0..csc.ncols {
                        let s0 = csc.col_ptr[c];
                        let e0 = csc.col_ptr[c + 1];
                        if s0 == e0 {
                            continue;
                        }
                        let x_c = s_alpha[c];
                        if x_c == K::ZERO {
                            continue;
                        }
                        let scaled_x = coeff * x_c;
                        for k in s0..e0 {
                            let r = csc.row_idx[k];
                            if r < n_eff {
                                eval_tbl[r] += scaled_x.scale_base_k(K::from(csc.vals[k]));
                            }
                        }
                    }
                }
            }

            Some(eval_tbl)
        } else {
            None
        };

        Self {
            cur_len: n_pad,
            eq_beta_r_tbl,
            eq_r_inputs_tbl,
            z1,
            f_var_tables,
            f_terms,
            eval_tbl,
            gamma_to_k,
            b,
            all_base,
        }
    }

    #[inline]
    fn fold_table_inplace(table: &mut Vec<K>, r: K) {
        debug_assert!(table.len() >= 2 && table.len() % 2 == 0);
        let half = table.len() / 2;
        for i in 0..half {
            let lo = table[2 * i];
            let hi = table[2 * i + 1];
            table[i] = lo + (hi - lo) * r;
        }
        table.truncate(half);
    }

    #[inline]
    fn fold_table_inplace_base(table: &mut Vec<K>, r: Fq) {
        debug_assert!(table.len() >= 2 && table.len() % 2 == 0);
        let half = table.len() / 2;
        for i in 0..half {
            let lo = table[2 * i].real();
            let hi = table[2 * i + 1].real();
            table[i] = K::from(lo + (hi - lo) * r);
        }
        table.truncate(half);
    }

    fn fold_inplace(&mut self, r: K) {
        if self.all_base && r.imag() == Fq::ZERO {
            let r0 = r.real();
            Self::fold_table_inplace_base(&mut self.eq_beta_r_tbl, r0);
            if let Some(tbl) = self.eq_r_inputs_tbl.as_mut() {
                Self::fold_table_inplace_base(tbl, r0);
            }
            for tbl in self.f_var_tables.iter_mut() {
                Self::fold_table_inplace_base(tbl, r0);
            }
            if let Some(tbl) = self.eval_tbl.as_mut() {
                Self::fold_table_inplace_base(tbl, r0);
            }
        } else {
            self.all_base = false;
            Self::fold_table_inplace(&mut self.eq_beta_r_tbl, r);
            if let Some(tbl) = self.eq_r_inputs_tbl.as_mut() {
                Self::fold_table_inplace(tbl, r);
            }
            for tbl in self.f_var_tables.iter_mut() {
                Self::fold_table_inplace(tbl, r);
            }
            if let Some(tbl) = self.eval_tbl.as_mut() {
                Self::fold_table_inplace(tbl, r);
            }
        }
        self.cur_len /= 2;
    }

    #[inline]
    fn poly_mul_affine_inplace_base(poly: &mut [Fq], a: Fq, b: Fq, current_deg: usize) {
        // Coeffs are low→high. Output truncates to input length:
        // new[0] = a*old[0]; new[d] = a*old[d] + b*old[d-1] (d>=1).
        let mut prev = Fq::ZERO;
        for coeff in poly.iter_mut().take(current_deg + 2) {
            let old = *coeff;
            *coeff = a * old + b * prev;
            prev = old;
        }
    }

    #[inline]
    fn poly_eval_base(coeffs: &[Fq], x: Fq) -> Fq {
        if coeffs.is_empty() {
            return Fq::ZERO;
        }
        let mut result = coeffs[coeffs.len() - 1];
        for &c in coeffs.iter().rev().skip(1) {
            result = result * x + c;
        }
        result
    }

    fn evals_row_phase_b2_base(&self, tail_len: usize, xs: &[K]) -> Vec<K> {
        let xs_base: Vec<Fq> = xs.iter().map(|&x| x.real()).collect();

        let f_max_term_deg: usize = self
            .f_terms
            .iter()
            .map(|term| {
                term.vars
                    .iter()
                    .map(|&(_, exp)| exp as usize)
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0);
        // eq_beta_r(X) adds one degree; Eval block is quadratic.
        let deg_max = core::cmp::max(2, f_max_term_deg + 1);

        const PAR_THRESHOLD: usize = 1 << 14;
        let coeffs_seq = |tail_len: usize| -> Vec<Fq> {
            let mut coeffs = vec![Fq::ZERO; deg_max + 1];
            let mut inner = vec![Fq::ZERO; deg_max + 1];
            let mut term_poly = vec![Fq::ZERO; deg_max + 1];

            for t in 0..tail_len {
                let idx = 2 * t;
                let e0 = self.eq_beta_r_tbl[idx].real();
                let e1 = self.eq_beta_r_tbl[idx + 1].real() - e0;

                inner.fill(Fq::ZERO);

                for term in &self.f_terms {
                    term_poly.fill(Fq::ZERO);
                    term_poly[0] = term.coeff.real();
                    let mut current_deg = 0usize;
                    for &(var_pos, exp) in &term.vars {
                        let tbl = &self.f_var_tables[var_pos];
                        let a = tbl[idx].real();
                        let b = tbl[idx + 1].real() - a;
                        for _ in 0..exp {
                            Self::poly_mul_affine_inplace_base(&mut term_poly, a, b, current_deg);
                            current_deg += 1;
                        }
                    }
                    for i in 0..=core::cmp::min(current_deg, deg_max) {
                        inner[i] += term_poly[i];
                    }
                }

                coeffs[0] += e0 * inner[0];
                for d in 1..=deg_max {
                    coeffs[d] += (e0 * inner[d]) + (e1 * inner[d - 1]);
                }

                if let (Some(eq_tbl), Some(eval_tbl)) = (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref()) {
                    let r0 = eq_tbl[idx].real();
                    let r1 = eq_tbl[idx + 1].real() - r0;
                    let v0 = eval_tbl[idx].real();
                    let v1 = eval_tbl[idx + 1].real() - v0;

                    let g = self.gamma_to_k.real();
                    coeffs[0] += g * (r0 * v0);
                    coeffs[1] += g * (r0 * v1 + r1 * v0);
                    coeffs[2] += g * (r1 * v1);
                }
            }

            coeffs
        };

        let coeffs = if tail_len >= PAR_THRESHOLD {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                (0..tail_len)
                    .into_par_iter()
                    .fold(
                        || {
                            (
                                vec![Fq::ZERO; deg_max + 1],
                                vec![Fq::ZERO; deg_max + 1],
                                vec![Fq::ZERO; deg_max + 1],
                            )
                        },
                        |(mut coeffs, mut inner, mut term_poly), t| {
                            let idx = 2 * t;
                            // eq_beta_r(X) = e0 + e1·X
                            let e0 = self.eq_beta_r_tbl[idx].real();
                            let e1 = self.eq_beta_r_tbl[idx + 1].real() - e0;

                            // inner(X) = f_prime(X) + nc_total(X)
                            inner.fill(Fq::ZERO);

                            // f_prime(X): expand sparse polynomial with affine substitutions.
                            for term in &self.f_terms {
                                term_poly.fill(Fq::ZERO);
                                term_poly[0] = term.coeff.real();
                                let mut current_deg = 0usize;
                                for &(var_pos, exp) in &term.vars {
                                    let tbl = &self.f_var_tables[var_pos];
                                    // v(X) = a + b·X
                                    let a = tbl[idx].real();
                                    let b = tbl[idx + 1].real() - a;
                                    for _ in 0..exp {
                                        Self::poly_mul_affine_inplace_base(&mut term_poly, a, b, current_deg);
                                        current_deg += 1;
                                    }
                                }
                                for i in 0..=core::cmp::min(current_deg, deg_max) {
                                    inner[i] += term_poly[i];
                                }
                            }

                            // coeffs += eq_beta_r(X) * inner(X)
                            coeffs[0] += e0 * inner[0];
                            for d in 1..=deg_max {
                                coeffs[d] += (e0 * inner[d]) + (e1 * inner[d - 1]);
                            }

                            // Eval: eq_r_inputs(X) * gamma_to_k * eval_tbl(X) (quadratic).
                            if let (Some(eq_tbl), Some(eval_tbl)) =
                                (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref())
                            {
                                let r0 = eq_tbl[idx].real();
                                let r1 = eq_tbl[idx + 1].real() - r0;
                                let v0 = eval_tbl[idx].real();
                                let v1 = eval_tbl[idx + 1].real() - v0;

                                let g = self.gamma_to_k.real();
                                coeffs[0] += g * (r0 * v0);
                                coeffs[1] += g * (r0 * v1 + r1 * v0);
                                coeffs[2] += g * (r1 * v1);
                            }

                            (coeffs, inner, term_poly)
                        },
                    )
                    .map(|(coeffs, _, _)| coeffs)
                    .reduce(
                        || vec![Fq::ZERO; deg_max + 1],
                        |mut a, b| {
                            for i in 0..=deg_max {
                                a[i] += b[i];
                            }
                            a
                        },
                    )
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                coeffs_seq(tail_len)
            }
        } else {
            coeffs_seq(tail_len)
        };

        xs_base
            .iter()
            .map(|&x| K::from(Self::poly_eval_base(&coeffs, x)))
            .collect()
    }

    fn evals_row_phase_b3_base(&self, tail_len: usize, xs: &[K]) -> Vec<K> {
        let xs_base: Vec<Fq> = xs.iter().map(|&x| x.real()).collect();

        let f_max_term_deg: usize = self
            .f_terms
            .iter()
            .map(|term| {
                term.vars
                    .iter()
                    .map(|&(_, exp)| exp as usize)
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0);
        // eq_beta_r(X) adds one degree; Eval block is quadratic.
        let deg_max = core::cmp::max(2, f_max_term_deg + 1);

        const PAR_THRESHOLD: usize = 1 << 14;
        let coeffs_seq = |tail_len: usize| -> Vec<Fq> {
            let mut coeffs = vec![Fq::ZERO; deg_max + 1];
            let mut inner = vec![Fq::ZERO; deg_max + 1];
            let mut term_poly = vec![Fq::ZERO; deg_max + 1];

            for t in 0..tail_len {
                let idx = 2 * t;
                let e0 = self.eq_beta_r_tbl[idx].real();
                let e1 = self.eq_beta_r_tbl[idx + 1].real() - e0;

                inner.fill(Fq::ZERO);

                for term in &self.f_terms {
                    term_poly.fill(Fq::ZERO);
                    term_poly[0] = term.coeff.real();
                    let mut current_deg = 0usize;
                    for &(var_pos, exp) in &term.vars {
                        let tbl = &self.f_var_tables[var_pos];
                        let a = tbl[idx].real();
                        let b = tbl[idx + 1].real() - a;
                        for _ in 0..exp {
                            Self::poly_mul_affine_inplace_base(&mut term_poly, a, b, current_deg);
                            current_deg += 1;
                        }
                    }
                    for i in 0..=core::cmp::min(current_deg, deg_max) {
                        inner[i] += term_poly[i];
                    }
                }

                coeffs[0] += e0 * inner[0];
                for d in 1..=deg_max {
                    coeffs[d] += (e0 * inner[d]) + (e1 * inner[d - 1]);
                }

                if let (Some(eq_tbl), Some(eval_tbl)) = (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref()) {
                    let r0 = eq_tbl[idx].real();
                    let r1 = eq_tbl[idx + 1].real() - r0;
                    let v0 = eval_tbl[idx].real();
                    let v1 = eval_tbl[idx + 1].real() - v0;

                    let g = self.gamma_to_k.real();
                    coeffs[0] += g * (r0 * v0);
                    coeffs[1] += g * (r0 * v1 + r1 * v0);
                    coeffs[2] += g * (r1 * v1);
                }
            }

            coeffs
        };

        let coeffs = if tail_len >= PAR_THRESHOLD {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                (0..tail_len)
                    .into_par_iter()
                    .fold(
                        || {
                            (
                                vec![Fq::ZERO; deg_max + 1],
                                vec![Fq::ZERO; deg_max + 1],
                                vec![Fq::ZERO; deg_max + 1],
                            )
                        },
                        |(mut coeffs, mut inner, mut term_poly), t| {
                            let idx = 2 * t;
                            // eq_beta_r(X) = e0 + e1·X
                            let e0 = self.eq_beta_r_tbl[idx].real();
                            let e1 = self.eq_beta_r_tbl[idx + 1].real() - e0;

                            // inner(X) = f_prime(X) + nc_total(X)
                            inner.fill(Fq::ZERO);

                            // f_prime(X): expand sparse polynomial with affine substitutions.
                            for term in &self.f_terms {
                                term_poly.fill(Fq::ZERO);
                                term_poly[0] = term.coeff.real();
                                let mut current_deg = 0usize;
                                for &(var_pos, exp) in &term.vars {
                                    let tbl = &self.f_var_tables[var_pos];
                                    // v(X) = a + b·X
                                    let a = tbl[idx].real();
                                    let b = tbl[idx + 1].real() - a;
                                    for _ in 0..exp {
                                        Self::poly_mul_affine_inplace_base(&mut term_poly, a, b, current_deg);
                                        current_deg += 1;
                                    }
                                }
                                for i in 0..=core::cmp::min(current_deg, deg_max) {
                                    inner[i] += term_poly[i];
                                }
                            }

                            // coeffs += eq_beta_r(X) * inner(X)
                            coeffs[0] += e0 * inner[0];
                            for d in 1..=deg_max {
                                coeffs[d] += (e0 * inner[d]) + (e1 * inner[d - 1]);
                            }

                            // Eval: eq_r_inputs(X) * gamma_to_k * eval_tbl(X) (quadratic).
                            if let (Some(eq_tbl), Some(eval_tbl)) =
                                (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref())
                            {
                                let r0 = eq_tbl[idx].real();
                                let r1 = eq_tbl[idx + 1].real() - r0;
                                let v0 = eval_tbl[idx].real();
                                let v1 = eval_tbl[idx + 1].real() - v0;

                                let g = self.gamma_to_k.real();
                                coeffs[0] += g * (r0 * v0);
                                coeffs[1] += g * (r0 * v1 + r1 * v0);
                                coeffs[2] += g * (r1 * v1);
                            }

                            (coeffs, inner, term_poly)
                        },
                    )
                    .map(|(coeffs, _, _)| coeffs)
                    .reduce(
                        || vec![Fq::ZERO; deg_max + 1],
                        |mut a, b| {
                            for i in 0..=deg_max {
                                a[i] += b[i];
                            }
                            a
                        },
                    )
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                coeffs_seq(tail_len)
            }
        } else {
            coeffs_seq(tail_len)
        };

        xs_base
            .iter()
            .map(|&x| K::from(Self::poly_eval_base(&coeffs, x)))
            .collect()
    }

    /// Multiply a polynomial by an affine `(a + b·x)` in-place.
    ///
    /// Coefficients are in low→high order. Output is truncated to the input length.
    #[inline]
    fn poly_mul_affine_inplace(poly: &mut [K], a: K, b: K, current_deg: usize) {
        let mut prev = K::ZERO;
        for coeff in poly.iter_mut().take(current_deg + 2) {
            let old = *coeff;
            *coeff = a * old + b * prev;
            prev = old;
        }
    }

    fn evals_row_phase_impl<Ff>(&self, xs: &[K], allow_base: bool) -> Vec<K>
    where
        Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
        K: From<Ff>,
    {
        debug_assert!(self.cur_len >= 2 && self.cur_len % 2 == 0);
        let tail_len = self.cur_len / 2;
        let xs_are_base = xs.iter().all(|&x| x.imag() == Fq::ZERO);
        let xs_all_base = allow_base && self.all_base && xs_are_base;

        // Fast path for b=2: build the univariate coefficients once per round,
        // then evaluate cheaply at all requested points.
        if self.b == 2 {
            if xs_all_base {
                return self.evals_row_phase_b2_base(tail_len, xs);
            }

            let f_max_term_deg: usize = self
                .f_terms
                .iter()
                .map(|term| {
                    term.vars
                        .iter()
                        .map(|&(_, exp)| exp as usize)
                        .sum::<usize>()
                })
                .max()
                .unwrap_or(0);
            // eq_beta_r(X) adds one degree; Eval block is quadratic.
            let deg_max = core::cmp::max(2, f_max_term_deg + 1);

            let mut coeffs = vec![K::ZERO; deg_max + 1];
            let mut inner = vec![K::ZERO; deg_max + 1];
            let mut term_poly = vec![K::ZERO; deg_max + 1];

            for t in 0..tail_len {
                // eq_beta_r(X) = e0 + e1·X
                let e0 = self.eq_beta_r_tbl[2 * t];
                let e1 = self.eq_beta_r_tbl[2 * t + 1] - e0;

                // inner(X) = f_prime(X)
                inner.fill(K::ZERO);

                // f_prime(X): expand sparse polynomial with affine substitutions.
                for term in &self.f_terms {
                    term_poly.fill(K::ZERO);
                    term_poly[0] = term.coeff;
                    let mut current_deg = 0usize;
                    for &(var_pos, exp) in &term.vars {
                        let tbl = &self.f_var_tables[var_pos];
                        // v(X) = a + b·X
                        let a = tbl[2 * t];
                        let b = tbl[2 * t + 1] - a;
                        for _ in 0..exp {
                            Self::poly_mul_affine_inplace(&mut term_poly, a, b, current_deg);
                            current_deg += 1;
                        }
                    }
                    for i in 0..=core::cmp::min(current_deg, deg_max) {
                        inner[i] += term_poly[i];
                    }
                }

                // coeffs += eq_beta_r(X) * inner(X)
                coeffs[0] += e0 * inner[0];
                for d in 1..=deg_max {
                    coeffs[d] += (e0 * inner[d]) + (e1 * inner[d - 1]);
                }

                // Eval: eq_r_inputs(X) * gamma_to_k * eval_tbl(X) (quadratic).
                if let (Some(eq_tbl), Some(eval_tbl)) = (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref()) {
                    let r0 = eq_tbl[2 * t];
                    let r1 = eq_tbl[2 * t + 1] - r0;
                    let v0 = eval_tbl[2 * t];
                    let v1 = eval_tbl[2 * t + 1] - v0;

                    let g = self.gamma_to_k;
                    coeffs[0] += g * (r0 * v0);
                    if deg_max >= 1 {
                        coeffs[1] += g * (r0 * v1 + r1 * v0);
                    }
                    if deg_max >= 2 {
                        coeffs[2] += g * (r1 * v1);
                    }
                }
            }

            return if xs_are_base {
                xs.iter()
                    .map(|&x| crate::sumcheck::poly_eval_k_base(&coeffs, x.real()))
                    .collect()
            } else {
                xs.iter()
                    .map(|&x| crate::sumcheck::poly_eval_k(&coeffs, x))
                    .collect()
            };
        }

        // Fast path for b=3: range polynomial is N(y) = y(y^2-1)(y^2-4) = y^5 - 5y^3 + 4y.
        // As in the b=2 case, we build the univariate coefficients once per round and then
        // evaluate at all requested points.
        if self.b == 3 {
            if xs_all_base {
                return self.evals_row_phase_b3_base(tail_len, xs);
            }

            let f_max_term_deg: usize = self
                .f_terms
                .iter()
                .map(|term| {
                    term.vars
                        .iter()
                        .map(|&(_, exp)| exp as usize)
                        .sum::<usize>()
                })
                .max()
                .unwrap_or(0);
            // eq_beta_r(X) adds one degree; Eval block is quadratic.
            let deg_max = core::cmp::max(2, f_max_term_deg + 1);

            let coeffs = {
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                {
                    (0..tail_len)
                        .into_par_iter()
                        .fold(
                            || {
                                (
                                    vec![K::ZERO; deg_max + 1],
                                    vec![K::ZERO; deg_max + 1],
                                    vec![K::ZERO; deg_max + 1],
                                )
                            },
                            |(mut coeffs, mut inner, mut term_poly), t| {
                                // eq_beta_r(X) = e0 + e1·X
                                let e0 = self.eq_beta_r_tbl[2 * t];
                                let e1 = self.eq_beta_r_tbl[2 * t + 1] - e0;

                                // inner(X) = f_prime(X)
                                inner.fill(K::ZERO);

                                // f_prime(X): expand sparse polynomial with affine substitutions.
                                for term in &self.f_terms {
                                    term_poly.fill(K::ZERO);
                                    term_poly[0] = term.coeff;
                                    let mut current_deg = 0usize;
                                    for &(var_pos, exp) in &term.vars {
                                        let tbl = &self.f_var_tables[var_pos];
                                        // v(X) = a + b·X
                                        let a = tbl[2 * t];
                                        let b = tbl[2 * t + 1] - a;
                                        for _ in 0..exp {
                                            Self::poly_mul_affine_inplace(&mut term_poly, a, b, current_deg);
                                            current_deg += 1;
                                        }
                                    }
                                    for i in 0..=core::cmp::min(current_deg, deg_max) {
                                        inner[i] += term_poly[i];
                                    }
                                }

                                // coeffs += eq_beta_r(X) * inner(X)
                                coeffs[0] += e0 * inner[0];
                                for d in 1..=deg_max {
                                    coeffs[d] += (e0 * inner[d]) + (e1 * inner[d - 1]);
                                }

                                // Eval: eq_r_inputs(X) * gamma_to_k * eval_tbl(X) (quadratic).
                                if let (Some(eq_tbl), Some(eval_tbl)) =
                                    (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref())
                                {
                                    let r0 = eq_tbl[2 * t];
                                    let r1 = eq_tbl[2 * t + 1] - r0;
                                    let v0 = eval_tbl[2 * t];
                                    let v1 = eval_tbl[2 * t + 1] - v0;

                                    let g = self.gamma_to_k;
                                    coeffs[0] += g * (r0 * v0);
                                    coeffs[1] += g * (r0 * v1 + r1 * v0);
                                    coeffs[2] += g * (r1 * v1);
                                }

                                (coeffs, inner, term_poly)
                            },
                        )
                        .map(|(coeffs, _, _)| coeffs)
                        .reduce(
                            || vec![K::ZERO; deg_max + 1],
                            |mut a, b| {
                                for i in 0..=deg_max {
                                    a[i] += b[i];
                                }
                                a
                            },
                        )
                }
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                {
                    let mut coeffs = vec![K::ZERO; deg_max + 1];
                    let mut inner = vec![K::ZERO; deg_max + 1];
                    let mut term_poly = vec![K::ZERO; deg_max + 1];

                    for t in 0..tail_len {
                        // eq_beta_r(X) = e0 + e1·X
                        let e0 = self.eq_beta_r_tbl[2 * t];
                        let e1 = self.eq_beta_r_tbl[2 * t + 1] - e0;

                        // inner(X) = f_prime(X)
                        inner.fill(K::ZERO);

                        // f_prime(X): expand sparse polynomial with affine substitutions.
                        for term in &self.f_terms {
                            term_poly.fill(K::ZERO);
                            term_poly[0] = term.coeff;
                            let mut current_deg = 0usize;
                            for &(var_pos, exp) in &term.vars {
                                let tbl = &self.f_var_tables[var_pos];
                                // v(X) = a + b·X
                                let a = tbl[2 * t];
                                let b = tbl[2 * t + 1] - a;
                                for _ in 0..exp {
                                    Self::poly_mul_affine_inplace(&mut term_poly, a, b, current_deg);
                                    current_deg += 1;
                                }
                            }
                            for i in 0..=core::cmp::min(current_deg, deg_max) {
                                inner[i] += term_poly[i];
                            }
                        }

                        // coeffs += eq_beta_r(X) * inner(X)
                        coeffs[0] += e0 * inner[0];
                        for d in 1..=deg_max {
                            coeffs[d] += (e0 * inner[d]) + (e1 * inner[d - 1]);
                        }

                        // Eval: eq_r_inputs(X) * gamma_to_k * eval_tbl(X) (quadratic).
                        if let (Some(eq_tbl), Some(eval_tbl)) = (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref())
                        {
                            let r0 = eq_tbl[2 * t];
                            let r1 = eq_tbl[2 * t + 1] - r0;
                            let v0 = eval_tbl[2 * t];
                            let v1 = eval_tbl[2 * t + 1] - v0;

                            let g = self.gamma_to_k;
                            coeffs[0] += g * (r0 * v0);
                            coeffs[1] += g * (r0 * v1 + r1 * v0);
                            coeffs[2] += g * (r1 * v1);
                        }
                    }

                    coeffs
                }
            };

            return if xs_are_base {
                xs.iter()
                    .map(|&x| crate::sumcheck::poly_eval_k_base(&coeffs, x.real()))
                    .collect()
            } else {
                xs.iter()
                    .map(|&x| crate::sumcheck::poly_eval_k(&coeffs, x))
                    .collect()
            };
        }

        // Generic fallback: evaluate directly at each x (slower, but supports any b).
        let f_arity = self.f_var_tables.len();

        // `xs` is typically very small (sumcheck evaluation points), so Rayon overhead dominates here.
        xs.iter()
            .map(|&x| {
                let one_minus = K::ONE - x;
                let mut var_vals = vec![K::ZERO; f_arity];
                let mut sum_x = K::ZERO;

                for t in 0..tail_len {
                    let eq_beta_r = one_minus * self.eq_beta_r_tbl[2 * t] + x * self.eq_beta_r_tbl[2 * t + 1];

                    // f variables at (prefix, x, tail)
                    for (pos, tbl) in self.f_var_tables.iter().enumerate() {
                        var_vals[pos] = one_minus * tbl[2 * t] + x * tbl[2 * t + 1];
                    }

                    // f_prime = f(m_vals)
                    let mut f_prime = K::ZERO;
                    for term in &self.f_terms {
                        let mut acc = term.coeff;
                        for &(var_pos, exp) in &term.vars {
                            let xi = var_vals[var_pos];
                            let mut p = xi;
                            for _ in 1..exp {
                                p *= xi;
                            }
                            acc *= p;
                        }
                        f_prime += acc;
                    }

                    let mut out = eq_beta_r * f_prime;

                    // Eval: eq_r_inputs(r') * gamma_to_k * eval_tbl(r')
                    if let (Some(eq_tbl), Some(eval_tbl)) = (self.eq_r_inputs_tbl.as_ref(), self.eval_tbl.as_ref()) {
                        let eq_r_inputs = one_minus * eq_tbl[2 * t] + x * eq_tbl[2 * t + 1];
                        if eq_r_inputs != K::ZERO {
                            let e = one_minus * eval_tbl[2 * t] + x * eval_tbl[2 * t + 1];
                            out += eq_r_inputs * (self.gamma_to_k * e);
                        }
                    }

                    sum_x += out;
                }

                sum_x
            })
            .collect()
    }

    #[inline]
    fn evals_row_phase<Ff>(&self, xs: &[K]) -> Vec<K>
    where
        Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
        K: From<Ff>,
    {
        self.evals_row_phase_impl::<Ff>(xs, true)
    }

    #[inline]
    fn evals_row_phase_force_generic<Ff>(&self, xs: &[K]) -> Vec<K>
    where
        Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
        K: From<Ff>,
    {
        self.evals_row_phase_impl::<Ff>(xs, false)
    }
}

/// Symmetric range polynomial: ∏_{t=-(b-1)}^{b-1} (y - t) = y · ∏_{t=1}^{b-1} (y² - t²)
/// using cached `t²` values for `t=1..(b-1)`.
#[inline]
fn range_product_cached(y: K, range_t_sq: &[K]) -> K {
    if range_t_sq.is_empty() {
        return y;
    }
    let y2 = y * y;
    let mut prod = y;
    for &tt2 in range_t_sq {
        prod *= y2 - tt2;
    }
    prod
}

#[inline]
fn eq_lin(a: K, b: K) -> K {
    (K::ONE - a) * (K::ONE - b) + a * b
}

/// Fold one Ajtai bit into-place for a digits table (size D).
#[inline]
fn fold_bit_inplace(digits: &mut [K; D], bit: usize, a: K) {
    let stride = 1usize << bit;
    let step = stride << 1;
    let n = D;
    let mut base = 0usize;
    while base < n {
        let mut off = 0usize;
        while off < stride {
            let i0 = base + off;
            if i0 >= n {
                break;
            }
            let i1 = i0 + stride;
            let lo = digits[i0];
            let hi = if i1 < n { digits[i1] } else { K::ZERO };
            digits[i0] = lo + (hi - lo) * a;
            off += 1;
        }
        base += step;
    }
}

/// Fold the current Ajtai bit into `digits_pref` (which already has the prefix folded),
/// then compute the tail-weighted sum of the resulting MLE "heads".
#[inline]
fn ajtai_tail_weighted_dot_prefolded(digits_pref: &[K; D], x: K, bit: usize, head_stride: usize, w_tail: &[K]) -> K {
    let mut tmp = *digits_pref;
    fold_bit_inplace(&mut tmp, bit, x);
    let mut acc = K::ZERO;
    for (t, &w) in w_tail.iter().enumerate() {
        let idx = t * head_stride;
        if idx < D {
            acc += w * tmp[idx];
        }
    }
    acc
}

/// Fold the current Ajtai bit into `digits_pref` (which already has the prefix folded),
/// then compute the tail-weighted sum of the range polynomial N(·) over the MLE heads.
#[inline]
fn ajtai_tail_weighted_range_prefolded(
    digits_pref: &[K; D],
    x: K,
    bit: usize,
    head_stride: usize,
    w_tail: &[K],
    range_t_sq: &[K],
) -> K {
    let mut tmp = *digits_pref;
    fold_bit_inplace(&mut tmp, bit, x);
    let mut acc = K::ZERO;
    for (t, &w) in w_tail.iter().enumerate() {
        let idx = t * head_stride;
        if idx < D {
            acc += w * range_product_cached(tmp[idx], range_t_sq);
        }
    }
    acc
}

#[inline]
fn chi_tail_weights(bits: &[K]) -> Vec<K> {
    let t = bits.len();
    let len = 1usize << t;
    let mut w = vec![K::ZERO; len];
    w[0] = K::ONE;
    for (i, &b) in bits.iter().enumerate() {
        let step = 1usize << i;
        let one_minus = K::ONE - b;
        for mask in 0..step {
            let v = w[mask];
            w[mask] = v * one_minus;
            w[mask + step] = v * b;
        }
    }
    w
}

/// Precomputation for a fixed r' (row assignment) - eliminates redundant v_j recomputation
struct RPrecomp {
    /// Y_eval[i][j][ρ] = (Z_i · v_j)[ρ] for Eval terms  
    y_eval: Vec<Vec<[K; D]>>,
    /// F' = f(z_1 · v_j) - independent of α'
    f_prime: K,
    /// eq(r', β_r) - independent of α'
    eq_beta_r: K,
    /// eq(r', r_inputs) if present - independent of α'
    eq_r_inputs: K,
}

/// Helper: compute eq for a boolean mask against a field vector
#[inline]
fn eq_points_bool_mask(mask: usize, points: &[K]) -> K {
    let mut prod = K::ONE;
    for (bit_idx, &p) in points.iter().enumerate() {
        let is_one = ((mask >> bit_idx) & 1) == 1;
        prod *= if is_one { p } else { K::ONE - p };
    }
    prod
}

pub struct OptimizedOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<F>,
{
    pub s: &'a CcsStructure<F>,
    pub params: &'a neo_params::NeoParams,
    // Witnesses in the same order as the engine: all MCS first, then ME
    pub mcs_witnesses: &'a [McsWitness<F>],
    pub me_witnesses: &'a [Mat<F>],
    // Challenges (α, β, γ)
    pub ch: Challenges,
    // Shared dims and degree bound for sumcheck
    pub ell_d: usize,
    pub ell_n: usize,
    pub d_sc: usize,
    // Round tracking
    pub round_idx: usize,
    // Collected row and Ajtai challenges r' and α'
    pub row_chals: Vec<K>,
    pub ajtai_chals: Vec<K>,
    // Input ME r (if any) for Eval gating
    pub r_inputs: Option<Vec<K>>,
    // Cached sparse formats for efficient matrix-vector products
    pub sparse: Arc<SparseCache<F>>,

    // Streaming row-phase state (folded in-place across row rounds)
    row_stream: RowStreamState,

    // Cached row-only precomputation for Ajtai rounds (r' fixed after row phase).
    ajtai_precomp: Option<RPrecomp>,
}

impl<'a, F> OptimizedOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<F>,
{
    pub fn new_with_sparse(
        s: &'a CcsStructure<F>,
        params: &'a neo_params::NeoParams,
        mcs_witnesses: &'a [McsWitness<F>],
        me_witnesses: &'a [Mat<F>],
        ch: Challenges,
        ell_d: usize,
        ell_n: usize,
        d_sc: usize,
        r_inputs: Option<&[K]>,
        sparse: Arc<SparseCache<F>>,
    ) -> Self {
        assert!(!mcs_witnesses.is_empty(), "need at least one MCS instance for F-term");

        let row_stream = RowStreamState::build(
            s,
            params.b,
            &ch,
            ell_d,
            ell_n,
            mcs_witnesses,
            me_witnesses,
            r_inputs,
            sparse.as_ref(),
        );

        Self {
            s,
            params,
            mcs_witnesses,
            me_witnesses,
            ch,
            ell_d,
            ell_n,
            d_sc,
            round_idx: 0,
            row_chals: Vec::with_capacity(ell_n),
            ajtai_chals: Vec::with_capacity(ell_d),
            r_inputs: r_inputs.map(|r| r.to_vec()),
            sparse,
            row_stream,
            ajtai_precomp: None,
        }
    }

    #[inline]
    fn num_rounds_total(&self) -> usize {
        self.ell_n + self.ell_d
    }

    #[inline]
    fn eq_points(p: &[K], q: &[K]) -> K {
        assert_eq!(p.len(), q.len(), "eq_points: length mismatch");
        let mut acc = K::ONE;
        for i in 0..p.len() {
            let (pi, qi) = (p[i], q[i]);
            acc *= (K::ONE - pi) * (K::ONE - qi) + pi * qi;
        }
        acc
    }

    /// Precompute all data that depends only on r' (not on α') for row phase optimization.
    /// This eliminates redundant v_j recomputation across all boolean α' assignments.
    fn precompute_for_r(&self, r_prime: &[K]) -> RPrecomp {
        let t = self.s.t();

        // Build χ_r table over the Boolean row domain.
        let chi_r = chi_tail_weights(r_prime);
        let n_sz = chi_r.len();

        // Compute eq(r', β_r) and eq(r', r_inputs)
        let eq_beta_r = Self::eq_points(r_prime, &self.ch.beta_r);
        let eq_r_inputs = match self.r_inputs {
            Some(ref r_in) => Self::eq_points(r_prime, r_in),
            None => K::ZERO,
        };

        // Compute all v_j = M_j^T · χ_r' once (sparse representation).
        //
        // NOTE: `evals_row_phase` already parallelizes over the remaining Boolean row assignments,
        // so this is intentionally sequential to avoid nested rayon parallelism.
        let n_eff = core::cmp::min(self.s.n, n_sz);
        let sparse = self.sparse.as_ref();
        if sparse.len() != t {
            panic!("optimized oracle sparse cache: matrix count mismatch");
        }

        // vjs_nz[j] = list of (col, vj[col]) pairs where vj[col] != 0.
        let mut vjs_nz: Vec<Vec<(usize, K)>> = Vec::with_capacity(t);

        // Identity matrix: v[c] = χ_r[c] for c < min(m, n_eff).
        let cap_id = core::cmp::min(self.s.m, n_eff);
        let mut v_id_nz = Vec::with_capacity(cap_id);
        for c in 0..cap_id {
            let v = chi_r[c];
            if v != K::ZERO {
                v_id_nz.push((c, v));
            }
        }

        for j in 0..t {
            // Identity sentinel: v_j = χ_r.
            let Some(csc) = sparse.csc(j) else {
                vjs_nz.push(v_id_nz.clone());
                continue;
            };

            let mut nz = Vec::<(usize, K)>::new();
            for c in 0..csc.ncols {
                let s = csc.col_ptr[c];
                let e = csc.col_ptr[c + 1];
                if s == e {
                    continue;
                }
                let mut acc = K::ZERO;
                for k in s..e {
                    let r = csc.row_idx[k];
                    if r < n_eff {
                        acc += chi_r[r].scale_base_k(K::from(csc.vals[k]));
                    }
                }
                if acc != K::ZERO {
                    nz.push((c, acc));
                }
            }
            vjs_nz.push(nz);
        }

        // Compute F' = f(z_1 · v_j) - independent of α'
        // (z1 is precomputed once per oracle build in `RowStreamState`).
        let z1 = &self.row_stream.z1;

        let mut m_vals = vec![K::ZERO; t];
        for (j, vj) in vjs_nz.iter().enumerate() {
            let mut acc = K::ZERO;
            for &(c, v) in vj {
                acc += z1[c] * v;
            }
            m_vals[j] = acc;
        }
        let f_prime = self.s.f.eval_in_ext::<K>(&m_vals);

        let all_witnesses: Vec<&Mat<F>> = self
            .mcs_witnesses
            .iter()
            .map(|w| &w.Z)
            .chain(self.me_witnesses.iter())
            .collect();

        // Precompute Y_eval[i][j][ρ] = (Z_i · v_j)[ρ] for all instances and matrices.
        let y_eval: Vec<Vec<[K; D]>> = {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                all_witnesses
                    .par_iter()
                    .map(|Zi| {
                        (0..t)
                            .map(|j| {
                                let mut y_row = [K::ZERO; D];
                                for rho in 0..D {
                                    let mut acc = K::ZERO;
                                    let z_row = Zi.row(rho);
                                    for &(c, v) in &vjs_nz[j] {
                                        acc += v.scale_base_k(K::from(z_row[c]));
                                    }
                                    y_row[rho] = acc;
                                }
                                y_row
                            })
                            .collect()
                    })
                    .collect()
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                all_witnesses
                    .iter()
                    .map(|Zi| {
                        (0..t)
                            .map(|j| {
                                let mut y_row = [K::ZERO; D];
                                for rho in 0..D {
                                    let mut acc = K::ZERO;
                                    let z_row = Zi.row(rho);
                                    for &(c, v) in &vjs_nz[j] {
                                        acc += v.scale_base_k(K::from(z_row[c]));
                                    }
                                    y_row[rho] = acc;
                                }
                                y_row
                            })
                            .collect()
                    })
                    .collect()
            }
        };

        RPrecomp {
            y_eval,
            f_prime,
            eq_beta_r,
            eq_r_inputs,
        }
    }

    /// Compute the univariate round polynomial values at given xs for a row-bit round
    /// by summing Q over the remaining Boolean variables, with the current variable set to x.
    fn evals_row_phase(&self, xs: &[K]) -> Vec<K> {
        debug_assert!(self.round_idx < self.ell_n, "row phase after all row bits");
        let expect_len = 1usize << (self.ell_n - self.round_idx);
        debug_assert_eq!(
            self.row_stream.cur_len, expect_len,
            "row_stream out of sync with round_idx"
        );
        self.row_stream.evals_row_phase::<F>(xs)
    }

    #[doc(hidden)]
    pub fn __test_row_phase_base_vs_generic(&self, xs: &[K]) -> (Vec<K>, Vec<K>) {
        debug_assert!(self.round_idx < self.ell_n, "__test_row_phase_* requires row phase");
        let base = self.row_stream.evals_row_phase::<F>(xs);
        let generic = self.row_stream.evals_row_phase_force_generic::<F>(xs);
        (base, generic)
    }

    #[doc(hidden)]
    pub fn __test_row_stream_all_base(&self) -> bool {
        self.row_stream.all_base
    }

    /// Compute the univariate round polynomial for an Ajtai-bit round.
    /// DP version: removes the 2^{free_a}·D work per x and keeps outputs bit-identical.
    fn evals_ajtai_phase(&mut self, xs: &[K]) -> Vec<K> {
        let j = self.round_idx - self.ell_n;
        debug_assert!(j < self.ell_d, "ajtai phase after all Ajtai bits");

        let free_a = self.ell_d - j - 1;
        let r_vec = &self.row_chals;

        // r'-only precomp reused across all Ajtai rounds (r' is fixed after row phase).
        if self.ajtai_precomp.is_none() {
            self.ajtai_precomp = Some(self.precompute_for_r(r_vec));
        }
        let pre = self
            .ajtai_precomp
            .as_ref()
            .expect("ajtai_precomp just populated");

        let k_total = self.mcs_witnesses.len() + self.me_witnesses.len();
        let t_mats = self.s.t();

        // Tail weights (independent of x)
        let w_beta_tail = chi_tail_weights(&self.ch.beta_a[j + 1..self.ell_d]);
        let w_alpha_tail = chi_tail_weights(&self.ch.alpha[j + 1..self.ell_d]);
        let tail_len = 1usize << free_a;
        debug_assert_eq!(w_beta_tail.len(), tail_len);
        debug_assert_eq!(w_alpha_tail.len(), tail_len);
        let head_stride = 1usize << (j + 1);

        // Prefix factors (independent of x)
        let mut eq_beta_pref = K::ONE;
        let mut eq_alpha_pref = K::ONE;
        for i in 0..j {
            eq_beta_pref *= eq_lin(self.ajtai_chals[i], self.ch.beta_a[i]);
            eq_alpha_pref *= eq_lin(self.ajtai_chals[i], self.ch.alpha[i]);
        }

        // Gamma powers (independent of x)
        let mut gamma_pow_i = vec![K::ONE; k_total];
        for i in 1..k_total {
            gamma_pow_i[i] = gamma_pow_i[i - 1] * self.ch.gamma;
        }

        let mut gamma_to_k = K::ONE;
        for _ in 0..k_total {
            gamma_to_k *= self.ch.gamma;
        }

        let mut gamma_k_pow_j = vec![K::ONE; t_mats];
        for jj in 1..t_mats {
            gamma_k_pow_j[jj] = gamma_k_pow_j[jj - 1] * gamma_to_k;
        }

        let prefix = &self.ajtai_chals[..j];
        let beta_j = self.ch.beta_a[j];
        let alpha_j = self.ch.alpha[j];
        // Flattened as [i_abs * t_mats + j_mat] to avoid nested Vec allocations.
        let mut y_eval_pref = Vec::<[K; D]>::with_capacity(k_total * t_mats);
        for i_abs in 0..k_total {
            for j_mat in 0..t_mats {
                let mut digits = pre.y_eval[i_abs][j_mat];
                for b in 0..j {
                    fold_bit_inplace(&mut digits, b, prefix[b]);
                }
                y_eval_pref.push(digits);
            }
        }

        let has_inputs = self.r_inputs.is_some();

        let eval_at = |x: K| {
            // eq((α',r'), β) factor across α' = (prefix, x, tail)
            let eq_beta_px = eq_beta_pref * eq_lin(x, beta_j);
            let eq_beta = pre.eq_beta_r * eq_beta_px;

            // eq((α',r'), (α,r)) factor if inputs present
            let eq_ar_px = if has_inputs {
                pre.eq_r_inputs * (eq_alpha_pref * eq_lin(x, alpha_j))
            } else {
                K::ZERO
            };

            // Base: eq_beta * F'
            let mut out = eq_beta * pre.f_prime;

            // --- Eval block: γ^k · eq_ar · Σ_{j_mat,i≥2} γ^{i-1} (γ^k)^{j_mat} · Σ_tail w_alpha(tail) · ẏ_{(i,j)}(...)
            if k_total >= 2 && eq_ar_px != K::ZERO {
                let mut inner = K::ZERO;
                for j_mat in 0..t_mats {
                    let mut sum_j = K::ZERO;
                    for i_abs in 1..k_total {
                        let digits = &y_eval_pref[i_abs * t_mats + j_mat];
                        let ydot = ajtai_tail_weighted_dot_prefolded(digits, x, j, head_stride, &w_alpha_tail);
                        sum_j += gamma_pow_i[i_abs] * gamma_k_pow_j[j_mat] * ydot;
                    }
                    inner += sum_j;
                }
                out += eq_ar_px * (gamma_to_k * inner);
            }

            out
        };

        // `xs` is typically very small (sumcheck evaluation points), so Rayon overhead dominates here.
        xs.iter().map(|&x| eval_at(x)).collect()
    }

    /// Build Π_CCS ME outputs at the finalized row point `r'` using the oracle's cached
    /// `precompute_for_r` results (no dense matrix scans).
    pub fn build_me_outputs_from_ajtai_precomp<L>(
        &mut self,
        mcs_list: &[McsInstance<Cmt, F>],
        me_inputs: &[MeInstance<Cmt, F, K>],
        s_col: &[K],
        fold_digest: [u8; 32],
        l: &L,
    ) -> Vec<MeInstance<Cmt, F, K>>
    where
        L: SModuleHomomorphism<F, Cmt>,
    {
        assert_eq!(
            mcs_list.len(),
            self.mcs_witnesses.len(),
            "ME output builder: mcs_list/mcs_witnesses length mismatch"
        );
        assert_eq!(
            me_inputs.len(),
            self.me_witnesses.len(),
            "ME output builder: me_inputs/me_witnesses length mismatch"
        );
        assert_eq!(
            self.row_chals.len(),
            self.ell_n,
            "ME output builder: row challenges not finalized"
        );

        // Ensure r'-only precomputation is available.
        if self.ajtai_precomp.is_none() {
            self.ajtai_precomp = Some(self.precompute_for_r(&self.row_chals));
        }
        let pre = self
            .ajtai_precomp
            .as_ref()
            .expect("ajtai_precomp just populated");

        let d_pad = 1usize << self.ell_d;
        assert!(
            d_pad >= D,
            "ME output builder: expected 2^ell_d >= D (2^{} = {d_pad}, D = {D})",
            self.ell_d
        );

        let chi_s = if s_col.is_empty() {
            None
        } else {
            Some(chi_tail_weights(s_col))
        };

        // Base-b recomposition cache for y_scalars.
        let base = K::from(F::from_u64(self.params.b as u64));
        let mut pow_cache = vec![K::ONE; D];
        for rho in 1..D {
            pow_cache[rho] = pow_cache[rho - 1] * base;
        }
        let recompose = |digits: &[K; D]| -> K {
            let mut acc = K::ZERO;
            for rho in 0..D {
                acc += digits[rho] * pow_cache[rho];
            }
            acc
        };

        let t_mats = self.s.t();
        let mut out = Vec::with_capacity(self.mcs_witnesses.len() + self.me_witnesses.len());

        // MCS outputs (keep order).
        for (idx, (inst, wit)) in mcs_list.iter().zip(self.mcs_witnesses.iter()).enumerate() {
            let X = l.project_x(&wit.Z, inst.m_in);
            let mut y = Vec::with_capacity(t_mats);
            let mut y_scalars = Vec::with_capacity(t_mats);

            for j in 0..t_mats {
                let digits = &pre.y_eval[idx][j];
                let mut yj = vec![K::ZERO; d_pad];
                yj[..D].copy_from_slice(digits);
                y.push(yj);
                y_scalars.push(recompose(digits));
            }

            let y_zcol = if let Some(chi_s) = chi_s.as_ref() {
                debug_assert!(chi_s.len() >= self.s.m, "chi_s too short for CCS width");
                let mut yz = neo_ccs::utils::mat_vec_mul_fk::<F, K>(
                    wit.Z.as_slice(),
                    wit.Z.rows(),
                    wit.Z.cols(),
                    &chi_s[..self.s.m],
                );
                yz.resize(d_pad, K::ZERO);
                yz
            } else {
                Vec::new()
            };

            out.push(MeInstance {
                c_step_coords: vec![],
                u_offset: 0,
                u_len: 0,
                c: inst.c.clone(),
                X,
                r: self.row_chals.clone(),
                s_col: s_col.to_vec(),
                y,
                y_scalars,
                y_zcol,
                m_in: inst.m_in,
                fold_digest,
            });
        }

        // ME outputs (keep order).
        let base_idx = self.mcs_witnesses.len();
        for (me_idx, inp) in me_inputs.iter().enumerate() {
            let idx = base_idx + me_idx;
            let mut y = Vec::with_capacity(t_mats);
            let mut y_scalars = Vec::with_capacity(t_mats);

            for j in 0..t_mats {
                let digits = &pre.y_eval[idx][j];
                let mut yj = vec![K::ZERO; d_pad];
                yj[..D].copy_from_slice(digits);
                y.push(yj);
                y_scalars.push(recompose(digits));
            }

            let y_zcol = if let Some(chi_s) = chi_s.as_ref() {
                debug_assert!(chi_s.len() >= self.s.m, "chi_s too short for CCS width");
                let Zi = &self.me_witnesses[me_idx];
                let mut yz =
                    neo_ccs::utils::mat_vec_mul_fk::<F, K>(Zi.as_slice(), Zi.rows(), Zi.cols(), &chi_s[..self.s.m]);
                yz.resize(d_pad, K::ZERO);
                yz
            } else {
                Vec::new()
            };

            out.push(MeInstance {
                c_step_coords: vec![],
                u_offset: 0,
                u_len: 0,
                c: inp.c.clone(),
                X: inp.X.clone(),
                r: self.row_chals.clone(),
                s_col: s_col.to_vec(),
                y,
                y_scalars,
                y_zcol,
                m_in: inp.m_in,
                fold_digest,
            });
        }

        out
    }
}

impl<'a, F> RoundOracle for OptimizedOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<F>,
{
    fn num_rounds(&self) -> usize {
        self.num_rounds_total()
    }
    fn degree_bound(&self) -> usize {
        self.d_sc
    }

    fn evals_at(&mut self, xs: &[K]) -> Vec<K> {
        if self.round_idx < self.ell_n {
            self.evals_row_phase(xs)
        } else {
            self.evals_ajtai_phase(xs)
        }
    }

    fn fold(&mut self, r_i: K) {
        if self.round_idx < self.ell_n {
            self.row_chals.push(r_i);
            self.row_stream.fold_inplace(r_i);
        } else {
            self.ajtai_chals.push(r_i);
        }
        self.round_idx += 1;
    }
}
