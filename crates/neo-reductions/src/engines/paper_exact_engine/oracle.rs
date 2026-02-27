//! Paper-exact RoundOracle for the SplitNcV1 FE channel.
//!
//! This oracle evaluates the FE-only polynomial by brute-force summing over the
//! remaining Boolean variables each round. It is suitable for testing and
//! cross-checking correctness against the optimized engine.
//!
//! Variable order (rounds): first the `ell_n` row bits, then the `ell_d` Ajtai bits.
//! NC/range terms are handled by the separate NC sumcheck channel in SplitNcV1.

#![allow(non_snake_case)]

use neo_math::{D, K};
use p3_field::{Field, PrimeCharacteristicRing};

use crate::optimized_engine::Challenges;
use crate::sumcheck::RoundOracle;
use neo_ccs::{CcsStructure, CcsWitness, Mat};

#[cfg(feature = "paper-exact")]
pub struct PaperExactOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<F>,
{
    pub s: &'a CcsStructure<F>,
    pub params: &'a neo_params::NeoParams,
    // Witnesses in the same order as the engine: all MCS first, then ME
    pub mcs_witnesses: &'a [CcsWitness<F>],
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
    // Cached SuperNeo evaluator backend (formula unchanged).
    superneo_cache: crate::superneo_eval::SuperneoEvalCache,
}

#[cfg(feature = "paper-exact")]
impl<'a, F> PaperExactOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<F>,
{
    pub fn new(
        s: &'a CcsStructure<F>,
        params: &'a neo_params::NeoParams,
        mcs_witnesses: &'a [CcsWitness<F>],
        me_witnesses: &'a [Mat<F>],
        ch: Challenges,
        ell_d: usize,
        ell_n: usize,
        d_sc: usize,
        r_inputs: Option<&[K]>,
    ) -> Self {
        assert!(!mcs_witnesses.is_empty(), "need at least one MCS instance for F-term");
        let superneo_cache = crate::superneo_eval::build_superneo_eval_cache(s).unwrap_or_else(|| {
            panic!(
                "PaperExactOracle requires SuperNeo-compatible CCS shape (m={}, matrices={})",
                s.m,
                s.matrices.len()
            )
        });
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
            superneo_cache,
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

    #[inline]
    fn eval_all_mats_ct_only(&self, z: &[K], chi_r: &[K]) -> Vec<K> {
        // FE channel uses only ct(M_j z); full ring rows are unnecessary here.
        crate::superneo_eval::eval_all_mats_cached(&self.superneo_cache, z, chi_r, self.s.n)
    }

    /// Evaluate FE-only Q at extension point (α′, r′), including Eval block.
    ///
    /// This SplitNcV1 FE oracle intentionally uses constant-term projections (`ct`) from
    /// ring evaluations as its scalar openings and excludes NC/range terms.
    ///
    /// Formula:
    ///   Q_fe = eq((α′,r′),β)·F' + γ^k Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · E_{(i,j)}
    /// with E_{(i,j)} = eq((α′,r′),(α,r)) · ẏ'_{(i,j)}(α′).
    /// NC/range terms are intentionally excluded in this FE channel oracle.
    fn eval_q_ext(&self, alpha_prime: &[K], r_prime: &[K]) -> K {
        // Build χ tables for α′ and r′
        let d_sz = 1usize << alpha_prime.len();
        let n_sz = 1usize << r_prime.len();
        assert!(
            d_sz >= D,
            "PaperExactOracle::eval_q_ext: alpha dimension too small (2^|alpha'|={} < D={})",
            d_sz,
            D
        );

        let mut chi_a = vec![K::ZERO; d_sz];
        for rho in 0..d_sz {
            let mut w = K::ONE;
            for bit in 0..alpha_prime.len() {
                let a = alpha_prime[bit];
                let is_one = ((rho >> bit) & 1) == 1;
                w *= if is_one { a } else { K::ONE - a };
            }
            chi_a[rho] = w;
        }

        let mut chi_r = vec![K::ZERO; n_sz];
        for row in 0..n_sz {
            let mut w = K::ONE;
            for bit in 0..r_prime.len() {
                let r = r_prime[bit];
                let is_one = ((row >> bit) & 1) == 1;
                w *= if is_one { r } else { K::ONE - r };
            }
            chi_r[row] = w;
        }

        // eq((α′,r′), β) and eq((α′,r′),(α,r))
        let eq_beta = Self::eq_points(alpha_prime, &self.ch.beta_a) * Self::eq_points(r_prime, &self.ch.beta_r);
        let k_mcs = self.mcs_witnesses.len();
        let k_total = k_mcs + self.me_witnesses.len();
        let eq_ar = if k_total > k_mcs {
            let r_in = self
                .r_inputs
                .as_ref()
                .expect("PaperExactOracle::eval_q_ext: missing shared ME input r");
            Self::eq_points(alpha_prime, &self.ch.alpha) * Self::eq_points(r_prime, r_in)
        } else {
            K::ZERO
        };

        // ---------------------------
        // FE-only:
        // F' := Σ_{i=1..k_mcs} γ^{i-1} · f( Ẽ(M_j z_i)(r') )_j
        // ---------------------------
        let mut gamma_pow_mcs = vec![K::ONE; k_mcs];
        for i in 1..k_mcs {
            gamma_pow_mcs[i] = gamma_pow_mcs[i - 1] * self.ch.gamma;
        }

        let mut F_prime = K::ZERO;
        for (mcs_idx, w) in self.mcs_witnesses.iter().enumerate() {
            let z_i = crate::common::decode_superneo_coeffs_from_witness_mat(&w.Z, self.s.m).unwrap_or_else(|e| {
                panic!(
                    "PaperExactOracle::eval_q_ext: invalid packed MCS witness[{mcs_idx}] shape for m={}: {e}",
                    self.s.m
                )
            });

            let m_vals = self.eval_all_mats_ct_only(&z_i, &chi_r);
            F_prime += gamma_pow_mcs[mcs_idx] * self.s.f.eval_in_ext::<K>(&m_vals);
        }

        // ---------------------------------------
        // Eval block: compute Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · ẏ'_{(i,j)}(α′)
        // and then multiply once by outer γ^k and by eq_ar.
        // ---------------------------------------
        let mut eval_inner_sum = K::ZERO;
        if k_total > k_mcs && eq_ar != K::ZERO {
            // γ^k
            let mut gamma_to_k = K::ONE;
            for _ in 0..k_total {
                gamma_to_k *= self.ch.gamma;
            }

            // Eval block runs over ME slots i ∈ [k_mcs, k_total) (0-based).
            for (i_abs, Zi) in self
                .mcs_witnesses
                .iter()
                .map(|w| &w.Z)
                .chain(self.me_witnesses.iter())
                .enumerate()
                .skip(k_mcs)
            {
                let z_i = crate::common::decode_superneo_coeffs_from_witness_mat(Zi, self.s.m).unwrap_or_else(|e| {
                    panic!(
                        "PaperExactOracle::eval_q_ext: invalid witness shape for m={}: {e}",
                        self.s.m
                    )
                });
                let y_by_j_ring =
                    crate::superneo_eval::eval_all_mats_ring_cached(&self.superneo_cache, &z_i, &chi_r, self.s.n);

                // inner weight = γ^{i-1} · (γ^k)^j  (0-based j)
                let mut gamma_i = K::ONE;
                for _ in 0..i_abs {
                    gamma_i *= self.ch.gamma;
                }
                let mut gamma_k_pow_j = K::ONE;
                for yj in y_by_j_ring.iter().take(self.s.t()) {
                    let mut y_eval = K::ZERO;
                    for rho in 0..core::cmp::min(D, d_sz) {
                        y_eval += yj[rho] * chi_a[rho];
                    }
                    eval_inner_sum += (gamma_i * gamma_k_pow_j) * y_eval;
                    gamma_k_pow_j *= gamma_to_k;
                }
            }

            // Multiply by the outer γ^k and by eq_ar
            eval_inner_sum = eq_ar * (gamma_to_k * eval_inner_sum);
        } else {
            eval_inner_sum = K::ZERO;
        }

        // Assemble FE-only Q_fe(α′, r′) = eq_beta * F' + Eval block.
        eq_beta * F_prime + eval_inner_sum
    }

    /// Compute the univariate round polynomial values at given xs for a row-bit round
    /// by summing Q over the remaining Boolean variables, with the current variable set to x.
    fn evals_row_phase(&self, xs: &[K]) -> Vec<K> {
        let fixed = self.round_idx; // number of fixed row bits so far
        debug_assert!(fixed < self.ell_n, "row phase after all row bits");

        let free_rows = self.ell_n - fixed - 1;
        let tail_sz = 1usize << free_rows;

        // Precompute all Ajtai boolean assignments (full {0,1}^{ell_d})
        let d_sz = 1usize << self.ell_d;
        let mut alphas_bool: Vec<Vec<K>> = Vec::with_capacity(d_sz);
        for a_mask in 0..d_sz {
            let mut a = vec![K::ZERO; self.ell_d];
            for bit in 0..self.ell_d {
                a[bit] = if ((a_mask >> bit) & 1) == 1 { K::ONE } else { K::ZERO };
            }
            alphas_bool.push(a);
        }

        xs.iter()
            .map(|&x| {
                let mut sum_x = K::ZERO;
                for r_tail in 0..tail_sz {
                    let mut r_vec = vec![K::ZERO; self.ell_n];
                    // prefix fixed
                    for i in 0..fixed {
                        r_vec[i] = self.row_chals[i];
                    }
                    // current variable
                    r_vec[fixed] = x;
                    // remaining bits as boolean mask
                    for k in 0..free_rows {
                        let bit = ((r_tail >> k) & 1) == 1;
                        r_vec[fixed + 1 + k] = if bit { K::ONE } else { K::ZERO };
                    }

                    // sum over all Ajtai boolean assignments
                    for a in alphas_bool.iter() {
                        sum_x += self.eval_q_ext(a, &r_vec);
                    }
                }
                sum_x
            })
            .collect()
    }

    /// Compute the univariate round polynomial values at given xs for an Ajtai-bit round
    /// by summing Q over the remaining Ajtai Boolean variables, with the current variable set to x.
    fn evals_ajtai_phase(&self, xs: &[K]) -> Vec<K> {
        let j = self.round_idx - self.ell_n; // number of fixed Ajtai bits so far
        debug_assert!(j < self.ell_d, "ajtai phase after all Ajtai bits");

        let free_a = self.ell_d - j - 1;
        let tail_sz = 1usize << free_a;

        // Fixed row vector is the fully collected row_chals
        let r_vec = self.row_chals.clone();

        xs.iter()
            .map(|&x| {
                let mut sum_x = K::ZERO;
                for a_tail in 0..tail_sz {
                    let mut a_vec = vec![K::ZERO; self.ell_d];
                    // prefix fixed
                    for i in 0..j {
                        a_vec[i] = self.ajtai_chals[i];
                    }
                    // current var
                    a_vec[j] = x;
                    // remaining bits (Boolean)
                    for k in 0..free_a {
                        let bit = ((a_tail >> k) & 1) == 1;
                        a_vec[j + 1 + k] = if bit { K::ONE } else { K::ZERO };
                    }
                    sum_x += self.eval_q_ext(&a_vec, &r_vec);
                }
                sum_x
            })
            .collect()
    }
}

#[cfg(feature = "paper-exact")]
impl<'a, F> RoundOracle for PaperExactOracle<'a, F>
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
        } else {
            self.ajtai_chals.push(r_i);
        }
        self.round_idx += 1;
    }
}
