//! Twist (and Shout) sumcheck oracles built from multilinear factor tables.
//!
//! This module provides oracles for the **index-bit addressing** architecture:
//! instead of materializing huge one-hot tables, we compute eq(bits, r_addr)
//! dynamically from committed bit columns.
//!
//! ## Key Oracles
//!
//! - `ProductRoundOracle`: Generic multilinear product sumcheck (used for address-domain checks)
//! - `AddressLookupOracle`: Shout address-domain lookup sumcheck
//! - `IndexAdapterOracleSparseTime`: Sparse-in-time IDX→OH adapter (time-domain)
//! - `LazyWeightedBitnessOracleSparseTime`: Sparse-in-time aggregated bitness checks (time-domain)
//! - `TwistReadCheckOracleSparseTime` / `TwistWriteCheckOracleSparseTime`: Twist time-lane checks (time-domain)
//! - `TwistValEvalOracleSparseTime` / `TwistTotalIncOracleSparseTime`: Twist val reconstruction (time-domain)

use crate::bit_ops::eq_bit_affine;
use crate::mle::{eq_single, lt_eval};
use crate::sparse_time::SparseIdxVec;
use neo_math::K;
use neo_reductions::sumcheck::RoundOracle;
use p3_field::Field;
use p3_field::PrimeCharacteristicRing;

macro_rules! impl_round_oracle_via_core {
    ($ty:ty) => {
        impl RoundOracle for $ty {
            fn evals_at(&mut self, points: &[K]) -> Vec<K> {
                self.core.evals_at(points)
            }
            fn num_rounds(&self) -> usize {
                self.core.num_rounds()
            }
            fn degree_bound(&self) -> usize {
                self.core.degree_bound()
            }
            fn fold(&mut self, r: K) {
                self.core.fold(r)
            }
        }
    };
}

// ============================================================================
// Core ProductRoundOracle
// ============================================================================

/// Helper that runs sumcheck for a product of multilinear factors.
///
/// Each factor table is a length-2^ℓ vector enumerating the factor on the
/// Boolean hypercube, using little-endian bit order.
pub struct ProductRoundOracle {
    factors: Vec<Vec<K>>,
    rounds_remaining: usize,
    degree_bound: usize,
    challenges: Vec<K>,
}

impl ProductRoundOracle {
    pub fn new(factors: Vec<Vec<K>>, degree_bound: usize) -> Self {
        let len = factors.first().map(|f| f.len()).unwrap_or(1);
        assert!(len.is_power_of_two(), "factor length must be a power of two");
        for f in factors.iter() {
            assert_eq!(f.len(), len, "all factors must have the same length");
        }
        let total_rounds = log2_pow2(len);
        Self {
            factors,
            rounds_remaining: total_rounds,
            degree_bound,
            challenges: Vec::with_capacity(total_rounds),
        }
    }

    pub fn value(&self) -> Option<K> {
        if self.factors.iter().all(|f| f.len() == 1) {
            let mut acc = K::ONE;
            for f in self.factors.iter() {
                acc *= f[0];
            }
            Some(acc)
        } else {
            None
        }
    }

    pub fn challenges(&self) -> &[K] {
        &self.challenges
    }

    pub fn sum_over_hypercube(&self) -> K {
        let n = self.factors.first().map(|f| f.len()).unwrap_or(1);
        let mut sum = K::ZERO;
        for t in 0..n {
            let mut prod = K::ONE;
            for f in &self.factors {
                prod *= f[t];
            }
            sum += prod;
        }
        sum
    }
}

impl RoundOracle for ProductRoundOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.rounds_remaining == 0 {
            // Return the single value for all points
            let val = self
                .value()
                .expect("ProductRoundOracle invariant broken: rounds_remaining==0 but value() is None");
            return vec![val; points.len()];
        }
        let half = 1usize << (self.rounds_remaining - 1);
        let mut ys = vec![K::ZERO; points.len()];

        for (idx_point, &x) in points.iter().enumerate() {
            let mut acc = K::ZERO;
            for pair in 0..half {
                let mut prod = K::ONE;
                for factor in self.factors.iter() {
                    let f0 = factor[2 * pair];
                    let f1 = factor[2 * pair + 1];
                    prod *= f0 + (f1 - f0) * x;
                }
                acc += prod;
            }
            ys[idx_point] = acc;
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.rounds_remaining
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.rounds_remaining == 0 {
            return;
        }
        let half = 1usize << (self.rounds_remaining - 1);
        for factor in self.factors.iter_mut() {
            for i in 0..half {
                let f0 = factor[2 * i];
                let f1 = factor[2 * i + 1];
                factor[i] = f0 + (f1 - f0) * r;
            }
            factor.truncate(half);
        }
        self.rounds_remaining -= 1;
        self.challenges.push(r);
    }
}

// ============================================================================
// Sparse-in-time helpers and oracles (Track A)
// ============================================================================

#[inline]
fn chi_at_bool_index(r: &[K], idx: usize) -> K {
    crate::mle::chi_at_index(r, idx)
}

/// Compute χ_{r_cycle}(t) children for the current time sumcheck round.
///
/// Variable order is little-endian (bit 0 first), matching `ProductRoundOracle`.
#[inline]
fn chi_cycle_children(r_cycle: &[K], bit_idx: usize, prefix_eq: K, pair_idx: usize) -> (K, K) {
    debug_assert!(bit_idx < r_cycle.len());

    // Higher bits (bit_idx+1..ell) come from pair_idx, little-endian.
    let mut suffix = K::ONE;
    let mut shift = 0usize;
    for b in (bit_idx + 1)..r_cycle.len() {
        let bit = (pair_idx >> shift) & 1;
        suffix *= if bit == 1 { r_cycle[b] } else { K::ONE - r_cycle[b] };
        shift += 1;
    }

    let r = r_cycle[bit_idx];
    let child0 = prefix_eq * (K::ONE - r) * suffix;
    let child1 = prefix_eq * r * suffix;
    (child0, child1)
}

fn gather_pairs_from_sparse(entries: &[(usize, K)]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(entries.len());
    let mut prev: Option<usize> = None;
    for &(idx, _v) in entries {
        let p = idx >> 1;
        if prev != Some(p) {
            out.push(p);
            prev = Some(p);
        }
    }
    out
}

/// Sparse Route A oracle for Shout value:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · val(t)
pub struct ShoutValueOracleSparse {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    val: SparseIdxVec<K>,
    degree_bound: usize,
    challenges: Vec<K>,
}

impl ShoutValueOracleSparse {
    pub fn new(r_cycle: &[K], has_lookup: SparseIdxVec<K>, val: SparseIdxVec<K>) -> (Self, K) {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);

        let mut claim = K::ZERO;
        for &(t, gate) in has_lookup.entries() {
            let v = val.get(t);
            if v == K::ZERO {
                continue;
            }
            claim += chi_at_bool_index(r_cycle, t) * gate * v;
        }

        (
            Self {
                bit_idx: 0,
                r_cycle: r_cycle.to_vec(),
                prefix_eq: K::ONE,
                has_lookup,
                val,
                degree_bound: 3,
                challenges: Vec::with_capacity(ell_n),
            },
            claim,
        )
    }
}

impl RoundOracle for ShoutValueOracleSparse {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let v = self.prefix_eq * self.has_lookup.singleton_value() * self.val.singleton_value();
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let (chi0_base, chi1_base) = if self.bit_idx < self.r_cycle.len() {
            // Per-pair child weights depend on higher-bit assignment (pair index).
            (K::ZERO, K::ZERO)
        } else {
            (K::ZERO, K::ZERO)
        };
        let _ = (chi0_base, chi1_base);

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = chi0 + (chi1 - chi0) * x;
                let gate_x = gate0 + (gate1 - gate0) * x;
                let val_x = val0 + (val1 - val0) * x;
                ys[i] += chi_x * gate_x * val_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.val.fold_round_in_place(r);
        self.challenges.push(r);
        self.bit_idx += 1;
    }
}

// ============================================================================
// Packed-key RV32 ADD Shout (time-domain)
// ============================================================================

/// Sparse Route A oracle for RV32 packed ADD correctness:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · (lhs(t) + rhs(t) - val(t) - carry(t)·2^32)
///
/// This is a "no width bloat" alternative to the 64-bit addr-bit Shout encoding for `ADD`:
/// instead of committing to 64 key bits, we commit to packed `lhs/rhs` plus a carry bit.
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedAddOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    carry: SparseIdxVec<K>,
    val: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedAddOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        carry: SparseIdxVec<K>,
        val: SparseIdxVec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(carry.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            rhs,
            carry,
            val,
            degree_bound: 3,
        }
    }
}

impl RoundOracle for Rv32PackedAddOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let two32 = K::from_u64(1u64 << 32);
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let rhs = self.rhs.singleton_value();
            let carry = self.carry.singleton_value();
            let val = self.val.singleton_value();
            let expr = lhs + rhs - val - carry * two32;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let two32 = K::from_u64(1u64 << 32);

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let rhs0 = self.rhs.get(child0);
            let rhs1 = self.rhs.get(child1);
            let carry0 = self.carry.get(child0);
            let carry1 = self.carry.get(child1);
            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);

            let expr0 = lhs0 + rhs0 - val0 - carry0 * two32;
            let expr1 = lhs1 + rhs1 - val1 - carry1 * two32;

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }
                let expr_x = interp(expr0, expr1, x);
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        self.rhs.fold_round_in_place(r);
        self.carry.fold_round_in_place(r);
        self.val.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for RV32 packed SUB correctness:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · (lhs(t) - rhs(t) - val(t) + borrow(t)·2^32)
///
/// This is a "no width bloat" alternative to the 64-bit addr-bit Shout encoding for `SUB`:
/// instead of committing to 64 key bits, we commit to packed `lhs/rhs` plus a borrow bit.
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedSubOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    borrow: SparseIdxVec<K>,
    val: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedSubOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        borrow: SparseIdxVec<K>,
        val: SparseIdxVec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(borrow.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            rhs,
            borrow,
            val,
            degree_bound: 3,
        }
    }
}

impl RoundOracle for Rv32PackedSubOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let two32 = K::from_u64(1u64 << 32);
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let rhs = self.rhs.singleton_value();
            let borrow = self.borrow.singleton_value();
            let val = self.val.singleton_value();
            let expr = lhs - rhs - val + borrow * two32;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let two32 = K::from_u64(1u64 << 32);

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let rhs0 = self.rhs.get(child0);
            let rhs1 = self.rhs.get(child1);
            let borrow0 = self.borrow.get(child0);
            let borrow1 = self.borrow.get(child1);
            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);

            let expr0 = lhs0 - rhs0 - val0 + borrow0 * two32;
            let expr1 = lhs1 - rhs1 - val1 + borrow1 * two32;

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }
                let expr_x = interp(expr0, expr1, x);
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        self.rhs.fold_round_in_place(r);
        self.borrow.fold_round_in_place(r);
        self.val.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

// ============================================================================
// Packed-key RV32 MUL Shout (time-domain)
// ============================================================================

/// Sparse Route A oracle for RV32 packed MUL correctness:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · (lhs(t)·rhs(t) - val(t) - carry(t)·2^32)
///
/// Where:
/// - `carry(t)` is the high 32 bits of the 64-bit product `lhs·rhs`, encoded as 32 Boolean columns,
/// - `val(t)` is the low 32 bits (the `MUL` result).
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedMulOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    carry_bits: Vec<SparseIdxVec<K>>,
    val: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedMulOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        carry_bits: Vec<SparseIdxVec<K>>,
        val: SparseIdxVec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);
        debug_assert_eq!(carry_bits.len(), 32);
        for (i, b) in carry_bits.iter().enumerate() {
            debug_assert_eq!(b.len(), 1usize << ell_n, "carry_bits[{i}] length must match time domain");
        }

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            rhs,
            carry_bits,
            val,
            // Degree bound:
            // - chi(t): multilinear (degree 1)
            // - has_lookup(t): multilinear (degree 1)
            // - lhs(t)·rhs(t): degree 2
            // - val(t), carry(t): multilinear (degree 1)
            // ⇒ total degree ≤ 1 + 1 + 2 = 4
            degree_bound: 4,
        }
    }
}

impl RoundOracle for Rv32PackedMulOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let two32 = K::from_u64(1u64 << 32);

        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let rhs = self.rhs.singleton_value();
            let val = self.val.singleton_value();

            let mut carry = K::ZERO;
            for (i, b) in self.carry_bits.iter().enumerate() {
                carry += b.singleton_value() * K::from_u64(1u64 << i);
            }

            let expr = lhs * rhs - val - carry * two32;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let rhs0 = self.rhs.get(child0);
            let rhs1 = self.rhs.get(child1);
            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);

            // Pre-fetch carry bit endpoints for this pair to avoid repeated sparse lookups per eval point.
            let mut c0s: [K; 32] = [K::ZERO; 32];
            let mut c1s: [K; 32] = [K::ZERO; 32];
            for (i, b) in self.carry_bits.iter().enumerate() {
                c0s[i] = b.get(child0);
                c1s[i] = b.get(child1);
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let lhs_x = interp(lhs0, lhs1, x);
                let rhs_x = interp(rhs0, rhs1, x);
                let val_x = interp(val0, val1, x);

                let mut carry_x = K::ZERO;
                for j in 0..32 {
                    let c_x = interp(c0s[j], c1s[j], x);
                    carry_x += c_x * K::from_u64(1u64 << j);
                }

                let expr_x = lhs_x * rhs_x - val_x - carry_x * two32;
                if expr_x == K::ZERO {
                    continue;
                }

                ys[i] += chi_x * gate_x * expr_x;
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        self.rhs.fold_round_in_place(r);
        for b in self.carry_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        self.val.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

// ============================================================================
// Packed-key RV32 MULHU Shout (time-domain)
// ============================================================================

/// Sparse Route A oracle for RV32 packed MULHU correctness (unsigned high 32 bits):
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · (lhs(t)·rhs(t) - lo(t) - val(t)·2^32)
///
/// Where:
/// - `lo(t)` is the low 32 bits of the 64-bit product `lhs·rhs`, encoded as 32 Boolean columns,
/// - `val(t)` is the upper 32 bits (the `MULHU` result).
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedMulhuOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    lo_bits: Vec<SparseIdxVec<K>>,
    val: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedMulhuOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        lo_bits: Vec<SparseIdxVec<K>>,
        val: SparseIdxVec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);
        debug_assert_eq!(lo_bits.len(), 32);
        for (i, b) in lo_bits.iter().enumerate() {
            debug_assert_eq!(b.len(), 1usize << ell_n, "lo_bits[{i}] length must match time domain");
        }

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            rhs,
            lo_bits,
            val,
            // Degree bound:
            // - chi(t): multilinear (degree 1)
            // - has_lookup(t): multilinear (degree 1)
            // - lhs(t)·rhs(t): degree 2
            // - lo(t), val(t): multilinear (degree 1)
            // ⇒ total degree ≤ 1 + 1 + 2 = 4
            degree_bound: 4,
        }
    }
}

impl RoundOracle for Rv32PackedMulhuOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let two32 = K::from_u64(1u64 << 32);

        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let rhs = self.rhs.singleton_value();
            let val = self.val.singleton_value();

            let mut lo = K::ZERO;
            for (i, b) in self.lo_bits.iter().enumerate() {
                lo += b.singleton_value() * K::from_u64(1u64 << i);
            }

            let expr = lhs * rhs - lo - val * two32;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let rhs0 = self.rhs.get(child0);
            let rhs1 = self.rhs.get(child1);
            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);

            // Pre-fetch lo bit endpoints for this pair to avoid repeated sparse lookups per eval point.
            let mut lo0s: [K; 32] = [K::ZERO; 32];
            let mut lo1s: [K; 32] = [K::ZERO; 32];
            for (i, b) in self.lo_bits.iter().enumerate() {
                lo0s[i] = b.get(child0);
                lo1s[i] = b.get(child1);
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let lhs_x = interp(lhs0, lhs1, x);
                let rhs_x = interp(rhs0, rhs1, x);
                let val_x = interp(val0, val1, x);

                let mut lo_x = K::ZERO;
                for j in 0..32 {
                    let b_x = interp(lo0s[j], lo1s[j], x);
                    lo_x += b_x * K::from_u64(1u64 << j);
                }

                let expr_x = lhs_x * rhs_x - lo_x - val_x * two32;
                if expr_x == K::ZERO {
                    continue;
                }

                ys[i] += chi_x * gate_x * expr_x;
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        self.rhs.fold_round_in_place(r);
        for b in self.lo_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        self.val.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

// ============================================================================
// Packed-key RV32 MULH / MULHSU helpers (time-domain)
// ============================================================================

/// Sparse Route A oracle for RV32 unsigned product decomposition:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · (lhs(t)·rhs(t) - lo(t) - hi(t)·2^32)
///
/// Where:
/// - `lo(t)` is the low 32 bits of the 64-bit product, encoded as 32 Boolean columns,
/// - `hi(t)` is a witness column intended to be the upper 32 bits.
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedMulHiOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    lo_bits: Vec<SparseIdxVec<K>>,
    hi: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedMulHiOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        lo_bits: Vec<SparseIdxVec<K>>,
        hi: SparseIdxVec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(hi.len(), 1usize << ell_n);
        debug_assert_eq!(lo_bits.len(), 32);
        for (i, b) in lo_bits.iter().enumerate() {
            debug_assert_eq!(b.len(), 1usize << ell_n, "lo_bits[{i}] length must match time domain");
        }

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            rhs,
            lo_bits,
            hi,
            // Degree bound:
            // - chi(t): multilinear (degree 1)
            // - has_lookup(t): multilinear (degree 1)
            // - lhs(t)·rhs(t): degree 2
            // ⇒ total degree ≤ 1 + 1 + 2 = 4
            degree_bound: 4,
        }
    }
}

impl RoundOracle for Rv32PackedMulHiOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let two32 = K::from_u64(1u64 << 32);

        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let rhs = self.rhs.singleton_value();
            let hi = self.hi.singleton_value();

            let mut lo = K::ZERO;
            for (i, b) in self.lo_bits.iter().enumerate() {
                lo += b.singleton_value() * K::from_u64(1u64 << i);
            }

            let expr = lhs * rhs - lo - hi * two32;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let rhs0 = self.rhs.get(child0);
            let rhs1 = self.rhs.get(child1);
            let hi0 = self.hi.get(child0);
            let hi1 = self.hi.get(child1);

            // Pre-fetch lo bit endpoints for this pair to avoid repeated sparse lookups per eval point.
            let mut lo0s: [K; 32] = [K::ZERO; 32];
            let mut lo1s: [K; 32] = [K::ZERO; 32];
            for (i, b) in self.lo_bits.iter().enumerate() {
                lo0s[i] = b.get(child0);
                lo1s[i] = b.get(child1);
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let lhs_x = interp(lhs0, lhs1, x);
                let rhs_x = interp(rhs0, rhs1, x);
                let hi_x = interp(hi0, hi1, x);

                let mut lo_x = K::ZERO;
                for j in 0..32 {
                    let b_x = interp(lo0s[j], lo1s[j], x);
                    lo_x += b_x * K::from_u64(1u64 << j);
                }

                let expr_x = lhs_x * rhs_x - lo_x - hi_x * two32;
                if expr_x == K::ZERO {
                    continue;
                }

                ys[i] += chi_x * gate_x * expr_x;
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        self.rhs.fold_round_in_place(r);
        for b in self.lo_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        self.hi.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for RV32 packed MULH signed correction:
///   Σ_t χ(t)·has_lookup(t)·(w0·(hi - s1·rhs - s2·lhs + k·2^32 - val) + w1·k(k-1)(k-2))
///
/// Where:
/// - `hi` is the upper 32 bits of the unsigned product `lhs·rhs`,
/// - `s1`, `s2` are witness sign bits (msb of lhs/rhs),
/// - `k ∈ {0,1,2}` accounts for mod-2^32 normalization.
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedMulhAdapterOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    lhs_sign: SparseIdxVec<K>,
    rhs_sign: SparseIdxVec<K>,
    hi: SparseIdxVec<K>,
    k: SparseIdxVec<K>,
    val: SparseIdxVec<K>,
    weights: [K; 2],
    degree_bound: usize,
}

impl Rv32PackedMulhAdapterOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        lhs_sign: SparseIdxVec<K>,
        rhs_sign: SparseIdxVec<K>,
        hi: SparseIdxVec<K>,
        k: SparseIdxVec<K>,
        val: SparseIdxVec<K>,
        weights: [K; 2],
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(lhs_sign.len(), 1usize << ell_n);
        debug_assert_eq!(rhs_sign.len(), 1usize << ell_n);
        debug_assert_eq!(hi.len(), 1usize << ell_n);
        debug_assert_eq!(k.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            rhs,
            lhs_sign,
            rhs_sign,
            hi,
            k,
            val,
            weights,
            // Degree bound:
            // - chi(t): multilinear (degree 1)
            // - has_lookup(t): multilinear (degree 1)
            // - eq expr: degree 2 (sign·rhs)
            // - range poly: degree 3
            // ⇒ total degree ≤ 1 + 1 + 3 = 5
            degree_bound: 5,
        }
    }
}

impl RoundOracle for Rv32PackedMulhAdapterOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let two32 = K::from_u64(1u64 << 32);
        let w0 = self.weights[0];
        let w1 = self.weights[1];

        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let rhs = self.rhs.singleton_value();
            let lhs_sign = self.lhs_sign.singleton_value();
            let rhs_sign = self.rhs_sign.singleton_value();
            let hi = self.hi.singleton_value();
            let k = self.k.singleton_value();
            let val = self.val.singleton_value();

            let eq_expr = hi - lhs_sign * rhs - rhs_sign * lhs + k * two32 - val;
            let range = k * (k - K::ONE) * (k - K::from_u64(2));
            let expr = w0 * eq_expr + w1 * range;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let rhs0 = self.rhs.get(child0);
            let rhs1 = self.rhs.get(child1);
            let lhs_sign0 = self.lhs_sign.get(child0);
            let lhs_sign1 = self.lhs_sign.get(child1);
            let rhs_sign0 = self.rhs_sign.get(child0);
            let rhs_sign1 = self.rhs_sign.get(child1);
            let hi0 = self.hi.get(child0);
            let hi1 = self.hi.get(child1);
            let k0 = self.k.get(child0);
            let k1 = self.k.get(child1);
            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let lhs_x = interp(lhs0, lhs1, x);
                let rhs_x = interp(rhs0, rhs1, x);
                let lhs_sign_x = interp(lhs_sign0, lhs_sign1, x);
                let rhs_sign_x = interp(rhs_sign0, rhs_sign1, x);
                let hi_x = interp(hi0, hi1, x);
                let k_x = interp(k0, k1, x);
                let val_x = interp(val0, val1, x);

                let eq_expr = hi_x - lhs_sign_x * rhs_x - rhs_sign_x * lhs_x + k_x * two32 - val_x;
                let range = k_x * (k_x - K::ONE) * (k_x - K::from_u64(2));
                let expr_x = w0 * eq_expr + w1 * range;
                if expr_x == K::ZERO {
                    continue;
                }

                ys[i] += chi_x * gate_x * expr_x;
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        self.rhs.fold_round_in_place(r);
        self.lhs_sign.fold_round_in_place(r);
        self.rhs_sign.fold_round_in_place(r);
        self.hi.fold_round_in_place(r);
        self.k.fold_round_in_place(r);
        self.val.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for RV32 packed MULHSU signed correction:
///   Σ_t χ(t)·has_lookup(t)·(hi - s·rhs - val + b·2^32)
///
/// Where:
/// - `hi` is the upper 32 bits of the unsigned product `lhs·rhs`,
/// - `s` is the witness sign bit of `lhs`,
/// - `b ∈ {0,1}` is a borrow bit for mod-2^32 normalization.
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedMulhsuAdapterOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    lhs_sign: SparseIdxVec<K>,
    hi: SparseIdxVec<K>,
    borrow: SparseIdxVec<K>,
    val: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedMulhsuAdapterOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        lhs_sign: SparseIdxVec<K>,
        hi: SparseIdxVec<K>,
        borrow: SparseIdxVec<K>,
        val: SparseIdxVec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(lhs_sign.len(), 1usize << ell_n);
        debug_assert_eq!(hi.len(), 1usize << ell_n);
        debug_assert_eq!(borrow.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            rhs,
            lhs_sign,
            hi,
            borrow,
            val,
            // Degree bound:
            // - chi(t): multilinear (degree 1)
            // - has_lookup(t): multilinear (degree 1)
            // - eq expr: degree 2 (sign·rhs)
            // ⇒ total degree ≤ 1 + 1 + 2 = 4
            degree_bound: 4,
        }
    }
}

impl RoundOracle for Rv32PackedMulhsuAdapterOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let two32 = K::from_u64(1u64 << 32);

        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let rhs = self.rhs.singleton_value();
            let lhs_sign = self.lhs_sign.singleton_value();
            let hi = self.hi.singleton_value();
            let borrow = self.borrow.singleton_value();
            let val = self.val.singleton_value();
            let expr = hi - lhs_sign * rhs - val + borrow * two32;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let rhs0 = self.rhs.get(child0);
            let rhs1 = self.rhs.get(child1);
            let lhs_sign0 = self.lhs_sign.get(child0);
            let lhs_sign1 = self.lhs_sign.get(child1);
            let hi0 = self.hi.get(child0);
            let hi1 = self.hi.get(child1);
            let borrow0 = self.borrow.get(child0);
            let borrow1 = self.borrow.get(child1);
            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let rhs_x = interp(rhs0, rhs1, x);
                let lhs_sign_x = interp(lhs_sign0, lhs_sign1, x);
                let hi_x = interp(hi0, hi1, x);
                let borrow_x = interp(borrow0, borrow1, x);
                let val_x = interp(val0, val1, x);

                let expr_x = hi_x - lhs_sign_x * rhs_x - val_x + borrow_x * two32;
                if expr_x == K::ZERO {
                    continue;
                }

                ys[i] += chi_x * gate_x * expr_x;
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        self.rhs.fold_round_in_place(r);
        self.lhs_sign.fold_round_in_place(r);
        self.hi.fold_round_in_place(r);
        self.borrow.fold_round_in_place(r);
        self.val.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for RV32 packed EQ correctness:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · ((lhs(t) - rhs(t))·inv(t) - (1 - val(t)))
///
/// Here `inv(t)` is a witness column intended to be:
/// - `inv = 0` when `lhs == rhs` (unconstrained in this case),
/// - `inv = 1/(lhs - rhs)` when `lhs != rhs`.
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedEqOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    inv: SparseIdxVec<K>,
    val: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedEqOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        inv: SparseIdxVec<K>,
        val: SparseIdxVec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(inv.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            rhs,
            inv,
            val,
            degree_bound: 4,
        }
    }
}

impl RoundOracle for Rv32PackedEqOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let rhs = self.rhs.singleton_value();
            let inv = self.inv.singleton_value();
            let val = self.val.singleton_value();
            let diff = lhs - rhs;
            let expr = diff * inv - (K::ONE - val);
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let rhs0 = self.rhs.get(child0);
            let rhs1 = self.rhs.get(child1);
            let inv0 = self.inv.get(child0);
            let inv1 = self.inv.get(child1);
            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);

            let diff0 = lhs0 - rhs0;
            let diff1 = lhs1 - rhs1;

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }
                let diff_x = interp(diff0, diff1, x);
                let inv_x = interp(inv0, inv1, x);
                let val_x = interp(val0, val1, x);
                let expr_x = diff_x * inv_x - (K::ONE - val_x);
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        self.rhs.fold_round_in_place(r);
        self.inv.fold_round_in_place(r);
        self.val.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for RV32 packed EQ "zero product" check:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · (lhs(t) - rhs(t)) · val(t)
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedEqAdapterOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    val: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedEqAdapterOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        val: SparseIdxVec<K>,
        degree_bound: usize,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            rhs,
            val,
            degree_bound,
        }
    }
}

impl RoundOracle for Rv32PackedEqAdapterOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let rhs = self.rhs.singleton_value();
            let val = self.val.singleton_value();
            let diff = lhs - rhs;
            let expr = diff * val;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let rhs0 = self.rhs.get(child0);
            let rhs1 = self.rhs.get(child1);
            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);

            let diff0 = lhs0 - rhs0;
            let diff1 = lhs1 - rhs1;

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }
                let diff_x = interp(diff0, diff1, x);
                let val_x = interp(val0, val1, x);
                let expr_x = diff_x * val_x;
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        self.rhs.fold_round_in_place(r);
        self.val.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for RV32 packed NEQ correctness:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · ((lhs(t) - rhs(t))·inv(t) - val(t))
///
/// Here `inv(t)` is a witness column intended to be:
/// - `inv = 0` when `lhs == rhs` (unconstrained in this case),
/// - `inv = 1/(lhs - rhs)` when `lhs != rhs`.
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedNeqOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    inv: SparseIdxVec<K>,
    val: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedNeqOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        inv: SparseIdxVec<K>,
        val: SparseIdxVec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(inv.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            rhs,
            inv,
            val,
            degree_bound: 4,
        }
    }
}

impl RoundOracle for Rv32PackedNeqOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let rhs = self.rhs.singleton_value();
            let inv = self.inv.singleton_value();
            let val = self.val.singleton_value();
            let diff = lhs - rhs;
            let expr = diff * inv - val;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let rhs0 = self.rhs.get(child0);
            let rhs1 = self.rhs.get(child1);
            let inv0 = self.inv.get(child0);
            let inv1 = self.inv.get(child1);
            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);

            let diff0 = lhs0 - rhs0;
            let diff1 = lhs1 - rhs1;

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }
                let diff_x = interp(diff0, diff1, x);
                let inv_x = interp(inv0, inv1, x);
                let val_x = interp(val0, val1, x);
                let expr_x = diff_x * inv_x - val_x;
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        self.rhs.fold_round_in_place(r);
        self.inv.fold_round_in_place(r);
        self.val.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for RV32 packed NEQ "zero product" check:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · (lhs(t) - rhs(t)) · (1 - val(t))
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedNeqAdapterOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    val: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedNeqAdapterOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        val: SparseIdxVec<K>,
        degree_bound: usize,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            rhs,
            val,
            degree_bound,
        }
    }
}

impl RoundOracle for Rv32PackedNeqAdapterOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let rhs = self.rhs.singleton_value();
            let val = self.val.singleton_value();
            let diff = lhs - rhs;
            let expr = diff * (K::ONE - val);
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let rhs0 = self.rhs.get(child0);
            let rhs1 = self.rhs.get(child1);
            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);

            let diff0 = lhs0 - rhs0;
            let diff1 = lhs1 - rhs1;

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }
                let diff_x = interp(diff0, diff1, x);
                let val_x = interp(val0, val1, x);
                let expr_x = diff_x * (K::ONE - val_x);
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        self.rhs.fold_round_in_place(r);
        self.val.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

// ============================================================================
// Packed-key RV32 SLTU Shout (time-domain)
// ============================================================================

/// Sparse Route A oracle for RV32 packed SLT (signed less-than) correctness.
///
/// We reduce signed comparison to unsigned comparison by XOR-biasing both operands with `2^31`
/// (flip the sign bit). Let:
///   lhs_b = lhs ⊕ 2^31, rhs_b = rhs ⊕ 2^31
/// then `(lhs as i32) < (rhs as i32)` iff `lhs_b < rhs_b` as unsigned.
///
/// With witness bits `lhs_sign`, `rhs_sign` (intended as the msb of lhs/rhs), we implement the
/// XOR-biasing arithmetically:
///   x ⊕ 2^31 = x + (1 - 2·msb(x))·2^31.
///
/// The correctness constraint is:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · (lhs_b(t) - rhs_b(t) - diff(t) + out(t)·2^32)
///
/// Where `out(t)` is the SLT result bit (1 iff lhs < rhs in signed order, else 0) and `diff(t)`
/// is the u32 difference `lhs_b - rhs_b (mod 2^32)`.
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedSltOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    lhs_sign: SparseIdxVec<K>,
    rhs_sign: SparseIdxVec<K>,
    diff: SparseIdxVec<K>,
    out: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedSltOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        lhs_sign: SparseIdxVec<K>,
        rhs_sign: SparseIdxVec<K>,
        diff: SparseIdxVec<K>,
        out: SparseIdxVec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(lhs_sign.len(), 1usize << ell_n);
        debug_assert_eq!(rhs_sign.len(), 1usize << ell_n);
        debug_assert_eq!(diff.len(), 1usize << ell_n);
        debug_assert_eq!(out.len(), 1usize << ell_n);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            rhs,
            lhs_sign,
            rhs_sign,
            diff,
            out,
            degree_bound: 3,
        }
    }
}

impl RoundOracle for Rv32PackedSltOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let two31 = K::from_u64(1u64 << 31);
        let two32 = K::from_u64(1u64 << 32);
        let two = K::from_u64(2);

        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let rhs = self.rhs.singleton_value();
            let lhs_sign = self.lhs_sign.singleton_value();
            let rhs_sign = self.rhs_sign.singleton_value();
            let diff = self.diff.singleton_value();
            let out = self.out.singleton_value();

            let lhs_b = lhs + (K::ONE - two * lhs_sign) * two31;
            let rhs_b = rhs + (K::ONE - two * rhs_sign) * two31;
            let expr = lhs_b - rhs_b - diff + out * two32;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let rhs0 = self.rhs.get(child0);
            let rhs1 = self.rhs.get(child1);
            let lhs_sign0 = self.lhs_sign.get(child0);
            let lhs_sign1 = self.lhs_sign.get(child1);
            let rhs_sign0 = self.rhs_sign.get(child0);
            let rhs_sign1 = self.rhs_sign.get(child1);
            let diff0 = self.diff.get(child0);
            let diff1 = self.diff.get(child1);
            let out0 = self.out.get(child0);
            let out1 = self.out.get(child1);

            let lhs_b0 = lhs0 + (K::ONE - two * lhs_sign0) * two31;
            let lhs_b1 = lhs1 + (K::ONE - two * lhs_sign1) * two31;
            let rhs_b0 = rhs0 + (K::ONE - two * rhs_sign0) * two31;
            let rhs_b1 = rhs1 + (K::ONE - two * rhs_sign1) * two31;

            let expr0 = lhs_b0 - rhs_b0 - diff0 + out0 * two32;
            let expr1 = lhs_b1 - rhs_b1 - diff1 + out1 * two32;

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }
                let expr_x = interp(expr0, expr1, x);
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        self.rhs.fold_round_in_place(r);
        self.lhs_sign.fold_round_in_place(r);
        self.rhs_sign.fold_round_in_place(r);
        self.diff.fold_round_in_place(r);
        self.out.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for RV32 packed SLTU correctness:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · (lhs(t) - rhs(t) - diff(t) + out(t)·2^32)
///
/// Where `out(t)` is the SLTU result bit (1 iff lhs < rhs, else 0) and `diff(t)` is the u32
/// difference `lhs - rhs (mod 2^32)`.
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedSltuOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    diff: SparseIdxVec<K>,
    out: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedSltuOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        diff: SparseIdxVec<K>,
        out: SparseIdxVec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(diff.len(), 1usize << ell_n);
        debug_assert_eq!(out.len(), 1usize << ell_n);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            rhs,
            diff,
            out,
            degree_bound: 3,
        }
    }
}

impl RoundOracle for Rv32PackedSltuOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let two32 = K::from_u64(1u64 << 32);
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let rhs = self.rhs.singleton_value();
            let diff = self.diff.singleton_value();
            let out = self.out.singleton_value();
            let expr = lhs - rhs - diff + out * two32;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let two32 = K::from_u64(1u64 << 32);

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let rhs0 = self.rhs.get(child0);
            let rhs1 = self.rhs.get(child1);
            let diff0 = self.diff.get(child0);
            let diff1 = self.diff.get(child1);
            let out0 = self.out.get(child0);
            let out1 = self.out.get(child1);

            let expr0 = lhs0 - rhs0 - diff0 + out0 * two32;
            let expr1 = lhs1 - rhs1 - diff1 + out1 * two32;

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }
                let expr_x = interp(expr0, expr1, x);
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        self.rhs.fold_round_in_place(r);
        self.diff.fold_round_in_place(r);
        self.out.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

// ============================================================================
// Packed-key RV32 SLL Shout (time-domain)
// ============================================================================

/// Sparse Route A oracle for RV32 packed SLL correctness:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · (lhs(t) · 2^{shamt(t)} - val(t) - carry(t)·2^32)
///
/// Where:
/// - `shamt(t)` is the shift amount (0..31), encoded as 5 Boolean columns,
/// - `carry(t)` is the high part of `(lhs << shamt)` as a u32 (range-checked separately),
/// - `val(t)` is the low 32 bits of `(lhs << shamt)`.
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedSllOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    shamt_bits: Vec<SparseIdxVec<K>>,
    carry_bits: Vec<SparseIdxVec<K>>,
    val: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedSllOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        shamt_bits: Vec<SparseIdxVec<K>>,
        carry_bits: Vec<SparseIdxVec<K>>,
        val: SparseIdxVec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);
        debug_assert_eq!(shamt_bits.len(), 5);
        for (i, b) in shamt_bits.iter().enumerate() {
            debug_assert_eq!(b.len(), 1usize << ell_n, "shamt_bits[{i}] length must match time domain");
        }
        debug_assert_eq!(carry_bits.len(), 32);
        for (i, b) in carry_bits.iter().enumerate() {
            debug_assert_eq!(b.len(), 1usize << ell_n, "carry_bits[{i}] length must match time domain");
        }

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            shamt_bits,
            carry_bits,
            val,
            // Degree bound:
            // - chi(t): multilinear (degree 1)
            // - has_lookup(t): multilinear (degree 1)
            // - lhs(t): multilinear (degree 1)
            // - 2^{shamt(t)}: product of 5 linear terms in the shamt bits (degree 5)
            // - val(t), carry(t): multilinear (degree 1)
            // ⇒ total degree ≤ 1 + 1 + max(1+5, 1, 1) = 8
            degree_bound: 8,
        }
    }
}

impl RoundOracle for Rv32PackedSllOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let two32 = K::from_u64(1u64 << 32);
        let pow2_const: [K; 5] = [
            K::from_u64(2),
            K::from_u64(4),
            K::from_u64(16),
            K::from_u64(256),
            K::from_u64(65536),
        ];

        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let val = self.val.singleton_value();

            let mut pow2 = K::ONE;
            for (b, c) in self.shamt_bits.iter().zip(pow2_const.iter()) {
                let bit = b.singleton_value();
                pow2 *= K::ONE + bit * (*c - K::ONE);
            }

            let mut carry = K::ZERO;
            for (i, b) in self.carry_bits.iter().enumerate() {
                carry += b.singleton_value() * K::from_u64(1u64 << i);
            }

            let expr = lhs * pow2 - val - carry * two32;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);

            // Pre-fetch bit endpoints for this pair to avoid repeated sparse lookups per eval point.
            let mut b0s: [K; 5] = [K::ZERO; 5];
            let mut b1s: [K; 5] = [K::ZERO; 5];
            for (i, b) in self.shamt_bits.iter().enumerate() {
                b0s[i] = b.get(child0);
                b1s[i] = b.get(child1);
            }
            let mut c0s: [K; 32] = [K::ZERO; 32];
            let mut c1s: [K; 32] = [K::ZERO; 32];
            for (i, b) in self.carry_bits.iter().enumerate() {
                c0s[i] = b.get(child0);
                c1s[i] = b.get(child1);
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let lhs_x = interp(lhs0, lhs1, x);
                let val_x = interp(val0, val1, x);

                let mut pow2_x = K::ONE;
                for j in 0..5 {
                    let b_x = interp(b0s[j], b1s[j], x);
                    pow2_x *= K::ONE + b_x * (pow2_const[j] - K::ONE);
                }

                let mut carry_x = K::ZERO;
                for j in 0..32 {
                    let c_x = interp(c0s[j], c1s[j], x);
                    carry_x += c_x * K::from_u64(1u64 << j);
                }

                let expr_x = lhs_x * pow2_x - val_x - carry_x * two32;
                if expr_x == K::ZERO {
                    continue;
                }

                ys[i] += chi_x * gate_x * expr_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        for b in self.shamt_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        for b in self.carry_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        self.val.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

// ============================================================================
// Packed-key RV32 SRL Shout (time-domain)
// ============================================================================

/// Sparse Route A oracle for RV32 packed SRL correctness:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · (lhs(t) - val(t)·2^{shamt(t)} - rem(t))
///
/// Where:
/// - `shamt(t)` is the shift amount (0..31), encoded as 5 Boolean columns,
/// - `rem(t)` is the remainder `lhs mod 2^{shamt}`, encoded as 32 Boolean columns,
/// - `val(t)` is the SRL result (`lhs >> shamt`).
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedSrlOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    shamt_bits: Vec<SparseIdxVec<K>>,
    rem_bits: Vec<SparseIdxVec<K>>,
    val: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedSrlOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        shamt_bits: Vec<SparseIdxVec<K>>,
        rem_bits: Vec<SparseIdxVec<K>>,
        val: SparseIdxVec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);
        debug_assert_eq!(shamt_bits.len(), 5);
        for (i, b) in shamt_bits.iter().enumerate() {
            debug_assert_eq!(b.len(), 1usize << ell_n, "shamt_bits[{i}] length must match time domain");
        }
        debug_assert_eq!(rem_bits.len(), 32);
        for (i, b) in rem_bits.iter().enumerate() {
            debug_assert_eq!(b.len(), 1usize << ell_n, "rem_bits[{i}] length must match time domain");
        }

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            shamt_bits,
            rem_bits,
            val,
            // Degree bound:
            // - chi(t): multilinear (degree 1)
            // - has_lookup(t): multilinear (degree 1)
            // - lhs(t): multilinear (degree 1)
            // - 2^{shamt(t)}: product of 5 linear terms in the shamt bits (degree 5)
            // - val(t), rem(t): multilinear (degree 1)
            // ⇒ total degree ≤ 1 + 1 + max(1+5, 1, 1) = 8
            degree_bound: 8,
        }
    }
}

impl RoundOracle for Rv32PackedSrlOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let pow2_const: [K; 5] = [
            K::from_u64(2),
            K::from_u64(4),
            K::from_u64(16),
            K::from_u64(256),
            K::from_u64(65536),
        ];

        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let val = self.val.singleton_value();

            let mut pow2 = K::ONE;
            for (b, c) in self.shamt_bits.iter().zip(pow2_const.iter()) {
                let bit = b.singleton_value();
                pow2 *= K::ONE + bit * (*c - K::ONE);
            }

            let mut rem = K::ZERO;
            for (i, b) in self.rem_bits.iter().enumerate() {
                rem += b.singleton_value() * K::from_u64(1u64 << i);
            }

            let expr = lhs - val * pow2 - rem;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);

            // Pre-fetch bit endpoints for this pair to avoid repeated sparse lookups per eval point.
            let mut b0s: [K; 5] = [K::ZERO; 5];
            let mut b1s: [K; 5] = [K::ZERO; 5];
            for (i, b) in self.shamt_bits.iter().enumerate() {
                b0s[i] = b.get(child0);
                b1s[i] = b.get(child1);
            }
            let mut r0s: [K; 32] = [K::ZERO; 32];
            let mut r1s: [K; 32] = [K::ZERO; 32];
            for (i, b) in self.rem_bits.iter().enumerate() {
                r0s[i] = b.get(child0);
                r1s[i] = b.get(child1);
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let lhs_x = interp(lhs0, lhs1, x);
                let val_x = interp(val0, val1, x);

                let mut pow2_x = K::ONE;
                for j in 0..5 {
                    let b_x = interp(b0s[j], b1s[j], x);
                    pow2_x *= K::ONE + b_x * (pow2_const[j] - K::ONE);
                }

                let mut rem_x = K::ZERO;
                for j in 0..32 {
                    let r_x = interp(r0s[j], r1s[j], x);
                    rem_x += r_x * K::from_u64(1u64 << j);
                }

                let expr_x = lhs_x - val_x * pow2_x - rem_x;
                if expr_x == K::ZERO {
                    continue;
                }

                ys[i] += chi_x * gate_x * expr_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        for b in self.shamt_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        for b in self.rem_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        self.val.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle that enforces the SRL remainder is < 2^{shamt}:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · Σ_s eq(shamt(t), s) · Σ_{i≥s} 2^i · rem_bit_i(t)
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedSrlAdapterOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    shamt_bits: Vec<SparseIdxVec<K>>,
    rem_bits: Vec<SparseIdxVec<K>>,
    degree_bound: usize,
}

impl Rv32PackedSrlAdapterOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        shamt_bits: Vec<SparseIdxVec<K>>,
        rem_bits: Vec<SparseIdxVec<K>>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(shamt_bits.len(), 5);
        for (i, b) in shamt_bits.iter().enumerate() {
            debug_assert_eq!(b.len(), 1usize << ell_n, "shamt_bits[{i}] length must match time domain");
        }
        debug_assert_eq!(rem_bits.len(), 32);
        for (i, b) in rem_bits.iter().enumerate() {
            debug_assert_eq!(b.len(), 1usize << ell_n, "rem_bits[{i}] length must match time domain");
        }

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            shamt_bits,
            rem_bits,
            // Degree bound: chi (1) + gate (1) + eq_s(shamt) (5) + tail(rem) (1) = 8.
            degree_bound: 8,
        }
    }
}

impl RoundOracle for Rv32PackedSrlAdapterOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();

            let mut shamt: [K; 5] = [K::ZERO; 5];
            for (i, b) in self.shamt_bits.iter().enumerate() {
                shamt[i] = b.singleton_value();
            }

            let mut rem: [K; 32] = [K::ZERO; 32];
            for (i, b) in self.rem_bits.iter().enumerate() {
                rem[i] = b.singleton_value();
            }

            // tail_sum[s] = Σ_{i≥s} 2^i · rem_i
            let mut tail_sum: [K; 32] = [K::ZERO; 32];
            let mut tail = K::ZERO;
            for i in (0..32).rev() {
                tail += rem[i] * K::from_u64(1u64 << i);
                tail_sum[i] = tail;
            }

            // eq_s(shamt) for s in 0..32
            let mut eq: [K; 32] = [K::ZERO; 32];
            for s in 0..32usize {
                let mut prod = K::ONE;
                for j in 0..5usize {
                    let b = shamt[j];
                    if ((s >> j) & 1) == 1 {
                        prod *= b;
                    } else {
                        prod *= K::ONE - b;
                    }
                }
                eq[s] = prod;
            }

            let mut expr = K::ZERO;
            for s in 0..32usize {
                expr += eq[s] * tail_sum[s];
            }

            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            // Pre-fetch bit endpoints for this pair to avoid repeated sparse lookups per eval point.
            let mut b0s: [K; 5] = [K::ZERO; 5];
            let mut b1s: [K; 5] = [K::ZERO; 5];
            for (i, b) in self.shamt_bits.iter().enumerate() {
                b0s[i] = b.get(child0);
                b1s[i] = b.get(child1);
            }
            let mut r0s: [K; 32] = [K::ZERO; 32];
            let mut r1s: [K; 32] = [K::ZERO; 32];
            for (i, b) in self.rem_bits.iter().enumerate() {
                r0s[i] = b.get(child0);
                r1s[i] = b.get(child1);
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let mut shamt: [K; 5] = [K::ZERO; 5];
                for j in 0..5usize {
                    shamt[j] = interp(b0s[j], b1s[j], x);
                }

                let mut rem: [K; 32] = [K::ZERO; 32];
                for j in 0..32usize {
                    rem[j] = interp(r0s[j], r1s[j], x);
                }

                // tail_sum[s] = Σ_{i≥s} 2^i · rem_i
                let mut tail_sum: [K; 32] = [K::ZERO; 32];
                let mut tail = K::ZERO;
                for j in (0..32).rev() {
                    tail += rem[j] * K::from_u64(1u64 << j);
                    tail_sum[j] = tail;
                }

                // eq_s(shamt) for s in 0..32
                let mut expr_x = K::ZERO;
                for s in 0..32usize {
                    let mut prod = K::ONE;
                    for j in 0..5usize {
                        let b = shamt[j];
                        if ((s >> j) & 1) == 1 {
                            prod *= b;
                        } else {
                            prod *= K::ONE - b;
                        }
                    }
                    expr_x += prod * tail_sum[s];
                }

                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        for b in self.shamt_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        for b in self.rem_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        self.bit_idx += 1;
    }
}

// ============================================================================
// Packed-key RV32 SRA Shout (time-domain)
// ============================================================================

/// Sparse Route A oracle for RV32 packed SRA correctness:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · (lhs(t) - val(t)·2^{shamt(t)} - rem(t) - sign(t)·2^32·(1 - 2^{shamt(t)}))
///
/// Where:
/// - `shamt(t)` is the shift amount (0..31), encoded as 5 Boolean columns,
/// - `sign(t)` is the sign bit of `lhs` (and the expected sign bit of `val`), encoded as 1 Boolean column,
/// - `rem(t)` is the remainder in the signed floor-division identity:
///     (lhs_signed) = (val_signed)·2^{shamt} + rem, with rem ∈ [0, 2^{shamt})
///   encoded as 31 Boolean columns (bits 0..30),
/// - `val(t)` is the SRA result (`(lhs as i32) >> shamt`, represented as u32 in the field).
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedSraOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    shamt_bits: Vec<SparseIdxVec<K>>,
    sign: SparseIdxVec<K>,
    rem_bits: Vec<SparseIdxVec<K>>,
    val: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedSraOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        shamt_bits: Vec<SparseIdxVec<K>>,
        sign: SparseIdxVec<K>,
        rem_bits: Vec<SparseIdxVec<K>>,
        val: SparseIdxVec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);
        debug_assert_eq!(sign.len(), 1usize << ell_n);
        debug_assert_eq!(shamt_bits.len(), 5);
        for (i, b) in shamt_bits.iter().enumerate() {
            debug_assert_eq!(b.len(), 1usize << ell_n, "shamt_bits[{i}] length must match time domain");
        }
        debug_assert_eq!(rem_bits.len(), 31);
        for (i, b) in rem_bits.iter().enumerate() {
            debug_assert_eq!(b.len(), 1usize << ell_n, "rem_bits[{i}] length must match time domain");
        }

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            shamt_bits,
            sign,
            rem_bits,
            val,
            // Degree bound:
            // - chi(t): multilinear (degree 1)
            // - has_lookup(t): multilinear (degree 1)
            // - lhs(t): multilinear (degree 1)
            // - 2^{shamt(t)}: product of 5 linear terms in the shamt bits (degree 5)
            // - val(t), rem(t), sign(t): multilinear (degree 1)
            // ⇒ total degree ≤ 1 + 1 + max(1+5, 1, 1+5, 1) = 8
            degree_bound: 8,
        }
    }
}

impl RoundOracle for Rv32PackedSraOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let two32 = K::from_u64(1u64 << 32);
        let pow2_const: [K; 5] = [
            K::from_u64(2),
            K::from_u64(4),
            K::from_u64(16),
            K::from_u64(256),
            K::from_u64(65536),
        ];

        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let val = self.val.singleton_value();
            let sign = self.sign.singleton_value();

            let mut pow2 = K::ONE;
            for (b, c) in self.shamt_bits.iter().zip(pow2_const.iter()) {
                let bit = b.singleton_value();
                pow2 *= K::ONE + bit * (*c - K::ONE);
            }

            let mut rem = K::ZERO;
            for (i, b) in self.rem_bits.iter().enumerate() {
                rem += b.singleton_value() * K::from_u64(1u64 << i);
            }

            let corr = sign * two32 * (K::ONE - pow2);
            let expr = lhs - val * pow2 - rem - corr;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);
            let sign0 = self.sign.get(child0);
            let sign1 = self.sign.get(child1);

            // Pre-fetch bit endpoints for this pair to avoid repeated sparse lookups per eval point.
            let mut b0s: [K; 5] = [K::ZERO; 5];
            let mut b1s: [K; 5] = [K::ZERO; 5];
            for (i, b) in self.shamt_bits.iter().enumerate() {
                b0s[i] = b.get(child0);
                b1s[i] = b.get(child1);
            }
            let mut r0s: [K; 31] = [K::ZERO; 31];
            let mut r1s: [K; 31] = [K::ZERO; 31];
            for (i, b) in self.rem_bits.iter().enumerate() {
                r0s[i] = b.get(child0);
                r1s[i] = b.get(child1);
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let lhs_x = interp(lhs0, lhs1, x);
                let val_x = interp(val0, val1, x);
                let sign_x = interp(sign0, sign1, x);

                let mut pow2_x = K::ONE;
                for j in 0..5 {
                    let b_x = interp(b0s[j], b1s[j], x);
                    pow2_x *= K::ONE + b_x * (pow2_const[j] - K::ONE);
                }

                let mut rem_x = K::ZERO;
                for j in 0..31 {
                    let r_x = interp(r0s[j], r1s[j], x);
                    rem_x += r_x * K::from_u64(1u64 << j);
                }

                let corr_x = sign_x * two32 * (K::ONE - pow2_x);
                let expr_x = lhs_x - val_x * pow2_x - rem_x - corr_x;
                if expr_x == K::ZERO {
                    continue;
                }

                ys[i] += chi_x * gate_x * expr_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        for b in self.shamt_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        self.sign.fold_round_in_place(r);
        for b in self.rem_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        self.val.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle that enforces the SRA remainder is < 2^{shamt}:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · Σ_s eq(shamt(t), s) · Σ_{i≥s} 2^i · rem_bit_i(t)
///
/// For SRA, we only carry 31 remainder bits (0..30). This is sufficient because `shamt ∈ [0,31]`
/// and the remainder range is always `< 2^{shamt} ≤ 2^{31}`.
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedSraAdapterOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    shamt_bits: Vec<SparseIdxVec<K>>,
    rem_bits: Vec<SparseIdxVec<K>>,
    degree_bound: usize,
}

impl Rv32PackedSraAdapterOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        shamt_bits: Vec<SparseIdxVec<K>>,
        rem_bits: Vec<SparseIdxVec<K>>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(shamt_bits.len(), 5);
        for (i, b) in shamt_bits.iter().enumerate() {
            debug_assert_eq!(b.len(), 1usize << ell_n, "shamt_bits[{i}] length must match time domain");
        }
        debug_assert_eq!(rem_bits.len(), 31);
        for (i, b) in rem_bits.iter().enumerate() {
            debug_assert_eq!(b.len(), 1usize << ell_n, "rem_bits[{i}] length must match time domain");
        }

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            shamt_bits,
            rem_bits,
            // Degree bound: chi (1) + gate (1) + eq_s(shamt) (5) + tail(rem) (1) = 8.
            degree_bound: 8,
        }
    }
}

impl RoundOracle for Rv32PackedSraAdapterOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();

            let mut shamt: [K; 5] = [K::ZERO; 5];
            for (i, b) in self.shamt_bits.iter().enumerate() {
                shamt[i] = b.singleton_value();
            }

            let mut rem: [K; 31] = [K::ZERO; 31];
            for (i, b) in self.rem_bits.iter().enumerate() {
                rem[i] = b.singleton_value();
            }

            // tail_sum[s] = Σ_{i≥s} 2^i · rem_i, with tail_sum[31]=0 (no bits >= 31).
            let mut tail_sum: [K; 32] = [K::ZERO; 32];
            let mut tail = K::ZERO;
            for i in (0..31).rev() {
                tail += rem[i] * K::from_u64(1u64 << i);
                tail_sum[i] = tail;
            }
            tail_sum[31] = K::ZERO;

            let mut expr = K::ZERO;
            for s in 0..32usize {
                let mut prod = K::ONE;
                for j in 0..5usize {
                    let b = shamt[j];
                    if ((s >> j) & 1) == 1 {
                        prod *= b;
                    } else {
                        prod *= K::ONE - b;
                    }
                }
                expr += prod * tail_sum[s];
            }

            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            // Pre-fetch bit endpoints for this pair to avoid repeated sparse lookups per eval point.
            let mut b0s: [K; 5] = [K::ZERO; 5];
            let mut b1s: [K; 5] = [K::ZERO; 5];
            for (i, b) in self.shamt_bits.iter().enumerate() {
                b0s[i] = b.get(child0);
                b1s[i] = b.get(child1);
            }
            let mut r0s: [K; 31] = [K::ZERO; 31];
            let mut r1s: [K; 31] = [K::ZERO; 31];
            for (i, b) in self.rem_bits.iter().enumerate() {
                r0s[i] = b.get(child0);
                r1s[i] = b.get(child1);
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let mut shamt: [K; 5] = [K::ZERO; 5];
                for j in 0..5usize {
                    shamt[j] = interp(b0s[j], b1s[j], x);
                }

                let mut rem: [K; 31] = [K::ZERO; 31];
                for j in 0..31usize {
                    rem[j] = interp(r0s[j], r1s[j], x);
                }

                // tail_sum[s] = Σ_{i≥s} 2^i · rem_i, with tail_sum[31]=0 (no bits >= 31).
                let mut tail_sum: [K; 32] = [K::ZERO; 32];
                let mut tail = K::ZERO;
                for j in (0..31).rev() {
                    tail += rem[j] * K::from_u64(1u64 << j);
                    tail_sum[j] = tail;
                }
                tail_sum[31] = K::ZERO;

                let mut expr_x = K::ZERO;
                for s in 0..32usize {
                    let mut prod = K::ONE;
                    for j in 0..5usize {
                        let b = shamt[j];
                        if ((s >> j) & 1) == 1 {
                            prod *= b;
                        } else {
                            prod *= K::ONE - b;
                        }
                    }
                    expr_x += prod * tail_sum[s];
                }

                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        for b in self.shamt_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        for b in self.rem_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        self.bit_idx += 1;
    }
}

// ============================================================================
// Packed-key RV32 DIV*/REM* Shout (time-domain)
// ============================================================================

/// Sparse Route A oracle for RV32 packed DIVU correctness:
///   Σ_t χ(t)·has_lookup(t)·( z(t)·(quot(t) - 0xFFFF_FFFF) + (1 - z(t))·(lhs(t) - rhs(t)·quot(t) - rem(t)) )
///
/// Where:
/// - `quot(t)` is the DIVU output (lane.val),
/// - `rem(t)` is an auxiliary witness column,
/// - `z(t)` is a witness bit intended to be 1 iff `rhs(t) == 0`.
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedDivuOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    rem: SparseIdxVec<K>,
    rhs_is_zero: SparseIdxVec<K>,
    quot: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedDivuOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        rem: SparseIdxVec<K>,
        rhs_is_zero: SparseIdxVec<K>,
        quot: SparseIdxVec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(rem.len(), 1usize << ell_n);
        debug_assert_eq!(rhs_is_zero.len(), 1usize << ell_n);
        debug_assert_eq!(quot.len(), 1usize << ell_n);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            rhs,
            rem,
            rhs_is_zero,
            quot,
            // Degree bound:
            // - chi(t): multilinear (degree 1)
            // - has_lookup(t): multilinear (degree 1)
            // - rhs(t)·quot(t): degree 2
            // - gated by (1 - z(t)): adds 1
            // ⇒ total degree ≤ 1 + 1 + 2 + 1 = 5
            degree_bound: 5,
        }
    }
}

impl RoundOracle for Rv32PackedDivuOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let all_ones = K::from_u64(u32::MAX as u64);

        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let rhs = self.rhs.singleton_value();
            let rem = self.rem.singleton_value();
            let z = self.rhs_is_zero.singleton_value();
            let quot = self.quot.singleton_value();

            let expr = z * (quot - all_ones) + (K::ONE - z) * (lhs - rhs * quot - rem);
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let rhs0 = self.rhs.get(child0);
            let rhs1 = self.rhs.get(child1);
            let rem0 = self.rem.get(child0);
            let rem1 = self.rem.get(child1);
            let z0 = self.rhs_is_zero.get(child0);
            let z1 = self.rhs_is_zero.get(child1);
            let quot0 = self.quot.get(child0);
            let quot1 = self.quot.get(child1);

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let lhs_x = interp(lhs0, lhs1, x);
                let rhs_x = interp(rhs0, rhs1, x);
                let rem_x = interp(rem0, rem1, x);
                let z_x = interp(z0, z1, x);
                let quot_x = interp(quot0, quot1, x);

                let expr_x = z_x * (quot_x - all_ones) + (K::ONE - z_x) * (lhs_x - rhs_x * quot_x - rem_x);
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        self.rhs.fold_round_in_place(r);
        self.rem.fold_round_in_place(r);
        self.rhs_is_zero.fold_round_in_place(r);
        self.quot.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for RV32 packed REMU correctness:
///   Σ_t χ(t)·has_lookup(t)·( z(t)·(rem(t) - lhs(t)) + (1 - z(t))·(lhs(t) - rhs(t)·quot(t) - rem(t)) )
///
/// Where:
/// - `rem(t)` is the REMU output (lane.val),
/// - `quot(t)` is an auxiliary witness column,
/// - `z(t)` is a witness bit intended to be 1 iff `rhs(t) == 0`.
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedRemuOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    quot: SparseIdxVec<K>,
    rhs_is_zero: SparseIdxVec<K>,
    rem: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedRemuOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        quot: SparseIdxVec<K>,
        rhs_is_zero: SparseIdxVec<K>,
        rem: SparseIdxVec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(quot.len(), 1usize << ell_n);
        debug_assert_eq!(rhs_is_zero.len(), 1usize << ell_n);
        debug_assert_eq!(rem.len(), 1usize << ell_n);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            rhs,
            quot,
            rhs_is_zero,
            rem,
            // Degree bound:
            // - chi(t): multilinear (degree 1)
            // - has_lookup(t): multilinear (degree 1)
            // - rhs(t)·quot(t): degree 2
            // - gated by (1 - z(t)): adds 1
            // ⇒ total degree ≤ 1 + 1 + 2 + 1 = 5
            degree_bound: 5,
        }
    }
}

impl RoundOracle for Rv32PackedRemuOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let rhs = self.rhs.singleton_value();
            let quot = self.quot.singleton_value();
            let z = self.rhs_is_zero.singleton_value();
            let rem = self.rem.singleton_value();

            let expr = z * (rem - lhs) + (K::ONE - z) * (lhs - rhs * quot - rem);
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let rhs0 = self.rhs.get(child0);
            let rhs1 = self.rhs.get(child1);
            let quot0 = self.quot.get(child0);
            let quot1 = self.quot.get(child1);
            let z0 = self.rhs_is_zero.get(child0);
            let z1 = self.rhs_is_zero.get(child1);
            let rem0 = self.rem.get(child0);
            let rem1 = self.rem.get(child1);

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let lhs_x = interp(lhs0, lhs1, x);
                let rhs_x = interp(rhs0, rhs1, x);
                let quot_x = interp(quot0, quot1, x);
                let z_x = interp(z0, z1, x);
                let rem_x = interp(rem0, rem1, x);

                let expr_x = z_x * (rem_x - lhs_x) + (K::ONE - z_x) * (lhs_x - rhs_x * quot_x - rem_x);
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        self.rhs.fold_round_in_place(r);
        self.quot.fold_round_in_place(r);
        self.rhs_is_zero.fold_round_in_place(r);
        self.rem.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for RV32 packed DIVU/REMU helpers (adapter):
///   Σ_t χ(t)·has_lookup(t)·Σ_i w_i · c_i(t)
///
/// Constraints:
/// - `c0 = rhs_is_zero·(1 - rhs_is_zero)`                 (boolean helper; redundant with bitness)
/// - `c1 = rhs_is_zero·rhs`                               (rhs_is_zero => rhs==0)
/// - `c2 = (1 - rhs_is_zero)·(rem - rhs - diff + 2^32)`    (remainder bound)
/// - `c3 = diff - Σ 2^i·diff_bit_i`                       (u32 decomposition)
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedDivRemuAdapterOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    rhs_is_zero: SparseIdxVec<K>,
    rem: SparseIdxVec<K>,
    diff: SparseIdxVec<K>,
    diff_bits: Vec<SparseIdxVec<K>>,
    weights: [K; 4],
    degree_bound: usize,
}

impl Rv32PackedDivRemuAdapterOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        rhs_is_zero: SparseIdxVec<K>,
        rem: SparseIdxVec<K>,
        diff: SparseIdxVec<K>,
        diff_bits: Vec<SparseIdxVec<K>>,
        weights: [K; 4],
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs_is_zero.len(), 1usize << ell_n);
        debug_assert_eq!(rem.len(), 1usize << ell_n);
        debug_assert_eq!(diff.len(), 1usize << ell_n);
        debug_assert_eq!(diff_bits.len(), 32);
        for (i, b) in diff_bits.iter().enumerate() {
            debug_assert_eq!(b.len(), 1usize << ell_n, "diff_bits[{i}] length must match time domain");
        }

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            rhs,
            rhs_is_zero,
            rem,
            diff,
            diff_bits,
            weights,
	            // Degree bound:
	            // - chi(t): multilinear (degree 1)
	            // - has_lookup(t): multilinear (degree 1)
	            // - remainder bound term multiplies by (1 - rhs_is_zero): degree 2
	            // ⇒ total degree ≤ 1 + 1 + 2 = 4
	            degree_bound: 4,
	        }
    }
}

impl RoundOracle for Rv32PackedDivRemuAdapterOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let two32 = K::from_u64(1u64 << 32);
        let w0 = self.weights[0];
        let w1 = self.weights[1];
        let w2 = self.weights[2];
        let w3 = self.weights[3];

        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let rhs = self.rhs.singleton_value();
            let z = self.rhs_is_zero.singleton_value();
            let rem = self.rem.singleton_value();
            let diff = self.diff.singleton_value();

            let mut sum = K::ZERO;
            for (i, b) in self.diff_bits.iter().enumerate() {
                sum += b.singleton_value() * K::from_u64(1u64 << i);
            }

            let c0 = z * (K::ONE - z);
            let c1 = z * rhs;
            let c2 = (K::ONE - z) * (rem - rhs - diff + two32);
            let c3 = diff - sum;

            let expr = w0 * c0 + w1 * c1 + w2 * c2 + w3 * c3;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let rhs0 = self.rhs.get(child0);
            let rhs1 = self.rhs.get(child1);
            let z0 = self.rhs_is_zero.get(child0);
            let z1 = self.rhs_is_zero.get(child1);
            let rem0 = self.rem.get(child0);
            let rem1 = self.rem.get(child1);
            let diff0 = self.diff.get(child0);
            let diff1 = self.diff.get(child1);

            let mut b0s: [K; 32] = [K::ZERO; 32];
            let mut b1s: [K; 32] = [K::ZERO; 32];
            for (j, b) in self.diff_bits.iter().enumerate() {
                b0s[j] = b.get(child0);
                b1s[j] = b.get(child1);
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let rhs_x = interp(rhs0, rhs1, x);
                let z_x = interp(z0, z1, x);
                let rem_x = interp(rem0, rem1, x);
                let diff_x = interp(diff0, diff1, x);

                let mut sum = K::ZERO;
                for j in 0..32 {
                    let b_x = interp(b0s[j], b1s[j], x);
                    sum += b_x * K::from_u64(1u64 << j);
                }

                let c0 = z_x * (K::ONE - z_x);
                let c1 = z_x * rhs_x;
                let c2 = (K::ONE - z_x) * (rem_x - rhs_x - diff_x + two32);
                let c3 = diff_x - sum;
                let expr_x = w0 * c0 + w1 * c1 + w2 * c2 + w3 * c3;
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.rhs.fold_round_in_place(r);
        self.rhs_is_zero.fold_round_in_place(r);
        self.rem.fold_round_in_place(r);
        self.diff.fold_round_in_place(r);
        for b in self.diff_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for RV32 packed DIV correctness (signed quotient output).
///
/// Uses auxiliary `q_abs` (unsigned quotient), sign bits, and a `q_is_zero` witness bit to
/// handle the `-0` normalization case.
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedDivOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs_sign: SparseIdxVec<K>,
    rhs_sign: SparseIdxVec<K>,
    rhs_is_zero: SparseIdxVec<K>,
    q_abs: SparseIdxVec<K>,
    q_is_zero: SparseIdxVec<K>,
    val: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedDivOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs_sign: SparseIdxVec<K>,
        rhs_sign: SparseIdxVec<K>,
        rhs_is_zero: SparseIdxVec<K>,
        q_abs: SparseIdxVec<K>,
        q_is_zero: SparseIdxVec<K>,
        val: SparseIdxVec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs_sign.len(), 1usize << ell_n);
        debug_assert_eq!(rhs_sign.len(), 1usize << ell_n);
        debug_assert_eq!(rhs_is_zero.len(), 1usize << ell_n);
        debug_assert_eq!(q_abs.len(), 1usize << ell_n);
        debug_assert_eq!(q_is_zero.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs_sign,
            rhs_sign,
            rhs_is_zero,
            q_abs,
            q_is_zero,
            val,
            // This oracle composes abs-quotient sign logic (degree 4) with rhs_is_zero gating.
            degree_bound: 7,
        }
    }
}

impl RoundOracle for Rv32PackedDivOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let two = K::from_u64(2);
        let two32 = K::from_u64(1u64 << 32);
        let all_ones = K::from_u64(u32::MAX as u64);

        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let s1 = self.lhs_sign.singleton_value();
            let s2 = self.rhs_sign.singleton_value();
            let z = self.rhs_is_zero.singleton_value();
            let q_abs = self.q_abs.singleton_value();
            let q0 = self.q_is_zero.singleton_value();
            let val = self.val.singleton_value();

            let div_sign = s1 + s2 - two * s1 * s2;
            let neg_q = (K::ONE - q0) * (two32 - q_abs);
            let q_signed = (K::ONE - div_sign) * q_abs + div_sign * neg_q;

            let expr = z * (val - all_ones) + (K::ONE - z) * (val - q_signed);
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let s10 = self.lhs_sign.get(child0);
            let s11 = self.lhs_sign.get(child1);
            let s20 = self.rhs_sign.get(child0);
            let s21 = self.rhs_sign.get(child1);
            let z0 = self.rhs_is_zero.get(child0);
            let z1 = self.rhs_is_zero.get(child1);
            let q0 = self.q_abs.get(child0);
            let q1 = self.q_abs.get(child1);
            let qz0 = self.q_is_zero.get(child0);
            let qz1 = self.q_is_zero.get(child1);
            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let s1 = interp(s10, s11, x);
                let s2 = interp(s20, s21, x);
                let z = interp(z0, z1, x);
                let q_abs = interp(q0, q1, x);
                let qz = interp(qz0, qz1, x);
                let val = interp(val0, val1, x);

                let div_sign = s1 + s2 - two * s1 * s2;
                let neg_q = (K::ONE - qz) * (two32 - q_abs);
                let q_signed = (K::ONE - div_sign) * q_abs + div_sign * neg_q;

                let expr_x = z * (val - all_ones) + (K::ONE - z) * (val - q_signed);
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs_sign.fold_round_in_place(r);
        self.rhs_sign.fold_round_in_place(r);
        self.rhs_is_zero.fold_round_in_place(r);
        self.q_abs.fold_round_in_place(r);
        self.q_is_zero.fold_round_in_place(r);
        self.val.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for RV32 packed REM correctness (signed remainder output).
///
/// Uses auxiliary `r_abs` (unsigned remainder), the dividend sign bit, and `r_is_zero` to handle `-0`.
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct Rv32PackedRemOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    lhs_sign: SparseIdxVec<K>,
    rhs_is_zero: SparseIdxVec<K>,
    r_abs: SparseIdxVec<K>,
    r_is_zero: SparseIdxVec<K>,
    val: SparseIdxVec<K>,
    degree_bound: usize,
}

impl Rv32PackedRemOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        lhs_sign: SparseIdxVec<K>,
        rhs_is_zero: SparseIdxVec<K>,
        r_abs: SparseIdxVec<K>,
        r_is_zero: SparseIdxVec<K>,
        val: SparseIdxVec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(lhs_sign.len(), 1usize << ell_n);
        debug_assert_eq!(rhs_is_zero.len(), 1usize << ell_n);
        debug_assert_eq!(r_abs.len(), 1usize << ell_n);
        debug_assert_eq!(r_is_zero.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            lhs_sign,
            rhs_is_zero,
            r_abs,
            r_is_zero,
            val,
            degree_bound: 7,
        }
    }
}

impl RoundOracle for Rv32PackedRemOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let two32 = K::from_u64(1u64 << 32);

        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let s = self.lhs_sign.singleton_value();
            let z = self.rhs_is_zero.singleton_value();
            let r_abs = self.r_abs.singleton_value();
            let r0 = self.r_is_zero.singleton_value();
            let val = self.val.singleton_value();

            let neg_r = (K::ONE - r0) * (two32 - r_abs);
            let r_signed = (K::ONE - s) * r_abs + s * neg_r;
            let expr = z * (val - lhs) + (K::ONE - z) * (val - r_signed);
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let s0 = self.lhs_sign.get(child0);
            let s1 = self.lhs_sign.get(child1);
            let z0 = self.rhs_is_zero.get(child0);
            let z1 = self.rhs_is_zero.get(child1);
            let r0_0 = self.r_abs.get(child0);
            let r0_1 = self.r_abs.get(child1);
            let rz0 = self.r_is_zero.get(child0);
            let rz1 = self.r_is_zero.get(child1);
            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let lhs_x = interp(lhs0, lhs1, x);
                let s_x = interp(s0, s1, x);
                let z_x = interp(z0, z1, x);
                let r_abs_x = interp(r0_0, r0_1, x);
                let rz_x = interp(rz0, rz1, x);
                let val_x = interp(val0, val1, x);

                let neg_r = (K::ONE - rz_x) * (two32 - r_abs_x);
                let r_signed = (K::ONE - s_x) * r_abs_x + s_x * neg_r;
                let expr_x = z_x * (val_x - lhs_x) + (K::ONE - z_x) * (val_x - r_signed);
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        self.lhs_sign.fold_round_in_place(r);
        self.rhs_is_zero.fold_round_in_place(r);
        self.r_abs.fold_round_in_place(r);
        self.r_is_zero.fold_round_in_place(r);
        self.val.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for RV32 packed DIV/REM helpers (adapter).
///
/// This enforces:
/// - zero-detection for `rhs`,
/// - zero-detection for a signed-output magnitude (`q_abs` for DIV or `r_abs` for REM) to handle `-0`,
/// - absolute-value division equation over u32 values when `rhs != 0`,
/// - remainder bound `r_abs < |rhs|` when `rhs != 0`,
/// - u32 decomposition of the remainder-bound witness `diff`.
pub struct Rv32PackedDivRemAdapterOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    rhs_is_zero: SparseIdxVec<K>,
    lhs_sign: SparseIdxVec<K>,
    rhs_sign: SparseIdxVec<K>,
    q_abs: SparseIdxVec<K>,
    r_abs: SparseIdxVec<K>,
    mag: SparseIdxVec<K>,
    mag_is_zero: SparseIdxVec<K>,
    diff: SparseIdxVec<K>,
    diff_bits: Vec<SparseIdxVec<K>>,
    weights: [K; 7],
    degree_bound: usize,
}

impl Rv32PackedDivRemAdapterOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        rhs_is_zero: SparseIdxVec<K>,
        lhs_sign: SparseIdxVec<K>,
        rhs_sign: SparseIdxVec<K>,
        q_abs: SparseIdxVec<K>,
        r_abs: SparseIdxVec<K>,
        mag: SparseIdxVec<K>,
        mag_is_zero: SparseIdxVec<K>,
        diff: SparseIdxVec<K>,
        diff_bits: Vec<SparseIdxVec<K>>,
        weights: [K; 7],
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs_is_zero.len(), 1usize << ell_n);
        debug_assert_eq!(lhs_sign.len(), 1usize << ell_n);
        debug_assert_eq!(rhs_sign.len(), 1usize << ell_n);
        debug_assert_eq!(q_abs.len(), 1usize << ell_n);
        debug_assert_eq!(r_abs.len(), 1usize << ell_n);
        debug_assert_eq!(mag.len(), 1usize << ell_n);
        debug_assert_eq!(mag_is_zero.len(), 1usize << ell_n);
        debug_assert_eq!(diff.len(), 1usize << ell_n);
        debug_assert_eq!(diff_bits.len(), 32);
        for (i, b) in diff_bits.iter().enumerate() {
            debug_assert_eq!(b.len(), 1usize << ell_n, "diff_bits[{i}] length must match time domain");
        }

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs,
            rhs,
            rhs_is_zero,
            lhs_sign,
            rhs_sign,
            q_abs,
            r_abs,
            mag,
            mag_is_zero,
            diff,
            diff_bits,
            weights,
            degree_bound: 6,
        }
    }
}

impl RoundOracle for Rv32PackedDivRemAdapterOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let two = K::from_u64(2);
        let two32 = K::from_u64(1u64 << 32);

	        if self.has_lookup.len() == 1 {
	            let gate = self.has_lookup.singleton_value();
	            let lhs = self.lhs.singleton_value();
	            let rhs = self.rhs.singleton_value();
	            let z = self.rhs_is_zero.singleton_value();
	            let lhs_sign = self.lhs_sign.singleton_value();
	            let rhs_sign = self.rhs_sign.singleton_value();
	            let q_abs = self.q_abs.singleton_value();
	            let r_abs = self.r_abs.singleton_value();
	            let mag = self.mag.singleton_value();
	            let mag_z = self.mag_is_zero.singleton_value();
	            let diff = self.diff.singleton_value();

            let mut sum = K::ZERO;
            for (i, b) in self.diff_bits.iter().enumerate() {
                sum += b.singleton_value() * K::from_u64(1u64 << i);
            }

	            let lhs_abs = lhs + lhs_sign * (two32 - two * lhs);
	            let rhs_abs = rhs + rhs_sign * (two32 - two * rhs);

	            let c0 = z * (K::ONE - z);
	            let c1 = z * rhs;
	            let c2 = mag_z * (K::ONE - mag_z);
	            let c3 = mag_z * mag;
	            let c4 = (K::ONE - z) * (lhs_abs - rhs_abs * q_abs - r_abs);
	            let c5 = (K::ONE - z) * (r_abs - rhs_abs - diff + two32);
	            let c6 = diff - sum;

            let w = &self.weights;
            let expr = w[0] * c0 + w[1] * c1 + w[2] * c2 + w[3] * c3 + w[4] * c4 + w[5] * c5 + w[6] * c6;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

	            let lhs0 = self.lhs.get(child0);
	            let lhs1 = self.lhs.get(child1);
	            let rhs0 = self.rhs.get(child0);
	            let rhs1 = self.rhs.get(child1);
	            let z0 = self.rhs_is_zero.get(child0);
	            let z1 = self.rhs_is_zero.get(child1);
	            let lhs_sign0 = self.lhs_sign.get(child0);
	            let lhs_sign1 = self.lhs_sign.get(child1);
            let rhs_sign0 = self.rhs_sign.get(child0);
            let rhs_sign1 = self.rhs_sign.get(child1);
            let q0 = self.q_abs.get(child0);
	            let q1 = self.q_abs.get(child1);
	            let r0 = self.r_abs.get(child0);
	            let r1 = self.r_abs.get(child1);
	            let mag0 = self.mag.get(child0);
	            let mag1 = self.mag.get(child1);
	            let mag_z0 = self.mag_is_zero.get(child0);
	            let mag_z1 = self.mag_is_zero.get(child1);
	            let diff0 = self.diff.get(child0);
	            let diff1 = self.diff.get(child1);

            let mut b0s: [K; 32] = [K::ZERO; 32];
            let mut b1s: [K; 32] = [K::ZERO; 32];
            for (j, b) in self.diff_bits.iter().enumerate() {
                b0s[j] = b.get(child0);
                b1s[j] = b.get(child1);
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
	                let gate_x = interp(gate0, gate1, x);
	                if gate_x == K::ZERO {
	                    continue;
	                }

	                let lhs = interp(lhs0, lhs1, x);
	                let rhs = interp(rhs0, rhs1, x);
	                let z = interp(z0, z1, x);
	                let lhs_sign = interp(lhs_sign0, lhs_sign1, x);
	                let rhs_sign = interp(rhs_sign0, rhs_sign1, x);
	                let q_abs = interp(q0, q1, x);
	                let r_abs = interp(r0, r1, x);
	                let mag = interp(mag0, mag1, x);
	                let mag_z = interp(mag_z0, mag_z1, x);
	                let diff = interp(diff0, diff1, x);

                let mut sum = K::ZERO;
                for j in 0..32 {
                    let b_x = interp(b0s[j], b1s[j], x);
                    sum += b_x * K::from_u64(1u64 << j);
                }

	                let lhs_abs = lhs + lhs_sign * (two32 - two * lhs);
	                let rhs_abs = rhs + rhs_sign * (two32 - two * rhs);

	                let c0 = z * (K::ONE - z);
	                let c1 = z * rhs;
	                let c2 = mag_z * (K::ONE - mag_z);
	                let c3 = mag_z * mag;
	                let c4 = (K::ONE - z) * (lhs_abs - rhs_abs * q_abs - r_abs);
	                let c5 = (K::ONE - z) * (r_abs - rhs_abs - diff + two32);
	                let c6 = diff - sum;

                let w = &self.weights;
                let expr_x = w[0] * c0 + w[1] * c1 + w[2] * c2 + w[3] * c3 + w[4] * c4 + w[5] * c5 + w[6] * c6;
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
	        self.has_lookup.fold_round_in_place(r);
	        self.lhs.fold_round_in_place(r);
	        self.rhs.fold_round_in_place(r);
	        self.rhs_is_zero.fold_round_in_place(r);
	        self.lhs_sign.fold_round_in_place(r);
	        self.rhs_sign.fold_round_in_place(r);
	        self.q_abs.fold_round_in_place(r);
	        self.r_abs.fold_round_in_place(r);
	        self.mag.fold_round_in_place(r);
	        self.mag_is_zero.fold_round_in_place(r);
	        self.diff.fold_round_in_place(r);
        for b in self.diff_bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        self.bit_idx += 1;
    }
}

// ============================================================================
// Packed-key RV32 bitwise Shout (AND/OR/XOR) via 2-bit digits (time-domain)
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rv32PackedBitwiseOp2 {
    And,
    Andn,
    Or,
    Xor,
}

#[inline]
fn rv32_two_bit_digit_bits(inv2: K, inv6: K, x: K) -> (K, K) {
    // Bits for x in {0,1,2,3}, represented as degree-3 polynomials over the field:
    // - bit0: 0,1,0,1
    // - bit1: 0,0,1,1
    //
    // Using Lagrange basis on {0,1,2,3}:
    //   L1(x) = x(x-2)(x-3)/2
    //   L2(x) = -x(x-1)(x-3)/2
    //   L3(x) = x(x-1)(x-2)/6
    //
    // Then:
    //   bit0(x) = L1(x) + L3(x)
    //   bit1(x) = L2(x) + L3(x)
    let xm1 = x - K::ONE;
    let xm2 = x - K::from_u64(2);
    let xm3 = x - K::from_u64(3);

    let x_xm1 = x * xm1;
    let l1 = (x * xm2 * xm3) * inv2;
    let l3 = (x_xm1 * xm2) * inv6;
    let l2 = -(x_xm1 * xm3) * inv2;

    let bit0 = l1 + l3;
    let bit1 = l2 + l3;
    (bit0, bit1)
}

#[inline]
fn rv32_two_bit_digit_op(inv2: K, inv6: K, op: Rv32PackedBitwiseOp2, a: K, b: K) -> K {
    let (a0, a1) = rv32_two_bit_digit_bits(inv2, inv6, a);
    let (b0, b1) = rv32_two_bit_digit_bits(inv2, inv6, b);

    let two = K::from_u64(2);
    match op {
        Rv32PackedBitwiseOp2::And => {
            let r0 = a0 * b0;
            let r1 = a1 * b1;
            r0 + two * r1
        }
        Rv32PackedBitwiseOp2::Andn => {
            let r0 = a0 * (K::ONE - b0);
            let r1 = a1 * (K::ONE - b1);
            r0 + two * r1
        }
        Rv32PackedBitwiseOp2::Or => {
            let r0 = a0 + b0 - a0 * b0;
            let r1 = a1 + b1 - a1 * b1;
            r0 + two * r1
        }
        Rv32PackedBitwiseOp2::Xor => {
            let r0 = a0 + b0 - two * a0 * b0;
            let r1 = a1 + b1 - two * a1 * b1;
            r0 + two * r1
        }
    }
}

#[inline]
fn rv32_digit4_range_poly(x: K) -> K {
    // Vanishes exactly on {0,1,2,3}.
    x * (x - K::ONE) * (x - K::from_u64(2)) * (x - K::from_u64(3))
}

pub struct Rv32PackedBitwiseAdapterOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    degree_bound: usize,

    has_lookup: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs: SparseIdxVec<K>,
    lhs_digits: Vec<SparseIdxVec<K>>,
    rhs_digits: Vec<SparseIdxVec<K>>,
    weights: Vec<K>,
}

impl Rv32PackedBitwiseAdapterOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs: SparseIdxVec<K>,
        lhs_digits: Vec<SparseIdxVec<K>>,
        rhs_digits: Vec<SparseIdxVec<K>>,
        weights: Vec<K>,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        debug_assert_eq!(rhs.len(), 1usize << ell_n);
        debug_assert_eq!(lhs_digits.len(), 16);
        debug_assert_eq!(rhs_digits.len(), 16);
        for (i, d) in lhs_digits.iter().enumerate() {
            debug_assert_eq!(d.len(), 1usize << ell_n, "lhs_digits[{i}] length must match time domain");
        }
        for (i, d) in rhs_digits.iter().enumerate() {
            debug_assert_eq!(d.len(), 1usize << ell_n, "rhs_digits[{i}] length must match time domain");
        }
        debug_assert_eq!(weights.len(), 34);

        // Degree bound:
        // - chi(t): multilinear (degree 1)
        // - has_lookup(t): multilinear (degree 1)
        // - digit4 range poly: degree 4
        // ⇒ total degree ≤ 1 + 1 + 4 = 6
        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            degree_bound: 6,
            has_lookup,
            lhs,
            rhs,
            lhs_digits,
            rhs_digits,
            weights,
        }
    }
}

impl RoundOracle for Rv32PackedBitwiseAdapterOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let w_lhs = self.weights[0];
        let w_rhs = self.weights[1];
        let w_digits = &self.weights[2..];

        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let lhs = self.lhs.singleton_value();
            let rhs = self.rhs.singleton_value();

            let mut lhs_recon = K::ZERO;
            let mut rhs_recon = K::ZERO;
            for i in 0..16usize {
                lhs_recon += self.lhs_digits[i].singleton_value() * K::from_u64(1u64 << (2 * i));
                rhs_recon += self.rhs_digits[i].singleton_value() * K::from_u64(1u64 << (2 * i));
            }

            let mut range_sum = K::ZERO;
            for (i, d) in self.lhs_digits.iter().enumerate() {
                range_sum += w_digits[i] * rv32_digit4_range_poly(d.singleton_value());
            }
            for (i, d) in self.rhs_digits.iter().enumerate() {
                range_sum += w_digits[16 + i] * rv32_digit4_range_poly(d.singleton_value());
            }

            let expr = w_lhs * (lhs - lhs_recon) + w_rhs * (rhs - rhs_recon) + range_sum;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);
            let rhs0 = self.rhs.get(child0);
            let rhs1 = self.rhs.get(child1);

            let mut a0s: [K; 16] = [K::ZERO; 16];
            let mut a1s: [K; 16] = [K::ZERO; 16];
            for (i, d) in self.lhs_digits.iter().enumerate() {
                a0s[i] = d.get(child0);
                a1s[i] = d.get(child1);
            }
            let mut b0s: [K; 16] = [K::ZERO; 16];
            let mut b1s: [K; 16] = [K::ZERO; 16];
            for (i, d) in self.rhs_digits.iter().enumerate() {
                b0s[i] = d.get(child0);
                b1s[i] = d.get(child1);
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let lhs_x = interp(lhs0, lhs1, x);
                let rhs_x = interp(rhs0, rhs1, x);

                let mut lhs_recon = K::ZERO;
                let mut rhs_recon = K::ZERO;
                let mut range_sum = K::ZERO;
                for j in 0..16usize {
                    let aj = interp(a0s[j], a1s[j], x);
                    let bj = interp(b0s[j], b1s[j], x);
                    let pow = K::from_u64(1u64 << (2 * j));
                    lhs_recon += aj * pow;
                    rhs_recon += bj * pow;

                    range_sum += w_digits[j] * rv32_digit4_range_poly(aj);
                    range_sum += w_digits[16 + j] * rv32_digit4_range_poly(bj);
                }

                let expr = w_lhs * (lhs_x - lhs_recon) + w_rhs * (rhs_x - rhs_recon) + range_sum;
                ys[i] += chi_x * gate_x * expr;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        self.rhs.fold_round_in_place(r);
        for d in self.lhs_digits.iter_mut() {
            d.fold_round_in_place(r);
        }
        for d in self.rhs_digits.iter_mut() {
            d.fold_round_in_place(r);
        }
        self.bit_idx += 1;
    }
}

pub struct Rv32PackedBitwiseOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    lhs_digits: Vec<SparseIdxVec<K>>,
    rhs_digits: Vec<SparseIdxVec<K>>,
    val: SparseIdxVec<K>,
    op: Rv32PackedBitwiseOp2,
    inv2: K,
    inv6: K,
    degree_bound: usize,
}

impl Rv32PackedBitwiseOracleSparseTime {
    fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs_digits: Vec<SparseIdxVec<K>>,
        rhs_digits: Vec<SparseIdxVec<K>>,
        val: SparseIdxVec<K>,
        op: Rv32PackedBitwiseOp2,
    ) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(val.len(), 1usize << ell_n);
        debug_assert_eq!(lhs_digits.len(), 16);
        debug_assert_eq!(rhs_digits.len(), 16);
        for (i, d) in lhs_digits.iter().enumerate() {
            debug_assert_eq!(d.len(), 1usize << ell_n, "lhs_digits[{i}] length must match time domain");
        }
        for (i, d) in rhs_digits.iter().enumerate() {
            debug_assert_eq!(d.len(), 1usize << ell_n, "rhs_digits[{i}] length must match time domain");
        }

        // Degree bound:
        // - chi(t): multilinear (degree 1)
        // - has_lookup(t): multilinear (degree 1)
        // - bitwise digit op: degree 6 (two degree-3 bit extractors multiplied)
        // ⇒ total degree ≤ 1 + 1 + 6 = 8
        let inv2 = K::from_u64(2).inverse();
        let inv6 = K::from_u64(6).inverse();
        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            lhs_digits,
            rhs_digits,
            val,
            op,
            inv2,
            inv6,
            degree_bound: 8,
        }
    }
}

impl RoundOracle for Rv32PackedBitwiseOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let val = self.val.singleton_value();

            let mut out = K::ZERO;
            for i in 0..16usize {
                let a = self.lhs_digits[i].singleton_value();
                let b = self.rhs_digits[i].singleton_value();
                let digit = rv32_two_bit_digit_op(self.inv2, self.inv6, self.op, a, b);
                out += digit * K::from_u64(1u64 << (2 * i));
            }
            let expr = out - val;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let val0 = self.val.get(child0);
            let val1 = self.val.get(child1);

            let mut a0s: [K; 16] = [K::ZERO; 16];
            let mut a1s: [K; 16] = [K::ZERO; 16];
            for (j, d) in self.lhs_digits.iter().enumerate() {
                a0s[j] = d.get(child0);
                a1s[j] = d.get(child1);
            }
            let mut b0s: [K; 16] = [K::ZERO; 16];
            let mut b1s: [K; 16] = [K::ZERO; 16];
            for (j, d) in self.rhs_digits.iter().enumerate() {
                b0s[j] = d.get(child0);
                b1s[j] = d.get(child1);
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }

                let val_x = interp(val0, val1, x);

                let mut out = K::ZERO;
                for j in 0..16usize {
                    let a = interp(a0s[j], a1s[j], x);
                    let b = interp(b0s[j], b1s[j], x);
                    let digit = rv32_two_bit_digit_op(self.inv2, self.inv6, self.op, a, b);
                    out += digit * K::from_u64(1u64 << (2 * j));
                }

                let expr = out - val_x;
                ys[i] += chi_x * gate_x * expr;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        for d in self.lhs_digits.iter_mut() {
            d.fold_round_in_place(r);
        }
        for d in self.rhs_digits.iter_mut() {
            d.fold_round_in_place(r);
        }
        self.val.fold_round_in_place(r);
        self.bit_idx += 1;
    }
}

pub struct Rv32PackedAndOracleSparseTime {
    core: Rv32PackedBitwiseOracleSparseTime,
}
impl Rv32PackedAndOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs_digits: Vec<SparseIdxVec<K>>,
        rhs_digits: Vec<SparseIdxVec<K>>,
        val: SparseIdxVec<K>,
    ) -> Self {
        Self {
            core: Rv32PackedBitwiseOracleSparseTime::new(
                r_cycle,
                has_lookup,
                lhs_digits,
                rhs_digits,
                val,
                Rv32PackedBitwiseOp2::And,
            ),
        }
    }
}
impl_round_oracle_via_core!(Rv32PackedAndOracleSparseTime);

pub struct Rv32PackedAndnOracleSparseTime {
    core: Rv32PackedBitwiseOracleSparseTime,
}
impl Rv32PackedAndnOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs_digits: Vec<SparseIdxVec<K>>,
        rhs_digits: Vec<SparseIdxVec<K>>,
        val: SparseIdxVec<K>,
    ) -> Self {
        Self {
            core: Rv32PackedBitwiseOracleSparseTime::new(
                r_cycle,
                has_lookup,
                lhs_digits,
                rhs_digits,
                val,
                Rv32PackedBitwiseOp2::Andn,
            ),
        }
    }
}
impl_round_oracle_via_core!(Rv32PackedAndnOracleSparseTime);

pub struct Rv32PackedOrOracleSparseTime {
    core: Rv32PackedBitwiseOracleSparseTime,
}
impl Rv32PackedOrOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs_digits: Vec<SparseIdxVec<K>>,
        rhs_digits: Vec<SparseIdxVec<K>>,
        val: SparseIdxVec<K>,
    ) -> Self {
        Self {
            core: Rv32PackedBitwiseOracleSparseTime::new(
                r_cycle,
                has_lookup,
                lhs_digits,
                rhs_digits,
                val,
                Rv32PackedBitwiseOp2::Or,
            ),
        }
    }
}
impl_round_oracle_via_core!(Rv32PackedOrOracleSparseTime);

pub struct Rv32PackedXorOracleSparseTime {
    core: Rv32PackedBitwiseOracleSparseTime,
}
impl Rv32PackedXorOracleSparseTime {
    pub fn new(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        lhs_digits: Vec<SparseIdxVec<K>>,
        rhs_digits: Vec<SparseIdxVec<K>>,
        val: SparseIdxVec<K>,
    ) -> Self {
        Self {
            core: Rv32PackedBitwiseOracleSparseTime::new(
                r_cycle,
                has_lookup,
                lhs_digits,
                rhs_digits,
                val,
                Rv32PackedBitwiseOp2::Xor,
            ),
        }
    }
}
impl_round_oracle_via_core!(Rv32PackedXorOracleSparseTime);

/// Sparse Route A oracle for u32 bit-decomposition consistency:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · (x(t) - Σ_i 2^i · bit_i(t))
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct U32DecompOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    x: SparseIdxVec<K>,
    bits: Vec<SparseIdxVec<K>>,
    weights: Vec<K>,
    degree_bound: usize,
}

impl U32DecompOracleSparseTime {
    pub fn new(r_cycle: &[K], has_lookup: SparseIdxVec<K>, x: SparseIdxVec<K>, bits: Vec<SparseIdxVec<K>>) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(x.len(), 1usize << ell_n);
        debug_assert_eq!(bits.len(), 32);
        for (i, b) in bits.iter().enumerate() {
            debug_assert_eq!(b.len(), 1usize << ell_n, "bits[{i}] length must match time domain");
        }
        let weights: Vec<K> = (0..32).map(|i| K::from_u64(1u64 << i)).collect();

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            x,
            bits,
            weights,
            degree_bound: 3,
        }
    }
}

impl RoundOracle for U32DecompOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let x = self.x.singleton_value();
            let mut sum = K::ZERO;
            for (b, w) in self.bits.iter().zip(self.weights.iter()) {
                sum += b.singleton_value() * *w;
            }
            let expr = x - sum;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let x0 = self.x.get(child0);
            let x1 = self.x.get(child1);

            let mut expr0 = x0;
            let mut expr1 = x1;
            for (b_col, w) in self.bits.iter().zip(self.weights.iter()) {
                let b0 = b_col.get(child0);
                let b1 = b_col.get(child1);
                expr0 -= b0 * *w;
                expr1 -= b1 * *w;
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }
                let expr_x = interp(expr0, expr1, x);
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.x.fold_round_in_place(r);
        for b in self.bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for u5 bit-decomposition consistency:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · (x(t) - Σ_i 2^i · bit_i(t))
///
/// Intended usage: set the claimed sum to 0 to enforce correctness.
pub struct U5DecompOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    x: SparseIdxVec<K>,
    bits: Vec<SparseIdxVec<K>>,
    weights: Vec<K>,
    degree_bound: usize,
}

impl U5DecompOracleSparseTime {
    pub fn new(r_cycle: &[K], has_lookup: SparseIdxVec<K>, x: SparseIdxVec<K>, bits: Vec<SparseIdxVec<K>>) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(x.len(), 1usize << ell_n);
        debug_assert_eq!(bits.len(), 5);
        for (i, b) in bits.iter().enumerate() {
            debug_assert_eq!(b.len(), 1usize << ell_n, "bits[{i}] length must match time domain");
        }
        let weights: Vec<K> = (0..5).map(|i| K::from_u64(1u64 << i)).collect();

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_lookup,
            x,
            bits,
            weights,
            degree_bound: 3,
        }
    }
}

impl RoundOracle for U5DecompOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let x = self.x.singleton_value();
            let mut sum = K::ZERO;
            for (b, w) in self.bits.iter().zip(self.weights.iter()) {
                sum += b.singleton_value() * *w;
            }
            let expr = x - sum;
            let v = self.prefix_eq * gate * expr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let x0 = self.x.get(child0);
            let x1 = self.x.get(child1);

            let mut expr0 = x0;
            let mut expr1 = x1;
            for (b_col, w) in self.bits.iter().zip(self.weights.iter()) {
                let b0 = b_col.get(child0);
                let b1 = b_col.get(child1);
                expr0 -= b0 * *w;
                expr1 -= b1 * *w;
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }
                let expr_x = interp(expr0, expr1, x);
                if expr_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * gate_x * expr_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        self.x.fold_round_in_place(r);
        for b in self.bits.iter_mut() {
            b.fold_round_in_place(r);
        }
        self.bit_idx += 1;
    }
}

/// Zero oracle over the time hypercube (for placeholder claims).
pub struct ZeroOracleSparseTime {
    rounds_remaining: usize,
    degree_bound: usize,
}

impl ZeroOracleSparseTime {
    pub fn new(num_rounds: usize, degree_bound: usize) -> Self {
        Self {
            rounds_remaining: num_rounds,
            degree_bound,
        }
    }
}

impl RoundOracle for ZeroOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        vec![K::ZERO; points.len()]
    }

    fn num_rounds(&self) -> usize {
        self.rounds_remaining
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, _r: K) {
        if self.rounds_remaining > 0 {
            self.rounds_remaining -= 1;
        }
    }
}

#[inline]
fn interp(f0: K, f1: K, x: K) -> K {
    f0 + (f1 - f0) * x
}

fn fill_time_point(out: &mut [K], prefix: &[K], bit_idx: usize, bit_value: K, pair_idx: usize) {
    debug_assert!(bit_idx < out.len(), "bit_idx out of range");
    debug_assert_eq!(prefix.len(), bit_idx, "prefix length mismatch");

    out[..bit_idx].copy_from_slice(prefix);
    out[bit_idx] = bit_value;

    let mut shift = 0usize;
    for b in (bit_idx + 1)..out.len() {
        let bit = (pair_idx >> shift) & 1;
        out[b] = if bit == 1 { K::ONE } else { K::ZERO };
        shift += 1;
    }
}

fn lt_eval_at_bool_index(idx: usize, point: &[K]) -> K {
    let mut suffix = K::ONE;
    let mut acc = K::ZERO;
    for i in (0..point.len()).rev() {
        let bit = (idx >> i) & 1;
        if bit == 0 {
            acc += point[i] * suffix;
            suffix *= K::ONE - point[i];
        } else {
            suffix *= point[i];
        }
    }
    acc
}

fn val_pre_from_inc_terms(init_at_r_addr: K, inc_terms: &[(usize, K)], t_point: &[K]) -> K {
    let mut acc = init_at_r_addr;
    for &(u, inc_u) in inc_terms.iter() {
        acc += inc_u * lt_eval_at_bool_index(u, t_point);
    }
    acc
}

fn build_inc_terms_at_r_addr(
    wa_bits: &[SparseIdxVec<K>],
    has_write: &SparseIdxVec<K>,
    inc_at_write_addr: &SparseIdxVec<K>,
    r_addr: &[K],
) -> Vec<(usize, K)> {
    let mut out: Vec<(usize, K)> = Vec::new();
    for &(t, has_w) in has_write.entries() {
        let inc_t = inc_at_write_addr.get(t);
        if has_w == K::ZERO || inc_t == K::ZERO {
            continue;
        }

        let mut eq_addr = K::ONE;
        for (b, col) in wa_bits.iter().enumerate() {
            let bit = col.get(t);
            eq_addr *= eq_bit_affine(bit, r_addr[b]);
        }

        let inc_at_r_addr = has_w * inc_t * eq_addr;
        if inc_at_r_addr != K::ZERO {
            out.push((t, inc_at_r_addr));
        }
    }
    out
}

// ============================================================================
// Sparse-in-time Route A oracles (Track A, shared CPU bus)
// ============================================================================

/// Per-lane Twist bus columns (sparse in time).
///
/// A "lane" is an independent per-step access slot with its own read + write ports.
#[derive(Clone, Debug)]
pub struct TwistLaneSparseCols {
    pub ra_bits: Vec<SparseIdxVec<K>>,
    pub wa_bits: Vec<SparseIdxVec<K>>,
    pub has_read: SparseIdxVec<K>,
    pub has_write: SparseIdxVec<K>,
    pub wv: SparseIdxVec<K>,
    pub rv: SparseIdxVec<K>,
    pub inc_at_write_addr: SparseIdxVec<K>,
}

/// Sparse Route A oracle for Shout adapter:
///   Σ_t χ_{r_cycle}(t) · has_lookup(t) · Π_b eq(addr_bits_b(t), r_addr_b)
pub struct IndexAdapterOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_lookup: SparseIdxVec<K>,
    addr_bits: Vec<SparseIdxVec<K>>,
    r_addr: Vec<K>,
    degree_bound: usize,
    challenges: Vec<K>,
}

impl IndexAdapterOracleSparseTime {
    pub fn new_with_gate(
        r_cycle: &[K],
        has_lookup: SparseIdxVec<K>,
        addr_bits: Vec<SparseIdxVec<K>>,
        r_addr: &[K],
    ) -> (Self, K) {
        let ell_n = r_cycle.len();
        let ell_addr = addr_bits.len();
        debug_assert_eq!(has_lookup.len(), 1usize << ell_n);
        debug_assert_eq!(r_addr.len(), ell_addr);
        for (b, col) in addr_bits.iter().enumerate() {
            debug_assert_eq!(
                col.len(),
                1usize << ell_n,
                "addr_bits[{b}] length must match time domain"
            );
        }

        let mut claim = K::ZERO;
        for &(t, gate) in has_lookup.entries() {
            let mut eq_addr = K::ONE;
            for (b, col) in addr_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.get(t), r_addr[b]);
            }
            claim += chi_at_bool_index(r_cycle, t) * gate * eq_addr;
        }

        (
            Self {
                bit_idx: 0,
                r_cycle: r_cycle.to_vec(),
                prefix_eq: K::ONE,
                has_lookup,
                addr_bits,
                r_addr: r_addr.to_vec(),
                degree_bound: 2 + ell_addr,
                challenges: Vec::with_capacity(ell_n),
            },
            claim,
        )
    }
}

impl RoundOracle for IndexAdapterOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let mut eq_addr = K::ONE;
            for (b, col) in self.addr_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.singleton_value(), self.r_addr[b]);
            }
            let v = self.prefix_eq * self.has_lookup.singleton_value() * eq_addr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            // Per-bit eq factors at the children (after any time folding).
            let mut eq0s: Vec<K> = Vec::with_capacity(self.addr_bits.len());
            let mut d_eqs: Vec<K> = Vec::with_capacity(self.addr_bits.len());
            for (b, col) in self.addr_bits.iter().enumerate() {
                let e0 = eq_bit_affine(col.get(child0), self.r_addr[b]);
                let e1 = eq_bit_affine(col.get(child1), self.r_addr[b]);
                eq0s.push(e0);
                d_eqs.push(e1 - e0);
            }

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                let gate_x = interp(gate0, gate1, x);
                let mut prod = chi_x * gate_x;
                for (e0, de) in eq0s.iter().zip(d_eqs.iter()) {
                    prod *= *e0 + *de * x;
                }
                ys[i] += prod;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_lookup.fold_round_in_place(r);
        for col in self.addr_bits.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.challenges.push(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for χ_{r_cycle}-weighted *aggregated* bitness:
///   Σ_t χ_{r_cycle}(t) · ( Σ_i w_i · col_i(t) · (col_i(t) - 1) )
///
/// This reduces O(#bit-columns) separate degree-3 sumchecks to a single degree-3 sumcheck.
/// The weights `w_i` MUST be derived deterministically from transcript-known data (e.g. r_cycle),
/// so prover/verifier agree on the same polynomial.
pub struct LazyWeightedBitnessOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    cols: Vec<SparseIdxVec<K>>,
    weights: Vec<K>,
    degree_bound: usize,
    challenges: Vec<K>,
}

impl LazyWeightedBitnessOracleSparseTime {
    pub fn new_with_cycle(r_cycle: &[K], cols: Vec<SparseIdxVec<K>>, weights: Vec<K>) -> Self {
        let ell_n = r_cycle.len();
        debug_assert_eq!(cols.len(), weights.len());
        for col in &cols {
            debug_assert_eq!(col.len(), 1usize << ell_n);
        }
        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            cols,
            weights,
            degree_bound: 3,
            challenges: Vec::with_capacity(ell_n),
        }
    }
}

impl RoundOracle for LazyWeightedBitnessOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.cols.is_empty() {
            return vec![K::ZERO; points.len()];
        }

        if self.cols[0].len() == 1 {
            let mut acc = K::ZERO;
            for (col, w) in self.cols.iter().zip(self.weights.iter()) {
                let b = col.singleton_value();
                acc += *w * b * (b - K::ONE);
            }
            let v = self.prefix_eq * acc;
            return vec![v; points.len()];
        }

        // Union of parent indices whose children contain any nonzero across the aggregated columns.
        let mut pairs: Vec<usize> = Vec::new();
        for col in &self.cols {
            for &(idx, _v) in col.entries() {
                pairs.push(idx >> 1);
            }
        }
        pairs.sort_unstable();
        pairs.dedup();

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }

                let mut bit_sum = K::ZERO;
                for (col, w) in self.cols.iter().zip(self.weights.iter()) {
                    let b0 = col.get(child0);
                    let b1 = col.get(child1);
                    if b0 == K::ZERO && b1 == K::ZERO {
                        continue;
                    }
                    let b_x = interp(b0, b1, x);
                    bit_sum += *w * b_x * (b_x - K::ONE);
                }

                ys[i] += chi_x * bit_sum;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        for col in self.cols.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.challenges.push(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for Twist read-check (time rounds only).
pub struct TwistReadCheckOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    degree_bound: usize,

    r_addr: Vec<K>,
    ra_bits: Vec<SparseIdxVec<K>>,
    has_read: SparseIdxVec<K>,
    rv: SparseIdxVec<K>,

    init_at_r_addr: K,
    inc_terms_at_r_addr: Vec<(usize, K)>,

    t_child0: Vec<K>,
    t_child1: Vec<K>,
    challenges: Vec<K>,
}

impl TwistReadCheckOracleSparseTime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        r_cycle: &[K],
        has_read: SparseIdxVec<K>,
        rv: SparseIdxVec<K>,
        ra_bits: Vec<SparseIdxVec<K>>,
        // Write stream (for Val_pre).
        has_write: SparseIdxVec<K>,
        inc_at_write_addr: SparseIdxVec<K>,
        wa_bits: Vec<SparseIdxVec<K>>,
        r_addr: &[K],
        init_at_r_addr: K,
    ) -> Self {
        let ell_n = r_cycle.len();
        let ell_addr = r_addr.len();
        debug_assert_eq!(has_read.len(), 1usize << ell_n);
        debug_assert_eq!(rv.len(), 1usize << ell_n);
        debug_assert_eq!(has_write.len(), 1usize << ell_n);
        debug_assert_eq!(inc_at_write_addr.len(), 1usize << ell_n);
        debug_assert_eq!(ra_bits.len(), ell_addr);
        debug_assert_eq!(wa_bits.len(), ell_addr);

        let inc_terms_at_r_addr = build_inc_terms_at_r_addr(&wa_bits, &has_write, &inc_at_write_addr, r_addr);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            degree_bound: 3 + ell_addr,
            r_addr: r_addr.to_vec(),
            ra_bits,
            has_read,
            rv,
            init_at_r_addr,
            inc_terms_at_r_addr,
            t_child0: vec![K::ZERO; ell_n],
            t_child1: vec![K::ZERO; ell_n],
            challenges: Vec::with_capacity(ell_n),
        }
    }

    pub fn new_with_inc_terms(
        r_cycle: &[K],
        has_read: SparseIdxVec<K>,
        rv: SparseIdxVec<K>,
        ra_bits: Vec<SparseIdxVec<K>>,
        r_addr: &[K],
        init_at_r_addr: K,
        inc_terms_at_r_addr: Vec<(usize, K)>,
    ) -> Self {
        let ell_n = r_cycle.len();
        let ell_addr = r_addr.len();
        debug_assert_eq!(has_read.len(), 1usize << ell_n);
        debug_assert_eq!(rv.len(), 1usize << ell_n);
        debug_assert_eq!(ra_bits.len(), ell_addr);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            degree_bound: 3 + ell_addr,
            r_addr: r_addr.to_vec(),
            ra_bits,
            has_read,
            rv,
            init_at_r_addr,
            inc_terms_at_r_addr,
            t_child0: vec![K::ZERO; ell_n],
            t_child1: vec![K::ZERO; ell_n],
            challenges: Vec::with_capacity(ell_n),
        }
    }
}

impl RoundOracle for TwistReadCheckOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_read.len() == 1 {
            let mut eq_addr = K::ONE;
            for (b, col) in self.ra_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.singleton_value(), self.r_addr[b]);
            }
            let t_point = self.challenges.as_slice();
            let val_pre = val_pre_from_inc_terms(self.init_at_r_addr, &self.inc_terms_at_r_addr, t_point);
            let diff = val_pre - self.rv.singleton_value();
            let v = self.prefix_eq * self.has_read.singleton_value() * diff * eq_addr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_read.entries());
        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_read.get(child0);
            let gate1 = self.has_read.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            fill_time_point(&mut self.t_child0, &self.challenges, self.bit_idx, K::ZERO, pair);
            fill_time_point(&mut self.t_child1, &self.challenges, self.bit_idx, K::ONE, pair);
            let val_pre0 = val_pre_from_inc_terms(self.init_at_r_addr, &self.inc_terms_at_r_addr, &self.t_child0);
            let val_pre1 = val_pre_from_inc_terms(self.init_at_r_addr, &self.inc_terms_at_r_addr, &self.t_child1);

            let diff0 = val_pre0 - self.rv.get(child0);
            let diff1 = val_pre1 - self.rv.get(child1);

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            let mut eq0s: Vec<K> = Vec::with_capacity(self.ra_bits.len());
            let mut d_eqs: Vec<K> = Vec::with_capacity(self.ra_bits.len());
            for (b, col) in self.ra_bits.iter().enumerate() {
                let e0 = eq_bit_affine(col.get(child0), self.r_addr[b]);
                let e1 = eq_bit_affine(col.get(child1), self.r_addr[b]);
                eq0s.push(e0);
                d_eqs.push(e1 - e0);
            }

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                let gate_x = interp(gate0, gate1, x);
                let diff_x = interp(diff0, diff1, x);
                let mut prod = chi_x * gate_x * diff_x;
                for (e0, de) in eq0s.iter().zip(d_eqs.iter()) {
                    prod *= *e0 + *de * x;
                }
                ys[i] += prod;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_read.fold_round_in_place(r);
        self.rv.fold_round_in_place(r);
        for col in self.ra_bits.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.challenges.push(r);
        self.bit_idx += 1;
    }
}

/// Sparse Route A oracle for Twist write-check (time rounds only).
pub struct TwistWriteCheckOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    degree_bound: usize,

    r_addr: Vec<K>,
    wa_bits: Vec<SparseIdxVec<K>>,
    has_write: SparseIdxVec<K>,
    wv: SparseIdxVec<K>,
    inc_at_write_addr: SparseIdxVec<K>,

    init_at_r_addr: K,
    inc_terms_at_r_addr: Vec<(usize, K)>,

    t_child0: Vec<K>,
    t_child1: Vec<K>,
    challenges: Vec<K>,
}

impl TwistWriteCheckOracleSparseTime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        r_cycle: &[K],
        has_write: SparseIdxVec<K>,
        wv: SparseIdxVec<K>,
        inc_at_write_addr: SparseIdxVec<K>,
        wa_bits: Vec<SparseIdxVec<K>>,
        r_addr: &[K],
        init_at_r_addr: K,
    ) -> Self {
        let ell_n = r_cycle.len();
        let ell_addr = r_addr.len();
        debug_assert_eq!(has_write.len(), 1usize << ell_n);
        debug_assert_eq!(wv.len(), 1usize << ell_n);
        debug_assert_eq!(inc_at_write_addr.len(), 1usize << ell_n);
        debug_assert_eq!(wa_bits.len(), ell_addr);

        let inc_terms_at_r_addr = build_inc_terms_at_r_addr(&wa_bits, &has_write, &inc_at_write_addr, r_addr);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            degree_bound: 3 + ell_addr,
            r_addr: r_addr.to_vec(),
            wa_bits,
            has_write,
            wv,
            inc_at_write_addr,
            init_at_r_addr,
            inc_terms_at_r_addr,
            t_child0: vec![K::ZERO; ell_n],
            t_child1: vec![K::ZERO; ell_n],
            challenges: Vec::with_capacity(ell_n),
        }
    }

    pub fn new_with_inc_terms(
        r_cycle: &[K],
        has_write: SparseIdxVec<K>,
        wv: SparseIdxVec<K>,
        inc_at_write_addr: SparseIdxVec<K>,
        wa_bits: Vec<SparseIdxVec<K>>,
        r_addr: &[K],
        init_at_r_addr: K,
        inc_terms_at_r_addr: Vec<(usize, K)>,
    ) -> Self {
        let ell_n = r_cycle.len();
        let ell_addr = r_addr.len();
        debug_assert_eq!(has_write.len(), 1usize << ell_n);
        debug_assert_eq!(wv.len(), 1usize << ell_n);
        debug_assert_eq!(inc_at_write_addr.len(), 1usize << ell_n);
        debug_assert_eq!(wa_bits.len(), ell_addr);

        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            degree_bound: 3 + ell_addr,
            r_addr: r_addr.to_vec(),
            wa_bits,
            has_write,
            wv,
            inc_at_write_addr,
            init_at_r_addr,
            inc_terms_at_r_addr,
            t_child0: vec![K::ZERO; ell_n],
            t_child1: vec![K::ZERO; ell_n],
            challenges: Vec::with_capacity(ell_n),
        }
    }
}

impl RoundOracle for TwistWriteCheckOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_write.len() == 1 {
            let mut eq_addr = K::ONE;
            for (b, col) in self.wa_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.singleton_value(), self.r_addr[b]);
            }
            let t_point = self.challenges.as_slice();
            let val_pre = val_pre_from_inc_terms(self.init_at_r_addr, &self.inc_terms_at_r_addr, t_point);
            let delta = self.wv.singleton_value() - val_pre - self.inc_at_write_addr.singleton_value();
            let v = self.prefix_eq * self.has_write.singleton_value() * delta * eq_addr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_write.entries());
        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_write.get(child0);
            let gate1 = self.has_write.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            fill_time_point(&mut self.t_child0, &self.challenges, self.bit_idx, K::ZERO, pair);
            fill_time_point(&mut self.t_child1, &self.challenges, self.bit_idx, K::ONE, pair);
            let val_pre0 = val_pre_from_inc_terms(self.init_at_r_addr, &self.inc_terms_at_r_addr, &self.t_child0);
            let val_pre1 = val_pre_from_inc_terms(self.init_at_r_addr, &self.inc_terms_at_r_addr, &self.t_child1);

            let delta0 = self.wv.get(child0) - val_pre0 - self.inc_at_write_addr.get(child0);
            let delta1 = self.wv.get(child1) - val_pre1 - self.inc_at_write_addr.get(child1);

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);

            let mut eq0s: Vec<K> = Vec::with_capacity(self.wa_bits.len());
            let mut d_eqs: Vec<K> = Vec::with_capacity(self.wa_bits.len());
            for (b, col) in self.wa_bits.iter().enumerate() {
                let e0 = eq_bit_affine(col.get(child0), self.r_addr[b]);
                let e1 = eq_bit_affine(col.get(child1), self.r_addr[b]);
                eq0s.push(e0);
                d_eqs.push(e1 - e0);
            }

            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                let gate_x = interp(gate0, gate1, x);
                let delta_x = interp(delta0, delta1, x);
                let mut prod = chi_x * gate_x * delta_x;
                for (e0, de) in eq0s.iter().zip(d_eqs.iter()) {
                    prod *= *e0 + *de * x;
                }
                ys[i] += prod;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single(r, self.r_cycle[self.bit_idx]);
        self.has_write.fold_round_in_place(r);
        self.wv.fold_round_in_place(r);
        self.inc_at_write_addr.fold_round_in_place(r);
        for col in self.wa_bits.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.challenges.push(r);
        self.bit_idx += 1;
    }
}

/// Sparse time-domain val-evaluation oracle:
///   Σ_t has_write(t) · inc(t) · LT(t, r_time) · Π_b eq(wa_bit_b(t), r_addr_b)
pub struct TwistValEvalOracleSparseTime {
    bit_idx: usize,
    degree_bound: usize,

    r_time: Vec<K>,
    r_addr: Vec<K>,

    wa_bits: Vec<SparseIdxVec<K>>,
    has_write: SparseIdxVec<K>,
    inc_at_write_addr: SparseIdxVec<K>,

    t_child0: Vec<K>,
    t_child1: Vec<K>,
    challenges: Vec<K>,
}

impl TwistValEvalOracleSparseTime {
    pub fn new(
        wa_bits: Vec<SparseIdxVec<K>>,
        has_write: SparseIdxVec<K>,
        inc_at_write_addr: SparseIdxVec<K>,
        r_addr: &[K],
        r_time: &[K],
    ) -> (Self, K) {
        let ell_n = r_time.len();
        let ell_addr = r_addr.len();
        debug_assert_eq!(has_write.len(), 1usize << ell_n);
        debug_assert_eq!(inc_at_write_addr.len(), 1usize << ell_n);
        debug_assert_eq!(wa_bits.len(), ell_addr);
        for (b, col) in wa_bits.iter().enumerate() {
            debug_assert_eq!(col.len(), 1usize << ell_n, "wa_bits[{b}] length must match time domain");
        }

        let mut claim = K::ZERO;
        for &(t, gate) in has_write.entries() {
            let inc_t = inc_at_write_addr.get(t);
            if gate == K::ZERO || inc_t == K::ZERO {
                continue;
            }
            let lt_t = lt_eval_at_bool_index(t, r_time);
            let mut eq_addr = K::ONE;
            for (b, col) in wa_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.get(t), r_addr[b]);
            }
            claim += gate * inc_t * lt_t * eq_addr;
        }

        (
            Self {
                bit_idx: 0,
                degree_bound: 3 + ell_addr,
                r_time: r_time.to_vec(),
                r_addr: r_addr.to_vec(),
                wa_bits,
                has_write,
                inc_at_write_addr,
                t_child0: vec![K::ZERO; ell_n],
                t_child1: vec![K::ZERO; ell_n],
                challenges: Vec::with_capacity(ell_n),
            },
            claim,
        )
    }
}

impl RoundOracle for TwistValEvalOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_write.len() == 1 {
            let mut eq_addr = K::ONE;
            for (b, col) in self.wa_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.singleton_value(), self.r_addr[b]);
            }
            let lt = lt_eval(&self.challenges, &self.r_time);
            let v = self.has_write.singleton_value() * self.inc_at_write_addr.singleton_value() * eq_addr * lt;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_write.entries());
        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_write.get(child0);
            let gate1 = self.has_write.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }
            let inc0 = self.inc_at_write_addr.get(child0);
            let inc1 = self.inc_at_write_addr.get(child1);

            fill_time_point(&mut self.t_child0, &self.challenges, self.bit_idx, K::ZERO, pair);
            fill_time_point(&mut self.t_child1, &self.challenges, self.bit_idx, K::ONE, pair);
            let lt0 = lt_eval(&self.t_child0, &self.r_time);
            let lt1 = lt_eval(&self.t_child1, &self.r_time);

            let mut eq0s: Vec<K> = Vec::with_capacity(self.wa_bits.len());
            let mut d_eqs: Vec<K> = Vec::with_capacity(self.wa_bits.len());
            for (b, col) in self.wa_bits.iter().enumerate() {
                let e0 = eq_bit_affine(col.get(child0), self.r_addr[b]);
                let e1 = eq_bit_affine(col.get(child1), self.r_addr[b]);
                eq0s.push(e0);
                d_eqs.push(e1 - e0);
            }

            for (i, &x) in points.iter().enumerate() {
                let gate_x = interp(gate0, gate1, x);
                let inc_x = interp(inc0, inc1, x);
                let lt_x = interp(lt0, lt1, x);
                let mut prod = gate_x * inc_x * lt_x;
                for (e0, de) in eq0s.iter().zip(d_eqs.iter()) {
                    prod *= *e0 + *de * x;
                }
                ys[i] += prod;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_time.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.has_write.fold_round_in_place(r);
        self.inc_at_write_addr.fold_round_in_place(r);
        for col in self.wa_bits.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.challenges.push(r);
        self.bit_idx += 1;
    }
}

/// Sparse time-domain total-increment oracle:
///   Σ_t has_write(t) · inc(t) · Π_b eq(wa_bit_b(t), r_addr_b)
pub struct TwistTotalIncOracleSparseTime {
    degree_bound: usize,

    r_addr: Vec<K>,
    wa_bits: Vec<SparseIdxVec<K>>,
    has_write: SparseIdxVec<K>,
    inc_at_write_addr: SparseIdxVec<K>,
}

impl TwistTotalIncOracleSparseTime {
    pub fn new(
        wa_bits: Vec<SparseIdxVec<K>>,
        has_write: SparseIdxVec<K>,
        inc_at_write_addr: SparseIdxVec<K>,
        r_addr: &[K],
    ) -> (Self, K) {
        let ell_n = log2_pow2(has_write.len());
        let ell_addr = r_addr.len();
        debug_assert_eq!(inc_at_write_addr.len(), 1usize << ell_n);
        debug_assert_eq!(wa_bits.len(), ell_addr);
        for (b, col) in wa_bits.iter().enumerate() {
            debug_assert_eq!(col.len(), 1usize << ell_n, "wa_bits[{b}] length must match time domain");
        }

        let mut claim = K::ZERO;
        for &(t, gate) in has_write.entries() {
            let inc_t = inc_at_write_addr.get(t);
            if gate == K::ZERO || inc_t == K::ZERO {
                continue;
            }
            let mut eq_addr = K::ONE;
            for (b, col) in wa_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.get(t), r_addr[b]);
            }
            claim += gate * inc_t * eq_addr;
        }

        (
            Self {
                degree_bound: 2 + ell_addr,
                r_addr: r_addr.to_vec(),
                wa_bits,
                has_write,
                inc_at_write_addr,
            },
            claim,
        )
    }
}

impl RoundOracle for TwistTotalIncOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_write.len() == 1 {
            let mut eq_addr = K::ONE;
            for (b, col) in self.wa_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.singleton_value(), self.r_addr[b]);
            }
            let v = self.has_write.singleton_value() * self.inc_at_write_addr.singleton_value() * eq_addr;
            return vec![v; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_write.entries());
        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_write.get(child0);
            let gate1 = self.has_write.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }
            let inc0 = self.inc_at_write_addr.get(child0);
            let inc1 = self.inc_at_write_addr.get(child1);

            let mut eq0s: Vec<K> = Vec::with_capacity(self.wa_bits.len());
            let mut d_eqs: Vec<K> = Vec::with_capacity(self.wa_bits.len());
            for (b, col) in self.wa_bits.iter().enumerate() {
                let e0 = eq_bit_affine(col.get(child0), self.r_addr[b]);
                let e1 = eq_bit_affine(col.get(child1), self.r_addr[b]);
                eq0s.push(e0);
                d_eqs.push(e1 - e0);
            }

            for (i, &x) in points.iter().enumerate() {
                let gate_x = interp(gate0, gate1, x);
                let inc_x = interp(inc0, inc1, x);
                let mut prod = gate_x * inc_x;
                for (e0, de) in eq0s.iter().zip(d_eqs.iter()) {
                    prod *= *e0 + *de * x;
                }
                ys[i] += prod;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        log2_pow2(self.has_write.len())
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.has_write.len() == 1 {
            return;
        }
        self.has_write.fold_round_in_place(r);
        self.inc_at_write_addr.fold_round_in_place(r);
        for col in self.wa_bits.iter_mut() {
            col.fold_round_in_place(r);
        }
    }
}

// ============================================================================
// Val-Evaluation Oracle (SPARSE version)
// ============================================================================

// ============================================================================
// Address-Lane Oracles (addr rounds first, time summed)
// ============================================================================
//
// These are used by the "Phase 2" Route A integration to avoid materializing a
// time×addr table when Twist must share `r_time` with CCS:
//   1) Run an address-lane sum-check first (ell_addr rounds) to bind `r_addr`
//      and produce the *time-lane claimed sums* (one per check).
//   2) Run a time-lane sum-check (ell_n rounds) at the fixed `r_addr`.
//
// Concretely, these oracles implement the address-lane prefix of the same
// read/write check polynomials as the 2D oracles, but with the time variables
// summed internally. This keeps address rounds efficient and avoids allocating
// `pow2_time * pow2_addr`.
//
// Degree in each address variable is ≤ 2:
// - `Val_pre(addr, t)` is multilinear in `addr` (degree 1 per bit),
// - `eq(addr, bits(t))` is multilinear in `addr`,
// so their product has degree ≤ 2 per address bit.
//
// Variable order is little-endian address bits (bit 0 first), matching the
// rest of the module.

fn update_prefix_weights_in_place(weights: &mut [K], addrs: &[usize], bit_idx: usize, r: K) {
    let r0 = K::ONE - r;
    for (w, &a) in weights.iter_mut().zip(addrs.iter()) {
        if ((a >> bit_idx) & 1) == 1 {
            *w *= r;
        } else {
            *w *= r0;
        }
    }
}

// ============================================================================
// Address-lane oracles (Track A sparse-in-time variants)
// ============================================================================

fn addr_from_sparse_bits_at_time(bit_cols: &[SparseIdxVec<K>], t: usize) -> usize {
    let mut out = 0usize;
    for (b, col) in bit_cols.iter().enumerate() {
        if col.get(t) == K::ONE {
            out |= 1usize << b;
        }
    }
    out
}

fn merge_sparse_time_indices(a: &[(usize, K)], b: &[(usize, K)]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() || j < b.len() {
        let next = match (a.get(i), b.get(j)) {
            (Some(&(ai, _)), Some(&(bj, _))) => {
                if ai <= bj {
                    i += 1;
                    ai
                } else {
                    j += 1;
                    bj
                }
            }
            (Some(&(ai, _)), None) => {
                i += 1;
                ai
            }
            (None, Some(&(bj, _))) => {
                j += 1;
                bj
            }
            (None, None) => break,
        };
        if out.last().copied() != Some(next) {
            out.push(next);
        }
    }
    out
}

/// Sparse-in-time address-lane prefix oracle for the Twist read-check.
///
/// Same proof semantics as `TwistReadCheckAddrOracle`, but internal time iteration scales with
/// activity (`nnz(has_read) + nnz(has_write)`), not `T = 2^ell_n`.
pub struct TwistReadCheckAddrOracleSparseTime {
    ell_addr: usize,
    bit_idx: usize,
    degree_bound: usize,

    mem_scratch: std::collections::HashMap<usize, K>,

    // Per-event (sparse time list) arrays, sorted by time index.
    eq_cycle: Vec<K>,
    has_read: Vec<K>,
    rv: Vec<K>,
    has_write: Vec<K>,
    inc_at_write_addr: Vec<K>,
    ra_addrs: Vec<usize>,
    wa_addrs: Vec<usize>,
    ra_prefix_w: Vec<K>,
    wa_prefix_w: Vec<K>,

    init_addrs: Vec<usize>,
    init_vals: Vec<K>,
    init_prefix_w: Vec<K>,
}

impl TwistReadCheckAddrOracleSparseTime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        init_sparse: Vec<(usize, K)>,
        r_cycle: &[K],
        has_read: SparseIdxVec<K>,
        rv: SparseIdxVec<K>,
        ra_bits: &[SparseIdxVec<K>],
        has_write: SparseIdxVec<K>,
        wa_bits: &[SparseIdxVec<K>],
        inc_at_write_addr: SparseIdxVec<K>,
    ) -> Self {
        let pow2_time = 1usize << r_cycle.len();
        assert_eq!(has_read.len(), pow2_time, "has_read length must match time domain");
        assert_eq!(rv.len(), pow2_time, "rv length must match time domain");
        assert_eq!(has_write.len(), pow2_time, "has_write length must match time domain");
        assert_eq!(
            inc_at_write_addr.len(),
            pow2_time,
            "inc_at_write_addr length must match time domain"
        );

        let ell_addr = ra_bits.len();
        assert_eq!(wa_bits.len(), ell_addr, "wa_bits/ra_bits length mismatch");
        let pow2_addr = 1usize << ell_addr;
        for (addr, _) in init_sparse.iter() {
            assert!(*addr < pow2_addr, "init address out of range");
        }
        for (b, col) in ra_bits.iter().enumerate() {
            assert_eq!(col.len(), pow2_time, "ra_bits[{b}] length mismatch");
        }
        for (b, col) in wa_bits.iter().enumerate() {
            assert_eq!(col.len(), pow2_time, "wa_bits[{b}] length mismatch");
        }

        let times = merge_sparse_time_indices(has_read.entries(), has_write.entries());
        let mut eq_cycle_out = Vec::with_capacity(times.len());
        let mut has_read_out = Vec::with_capacity(times.len());
        let mut rv_out = Vec::with_capacity(times.len());
        let mut has_write_out = Vec::with_capacity(times.len());
        let mut inc_out = Vec::with_capacity(times.len());
        let mut ra_addrs = Vec::with_capacity(times.len());
        let mut wa_addrs = Vec::with_capacity(times.len());

        for &t in times.iter() {
            eq_cycle_out.push(chi_at_bool_index(r_cycle, t));
            let hr = has_read.get(t);
            let hw = has_write.get(t);
            has_read_out.push(hr);
            rv_out.push(rv.get(t));
            has_write_out.push(hw);
            inc_out.push(inc_at_write_addr.get(t));

            ra_addrs.push(if hr != K::ZERO {
                addr_from_sparse_bits_at_time(ra_bits, t)
            } else {
                0
            });
            wa_addrs.push(if hw != K::ZERO {
                addr_from_sparse_bits_at_time(wa_bits, t)
            } else {
                0
            });
        }

        let (init_addrs, init_vals): (Vec<usize>, Vec<K>) = init_sparse.into_iter().unzip();
        Self {
            ell_addr,
            bit_idx: 0,
            degree_bound: 2,
            mem_scratch: std::collections::HashMap::with_capacity(init_addrs.len()),
            eq_cycle: eq_cycle_out,
            has_read: has_read_out,
            rv: rv_out,
            has_write: has_write_out,
            inc_at_write_addr: inc_out,
            ra_addrs,
            wa_addrs,
            ra_prefix_w: vec![K::ONE; times.len()],
            wa_prefix_w: vec![K::ONE; times.len()],
            init_prefix_w: vec![K::ONE; init_addrs.len()],
            init_addrs,
            init_vals,
        }
    }
}

impl RoundOracle for TwistReadCheckAddrOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.num_rounds() == 0 {
            let mut mem = K::ZERO;
            for (&val, &w) in self.init_vals.iter().zip(self.init_prefix_w.iter()) {
                mem += val * w;
            }

            let mut sum = K::ZERO;
            for i in 0..self.eq_cycle.len() {
                let eq_t = self.eq_cycle[i];
                let gate_r = self.has_read[i];
                if gate_r != K::ZERO {
                    sum += eq_t * gate_r * self.ra_prefix_w[i] * (mem - self.rv[i]);
                }

                let gate_w = self.has_write[i];
                if gate_w != K::ZERO {
                    mem += self.inc_at_write_addr[i] * gate_w * self.wa_prefix_w[i];
                }
            }
            return vec![sum; points.len()];
        }

        let bit_idx = self.bit_idx;
        let mut ys = vec![K::ZERO; points.len()];

        self.mem_scratch.clear();
        let mem = &mut self.mem_scratch;
        for ((&addr, &val), &w) in self
            .init_addrs
            .iter()
            .zip(self.init_vals.iter())
            .zip(self.init_prefix_w.iter())
        {
            let idx = addr >> bit_idx;
            let contrib = val * w;
            if contrib != K::ZERO {
                *mem.entry(idx).or_insert(K::ZERO) += contrib;
            }
        }

        for i in 0..self.eq_cycle.len() {
            let eq_t = self.eq_cycle[i];

            let gate_r = self.has_read[i];
            if gate_r != K::ZERO {
                let ra = self.ra_addrs[i];
                let base = ra >> (bit_idx + 1);
                let idx0 = base * 2;
                let idx1 = idx0 + 1;
                let v0 = mem.get(&idx0).copied().unwrap_or(K::ZERO);
                let v1 = mem.get(&idx1).copied().unwrap_or(K::ZERO);
                let dv = v1 - v0;
                let rv_t = self.rv[i];
                let prefix = self.ra_prefix_w[i];
                let bit = (ra >> bit_idx) & 1;

                for (j, &x) in points.iter().enumerate() {
                    let val_x = v0 + dv * x;
                    let addr_factor = if bit == 1 { x } else { K::ONE - x };
                    ys[j] += eq_t * gate_r * prefix * addr_factor * (val_x - rv_t);
                }
            }

            let gate_w = self.has_write[i];
            if gate_w != K::ZERO {
                let wa = self.wa_addrs[i];
                let idx = wa >> bit_idx;
                let delta = self.inc_at_write_addr[i] * gate_w * self.wa_prefix_w[i];
                if delta != K::ZERO {
                    *mem.entry(idx).or_insert(K::ZERO) += delta;
                }
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.ell_addr.saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        update_prefix_weights_in_place(&mut self.init_prefix_w, &self.init_addrs, self.bit_idx, r);
        update_prefix_weights_in_place(&mut self.ra_prefix_w, &self.ra_addrs, self.bit_idx, r);
        update_prefix_weights_in_place(&mut self.wa_prefix_w, &self.wa_addrs, self.bit_idx, r);
        self.bit_idx += 1;
    }
}

/// Sparse-in-time address-lane prefix oracle for the Twist write-check.
pub struct TwistWriteCheckAddrOracleSparseTime {
    ell_addr: usize,
    bit_idx: usize,
    degree_bound: usize,

    mem_scratch: std::collections::HashMap<usize, K>,

    eq_cycle: Vec<K>,
    has_write: Vec<K>,
    wv: Vec<K>,
    inc_at_write_addr: Vec<K>,
    wa_addrs: Vec<usize>,
    wa_prefix_w: Vec<K>,

    init_addrs: Vec<usize>,
    init_vals: Vec<K>,
    init_prefix_w: Vec<K>,
}

impl TwistWriteCheckAddrOracleSparseTime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        init_sparse: Vec<(usize, K)>,
        r_cycle: &[K],
        has_write: SparseIdxVec<K>,
        wv: SparseIdxVec<K>,
        wa_bits: &[SparseIdxVec<K>],
        inc_at_write_addr: SparseIdxVec<K>,
    ) -> Self {
        let pow2_time = 1usize << r_cycle.len();
        assert_eq!(has_write.len(), pow2_time, "has_write length must match time domain");
        assert_eq!(wv.len(), pow2_time, "wv length must match time domain");
        assert_eq!(
            inc_at_write_addr.len(),
            pow2_time,
            "inc_at_write_addr length must match time domain"
        );

        let ell_addr = wa_bits.len();
        let pow2_addr = 1usize << ell_addr;
        for (addr, _) in init_sparse.iter() {
            assert!(*addr < pow2_addr, "init address out of range");
        }
        for (b, col) in wa_bits.iter().enumerate() {
            assert_eq!(col.len(), pow2_time, "wa_bits[{b}] length mismatch");
        }

        let times: Vec<usize> = has_write.entries().iter().map(|(t, _)| *t).collect();
        let mut eq_cycle_out = Vec::with_capacity(times.len());
        let mut has_write_out = Vec::with_capacity(times.len());
        let mut wv_out = Vec::with_capacity(times.len());
        let mut inc_out = Vec::with_capacity(times.len());
        let mut wa_addrs = Vec::with_capacity(times.len());

        for &t in times.iter() {
            eq_cycle_out.push(chi_at_bool_index(r_cycle, t));
            let hw = has_write.get(t);
            has_write_out.push(hw);
            wv_out.push(wv.get(t));
            inc_out.push(inc_at_write_addr.get(t));
            wa_addrs.push(addr_from_sparse_bits_at_time(wa_bits, t));
        }

        let (init_addrs, init_vals): (Vec<usize>, Vec<K>) = init_sparse.into_iter().unzip();
        Self {
            ell_addr,
            bit_idx: 0,
            degree_bound: 2,
            mem_scratch: std::collections::HashMap::with_capacity(init_addrs.len()),
            eq_cycle: eq_cycle_out,
            has_write: has_write_out,
            wv: wv_out,
            inc_at_write_addr: inc_out,
            wa_addrs,
            init_prefix_w: vec![K::ONE; init_addrs.len()],
            init_addrs,
            init_vals,
            wa_prefix_w: vec![K::ONE; times.len()],
        }
    }
}

impl RoundOracle for TwistWriteCheckAddrOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.num_rounds() == 0 {
            let mut mem = K::ZERO;
            for (&val, &w) in self.init_vals.iter().zip(self.init_prefix_w.iter()) {
                mem += val * w;
            }

            let mut sum = K::ZERO;
            for i in 0..self.eq_cycle.len() {
                let delta = self.wv[i] - mem - self.inc_at_write_addr[i];
                sum += self.eq_cycle[i] * self.has_write[i] * self.wa_prefix_w[i] * delta;

                let gate_w = self.has_write[i];
                if gate_w != K::ZERO {
                    mem += self.inc_at_write_addr[i] * gate_w * self.wa_prefix_w[i];
                }
            }
            return vec![sum; points.len()];
        }

        let bit_idx = self.bit_idx;
        let mut ys = vec![K::ZERO; points.len()];

        self.mem_scratch.clear();
        let mem = &mut self.mem_scratch;
        for ((&addr, &val), &w) in self
            .init_addrs
            .iter()
            .zip(self.init_vals.iter())
            .zip(self.init_prefix_w.iter())
        {
            let idx = addr >> bit_idx;
            let contrib = val * w;
            if contrib != K::ZERO {
                *mem.entry(idx).or_insert(K::ZERO) += contrib;
            }
        }

        for i in 0..self.eq_cycle.len() {
            let eq_t = self.eq_cycle[i];
            let gate = self.has_write[i];
            if gate != K::ZERO {
                let wa = self.wa_addrs[i];
                let base = wa >> (bit_idx + 1);
                let idx0 = base * 2;
                let idx1 = idx0 + 1;
                let v0 = mem.get(&idx0).copied().unwrap_or(K::ZERO);
                let v1 = mem.get(&idx1).copied().unwrap_or(K::ZERO);
                let dv = v1 - v0;
                let wv_t = self.wv[i];
                let inc_t = self.inc_at_write_addr[i];
                let prefix = self.wa_prefix_w[i];
                let bit = (wa >> bit_idx) & 1;

                for (j, &x) in points.iter().enumerate() {
                    let val_x = v0 + dv * x;
                    let addr_factor = if bit == 1 { x } else { K::ONE - x };
                    ys[j] += eq_t * gate * prefix * addr_factor * (wv_t - val_x - inc_t);
                }
            }

            let gate_w = self.has_write[i];
            if gate_w != K::ZERO {
                let wa = self.wa_addrs[i];
                let idx = wa >> bit_idx;
                let delta = self.inc_at_write_addr[i] * gate_w * self.wa_prefix_w[i];
                if delta != K::ZERO {
                    *mem.entry(idx).or_insert(K::ZERO) += delta;
                }
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.ell_addr.saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        update_prefix_weights_in_place(&mut self.init_prefix_w, &self.init_addrs, self.bit_idx, r);
        update_prefix_weights_in_place(&mut self.wa_prefix_w, &self.wa_addrs, self.bit_idx, r);
        self.bit_idx += 1;
    }
}

/// Multi-lane variant of `TwistReadCheckAddrOracleSparseTime`.
///
/// This oracle supports multiple Twist access lanes per CPU step by treating each lane's read/write
/// activity as an independent sparse event stream, ordered by `(time, op_kind, lane)` where
/// `op_kind` is read-before-write.
///
/// Semantics match Track A: reads observe pre-state at time `t`, writes are applied after reads at
/// the same `t`. Multiple writes to the same address at the same `t` are **not** supported by this
/// oracle (the caller must disallow or canonicalize them).
pub struct TwistReadCheckAddrOracleSparseTimeMultiLane {
    ell_addr: usize,
    bit_idx: usize,
    degree_bound: usize,

    mem_scratch: std::collections::HashMap<usize, K>,

    // Per-event (sparse time list) arrays, sorted by (time, op_kind, lane).
    eq_cycle: Vec<K>,
    has_read: Vec<K>,
    rv: Vec<K>,
    has_write: Vec<K>,
    inc_at_write_addr: Vec<K>,
    ra_addrs: Vec<usize>,
    wa_addrs: Vec<usize>,
    ra_prefix_w: Vec<K>,
    wa_prefix_w: Vec<K>,

    init_addrs: Vec<usize>,
    init_vals: Vec<K>,
    init_prefix_w: Vec<K>,
}

impl TwistReadCheckAddrOracleSparseTimeMultiLane {
    pub fn new(init_sparse: Vec<(usize, K)>, r_cycle: &[K], lanes: &[TwistLaneSparseCols]) -> Self {
        assert!(!lanes.is_empty(), "multi-lane Twist oracle requires at least 1 lane");

        let pow2_time = 1usize << r_cycle.len();
        let ell_addr = lanes[0].ra_bits.len();
        let pow2_addr = 1usize << ell_addr;

        for (addr, _) in init_sparse.iter() {
            assert!(*addr < pow2_addr, "init address out of range");
        }

        for (lane_idx, lane) in lanes.iter().enumerate() {
            assert_eq!(
                lane.has_read.len(),
                pow2_time,
                "has_read length must match time domain (lane={lane_idx})"
            );
            assert_eq!(
                lane.rv.len(),
                pow2_time,
                "rv length must match time domain (lane={lane_idx})"
            );
            assert_eq!(
                lane.has_write.len(),
                pow2_time,
                "has_write length must match time domain (lane={lane_idx})"
            );
            assert_eq!(
                lane.inc_at_write_addr.len(),
                pow2_time,
                "inc_at_write_addr length must match time domain (lane={lane_idx})"
            );
            assert_eq!(
                lane.ra_bits.len(),
                ell_addr,
                "ra_bits count must match ell_addr (lane={lane_idx})"
            );
            assert_eq!(
                lane.wa_bits.len(),
                ell_addr,
                "wa_bits count must match ell_addr (lane={lane_idx})"
            );
            for (b, col) in lane.ra_bits.iter().enumerate() {
                assert_eq!(col.len(), pow2_time, "ra_bits[{b}] length mismatch (lane={lane_idx})");
            }
            for (b, col) in lane.wa_bits.iter().enumerate() {
                assert_eq!(col.len(), pow2_time, "wa_bits[{b}] length mismatch (lane={lane_idx})");
            }
        }

        // Collect per-lane sparse events: reads first, then writes, at each time.
        let mut events: Vec<(usize, u8, usize)> = Vec::new();
        for (lane_idx, lane) in lanes.iter().enumerate() {
            events.extend(
                lane.has_read
                    .entries()
                    .iter()
                    .map(|&(t, _)| (t, 0u8, lane_idx)),
            );
            events.extend(
                lane.has_write
                    .entries()
                    .iter()
                    .map(|&(t, _)| (t, 1u8, lane_idx)),
            );
        }
        events.sort_unstable_by_key(|(t, kind, lane)| (*t, *kind, *lane));

        let mut eq_cycle_out = Vec::with_capacity(events.len());
        let mut has_read_out = Vec::with_capacity(events.len());
        let mut rv_out = Vec::with_capacity(events.len());
        let mut has_write_out = Vec::with_capacity(events.len());
        let mut inc_out = Vec::with_capacity(events.len());
        let mut ra_addrs = Vec::with_capacity(events.len());
        let mut wa_addrs = Vec::with_capacity(events.len());

        for (t, kind, lane_idx) in events.into_iter() {
            let lane = &lanes[lane_idx];
            eq_cycle_out.push(chi_at_bool_index(r_cycle, t));
            if kind == 0 {
                // Read
                let hr = lane.has_read.get(t);
                has_read_out.push(hr);
                rv_out.push(lane.rv.get(t));
                has_write_out.push(K::ZERO);
                inc_out.push(K::ZERO);
                ra_addrs.push(if hr != K::ZERO {
                    addr_from_sparse_bits_at_time(&lane.ra_bits, t)
                } else {
                    0
                });
                wa_addrs.push(0);
            } else {
                // Write
                let hw = lane.has_write.get(t);
                has_read_out.push(K::ZERO);
                rv_out.push(K::ZERO);
                has_write_out.push(hw);
                inc_out.push(lane.inc_at_write_addr.get(t));
                ra_addrs.push(0);
                wa_addrs.push(if hw != K::ZERO {
                    addr_from_sparse_bits_at_time(&lane.wa_bits, t)
                } else {
                    0
                });
            }
        }

        let events_len = eq_cycle_out.len();
        let (init_addrs, init_vals): (Vec<usize>, Vec<K>) = init_sparse.into_iter().unzip();
        Self {
            ell_addr,
            bit_idx: 0,
            degree_bound: 2,
            mem_scratch: std::collections::HashMap::with_capacity(init_addrs.len()),
            eq_cycle: eq_cycle_out,
            has_read: has_read_out,
            rv: rv_out,
            has_write: has_write_out,
            inc_at_write_addr: inc_out,
            ra_addrs,
            wa_addrs,
            ra_prefix_w: vec![K::ONE; events_len],
            wa_prefix_w: vec![K::ONE; events_len],
            init_prefix_w: vec![K::ONE; init_addrs.len()],
            init_addrs,
            init_vals,
        }
    }
}

impl RoundOracle for TwistReadCheckAddrOracleSparseTimeMultiLane {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.num_rounds() == 0 {
            let mut mem = K::ZERO;
            for (&val, &w) in self.init_vals.iter().zip(self.init_prefix_w.iter()) {
                mem += val * w;
            }

            let mut sum = K::ZERO;
            for i in 0..self.eq_cycle.len() {
                let diff = mem - self.rv[i];
                sum += self.eq_cycle[i] * self.has_read[i] * self.ra_prefix_w[i] * diff;

                let gate_w = self.has_write[i];
                if gate_w != K::ZERO {
                    mem += self.inc_at_write_addr[i] * gate_w * self.wa_prefix_w[i];
                }
            }
            return vec![sum; points.len()];
        }

        let bit_idx = self.bit_idx;
        let mut ys = vec![K::ZERO; points.len()];

        self.mem_scratch.clear();
        let mem = &mut self.mem_scratch;
        for ((&addr, &val), &w) in self
            .init_addrs
            .iter()
            .zip(self.init_vals.iter())
            .zip(self.init_prefix_w.iter())
        {
            let idx = addr >> bit_idx;
            let contrib = val * w;
            if contrib != K::ZERO {
                *mem.entry(idx).or_insert(K::ZERO) += contrib;
            }
        }

        for i in 0..self.eq_cycle.len() {
            let eq_t = self.eq_cycle[i];
            let gate_r = self.has_read[i];
            if gate_r != K::ZERO {
                let ra = self.ra_addrs[i];
                let base = ra >> (bit_idx + 1);
                let idx0 = base * 2;
                let idx1 = idx0 + 1;
                let v0 = mem.get(&idx0).copied().unwrap_or(K::ZERO);
                let v1 = mem.get(&idx1).copied().unwrap_or(K::ZERO);
                let dv = v1 - v0;
                let rv_t = self.rv[i];
                let prefix = self.ra_prefix_w[i];
                let bit = (ra >> bit_idx) & 1;

                for (j, &x) in points.iter().enumerate() {
                    let val_x = v0 + dv * x;
                    let addr_factor = if bit == 1 { x } else { K::ONE - x };
                    ys[j] += eq_t * gate_r * prefix * addr_factor * (val_x - rv_t);
                }
            }

            let gate_w = self.has_write[i];
            if gate_w != K::ZERO {
                let wa = self.wa_addrs[i];
                let idx = wa >> bit_idx;
                let delta = self.inc_at_write_addr[i] * gate_w * self.wa_prefix_w[i];
                if delta != K::ZERO {
                    *mem.entry(idx).or_insert(K::ZERO) += delta;
                }
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.ell_addr.saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        update_prefix_weights_in_place(&mut self.init_prefix_w, &self.init_addrs, self.bit_idx, r);
        update_prefix_weights_in_place(&mut self.ra_prefix_w, &self.ra_addrs, self.bit_idx, r);
        update_prefix_weights_in_place(&mut self.wa_prefix_w, &self.wa_addrs, self.bit_idx, r);
        self.bit_idx += 1;
    }
}

/// Multi-lane variant of `TwistWriteCheckAddrOracleSparseTime`.
///
/// Semantics are identical: multiple writes per time are allowed as independent sparse events.
pub struct TwistWriteCheckAddrOracleSparseTimeMultiLane {
    ell_addr: usize,
    bit_idx: usize,
    degree_bound: usize,

    mem_scratch: std::collections::HashMap<usize, K>,

    time_idxs: Vec<usize>,
    eq_cycle: Vec<K>,
    has_write: Vec<K>,
    wv: Vec<K>,
    inc_at_write_addr: Vec<K>,
    wa_addrs: Vec<usize>,
    wa_prefix_w: Vec<K>,

    init_addrs: Vec<usize>,
    init_vals: Vec<K>,
    init_prefix_w: Vec<K>,
}

impl TwistWriteCheckAddrOracleSparseTimeMultiLane {
    pub fn new(init_sparse: Vec<(usize, K)>, r_cycle: &[K], lanes: &[TwistLaneSparseCols]) -> Self {
        assert!(!lanes.is_empty(), "multi-lane Twist oracle requires at least 1 lane");

        let pow2_time = 1usize << r_cycle.len();
        let ell_addr = lanes[0].wa_bits.len();
        let pow2_addr = 1usize << ell_addr;

        for (addr, _) in init_sparse.iter() {
            assert!(*addr < pow2_addr, "init address out of range");
        }

        for (lane_idx, lane) in lanes.iter().enumerate() {
            assert_eq!(
                lane.has_write.len(),
                pow2_time,
                "has_write length must match time domain (lane={lane_idx})"
            );
            assert_eq!(
                lane.wv.len(),
                pow2_time,
                "wv length must match time domain (lane={lane_idx})"
            );
            assert_eq!(
                lane.inc_at_write_addr.len(),
                pow2_time,
                "inc_at_write_addr length must match time domain (lane={lane_idx})"
            );
            assert_eq!(
                lane.wa_bits.len(),
                ell_addr,
                "wa_bits count must match ell_addr (lane={lane_idx})"
            );
            for (b, col) in lane.wa_bits.iter().enumerate() {
                assert_eq!(col.len(), pow2_time, "wa_bits[{b}] length mismatch (lane={lane_idx})");
            }
        }

        let mut events: Vec<(usize, usize)> = Vec::new();
        for (lane_idx, lane) in lanes.iter().enumerate() {
            events.extend(lane.has_write.entries().iter().map(|&(t, _)| (t, lane_idx)));
        }
        events.sort_unstable_by_key(|(t, lane)| (*t, *lane));

        let mut eq_cycle_out = Vec::with_capacity(events.len());
        let mut has_write_out = Vec::with_capacity(events.len());
        let mut wv_out = Vec::with_capacity(events.len());
        let mut inc_out = Vec::with_capacity(events.len());
        let mut wa_addrs = Vec::with_capacity(events.len());
        let mut time_idxs = Vec::with_capacity(events.len());

        for (t, lane_idx) in events.into_iter() {
            let lane = &lanes[lane_idx];
            let hw = lane.has_write.get(t);
            time_idxs.push(t);
            eq_cycle_out.push(chi_at_bool_index(r_cycle, t));
            has_write_out.push(hw);
            wv_out.push(lane.wv.get(t));
            inc_out.push(lane.inc_at_write_addr.get(t));
            wa_addrs.push(addr_from_sparse_bits_at_time(&lane.wa_bits, t));
        }

        let events_len = eq_cycle_out.len();
        let (init_addrs, init_vals): (Vec<usize>, Vec<K>) = init_sparse.into_iter().unzip();
        Self {
            ell_addr,
            bit_idx: 0,
            degree_bound: 2,
            mem_scratch: std::collections::HashMap::with_capacity(init_addrs.len()),
            time_idxs,
            eq_cycle: eq_cycle_out,
            has_write: has_write_out,
            wv: wv_out,
            inc_at_write_addr: inc_out,
            wa_addrs,
            init_prefix_w: vec![K::ONE; init_addrs.len()],
            init_addrs,
            init_vals,
            wa_prefix_w: vec![K::ONE; events_len],
        }
    }
}

impl RoundOracle for TwistWriteCheckAddrOracleSparseTimeMultiLane {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.num_rounds() == 0 {
            let mut mem = K::ZERO;
            for (&val, &w) in self.init_vals.iter().zip(self.init_prefix_w.iter()) {
                mem += val * w;
            }

            let mut sum = K::ZERO;
            let mut i = 0usize;
            while i < self.eq_cycle.len() {
                let t = self.time_idxs[i];
                let start = i;
                while i < self.eq_cycle.len() && self.time_idxs[i] == t {
                    i += 1;
                }
                let end = i;

                // Evaluate write-check terms at time t using pre-state mem.
                for k in start..end {
                    let delta = self.wv[k] - mem - self.inc_at_write_addr[k];
                    sum += self.eq_cycle[k] * self.has_write[k] * self.wa_prefix_w[k] * delta;
                }

                // Apply all writes at time t after checks.
                for k in start..end {
                    let gate_w = self.has_write[k];
                    if gate_w != K::ZERO {
                        mem += self.inc_at_write_addr[k] * gate_w * self.wa_prefix_w[k];
                    }
                }
            }
            return vec![sum; points.len()];
        }

        let bit_idx = self.bit_idx;
        let mut ys = vec![K::ZERO; points.len()];

        self.mem_scratch.clear();
        let mem = &mut self.mem_scratch;
        for ((&addr, &val), &w) in self
            .init_addrs
            .iter()
            .zip(self.init_vals.iter())
            .zip(self.init_prefix_w.iter())
        {
            let idx = addr >> bit_idx;
            let contrib = val * w;
            if contrib != K::ZERO {
                *mem.entry(idx).or_insert(K::ZERO) += contrib;
            }
        }

        let mut i = 0usize;
        while i < self.eq_cycle.len() {
            let t = self.time_idxs[i];
            let start = i;
            while i < self.eq_cycle.len() && self.time_idxs[i] == t {
                i += 1;
            }
            let end = i;

            // Evaluate write-check terms at time t using pre-state mem.
            for k in start..end {
                let eq_t = self.eq_cycle[k];
                let gate = self.has_write[k];
                if gate != K::ZERO {
                    let wa = self.wa_addrs[k];
                    let base = wa >> (bit_idx + 1);
                    let idx0 = base * 2;
                    let idx1 = idx0 + 1;
                    let v0 = mem.get(&idx0).copied().unwrap_or(K::ZERO);
                    let v1 = mem.get(&idx1).copied().unwrap_or(K::ZERO);
                    let dv = v1 - v0;
                    let wv_t = self.wv[k];
                    let inc_t = self.inc_at_write_addr[k];
                    let prefix = self.wa_prefix_w[k];
                    let bit = (wa >> bit_idx) & 1;

                    for (j, &x) in points.iter().enumerate() {
                        let val_x = v0 + dv * x;
                        let addr_factor = if bit == 1 { x } else { K::ONE - x };
                        ys[j] += eq_t * gate * prefix * addr_factor * (wv_t - val_x - inc_t);
                    }
                }
            }

            // Apply all writes at time t after checks.
            for k in start..end {
                let gate_w = self.has_write[k];
                if gate_w != K::ZERO {
                    let wa = self.wa_addrs[k];
                    let idx = wa >> bit_idx;
                    let delta = self.inc_at_write_addr[k] * gate_w * self.wa_prefix_w[k];
                    if delta != K::ZERO {
                        *mem.entry(idx).or_insert(K::ZERO) += delta;
                    }
                }
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        self.ell_addr.saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        update_prefix_weights_in_place(&mut self.init_prefix_w, &self.init_addrs, self.bit_idx, r);
        update_prefix_weights_in_place(&mut self.wa_prefix_w, &self.wa_addrs, self.bit_idx, r);
        self.bit_idx += 1;
    }
}

// ============================================================================
// Address-Domain Lookup Oracle (New Architecture)
// ============================================================================

/// Address-domain lookup oracle for Shout.
///
/// Uses the identity:
///   val̃(r_cycle) = Σ_{a∈{0,1}^{ℓ_A}} Table(a) · Ã(r_cycle, a)
///
/// where Ã(r_cycle, a) is the MLE of the one-hot adapter matrix A(t,a) = 1[a = addr(t)].
///
/// The sum-check is over address variables (ℓ_A rounds), not time variables.
/// At the end, the verifier checks: S_final = Tablẽ(r_addr) · Ã(r_cycle, r_addr)
///
/// ## Advantages over time-domain approach:
/// - No need to commit `table_at_addr[t]` column
/// - Direct verification using table MLE (public) and adapter evaluation
/// - Sum-check domain is address space (often smaller than time)
pub struct AddressLookupOracle {
    /// Product oracle for Table(a) · weight(a) over address space
    core: ProductRoundOracle,
}

impl AddressLookupOracle {
    /// Create a new address-domain lookup oracle from sparse-in-time columns.
    ///
    /// Track A: proof statement/verifier semantics are unchanged; only time iteration is sparse.
    pub fn new(
        addr_bits: &[SparseIdxVec<K>],
        has_lookup: &SparseIdxVec<K>,
        table: &[K],
        r_cycle: &[K],
        ell_addr: usize,
    ) -> (Self, K) {
        let pow2_cycle = 1usize << r_cycle.len();
        let pow2_addr = 1usize << ell_addr;

        assert_eq!(addr_bits.len(), ell_addr, "addr_bits count must match ell_addr");
        for (b, col) in addr_bits.iter().enumerate() {
            assert_eq!(col.len(), pow2_cycle, "addr_bits[{b}] length must match cycle domain");
        }
        assert_eq!(
            has_lookup.len(),
            pow2_cycle,
            "has_lookup length must match cycle domain"
        );

        let mut claimed_sum = K::ZERO;
        let mut weight_table = vec![K::ZERO; pow2_addr];

        for &(t, gate) in has_lookup.entries() {
            if gate == K::ZERO {
                continue;
            }
            let weight_t = chi_at_bool_index(r_cycle, t) * gate;
            if weight_t == K::ZERO {
                continue;
            }

            let mut addr_t = 0usize;
            for (b, col) in addr_bits.iter().enumerate() {
                if col.get(t) == K::ONE {
                    addr_t |= 1usize << b;
                }
            }
            if addr_t < pow2_addr {
                weight_table[addr_t] += weight_t;
            }
        }

        for addr in 0..pow2_addr.min(table.len()) {
            claimed_sum += table[addr] * weight_table[addr];
        }

        let mut table_k: Vec<K> = table.iter().copied().collect();
        table_k.resize(pow2_addr, K::ZERO);
        let core = ProductRoundOracle::new(vec![table_k, weight_table], 2);

        (Self { core }, claimed_sum)
    }

    /// Get the final value after all rounds (should equal Tablẽ(r_addr) · Ã(r_cycle, r_addr))
    pub fn final_value(&self) -> Option<K> {
        self.core.value()
    }

    /// Get the challenges accumulated during sum-check (= r_addr)
    pub fn challenges(&self) -> &[K] {
        self.core.challenges()
    }
}

impl_round_oracle_via_core!(AddressLookupOracle);

/// Compute the MLE of a table at a random point.
///
/// Tablẽ(r) = Σ_{a∈{0,1}^ℓ} Table[a] · eq(r, a)
pub fn table_mle_eval(table: &[K], r_addr: &[K]) -> K {
    let ell = r_addr.len();
    let pow2 = 1usize << ell;

    let mut result = K::ZERO;
    for (idx, &val) in table.iter().enumerate().take(pow2) {
        // eq(r, idx) = χ_r[idx]
        let weight = crate::mle::chi_at_index(r_addr, idx);
        result += val * weight;
    }
    result
}

fn log2_pow2(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    debug_assert!(n.is_power_of_two(), "expected power of two, got {n}");
    n.trailing_zeros() as usize
}
