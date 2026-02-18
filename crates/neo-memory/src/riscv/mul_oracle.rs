//! Sparse-time address-domain lookup oracle for small, fully-enumerable subtables.
//!
//! Used by `Mul8` and `Add8Acc` subtables in the RV32M decomposition.
//! Unlike the implicit RISC-V opcode oracle that relies on closed-form MLE evaluation,
//! this oracle materializes the table and folds it alongside the sparse weight map.

use std::collections::HashMap;

use neo_math::K;
use neo_reductions::error::PiCcsError;
use neo_reductions::sumcheck::RoundOracle;
use p3_field::PrimeCharacteristicRing;

use crate::mle::chi_at_index;
use crate::sparse_time::SparseIdxVec;

/// Address-domain lookup oracle for small, dense subtables with sparse-time weights.
///
/// The proven identity is:
/// ```text
/// Σ_a Table(a) · weight(a) = claimed_sum
/// ```
///
/// The table is folded in-place alongside the weight map on each sumcheck round.
/// Total MLE cost is O(2^{d+1}) across all rounds (geometric series).
pub struct SmallTableAddressLookupOracleSparse {
    #[allow(dead_code)]
    ell_total: usize,
    rounds_remaining: usize,
    /// Dense table, folded each round: starts at 2^d entries, halved per fold.
    table: Vec<K>,
    /// Sparse weight map on remaining Boolean hypercube.
    weights: HashMap<u64, K>,
}

impl SmallTableAddressLookupOracleSparse {
    /// Construct the oracle from sparse-time columns and a pre-built dense table.
    ///
    /// `table_u64` must have exactly `2^ell_addr` entries.
    pub fn new_sparse_time(
        ell_addr: usize,
        table_u64: &[u64],
        addr_bits: &[SparseIdxVec<K>],
        has_lookup: &SparseIdxVec<K>,
        r_cycle: &[K],
    ) -> Result<(Self, K), PiCcsError> {
        if ell_addr > 20 {
            return Err(PiCcsError::InvalidInput(format!(
                "SmallTableOracle: ell_addr={ell_addr} too large (max 20 for dense table)"
            )));
        }
        let expected_size = 1usize << ell_addr;
        if table_u64.len() != expected_size {
            return Err(PiCcsError::InvalidInput(format!(
                "SmallTableOracle: table.len()={} != 2^ell_addr={}",
                table_u64.len(),
                expected_size
            )));
        }
        if addr_bits.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "SmallTableOracle: addr_bits.len()={} != ell_addr={}",
                addr_bits.len(),
                ell_addr
            )));
        }

        let pow2_cycle = 1usize
            .checked_shl(r_cycle.len() as u32)
            .ok_or_else(|| PiCcsError::InvalidInput("SmallTableOracle: 2^ell_cycle overflow".into()))?;
        if has_lookup.len() != pow2_cycle {
            return Err(PiCcsError::InvalidInput(format!(
                "SmallTableOracle: has_lookup.len()={} != 2^ell_cycle={pow2_cycle}",
                has_lookup.len()
            )));
        }
        for (i, col) in addr_bits.iter().enumerate() {
            if col.len() != pow2_cycle {
                return Err(PiCcsError::InvalidInput(format!(
                    "SmallTableOracle: addr_bits[{i}].len()={} != 2^ell_cycle={pow2_cycle}",
                    col.len()
                )));
            }
        }

        let table: Vec<K> = table_u64.iter().map(|&v| K::from_u64(v)).collect();

        let mut weights: HashMap<u64, K> = HashMap::new();
        for &(t, gate) in has_lookup.entries() {
            if gate == K::ZERO {
                continue;
            }

            let mut addr_t: u64 = 0;
            for b in 0..ell_addr {
                if addr_bits[b].get(t) == K::ONE {
                    addr_t |= 1u64 << b;
                }
            }

            let w_t = chi_at_index(r_cycle, t) * gate;
            if w_t != K::ZERO {
                *weights.entry(addr_t).or_insert(K::ZERO) += w_t;
            }
        }

        let mut claimed_sum = K::ZERO;
        for (&addr, &w) in weights.iter() {
            claimed_sum += table[addr as usize] * w;
        }

        Ok((
            Self {
                ell_total: ell_addr,
                rounds_remaining: ell_addr,
                table,
                weights,
            },
            claimed_sum,
        ))
    }

    /// O(1) table lookup into the folded table.
    fn table_endpoint(&self, cur_bit: u64, rest_bits: u64) -> K {
        let idx = (cur_bit & 1) | (rest_bits << 1);
        self.table[idx as usize]
    }
}

impl RoundOracle for SmallTableAddressLookupOracleSparse {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.rounds_remaining == 0 {
            let w = self.weights.get(&0).copied().unwrap_or(K::ZERO);
            let t = self.table[0];
            let v = t * w;
            return vec![v; points.len()];
        }

        let mut pair_weights: HashMap<u64, (K, K)> = HashMap::new();
        for (&idx, &w) in self.weights.iter() {
            let pair = idx >> 1;
            let entry = pair_weights.entry(pair).or_insert((K::ZERO, K::ZERO));
            if (idx & 1) == 0 {
                entry.0 = w;
            } else {
                entry.1 = w;
            }
        }

        let mut c0 = K::ZERO;
        let mut c1 = K::ZERO;
        let mut c2 = K::ZERO;

        for (pair, (w0, w1)) in pair_weights.into_iter() {
            if w0 == K::ZERO && w1 == K::ZERO {
                continue;
            }
            let t0 = self.table_endpoint(0, pair);
            let t1 = self.table_endpoint(1, pair);
            let dw = w1 - w0;
            let dt = t1 - t0;

            c0 += t0 * w0;
            c1 += t0 * dw + w0 * dt;
            c2 += dt * dw;
        }

        points
            .iter()
            .map(|&x| {
                let x2 = x * x;
                c0 + c1 * x + c2 * x2
            })
            .collect()
    }

    fn num_rounds(&self) -> usize {
        self.rounds_remaining
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn fold(&mut self, r: K) {
        if self.rounds_remaining == 0 {
            return;
        }

        // Fold weights (sparse).
        let one_minus_r = K::ONE - r;
        let mut next_weights: HashMap<u64, K> = HashMap::with_capacity(self.weights.len());
        for (&idx, &w) in self.weights.iter() {
            let pair = idx >> 1;
            let contrib = if (idx & 1) == 0 { w * one_minus_r } else { w * r };
            if contrib == K::ZERO {
                continue;
            }
            *next_weights.entry(pair).or_insert(K::ZERO) += contrib;
        }
        self.weights = next_weights;

        // Fold table (dense): table'[i] = (1-r) * table[2*i] + r * table[2*i+1].
        let half = self.table.len() / 2;
        let mut next_table = Vec::with_capacity(half);
        for i in 0..half {
            next_table.push(self.table[2 * i] * one_minus_r + self.table[2 * i + 1] * r);
        }
        self.table = next_table;

        self.rounds_remaining -= 1;
    }
}
