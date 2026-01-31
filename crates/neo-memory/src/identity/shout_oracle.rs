use std::collections::HashMap;

use neo_math::K;
use neo_reductions::error::PiCcsError;
use neo_reductions::sumcheck::RoundOracle;
use p3_field::PrimeCharacteristicRing;

use crate::sparse_time::SparseIdxVec;

/// Address-domain lookup oracle for Shout with an implicit identity table: `table[addr] = addr`.
///
/// This is a SPARSE variant of the generic address-domain lookup oracle that avoids
/// materializing the full `2^ell_addr` weight table (and thus supports large address domains).
///
/// The proven identity is:
/// ```text
/// Σ_a Table(a) · weight(a) = claimed_sum
/// ```
/// where `Table(a) = a` and `weight` is derived from the committed `(addr_bits, has_lookup)`
/// columns and `r_cycle`.
pub struct IdentityAddressLookupOracleSparse {
    rounds_remaining: usize,

    // prefix_value = Σ_{i < bound_len} 2^i · r_i
    prefix_value: K,
    // coeff_cur = 2^{bound_len}
    coeff_cur: K,

    /// Sparse truth table of `weight` on the remaining Boolean hypercube.
    /// Keys are indices in little-endian order over the remaining variables.
    weights: HashMap<u64, K>,
}

impl IdentityAddressLookupOracleSparse {
    /// Construct the oracle and its claimed sum from sparse-in-time decoded columns.
    ///
    /// This avoids building dense `2^ell_cycle` vectors and iterates only over `has_lookup`'s nonzero entries.
    pub fn new_sparse_time(
        ell_total: usize,
        addr_bits: &[SparseIdxVec<K>],
        has_lookup: &SparseIdxVec<K>,
        r_cycle: &[K],
    ) -> Result<(Self, K), PiCcsError> {
        if ell_total == 0 || ell_total > 64 {
            return Err(PiCcsError::InvalidInput(format!(
                "Identity implicit Shout: ell_total must be in 1..=64, got {ell_total}"
            )));
        }
        if addr_bits.len() != ell_total {
            return Err(PiCcsError::InvalidInput(format!(
                "Identity implicit Shout: addr_bits.len()={} != ell_total={ell_total}",
                addr_bits.len()
            )));
        }

        let pow2_cycle = 1usize
            .checked_shl(r_cycle.len() as u32)
            .ok_or_else(|| PiCcsError::InvalidInput("Identity implicit Shout: 2^ell_cycle overflow".into()))?;
        if has_lookup.len() != pow2_cycle {
            return Err(PiCcsError::InvalidInput(format!(
                "Identity implicit Shout: has_lookup.len()={} != 2^ell_cycle={pow2_cycle}",
                has_lookup.len()
            )));
        }
        for (i, col) in addr_bits.iter().enumerate() {
            if col.len() != pow2_cycle {
                return Err(PiCcsError::InvalidInput(format!(
                    "Identity implicit Shout: addr_bits[{i}].len()={} != 2^ell_cycle={pow2_cycle}",
                    col.len()
                )));
            }
        }

        // Build sparse weight table: weight[a] = Σ_t χ(r_cycle,t) * has_lookup(t) for steps hitting address a.
        let mut weights: HashMap<u64, K> = HashMap::new();
        for &(t, gate) in has_lookup.entries() {
            if gate == K::ZERO {
                continue;
            }

            let mut addr_t: u64 = 0;
            for b in 0..ell_total {
                if addr_bits[b].get(t) == K::ONE {
                    addr_t |= 1u64 << b;
                }
            }

            let w_t = crate::mle::chi_at_index(r_cycle, t) * gate;
            if w_t != K::ZERO {
                *weights.entry(addr_t).or_insert(K::ZERO) += w_t;
            }
        }

        // Claimed sum = Σ_a Table(a) * weight(a) = Σ_a a * weight(a).
        let mut claimed_sum = K::ZERO;
        for (&addr, &w) in weights.iter() {
            claimed_sum += K::from_u64(addr) * w;
        }

        Ok((
            Self {
                rounds_remaining: ell_total,
                prefix_value: K::ZERO,
                coeff_cur: K::ONE,
                weights,
            },
            claimed_sum,
        ))
    }

    #[inline]
    fn rest_contrib(&self, rest_bits: u64) -> K {
        // With little-endian ordering, the remaining bits represent an integer `rest_bits`.
        // Each unit in `rest_bits` contributes a factor of 2^{bound_len+1} = 2 * coeff_cur.
        (self.coeff_cur + self.coeff_cur) * K::from_u64(rest_bits)
    }
}

impl RoundOracle for IdentityAddressLookupOracleSparse {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.rounds_remaining == 0 {
            let w = self.weights.get(&0).copied().unwrap_or(K::ZERO);
            let v = self.prefix_value * w;
            return vec![v; points.len()];
        }

        // Group weights by (rest_bits = idx >> 1) into (w0, w1).
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

        // Accumulate quadratic coefficients over all active pairs.
        let mut c0 = K::ZERO;
        let mut c1 = K::ZERO;
        let mut c2 = K::ZERO;

        // For the identity table, dt = t1 - t0 is constant: 2^{bound_len}.
        let dt = self.coeff_cur;

        for (pair, (w0, w1)) in pair_weights.into_iter() {
            if w0 == K::ZERO && w1 == K::ZERO {
                continue;
            }

            let t0 = self.prefix_value + self.rest_contrib(pair);
            let dw = w1 - w0;

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

        let one_minus_r = K::ONE - r;
        let mut next: HashMap<u64, K> = HashMap::with_capacity(self.weights.len());
        for (&idx, &w) in self.weights.iter() {
            let pair = idx >> 1;
            let contrib = if (idx & 1) == 0 { w * one_minus_r } else { w * r };
            if contrib == K::ZERO {
                continue;
            }
            *next.entry(pair).or_insert(K::ZERO) += contrib;
        }

        self.weights = next;

        // Bind this address bit into the identity Table~.
        self.prefix_value += self.coeff_cur * r;
        self.coeff_cur = self.coeff_cur + self.coeff_cur;

        self.rounds_remaining -= 1;
    }
}
