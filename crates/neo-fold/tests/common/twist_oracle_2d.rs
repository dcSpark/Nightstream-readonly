//! Dense (time×addr) Twist helpers/oracles.
//!
//! These are used only by tests (e.g. `ts_route_a_negative.rs`) to compare against
//! or regress legacy “materialize time×addr” behavior. They are intentionally kept
//! out of `neo-memory` production modules.

use neo_math::K;
use neo_memory::twist_oracle::{build_eq_table, ProductRoundOracle};
use neo_reductions::sumcheck::RoundOracle;
use p3_field::PrimeCharacteristicRing;

/// Lift a time-domain factor to the flattened time×addr domain with time bits first.
pub fn lift_time_factor_to_time_addr(time: &[K], pow2_addr: usize) -> Vec<K> {
    let mut out = Vec::with_capacity(time.len() * pow2_addr);
    for _ in 0..pow2_addr {
        out.extend_from_slice(time);
    }
    out
}

/// Lift a time-domain factor to time×addr domain while selecting address bit `bit_idx`.
///
/// Layout: `flat_index = t + pow2_time * addr`. The lifted factor equals:
/// - `time_bitcol[t]` when `addr_bit=1`
/// - `(1 - time_bitcol[t])` when `addr_bit=0`
pub fn lift_time_factor_with_addr_bit(time_bitcol: &[K], bit_idx: usize, pow2_addr: usize) -> Vec<K> {
    let pow2_time = time_bitcol.len();
    let mut out = Vec::with_capacity(pow2_time * pow2_addr);
    for addr in 0..pow2_addr {
        let bit = (addr >> bit_idx) & 1;
        if bit == 0 {
            for t in 0..pow2_time {
                out.push(K::ONE - time_bitcol[t]);
            }
        } else {
            out.extend_from_slice(time_bitcol);
        }
    }
    out
}

/// Build Val(k, t) table (pre-write) over flattened time×addr domain, time bits first.
/// Build Val(k, t) table (pre-write) over flattened time×addr domain, time bits first,
/// using sparse increments as the source of truth.
pub fn build_val_table_pre_write_from_inc(
    init: &[K],
    has_write: &[K],
    wa_bits: &[Vec<K>],
    inc_at_write_addr: &[K],
    pow2_time: usize,
    pow2_addr: usize,
) -> Vec<K> {
    assert_eq!(has_write.len(), pow2_time, "has_write length mismatch");
    assert_eq!(inc_at_write_addr.len(), pow2_time, "inc_at_write_addr length mismatch");
    for col in wa_bits {
        assert_eq!(col.len(), pow2_time, "wa_bits column length mismatch");
    }

    let mut mem = vec![K::ZERO; pow2_addr];
    if !init.is_empty() {
        let init_len = init.len().min(pow2_addr);
        mem[..init_len].copy_from_slice(&init[..init_len]);
    }

    let mut out = vec![K::ZERO; pow2_time * pow2_addr];
    for t in 0..pow2_time {
        for addr in 0..pow2_addr {
            out[addr * pow2_time + t] = mem[addr];
        }

        if has_write[t] == K::ONE {
            let mut addr = 0usize;
            for (b, col) in wa_bits.iter().enumerate() {
                if col[t] == K::ONE {
                    addr |= 1usize << b;
                }
            }
            if addr < pow2_addr {
                mem[addr] += inc_at_write_addr[t];
            }
        }
    }

    out
}

/// Read-check over time×addr with time bits first.
pub struct TwistReadCheck2DOracle {
    core: ProductRoundOracle,
}

impl TwistReadCheck2DOracle {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        r_cycle: &[K],
        has_read: &[K],
        ra_bits: &[Vec<K>],
        rv: &[K],
        val_table: &[K],
        pow2_time: usize,
        pow2_addr: usize,
    ) -> Self {
        assert_eq!(has_read.len(), pow2_time, "has_read length mismatch");
        assert_eq!(rv.len(), pow2_time, "rv length mismatch");
        assert_eq!(ra_bits.len(), (pow2_addr.trailing_zeros() as usize));
        assert_eq!(val_table.len(), pow2_time * pow2_addr, "val_table length mismatch");

        let eq_cycle = build_eq_table(r_cycle);
        let eq_cycle_l = lift_time_factor_to_time_addr(&eq_cycle, pow2_addr);
        let has_read_l = lift_time_factor_to_time_addr(has_read, pow2_addr);

        let mut diff = vec![K::ZERO; pow2_time * pow2_addr];
        for addr in 0..pow2_addr {
            for t in 0..pow2_time {
                let idx = addr * pow2_time + t;
                diff[idx] = val_table[idx] - rv[t];
            }
        }

        let mut factors = Vec::with_capacity(3 + ra_bits.len());
        factors.push(eq_cycle_l);
        factors.push(has_read_l);
        factors.push(diff);
        for (b, col) in ra_bits.iter().enumerate() {
            factors.push(lift_time_factor_with_addr_bit(col, b, pow2_addr));
        }

        let degree_bound = factors.len();
        let core = ProductRoundOracle::new(factors, degree_bound);
        Self { core }
    }
}

impl RoundOracle for TwistReadCheck2DOracle {
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

/// Write-check over time×addr with time bits first.
pub struct TwistWriteCheck2DOracle {
    core: ProductRoundOracle,
}

impl TwistWriteCheck2DOracle {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        r_cycle: &[K],
        has_write: &[K],
        wa_bits: &[Vec<K>],
        wv: &[K],
        inc_at_write_addr: &[K],
        val_table: &[K],
        pow2_time: usize,
        pow2_addr: usize,
    ) -> Self {
        assert_eq!(has_write.len(), pow2_time, "has_write length mismatch");
        assert_eq!(wv.len(), pow2_time, "wv length mismatch");
        assert_eq!(inc_at_write_addr.len(), pow2_time, "inc length mismatch");
        assert_eq!(wa_bits.len(), (pow2_addr.trailing_zeros() as usize));
        assert_eq!(val_table.len(), pow2_time * pow2_addr, "val_table length mismatch");

        let eq_cycle = build_eq_table(r_cycle);
        let eq_cycle_l = lift_time_factor_to_time_addr(&eq_cycle, pow2_addr);
        let has_write_l = lift_time_factor_to_time_addr(has_write, pow2_addr);

        let mut delta = vec![K::ZERO; pow2_time * pow2_addr];
        for addr in 0..pow2_addr {
            for t in 0..pow2_time {
                let idx = addr * pow2_time + t;
                delta[idx] = wv[t] - val_table[idx] - inc_at_write_addr[t];
            }
        }

        let mut factors = Vec::with_capacity(3 + wa_bits.len());
        factors.push(eq_cycle_l);
        factors.push(has_write_l);
        factors.push(delta);
        for (b, col) in wa_bits.iter().enumerate() {
            factors.push(lift_time_factor_with_addr_bit(col, b, pow2_addr));
        }

        let degree_bound = factors.len();
        let core = ProductRoundOracle::new(factors, degree_bound);
        Self { core }
    }
}

impl RoundOracle for TwistWriteCheck2DOracle {
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
