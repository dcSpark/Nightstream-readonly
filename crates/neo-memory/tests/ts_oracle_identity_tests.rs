//! Oracle-level identity tests for Twist/Shout (fast, deterministic).

use neo_math::K;
use neo_memory::mle::lt_eval;
use neo_memory::twist_oracle::{
    build_eq_table, compute_eq_from_bits, IndexAdapterOracle, TwistTotalIncOracleSparse, TwistValEvalOracleSparse,
};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

fn k(u: u64) -> K {
    K::from(Goldilocks::from_u64(u))
}

fn mask_to_bits(mask: usize, ell: usize) -> Vec<K> {
    (0..ell)
        .map(|i| if ((mask >> i) & 1) == 1 { K::ONE } else { K::ZERO })
        .collect()
}

#[test]
fn shout_adapter_oracle_matches_naive_sum() {
    // ell_n=2 => pow2_cycle=4, ell_addr=2 => 4 addresses.
    let pow2_cycle = 4usize;
    let ell_n = 2usize;
    let ell_addr = 2usize;

    let has_lookup = vec![K::ONE, K::ZERO, K::ONE, K::ZERO];

    // Addresses: [2, x, 1, x] where x is irrelevant when has_lookup=0.
    // addr_bits are little-endian (bit 0 first).
    let addr_bits: Vec<Vec<K>> = vec![
        vec![K::ZERO, K::ZERO, K::ONE, K::ZERO], // bit 0
        vec![K::ONE, K::ZERO, K::ZERO, K::ZERO], // bit 1
    ];

    let r_cycle = vec![k(5), k(7)];
    let r_addr = vec![k(11), k(13)];

    let oracle = IndexAdapterOracle::new_with_gate(&addr_bits, &has_lookup, &r_cycle, &r_addr);
    let claimed = oracle.compute_claim();

    let chi_cycle = build_eq_table(&r_cycle);
    assert_eq!(chi_cycle.len(), pow2_cycle);
    let eq_addr = compute_eq_from_bits(&addr_bits, &r_addr);
    assert_eq!(eq_addr.len(), pow2_cycle);

    let mut naive = K::ZERO;
    for t in 0..pow2_cycle {
        naive += chi_cycle[t] * has_lookup[t] * eq_addr[t];
    }

    assert_eq!(claimed, naive);
    assert_eq!(ell_n, r_cycle.len());
    assert_eq!(ell_addr, r_addr.len());
}

#[test]
fn twist_val_eval_sparse_oracle_matches_naive_sum() {
    // ell_n=2 => pow2_cycle=4, ell_addr=2 => 4 addresses.
    let pow2_cycle = 4usize;
    let ell_n = 2usize;
    let ell_addr = 2usize;

    let has_write = vec![K::ONE, K::ZERO, K::ONE, K::ONE];
    let inc_at_write_addr = vec![k(5), K::ZERO, k(11), k(13)];

    // Write addresses (little-endian bits): [2, x, 1, 3] where x is irrelevant when has_write=0.
    let wa_bits: Vec<Vec<K>> = vec![
        vec![K::ZERO, K::ZERO, K::ONE, K::ONE], // bit 0
        vec![K::ONE, K::ZERO, K::ZERO, K::ONE], // bit 1
    ];

    let r_addr = vec![k(3), k(9)];
    let r_time = vec![k(2), k(4)];

    let (_oracle, claimed) =
        TwistValEvalOracleSparse::new(&wa_bits, has_write.clone(), inc_at_write_addr.clone(), &r_addr, &r_time);

    let eq_wa = compute_eq_from_bits(&wa_bits, &r_addr);
    assert_eq!(eq_wa.len(), pow2_cycle);

    let mut naive = K::ZERO;
    for t in 0..pow2_cycle {
        let t_bits = mask_to_bits(t, ell_n);
        let lt = lt_eval(&t_bits, &r_time);
        naive += has_write[t] * inc_at_write_addr[t] * eq_wa[t] * lt;
    }

    assert_eq!(claimed, naive);
    assert_eq!(ell_addr, r_addr.len());
}

#[test]
fn twist_total_inc_sparse_oracle_matches_naive_sum() {
    let pow2_cycle = 4usize;

    let has_write = vec![K::ONE, K::ZERO, K::ONE, K::ONE];
    let inc_at_write_addr = vec![k(5), K::ZERO, k(11), k(13)];

    let wa_bits: Vec<Vec<K>> = vec![
        vec![K::ZERO, K::ZERO, K::ONE, K::ONE], // bit 0
        vec![K::ONE, K::ZERO, K::ZERO, K::ONE], // bit 1
    ];

    let r_addr = vec![k(3), k(9)];

    let (_oracle, claimed) = TwistTotalIncOracleSparse::new(&wa_bits, has_write.clone(), inc_at_write_addr.clone(), &r_addr);

    let eq_wa = compute_eq_from_bits(&wa_bits, &r_addr);
    assert_eq!(eq_wa.len(), pow2_cycle);

    let mut naive = K::ZERO;
    for t in 0..pow2_cycle {
        naive += has_write[t] * inc_at_write_addr[t] * eq_wa[t];
    }

    assert_eq!(claimed, naive);
}

