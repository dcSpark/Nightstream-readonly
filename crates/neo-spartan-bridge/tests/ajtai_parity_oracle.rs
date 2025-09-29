//! Ajtai <row, z> parity oracle (host-side)
//!
//! Verifies that for a small PP-backed instance, rows derived from PP match the
//! commitment coordinates via dot(row, z_digits). This catches layout/digitization
//! drift without involving Spartan.

#![allow(deprecated)]

use neo_ccs::{MEInstance, MEWitness};
use p3_goldilocks::Goldilocks as F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Build tiny honest instance (same as in transcript test)
fn tiny_honest_instance() -> (MEInstance, MEWitness, std::sync::Arc<neo_ajtai::PP<neo_math::Rq>>) {
    let d = neo_math::ring::D; let m = 1usize; let kappa = 2usize;
    let mut z_digits: Vec<i64> = Vec::with_capacity(d*m);
    for i in 0..(d*m) { z_digits.push(match i % 3 { 0 => -1, 1 => 0, _ => 1 }); }
    let to_f = |zi: i64| if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
    let z_f: Vec<F> = z_digits.iter().copied().map(to_f).collect();

    use rand::SeedableRng; use rand::rngs::StdRng;
    let mut rng = StdRng::from_seed([7u8; 32]);
    let pp = std::sync::Arc::new(neo_ajtai::setup(&mut rng, d, kappa, m).expect("setup"));
    let c = neo_ajtai::commit(&pp, &z_f);

    let me = MEInstance {
        c_coords: c.data.clone(),
        y_outputs: vec![], r_point: vec![], base_b: 2,
        header_digest: [0u8; 32], c_step_coords: vec![], u_offset: 0, u_len: 0,
    };
    let wit = MEWitness { z_digits, weight_vectors: vec![], ajtai_rows: None };
    (me, wit, pp)
}

/// Dot product helper: signed z_digits · row
fn dot_row_z(row: &[F], z_digits: &[i64]) -> F {
    let mut acc = F::ZERO;
    for (a, &zi) in row.iter().zip(z_digits.iter()) {
        let zf = if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
        acc += *a * zf;
    }
    acc
}

/// Check a few (row, z) pairs against c_coords[i]. z_len_unpadded := d * m (no padding).
fn ajtai_parity_check_pp(
    me: &MEInstance,
    wit: &MEWitness,
    pp: &neo_ajtai::PP<neo_math::Rq>,
    z_len_unpadded: usize,
    sample_rows: &[usize],
) {
    assert!(!me.c_coords.is_empty(), "no c_coords to check");
    for &i in sample_rows {
        assert!(i < me.c_coords.len(), "sample row {} out of bounds", i);
        // Compute the exact Ajtai row for the unpadded z length
        let row = neo_ajtai::compute_single_ajtai_row(pp, i, z_len_unpadded, me.c_coords.len())
            .expect("compute_single_ajtai_row");
        let lhs = dot_row_z(&row, &wit.z_digits[..z_len_unpadded]);
        let rhs = me.c_coords[i];
        assert_eq!(lhs, rhs, "Ajtai parity mismatch at row {}: L·z={} vs c_coords[{}]={}", 
            i, lhs.as_canonical_u64(), i, rhs.as_canonical_u64());
    }
}

#[test]
fn ajtai_parity_pp_rows_match() {
    let d = neo_math::ring::D; let m = 1usize; let z_len_unpadded = d*m;
    let (me, wit, pp) = tiny_honest_instance();
    let n = me.c_coords.len().max(1);
    let mids = [0, n/2, n.saturating_sub(1)];
    ajtai_parity_check_pp(&me, &wit, &pp, z_len_unpadded, &mids);
}
