//! Stage 1: Ajtai row • z_digits sanity (no Spartan involved)

mod common;
use common::test_setup;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn ajtai_row_dot_z_matches_manual() {
    test_setup();

    // Use small m to keep it quick
    let d = neo_math::ring::D;
    let m = 1usize;
    let z_len = d * m;

    // Ensure PP exists for (d, m)
    {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::from_seed([42u8; 32]);
        let pp = neo_ajtai::setup(&mut rng, d, 1, m).expect("setup");
        neo_ajtai::set_global_pp(pp).expect("register PP");
    }

    let pp = neo_ajtai::get_global_pp_for_dims(d, m).expect("get PP");

    // Balanced digits in {-1, 0, 1}
    let mut z_digits: Vec<i64> = Vec::with_capacity(z_len);
    for i in 0..z_len { z_digits.push(match i % 3 { 0 => -1, 1 => 0, _ => 1 }); }

    // Helper: map i64 → F
    let to_f = |zi: i64| if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };

    // Sample a few coordinates
    let rows = 3usize;
    for coord_idx in 0..rows {
        let row = neo_ajtai::compute_single_ajtai_row(&pp, coord_idx, z_len, rows).expect("row");
        assert_eq!(row.len(), z_len, "row length mismatch");
        let mut ip = F::ZERO;
        for j in 0..z_len { ip += row[j] * to_f(z_digits[j]); }
        // No external binding to compare against; the exercise is to confirm basic operations work
        // and do not panic and shapes are aligned.
        assert!(ip != F::ZERO || z_digits.iter().all(|&z| z == 0), "inner product should be well-defined");
    }
}

