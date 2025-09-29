use p3_goldilocks::Goldilocks as F;
use p3_field::PrimeCharacteristicRing;
use neo_spartan_bridge::ajtai_prg;

#[test]
fn prg_rows_parity_and_ip() {
    let seed = [42u8; 32];
    let row_len = 64usize;
    let rows = 7usize;

    // Construct small signed z_digits in [-2,2]
    let mut z = vec![0i64; row_len];
    for (i, zi) in z.iter_mut().enumerate() {
        let v = ((i * 31 + 7) % 5) as i64 - 2;
        *zi = v;
    }

    for i in 0..rows {
        let r1 = ajtai_prg::expand_row_from_seed(seed, i as u32, row_len);
        let r2 = ajtai_prg::expand_row_from_seed(seed, i as u32, row_len);
        assert_eq!(r1, r2, "PRG must be deterministic for row {}", i);
        // Simple inner product sanity (non-zero typical)
        let mut ip = F::ZERO;
        for (a, &zi) in r1.iter().zip(z.iter()) {
            let zf = if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
            ip += *a * zf;
        }
        // not a strict requirement, but sanity check non-trivial rows
        assert_ne!(ip, F::ZERO, "inner product unexpectedly zero (sanity)");
    }
}
