use neo_ajtai::prg::expand_row_v1;

#[test]
fn ajtai_prg_determinism_v1() {
    let seed = [42u8; 32];
    let r0 = expand_row_v1(&seed, 0, 10);
    let r0_b = expand_row_v1(&seed, 0, 10);
    assert_eq!(r0, r0_b, "same seed+row_idx must produce identical row");

    let r1 = expand_row_v1(&seed, 1, 10);
    assert_ne!(r0, r1, "different row_idx must produce different rows");

    let mut seed2 = seed;
    seed2[0] ^= 1;
    let r0_seed2 = expand_row_v1(&seed2, 0, 10);
    assert_ne!(r0, r0_seed2, "different seed must produce different rows");
}

#[test]
fn ajtai_prg_length_v1() {
    let seed = [7u8; 32];
    for len in [1usize, 2, 3, 4, 5, 8, 9, 16, 17] {
        let row = expand_row_v1(&seed, 123, len);
        assert_eq!(row.len(), len);
    }
}

