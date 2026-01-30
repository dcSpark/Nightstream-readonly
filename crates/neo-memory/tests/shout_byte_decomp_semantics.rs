//! Shout semantic checks for explicit tables.
//!
//! This test builds explicit Shout tables plus a corresponding Ajtai-encoded `LutWitness`, then
//! runs `neo_memory::shout::check_shout_semantics` to get precise, step-indexed failures (instead
//! of only seeing a generic sumcheck mismatch later in the pipeline).
//!
//! Tables exercised here:
//! - Byte identity (`k=256`, `n_side=256`, `ell=8`): `table[addr] = addr`.
//!   Useful for byte range-checks / byte decomposition.
//! - Nibble square (`k=16`, `n_side=16`, `ell=4`): `table[addr] = addr^2`.
//!   Useful for 4-bit range-checks / nibble decomposition with a non-trivial mapping.
//!
//! Running:
//! - Run this integration test binary: `cargo test -p neo-memory --test shout_byte_decomp_semantics --release`
//! - Or filter by name (this test is prefixed accordingly): `cargo test -p neo-memory shout_byte_decomp_semantics --release`

use neo_math::F;
use neo_memory::ajtai::encode_vector_balanced_to_mat;
use neo_memory::shout::check_shout_semantics;
use neo_memory::witness::{LutInstance, LutWitness};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

fn build_single_lane_explicit_lut_witness(
    params: &NeoParams,
    n_side: usize,
    ell: usize,
    table: Vec<F>,
    addrs: &[u64],
) -> (LutInstance<(), F>, LutWitness<F>, Vec<F>) {
    assert!(n_side.is_power_of_two(), "n_side must be power-of-two");
    assert_eq!(n_side.trailing_zeros() as usize, ell, "ell must be log2(n_side)");

    let steps = addrs.len();
    for &addr in addrs {
        assert!(
            (addr as usize) < n_side,
            "addr out of range: addr={addr}, n_side={n_side}"
        );
    }

    let inst = LutInstance::<(), F> {
        comms: Vec::new(),
        k: n_side,
        d: 1,
        n_side,
        steps,
        lanes: 1,
        ell,
        table_spec: None,
        table: table.clone(),
    };

    // Layout: [addr_bits(ell), has_lookup, val].
    let mut mats = Vec::with_capacity(ell + 2);

    // addr_bits are little-endian.
    for bit in 0..ell {
        let col: Vec<F> = addrs
            .iter()
            .map(|&addr| if ((addr >> bit) & 1) == 1 { F::ONE } else { F::ZERO })
            .collect();
        mats.push(encode_vector_balanced_to_mat(params, &col));
    }

    let has_lookup: Vec<F> = vec![F::ONE; steps];
    mats.push(encode_vector_balanced_to_mat(params, &has_lookup));

    let expected_vals: Vec<F> = addrs.iter().map(|&addr| table[addr as usize]).collect();
    mats.push(encode_vector_balanced_to_mat(params, &expected_vals));

    (inst, LutWitness { mats }, expected_vals)
}

#[test]
fn shout_byte_decomp_semantics_explicit_tables_byte_identity_and_nibble_square() {
    let params = NeoParams::goldilocks_127();

    // Byte identity table: for any byte `b`, lookup(b) = b.
    // Repeated bytes are allowed: Shout does not require distinct keys.
    let byte_identity: Vec<F> = (0u64..256).map(F::from_u64).collect();
    let bytes: Vec<u8> = vec![0x00, 0xAB, 0xAB, 0x00, 0xFF, 0xAB, 0xFF, 0x00];
    let addrs: Vec<u64> = bytes.iter().map(|&b| b as u64).collect();
    let (inst, wit, expected_vals) =
        build_single_lane_explicit_lut_witness(&params, /*n_side=*/ 256, /*ell=*/ 8, byte_identity, &addrs);
    check_shout_semantics(&params, &inst, &wit, &expected_vals).expect("byte identity semantics");

    // Nibble square table: for any 4-bit `x`, lookup(x) = x^2 (in the base field).
    let nibble_square: Vec<F> = (0u64..16).map(|x| F::from_u64(x * x)).collect();
    let addrs: Vec<u64> = vec![0, 1, 2, 2, 3, 15, 15, 4, 0];
    let (inst, wit, expected_vals) =
        build_single_lane_explicit_lut_witness(&params, /*n_side=*/ 16, /*ell=*/ 4, nibble_square, &addrs);
    check_shout_semantics(&params, &inst, &wit, &expected_vals).expect("nibble square semantics");
}
