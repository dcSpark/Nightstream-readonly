#![allow(deprecated)] // Use legacy ProofBundle helpers in tests

use neo_spartan_bridge::{compress_me_to_spartan, verify_me_spartan, encode_bridge_io_header};
use neo_ccs::{MEInstance, MEWitness};
use p3_goldilocks::Goldilocks as F;
use p3_field::PrimeCharacteristicRing;

fn tiny_me_instance() -> (MEInstance, MEWitness) {
    // Witness digits (len = 8 = 2^3)
    let z = vec![1i64, 2, 3, 0, -1, 1, 0, 2];

    // Two Ajtai rows (unit vectors) so constraints become z0=c0 and z1=c1
    let ajtai_rows = vec![
        vec![F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
        vec![F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
    ];
    let dot = |row: &[F]| -> F {
        row.iter().zip(z.iter()).fold(F::ZERO, |acc, (a, &zi)| {
            let zi_f = if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
            acc + (*a) * zi_f
        })
    };
    let c_coords = vec![dot(&ajtai_rows[0]), dot(&ajtai_rows[1])];

    // Two weight vectors -> two y outputs (arbitrary but deterministic)
    let w0 = vec![F::ONE, F::ONE, F::ONE, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO];
    let w1 = vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ONE];
    let dotf = |row: &[F]| -> F { row.iter().zip(z.iter()).fold(F::ZERO, |acc, (a, &zi)| {
        let zi_f = if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
        acc + (*a) * zi_f
    })};
    let y_outputs = vec![dotf(&w0), dotf(&w1)];

    let me = MEInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c_coords,
        y_outputs,
        r_point: vec![F::from_u64(3); 2],
        base_b: 4, // => digits in [-3,3]
        header_digest: [0u8; 32],
    };
    let wit = MEWitness { z_digits: z, weight_vectors: vec![w0, w1], ajtai_rows: Some(ajtai_rows) };
    (me, wit)
}

#[test]
fn public_io_flip_byte_rejected() {
    let (me, wit) = tiny_me_instance();
    let mut bundle = compress_me_to_spartan(&me, &wit).expect("prove");

    // Sanity: header encoding matches canonical
    let canonical = encode_bridge_io_header(&me);
    assert_eq!(canonical, bundle.public_io_bytes, "canonical header mismatch");

    // Flip first byte in bound public IO
    assert!(!bundle.public_io_bytes.is_empty());
    bundle.public_io_bytes[0] ^= 1;

    // Verify must return false when bound IO is tampered
    let ok = verify_me_spartan(&bundle).expect("verify runs");
    assert!(!ok, "verification must reject tampered public_io_bytes");
}

#[test]
fn public_io_matches_snark_encoding() {
    let (me, wit) = tiny_me_instance();
    let bundle = compress_me_to_spartan(&me, &wit).expect("prove");
    let canonical = encode_bridge_io_header(&me);
    assert_eq!(canonical, bundle.public_io_bytes, "SNARK public IO must equal canonical encoder");
}

