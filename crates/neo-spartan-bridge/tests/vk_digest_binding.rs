#![allow(deprecated)]

use neo_spartan_bridge::{compress_me_to_lean_proof, verify_lean_proof};
use neo_ccs::{MEInstance, MEWitness};
use p3_goldilocks::Goldilocks as F;
use p3_field::PrimeCharacteristicRing;

fn tiny_me_instance() -> (MEInstance, MEWitness) {
    // Simple consistent instance (same as in other tests)
    let z = vec![1i64, 0, -1, 2, 0, 1, 0, -2];
    let ajtai_rows = vec![
        vec![F::ONE; 8],
        vec![F::ZERO, F::ONE, F::ZERO, F::ONE, F::ZERO, F::ONE, F::ZERO, F::ONE],
    ];
    let dot = |row: &[F]| -> F {
        row.iter().zip(z.iter()).fold(F::ZERO, |acc, (a, &zi)| {
            let zi_f = if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
            acc + (*a) * zi_f
        })
    };
    let c_coords = vec![dot(&ajtai_rows[0]), dot(&ajtai_rows[1])];

    // Two weight vectors for two y outputs
    let w0 = vec![F::ONE, F::ZERO, F::ONE, F::ZERO, F::ONE, F::ZERO, F::ONE, F::ZERO];
    let w1 = vec![F::ZERO, F::ONE, F::ZERO, F::ONE, F::ZERO, F::ONE, F::ZERO, F::ONE];
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
        r_point: vec![F::from_u64(7); 2],
        base_b: 4,
        header_digest: [0u8; 32],
    };
    let wit = MEWitness { z_digits: z, weight_vectors: vec![w0, w1], ajtai_rows: Some(ajtai_rows) };
    (me, wit)
}

#[test]
fn vk_digest_binding_ok_and_flip_fails() {
    let (me, wit) = tiny_me_instance();
    let mut proof = compress_me_to_lean_proof(&me, &wit).expect("lean proof");

    // Sanity: verifies as-is
    assert!(verify_lean_proof(&proof).expect("verify runs"));

    // Flip a byte in vk_digest → must error on binding check
    proof.vk_digest[0] ^= 0xA5;
    let err = verify_lean_proof(&proof).err().expect("should error");
    let msg = format!("{err}");
    assert!(msg.contains("VK digest mismatch"), "unexpected error: {msg}");
}
