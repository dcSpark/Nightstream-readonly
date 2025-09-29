//! Verify-time public IO tamper: mutate `Proof.public_io_bytes` and expect verify to fail.
#![allow(deprecated)]

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;

#[test]
fn public_io_tamper_rejected() {
    use neo_ccs::{MEInstance, MEWitness};

    // Enable PRG-derived Ajtai rows inside the circuit to avoid requiring PP
    std::env::set_var("NEO_ENABLE_PRG_ROWS", "1");

    // Small PRG-mode instance
    let seed = [13u8; 32];
    let base_b = 3u64;
    let z_len = 8usize; // power of two

    let mut z_digits = vec![0i64; z_len];
    for (i, zi) in z_digits.iter_mut().enumerate() {
        *zi = match i % 5 { 0 => -2, 1 => -1, 2 => 0, 3 => 1, _ => 2 };
    }

    // Derive c_coords from PRG rows to bind ⟨row_i, z⟩ = c_i
    let rows = 4usize;
    let mut c_coords: Vec<neo_math::F> = Vec::with_capacity(rows);
    for i in 0..rows {
        let row = neo_spartan_bridge::ajtai_prg::expand_row_from_seed(seed, i as u32, z_len);
        let mut acc = F::ZERO;
        for (a, &zv) in row.iter().zip(z_digits.iter()) {
            let zf = if zv >= 0 { F::from_u64(zv as u64) } else { -F::from_u64((-zv) as u64) };
            acc += *a * zf;
        }
        c_coords.push(neo_math::F::from_u64(acc.as_canonical_u64()));
    }

    let me = MEInstance {
        c_coords: c_coords.clone(),
        y_outputs: vec![],
        r_point: vec![],
        base_b,
        header_digest: seed,
        c_step_coords: c_coords.clone(),
        u_offset: 0,
        u_len: 0,
    };
    let wit = MEWitness { z_digits, weight_vectors: vec![], ajtai_rows: None };

    // Prove (lean proof with bound public IO)
    let proof = neo_spartan_bridge::compress_me_to_lean_proof_with_pp(&me, &wit, None)
        .expect("prove ok");
    assert!(neo_spartan_bridge::verify_lean_proof(&proof).expect("verify runs"));

    // Replay attack: mutate public_io bytes without touching the proof bytes
    let mut tampered = bincode::deserialize::<neo_spartan_bridge::Proof>(&bincode::serialize(&proof).unwrap()).unwrap();
    if !tampered.public_io_bytes.is_empty() {
        tampered.public_io_bytes[0] ^= 1; // flip one bit
    } else {
        // Should not happen in practice; keep test robust
        tampered.public_io_bytes = vec![1u8];
    }

    // Verifier must reject because it re-encodes public values from Spartan
    let ok = neo_spartan_bridge::verify_lean_proof(&tampered).expect("verify runs");
    assert!(!ok, "verifier must reject when public_io_bytes are tampered");
}
