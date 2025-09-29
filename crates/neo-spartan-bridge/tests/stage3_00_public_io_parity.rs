//! Stage 3: Public IO parity (no EV) — circuit vs encoder bytes must match exactly

mod common;
use common::test_setup;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use spartan2::traits::circuit::SpartanCircuit;

#[test]
fn public_io_parity_no_ev() {
    test_setup();

    // Build a minimal legacy ME instance
    #[allow(deprecated)]
    let me = neo_ccs::MEInstance {
        c_coords: vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)],
        y_outputs: vec![F::from_u64(10), F::from_u64(11)],
        r_point: vec![F::from_u64(7), F::from_u64(8)],
        base_b: 2,
        header_digest: [42u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };

    #[allow(deprecated)]
    let wit = neo_ccs::MEWitness { z_digits: vec![], weight_vectors: vec![], ajtai_rows: None };

    // Encode via canonical encoder
    let encoded = neo_spartan_bridge::encode_bridge_io_header(&me);

    // Compute via circuit public_values()
    #[allow(deprecated)]
    let circuit = neo_spartan_bridge::me_to_r1cs::MeCircuit::new(me.clone(), wit, None, me.header_digest);
    let scalars = circuit.public_values().expect("public_values");
    // Flatten to bytes (LE u64 per scalar)
    let mut via_circuit = Vec::with_capacity(scalars.len() * 8);
    for s in scalars { via_circuit.extend_from_slice(&s.to_canonical_u64().to_le_bytes()); }

    assert_eq!(via_circuit, encoded, "public IO bytes mismatch (no-EV)");
}

