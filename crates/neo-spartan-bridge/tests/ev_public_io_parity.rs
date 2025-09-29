#![allow(deprecated)]

use neo_spartan_bridge::{encode_bridge_io_header_with_ev};
use neo_spartan_bridge::me_to_r1cs::{IvcEvEmbed, MeCircuit};
use neo_ccs::{MEInstance, MEWitness};
use p3_goldilocks::Goldilocks as F;
use p3_field::PrimeCharacteristicRing;
use spartan2::traits::circuit::SpartanCircuit;

fn tiny_me_instance() -> (MEInstance, MEWitness) {
    let z = vec![1i64, 2, 3, 0, -1, 1, 0, 2];
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
        r_point: vec![F::from_u64(5); 2],
        base_b: 4,
        header_digest: [0u8; 32],
    };
    let wit = MEWitness { z_digits: z, weight_vectors: vec![w0, w1], ajtai_rows: Some(ajtai_rows) };
    (me, wit)
}

#[test]
fn ev_public_io_parity_roundtrip() {
    let (me, wit) = tiny_me_instance();
    let ev = IvcEvEmbed {
        rho: F::from_u64(9),
        y_prev: vec![F::from_u64(3), F::from_u64(4)],
        y_next: vec![F::from_u64(12), F::from_u64(22)],
        y_step_public: None,
        fold_chain_digest: Some([0x55; 32]),
        acc_c_prev: Some(vec![F::from_u64(7), F::from_u64(8)]),
        acc_c_step: Some(vec![F::from_u64(0), F::from_u64(1)]),
        acc_c_next: Some(vec![F::from_u64(7), F::from_u64(17)]),
        rho_eff: Some(F::from_u64(1)),
    };

    // Canonical encoding (host)
    let expected = encode_bridge_io_header_with_ev(&me, Some(&ev));

    // Circuit public values (in-circuit layout)
    let circ = MeCircuit::new(me.clone(), wit.clone(), None, me.header_digest).with_ev(Some(ev));
    let publics = circ.public_values().expect("public_values");
    let mut actual = Vec::with_capacity(publics.len() * 8);
    for x in publics { actual.extend_from_slice(&x.to_canonical_u64().to_le_bytes()); }

    assert_eq!(expected, actual, "EV-extended public IO must match circuit public_values layout");
}

