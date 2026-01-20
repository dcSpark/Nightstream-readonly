#![allow(non_snake_case)]

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, Mat, MeInstance};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

#[test]
fn rlc_public_mixes_y_zcol_when_present() {
    let params = NeoParams::goldilocks_127();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let d_pad = 1usize << ell_d;

    // Minimal CCS structure: only `t` matters for Π_RLC public mixing.
    let s = CcsStructure::new(vec![Mat::identity(1)], neo_ccs::poly::SparsePoly::new(1, vec![])).unwrap();

    let m_in = 1usize;
    let r = vec![K::from(F::from_u64(3)), K::from(F::from_u64(5))];
    let s_col = vec![K::from(F::from_u64(7))];

    let mut X0 = Mat::zero(D, m_in, F::ZERO);
    let mut X1 = Mat::zero(D, m_in, F::ZERO);
    X0[(0, 0)] = F::from_u64(11);
    X1[(0, 0)] = F::from_u64(13);

    let mut y0 = vec![vec![K::ZERO; d_pad]];
    let mut y1 = vec![vec![K::ZERO; d_pad]];
    y0[0][0] = K::from(F::from_u64(17));
    y1[0][0] = K::from(F::from_u64(19));

    let y_scalars0 = vec![K::ZERO];
    let y_scalars1 = vec![K::ZERO];

    let mut y_zcol0 = vec![K::ZERO; d_pad];
    let mut y_zcol1 = vec![K::ZERO; d_pad];
    y_zcol0[0] = K::from(F::from_u64(23));
    y_zcol1[0] = K::from(F::from_u64(29));

    let inst0 = MeInstance::<Commitment, F, K> {
        c: Commitment::zeros(params.d as usize, 1),
        X: X0,
        r: r.clone(),
        s_col: s_col.clone(),
        y: y0,
        y_scalars: y_scalars0,
        y_zcol: y_zcol0.clone(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };
    let inst1 = MeInstance::<Commitment, F, K> {
        c: Commitment::zeros(params.d as usize, 1),
        X: X1,
        r,
        s_col,
        y: y1,
        y_scalars: y_scalars1,
        y_zcol: y_zcol1.clone(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };

    // ρ_0 = I, ρ_1 = 2·I.
    let rho0 = Mat::identity(D);
    let mut rho1 = Mat::identity(D);
    for i in 0..D {
        rho1.set(i, i, F::from_u64(2));
    }
    let rhos = vec![rho0, rho1];

    let out = neo_reductions::api::rlc_public(
        &s,
        &params,
        &rhos,
        &[inst0, inst1],
        |_rhos, _cs| Commitment::zeros(params.d as usize, 1),
        ell_d,
    )
    .expect("rlc_public");

    // Expect y_zcol_out = y_zcol0 + 2·y_zcol1 (first D entries; rest padding stays 0).
    let mut expected = vec![K::ZERO; d_pad];
    for rho in 0..D {
        expected[rho] = y_zcol0[rho] + K::from(F::from_u64(2)) * y_zcol1[rho];
    }
    assert_eq!(out.y_zcol, expected);
    assert_eq!(out.s_col.len(), 1);
}
