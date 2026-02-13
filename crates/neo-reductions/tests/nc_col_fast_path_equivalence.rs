#![allow(non_snake_case)]

use neo_ccs::{CcsStructure, Mat, McsWitness, SparsePoly};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::utils::build_dims_and_policy;
use neo_reductions::optimized_engine::oracle::NcOracle;
use neo_reductions::optimized_engine::Challenges;
use neo_reductions::sumcheck::RoundOracle;
use p3_field::PrimeCharacteristicRing;

fn identity_left(n: usize, m: usize) -> Mat<F> {
    let mut mat = Mat::zero(n, m, F::ZERO);
    for i in 0..n.min(m) {
        mat.set(i, i, F::ONE);
    }
    mat
}

fn run_fast_vs_generic(b: u32) {
    let n = 4usize;
    let m = 8usize;

    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.b = b;

    let s = CcsStructure::new(vec![identity_left(n, m)], SparsePoly::new(1, vec![])).expect("ccs");
    let dims = build_dims_and_policy(&params, &s).expect("dims");

    let mut data = Vec::with_capacity(D * m);
    for rho in 0..D {
        for c in 0..m {
            data.push(F::from_u64(7 + (rho as u64) * 19 + (c as u64) * 23));
        }
    }
    let Z = Mat::from_row_major(D, m, data);
    let mcs_witnesses = vec![McsWitness { w: vec![F::ZERO; m], Z }];

    let ch = Challenges {
        alpha: (0..dims.ell_d)
            .map(|i| K::from(F::from_u64(100 + i as u64)))
            .collect(),
        beta_a: (0..dims.ell_d)
            .map(|i| K::from(F::from_u64(200 + i as u64)))
            .collect(),
        beta_r: (0..dims.ell_n)
            .map(|i| K::from(F::from_u64(300 + i as u64)))
            .collect(),
        beta_m: (0..dims.ell_m)
            .map(|i| K::from(F::from_u64(400 + i as u64)))
            .collect(),
        gamma: K::from(F::from_u64(777)),
    };

    let mut oracle = NcOracle::new(&s, &params, &mcs_witnesses, &[], ch, dims.ell_d, dims.ell_m, dims.d_sc);
    let xs = vec![
        K::from(F::ZERO),
        K::from(F::ONE),
        K::from(F::from_u64(2)),
        K::from(F::from_u64(5)),
        K::from(F::from_u64(9)),
    ];

    for round in 0..dims.ell_m {
        let (fast, generic) = oracle
            .__test_col_phase_fast_vs_generic(&xs)
            .expect("must be in NC column phase");
        assert_eq!(
            fast, generic,
            "NcOracle fast col-phase mismatch at b={b}, round={round}"
        );
        oracle.fold(K::from(F::from_u64(900 + round as u64)));
    }
}

#[test]
fn nc_col_phase_fast_path_matches_generic_b2() {
    run_fast_vs_generic(2);
}

#[test]
fn nc_col_phase_fast_path_matches_generic_b3() {
    run_fast_vs_generic(3);
}
