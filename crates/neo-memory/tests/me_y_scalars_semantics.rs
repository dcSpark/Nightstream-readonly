#![allow(non_snake_case)]

use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::{CcsStructure, Mat};
use neo_math::{D, F, K};
use neo_memory::ajtai::encode_vector_for_ccs_m;
use neo_memory::mle::compute_me_y_for_ccs;
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

fn dense_mat(rows: usize, cols: usize, seed: u64) -> Mat<F> {
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let v = if (r + 3 * c + 1) % 5 == 0 {
                F::from_u64(seed + (r as u64) * 17 + (c as u64) * 23 + 1)
            } else {
                F::ZERO
            };
            data.push(v);
        }
    }
    Mat::from_row_major(rows, cols, data)
}

fn build_Z(params: &NeoParams, seed: u64, m: usize) -> Mat<F> {
    let z: Vec<F> = (0..m)
        .map(|c| F::from_u64(seed + (c as u64) * 11 + 1))
        .collect();
    encode_vector_for_ccs_m(params, m, &z).expect("encode witness")
}

#[test]
fn compute_me_y_for_ccs_superneo_shape_uses_constant_term() {
    let params = NeoParams::goldilocks_127();
    let n = 8usize;
    let m = D; // SuperNeo-compatible width
    let s = CcsStructure::new(
        vec![dense_mat(n, m, 101), dense_mat(n, m, 202)],
        SparsePoly::new(
            2,
            vec![Term {
                coeff: F::ONE,
                exps: vec![1, 0],
            }],
        ),
    )
    .expect("valid CCS");

    let Z = build_Z(&params, 303, m);
    let r = vec![
        K::from(F::from_u64(3)),
        K::from(F::from_u64(5)),
        K::from(F::from_u64(7)),
    ]; // n_pad=8

    let (y_vecs, y_scalars) = compute_me_y_for_ccs(&params, &s, &Z, &r);
    assert_eq!(y_vecs.len(), y_scalars.len());
    for j in 0..y_vecs.len() {
        assert_eq!(
            y_scalars[j], y_vecs[j][0],
            "expected constant-term scalar semantics for superneo-compatible shape at j={j}"
        );
    }
}

#[test]
fn compute_me_y_for_ccs_nondiv_width_still_uses_constant_term() {
    let params = NeoParams::goldilocks_127();
    let n = 8usize;
    let m = 8usize; // non-divisible width uses packed ceil(m/D) embedding.
    let s = CcsStructure::new(
        vec![dense_mat(n, m, 707), dense_mat(n, m, 909)],
        SparsePoly::new(
            2,
            vec![Term {
                coeff: F::ONE,
                exps: vec![1, 0],
            }],
        ),
    )
    .expect("valid CCS");

    let Z = build_Z(&params, 1201, m);
    let r = vec![
        K::from(F::from_u64(2)),
        K::from(F::from_u64(13)),
        K::from(F::from_u64(17)),
    ]; // n_pad=8

    let (y_vecs, y_scalars) = compute_me_y_for_ccs(&params, &s, &Z, &r);
    assert_eq!(y_vecs.len(), y_scalars.len());
    for j in 0..y_vecs.len() {
        assert_eq!(
            y_scalars[j], y_vecs[j][0],
            "expected constant-term scalar semantics at j={j}"
        );
    }
}
