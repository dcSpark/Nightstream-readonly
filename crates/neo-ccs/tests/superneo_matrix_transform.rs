use neo_ccs::{sparse::CcsMatrix, CcsStructure, Mat, SparsePoly};
use neo_math::{ct, Rq, D, F};
use p3_field::PrimeCharacteristicRing;

fn deterministic_vec(len: usize, seed: u64) -> Vec<F> {
    let mut x = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push(F::from_u64(x));
    }
    out
}

fn field_mul_rows(matrix: &CcsMatrix<F>, x: &[F], nrows: usize) -> Vec<F> {
    let mut out = vec![F::ZERO; nrows];
    matrix.add_mul_into(x, &mut out, nrows);
    out
}

fn ring_ct_row_eval(matrix: &CcsMatrix<F>, row: usize, z_ring: &[Rq], ncols: usize) -> F {
    let n_ring_cols = ncols / D;
    let mut row_blocks = vec![[F::ZERO; D]; n_ring_cols];

    match matrix {
        CcsMatrix::Identity { .. } => {
            panic!("identity sentinel should not appear after superneo transform")
        }
        CcsMatrix::Csc(m) => {
            for c in 0..m.ncols {
                let s = m.col_ptr[c];
                let e = m.col_ptr[c + 1];
                let block = c / D;
                let local = c % D;
                for k in s..e {
                    if m.row_idx[k] == row {
                        row_blocks[block][local] += m.vals[k];
                    }
                }
            }
        }
    }

    let mut acc = Rq::zero();
    for (blk, coeffs) in row_blocks.iter().enumerate() {
        let mr = Rq(*coeffs);
        acc = acc + mr.mul(&z_ring[blk]);
    }
    ct(&acc)
}

fn to_ring_blocks(z: &[F]) -> Vec<Rq> {
    assert!(z.len().is_multiple_of(D));
    let mut out = Vec::with_capacity(z.len() / D);
    for chunk in z.chunks_exact(D) {
        let mut coeffs = [F::ZERO; D];
        coeffs.copy_from_slice(chunk);
        out.push(Rq(coeffs));
    }
    out
}

#[test]
fn superneo_transform_identity_matrix_recovers_z_via_ct() {
    let m = 2 * D;
    let n = m;
    let s = CcsStructure::new(vec![Mat::identity(n)], SparsePoly::new(1, vec![])).expect("valid CCS");
    let s_bar = s.transform_matrices_superneo().expect("superneo transform");

    let z = deterministic_vec(m, 0x1111_2222_3333_4444);
    let z_ring = to_ring_blocks(&z);

    let matrix_bar = &s_bar.matrices[0];
    for (r, zr) in z.iter().enumerate().take(n) {
        let got = ring_ct_row_eval(matrix_bar, r, &z_ring, m);
        assert_eq!(*zr, got, "identity row mismatch at row={r}");
    }
}

#[test]
fn superneo_transform_general_matrix_matches_field_matrix_vector_product() {
    let n = 4usize;
    let m = 2 * D;
    let mut mat = Mat::zero(n, m, F::ZERO);
    for r in 0..n {
        for c in (r..m).step_by(17) {
            // Keep matrix sparse but nontrivial across both D-blocks.
            mat[(r, c)] = F::from_u64((r as u64 + 3) * (c as u64 + 5));
        }
    }

    let s = CcsStructure::new(vec![mat], SparsePoly::new(1, vec![])).expect("valid CCS");
    let s_bar = s.transform_matrices_superneo().expect("superneo transform");

    let z = deterministic_vec(m, 0x9999_aaaa_bbbb_cccc);
    let z_ring = to_ring_blocks(&z);

    let y_field = field_mul_rows(&s.matrices[0], &z, n);
    let matrix_bar = &s_bar.matrices[0];
    for (r, y) in y_field.iter().enumerate().take(n) {
        let got = ring_ct_row_eval(matrix_bar, r, &z_ring, m);
        assert_eq!(*y, got, "row mismatch at row={r}");
    }
}
