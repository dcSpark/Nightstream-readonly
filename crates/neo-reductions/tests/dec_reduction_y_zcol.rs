#![allow(non_snake_case)]

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, Mat, MeInstance};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

fn project_x_first_cols(Z: &Mat<F>, m_in: usize) -> Mat<F> {
    let mut X = Mat::zero(D, m_in, F::ZERO);
    for r in 0..D {
        for c in 0..m_in {
            X[(r, c)] = Z[(r, c)];
        }
    }
    X
}

fn eval_zcol(Z: &Mat<F>, s_col: &[K], m: usize, ell_d: usize) -> Vec<K> {
    use neo_ccs::utils::{mat_vec_mul_fk, tensor_point};
    let chi_s = tensor_point::<K>(s_col);
    let mut y = mat_vec_mul_fk::<F, K>(Z.as_slice(), Z.rows(), Z.cols(), &chi_s[..m]);
    y.resize(1usize << ell_d, K::ZERO);
    y
}

#[test]
fn dec_reduction_emits_and_checks_y_zcol() {
    let params = NeoParams::goldilocks_127();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let d_pad = 1usize << ell_d;

    // CCS: n=m=2, t=1 (identity). Only shapes matter for this test.
    let s = CcsStructure::new(vec![Mat::identity(2)], neo_ccs::poly::SparsePoly::new(1, vec![])).unwrap();

    let m_in = 1usize;
    let r = vec![K::from(F::from_u64(3))]; // ell_n = 1
    let s_col = vec![K::from(F::from_u64(5))]; // ell_m = 1 (since m=2)

    // Two DEC digits Z0, Z1 (D×m).
    let m = s.m;
    let mut z0 = Mat::zero(D, m, F::ZERO);
    let mut z1 = Mat::zero(D, m, F::ZERO);
    for rho in 0..D {
        for c in 0..m {
            z0[(rho, c)] = F::from_u64((1 + rho as u64 + 3 * c as u64) % 97);
            z1[(rho, c)] = F::from_u64((7 + rho as u64 + 5 * c as u64) % 97);
        }
    }
    let Z_split = vec![z0.clone(), z1.clone()];

    // Parent witness Z = Z0 + b·Z1.
    let bF = F::from_u64(params.b as u64);
    let mut Z_parent = Mat::zero(D, m, F::ZERO);
    for rho in 0..D {
        for c in 0..m {
            Z_parent[(rho, c)] = z0[(rho, c)] + bF * z1[(rho, c)];
        }
    }

    let X_parent = project_x_first_cols(&Z_parent, m_in);
    let (y_parent, y_scalars_parent) = neo_reductions::common::compute_y_from_Z_and_r(&s, &Z_parent, &r, ell_d, params.b);
    assert_eq!(y_parent.len(), s.t());
    assert_eq!(y_parent[0].len(), d_pad);

    let y_zcol_parent = eval_zcol(&Z_parent, &s_col, m, ell_d);
    assert_eq!(y_zcol_parent.len(), d_pad);

    let parent = MeInstance::<Commitment, F, K> {
        c: Commitment::zeros(params.d as usize, 1),
        X: X_parent,
        r: r.clone(),
        s_col: s_col.clone(),
        y: y_parent,
        y_scalars: y_scalars_parent,
        y_zcol: y_zcol_parent.clone(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };

    let (children, ok_y, ok_X) = neo_reductions::optimized_engine::dec_reduction_paper_exact::<F>(
        &s,
        &params,
        &parent,
        &Z_split,
        ell_d,
    );
    assert!(ok_y && ok_X, "DEC must pass y/X checks (including y_zcol)");
    assert_eq!(children.len(), 2);
    for (i, child) in children.iter().enumerate() {
        assert_eq!(child.s_col, s_col, "child s_col must match parent");
        assert_eq!(child.y_zcol.len(), d_pad, "child y_zcol must be padded");
        let want = eval_zcol(&Z_split[i], &s_col, m, ell_d);
        assert_eq!(child.y_zcol, want, "child y_zcol must equal Z_i · chi(s_col)");
    }

    // Explicitly check y_zcol decomposition: parent = child0 + b·child1.
    let bK = K::from(bF);
    let mut lhs = vec![K::ZERO; d_pad];
    for t in 0..d_pad {
        lhs[t] = children[0].y_zcol[t] + bK * children[1].y_zcol[t];
    }
    assert_eq!(lhs, y_zcol_parent);
}
