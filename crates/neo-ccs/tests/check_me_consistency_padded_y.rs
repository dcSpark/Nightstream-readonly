#![allow(non_snake_case)]

use neo_ccs::{
    poly::SparsePoly, poly::Term, relations::check_me_consistency, traits::SModuleHomomorphism, CcsStructure, Mat,
    MeInstance, MeWitness,
};
use neo_math::ring::D;
use neo_math::K;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;

struct TestL;

impl SModuleHomomorphism<Fq, Vec<Fq>> for TestL {
    fn commit(&self, z: &Mat<Fq>) -> Vec<Fq> {
        z.as_slice().to_vec()
    }

    fn project_x(&self, z: &Mat<Fq>, min: usize) -> Mat<Fq> {
        let (d, m) = (z.rows(), z.cols());
        assert!(min <= m);
        let mut out = Mat::zero(d, min, Fq::ZERO);
        for c in 0..min {
            for r in 0..d {
                out[(r, c)] = z[(r, c)];
            }
        }
        out
    }
}

#[test]
fn me_consistency_accepts_padded_y_rows() {
    // CCS: n=4 (power of two), m=3, t=1, f(y)=y0 (linear)
    let n = 4usize;
    let m = 3usize;
    let m0 = Mat::from_row_major(
        n,
        m,
        vec![
            Fq::ONE,
            Fq::ZERO,
            Fq::ZERO,
            Fq::ZERO,
            Fq::ONE,
            Fq::ZERO,
            Fq::ZERO,
            Fq::ZERO,
            Fq::ONE,
            Fq::ONE,
            Fq::ONE,
            Fq::ONE,
        ],
    );
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: Fq::ONE,
            exps: vec![1],
        }],
    );
    let s = CcsStructure::new(vec![m0.clone()], f).unwrap();

    // Construct Z (d×m) with small entries
    let d = D;
    let mut Z = Mat::zero(d, m, Fq::ZERO);
    for c in 0..m {
        for r in 0..d {
            Z[(r, c)] = Fq::from_u64(((r + c) % 5) as u64);
        }
    }

    let L = TestL;

    let m_in = 1usize;
    let c = L.commit(&Z);
    let X = L.project_x(&Z, m_in);

    // Choose r ∈ K^ell with ell=log2(n)=2.
    let r = vec![K::from(Fq::from_u64(3)), K::from(Fq::from_u64(5))]; // arbitrary
    let rb = neo_ccs::utils::tensor_point::<K>(&r);

    // v = M^T rb (in K^m)
    let v_k = {
        let mut v = vec![K::ZERO; m];
        for row in 0..n {
            let rb_r = rb[row];
            let row_slice = m0.row(row).to_vec();
            for cidx in 0..m {
                v[cidx] += K::from(row_slice[cidx]) * rb_r;
            }
        }
        v
    };

    // y = Z * v (in K^d)
    let y0 = neo_ccs::utils::mat_vec_mul_fk::<Fq, K>(Z.as_slice(), d, m, &v_k);

    // Pad y to 2^{ell_d} (typically 64 for D=54).
    let mut y0_padded = y0.clone();
    let d_pad = D.next_power_of_two();
    y0_padded.resize(d_pad, K::ZERO);

    let inst = MeInstance::<_, Fq, K> {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c,
        X,
        r,
        s_col: vec![],
        y: vec![y0_padded.clone()],
        y_scalars: vec![K::ZERO],
        y_zcol: vec![],
        m_in,
        fold_digest: [0u8; 32],
    };
    let wit = MeWitness::<Fq> { Z: Z.clone() };

    assert!(check_me_consistency(&s, &L, &inst, &wit).is_ok());

    // Non-zero padding must be rejected.
    let mut inst_bad = inst.clone();
    inst_bad.y[0][D] += K::ONE;
    assert!(check_me_consistency(&s, &L, &inst_bad, &wit).is_err());
}
