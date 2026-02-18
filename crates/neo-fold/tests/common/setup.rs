#![allow(dead_code)]

use std::sync::Arc;

use neo_ajtai::{s_lincomb, s_mul, setup as ajtai_setup, AjtaiSModule, Commitment as Cmt};
use neo_ccs::relations::CcsStructure;
use neo_ccs::sparse::{CcsMatrix, CscMat};
use neo_ccs::Mat;
use neo_fold::shard::CommitMixers;
use neo_math::ring::{cf_inv, Rq as RqEl};
use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub type Mixers = CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>;

pub fn setup_ajtai_committer(params: &NeoParams, m: usize) -> AjtaiSModule {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, m).expect("Ajtai setup should succeed");
    AjtaiSModule::new(Arc::new(pp))
}

pub fn rot_matrix_to_rq(mat: &Mat<F>) -> RqEl {
    debug_assert_eq!(mat.rows(), D);
    debug_assert_eq!(mat.cols(), D);

    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}

pub fn default_mixers() -> Mixers {
    fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Cmt]) -> Cmt {
        assert!(!cs.is_empty(), "mix_rhos_commits: empty commitments");
        if cs.len() == 1 {
            return cs[0].clone();
        }
        let rq_els: Vec<RqEl> = rhos.iter().map(rot_matrix_to_rq).collect();
        s_lincomb(&rq_els, cs).expect("s_lincomb should succeed")
    }

    fn combine_b_pows(cs: &[Cmt], b: u32) -> Cmt {
        assert!(!cs.is_empty(), "combine_b_pows: empty commitments");
        let mut acc = cs[0].clone();
        let mut pow = F::from_u64(b as u64);
        for c in cs.iter().skip(1) {
            let rq_pow = RqEl::from_field_scalar(pow);
            let term = s_mul(&rq_pow, c);
            acc.add_inplace(&term);
            pow *= F::from_u64(b as u64);
        }
        acc
    }

    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

/// Test-only helper: widen CCS column count by appending all-zero columns to every matrix.
/// This is used by legacy no-shared packed-bus tests when bus tails need more per-step columns
/// than the optimized RV32 trace core now commits.
pub fn widen_ccs_cols_for_test(ccs: &mut CcsStructure<F>, target_m: usize) {
    if target_m <= ccs.m {
        return;
    }
    for mat in &mut ccs.matrices {
        match mat {
            CcsMatrix::Identity { n } => {
                let nrows = *n;
                let diag = nrows.min(target_m);
                let mut col_ptr = Vec::with_capacity(target_m + 1);
                for c in 0..=target_m {
                    col_ptr.push(c.min(diag));
                }
                let row_idx: Vec<usize> = (0..diag).collect();
                let vals = vec![F::ONE; diag];
                *mat = CcsMatrix::Csc(CscMat {
                    nrows,
                    ncols: target_m,
                    col_ptr,
                    row_idx,
                    vals,
                });
            }
            CcsMatrix::Csc(csc) => {
                if csc.ncols > target_m {
                    continue;
                }
                let nnz = *csc.col_ptr.last().unwrap_or(&0);
                csc.col_ptr.resize(target_m + 1, nnz);
                csc.ncols = target_m;
            }
        }
    }
    ccs.m = target_m;
}
