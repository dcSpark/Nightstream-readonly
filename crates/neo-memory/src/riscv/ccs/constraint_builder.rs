use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::relations::CcsStructure;
use neo_ccs::sparse::{CcsMatrix, CscMat};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

#[derive(Clone, Debug)]
pub(super) struct Constraint<Ff: PrimeCharacteristicRing + Copy> {
    pub condition_col: usize,
    pub negate_condition: bool,
    pub additional_condition_cols: Vec<usize>,
    pub b_terms: Vec<(usize, Ff)>,
    pub c_terms: Vec<(usize, Ff)>,
}

impl<Ff: PrimeCharacteristicRing + Copy> Constraint<Ff> {
    pub fn terms(condition_col: usize, negate_condition: bool, b_terms: Vec<(usize, Ff)>) -> Self {
        Self {
            condition_col,
            negate_condition,
            additional_condition_cols: Vec::new(),
            b_terms,
            c_terms: Vec::new(),
        }
    }
}

pub(super) fn build_r1cs_ccs(
    constraints: &[Constraint<F>],
    n: usize,
    m: usize,
    const_one_col: usize,
) -> Result<CcsStructure<F>, String> {
    if m == 0 {
        return Err("RV32 trace CCS: m must be >= 1".into());
    }
    if n == 0 {
        return Err("RV32 trace CCS: n must be >= 1".into());
    }
    if const_one_col >= m {
        return Err(format!("RV32 trace CCS: const_one_col({const_one_col}) must be < m({m})"));
    }
    if constraints.len() > n {
        return Err(format!(
            "RV32 trace CCS: too many constraints ({}) for CCS with n={} m={}",
            constraints.len(),
            n,
            m
        ));
    }

    // NOTE: This circuit can have very large `m`. Do not materialize dense `n×m` matrices:
    // on wasm32 this can panic with "capacity overflow", and on native it is extremely slow.
    let mut a_trips: Vec<(usize, usize, F)> = Vec::new();
    let mut b_trips: Vec<(usize, usize, F)> = Vec::new();
    let mut c_trips: Vec<(usize, usize, F)> = Vec::new();

    for (row, c) in constraints.iter().enumerate() {
        if c.negate_condition {
            a_trips.push((row, const_one_col, F::ONE));
            a_trips.push((row, c.condition_col, -F::ONE));
            for &col in &c.additional_condition_cols {
                a_trips.push((row, col, -F::ONE));
            }
        } else {
            a_trips.push((row, c.condition_col, F::ONE));
            for &col in &c.additional_condition_cols {
                a_trips.push((row, col, F::ONE));
            }
        }

        for &(col, coeff) in &c.b_terms {
            b_trips.push((row, col, coeff));
        }
        for &(col, coeff) in &c.c_terms {
            c_trips.push((row, col, coeff));
        }
    }

    let a = CcsMatrix::Csc(CscMat::from_triplets(a_trips, n, m));
    let b = CcsMatrix::Csc(CscMat::from_triplets(b_trips, n, m));
    let c = CcsMatrix::Csc(CscMat::from_triplets(c_trips, n, m));

    // Base polynomial f(X1,X2,X3) = X1 * X2 - X3
    let f_base = SparsePoly::new(
        3,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![1, 1, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 1],
            },
        ],
    );

    let matrices = vec![a, b, c];

    CcsStructure::new_sparse(matrices, f_base).map_err(|e| format!("RV32 trace CCS: invalid structure: {e:?}"))
}
