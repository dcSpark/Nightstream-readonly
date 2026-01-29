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
    pub fn eq_const(condition_col: usize, const_one_col: usize, left: usize, c: u64) -> Self {
        Self {
            condition_col,
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(left, Ff::ONE), (const_one_col, -Ff::from_u64(c))],
            c_terms: Vec::new(),
        }
    }

    pub fn zero(condition_col: usize, col: usize) -> Self {
        Self {
            condition_col,
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(col, Ff::ONE)],
            c_terms: Vec::new(),
        }
    }

    pub fn terms(condition_col: usize, negate_condition: bool, b_terms: Vec<(usize, Ff)>) -> Self {
        Self {
            condition_col,
            negate_condition,
            additional_condition_cols: Vec::new(),
            b_terms,
            c_terms: Vec::new(),
        }
    }

    pub fn terms_or(condition_cols: &[usize], negate_condition: bool, b_terms: Vec<(usize, Ff)>) -> Self {
        assert!(!condition_cols.is_empty(), "need at least one condition column");
        Self {
            condition_col: condition_cols[0],
            negate_condition,
            additional_condition_cols: condition_cols[1..].to_vec(),
            b_terms,
            c_terms: Vec::new(),
        }
    }

    pub fn mul(left: usize, right: usize, out: usize) -> Self {
        Self {
            condition_col: left,
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(right, Ff::ONE)],
            c_terms: vec![(out, Ff::ONE)],
        }
    }
}

pub(super) fn build_identity_first_r1cs_ccs(
    constraints: &[Constraint<F>],
    m: usize,
    const_one_col: usize,
) -> Result<CcsStructure<F>, String> {
    let n = m;
    if constraints.len() > n {
        return Err(format!(
            "RV32 B1 CCS: too many constraints ({}) for square CCS with m=n={}",
            constraints.len(),
            n
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

    let i_n = CcsMatrix::Identity { n };
    let a = CcsMatrix::Csc(CscMat::from_triplets(a_trips, n, m));
    let b = CcsMatrix::Csc(CscMat::from_triplets(b_trips, n, m));
    let c = CcsMatrix::Csc(CscMat::from_triplets(c_trips, n, m));

    let f = SparsePoly::new(
        4,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![0, 1, 1, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 0, 1],
            },
        ],
    );

    CcsStructure::new_sparse(vec![i_n, a, b, c], f)
        .map_err(|e| format!("RV32 B1 CCS: invalid structure: {e:?}"))
}

