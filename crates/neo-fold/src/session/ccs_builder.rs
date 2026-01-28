use neo_ccs::{CcsMatrix, CcsStructure, CscMat, SparsePoly, Term};
use p3_field::{Field, PrimeCharacteristicRing};

/// A single R1CS row: A(z) * B(z) = C(z), where each side is a sparse linear form.
#[derive(Clone, Debug)]
pub struct R1csRow<F> {
    pub a_terms: Vec<(usize, F)>,
    pub b_terms: Vec<(usize, F)>,
    pub c_terms: Vec<(usize, F)>,
}

/// Minimal, ergonomic R1CS→CCS builder intended for `neo_fold::session` users.
///
/// This builder:
/// - lets callers add R1CS constraints without manually managing matrices, and
/// - emits a CCS (n×m) with an optional reserved all-zero tail row region for shared-CPU-bus injection.
#[derive(Clone, Debug)]
pub struct CcsBuilder<F> {
    m_in: usize,
    const_one_col: usize,
    rows: Vec<R1csRow<F>>,
}

impl<F> CcsBuilder<F>
where
    F: Field + PrimeCharacteristicRing + Copy + Send + Sync,
{
    pub fn new(m_in: usize, const_one_col: usize) -> Result<Self, String> {
        if m_in == 0 {
            return Err("CcsBuilder: m_in must be >= 1".into());
        }
        if const_one_col >= m_in {
            return Err(format!(
                "CcsBuilder: const_one_col({const_one_col}) must be < m_in({m_in})"
            ));
        }
        Ok(Self {
            m_in,
            const_one_col,
            rows: Vec::new(),
        })
    }

    #[inline]
    pub fn m_in(&self) -> usize {
        self.m_in
    }

    #[inline]
    pub fn const_one_col(&self) -> usize {
        self.const_one_col
    }

    #[inline]
    pub fn rows_len(&self) -> usize {
        self.rows.len()
    }

    pub fn r1cs_terms(
        &mut self,
        a_terms: impl IntoIterator<Item = (usize, F)>,
        b_terms: impl IntoIterator<Item = (usize, F)>,
        c_terms: impl IntoIterator<Item = (usize, F)>,
    ) -> &mut Self {
        self.rows.push(R1csRow {
            a_terms: a_terms.into_iter().collect(),
            b_terms: b_terms.into_iter().collect(),
            c_terms: c_terms.into_iter().collect(),
        });
        self
    }

    pub fn r1cs_cols(&mut self, a_cols: &[usize], b_cols: &[usize], c_cols: &[usize]) -> &mut Self {
        self.r1cs_terms(
            a_cols.iter().copied().map(|c| (c, F::ONE)),
            b_cols.iter().copied().map(|c| (c, F::ONE)),
            c_cols.iter().copied().map(|c| (c, F::ONE)),
        )
    }

    /// Add an unconditional equality constraint: `left == right`.
    ///
    /// Encoded as: (left - right) * 1 = 0.
    pub fn eq(&mut self, left: usize, right: usize) -> &mut Self {
        self.r1cs_terms(
            [(left, F::ONE), (right, -F::ONE)],
            [(self.const_one_col, F::ONE)],
            [],
        )
    }

    /// Add a per-lane "continuity" constraint:
    /// `before[j+1] == after[j]` for j=0..N-2.
    pub fn lane_continuity<const N: usize>(&mut self, before: super::Lane<N>, after: super::Lane<N>) -> &mut Self {
        for j in 0..(N.saturating_sub(1)) {
            self.eq(before.at(j + 1), after.at(j));
        }
        self
    }

    /// Build a (potentially rectangular) CCS with:
    /// - witness width `m = m_min`, and
    /// - constraint rows `n = cpu_rows + reserved_trailing_rows`.
    ///
    /// The last `reserved_trailing_rows` are left all-zero in A/B/C to support shared-bus
    /// constraint injection via `extend_ccs_with_shared_cpu_bus_constraints`.
    pub fn build_rect(self, m_min: usize, reserved_trailing_rows: usize) -> Result<CcsStructure<F>, String> {
        let cpu_rows = self.rows.len();
        if cpu_rows == 0 {
            return Err("CcsBuilder: no constraints added".into());
        }
        let n = cpu_rows
            .checked_add(reserved_trailing_rows)
            .ok_or_else(|| "CcsBuilder: cpu_rows + reserved_trailing_rows overflow".to_string())?;
        let m = m_min;

        if self.m_in > m {
            return Err(format!(
                "CcsBuilder: m_in({}) exceeds CCS width m({m})",
                self.m_in
            ));
        }

        let mut a_trips: Vec<(usize, usize, F)> = Vec::new();
        let mut b_trips: Vec<(usize, usize, F)> = Vec::new();
        let mut c_trips: Vec<(usize, usize, F)> = Vec::new();

        for (row, r) in self.rows.iter().enumerate() {
            for &(col, coeff) in &r.a_terms {
                if col >= m {
                    return Err(format!("CcsBuilder: A term col {col} out of range (m={m})"));
                }
                a_trips.push((row, col, coeff));
            }
            for &(col, coeff) in &r.b_terms {
                if col >= m {
                    return Err(format!("CcsBuilder: B term col {col} out of range (m={m})"));
                }
                b_trips.push((row, col, coeff));
            }
            for &(col, coeff) in &r.c_terms {
                if col >= m {
                    return Err(format!("CcsBuilder: C term col {col} out of range (m={m})"));
                }
                c_trips.push((row, col, coeff));
            }
        }

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

        // Match `neo_ccs::r1cs_to_ccs` behavior: insert identity-first only when square.
        let (matrices, f) = if n == m {
            (
                vec![
                    CcsMatrix::Identity { n },
                    CcsMatrix::Csc(CscMat::from_triplets(a_trips, n, m)),
                    CcsMatrix::Csc(CscMat::from_triplets(b_trips, n, m)),
                    CcsMatrix::Csc(CscMat::from_triplets(c_trips, n, m)),
                ],
                f_base.insert_var_at_front(),
            )
        } else {
            (
                vec![
                    CcsMatrix::Csc(CscMat::from_triplets(a_trips, n, m)),
                    CcsMatrix::Csc(CscMat::from_triplets(b_trips, n, m)),
                    CcsMatrix::Csc(CscMat::from_triplets(c_trips, n, m)),
                ],
                f_base,
            )
        };

        CcsStructure::new_sparse(matrices, f).map_err(|e| format!("CcsBuilder: invalid CCS: {e:?}"))
    }
}
