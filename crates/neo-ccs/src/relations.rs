use p3_field::Field;

use neo_math::D;

use crate::{
    error::{CcsError, RelationError},
    matrix::{Mat, MatRef},
    poly::SparsePoly,
    sparse::{CcsMatrix, CscMat},
    traits::SModuleHomomorphism,
    utils::{mat_vec_mul_fk, tensor_point},
};

/// CCS structure: matrices {M_j} and a sparse polynomial `f` in `t` variables.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CcsStructure<F> {
    /// M_j ∈ F^{n×m}, j = 0..t-1
    pub matrices: Vec<CcsMatrix<F>>,
    /// Degree-`<u` polynomial in t variables.
    pub f: SparsePoly<F>,
    /// n (rows)
    pub n: usize,
    /// m (cols)
    pub m: usize,
}

impl<F: Field> CcsStructure<F> {
    /// Create a CCS structure; validates matrix shapes & polynomial arity.
    pub fn new(matrices: Vec<Mat<F>>, f: SparsePoly<F>) -> Result<Self, RelationError>
    where
        F: p3_field::PrimeCharacteristicRing + Copy + Eq + Send + Sync,
    {
        if matrices.is_empty() {
            return Err(RelationError::InvalidStructure);
        }
        let n = matrices[0].rows();
        let m = matrices[0].cols();
        for mj in matrices.iter() {
            if mj.rows() != n || mj.cols() != m {
                return Err(RelationError::InvalidStructure);
            }
            if mj.rows() == 0 || mj.cols() == 0 {
                return Err(RelationError::InvalidStructure);
            }
        }
        let t = matrices.len();
        if f.arity() != t {
            return Err(RelationError::PolyArity {
                poly_arity: f.arity(),
                t,
            });
        }

        let matrices = matrices
            .into_iter()
            .map(|mj| {
                if mj.is_identity_hint() {
                    CcsMatrix::Identity { n: mj.rows() }
                } else {
                    CcsMatrix::Csc(CscMat::from_dense_row_major(&mj))
                }
            })
            .collect();

        Ok(Self { matrices, f, n, m })
    }

    /// Create a CCS structure from sparse matrices (CSC / identity).
    pub fn new_sparse(matrices: Vec<CcsMatrix<F>>, f: SparsePoly<F>) -> Result<Self, RelationError> {
        if matrices.is_empty() {
            return Err(RelationError::InvalidStructure);
        }
        let n = matrices[0].rows();
        let m = matrices[0].cols();
        for mj in matrices.iter() {
            if mj.rows() != n || mj.cols() != m {
                return Err(RelationError::InvalidStructure);
            }
            if mj.rows() == 0 || mj.cols() == 0 {
                return Err(RelationError::InvalidStructure);
            }
        }
        let t = matrices.len();
        if f.arity() != t {
            return Err(RelationError::PolyArity {
                poly_arity: f.arity(),
                t,
            });
        }
        Ok(Self { matrices, f, n, m })
    }

    /// Number of matrices (arity of `f`).
    pub fn t(&self) -> usize {
        self.matrices.len()
    }

    /// Maximum degree of the CCS polynomial.
    pub fn max_degree(&self) -> u32 {
        self.f.max_degree()
    }

    /// Ensure the first matrix is the identity I_n, as assumed by paper's NC semantics.
    /// If not, insert I_n at index 0 and shift the polynomial arity/variables accordingly.
    pub fn ensure_identity_first(&self) -> Result<Self, RelationError>
    where
        F: p3_field::PrimeCharacteristicRing + Copy + Eq + Clone,
    {
        // If not square, we cannot insert a true identity; leave structure unchanged.
        if self.n != self.m {
            return Ok(self.clone());
        }
        let is_id0 = self
            .matrices
            .first()
            .map(|m0| m0.is_identity())
            .unwrap_or(false);
        if is_id0 {
            return Ok(self.clone());
        }
        // Insert identity at position 0
        let mut matrices = self.matrices.clone();
        matrices.insert(0, CcsMatrix::Identity { n: self.n });
        // Shift polynomial variables by inserting a dummy variable at the front
        let f = self.f.insert_var_at_front();
        Ok(CcsStructure {
            matrices,
            f,
            n: self.n,
            m: self.m,
        })
    }

    /// Owned variant of `ensure_identity_first` that avoids cloning when `M₀` is already identity.
    ///
    /// This is useful in hot paths where callers already own a `CcsStructure` and only need to
    /// normalize it (if necessary) for Ajtai/NC semantics.
    pub fn ensure_identity_first_owned(mut self) -> Result<Self, RelationError>
    where
        F: p3_field::PrimeCharacteristicRing + Copy + Eq + Clone,
    {
        // If not square, we cannot insert a true identity; leave structure unchanged.
        if self.n != self.m {
            return Ok(self);
        }
        let is_id0 = self
            .matrices
            .first()
            .map(|m0| m0.is_identity())
            .unwrap_or(false);
        if is_id0 {
            return Ok(self);
        }
        self.matrices.insert(0, CcsMatrix::Identity { n: self.n });
        self.f = self.f.insert_var_at_front();
        Ok(self)
    }

    /// **STRICT** validation: Assert that M₀ = I_n for Ajtai/NC pipeline.
    ///
    /// The Ajtai norm constraint (NC) layer assumes the first matrix is the identity
    /// for digit-range checks. If this invariant is violated, the sumcheck will fail
    /// with a mysterious error later. This function fails fast with a clear error message.
    ///
    /// # Errors
    /// - Returns error if n ≠ m (non-square CCS cannot have square identity)
    /// - Returns error if matrices list is empty
    /// - Returns error if M₀ is not the identity matrix I_n
    ///
    /// # Example
    /// ```ignore
    /// // Before using CCS in Ajtai/NC pipeline:
    /// ccs.assert_m0_is_identity_for_nc()?;
    /// ```
    pub fn assert_m0_is_identity_for_nc(&self) -> Result<(), RelationError>
    where
        F: p3_field::PrimeCharacteristicRing + Copy + Eq,
    {
        // Check 1: Square CCS required for identity to even make sense
        if self.n != self.m {
            return Err(RelationError::Message(format!(
                "Ajtai NC requires square CCS (n_constraints == n_vars), got {}×{}. \
                 You may need to pad your R1CS to square dimensions.",
                self.n, self.m
            )));
        }

        // Check 2: Must have at least one matrix
        if self.matrices.is_empty() {
            return Err(RelationError::Message(
                "Ajtai NC expects at least one matrix (M₀) in CCS".into(),
            ));
        }

        // Check 3: M₀ must be the identity matrix
        if !self.matrices[0].is_identity() {
            return Err(RelationError::Message(
                "Ajtai NC requires M₀ = I_n (identity matrix). \
                 Your CCS has a non-identity first matrix. \
                 This usually happens with rectangular R1CS or when r1cs_to_ccs \
                 doesn't produce identity-first form. \
                 Try: (1) ensure n==m in R1CS, or (2) call ensure_identity_first() \
                 before this check."
                    .into(),
            ));
        }

        Ok(())
    }
}

/// MCS instance: (c, x) with public inputs x ⊂ z (see Def. 17).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct McsInstance<C, F> {
    /// Commitment to Z (Ajtai over decomposition).
    pub c: C,
    /// Public inputs x ∈ F^{m_in}; z = x || w.
    pub x: Vec<F>,
    /// m_in
    pub m_in: usize,
}

/// MCS witness: w and its decomposition Z = Decomp_b(z) (we need Z for consistency checks).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[allow(non_snake_case)]
pub struct McsWitness<F> {
    /// Private witness w ∈ F^{m - m_in}.
    pub w: Vec<F>,
    /// Z ∈ F^{d×m}: decomposition matrix of z = x || w.
    pub Z: Mat<F>,
}

/// ME instance: (c, X, r, {y_j}). See Def. 18.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[allow(non_snake_case)]
pub struct MeInstance<C, F, K> {
    /// Commitment to Z.
    pub c: C,
    /// X = L_x(Z) ∈ F^{d×m_in}
    pub X: Mat<F>,
    /// r ∈ K^{log n}
    pub r: Vec<K>,
    /// s_col ∈ K^{log m}: column-domain point used for the digit-range (NC) check.
    ///
    /// Legacy (square/identity-first) pipelines may leave this empty.
    #[serde(default)]
    pub s_col: Vec<K>,
    /// y_j digit rows for j=0..t-1.
    ///
    /// Callers may store either:
    /// - the unpadded length `d` (= `Z.rows()`), or
    /// - the Ajtai-padded length `2^{ell_d}` (typically `D.next_power_of_two()`),
    ///   in which case the tail must be all zeros.
    pub y: Vec<Vec<K>>,
    /// **SECURITY**: Y_j(r) = ⟨(M_j z), χ_r⟩ ∈ K scalars for CCS terminal verification
    /// These are the CORRECT values needed for sum-check terminal check, not sums of y vector components
    pub y_scalars: Vec<K>,
    /// y_zcol := Z · χ_{s_col} ∈ K^{d} (digit rows, typically padded to 2^{ell_d}).
    ///
    /// Legacy (square/identity-first) pipelines may leave this empty.
    #[serde(default)]
    pub y_zcol: Vec<K>,
    /// m_in
    pub m_in: usize,
    /// **SECURITY**: Transcript-derived digest binding this ME to the folding proof
    pub fold_digest: [u8; 32],
    /// **PATTERN A**: Pre-commitment coordinates for linear link constraints
    /// c_step_coords[i] are the coordinates of the pre-commitment (with ρ=0 for EV part)
    /// Used to enforce: c_full[i] - c_step_coords[i] = ⟨L_i, U⟩ where U = ρ·y_step
    pub c_step_coords: Vec<F>,
    /// Pattern A: Offset where ρ-dependent part starts in witness vector (unused in Pattern B)
    pub u_offset: usize,
    /// Pattern A: Length of the ρ-dependent part (unused in Pattern B)
    pub u_len: usize,
}

/// ME witness: Z.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[allow(non_snake_case)]
pub struct MeWitness<F> {
    /// Z ∈ F^{d×m}
    pub Z: Mat<F>,
}

/// Check `c == L(Z)` for MCS.
/// Note: The critical Z == Decomp_b(z) check is now handled in the folding pipeline
/// where both neo-ccs and neo-ajtai dependencies are available.
pub fn check_mcs_opening<F: Field, C, L: SModuleHomomorphism<F, C>>(
    l: &L,
    inst: &McsInstance<C, F>,
    wit: &McsWitness<F>,
) -> Result<Vec<F>, CcsError>
where
    C: PartialEq,
{
    // shape sanity
    let m = inst.m_in + wit.w.len();
    if wit.Z.cols() != m {
        return Err(CcsError::Dim {
            context: "Z (cols) vs m_in + |w|",
            expected: (wit.Z.rows(), m),
            got: (wit.Z.rows(), wit.Z.cols()),
        });
    }
    // z = x || w
    if inst.x.len() != inst.m_in {
        return Err(CcsError::Len {
            context: "x (public)",
            expected: inst.m_in,
            got: inst.x.len(),
        });
    }
    let mut z = inst.x.clone();
    z.extend_from_slice(&wit.w);

    // === COMMITMENT BINDING ===
    let c_star = l.commit(&wit.Z);
    if c_star != inst.c {
        return Err(CcsError::Relation("c != L(Z)".into()));
    }

    Ok(z)
}

/// Check `X == L_x(Z)` and `y_j == Z M_j^T r^b` for ME (Def. 18).
pub fn check_me_consistency<F: Field, K: Field + From<F>, C, L: SModuleHomomorphism<F, C>>(
    s: &CcsStructure<F>,
    l: &L,
    inst: &MeInstance<C, F, K>,
    wit: &MeWitness<F>,
) -> Result<(), CcsError>
where
    C: PartialEq,
{
    // X = L_x(Z)
    let x_star = l.project_x(&wit.Z, inst.m_in);
    if x_star.as_slice() != inst.X.as_slice() {
        return Err(CcsError::Relation("X != L_x(Z)".into()));
    }
    // c == L(Z) (always true in Π_CCS/Π_RLC composition; enforce here)
    let c_star = l.commit(&wit.Z);
    if c_star != inst.c {
        return Err(CcsError::Relation("c != L(Z)".into()));
    }

    // y_j == Z M_j^T r^b
    // Allow arbitrary n by deriving ℓ from the next power of two.
    // χ_r is length 2^ℓ, and we consume only the first n entries.
    let n_pad = s.n.next_power_of_two();
    let ell = n_pad.trailing_zeros() as usize;
    if inst.r.len() != ell {
        return Err(CcsError::Len {
            context: "r (extension point)",
            expected: ell,
            got: inst.r.len(),
        });
    }

    // Optional NC channel: y_zcol == Z · χ_{s_col} (column-domain).
    //
    // This is only checked when both `s_col` and `y_zcol` are present, so legacy callers
    // can omit these fields without failing consistency checks.
    if !(inst.s_col.is_empty() && inst.y_zcol.is_empty()) {
        if inst.s_col.is_empty() || inst.y_zcol.is_empty() {
            return Err(CcsError::Relation(
                "incomplete NC channel: expected both s_col and y_zcol".into(),
            ));
        }

        // Column-domain length is derived from CCS width `m` (not `n`).
        let m_pad = s.m.next_power_of_two().max(2);
        let ell_m = m_pad.trailing_zeros() as usize;
        if inst.s_col.len() != ell_m {
            return Err(CcsError::Len {
                context: "s_col (column extension point)",
                expected: ell_m,
                got: inst.s_col.len(),
            });
        }

        // Ajtai padding length for digit rows (matches `1 << ell_d` used by Π_CCS dims).
        let d_pad = D.next_power_of_two();
        let ell_d = d_pad.trailing_zeros() as usize;
        let d_pad = 1usize << ell_d;
        if inst.y_zcol.len() != d_pad {
            return Err(CcsError::Len {
                context: "y_zcol (padded digit rows)",
                expected: d_pad,
                got: inst.y_zcol.len(),
            });
        }

        // Compute y_zcol = Z · χ_{s_col}.
        let chi_s = crate::utils::tensor_point::<K>(&inst.s_col);
        // Consume only the first `m` entries (outside-of-range are implicitly zero).
        use crate::utils::mat_vec_mul_fk;
        let mut y_star = mat_vec_mul_fk::<F, K>(wit.Z.as_slice(), wit.Z.rows(), wit.Z.cols(), &chi_s[..s.m]);
        y_star.resize(d_pad, K::ZERO);

        if y_star.as_slice() != inst.y_zcol.as_slice() {
            return Err(CcsError::Relation("y_zcol != Z · χ_{s_col}".into()));
        }
    }
    let rb = tensor_point::<K>(&inst.r); // K^n

    // for each j: v := M_j^T r^b ∈ K^m; then y_j = Z v ∈ K^d
    if inst.y.len() != s.t() {
        return Err(CcsError::Len {
            context: "|y|",
            expected: s.t(),
            got: inst.y.len(),
        });
    }

    // Ajtai padding length for digit rows (matches `1 << ell_d` used by Π_CCS dims).
    let d_pad = D.next_power_of_two();

    for (j, mj) in s.matrices.iter().enumerate() {
        // v = M_j^T r^b (consume only the first n rows of χ_r)
        let mut v_k_m = vec![K::ZERO; s.m];
        mj.add_mul_transpose_into(&rb, &mut v_k_m, s.n);
        // y*_j = Z v_k_m
        let z_ref = MatRef::from_mat(&wit.Z);
        let y_star = mat_vec_mul_fk::<F, K>(z_ref.data, z_ref.rows, z_ref.cols, &v_k_m);
        let yj = &inst.y[j];
        let d = y_star.len();
        if yj.len() < d {
            return Err(CcsError::Len {
                context: "y[j] (digit row)",
                expected: d,
                got: yj.len(),
            });
        }
        if yj.len() != d && yj.len() != d_pad {
            return Err(CcsError::Len {
                context: "y[j] (digit row)",
                expected: d_pad,
                got: yj.len(),
            });
        }
        if y_star.as_slice() != &yj[..d] {
            return Err(CcsError::Relation("y_j != Z M_j^T r^b".into()));
        }
        if yj[d..].iter().any(|&x| x != K::ZERO) {
            return Err(CcsError::Relation("y_j != Z M_j^T r^b".into()));
        }
    }
    Ok(())
}

/// **MUST**: Verify CCS satisfiability `f(M z) = 0` **row-wise** with public inputs `x`.
///
/// This matches Def. 17's condition `f(Mg_1 z, …, Mg_t z) ∈ ZS_n` by simply
/// checking that for each row i, `f((M_1 z)[i], …, (M_t z)[i]) == 0`.
pub fn check_ccs_rowwise_zero<F: Field>(s: &CcsStructure<F>, x: &[F], w: &[F]) -> Result<(), CcsError> {
    if x.len() + w.len() != s.m {
        return Err(CcsError::Len {
            context: "z = x||w length",
            expected: s.m,
            got: x.len() + w.len(),
        });
    }
    let mut z = x.to_vec();
    z.extend_from_slice(w);

    // Compute M_j z for every j
    let mut mz: Vec<Vec<F>> = Vec::with_capacity(s.t());
    for mj in &s.matrices {
        let mut v = vec![F::ZERO; s.n];
        mj.add_mul_into(&z, &mut v, s.n);
        mz.push(v);
    }

    // Row-wise: for each i, evaluate f( (M_1 z)[i], ..., (M_t z)[i] ) == 0
    for i in 0..s.n {
        let mut point = Vec::with_capacity(s.t());
        for j in 0..s.t() {
            point.push(mz[j][i]);
        }
        let val = s.f.eval(&point);
        if val != F::ZERO {
            return Err(CcsError::RowFail { row: i });
        }
    }
    Ok(())
}

/// **MUST**: Verify **relaxed CCS** `f(M z) = e * u` row-wise (defaults `u=0`, `e=1`).
///
/// This corresponds to the usual relaxed CCS used in Nova/HyperNova/Neo.
pub fn check_ccs_rowwise_relaxed<F: Field>(
    s: &CcsStructure<F>,
    x: &[F],
    w: &[F],
    u: Option<&[F]>,
    e: Option<F>,
) -> Result<(), CcsError> {
    let e = e.unwrap_or(F::ONE);
    let zero_u: Vec<F>;
    let u = match u {
        Some(u) => {
            if u.len() != s.n {
                return Err(CcsError::Len {
                    context: "u (slack)",
                    expected: s.n,
                    got: u.len(),
                });
            }
            u
        }
        None => {
            zero_u = vec![F::ZERO; s.n];
            &zero_u
        }
    };
    if x.len() + w.len() != s.m {
        return Err(CcsError::Len {
            context: "z = x||w length",
            expected: s.m,
            got: x.len() + w.len(),
        });
    }
    let mut z = x.to_vec();
    z.extend_from_slice(w);

    // M_j z for every j
    let mut mz: Vec<Vec<F>> = Vec::with_capacity(s.t());
    for mj in &s.matrices {
        let mut v = vec![F::ZERO; s.n];
        mj.add_mul_into(&z, &mut v, s.n);
        mz.push(v);
    }

    // Row-wise: f( (M_1 z)[i], ..., (M_t z)[i] ) == e * u[i]
    for i in 0..s.n {
        let mut point = Vec::with_capacity(s.t());
        for j in 0..s.t() {
            point.push(mz[j][i]);
        }
        let val = s.f.eval(&point);
        if val != e * u[i] {
            return Err(CcsError::RowFail { row: i });
        }
    }
    Ok(())
}
