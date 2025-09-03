use p3_goldilocks::Goldilocks as Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand::{RngCore, CryptoRng};
use crate::types::{PP, Commitment};
use crate::error::{AjtaiError, AjtaiResult};


/// Bring in ring & S-action APIs from neo-math.
use neo_math::ring::{Rq as RqEl, cf_inv as cf_unmap, cf, D, ETA};
use neo_math::s_action::SAction;

/// Sample a uniform element from F_q using rejection sampling to avoid bias.
#[inline]
fn sample_uniform_fq<R: RngCore + CryptoRng>(rng: &mut R) -> Fq {
    // Rejection sampling: draw u64; accept if < q; otherwise redraw.
    const Q: u64 = <Fq as PrimeField64>::ORDER_U64; // 2^64 - 2^32 + 1
    loop {
        let x = rng.next_u64();
        if x < Q { return Fq::from_u64(x); }
    }
}

/// Rotation "one-step" for Φ₈₁(X) = X^54 + X^27 + 1
/// 
/// Turns column t into column t+1 in O(d) (no ring multiply).
/// For Φ₈₁, the step is: next[0] = -v_{d-1}, next[27] = v_{26} - v_{d-1},
/// next[k] = v_{k-1} for k ∈ {1,...,d-1}\{27}.
#[inline]
fn rot_step_phi_81(cur: &[Fq; D], next: &mut [Fq; D]) {
    let last = cur[D - 1];
    // shift: next[k] = cur[k-1] for k>=1; next[0] = 0
    next[0] = Fq::ZERO;
    for k in 1..D { next[k] = cur[k - 1]; }
    // cyclotomic corrections for X^54 ≡ -X^27 - 1
    next[0] -= last;        // -1 * last
    next[27] -= last;       // -X^27 * last
}

/// Optional: if you ever support X^D + 1 rings (AGL/Mersenne), use this fallback
#[allow(dead_code)]
#[inline]
fn rot_step_xd_plus_1(cur: &[Fq; D], next: &mut [Fq; D]) {
    let last = cur[D - 1];
    next[0] = Fq::ZERO;
    for k in 1..D { next[k] = cur[k - 1]; }
    next[0] -= last; // X^D ≡ -1
}

/// Rotation step dispatcher - compile-time constant for η=81 ⇒ D=54
#[inline]
fn rot_step(cur: &[Fq; D], next: &mut [Fq; D]) {
    // Compile-time constant in this repo (η=81 ⇒ D=54); keep a readable switch.
    if ETA == 81 { rot_step_phi_81(cur, next) }
    else { rot_step_xd_plus_1(cur, next) } // safe fallback if you later add AGL
}

/// MUST: Setup(κ,m) → sample M ← R_q^{κ×m} uniformly (Def. 9).
pub fn setup<R: RngCore + CryptoRng>(rng: &mut R, d: usize, kappa: usize, m: usize) -> AjtaiResult<PP<RqEl>> {
    // Ensure d matches the fixed ring dimension from neo-math
    if d != neo_math::ring::D {
        return Err(AjtaiError::InvalidDimensions("d parameter must match ring dimension D"));
    }
    let mut rows = Vec::with_capacity(kappa);
    for _ in 0..kappa {
        let mut row = Vec::with_capacity(m);
        for _ in 0..m {
            // sample ring element uniformly by sampling d random coefficients in F_q and mapping via cf^{-1}
            let coeffs_vec: Vec<Fq> = (0..neo_math::ring::D)
                .map(|_| sample_uniform_fq(rng))
                .collect();
            let coeffs: [Fq; neo_math::ring::D] = coeffs_vec
                .try_into()
                .map_err(|_| AjtaiError::InvalidDimensions("Failed to create coefficient array"))?;
            row.push(cf_unmap(coeffs));
        }
        rows.push(row);
    }
    Ok(PP { kappa, m, d, m_rows: rows })
}

// Variable-time optimization removed for security and simplicity

/// MUST: Commit(pp, Z) = cf(M · cf^{-1}(Z)) as c ∈ F_q^{d×κ}.  S-homomorphic over S by construction.
/// Uses constant-time dense computation for all inputs (audit-ready).
/// Returns error if Z dimensions don't match expected d×m.
#[allow(non_snake_case)]
pub fn try_commit(pp: &PP<RqEl>, Z: &[Fq]) -> AjtaiResult<Commitment> {
    // Z is d×m (column-major by (col*d + row)), output c is d×kappa (column-major)
    let d = pp.d; let m = pp.m;
    if Z.len() != d*m {
        return Err(AjtaiError::SizeMismatch { 
            expected: d*m, 
            actual: Z.len() 
        });
    }
    
    Ok(commit_dense(pp, Z))
}

/// Convenience wrapper that panics on dimension mismatch (for tests and controlled environments).
#[allow(non_snake_case)]
pub fn commit(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    try_commit(pp, Z).expect("commit: Z dimensions must match d×m")
}

/// Constant-time dense commit implementation  
/// Computes c = cf(M · cf^{-1}(Z)) using S-action matrix multiplication
#[allow(non_snake_case)]
fn commit_dense(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    let d = pp.d; let m = pp.m; let kappa = pp.kappa;

    // Pre-extract columns of Z (digits per column)
    let cols: Vec<&[Fq]> = (0..m).map(|j| &Z[j*d .. (j+1)*d]).collect();

    let mut c = Commitment::zeros(d, kappa);

    // For each Ajtai row i, compute Σ_j S-action(a_ij) · Z_col_j into c_col_i.
    for i in 0..kappa {
        let acc = c.col_mut(i);
        #[allow(clippy::needless_range_loop)]
        for j in 0..m {
            let a_ij = &pp.m_rows[i][j];           // R_q element
            let s_action = SAction::from_ring(*a_ij);  // Create S-action from ring element
            let v: [Fq; neo_math::ring::D] = cols[j].try_into().expect("column length should be d"); // digits for column j

            // Apply S-action to the coefficient vector
            let result = s_action.apply_vec(&v);
            
            // Constant-time accumulation (no secret-dependent branching).
            for (a, &r) in acc.iter_mut().zip(&result) {
                *a += r;
            }
        }
    }
    c
}

/// MUST: Verify opening by recomputing commitment (binding implies uniqueness).
#[allow(non_snake_case)]
pub fn verify_open(pp: &PP<RqEl>, c: &Commitment, Z: &[Fq]) -> bool {
    &commit(pp, Z) == c
}

/// MUST: Verify split opening: c == Σ b^{i-1} c_i and Z == Σ b^{i-1} Z_i, with ||Z_i||_∞<b (range assertions done by caller).
#[allow(non_snake_case)]
pub fn verify_split_open(pp: &PP<RqEl>, c: &Commitment, b: u32, c_is: &[Commitment], Z_is: &[Vec<Fq>]) -> bool {
    let k = c_is.len();
    if k != Z_is.len() { return false; }
    // Check shapes
    for ci in c_is { if ci.d != c.d || ci.kappa != c.kappa { return false; } }
    // Recompose commitment
    let mut acc = Commitment::zeros(c.d, c.kappa);
    let mut pow = Fq::ONE;
    let b_f = Fq::from_u64(b as u64);
    #[allow(clippy::needless_range_loop)]
    for i in 0..k {
        for (a, &x) in acc.data.iter_mut().zip(&c_is[i].data) { *a += x * pow; }
        pow *= b_f;
    }
    if &acc != c { return false; }
    // Recompose Z and check commit again
    let d = pp.d; let m = pp.m;
    let mut Z = vec![Fq::ZERO; d*m];
    let mut pow = Fq::ONE;
    for Zi in Z_is {
        if Zi.len() != d*m { return false; }
        for (a, &x) in Z.iter_mut().zip(Zi) { *a += x * pow; }
        pow *= b_f;
    }
    &commit(pp, &Z) == c
}

/// MUST: S-homomorphism: ρ·L(Z) = L(ρ·Z).  We expose helpers for left-multiplying commitments.
/// Since we don't have direct access to SMatrix, we use SAction to operate on the commitment data.
pub fn s_mul(rho_ring: &RqEl, c: &Commitment) -> Commitment {
    let d = c.d; let kappa = c.kappa;
    let mut out = Commitment::zeros(d, kappa);
    let s_action = SAction::from_ring(*rho_ring);
    
    for col in 0..kappa {
        let src: [Fq; neo_math::ring::D] = c.col(col).try_into().expect("column length should be d");
        let dst_result = s_action.apply_vec(&src);
        let dst = out.col_mut(col);
        dst.copy_from_slice(&dst_result);
    }
    out
}

/// Reference implementation for differential testing against the optimized commit
/// 
/// Implements the specification directly: c = cf(M · cf^{-1}(Z))
/// This verifies the fundamental S-action isomorphism cf(a·b) = rot(a)·cf(b) 
/// at the commitment level - exactly the algebra the construction relies on.
/// 
/// ⚠️  FOR TESTING ONLY - NOT CONSTANT TIME ⚠️
/// Available for unit tests and with the 'testing' feature for integration tests
#[cfg(any(test, feature = "testing"))]
#[allow(non_snake_case)]
pub fn commit_spec(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    let d = pp.d; let m = pp.m;
    let mut c = Commitment::zeros(d, pp.kappa);
    
    for i in 0..pp.kappa {
        let acc_i = c.col_mut(i);
        for j in 0..m {
            let s = SAction::from_ring(pp.m_rows[i][j]);
            let v: [Fq; neo_math::ring::D] = Z[j*d..(j+1)*d].try_into().unwrap();
            let w = s.apply_vec(&v);
            for (a, &x) in acc_i.iter_mut().zip(&w) { *a += x; }
        }
    }
    c
}

pub fn s_lincomb(rhos: &[RqEl], cs: &[Commitment]) -> AjtaiResult<Commitment> {
    if rhos.is_empty() {
        return Err(AjtaiError::EmptyInput);
    }
    if rhos.len() != cs.len() {
        return Err(AjtaiError::SizeMismatch { 
            expected: rhos.len(), 
            actual: cs.len() 
        });
    }
    if cs.is_empty() {
        return Err(AjtaiError::EmptyInput);
    }
    
    let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
    for (rho, c) in rhos.iter().zip(cs) {
        let term = s_mul(rho, c);
        acc.add_inplace(&term);
    }
    Ok(acc)
}

/// Constant-time masked columns accumulation (streaming).
///
/// c = cf(M · cf^{-1}(Z)) computed as:
///   for i in 0..kappa, j in 0..m:
///     col <- cf(a_ij)       // column 0 of rot(a_ij)
///     for t in 0..d-1:
///       acc += Z[j*d + t] * col
///       col <- next column via rot_step()
///
/// **Constant-Time Guarantees:**
/// - Fixed iteration counts (no secret-dependent branching)
/// - No secret-dependent memory accesses
/// - Identical execution flow regardless of Z values (sparsity, magnitude)
/// - Assumes underlying field arithmetic is constant-time (true for Goldilocks)
/// 
/// This implements the identity cf(a·b) = rot(a)·cf(b) = Σ(t=0 to d-1) b_t · col_t(rot(a))
#[allow(non_snake_case)]
pub fn commit_masked_ct(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    let d = pp.d; let m = pp.m; let kappa = pp.kappa;
    debug_assert_eq!(d, D);
    assert_eq!(Z.len(), d * m, "Z must be d×m");

    let mut C = Commitment::zeros(d, kappa);

    // For each Ajtai row i and message column j
    for i in 0..kappa {
        let acc_i = C.col_mut(i);
        for j in 0..m {
            // Start from col_0 = cf(a_ij)
            let mut col = cf(pp.m_rows[i][j]);
            let mut nxt = [Fq::ZERO; D];

            // Loop over all base-digits t (constant-time)
            let base = j * d;
            for t in 0..d {
                let mask = Z[base + t];        // any Fq digit (0, ±1, small, or general)
                // acc += mask * col   (branch-free masked add)
                for r in 0..d {
                    // single FMA-like op on the field
                    acc_i[r] += col[r] * mask;
                }
                // Advance to the next rotation column in O(d)
                rot_step(&col, &mut nxt);
                core::mem::swap(&mut col, &mut nxt); // Cheaper than copying [Fq; D]
            }
        }
    }
    C
}

/// Fill `cols` with the d rotation columns of rot(a): cols[t] = cf(a * X^t).
#[inline]
fn precompute_rot_columns(a: RqEl, cols: &mut [[Fq; D]]) {
    let mut col = cf(a);
    let mut nxt = [Fq::ZERO; D];
    for t in 0..D {
        cols[t] = col;
        rot_step(&col, &mut nxt);
        col = nxt;
    }
}

/// Constant-time commit using precomputed rotation columns per (i,j).
///
/// Space/time trade: uses a stack-allocated `[[Fq; D]; D]` scratch per (i,j) to
/// remove per-step rot_step(), keeping the same constant-time masked adds.
/// 
/// **Constant-Time Guarantees:**
/// - Fixed iteration counts (no secret-dependent branching)  
/// - No secret-dependent memory accesses
/// - Identical execution flow regardless of Z values (sparsity, magnitude)
/// - Assumes underlying field arithmetic is constant-time (true for Goldilocks)
/// 
/// This implements the same identity cf(a·b) = rot(a)·cf(b) = Σ(t=0 to d-1) b_t · col_t(rot(a))
/// but precomputes all rotation columns once per (i,j) pair for better cache locality.
#[allow(non_snake_case)]
pub fn commit_precomp_ct(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    let d = pp.d; let m = pp.m; let kappa = pp.kappa;
    debug_assert_eq!(d, D);
    assert_eq!(Z.len(), d * m, "Z must be d×m");

    let mut C = Commitment::zeros(d, kappa);

    // Stack-allocated scratch for columns of rot(a_ij) 
    // 54×54×8 ≈ 23 KiB fits comfortably on most target stacks
    let mut cols = [[Fq::ZERO; D]; D]; // d columns, each length d

    for i in 0..kappa {
        let acc_i = C.col_mut(i);
        for j in 0..m {
            precompute_rot_columns(pp.m_rows[i][j], &mut cols);
            let base = j * d;
            // Constant schedule: always loop over all t
            for t in 0..d {
                let mask = Z[base + t];
                let col_t = &cols[t];
                for r in 0..d {
                    acc_i[r] += col_t[r] * mask;
                }
            }
        }
    }
    C
}