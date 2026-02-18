use super::*;

pub(crate) enum CcsOracleDispatch<'a> {
    Optimized(neo_reductions::engines::optimized_engine::oracle::OptimizedOracle<'a, F>),
    #[cfg(feature = "paper-exact")]
    PaperExact(neo_reductions::engines::paper_exact_engine::oracle::PaperExactOracle<'a, F>),
}

impl<'a> RoundOracle for CcsOracleDispatch<'a> {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        match self {
            Self::Optimized(oracle) => oracle.evals_at(points),
            #[cfg(feature = "paper-exact")]
            Self::PaperExact(oracle) => oracle.evals_at(points),
        }
    }

    fn num_rounds(&self) -> usize {
        match self {
            Self::Optimized(oracle) => oracle.num_rounds(),
            #[cfg(feature = "paper-exact")]
            Self::PaperExact(oracle) => oracle.num_rounds(),
        }
    }

    fn degree_bound(&self) -> usize {
        match self {
            Self::Optimized(oracle) => oracle.degree_bound(),
            #[cfg(feature = "paper-exact")]
            Self::PaperExact(oracle) => oracle.degree_bound(),
        }
    }

    fn fold(&mut self, r: K) {
        match self {
            Self::Optimized(oracle) => oracle.fold(r),
            #[cfg(feature = "paper-exact")]
            Self::PaperExact(oracle) => oracle.fold(r),
        }
    }
}

// ============================================================================
// Utilities
// ============================================================================

pub use crate::memory_sidecar::memory::absorb_step_memory;

// ============================================================================
// Optional step-to-step (cross-chunk) linking
// ============================================================================

/// Optional verifier-side linking constraints across adjacent shard steps.
///
/// This is intended for chunked CPU circuits that expose boundary state as part of the public
/// input vector `x` per step, and need the verifier to enforce that the state chains across steps.
#[derive(Clone, Debug)]
pub struct StepLinkingConfig {
    /// Equalities on adjacent steps: require `steps[i].x[prev_idx] == steps[i+1].x[next_idx]`.
    pub prev_next_equalities: Vec<(usize, usize)>,
}

impl StepLinkingConfig {
    pub fn new(prev_next_equalities: Vec<(usize, usize)>) -> Self {
        Self { prev_next_equalities }
    }
}

pub fn check_step_linking(steps: &[StepInstanceBundle<Cmt, F, K>], cfg: &StepLinkingConfig) -> Result<(), PiCcsError> {
    if steps.len() <= 1 || cfg.prev_next_equalities.is_empty() {
        return Ok(());
    }
    for (i, (prev, next)) in steps.iter().zip(steps.iter().skip(1)).enumerate() {
        let prev_x = &prev.mcs_inst.x;
        let next_x = &next.mcs_inst.x;
        for &(prev_idx, next_idx) in &cfg.prev_next_equalities {
            if prev_idx >= prev_x.len() || next_idx >= next_x.len() {
                return Err(PiCcsError::InvalidInput(format!(
                    "step linking index out of range at boundary {i}: prev_x.len()={}, next_x.len()={}, pair=({prev_idx},{next_idx})",
                    prev_x.len(),
                    next_x.len(),
                )));
            }
            if prev_x[prev_idx] != next_x[next_idx] {
                return Err(PiCcsError::ProtocolError(format!(
                    "step linking failed at boundary {i}: prev_x[{prev_idx}] != next_x[{next_idx}]",
                )));
            }
        }
    }
    Ok(())
}

/// Commitment mixers so the coordinator stays scheme-agnostic.
/// - `mix_rhos_commits(ρ, cs)` returns Σ ρ_i · c_i  (S-action).
/// - `combine_b_pows(cs, b)` returns Σ \bar b^{i-1} c_i  (DEC check).
#[derive(Clone, Copy)]
pub struct CommitMixers<MR, MB>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
    MB: Fn(&[Cmt], u32) -> Cmt,
{
    pub mix_rhos_commits: MR,
    pub combine_b_pows: MB,
}

pub fn normalize_me_claims(
    me_claims: &mut [MeInstance<Cmt, F, K>],
    ell_n: usize,
    ell_d: usize,
    t: usize,
) -> Result<(), PiCcsError> {
    let y_pad = 1usize << ell_d;
    for (i, me) in me_claims.iter_mut().enumerate() {
        if me.r.len() != ell_n {
            return Err(PiCcsError::InvalidInput(format!(
                "ME[{}] r.len()={}, expected ell_n={}",
                i,
                me.r.len(),
                ell_n
            )));
        }
        if me.y.len() > t {
            return Err(PiCcsError::InvalidInput(format!(
                "ME[{}] y.len()={}, expected <= t={}",
                i,
                me.y.len(),
                t
            )));
        }
        for (j, row) in me.y.iter_mut().enumerate() {
            if row.len() > y_pad {
                return Err(PiCcsError::InvalidInput(format!(
                    "ME[{}] y[{}].len()={}, expected <= {}",
                    i,
                    j,
                    row.len(),
                    y_pad
                )));
            }
            row.resize(y_pad, K::ZERO);
        }
        me.y.resize_with(t, || vec![K::ZERO; y_pad]);
        if me.y_scalars.len() > t {
            return Err(PiCcsError::InvalidInput(format!(
                "ME[{}] y_scalars.len()={}, expected <= t={}",
                i,
                me.y_scalars.len(),
                t
            )));
        }
        me.y_scalars.resize(t, K::ZERO);
    }
    Ok(())
}

pub(crate) fn validate_me_batch_invariants(batch: &[MeInstance<Cmt, F, K>], context: &str) -> Result<(), PiCcsError> {
    if batch.is_empty() {
        return Ok(());
    }
    let me0 = &batch[0];
    let r0 = &me0.r;
    let m_in0 = me0.m_in;
    let y_len0 = me0.y.len();
    let y_row_len0 = me0.y.first().map(|r| r.len()).unwrap_or(0);
    let y_scalars_len0 = me0.y_scalars.len();

    if me0.X.rows() != D {
        return Err(PiCcsError::ProtocolError(format!(
            "{}: ME claim 0 has X.rows()={}, expected D={}",
            context,
            me0.X.rows(),
            D
        )));
    }
    if me0.X.cols() != m_in0 {
        return Err(PiCcsError::ProtocolError(format!(
            "{}: ME claim 0 has X.cols()={}, expected m_in={}",
            context,
            me0.X.cols(),
            m_in0
        )));
    }

    for (i, me) in batch.iter().enumerate().skip(1) {
        if me.r != *r0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has different r than claim 0 (r-alignment required for RLC)",
                context, i
            )));
        }
        if me.m_in != m_in0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has m_in={}, expected {}",
                context, i, me.m_in, m_in0
            )));
        }
        if me.X.rows() != D || me.X.cols() != m_in0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has X shape {}x{}, expected {}x{}",
                context,
                i,
                me.X.rows(),
                me.X.cols(),
                D,
                m_in0
            )));
        }
        if me.y.len() != y_len0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has y.len()={}, expected {}",
                context,
                i,
                me.y.len(),
                y_len0
            )));
        }
        for (j, row) in me.y.iter().enumerate() {
            if row.len() != y_row_len0 {
                return Err(PiCcsError::ProtocolError(format!(
                    "{}: ME claim {} has y[{}].len()={}, expected {}",
                    context,
                    i,
                    j,
                    row.len(),
                    y_row_len0
                )));
            }
        }
        if me.y_scalars.len() != y_scalars_len0 {
            return Err(PiCcsError::ProtocolError(format!(
                "{}: ME claim {} has y_scalars.len()={}, expected {}",
                context,
                i,
                me.y_scalars.len(),
                y_scalars_len0
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RlcLane {
    Main,
    Val,
}

#[inline]
pub(crate) fn balanced_divrem_i64(v: i64, b: i64) -> (i64, i64) {
    debug_assert!(b >= 2);
    let mut r = v % b;
    let mut q = (v - r) / b;
    let half = b / 2;
    if r > half {
        r -= b;
        q += 1;
    } else if r < -half {
        r += b;
        q -= 1;
    }
    (r, q)
}

#[inline]
pub(crate) fn balanced_divrem_i128(v: i128, b: i128) -> (i128, i128) {
    debug_assert!(b >= 2);
    let mut r = v % b;
    let mut q = (v - r) / b;
    let half = b / 2;
    if r > half {
        r -= b;
        q += 1;
    } else if r < -half {
        r += b;
        q -= 1;
    }
    (r, q)
}

#[inline]
pub(crate) fn f_from_i64(x: i64) -> F {
    if x >= 0 {
        F::from_u64(x as u64)
    } else {
        F::ZERO - F::from_u64((-x) as u64)
    }
}

#[inline]
pub(crate) fn verify_me_y_scalars_canonical(
    me: &MeInstance<Cmt, F, K>,
    b: u32,
    step_idx: usize,
    context: &str,
) -> Result<(), PiCcsError> {
    if me.y_scalars.len() != me.y.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "step {}: {}: y_scalars.len()={} must equal y.len()={}",
            step_idx,
            context,
            me.y_scalars.len(),
            me.y.len()
        )));
    }
    let bK = K::from(F::from_u64(b as u64));
    for (j, row) in me.y.iter().enumerate() {
        if row.len() < D {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: {}: y[{}].len()={} must be >= D={}",
                step_idx,
                context,
                j,
                row.len(),
                D
            )));
        }
        let mut expect = K::ZERO;
        let mut pow = K::ONE;
        for rho in 0..D {
            expect += pow * row[rho];
            pow *= bK;
        }
        if me.y_scalars[j] != expect {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: {}: non-canonical y_scalars at row {}",
                step_idx, context, j
            )));
        }
    }
    Ok(())
}

pub(crate) fn dec_stream_no_witness<MB>(
    params: &NeoParams,
    s: &CcsStructure<F>,
    parent: &MeInstance<Cmt, F, K>,
    Z_mix: &Mat<F>,
    ell_d: usize,
    k_dec: usize,
    combine_b_pows: MB,
    sparse: Option<&SparseCache<F>>,
) -> Result<(Vec<MeInstance<Cmt, F, K>>, Vec<Cmt>, bool, bool, bool), PiCcsError>
where
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    if k_dec == 0 {
        return Err(PiCcsError::InvalidInput("DEC: k_dec must be > 0".into()));
    }
    if Z_mix.rows() != D || Z_mix.cols() != s.m {
        return Err(PiCcsError::InvalidInput(format!(
            "DEC: Z_mix must have shape D×m = {}×{} (got {}×{})",
            D,
            s.m,
            Z_mix.rows(),
            Z_mix.cols()
        )));
    }

    let d_pad = 1usize << ell_d;
    let want_nc_channel = !(parent.s_col.is_empty() && parent.y_zcol.is_empty());
    if want_nc_channel && (parent.s_col.is_empty() || parent.y_zcol.is_empty()) {
        return Err(PiCcsError::InvalidInput(
            "DEC: incomplete NC channel on parent (expected both s_col and y_zcol)".into(),
        ));
    }
    if want_nc_channel && parent.y_zcol.len() != d_pad {
        return Err(PiCcsError::InvalidInput(format!(
            "DEC: parent y_zcol length mismatch (expected {}, got {})",
            d_pad,
            parent.y_zcol.len()
        )));
    }

    enum PpAccess {
        Seeded {
            kappa: usize,
            chunk_size: usize,
            chunk_seeds_by_row: Vec<Vec<[u8; 32]>>,
        },
        Loaded {
            pp: Arc<neo_ajtai::PP<neo_math::ring::Rq>>,
        },
    }

    let pp_access = if let Some(pp) = try_get_loaded_global_pp_for_dims(D, s.m) {
        if pp.kappa == 0 {
            return Err(PiCcsError::InvalidInput("DEC: PP.kappa must be > 0".into()));
        }
        PpAccess::Loaded { pp }
    } else if let Ok((kappa, seed)) = get_global_pp_seeded_params_for_dims(D, s.m) {
        if kappa == 0 {
            return Err(PiCcsError::InvalidInput("DEC: PP.kappa must be > 0".into()));
        }
        let (chunk_size, chunk_seeds_by_row) = seeded_pp_chunk_seeds(seed, kappa, s.m);
        PpAccess::Seeded {
            kappa,
            chunk_size,
            chunk_seeds_by_row,
        }
    } else {
        // Fallback: non-seeded entry. This will materialize PP if needed.
        let pp = get_global_pp_for_dims(D, s.m).map_err(|e| {
            PiCcsError::InvalidInput(format!("DEC: Ajtai PP unavailable for (d,m)=({},{}) ({})", D, s.m, e))
        })?;
        if pp.kappa == 0 {
            return Err(PiCcsError::InvalidInput("DEC: PP.kappa must be > 0".into()));
        }
        PpAccess::Loaded { pp }
    };

    // Build χ_r and v_j = M_j^T · χ_r (same as the reference DEC).
    let ell_n = parent.r.len();
    let n_sz = 1usize
        .checked_shl(ell_n as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("DEC: 2^ell_n overflow".into()))?;
    let n_eff = core::cmp::min(s.n, n_sz);

    // χ_r table over the row/time hypercube.
    //
    // IMPORTANT: Use the same bit order as `eq_points_bool_mask` / `chi_tail_weights`
    // (bit 0 = LSB) so CSC column traversals match the reference DEC.
    #[inline]
    fn chi_tail_weights(bits: &[K]) -> Vec<K> {
        let t = bits.len();
        let len = 1usize << t;
        let mut w = vec![K::ZERO; len];
        w[0] = K::ONE;
        for (i, &b) in bits.iter().enumerate() {
            let step = 1usize << i;
            let one_minus = K::ONE - b;
            for mask in 0..step {
                let v = w[mask];
                w[mask] = v * one_minus;
                w[mask + step] = v * b;
            }
        }
        w
    }

    let chi_r = chi_tail_weights(&parent.r);
    debug_assert_eq!(chi_r.len(), n_sz);

    let chi_s = if want_nc_channel {
        let chi = chi_tail_weights(&parent.s_col);
        if chi.len() < s.m {
            return Err(PiCcsError::InvalidInput(format!(
                "DEC: chi(s_col) too short for CCS width (need >= {}, got {})",
                s.m,
                chi.len()
            )));
        }
        chi
    } else {
        Vec::new()
    };

    let t_mats = s.t();

    enum VjsAccess<'a> {
        Dense(Vec<Vec<K>>),
        Sparse {
            cap: usize,
            cache: &'a SparseCache<F>,
        },
    }

    let vjs_access = if let Some(cache) = sparse {
        if cache.len() != t_mats {
            return Err(PiCcsError::InvalidInput(format!(
                "DEC: sparse cache matrix count mismatch: got {}, expected {}",
                cache.len(),
                t_mats
            )));
        }
        let cap = core::cmp::min(s.m, n_eff);
        VjsAccess::Sparse { cap, cache }
    } else {
        let mut vjs: Vec<Vec<K>> = vec![vec![K::ZERO; s.m]; t_mats];
        for j in 0..t_mats {
            s.matrices[j].add_mul_transpose_into(&chi_r, &mut vjs[j], n_eff);
        }
        VjsAccess::Dense(vjs)
    };

    // Base-b powers in K for y_scalar recomposition.
    let bF = F::from_u64(params.b as u64);
    let bK = K::from(bF);
    let mut pow_b_k = [K::ONE; D];
    for rho in 1..D {
        pow_b_k[rho] = pow_b_k[rho - 1] * bK;
    }

    // Precompute parameters for bounded signed decoding of Z_mix entries.
    let b_u = params.b as u128;
    let mut B_u: u128 = 1;
    for _ in 0..k_dec {
        B_u = B_u.saturating_mul(b_u);
    }
    let p: u128 = F::ORDER_U64 as u128;

    // Fast row-major access.
    let z_rows: Vec<&[F]> = (0..D).map(|r| Z_mix.row(r)).collect();

    struct Acc {
        commit: Vec<[F; D]>, // [digit][kappa] -> [D]
        y: Vec<[K; D]>,      // [digit][t] -> [D]
        y_zcol: Vec<[K; D]>, // [digit] -> [D]
        any_nonzero: Vec<bool>,
        vj: Vec<K>,          // scratch: t
        digits: Vec<i32>,    // scratch: k*D (balanced digits)
        rot_next: [F; D],    // scratch: rotation step output (written fully each time)
        err: Option<String>, // first error wins
    }

    impl Acc {
        fn new(k_dec: usize, kappa: usize, t: usize) -> Self {
            Self {
                commit: vec![[F::ZERO; D]; k_dec * kappa],
                y: vec![[K::ZERO; D]; k_dec * t],
                y_zcol: vec![[K::ZERO; D]; k_dec],
                any_nonzero: vec![false; k_dec],
                vj: vec![K::ZERO; t],
                digits: vec![0i32; k_dec * D],
                rot_next: [F::ZERO; D],
                err: None,
            }
        }

        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        fn add_inplace(&mut self, rhs: &Acc, k_dec: usize, kappa: usize, t: usize) {
            for (dst, src) in self.commit.iter_mut().zip(rhs.commit.iter()) {
                for r in 0..D {
                    dst[r] += src[r];
                }
            }
            for (dst, src) in self.y.iter_mut().zip(rhs.y.iter()) {
                for r in 0..D {
                    dst[r] += src[r];
                }
            }
            for (dst, src) in self.y_zcol.iter_mut().zip(rhs.y_zcol.iter()) {
                for r in 0..D {
                    dst[r] += src[r];
                }
            }
            for i in 0..k_dec {
                self.any_nonzero[i] |= rhs.any_nonzero[i];
            }
            if self.err.is_none() {
                self.err = rhs.err.clone();
            }
            // silence unused warnings when parameters are const-propagated
            let _ = (k_dec, kappa, t);
        }
    }

    let m = s.m;
    let b_i64 = params.b as i64;
    let b_i128 = params.b as i128;

    // Specialized rot_step for Φ₈₁(X) = X^54 + X^27 + 1 (η=81, D=54).
    // Mirrors `neo_ajtai::commit::rot_step_phi_81` but kept local to avoid pulling a large
    // D×D scratch table (`precompute_rot_columns`) into the hot DEC streaming loop.
    #[inline]
    fn rot_step_phi_81(cur: &[F; D], next: &mut [F; D]) {
        let last = cur[D - 1];
        next[0] = F::ZERO;
        next[1..D].copy_from_slice(&cur[..(D - 1)]);
        next[0] -= last;
        next[27] -= last;
    }

    #[inline]
    fn acc_add_assign(acc: &mut [F; D], col: &[F; D]) {
        type P = <F as Field>::Packing;
        let prefix_len = D - (D % P::WIDTH);
        let (acc_prefix, acc_suffix) = acc.split_at_mut(prefix_len);
        let (col_prefix, col_suffix) = col.split_at(prefix_len);

        for (a, b) in P::pack_slice_mut(acc_prefix)
            .iter_mut()
            .zip(P::pack_slice(col_prefix).iter())
        {
            *a += *b;
        }
        for (a, &b) in acc_suffix.iter_mut().zip(col_suffix.iter()) {
            *a += b;
        }
    }

    #[inline]
    fn acc_sub_assign(acc: &mut [F; D], col: &[F; D]) {
        type P = <F as Field>::Packing;
        let prefix_len = D - (D % P::WIDTH);
        let (acc_prefix, acc_suffix) = acc.split_at_mut(prefix_len);
        let (col_prefix, col_suffix) = col.split_at(prefix_len);

        for (a, b) in P::pack_slice_mut(acc_prefix)
            .iter_mut()
            .zip(P::pack_slice(col_prefix).iter())
        {
            *a -= *b;
        }
        for (a, &b) in acc_suffix.iter_mut().zip(col_suffix.iter()) {
            *a -= b;
        }
    }

    #[inline]
    fn acc_mul_add_assign(acc: &mut [F; D], col: &[F; D], scalar: F) {
        type P = <F as Field>::Packing;
        let prefix_len = D - (D % P::WIDTH);
        let (acc_prefix, acc_suffix) = acc.split_at_mut(prefix_len);
        let (col_prefix, col_suffix) = col.split_at(prefix_len);
        let scalar_p: P = scalar.into();

        for (a, b) in P::pack_slice_mut(acc_prefix)
            .iter_mut()
            .zip(P::pack_slice(col_prefix).iter())
        {
            *a += *b * scalar_p;
        }
        for (a, &b) in acc_suffix.iter_mut().zip(col_suffix.iter()) {
            *a += b * scalar;
        }
    }

    let (kappa, acc) = match &pp_access {
        PpAccess::Loaded { pp } => {
            let kappa = pp.kappa;
            let process_col = |mut st: Acc, col: usize| -> Acc {
                if st.err.is_some() {
                    return st;
                }

                // Decompose the column's D entries into balanced base-b digits for each DEC child.
                for rho in 0..D {
                    let u = z_rows[rho][col].as_canonical_u64() as u128;
                    if B_u <= i64::MAX as u128 {
                        let val_opt: Option<i64> = if u < B_u {
                            Some(u as i64)
                        } else if p.checked_sub(u).map(|w| w < B_u).unwrap_or(false) {
                            Some(-((p - u) as i64))
                        } else {
                            None
                        };
                        let mut v = match val_opt {
                            Some(v) => v,
                            None => {
                                st.err = Some(format!(
                                    "DEC split: Z_mix[{},{}] is out of range for k_rho={}, b={}",
                                    rho, col, k_dec, params.b
                                ));
                                return st;
                            }
                        };
                        for i in 0..k_dec {
                            if v == 0 {
                                st.digits[i * D + rho] = 0;
                                continue;
                            }
                            let (r_i, q) = balanced_divrem_i64(v, b_i64);
                            if r_i != 0 {
                                st.any_nonzero[i] = true;
                            }
                            st.digits[i * D + rho] = r_i as i32;
                            v = q;
                        }
                        if v != 0 {
                            st.err = Some(format!(
                                "DEC split: Z_mix[{},{}] needs more than k_rho={} digits in base b={}",
                                rho, col, k_dec, params.b
                            ));
                            return st;
                        }
                    } else {
                        let val_opt: Option<i128> = if u < B_u {
                            Some(u as i128)
                        } else if p.checked_sub(u).map(|w| w < B_u).unwrap_or(false) {
                            Some(-((p - u) as i128))
                        } else {
                            None
                        };
                        let mut v = match val_opt {
                            Some(v) => v,
                            None => {
                                st.err = Some(format!(
                                    "DEC split: Z_mix[{},{}] is out of range for k_rho={}, b={}",
                                    rho, col, k_dec, params.b
                                ));
                                return st;
                            }
                        };
                        for i in 0..k_dec {
                            if v == 0 {
                                st.digits[i * D + rho] = 0;
                                continue;
                            }
                            let (r_i, q) = balanced_divrem_i128(v, b_i128);
                            if r_i != 0 {
                                st.any_nonzero[i] = true;
                            }
                            st.digits[i * D + rho] = r_i as i32;
                            v = q;
                        }
                        if v != 0 {
                            st.err = Some(format!(
                                "DEC split: Z_mix[{},{}] needs more than k_rho={} digits in base b={}",
                                rho, col, k_dec, params.b
                            ));
                            return st;
                        }
                    }
                }

                // vj[col] := M_j^T · χ_r (compute per column to avoid materializing all vjs).
                match &vjs_access {
                    VjsAccess::Dense(vjs) => {
                        for j in 0..t_mats {
                            st.vj[j] = vjs[j][col];
                        }
                    }
                    VjsAccess::Sparse { cap, cache } => {
                        for j in 0..t_mats {
                            st.vj[j] = if let Some(csc) = cache.csc(j) {
                                let mut sum = K::ZERO;
                                let s = csc.col_ptr[col];
                                let e = csc.col_ptr[col + 1];
                                for k in s..e {
                                    let r = csc.row_idx[k];
                                    if r < n_eff {
                                        sum += K::from(csc.vals[k]) * chi_r[r];
                                    }
                                }
                                sum
                            } else if col < *cap {
                                chi_r[col]
                            } else {
                                K::ZERO
                            };
                        }
                    }
                }

                // y_(i,j)[rho] += Z_i[rho,col] * vj[col]
                for i in 0..k_dec {
                    let y_base = i * t_mats;
                    for rho in 0..D {
                        let digit = st.digits[i * D + rho];
                        if digit == 0 {
                            continue;
                        }
                        for j in 0..t_mats {
                            let vj = st.vj[j];
                            if vj != K::ZERO {
                                match digit {
                                    1 => st.y[y_base + j][rho] += vj,
                                    -1 => st.y[y_base + j][rho] -= vj,
                                    _ => st.y[y_base + j][rho] += vj.scale_base(f_from_i64(digit as i64)),
                                }
                            }
                        }
                    }
                }

                // y_zcol_i[rho] += Z_i[rho,col] * χ_{s_col}[col] (optional).
                if !chi_s.is_empty() {
                    let w_col = chi_s[col];
                    if w_col != K::ZERO {
                        for i in 0..k_dec {
                            for rho in 0..D {
                                let digit = st.digits[i * D + rho];
                                if digit == 0 {
                                    continue;
                                }
                                match digit {
                                    1 => st.y_zcol[i][rho] += w_col,
                                    -1 => st.y_zcol[i][rho] -= w_col,
                                    _ => st.y_zcol[i][rho] += w_col.scale_base(f_from_i64(digit as i64)),
                                }
                            }
                        }
                    }
                }

                // Commitment accumulators per digit.
                for kr in 0..kappa {
                    let mut rot_col = neo_math::ring::cf(pp.m_rows[kr][col]);
                    for rho in 0..D {
                        for i in 0..k_dec {
                            let digit = st.digits[i * D + rho];
                            if digit == 0 {
                                continue;
                            }
                            let acc = &mut st.commit[i * kappa + kr];
                            match digit {
                                1 => acc_add_assign(acc, &rot_col),
                                -1 => acc_sub_assign(acc, &rot_col),
                                _ => acc_mul_add_assign(acc, &rot_col, f_from_i64(digit as i64)),
                            }
                        }
                        rot_step_phi_81(&rot_col, &mut st.rot_next);
                        core::mem::swap(&mut rot_col, &mut st.rot_next);
                    }
                }

                st
            };

            let acc = {
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                {
                    (0..m)
                        .into_par_iter()
                        .fold(|| Acc::new(k_dec, kappa, t_mats), |st, col| process_col(st, col))
                        .reduce(
                            || Acc::new(k_dec, kappa, t_mats),
                            |mut a, b| {
                                if a.err.is_none() {
                                    a.add_inplace(&b, k_dec, kappa, t_mats);
                                }
                                a
                            },
                        )
                }
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                {
                    let mut st = Acc::new(k_dec, kappa, t_mats);
                    for col in 0..m {
                        st = process_col(st, col);
                    }
                    st
                }
            };
            (kappa, acc)
        }
        PpAccess::Seeded {
            kappa,
            chunk_size,
            chunk_seeds_by_row,
        } => {
            let kappa = *kappa;
            let chunk_size = *chunk_size;
            let num_chunks = (m + chunk_size - 1) / chunk_size;

            let process_chunk = |mut st: Acc, chunk_idx: usize| -> Acc {
                if st.err.is_some() {
                    return st;
                }

                let start = chunk_idx * chunk_size;
                let end = core::cmp::min(m, start + chunk_size);

                let mut rngs: Vec<ChaCha8Rng> = (0..kappa)
                    .map(|kr| ChaCha8Rng::from_seed(chunk_seeds_by_row[kr][chunk_idx]))
                    .collect();

                for col in start..end {
                    // Decompose the column's D entries into balanced base-b digits for each DEC child.
                    for rho in 0..D {
                        let u = z_rows[rho][col].as_canonical_u64() as u128;
                        if B_u <= i64::MAX as u128 {
                            let val_opt: Option<i64> = if u < B_u {
                                Some(u as i64)
                            } else if p.checked_sub(u).map(|w| w < B_u).unwrap_or(false) {
                                Some(-((p - u) as i64))
                            } else {
                                None
                            };
                            let mut v = match val_opt {
                                Some(v) => v,
                                None => {
                                    st.err = Some(format!(
                                        "DEC split: Z_mix[{},{}] is out of range for k_rho={}, b={}",
                                        rho, col, k_dec, params.b
                                    ));
                                    return st;
                                }
                            };
                            for i in 0..k_dec {
                                if v == 0 {
                                    st.digits[i * D + rho] = 0;
                                    continue;
                                }
                                let (r_i, q) = balanced_divrem_i64(v, b_i64);
                                if r_i != 0 {
                                    st.any_nonzero[i] = true;
                                }
                                st.digits[i * D + rho] = r_i as i32;
                                v = q;
                            }
                            if v != 0 {
                                st.err = Some(format!(
                                    "DEC split: Z_mix[{},{}] needs more than k_rho={} digits in base b={}",
                                    rho, col, k_dec, params.b
                                ));
                                return st;
                            }
                        } else {
                            let val_opt: Option<i128> = if u < B_u {
                                Some(u as i128)
                            } else if p.checked_sub(u).map(|w| w < B_u).unwrap_or(false) {
                                Some(-((p - u) as i128))
                            } else {
                                None
                            };
                            let mut v = match val_opt {
                                Some(v) => v,
                                None => {
                                    st.err = Some(format!(
                                        "DEC split: Z_mix[{},{}] is out of range for k_rho={}, b={}",
                                        rho, col, k_dec, params.b
                                    ));
                                    return st;
                                }
                            };
                            for i in 0..k_dec {
                                if v == 0 {
                                    st.digits[i * D + rho] = 0;
                                    continue;
                                }
                                let (r_i, q) = balanced_divrem_i128(v, b_i128);
                                if r_i != 0 {
                                    st.any_nonzero[i] = true;
                                }
                                st.digits[i * D + rho] = r_i as i32;
                                v = q;
                            }
                            if v != 0 {
                                st.err = Some(format!(
                                    "DEC split: Z_mix[{},{}] needs more than k_rho={} digits in base b={}",
                                    rho, col, k_dec, params.b
                                ));
                                return st;
                            }
                        }
                    }

                    // vj[col] := M_j^T · χ_r (compute per column to avoid materializing all vjs).
                    match &vjs_access {
                        VjsAccess::Dense(vjs) => {
                            for j in 0..t_mats {
                                st.vj[j] = vjs[j][col];
                            }
                        }
                        VjsAccess::Sparse { cap, cache } => {
                            for j in 0..t_mats {
                                st.vj[j] = if let Some(csc) = cache.csc(j) {
                                    let mut sum = K::ZERO;
                                    let s = csc.col_ptr[col];
                                    let e = csc.col_ptr[col + 1];
                                    for k in s..e {
                                        let r = csc.row_idx[k];
                                        if r < n_eff {
                                            sum += K::from(csc.vals[k]) * chi_r[r];
                                        }
                                    }
                                    sum
                                } else if col < *cap {
                                    chi_r[col]
                                } else {
                                    K::ZERO
                                };
                            }
                        }
                    }

                    // y_(i,j)[rho] += Z_i[rho,col] * vj[col]
                    for i in 0..k_dec {
                        let y_base = i * t_mats;
                        for rho in 0..D {
                            let digit = st.digits[i * D + rho];
                            if digit == 0 {
                                continue;
                            }
                            for j in 0..t_mats {
                                let vj = st.vj[j];
                                if vj != K::ZERO {
                                    match digit {
                                        1 => st.y[y_base + j][rho] += vj,
                                        -1 => st.y[y_base + j][rho] -= vj,
                                        _ => st.y[y_base + j][rho] += vj.scale_base(f_from_i64(digit as i64)),
                                    }
                                }
                            }
                        }
                    }

                    // y_zcol_i[rho] += Z_i[rho,col] * χ_{s_col}[col] (optional).
                    if !chi_s.is_empty() {
                        let w_col = chi_s[col];
                        if w_col != K::ZERO {
                            for i in 0..k_dec {
                                for rho in 0..D {
                                    let digit = st.digits[i * D + rho];
                                    if digit == 0 {
                                        continue;
                                    }
                                    match digit {
                                        1 => st.y_zcol[i][rho] += w_col,
                                        -1 => st.y_zcol[i][rho] -= w_col,
                                        _ => st.y_zcol[i][rho] += w_col.scale_base(f_from_i64(digit as i64)),
                                    }
                                }
                            }
                        }
                    }

                    // Commitment accumulators per digit.
                    for kr in 0..kappa {
                        let a_kr_col = sample_uniform_rq(&mut rngs[kr]);
                        let mut rot_col = neo_math::ring::cf(a_kr_col);
                        for rho in 0..D {
                            for i in 0..k_dec {
                                let digit = st.digits[i * D + rho];
                                if digit == 0 {
                                    continue;
                                }
                                let acc = &mut st.commit[i * kappa + kr];
                                match digit {
                                    1 => acc_add_assign(acc, &rot_col),
                                    -1 => acc_sub_assign(acc, &rot_col),
                                    _ => acc_mul_add_assign(acc, &rot_col, f_from_i64(digit as i64)),
                                }
                            }
                            rot_step_phi_81(&rot_col, &mut st.rot_next);
                            core::mem::swap(&mut rot_col, &mut st.rot_next);
                        }
                    }
                }

                st
            };

            let acc = {
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                {
                    (0..num_chunks)
                        .into_par_iter()
                        .fold(
                            || Acc::new(k_dec, kappa, t_mats),
                            |st, chunk_idx| process_chunk(st, chunk_idx),
                        )
                        .reduce(
                            || Acc::new(k_dec, kappa, t_mats),
                            |mut a, b| {
                                if a.err.is_none() {
                                    a.add_inplace(&b, k_dec, kappa, t_mats);
                                }
                                a
                            },
                        )
                }
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                {
                    let mut st = Acc::new(k_dec, kappa, t_mats);
                    for chunk_idx in 0..num_chunks {
                        st = process_chunk(st, chunk_idx);
                    }
                    st
                }
            };
            (kappa, acc)
        }
    };

    if let Some(err) = acc.err {
        return Err(PiCcsError::ProtocolError(err));
    }

    // Commitments c_i from accumulated columns.
    let mut child_cs: Vec<Cmt> = Vec::with_capacity(k_dec);
    for i in 0..k_dec {
        if !acc.any_nonzero[i] {
            child_cs.push(Cmt::zeros(D, kappa));
            continue;
        }
        let mut c = Cmt::zeros(D, kappa);
        for kr in 0..kappa {
            c.col_mut(kr).copy_from_slice(&acc.commit[i * kappa + kr]);
        }
        child_cs.push(c);
    }

    // X_i: project first m_in columns from Z_i (small; compute sequentially).
    let m_in = parent.m_in;
    let mut xs_row_major: Vec<Vec<F>> = vec![vec![F::ZERO; D * m_in]; k_dec];
    for col in 0..m_in {
        for rho in 0..D {
            let u = z_rows[rho][col].as_canonical_u64() as u128;
            if B_u <= i64::MAX as u128 {
                let val_opt: Option<i64> = if u < B_u {
                    Some(u as i64)
                } else if p.checked_sub(u).map(|w| w < B_u).unwrap_or(false) {
                    Some(-((p - u) as i64))
                } else {
                    None
                };
                let mut v = val_opt.ok_or_else(|| {
                    PiCcsError::ProtocolError(format!(
                        "DEC split(X): Z_mix[{},{}] out of range for k_rho={}, b={}",
                        rho, col, k_dec, params.b
                    ))
                })?;
                for i in 0..k_dec {
                    if v == 0 {
                        break;
                    }
                    let (r_i, q) = balanced_divrem_i64(v, b_i64);
                    xs_row_major[i][rho * m_in + col] = f_from_i64(r_i);
                    v = q;
                }
                if v != 0 {
                    return Err(PiCcsError::ProtocolError(format!(
                        "DEC split(X): Z_mix[{},{}] needs more than k_rho={} digits in base b={}",
                        rho, col, k_dec, params.b
                    )));
                }
            } else {
                let val_opt: Option<i128> = if u < B_u {
                    Some(u as i128)
                } else if p.checked_sub(u).map(|w| w < B_u).unwrap_or(false) {
                    Some(-((p - u) as i128))
                } else {
                    None
                };
                let mut v = val_opt.ok_or_else(|| {
                    PiCcsError::ProtocolError(format!(
                        "DEC split(X): Z_mix[{},{}] out of range for k_rho={}, b={}",
                        rho, col, k_dec, params.b
                    ))
                })?;
                for i in 0..k_dec {
                    if v == 0 {
                        break;
                    }
                    let (r_i, q) = balanced_divrem_i128(v, b_i128);
                    xs_row_major[i][rho * m_in + col] = f_from_i64(r_i as i64);
                    v = q;
                }
                if v != 0 {
                    return Err(PiCcsError::ProtocolError(format!(
                        "DEC split(X): Z_mix[{},{}] needs more than k_rho={} digits in base b={}",
                        rho, col, k_dec, params.b
                    )));
                }
            }
        }
    }

    let parent_r = parent.r.clone();
    let fold_digest = parent.fold_digest;

    let mut children: Vec<MeInstance<Cmt, F, K>> = Vec::with_capacity(k_dec);
    for i in 0..k_dec {
        let Xi = Mat::from_row_major(D, m_in, xs_row_major[i].clone());
        let mut y_i: Vec<Vec<K>> = Vec::with_capacity(t_mats);
        let mut y_scalars_i: Vec<K> = Vec::with_capacity(t_mats);
        for j in 0..t_mats {
            let mut yj = vec![K::ZERO; d_pad];
            let row = &acc.y[i * t_mats + j];
            for rho in 0..D {
                yj[rho] = row[rho];
            }
            let mut sc = K::ZERO;
            for rho in 0..D {
                sc += yj[rho] * pow_b_k[rho];
            }
            y_i.push(yj);
            y_scalars_i.push(sc);
        }

        let y_zcol = if chi_s.is_empty() {
            Vec::new()
        } else {
            let mut yz = vec![K::ZERO; d_pad];
            let row = &acc.y_zcol[i];
            for rho in 0..D {
                yz[rho] = row[rho];
            }
            yz
        };

        children.push(MeInstance::<Cmt, F, K> {
            c_step_coords: vec![],
            u_offset: 0,
            u_len: 0,
            c: child_cs[i].clone(),
            X: Xi,
            r: parent_r.clone(),
            s_col: parent.s_col.clone(),
            y: y_i,
            y_scalars: y_scalars_i,
            y_zcol,
            m_in,
            fold_digest,
        });
    }

    // Public checks (mirror paper-exact DEC).
    let mut ok_y = true;
    for j in 0..t_mats {
        let mut lhs = vec![K::ZERO; d_pad];
        let mut pow = K::ONE;
        for i in 0..k_dec {
            for t in 0..d_pad {
                lhs[t] += pow * children[i].y[j][t];
            }
            pow *= bK;
        }
        if lhs != parent.y[j] {
            ok_y = false;
            break;
        }
    }

    // y_zcol: column-domain opening must also decompose (when present).
    if ok_y && !chi_s.is_empty() {
        let mut lhs = vec![K::ZERO; d_pad];
        let mut pow = K::ONE;
        for i in 0..k_dec {
            for t in 0..d_pad {
                lhs[t] += pow * children[i].y_zcol[t];
            }
            pow *= bK;
        }
        if lhs != parent.y_zcol {
            ok_y = false;
        }
    }

    let mut lhs_X = Mat::zero(D, m_in, F::ZERO);
    let mut pow = F::ONE;
    for i in 0..k_dec {
        for r in 0..D {
            for c in 0..m_in {
                lhs_X[(r, c)] += pow * children[i].X[(r, c)];
            }
        }
        pow *= bF;
    }
    let ok_X = lhs_X.as_slice() == parent.X.as_slice();

    let ok_c = combine_b_pows(&child_cs, params.b) == parent.c;
    Ok((children, child_cs, ok_y, ok_X, ok_c))
}

pub(crate) fn bind_rlc_inputs(
    tr: &mut Poseidon2Transcript,
    lane: RlcLane,
    step_idx: usize,
    me_inputs: &[MeInstance<Cmt, F, K>],
) -> Result<(), PiCcsError> {
    let lane_scope: &'static [u8] = match lane {
        RlcLane::Main => b"main",
        RlcLane::Val => b"val",
    };

    // v2: binds NC-channel fields (s_col, y_zcol) so RLC challenges depend on the full instance.
    tr.append_message(b"fold/rlc_inputs/v2", lane_scope);
    tr.append_u64s(b"step_idx", &[step_idx as u64]);
    tr.append_u64s(b"me_count", &[me_inputs.len() as u64]);

    for me in me_inputs {
        tr.append_fields(b"c_data", &me.c.data);
        tr.append_u64s(b"m_in", &[me.m_in as u64]);
        tr.append_message(b"me_fold_digest", &me.fold_digest);

        let r_coeffs_per_limb = me.r.first().map(|v| v.as_coeffs().len()).unwrap_or(0);
        tr.append_fields_iter(
            b"r_limb",
            me.r.len()
                .checked_mul(r_coeffs_per_limb)
                .ok_or_else(|| PiCcsError::ProtocolError("r_limb length overflow".into()))?,
            me.r.iter().flat_map(|limb| limb.as_coeffs()),
        );

        tr.append_u64s(b"s_col_len", &[me.s_col.len() as u64]);
        let s_col_coeffs_per_elem = me.s_col.first().map(|v| v.as_coeffs().len()).unwrap_or(0);
        tr.append_fields_iter(
            b"s_col_elem",
            me.s_col
                .len()
                .checked_mul(s_col_coeffs_per_elem)
                .ok_or_else(|| PiCcsError::ProtocolError("s_col_elem length overflow".into()))?,
            me.s_col.iter().flat_map(|sc| sc.as_coeffs()),
        );

        tr.append_u64s(b"y_zcol_len", &[me.y_zcol.len() as u64]);
        let y_zcol_coeffs_per_elem = me.y_zcol.first().map(|v| v.as_coeffs().len()).unwrap_or(0);
        tr.append_fields_iter(
            b"y_zcol_elem",
            me.y_zcol
                .len()
                .checked_mul(y_zcol_coeffs_per_elem)
                .ok_or_else(|| PiCcsError::ProtocolError("y_zcol_elem length overflow".into()))?,
            me.y_zcol.iter().flat_map(|yz| yz.as_coeffs()),
        );

        tr.append_fields(b"X", me.X.as_slice());

        let y_elem_coeffs_per_elem =
            me.y.iter()
                .find_map(|row| row.first())
                .map(|v| v.as_coeffs().len())
                .unwrap_or(0);
        let y_elem_count = me.y.iter().map(Vec::len).sum::<usize>();
        tr.append_fields_iter(
            b"y_elem",
            y_elem_count
                .checked_mul(y_elem_coeffs_per_elem)
                .ok_or_else(|| PiCcsError::ProtocolError("y_elem length overflow".into()))?,
            me.y.iter()
                .flat_map(|row| row.iter().flat_map(|v| v.as_coeffs())),
        );

        let y_scalar_coeffs_per_elem = me
            .y_scalars
            .first()
            .map(|v| v.as_coeffs().len())
            .unwrap_or(0);
        tr.append_fields_iter(
            b"y_scalar",
            me.y_scalars
                .len()
                .checked_mul(y_scalar_coeffs_per_elem)
                .ok_or_else(|| PiCcsError::ProtocolError("y_scalar length overflow".into()))?,
            me.y_scalars.iter().flat_map(|ysc| ysc.as_coeffs()),
        );

        tr.append_u64s(b"c_step_coords_len", &[me.c_step_coords.len() as u64]);
        tr.append_fields(b"c_step_coords", &me.c_step_coords);
        tr.append_u64s(b"u_offset", &[me.u_offset as u64]);
        tr.append_u64s(b"u_len", &[me.u_len as u64]);
    }

    Ok(())
}
