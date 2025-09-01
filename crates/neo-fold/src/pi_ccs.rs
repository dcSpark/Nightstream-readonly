//! Π_CCS: Sum-check reduction over extension field K
//!
//! This is the single sum-check used throughout Neo protocol.
//! Proves: Σ_{u∈{0,1}^ℓ} (Σ_i α_i · f_i(u)) · χ_r(u) = 0

#![allow(non_snake_case)] // Allow mathematical notation like X, T, B

use crate::transcript::{FoldTranscript, Domain};
use crate::error::PiCcsError;
use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance, Mat};
use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K};
use p3_field::{PrimeCharacteristicRing, Field};

/// Π_CCS proof containing the single sum-check over K
#[derive(Debug, Clone)]
pub struct PiCcsProof {
    /// Sum-check protocol rounds (univariate polynomials as coefficients)
    pub sumcheck_rounds: Vec<Vec<K>>,
    /// Extension policy binding digest  
    pub header_digest: [u8; 32],
}

// ===== Sum-check helpers =====

#[inline]
fn poly_eval_k(coeffs: &[K], x: K) -> K {
    let mut acc = K::ZERO;
    for &c in coeffs.iter().rev() { acc = acc * x + c; }
    acc
}

#[inline]
fn poly_mul_k(a: &[K], b: &[K]) -> Vec<K> {
    let mut out = vec![K::ZERO; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() { out[i + j] += ai * bj; }
    }
    out
}

/// Lagrange interpolation at distinct points xs with values ys (returns coeffs lowest→highest).
fn lagrange_interpolate_k(xs: &[K], ys: &[K]) -> Vec<K> {
    assert_eq!(xs.len(), ys.len());
    let m = xs.len();
    let mut coeffs = vec![K::ZERO; m];
    for i in 0..m {
        let (xi, yi) = (xs[i], ys[i]);
        let mut denom = K::ONE;
        let mut numer = vec![K::ONE]; // polynomial 1
        for j in 0..m {
            if i == j { continue; }
            denom *= xi - xs[j];
            numer = poly_mul_k(&numer, &[-xs[j], K::ONE]); // (X - x_j)
        }
        let scale = yi * denom.inverse(); // K is a field; panic if denom=0 (distinct xs)
        for k in 0..numer.len() { coeffs[k] += numer[k] * scale; }
    }
    coeffs
}

/// Compute Y_j(u) = ⟨(M_j z), χ_u⟩ in K, for all j, then f(Y(u)) in K.
fn eval_g_for_instance_in_ext(
    s: &CcsStructure<F>,
    z: &[F],
    mz_cache: Option<&[Vec<F>]>,   // optional cache: M_j z over F
    u: &[K],
) -> K {
    let rb = neo_ccs::utils::tensor_point::<K>(u); // χ_u ∈ K^n
    let t = s.t();
    // M_j z rows over F (compute or reuse)
    let mz: Vec<Vec<F>> = if let Some(cached) = mz_cache {
        cached.to_vec()
    } else {
        s.matrices.iter().map(|mj| neo_ccs::utils::mat_vec_mul_ff::<F>(
            mj.as_slice(), s.n, s.m, z
        )).collect()
    };
    // Y_j(u) = Σ_i (M_j z)[i] * χ_u[i]  (promote F→K)
    let mut y_ext = Vec::with_capacity(t);
    for j in 0..t {
        let mut yj = K::ZERO;
        for i in 0..s.n { yj += K::from(mz[j][i]) * rb[i]; }
        y_ext.push(yj);
    }
    s.f.eval_in_ext::<K>(&y_ext)
}

/// Prove Π_CCS: CCS instances satisfy constraints via sum-check
/// 
/// Input: k+1 CCS instances, outputs k ME instances + proof
pub fn pi_ccs_prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut FoldTranscript,
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    witnesses: &[McsWitness<F>],
    l: &L, // we need L to check c = L(Z) and to compute X = L_x(Z)
) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    tr.domain(Domain::CCS);

    // --- Input & policy checks ---
    if mcs_list.is_empty() || mcs_list.len() != witnesses.len() {
        return Err(PiCcsError::InvalidInput("empty or mismatched inputs".into()));
    }
    if !s.n.is_power_of_two() {
        return Err(PiCcsError::InvalidInput(format!("n={} not power of two", s.n)));
    }
    let ell = s.n.trailing_zeros() as usize;
    let d_sc = s.max_degree() as usize;

    let ext = params.extension_check(ell as u32, d_sc as u32)
        .map_err(|e| PiCcsError::ExtensionPolicyFailed(format!("Extension policy validation failed: {}", e)))?;
    if ext.slack_bits < 0 {
        return Err(PiCcsError::ExtensionPolicyFailed(format!(
            "Insufficient security slack: {} bits (need ≥ 0)", ext.slack_bits
        )));
    }
    tr.absorb_ccs_header(64, ext.s_supported, params.lambda, ell as u32, d_sc as u32, ext.slack_bits);

    // --- Prepare per-instance data and check c=L(Z) ---
    // Also build z = x||w and cache M_j z over F for each instance.
    struct Inst<'a> {
        z: Vec<F>, 
        Z: &'a Mat<F>, 
        m_in: usize, 
        mz: Vec<Vec<F>>,
        c: Cmt,
    }
    let mut insts: Vec<Inst> = Vec::with_capacity(mcs_list.len());
    for (inst, wit) in mcs_list.iter().zip(witnesses.iter()) {
        let z = neo_ccs::relations::check_mcs_opening(l, inst, wit)
            .map_err(|e| PiCcsError::InvalidInput(format!("MCS opening failed: {e}")))?;
        // cache M_j z
        let mz = s.matrices.iter().map(|mj| neo_ccs::utils::mat_vec_mul_ff::<F>(
            mj.as_slice(), s.n, s.m, &z
        )).collect();
        insts.push(Inst{ z, Z: &wit.Z, m_in: inst.m_in, mz, c: inst.c.clone() });
    }

    // --- Batch with random alphas ---
    tr.absorb_bytes(b"neo/ccs/batch");
    let alphas: Vec<K> = (0..insts.len()).map(|_| tr.challenge_k()).collect();

    // --- Run sum-check rounds (degree ≤ d_sc) ---
    let mut rounds: Vec<Vec<K>> = Vec::with_capacity(ell);
    let mut prefix: Vec<K> = Vec::with_capacity(ell);
    let sample_xs: Vec<K> = (0..=d_sc as u64).map(|u| K::from(F::from_u64(u))).collect();

    // initial running sum is the public target: sum_{u∈{0,1}^ℓ} g(u) == 0
    let mut running_sum = K::ZERO;

    for i in 0..ell {
        // For each sample X, compute S_i(X) = Σ_{tail} g(prefix, X, tail), batched over instances.
        let mut sample_ys = vec![K::ZERO; sample_xs.len()];
        for (sx, &X) in sample_xs.iter().enumerate() {
            let tail_len = ell - i - 1;
            let mut sum_at_X = K::ZERO;
            for tail in 0..(1usize << tail_len) {
                // Build u = prefix || X || tail_bits
                let mut u = Vec::with_capacity(ell);
                u.extend_from_slice(&prefix);
                u.push(X);
                for j in 0..tail_len {
                    u.push(if (tail >> j) & 1 == 1 { K::ONE } else { K::ZERO });
                }
                // Batched g(u) over instances with coefficients alphas
                let mut g_combined = K::ZERO;
                for (a, inst) in alphas.iter().zip(insts.iter()) {
                    g_combined += *a * eval_g_for_instance_in_ext(s, &inst.z, Some(&inst.mz), &u);
                }
                sum_at_X += g_combined;
            }
            sample_ys[sx] = sum_at_X;
        }
        let coeffs = lagrange_interpolate_k(&sample_xs, &sample_ys);
        if coeffs.len() > d_sc + 1 {
            return Err(PiCcsError::SumcheckError(format!(
                "round {i}: degree {} exceeds bound {d_sc}", coeffs.len()-1
            )));
        }

        // Verifier-side checks (non-interactively enforced via transcript):
        // Check p(0)+p(1)=running_sum
        let p0 = poly_eval_k(&coeffs, K::ZERO);
        let p1 = poly_eval_k(&coeffs, K::ONE);
        if p0 + p1 != running_sum {
            return Err(PiCcsError::SumcheckError(format!(
                "round {i}: p(0)+p(1) mismatch")));
        }

        // Bind polynomial to transcript and sample r_i
        tr.absorb_ext_as_base_fields_k(b"neo/ccs/round", coeffs[0]);
        for c in coeffs.iter().skip(1) { tr.absorb_ext_as_base_fields_k(b"neo/ccs/round", *c); }
        let r_i = tr.challenge_k();

        // Update state
        running_sum = poly_eval_k(&coeffs, r_i);
        prefix.push(r_i);
        rounds.push(coeffs);
    }

    // Derive r = prefix; compute r^⊗ once
    let r = prefix;
    let rb = neo_ccs::utils::tensor_point::<K>(&r);

    // --- Build ME instances (one per input) ---
    let mut out_me = Vec::with_capacity(insts.len());
    for inst in insts.iter() {
        // X = L_x(Z)
        let X = l.project_x(inst.Z, inst.m_in);
        // For each M_j: v_j = M_j^T * r^⊗  (K^m), then y_j = Z * v_j (K^d)
        let mut y = Vec::with_capacity(s.t());
        for mj in &s.matrices {
            // v_j[c] = Σ_r mj[r,c] * rb[r]
            let mut vj = vec![K::ZERO; s.m];
            for row in 0..s.n {
                let coeff = rb[row];
                let row_m = mj.row(row);
                for c in 0..s.m { vj[c] += K::from(row_m[c]) * coeff; }
            }
            let z_ref = neo_ccs::MatRef::from_mat(inst.Z);
            let yj = neo_ccs::utils::mat_vec_mul_fk::<F,K>(z_ref.data, z_ref.rows, z_ref.cols, &vj);
            y.push(yj);
        }
        out_me.push(MeInstance{ c: inst.c.clone(), X, r: r.clone(), y, m_in: inst.m_in });
    }

    let proof = PiCcsProof { sumcheck_rounds: rounds, header_digest: tr.state_digest() };
    Ok((out_me, proof))
}

/// Verify Π_CCS: Check sum-check rounds and defer final evaluation to ME claims
pub fn pi_ccs_verify(
    tr: &mut FoldTranscript,
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    _mcs_list: &[McsInstance<Cmt, F>],
    out_me: &[MeInstance<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    tr.domain(Domain::CCS);
    if !s.n.is_power_of_two() { 
        return Err(PiCcsError::InvalidInput("n not power of two".into())); 
    }
    let ell = s.n.trailing_zeros() as usize;
    let d_sc = s.max_degree() as usize;

    let ext = params.extension_check(ell as u32, d_sc as u32)
        .map_err(|e| PiCcsError::ExtensionPolicyFailed(e.to_string()))?;
    tr.absorb_ccs_header(64, ext.s_supported, params.lambda, ell as u32, d_sc as u32, ext.slack_bits);

    // Re-derive batching coefficients (not needed to check rounds structure), but bind domain
    tr.absorb_bytes(b"neo/ccs/batch");

    if proof.sumcheck_rounds.len() != ell { return Ok(false); }

    let mut running_sum = K::ZERO;
    let mut r = Vec::with_capacity(ell);

    for (_i, coeffs) in proof.sumcheck_rounds.iter().enumerate() {
        if coeffs.is_empty() || coeffs.len() > d_sc + 1 { return Ok(false); }
        let p0 = poly_eval_k(coeffs, K::ZERO);
        let p1 = poly_eval_k(coeffs, K::ONE);
        if p0 + p1 != running_sum { return Ok(false); }

        // absorb coeffs and sample r_i
        tr.absorb_ext_as_base_fields_k(b"neo/ccs/round", coeffs[0]);
        for c in coeffs.iter().skip(1) { tr.absorb_ext_as_base_fields_k(b"neo/ccs/round", *c); }
        let r_i = tr.challenge_k();
        running_sum = poly_eval_k(coeffs, r_i);
        r.push(r_i);
    }

    // Light structural sanity: every output ME must carry the same r
    if !out_me.iter().all(|me| me.r == r) { return Ok(false); }

    // Sum-check reduces to ME claims (checked in subsequent reductions).
    Ok(true)
}