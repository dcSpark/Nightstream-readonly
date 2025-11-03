//! Paper-exact prove implementation for PiCcsEngine.
//!
//! This module contains the prove logic for the paper-exact engine,
//! which runs the sumcheck protocol using the paper-exact oracle
//! to evaluate true per-round polynomials.

#![allow(non_snake_case)]

use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance, Mat};
use neo_ajtai::Commitment as Cmt;
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_math::{F, K};
use neo_transcript::Transcript;
use neo_math::KExtensions;
use p3_field::PrimeCharacteristicRing;
use crate::sumcheck::RoundOracle;
use crate::error::PiCcsError;
use crate::optimized_engine::PiCcsProof;

/// Paper-exact prove implementation.
///
/// This function runs the sumcheck protocol using the paper-exact oracle,
/// which directly evaluates the polynomial Q over the Boolean hypercube
/// without optimizations.
pub fn paper_exact_prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    mcs_witnesses: &[McsWitness<F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    // Dims + transcript binding
    let dims = crate::optimized_engine::context::build_dims_and_policy(params, s)?;
    crate::optimized_engine::transcript::bind_header_and_instances(
        tr, params, s, mcs_list, dims.ell, dims.d_sc, 0,
    )?;
    crate::optimized_engine::transcript::bind_me_inputs(tr, me_inputs)?;

    // Sample challenges
    let ch = crate::optimized_engine::transcript::sample_challenges(tr, dims.ell_d, dims.ell)?;

    // Validate ME input r (if provided)
    for (idx, me) in me_inputs.iter().enumerate() {
        if me.r.len() != dims.ell_n {
            return Err(PiCcsError::InvalidInput(format!(
                "ME input r length mismatch at accumulator #{}: expected ell_n = {}, got {}",
                idx, dims.ell_n, me.r.len()
            )));
        }
    }

    // Initial sum: literal hypercube sum (paper-exact)
    let initial_sum = crate::paper_exact_engine::sum_q_over_hypercube_paper_exact(
        s,
        params,
        mcs_witnesses,
        me_witnesses,
        &ch,
        dims.ell_d,
        dims.ell_n,
        me_inputs.get(0).map(|mi| mi.r.as_slice()),
    );

    // Bind initial sum to transcript
    tr.append_fields(b"sumcheck/initial_sum", &initial_sum.as_coeffs());

    // Paper-exact oracle to evaluate true per-round polynomials
    let mut oracle = crate::paper_exact_engine::oracle::PaperExactOracle::new(
        s,
        params,
        mcs_witnesses,
        me_witnesses,
        ch.clone(),
        dims.ell_d,
        dims.ell_n,
        dims.d_sc,
        me_inputs.get(0).map(|mi| mi.r.as_slice()),
    );

    let mut running_sum = initial_sum;
    let mut sumcheck_rounds: Vec<Vec<K>> = Vec::with_capacity(oracle.num_rounds());
    let mut sumcheck_chals: Vec<K> = Vec::with_capacity(oracle.num_rounds());

    for _ in 0..oracle.num_rounds() {
        let deg = oracle.degree_bound();
        let xs: Vec<K> = (0..=deg).map(|t| K::from(F::from_u64(t as u64))).collect();
        let ys = oracle.evals_at(&xs);
        if ys[0] + ys[1] != running_sum {
            return Err(PiCcsError::SumcheckError(
                "round invariant failed: p(0)+p(1) ≠ running_sum (paper-exact)".into(),
            ));
        }
        let coeffs = crate::optimized_engine::interpolate_univariate(&xs, &ys);
        for &c in &coeffs {
            tr.append_fields(b"sumcheck/round/coeff", &c.as_coeffs());
        }
        let c0 = tr.challenge_field(b"sumcheck/challenge/0");
        let c1 = tr.challenge_field(b"sumcheck/challenge/1");
        let r_i = neo_math::from_complex(c0, c1);
        sumcheck_chals.push(r_i);
        // Horner eval
        let mut val = K::ZERO;
        for &c in coeffs.iter().rev() { val = val * r_i + c; }
        running_sum = val;
        oracle.fold(r_i);
        sumcheck_rounds.push(coeffs);
    }

    // Build outputs literally at r′ using paper-exact helper
    let fold_digest = tr.digest32();
    let (r_prime, _alpha_prime) = sumcheck_chals.split_at(dims.ell_n);
    let out_me = crate::paper_exact_engine::build_me_outputs_paper_exact(
        s,
        params,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        r_prime,
        dims.ell_d,
        fold_digest,
        log,
    );

    let mut proof = PiCcsProof::new(sumcheck_rounds, Some(initial_sum));
    proof.sumcheck_challenges = sumcheck_chals;
    proof.challenges_public = ch;
    proof.sumcheck_final = running_sum;
    proof.header_digest = fold_digest.to_vec();

    Ok((out_me, proof))
}

