//! Optimized engine implementation for Π_CCS
//!
//! This module contains the optimized implementation of the CCS reduction protocol.
//! It has been refactored from the original `pi_ccs` module structure to allow
//! for better organization and testing against the paper-exact reference implementation.

use neo_math::{F, K, KExtensions};
use neo_transcript::Transcript;
use crate::sumcheck::RoundOracle;
use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance, Mat};
use neo_ajtai::Commitment as Cmt;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use crate::error::PiCcsError;

pub mod context;         
pub mod transcript;      
pub mod precompute;      
pub mod checks;          
pub mod terminal;        
pub mod outputs;         
pub mod sumcheck_driver; 
pub mod oracle;          
pub mod transcript_replay; 

pub mod nc_core;         
pub mod nc_constraints;  
pub mod sparse_matrix;   
pub mod eq_weights;      

// Re-export commonly used public items
pub use oracle::GenericCcsOracle;
pub use transcript::Challenges;
pub use sparse_matrix::{Csr, to_csr};
pub use transcript_replay::{
    TranscriptTail,
    pi_ccs_derive_transcript_tail,
    pi_ccs_derive_transcript_tail_with_me_inputs,
    pi_ccs_derive_transcript_tail_with_me_inputs_and_label,
    pi_ccs_derive_transcript_tail_from_bound_transcript,
    pi_ccs_compute_terminal_claim_r1cs_or_ccs,
};

/// Proof structure for the Π_CCS protocol
#[derive(Debug, Clone)]
pub struct PiCcsProof {
    /// Sumcheck rounds (each round is a vector of polynomial coefficients)
    pub sumcheck_rounds: Vec<Vec<K>>,
    
    /// Initial sum over the Boolean hypercube (optional, can be derived from round 0)
    pub sc_initial_sum: Option<K>,
    
    /// Sumcheck challenges (r' || α' from the sumcheck protocol)
    pub sumcheck_challenges: Vec<K>,
    
    /// Public challenges (α, β, γ)
    pub challenges_public: Challenges,
    
    /// Final running sum after all sumcheck rounds
    pub sumcheck_final: K,
    
    /// Header digest for binding
    pub header_digest: Vec<u8>,
    
    /// Additional proof data (if needed)
    pub _extra: Option<Vec<u8>>,
}

impl PiCcsProof {
    /// Create a new proof
    pub fn new(sumcheck_rounds: Vec<Vec<K>>, sc_initial_sum: Option<K>) -> Self {
        Self {
            sumcheck_rounds,
            sc_initial_sum,
            sumcheck_challenges: Vec::new(),
            challenges_public: Challenges {
                alpha: Vec::new(),
                beta_a: Vec::new(),
                beta_r: Vec::new(),
                gamma: K::ZERO,
            },
            sumcheck_final: K::ZERO,
            header_digest: Vec::new(),
            _extra: None,
        }
    }
}

/// Naive Lagrange interpolation over K to monomial coefficients.
/// Returns coefficients c such that p(x) = Σ c[i] x^i matches (xs, ys).
pub(crate) fn interpolate_univariate(xs: &[K], ys: &[K]) -> Vec<K> {
    assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    let mut coeffs = vec![K::ZERO; n];
    // Build Lagrange basis polynomials ℓ_i(x) and accumulate y_i·ℓ_i(x)
    for i in 0..n {
        // Numerator: prod_{j≠i} (x - x_j)
        let mut numer = vec![K::ZERO; n];
        numer[0] = K::ONE; // degree 0 poly = 1
        let mut cur_deg = 0usize;
        for j in 0..n {
            if i == j { continue; }
            // Multiply numer by (x - x_j)
            let xj = xs[j];
            let mut next = vec![K::ZERO; n];
            for d in 0..=cur_deg {
                // x * numer[d]
                next[d + 1] += numer[d];
                // -x_j * numer[d]
                next[d] += -xj * numer[d];
            }
            numer = next;
            cur_deg += 1;
        }
        // Denominator: prod_{j≠i} (x_i - x_j)
        let mut denom = K::ONE;
        for j in 0..n { if i != j { denom *= xs[i] - xs[j]; } }
        let scale = ys[i] * denom.inv();
        for d in 0..=cur_deg { coeffs[d] += scale * numer[d]; }
    }
    coeffs
}

/// Stub for pi_ccs_prove function
/// TODO: This needs to be properly implemented based on the original code
pub fn pi_ccs_prove<L>(
    tr: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    mcs_witnesses: &[McsWitness<F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError>
where
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
{
    // 0) Dims + transcript binding
    let dims = context::build_dims_and_policy(params, s)?;
    transcript::bind_header_and_instances(tr, params, s, mcs_list, dims.ell, dims.d_sc, 0)?;
    transcript::bind_me_inputs(tr, me_inputs)?;

    // 1) Sample public challenges
    let ch = transcript::sample_challenges(tr, dims.ell_d, dims.ell)?;

    // 2) Precompute sparse matrices and instance caches
    let mats_csr: Vec<_> = s
        .matrices
        .iter()
        .map(|m| sparse_matrix::to_csr(m, s.n, s.m))
        .collect();
    let insts = precompute::prepare_instances(s, params, mcs_list, mcs_witnesses, &mats_csr, log)?;

    let mle_partials = precompute::build_mle_partials_first_inst(s, dims.ell_n, &insts)?;
    let beta_block = precompute::precompute_beta_block(
        s,
        params,
        &insts,
        mcs_witnesses,
        me_witnesses,
        &ch,
        dims.ell_d,
        dims.ell_n,
    )?;
    let eval_row_partial = precompute::precompute_eval_row_partial(
        s,
        me_witnesses,
        &ch,
        mcs_list.len() + me_inputs.len(),
        dims.ell_n,
    )?;

    // Equality tables (full tensors) for gates
    let mut w_beta_a_partial = neo_ccs::utils::tensor_point::<K>(&ch.beta_a);
    let mut w_alpha_a_partial = neo_ccs::utils::tensor_point::<K>(&ch.alpha);
    let mut w_beta_r_partial = neo_ccs::utils::tensor_point::<K>(&ch.beta_r);
    // For Eval row gate, use r from ME inputs (if any) or a single zero to disable
    let mut w_eval_r_partial = if let Some(first) = me_inputs.get(0) {
        neo_ccs::utils::tensor_point::<K>(&first.r)
    } else {
        vec![K::ZERO]
    };

    // Padding to exact powers of two
    w_beta_a_partial = precompute::pad_to_pow2_k(w_beta_a_partial, dims.ell_d)?;
    w_alpha_a_partial = precompute::pad_to_pow2_k(w_alpha_a_partial, dims.ell_d)?;
    w_beta_r_partial = precompute::pad_to_pow2_k(w_beta_r_partial, dims.ell_n)?;
    if w_eval_r_partial.len() > 1 {
        w_eval_r_partial = precompute::pad_to_pow2_k(w_eval_r_partial, dims.ell_n)?;
    }

    // NC Ajtai rows (full) and gamma powers
    let mut nc_y_matrices = precompute::precompute_nc_full_rows(s, mcs_witnesses, me_witnesses, dims.ell_n)?;
    let mut nc_row_gamma_pows = Vec::with_capacity(mcs_list.len() + me_witnesses.len());
    {
        let mut g = ch.gamma;
        for _ in 0..(mcs_list.len() + me_witnesses.len()) {
            nc_row_gamma_pows.push(g);
            g *= ch.gamma;
        }
    }

    // Compute initial sum claim
    let initial_sum = precompute::compute_initial_sum_components(
        &beta_block,
        me_inputs.get(0).map(|mi| mi.r.as_slice()),
        &eval_row_partial,
    )?;

    // 3) Instantiate oracle
    let me_offset = mcs_witnesses.len();
    let z_witnesses: Vec<&neo_ccs::Mat<F>> = mcs_witnesses
        .iter()
        .map(|w| &w.Z)
        .chain(me_witnesses.iter())
        .collect();
    let csr_m1 = mats_csr
        .get(0)
        .ok_or_else(|| PiCcsError::InvalidInput("no M_1 matrix".into()))?;
    let mut oracle = crate::optimized_engine::oracle::GenericCcsOracle::new(
        s,
        mle_partials,
        w_beta_a_partial,
        w_alpha_a_partial,
        w_beta_r_partial,
        w_eval_r_partial,
        eval_row_partial,
        z_witnesses,
        csr_m1,
        &mats_csr,
        nc_y_matrices.drain(..).collect(),
        nc_row_gamma_pows,
        ch.gamma,
        mcs_list.len() + me_inputs.len(),
        params.b,
        dims.ell_d,
        dims.ell_n,
        dims.d_sc,
        me_offset,
        initial_sum,
        beta_block.f_at_beta_r,
        beta_block.nc_sum_hypercube,
    );

    // 4) Sumcheck rounds: interpolate true univariate polynomials per round
    // Bind initial sum to transcript as done by verifier
    tr.append_fields(b"sumcheck/initial_sum", &initial_sum.as_coeffs());
    let mut running_sum = initial_sum;
    let mut sumcheck_rounds: Vec<Vec<K>> = Vec::with_capacity(oracle.num_rounds());
    let mut sumcheck_chals: Vec<K> = Vec::with_capacity(oracle.num_rounds());

    for _round_idx in 0..oracle.num_rounds() {
        let deg = oracle.degree_bound();
        // Sample at deg+1 distinct points: 0,1,2,...,deg
        let xs: Vec<K> = (0..=deg)
            .map(|t| K::from(F::from_u64(t as u64)))
            .collect();
        let ys = oracle.evals_at(&xs);
        // Round-0 invariant check: p(0)+p(1) must equal running_sum
        let y0 = ys[0];
        let y1 = ys[1 % ys.len()];
        if y0 + y1 != running_sum {
            return Err(PiCcsError::SumcheckError(
                "round invariant failed: p(0)+p(1) ≠ running_sum".into(),
            ));
        }
        let coeffs = interpolate_univariate(&xs, &ys);
        // Append coeffs to transcript (same framing as verifier)
        for &c in &coeffs {
            tr.append_fields(b"sumcheck/round/coeff", &c.as_coeffs());
        }
        // Sample challenge for this round and update running sum
        let c0 = tr.challenge_field(b"sumcheck/challenge/0");
        let c1 = tr.challenge_field(b"sumcheck/challenge/1");
        let r_i = neo_math::from_complex(c0, c1);
        sumcheck_chals.push(r_i);
        // Horner eval with our coeffs
        let mut val = K::ZERO;
        for &c in coeffs.iter().rev() {
            val = val * r_i + c;
        }
        running_sum = val;
        oracle.fold(r_i);
        sumcheck_rounds.push(coeffs);
    }

    // 5) Capture fold digest and build outputs at r′ (row) using r from transcript tail
    let fold_digest = tr.digest32();
    let (r_prime, _alpha_prime) = sumcheck_chals.split_at(dims.ell_n);

    let out_me = crate::optimized_engine::outputs::build_me_outputs(
        tr,
        s,
        params,
        &mats_csr,
        &insts,
        me_inputs,
        me_witnesses,
        r_prime,
        dims.ell_d,
        fold_digest,
        log,
    )?;

    let mut proof = PiCcsProof::new(sumcheck_rounds, Some(initial_sum));
    proof.sumcheck_challenges = sumcheck_chals;
    proof.challenges_public = ch.clone();
    proof.sumcheck_final = running_sum;
    proof.header_digest = fold_digest.to_vec();

    Ok((out_me, proof))
}

/// Stub for pi_ccs_prove_simple function (k=1 case, no ME inputs)
/// TODO: This needs to be properly implemented based on the original code
pub fn pi_ccs_prove_simple<L>(
    tr: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    mcs_witnesses: &[McsWitness<F>],
    log: &L,
) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError>
where
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
{
    pi_ccs_prove(tr, params, s, mcs_list, mcs_witnesses, &[], &[], log)
}

/// Stub for pi_ccs_verify function
/// TODO: This needs to be properly implemented based on the original code
pub fn pi_ccs_verify(
    tr: &mut Poseidon2Transcript,
    params: &neo_params::NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_outputs: &[MeInstance<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    // 0) Bind transcript and sample challenges exactly like prover
    let dims = context::build_dims_and_policy(params, s)?;
    transcript::bind_header_and_instances(tr, params, s, mcs_list, dims.ell, dims.d_sc, 0)?;
    transcript::bind_me_inputs(tr, me_inputs)?;
    let ch = transcript::sample_challenges(tr, dims.ell_d, dims.ell)?;

    // 1) Bind initial sum and verify rounds to derive r′||α′
    let claimed_initial = match proof.sc_initial_sum {
        Some(x) => x,
        None => {
            if let Some(round0) = proof.sumcheck_rounds.get(0) {
                use crate::sumcheck::poly_eval_k;
                poly_eval_k(round0, K::ZERO) + poly_eval_k(round0, K::ONE)
            } else {
                K::ZERO
            }
        }
    };
    tr.append_fields(b"sumcheck/initial_sum", &claimed_initial.as_coeffs());
    let (r_all, running_sum, ok) = crate::sumcheck::verify_sumcheck_rounds(
        tr,
        dims.d_sc,
        claimed_initial,
        &proof.sumcheck_rounds,
    );
    if !ok { return Err(PiCcsError::SumcheckError("rounds invalid".into())); }

    // 2) Terminal identity check using optimized RHS assembly
    let (r_prime, alpha_prime) = r_all.split_at(dims.ell_n);
    let rhs = crate::optimized_engine::terminal::rhs_Q_apr(
        s,
        &ch,
        r_prime,
        alpha_prime,
        mcs_list,
        me_inputs,
        me_outputs,
        params,
    )?;

    Ok(running_sum == rhs)
}
