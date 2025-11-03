//! Cross-check engine wrapper for validation during development.
//!
//! This module provides the CrossCheckEngine, which runs the optimized engine
//! and validates key identities against paper-exact helpers. This is useful
//! for debugging and ensuring correctness.

#![allow(non_snake_case)]

use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance, Mat};
use neo_ajtai::Commitment as Cmt;
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_math::{F, K};
use crate::error::PiCcsError;
use crate::engines::optimized_engine::PiCcsProof;
use crate::engines::PiCcsEngine;

/// Runtime cross-check configuration.
#[derive(Clone, Debug, Default)]
pub struct CrosscheckCfg {
    pub fail_fast: bool,
    pub initial_sum: bool,
    pub per_round: bool,
    pub terminal: bool,
    pub outputs: bool,
}

/// Cross-check engine wrapper that runs the optimized engine
/// and validates key identities against paper-exact helpers.
#[derive(Clone, Debug)]
pub struct CrossCheckEngine<I, R> {
    pub inner: I,      // Optimized engine
    pub ref_oracle: R, // PaperExact engine (reference)
    pub cfg: CrosscheckCfg,
}

/// Implementation of prove logic for cross-checking.
///
/// # How Crosscheck Works
///
/// 1. **Runs the optimized engine** to produce a proof
///    - Any transcript dumps you see are from the optimized prover
///    - The proof contains sumcheck rounds and a final sumcheck value
///
/// 2. **Extracts challenges** by replaying the optimized engine's transcript
///    - This gives us the random points (r', α') used in the proof
///
/// 3. **Validates specific computations** by comparing:
///    - Optimized engine's formulas (efficient, from public outputs)
///    - Paper-exact engine's formulas (direct, from witnesses)
///
/// The paper-exact engine never produces a full proof - it only computes
/// specific values for comparison against the optimized engine's proof.
pub fn crosscheck_prove<I, L>(
    inner: &I,
    cfg: &CrosscheckCfg,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    mcs_witnesses: &[McsWitness<F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError>
where
    I: PiCcsEngine,
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
{
    // 1) Run optimized path
    let (out_me_opt, proof) = inner
        .prove(tr, params, s, mcs_list, mcs_witnesses, me_inputs, me_witnesses, log)?;

    // 2) Deterministically reconstruct the transcript tail to obtain r' || α'
    let tail = crate::engines::optimized_engine::transcript_replay::pi_ccs_derive_transcript_tail_with_me_inputs(
        params, s, mcs_list, me_inputs, &proof,
    )?;

    // Dimension split for (r', α')
    let dims = crate::engines::optimized_engine::context::build_dims_and_policy(params, s)?;
    let (r_prime, alpha_prime) = tail.r.split_at(dims.ell_n);

    // 3) Optional cross-checks
    if cfg.initial_sum {
        let lhs_exact = crate::engines::paper_exact_engine::sum_q_over_hypercube_paper_exact(
            s,
            params,
            mcs_witnesses,
            me_witnesses,
            &proof.challenges_public,
            dims.ell_d,
            dims.ell_n,
            me_inputs.get(0).map(|mi| mi.r.as_slice()),
        );
        if lhs_exact != tail.initial_sum {
            return Err(PiCcsError::ProtocolError(
                "crosscheck: initial sum mismatch (optimized vs paper-exact)".into(),
            ));
        }
    }

    if cfg.terminal {
        // Check if detailed logging is enabled via environment variable
        let detailed_log = std::env::var("NEO_CROSSCHECK_DETAIL").is_ok();
        
        if detailed_log {
            eprintln!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
            eprintln!("║           CROSSCHECK: Computing Terminal Evaluation Q(α', r')            ║");
            eprintln!("╚═══════════════════════════════════════════════════════════════════════════╝");
            eprintln!();
        }
        
        // Optimized RHS assembled from outputs
        if detailed_log {
            eprintln!();
            eprintln!("╔════════════════════════════════════════════════════════════════════════════╗");
            eprintln!("║                    OPTIMIZED ENGINE (from ME outputs)                      ║");
            eprintln!("╚════════════════════════════════════════════════════════════════════════════╝");
        }
        let rhs_opt = crate::engines::optimized_engine::terminal::rhs_Q_apr(
            s,
            &proof.challenges_public,
            r_prime,
            alpha_prime,
            mcs_list,
            me_inputs,
            &out_me_opt,
            params,
        )?;
        if detailed_log {
            eprintln!("  [Optimized] Result: {:?}", rhs_opt);
        }

        // Paper-exact LHS directly from witnesses
        if detailed_log {
            eprintln!();
            eprintln!("╔════════════════════════════════════════════════════════════════════════════╗");
            eprintln!("║                 PAPER-EXACT ENGINE (from witnesses)                        ║");
            eprintln!("╚════════════════════════════════════════════════════════════════════════════╝");
        }
        
        // IMPORTANT: Pass the ME inputs' r value so the Eval block is properly gated
        let r_inputs = me_inputs.get(0).map(|mi| mi.r.as_slice());
        let (lhs_exact, _rhs_unused) = crate::engines::paper_exact_engine::q_eval_at_ext_point_paper_exact_with_inputs(
            s,
            params,
            mcs_witnesses,
            me_witnesses,
            alpha_prime,
            r_prime,
            &proof.challenges_public,
            r_inputs,
        );
        if detailed_log {
            eprintln!("  [Paper-exact] Result: {:?}", lhs_exact);
            eprintln!();
            eprintln!("╔════════════════════════════════════════════════════════════════════════════╗");
            eprintln!("║                           COMPARISON                                       ║");
            eprintln!("╚════════════════════════════════════════════════════════════════════════════╝");
            eprintln!("  Sumcheck final value (from optimized proof): {:?}", tail.running_sum);
            eprintln!("  Optimized Q(α', r'):                         {:?}", rhs_opt);
            eprintln!("  Paper-exact Q(α', r'):                       {:?}", lhs_exact);
            eprintln!("  Match: {}", rhs_opt == lhs_exact && rhs_opt == tail.running_sum);
            eprintln!();
        }

        if rhs_opt != lhs_exact || rhs_opt != tail.running_sum {
            eprintln!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
            eprintln!("║              CROSSCHECK: Terminal Evaluation Claim Mismatch               ║");
            eprintln!("╚═══════════════════════════════════════════════════════════════════════════╝");
            eprintln!();
            eprintln!("Background:");
            eprintln!("  The optimized engine ran and produced a proof via sumcheck.");
            eprintln!("  Now we're verifying that Q(α', r') can be computed correctly.");
            eprintln!();
            eprintln!("Three ways to compute/check Q(α', r'):");
            eprintln!();
            eprintln!("  1. Optimized terminal formula (from output ME instances):");
            eprintln!("     → {:?}", rhs_opt);
            eprintln!();
            eprintln!("  2. Paper-exact direct evaluation (from witnesses):");
            eprintln!("     → {:?}", lhs_exact);
            eprintln!();
            eprintln!("  3. Sumcheck final value (from optimized engine's proof):");
            eprintln!("     → {:?}", tail.running_sum);
            eprintln!();
            eprintln!("Comparisons:");
            eprintln!("  Optimized terminal == Paper-exact:   {}", rhs_opt == lhs_exact);
            eprintln!("  Optimized terminal == Sumcheck final: {}", rhs_opt == tail.running_sum);
            eprintln!("  Paper-exact == Sumcheck final:        {}", lhs_exact == tail.running_sum);
            eprintln!();
            eprintln!("Expected: All three should match.");
            eprintln!("Actual:   Mismatch detected!");
            eprintln!();
            if rhs_opt == tail.running_sum && lhs_exact != tail.running_sum {
                eprintln!("Diagnosis: Optimized engine is self-consistent, but paper-exact");
                eprintln!("           formula produces a different value.");
            } else if lhs_exact == tail.running_sum && rhs_opt != tail.running_sum {
                eprintln!("Diagnosis: Paper-exact matches sumcheck, but optimized terminal");
                eprintln!("           formula produces a different value.");
            } else {
                eprintln!("Diagnosis: Complex mismatch - neither matches sumcheck final value.");
            }
            eprintln!();
            eprintln!("Tip: Set NEO_CROSSCHECK_DETAIL=1 for detailed step-by-step computation logs.");
            eprintln!("═══════════════════════════════════════════════════════════════════════════\n");
            
            return Err(PiCcsError::ProtocolError(
                "crosscheck: terminal evaluation claim mismatch".into(),
            ));
        }
    }

    if cfg.outputs {
        let fold_digest: [u8; 32] = proof
            .header_digest
            .as_slice()
            .try_into()
            .unwrap_or([0u8; 32]);

        let out_me_ref = crate::engines::paper_exact_engine::build_me_outputs_paper_exact(
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

        if out_me_ref.len() != out_me_opt.len() {
            return Err(PiCcsError::ProtocolError(
                "crosscheck: outputs length mismatch".into(),
            ));
        }
        for (a, b) in out_me_ref.iter().zip(out_me_opt.iter()) {
            if a.m_in != b.m_in || a.r != b.r || a.c.data != b.c.data {
                return Err(PiCcsError::ProtocolError(
                    "crosscheck: output metadata mismatch".into(),
                ));
            }
            for (ya, yb) in a.y.iter().zip(b.y.iter()) {
                if ya.len() != yb.len() {
                    return Err(PiCcsError::ProtocolError(
                        "crosscheck: y row length mismatch".into(),
                    ));
                }
                if ya != yb {
                    return Err(PiCcsError::ProtocolError(
                        "crosscheck: y row content mismatch".into(),
                    ));
                }
            }
            if a.y_scalars != b.y_scalars {
                return Err(PiCcsError::ProtocolError(
                    "crosscheck: y_scalars mismatch".into(),
                ));
            }
            // X matrix equality (dense, small D)
            if a.X.rows() != b.X.rows() || a.X.cols() != b.X.cols() {
                return Err(PiCcsError::ProtocolError(
                    "crosscheck: X dims mismatch".into(),
                ));
            }
            for r in 0..a.X.rows() {
                for c in 0..a.X.cols() {
                    if a.X[(r, c)] != b.X[(r, c)] {
                        return Err(PiCcsError::ProtocolError(
                            format!(
                                "crosscheck: X mismatch at ({},{})", r, c
                            ),
                        ));
                    }
                }
            }
        }
    }

    Ok((out_me_opt, proof))
}

/// Implementation of verify logic for cross-checking.
/// Verification remains the optimized path; cross-checking is a prover-side aid.
pub fn crosscheck_verify<I>(
    inner: &I,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_outputs: &[MeInstance<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError>
where
    I: PiCcsEngine,
{
    inner.verify(tr, params, s, mcs_list, me_inputs, me_outputs, proof)
}
