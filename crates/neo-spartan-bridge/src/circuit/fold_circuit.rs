//! Main FoldRun circuit implementation
//!
//! This synthesizes R1CS constraints to verify an entire FoldRun:
//! - For each fold step:
//!   - Verify Π-CCS terminal identity
//!   - Verify sumcheck rounds
//!   - Verify RLC equalities
//!   - Verify DEC equalities
//! - Verify accumulator chaining between steps

use bellpepper_core::{ConstraintSystem, SynthesisError, Variable};
use crate::circuit::witness::{FoldRunWitness, FoldRunInstance};
use crate::gadgets::k_field::{KNum, KNumVar, alloc_k, k_add as k_add_raw, k_mul as k_mul_raw, k_scalar_mul as k_scalar_mul_raw};
use crate::gadgets::pi_ccs::{sumcheck_round_gadget, sumcheck_eval_gadget};
use crate::error::{Result, SpartanBridgeError};
use neo_fold::folding::FoldStep;
use neo_math::F as NeoF;
use neo_ccs::Mat;
use p3_field::{PrimeField64, PrimeCharacteristicRing};
use crate::CircuitF;

// Spartan2 integration: implement SpartanCircuit over Goldilocks + Hash-MLE PCS.
use bellpepper_core::num::AllocatedNum;
use spartan2::provider::GoldilocksP3MerkleMleEngine;
use spartan2::traits::circuit::SpartanCircuit as SpartanCircuitTrait;

/// Sparse representation of the CCS polynomial f in the circuit field.
///
/// Each term is coeff * ∏_j m_j^{exps[j]}.
#[derive(Clone, Debug)]
pub struct CircuitPolyTerm {
    pub coeff: CircuitF,
    pub exps: Vec<u32>,
}

/// Main circuit for verifying a FoldRun
#[derive(Clone)]
pub struct FoldRunCircuit {
    /// Public instance
    pub instance: FoldRunInstance,
    
    /// Private witness
    pub witness: Option<FoldRunWitness>,
    
    /// Delta constant for K-field multiplication (u^2 = δ)
    /// For Goldilocks K, δ = 7
    pub delta: CircuitF,
    
    /// Base parameter b for DEC decomposition
    pub base_b: u32,

    /// CCS polynomial f, converted to circuit field coefficients.
    pub poly_f: Vec<CircuitPolyTerm>,
}

impl FoldRunCircuit {
    pub fn new(
        instance: FoldRunInstance,
        witness: Option<FoldRunWitness>,
        delta: CircuitF,
        base_b: u32,
        poly_f: Vec<CircuitPolyTerm>,
    ) -> Self {
        Self {
            instance,
            witness,
            delta,
            base_b,
            poly_f,
        }
    }

    /// Synthesize the full FoldRun circuit
    ///
    /// This is the main entry point for constraint generation
    pub fn synthesize<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
    ) -> Result<()> {
        // Allocate public inputs
        self.allocate_public_inputs(cs)?;

        // Get witness or error
        let witness = self.witness.as_ref()
            .ok_or_else(|| SpartanBridgeError::InvalidInput("Missing witness".into()))?;

        // Verify each fold step
        for (step_idx, step) in witness.fold_run.steps.iter().enumerate() {
            self.verify_fold_step(cs, step_idx, step, witness)?;
        }

        // Verify accumulator chaining across all steps
        self.verify_accumulator_chaining(cs, witness)?;

        Ok(())
    }

    /// Allocate and constrain public inputs
    /// 
    /// Currently we expose a minimal set of public inputs:
    /// - Params digest
    /// - CCS digest
    /// - MCS digest
    ///
    /// The full accumulator and Π-CCS challenges remain private witness data for now.
    fn allocate_public_inputs<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
    ) -> Result<()> {
        // Helper to expose a 32-byte digest as 4 field elements by chunking into u64.
        let mut alloc_digest =
            |label: &str, digest: &[u8; 32]| -> Result<()> {
                for (i, chunk) in digest.chunks(8).enumerate() {
                    let mut limb_bytes = [0u8; 8];
                    limb_bytes.copy_from_slice(chunk);
                    let limb_u64 = u64::from_le_bytes(limb_bytes);
                    let value = CircuitF::from(limb_u64);
                    // Allocate as public input.
                    let _ = cs.alloc_input(
                        || format!("{}_limb_{}", label, i),
                        || Ok(value),
                    )?;
                }
                Ok(())
            };

        alloc_digest("params_digest", &self.instance.params_digest)?;
        alloc_digest("ccs_digest", &self.instance.ccs_digest)?;
        alloc_digest("mcs_digest", &self.instance.mcs_digest)?;

        Ok(())
    }

    /// Verify a single fold step
    fn verify_fold_step<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        step: &neo_fold::folding::FoldStep,
        witness: &FoldRunWitness,
    ) -> Result<()> {
        // Get the Π-CCS proof and challenges for this step
        let pi_ccs_proof = &witness.pi_ccs_proofs[step_idx];
        let challenges = &self.instance.pi_ccs_challenges[step_idx];

        // 1a. Bind the public claimed initial sum T from ME inputs and challenges.
        let initial_sum_var = self.verify_initial_sum_binding(
            cs,
            step_idx,
            pi_ccs_proof,
            challenges,
            witness,
        )?;

        // 1b. Run sumcheck rounds algebra starting from T to obtain the final running sum.
        let sumcheck_final_var = self.verify_sumcheck_rounds(
            cs,
            step_idx,
            pi_ccs_proof,
            challenges,
            &initial_sum_var,
        )?;

        // 1c. Verify Π-CCS terminal identity: final running sum == RHS(α,β,γ,r',α', out_me, inputs).
        self.verify_terminal_identity(
            cs,
            step_idx,
            step,
            pi_ccs_proof,
            challenges,
            witness,
            &sumcheck_final_var,
        )?;

        // 2. Verify RLC equalities
        self.verify_rlc(cs, step_idx, step, witness)?;

        // 3. Verify DEC equalities
        self.verify_dec(cs, step_idx, step, witness)?;

        Ok(())
    }

    /// Verify sumcheck rounds for a step
    fn verify_sumcheck_rounds<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        proof: &neo_reductions::PiCcsProof,
        challenges: &crate::circuit::witness::PiCcsChallenges,
        initial_sum: &KNumVar,
    ) -> Result<KNumVar> {
        // Sanity: number of rounds in the proof must match the number of
        // advertised challenges for this step.
        if proof.sumcheck_rounds.len() != challenges.sumcheck_challenges.len() {
            return Err(SpartanBridgeError::InvalidInput(
                format!(
                    "Sumcheck rounds/challenges length mismatch at step {}: rounds={}, challenges={}",
                    step_idx,
                    proof.sumcheck_rounds.len(),
                    challenges.sumcheck_challenges.len()
                ),
            ));
        }

        // Start running sum from the in-circuit initial sum variable (T).
        let mut claimed_sum = initial_sum.clone();

        // For each sumcheck round
        for (round_idx, round_poly) in proof.sumcheck_rounds.iter().enumerate() {
            // Allocate polynomial coefficients
            let mut coeffs = Vec::new();
            for (coeff_idx, coeff) in round_poly.iter().enumerate() {
                let coeff_var = self.alloc_k_from_neo(
                    cs,
                    *coeff,
                    &format!("step_{}_round_{}_coeff_{}", step_idx, round_idx, coeff_idx),
                ).map_err(|e| SpartanBridgeError::SynthesisError(format!("{:?}", e)))?;
                coeffs.push(coeff_var);
            }

            // Verify p(0) + p(1) = claimed_sum
            sumcheck_round_gadget(
                cs,
                &coeffs,
                round_poly,
                &claimed_sum,
                self.delta,
                &format!("step_{}_round_{}", step_idx, round_idx),
            ).map_err(|e| SpartanBridgeError::from(e))?;

            // Allocate challenge for this round
            let challenge = self.alloc_k_from_neo(
                cs,
                challenges.sumcheck_challenges[round_idx],
                &format!("step_{}_round_{}_challenge", step_idx, round_idx),
            )?;

            // Compute next claimed sum = p(challenge) via gadget (returns KNumVar)
            let next_sum = sumcheck_eval_gadget(
                cs,
                &coeffs,
                round_poly,
                &challenge,
                challenges.sumcheck_challenges[round_idx],
                self.delta,
                &format!("step_{}_round_{}_eval", step_idx, round_idx),
            ).map_err(|e| SpartanBridgeError::from(e))?;

            // Update claimed_sum for next round
            claimed_sum = next_sum;
        }

        // After all rounds, optionally check that the prover's scalar
        // sumcheck_final matches the in-circuit running sum. This is a
        // consistency check only; the algebraic binding is via the KNumVar
        // returned from this method.
        let final_sum_expected = self.alloc_k_from_neo(
            cs,
            proof.sumcheck_final,
            &format!("step_{}_final_sum", step_idx),
        )?;

        self.enforce_k_eq(
            cs,
            &claimed_sum,
            &final_sum_expected,
            &format!("step_{}_final_sum_matches_scalar", step_idx),
        );

        Ok(claimed_sum)
    }

    /// Verify that sc_initial_sum equals the public claimed initial sum T
    /// computed from ME inputs and α, mirroring `claimed_initial_sum_from_inputs`.
    fn verify_initial_sum_binding<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        proof: &neo_reductions::PiCcsProof,
        challenges: &crate::circuit::witness::PiCcsChallenges,
        witness: &FoldRunWitness,
    ) -> Result<KNumVar> {
        use core::cmp::min;
        use neo_math::K as NeoK;

        // ME inputs for this step (in host representation):
        // - step 0: public initial accumulator,
        // - step i>0: DEC children of previous step.
        let me_inputs: &[neo_ccs::MeInstance<neo_ajtai::Commitment, NeoF, NeoK>] =
            if step_idx == 0 {
                &self.instance.initial_accumulator
            } else {
                &witness.fold_run.steps[step_idx - 1].dec_children
            };

        // Host-side computation of the public claimed initial sum T, mirroring
        // `neo_reductions::optimized_engine::claimed_initial_sum_from_inputs`.
        let t_native: NeoK = {
            let k_total = 1 + me_inputs.len(); // first slot is the MCS instance
            if k_total < 2 {
                NeoK::ZERO
            } else {
                // Build χ_{α} over the Ajtai domain.
                let d_sz = 1usize
                    .checked_shl(challenges.alpha.len() as u32)
                    .unwrap_or(0);
                let mut chi_a = vec![NeoK::ZERO; d_sz];
                for rho in 0..d_sz {
                    let mut w = NeoK::ONE;
                    for (bit, &a) in challenges.alpha.iter().enumerate() {
                        let is_one = ((rho >> bit) & 1) == 1;
                        w *= if is_one { a } else { NeoK::ONE - a };
                    }
                    chi_a[rho] = w;
                }

                // γ^k
                let mut gamma_to_k = NeoK::ONE;
                for _ in 0..k_total {
                    gamma_to_k *= challenges.gamma;
                }

                // Number of matrices t: use y-table length from ME inputs.
                let t = if me_inputs.is_empty() {
                    0
                } else {
                    me_inputs[0].y.len()
                };

                // Inner weighted sum over (j, i>=2)
                let mut inner = NeoK::ZERO;
                for j in 0..t {
                    for (idx, out) in me_inputs.iter().enumerate() {
                        // me_inputs[idx] corresponds to instance i = idx + 2 in the paper
                        // (i=1 is the MCS instance, not in me_inputs)
                        let i_abs = idx + 2;

                        // ẏ_{(i,j)}(α) = ⟨ y_{(i,j)}, χ_{α} ⟩
                        let yj = &out.y[j];
                        let mut y_eval = NeoK::ZERO;
                        let limit = min(d_sz, yj.len());
                        for rho in 0..limit {
                            y_eval += yj[rho] * chi_a[rho];
                        }

                        // weight = γ^{i-1} · (γ^k)^j  (0-based j)
                        let mut weight = NeoK::ONE;
                        // γ^{i-1}
                        for _ in 0..(i_abs - 1) {
                            weight *= challenges.gamma;
                        }
                        // (γ^k)^j
                        for _ in 0..j {
                            weight *= gamma_to_k;
                        }

                        inner += weight * y_eval;
                    }
                }

                // Multiply by a single outer γ^k to match
                // `claimed_initial_sum_from_inputs` in the optimized engine.
                let mut result = NeoK::ONE;
                for _ in 0..k_total {
                    result *= challenges.gamma;
                }
                result * inner
            }
        };

        // Allocate T as a K constant derived from the host-side T_native.
        let t_var = self.alloc_k_from_neo(
            cs,
            t_native,
            &format!("step_{}_T", step_idx),
        )?;

        // If the proof provided a scalar sc_initial_sum, enforce that it
        // matches the public T. This mirrors the native verifier's optional
        // tightness check.
        if let Some(sc_initial) = proof.sc_initial_sum {
            let sc_initial_var = self.alloc_k_from_neo(
                cs,
                sc_initial,
                &format!("step_{}_sc_initial_sum_binding", step_idx),
            )?;

            self.enforce_k_eq(
                cs,
                &sc_initial_var,
                &t_var,
                &format!("step_{}_initial_sum_matches_T", step_idx),
            );
        }

        Ok(t_var)
    }

    /// KNumVar version of `claimed_initial_sum_from_inputs`, using only the ME
    /// inputs' y-vectors and α, γ from the challenges.
    fn claimed_initial_sum_gadget<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        alpha: &[KNumVar],
        gamma: &KNumVar,
        me_inputs_y: &[Vec<Vec<KNumVar>>],
    ) -> Result<KNumVar> {
        use core::cmp::min;

        let k_total = 1 + me_inputs_y.len(); // 1 MCS + |ME|
        if k_total < 2 {
            // No Eval block when k=1 → T = 0.
            return self.k_zero(cs, &format!("step_{}_T_zero", step_idx));
        }

        // Build χ_α over Ajtai domain.
        let d_sz = 1usize << alpha.len();
        let mut chi_alpha: Vec<KNumVar> = Vec::with_capacity(d_sz);

        for rho in 0..d_sz {
            let mut w = self.k_one(
                cs,
                &format!("step_{}_chi_alpha_{}_init", step_idx, rho),
            )?;
            for (bit, a_bit) in alpha.iter().enumerate() {
                let bit_is_one = ((rho >> bit) & 1) == 1;
                let factor = if bit_is_one {
                    a_bit.clone()
                } else {
                    self.k_one_minus(
                        cs,
                        a_bit,
                        &format!(
                            "step_{}_chi_alpha_{}_bit{}_one_minus",
                            step_idx, rho, bit
                        ),
                    )?
                };
                w = self.k_mul(
                    cs,
                    &w,
                    &factor,
                    &format!(
                        "step_{}_chi_alpha_{}_bit{}",
                        step_idx, rho, bit
                    ),
                )?;
            }
            chi_alpha.push(w);
        }

        // γ^k_total
        let mut gamma_to_k = self.k_one(
            cs,
            &format!("step_{}_gamma_to_k_init", step_idx),
        )?;
        for e in 0..k_total {
            gamma_to_k = self.k_mul(
                cs,
                &gamma_to_k,
                gamma,
                &format!("step_{}_gamma_to_k_pow{}", step_idx, e + 1),
            )?;
        }

        // Inner weighted sum over (j, i>=2).
        let t = if me_inputs_y.is_empty() {
            0
        } else {
            me_inputs_y[0].len()
        };

        let mut inner = self.k_zero(
            cs,
            &format!("step_{}_T_inner_init", step_idx),
        )?;

        for j in 0..t {
            // (γ^k_total)^j – shared across all i for this j.
            let mut gamma_k_j = self.k_one(
                cs,
                &format!("step_{}_T_gamma_k_j_init_j{}", step_idx, j),
            )?;
            for pow_idx in 0..j {
                gamma_k_j = self.k_mul(
                    cs,
                    &gamma_k_j,
                    &gamma_to_k,
                    &format!(
                        "step_{}_T_gamma_k_j_step_j{}_{}",
                        step_idx, j, pow_idx
                    ),
                )?;
            }

            for (idx, y_table) in me_inputs_y.iter().enumerate() {
                // me_inputs[idx] corresponds to instance i = idx + 2 in the paper.
                let i_abs = idx + 2;
                let row = &y_table[j];
                let limit = min(d_sz, row.len());

                // y_eval = ⟨ y_{(i,j)}, χ_α ⟩
                let mut y_eval = self.k_zero(
                    cs,
                    &format!("step_{}_T_y_eval_j{}_i{}", step_idx, j, i_abs),
                )?;
                for rho in 0..limit {
                    let prod = self.k_mul(
                        cs,
                        &row[rho],
                        &chi_alpha[rho],
                        &format!(
                            "step_{}_T_y_eval_j{}_i{}_rho{}",
                            step_idx, j, i_abs, rho
                        ),
                    )?;
                    y_eval = self.k_add(
                        cs,
                        &y_eval,
                        &prod,
                        &format!(
                            "step_{}_T_y_eval_acc_j{}_i{}_rho{}",
                            step_idx, j, i_abs, rho
                        ),
                    )?;
                }

                // γ^{i-1}
                let mut gamma_i = self.k_one(
                    cs,
                    &format!("step_{}_T_gamma_i_init_j{}_i{}", step_idx, j, i_abs),
                )?;
                for pow_idx in 0..(i_abs - 1) {
                    gamma_i = self.k_mul(
                        cs,
                        &gamma_i,
                        gamma,
                        &format!("step_{}_T_gamma_i_step_j{}_i{}_{}", step_idx, j, i_abs, pow_idx),
                    )?;
                }

                let weight = self.k_mul(
                    cs,
                    &gamma_i,
                    &gamma_k_j,
                    &format!("step_{}_T_weight_j{}_i{}", step_idx, j, i_abs),
                )?;

                let contrib = self.k_mul(
                    cs,
                    &weight,
                    &y_eval,
                    &format!("step_{}_T_contrib_j{}_i{}", step_idx, j, i_abs),
                )?;

                inner = self.k_add(
                    cs,
                    &inner,
                    &contrib,
                    &format!("step_{}_T_inner_acc_j{}_i{}", step_idx, j, i_abs),
                )?;
            }
        }

        // T = γ^k_total · inner
        let t = self.k_mul(
            cs,
            &gamma_to_k,
            &inner,
            &format!("step_{}_T_with_outer_gamma_k", step_idx),
        )?;
        Ok(t)
    }

    /// Build χ-table for an Ajtai challenge vector (α or α') using native K values.
    ///
    /// χ_α[ρ] = ∏_bit (α_bit if ρ_bit=1 else 1-α_bit), computed in the host field and
    /// lifted into K variables. This avoids needing in-circuit K arithmetic for χ.
    fn build_chi_table<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        alpha_vars: &[KNumVar],
        alpha_values: &[neo_math::K],
        label: &str,
    ) -> Result<Vec<KNumVar>> {
        if alpha_vars.len() != alpha_values.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "chi-table alpha length mismatch at step {}: vars={}, values={}",
                step_idx,
                alpha_vars.len(),
                alpha_values.len()
            )));
        }

        let d_sz = 1usize << alpha_values.len();
        let mut chi: Vec<KNumVar> = Vec::with_capacity(d_sz);

        for rho in 0..d_sz {
            // Host-side χ value for this rho.
            let mut w_native = neo_math::K::ONE;
            for (bit, &a_val) in alpha_values.iter().enumerate() {
                let bit_is_one = ((rho >> bit) & 1) == 1;
                let term = if bit_is_one {
                    a_val
                } else {
                    neo_math::K::ONE - a_val
                };
                w_native *= term;
            }

            let w_var = self.alloc_k_from_neo(
                cs,
                w_native,
                &format!("step_{}_{}_chi_{}", step_idx, label, rho),
            )?;
            chi.push(w_var);
        }

        Ok(chi)
    }

    /// Equality polynomial eq_points over K, using the same formula as the
    /// native `eq_points`: ∏_i [(1-p_i)*(1-q_i) + p_i*q_i].
    ///
    /// The value is computed natively from `p_vals`/`q_vals` and lifted into K;
    /// the circuit treats it as a derived witness scalar.
    fn eq_points<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        p: &[KNumVar],
        q: &[KNumVar],
        p_vals: &[neo_math::K],
        q_vals: &[neo_math::K],
        label: &str,
    ) -> Result<KNumVar> {
        if p.len() != q.len() || p_vals.len() != q_vals.len() || p.len() != p_vals.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "eq_points length mismatch at step {}: p_vars={}, q_vars={}, p_vals={}, q_vals={}",
                step_idx,
                p.len(),
                q.len(),
                p_vals.len(),
                q_vals.len(),
            )));
        }

        // Native eq value.
        let mut acc_native = neo_math::K::ONE;
        for (&pi, &qi) in p_vals.iter().zip(q_vals.iter()) {
            let term = (neo_math::K::ONE - pi) * (neo_math::K::ONE - qi) + pi * qi;
            acc_native *= term;
        }

        let acc_var = self.alloc_k_from_neo(
            cs,
            acc_native,
            &format!("step_{}_{}_eq", step_idx, label),
        )?;

        Ok(acc_var)
    }

    /// Range product gadget: ∏_{t=-(b-1)}^{b-1} (val - t) over K.
    fn range_product<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        val: &KNumVar,
        label: &str,
    ) -> Result<KNumVar> {
        let mut acc = self.k_one(
            cs,
            &format!("step_{}_{}_range_init", step_idx, label),
        )?;

        let b = self.base_b as i32;
        for t in (-(b - 1))..=(b - 1) {
            let abs = t.abs() as u64;
            let base = if t >= 0 {
                CircuitF::from(abs)
            } else {
                CircuitF::from(0u64) - CircuitF::from(abs)
            };
            let t_k = self.k_const(
                cs,
                base,
                &format!("step_{}_{}_t_{}", step_idx, label, t),
            )?;

            let diff = self.k_sub(
                cs,
                val,
                &t_k,
                &format!("step_{}_{}_val_minus_t_{}", step_idx, label, t),
            )?;

            acc = self.k_mul(
                cs,
                &acc,
                &diff,
                &format!("step_{}_{}_range_acc_{}", step_idx, label, t),
            )?;
        }

        Ok(acc)
    }

    /// Evaluate the CCS polynomial f at the given m-values in K.
    fn eval_poly_f_in_k<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        m_vals: &[KNumVar],
    ) -> Result<KNumVar> {
        let mut acc = self.k_zero(
            cs,
            &format!("step_{}_F_prime_init", step_idx),
        )?;

        for (term_idx, term) in self.poly_f.iter().enumerate() {
            // term_val = ∏_j m_j^{exp_j}
            let mut term_val = self.k_one(
                cs,
                &format!("step_{}_F_term{}_init", step_idx, term_idx),
            )?;

            for (var_idx, &exp) in term.exps.iter().enumerate() {
                if exp == 0 {
                    continue;
                }
                let base = &m_vals[var_idx];
                let pow = self.k_pow(
                    cs,
                    base,
                    exp,
                    &format!(
                        "step_{}_F_term{}_var{}_pow",
                        step_idx, term_idx, var_idx
                    ),
                )?;
                term_val = self.k_mul(
                    cs,
                    &term_val,
                    &pow,
                    &format!(
                        "step_{}_F_term{}_var{}_mul",
                        step_idx, term_idx, var_idx
                    ),
                )?;
            }

            // Scale by coefficient.
            let scaled = self.k_scalar_mul(
                cs,
                term.coeff,
                &term_val,
                &format!("step_{}_F_term{}_scaled", step_idx, term_idx),
            )?;

            acc = self.k_add(
                cs,
                &acc,
                &scaled,
                &format!("step_{}_F_acc_term{}", step_idx, term_idx),
            )?;
        }

        Ok(acc)
    }

    /// Verify Π-CCS terminal identity for a step:
    /// sumcheck_final == rhs_terminal_identity_paper_exact(α,β,γ,r',α', outputs, inputs.r).
    fn verify_terminal_identity<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        step: &FoldStep,
        _proof: &neo_reductions::PiCcsProof,
        challenges: &crate::circuit::witness::PiCcsChallenges,
        witness: &FoldRunWitness,
        sumcheck_final: &KNumVar,
    ) -> Result<()> {
        use neo_math::{D, K as NeoK, F as NeoBaseF};

        // ME inputs for this step (needed only for r in eq((α',r'),(α,r))).
        let me_inputs = if step_idx == 0 {
            &self.instance.initial_accumulator
        } else {
            &witness.fold_run.steps[step_idx - 1].dec_children
        };

        // Outputs y' for this step.
        let out_me = &step.ccs_out;
        if out_me.is_empty() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "Terminal identity at step {}: empty outputs",
                step_idx
            )));
        }

        // Allocate α' and r' (used to tie them to sumcheck challenges).
        let mut alpha_prime_vars = Vec::with_capacity(challenges.alpha_prime.len());
        for (i, &k) in challenges.alpha_prime.iter().enumerate() {
            alpha_prime_vars.push(self.alloc_k_from_neo(
                cs,
                k,
                &format!("step_{}_alpha_prime_{}", step_idx, i),
            )?);
        }

        let mut r_prime_vars = Vec::with_capacity(challenges.r_prime.len());
        for (i, &k) in challenges.r_prime.iter().enumerate() {
            r_prime_vars.push(self.alloc_k_from_neo(
                cs,
                k,
                &format!("step_{}_r_prime_{}", step_idx, i),
            )?);
        }

        // Enforce that (r_prime, alpha_prime) is exactly the split of the
        // per-round sumcheck challenges vector.
        let total_chals = challenges.sumcheck_challenges.len();
        let rows = r_prime_vars.len();
        let ajtai = alpha_prime_vars.len();
        if total_chals != rows + ajtai {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "sumcheck_challenges length mismatch at step {}: expected {} (rows + ajtai), got {}",
                step_idx,
                rows + ajtai,
                total_chals
            )));
        }
        // r' == prefix of sumcheck_challenges
        for i in 0..rows {
            let sc_val = self.alloc_k_from_neo(
                cs,
                challenges.sumcheck_challenges[i],
                &format!("step_{}_sc_round_chal_row_{}", step_idx, i),
            )?;
            self.enforce_k_eq(
                cs,
                &r_prime_vars[i],
                &sc_val,
                &format!("step_{}_r_prime_matches_sc_{}", step_idx, i),
            );
        }
        // α' == suffix of sumcheck_challenges
        for j in 0..ajtai {
            let idx = rows + j;
            let sc_val = self.alloc_k_from_neo(
                cs,
                challenges.sumcheck_challenges[idx],
                &format!("step_{}_sc_round_chal_ajtai_{}", step_idx, j),
            )?;
            self.enforce_k_eq(
                cs,
                &alpha_prime_vars[j],
                &sc_val,
                &format!("step_{}_alpha_prime_matches_sc_{}", step_idx, j),
            );
        }

        // --- Native computation of RHS terminal identity (Step 4) ---
        //
        // We mirror `rhs_terminal_identity_paper_exact` from the Neo reductions
        // engine, but using only the data available to the circuit:
        //   - out_me.y (outputs)
        //   - me_inputs[0].r (for eq((α',r'),(α,r)))
        //   - challenges (α, β, γ, α', r')
        //   - base_b and poly_f (to evaluate F').

        // eq((α',r'), β) = eq(α', β_a) * eq(r', β_r).
        let eq_points_native = |p: &[NeoK], q: &[NeoK]| -> NeoK {
            if p.len() != q.len() {
                return NeoK::ZERO;
            }
            let mut acc = NeoK::ONE;
            for (&pi, &qi) in p.iter().zip(q.iter()) {
                let term = (NeoK::ONE - pi) * (NeoK::ONE - qi) + pi * qi;
                acc *= term;
            }
            acc
        };

        let eq_aprp_beta_native = if challenges.alpha_prime.is_empty() && challenges.r_prime.is_empty() {
            NeoK::ONE
        } else {
            let e1 = eq_points_native(&challenges.alpha_prime, &challenges.beta_a);
            let e2 = eq_points_native(&challenges.r_prime, &challenges.beta_r);
            e1 * e2
        };

        // eq((α',r'),(α,r)) if we have ME inputs; else 0 (Eval' block vanishes).
        let eq_aprp_ar_native = if let Some(first_input) = me_inputs.first() {
            let e1 = eq_points_native(&challenges.alpha_prime, &challenges.alpha);
            let e2 = eq_points_native(&challenges.r_prime, &first_input.r);
            e1 * e2
        } else {
            NeoK::ZERO
        };

        // --- F' from first output's y'[i=1] ---
        // Recompose m_j from Ajtai digits with base-b, then evaluate f via poly_f.
        let t = out_me[0].y.len();

        let bK = NeoK::from(NeoBaseF::from_u64(self.base_b as u64));
        let mut m_vals_native: Vec<NeoK> = Vec::with_capacity(t);
        for j in 0..t {
            let row = &out_me[0].y[j]; // K^d (padded)
            let mut acc = NeoK::ZERO;
            let mut pow = NeoK::ONE;
            for rho in 0..D {
                let digit = row.get(rho).copied().unwrap_or(NeoK::ZERO);
                acc += pow * digit;
                pow *= bK;
            }
            m_vals_native.push(acc);
        }

        // Evaluate CCS polynomial f at m_vals_native using poly_f (coeffs in base field).
        let mut F_prime_native = NeoK::ZERO;
        for term in &self.poly_f {
            // term_val = ∏_j m_j^{exp_j}
            let mut term_val = NeoK::ONE;
            for (var_idx, &exp) in term.exps.iter().enumerate() {
                if exp == 0 {
                    continue;
                }
                let base = m_vals_native[var_idx];
                let mut pow_val = NeoK::ONE;
                for _ in 0..exp {
                    pow_val *= base;
                }
                term_val *= pow_val;
            }

            // Scale by coefficient (lift CircuitF coeff back to Neo base field, then to K).
            let coeff_base = NeoBaseF::from_u64(term.coeff.to_canonical_u64());
            let coeff_k = NeoK::from(coeff_base);
            F_prime_native += coeff_k * term_val;
        }

        // --- Σ γ^i · N_i' over outputs.
        // N_i' = ∏_{t} ( ẏ'_{(i,1)}(α') - t ), with ẏ' evaluated at α' as MLE.
        let d_sz = 1usize << challenges.alpha_prime.len();
        let mut chi_alpha_prime_native = vec![NeoK::ZERO; d_sz];
        for rho in 0..d_sz {
            let mut w = NeoK::ONE;
            for (bit, &a) in challenges.alpha_prime.iter().enumerate() {
                let bit_is_one = ((rho >> bit) & 1) == 1;
                w *= if bit_is_one { a } else { NeoK::ONE - a };
            }
            chi_alpha_prime_native[rho] = w;
        }

        let range_product_native = |val: NeoK, b: u32| -> NeoK {
            let lo = -((b as i64) - 1);
            let hi = (b as i64) - 1;
            let mut prod = NeoK::ONE;
            for t_i in lo..=hi {
                let base = NeoBaseF::from_i64(t_i);
                prod *= val - NeoK::from(base);
            }
            prod
        };

        let mut nc_prime_sum_native = NeoK::ZERO;
        {
            let mut g = challenges.gamma; // γ^1
            for out in out_me {
                // ẏ'_{(i,1)}(α') = Σ_ρ y'_{(i,1)}[ρ] · χ_{α'}[ρ]
                let y1 = &out.y[0];
                let limit = core::cmp::min(chi_alpha_prime_native.len(), y1.len());
                let mut y_eval = NeoK::ZERO;
                for rho in 0..limit {
                    y_eval += y1[rho] * chi_alpha_prime_native[rho];
                }
                let Ni = range_product_native(y_eval, self.base_b);
                nc_prime_sum_native += g * Ni;
                g *= challenges.gamma;
            }
        }

        // --- Eval' block ---
        // γ^k · Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · E_{(i,j)} with
        // E_{(i,j)} = eq((α',r'),(α,r)) · ẏ'_{(i,j)}(α').
        let mut eval_sum_native = NeoK::ZERO;
        let k_total = out_me.len();
        if me_inputs.first().is_some() && k_total >= 2 {
            // Precompute γ^k
            let mut gamma_to_k = NeoK::ONE;
            for _ in 0..k_total {
                gamma_to_k *= challenges.gamma;
            }

            for j in 0..t {
                for (i_abs, out) in out_me.iter().enumerate().skip(1) {
                    let y = &out.y[j];
                    let mut y_eval = NeoK::ZERO;
                    let limit = core::cmp::min(chi_alpha_prime_native.len(), y.len());
                    for rho in 0..limit {
                        y_eval += y[rho] * chi_alpha_prime_native[rho];
                    }

                    // weight = γ^{i-1} * (γ^k)^j  (0-based indices)
                    let mut weight = NeoK::ONE;
                    for _ in 0..i_abs {
                        weight *= challenges.gamma;
                    }
                    for _ in 0..j {
                        weight *= gamma_to_k;
                    }

                    eval_sum_native += weight * y_eval;
                }
            }
        }

        // Assemble RHS in native K:
        // v = eq((α',r'), β)·(F' + Σ γ^i N_i') + γ^k · eq((α',r'), (α,r)) · Eval'.
        let mut gamma_to_k_outer = NeoK::ONE;
        for _ in 0..k_total {
            gamma_to_k_outer *= challenges.gamma;
        }
        let rhs_native =
            eq_aprp_beta_native * (F_prime_native + nc_prime_sum_native)
                + eq_aprp_ar_native * (gamma_to_k_outer * eval_sum_native);

        // Lift RHS into the circuit as a single KNumVar and bind to sumcheck_final.
        let rhs = self.alloc_k_from_neo(
            cs,
            rhs_native,
            &format!("step_{}_rhs_native", step_idx),
        )?;

        // Enforce that the in-circuit final running sum from sumcheck
        // rounds equals the RHS terminal identity.
        self.enforce_k_eq(
            cs,
            sumcheck_final,
            &rhs,
            &format!("step_{}_final_sum_matches_rhs", step_idx),
        );

        Ok(())
    }

    /// Verify RLC equalities for a step
    ///
    /// Enforces the public RLC relations used in `rlc_public`:
    /// - r is preserved: parent.r == child.r for all inputs
    /// - X_parent = Σ_i ρ_i · X_i
    /// - y_parent[j] = Σ_i ρ_i · y_(i,j) (first D digits)
    fn verify_rlc<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        step: &FoldStep,
        _witness: &FoldRunWitness,
    ) -> Result<()> {
        let parent = &step.rlc_parent;
        let children = &step.ccs_out;
        let rhos: &[Mat<NeoF>] = &step.rlc_rhos;

        if children.is_empty() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "RLC at step {} has no children",
                step_idx
            )));
        }
        if children.len() != rhos.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "RLC at step {}: rhos/children length mismatch (rhos={}, children={})",
                step_idx,
                rhos.len(),
                children.len()
            )));
        }

        let d = parent.X.rows();
        let m_in = parent.m_in;

        // Dimension sanity checks
        for (i, child) in children.iter().enumerate() {
            if child.X.rows() != d || child.X.cols() != parent.X.cols() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC X dimension mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
            if child.m_in != m_in {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC m_in mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
        }
        for (i, rho) in rhos.iter().enumerate() {
            if rho.rows() != d || rho.cols() != d {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC ρ dimension mismatch at step {}, matrix {}",
                    step_idx, i
                )));
            }
        }

        // Allocate X matrices for parent and children
        let parent_X_vars = self.alloc_matrix_from_neo(
            cs,
            &parent.X,
            &format!("step_{}_rlc_parent_X", step_idx),
        )?;

        let mut child_X_vars = Vec::with_capacity(children.len());
        for (i, child) in children.iter().enumerate() {
            child_X_vars.push(self.alloc_matrix_from_neo(
                cs,
                &child.X,
                &format!("step_{}_rlc_child_{}_X", step_idx, i),
            )?);
        }

        // Enforce X_parent = Σ_i ρ_i · X_i
        for row in 0usize..d {
            for col in 0usize..m_in {
                cs.enforce(
                    || format!("step_{}_rlc_X_{}_{}", step_idx, row, col),
                    |lc| {
                        let mut res = lc;
                        for (i, rho) in rhos.iter().enumerate() {
                            for k in 0..d {
                                let coeff = Self::neo_f_to_circuit(&rho[(row, k)]);
                                if coeff != CircuitF::from(0u64) {
                                    res = res + (coeff, child_X_vars[i][k][col]);
                                }
                            }
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_X_vars[row][col],
                );
            }
        }

        // Enforce r preservation: parent.r == child.r for all inputs
        let r_len = parent.r.len();
        for (i, child) in children.iter().enumerate() {
            if child.r.len() != r_len {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC r length mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
            for idx in 0..r_len {
                let parent_r_var = self.alloc_k_from_neo(
                    cs,
                    parent.r[idx],
                    &format!("step_{}_rlc_parent_r_{}_{}", step_idx, i, idx),
                )?;
                let child_r_var = self.alloc_k_from_neo(
                    cs,
                    child.r[idx],
                    &format!("step_{}_rlc_child_{}_r_{}", step_idx, i, idx),
                )?;
                self.enforce_k_eq(
                    cs,
                    &parent_r_var,
                    &child_r_var,
                    &format!("step_{}_rlc_r_eq_child_{}_idx_{}", step_idx, i, idx),
                );
            }
        }

        // Enforce y_parent[j] = Σ_i ρ_i · y_(i,j) on the first D digits.
        // Use the ring's D dimension and y-vector lengths.
        let t = parent.y.len();
        if t == 0 {
            return Ok(());
        }
        let d_pad = parent.y[0].len();
        if d_pad == 0 {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "RLC at step {}: empty y vectors in parent",
                step_idx
            )));
        }
        let d_ring = neo_math::D.min(d_pad);

        let parent_y_vars = self.alloc_y_table_from_neo(
            cs,
            &parent.y,
            &format!("step_{}_rlc_parent_y", step_idx),
        )?;

        let mut children_y_vars = Vec::with_capacity(children.len());
        for (i, child) in children.iter().enumerate() {
            if child.y.len() != t {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC y length mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
            children_y_vars.push(self.alloc_y_table_from_neo(
                cs,
                &child.y,
                &format!("step_{}_rlc_child_{}_y", step_idx, i),
            )?);
        }

        for j in 0..t {
            if parent.y[j].len() != d_pad {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC parent y[j] length mismatch at step {}, j={}",
                    step_idx, j
                )));
            }
            for r_idx in 0..d_ring {
                // c0 component
                cs.enforce(
                    || format!("step_{}_rlc_y_c0_j{}_r{}", step_idx, j, r_idx),
                    |lc| {
                        let mut res = lc;
                        for (i, rho) in rhos.iter().enumerate() {
                            for k in 0..d_ring {
                                let coeff = Self::neo_f_to_circuit(&rho[(r_idx, k)]);
                                if coeff != CircuitF::from(0u64) {
                                    res = res + (coeff, children_y_vars[i][j][k].c0);
                                }
                            }
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_y_vars[j][r_idx].c0,
                );

                // c1 component
                cs.enforce(
                    || format!("step_{}_rlc_y_c1_j{}_r{}", step_idx, j, r_idx),
                    |lc| {
                        let mut res = lc;
                        for (i, rho) in rhos.iter().enumerate() {
                            for k in 0..d_ring {
                                let coeff = Self::neo_f_to_circuit(&rho[(r_idx, k)]);
                                if coeff != CircuitF::from(0u64) {
                                    res = res + (coeff, children_y_vars[i][j][k].c1);
                                }
                            }
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_y_vars[j][r_idx].c1,
                );
            }
        }

        Ok(())
    }

    /// Verify DEC equalities for a step
    ///
    /// Enforces the public DEC relations (ignoring commitments for now):
    /// - r is preserved: parent.r == child.r for all children
    /// - X_parent = Σ_i b^i · X_i
    /// - y_parent[j] = Σ_i b^i · y_(i,j)
    fn verify_dec<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        step: &FoldStep,
        _witness: &FoldRunWitness,
    ) -> Result<()> {
        let parent = &step.rlc_parent;
        let children = &step.dec_children;

        if children.is_empty() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "DEC at step {} has no children",
                step_idx
            )));
        }

        let d = parent.X.rows();
        let m_in = parent.m_in;

        for (i, child) in children.iter().enumerate() {
            if child.X.rows() != d || child.X.cols() != parent.X.cols() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "DEC X dimension mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
            if child.m_in != m_in {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "DEC m_in mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
        }

        // Allocate X matrices
        let parent_X_vars = self.alloc_matrix_from_neo(
            cs,
            &parent.X,
            &format!("step_{}_dec_parent_X", step_idx),
        )?;

        let mut child_X_vars = Vec::with_capacity(children.len());
        for (i, child) in children.iter().enumerate() {
            child_X_vars.push(self.alloc_matrix_from_neo(
                cs,
                &child.X,
                &format!("step_{}_dec_child_{}_X", step_idx, i),
            )?);
        }

        // X_parent = Σ b^i · X_i
        for row in 0usize..d {
            for col in 0usize..m_in {
                cs.enforce(
                    || format!("step_{}_dec_X_{}_{}", step_idx, row, col),
                    |lc| {
                        let mut res = lc;
                        let mut pow = CircuitF::from(1u64);
                        for child_vars in child_X_vars.iter() {
                            res = res + (pow, child_vars[row][col]);
                            pow *= CircuitF::from(self.base_b as u64);
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_X_vars[row][col],
                );
            }
        }

        // r preservation: parent.r == child.r for all children
        let r_len = parent.r.len();
        for (i, child) in children.iter().enumerate() {
            if child.r.len() != r_len {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "DEC r length mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
            for idx in 0..r_len {
                let parent_r_var = self.alloc_k_from_neo(
                    cs,
                    parent.r[idx],
                    &format!("step_{}_dec_parent_r_{}_{}", step_idx, i, idx),
                )?;
                let child_r_var = self.alloc_k_from_neo(
                    cs,
                    child.r[idx],
                    &format!("step_{}_dec_child_{}_r_{}", step_idx, i, idx),
                )?;
                self.enforce_k_eq(
                    cs,
                    &parent_r_var,
                    &child_r_var,
                    &format!("step_{}_dec_r_eq_child_{}_idx_{}", step_idx, i, idx),
                );
            }
        }

        // y_parent[j] = Σ b^i · y_(i,j)
        let t = parent.y.len();
        if t == 0 {
            return Ok(());
        }
        let d_pad = parent.y[0].len();
        if d_pad == 0 {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "DEC at step {}: empty y vectors in parent",
                step_idx
            )));
        }

        let parent_y_vars = self.alloc_y_table_from_neo(
            cs,
            &parent.y,
            &format!("step_{}_dec_parent_y", step_idx),
        )?;

        let mut children_y_vars = Vec::with_capacity(children.len());
        for (i, child) in children.iter().enumerate() {
            if child.y.len() != t {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "DEC y length mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
            for y_j in &child.y {
                if y_j.len() != d_pad {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "DEC child y[j] length mismatch at step {}, child {}",
                        step_idx, i
                    )));
                }
            }
            children_y_vars.push(self.alloc_y_table_from_neo(
                cs,
                &child.y,
                &format!("step_{}_dec_child_{}_y", step_idx, i),
            )?);
        }

        for j in 0..t {
            for r_idx in 0..d_pad {
                // c0 component
                cs.enforce(
                    || format!("step_{}_dec_y_c0_j{}_r{}", step_idx, j, r_idx),
                    |lc| {
                        let mut res = lc;
                        let mut pow = CircuitF::from(1u64);
                        for child_y in children_y_vars.iter() {
                            res = res + (pow, child_y[j][r_idx].c0);
                            pow *= CircuitF::from(self.base_b as u64);
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_y_vars[j][r_idx].c0,
                );

                // c1 component
                cs.enforce(
                    || format!("step_{}_dec_y_c1_j{}_r{}", step_idx, j, r_idx),
                    |lc| {
                        let mut res = lc;
                        let mut pow = CircuitF::from(1u64);
                        for child_y in children_y_vars.iter() {
                            res = res + (pow, child_y[j][r_idx].c1);
                            pow *= CircuitF::from(self.base_b as u64);
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_y_vars[j][r_idx].c1,
                );
            }
        }

        Ok(())
    }

    /// Verify accumulator chaining across steps.
    ///
    /// Currently enforces that the final accumulator claimed in the public
    /// instance matches the `FoldRun`'s final outputs. Full cross-step chaining
    /// (linking DEC children to the next step's inputs) is left for future work
    /// once those inputs are explicitly exposed in the witness.
    fn verify_accumulator_chaining<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        witness: &FoldRunWitness,
    ) -> Result<()> {
        let final_public = &self.instance.final_accumulator;
        let final_witness = &witness.fold_run.final_outputs;

        if final_public.len() != final_witness.len() {
            return Err(SpartanBridgeError::InvalidInput(
                "Final accumulator length mismatch".into(),
            ));
        }

        for (idx, (pub_me, wit_me)) in final_public.iter().zip(final_witness.iter()).enumerate() {
            // Enforce X equality
            if pub_me.X.rows() != wit_me.X.rows() || pub_me.X.cols() != wit_me.X.cols() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "Final accumulator X dimension mismatch at index {}",
                    idx
                )));
            }

            let pub_X_vars = self.alloc_matrix_from_neo(
                cs,
                &pub_me.X,
                &format!("acc_final_pub_{}_X", idx),
            )?;
            let wit_X_vars = self.alloc_matrix_from_neo(
                cs,
                &wit_me.X,
                &format!("acc_final_wit_{}_X", idx),
            )?;

            for r in 0..pub_me.X.rows() {
                for c in 0..pub_me.X.cols() {
                    cs.enforce(
                        || format!("acc_final_X_eq_{}_{}_{}", idx, r, c),
                        |lc| lc + pub_X_vars[r][c],
                        |lc| lc + CS::one(),
                        |lc| lc + wit_X_vars[r][c],
                    );
                }
            }

            // Enforce r equality
            if pub_me.r.len() != wit_me.r.len() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "Final accumulator r length mismatch at index {}",
                    idx
                )));
            }
            for j in 0..pub_me.r.len() {
                let pub_r = self.alloc_k_from_neo(
                    cs,
                    pub_me.r[j],
                    &format!("acc_final_pub_{}_r_{}", idx, j),
                )?;
                let wit_r = self.alloc_k_from_neo(
                    cs,
                    wit_me.r[j],
                    &format!("acc_final_wit_{}_r_{}", idx, j),
                )?;
                self.enforce_k_eq(
                    cs,
                    &pub_r,
                    &wit_r,
                    &format!("acc_final_r_eq_{}_{}", idx, j),
                );
            }

            // Enforce y equality
            if pub_me.y.len() != wit_me.y.len() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "Final accumulator y length mismatch at index {}",
                    idx
                )));
            }

            let pub_y_vars = self.alloc_y_table_from_neo(
                cs,
                &pub_me.y,
                &format!("acc_final_pub_{}_y", idx),
            )?;
            let wit_y_vars = self.alloc_y_table_from_neo(
                cs,
                &wit_me.y,
                &format!("acc_final_wit_{}_y", idx),
            )?;

            for j in 0..pub_me.y.len() {
                if pub_me.y[j].len() != wit_me.y[j].len() {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "Final accumulator y[j] length mismatch at index {}, j={}",
                        idx, j
                    )));
                }
                for r_idx in 0..pub_me.y[j].len() {
                    self.enforce_k_eq(
                        cs,
                        &pub_y_vars[j][r_idx],
                        &wit_y_vars[j][r_idx],
                        &format!("acc_final_y_eq_{}_{}_{}", idx, j, r_idx),
                    );
                }
            }
        }

        Ok(())
    }

    /// Helper: convert Neo base-field element to circuit field
    fn neo_f_to_circuit(f: &NeoF) -> CircuitF {
        CircuitF::from(f.as_canonical_u64())
    }

    /// Helper: allocate a K element from neo_math::K
    fn alloc_k_from_neo<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        k: neo_math::K,
        label: &str,
    ) -> Result<KNumVar> {
        let k_num = KNum::<CircuitF>::from_neo_k(k);
        alloc_k(cs, Some(k_num), label)
            .map_err(SpartanBridgeError::BellpepperError)
    }

    /// Helper: allocate a dense matrix of NeoF as circuit variables.
    fn alloc_matrix_from_neo<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        mat: &Mat<NeoF>,
        label: &str,
    ) -> Result<Vec<Vec<Variable>>> {
        let rows = mat.rows();
        let cols = mat.cols();
        let mut vars = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row_vars = Vec::with_capacity(cols);
            for c in 0..cols {
                let value = Self::neo_f_to_circuit(&mat[(r, c)]);
                let var = cs.alloc(
                    || format!("{}_{}_{}", label, r, c),
                    || Ok(value),
                )?;
                row_vars.push(var);
            }
            vars.push(row_vars);
        }
        Ok(vars)
    }

    /// Helper: allocate a table of K elements (y-vectors) from neo_math::K.
    fn alloc_y_table_from_neo<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        y: &[Vec<neo_math::K>],
        label: &str,
    ) -> Result<Vec<Vec<KNumVar>>> {
        let mut table = Vec::with_capacity(y.len());
        for (j, row) in y.iter().enumerate() {
            let mut row_vars = Vec::with_capacity(row.len());
            for (idx, k_val) in row.iter().enumerate() {
                let var = self.alloc_k_from_neo(
                    cs,
                    *k_val,
                    &format!("{}_{}_{}", label, j, idx),
                )?;
                row_vars.push(var);
            }
            table.push(row_vars);
        }
        Ok(table)
    }

    /// Helper: enforce equality of two KNumVars.
    fn enforce_k_eq<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        a: &KNumVar,
        b: &KNumVar,
        label: &str,
    ) {
        cs.enforce(
            || format!("{}_c0", label),
            |lc| lc + a.c0,
            |lc| lc + CS::one(),
            |lc| lc + b.c0,
        );
        cs.enforce(
            || format!("{}_c1", label),
            |lc| lc + a.c1,
            |lc| lc + CS::one(),
            |lc| lc + b.c1,
        );
    }

    /// Helper: allocate a constant K element from a base-field value.
    fn k_const<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        c0: CircuitF,
        label: &str,
    ) -> Result<KNumVar> {
        let k_num = KNum::<CircuitF>::from_f(c0);
        alloc_k(cs, Some(k_num), label).map_err(SpartanBridgeError::BellpepperError)
    }

    /// Helper: K zero.
    fn k_zero<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        label: &str,
    ) -> Result<KNumVar> {
        self.k_const(cs, CircuitF::from(0u64), label)
    }

    /// Helper: K one.
    fn k_one<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        label: &str,
    ) -> Result<KNumVar> {
        self.k_const(cs, CircuitF::from(1u64), label)
    }

    /// Helper: K addition: r = a + b.
    fn k_add<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        a: &KNumVar,
        b: &KNumVar,
        label: &str,
    ) -> Result<KNumVar> {
        k_add_raw(cs, a, b, None, label).map_err(SpartanBridgeError::BellpepperError)
    }

    /// Helper: K multiplication: r = a * b.
    fn k_mul<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        a: &KNumVar,
        b: &KNumVar,
        label: &str,
    ) -> Result<KNumVar> {
        k_mul_raw(cs, a, b, self.delta, None, label).map_err(SpartanBridgeError::BellpepperError)
    }

    /// Helper: K scalar multiplication: r = k * a.
    fn k_scalar_mul<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        k: CircuitF,
        a: &KNumVar,
        label: &str,
    ) -> Result<KNumVar> {
        k_scalar_mul_raw(cs, k, a, None, label).map_err(SpartanBridgeError::BellpepperError)
    }

    /// Helper: K subtraction: r = a - b.
    fn k_sub<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        a: &KNumVar,
        b: &KNumVar,
        label: &str,
    ) -> Result<KNumVar> {
        // Compute -b via scalar multiplication by -1, then add.
        let minus_one = CircuitF::from(0u64) - CircuitF::from(1u64);
        let neg_b = self.k_scalar_mul(
            cs,
            minus_one,
            b,
            &format!("{}_neg_b", label),
        )?;
        self.k_add(cs, a, &neg_b, label)
    }

    /// Helper: 1 - a.
    fn k_one_minus<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        a: &KNumVar,
        label: &str,
    ) -> Result<KNumVar> {
        let one = self.k_one(cs, &format!("{}_one", label))?;
        self.k_sub(cs, &one, a, &format!("{}_1_minus", label))
    }

    /// Helper: K exponentiation by small integer exponent: base^exp.
    fn k_pow<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        base: &KNumVar,
        exp: u32,
        label: &str,
    ) -> Result<KNumVar> {
        if exp == 0 {
            return self.k_one(cs, &format!("{}_pow0", label));
        }
        let mut acc = base.clone();
        for i in 1..exp {
            acc = self.k_mul(
                cs,
                &acc,
                base,
                &format!("{}_pow_step_{}", label, i),
            )?;
        }
        Ok(acc)
    }
}

/// Implement Spartan2's `SpartanCircuit` trait for `FoldRunCircuit` using the
/// Goldilocks + Hash-MLE PCS engine. This lets Spartan2 treat the FoldRun
/// circuit as an R1CS provider.
impl SpartanCircuitTrait<GoldilocksP3MerkleMleEngine> for FoldRunCircuit {
    fn public_values(&self) -> std::result::Result<Vec<CircuitF>, SynthesisError> {
        // Must mirror `allocate_public_inputs`: expose params/CCS/MCS digests as
        // 4 little-endian u64 limbs each, in that order.
        fn append_digest(out: &mut Vec<CircuitF>, digest: &[u8; 32]) {
            for chunk in digest.chunks(8) {
                let mut limb_bytes = [0u8; 8];
                limb_bytes.copy_from_slice(chunk);
                let limb_u64 = u64::from_le_bytes(limb_bytes);
                out.push(CircuitF::from(limb_u64));
            }
        }

        let mut vals = Vec::with_capacity(12);
        append_digest(&mut vals, &self.instance.params_digest);
        append_digest(&mut vals, &self.instance.ccs_digest);
        append_digest(&mut vals, &self.instance.mcs_digest);
        Ok(vals)
    }

    fn shared<CS: ConstraintSystem<CircuitF>>(
        &self,
        _cs: &mut CS,
    ) -> std::result::Result<Vec<AllocatedNum<CircuitF>>, SynthesisError> {
        // This circuit does not use "shared" variables across multiple Spartan
        // circuits; all variables are allocated inside `synthesize`.
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<CircuitF>>(
        &self,
        _cs: &mut CS,
        _shared: &[AllocatedNum<CircuitF>],
    ) -> std::result::Result<Vec<AllocatedNum<CircuitF>>, SynthesisError> {
        // We do not distinguish precommitted variables; everything is handled
        // directly in `synthesize`.
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        // Spartan's transcript-driven "challenges" are not used by this
        // circuit; all randomness comes from the Neo folding transcript
        // (already baked into `FoldRunInstance` / `FoldRunWitness`).
        0
    }

    fn synthesize<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        _shared: &[AllocatedNum<CircuitF>],
        _precommitted: &[AllocatedNum<CircuitF>],
        _challenges: Option<&[CircuitF]>,
    ) -> std::result::Result<(), SynthesisError> {
        // Delegate to the existing FoldRunCircuit::synthesize and map any
        // bridge errors back into a SynthesisError understood by Spartan2.
        self.synthesize(cs).map_err(|e| match e {
            SpartanBridgeError::BellpepperError(inner) => inner,
            // For higher-level bridge errors (invalid input, etc.), treat them
            // as unsatisfiable circuits from the SNARK's perspective.
            _ => SynthesisError::Unsatisfiable,
        })
    }
}
