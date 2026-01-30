//! Main FoldRun circuit implementation
//!
//! This synthesizes R1CS constraints to verify an entire FoldRun:
//! - For each fold step:
//!   - Verify Π-CCS terminal identity
//!   - Verify sumcheck rounds
//!   - Verify RLC equalities
//!   - Verify DEC equalities
//! - Verify accumulator chaining between steps

use crate::circuit::witness::{FoldRunInstance, FoldRunWitness};
use crate::error::{Result, SpartanBridgeError};
use crate::gadgets::k_field::{alloc_k, k_add as k_add_raw, KNum, KNumVar};
use crate::gadgets::pi_ccs::{sumcheck_eval_gadget, sumcheck_round_gadget};
use crate::CircuitF;
use bellpepper_core::{ConstraintSystem, SynthesisError};
use neo_ccs::Mat;
use neo_fold::shard::FoldStep;
use neo_math::F as NeoF;
use p3_field::PrimeCharacteristicRing;

// Import helper functions from separate module
use super::fold_circuit_helpers as helpers;

// Spartan2 integration: implement SpartanCircuit over Goldilocks + Hash-MLE PCS.
use bellpepper_core::num::AllocatedNum;
use spartan2::provider::GoldilocksP3MerkleMleEngine;
use spartan2::traits::circuit::SpartanCircuit as SpartanCircuitTrait;

/// Sparse representation of the CCS polynomial f in the circuit field.
///
/// Each term is coeff * ∏_j m_j^{exps[j]}.
#[derive(Clone, Debug)]
pub struct CircuitPolyTerm {
    /// Coefficient in the circuit field (Spartan's Goldilocks).
    pub coeff: CircuitF,
    /// Same coefficient in Neo's base field, for native K computations.
    pub coeff_native: NeoF,
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
    pub fn synthesize<CS: ConstraintSystem<CircuitF>>(&self, cs: &mut CS) -> Result<()> {
        // Allocate public inputs
        self.allocate_public_inputs(cs)?;

        // Get witness or error
        let witness = self
            .witness
            .as_ref()
            .ok_or_else(|| SpartanBridgeError::InvalidInput("Missing witness".into()))?;

        // Verify each fold step
        for (step_idx, step) in witness.fold_run.steps.iter().enumerate() {
            self.verify_fold_step(cs, step_idx, &step.fold, witness)?;
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
    ///
    /// The full accumulator and Π-CCS challenges remain private witness data for now.
    fn allocate_public_inputs<CS: ConstraintSystem<CircuitF>>(&self, cs: &mut CS) -> Result<()> {
        // Helper to expose a 32-byte digest as 8 field elements by chunking into u32.
        let mut alloc_digest = |label: &str, digest: &[u8; 32]| -> Result<()> {
            for (i, chunk) in digest.chunks(4).enumerate() {
                let mut limb_bytes = [0u8; 4];
                limb_bytes.copy_from_slice(chunk);
                let limb_u32 = u32::from_le_bytes(limb_bytes);
                let value = CircuitF::from(limb_u32 as u64);
                // Allocate as public input.
                let _ = cs.alloc_input(|| format!("{}_limb_{}", label, i), || Ok(value))?;
            }
            Ok(())
        };

        alloc_digest("params_digest", &self.instance.params_digest)?;
        alloc_digest("ccs_digest", &self.instance.ccs_digest)?;

        Ok(())
    }

    /// Verify a single fold step
    fn verify_fold_step<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        step: &FoldStep,
        witness: &FoldRunWitness,
    ) -> Result<()> {
        // Get the Π-CCS proof and challenges for this step.
        let pi_ccs_proof = &step.ccs_proof;
        let challenges = &self.instance.pi_ccs_challenges[step_idx];

        if pi_ccs_proof.variant != neo_reductions::optimized_engine::PiCcsProofVariant::SplitNcV1 {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "FoldRunCircuit currently requires SplitNcV1 Π-CCS proofs (got {:?})",
                pi_ccs_proof.variant
            )));
        }

        // Allocate all per-step challenges once so all sub-gadgets are forced to
        // use the same variables (no duplicated allocations with unconstrained equality).
        let mut alpha_vars = Vec::with_capacity(challenges.alpha.len());
        for (i, &val) in challenges.alpha.iter().enumerate() {
            alpha_vars.push(helpers::alloc_k_from_neo(
                cs,
                val,
                &format!("step_{}_alpha_{}", step_idx, i),
            )?);
        }

        let mut beta_a_vars = Vec::with_capacity(challenges.beta_a.len());
        for (i, &val) in challenges.beta_a.iter().enumerate() {
            beta_a_vars.push(helpers::alloc_k_from_neo(
                cs,
                val,
                &format!("step_{}_beta_a_{}", step_idx, i),
            )?);
        }

        let mut beta_r_vars = Vec::with_capacity(challenges.beta_r.len());
        for (i, &val) in challenges.beta_r.iter().enumerate() {
            beta_r_vars.push(helpers::alloc_k_from_neo(
                cs,
                val,
                &format!("step_{}_beta_r_{}", step_idx, i),
            )?);
        }

        let mut beta_m_vars = Vec::with_capacity(challenges.beta_m.len());
        for (i, &val) in challenges.beta_m.iter().enumerate() {
            beta_m_vars.push(helpers::alloc_k_from_neo(
                cs,
                val,
                &format!("step_{}_beta_m_{}", step_idx, i),
            )?);
        }

        let gamma_var = helpers::alloc_k_from_neo(cs, challenges.gamma, &format!("step_{}_gamma", step_idx))?;

        let mut sumcheck_chal_vars = Vec::with_capacity(challenges.sumcheck_challenges.len());
        for (i, &val) in challenges.sumcheck_challenges.iter().enumerate() {
            sumcheck_chal_vars.push(helpers::alloc_k_from_neo(
                cs,
                val,
                &format!("step_{}_sumcheck_chal_{}", step_idx, i),
            )?);
        }

        let mut sumcheck_chal_nc_vars = Vec::with_capacity(challenges.sumcheck_challenges_nc.len());
        for (i, &val) in challenges.sumcheck_challenges_nc.iter().enumerate() {
            sumcheck_chal_nc_vars.push(helpers::alloc_k_from_neo(
                cs,
                val,
                &format!("step_{}_sumcheck_nc_chal_{}", step_idx, i),
            )?);
        }

        // 1a. Bind the public claimed initial sum T from ME inputs and challenges.
        let initial_sum_var =
            self.verify_initial_sum_binding(cs, step_idx, pi_ccs_proof, challenges, witness, &alpha_vars, &gamma_var)?;

        // 1b. Run sumcheck rounds algebra starting from T to obtain the final running sum.
        let sumcheck_final_var = self.verify_sumcheck_rounds(
            cs,
            step_idx,
            pi_ccs_proof,
            challenges,
            &sumcheck_chal_vars,
            &initial_sum_var,
        )?;

        // Pre-allocate y-tables and y_zcol vectors for Π-CCS outputs once so they can be shared
        // between terminal-identity, NC, and RLC gadgets.
        let mut ccs_out_y_vars: Vec<Vec<Vec<KNumVar>>> = Vec::with_capacity(step.ccs_out.len());
        let mut ccs_out_y_zcol_vars: Vec<Vec<KNumVar>> = Vec::with_capacity(step.ccs_out.len());
        for (i, child) in step.ccs_out.iter().enumerate() {
            let y_table =
                helpers::alloc_y_table_from_neo(cs, &child.y, &format!("step_{}_ccs_out_child_{}_y", step_idx, i))?;
            ccs_out_y_vars.push(y_table);

            let mut yz_vars = Vec::with_capacity(child.y_zcol.len());
            for (idx, &val) in child.y_zcol.iter().enumerate() {
                yz_vars.push(helpers::alloc_k_from_neo(
                    cs,
                    val,
                    &format!("step_{}_ccs_out_child_{}_y_zcol_{}", step_idx, i, idx),
                )?);
            }
            ccs_out_y_zcol_vars.push(yz_vars);
        }

        // 1c. Verify Π-CCS FE terminal identity: final running sum == RHS_fe(α,β,γ,r',α', out_me, inputs.r).
        self.verify_terminal_identity(
            cs,
            step_idx,
            step,
            pi_ccs_proof,
            challenges,
            witness,
            &sumcheck_final_var,
            &ccs_out_y_vars,
            &alpha_vars,
            &beta_a_vars,
            &beta_r_vars,
            &gamma_var,
            &sumcheck_chal_vars,
        )?;

        // 1d. Verify Π-CCS NC sumcheck + NC terminal identity (split-NC variant).
        let initial_sum_nc_var = helpers::k_zero(cs, &format!("step_{}_nc_initial_sum_zero", step_idx))?;
        if let Some(sc_initial_nc) = pi_ccs_proof.sc_initial_sum_nc {
            let sc_initial_nc_var =
                helpers::alloc_k_from_neo(cs, sc_initial_nc, &format!("step_{}_sc_initial_sum_nc", step_idx))?;
            helpers::enforce_k_eq(
                cs,
                &sc_initial_nc_var,
                &initial_sum_nc_var,
                &format!("step_{}_nc_initial_sum_matches_zero", step_idx),
            );
        }
        let sumcheck_final_nc_var = self.verify_sumcheck_rounds_nc(
            cs,
            step_idx,
            pi_ccs_proof,
            challenges,
            &sumcheck_chal_nc_vars,
            &initial_sum_nc_var,
        )?;
        self.verify_terminal_identity_nc(
            cs,
            step_idx,
            step,
            pi_ccs_proof,
            challenges,
            &sumcheck_final_nc_var,
            &ccs_out_y_zcol_vars,
            &beta_a_vars,
            &beta_m_vars,
            &gamma_var,
            &sumcheck_chal_nc_vars,
        )?;

        // 2. Verify RLC equalities
        self.verify_rlc(cs, step_idx, step, witness, &ccs_out_y_vars, &ccs_out_y_zcol_vars)?;

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
        sumcheck_challenges: &[KNumVar],
        initial_sum: &KNumVar,
    ) -> Result<KNumVar> {
        // Sanity: number of rounds in the proof must match the number of
        // advertised challenges for this step.
        if proof.sumcheck_rounds.len() != challenges.sumcheck_challenges.len()
            || proof.sumcheck_rounds.len() != sumcheck_challenges.len()
        {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "Sumcheck rounds/challenges length mismatch at step {}: rounds={}, challenges={}",
                step_idx,
                proof.sumcheck_rounds.len(),
                challenges.sumcheck_challenges.len()
            )));
        }

        // Start running sum from the in-circuit initial sum variable (T).
        let mut claimed_sum = initial_sum.clone();

        // For each sumcheck round
        for (round_idx, round_poly) in proof.sumcheck_rounds.iter().enumerate() {
            // Allocate polynomial coefficients
            let mut coeffs = Vec::new();
            for (coeff_idx, coeff) in round_poly.iter().enumerate() {
                let coeff_var = helpers::alloc_k_from_neo(
                    cs,
                    *coeff,
                    &format!("step_{}_round_{}_coeff_{}", step_idx, round_idx, coeff_idx),
                )
                .map_err(|e| SpartanBridgeError::SynthesisError(format!("{:?}", e)))?;
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
            )
            .map_err(|e| SpartanBridgeError::from(e))?;

            let challenge = &sumcheck_challenges[round_idx];

            // Compute next claimed sum = p(challenge) via gadget (returns KNumVar)
            let next_sum = sumcheck_eval_gadget(
                cs,
                &coeffs,
                round_poly,
                challenge,
                challenges.sumcheck_challenges[round_idx],
                self.delta,
                &format!("step_{}_round_{}_eval", step_idx, round_idx),
            )
            .map_err(|e| SpartanBridgeError::from(e))?;

            // Update claimed_sum for next round
            claimed_sum = next_sum;
        }

        // After all rounds, optionally check that the prover's scalar
        // sumcheck_final matches the in-circuit running sum. This is a
        // consistency check only; the algebraic binding is via the KNumVar
        // returned from this method.
        let final_sum_expected =
            helpers::alloc_k_from_neo(cs, proof.sumcheck_final, &format!("step_{}_final_sum", step_idx))?;

        helpers::enforce_k_eq(
            cs,
            &claimed_sum,
            &final_sum_expected,
            &format!("step_{}_final_sum_matches_scalar", step_idx),
        );

        Ok(claimed_sum)
    }

    /// Verify NC sumcheck rounds for a step (SplitNcV1 only).
    fn verify_sumcheck_rounds_nc<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        proof: &neo_reductions::PiCcsProof,
        challenges: &crate::circuit::witness::PiCcsChallenges,
        sumcheck_challenges_nc: &[KNumVar],
        initial_sum: &KNumVar,
    ) -> Result<KNumVar> {
        if proof.sumcheck_rounds_nc.len() != challenges.sumcheck_challenges_nc.len()
            || proof.sumcheck_rounds_nc.len() != sumcheck_challenges_nc.len()
        {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "NC sumcheck rounds/challenges length mismatch at step {}: rounds={}, challenges={}",
                step_idx,
                proof.sumcheck_rounds_nc.len(),
                challenges.sumcheck_challenges_nc.len()
            )));
        }

        let mut claimed_sum = initial_sum.clone();
        for (round_idx, round_poly) in proof.sumcheck_rounds_nc.iter().enumerate() {
            let mut coeffs = Vec::new();
            for (coeff_idx, coeff) in round_poly.iter().enumerate() {
                let coeff_var = helpers::alloc_k_from_neo(
                    cs,
                    *coeff,
                    &format!("step_{}_nc_round_{}_coeff_{}", step_idx, round_idx, coeff_idx),
                )
                .map_err(|e| SpartanBridgeError::SynthesisError(format!("{:?}", e)))?;
                coeffs.push(coeff_var);
            }

            sumcheck_round_gadget(
                cs,
                &coeffs,
                round_poly,
                &claimed_sum,
                self.delta,
                &format!("step_{}_nc_round_{}", step_idx, round_idx),
            )
            .map_err(SpartanBridgeError::from)?;

            let challenge = &sumcheck_challenges_nc[round_idx];

            let next_sum = sumcheck_eval_gadget(
                cs,
                &coeffs,
                round_poly,
                challenge,
                challenges.sumcheck_challenges_nc[round_idx],
                self.delta,
                &format!("step_{}_nc_round_{}_eval", step_idx, round_idx),
            )
            .map_err(SpartanBridgeError::from)?;

            claimed_sum = next_sum;
        }

        let final_sum_expected =
            helpers::alloc_k_from_neo(cs, proof.sumcheck_final_nc, &format!("step_{}_nc_final_sum", step_idx))?;
        helpers::enforce_k_eq(
            cs,
            &claimed_sum,
            &final_sum_expected,
            &format!("step_{}_nc_final_sum_matches_scalar", step_idx),
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
        alpha_vars: &[KNumVar],
        gamma_var: &KNumVar,
    ) -> Result<KNumVar> {
        // Sanity: the caller must provide exactly one allocation for each
        // logical challenge.
        if alpha_vars.len() != challenges.alpha.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "alpha length mismatch at step {}: vars={}, values={}",
                step_idx,
                alpha_vars.len(),
                challenges.alpha.len()
            )));
        }

        // 2. Select ME inputs for this step (same choice as the native
        //    verifier): step 0 uses the public initial accumulator; later
        //    steps use DEC children from the previous step.
        let me_inputs = if step_idx == 0 {
            &self.instance.initial_accumulator
        } else {
            &witness.fold_run.steps[step_idx - 1].fold.dec_children
        };

        // 3. Allocate the ME inputs' y-tables as KNumVar arrays so they can
        //    be fed into the in-circuit claimed_initial_sum_gadget.
        let mut me_inputs_y_vars: Vec<Vec<Vec<KNumVar>>> = Vec::with_capacity(me_inputs.len());
        let mut me_inputs_y_vals: Vec<Vec<Vec<neo_math::K>>> = Vec::with_capacity(me_inputs.len());
        for (i, input) in me_inputs.iter().enumerate() {
            let y_table =
                helpers::alloc_y_table_from_neo(cs, &input.y, &format!("step_{}_me_input_{}_y", step_idx, i))?;
            me_inputs_y_vars.push(y_table);
            me_inputs_y_vals.push(input.y.clone());
        }

        // 4. Derive T in-circuit from (α, γ, ME y) using the gadget version
        //    of `claimed_initial_sum_from_inputs`.
        let t_var = self.claimed_initial_sum_gadget(
            cs,
            step_idx,
            alpha_vars,
            &challenges.alpha,
            gamma_var,
            challenges.gamma,
            &me_inputs_y_vars,
            &me_inputs_y_vals,
        )?;

        // 5. If the proof provided a scalar sc_initial_sum, enforce that it
        //    matches the in-circuit T. This mirrors the native verifier's
        //    optional tightness check.
        if let Some(sc_initial) = proof.sc_initial_sum {
            let sc_initial_var =
                helpers::alloc_k_from_neo(cs, sc_initial, &format!("step_{}_sc_initial_sum_binding", step_idx))?;

            helpers::enforce_k_eq(
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
        alpha_vars: &[KNumVar],
        alpha_vals: &[neo_math::K],
        gamma_var: &KNumVar,
        gamma_val: neo_math::K,
        me_inputs_y_vars: &[Vec<Vec<KNumVar>>],
        me_inputs_y_vals: &[Vec<Vec<neo_math::K>>],
    ) -> Result<KNumVar> {
        use core::cmp::min;
        use neo_math::K as NeoK;

        if alpha_vars.len() != alpha_vals.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "claimed_initial_sum_gadget alpha length mismatch at step {}: vars={}, vals={}",
                step_idx,
                alpha_vars.len(),
                alpha_vals.len()
            )));
        }
        if me_inputs_y_vars.len() != me_inputs_y_vals.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "claimed_initial_sum_gadget me_inputs length mismatch at step {}: vars={}, vals={}",
                step_idx,
                me_inputs_y_vars.len(),
                me_inputs_y_vals.len()
            )));
        }

        let k_total = 1 + me_inputs_y_vars.len(); // 1 MCS + |ME|
        if k_total < 2 {
            // No Eval block when k=1 → T = 0.
            return helpers::k_zero(cs, &format!("step_{}_T_zero", step_idx));
        }

        // Build χ_α over Ajtai domain.
        let d_sz = 1usize << alpha_vars.len();
        let mut chi_alpha_vars: Vec<KNumVar> = Vec::with_capacity(d_sz);
        let mut chi_alpha_vals: Vec<NeoK> = Vec::with_capacity(d_sz);

        // χ_α[ρ] = ∏_bit (α_bit if ρ_bit=1 else 1-α_bit)
        for rho in 0..d_sz {
            // Start with w = 1 in both the native and in-circuit representation.
            let mut w_val = NeoK::ONE;
            let mut w_var = helpers::k_one(cs, &format!("step_{}_chi_alpha_{}_init", step_idx, rho))?;

            for (bit, (a_var, &a_val)) in alpha_vars.iter().zip(alpha_vals.iter()).enumerate() {
                let bit_is_one = ((rho >> bit) & 1) == 1;
                // Factor value in K.
                let (factor_var, factor_val) = if bit_is_one {
                    (a_var.clone(), a_val)
                } else {
                    // factor = 1 - α_bit, enforced via an explicit relation
                    // factor + α_bit = 1 in K.
                    let factor_val = NeoK::ONE - a_val;
                    let factor_hint = KNum::<CircuitF>::from_neo_k(factor_val);
                    let factor_var = alloc_k(
                        cs,
                        Some(factor_hint),
                        &format!("step_{}_chi_alpha_{}_bit{}_factor", step_idx, rho, bit),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;

                    // one_var with explicit K value 1.
                    let one_var = helpers::k_one(cs, &format!("step_{}_chi_alpha_{}_bit{}_one", step_idx, rho, bit))?;

                    // sum = factor + α_bit with native hint, enforce sum == 1.
                    let sum_val = factor_val + a_val;
                    let sum_hint = KNum::<CircuitF>::from_neo_k(sum_val);
                    let sum_var = k_add_raw(
                        cs,
                        &factor_var,
                        a_var,
                        Some(sum_hint),
                        &format!("step_{}_chi_alpha_{}_bit{}_sum", step_idx, rho, bit),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                    helpers::enforce_k_eq(
                        cs,
                        &sum_var,
                        &one_var,
                        &format!("step_{}_chi_alpha_{}_bit{}_one_minus", step_idx, rho, bit),
                    );
                    (factor_var, factor_val)
                };

                // w <- w * factor with native hint.
                let (new_w_var, new_w_val) = helpers::k_mul_with_hint(
                    cs,
                    &w_var,
                    w_val,
                    &factor_var,
                    factor_val,
                    self.delta,
                    &format!("step_{}_chi_alpha_{}_bit{}", step_idx, rho, bit),
                )?;
                w_var = new_w_var;
                w_val = new_w_val;
            }

            chi_alpha_vars.push(w_var);
            chi_alpha_vals.push(w_val);
        }

        // γ^k_total (used in the weights): computed in-circuit so it is bound to γ.
        let mut gamma_to_k = helpers::k_one(cs, &format!("step_{}_T_gamma_to_k_init", step_idx))?;
        let mut gamma_to_k_val = NeoK::ONE;
        for pow_idx in 0..k_total {
            let (new_gamma_to_k, new_gamma_to_k_val) = helpers::k_mul_with_hint(
                cs,
                &gamma_to_k,
                gamma_to_k_val,
                gamma_var,
                gamma_val,
                self.delta,
                &format!("step_{}_T_gamma_to_k_mul_{}", step_idx, pow_idx),
            )?;
            gamma_to_k = new_gamma_to_k;
            gamma_to_k_val = new_gamma_to_k_val;
        }

        // Inner weighted sum over (j, i>=2).
        let t = if me_inputs_y_vars.is_empty() {
            0
        } else {
            me_inputs_y_vars[0].len()
        };

        let mut inner_val = NeoK::ZERO;
        let mut inner = helpers::k_zero(cs, &format!("step_{}_T_inner_init", step_idx))?;

        for j in 0..t {
            // (γ^k_total)^j – shared across all i for this j, tracked natively and then lifted.
            let mut gamma_k_j_val = NeoK::ONE;
            let mut gamma_k_j = helpers::k_one(cs, &format!("step_{}_T_gamma_k_j_init_j{}", step_idx, j))?;
            for pow_idx in 0..j {
                let (new_gamma_k_j, new_gamma_k_j_val) = helpers::k_mul_with_hint(
                    cs,
                    &gamma_k_j,
                    gamma_k_j_val,
                    &gamma_to_k,
                    gamma_to_k_val,
                    self.delta,
                    &format!("step_{}_T_gamma_k_j_step_j{}_{}", step_idx, j, pow_idx),
                )?;
                gamma_k_j = new_gamma_k_j;
                gamma_k_j_val = new_gamma_k_j_val;
            }

            for (idx, (y_table_vars, y_table_vals)) in me_inputs_y_vars
                .iter()
                .zip(me_inputs_y_vals.iter())
                .enumerate()
            {
                // me_inputs[idx] corresponds to instance i = idx + 2 in the paper.
                let i_abs = idx + 2;
                let row_vars = &y_table_vars[j];
                let row_vals = &y_table_vals[j];
                let limit = min(d_sz, min(row_vars.len(), row_vals.len()));

                // y_eval = ⟨ y_{(i,j)}, χ_α ⟩
                let mut y_eval_val = NeoK::ZERO;
                let mut y_eval = helpers::k_zero(cs, &format!("step_{}_T_y_eval_j{}_i{}", step_idx, j, i_abs))?;
                for rho in 0..limit {
                    let (prod, prod_val) = helpers::k_mul_with_hint(
                        cs,
                        &row_vars[rho],
                        row_vals[rho],
                        &chi_alpha_vars[rho],
                        chi_alpha_vals[rho],
                        self.delta,
                        &format!("step_{}_T_y_eval_j{}_i{}_rho{}", step_idx, j, i_abs, rho),
                    )?;

                    y_eval_val += prod_val;
                    let y_eval_hint = KNum::<CircuitF>::from_neo_k(y_eval_val);
                    y_eval = k_add_raw(
                        cs,
                        &y_eval,
                        &prod,
                        Some(y_eval_hint),
                        &format!("step_{}_T_y_eval_acc_j{}_i{}_rho{}", step_idx, j, i_abs, rho),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                }

                // γ^{i-1}
                let mut gamma_i_val = NeoK::ONE;
                let mut gamma_i = helpers::k_one(cs, &format!("step_{}_T_gamma_i_init_j{}_i{}", step_idx, j, i_abs))?;
                for pow_idx in 0..(i_abs - 1) {
                    let (new_gamma_i, new_gamma_i_val) = helpers::k_mul_with_hint(
                        cs,
                        &gamma_i,
                        gamma_i_val,
                        gamma_var,
                        gamma_val,
                        self.delta,
                        &format!("step_{}_T_gamma_i_step_j{}_i{}_{}", step_idx, j, i_abs, pow_idx),
                    )?;
                    gamma_i = new_gamma_i;
                    gamma_i_val = new_gamma_i_val;
                }

                let (weight, weight_val) = helpers::k_mul_with_hint(
                    cs,
                    &gamma_i,
                    gamma_i_val,
                    &gamma_k_j,
                    gamma_k_j_val,
                    self.delta,
                    &format!("step_{}_T_weight_j{}_i{}", step_idx, j, i_abs),
                )?;

                let (contrib, contrib_val) = helpers::k_mul_with_hint(
                    cs,
                    &weight,
                    weight_val,
                    &y_eval,
                    y_eval_val,
                    self.delta,
                    &format!("step_{}_T_contrib_j{}_i{}", step_idx, j, i_abs),
                )?;

                inner_val += contrib_val;
                let inner_hint = KNum::<CircuitF>::from_neo_k(inner_val);
                inner = k_add_raw(
                    cs,
                    &inner,
                    &contrib,
                    Some(inner_hint),
                    &format!("step_{}_T_inner_acc_j{}_i{}", step_idx, j, i_abs),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
            }
        }

        // T = γ^{k_total} * inner, matching the native `claimed_initial_sum_from_inputs`.
        let (t_var, _t_val) = helpers::k_mul_with_hint(
            cs,
            &gamma_to_k,
            gamma_to_k_val,
            &inner,
            inner_val,
            self.delta,
            &format!("step_{}_T_scale_by_gamma_k", step_idx),
        )?;

        Ok(t_var)
    }

    /// Build χ-table for an Ajtai challenge vector (α or α') using native K values.
    ///
    /// χ_α[ρ] = ∏_bit (α_bit if ρ_bit=1 else 1-α_bit), computed in the host field and
    /// lifted into K variables. This avoids needing in-circuit K arithmetic for χ.
    #[allow(dead_code)]
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
                let term = if bit_is_one { a_val } else { neo_math::K::ONE - a_val };
                w_native *= term;
            }

            let w_var = helpers::alloc_k_from_neo(cs, w_native, &format!("step_{}_{}_chi_{}", step_idx, label, rho))?;
            chi.push(w_var);
        }

        Ok(chi)
    }

    /// Equality polynomial eq_points over K, using the same formula as the
    /// native `eq_points`: ∏_i [(1-p_i)*(1-q_i) + p_i*q_i].
    ///
    /// This version constrains `eq` in terms of the K variables `p` and `q`,
    /// while using `p_vals`/`q_vals` as native hints for intermediate K ops.
    fn eq_points<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        p: &[KNumVar],
        q: &[KNumVar],
        p_vals: &[neo_math::K],
        q_vals: &[neo_math::K],
        label: &str,
    ) -> Result<(KNumVar, neo_math::K)> {
        use neo_math::K as NeoK;

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

        // eq over empty vectors is 1.
        if p.is_empty() {
            let one_var = helpers::k_one(cs, &format!("step_{}_{}_eq_one", step_idx, label))?;
            return Ok((one_var, NeoK::ONE));
        }

        // Canonical K-constant 1, shared across all coordinates.
        let one_var = helpers::k_one(cs, &format!("step_{}_{}_one_const", step_idx, label))?;

        // acc = 1 in K.
        let mut acc_var = one_var.clone();
        let mut acc_native = NeoK::ONE;

        for i in 0..p.len() {
            let pi_var = &p[i];
            let qi_var = &q[i];
            let pi_val = p_vals[i];
            let qi_val = q_vals[i];

            // 1 - p_i
            let one_minus_pi_val = NeoK::ONE - pi_val;
            let one_minus_pi_hint = KNum::<CircuitF>::from_neo_k(one_minus_pi_val);
            let one_minus_pi = alloc_k(
                cs,
                Some(one_minus_pi_hint),
                &format!("step_{}_{}_one_minus_p_{}", step_idx, label, i),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            // Enforce (1 - p_i) + p_i = 1 in K.
            let sum_p_val = one_minus_pi_val + pi_val; // native 1
            let sum_p_hint = KNum::<CircuitF>::from_neo_k(sum_p_val);
            let sum_p = k_add_raw(
                cs,
                &one_minus_pi,
                pi_var,
                Some(sum_p_hint),
                &format!("step_{}_{}_one_minus_p_sum_{}", step_idx, label, i),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
            helpers::enforce_k_eq(
                cs,
                &sum_p,
                &one_var,
                &format!("step_{}_{}_one_minus_p_check_{}", step_idx, label, i),
            );

            // 1 - q_i
            let one_minus_qi_val = NeoK::ONE - qi_val;
            let one_minus_qi_hint = KNum::<CircuitF>::from_neo_k(one_minus_qi_val);
            let one_minus_qi = alloc_k(
                cs,
                Some(one_minus_qi_hint),
                &format!("step_{}_{}_one_minus_q_{}", step_idx, label, i),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            let sum_q_val = one_minus_qi_val + qi_val; // native 1
            let sum_q_hint = KNum::<CircuitF>::from_neo_k(sum_q_val);
            let sum_q = k_add_raw(
                cs,
                &one_minus_qi,
                qi_var,
                Some(sum_q_hint),
                &format!("step_{}_{}_one_minus_q_sum_{}", step_idx, label, i),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
            helpers::enforce_k_eq(
                cs,
                &sum_q,
                &one_var,
                &format!("step_{}_{}_one_minus_q_check_{}", step_idx, label, i),
            );

            // (1 - p_i)*(1 - q_i)
            let (prod1_var, prod1_val) = helpers::k_mul_with_hint(
                cs,
                &one_minus_pi,
                one_minus_pi_val,
                &one_minus_qi,
                one_minus_qi_val,
                self.delta,
                &format!("step_{}_{}_prod1_{}", step_idx, label, i),
            )?;

            // p_i * q_i
            let (pq_var, pq_val) = helpers::k_mul_with_hint(
                cs,
                pi_var,
                pi_val,
                qi_var,
                qi_val,
                self.delta,
                &format!("step_{}_{}_pq_{}", step_idx, label, i),
            )?;

            // term_i = (1-p_i)*(1-q_i) + p_i*q_i
            let term_val = prod1_val + pq_val;
            let term_hint = KNum::<CircuitF>::from_neo_k(term_val);
            let term_var = k_add_raw(
                cs,
                &prod1_var,
                &pq_var,
                Some(term_hint),
                &format!("step_{}_{}_term_{}", step_idx, label, i),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            // acc *= term_i
            let (new_acc_var, new_acc_native) = helpers::k_mul_with_hint(
                cs,
                &acc_var,
                acc_native,
                &term_var,
                term_val,
                self.delta,
                &format!("step_{}_{}_eq_acc_step_{}", step_idx, label, i),
            )?;
            acc_var = new_acc_var;
            acc_native = new_acc_native;
        }

        Ok((acc_var, acc_native))
    }

    /// Recompose a single Ajtai y-row in base-b into a K element:
    /// m = Σ_{ℓ} b^ℓ · y[ℓ], using native K hints for the result and
    /// enforcing linear relations on the limbs.
    fn recompose_y_row_base_b<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        _j: usize,
        y_row_vars: &[KNumVar],
        y_row_vals: &[neo_math::K],
        label: &str,
    ) -> Result<(KNumVar, neo_math::K)> {
        use neo_math::{F as NeoF, K as NeoK};

        let len = core::cmp::min(y_row_vars.len(), y_row_vals.len());
        if len == 0 {
            let zero = helpers::k_zero(cs, &format!("step_{}_{}_empty_row", step_idx, label))?;
            return Ok((zero, NeoK::ZERO));
        }

        // Native recomposition in K: m_native = Σ b^ℓ * y[ℓ].
        let base_native_f: NeoF = NeoF::from_u64(self.base_b as u64);
        let base_native_k: NeoK = NeoK::from(base_native_f);

        let mut pow_k = NeoK::ONE;
        let mut m_native = NeoK::ZERO;
        for ell in 0..len {
            m_native += pow_k * y_row_vals[ell];
            pow_k *= base_native_k;
        }

        // Allocate m with the correct native value as a hint.
        let m_var = helpers::alloc_k_from_neo(cs, m_native, &format!("step_{}_{}_val", step_idx, label))?;

        // Enforce limb-wise base-b recomposition:
        // m.c0 = Σ b^ℓ * y[ℓ].c0,  m.c1 = Σ b^ℓ * y[ℓ].c1.
        let base_circ = CircuitF::from(self.base_b as u64);

        // c0 component
        cs.enforce(
            || format!("step_{}_{}_recompose_c0", step_idx, label),
            |lc| {
                let mut res = lc;
                // ℓ = 0 term has coefficient 1.
                res = res + (CircuitF::from(1u64), y_row_vars[0].c0);
                let mut pow = base_circ;
                for ell in 1..len {
                    res = res + (pow, y_row_vars[ell].c0);
                    pow *= base_circ;
                }
                res
            },
            |lc| lc + CS::one(),
            |lc| lc + m_var.c0,
        );

        // c1 component
        cs.enforce(
            || format!("step_{}_{}_recompose_c1", step_idx, label),
            |lc| {
                let mut res = lc;
                res = res + (CircuitF::from(1u64), y_row_vars[0].c1);
                let mut pow = base_circ;
                for ell in 1..len {
                    res = res + (pow, y_row_vars[ell].c1);
                    pow *= base_circ;
                }
                res
            },
            |lc| lc + CS::one(),
            |lc| lc + m_var.c1,
        );

        Ok((m_var, m_native))
    }

    /// Range product gadget: ∏_{t=-(b-1)}^{b-1} (val - t) over K, using native
    /// K hints and explicit limb-wise linear constraints for (val - t).
    #[allow(dead_code)]
    fn range_product<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        val: &KNumVar,
        val_native: neo_math::K,
        label: &str,
    ) -> Result<(KNumVar, neo_math::K)> {
        use neo_math::{F as NeoF, K as NeoK};

        let mut acc_var = helpers::k_one(cs, &format!("step_{}_{}_range_init", step_idx, label))?;
        let mut acc_native = NeoK::ONE;

        let b = self.base_b as i32;

        for t in (-(b - 1))..=(b - 1) {
            let abs = t.abs() as u64;
            let base_f: NeoF = NeoF::from_u64(abs);
            let t_native = if t >= 0 {
                NeoK::from(base_f)
            } else {
                NeoK::from(base_f) * NeoK::from(NeoF::from_u64(0u64)) - NeoK::from(base_f)
            };

            let diff_native = val_native - t_native;

            // Allocate diff with its native value.
            let diff_var = helpers::alloc_k_from_neo(
                cs,
                diff_native,
                &format!("step_{}_{}_val_minus_t_{}", step_idx, label, t),
            )?;

            // Enforce diff = val - t limb-wise in K (t is a constant).
            let t_hint = KNum::<CircuitF>::from_neo_k(t_native);
            let minus_t_c0 = CircuitF::from(0u64) - t_hint.c0;
            let minus_t_c1 = CircuitF::from(0u64) - t_hint.c1;

            // c0: diff.c0 = val.c0 - t.c0
            cs.enforce(
                || format!("step_{}_{}_range_diff_t{}_c0", step_idx, label, t),
                |lc| lc + val.c0 + (minus_t_c0, CS::one()),
                |lc| lc + CS::one(),
                |lc| lc + diff_var.c0,
            );
            // c1: diff.c1 = val.c1 - t.c1
            cs.enforce(
                || format!("step_{}_{}_range_diff_t{}_c1", step_idx, label, t),
                |lc| lc + val.c1 + (minus_t_c1, CS::one()),
                |lc| lc + CS::one(),
                |lc| lc + diff_var.c1,
            );

            // acc *= diff with native hints.
            let (new_acc_var, new_acc_native) = helpers::k_mul_with_hint(
                cs,
                &acc_var,
                acc_native,
                &diff_var,
                diff_native,
                self.delta,
                &format!("step_{}_{}_range_acc_{}", step_idx, label, t),
            )?;
            acc_var = new_acc_var;
            acc_native = new_acc_native;
        }

        Ok((acc_var, acc_native))
    }

    /// Evaluate the CCS polynomial f at the given m-values in K, using native K
    /// hints for all intermediate products to keep the witness consistent.
    #[allow(dead_code)]
    fn eval_poly_f_in_k<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        m_vals: &[KNumVar],
        m_vals_native: &[neo_math::K],
    ) -> Result<(KNumVar, neo_math::K)> {
        use neo_math::K as NeoK;

        if m_vals.len() != m_vals_native.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "eval_poly_f_in_k length mismatch at step {}: m_vals={}, m_vals_native={}",
                step_idx,
                m_vals.len(),
                m_vals_native.len()
            )));
        }

        // Compute each monomial term_val = ∏_j m_j^{exp_j} with K hints.
        let mut term_vars: Vec<KNumVar> = Vec::with_capacity(self.poly_f.len());
        let mut term_natives: Vec<NeoK> = Vec::with_capacity(self.poly_f.len());

        for (term_idx, term) in self.poly_f.iter().enumerate() {
            let mut term_var = helpers::k_one(cs, &format!("step_{}_F_term{}_init", step_idx, term_idx))?;
            let mut term_native = NeoK::ONE;

            for (var_idx, &exp) in term.exps.iter().enumerate() {
                if exp == 0 {
                    continue;
                }
                let base_var = &m_vals[var_idx];
                let base_native = m_vals_native[var_idx];

                // pow = base^exp
                let mut pow_var = helpers::k_one(
                    cs,
                    &format!("step_{}_F_term{}_var{}_pow_init", step_idx, term_idx, var_idx),
                )?;
                let mut pow_native = NeoK::ONE;
                for e in 0..exp {
                    let (new_pow_var, new_pow_native) = helpers::k_mul_with_hint(
                        cs,
                        &pow_var,
                        pow_native,
                        base_var,
                        base_native,
                        self.delta,
                        &format!("step_{}_F_term{}_var{}_pow_mul{}", step_idx, term_idx, var_idx, e),
                    )?;
                    pow_var = new_pow_var;
                    pow_native = new_pow_native;
                }

                // term *= pow
                let (new_term_var, new_term_native) = helpers::k_mul_with_hint(
                    cs,
                    &term_var,
                    term_native,
                    &pow_var,
                    pow_native,
                    self.delta,
                    &format!("step_{}_F_term{}_var{}_mul", step_idx, term_idx, var_idx),
                )?;
                term_var = new_term_var;
                term_native = new_term_native;
            }

            term_vars.push(term_var);
            term_natives.push(term_native);
        }

        // Native F' value: F'(m) = Σ coeff * term_native.
        let mut F_prime_native = NeoK::ZERO;
        for (term_idx, term) in self.poly_f.iter().enumerate() {
            let coeff_k: NeoK = NeoK::from(term.coeff_native);
            F_prime_native += coeff_k * term_natives[term_idx];
        }

        let F_prime_var = helpers::alloc_k_from_neo(cs, F_prime_native, &format!("step_{}_F_prime", step_idx))?;

        // Enforce F' limb-wise as Σ coeff * term_j.
        cs.enforce(
            || format!("step_{}_F_prime_c0_check", step_idx),
            |lc| {
                let mut res = lc;
                for (term_idx, term) in self.poly_f.iter().enumerate() {
                    let coeff_circ = term.coeff;
                    res = res + (coeff_circ, term_vars[term_idx].c0);
                }
                res
            },
            |lc| lc + CS::one(),
            |lc| lc + F_prime_var.c0,
        );

        cs.enforce(
            || format!("step_{}_F_prime_c1_check", step_idx),
            |lc| {
                let mut res = lc;
                for (term_idx, term) in self.poly_f.iter().enumerate() {
                    let coeff_circ = term.coeff;
                    res = res + (coeff_circ, term_vars[term_idx].c1);
                }
                res
            },
            |lc| lc + CS::one(),
            |lc| lc + F_prime_var.c1,
        );

        Ok((F_prime_var, F_prime_native))
    }

    /// Verify Π-CCS FE terminal identity for a step (SplitNcV1):
    /// sumcheck_final == rhs_terminal_identity_fe_paper_exact(α,β,γ,r',α', outputs, inputs.r).
    fn verify_terminal_identity<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        step: &FoldStep,
        _proof: &neo_reductions::PiCcsProof,
        challenges: &crate::circuit::witness::PiCcsChallenges,
        witness: &FoldRunWitness,
        sumcheck_final: &KNumVar,
        out_y_vars: &[Vec<Vec<KNumVar>>],
        alpha_vars: &[KNumVar],
        beta_a_vars: &[KNumVar],
        beta_r_vars: &[KNumVar],
        gamma_var: &KNumVar,
        sumcheck_challenges: &[KNumVar],
    ) -> Result<()> {
        use neo_math::K as NeoK;

        // ME inputs for this step (needed only for r in eq((α',r'),(α,r))).
        let me_inputs = if step_idx == 0 {
            &self.instance.initial_accumulator
        } else {
            &witness.fold_run.steps[step_idx - 1].fold.dec_children
        };

        // Outputs y' for this step.
        let out_me = &step.ccs_out;
        if out_me.is_empty() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "Terminal identity at step {}: empty outputs",
                step_idx
            )));
        }

        // Split the per-round sumcheck challenges into the views used by Π-CCS:
        // r' is the row-domain prefix and α' is the Ajtai-domain suffix.
        if sumcheck_challenges.len() != challenges.sumcheck_challenges.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "sumcheck_challenges length mismatch at step {}: vars={}, values={}",
                step_idx,
                sumcheck_challenges.len(),
                challenges.sumcheck_challenges.len()
            )));
        }
        let rows = challenges.r_prime.len();
        let ajtai = challenges.alpha_prime.len();
        if sumcheck_challenges.len() != rows + ajtai {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "sumcheck_challenges split mismatch at step {}: expected {} (rows + ajtai), got {}",
                step_idx,
                rows + ajtai,
                sumcheck_challenges.len()
            )));
        }
        let (r_prime_vars, alpha_prime_vars) = sumcheck_challenges.split_at(rows);

        // --- Scalar equality polynomials eq((α',r'),β) and eq((α',r'),(α,r)) ---
        //
        // Computed in-circuit over K using eq_points, with native K values only
        // used as hints for k_mul_with_hint.
        if beta_a_vars.len() != challenges.beta_a.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "beta_a length mismatch at step {}: vars={}, values={}",
                step_idx,
                beta_a_vars.len(),
                challenges.beta_a.len()
            )));
        }
        if beta_r_vars.len() != challenges.beta_r.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "beta_r length mismatch at step {}: vars={}, values={}",
                step_idx,
                beta_r_vars.len(),
                challenges.beta_r.len()
            )));
        }

        // eq((α',r'), β) = eq(α', β_a) * eq(r', β_r) in K.
        let (eq_alpha_prime_beta_a, eq_alpha_prime_beta_a_native) = self.eq_points(
            cs,
            step_idx,
            alpha_prime_vars,
            beta_a_vars,
            &challenges.alpha_prime,
            &challenges.beta_a,
            "eq_alpha_prime_beta_a",
        )?;

        let (eq_r_prime_beta_r, eq_r_prime_beta_r_native) = self.eq_points(
            cs,
            step_idx,
            r_prime_vars,
            beta_r_vars,
            &challenges.r_prime,
            &challenges.beta_r,
            "eq_r_prime_beta_r",
        )?;

        let (eq_aprp_beta, eq_aprp_beta_native) = helpers::k_mul_with_hint(
            cs,
            &eq_alpha_prime_beta_a,
            eq_alpha_prime_beta_a_native,
            &eq_r_prime_beta_r,
            eq_r_prime_beta_r_native,
            self.delta,
            &format!("step_{}_eq_aprp_beta", step_idx),
        )?;

        // eq((α',r'),(α,r)) if we have ME inputs; else 0 (Eval' block vanishes).
        let (eq_aprp_ar, eq_aprp_ar_native) = if let Some(first_input) = me_inputs.first() {
            if alpha_vars.len() != challenges.alpha.len() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "alpha length mismatch at step {}: vars={}, values={}",
                    step_idx,
                    alpha_vars.len(),
                    challenges.alpha.len()
                )));
            }

            // Allocate r (from the first ME input) as K variables.
            let mut r_vars = Vec::with_capacity(first_input.r.len());
            for (i, &k) in first_input.r.iter().enumerate() {
                r_vars.push(helpers::alloc_k_from_neo(cs, k, &format!("step_{}_r_{}", step_idx, i))?);
            }

            let (eq_alpha_prime_alpha, eq_alpha_prime_alpha_native) = self.eq_points(
                cs,
                step_idx,
                alpha_prime_vars,
                alpha_vars,
                &challenges.alpha_prime,
                &challenges.alpha,
                "eq_alpha_prime_alpha",
            )?;

            let (eq_r_prime_r, eq_r_prime_r_native) = self.eq_points(
                cs,
                step_idx,
                r_prime_vars,
                &r_vars,
                &challenges.r_prime,
                &first_input.r,
                "eq_r_prime_r",
            )?;

            helpers::k_mul_with_hint(
                cs,
                &eq_alpha_prime_alpha,
                eq_alpha_prime_alpha_native,
                &eq_r_prime_r,
                eq_r_prime_r_native,
                self.delta,
                &format!("step_{}_eq_aprp_ar", step_idx),
            )?
        } else {
            // No ME inputs ⇒ Eval' block vanishes; force eq((α',r'),(α,r)) = 0 so
            // that the whole Eval' contribution is zero.
            let zero_var = helpers::k_zero(cs, &format!("step_{}_eq_aprp_ar_zero", step_idx))?;
            (zero_var, NeoK::ZERO)
        };

        // --- γ and γ^k_total ---
        let gamma_val = challenges.gamma;
        let k_total = out_me.len();
        let mut gamma_k_total = helpers::k_one(cs, &format!("step_{}_gamma_k_total_init", step_idx))?;
        let mut gamma_k_total_val = neo_math::K::ONE;
        for pow_idx in 0..k_total {
            let (new_gk, new_gk_val) = helpers::k_mul_with_hint(
                cs,
                &gamma_k_total,
                gamma_k_total_val,
                gamma_var,
                gamma_val,
                self.delta,
                &format!("step_{}_gamma_k_total_mul_{}", step_idx, pow_idx),
            )?;
            gamma_k_total = new_gk;
            gamma_k_total_val = new_gk_val;
        }

        // --- F' from first output's y'[i=1] in-circuit ---
        //
        // Recompose m_j from Ajtai digits with base-b using only the first D
        // digits, then evaluate f via poly_f in K.
        let t = out_me[0].y.len();
        let (F_prime, F_prime_native) = if t == 0 {
            (
                helpers::k_zero(cs, &format!("step_{}_F_prime_zero", step_idx))?,
                NeoK::ZERO,
            )
        } else {
            // Use the shared y-table allocation for the first output.
            let first_out_y = &out_y_vars[0];
            let d_pad = first_out_y[0].len();
            let d_ring = neo_math::D.min(d_pad);

            // Native y-table for the first output (for K hints).
            let first_out_y_vals = &out_me[0].y;

            let mut m_vals: Vec<KNumVar> = Vec::with_capacity(t);
            let mut m_vals_native: Vec<NeoK> = Vec::with_capacity(t);
            for j in 0..t {
                if first_out_y[j].len() != d_pad {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "Terminal identity at step {}: inconsistent y row length in output 0 (j={})",
                        step_idx, j
                    )));
                }
                if first_out_y_vals[j].len() != d_pad {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "Terminal identity at step {}: inconsistent y row length in output 0 values (j={})",
                        step_idx, j
                    )));
                }
                let (m_j, m_j_native) = self.recompose_y_row_base_b(
                    cs,
                    step_idx,
                    j,
                    &first_out_y[j][..d_ring],
                    &first_out_y_vals[j][..d_ring],
                    &format!("step_{}_F_m_j{}", step_idx, j),
                )?;
                m_vals.push(m_j);
                m_vals_native.push(m_j_native);
            }

            self.eval_poly_f_in_k(cs, step_idx, &m_vals, &m_vals_native)?
        };

        // --- χ_{α'} table in K for Ajtai domain ---
        //
        // χ_{α'}[ρ] = ∏_bit (α'_bit if ρ_bit=1 else 1-α'_bit), with explicit
        // K equality constraints using native K hints (mirrors χ_α gadget).
        let d_sz = 1usize << challenges.alpha_prime.len();
        let mut chi_alpha_prime: Vec<KNumVar> = Vec::with_capacity(d_sz);
        let mut chi_alpha_prime_vals: Vec<NeoK> = Vec::with_capacity(d_sz);
        for rho in 0..d_sz {
            let mut w_val = NeoK::ONE;
            let mut w_var = helpers::k_one(cs, &format!("step_{}_chi_alpha_prime_{}_init", step_idx, rho))?;

            for (bit, (a_var, &a_val)) in alpha_prime_vars
                .iter()
                .zip(challenges.alpha_prime.iter())
                .enumerate()
            {
                let bit_is_one = ((rho >> bit) & 1) == 1;
                // Factor value in K.
                let (factor_var, factor_val) = if bit_is_one {
                    (a_var.clone(), a_val)
                } else {
                    // factor = 1 - α'_bit, enforced via factor + α'_bit = 1 in K.
                    let factor_val = NeoK::ONE - a_val;
                    let factor_hint = KNum::<CircuitF>::from_neo_k(factor_val);
                    let factor_var = alloc_k(
                        cs,
                        Some(factor_hint),
                        &format!("step_{}_chi_alpha_prime_{}_bit{}_factor", step_idx, rho, bit),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;

                    let one_var =
                        helpers::k_one(cs, &format!("step_{}_chi_alpha_prime_{}_bit{}_one", step_idx, rho, bit))?;

                    // sum = factor + α'_bit with native hint, enforce sum == 1.
                    let sum_val = factor_val + a_val;
                    let sum_hint = KNum::<CircuitF>::from_neo_k(sum_val);
                    let sum_var = k_add_raw(
                        cs,
                        &factor_var,
                        a_var,
                        Some(sum_hint),
                        &format!("step_{}_chi_alpha_prime_{}_bit{}_sum", step_idx, rho, bit),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                    helpers::enforce_k_eq(
                        cs,
                        &sum_var,
                        &one_var,
                        &format!("step_{}_chi_alpha_prime_{}_bit{}_one_minus", step_idx, rho, bit),
                    );

                    (factor_var, factor_val)
                };

                // w <- w * factor with native hint.
                let (new_w_var, new_w_val) = helpers::k_mul_with_hint(
                    cs,
                    &w_var,
                    w_val,
                    &factor_var,
                    factor_val,
                    self.delta,
                    &format!("step_{}_chi_alpha_prime_{}_bit{}", step_idx, rho, bit),
                )?;
                w_var = new_w_var;
                w_val = new_w_val;
            }

            chi_alpha_prime.push(w_var);
            chi_alpha_prime_vals.push(w_val);
        }

        // SplitNcV1: NC is handled by a dedicated NC sumcheck over y_zcol, so the FE RHS excludes NC.

        // --- Eval' block in K ---
        //
        // γ^k · Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · E_{(i,j)} with
        // E_{(i,j)} = eq((α',r'),(α,r)) · ẏ'_{(i,j)}(α').
        let mut eval_sum = helpers::k_zero(cs, &format!("step_{}_Eval_sum_init", step_idx))?;
        let mut eval_sum_native = NeoK::ZERO;

        if me_inputs.first().is_some() && k_total >= 2 {
            for j in 0..t {
                // Precompute (γ^k_total)^j once per j and reuse across outputs.
                let mut gamma_k_j_val = NeoK::ONE;
                let mut gamma_k_j = helpers::k_one(cs, &format!("step_{}_Eval_gamma_k_j_init_j{}", step_idx, j))?;
                for pow_idx in 0..j {
                    let (new_gamma_k_j, new_gamma_k_j_val) = helpers::k_mul_with_hint(
                        cs,
                        &gamma_k_j,
                        gamma_k_j_val,
                        &gamma_k_total,
                        gamma_k_total_val,
                        self.delta,
                        &format!("step_{}_Eval_gamma_k_j_step_j{}_{}", step_idx, j, pow_idx),
                    )?;
                    gamma_k_j = new_gamma_k_j;
                    gamma_k_j_val = new_gamma_k_j_val;
                }

                for (i_abs, out_y) in out_y_vars.iter().enumerate().skip(1) {
                    if out_y.len() != t {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "Terminal identity at step {}: y length mismatch in output {}",
                            step_idx, i_abs
                        )));
                    }
                    let row = &out_y[j];
                    let limit = core::cmp::min(chi_alpha_prime.len(), row.len());

                    let mut y_eval = helpers::k_zero(cs, &format!("step_{}_Eval_y_eval_j{}_i{}", step_idx, j, i_abs))?;
                    let mut y_eval_val = NeoK::ZERO;
                    for rho in 0..limit {
                        let y_val = out_me[i_abs].y[j][rho];
                        let chi_val = chi_alpha_prime_vals[rho];
                        let (prod_var, prod_val) = helpers::k_mul_with_hint(
                            cs,
                            &row[rho],
                            y_val,
                            &chi_alpha_prime[rho],
                            chi_val,
                            self.delta,
                            &format!("step_{}_Eval_y_eval_j{}_i{}_rho{}", step_idx, j, i_abs, rho),
                        )?;
                        y_eval_val += prod_val;
                        let hint = KNum::<CircuitF>::from_neo_k(y_eval_val);
                        y_eval = k_add_raw(
                            cs,
                            &y_eval,
                            &prod_var,
                            Some(hint),
                            &format!("step_{}_Eval_y_eval_acc_j{}_i{}_rho{}", step_idx, j, i_abs, rho),
                        )
                        .map_err(SpartanBridgeError::BellpepperError)?;
                    }

                    // weight = γ^{i_abs} * (γ^k_total)^j  (0-based indices)
                    let mut gamma_i =
                        helpers::k_one(cs, &format!("step_{}_Eval_gamma_i_init_j{}_i{}", step_idx, j, i_abs))?;
                    let mut gamma_i_val = NeoK::ONE;
                    for pow_idx in 0..i_abs {
                        let (new_gamma_i, new_gamma_i_val) = helpers::k_mul_with_hint(
                            cs,
                            &gamma_i,
                            gamma_i_val,
                            gamma_var,
                            gamma_val,
                            self.delta,
                            &format!("step_{}_Eval_gamma_i_step_j{}_i{}_{}", step_idx, j, i_abs, pow_idx),
                        )?;
                        gamma_i = new_gamma_i;
                        gamma_i_val = new_gamma_i_val;
                    }

                    let (weight, weight_val) = helpers::k_mul_with_hint(
                        cs,
                        &gamma_i,
                        gamma_i_val,
                        &gamma_k_j,
                        gamma_k_j_val,
                        self.delta,
                        &format!("step_{}_Eval_weight_j{}_i{}", step_idx, j, i_abs),
                    )?;

                    let (contrib, contrib_val) = helpers::k_mul_with_hint(
                        cs,
                        &weight,
                        weight_val,
                        &y_eval,
                        y_eval_val,
                        self.delta,
                        &format!("step_{}_Eval_contrib_j{}_i{}", step_idx, j, i_abs),
                    )?;

                    eval_sum_native += contrib_val;
                    let eval_hint = KNum::<CircuitF>::from_neo_k(eval_sum_native);
                    eval_sum = k_add_raw(
                        cs,
                        &eval_sum,
                        &contrib,
                        Some(eval_hint),
                        &format!("step_{}_Eval_acc_j{}_i{}", step_idx, j, i_abs),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                }
            }
        }

        // Assemble RHS_fe in K:
        // v_fe = eq((α',r'), β)·F' + γ^k · eq((α',r'), (α,r)) · Eval'.
        let (left, left_native) = helpers::k_mul_with_hint(
            cs,
            &eq_aprp_beta,
            eq_aprp_beta_native,
            &F_prime,
            F_prime_native,
            self.delta,
            &format!("step_{}_RHS_left", step_idx),
        )?;

        // Eval' := γ^k · Σ_{j,i} γ^{i-1 + j·k} · ẏ'_{(i,j)}(α').
        let (eval_sum_scaled, eval_sum_scaled_native) = helpers::k_mul_with_hint(
            cs,
            &gamma_k_total,
            gamma_k_total_val,
            &eval_sum,
            eval_sum_native,
            self.delta,
            &format!("step_{}_Eval_sum_scaled", step_idx),
        )?;
        let (right, right_native) = helpers::k_mul_with_hint(
            cs,
            &eq_aprp_ar,
            eq_aprp_ar_native,
            &eval_sum_scaled,
            eval_sum_scaled_native,
            self.delta,
            &format!("step_{}_RHS_right", step_idx),
        )?;

        // rhs = left + right with native hint.
        let rhs_native = left_native + right_native;
        let rhs_hint = KNum::<CircuitF>::from_neo_k(rhs_native);
        let rhs = k_add_raw(
            cs,
            &left,
            &right,
            Some(rhs_hint),
            &format!("step_{}_RHS_total", step_idx),
        )
        .map_err(SpartanBridgeError::BellpepperError)?;

        // Enforce that the in-circuit final running sum from sumcheck
        // rounds equals the RHS terminal identity.
        helpers::enforce_k_eq(
            cs,
            sumcheck_final,
            &rhs,
            &format!("step_{}_final_sum_matches_rhs", step_idx),
        );

        Ok(())
    }

    /// Verify Π-CCS NC terminal identity for a step (SplitNcV1).
    ///
    /// sumcheck_final_nc == rhs_terminal_identity_nc_paper_exact(β_a,β_m,γ,s_col',α'_nc, outputs.y_zcol).
    fn verify_terminal_identity_nc<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        step: &FoldStep,
        _proof: &neo_reductions::PiCcsProof,
        challenges: &crate::circuit::witness::PiCcsChallenges,
        sumcheck_final_nc: &KNumVar,
        out_y_zcol_vars: &[Vec<KNumVar>],
        beta_a_vars: &[KNumVar],
        beta_m_vars: &[KNumVar],
        gamma_var: &KNumVar,
        sumcheck_challenges_nc: &[KNumVar],
    ) -> Result<()> {
        use neo_math::K as NeoK;

        let out_me = &step.ccs_out;
        if out_me.is_empty() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "NC terminal identity at step {}: empty outputs",
                step_idx
            )));
        }
        if out_y_zcol_vars.len() != out_me.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "NC terminal identity at step {}: y_zcol vars/output length mismatch (vars={}, outputs={})",
                step_idx,
                out_y_zcol_vars.len(),
                out_me.len()
            )));
        }

        // Split the NC sumcheck challenges into the views used by Π-CCS (SplitNcV1):
        // s_col' is the column-domain prefix and α'_nc is the Ajtai-domain suffix.
        if sumcheck_challenges_nc.len() != challenges.sumcheck_challenges_nc.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "sumcheck_challenges_nc length mismatch at step {}: vars={}, values={}",
                step_idx,
                sumcheck_challenges_nc.len(),
                challenges.sumcheck_challenges_nc.len()
            )));
        }
        let cols = challenges.s_col_prime.len();
        let ajtai = challenges.alpha_prime_nc.len();
        if sumcheck_challenges_nc.len() != cols + ajtai {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "sumcheck_challenges_nc split mismatch at step {}: expected {} (cols + ajtai), got {}",
                step_idx,
                cols + ajtai,
                sumcheck_challenges_nc.len()
            )));
        }
        let (s_col_prime_vars, alpha_prime_vars) = sumcheck_challenges_nc.split_at(cols);

        if beta_a_vars.len() != challenges.beta_a.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "beta_a length mismatch at step {}: vars={}, values={}",
                step_idx,
                beta_a_vars.len(),
                challenges.beta_a.len()
            )));
        }
        if beta_m_vars.len() != challenges.beta_m.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "beta_m length mismatch at step {}: vars={}, values={}",
                step_idx,
                beta_m_vars.len(),
                challenges.beta_m.len()
            )));
        }

        let (eq_alpha_prime_beta_a, eq_alpha_prime_beta_a_native) = self.eq_points(
            cs,
            step_idx,
            alpha_prime_vars,
            beta_a_vars,
            &challenges.alpha_prime_nc,
            &challenges.beta_a,
            "eq_alpha_prime_nc_beta_a",
        )?;

        let (eq_s_col_beta_m, eq_s_col_beta_m_native) = self.eq_points(
            cs,
            step_idx,
            s_col_prime_vars,
            beta_m_vars,
            &challenges.s_col_prime,
            &challenges.beta_m,
            "eq_s_col_beta_m",
        )?;

        let (eq_apsp_beta, eq_apsp_beta_native) = helpers::k_mul_with_hint(
            cs,
            &eq_alpha_prime_beta_a,
            eq_alpha_prime_beta_a_native,
            &eq_s_col_beta_m,
            eq_s_col_beta_m_native,
            self.delta,
            &format!("step_{}_eq_apsp_beta", step_idx),
        )?;

        // χ_{α'_nc} table in K for Ajtai domain.
        let d_sz = 1usize << challenges.alpha_prime_nc.len();
        let mut chi_alpha_prime: Vec<KNumVar> = Vec::with_capacity(d_sz);
        let mut chi_alpha_prime_vals: Vec<NeoK> = Vec::with_capacity(d_sz);
        for rho in 0..d_sz {
            let mut w_val = NeoK::ONE;
            let mut w_var = helpers::k_one(cs, &format!("step_{}_chi_alpha_prime_nc_{}_init", step_idx, rho))?;

            for (bit, (a_var, &a_val)) in alpha_prime_vars
                .iter()
                .zip(challenges.alpha_prime_nc.iter())
                .enumerate()
            {
                let bit_is_one = ((rho >> bit) & 1) == 1;
                let (factor_var, factor_val) = if bit_is_one {
                    (a_var.clone(), a_val)
                } else {
                    let factor_val = NeoK::ONE - a_val;
                    let factor_hint = KNum::<CircuitF>::from_neo_k(factor_val);
                    let factor_var = alloc_k(
                        cs,
                        Some(factor_hint),
                        &format!("step_{}_chi_alpha_prime_nc_{}_bit{}_factor", step_idx, rho, bit),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;

                    let one_var = helpers::k_one(
                        cs,
                        &format!("step_{}_chi_alpha_prime_nc_{}_bit{}_one", step_idx, rho, bit),
                    )?;

                    let sum_val = factor_val + a_val;
                    let sum_hint = KNum::<CircuitF>::from_neo_k(sum_val);
                    let sum_var = k_add_raw(
                        cs,
                        &factor_var,
                        a_var,
                        Some(sum_hint),
                        &format!("step_{}_chi_alpha_prime_nc_{}_bit{}_sum", step_idx, rho, bit),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                    helpers::enforce_k_eq(
                        cs,
                        &sum_var,
                        &one_var,
                        &format!("step_{}_chi_alpha_prime_nc_{}_bit{}_one_minus", step_idx, rho, bit),
                    );
                    (factor_var, factor_val)
                };

                let (new_w_var, new_w_val) = helpers::k_mul_with_hint(
                    cs,
                    &w_var,
                    w_val,
                    &factor_var,
                    factor_val,
                    self.delta,
                    &format!("step_{}_chi_alpha_prime_nc_{}_bit{}", step_idx, rho, bit),
                )?;
                w_var = new_w_var;
                w_val = new_w_val;
            }

            chi_alpha_prime.push(w_var);
            chi_alpha_prime_vals.push(w_val);
        }

        // Enforce output s_col == s_col'.
        for (i_idx, out) in out_me.iter().enumerate() {
            if out.s_col.len() != challenges.s_col_prime.len() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "NC terminal identity at step {}: out[{}].s_col.len() != s_col_prime.len()",
                    step_idx, i_idx
                )));
            }
            for j in 0..out.s_col.len() {
                let out_s = helpers::alloc_k_from_neo(
                    cs,
                    out.s_col[j],
                    &format!("step_{}_out_{}_s_col_{}", step_idx, i_idx, j),
                )?;
                helpers::enforce_k_eq(
                    cs,
                    &out_s,
                    &s_col_prime_vars[j],
                    &format!("step_{}_out_{}_s_col_matches_prime_{}", step_idx, i_idx, j),
                );
            }
        }

        // Σ γ^i · N_i over outputs, where N_i = range_product(⟨y_zcol_i, χ_{α'_nc}⟩).
        let gamma_val = challenges.gamma;

        let mut nc_sum = helpers::k_zero(cs, &format!("step_{}_nc_sum_init", step_idx))?;
        let mut nc_sum_native = NeoK::ZERO;

        let mut g = (*gamma_var).clone();
        let mut g_native = gamma_val;
        for (i_idx, out) in out_me.iter().enumerate() {
            let y = &out_y_zcol_vars[i_idx];
            let y_vals = &out.y_zcol;
            let limit = core::cmp::min(chi_alpha_prime.len(), core::cmp::min(y.len(), y_vals.len()));
            if limit == 0 {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "NC terminal identity at step {}: out[{}] y_zcol is empty",
                    step_idx, i_idx
                )));
            }

            let mut y_eval = helpers::k_zero(cs, &format!("step_{}_nc_y_eval_i{}", step_idx, i_idx))?;
            let mut y_eval_val = NeoK::ZERO;
            for rho in 0..limit {
                let (prod_var, prod_val) = helpers::k_mul_with_hint(
                    cs,
                    &y[rho],
                    y_vals[rho],
                    &chi_alpha_prime[rho],
                    chi_alpha_prime_vals[rho],
                    self.delta,
                    &format!("step_{}_nc_y_eval_i{}_rho{}", step_idx, i_idx, rho),
                )?;
                y_eval_val += prod_val;
                let hint = KNum::<CircuitF>::from_neo_k(y_eval_val);
                y_eval = k_add_raw(
                    cs,
                    &y_eval,
                    &prod_var,
                    Some(hint),
                    &format!("step_{}_nc_y_eval_acc_i{}_rho{}", step_idx, i_idx, rho),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
            }

            let (Ni, Ni_native) = self.range_product(
                cs,
                step_idx,
                &y_eval,
                y_eval_val,
                &format!("step_{}_nc_range_i{}", step_idx, i_idx),
            )?;

            let (gNi, gNi_native) = helpers::k_mul_with_hint(
                cs,
                &g,
                g_native,
                &Ni,
                Ni_native,
                self.delta,
                &format!("step_{}_nc_weighted_i{}", step_idx, i_idx),
            )?;

            nc_sum_native += gNi_native;
            let nc_hint = KNum::<CircuitF>::from_neo_k(nc_sum_native);
            nc_sum = k_add_raw(
                cs,
                &nc_sum,
                &gNi,
                Some(nc_hint),
                &format!("step_{}_nc_acc_i{}", step_idx, i_idx),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            let (new_g, new_g_native) = helpers::k_mul_with_hint(
                cs,
                &g,
                g_native,
                gamma_var,
                gamma_val,
                self.delta,
                &format!("step_{}_nc_gamma_step_i{}", step_idx, i_idx),
            )?;
            g = new_g;
            g_native = new_g_native;
        }

        let (rhs, rhs_native) = helpers::k_mul_with_hint(
            cs,
            &eq_apsp_beta,
            eq_apsp_beta_native,
            &nc_sum,
            nc_sum_native,
            self.delta,
            &format!("step_{}_nc_rhs", step_idx),
        )?;

        helpers::enforce_k_eq(
            cs,
            sumcheck_final_nc,
            &rhs,
            &format!("step_{}_nc_final_sum_matches_rhs", step_idx),
        );

        // Silence unused native hint (kept for symmetry with FE terminal identity).
        let _ = rhs_native;

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
        children_y_vars: &[Vec<Vec<KNumVar>>],
        children_y_zcol_vars: &[Vec<KNumVar>],
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
        let parent_X_vars = helpers::alloc_matrix_from_neo(cs, &parent.X, &format!("step_{}_rlc_parent_X", step_idx))?;

        let mut child_X_vars = Vec::with_capacity(children.len());
        for (i, child) in children.iter().enumerate() {
            child_X_vars.push(helpers::alloc_matrix_from_neo(
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
                                let coeff = helpers::neo_f_to_circuit(&rho[(row, k)]);
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
                let parent_r_var = helpers::alloc_k_from_neo(
                    cs,
                    parent.r[idx],
                    &format!("step_{}_rlc_parent_r_{}_{}", step_idx, i, idx),
                )?;
                let child_r_var = helpers::alloc_k_from_neo(
                    cs,
                    child.r[idx],
                    &format!("step_{}_rlc_child_{}_r_{}", step_idx, i, idx),
                )?;
                helpers::enforce_k_eq(
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

        let parent_y_vars = helpers::alloc_y_table_from_neo(cs, &parent.y, &format!("step_{}_rlc_parent_y", step_idx))?;

        if children_y_vars.len() != children.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "RLC at step {}: children_y_vars/children length mismatch (vars={}, children={})",
                step_idx,
                children_y_vars.len(),
                children.len()
            )));
        }
        if children_y_zcol_vars.len() != children.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "RLC at step {}: children_y_zcol_vars/children length mismatch (vars={}, children={})",
                step_idx,
                children_y_zcol_vars.len(),
                children.len()
            )));
        }
        for (i, child) in children.iter().enumerate() {
            if child.y.len() != t {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC y length mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
        }

        // Enforce s_col preservation: parent.s_col == child.s_col for all inputs (when present).
        let s_col_len = parent.s_col.len();
        for (i, child) in children.iter().enumerate() {
            if child.s_col.len() != s_col_len {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC s_col length mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
            for idx in 0..s_col_len {
                let parent_s = helpers::alloc_k_from_neo(
                    cs,
                    parent.s_col[idx],
                    &format!("step_{}_rlc_parent_s_col_{}_{}", step_idx, i, idx),
                )?;
                let child_s = helpers::alloc_k_from_neo(
                    cs,
                    child.s_col[idx],
                    &format!("step_{}_rlc_child_{}_s_col_{}", step_idx, i, idx),
                )?;
                helpers::enforce_k_eq(
                    cs,
                    &parent_s,
                    &child_s,
                    &format!("step_{}_rlc_s_col_eq_child_{}_idx_{}", step_idx, i, idx),
                );
            }
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
                                let coeff = helpers::neo_f_to_circuit(&rho[(r_idx, k)]);
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
                                let coeff = helpers::neo_f_to_circuit(&rho[(r_idx, k)]);
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

        // Enforce y_zcol_parent = Σ_i ρ_i · y_zcol_i (first D digits), when present.
        let want_nc_channel = !(parent.s_col.is_empty() && parent.y_zcol.is_empty());
        if want_nc_channel {
            if parent.y_zcol.len() != d_pad {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC at step {}: parent y_zcol length mismatch (got {}, expected {})",
                    step_idx,
                    parent.y_zcol.len(),
                    d_pad
                )));
            }
            for (i, child) in children.iter().enumerate() {
                if child.y_zcol.len() != d_pad {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "RLC at step {}: child {} y_zcol length mismatch (got {}, expected {})",
                        step_idx,
                        i,
                        child.y_zcol.len(),
                        d_pad
                    )));
                }
                if children_y_zcol_vars[i].len() != d_pad {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "RLC at step {}: child {} y_zcol vars length mismatch (got {}, expected {})",
                        step_idx,
                        i,
                        children_y_zcol_vars[i].len(),
                        d_pad
                    )));
                }
            }

            let mut parent_yz_vars = Vec::with_capacity(d_ring);
            for idx in 0..d_ring {
                parent_yz_vars.push(helpers::alloc_k_from_neo(
                    cs,
                    parent.y_zcol[idx],
                    &format!("step_{}_rlc_parent_y_zcol_{}", step_idx, idx),
                )?);
            }

            for r_idx in 0..d_ring {
                cs.enforce(
                    || format!("step_{}_rlc_y_zcol_c0_r{}", step_idx, r_idx),
                    |lc| {
                        let mut res = lc;
                        for (i, rho) in rhos.iter().enumerate() {
                            for k in 0..d_ring {
                                let coeff = helpers::neo_f_to_circuit(&rho[(r_idx, k)]);
                                if coeff != CircuitF::from(0u64) {
                                    res = res + (coeff, children_y_zcol_vars[i][k].c0);
                                }
                            }
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_yz_vars[r_idx].c0,
                );
                cs.enforce(
                    || format!("step_{}_rlc_y_zcol_c1_r{}", step_idx, r_idx),
                    |lc| {
                        let mut res = lc;
                        for (i, rho) in rhos.iter().enumerate() {
                            for k in 0..d_ring {
                                let coeff = helpers::neo_f_to_circuit(&rho[(r_idx, k)]);
                                if coeff != CircuitF::from(0u64) {
                                    res = res + (coeff, children_y_zcol_vars[i][k].c1);
                                }
                            }
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_yz_vars[r_idx].c1,
                );
            }
        } else if !children_y_zcol_vars.iter().all(|v| v.is_empty()) {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "RLC at step {}: unexpected y_zcol channel on child",
                step_idx
            )));
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
        let parent_X_vars = helpers::alloc_matrix_from_neo(cs, &parent.X, &format!("step_{}_dec_parent_X", step_idx))?;

        let mut child_X_vars = Vec::with_capacity(children.len());
        for (i, child) in children.iter().enumerate() {
            child_X_vars.push(helpers::alloc_matrix_from_neo(
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
                let parent_r_var = helpers::alloc_k_from_neo(
                    cs,
                    parent.r[idx],
                    &format!("step_{}_dec_parent_r_{}_{}", step_idx, i, idx),
                )?;
                let child_r_var = helpers::alloc_k_from_neo(
                    cs,
                    child.r[idx],
                    &format!("step_{}_dec_child_{}_r_{}", step_idx, i, idx),
                )?;
                helpers::enforce_k_eq(
                    cs,
                    &parent_r_var,
                    &child_r_var,
                    &format!("step_{}_dec_r_eq_child_{}_idx_{}", step_idx, i, idx),
                );
            }
        }

        // Optional NC channel: parent.s_col == child.s_col (when present), and reject partial presence.
        let want_nc_channel = !(parent.s_col.is_empty() && parent.y_zcol.is_empty());
        if want_nc_channel {
            let s_len = parent.s_col.len();
            for (i, child) in children.iter().enumerate() {
                if child.s_col.len() != s_len {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "DEC s_col length mismatch at step {}, child {}",
                        step_idx, i
                    )));
                }
                for idx in 0..s_len {
                    let parent_s = helpers::alloc_k_from_neo(
                        cs,
                        parent.s_col[idx],
                        &format!("step_{}_dec_parent_s_col_{}_{}", step_idx, i, idx),
                    )?;
                    let child_s = helpers::alloc_k_from_neo(
                        cs,
                        child.s_col[idx],
                        &format!("step_{}_dec_child_{}_s_col_{}", step_idx, i, idx),
                    )?;
                    helpers::enforce_k_eq(
                        cs,
                        &parent_s,
                        &child_s,
                        &format!("step_{}_dec_s_col_eq_child_{}_idx_{}", step_idx, i, idx),
                    );
                }
            }
        } else if children
            .iter()
            .any(|ch| !(ch.s_col.is_empty() && ch.y_zcol.is_empty()))
        {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "DEC at step {}: unexpected y_zcol channel on child",
                step_idx
            )));
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

        let parent_y_vars = helpers::alloc_y_table_from_neo(cs, &parent.y, &format!("step_{}_dec_parent_y", step_idx))?;

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
            children_y_vars.push(helpers::alloc_y_table_from_neo(
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

        // y_zcol_parent = Σ b^i · y_zcol_i (when present).
        if want_nc_channel {
            if parent.y_zcol.len() != d_pad {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "DEC at step {}: parent y_zcol length mismatch (got {}, expected {})",
                    step_idx,
                    parent.y_zcol.len(),
                    d_pad
                )));
            }

            let mut parent_yz_vars = Vec::with_capacity(d_pad);
            for idx in 0..d_pad {
                parent_yz_vars.push(helpers::alloc_k_from_neo(
                    cs,
                    parent.y_zcol[idx],
                    &format!("step_{}_dec_parent_y_zcol_{}", step_idx, idx),
                )?);
            }

            let mut child_yz_vars: Vec<Vec<KNumVar>> = Vec::with_capacity(children.len());
            for (i, child) in children.iter().enumerate() {
                if child.y_zcol.len() != d_pad {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "DEC at step {}: child {} y_zcol length mismatch (got {}, expected {})",
                        step_idx,
                        i,
                        child.y_zcol.len(),
                        d_pad
                    )));
                }
                let mut vars = Vec::with_capacity(d_pad);
                for (idx, &val) in child.y_zcol.iter().enumerate() {
                    vars.push(helpers::alloc_k_from_neo(
                        cs,
                        val,
                        &format!("step_{}_dec_child_{}_y_zcol_{}", step_idx, i, idx),
                    )?);
                }
                child_yz_vars.push(vars);
            }

            for r_idx in 0..d_pad {
                cs.enforce(
                    || format!("step_{}_dec_y_zcol_c0_r{}", step_idx, r_idx),
                    |lc| {
                        let mut res = lc;
                        let mut pow = CircuitF::from(1u64);
                        for child_vars in child_yz_vars.iter() {
                            res = res + (pow, child_vars[r_idx].c0);
                            pow *= CircuitF::from(self.base_b as u64);
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_yz_vars[r_idx].c0,
                );
                cs.enforce(
                    || format!("step_{}_dec_y_zcol_c1_r{}", step_idx, r_idx),
                    |lc| {
                        let mut res = lc;
                        let mut pow = CircuitF::from(1u64);
                        for child_vars in child_yz_vars.iter() {
                            res = res + (pow, child_vars[r_idx].c1);
                            pow *= CircuitF::from(self.base_b as u64);
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_yz_vars[r_idx].c1,
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
        let final_witness = witness
            .fold_run
            .compute_final_outputs(&self.instance.initial_accumulator);

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

            let pub_X_vars = helpers::alloc_matrix_from_neo(cs, &pub_me.X, &format!("acc_final_pub_{}_X", idx))?;
            let wit_X_vars = helpers::alloc_matrix_from_neo(cs, &wit_me.X, &format!("acc_final_wit_{}_X", idx))?;

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
                let pub_r = helpers::alloc_k_from_neo(cs, pub_me.r[j], &format!("acc_final_pub_{}_r_{}", idx, j))?;
                let wit_r = helpers::alloc_k_from_neo(cs, wit_me.r[j], &format!("acc_final_wit_{}_r_{}", idx, j))?;
                helpers::enforce_k_eq(cs, &pub_r, &wit_r, &format!("acc_final_r_eq_{}_{}", idx, j));
            }

            // Enforce s_col equality (NC channel binding).
            if pub_me.s_col.len() != wit_me.s_col.len() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "Final accumulator s_col length mismatch at index {}",
                    idx
                )));
            }
            for j in 0..pub_me.s_col.len() {
                let pub_s =
                    helpers::alloc_k_from_neo(cs, pub_me.s_col[j], &format!("acc_final_pub_{}_s_col_{}", idx, j))?;
                let wit_s =
                    helpers::alloc_k_from_neo(cs, wit_me.s_col[j], &format!("acc_final_wit_{}_s_col_{}", idx, j))?;
                helpers::enforce_k_eq(cs, &pub_s, &wit_s, &format!("acc_final_s_col_eq_{}_{}", idx, j));
            }

            // Enforce y equality
            if pub_me.y.len() != wit_me.y.len() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "Final accumulator y length mismatch at index {}",
                    idx
                )));
            }

            let pub_y_vars = helpers::alloc_y_table_from_neo(cs, &pub_me.y, &format!("acc_final_pub_{}_y", idx))?;
            let wit_y_vars = helpers::alloc_y_table_from_neo(cs, &wit_me.y, &format!("acc_final_wit_{}_y", idx))?;

            for j in 0..pub_me.y.len() {
                if pub_me.y[j].len() != wit_me.y[j].len() {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "Final accumulator y[j] length mismatch at index {}, j={}",
                        idx, j
                    )));
                }
                for r_idx in 0..pub_me.y[j].len() {
                    helpers::enforce_k_eq(
                        cs,
                        &pub_y_vars[j][r_idx],
                        &wit_y_vars[j][r_idx],
                        &format!("acc_final_y_eq_{}_{}_{}", idx, j, r_idx),
                    );
                }
            }

            // Enforce y_zcol equality (when present).
            if pub_me.y_zcol.len() != wit_me.y_zcol.len() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "Final accumulator y_zcol length mismatch at index {}",
                    idx
                )));
            }
            for r_idx in 0..pub_me.y_zcol.len() {
                let pub_yz = helpers::alloc_k_from_neo(
                    cs,
                    pub_me.y_zcol[r_idx],
                    &format!("acc_final_pub_{}_y_zcol_{}", idx, r_idx),
                )?;
                let wit_yz = helpers::alloc_k_from_neo(
                    cs,
                    wit_me.y_zcol[r_idx],
                    &format!("acc_final_wit_{}_y_zcol_{}", idx, r_idx),
                )?;
                helpers::enforce_k_eq(cs, &pub_yz, &wit_yz, &format!("acc_final_y_zcol_eq_{}_{}", idx, r_idx));
            }
        }

        Ok(())
    }
}

/// Implement Spartan2's `SpartanCircuit` trait for `FoldRunCircuit` using the
/// Goldilocks + Hash-MLE PCS engine. This lets Spartan2 treat the FoldRun
/// circuit as an R1CS provider.
impl SpartanCircuitTrait<GoldilocksP3MerkleMleEngine> for FoldRunCircuit {
    fn public_values(&self) -> std::result::Result<Vec<CircuitF>, SynthesisError> {
        // Must mirror `allocate_public_inputs`: expose params/CCS digests as
        // 8 little-endian u32 limbs each, in that order.
        fn append_digest(out: &mut Vec<CircuitF>, digest: &[u8; 32]) {
            for chunk in digest.chunks(4) {
                let mut limb_bytes = [0u8; 4];
                limb_bytes.copy_from_slice(chunk);
                let limb_u32 = u32::from_le_bytes(limb_bytes);
                out.push(CircuitF::from(limb_u32 as u64));
            }
        }

        let mut vals = Vec::with_capacity(16);
        append_digest(&mut vals, &self.instance.params_digest);
        append_digest(&mut vals, &self.instance.ccs_digest);
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
