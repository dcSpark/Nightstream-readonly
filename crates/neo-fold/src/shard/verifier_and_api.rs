use super::*;

pub fn fold_shard_prove<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<ShardProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let (proof, _final_main_wits, _val_lane_wits) = fold_shard_prove_impl(
        false,
        mode,
        tr,
        params,
        s_me,
        steps,
        0,
        acc_init,
        acc_wit_init,
        l,
        mixers,
        None,
        None,
        None,
    )?;
    Ok(proof)
}

pub(crate) fn fold_shard_prove_with_context<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
    ctx: &ShardProverContext,
) -> Result<ShardProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let (proof, _final_main_wits, _val_lane_wits) = fold_shard_prove_impl(
        false,
        mode,
        tr,
        params,
        s_me,
        steps,
        0,
        acc_init,
        acc_wit_init,
        l,
        mixers,
        None,
        Some(ctx),
        None,
    )?;
    Ok(proof)
}

pub(crate) fn fold_shard_prove_with_context_and_step_timings<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
    ctx: &ShardProverContext,
) -> Result<(ShardProof, Vec<f64>), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let mut step_prove_ms = Vec::with_capacity(steps.len());
    let (proof, _final_main_wits, _val_lane_wits) = fold_shard_prove_impl(
        false,
        mode,
        tr,
        params,
        s_me,
        steps,
        0,
        acc_init,
        acc_wit_init,
        l,
        mixers,
        None,
        Some(ctx),
        Some(&mut step_prove_ms),
    )?;
    Ok((proof, step_prove_ms))
}

pub fn fold_shard_prove_with_output_binding<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    final_memory_state: &[F],
) -> Result<ShardProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let (proof, _final_main_wits, _val_lane_wits) = fold_shard_prove_impl(
        false,
        mode,
        tr,
        params,
        s_me,
        steps,
        0,
        acc_init,
        acc_wit_init,
        l,
        mixers,
        Some((ob_cfg, final_memory_state)),
        None,
        None,
    )?;
    Ok(proof)
}

pub(crate) fn fold_shard_prove_with_output_binding_with_context<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    final_memory_state: &[F],
    ctx: &ShardProverContext,
) -> Result<ShardProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let (proof, _final_main_wits, _val_lane_wits) = fold_shard_prove_impl(
        false,
        mode,
        tr,
        params,
        s_me,
        steps,
        0,
        acc_init,
        acc_wit_init,
        l,
        mixers,
        Some((ob_cfg, final_memory_state)),
        Some(ctx),
        None,
    )?;
    Ok(proof)
}

pub fn fold_shard_prove_with_witnesses<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<(ShardProof, ShardFoldOutputs<Cmt, F, K>, ShardFoldWitnesses<F>), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let (proof, final_main_wits, val_lane_wits) = fold_shard_prove_impl(
        true,
        mode,
        tr,
        params,
        s_me,
        steps,
        0,
        acc_init,
        acc_wit_init,
        l,
        mixers,
        None,
        None,
        None,
    )?;
    let outputs = proof.compute_fold_outputs(acc_init);
    if outputs.obligations.main.len() != final_main_wits.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "final main witness count mismatch (have {}, need {})",
            final_main_wits.len(),
            outputs.obligations.main.len()
        )));
    }
    if outputs.obligations.val.len() != val_lane_wits.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "val-lane witness count mismatch (have {}, need {})",
            val_lane_wits.len(),
            outputs.obligations.val.len()
        )));
    }
    Ok((
        proof,
        outputs,
        ShardFoldWitnesses {
            final_main_wits,
            val_lane_wits,
        },
    ))
}

/// Same as `fold_shard_prove_with_witnesses`, but offsets the per-step transcript index by `step_idx_offset`.
///
/// This is useful for "continuation" style proving across multiple calls while preserving a globally
/// increasing step index for transcript domain separation.
pub fn fold_shard_prove_with_witnesses_with_step_offset<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
    step_idx_offset: usize,
) -> Result<(ShardProof, ShardFoldOutputs<Cmt, F, K>, ShardFoldWitnesses<F>), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let (proof, final_main_wits, val_lane_wits) = fold_shard_prove_impl(
        true,
        mode,
        tr,
        params,
        s_me,
        steps,
        step_idx_offset,
        acc_init,
        acc_wit_init,
        l,
        mixers,
        None,
        None,
        None,
    )?;
    let outputs = proof.compute_fold_outputs(acc_init);
    if outputs.obligations.main.len() != final_main_wits.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "final main witness count mismatch (have {}, need {})",
            final_main_wits.len(),
            outputs.obligations.main.len()
        )));
    }
    if outputs.obligations.val.len() != val_lane_wits.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "val-lane witness count mismatch (have {}, need {})",
            val_lane_wits.len(),
            outputs.obligations.val.len()
        )));
    }
    Ok((
        proof,
        outputs,
        ShardFoldWitnesses {
            final_main_wits,
            val_lane_wits,
        },
    ))
}

// ============================================================================
// Shard Verification
// ============================================================================

pub(crate) fn fold_shard_verify_impl<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    step_idx_offset: usize,
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: Option<&crate::output_binding::OutputBindingConfig>,
    prover_ctx: Option<&ShardProverContext>,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    for (step_idx, step) in steps.iter().enumerate() {
        if step.lut_insts.is_empty() && step.mem_insts.is_empty() {
            continue;
        }
        let is_shared_step = step.lut_insts.iter().all(|inst| inst.comms.is_empty())
            && step.mem_insts.iter().all(|inst| inst.comms.is_empty());
        if !is_shared_step {
            return Err(PiCcsError::InvalidInput(format!(
                "legacy no-shared CPU bus mode was removed; step_idx={step_idx} must use shared-bus statement format"
            )));
        }
    }
    tr.append_message(b"shard/cpu_bus_mode", &[1u8]);
    let (s, cpu_bus) = crate::memory_sidecar::cpu_bus::prepare_ccs_for_shared_cpu_bus_steps(s_me, steps)?;
    let dims = utils::build_dims_and_policy(params, s)?;
    let utils::Dims {
        ell_d,
        ell_n,
        ell_m,
        ell,
        d_sc,
        ..
    } = dims;
    let ring = ccs::RotRing::goldilocks();

    if steps.len() != proof.steps.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "step count mismatch: public {} vs proof {}",
            steps.len(),
            proof.steps.len()
        )));
    }
    if ob_cfg.is_some() && steps.is_empty() {
        return Err(PiCcsError::InvalidInput("output binding requires >= 1 step".into()));
    }
    if ob_cfg.is_none() && proof.output_proof.is_some() {
        return Err(PiCcsError::InvalidInput(
            "shard proof contains output binding, but verifier did not supply OutputBindingConfig".into(),
        ));
    }
    if ob_cfg.is_some() && proof.output_proof.is_none() {
        return Err(PiCcsError::InvalidInput(
            "verifier supplied OutputBindingConfig, but shard proof has no output binding".into(),
        ));
    }

    let mut accumulator = acc_init.to_vec();
    let mut val_lane_obligations: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let ccs_sparse_cache: Option<Arc<SparseCache<F>>> = if mode_uses_sparse_cache(&mode) {
        Some(
            prover_ctx
                .and_then(|ctx| ctx.ccs_sparse_cache.clone())
                .unwrap_or_else(|| Arc::new(SparseCache::build(s))),
        )
    } else {
        None
    };
    let ccs_mat_digest = prover_ctx
        .map(|ctx| ctx.ccs_mat_digest.clone())
        .unwrap_or_else(|| utils::digest_ccs_matrices_with_sparse_cache(s, ccs_sparse_cache.as_deref()));

    for (idx, (step, step_proof)) in steps.iter().zip(proof.steps.iter()).enumerate() {
        let step_idx = step_idx_offset
            .checked_add(idx)
            .ok_or_else(|| PiCcsError::InvalidInput("step index overflow".into()))?;
        let has_prev = idx > 0;
        absorb_step_memory(tr, step);

        let include_ob = ob_cfg.is_some() && (idx + 1 == steps.len());
        let mut ob_state: Option<neo_memory::output_check::OutputSumcheckState> = None;
        let mut ob_inc_total_degree_bound: Option<usize> = None;

        if include_ob {
            let cfg =
                ob_cfg.ok_or_else(|| PiCcsError::InvalidInput("output binding enabled but config missing".into()))?;
            let ob_proof = proof
                .output_proof
                .as_ref()
                .ok_or_else(|| PiCcsError::InvalidInput("output binding enabled but proof missing".into()))?;

            if cfg.mem_idx >= step.mem_insts.len() {
                return Err(PiCcsError::InvalidInput("output binding mem_idx out of range".into()));
            }
            let mem_inst = step
                .mem_insts
                .get(cfg.mem_idx)
                .ok_or_else(|| PiCcsError::InvalidInput("output binding mem_idx out of range".into()))?;
            let expected_k = 1usize
                .checked_shl(cfg.num_bits as u32)
                .ok_or_else(|| PiCcsError::InvalidInput("output binding: 2^num_bits overflow".into()))?;
            if mem_inst.k != expected_k {
                return Err(PiCcsError::InvalidInput(format!(
                    "output binding: cfg.num_bits implies k={}, but mem_inst.k={}",
                    expected_k, mem_inst.k
                )));
            }
            let ell_addr = mem_inst.twist_layout().lanes[0].ell_addr;
            if ell_addr != cfg.num_bits {
                return Err(PiCcsError::InvalidInput(format!(
                    "output binding: cfg.num_bits={}, but twist_layout.ell_addr={}",
                    cfg.num_bits, ell_addr
                )));
            }

            tr.append_message(b"shard/output_binding_start", &(step_idx as u64).to_le_bytes());
            tr.append_u64s(b"output_binding/mem_idx", &[cfg.mem_idx as u64]);
            tr.append_u64s(b"output_binding/num_bits", &[cfg.num_bits as u64]);

            let state = neo_memory::output_check::verify_output_sumcheck_rounds_get_state(
                tr,
                cfg.num_bits,
                cfg.program_io.clone(),
                &ob_proof.output_sc,
            )
            .map_err(|e| PiCcsError::ProtocolError(format!("output sumcheck failed: {e:?}")))?;
            ob_inc_total_degree_bound = Some(2 + ell_addr);
            ob_state = Some(state);
        }

        let mcs_inst = &step.mcs_inst;

        // --------------------------------------------------------------------
        // Route A: Verify shared-challenge batched sum-check (time/row rounds),
        // then finish CCS Ajtai rounds, then proceed with RLC→DEC as before.
        // --------------------------------------------------------------------

        // Bind CCS header + ME inputs and sample public challenges.
        utils::bind_header_and_instances_with_digest(
            tr,
            params,
            &s,
            core::slice::from_ref(mcs_inst),
            dims,
            &ccs_mat_digest,
        )?;
        utils::bind_me_inputs(tr, &accumulator)?;
        let mut ch = utils::sample_challenges(tr, ell_d, ell)?;
        if step_proof.fold.ccs_proof.variant == crate::optimized_engine::PiCcsProofVariant::SplitNcV1 {
            ch.beta_m = utils::sample_beta_m(tr, ell_m)?;
        }
        let expected_ch = &step_proof.fold.ccs_proof.challenges_public;
        if expected_ch.alpha != ch.alpha
            || expected_ch.beta_a != ch.beta_a
            || expected_ch.beta_r != ch.beta_r
            || expected_ch.beta_m != ch.beta_m
            || expected_ch.gamma != ch.gamma
        {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS challenges_public mismatch",
                idx
            )));
        }

        // Public initial sum T for CCS sumcheck (engine-selected).
        let claimed_initial = match &mode {
            FoldingMode::Optimized => crate::optimized_engine::claimed_initial_sum_from_inputs(&s, &ch, &accumulator),
            #[cfg(feature = "paper-exact")]
            FoldingMode::PaperExact => {
                crate::paper_exact_engine::claimed_initial_sum_from_inputs(&s, &ch, &accumulator)
            }
            #[cfg(feature = "paper-exact")]
            FoldingMode::OptimizedWithCrosscheck(_) => {
                crate::optimized_engine::claimed_initial_sum_from_inputs(&s, &ch, &accumulator)
            }
        };
        if let Some(x) = step_proof.fold.ccs_proof.sc_initial_sum {
            if x != claimed_initial {
                return Err(PiCcsError::SumcheckError(
                    "initial sum mismatch: proof claims different value than public T".into(),
                ));
            }
        }
        tr.append_fields(b"sumcheck/initial_sum", &claimed_initial.as_coeffs());

        // Route A memory checks use a separate transcript-derived cycle point `r_cycle`
        // to form χ_{r_cycle}(t) weights inside their sum-check polynomials.
        let r_cycle: Vec<K> =
            ts::sample_ext_point(tr, b"route_a/r_cycle", b"route_a/cycle/0", b"route_a/cycle/1", ell_n);

        let shout_pre = crate::memory_sidecar::memory::verify_shout_addr_pre_time(tr, step, &step_proof.mem, step_idx)?;
        let twist_pre = crate::memory_sidecar::memory::verify_twist_addr_pre_time(tr, step, &step_proof.mem)?;
        let wb_enabled = crate::memory_sidecar::memory::wb_wp_required_for_step_instance(step);
        let wp_enabled = crate::memory_sidecar::memory::wb_wp_required_for_step_instance(step);
        let decode_stage_enabled = crate::memory_sidecar::memory::decode_stage_required_for_step_instance(step);
        let width_stage_enabled = crate::memory_sidecar::memory::width_stage_required_for_step_instance(step);
        let control_stage_enabled = crate::memory_sidecar::memory::control_stage_required_for_step_instance(step);
        let crate::memory_sidecar::route_a_time::RouteABatchedTimeVerifyOutput { r_time, final_values } =
            crate::memory_sidecar::route_a_time::verify_route_a_batched_time(
                tr,
                step_idx,
                ell_n,
                d_sc,
                claimed_initial,
                step,
                &step_proof.batched_time,
                wb_enabled,
                wp_enabled,
                decode_stage_enabled,
                width_stage_enabled,
                control_stage_enabled,
                ob_inc_total_degree_bound,
            )?;

        // CCS proof structure consistency with batched time proof.
        let want_rounds_total = ell_n + ell_d;
        if step_proof.fold.ccs_proof.sumcheck_rounds.len() != want_rounds_total {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: CCS sumcheck_rounds.len()={}, expected {}",
                idx,
                step_proof.fold.ccs_proof.sumcheck_rounds.len(),
                want_rounds_total
            )));
        }
        if step_proof.fold.ccs_proof.sumcheck_challenges.len() != want_rounds_total {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: CCS sumcheck_challenges.len()={}, expected {}",
                idx,
                step_proof.fold.ccs_proof.sumcheck_challenges.len(),
                want_rounds_total
            )));
        }
        for (round_idx, (a, b)) in step_proof
            .fold
            .ccs_proof
            .sumcheck_rounds
            .iter()
            .take(ell_n)
            .zip(step_proof.batched_time.round_polys[0].iter())
            .enumerate()
        {
            if a != b {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: CCS time round poly mismatch at round {}",
                    idx, round_idx
                )));
            }
        }

        if step_proof.fold.ccs_proof.sumcheck_challenges[..ell_n] != r_time {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: CCS time challenges mismatch with r_time",
                idx
            )));
        }

        let expected_k = accumulator.len() + 1;
        if step_proof.fold.ccs_out.len() != expected_k {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS returned {} outputs; expected k={}",
                idx,
                step_proof.fold.ccs_out.len(),
                expected_k
            )));
        }
        if step_proof.fold.ccs_out.is_empty() {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS produced empty ccs_out",
                idx
            )));
        }
        if step_proof.fold.ccs_out[0].r != r_time {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS output r != r_time (Route A requires shared r)",
                idx
            )));
        }

        // Bind Π_CCS outputs to the public MCS instance and carried ME inputs.
        //
        // - Commitments must match (Π_CCS does not change commitments).
        // - `X` must match the digit-decomposition of public `x` for the MCS output.
        // - `X` must match the carried ME inputs for subsequent outputs.
        {
            let out0 = &step_proof.fold.ccs_out[0];
            if out0.c != mcs_inst.c {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: Π_CCS output[0].c does not match mcs_inst.c",
                    idx
                )));
            }
            if out0.m_in != mcs_inst.m_in {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: Π_CCS output[0].m_in={}, expected {}",
                    idx, out0.m_in, mcs_inst.m_in
                )));
            }
            if out0.X.rows() != D || out0.X.cols() != mcs_inst.m_in {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: Π_CCS output[0].X has shape {}×{}, expected {}×{}",
                    idx,
                    out0.X.rows(),
                    out0.X.cols(),
                    D,
                    mcs_inst.m_in
                )));
            }

            for (i, inp) in accumulator.iter().enumerate() {
                let out = &step_proof.fold.ccs_out[i + 1];
                if out.c != inp.c {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: Π_CCS output[{}].c does not match accumulator[{}].c",
                        idx,
                        i + 1,
                        i
                    )));
                }
                if out.m_in != inp.m_in {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: Π_CCS output[{}].m_in={}, expected {}",
                        idx,
                        i + 1,
                        out.m_in,
                        inp.m_in
                    )));
                }
                if out.X != inp.X {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: Π_CCS output[{}].X does not match accumulator[{}].X",
                        idx,
                        i + 1,
                        i
                    )));
                }
            }
        }

        // Finish CCS Ajtai rounds alone (continuing transcript state after batched rounds).
        let ajtai_rounds = &step_proof.fold.ccs_proof.sumcheck_rounds[ell_n..];
        let (ajtai_chals, running_sum, ok) =
            verify_sumcheck_rounds_ds(tr, b"ccs/ajtai", step_idx, d_sc, final_values[0], ajtai_rounds);
        if !ok {
            return Err(PiCcsError::SumcheckError("Π_CCS Ajtai rounds invalid".into()));
        }

        // Verify stored sumcheck challenges/final match transcript-derived values.
        let mut r_all = r_time.clone();
        r_all.extend_from_slice(&ajtai_chals);
        if r_all != step_proof.fold.ccs_proof.sumcheck_challenges {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS sumcheck challenges mismatch",
                idx
            )));
        }
        if running_sum != step_proof.fold.ccs_proof.sumcheck_final {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS sumcheck_final mismatch",
                idx
            )));
        }

        // Validate ME input r length (required by RHS assembly if k>1).
        for (i, me) in accumulator.iter().enumerate() {
            if me.r.len() != ell_n {
                return Err(PiCcsError::InvalidInput(format!(
                    "step {}: ME input r length mismatch at accumulator #{}: expected {}, got {}",
                    idx,
                    i,
                    ell_n,
                    me.r.len()
                )));
            }
        }

        if step_proof.fold.ccs_proof.variant != crate::optimized_engine::PiCcsProofVariant::SplitNcV1 {
            return Err(PiCcsError::ProtocolError("unsupported Π_CCS proof variant".into()));
        }

        // FE-only terminal identity.
        let rhs_fe = crate::paper_exact_engine::rhs_terminal_identity_fe_paper_exact(
            &s,
            params,
            &ch,
            &r_time,
            &ajtai_chals,
            &step_proof.fold.ccs_out,
            accumulator.first().map(|mi| mi.r.as_slice()),
        );
        if running_sum != rhs_fe {
            return Err(PiCcsError::SumcheckError(
                "Π_CCS FE-only terminal identity check failed".into(),
            ));
        }

        // NC-only sumcheck + terminal identity.
        if step_proof.fold.ccs_proof.sumcheck_rounds_nc.is_empty() {
            return Err(PiCcsError::InvalidInput(
                "Π_CCS SplitNcV1 requires non-empty sumcheck_rounds_nc".into(),
            ));
        }
        if let Some(x) = step_proof.fold.ccs_proof.sc_initial_sum_nc {
            if x != K::ZERO {
                return Err(PiCcsError::InvalidInput(
                    "Π_CCS SplitNcV1 requires sc_initial_sum_nc == 0".into(),
                ));
            }
        }
        let want_nc_rounds_total = ell_m
            .checked_add(ell_d)
            .ok_or_else(|| PiCcsError::ProtocolError("ell_m + ell_d overflow".into()))?;
        if step_proof.fold.ccs_proof.sumcheck_rounds_nc.len() != want_nc_rounds_total {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: Π_CCS NC sumcheck_rounds_nc.len()={}, expected {}",
                idx,
                step_proof.fold.ccs_proof.sumcheck_rounds_nc.len(),
                want_nc_rounds_total
            )));
        }
        if step_proof.fold.ccs_proof.sumcheck_challenges_nc.len() != want_nc_rounds_total {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: Π_CCS NC sumcheck_challenges_nc.len()={}, expected {}",
                idx,
                step_proof.fold.ccs_proof.sumcheck_challenges_nc.len(),
                want_nc_rounds_total
            )));
        }

        let (nc_chals, running_sum_nc, ok_nc) = verify_sumcheck_rounds_ds(
            tr,
            b"ccs/nc",
            step_idx,
            d_sc,
            K::ZERO,
            &step_proof.fold.ccs_proof.sumcheck_rounds_nc,
        );
        if !ok_nc {
            return Err(PiCcsError::SumcheckError("Π_CCS NC rounds invalid".into()));
        }

        if nc_chals != step_proof.fold.ccs_proof.sumcheck_challenges_nc {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS NC sumcheck challenges mismatch",
                idx
            )));
        }
        if running_sum_nc != step_proof.fold.ccs_proof.sumcheck_final_nc {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_CCS sumcheck_final_nc mismatch",
                idx
            )));
        }

        let (s_col_prime, alpha_prime_nc) = nc_chals.split_at(ell_m);
        let d_pad = 1usize
            .checked_shl(ell_d as u32)
            .ok_or_else(|| PiCcsError::ProtocolError("2^ell_d overflow".into()))?;
        for (out_idx, out) in step_proof.fold.ccs_out.iter().enumerate() {
            if out.r != r_time {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: Π_CCS output[{out_idx}] r != r_time",
                    idx
                )));
            }
            if out.s_col.as_slice() != s_col_prime {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: Π_CCS output[{out_idx}] s_col mismatch",
                    idx
                )));
            }
            if out.y_zcol.len() != d_pad {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: Π_CCS output[{out_idx}] y_zcol.len()={}, expected {}",
                    idx,
                    out.y_zcol.len(),
                    d_pad
                )));
            }
        }

        let rhs_nc = crate::paper_exact_engine::rhs_terminal_identity_nc_paper_exact(
            params,
            &ch,
            s_col_prime,
            alpha_prime_nc,
            &step_proof.fold.ccs_out,
        );
        if running_sum_nc != rhs_nc {
            return Err(PiCcsError::SumcheckError(
                "Π_CCS NC terminal identity check failed".into(),
            ));
        }

        let observed_digest = tr.digest32();
        if observed_digest != step_proof.fold.ccs_proof.header_digest.as_slice() {
            return Err(PiCcsError::ProtocolError("Π_CCS header digest mismatch".into()));
        }
        let expected_digest: [u8; 32] = step_proof
            .fold
            .ccs_proof
            .header_digest
            .as_slice()
            .try_into()
            .map_err(|_| PiCcsError::ProtocolError("Π_CCS header digest must be 32 bytes".into()))?;
        for (out_idx, out) in step_proof.fold.ccs_out.iter().enumerate() {
            if out.fold_digest != expected_digest {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: Π_CCS output[{out_idx}] fold_digest mismatch",
                    idx
                )));
            }
        }

        // Verify mem proofs (shared CPU bus only).
        let prev_step = (idx > 0).then(|| &steps[idx - 1]);
        let mem_out = crate::memory_sidecar::memory::verify_route_a_memory_step(
            tr,
            &cpu_bus,
            s.m,
            s.t(),
            step,
            prev_step,
            &step_proof.fold.ccs_out[0],
            &r_time,
            &r_cycle,
            &final_values,
            &step_proof.batched_time.claimed_sums,
            1, // claim 0 is CCS/time
            &step_proof.mem,
            &shout_pre,
            &twist_pre,
            step_idx,
        )?;

        let expected_consumed = if include_ob {
            final_values
                .len()
                .checked_sub(1)
                .ok_or_else(|| PiCcsError::ProtocolError("missing output binding claim".into()))?
        } else {
            final_values.len()
        };
        if mem_out.claim_idx_end != expected_consumed {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: batched claim index mismatch (consumed {}, expected {})",
                idx, mem_out.claim_idx_end, expected_consumed
            )));
        }

        if include_ob {
            let cfg =
                ob_cfg.ok_or_else(|| PiCcsError::InvalidInput("output binding enabled but config missing".into()))?;
            let ob_state = ob_state
                .take()
                .ok_or_else(|| PiCcsError::ProtocolError("output sumcheck state missing".into()))?;

            let inc_idx = final_values
                .len()
                .checked_sub(1)
                .ok_or_else(|| PiCcsError::ProtocolError("missing inc_total claim".into()))?;
            if step_proof.batched_time.labels.get(inc_idx).copied() != Some(crate::output_binding::OB_INC_TOTAL_LABEL) {
                return Err(PiCcsError::ProtocolError("output binding claim not last".into()));
            }

            let inc_total_claim = *step_proof
                .batched_time
                .claimed_sums
                .get(inc_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("missing inc_total claimed_sum".into()))?;
            let inc_total_final = *final_values
                .get(inc_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("missing inc_total final_value".into()))?;

            let twist_open = mem_out
                .twist_time_openings
                .get(cfg.mem_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("missing twist_time_openings for mem_idx".into()))?;
            let inc_terminal = crate::output_binding::inc_terminal_from_time_openings(twist_open, &ob_state.r_prime)
                .map_err(|e| PiCcsError::ProtocolError(format!("inc_total terminal mismatch: {e:?}")))?;
            if inc_total_final != inc_terminal {
                return Err(PiCcsError::ProtocolError("inc_total terminal mismatch".into()));
            }

            let mem_inst = step
                .mem_insts
                .get(cfg.mem_idx)
                .ok_or_else(|| PiCcsError::InvalidInput("output binding mem_idx out of range".into()))?;
            let expected_k = 1usize
                .checked_shl(cfg.num_bits as u32)
                .ok_or_else(|| PiCcsError::InvalidInput("output binding: 2^num_bits overflow".into()))?;
            if mem_inst.k != expected_k {
                return Err(PiCcsError::InvalidInput(format!(
                    "output binding: cfg.num_bits implies k={}, but mem_inst.k={}",
                    expected_k, mem_inst.k
                )));
            }
            let ell_addr = mem_inst.twist_layout().lanes[0].ell_addr;
            if ell_addr != cfg.num_bits {
                return Err(PiCcsError::InvalidInput(format!(
                    "output binding: cfg.num_bits={}, but twist_layout.ell_addr={}",
                    cfg.num_bits, ell_addr
                )));
            }
            let val_init = crate::output_binding::val_init_from_mem_init(&mem_inst.init, mem_inst.k, &ob_state.r_prime)
                .map_err(|e| PiCcsError::ProtocolError(format!("MemInit eval failed: {e:?}")))?;

            let val_final_at_r_prime = val_init + inc_total_claim;
            let expected_out = ob_state.eq_eval * ob_state.io_mask_eval * (val_final_at_r_prime - ob_state.val_io_eval);
            if expected_out != ob_state.output_final {
                return Err(PiCcsError::ProtocolError("output binding final check failed".into()));
            }
        }

        validate_me_batch_invariants(&step_proof.fold.ccs_out, "verify step ccs outputs")?;
        verify_rlc_dec_lane(
            RlcLane::Main,
            tr,
            params,
            &s,
            &ring,
            ell_d,
            mixers,
            step_idx,
            &step_proof.fold.ccs_out,
            &step_proof.fold.rlc_rhos,
            &step_proof.fold.rlc_parent,
            &step_proof.fold.dec_children,
        )?;

        accumulator = step_proof.fold.dec_children.clone();

        // Phase 2: Verify folding lanes for ME claims evaluated at r_val.
        if step_proof.mem.val_me_claims.is_empty() {
            if !step_proof.val_fold.is_empty() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: unexpected val_fold proof(s) (no r_val ME claims)",
                    idx
                )));
            }
        } else {
            tr.append_message(b"fold/val_lane_start", &(step_idx as u64).to_le_bytes());
            let expected = 1usize + usize::from(has_prev);
            if step_proof.mem.val_me_claims.len() != expected {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: val_me_claims count mismatch in shared-bus mode (have {}, expected {})",
                    idx,
                    step_proof.mem.val_me_claims.len(),
                    expected
                )));
            }
            if step_proof.val_fold.len() != expected {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: val_fold count mismatch in shared-bus mode (have {}, expected {})",
                    idx,
                    step_proof.val_fold.len(),
                    expected
                )));
            }

            for (claim_idx, (me, proof)) in step_proof
                .mem
                .val_me_claims
                .iter()
                .zip(step_proof.val_fold.iter())
                .enumerate()
            {
                let ctx = match claim_idx {
                    0 => "cpu",
                    1 => "cpu_prev",
                    _ => {
                        return Err(PiCcsError::ProtocolError(
                            "unexpected extra r_val ME claim in shared-bus mode".into(),
                        ));
                    }
                };
                tr.append_message(b"fold/val_lane_claim_idx", &(claim_idx as u64).to_le_bytes());
                tr.append_message(b"fold/val_lane_claim_ctx", ctx.as_bytes());
                verify_rlc_dec_lane(
                    RlcLane::Val,
                    tr,
                    params,
                    &s,
                    &ring,
                    ell_d,
                    mixers,
                    step_idx,
                    core::slice::from_ref(me),
                    &proof.rlc_rhos,
                    &proof.rlc_parent,
                    &proof.dec_children,
                )
                .map_err(|e| {
                    PiCcsError::ProtocolError(format!(
                        "step {} val_fold(shared) claim {} ({ctx}) verify failed: {e:?}",
                        idx, claim_idx
                    ))
                })?;
                val_lane_obligations.extend_from_slice(&proof.dec_children);
            }
        }

        if step_proof.mem.wb_me_claims.is_empty() {
            if !step_proof.wb_fold.is_empty() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: unexpected wb_fold proof(s) (no WB ME claims)",
                    idx
                )));
            }
        } else {
            if step_proof.wb_fold.len() != step_proof.mem.wb_me_claims.len() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: wb_fold count mismatch (have {}, expected {})",
                    idx,
                    step_proof.wb_fold.len(),
                    step_proof.mem.wb_me_claims.len()
                )));
            }
            tr.append_message(b"fold/wb_lane_start", &(step_idx as u64).to_le_bytes());
            for (claim_idx, (me, proof)) in step_proof
                .mem
                .wb_me_claims
                .iter()
                .zip(step_proof.wb_fold.iter())
                .enumerate()
            {
                tr.append_message(b"fold/wb_lane_claim_idx", &(claim_idx as u64).to_le_bytes());
                verify_rlc_dec_lane(
                    RlcLane::Val,
                    tr,
                    params,
                    &s,
                    &ring,
                    ell_d,
                    mixers,
                    step_idx,
                    core::slice::from_ref(me),
                    &proof.rlc_rhos,
                    &proof.rlc_parent,
                    &proof.dec_children,
                )
                .map_err(|e| {
                    PiCcsError::ProtocolError(format!("step {} wb_fold claim {} verify failed: {e:?}", idx, claim_idx))
                })?;
                val_lane_obligations.extend_from_slice(&proof.dec_children);
            }
        }

        if step_proof.mem.wp_me_claims.is_empty() {
            if !step_proof.wp_fold.is_empty() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: unexpected wp_fold proof(s) (no WP ME claims)",
                    idx
                )));
            }
        } else {
            if step_proof.wp_fold.len() != step_proof.mem.wp_me_claims.len() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: wp_fold count mismatch (have {}, expected {})",
                    idx,
                    step_proof.wp_fold.len(),
                    step_proof.mem.wp_me_claims.len()
                )));
            }
            tr.append_message(b"fold/wp_lane_start", &(step_idx as u64).to_le_bytes());
            for (claim_idx, (me, proof)) in step_proof
                .mem
                .wp_me_claims
                .iter()
                .zip(step_proof.wp_fold.iter())
                .enumerate()
            {
                tr.append_message(b"fold/wp_lane_claim_idx", &(claim_idx as u64).to_le_bytes());
                verify_rlc_dec_lane(
                    RlcLane::Val,
                    tr,
                    params,
                    &s,
                    &ring,
                    ell_d,
                    mixers,
                    step_idx,
                    core::slice::from_ref(me),
                    &proof.rlc_rhos,
                    &proof.rlc_parent,
                    &proof.dec_children,
                )
                .map_err(|e| {
                    PiCcsError::ProtocolError(format!("step {} wp_fold claim {} verify failed: {e:?}", idx, claim_idx))
                })?;
                val_lane_obligations.extend_from_slice(&proof.dec_children);
            }
        }

        tr.append_message(b"fold/step_done", &(step_idx as u64).to_le_bytes());
    }

    Ok(ShardFoldOutputs {
        obligations: ShardObligations {
            main: accumulator,
            val: val_lane_obligations,
        },
    })
}

pub fn fold_shard_verify<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_verify_impl(mode, tr, params, s_me, steps, 0, acc_init, proof, mixers, None, None)
}

/// Same as `fold_shard_verify`, but offsets the per-step transcript index by `step_idx_offset`.
pub fn fold_shard_verify_with_step_offset<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    step_idx_offset: usize,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_verify_impl(
        mode,
        tr,
        params,
        s_me,
        steps,
        step_idx_offset,
        acc_init,
        proof,
        mixers,
        None,
        None,
    )
}

pub fn fold_shard_verify_with_step_linking<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    step_linking: &StepLinkingConfig,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    check_step_linking(steps, step_linking)?;
    fold_shard_verify(mode, tr, params, s_me, steps, acc_init, proof, mixers)
}

pub fn fold_shard_verify_with_output_binding<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_verify_impl(
        mode,
        tr,
        params,
        s_me,
        steps,
        0,
        acc_init,
        proof,
        mixers,
        Some(ob_cfg),
        None,
    )
}

pub(crate) fn fold_shard_verify_with_context<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    prover_ctx: &ShardProverContext,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_verify_impl(
        mode,
        tr,
        params,
        s_me,
        steps,
        0,
        acc_init,
        proof,
        mixers,
        None,
        Some(prover_ctx),
    )
}

pub(crate) fn fold_shard_verify_with_step_linking_with_context<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    step_linking: &StepLinkingConfig,
    prover_ctx: &ShardProverContext,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    check_step_linking(steps, step_linking)?;
    fold_shard_verify_with_context(mode, tr, params, s_me, steps, acc_init, proof, mixers, prover_ctx)
}

pub(crate) fn fold_shard_verify_with_output_binding_with_context<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    prover_ctx: &ShardProverContext,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    fold_shard_verify_impl(
        mode,
        tr,
        params,
        s_me,
        steps,
        0,
        acc_init,
        proof,
        mixers,
        Some(ob_cfg),
        Some(prover_ctx),
    )
}

pub(crate) fn fold_shard_verify_with_output_binding_and_step_linking_with_context<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    step_linking: &StepLinkingConfig,
    prover_ctx: &ShardProverContext,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    check_step_linking(steps, step_linking)?;
    fold_shard_verify_with_output_binding_with_context(
        mode, tr, params, s_me, steps, acc_init, proof, mixers, ob_cfg, prover_ctx,
    )
}

pub fn fold_shard_verify_with_output_binding_and_step_linking<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    step_linking: &StepLinkingConfig,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    check_step_linking(steps, step_linking)?;
    fold_shard_verify_with_output_binding(mode, tr, params, s_me, steps, acc_init, proof, mixers, ob_cfg)
}

pub fn fold_shard_verify_and_finalize<MR, MB, Fin>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    finalizer: &mut Fin,
) -> Result<(), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
    Fin: ObligationFinalizer<Cmt, F, K, Error = PiCcsError>,
{
    let outputs = fold_shard_verify(mode, tr, params, s_me, steps, acc_init, proof, mixers)?;
    let report = finalizer.finalize(&outputs.obligations)?;
    outputs
        .obligations
        .require_all_finalized(report.did_finalize_main, report.did_finalize_val)?;
    Ok(())
}

pub fn fold_shard_verify_and_finalize_with_step_linking<MR, MB, Fin>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    step_linking: &StepLinkingConfig,
    finalizer: &mut Fin,
) -> Result<(), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
    Fin: ObligationFinalizer<Cmt, F, K, Error = PiCcsError>,
{
    check_step_linking(steps, step_linking)?;
    fold_shard_verify_and_finalize(mode, tr, params, s_me, steps, acc_init, proof, mixers, finalizer)
}

pub fn fold_shard_verify_and_finalize_with_output_binding<MR, MB, Fin>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    finalizer: &mut Fin,
) -> Result<(), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
    Fin: ObligationFinalizer<Cmt, F, K, Error = PiCcsError>,
{
    let outputs =
        fold_shard_verify_with_output_binding(mode, tr, params, s_me, steps, acc_init, proof, mixers, ob_cfg)?;
    let report = finalizer.finalize(&outputs.obligations)?;
    outputs
        .obligations
        .require_all_finalized(report.did_finalize_main, report.did_finalize_val)?;
    Ok(())
}

pub fn fold_shard_verify_and_finalize_with_output_binding_and_step_linking<MR, MB, Fin>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    step_linking: &StepLinkingConfig,
    finalizer: &mut Fin,
) -> Result<(), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
    Fin: ObligationFinalizer<Cmt, F, K, Error = PiCcsError>,
{
    check_step_linking(steps, step_linking)?;
    fold_shard_verify_and_finalize_with_output_binding(
        mode, tr, params, s_me, steps, acc_init, proof, mixers, ob_cfg, finalizer,
    )
}
