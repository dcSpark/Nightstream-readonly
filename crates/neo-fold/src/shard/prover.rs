use super::*;

#[derive(Clone)]
pub(crate) struct ShardProverContext {
    pub ccs_mat_digest: Vec<F>,
    pub ccs_sparse_cache: Option<Arc<SparseCache<F>>>,
}

#[inline]
pub(crate) fn mode_uses_sparse_cache(mode: &FoldingMode) -> bool {
    match mode {
        FoldingMode::Optimized => true,
        #[cfg(feature = "paper-exact")]
        FoldingMode::OptimizedWithCrosscheck(_) => true,
        #[cfg(feature = "paper-exact")]
        FoldingMode::PaperExact => false,
    }
}

pub(crate) fn fold_shard_prove_impl<L, MR, MB>(
    collect_val_lane_wits: bool,
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepWitnessBundle<Cmt, F, K>],
    step_idx_offset: usize,
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
    ob: Option<(&crate::output_binding::OutputBindingConfig, &[F])>,
    prover_ctx: Option<&ShardProverContext>,
    mut step_prove_ms_out: Option<&mut Vec<f64>>,
) -> Result<(ShardProof, Vec<Mat<F>>, Vec<Mat<F>>), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    for (step_idx, step) in steps.iter().enumerate() {
        if step.lut_instances.is_empty() && step.mem_instances.is_empty() {
            continue;
        }
        let is_shared_step = step
            .lut_instances
            .iter()
            .all(|(inst, wit)| inst.comms.is_empty() && wit.mats.is_empty())
            && step
                .mem_instances
                .iter()
                .all(|(inst, wit)| inst.comms.is_empty() && wit.mats.is_empty());
        if !is_shared_step {
            return Err(PiCcsError::InvalidInput(format!(
                "legacy no-shared CPU bus mode was removed; step_idx={step_idx} must use shared-bus witness format"
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
    if mode_uses_sparse_cache(&mode) && ccs_sparse_cache.is_none() {
        return Err(PiCcsError::ProtocolError(
            "missing SparseCache for optimized mode".into(),
        ));
    }
    let k_dec = params.k_rho as usize;
    let ring = ccs::RotRing::goldilocks();

    if acc_init.len() != acc_wit_init.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "acc_init.len()={} != acc_wit_init.len()={}",
            acc_init.len(),
            acc_wit_init.len()
        )));
    }

    // Initialize accumulator
    let mut accumulator = acc_init.to_vec();
    let mut accumulator_wit = acc_wit_init.to_vec();

    let mut step_proofs = Vec::with_capacity(steps.len());
    let mut val_lane_wits: Vec<Mat<F>> = Vec::new();
    let mut prev_twist_decoded: Option<Vec<crate::memory_sidecar::memory::TwistDecodedColsSparse>> = None;
    let mut output_proof: Option<neo_memory::output_check::OutputBindingProof> = None;

    if ob.is_some() && steps.is_empty() {
        return Err(PiCcsError::InvalidInput("output binding requires >= 1 step".into()));
    }

    for (idx, step) in steps.iter().enumerate() {
        let step_idx = step_idx_offset
            .checked_add(idx)
            .ok_or_else(|| PiCcsError::InvalidInput("step index overflow".into()))?;
        let step_start = time_now();
        crate::memory_sidecar::memory::absorb_step_memory_witness(tr, step);

        let include_ob = ob.is_some() && (idx + 1 == steps.len());
        let mut wb_time_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut wp_time_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut decode_decode_fields_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut decode_decode_immediates_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> =
            None;
        let mut width_bitness_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut width_quiescence_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut width_load_semantics_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut width_store_semantics_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut control_next_pc_linear_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut control_next_pc_control_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> =
            None;
        let mut control_branch_semantics_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> =
            None;
        let mut control_control_writeback_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> =
            None;
        let mut ob_time_claim: Option<crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim> = None;
        let mut ob_r_prime: Option<Vec<K>> = None;

        // Output binding is injected only on the final step, and must run before sampling Route-A `r_time`.
        if include_ob {
            let (cfg, final_memory_state) =
                ob.ok_or_else(|| PiCcsError::InvalidInput("output binding enabled but config missing".into()))?;

            if output_proof.is_some() {
                return Err(PiCcsError::ProtocolError(
                    "output binding already attached (internal error)".into(),
                ));
            }

            if cfg.mem_idx >= step.mem_instances.len() {
                return Err(PiCcsError::InvalidInput("output binding mem_idx out of range".into()));
            }
            let expected_k = 1usize
                .checked_shl(cfg.num_bits as u32)
                .ok_or_else(|| PiCcsError::InvalidInput("output binding: 2^num_bits overflow".into()))?;
            if final_memory_state.len() != expected_k {
                return Err(PiCcsError::InvalidInput(format!(
                    "output binding: final_memory_state.len()={} != 2^num_bits={}",
                    final_memory_state.len(),
                    expected_k
                )));
            }
            let mem_inst = &step.mem_instances[cfg.mem_idx].0;
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

            let (output_sc, r_prime) = neo_memory::output_check::generate_output_sumcheck_proof_and_challenges(
                tr,
                cfg.num_bits,
                cfg.program_io.clone(),
                final_memory_state,
            )
            .map_err(|e| PiCcsError::ProtocolError(format!("output sumcheck failed: {e:?}")))?;

            output_proof = Some(neo_memory::output_check::OutputBindingProof { output_sc });
            ob_r_prime = Some(r_prime);
        }

        let (mcs_inst, mcs_wit) = &step.mcs;

        // k = accumulator.len() + 1
        let k = accumulator.len() + 1;

        // --------------------------------------------------------------------
        // Route A: Shared-challenge batched sum-check for time/row rounds.
        // --------------------------------------------------------------------
        //
        // 1) Bind CCS header + ME inputs
        // 2) Sample CCS challenges (α, β, γ) and bind initial sum
        // 3) Build CCS oracle + lazy Twist/Shout oracles
        // 4) Run ONE batched sum-check for the first ell_n rounds (row/time)
        // 5) Finish CCS alone for remaining ell_d Ajtai rounds
        // 6) Emit CCS + memory ME claims at the shared r_time and fold via RLC/DEC

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
        ch.beta_m = utils::sample_beta_m(tr, ell_m)?;
        let ccs_initial_sum = claimed_initial_sum_from_inputs(&s, &ch, &accumulator);
        tr.append_fields(b"sumcheck/initial_sum", &ccs_initial_sum.as_coeffs());

        // Route A memory checks use a separate transcript-derived cycle point `r_cycle`
        // to form χ_{r_cycle}(t) weights inside their sum-check polynomials.
        let r_cycle: Vec<K> =
            ts::sample_ext_point(tr, b"route_a/r_cycle", b"route_a/cycle/0", b"route_a/cycle/1", ell_n);

        // CCS oracle (engine-selected).
        //
        // Keep the optimized oracle concrete so we can build outputs from its Ajtai precompute.
        let mut ccs_oracle: CcsOracleDispatch<'_> = match mode.clone() {
            FoldingMode::Optimized => {
                let sparse = ccs_sparse_cache
                    .as_ref()
                    .ok_or_else(|| PiCcsError::ProtocolError("missing SparseCache for optimized mode".into()))?;
                CcsOracleDispatch::Optimized(
                    neo_reductions::engines::optimized_engine::oracle::OptimizedOracle::new_with_sparse(
                        &s,
                        params,
                        core::slice::from_ref(mcs_wit),
                        &accumulator_wit,
                        ch.clone(),
                        ell_d,
                        ell_n,
                        d_sc,
                        accumulator.first().map(|mi| mi.r.as_slice()),
                        sparse.clone(),
                    ),
                )
            }
            #[cfg(feature = "paper-exact")]
            FoldingMode::PaperExact => CcsOracleDispatch::PaperExact(
                neo_reductions::engines::paper_exact_engine::oracle::PaperExactOracle::new(
                    &s,
                    params,
                    core::slice::from_ref(mcs_wit),
                    &accumulator_wit,
                    ch.clone(),
                    ell_d,
                    ell_n,
                    d_sc,
                    accumulator.first().map(|mi| mi.r.as_slice()),
                ),
            ),
            #[cfg(feature = "paper-exact")]
            FoldingMode::OptimizedWithCrosscheck(_) => {
                let sparse = ccs_sparse_cache
                    .as_ref()
                    .ok_or_else(|| PiCcsError::ProtocolError("missing SparseCache for optimized mode".into()))?;
                CcsOracleDispatch::Optimized(
                    neo_reductions::engines::optimized_engine::oracle::OptimizedOracle::new_with_sparse(
                        &s,
                        params,
                        core::slice::from_ref(mcs_wit),
                        &accumulator_wit,
                        ch.clone(),
                        ell_d,
                        ell_n,
                        d_sc,
                        accumulator.first().map(|mi| mi.r.as_slice()),
                        sparse.clone(),
                    ),
                )
            }
        };

        let shout_pre = crate::memory_sidecar::memory::prove_shout_addr_pre_time(
            tr, params, step, &cpu_bus, ell_n, &r_cycle, step_idx,
        )?;

        let twist_pre =
            crate::memory_sidecar::memory::prove_twist_addr_pre_time(tr, params, step, &cpu_bus, ell_n, &r_cycle)
                .map_err(|e| PiCcsError::ProtocolError(format!("twist addr-pre failed at step_idx={step_idx}: {e}")))?;
        let twist_read_claims: Vec<K> = twist_pre.iter().map(|p| p.read_check_claim_sum).collect();
        let twist_write_claims: Vec<K> = twist_pre.iter().map(|p| p.write_check_claim_sum).collect();
        let mut mem_oracles = crate::memory_sidecar::memory::build_route_a_memory_oracles(
            params, step, ell_n, &r_cycle, &shout_pre, &twist_pre,
        )?;

        let (wb_time_claim_built, wp_time_claim_built) =
            crate::memory_sidecar::memory::build_route_a_wb_wp_time_claims(params, step, &r_cycle)?;
        let wb_wp_required = crate::memory_sidecar::memory::wb_wp_required_for_step_witness(step);
        if wb_wp_required && (wb_time_claim_built.is_none() || wp_time_claim_built.is_none()) {
            return Err(PiCcsError::ProtocolError(
                "WB/WP claims are required in RV32 trace mode but were not built".into(),
            ));
        }
        if let Some((oracle, _claimed_sum)) = wb_time_claim_built {
            wb_time_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"wb/booleanity",
            });
        }
        if let Some((oracle, _claimed_sum)) = wp_time_claim_built {
            wp_time_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"wp/quiescence",
            });
        }
        let (decode_decode_fields_built, decode_decode_immediates_built) =
            crate::memory_sidecar::memory::build_route_a_decode_time_claims(params, step, &r_cycle)?;
        let decode_required = crate::memory_sidecar::memory::decode_stage_required_for_step_witness(step);
        if decode_required && (decode_decode_fields_built.is_none() || decode_decode_immediates_built.is_none()) {
            return Err(PiCcsError::ProtocolError(
                "decode stage claims are required in RV32 trace mode but were not built".into(),
            ));
        }
        if let Some((oracle, _claimed_sum)) = decode_decode_fields_built {
            decode_decode_fields_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"decode/fields",
            });
        }
        if let Some((oracle, _claimed_sum)) = decode_decode_immediates_built {
            decode_decode_immediates_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"decode/immediates",
            });
        }
        let (
            width_bitness_built,
            width_quiescence_built,
            _width_selector_linkage_built,
            width_load_semantics_built,
            width_store_semantics_built,
        ) = crate::memory_sidecar::memory::build_route_a_width_time_claims(params, step, &r_cycle)?;
        let width_required = crate::memory_sidecar::memory::width_stage_required_for_step_witness(step);
        if width_required
            && (width_bitness_built.is_none()
                || width_quiescence_built.is_none()
                || width_load_semantics_built.is_none()
                || width_store_semantics_built.is_none())
        {
            return Err(PiCcsError::ProtocolError(
                "width stage claims are required in RV32 trace mode but were not built".into(),
            ));
        }
        if let Some((oracle, _claimed_sum)) = width_bitness_built {
            width_bitness_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"width/bitness",
            });
        }
        if let Some((oracle, _claimed_sum)) = width_quiescence_built {
            width_quiescence_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"width/quiescence",
            });
        }
        if let Some((oracle, _claimed_sum)) = width_load_semantics_built {
            width_load_semantics_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"width/load_semantics",
            });
        }
        if let Some((oracle, _claimed_sum)) = width_store_semantics_built {
            width_store_semantics_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"width/store_semantics",
            });
        }
        let (
            control_next_pc_linear_built,
            control_next_pc_control_built,
            control_branch_semantics_built,
            control_control_writeback_built,
        ) = crate::memory_sidecar::memory::build_route_a_control_time_claims(params, step, &r_cycle)?;
        let control_required = crate::memory_sidecar::memory::control_stage_required_for_step_witness(step);
        if control_required
            && (control_next_pc_linear_built.is_none()
                || control_next_pc_control_built.is_none()
                || control_branch_semantics_built.is_none()
                || control_control_writeback_built.is_none())
        {
            return Err(PiCcsError::ProtocolError(
                "control stage claims are required in RV32 trace mode but were not built".into(),
            ));
        }
        if let Some((oracle, _claimed_sum)) = control_next_pc_linear_built {
            control_next_pc_linear_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"control/next_pc_linear",
            });
        }
        if let Some((oracle, _claimed_sum)) = control_next_pc_control_built {
            control_next_pc_control_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"control/next_pc_control",
            });
        }
        if let Some((oracle, _claimed_sum)) = control_branch_semantics_built {
            control_branch_semantics_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"control/branch_semantics",
            });
        }
        if let Some((oracle, _claimed_sum)) = control_control_writeback_built {
            control_control_writeback_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle,
                claimed_sum: K::ZERO,
                label: b"control/writeback",
            });
        }

        if include_ob {
            let (cfg, _final_memory_state) =
                ob.ok_or_else(|| PiCcsError::InvalidInput("output binding enabled but config missing".into()))?;
            let r_prime = ob_r_prime
                .as_ref()
                .ok_or_else(|| PiCcsError::ProtocolError("output binding r_prime missing".into()))?;
            let pre = twist_pre
                .get(cfg.mem_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("output binding mem_idx out of range for twist_pre".into()))?;

            if pre.decoded.lanes.is_empty() {
                return Err(PiCcsError::ProtocolError(
                    "output binding: Twist decoded lanes empty".into(),
                ));
            }

            let mut oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(pre.decoded.lanes.len());
            let mut claimed_sum = K::ZERO;
            for lane in pre.decoded.lanes.iter() {
                let (oracle, claim) = neo_memory::twist_oracle::TwistTotalIncOracleSparseTime::new(
                    lane.wa_bits.clone(),
                    lane.has_write.clone(),
                    lane.inc_at_write_addr.clone(),
                    r_prime,
                );
                oracles.push(Box::new(oracle));
                claimed_sum += claim;
            }
            let oracle = crate::memory_sidecar::memory::SumRoundOracle::new(oracles);

            ob_time_claim = Some(crate::memory_sidecar::route_a_time::ExtraBatchedTimeClaim {
                oracle: Box::new(oracle),
                claimed_sum,
                label: crate::output_binding::OB_INC_TOTAL_LABEL,
            });
        }

        let crate::memory_sidecar::route_a_time::RouteABatchedTimeProverOutput {
            r_time,
            per_claim_results,
            proof: batched_time,
        } = crate::memory_sidecar::route_a_time::prove_route_a_batched_time(
            tr,
            step_idx,
            ell_n,
            d_sc,
            ccs_initial_sum,
            &mut ccs_oracle,
            &mut mem_oracles,
            step,
            twist_read_claims,
            twist_write_claims,
            wb_time_claim,
            wp_time_claim,
            decode_decode_fields_claim,
            decode_decode_immediates_claim,
            width_bitness_claim,
            width_quiescence_claim,
            None,
            width_load_semantics_claim,
            width_store_semantics_claim,
            control_next_pc_linear_claim,
            control_next_pc_control_claim,
            control_branch_semantics_claim,
            control_control_writeback_claim,
            ob_time_claim,
        )?;

        // Finish CCS Ajtai rounds alone, continuing from the CCS oracle state after ell_n folds.
        let ccs_time_rounds = per_claim_results
            .first()
            .map(|r| r.round_polys.clone())
            .unwrap_or_default();
        let mut sumcheck_rounds = ccs_time_rounds;
        let mut sumcheck_chals = r_time.clone();
        let ajtai_initial_sum = per_claim_results
            .first()
            .map(|r| r.final_value)
            .unwrap_or(ccs_initial_sum);

        let mut ccs_ajtai = RoundOraclePrefix::new(&mut ccs_oracle, ell_d);
        let (ajtai_rounds, ajtai_chals) =
            run_sumcheck_prover_ds(tr, b"ccs/ajtai", step_idx, &mut ccs_ajtai, ajtai_initial_sum)?;
        let mut running_sum = ajtai_initial_sum;
        for (round_poly, &r_i) in ajtai_rounds.iter().zip(ajtai_chals.iter()) {
            running_sum = poly_eval_k(round_poly, r_i);
        }
        sumcheck_rounds.extend_from_slice(&ajtai_rounds);
        sumcheck_chals.extend_from_slice(&ajtai_chals);

        // --------------------------------------------------------------------
        // NC-only sumcheck (digit-range / norm-check) over {0,1}^{ell_m + ell_d}.
        // --------------------------------------------------------------------
        let mut ccs_nc_oracle = neo_reductions::engines::optimized_engine::oracle::NcOracle::new(
            &s,
            params,
            core::slice::from_ref(mcs_wit),
            &accumulator_wit,
            ch.clone(),
            ell_d,
            ell_m,
            d_sc,
        );
        let (sumcheck_rounds_nc, sumcheck_chals_nc) =
            run_sumcheck_prover_ds(tr, b"ccs/nc", step_idx, &mut ccs_nc_oracle, K::ZERO)?;
        let mut running_sum_nc = K::ZERO;
        for (round_poly, &r_i) in sumcheck_rounds_nc.iter().zip(sumcheck_chals_nc.iter()) {
            running_sum_nc = poly_eval_k(round_poly, r_i);
        }
        let (s_col, _alpha_prime_nc) = sumcheck_chals_nc.split_at(ell_m);

        // Build CCS ME outputs at r_time.
        let fold_digest = tr.digest32();
        let mut ccs_out = match &mut ccs_oracle {
            CcsOracleDispatch::Optimized(oracle) => oracle.build_me_outputs_from_ajtai_precomp(
                core::slice::from_ref(mcs_inst),
                &accumulator,
                s_col,
                fold_digest,
                l,
            ),
            #[cfg(feature = "paper-exact")]
            CcsOracleDispatch::PaperExact(_) => build_me_outputs_paper_exact(
                &s,
                params,
                core::slice::from_ref(mcs_inst),
                core::slice::from_ref(mcs_wit),
                &accumulator,
                &accumulator_wit,
                &r_time,
                s_col,
                ell_d,
                fold_digest,
                l,
            ),
        };

        // CCS oracle borrows accumulator_wit; drop before updating accumulator_wit at the end.
        drop(ccs_oracle);

        let mut trace_linkage_t_len: Option<usize> = None;

        // Shared CPU bus: append "implicit openings" for all bus columns without materializing
        // bus copyout matrices into the CCS.
        if cpu_bus.bus_cols > 0 {
            let core_t = s.t();
            if ccs_out.len() != 1 + accumulator_wit.len() {
                return Err(PiCcsError::ProtocolError(format!(
                    "CCS output count mismatch for bus openings (ccs_out.len()={}, expected {})",
                    ccs_out.len(),
                    1 + accumulator_wit.len()
                )));
            }

            crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
                params,
                &cpu_bus,
                core_t,
                &mcs_wit.Z,
                &mut ccs_out[0],
            )?;
            for (out, Z) in ccs_out.iter_mut().skip(1).zip(accumulator_wit.iter()) {
                crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(params, &cpu_bus, core_t, Z, out)?;
            }
        }

        // For RV32 trace wiring CCS, append time-combined openings for trace columns needed to
        // link Twist/Shout sidecars at r_time. In shared-bus mode this is appended after bus openings.
        if (!step.mem_instances.is_empty() || !step.lut_instances.is_empty()) && mcs_inst.m_in == 5 {
            // Infer that the CPU witness is the RV32 trace column-major layout:
            // z = [x (m_in) | trace_cols * t_len]
            let m_in = mcs_inst.m_in;
            let t_len = step
                .mem_instances
                .first()
                .map(|(inst, _wit)| inst.steps)
                .or_else(|| {
                    // Shout event-table instances may have `steps != t_len`; prefer a non-event-table
                    // instance if present, otherwise fall back to inferring from the trace layout.
                    step.lut_instances
                        .iter()
                        .find(|(inst, _wit)| {
                            !matches!(inst.table_spec, Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. }))
                        })
                        .map(|(inst, _wit)| inst.steps)
                })
                .or_else(|| {
                    // Trace CCS layout inference: z = [x (m_in) | trace_cols * t_len]
                    let trace = Rv32TraceLayout::new();
                    let w = s.m.checked_sub(m_in)?;
                    if trace.cols == 0 || w % trace.cols != 0 {
                        return None;
                    }
                    Some(w / trace.cols)
                })
                .ok_or_else(|| PiCcsError::InvalidInput("missing mem/lut instances".into()))?;
            if t_len == 0 {
                return Err(PiCcsError::InvalidInput("trace linkage requires steps>=1".into()));
            }
            for (i, (inst, _wit)) in step.mem_instances.iter().enumerate() {
                if inst.steps != t_len {
                    return Err(PiCcsError::InvalidInput(format!(
                        "trace linkage requires stable steps across mem instances (mem_idx={i} has steps={}, expected {t_len})",
                        inst.steps
                    )));
                }
            }

            let trace = Rv32TraceLayout::new();
            let trace_len = trace
                .cols
                .checked_mul(t_len)
                .ok_or_else(|| PiCcsError::InvalidInput("trace cols * t_len overflow".into()))?;
            let expected_m = m_in
                .checked_add(trace_len)
                .ok_or_else(|| PiCcsError::InvalidInput("m_in + trace_len overflow".into()))?;
            if s.m < expected_m {
                return Err(PiCcsError::InvalidInput(format!(
                    "trace linkage expects m >= m_in + trace.cols*t_len (m={}; min_m={expected_m} for t_len={t_len}, trace_cols={})",
                    s.m, trace.cols
                )));
            }

            let trace_cols_to_open_dense: Vec<usize> = vec![
                trace.active,
                trace.cycle,
                trace.pc_before,
                trace.instr_word,
                trace.rs1_addr,
                trace.rs1_val,
                trace.rs2_addr,
                trace.rs2_val,
                trace.rd_addr,
                trace.rd_val,
                trace.ram_addr,
                trace.ram_rv,
                trace.ram_wv,
            ];
            let trace_cols_to_open_shout: Vec<usize> = vec![
                trace.shout_has_lookup,
                trace.shout_val,
                trace.shout_lhs,
                trace.shout_rhs,
            ];
            let trace_cols_to_open_all: Vec<usize> = trace_cols_to_open_dense
                .iter()
                .chain(trace_cols_to_open_shout.iter())
                .copied()
                .collect();
            let core_t = s.t();
            let trace_open_base = core_t + cpu_bus.bus_cols;
            let col_base = m_in; // trace_base in the RV32 trace layout

            // Event-table style micro-optimization: Shout trace columns are constrained to be 0
            // whenever `shout_has_lookup == 0`, so we can compute their openings by summing only
            // over the active lookup rows.
            let active_shout_js: Vec<usize> = {
                let d = neo_math::D;
                let mut out: Vec<usize> = Vec::new();
                let col_offset = trace
                    .shout_has_lookup
                    .checked_mul(t_len)
                    .ok_or_else(|| PiCcsError::InvalidInput("trace col_id * t_len overflow".into()))?;
                for j in 0..t_len {
                    let z_idx = col_base
                        .checked_add(col_offset)
                        .and_then(|x| x.checked_add(j))
                        .ok_or_else(|| PiCcsError::InvalidInput("trace z index overflow".into()))?;
                    if z_idx >= mcs_wit.Z.cols() {
                        return Err(PiCcsError::InvalidInput(format!(
                            "trace openings: z_idx out of range (z_idx={z_idx}, m={})",
                            mcs_wit.Z.cols()
                        )));
                    }

                    let mut any = false;
                    for rho in 0..d {
                        if mcs_wit.Z[(rho, z_idx)] != F::ZERO {
                            any = true;
                            break;
                        }
                    }
                    if any {
                        out.push(j);
                    }
                }
                out
            };

            crate::memory_sidecar::cpu_bus::append_col_major_time_openings_to_me_instance(
                params,
                m_in,
                t_len,
                col_base,
                &trace_cols_to_open_dense,
                trace_open_base,
                &mcs_wit.Z,
                &mut ccs_out[0],
            )?;
            crate::memory_sidecar::cpu_bus::append_col_major_time_openings_to_me_instance_at_js(
                params,
                m_in,
                t_len,
                col_base,
                &trace_cols_to_open_shout,
                trace_open_base + trace_cols_to_open_dense.len(),
                &mcs_wit.Z,
                &mut ccs_out[0],
                &active_shout_js,
            )?;
            for (out, Z) in ccs_out.iter_mut().skip(1).zip(accumulator_wit.iter()) {
                crate::memory_sidecar::cpu_bus::append_col_major_time_openings_to_me_instance(
                    params,
                    m_in,
                    t_len,
                    col_base,
                    &trace_cols_to_open_all,
                    trace_open_base,
                    Z,
                    out,
                )?;
            }
            trace_linkage_t_len = Some(t_len);
        }

        if ccs_out.len() != k {
            return Err(PiCcsError::ProtocolError(format!(
                "Π_CCS returned {} outputs; expected k={k}",
                ccs_out.len()
            )));
        }

        let mut ccs_proof = crate::PiCcsProof::new(sumcheck_rounds, Some(ccs_initial_sum));
        ccs_proof.variant = crate::optimized_engine::PiCcsProofVariant::SplitNcV1;
        ccs_proof.sumcheck_challenges = sumcheck_chals;
        ccs_proof.sumcheck_rounds_nc = sumcheck_rounds_nc;
        ccs_proof.sc_initial_sum_nc = Some(K::ZERO);
        ccs_proof.sumcheck_challenges_nc = sumcheck_chals_nc;
        ccs_proof.challenges_public = ch;
        ccs_proof.sumcheck_final = running_sum;
        ccs_proof.sumcheck_final_nc = running_sum_nc;
        ccs_proof.header_digest = fold_digest.to_vec();

        #[cfg(feature = "paper-exact")]
        if let FoldingMode::OptimizedWithCrosscheck(cfg) = &mode {
            crosscheck_route_a_ccs_step(
                cfg,
                step_idx,
                params,
                &s,
                &cpu_bus,
                mcs_inst,
                mcs_wit,
                &accumulator,
                &accumulator_wit,
                &ccs_out,
                &ccs_proof,
                ell_d,
                ell_n,
                ell_m,
                d_sc,
                fold_digest,
                l,
            )?;
        }

        // Witnesses for CCS outputs: [Z_mcs, Z_seed...] (borrow; avoid multi-GB clones)
        let mut outs_Z: Vec<&Mat<F>> = Vec::with_capacity(k);
        outs_Z.push(&mcs_wit.Z);
        outs_Z.extend(accumulator_wit.iter());

        // Memory sidecar: emit ME claims at the shared r_time (no fixed-challenge sumcheck).
        let prev_step = (idx > 0).then(|| &steps[idx - 1]);
        let prev_twist_decoded_ref = prev_twist_decoded.as_deref();
        let mut mem_proof = crate::memory_sidecar::memory::finalize_route_a_memory_prover(
            tr,
            params,
            &cpu_bus,
            &s,
            step,
            prev_step,
            prev_twist_decoded_ref,
            &mut mem_oracles,
            &shout_pre.addr_pre,
            &twist_pre,
            &r_time,
            mcs_inst.m_in,
            step_idx,
        )?;
        prev_twist_decoded = Some(twist_pre.into_iter().map(|p| p.decoded).collect());

        // Normalize ME claim shapes for per-claim folding lanes.
        for me in mem_proof.val_me_claims.iter_mut() {
            let t = me.y.len();
            normalize_me_claims(core::slice::from_mut(me), ell_n, ell_d, t)?;
        }
        for me in mem_proof.wb_me_claims.iter_mut() {
            let t = me.y.len();
            normalize_me_claims(core::slice::from_mut(me), ell_n, ell_d, t)?;
        }
        for me in mem_proof.wp_me_claims.iter_mut() {
            let t = me.y.len();
            normalize_me_claims(core::slice::from_mut(me), ell_n, ell_d, t)?;
        }

        validate_me_batch_invariants(&ccs_out, "prove step ccs outputs")?;

        let want_main_wits = collect_val_lane_wits || idx + 1 < steps.len();
        let (main_fold, Z_split) = prove_rlc_dec_lane(
            &mode,
            RlcLane::Main,
            tr,
            params,
            &s,
            ccs_sparse_cache.as_deref(),
            Some(&cpu_bus),
            &ring,
            ell_d,
            k_dec,
            step_idx,
            trace_linkage_t_len,
            &ccs_out,
            &outs_Z,
            want_main_wits,
            l,
            mixers,
        )?;
        let RlcDecProof {
            rlc_rhos: rhos,
            rlc_parent: parent_pub,
            dec_children: children,
        } = main_fold;

        let has_prev = prev_step.is_some();

        // --------------------------------------------------------------------
        // Phase 2: Second folding lane for Twist val-eval ME claims at r_val.
        // --------------------------------------------------------------------
        let mut val_fold: Vec<RlcDecProof> = Vec::new();
        if !mem_proof.val_me_claims.is_empty() {
            tr.append_message(b"fold/val_lane_start", &(step_idx as u64).to_le_bytes());
            let expected = 1usize + usize::from(has_prev);
            if mem_proof.val_me_claims.len() != expected {
                return Err(PiCcsError::ProtocolError(format!(
                    "Twist(val) claim count mismatch (have {}, expected {})",
                    mem_proof.val_me_claims.len(),
                    expected
                )));
            }
            let can_reuse_main_lane_dec =
                ccs_out.len() == 1 && outs_Z.len() == 1 && !Z_split.is_empty() && children.len() == Z_split.len();
            let shared_val_lane_child_cs: Option<Vec<Cmt>> = if can_reuse_main_lane_dec {
                Some(children.iter().map(|child| child.c.clone()).collect())
            } else {
                None
            };

            for (claim_idx, me) in mem_proof.val_me_claims.iter().enumerate() {
                let (wit, ctx) = match claim_idx {
                    0 => (&mcs_wit.Z, "cpu"),
                    1 => {
                        let prev = prev_step
                            .ok_or_else(|| PiCcsError::ProtocolError("missing prev_step for r_val claim".into()))?;
                        (&prev.mcs.1.Z, "cpu_prev")
                    }
                    _ => {
                        return Err(PiCcsError::ProtocolError(
                            "unexpected extra r_val ME claim in shared-bus mode".into(),
                        ));
                    }
                };
                tr.append_message(b"fold/val_lane_claim_idx", &(claim_idx as u64).to_le_bytes());
                tr.append_message(b"fold/val_lane_claim_ctx", ctx.as_bytes());

                // Reuse main-lane split/commit artifacts for the current-step shared-bus
                // val lane so we don't pay an extra full split+commit.
                if claim_idx == 0 {
                    if let Some(child_cs) = shared_val_lane_child_cs.as_ref() {
                        bind_rlc_inputs(tr, RlcLane::Val, step_idx, core::slice::from_ref(me))?;
                        let rlc_rhos = ccs::sample_rot_rhos_n(tr, params, &ring, 1)?;
                        let mut rlc_parent = ccs::rlc_public(
                            &s,
                            params,
                            &rlc_rhos,
                            core::slice::from_ref(me),
                            mixers.mix_rhos_commits,
                            ell_d,
                        )?;
                        let (mut dec_children, ok_y, ok_x, ok_c) = ccs::dec_children_with_commit_cached(
                            mode.clone(),
                            &s,
                            params,
                            &rlc_parent,
                            &Z_split,
                            ell_d,
                            child_cs,
                            mixers.combine_b_pows,
                            ccs_sparse_cache.as_deref(),
                        );
                        if !(ok_y && ok_x && ok_c) {
                            return Err(PiCcsError::ProtocolError(format!(
                                "DEC(val) public check failed at step {} (y={}, X={}, c={})",
                                step_idx, ok_y, ok_x, ok_c
                            )));
                        }
                        if cpu_bus.bus_cols > 0 {
                            let core_t = s.t();
                            crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
                                params,
                                &cpu_bus,
                                core_t,
                                wit,
                                &mut rlc_parent,
                            )?;
                            for (child, zi) in dec_children.iter_mut().zip(Z_split.iter()) {
                                crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
                                    params, &cpu_bus, core_t, zi, child,
                                )?;
                            }
                        }
                        if collect_val_lane_wits {
                            val_lane_wits.extend(Z_split.iter().cloned());
                        }
                        val_fold.push(RlcDecProof {
                            rlc_rhos,
                            rlc_parent,
                            dec_children,
                        });
                        continue;
                    }
                }

                let (proof, mut Z_split_val) = prove_rlc_dec_lane(
                    &mode,
                    RlcLane::Val,
                    tr,
                    params,
                    &s,
                    ccs_sparse_cache.as_deref(),
                    Some(&cpu_bus),
                    &ring,
                    ell_d,
                    k_dec,
                    step_idx,
                    None,
                    core::slice::from_ref(me),
                    core::slice::from_ref(&wit),
                    collect_val_lane_wits,
                    l,
                    mixers,
                )?;
                if collect_val_lane_wits {
                    val_lane_wits.extend(Z_split_val.drain(..));
                }
                val_fold.push(proof);
            }
        }

        // Additional WB/WP folding lane(s): CPU ME openings used by wb/booleanity and
        // wp/quiescence stages. These lanes share the same witness matrix (`mcs_wit.Z`),
        // so precompute DEC digit witnesses + child commitments once per step.
        let mut wb_wp_dec_wits: Option<Vec<Mat<F>>> = None;
        let mut wb_wp_child_cs: Option<Vec<Cmt>> = None;
        if !mem_proof.wb_me_claims.is_empty() || !mem_proof.wp_me_claims.is_empty() {
            let (dec_wits, digit_nonzero) = ccs::split_b_matrix_k_with_nonzero_flags(&mcs_wit.Z, k_dec, params.b)?;
            let zero_c = Cmt::zeros(mcs_inst.c.d, mcs_inst.c.kappa);
            let child_cs: Vec<Cmt> = {
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                {
                    const PAR_CHILD_COMMIT_THRESHOLD: usize = 32;
                    let use_parallel = dec_wits.len() >= PAR_CHILD_COMMIT_THRESHOLD && rayon::current_num_threads() > 1;
                    if use_parallel {
                        dec_wits
                            .par_iter()
                            .enumerate()
                            .map(|(idx, Zi)| {
                                if digit_nonzero[idx] {
                                    l.commit(Zi)
                                } else {
                                    zero_c.clone()
                                }
                            })
                            .collect()
                    } else {
                        dec_wits
                            .iter()
                            .enumerate()
                            .map(|(idx, Zi)| {
                                if digit_nonzero[idx] {
                                    l.commit(Zi)
                                } else {
                                    zero_c.clone()
                                }
                            })
                            .collect()
                    }
                }
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                {
                    dec_wits
                        .iter()
                        .enumerate()
                        .map(|(idx, Zi)| {
                            if digit_nonzero[idx] {
                                l.commit(Zi)
                            } else {
                                zero_c.clone()
                            }
                        })
                        .collect()
                }
            };
            wb_wp_dec_wits = Some(dec_wits);
            wb_wp_child_cs = Some(child_cs);
        }

        // Additional WB folding lane(s): CPU ME openings used by wb/booleanity stage.
        let mut wb_fold: Vec<RlcDecProof> = Vec::new();
        if !mem_proof.wb_me_claims.is_empty() {
            let trace = Rv32TraceLayout::new();
            let t_len = crate::memory_sidecar::memory::infer_rv32_trace_t_len_for_wb_wp(step, &trace)?;
            let wb_cols = crate::memory_sidecar::memory::rv32_trace_wb_columns(&trace);
            let core_t = s.t();
            let m_in = mcs_inst.m_in;
            let dec_wits = wb_wp_dec_wits
                .as_ref()
                .ok_or_else(|| PiCcsError::ProtocolError("WB fold missing shared DEC witnesses".into()))?;
            let child_cs = wb_wp_child_cs
                .as_ref()
                .ok_or_else(|| PiCcsError::ProtocolError("WB fold missing shared DEC commitments".into()))?;
            tr.append_message(b"fold/wb_lane_start", &(step_idx as u64).to_le_bytes());
            for (claim_idx, me) in mem_proof.wb_me_claims.iter().enumerate() {
                tr.append_message(b"fold/wb_lane_claim_idx", &(claim_idx as u64).to_le_bytes());
                bind_rlc_inputs(tr, RlcLane::Val, step_idx, core::slice::from_ref(me))?;
                let rlc_rhos = ccs::sample_rot_rhos_n(tr, params, &ring, 1)?;
                let rlc_parent = ccs::rlc_public(
                    &s,
                    params,
                    &rlc_rhos,
                    core::slice::from_ref(me),
                    mixers.mix_rhos_commits,
                    ell_d,
                )?;
                let (mut dec_children, ok_y, ok_x, ok_c) = ccs::dec_children_with_commit_cached(
                    mode.clone(),
                    &s,
                    params,
                    &rlc_parent,
                    dec_wits,
                    ell_d,
                    child_cs,
                    mixers.combine_b_pows,
                    ccs_sparse_cache.as_deref(),
                );
                if !(ok_y && ok_x && ok_c) {
                    return Err(PiCcsError::ProtocolError(format!(
                        "DEC(val) public check failed at step {} (y={}, X={}, c={})",
                        step_idx, ok_y, ok_x, ok_c
                    )));
                }
                if dec_children.len() != dec_wits.len() {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: WB fold requires materialized DEC witnesses (children={}, wits={})",
                        step_idx,
                        dec_children.len(),
                        dec_wits.len()
                    )));
                }
                for (child, zi) in dec_children.iter_mut().zip(dec_wits.iter()) {
                    crate::memory_sidecar::cpu_bus::append_col_major_time_openings_to_me_instance(
                        params, m_in, t_len, m_in, &wb_cols, core_t, zi, child,
                    )?;
                }
                if collect_val_lane_wits {
                    val_lane_wits.extend(dec_wits.iter().cloned());
                }
                wb_fold.push(RlcDecProof {
                    rlc_rhos,
                    rlc_parent,
                    dec_children,
                });
            }
        }

        // Additional WP folding lane(s): CPU ME openings used by wp/quiescence stage.
        let mut wp_fold: Vec<RlcDecProof> = Vec::new();
        if !mem_proof.wp_me_claims.is_empty() {
            let trace = Rv32TraceLayout::new();
            let t_len = crate::memory_sidecar::memory::infer_rv32_trace_t_len_for_wb_wp(step, &trace)?;
            let mut wp_open_cols = crate::memory_sidecar::memory::rv32_trace_wp_opening_columns(&trace);
            if control_required {
                wp_open_cols.extend(crate::memory_sidecar::memory::rv32_trace_control_extra_opening_columns(
                    &trace,
                ));
            }
            if decode_required {
                let decode_layout = Rv32DecodeSidecarLayout::new();
                let (_decode_open_cols, decode_lut_indices) =
                    crate::memory_sidecar::memory::resolve_shared_decode_lookup_lut_indices(step, &decode_layout)?;
                let bus = crate::memory_sidecar::memory::build_bus_layout_for_step_witness(step, t_len)?;
                if bus.shout_cols.len() != step.lut_instances.len() {
                    return Err(PiCcsError::ProtocolError(
                        "W2(shared): bus layout shout lane count drift in WP fold".into(),
                    ));
                }
                let bus_base_delta = bus
                    .bus_base
                    .checked_sub(mcs_inst.m_in)
                    .ok_or_else(|| PiCcsError::ProtocolError("W2(shared): bus_base underflow in WP fold".into()))?;
                if bus_base_delta % t_len != 0 {
                    return Err(PiCcsError::ProtocolError(format!(
                        "W2(shared): bus_base alignment mismatch in WP fold (bus_base_delta={}, t_len={t_len})",
                        bus_base_delta
                    )));
                }
                let bus_col_offset = bus_base_delta / t_len;
                for &lut_idx in decode_lut_indices.iter() {
                    let inst_cols = bus.shout_cols.get(lut_idx).ok_or_else(|| {
                        PiCcsError::ProtocolError(
                            "W2(shared): missing shout cols for decode lookup table in WP fold".into(),
                        )
                    })?;
                    let lane0 = inst_cols.lanes.get(0).ok_or_else(|| {
                        PiCcsError::ProtocolError(
                            "W2(shared): expected one shout lane for decode lookup table in WP fold".into(),
                        )
                    })?;
                    wp_open_cols.push(bus_col_offset + lane0.primary_val());
                }
            }
            if width_required {
                wp_open_cols.extend(crate::memory_sidecar::memory::width_lookup_bus_val_cols_witness(
                    step, t_len,
                )?);
            }
            let core_t = s.t();
            let m_in = mcs_inst.m_in;
            let dec_wits = wb_wp_dec_wits
                .as_ref()
                .ok_or_else(|| PiCcsError::ProtocolError("WP fold missing shared DEC witnesses".into()))?;
            let child_cs = wb_wp_child_cs
                .as_ref()
                .ok_or_else(|| PiCcsError::ProtocolError("WP fold missing shared DEC commitments".into()))?;
            tr.append_message(b"fold/wp_lane_start", &(step_idx as u64).to_le_bytes());
            for (claim_idx, me) in mem_proof.wp_me_claims.iter().enumerate() {
                tr.append_message(b"fold/wp_lane_claim_idx", &(claim_idx as u64).to_le_bytes());
                bind_rlc_inputs(tr, RlcLane::Val, step_idx, core::slice::from_ref(me))?;
                let rlc_rhos = ccs::sample_rot_rhos_n(tr, params, &ring, 1)?;
                let rlc_parent = ccs::rlc_public(
                    &s,
                    params,
                    &rlc_rhos,
                    core::slice::from_ref(me),
                    mixers.mix_rhos_commits,
                    ell_d,
                )?;
                let (mut dec_children, ok_y, ok_x, ok_c) = ccs::dec_children_with_commit_cached(
                    mode.clone(),
                    &s,
                    params,
                    &rlc_parent,
                    dec_wits,
                    ell_d,
                    child_cs,
                    mixers.combine_b_pows,
                    ccs_sparse_cache.as_deref(),
                );
                if !(ok_y && ok_x && ok_c) {
                    return Err(PiCcsError::ProtocolError(format!(
                        "DEC(val) public check failed at step {} (y={}, X={}, c={})",
                        step_idx, ok_y, ok_x, ok_c
                    )));
                }
                if dec_children.len() != dec_wits.len() {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: WP fold requires materialized DEC witnesses (children={}, wits={})",
                        step_idx,
                        dec_children.len(),
                        dec_wits.len()
                    )));
                }
                for (child, zi) in dec_children.iter_mut().zip(dec_wits.iter()) {
                    crate::memory_sidecar::cpu_bus::append_col_major_time_openings_to_me_instance(
                        params,
                        m_in,
                        t_len,
                        m_in,
                        &wp_open_cols,
                        core_t,
                        zi,
                        child,
                    )?;
                }
                if collect_val_lane_wits {
                    val_lane_wits.extend(dec_wits.iter().cloned());
                }
                wp_fold.push(RlcDecProof {
                    rlc_rhos,
                    rlc_parent,
                    dec_children,
                });
            }
        }

        accumulator = children.clone();
        accumulator_wit = if want_main_wits { Z_split } else { Vec::new() };

        step_proofs.push(StepProof {
            fold: FoldStep {
                ccs_out,
                ccs_proof,
                rlc_rhos: rhos,
                rlc_parent: parent_pub,
                dec_children: children,
            },
            mem: mem_proof,
            batched_time,
            val_fold,
            wb_fold,
            wp_fold,
        });

        tr.append_message(b"fold/step_done", &(step_idx as u64).to_le_bytes());
        if let Some(out) = step_prove_ms_out.as_deref_mut() {
            out.push(elapsed_ms(step_start));
        }
    }

    Ok((
        ShardProof {
            steps: step_proofs,
            output_proof,
        },
        accumulator_wit,
        val_lane_wits,
    ))
}
