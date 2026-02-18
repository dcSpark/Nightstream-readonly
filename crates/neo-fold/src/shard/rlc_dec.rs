use super::*;

pub(crate) fn prove_rlc_dec_lane<L, MR, MB>(
    mode: &FoldingMode,
    lane: RlcLane,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    ccs_sparse_cache: Option<&SparseCache<F>>,
    cpu_bus: Option<&neo_memory::cpu::BusLayout>,
    ring: &ccs::RotRing,
    ell_d: usize,
    k_dec: usize,
    step_idx: usize,
    trace_linkage_t_len: Option<usize>,
    me_inputs: &[MeInstance<Cmt, F, K>],
    wit_inputs: &[&Mat<F>],
    want_witnesses: bool,
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<(RlcDecProof, Vec<Mat<F>>), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    if me_inputs.is_empty() {
        let prefix = match lane {
            RlcLane::Main => "",
            RlcLane::Val => "val-lane ",
        };
        return Err(PiCcsError::InvalidInput(format!(
            "step {}: {prefix}RLC input batch is empty",
            step_idx
        )));
    }
    if wit_inputs.len() != me_inputs.len() {
        let prefix = match lane {
            RlcLane::Main => "",
            RlcLane::Val => "val-lane ",
        };
        return Err(PiCcsError::InvalidInput(format!(
            "step {}: {prefix}RLC witness count mismatch (me_inputs.len()={}, wit_inputs.len()={})",
            step_idx,
            me_inputs.len(),
            wit_inputs.len()
        )));
    }

    bind_rlc_inputs(tr, lane, step_idx, me_inputs)?;
    let rlc_rhos = ccs::sample_rot_rhos_n(tr, params, ring, me_inputs.len())?;
    let (mut rlc_parent, Z_mix) = if me_inputs.len() == 1 {
        if rlc_rhos.len() != 1 {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: Π_RLC(k=1): |rhos| must equal |inputs|",
                step_idx
            )));
        }
        let inp = &me_inputs[0];

        // Match `neo_reductions::api::rlc_with_commit` semantics for k=1 without cloning Z.
        let inputs_c = vec![inp.c.clone()];
        let c = (mixers.mix_rhos_commits)(&rlc_rhos, &inputs_c);

        let t = inp.y.len();
        if t < s.t() {
            return Err(PiCcsError::InvalidInput(format!(
                "step {}: Π_RLC(k=1): ME y.len() must be >= s.t() (got {}, s.t()={})",
                step_idx,
                t,
                s.t()
            )));
        }
        for (j, row) in inp.y.iter().enumerate() {
            if row.len() < D {
                return Err(PiCcsError::InvalidInput(format!(
                    "step {}: Π_RLC(k=1): ME y[{}].len()={} must be >= D={}",
                    step_idx,
                    j,
                    row.len(),
                    D
                )));
            }
        }
        verify_me_y_scalars_canonical(inp, params.b, step_idx, "Π_RLC(k=1)")?;

        let out = MeInstance::<Cmt, F, K> {
            c_step_coords: vec![],
            u_offset: 0,
            u_len: 0,
            c,
            X: inp.X.clone(),
            r: inp.r.clone(),
            s_col: inp.s_col.clone(),
            y: inp.y.clone(),
            y_scalars: inp.y_scalars.clone(),
            y_zcol: inp.y_zcol.clone(),
            m_in: inp.m_in,
            fold_digest: inp.fold_digest,
        };

        (out, Cow::Borrowed(wit_inputs[0]))
    } else {
        let (out, Z_mix) = {
            #[cfg(feature = "paper-exact")]
            {
                if matches!(mode, FoldingMode::PaperExact) {
                    // Keep paper-exact dispatch through the public API.
                    let wit_owned: Vec<Mat<F>> = wit_inputs.iter().map(|m| (*m).clone()).collect();
                    ccs::rlc_with_commit(
                        mode.clone(),
                        s,
                        params,
                        &rlc_rhos,
                        me_inputs,
                        &wit_owned,
                        ell_d,
                        mixers.mix_rhos_commits,
                    )?
                } else {
                    neo_reductions::optimized_engine::rlc_reduction_optimized_with_commit_mix(
                        s,
                        params,
                        &rlc_rhos,
                        me_inputs,
                        wit_inputs,
                        ell_d,
                        mixers.mix_rhos_commits,
                    )
                }
            }
            #[cfg(not(feature = "paper-exact"))]
            {
                neo_reductions::optimized_engine::rlc_reduction_optimized_with_commit_mix(
                    s,
                    params,
                    &rlc_rhos,
                    me_inputs,
                    wit_inputs,
                    ell_d,
                    mixers.mix_rhos_commits,
                )
            }
        };
        (out, Cow::Owned(Z_mix))
    };

    let Z_mix = Z_mix.as_ref();

    let inputs_have_extra_y = me_inputs.iter().any(|me| me.y.len() > s.t());
    let can_stream_dec = !want_witnesses
        && has_global_pp_for_dims(D, s.m)
        && !cpu_bus.map(|b| b.bus_cols > 0).unwrap_or(false)
        && !inputs_have_extra_y;

    let materialize_dec = || -> Result<(Vec<MeInstance<Cmt, F, K>>, bool, bool, bool, Vec<Mat<F>>), PiCcsError> {
        // Standard DEC: materialize digit matrices (needed when carrying witnesses forward).
        let (Z_split, digit_nonzero) = ccs::split_b_matrix_k_with_nonzero_flags(Z_mix, k_dec, params.b)?;
        let zero_c = Cmt::zeros(rlc_parent.c.d, rlc_parent.c.kappa);
        let child_cs: Vec<Cmt> = {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                const PAR_CHILD_COMMIT_THRESHOLD: usize = 32;
                let use_parallel = Z_split.len() >= PAR_CHILD_COMMIT_THRESHOLD && rayon::current_num_threads() > 1;
                if use_parallel {
                    Z_split
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
                    Z_split
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
                Z_split
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
        let (dec_children, ok_y, ok_X, ok_c) = ccs::dec_children_with_commit_cached(
            mode.clone(),
            s,
            params,
            &rlc_parent,
            &Z_split,
            ell_d,
            &child_cs,
            mixers.combine_b_pows,
            ccs_sparse_cache,
        );
        Ok((dec_children, ok_y, ok_X, ok_c, Z_split))
    };

    let (mut dec_children, ok_y, ok_X, ok_c, maybe_wits) = if can_stream_dec {
        // Memory-optimized DEC: compute children + commitments without materializing Z_split.
        // If public consistency checks fail (e.g. global PP mismatch vs local committer),
        // fall back to the materialized path for correctness.
        let (children, _child_cs, ok_y, ok_X, ok_c) = dec_stream_no_witness(
            params,
            s,
            &rlc_parent,
            Z_mix,
            ell_d,
            k_dec,
            mixers.combine_b_pows,
            ccs_sparse_cache,
        )?;
        if ok_y && ok_X && ok_c {
            (children, ok_y, ok_X, ok_c, Vec::new())
        } else {
            materialize_dec()?
        }
    } else {
        materialize_dec()?
    };
    if !(ok_y && ok_X && ok_c) {
        let lane_label = match lane {
            RlcLane::Main => "DEC",
            RlcLane::Val => "DEC(val)",
        };
        return Err(PiCcsError::ProtocolError(format!(
            "{} public check failed at step {} (y={}, X={}, c={})",
            lane_label, step_idx, ok_y, ok_X, ok_c
        )));
    }

    // Shared CPU bus: carry the implicit bus openings through Π_RLC/Π_DEC so they remain
    // part of the folded instance (and are checked by public DEC verification).
    if let Some(bus) = cpu_bus {
        if bus.bus_cols > 0 {
            let core_t = s.t();
            crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
                params,
                bus,
                core_t,
                Z_mix,
                &mut rlc_parent,
            )?;
            for (child, Zi) in dec_children.iter_mut().zip(maybe_wits.iter()) {
                crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(params, bus, core_t, Zi, child)?;
            }
        }
    }

    // If the main lane carries RV32 trace linkage openings, propagate them through Π_DEC so child
    // instances keep the same extra y/y_scalars length (after optional shared-bus openings).
    if matches!(lane, RlcLane::Main) && trace_linkage_t_len.is_some() {
        let core_t = s.t();
        let trace_open_base = core_t + cpu_bus.map_or(0usize, |bus| bus.bus_cols);
        let trace = Rv32TraceLayout::new();
        let trace_cols_to_open: Vec<usize> = vec![
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
            trace.shout_has_lookup,
            trace.shout_val,
            trace.shout_lhs,
            trace.shout_rhs,
        ];

        let want_len = trace_open_base + trace_cols_to_open.len();
        let has_base_only = rlc_parent.y.len() == trace_open_base && rlc_parent.y_scalars.len() == trace_open_base;
        let has_trace_openings = rlc_parent.y.len() == want_len && rlc_parent.y_scalars.len() == want_len;
        if has_base_only || has_trace_openings {
            let m_in = rlc_parent.m_in;
            if m_in != 5 {
                return Err(PiCcsError::InvalidInput(format!(
                    "trace linkage openings expect m_in=5 (got {m_in})"
                )));
            }
            let t_len = trace_linkage_t_len
                .ok_or_else(|| PiCcsError::ProtocolError("trace linkage openings require explicit t_len".into()))?;
            if t_len == 0 {
                return Err(PiCcsError::InvalidInput("trace linkage expects t_len >= 1".into()));
            }
            let trace_len = trace
                .cols
                .checked_mul(t_len)
                .ok_or_else(|| PiCcsError::InvalidInput("trace cols * t_len overflow".into()))?;
            let min_m = m_in
                .checked_add(trace_len)
                .ok_or_else(|| PiCcsError::InvalidInput("m_in + trace_len overflow".into()))?;
            if s.m < min_m {
                return Err(PiCcsError::InvalidInput(format!(
                    "trace linkage openings require m >= m_in + trace.cols*t_len (m={}, min_m={} for t_len={}, trace_cols={})",
                    s.m, min_m, t_len, trace.cols
                )));
            }

            crate::memory_sidecar::cpu_bus::append_col_major_time_openings_to_me_instance(
                params,
                m_in,
                t_len,
                /*col_base=*/ m_in,
                &trace_cols_to_open,
                trace_open_base,
                Z_mix,
                &mut rlc_parent,
            )?;
            if dec_children.len() != maybe_wits.len() {
                return Err(PiCcsError::ProtocolError(
                    "trace linkage requires materialized DEC witnesses".into(),
                ));
            }
            for (child, Zi) in dec_children.iter_mut().zip(maybe_wits.iter()) {
                crate::memory_sidecar::cpu_bus::append_col_major_time_openings_to_me_instance(
                    params,
                    m_in,
                    t_len,
                    /*col_base=*/ m_in,
                    &trace_cols_to_open,
                    trace_open_base,
                    Zi,
                    child,
                )?;
            }
        } else {
            return Err(PiCcsError::InvalidInput(format!(
                "trace linkage openings expect parent y/y_scalars len to be base={} or base+trace_openings={} (got y.len()={}, y_scalars.len()={})",
                trace_open_base,
                want_len,
                rlc_parent.y.len(),
                rlc_parent.y_scalars.len(),
            )));
        }
    }

    Ok((
        RlcDecProof {
            rlc_rhos,
            rlc_parent,
            dec_children,
        },
        maybe_wits,
    ))
}

pub(crate) fn verify_rlc_dec_lane<MR, MB>(
    lane: RlcLane,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    ring: &ccs::RotRing,
    ell_d: usize,
    mixers: CommitMixers<MR, MB>,
    step_idx: usize,
    rlc_inputs: &[MeInstance<Cmt, F, K>],
    rlc_rhos: &[Mat<F>],
    rlc_parent: &MeInstance<Cmt, F, K>,
    dec_children: &[MeInstance<Cmt, F, K>],
) -> Result<(), PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    bind_rlc_inputs(tr, lane, step_idx, rlc_inputs)?;

    if rlc_rhos.len() != rlc_inputs.len() {
        let prefix = match lane {
            RlcLane::Main => "",
            RlcLane::Val => "val-lane ",
        };
        return Err(PiCcsError::InvalidInput(format!(
            "step {}: {}RLC ρ count mismatch (expected {}, got {})",
            step_idx,
            prefix,
            rlc_inputs.len(),
            rlc_rhos.len()
        )));
    }

    for (i, me) in rlc_inputs.iter().enumerate() {
        verify_me_y_scalars_canonical(
            me,
            params.b,
            step_idx,
            &format!(
                "{}RLC input[{i}]",
                match lane {
                    RlcLane::Main => "",
                    RlcLane::Val => "val-lane ",
                }
            ),
        )?;
    }

    let rhos_from_tr = ccs::sample_rot_rhos_n(tr, params, ring, rlc_inputs.len())?;
    for (j, (sampled, stored)) in rhos_from_tr.iter().zip(rlc_rhos.iter()).enumerate() {
        if sampled.as_slice() != stored.as_slice() {
            return Err(PiCcsError::ProtocolError(match lane {
                RlcLane::Main => format!("step {}: RLC ρ #{} mismatch: transcript vs proof", step_idx, j),
                RlcLane::Val => format!("step {}: val-lane RLC ρ #{} mismatch: transcript vs proof", step_idx, j),
            }));
        }
    }

    let parent_pub = ccs::rlc_public(s, params, rlc_rhos, rlc_inputs, mixers.mix_rhos_commits, ell_d)?;

    let prefix = match lane {
        RlcLane::Main => "",
        RlcLane::Val => "val-lane ",
    };
    if parent_pub.m_in != rlc_parent.m_in {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC m_in mismatch (public={}, proof={})",
            step_idx, parent_pub.m_in, rlc_parent.m_in
        )));
    }
    if parent_pub.fold_digest != rlc_parent.fold_digest {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC fold_digest mismatch",
            step_idx
        )));
    }
    if parent_pub.c_step_coords != rlc_parent.c_step_coords {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC c_step_coords mismatch",
            step_idx
        )));
    }
    if parent_pub.u_offset != rlc_parent.u_offset {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC u_offset mismatch",
            step_idx
        )));
    }
    if parent_pub.u_len != rlc_parent.u_len {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC u_len mismatch",
            step_idx
        )));
    }
    if parent_pub.X != rlc_parent.X {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC X mismatch",
            step_idx
        )));
    }
    if parent_pub.c != rlc_parent.c {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC commitment mismatch",
            step_idx
        )));
    }
    if parent_pub.r != rlc_parent.r {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC r mismatch",
            step_idx
        )));
    }
    if parent_pub.s_col != rlc_parent.s_col {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC s_col mismatch",
            step_idx
        )));
    }
    if parent_pub.y != rlc_parent.y {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC y mismatch",
            step_idx
        )));
    }
    if parent_pub.y_scalars != rlc_parent.y_scalars {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC y_scalars mismatch",
            step_idx
        )));
    }
    if parent_pub.y_zcol != rlc_parent.y_zcol {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC y_zcol mismatch",
            step_idx
        )));
    }

    if rlc_parent.X.rows() != D || rlc_parent.X.cols() != rlc_parent.m_in {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: {prefix}RLC parent X shape {}x{} does not match m_in={}",
            step_idx,
            rlc_parent.X.rows(),
            rlc_parent.X.cols(),
            rlc_parent.m_in
        )));
    }
    if !dec_children.is_empty() {
        validate_me_batch_invariants(dec_children, "verify step dec children")?;
        for (child_idx, child) in dec_children.iter().enumerate() {
            if child.m_in != rlc_parent.m_in {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: {prefix}DEC child[{child_idx}] has m_in={}, expected {}",
                    step_idx, child.m_in, rlc_parent.m_in
                )));
            }
            if child.fold_digest != rlc_parent.fold_digest {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: {prefix}DEC child[{child_idx}] fold_digest mismatch",
                    step_idx
                )));
            }
        }
    }

    if !ccs::verify_dec_public(s, params, rlc_parent, dec_children, mixers.combine_b_pows, ell_d) {
        return Err(PiCcsError::ProtocolError(match lane {
            RlcLane::Main => format!("step {}: DEC public check failed", step_idx),
            RlcLane::Val => format!("step {}: val-lane DEC public check failed", step_idx),
        }));
    }

    Ok(())
}

#[cfg(feature = "paper-exact")]
pub(crate) fn crosscheck_route_a_ccs_step<L>(
    cfg: &neo_reductions::engines::CrosscheckCfg,
    step_idx: usize,
    params: &NeoParams,
    s: &CcsStructure<F>,
    cpu_bus: &neo_memory::cpu::BusLayout,
    mcs_inst: &neo_ccs::McsInstance<Cmt, F>,
    mcs_wit: &neo_ccs::McsWitness<F>,
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    ccs_out: &[MeInstance<Cmt, F, K>],
    ccs_proof: &crate::PiCcsProof,
    ell_d: usize,
    ell_n: usize,
    ell_m: usize,
    d_sc: usize,
    fold_digest: [u8; 32],
    log: &L,
) -> Result<(), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt> + Sync,
{
    let want_rounds_total = ell_n
        .checked_add(ell_d)
        .ok_or_else(|| PiCcsError::ProtocolError("ell_n + ell_d overflow".into()))?;
    if ccs_proof.sumcheck_rounds.len() != want_rounds_total {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: crosscheck expects {} CCS sumcheck rounds, got {}",
            step_idx,
            want_rounds_total,
            ccs_proof.sumcheck_rounds.len(),
        )));
    }
    if ccs_proof.sumcheck_challenges.len() != want_rounds_total {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: crosscheck expects {} CCS sumcheck challenges, got {}",
            step_idx,
            want_rounds_total,
            ccs_proof.sumcheck_challenges.len(),
        )));
    }
    let (s_col_prime, alpha_prime_nc) = if ccs_proof.variant == crate::optimized_engine::PiCcsProofVariant::SplitNcV1 {
        let want_nc_rounds_total = ell_m
            .checked_add(ell_d)
            .ok_or_else(|| PiCcsError::ProtocolError("ell_m + ell_d overflow".into()))?;
        if ccs_proof.sumcheck_rounds_nc.len() != want_nc_rounds_total {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: crosscheck expects {} NC sumcheck rounds, got {}",
                step_idx,
                want_nc_rounds_total,
                ccs_proof.sumcheck_rounds_nc.len(),
            )));
        }
        if ccs_proof.sumcheck_challenges_nc.len() != want_nc_rounds_total {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: crosscheck expects {} NC sumcheck challenges, got {}",
                step_idx,
                want_nc_rounds_total,
                ccs_proof.sumcheck_challenges_nc.len(),
            )));
        }
        ccs_proof.sumcheck_challenges_nc.split_at(ell_m)
    } else {
        (&[][..], &[][..])
    };

    let (r_prime, alpha_prime) = ccs_proof.sumcheck_challenges.split_at(ell_n);
    let r_inputs = me_inputs.first().map(|mi| mi.r.as_slice());

    // Crosscheck initial-sum parity is most informative once there is at least one carried ME
    // input. For empty-accumulator starts, optimized and paper-exact route through different
    // constant-term paths and can diverge without indicating a soundness issue.
    if cfg.initial_sum && !me_inputs.is_empty() {
        let lhs_exact = crate::paper_exact_engine::sum_q_over_hypercube_paper_exact(
            s,
            params,
            core::slice::from_ref(mcs_wit),
            me_witnesses,
            &ccs_proof.challenges_public,
            ell_d,
            ell_n,
            r_inputs,
        );
        let initial_sum_prover = ccs_proof
            .sumcheck_rounds
            .first()
            .map(|p0| poly_eval_k(p0, K::ZERO) + poly_eval_k(p0, K::ONE))
            .ok_or_else(|| PiCcsError::ProtocolError("crosscheck: missing sumcheck round 0".into()))?;
        if lhs_exact != initial_sum_prover {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: crosscheck initial sum mismatch (optimized vs paper-exact)",
                step_idx
            )));
        }
    }

    if cfg.per_round {
        let mut paper_oracle = crate::paper_exact_engine::oracle::PaperExactOracle::new(
            s,
            params,
            core::slice::from_ref(mcs_wit),
            me_witnesses,
            ccs_proof.challenges_public.clone(),
            ell_d,
            ell_n,
            d_sc,
            r_inputs,
        );

        let mut any_mismatch = false;
        for (round_idx, (opt_coeffs, &challenge)) in ccs_proof
            .sumcheck_rounds
            .iter()
            .zip(ccs_proof.sumcheck_challenges.iter())
            .enumerate()
        {
            let deg = paper_oracle.degree_bound();
            let xs: Vec<K> = (0..=deg).map(|t| K::from(F::from_u64(t as u64))).collect();
            let paper_evals = paper_oracle.evals_at(&xs);

            for (&x, &expected) in xs.iter().zip(paper_evals.iter()) {
                let actual = poly_eval_k(opt_coeffs, x);
                if actual != expected {
                    any_mismatch = true;
                    if cfg.fail_fast {
                        return Err(PiCcsError::ProtocolError(format!(
                            "step {}: crosscheck round {} polynomial mismatch",
                            step_idx, round_idx
                        )));
                    }
                }
            }

            paper_oracle.fold(challenge);
        }
        if any_mismatch {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: crosscheck per-round polynomial mismatch",
                step_idx
            )));
        }
    }

    if cfg.terminal {
        let running_sum_prover = if let Some(initial) = ccs_proof.sc_initial_sum {
            let mut running = initial;
            for (coeffs, &ri) in ccs_proof
                .sumcheck_rounds
                .iter()
                .zip(ccs_proof.sumcheck_challenges.iter())
            {
                running = poly_eval_k(coeffs, ri);
            }
            running
        } else {
            ccs_proof
                .sumcheck_rounds
                .first()
                .map(|p0| poly_eval_k(p0, K::ZERO) + poly_eval_k(p0, K::ONE))
                .unwrap_or(K::ZERO)
        };

        let rhs_fe = crate::paper_exact_engine::rhs_terminal_identity_fe_paper_exact(
            s,
            params,
            &ccs_proof.challenges_public,
            r_prime,
            alpha_prime,
            ccs_out,
            r_inputs,
        );
        let (lhs_fe, _rhs_unused) = crate::paper_exact_engine::q_eval_at_ext_point_fe_paper_exact_with_inputs(
            s,
            params,
            core::slice::from_ref(mcs_wit),
            me_witnesses,
            alpha_prime,
            r_prime,
            &ccs_proof.challenges_public,
            r_inputs,
        );
        if rhs_fe != lhs_fe || rhs_fe != running_sum_prover {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: crosscheck FE terminal evaluation claim mismatch",
                step_idx
            )));
        }

        let rhs_nc = crate::paper_exact_engine::rhs_terminal_identity_nc_paper_exact(
            params,
            &ccs_proof.challenges_public,
            s_col_prime,
            alpha_prime_nc,
            ccs_out,
        );
        if rhs_nc != ccs_proof.sumcheck_final_nc {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: crosscheck NC terminal evaluation claim mismatch",
                step_idx
            )));
        }
    }

    if cfg.outputs {
        let mut out_me_ref = build_me_outputs_paper_exact(
            s,
            params,
            core::slice::from_ref(mcs_inst),
            core::slice::from_ref(mcs_wit),
            me_inputs,
            me_witnesses,
            r_prime,
            s_col_prime,
            ell_d,
            fold_digest,
            log,
        );

        if cpu_bus.bus_cols > 0 {
            let core_t = s.t();
            if out_me_ref.len() != 1 + me_witnesses.len() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck CCS output count mismatch for bus openings (out_me_ref.len()={}, expected {})",
                    step_idx,
                    out_me_ref.len(),
                    1 + me_witnesses.len()
                )));
            }

            crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
                params,
                cpu_bus,
                core_t,
                &mcs_wit.Z,
                &mut out_me_ref[0],
            )?;
            for (out, Z) in out_me_ref.iter_mut().skip(1).zip(me_witnesses.iter()) {
                crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(params, cpu_bus, core_t, Z, out)?;
            }

            let trace = Rv32TraceLayout::new();
            let trace_cols_to_open: Vec<usize> = vec![
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
                trace.shout_has_lookup,
                trace.shout_val,
                trace.shout_lhs,
                trace.shout_rhs,
            ];
            let want_with_trace = core_t + cpu_bus.bus_cols + trace_cols_to_open.len();
            if ccs_out
                .first()
                .map(|me| me.y_scalars.len() == want_with_trace)
                .unwrap_or(false)
            {
                let m_in = mcs_inst.m_in;
                let bus_region_len = cpu_bus
                    .bus_cols
                    .checked_mul(cpu_bus.chunk_size)
                    .ok_or_else(|| PiCcsError::ProtocolError("crosscheck bus region overflow".into()))?;
                let trace_region =
                    s.m.checked_sub(m_in)
                        .and_then(|v| v.checked_sub(bus_region_len))
                        .ok_or_else(|| PiCcsError::ProtocolError("crosscheck trace region underflow".into()))?;
                if trace.cols == 0 || trace_region % trace.cols != 0 {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: crosscheck cannot infer trace t_len (trace_region={}, trace_cols={})",
                        step_idx, trace_region, trace.cols
                    )));
                }
                let t_len = trace_region / trace.cols;
                let trace_open_base = core_t + cpu_bus.bus_cols;
                crate::memory_sidecar::cpu_bus::append_col_major_time_openings_to_me_instance(
                    params,
                    m_in,
                    t_len,
                    m_in,
                    &trace_cols_to_open,
                    trace_open_base,
                    &mcs_wit.Z,
                    &mut out_me_ref[0],
                )?;
                for (out, Z) in out_me_ref.iter_mut().skip(1).zip(me_witnesses.iter()) {
                    crate::memory_sidecar::cpu_bus::append_col_major_time_openings_to_me_instance(
                        params,
                        m_in,
                        t_len,
                        m_in,
                        &trace_cols_to_open,
                        trace_open_base,
                        Z,
                        out,
                    )?;
                }
            }
        }

        if out_me_ref.len() != ccs_out.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: crosscheck output length mismatch (paper={}, optimized={})",
                step_idx,
                out_me_ref.len(),
                ccs_out.len()
            )));
        }

        for (idx, (a, b)) in out_me_ref.iter().zip(ccs_out.iter()).enumerate() {
            if a.m_in != b.m_in {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck output[{idx}] m_in mismatch (paper={}, optimized={})",
                    step_idx, a.m_in, b.m_in
                )));
            }
            if a.r != b.r {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck output[{idx}] r mismatch",
                    step_idx
                )));
            }
            if a.s_col != b.s_col {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck output[{idx}] s_col mismatch",
                    step_idx
                )));
            }
            if a.c.data != b.c.data {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck output[{idx}] commitment mismatch",
                    step_idx
                )));
            }
            if a.y.len() != b.y.len() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck output[{idx}] y.len mismatch (paper={}, optimized={})",
                    step_idx,
                    a.y.len(),
                    b.y.len()
                )));
            }
            for (j, (ya, yb)) in a.y.iter().zip(b.y.iter()).enumerate() {
                if ya != yb {
                    return Err(PiCcsError::ProtocolError(format!(
                        "step {}: crosscheck output[{idx}] y row {j} mismatch",
                        step_idx
                    )));
                }
            }
            if a.y_scalars != b.y_scalars {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck output[{idx}] y_scalars mismatch",
                    step_idx
                )));
            }
            if a.y_zcol != b.y_zcol {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck output[{idx}] y_zcol mismatch",
                    step_idx
                )));
            }
            if a.X.rows() != b.X.rows() || a.X.cols() != b.X.cols() {
                return Err(PiCcsError::ProtocolError(format!(
                    "step {}: crosscheck output[{idx}] X dims mismatch (paper={}x{}, optimized={}x{})",
                    step_idx,
                    a.X.rows(),
                    a.X.cols(),
                    b.X.rows(),
                    b.X.cols()
                )));
            }
            for r in 0..a.X.rows() {
                for c in 0..a.X.cols() {
                    if a.X[(r, c)] != b.X[(r, c)] {
                        return Err(PiCcsError::ProtocolError(format!(
                            "step {}: crosscheck output[{idx}] X mismatch at ({},{})",
                            step_idx, r, c
                        )));
                    }
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// Shard Proving
// ============================================================================
