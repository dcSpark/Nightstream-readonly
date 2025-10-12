//! NIVC finalization pipeline - generate final SNARK proof

use crate::{F, NeoParams};
use super::super::api::{NivcProgram, NivcChainProof};
use neo_spartan_bridge::pi_ccs_embed as piccs;
use p3_field::{PrimeField64, PrimeCharacteristicRing};

mod helpers;
use helpers::*;

/// Options for NIVC final proof
pub struct NivcFinalizeOptions { 
    pub embed_ivc_ev: bool 
}

/// Generate a succinct final SNARK proof for the NIVC chain with options.
/// Returns: (lean proof, augmented CCS, final public input)
pub fn finalize_with_options(
    program: &NivcProgram,
    params: &NeoParams,
    chain: NivcChainProof,
    opts: NivcFinalizeOptions,
) -> anyhow::Result<Option<(crate::Proof, neo_ccs::CcsStructure<F>, Vec<F>)>> {
    if chain.steps.is_empty() { return Ok(None); }
    let last = chain.steps.last().unwrap();
    let j = last.which_type;
    anyhow::ensure!(j < program.len(), "invalid which_type in last step");
    let spec = &program.steps[j];

    // Gather data from last step proof
    let mut rho = if last.inner.step_rho != F::ZERO { 
        last.inner.step_rho 
    } else { 
        F::ONE 
    };
    let y_prev = last.inner.step_y_prev.clone();
    let y_next = last.inner.step_y_next.clone();
    let step_x = last.inner.step_public_input.clone();
    let y_len = y_prev.len();

    // Reconstruct augmented CCS
    let augmented_ccs = build_augmented_ccs(
        &spec.ccs,
        step_x.len(),
        &spec.binding.y_step_offsets,
        &spec.binding.y_prev_witness_indices,
        &spec.binding.step_program_input_witness_indices,
        y_len,
        spec.binding.const1_witness_index,
    )?;

    // Build final public input for the final SNARK
    let final_public_input = crate::ivc::build_final_snark_public_input(&step_x, rho, &y_prev, &y_next);

    // Extract ME and witness
    let (final_me, final_me_wit) = pick_me_and_witness(last, &chain, j)?;

    // Bridge adapter: modern → legacy
    let mut me_for_bridge = final_me.clone();
    me_for_bridge.c.data = last.inner.c_step_coords.clone();
    
    let (mut legacy_me, mut legacy_wit, _pp) = crate::adapt_from_modern(
        std::slice::from_ref(&me_for_bridge),
        std::slice::from_ref(final_me_wit),
        &augmented_ccs,
        params,
        &[],
        None,
    ).map_err(|e| anyhow::anyhow!("Bridge adapter failed: {}", e))?;

    // Align Ajtai binding target
    if !opts.embed_ivc_ev {
        #[allow(deprecated)]
        {
            let z_len_dm = final_me_wit.Z.rows() * final_me_wit.Z.cols();
            let rows = legacy_me.c_coords.len();
            let mut new_coords = Vec::with_capacity(rows);
            for i in 0..rows {
                match neo_ajtai::compute_single_ajtai_row(&_pp, i, z_len_dm, rows) {
                    Ok(row) => {
                        let mut acc_f = neo_math::F::ZERO;
                        for (j, &a) in row.iter().enumerate() {
                            if j >= legacy_wit.z_digits.len() { break; }
                            let zi = legacy_wit.z_digits[j];
                            let zf = if zi >= 0 { neo_math::F::from_u64(zi as u64) } else { -neo_math::F::from_u64((-zi) as u64) };
                            acc_f += a * zf;
                        }
                        new_coords.push(acc_f);
                    }
                    Err(e) => anyhow::bail!("Failed to compute Ajtai row {}: {}", i, e),
                }
            }
            legacy_me.c_coords = new_coords;
        }
    }

    // Bind proof to augmented CCS + public input
    let context_digest = crate::context_digest_v1(&augmented_ccs, &final_public_input);
    #[allow(deprecated)]
    { legacy_me.header_digest = context_digest; }

    // Power-of-two padding if embedding EV
    if opts.embed_ivc_ev {
        #[allow(deprecated)]
        let original_len = legacy_wit.z_digits.len();
        #[allow(deprecated)]
        let target_len = if original_len <= 1 { 1 } else { original_len.next_power_of_two() };
        if target_len > original_len {
            #[allow(deprecated)]
            legacy_wit.z_digits.resize(target_len, 0i64);
        }
    }

    // Compress to lean proof
    let ajtai_pp_arc = std::sync::Arc::new(_pp.clone());
    let lean = if opts.embed_ivc_ev {
        anyhow::ensure!(rho != F::ZERO, "ρ is zero; EV embedding not supported");

        // Optionally bind ρ to fold-chain transcript digest
        let fold_digest_opt = last.inner.folding_proof
            .as_ref()
            .map(|fp| neo_fold::folding_proof_digest(fp));
        if let Some(d) = &fold_digest_opt {
            let mut limbs: Vec<u64> = Vec::new();
            for chunk in b"neo/ev/rho_from_digest/v1".chunks(8) {
                let mut b = [0u8; 8];
                b[..chunk.len()].copy_from_slice(chunk);
                limbs.push(u64::from_le_bytes(b));
            }
            for chunk in d.chunks(8) {
                let mut b = [0u8; 8];
                b[..chunk.len()].copy_from_slice(chunk);
                limbs.push(u64::from_le_bytes(b));
            }
            let packed: Vec<u8> = limbs.iter().flat_map(|&x| x.to_le_bytes()).collect();
            let h = neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash_packed_bytes(&packed);
            let mut derived = F::from_u64(h[0].as_canonical_u64());
            if derived == F::ZERO { derived = F::ONE; }
            rho = derived;
        }

        // Build EV embed
        let y_step_public = {
            let rho_inv = F::ONE / rho;
            Some(y_next.iter().zip(y_prev.iter()).map(|(n,p)| (*n - *p) * rho_inv).collect::<Vec<_>>())
        };

        let c_next_vec = last.inner.next_accumulator.c_coords.clone();
        let c_step_from_z: Vec<F> = {
            let d_pp = final_me_wit.Z.rows();
            let m_pp = final_me_wit.Z.cols();
            let z_len_dm = d_pp * m_pp;
            let rows = last.inner.c_step_coords.len();
            let mut out = Vec::with_capacity(rows);
            for i in 0..rows {
                match neo_ajtai::compute_single_ajtai_row(&_pp, i, z_len_dm, rows) {
                    Ok(row) => {
                        let mut acc_f = F::ZERO;
                        #[allow(deprecated)]
                        for j in 0..core::cmp::min(z_len_dm, legacy_wit.z_digits.len()) {
                            let a = row[j];
                            let zi = legacy_wit.z_digits[j];
                            let zf = if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
                            acc_f += a * zf;
                        }
                        out.push(acc_f);
                    }
                    Err(e) => anyhow::bail!("Ajtai row {} compute failed: {}", i, e),
                }
            }
            out
        };
        
        let c_prev_from_z: Vec<F> = {
            let mut v = Vec::with_capacity(c_next_vec.len());
            for i in 0..c_next_vec.len() { 
                v.push(c_next_vec[i] - rho * c_step_from_z[i]); 
            }
            v
        };

        let ev_embed = neo_spartan_bridge::IvcEvEmbed {
            rho,
            y_prev: y_prev.clone(),
            y_next: y_next.clone(),
            y_step_public,
            fold_chain_digest: fold_digest_opt,
            acc_c_prev: Some(c_prev_from_z.clone()),
            acc_c_step: Some(c_step_from_z.clone()),
            acc_c_next: Some(c_next_vec.clone()),
            rho_eff: None,
        };

        // Linkage
        let linkage = Some(neo_spartan_bridge::IvcLinkageInputs {
            x_indices_abs: spec.binding.step_program_input_witness_indices.clone(),
            y_prev_indices_abs: spec.binding.y_prev_witness_indices.clone(),
            const1_index_abs: None,
            step_io: last.step_io.clone(),
        });

        #[allow(deprecated)]
        { legacy_me.header_digest = context_digest; }

        // Build Pi-CCS embed
        let pi_ccs_embed_opt = build_pi_ccs_embed(&augmented_ccs);

        // Canonicalize weight vectors
        {
            let mut mats = Vec::with_capacity(augmented_ccs.matrices.len());
            for mj in &augmented_ccs.matrices {
                let rows = mj.rows();
                let cols = mj.cols();
                let mut entries = Vec::new();
                for r in 0..rows { 
                    for c in 0..cols {
                        let a = mj[(r, c)];
                        if a != F::ZERO { 
                            entries.push((r as u32, c as u32, a)); 
                        }
                    }
                }
                mats.push(piccs::CcsCsr { rows, cols, entries });
            }
            
            // Compute canonicalized weight vectors inline
            let n_rows = mats.first().map(|m| m.rows).unwrap_or(0);
            let d = neo_math::ring::D;
            #[allow(deprecated)]
            let _pairs_avail = legacy_me.r_point.len() / 2;
            let mut ell_needed = 0usize; 
            while (1usize << ell_needed) < n_rows { ell_needed += 1; }
            #[allow(deprecated)]
            let r_pairs: Vec<(F, F)> = (0..ell_needed).map(|t| (legacy_me.r_point[2*t], legacy_me.r_point[2*t+1])).collect();
            #[allow(deprecated)]
            let base_b = F::from_u64(legacy_me.base_b as u64);
            let pow_b = piccs::pow_table(base_b, d);

            let compute_chi = |row_i: usize| -> (F, F) {
                let mut re = F::ONE; let mut im = F::ZERO; let mut mask = row_i;
                for t in 0..ell_needed {
                    let (rt_re, rt_im) = r_pairs[t];
                    let bit = (mask & 1) == 1;
                    let tr = if bit { rt_re } else { F::ONE - rt_re };
                    let ti = if bit { rt_im } else { -rt_im };
                    let new_re = re * tr - im * ti;
                    let new_im = re * ti + im * tr;
                    re = new_re; im = new_im; mask >>= 1;
                }
                (re, im)
            };

            let mut new_weights: Vec<Vec<F>> = Vec::with_capacity(2 * mats.len());
            for mj in mats.iter() {
                let mut by_col: Vec<Vec<(usize, F)>> = vec![Vec::new(); mj.cols];
                for &(r_idx, c_idx, a) in &mj.entries { by_col[c_idx as usize].push((r_idx as usize, a)); }
                let mut v_re = vec![F::ZERO; mj.cols];
                let mut v_im = vec![F::ZERO; mj.cols];
                for c in 0..mj.cols {
                    let mut acc_re = F::ZERO; let mut acc_im = F::ZERO;
                    for (r_i, a) in &by_col[c] {
                        let (chi_re, chi_im) = compute_chi(*r_i);
                        acc_re += *a * chi_re; acc_im += *a * chi_im;
                    }
                    v_re[c] = acc_re; v_im[c] = acc_im;
                }
                let mut w_re = vec![F::ZERO; d * mj.cols];
                let mut w_im = vec![F::ZERO; d * mj.cols];
                for c in 0..mj.cols { for r in 0..d { let idx = c*d + r; w_re[idx] = v_re[c] * pow_b[r]; w_im[idx] = v_im[c] * pow_b[r]; } }
                new_weights.push(w_re);
                new_weights.push(w_im);
            }
            #[allow(deprecated)]
            { legacy_wit.weight_vectors = new_weights; }
        }

        // Guard check
        if cfg!(debug_assertions) {
            neo_spartan_bridge::guards::assert_public_io_parity(
                &legacy_me, &legacy_wit, Some(&ev_embed), Some(ajtai_pp_arc.clone())
            )?;
        }

        neo_spartan_bridge::compress_ivc_verifier_to_lean_proof_with_linkage_and_pi_ccs(
            &legacy_me, &legacy_wit, Some(ajtai_pp_arc), Some(ev_embed), None, linkage, pi_ccs_embed_opt,
        )?
    } else {
        neo_spartan_bridge::compress_me_to_lean_proof_with_pp(&legacy_me, &legacy_wit, Some(ajtai_pp_arc))?
    };

    let proof = crate::Proof {
        v: 2,
        circuit_key: lean.circuit_key,
        vk_digest: lean.vk_digest,
        public_io: lean.public_io_bytes,
        proof_bytes: lean.proof_bytes,
        public_results: vec![],
        meta: crate::ProofMeta { 
            num_y_compact: last.inner.step_proof.meta.num_y_compact, 
            num_app_outputs: 0 
        },
    };
    Ok(Some((proof, augmented_ccs, final_public_input)))
}

/// Generate a succinct final SNARK proof for the NIVC chain (default: embed_ivc_ev = true).
pub fn finalize(
    program: &NivcProgram,
    params: &NeoParams,
    chain: NivcChainProof,
) -> anyhow::Result<Option<(crate::Proof, neo_ccs::CcsStructure<F>, Vec<F>)>> {
    finalize_with_options(program, params, chain, NivcFinalizeOptions { embed_ivc_ev: true })
}

