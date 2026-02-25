//! Shared helpers for Twist/Shout.
//!
//! This module is intentionally small and mechanical: it centralizes the
//! duplicated transcript sampling, ME-opening construction, CCS padding, and
//! Ajtai decode+pad utilities used by both protocols.

use neo_ajtai::Commitment as AjtaiCmt;
use neo_ccs::{matrix::Mat, relations::CeClaim, CcsStructure};
use neo_math::{from_complex, F as BaseField, K as KElem};
use neo_params::NeoParams;
use neo_reductions::error::PiCcsError;
use neo_transcript::{Poseidon2Transcript, Transcript, TranscriptProtocol};
use p3_field::{PrimeCharacteristicRing, PrimeField};

use crate::ajtai::decode_vector_for_ccs_m as ajtai_decode_vector_for_ccs_m;

// ============================================================================
// Transcript sampling helpers
// ============================================================================

pub fn sample_ext_point(
    tr: &mut Poseidon2Transcript,
    label: &'static [u8],
    coord0_label: &'static [u8],
    coord1_label: &'static [u8],
    len: usize,
) -> Vec<KElem> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        tr.append_message(label, &i.to_le_bytes());
        let c0 = tr.challenge_field(coord0_label);
        let c1 = tr.challenge_field(coord1_label);
        out.push(from_complex(c0, c1));
    }
    out
}

pub fn sample_base_addr_point(
    tr: &mut Poseidon2Transcript,
    label: &'static [u8],
    coord0_label: &'static [u8],
    len: usize,
) -> Vec<KElem> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        tr.append_message(label, &i.to_le_bytes());
        let c0 = tr.challenge_field(coord0_label);
        out.push(from_complex(c0, BaseField::ZERO));
    }
    out
}

// ============================================================================
// Transcript binding helpers
// ============================================================================

pub fn absorb_ajtai_commitments(
    tr: &mut Poseidon2Transcript,
    count_label: &'static [u8],
    idx_label: &'static [u8],
    comms: &[AjtaiCmt],
) {
    tr.append_message(count_label, &(comms.len() as u64).to_le_bytes());
    for (i, comm) in comms.iter().enumerate() {
        tr.append_message(idx_label, &(i as u64).to_le_bytes());
        tr.absorb_commit_coords(&comm.data);
    }
}

// ============================================================================
// CCS padding + ME opening
// ============================================================================

pub fn require_mat_layout_for_ccs_width(
    mat: &Mat<BaseField>,
    target_cols: usize,
) -> Result<Mat<BaseField>, PiCcsError> {
    // Keep this helper as the single shape gate for Route-A claim emission:
    // it now enforces the same strict layout policy as reductions.
    neo_reductions::common::witness_mat_layout(mat, target_cols).map_err(|e| {
        PiCcsError::InvalidInput(format!(
            "require_mat_layout_for_ccs_width: witness shape incompatible with logical CCS width m={target_cols}: {e}"
        ))
    })?;
    Ok(mat.clone())
}

/// Shared ME-opening constructor.
///
/// `digest_label` must remain domain-separated (e.g. `b"twist/me_digest"`, `b"shout/me_digest"`).
pub fn mk_me_opening_with_ccs<Cmt, KOut>(
    tr: &Poseidon2Transcript,
    digest_label: &'static [u8],
    params: &NeoParams,
    s: &CcsStructure<BaseField>,
    comm: &Cmt,
    mat: &Mat<BaseField>,
    r: &[KElem],
    m_in: usize,
) -> Result<CeClaim<Cmt, BaseField, KOut>, PiCcsError>
where
    KOut: From<KElem> + Clone,
    Cmt: Clone,
{
    let d = params.d as usize;
    let t = s.t();
    let y_pad = d.next_power_of_two();
    let ell_d = y_pad.trailing_zeros() as usize;

    // Pad witness to CCS width
    let z_padded = require_mat_layout_for_ccs_width(mat, s.m)?;

    // X = L_x(Z) over logical witness columns.
    let x_mat = neo_reductions::common::project_x_from_witness_mat(&z_padded, s.m, m_in).map_err(|e| {
        PiCcsError::InvalidInput(format!(
            "mk_me_opening_with_ccs: X projection failed for m={}, m_in={}: {e}",
            s.m, m_in
        ))
    })?;

    let (mut y_ring_k, mut ct_k) = neo_reductions::common::compute_y_from_Z_and_r(s, &z_padded, r, ell_d, params.b);
    y_ring_k.resize_with(t, || vec![KElem::ZERO; y_pad]);
    ct_k.resize(t, KElem::ZERO);
    let y_ring: Vec<Vec<KOut>> = y_ring_k
        .into_iter()
        .map(|yj| yj.into_iter().map(KOut::from).collect())
        .collect();

    let ct: Vec<KOut> = ct_k.into_iter().map(KOut::from).collect();

    let fold_digest = {
        let mut fork = tr.fork(digest_label);
        fork.digest32()
    };

    Ok(CeClaim {
        c: comm.clone(),
        X: x_mat,
        r: r.iter().copied().map(KOut::from).collect(),
        s_col: Vec::new(),
        y_ring,
        ct,
        aux_openings: Vec::new(),
        y_zcol: Vec::new(),
        m_in,
        fold_digest,
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    })
}

/// Decode address bits into flattened addresses (shared by semantic checkers).
pub fn decode_addrs_from_bits<F: PrimeField>(
    params: &NeoParams,
    addr_bit_mats: &[Mat<F>],
    d: usize,
    ell: usize,
    n_side: usize,
    steps: usize,
) -> Result<Vec<u64>, PiCcsError> {
    let decoded: Vec<Vec<F>> = addr_bit_mats
        .iter()
        .map(|m| ajtai_decode_vector_for_ccs_m(params, steps, m).map_err(PiCcsError::InvalidInput))
        .collect::<Result<Vec<_>, _>>()?;

    let mut addrs = vec![0u64; steps];
    for dim in 0..d {
        let base = dim * ell;
        let stride = (n_side as u64)
            .checked_pow(dim as u32)
            .ok_or_else(|| PiCcsError::InvalidInput("decode_addrs_from_bits: stride overflow".into()))?;
        for b in 0..ell {
            let col = &decoded[base + b];
            let bit_weight = 1u64
                .checked_shl(b as u32)
                .ok_or_else(|| PiCcsError::InvalidInput("decode_addrs_from_bits: bit_weight overflow".into()))?;
            for j in 0..steps.min(col.len()) {
                if col[j] == F::ONE {
                    let delta = bit_weight.checked_mul(stride).ok_or_else(|| {
                        PiCcsError::InvalidInput("decode_addrs_from_bits: address contribution overflow".into())
                    })?;
                    addrs[j] = addrs[j]
                        .checked_add(delta)
                        .ok_or_else(|| PiCcsError::InvalidInput("decode_addrs_from_bits: address overflow".into()))?;
                }
            }
        }
    }
    Ok(addrs)
}

// ============================================================================
// Convenience helpers
// ============================================================================

pub fn emit_me_claims_for_mats<Cmt>(
    tr: &Poseidon2Transcript,
    digest_label: &'static [u8],
    params: &NeoParams,
    s: &CcsStructure<BaseField>,
    comms: &[Cmt],
    mats: &[Mat<BaseField>],
    r: &[KElem],
    m_in: usize,
) -> Result<Vec<CeClaim<Cmt, BaseField, KElem>>, PiCcsError>
where
    Cmt: Clone,
{
    if comms.len() < mats.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "emit_me_claims_for_mats: comms.len()={} < mats.len()={}",
            comms.len(),
            mats.len()
        )));
    }

    let mut out = Vec::with_capacity(mats.len());
    for (i, mat) in mats.iter().enumerate() {
        out.push(mk_me_opening_with_ccs::<Cmt, KElem>(
            tr,
            digest_label,
            params,
            s,
            &comms[i],
            mat,
            r,
            m_in,
        )?);
    }
    Ok(out)
}
