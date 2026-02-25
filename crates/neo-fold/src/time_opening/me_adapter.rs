use crate::shard_proof_types::{OpeningDomain, TimeOpeningProof};
use crate::PiCcsError;
use neo_ajtai::{s_mul, Commitment as Cmt};
use neo_ccs::Mat;
use neo_math::{ring::cf_inv, ring::Rq as RqEl, D, F, K};
use p3_field::PrimeCharacteristicRing;

#[inline]
pub fn build_small_chi_table(point: &[K]) -> Result<Option<Vec<K>>, PiCcsError> {
    let n_pad = 1usize
        .checked_shl(point.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("time/opening eval: 2^ell_n overflow".into()))?;
    if n_pad <= (1usize << 16) {
        Ok(Some(neo_memory::mle::build_chi_table(point)))
    } else {
        Ok(None)
    }
}

#[inline]
fn chi_at(point: &[K], row: usize, chi_table: Option<&[K]>) -> K {
    if let Some(chi) = chi_table {
        chi[row]
    } else {
        chi_for_row_index(point, row)
    }
}

#[inline]
pub fn cpu_time_row_weights(point: &[K], m_in: usize, t: usize, chi_table: Option<&[K]>) -> Result<Vec<K>, PiCcsError> {
    let n_pad = 1usize
        .checked_shl(point.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("time/opening CPU weights: 2^ell_n overflow".into()))?;
    if m_in.checked_add(t).map_or(true, |end| end > n_pad) {
        return Err(PiCcsError::InvalidInput(format!(
            "time/opening CPU weights: row range out of bounds (m_in={}, t={}, n_pad={})",
            m_in, t, n_pad
        )));
    }
    let mut out = vec![K::ZERO; t];
    for (j, slot) in out.iter_mut().enumerate() {
        *slot = chi_at(point, m_in + j, chi_table);
    }
    Ok(out)
}

#[inline]
pub fn mem_time_row_weights(
    point: &[K],
    bus: &neo_memory::cpu::BusLayout,
    chi_table: Option<&[K]>,
) -> Result<Vec<K>, PiCcsError> {
    let n_pad = 1usize
        .checked_shl(point.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("time/opening MEM weights: 2^ell_n overflow".into()))?;
    let mut out = vec![K::ZERO; bus.chunk_size];
    for (j, slot) in out.iter_mut().enumerate() {
        let row = bus.time_index(j);
        if row >= n_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "time/opening MEM weights: bus time row out of range (row={}, n_pad={})",
                row, n_pad
            )));
        }
        *slot = chi_at(point, row, chi_table);
    }
    Ok(out)
}

#[inline]
pub fn eval_mat_digits_from_row_weights(weights: &[K], z: &Mat<F>) -> Result<Vec<K>, PiCcsError> {
    if z.cols() != weights.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "time/opening eval(digits): z.cols()={} != weights.len()={}",
            z.cols(),
            weights.len()
        )));
    }
    let mut out = vec![K::ZERO; z.rows()];
    for (j, &w) in weights.iter().enumerate() {
        if w == K::ZERO {
            continue;
        }
        for rho in 0..z.rows() {
            out[rho] += w * K::from(z[(rho, j)]);
        }
    }
    Ok(out)
}

#[inline]
pub fn chi_for_row_index(point: &[K], idx: usize) -> K {
    let mut acc = K::ONE;
    for (bit, &ri) in point.iter().enumerate() {
        let is_one = ((idx >> bit) & 1) == 1;
        acc *= if is_one { ri } else { K::ONE - ri };
    }
    acc
}

#[inline]
pub fn eval_cpu_time_vector_at_point(point: &[K], m_in: usize, col: &[F]) -> Result<K, PiCcsError> {
    let n_pad = 1usize
        .checked_shl(point.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("time/opening CPU eval: 2^ell_n overflow".into()))?;
    if m_in.checked_add(col.len()).map_or(true, |end| end > n_pad) {
        return Err(PiCcsError::InvalidInput(format!(
            "time/opening CPU eval: row range out of bounds (m_in={}, t={}, n_pad={})",
            m_in,
            col.len(),
            n_pad
        )));
    }
    let chi_table = if n_pad <= (1usize << 16) {
        Some(neo_memory::mle::build_chi_table(point))
    } else {
        None
    };
    let mut acc = K::ZERO;
    for (j, &v) in col.iter().enumerate() {
        let row = m_in + j;
        let w = if let Some(ref chi) = chi_table {
            chi[row]
        } else {
            chi_for_row_index(point, row)
        };
        acc += w * K::from(v);
    }
    Ok(acc)
}

#[inline]
pub fn eval_mem_time_vector_at_point(
    point: &[K],
    bus: &neo_memory::cpu::BusLayout,
    col: &[F],
) -> Result<K, PiCcsError> {
    if col.len() != bus.chunk_size {
        return Err(PiCcsError::InvalidInput(format!(
            "time/opening MEM eval: column length mismatch (len={}, chunk_size={})",
            col.len(),
            bus.chunk_size
        )));
    }
    let n_pad = 1usize
        .checked_shl(point.len() as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("time/opening MEM eval: 2^ell_n overflow".into()))?;
    let chi_table = if n_pad <= (1usize << 16) {
        Some(neo_memory::mle::build_chi_table(point))
    } else {
        None
    };
    let mut acc = K::ZERO;
    for (j, &v) in col.iter().enumerate() {
        let row = bus.time_index(j);
        if row >= n_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "time/opening MEM eval: bus time row out of range (row={}, n_pad={})",
                row, n_pad
            )));
        }
        let w = if let Some(ref chi) = chi_table {
            chi[row]
        } else {
            chi_for_row_index(point, row)
        };
        acc += w * K::from(v);
    }
    Ok(acc)
}

#[inline]
pub fn eval_time_vector_at_point(
    domain: OpeningDomain,
    point: &[K],
    m_in: usize,
    bus: &neo_memory::cpu::BusLayout,
    col: &[F],
) -> Result<K, PiCcsError> {
    match domain {
        OpeningDomain::Cpu => eval_cpu_time_vector_at_point(point, m_in, col),
        OpeningDomain::Mem => eval_mem_time_vector_at_point(point, bus, col),
    }
}

#[inline]
fn rot_matrix_to_rq(mat: &Mat<F>) -> Result<RqEl, PiCcsError> {
    if mat.rows() != D || mat.cols() != D {
        return Err(PiCcsError::InvalidInput(format!(
            "time/opening: rotation matrix must be {D}x{D} (got {}x{})",
            mat.rows(),
            mat.cols()
        )));
    }
    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    Ok(cf_inv(coeffs))
}

#[inline]
pub fn add_rot_scaled_commitment(acc: &mut Option<Cmt>, c: &Cmt, rho: &Mat<F>) -> Result<(), PiCcsError> {
    let rho_rq = rot_matrix_to_rq(rho)?;
    let term = s_mul(&rho_rq, c);
    if let Some(out) = acc.as_mut() {
        debug_assert_eq!(out.d, term.d);
        debug_assert_eq!(out.kappa, term.kappa);
        out.add_inplace(&term);
    } else {
        *acc = Some(term);
    }
    Ok(())
}

#[inline]
pub fn apply_rot_to_digits(rho: &Mat<F>, digits: &[K]) -> Result<Vec<K>, PiCcsError> {
    if rho.rows() != D || rho.cols() != D {
        return Err(PiCcsError::InvalidInput(format!(
            "time/opening: rotation matrix must be {D}x{D} (got {}x{})",
            rho.rows(),
            rho.cols()
        )));
    }
    if digits.len() != D {
        return Err(PiCcsError::InvalidInput(format!(
            "time/opening: digits.len()={} != D={D}",
            digits.len()
        )));
    }
    let mut out = vec![K::ZERO; D];
    for r in 0..D {
        let mut acc = K::ZERO;
        for k in 0..D {
            acc += K::from(rho[(r, k)]) * digits[k];
        }
        out[r] = acc;
    }
    Ok(out)
}

pub fn build_logical_col_pos(time_col_ids: &[usize]) -> Result<std::collections::BTreeMap<usize, usize>, PiCcsError> {
    let mut logical_col_pos = std::collections::BTreeMap::<usize, usize>::new();
    for (abs_pos, &col_id) in time_col_ids.iter().enumerate() {
        if logical_col_pos.insert(col_id, abs_pos).is_some() {
            return Err(PiCcsError::ProtocolError(format!(
                "time/opening: duplicate logical col_id={col_id} in time_col_ids",
            )));
        }
    }
    Ok(logical_col_pos)
}

pub fn domain_for_col_ids(
    col_ids: &[usize],
    logical_col_pos: &std::collections::BTreeMap<usize, usize>,
    cpu_cols_len: usize,
) -> Result<OpeningDomain, PiCcsError> {
    let mut domain: Option<OpeningDomain> = None;
    for &col_id in col_ids.iter() {
        let abs_pos = logical_col_pos.get(&col_id).copied().ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "time/opening: logical col_id {} not present in time_col_ids",
                col_id
            ))
        })?;
        let cur = if abs_pos < cpu_cols_len {
            OpeningDomain::Cpu
        } else {
            OpeningDomain::Mem
        };
        match domain {
            None => domain = Some(cur),
            Some(prev) if prev == cur => {}
            Some(_) => {
                return Err(PiCcsError::ProtocolError(
                    "time/opening: mixed CPU/MEM ids in a single opening claim".into(),
                ));
            }
        }
    }
    domain.ok_or_else(|| PiCcsError::ProtocolError("time/opening: empty col_ids".into()))
}

#[inline]
pub fn recompose_digits_to_scalar(digits: &[K], b: u32) -> K {
    let b_k = K::from(F::from_u64(b as u64));
    let mut pow = K::ONE;
    let mut acc = K::ZERO;
    for &d in digits {
        acc += pow * d;
        pow *= b_k;
    }
    acc
}

#[inline]
fn eval_cpu_time_mat_digits_at_point_with_chi(
    point: &[K],
    m_in: usize,
    z: &Mat<F>,
    chi_table: Option<&[K]>,
) -> Result<Vec<K>, PiCcsError> {
    let weights = cpu_time_row_weights(point, m_in, z.cols(), chi_table)?;
    eval_mat_digits_from_row_weights(weights.as_slice(), z)
}

#[inline]
fn eval_mem_time_mat_digits_at_point_with_chi(
    point: &[K],
    bus: &neo_memory::cpu::BusLayout,
    z: &Mat<F>,
    chi_table: Option<&[K]>,
) -> Result<Vec<K>, PiCcsError> {
    if z.cols() != bus.chunk_size {
        return Err(PiCcsError::InvalidInput(format!(
            "time/opening MEM eval(digits): z.cols()={} != chunk_size={}",
            z.cols(),
            bus.chunk_size
        )));
    }
    let weights = mem_time_row_weights(point, bus, chi_table)?;
    eval_mat_digits_from_row_weights(weights.as_slice(), z)
}

#[inline]
pub fn eval_time_mat_digits_at_point(
    domain: OpeningDomain,
    point: &[K],
    m_in: usize,
    bus: &neo_memory::cpu::BusLayout,
    z: &Mat<F>,
) -> Result<Vec<K>, PiCcsError> {
    let chi_storage = build_small_chi_table(point)?;
    eval_time_mat_digits_at_point_with_chi(domain, point, m_in, bus, z, chi_storage.as_deref())
}

#[inline]
pub fn eval_time_mat_digits_at_point_with_chi(
    domain: OpeningDomain,
    point: &[K],
    m_in: usize,
    bus: &neo_memory::cpu::BusLayout,
    z: &Mat<F>,
    chi_table: Option<&[K]>,
) -> Result<Vec<K>, PiCcsError> {
    match domain {
        OpeningDomain::Cpu => eval_cpu_time_mat_digits_at_point_with_chi(point, m_in, z, chi_table),
        OpeningDomain::Mem => eval_mem_time_mat_digits_at_point_with_chi(point, bus, z, chi_table),
    }
}

pub struct ClaimCommitEval {
    pub domain: OpeningDomain,
    pub commitment: Cmt,
    pub eval_scalar: K,
    pub eval_digits: Vec<K>,
}

pub fn claim_commitment_and_eval(
    open_pf: &TimeOpeningProof,
    coeffs: &[Mat<F>],
    logical_col_pos: &std::collections::BTreeMap<usize, usize>,
    cpu_cols_len: usize,
    time_cpu_commitments: &[Cmt],
    time_mem_commitments: &[Cmt],
    b: u32,
) -> Result<ClaimCommitEval, PiCcsError> {
    if open_pf.col_ids.len() != open_pf.evals.len()
        || open_pf.col_ids.len() != coeffs.len()
        || open_pf.col_ids.len() != open_pf.digit_evals.len()
    {
        return Err(PiCcsError::ProtocolError(
            "time/opening claim: malformed col_ids/evals/digit_evals/coeffs lengths".into(),
        ));
    }
    let domain = domain_for_col_ids(&open_pf.col_ids, logical_col_pos, cpu_cols_len)?;

    let mut claim_commitment: Option<Cmt> = None;
    let mut claim_digits = vec![K::ZERO; D];
    for (i, &col_id) in open_pf.col_ids.iter().enumerate() {
        if open_pf.digit_evals[i].len() != D {
            return Err(PiCcsError::ProtocolError(format!(
                "time/opening claim: digit_evals[{i}] len {} != D={D}",
                open_pf.digit_evals[i].len()
            )));
        }
        let recomposed = recompose_digits_to_scalar(open_pf.digit_evals[i].as_slice(), b);
        if recomposed != open_pf.evals[i] {
            return Err(PiCcsError::ProtocolError(format!(
                "time/opening claim: digit_evals[{i}] recomposition mismatch"
            )));
        }
        let rotated_digits = apply_rot_to_digits(&coeffs[i], open_pf.digit_evals[i].as_slice())?;
        for rho in 0..D {
            claim_digits[rho] += rotated_digits[rho];
        }

        let abs_pos = logical_col_pos.get(&col_id).copied().ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "time/opening claim: logical col_id {} not present in time_col_ids",
                col_id
            ))
        })?;
        let c_ref = if abs_pos < cpu_cols_len {
            time_cpu_commitments.get(abs_pos).ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "time/opening claim: CPU commitment index {} out of range",
                    abs_pos
                ))
            })?
        } else {
            let mem_idx = abs_pos - cpu_cols_len;
            time_mem_commitments.get(mem_idx).ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "time/opening claim: MEM commitment index {} out of range",
                    mem_idx
                ))
            })?
        };
        add_rot_scaled_commitment(&mut claim_commitment, c_ref, &coeffs[i])?;
    }

    let claim_commitment = claim_commitment
        .ok_or_else(|| PiCcsError::ProtocolError("time/opening claim: empty opening proof is not allowed".into()))?;
    let claim_eval = recompose_digits_to_scalar(claim_digits.as_slice(), b);
    Ok(ClaimCommitEval {
        domain,
        commitment: claim_commitment,
        eval_scalar: claim_eval,
        eval_digits: claim_digits,
    })
}
