use crate::shard_proof_types::{
    OpeningClaimEntry, OpeningClaimManifest, OpeningDomain, TimeOpeningProof, TimeOpeningSource, TimePointOpening,
};
use crate::PiCcsError;
use neo_math::KExtensions;
use neo_transcript::{Poseidon2Transcript, Transcript};

fn digest_manifest_entries(entries: &[OpeningClaimEntry]) -> [u8; 32] {
    let mut h = Poseidon2Transcript::new(b"time_openings/manifest_digest");
    h.append_message(b"time_openings/manifest_digest/version", b"v2");
    h.append_u64s(b"time_openings/manifest_digest/entries_len", &[entries.len() as u64]);
    for (idx, entry) in entries.iter().enumerate() {
        let dom = match entry.domain {
            OpeningDomain::Cpu => 1u64,
            OpeningDomain::Mem => 2u64,
        };
        let src = match entry.source {
            TimeOpeningSource::Unknown => 0u64,
            TimeOpeningSource::CommittedOpening => 1u64,
            TimeOpeningSource::VirtualReducedOpening => 2u64,
        };
        h.append_u64s(b"time_openings/manifest_digest/entry_idx", &[idx as u64]);
        h.append_u64s(b"time_openings/manifest_digest/domain", &[dom]);
        h.append_u64s(b"time_openings/manifest_digest/source", &[src]);
        h.append_u64s(b"time_openings/manifest_digest/point_len", &[entry.point.len() as u64]);
        let point_coeffs_per_elem = entry
            .point
            .first()
            .map(|v| v.as_coeffs().len())
            .unwrap_or(0);
        h.append_fields_iter(
            b"time_openings/manifest_digest/point",
            entry.point.len().saturating_mul(point_coeffs_per_elem),
            entry.point.iter().flat_map(|v| v.as_coeffs()),
        );
        let col_ids_u64: Vec<u64> = entry.col_ids.iter().map(|&id| id as u64).collect();
        h.append_u64s(
            b"time_openings/manifest_digest/col_ids_len",
            &[col_ids_u64.len() as u64],
        );
        h.append_u64s(b"time_openings/manifest_digest/col_ids", &col_ids_u64);
    }
    h.digest32()
}

fn infer_domain_for_col_ids(
    col_ids: &[usize],
    time_col_ids: &[usize],
    cpu_commitment_len: usize,
) -> Result<OpeningDomain, PiCcsError> {
    let mut pos = std::collections::BTreeMap::<usize, usize>::new();
    for (i, &id) in time_col_ids.iter().enumerate() {
        if pos.insert(id, i).is_some() {
            return Err(PiCcsError::ProtocolError(format!(
                "time/opening manifest: duplicate logical column id {id} in time_col_ids"
            )));
        }
    }
    let mut domain: Option<OpeningDomain> = None;
    for &id in col_ids.iter() {
        let abs_pos = pos.get(&id).copied().ok_or_else(|| {
            PiCcsError::ProtocolError(format!("time/opening manifest: col_id={id} missing from time_col_ids"))
        })?;
        let d = if abs_pos < cpu_commitment_len {
            OpeningDomain::Cpu
        } else {
            OpeningDomain::Mem
        };
        match domain {
            None => domain = Some(d),
            Some(prev) if prev == d => {}
            Some(_) => {
                return Err(PiCcsError::ProtocolError(
                    "time/opening manifest: mixed CPU/MEM col_ids in a single opening claim".into(),
                ))
            }
        }
    }
    domain.ok_or_else(|| PiCcsError::ProtocolError("time/opening manifest: empty col_ids".into()))
}

pub fn build_opening_claim_manifest(
    fold_openings: &[TimePointOpening],
    opening_proofs: &[TimeOpeningProof],
    time_col_ids: &[usize],
    cpu_commitment_len: usize,
) -> Result<OpeningClaimManifest, PiCcsError> {
    let mut entries = Vec::<OpeningClaimEntry>::with_capacity(opening_proofs.len());
    for (idx, pf) in opening_proofs.iter().enumerate() {
        if pf.point.is_empty() {
            return Err(PiCcsError::ProtocolError(format!(
                "time/opening manifest: opening_proofs[{idx}] has empty point"
            )));
        }
        if pf.col_ids.is_empty() || pf.col_ids.len() != pf.evals.len() || pf.col_ids.len() != pf.digit_evals.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "time/opening manifest: opening_proofs[{idx}] malformed"
            )));
        }
        if !pf.col_ids.windows(2).all(|w| w[0] < w[1]) {
            return Err(PiCcsError::ProtocolError(format!(
                "time/opening manifest: opening_proofs[{idx}] col_ids must be strictly sorted"
            )));
        }
        for (digit_idx, digit_eval) in pf.digit_evals.iter().enumerate() {
            if digit_eval.len() != neo_math::D {
                return Err(PiCcsError::ProtocolError(format!(
                    "time/opening manifest: opening_proofs[{idx}] digit_evals[{digit_idx}] len {} != D={}",
                    digit_eval.len(),
                    neo_math::D
                )));
            }
        }
        let src = {
            let mut matched_src = None;
            for opening in fold_openings.iter().filter(|o| o.point == pf.point) {
                if opening.col_ids.len() != pf.col_ids.len() {
                    continue;
                }
                let mut opening_norm = opening.col_ids.clone();
                opening_norm.sort_unstable();
                if opening_norm != pf.col_ids {
                    continue;
                }
                if matched_src.is_some() {
                    return Err(PiCcsError::ProtocolError(format!(
                        "time/opening manifest: duplicate TimePointOpening matches opening_proofs[{idx}]"
                    )));
                }
                matched_src = Some(opening.source);
            }
            matched_src.ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "time/opening manifest: no matching TimePointOpening for opening_proofs[{idx}]"
                ))
            })?
        };
        if src != TimeOpeningSource::CommittedOpening {
            return Err(PiCcsError::ProtocolError(format!(
                "time/opening manifest: non-committed opening source {:?} is not accepted in canonical mode",
                src
            )));
        }
        let domain = infer_domain_for_col_ids(&pf.col_ids, time_col_ids, cpu_commitment_len)?;
        entries.push(OpeningClaimEntry {
            point: pf.point.clone(),
            col_ids: pf.col_ids.clone(),
            source: src,
            domain,
        });
    }

    for i in 0..entries.len() {
        for j in (i + 1)..entries.len() {
            if entries[i].domain == entries[j].domain
                && entries[i].point == entries[j].point
                && entries[i].col_ids == entries[j].col_ids
            {
                return Err(PiCcsError::ProtocolError(
                    "time/opening manifest: duplicate (point, col_ids, domain) claim".into(),
                ));
            }
            if entries[i].point == entries[j].point && entries[i].col_ids == entries[j].col_ids {
                return Err(PiCcsError::ProtocolError(
                    "time/opening manifest: conflicting domain for identical (point, col_ids) claim".into(),
                ));
            }
        }
    }

    let digest = digest_manifest_entries(&entries);
    Ok(OpeningClaimManifest { entries, digest })
}

pub fn bind_opening_claim_manifest(tr: &mut Poseidon2Transcript, step_idx: usize, manifest: &OpeningClaimManifest) {
    tr.append_message(b"time_openings/manifest_bind/v1", &[]);
    tr.append_u64s(b"time_openings/manifest_bind/step_idx", &[step_idx as u64]);
    tr.append_u64s(
        b"time_openings/manifest_bind/entries_len",
        &[manifest.entries.len() as u64],
    );
    for (entry_idx, entry) in manifest.entries.iter().enumerate() {
        tr.append_u64s(b"time_openings/manifest_bind/entry_idx", &[entry_idx as u64]);
        let dom = match entry.domain {
            OpeningDomain::Cpu => 0u64,
            OpeningDomain::Mem => 1u64,
        };
        let src = match entry.source {
            TimeOpeningSource::Unknown => 0u64,
            TimeOpeningSource::CommittedOpening => 1u64,
            TimeOpeningSource::VirtualReducedOpening => 2u64,
        };
        tr.append_u64s(b"time_openings/manifest_bind/domain", &[dom]);
        tr.append_u64s(b"time_openings/manifest_bind/source", &[src]);
        tr.append_u64s(b"time_openings/manifest_bind/point_len", &[entry.point.len() as u64]);
        let point_coeffs_per_elem = entry
            .point
            .first()
            .map(|v| v.as_coeffs().len())
            .unwrap_or(0);
        tr.append_fields_iter(
            b"time_openings/manifest_bind/point",
            entry.point.len().saturating_mul(point_coeffs_per_elem),
            entry.point.iter().flat_map(|v| v.as_coeffs()),
        );
        let col_ids_u64: Vec<u64> = entry.col_ids.iter().map(|&id| id as u64).collect();
        tr.append_u64s(b"time_openings/manifest_bind/col_ids", &col_ids_u64);
    }
    tr.append_message(b"time_openings/manifest_bind/digest", &manifest.digest);
}
