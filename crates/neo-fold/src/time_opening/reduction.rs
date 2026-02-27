use crate::memory_sidecar::sumcheck_ds::{run_sumcheck_prover_ds, verify_sumcheck_rounds_ds};
use crate::shard_proof_types::{
    OpeningClaimManifest, OpeningDomain, OpeningReductionGroup, OpeningReductionProof, OpeningUnificationProof,
};
use crate::PiCcsError;
use neo_ccs::Mat;
use neo_math::{from_complex, KExtensions, F, K};
use neo_params::NeoParams;
use neo_reductions as ccs;
use neo_reductions::sumcheck::RoundOracle;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

fn group_digest(
    manifest_digest: &[u8; 32],
    point: &[neo_math::K],
    domain: OpeningDomain,
    claim_indices: &[usize],
) -> [u8; 32] {
    let mut h = Poseidon2Transcript::new(b"stage8/reduction/group_digest");
    h.append_message(b"stage8/reduction/group_digest/version", b"v2");
    h.append_message(b"stage8/reduction/group_digest/manifest_digest", manifest_digest);
    let dom = match domain {
        OpeningDomain::Cpu => 1u64,
        OpeningDomain::Mem => 2u64,
    };
    h.append_u64s(b"stage8/reduction/group_digest/domain", &[dom]);
    h.append_u64s(b"stage8/reduction/group_digest/point_len", &[point.len() as u64]);
    let point_coeffs_per_elem = point.first().map(|v| v.as_coeffs().len()).unwrap_or(0);
    h.append_fields_iter(
        b"stage8/reduction/group_digest/point",
        point.len().saturating_mul(point_coeffs_per_elem),
        point.iter().flat_map(|v| v.as_coeffs()),
    );
    let claim_indices_u64: Vec<u64> = claim_indices.iter().map(|&idx| idx as u64).collect();
    h.append_u64s(
        b"stage8/reduction/group_digest/claim_indices_len",
        &[claim_indices_u64.len() as u64],
    );
    h.append_u64s(b"stage8/reduction/group_digest/claim_indices", &claim_indices_u64);
    h.digest32()
}

pub fn build_opening_reduction(manifest: &OpeningClaimManifest) -> Result<OpeningReductionProof, PiCcsError> {
    let mut groups: Vec<OpeningReductionGroup> = Vec::new();
    for (idx, entry) in manifest.entries.iter().enumerate() {
        if let Some(group) = groups.iter_mut().find(|g| g.point == entry.point) {
            group.claim_indices.push(idx);
        } else {
            groups.push(OpeningReductionGroup {
                point: entry.point.clone(),
                // Canonical Stage-8 reduction domain:
                // time rows are contiguous in Route-A (`time_index(j)=m_in+j`),
                // so mixed CPU/MEM opening claims can share one reduction domain.
                domain: OpeningDomain::Cpu,
                claim_indices: vec![idx],
                group_digest: [0u8; 32],
            });
        }
    }

    for (group_idx, group) in groups.iter().enumerate() {
        if group.claim_indices.is_empty() {
            return Err(PiCcsError::ProtocolError(format!(
                "time/opening reduction: group[{group_idx}] has no claims"
            )));
        }
        if !group.claim_indices.windows(2).all(|w| w[0] < w[1]) {
            return Err(PiCcsError::ProtocolError(format!(
                "time/opening reduction: group[{group_idx}] claim indices must be strictly increasing"
            )));
        }
        match group.domain {
            OpeningDomain::Cpu | OpeningDomain::Mem => {}
        }
    }
    for group in groups.iter_mut() {
        group.group_digest = group_digest(
            &manifest.digest,
            group.point.as_slice(),
            group.domain,
            group.claim_indices.as_slice(),
        );
    }

    Ok(OpeningReductionProof { groups })
}

pub fn bind_opening_reduction_and_sample_group_coeffs(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    step_idx: usize,
    manifest_len: usize,
    manifest_digest: &[u8; 32],
    reduction: &OpeningReductionProof,
) -> Result<Vec<Vec<Mat<F>>>, PiCcsError> {
    tr.append_message(b"stage8/reduction_bind/v1", &[]);
    tr.append_u64s(b"stage8/reduction_bind/step_idx", &[step_idx as u64]);
    tr.append_u64s(b"stage8/reduction_bind/group_len", &[reduction.groups.len() as u64]);

    let mut seen = vec![false; manifest_len];
    let mut out_coeffs: Vec<Vec<Mat<F>>> = Vec::with_capacity(reduction.groups.len());
    let ring = ccs::RotRing::goldilocks();
    for (group_idx, group) in reduction.groups.iter().enumerate() {
        let expected_digest = group_digest(
            manifest_digest,
            group.point.as_slice(),
            group.domain,
            group.claim_indices.as_slice(),
        );
        if group.group_digest != expected_digest {
            return Err(PiCcsError::ProtocolError(format!(
                "time/opening reduction: group[{group_idx}] digest mismatch"
            )));
        }

        tr.append_u64s(b"stage8/reduction_bind/group_idx", &[group_idx as u64]);
        let dom = match group.domain {
            OpeningDomain::Cpu => 0u64,
            OpeningDomain::Mem => 1u64,
        };
        tr.append_u64s(b"stage8/reduction_bind/domain", &[dom]);
        tr.append_u64s(b"stage8/reduction_bind/point_len", &[group.point.len() as u64]);
        let point_coeffs_per_elem = group
            .point
            .first()
            .map(|v| v.as_coeffs().len())
            .unwrap_or(0);
        tr.append_fields_iter(
            b"stage8/reduction_bind/point",
            group.point.len().saturating_mul(point_coeffs_per_elem),
            group.point.iter().flat_map(|v| v.as_coeffs()),
        );
        tr.append_message(b"stage8/reduction_bind/group_digest", &group.group_digest);

        let mut idx_u64 = Vec::with_capacity(group.claim_indices.len());
        for &claim_idx in group.claim_indices.iter() {
            if claim_idx >= manifest_len {
                return Err(PiCcsError::ProtocolError(format!(
                    "time/opening reduction: claim index {} out of range for manifest_len={}",
                    claim_idx, manifest_len
                )));
            }
            if seen[claim_idx] {
                return Err(PiCcsError::ProtocolError(format!(
                    "time/opening reduction: duplicate claim index {}",
                    claim_idx
                )));
            }
            seen[claim_idx] = true;
            idx_u64.push(claim_idx as u64);
        }
        tr.append_u64s(b"stage8/reduction_bind/claim_indices", &idx_u64);
        tr.append_message(b"stage8/reduction/rho", &(group_idx as u64).to_le_bytes());
        let rhos = ccs::sample_rot_rhos_n(tr, params, &ring, group.claim_indices.len())?;
        for i in 0..rhos.len() {
            for j in (i + 1)..rhos.len() {
                if rhos[i] == rhos[j] {
                    return Err(PiCcsError::ProtocolError(format!(
                        "time/opening reduction: duplicate rho matrices in group {group_idx} (indices {i},{j})"
                    )));
                }
            }
        }
        out_coeffs.push(rhos);
    }

    if seen.iter().any(|v| !*v) {
        return Err(PiCcsError::ProtocolError(
            "time/opening reduction: groups do not cover all manifest entries".into(),
        ));
    }

    Ok(out_coeffs)
}

#[inline]
fn ceil_log2_at_least_1(n: usize) -> usize {
    let need = n.max(1).next_power_of_two();
    (need.trailing_zeros() as usize).max(1)
}

fn group_value(group: &OpeningReductionGroup) -> K {
    let mut h = Poseidon2Transcript::new(b"stage8/reduction/group_value");
    h.append_message(b"stage8/reduction/group_value/version", b"v1");
    h.append_message(b"stage8/reduction/group_value/digest", &group.group_digest);
    let dom = match group.domain {
        OpeningDomain::Cpu => 0u64,
        OpeningDomain::Mem => 1u64,
    };
    h.append_u64s(b"stage8/reduction/group_value/domain", &[dom]);
    h.append_u64s(b"stage8/reduction/group_value/point_len", &[group.point.len() as u64]);
    let point_coeffs_per_elem = group
        .point
        .first()
        .map(|v| v.as_coeffs().len())
        .unwrap_or(0);
    h.append_fields_iter(
        b"stage8/reduction/group_value/point",
        group.point.len().saturating_mul(point_coeffs_per_elem),
        group.point.iter().flat_map(|v| v.as_coeffs()),
    );
    let idx_u64: Vec<u64> = group.claim_indices.iter().map(|&idx| idx as u64).collect();
    h.append_u64s(
        b"stage8/reduction/group_value/claim_indices_len",
        &[idx_u64.len() as u64],
    );
    h.append_u64s(b"stage8/reduction/group_value/claim_indices", &idx_u64);
    let c = h.challenge_field(b"stage8/reduction/group_value/0");
    let d = h.challenge_field(b"stage8/reduction/group_value/1");
    from_complex(c, d)
}

fn bind_opening_unification_statement(
    tr: &mut Poseidon2Transcript,
    step_idx: usize,
    reduction: &OpeningReductionProof,
    ell_sel: usize,
    values: &[K],
) {
    tr.append_message(b"stage8/reduction_unify_bind/v1", &[]);
    tr.append_u64s(b"stage8/reduction_unify_bind/step_idx", &[step_idx as u64]);
    tr.append_u64s(
        b"stage8/reduction_unify_bind/group_len",
        &[reduction.groups.len() as u64],
    );
    tr.append_u64s(b"stage8/reduction_unify_bind/ell_sel", &[ell_sel as u64]);
    for (idx, (group, value)) in reduction.groups.iter().zip(values.iter()).enumerate() {
        tr.append_u64s(b"stage8/reduction_unify_bind/group_idx", &[idx as u64]);
        let dom = match group.domain {
            OpeningDomain::Cpu => 0u64,
            OpeningDomain::Mem => 1u64,
        };
        tr.append_u64s(b"stage8/reduction_unify_bind/domain", &[dom]);
        tr.append_u64s(b"stage8/reduction_unify_bind/point_len", &[group.point.len() as u64]);
        let point_coeffs_per_elem = group
            .point
            .first()
            .map(|v| v.as_coeffs().len())
            .unwrap_or(0);
        tr.append_fields_iter(
            b"stage8/reduction_unify_bind/point",
            group.point.len().saturating_mul(point_coeffs_per_elem),
            group.point.iter().flat_map(|v| v.as_coeffs()),
        );
        let idx_u64: Vec<u64> = group
            .claim_indices
            .iter()
            .map(|&claim_idx| claim_idx as u64)
            .collect();
        tr.append_u64s(
            b"stage8/reduction_unify_bind/claim_indices_len",
            &[idx_u64.len() as u64],
        );
        tr.append_u64s(b"stage8/reduction_unify_bind/claim_indices", &idx_u64);
        tr.append_message(b"stage8/reduction_unify_bind/group_digest", &group.group_digest);
        tr.append_fields(b"stage8/reduction_unify_bind/group_value", &value.as_coeffs());
    }
}

#[derive(Clone, Debug)]
struct GroupSelectorOracle {
    values: Vec<K>,
    ell_sel: usize,
    prefix: Vec<K>,
}

impl GroupSelectorOracle {
    fn new(values: Vec<K>, ell_sel: usize) -> Self {
        Self {
            values,
            ell_sel,
            prefix: Vec::with_capacity(ell_sel),
        }
    }

    #[inline]
    fn bit_at(index: usize, bit: usize) -> bool {
        ((index >> bit) & 1usize) == 1
    }

    #[inline]
    fn bit_weight(bit: bool, x: K) -> K {
        if bit {
            x
        } else {
            K::ONE - x
        }
    }

    fn eval_at_point(values: &[K], r: &[K]) -> K {
        let mut acc = K::ZERO;
        for (group_idx, value) in values.iter().enumerate() {
            let mut w = K::ONE;
            for (bit_idx, &rv) in r.iter().enumerate() {
                w *= Self::bit_weight(Self::bit_at(group_idx, bit_idx), rv);
            }
            acc += w * *value;
        }
        acc
    }
}

impl RoundOracle for GroupSelectorOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.prefix.len() >= self.ell_sel {
            return vec![K::ZERO; points.len()];
        }
        let round_idx = self.prefix.len();
        let mut out = vec![K::ZERO; points.len()];
        for (group_idx, value) in self.values.iter().enumerate() {
            let mut pref_w = K::ONE;
            for (bit_idx, &bound) in self.prefix.iter().enumerate() {
                pref_w *= Self::bit_weight(Self::bit_at(group_idx, bit_idx), bound);
            }
            for (i, &x) in points.iter().enumerate() {
                out[i] += pref_w * Self::bit_weight(Self::bit_at(group_idx, round_idx), x) * *value;
            }
        }
        out
    }

    fn num_rounds(&self) -> usize {
        self.ell_sel
    }

    fn degree_bound(&self) -> usize {
        1
    }

    fn fold(&mut self, r: K) {
        self.prefix.push(r);
    }
}

pub fn prove_opening_unification_sumcheck(
    tr: &mut Poseidon2Transcript,
    step_idx: usize,
    reduction: &OpeningReductionProof,
) -> Result<OpeningUnificationProof, PiCcsError> {
    if reduction.groups.is_empty() {
        return Ok(OpeningUnificationProof::default());
    }
    let values: Vec<K> = reduction.groups.iter().map(group_value).collect();
    let ell_sel = ceil_log2_at_least_1(reduction.groups.len());
    bind_opening_unification_statement(tr, step_idx, reduction, ell_sel, &values);
    let claimed_sum = values.iter().copied().fold(K::ZERO, |acc, v| acc + v);
    let mut oracle = GroupSelectorOracle::new(values, ell_sel);
    let (round_polys, r_unify) =
        run_sumcheck_prover_ds(tr, b"stage8/reduction_unify", step_idx, &mut oracle, claimed_sum)?;
    if round_polys.len() != ell_sel || r_unify.len() != ell_sel {
        return Err(PiCcsError::ProtocolError(format!(
            "stage8/reduction_unify prove: round/challenge length mismatch (rounds={}, r_unify={}, ell_sel={})",
            round_polys.len(),
            r_unify.len(),
            ell_sel
        )));
    }
    Ok(OpeningUnificationProof {
        claimed_sum,
        round_polys,
        r_unify,
    })
}

pub fn verify_opening_unification_sumcheck(
    tr: &mut Poseidon2Transcript,
    step_idx: usize,
    reduction: &OpeningReductionProof,
    proof: &OpeningUnificationProof,
) -> Result<(), PiCcsError> {
    if reduction.groups.is_empty() {
        if proof.round_polys.is_empty() && proof.r_unify.is_empty() && proof.claimed_sum == K::ZERO {
            return Ok(());
        }
        return Err(PiCcsError::ProtocolError(
            "stage8/reduction_unify verify: expected empty proof when there are no reduction groups".into(),
        ));
    }
    let values: Vec<K> = reduction.groups.iter().map(group_value).collect();
    let ell_sel = ceil_log2_at_least_1(reduction.groups.len());
    bind_opening_unification_statement(tr, step_idx, reduction, ell_sel, &values);
    let expected_sum = values.iter().copied().fold(K::ZERO, |acc, v| acc + v);
    if proof.claimed_sum != expected_sum {
        return Err(PiCcsError::ProtocolError(
            "stage8/reduction_unify verify: claimed_sum mismatch".into(),
        ));
    }
    if proof.round_polys.len() != ell_sel {
        return Err(PiCcsError::ProtocolError(format!(
            "stage8/reduction_unify verify: expected {} rounds, got {}",
            ell_sel,
            proof.round_polys.len()
        )));
    }
    let (r_unify, final_value, ok) = verify_sumcheck_rounds_ds(
        tr,
        b"stage8/reduction_unify",
        step_idx,
        1,
        proof.claimed_sum,
        &proof.round_polys,
    );
    if !ok {
        return Err(PiCcsError::ProtocolError(
            "stage8/reduction_unify verify: sumcheck verification failed".into(),
        ));
    }
    if r_unify != proof.r_unify {
        return Err(PiCcsError::ProtocolError(
            "stage8/reduction_unify verify: r_unify mismatch".into(),
        ));
    }
    let expected_final = GroupSelectorOracle::eval_at_point(&values, &proof.r_unify);
    if final_value != expected_final {
        return Err(PiCcsError::ProtocolError(
            "stage8/reduction_unify verify: final value mismatch".into(),
        ));
    }
    Ok(())
}
