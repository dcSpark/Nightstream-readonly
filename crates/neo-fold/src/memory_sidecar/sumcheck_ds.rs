use neo_math::{KExtensions, K};
use neo_reductions::sumcheck::{
    run_batched_sumcheck_prover, run_sumcheck_prover, verify_batched_sumcheck_rounds, verify_sumcheck_rounds,
    BatchedClaim, BatchedClaimResult, RoundOracle,
};
use neo_transcript::{Poseidon2Transcript, Transcript};

use crate::PiCcsError;

/// Call before *any* sumcheck to domain-separate it from everything else.
fn sc_start(tr: &mut Poseidon2Transcript, domain: &'static [u8], inst_idx: usize) {
    tr.append_message(b"sc/domain", domain);
    tr.append_message(b"sc/inst_idx", &(inst_idx as u64).to_le_bytes());
}

fn sc_end(tr: &mut Poseidon2Transcript, domain: &'static [u8], inst_idx: usize) {
    tr.append_message(b"sc/domain_end", domain);
    tr.append_message(b"sc/inst_idx_end", &(inst_idx as u64).to_le_bytes());
}

pub fn run_sumcheck_prover_ds<O: RoundOracle>(
    tr: &mut Poseidon2Transcript,
    domain: &'static [u8],
    inst_idx: usize,
    oracle: &mut O,
    claimed_sum: K,
) -> Result<(Vec<Vec<K>>, Vec<K>), PiCcsError> {
    sc_start(tr, domain, inst_idx);
    tr.append_fields(b"sc/claimed_sum", &claimed_sum.as_coeffs());
    let out = run_sumcheck_prover(tr, oracle, claimed_sum)
        .map_err(|e| PiCcsError::SumcheckError(format!("{domain:?}: {e}")))?;
    sc_end(tr, domain, inst_idx);
    Ok(out)
}

pub fn verify_sumcheck_rounds_ds(
    tr: &mut Poseidon2Transcript,
    domain: &'static [u8],
    inst_idx: usize,
    degree_bound: usize,
    claimed_sum: K,
    rounds: &[Vec<K>],
) -> (Vec<K>, K, bool) {
    sc_start(tr, domain, inst_idx);
    tr.append_fields(b"sc/claimed_sum", &claimed_sum.as_coeffs());
    let out = verify_sumcheck_rounds(tr, degree_bound, claimed_sum, rounds);
    sc_end(tr, domain, inst_idx);
    out
}

pub fn run_batched_sumcheck_prover_ds<'a>(
    tr: &mut Poseidon2Transcript,
    domain: &'static [u8],
    inst_idx: usize,
    claims: &mut [BatchedClaim<'a>],
) -> Result<(Vec<K>, Vec<BatchedClaimResult>), PiCcsError> {
    sc_start(tr, domain, inst_idx);
    let out =
        run_batched_sumcheck_prover(tr, claims).map_err(|e| {
            PiCcsError::SumcheckError(format!(
                "{domain:?} (inst_idx={inst_idx}): {e}"
            ))
        })?;
    sc_end(tr, domain, inst_idx);
    Ok(out)
}

pub fn verify_batched_sumcheck_rounds_ds(
    tr: &mut Poseidon2Transcript,
    domain: &'static [u8],
    inst_idx: usize,
    round_polys: &[Vec<Vec<K>>],
    claimed_sums: &[K],
    labels: &[&[u8]],
    degree_bounds: &[usize],
) -> (Vec<K>, Vec<K>, bool) {
    sc_start(tr, domain, inst_idx);
    let out = verify_batched_sumcheck_rounds(tr, round_polys, claimed_sums, labels, degree_bounds);
    sc_end(tr, domain, inst_idx);
    out
}
