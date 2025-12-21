//! Twist argument for read/write memory correctness (Route A).
//!
//! This module intentionally supports only Route A integration inside
//! `neo-fold::shard`. The legacy fixed-challenge APIs were removed.
//!
//! Current Route A semantics: per-chunk Twist with a virtual `Val` (via sparse
//! increments + LT-based val-eval) and read/write zero-checks.
//!
//! `init_vals` are provided per chunk. Cross-chunk rollover is enforced by the
//! Route A integration in `neo-fold::shard` (not by this per-chunk argument).

use crate::ajtai::decode_vector as ajtai_decode_vector;
use crate::ts_common as ts;
use crate::twist_oracle::{
    compute_eq_from_bits, table_mle_eval, LazyBitnessOracle, TwistReadCheckAddrOracle, TwistReadCheckOracle,
    TwistWriteCheckAddrOracle, TwistWriteCheckOracle,
};
use crate::witness::{MemInstance, MemWitness};
use neo_ajtai::Commitment as AjtaiCmt;
use neo_ccs::matrix::Mat;
use neo_math::{F as BaseField, K as KElem};
use neo_params::NeoParams;
use neo_reductions::error::PiCcsError;
use neo_transcript::Poseidon2Transcript;
use p3_field::{PrimeCharacteristicRing, PrimeField};
use serde::{Deserialize, Serialize};

// ============================================================================
// Input validation
// ============================================================================

fn validate_index_bit_addressing<Cmt, F: PrimeCharacteristicRing + PartialEq>(
    inst: &MemInstance<Cmt, F>,
) -> Result<(), PiCcsError> {
    crate::addr::validate_pow2_bit_addressing("Twist", inst.n_side, inst.d, inst.ell, inst.k)?;
    if inst.init_vals.len() != inst.k {
        return Err(PiCcsError::InvalidInput(format!(
            "Twist: init_vals.len()={} must equal k={}",
            inst.init_vals.len(),
            inst.k
        )));
    }

    Ok(())
}

// ============================================================================
// Transcript binding
// ============================================================================

/// Absorb all Twist commitments into the transcript.
///
/// Must be called before sampling any challenge used to open these commitments.
pub fn absorb_commitments<F>(tr: &mut Poseidon2Transcript, inst: &MemInstance<AjtaiCmt, F>) {
    ts::absorb_ajtai_commitments(tr, b"twist/absorb_commitments", b"twist/comm_idx", &inst.comms);
}

// ============================================================================
// Proof metadata (Route A)
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchedAddrProof<F> {
    /// claimed_sums[0] = read_addr_claim_sum
    /// claimed_sums[1] = write_addr_claim_sum
    pub claimed_sums: Vec<F>,
    pub round_polys: Vec<Vec<Vec<F>>>,
    pub r_addr: Vec<F>,
}

impl<F: Default> Default for BatchedAddrProof<F> {
    fn default() -> Self {
        Self {
            claimed_sums: Vec::new(),
            round_polys: Vec::new(),
            r_addr: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TwistProof<F> {
    pub me_claim_count: usize,
    pub addr_batch: BatchedAddrProof<F>,
    pub val_eval: Option<TwistValEvalProof<F>>,
}

impl<F: Default> Default for TwistProof<F> {
    fn default() -> Self {
        Self {
            me_claim_count: 0,
            addr_batch: BatchedAddrProof::default(),
            val_eval: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TwistValEvalProof<F> {
    /// V = Val(r_addr, r_time) (pre-write).
    pub claimed_val: F,
    /// Σ_t Inc(r_addr, t) · LT(t, r_time) (init term excluded).
    pub claimed_inc_sum_lt: F,
    /// Sum-check rounds for the LT-weighted claim (ell_n rounds, cycle/time variables).
    pub rounds_lt: Vec<Vec<F>>,

    /// Σ_t Inc(r_addr, t) (total increment over the whole chunk).
    pub claimed_inc_sum_total: F,
    /// Sum-check rounds for the total-increment claim (ell_n rounds, cycle/time variables).
    pub rounds_total: Vec<Vec<F>>,

    /// Challenge point for this sum-check (cycle/time variables).
    pub r_val: Vec<F>,
}

// ============================================================================
// Witness layout helpers
// ============================================================================

#[derive(Clone, Debug)]
pub struct TwistWitnessParts<'a, F> {
    pub ra_bit_mats: &'a [Mat<F>],
    pub wa_bit_mats: &'a [Mat<F>],
    pub has_read_mat: &'a Mat<F>,
    pub has_write_mat: &'a Mat<F>,
    pub wv_mat: &'a Mat<F>,
    pub rv_mat: &'a Mat<F>,
    pub inc_at_write_addr_mat: &'a Mat<F>,
}

/// Layout: `[ra_bits (d*ell), wa_bits (d*ell), has_read, has_write, wv, rv, inc_at_write_addr]`.
pub fn split_mem_mats<'a, F: Clone>(
    inst: &MemInstance<impl Clone, F>,
    wit: &'a MemWitness<F>,
) -> TwistWitnessParts<'a, F> {
    let ell_addr = inst.d * inst.ell;
    let expected = 2 * ell_addr + 5;
    assert_eq!(
        wit.mats.len(),
        expected,
        "MemWitness has {} matrices, expected {} (2*d*ell={} + has_read + has_write + wv + rv + inc_at_write_addr)",
        wit.mats.len(),
        expected,
        2 * ell_addr
    );

    TwistWitnessParts {
        ra_bit_mats: &wit.mats[..ell_addr],
        wa_bit_mats: &wit.mats[ell_addr..2 * ell_addr],
        has_read_mat: &wit.mats[2 * ell_addr],
        has_write_mat: &wit.mats[2 * ell_addr + 1],
        wv_mat: &wit.mats[2 * ell_addr + 2],
        rv_mat: &wit.mats[2 * ell_addr + 3],
        inc_at_write_addr_mat: &wit.mats[2 * ell_addr + 4],
    }
}

// ============================================================================
// Semantic checker (debug/tests)
// ============================================================================

pub fn check_twist_semantics<F: PrimeField>(
    params: &NeoParams,
    inst: &MemInstance<impl Clone, F>,
    wit: &MemWitness<F>,
) -> Result<(), PiCcsError> {
    validate_index_bit_addressing(inst)?;

    let parts = split_mem_mats(inst, wit);
    let steps = inst.steps;
    let k = inst.k;

    let has_read = ajtai_decode_vector(params, parts.has_read_mat);
    let has_write = ajtai_decode_vector(params, parts.has_write_mat);
    let wv = ajtai_decode_vector(params, parts.wv_mat);
    let rv = ajtai_decode_vector(params, parts.rv_mat);
    let inc_at_write_addr = ajtai_decode_vector(params, parts.inc_at_write_addr_mat);

    // Bitness of address bits.
    for mat in parts.ra_bit_mats.iter().chain(parts.wa_bit_mats.iter()) {
        let v = ajtai_decode_vector(params, mat);
        for (j, &x) in v.iter().enumerate() {
            if x != F::ZERO && x != F::ONE {
                return Err(PiCcsError::InvalidInput(format!(
                    "Twist: non-binary value in address bit column at step {j}: {x:?}"
                )));
            }
        }
    }

    // Decode addresses.
    let read_addrs = ts::decode_addrs_from_bits(params, parts.ra_bit_mats, inst.d, inst.ell, inst.n_side, steps);
    let write_addrs = ts::decode_addrs_from_bits(params, parts.wa_bit_mats, inst.d, inst.ell, inst.n_side, steps);

    // Route A prototype: per-chunk init values are provided in the instance.
    let mut mem = inst.init_vals.clone();
    for j in 0..steps {
        if has_read[j] == F::ONE {
            let addr = read_addrs[j] as usize;
            if addr < k && rv[j] != mem[addr] {
                return Err(PiCcsError::InvalidInput(format!(
                    "Twist: read mismatch at step {j}: rv={:?}, mem[{addr}]={:?}",
                    rv[j], mem[addr]
                )));
            }
        }

        if has_write[j] == F::ONE {
            let addr = write_addrs[j] as usize;
            if addr < k {
                let expected_inc = wv[j] - mem[addr];
                if inc_at_write_addr[j] != expected_inc {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Twist: inc mismatch at step {j}, addr {addr}: got {:?}, expected {:?}",
                        inc_at_write_addr[j], expected_inc
                    )));
                }
                mem[addr] = wv[j];
            }
        }
    }

    Ok(())
}

// ============================================================================
// Route A decoded columns (time-domain only).
// ============================================================================

#[derive(Clone, Debug)]
pub struct TwistDecodedCols {
    pub ra_bits: Vec<Vec<KElem>>,
    pub wa_bits: Vec<Vec<KElem>>,
    pub has_read: Vec<KElem>,
    pub has_write: Vec<KElem>,
    pub wv: Vec<KElem>,
    pub rv: Vec<KElem>,
    pub inc_at_write_addr: Vec<KElem>,
}

/// Compute Val(r_addr, t) on all Boolean t in the padded time domain.
///
/// Semantics: "pre-write" value at time t.
/// Update order matches `check_twist_semantics`: read sees pre-write, then write updates.
///
/// Inputs are decoded K-vectors (length pow2_cycle).
pub fn compute_val_at_r_addr_pre_write(
    wa_bits: &[Vec<KElem>],      // len = ell_addr, each len = pow2_cycle
    has_write: &[KElem],         // len = pow2_cycle
    inc_at_write_addr: &[KElem], // len = pow2_cycle
    r_addr: &[KElem],            // len = ell_addr
    init_at_r_addr: KElem,
) -> Vec<KElem> {
    let pow2_cycle = has_write.len();
    assert_eq!(
        inc_at_write_addr.len(),
        pow2_cycle,
        "inc_at_write_addr length must match has_write"
    );

    // inc_at_r_addr[t] = has_write[t] * eq(wa_bits(t), r_addr) * inc_at_write_addr[t]
    let inc_at_r_addr =
        crate::twist_oracle::build_inc_at_r_addr_sparse(wa_bits, has_write, inc_at_write_addr, r_addr, pow2_cycle);

    // Prefix-sum (pre-write): val[t] = init + Σ_{u < t} inc[u]
    let mut out = vec![KElem::ZERO; pow2_cycle];
    let mut acc = init_at_r_addr;
    for t in 0..pow2_cycle {
        out[t] = acc;
        acc += inc_at_r_addr[t];
    }
    out
}

/// Decode committed Twist columns into K-extension vectors padded to 2^ell_cycle.
pub fn decode_twist_cols<Cmt: Clone>(
    params: &NeoParams,
    inst: &MemInstance<Cmt, BaseField>,
    wit: &MemWitness<BaseField>,
    ell_cycle: usize,
) -> Result<TwistDecodedCols, PiCcsError> {
    validate_index_bit_addressing(inst)?;

    #[cfg(debug_assertions)]
    check_twist_semantics(params, inst, wit)?;

    let ell_addr = inst.d * inst.ell;
    let pow2_cycle = 1usize << ell_cycle;
    if inst.steps > pow2_cycle {
        return Err(PiCcsError::InvalidInput(format!(
            "Twist(Route A): steps={} exceeds 2^ell_cycle={pow2_cycle}",
            inst.steps
        )));
    }

    let pow2_addr = 1usize
        .checked_shl(ell_addr as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): 2^ell_addr overflow".into()))?;
    if pow2_addr != inst.k || pow2_addr != inst.init_vals.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "Twist(Route A): expected 2^(d*ell) == k == init_vals.len(), got 2^(d*ell)={pow2_addr}, k={}, init_vals.len()={}",
            inst.k,
            inst.init_vals.len()
        )));
    }

    let parts = split_mem_mats(inst, wit);

    // Decode committed columns (time-domain vectors, padded to pow2_cycle).
    let ra_bits = ts::decode_mats_to_k_padded(params, parts.ra_bit_mats, pow2_cycle);
    let wa_bits = ts::decode_mats_to_k_padded(params, parts.wa_bit_mats, pow2_cycle);
    let has_read = ts::decode_mat_to_k_padded(params, parts.has_read_mat, pow2_cycle);
    let has_write = ts::decode_mat_to_k_padded(params, parts.has_write_mat, pow2_cycle);
    let wv = ts::decode_mat_to_k_padded(params, parts.wv_mat, pow2_cycle);
    let rv = ts::decode_mat_to_k_padded(params, parts.rv_mat, pow2_cycle);
    let inc_at_write_addr = ts::decode_mat_to_k_padded(params, parts.inc_at_write_addr_mat, pow2_cycle);

    Ok(TwistDecodedCols {
        ra_bits,
        wa_bits,
        has_read,
        has_write,
        wv,
        rv,
        inc_at_write_addr,
    })
}

// ============================================================================
// Route A oracles v3 (Phase 2): addr-pre + time-only checks (no time×addr table).
// ============================================================================

pub struct RouteATwistOraclesV3 {
    pub read_check: TwistReadCheckOracle,
    pub read_check_claim_sum: KElem,
    pub write_check: TwistWriteCheckOracle,
    pub write_check_claim_sum: KElem,
    pub bitness: Vec<LazyBitnessOracle>,
    pub ell_addr: usize,
}

pub struct RouteATwistAddrOraclesV3 {
    pub read_addr: TwistReadCheckAddrOracle,
    pub write_addr: TwistWriteCheckAddrOracle,
    pub ell_addr: usize,
}

pub fn build_route_a_twist_addr_oracles_v3<Cmt: Clone>(
    inst: &MemInstance<Cmt, BaseField>,
    decoded: &TwistDecodedCols,
    r_cycle: &[KElem],
) -> Result<RouteATwistAddrOraclesV3, PiCcsError> {
    validate_index_bit_addressing(inst)?;
    let ell_addr = inst.d * inst.ell;
    let pow2_addr = 1usize
        .checked_shl(ell_addr as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): 2^ell_addr overflow".into()))?;

    let expected_pow2_time = 1usize << r_cycle.len();
    if decoded.has_read.len() != expected_pow2_time
        || decoded.has_write.len() != expected_pow2_time
        || decoded.rv.len() != expected_pow2_time
        || decoded.wv.len() != expected_pow2_time
        || decoded.inc_at_write_addr.len() != expected_pow2_time
    {
        return Err(PiCcsError::InvalidInput(format!(
            "Twist(Route A): decoded column length mismatch with r_cycle (expected {}, got has_read={}, has_write={}, rv={}, wv={}, inc_at_write_addr={})",
            expected_pow2_time,
            decoded.has_read.len(),
            decoded.has_write.len(),
            decoded.rv.len(),
            decoded.wv.len(),
            decoded.inc_at_write_addr.len()
        )));
    }

    let init_table_k: Vec<KElem> = inst.init_vals.iter().map(|&v| v.into()).collect();
    if init_table_k.len() != pow2_addr {
        return Err(PiCcsError::InvalidInput(format!(
            "Twist(Route A): init table length mismatch (got {}, expected pow2_addr={})",
            init_table_k.len(),
            pow2_addr
        )));
    }

    let read_addr = TwistReadCheckAddrOracle::new(
        init_table_k.clone(),
        r_cycle,
        decoded.has_read.clone(),
        decoded.rv.clone(),
        &decoded.ra_bits,
        decoded.has_write.clone(),
        &decoded.wa_bits,
        decoded.inc_at_write_addr.clone(),
    );

    let write_addr = TwistWriteCheckAddrOracle::new(
        init_table_k,
        r_cycle,
        decoded.has_write.clone(),
        decoded.wv.clone(),
        &decoded.wa_bits,
        decoded.inc_at_write_addr.clone(),
    );

    Ok(RouteATwistAddrOraclesV3 {
        read_addr,
        write_addr,
        ell_addr,
    })
}

pub fn build_route_a_twist_oracles_v3<Cmt: Clone>(
    inst: &MemInstance<Cmt, BaseField>,
    decoded: &TwistDecodedCols,
    r_cycle: &[KElem],
    r_addr: &[KElem],
    read_check_claim_sum: KElem,
    write_check_claim_sum: KElem,
) -> Result<RouteATwistOraclesV3, PiCcsError> {
    validate_index_bit_addressing(inst)?;
    let ell_addr = inst.d * inst.ell;
    if r_addr.len() != ell_addr {
        return Err(PiCcsError::InvalidInput(format!(
            "Twist(Route A): r_addr.len()={}, expected ell_addr={}",
            r_addr.len(),
            ell_addr
        )));
    }

    let expected_pow2_time = 1usize << r_cycle.len();
    if decoded.has_read.len() != expected_pow2_time
        || decoded.has_write.len() != expected_pow2_time
        || decoded.rv.len() != expected_pow2_time
        || decoded.wv.len() != expected_pow2_time
        || decoded.inc_at_write_addr.len() != expected_pow2_time
    {
        return Err(PiCcsError::InvalidInput(format!(
            "Twist(Route A): decoded column length mismatch with r_cycle (expected {}, got has_read={}, has_write={}, rv={}, wv={}, inc_at_write_addr={})",
            expected_pow2_time,
            decoded.has_read.len(),
            decoded.has_write.len(),
            decoded.rv.len(),
            decoded.wv.len(),
            decoded.inc_at_write_addr.len()
        )));
    }

    // Compute Val_pre(r_addr, t) for all boolean time indices t.
    let init_table_k: Vec<KElem> = inst.init_vals.iter().map(|&v| v.into()).collect();
    let init_at_r_addr = table_mle_eval(&init_table_k, r_addr);
    let eq_wa = compute_eq_from_bits(&decoded.wa_bits, r_addr);
    let mut cur = init_at_r_addr;
    let mut val_pre_at_r_addr: Vec<KElem> = Vec::with_capacity(expected_pow2_time);
    for t in 0..expected_pow2_time {
        val_pre_at_r_addr.push(cur);
        cur += decoded.has_write[t] * decoded.inc_at_write_addr[t] * eq_wa[t];
    }

    let read_check = TwistReadCheckOracle::new(
        &decoded.ra_bits,
        val_pre_at_r_addr.clone(),
        decoded.rv.clone(),
        decoded.has_read.clone(),
        r_cycle,
        r_addr,
    );
    let write_check = TwistWriteCheckOracle::new(
        &decoded.wa_bits,
        decoded.wv.clone(),
        val_pre_at_r_addr,
        decoded.inc_at_write_addr.clone(),
        decoded.has_write.clone(),
        r_cycle,
        r_addr,
    );

    // Bitness for ra_bits + wa_bits + has_read + has_write, χ_{r_cycle}-weighted.
    let mut bitness = Vec::with_capacity(2 * ell_addr + 2);
    for bits in decoded
        .ra_bits
        .iter()
        .cloned()
        .chain(decoded.wa_bits.iter().cloned())
    {
        bitness.push(LazyBitnessOracle::new_with_cycle(r_cycle, bits));
    }
    bitness.push(LazyBitnessOracle::new_with_cycle(r_cycle, decoded.has_read.clone()));
    bitness.push(LazyBitnessOracle::new_with_cycle(r_cycle, decoded.has_write.clone()));

    Ok(RouteATwistOraclesV3 {
        read_check,
        read_check_claim_sum,
        write_check,
        write_check_claim_sum,
        bitness,
        ell_addr,
    })
}
