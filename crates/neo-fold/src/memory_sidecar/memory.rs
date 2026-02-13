use crate::memory_sidecar::claim_plan::RouteATimeClaimPlan;
use crate::memory_sidecar::shout_paging::plan_shout_addr_pages;
use crate::memory_sidecar::sumcheck_ds::{run_batched_sumcheck_prover_ds, verify_batched_sumcheck_rounds_ds};
use crate::memory_sidecar::transcript::{bind_batched_claim_sums, bind_twist_val_eval_claim_sums, digest_fields};
use crate::memory_sidecar::utils::{bitness_weights, RoundOraclePrefix};
use crate::shard_proof_types::{
    MemOrLutProof, MemSidecarProof, ShoutAddrPreGroupProof, ShoutAddrPreProof, ShoutProofK, TwistProofK,
};
use crate::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, MeInstance};
use neo_math::{F, K};
use neo_memory::bit_ops::{eq_bit_affine, eq_bits_prod};
use neo_memory::cpu::{build_bus_layout_for_instances_with_shout_and_twist_lanes, BusLayout};
use neo_memory::identity::shout_oracle::IdentityAddressLookupOracleSparse;
use neo_memory::mle::{eq_points, lt_eval};
use neo_memory::riscv::lookups::{PROG_ID, RAM_ID, REG_ID};
use neo_memory::riscv::shout_oracle::RiscvAddressLookupOracleSparse;
use neo_memory::riscv::trace::Rv32TraceLayout;
use neo_memory::sparse_time::SparseIdxVec;
use neo_memory::ts_common as ts;
use neo_memory::twist_oracle::{
    AddressLookupOracle, IndexAdapterOracleSparseTime, LazyWeightedBitnessOracleSparseTime,
    Rv32PackedAddOracleSparseTime, Rv32PackedAndOracleSparseTime, Rv32PackedAndnOracleSparseTime,
    Rv32PackedBitwiseAdapterOracleSparseTime, Rv32PackedDivOracleSparseTime, Rv32PackedDivRemAdapterOracleSparseTime,
    Rv32PackedDivRemuAdapterOracleSparseTime, Rv32PackedDivuOracleSparseTime, Rv32PackedEqAdapterOracleSparseTime,
    Rv32PackedEqOracleSparseTime, Rv32PackedMulHiOracleSparseTime, Rv32PackedMulOracleSparseTime,
    Rv32PackedMulhAdapterOracleSparseTime, Rv32PackedMulhsuAdapterOracleSparseTime, Rv32PackedMulhuOracleSparseTime,
    Rv32PackedNeqAdapterOracleSparseTime, Rv32PackedNeqOracleSparseTime, Rv32PackedOrOracleSparseTime,
    Rv32PackedRemOracleSparseTime, Rv32PackedRemuOracleSparseTime, Rv32PackedSllOracleSparseTime,
    Rv32PackedSltOracleSparseTime, Rv32PackedSltuOracleSparseTime, Rv32PackedSraAdapterOracleSparseTime,
    Rv32PackedSraOracleSparseTime, Rv32PackedSrlAdapterOracleSparseTime, Rv32PackedSrlOracleSparseTime,
    Rv32PackedSubOracleSparseTime, Rv32PackedXorOracleSparseTime, ShoutValueOracleSparse, TwistLaneSparseCols,
    TwistReadCheckAddrOracleSparseTimeMultiLane, TwistReadCheckOracleSparseTime, TwistTotalIncOracleSparseTime,
    TwistValEvalOracleSparseTime, TwistWriteCheckAddrOracleSparseTimeMultiLane, TwistWriteCheckOracleSparseTime,
    U32DecompOracleSparseTime, ZeroOracleSparseTime,
};
use neo_memory::witness::{LutInstance, LutTableSpec, MemInstance, StepInstanceBundle, StepWitnessBundle};
use neo_memory::{eval_init_at_r_addr, twist, BatchedAddrProof, MemInit};
use neo_params::NeoParams;
use neo_reductions::sumcheck::{BatchedClaim, RoundOracle};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::Field;
use p3_field::PrimeCharacteristicRing;

// ============================================================================
// Transcript binding
// ============================================================================

fn bind_shout_table_spec(tr: &mut Poseidon2Transcript, spec: &Option<LutTableSpec>) {
    let Some(spec) = spec else {
        return;
    };

    tr.append_message(b"shout/table_spec/tag", &[1u8]);
    match spec {
        LutTableSpec::RiscvOpcode { opcode, xlen } => {
            let opcode_id = neo_memory::riscv::lookups::RiscvShoutTables::new(*xlen)
                .opcode_to_id(*opcode)
                .0 as u64;

            tr.append_message(b"shout/table_spec/riscv/tag", &[1u8]);
            tr.append_message(b"shout/table_spec/riscv/opcode_id", &opcode_id.to_le_bytes());
            tr.append_message(b"shout/table_spec/riscv/xlen", &(*xlen as u64).to_le_bytes());
        }
        LutTableSpec::RiscvOpcodePacked { opcode, xlen } => {
            let opcode_id = neo_memory::riscv::lookups::RiscvShoutTables::new(*xlen)
                .opcode_to_id(*opcode)
                .0 as u64;

            tr.append_message(b"shout/table_spec/riscv_packed/tag", &[1u8]);
            tr.append_message(b"shout/table_spec/riscv_packed/opcode_id", &opcode_id.to_le_bytes());
            tr.append_message(b"shout/table_spec/riscv_packed/xlen", &(*xlen as u64).to_le_bytes());
        }
        LutTableSpec::RiscvOpcodeEventTablePacked {
            opcode,
            xlen,
            time_bits,
        } => {
            let opcode_id = neo_memory::riscv::lookups::RiscvShoutTables::new(*xlen)
                .opcode_to_id(*opcode)
                .0 as u64;

            tr.append_message(b"shout/table_spec/riscv_event_table_packed/tag", &[1u8]);
            tr.append_message(
                b"shout/table_spec/riscv_event_table_packed/opcode_id",
                &opcode_id.to_le_bytes(),
            );
            tr.append_message(
                b"shout/table_spec/riscv_event_table_packed/xlen",
                &(*xlen as u64).to_le_bytes(),
            );
            tr.append_message(
                b"shout/table_spec/riscv_event_table_packed/time_bits",
                &(*time_bits as u64).to_le_bytes(),
            );
        }
        LutTableSpec::IdentityU32 => {
            tr.append_message(b"shout/table_spec/identity_u32/tag", &[1u8]);
        }
    }
}

fn absorb_step_memory_impl<'a, LI, MI>(tr: &mut Poseidon2Transcript, mut lut_insts: LI, mut mem_insts: MI)
where
    LI: ExactSizeIterator<Item = &'a LutInstance<Cmt, F>>,
    MI: ExactSizeIterator<Item = &'a MemInstance<Cmt, F>>,
{
    tr.append_message(b"step/absorb_memory_start", &[]);
    tr.append_message(b"step/lut_count", &(lut_insts.len() as u64).to_le_bytes());
    for (i, inst) in lut_insts.by_ref().enumerate() {
        // Bind public LUT parameters before any challenges.
        tr.append_message(b"step/lut_idx", &(i as u64).to_le_bytes());
        tr.append_message(b"shout/k", &(inst.k as u64).to_le_bytes());
        tr.append_message(b"shout/d", &(inst.d as u64).to_le_bytes());
        tr.append_message(b"shout/n_side", &(inst.n_side as u64).to_le_bytes());
        tr.append_message(b"shout/steps", &(inst.steps as u64).to_le_bytes());
        tr.append_message(b"shout/ell", &(inst.ell as u64).to_le_bytes());
        tr.append_message(b"shout/lanes", &(inst.lanes.max(1) as u64).to_le_bytes());
        bind_shout_table_spec(tr, &inst.table_spec);
        let table_digest = digest_fields(b"shout/table", &inst.table);
        tr.append_message(b"shout/table_digest", &table_digest);

        // Bind commitments so Route-A challenges (r_cycle, addr/time points) are sampled after them.
        tr.append_message(b"shout/comms_len", &(inst.comms.len() as u64).to_le_bytes());
        for (j, comm) in inst.comms.iter().enumerate() {
            tr.append_message(b"shout/comm_idx", &(j as u64).to_le_bytes());
            tr.append_fields(b"shout/comm_data", &comm.data);
        }
    }
    tr.append_message(b"step/mem_count", &(mem_insts.len() as u64).to_le_bytes());
    for (i, inst) in mem_insts.by_ref().enumerate() {
        // Bind public memory parameters before any challenges.
        tr.append_message(b"step/mem_idx", &(i as u64).to_le_bytes());
        tr.append_message(b"twist/mem_id", &(inst.mem_id as u64).to_le_bytes());
        tr.append_message(b"twist/k", &(inst.k as u64).to_le_bytes());
        tr.append_message(b"twist/d", &(inst.d as u64).to_le_bytes());
        tr.append_message(b"twist/n_side", &(inst.n_side as u64).to_le_bytes());
        tr.append_message(b"twist/steps", &(inst.steps as u64).to_le_bytes());
        tr.append_message(b"twist/ell", &(inst.ell as u64).to_le_bytes());
        tr.append_message(b"twist/lanes", &(inst.lanes.max(1) as u64).to_le_bytes());
        let init_digest = match &inst.init {
            MemInit::Zero => digest_fields(b"twist/init/zero", &[]),
            MemInit::Sparse(pairs) => {
                let mut fs = Vec::with_capacity(2 * pairs.len());
                for (addr, val) in pairs.iter() {
                    fs.push(F::from_u64(*addr));
                    fs.push(*val);
                }
                digest_fields(b"twist/init/sparse", &fs)
            }
        };
        tr.append_message(b"twist/init_digest", &init_digest);

        // Bind commitments so Route-A challenges (r_cycle, addr/time points) are sampled after them.
        tr.append_message(b"twist/comms_len", &(inst.comms.len() as u64).to_le_bytes());
        for (j, comm) in inst.comms.iter().enumerate() {
            tr.append_message(b"twist/comm_idx", &(j as u64).to_le_bytes());
            tr.append_fields(b"twist/comm_data", &comm.data);
        }
    }
    tr.append_message(b"step/absorb_memory_done", &[]);
}

pub fn absorb_step_memory(tr: &mut Poseidon2Transcript, step: &StepInstanceBundle<Cmt, F, K>) {
    absorb_step_memory_impl(tr, step.lut_insts.iter(), step.mem_insts.iter());
}

pub(crate) fn absorb_step_memory_witness(tr: &mut Poseidon2Transcript, step: &StepWitnessBundle<Cmt, F, K>) {
    absorb_step_memory_impl(
        tr,
        step.lut_instances.iter().map(|(inst, _)| inst),
        step.mem_instances.iter().map(|(inst, _)| inst),
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rv32PackedShoutOp {
    And,
    Andn,
    Add,
    Or,
    Sub,
    Xor,
    Eq,
    Neq,
    Slt,
    Sll,
    Srl,
    Sra,
    Sltu,
    Mul,
    Mulh,
    Mulhu,
    Mulhsu,
    Div,
    Divu,
    Rem,
    Remu,
}

fn rv32_packed_shout_layout(spec: &Option<LutTableSpec>) -> Result<Option<(Rv32PackedShoutOp, usize)>, PiCcsError> {
    let (opcode, xlen, time_bits) = match spec {
        Some(LutTableSpec::RiscvOpcodePacked { opcode, xlen }) => (*opcode, *xlen, 0usize),
        Some(LutTableSpec::RiscvOpcodeEventTablePacked {
            opcode,
            xlen,
            time_bits,
        }) => (*opcode, *xlen, *time_bits),
        _ => return Ok(None),
    };

    if xlen != 32 {
        return Err(PiCcsError::InvalidInput(format!(
            "packed RISC-V Shout is only supported for RV32 (xlen=32) in Route A (got xlen={xlen})"
        )));
    }
    if time_bits == 0 {
        // `RiscvOpcodePacked` uses `time_bits=0` (no prefix). Event-table packed must be >= 1.
        if matches!(spec, Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. })) {
            return Err(PiCcsError::InvalidInput(
                "RiscvOpcodeEventTablePacked requires time_bits >= 1".into(),
            ));
        }
    }

    let op = match opcode {
        neo_memory::riscv::lookups::RiscvOpcode::And => Rv32PackedShoutOp::And,
        neo_memory::riscv::lookups::RiscvOpcode::Andn => Rv32PackedShoutOp::Andn,
        neo_memory::riscv::lookups::RiscvOpcode::Add => Rv32PackedShoutOp::Add,
        neo_memory::riscv::lookups::RiscvOpcode::Or => Rv32PackedShoutOp::Or,
        neo_memory::riscv::lookups::RiscvOpcode::Sub => Rv32PackedShoutOp::Sub,
        neo_memory::riscv::lookups::RiscvOpcode::Xor => Rv32PackedShoutOp::Xor,
        neo_memory::riscv::lookups::RiscvOpcode::Eq => Rv32PackedShoutOp::Eq,
        neo_memory::riscv::lookups::RiscvOpcode::Neq => Rv32PackedShoutOp::Neq,
        neo_memory::riscv::lookups::RiscvOpcode::Slt => Rv32PackedShoutOp::Slt,
        neo_memory::riscv::lookups::RiscvOpcode::Sll => Rv32PackedShoutOp::Sll,
        neo_memory::riscv::lookups::RiscvOpcode::Srl => Rv32PackedShoutOp::Srl,
        neo_memory::riscv::lookups::RiscvOpcode::Sra => Rv32PackedShoutOp::Sra,
        neo_memory::riscv::lookups::RiscvOpcode::Sltu => Rv32PackedShoutOp::Sltu,
        neo_memory::riscv::lookups::RiscvOpcode::Mul => Rv32PackedShoutOp::Mul,
        neo_memory::riscv::lookups::RiscvOpcode::Mulh => Rv32PackedShoutOp::Mulh,
        neo_memory::riscv::lookups::RiscvOpcode::Mulhu => Rv32PackedShoutOp::Mulhu,
        neo_memory::riscv::lookups::RiscvOpcode::Mulhsu => Rv32PackedShoutOp::Mulhsu,
        neo_memory::riscv::lookups::RiscvOpcode::Div => Rv32PackedShoutOp::Div,
        neo_memory::riscv::lookups::RiscvOpcode::Divu => Rv32PackedShoutOp::Divu,
        neo_memory::riscv::lookups::RiscvOpcode::Rem => Rv32PackedShoutOp::Rem,
        neo_memory::riscv::lookups::RiscvOpcode::Remu => Rv32PackedShoutOp::Remu,
        _ => {
            return Err(PiCcsError::InvalidInput(format!(
                "packed RISC-V Shout is only supported for selected RV32 ops in Route A (got opcode={opcode:?})"
            )));
        }
    };

    Ok(Some((op, time_bits)))
}

fn rv32_shout_table_id_from_spec(spec: &Option<LutTableSpec>) -> Result<u32, PiCcsError> {
    let (opcode, xlen) = match spec {
        Some(LutTableSpec::RiscvOpcode { opcode, xlen }) => (*opcode, *xlen),
        Some(LutTableSpec::RiscvOpcodePacked { opcode, xlen }) => (*opcode, *xlen),
        Some(LutTableSpec::RiscvOpcodeEventTablePacked { opcode, xlen, .. }) => (*opcode, *xlen),
        Some(LutTableSpec::IdentityU32) => {
            return Err(PiCcsError::InvalidInput(
                "trace linkage expects RISC-V shout table specs (IdentityU32 is unsupported)".into(),
            ));
        }
        None => {
            return Err(PiCcsError::InvalidInput(
                "trace linkage requires LutTableSpec on no-shared-bus shout instances".into(),
            ));
        }
    };

    if xlen != 32 {
        return Err(PiCcsError::InvalidInput(format!(
            "trace linkage expects RV32 shout specs (got xlen={xlen})"
        )));
    }
    Ok(neo_memory::riscv::lookups::RiscvShoutTables::new(xlen)
        .opcode_to_id(opcode)
        .0)
}

// ============================================================================
// Prover helpers
// ============================================================================

pub(crate) struct ShoutDecodedColsSparse {
    pub lanes: Vec<ShoutLaneSparseCols>,
}

pub(crate) struct ShoutLaneSparseCols {
    pub addr_bits: Vec<SparseIdxVec<K>>,
    pub has_lookup: SparseIdxVec<K>,
    pub val: SparseIdxVec<K>,
}

pub(crate) struct TwistDecodedColsSparse {
    pub lanes: Vec<TwistLaneSparseCols>,
}

pub(crate) struct SumRoundOracle {
    oracles: Vec<Box<dyn RoundOracle>>,
    num_rounds: usize,
    degree_bound: usize,
}

impl SumRoundOracle {
    pub(crate) fn new(oracles: Vec<Box<dyn RoundOracle>>) -> Self {
        if oracles.is_empty() {
            panic!("SumRoundOracle requires at least one oracle");
        }

        let num_rounds = oracles[0].num_rounds();
        let degree_bound = oracles[0].degree_bound();
        for (idx, o) in oracles.iter().enumerate().skip(1) {
            if o.num_rounds() != num_rounds {
                panic!(
                    "SumRoundOracle num_rounds mismatch at idx={idx} (got {}, expected {num_rounds})",
                    o.num_rounds()
                );
            }
            if o.degree_bound() != degree_bound {
                panic!(
                    "SumRoundOracle degree_bound mismatch at idx={idx} (got {}, expected {degree_bound})",
                    o.degree_bound()
                );
            }
        }

        Self {
            oracles,
            num_rounds,
            degree_bound,
        }
    }
}

impl RoundOracle for SumRoundOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let mut acc = vec![K::ZERO; points.len()];
        for o in self.oracles.iter_mut() {
            let ys = o.evals_at(points);
            if ys.len() != acc.len() {
                panic!(
                    "SumRoundOracle eval length mismatch (got {}, expected {})",
                    ys.len(),
                    acc.len()
                );
            }
            for (a, y) in acc.iter_mut().zip(ys) {
                *a += y;
            }
        }
        acc
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        for o in self.oracles.iter_mut() {
            o.fold(r);
        }
        self.num_rounds = self.oracles[0].num_rounds();
    }
}

#[inline]
fn interp(a0: K, a1: K, x: K) -> K {
    a0 + (a1 - a0) * x
}

fn log2_pow2(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    debug_assert!(n.is_power_of_two(), "expected power of two, got {n}");
    n.trailing_zeros() as usize
}

fn gather_pairs_from_sparse(entries: &[(usize, K)]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(entries.len());
    let mut prev: Option<usize> = None;
    for &(idx, _v) in entries {
        let p = idx >> 1;
        if prev != Some(p) {
            out.push(p);
            prev = Some(p);
        }
    }
    out
}

/// Sparse time-domain oracle for event-table RV32 Shout hash linkage:
///   Σ_t has_lookup(t) · (1 + α·val(t) + β·lhs(t) + γ·rhs(t)) · Π_b eq(time_bit_b(t), r_addr_b)
///
/// Intended usage:
/// - `time_bit_b(t)` encodes the original cycle index of event row `t` (little-endian).
/// - `r_addr` is set to `r_cycle` so the claim is an MLE evaluation over cycle indices.
struct ShoutEventTableHashOracleSparseTime {
    degree_bound: usize,
    r_addr: Vec<K>,

    time_bits: Vec<SparseIdxVec<K>>,
    has_lookup: SparseIdxVec<K>,
    val: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs_terms: Vec<(SparseIdxVec<K>, K)>,

    alpha: K,
    beta: K,
    gamma: K,
}

impl ShoutEventTableHashOracleSparseTime {
    fn new(
        r_addr: &[K],
        time_bits: Vec<SparseIdxVec<K>>,
        has_lookup: SparseIdxVec<K>,
        val: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs_terms: Vec<(SparseIdxVec<K>, K)>,
        alpha: K,
        beta: K,
        gamma: K,
    ) -> (Self, K) {
        let ell_n = log2_pow2(has_lookup.len());
        debug_assert_eq!(val.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        for (i, col) in time_bits.iter().enumerate() {
            debug_assert_eq!(col.len(), 1usize << ell_n, "time_bits[{i}] length mismatch");
        }
        for (i, (col, _w)) in rhs_terms.iter().enumerate() {
            debug_assert_eq!(col.len(), 1usize << ell_n, "rhs_terms[{i}] length mismatch");
        }
        debug_assert_eq!(time_bits.len(), r_addr.len(), "time_bits/r_addr length mismatch");

        let mut claim = K::ZERO;
        for &(t, gate) in has_lookup.entries() {
            if gate == K::ZERO {
                continue;
            }

            let v_t = val.get(t);
            let lhs_t = lhs.get(t);
            let mut rhs_t = K::ZERO;
            for (col, w) in rhs_terms.iter() {
                rhs_t += *w * col.get(t);
            }

            let hash_t = K::ONE + alpha * v_t + beta * lhs_t + gamma * rhs_t;
            if hash_t == K::ZERO {
                continue;
            }

            let mut eq_addr = K::ONE;
            for (b, col) in time_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.get(t), r_addr[b]);
            }

            claim += gate * hash_t * eq_addr;
        }

        (
            Self {
                degree_bound: 2 + r_addr.len(),
                r_addr: r_addr.to_vec(),
                time_bits,
                has_lookup,
                val,
                lhs,
                rhs_terms,
                alpha,
                beta,
                gamma,
            },
            claim,
        )
    }
}

impl RoundOracle for ShoutEventTableHashOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let v = self.val.singleton_value();
            let lhs = self.lhs.singleton_value();
            let mut rhs = K::ZERO;
            for (col, w) in self.rhs_terms.iter() {
                rhs += *w * col.singleton_value();
            }
            let hash = gate * (K::ONE + self.alpha * v + self.beta * lhs + self.gamma * rhs);

            let mut eq_addr = K::ONE;
            for (b, col) in self.time_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.singleton_value(), self.r_addr[b]);
            }

            let out = hash * eq_addr;
            return vec![out; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let v0 = self.val.get(child0);
            let v1 = self.val.get(child1);
            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);

            let mut rhs0 = K::ZERO;
            let mut rhs1 = K::ZERO;
            for (col, w) in self.rhs_terms.iter() {
                rhs0 += *w * col.get(child0);
                rhs1 += *w * col.get(child1);
            }

            let mut eq0s: Vec<K> = Vec::with_capacity(self.time_bits.len());
            let mut d_eqs: Vec<K> = Vec::with_capacity(self.time_bits.len());
            for (b, col) in self.time_bits.iter().enumerate() {
                let e0 = eq_bit_affine(col.get(child0), self.r_addr[b]);
                let e1 = eq_bit_affine(col.get(child1), self.r_addr[b]);
                eq0s.push(e0);
                d_eqs.push(e1 - e0);
            }

            for (i, &x) in points.iter().enumerate() {
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }
                let v_x = interp(v0, v1, x);
                let lhs_x = interp(lhs0, lhs1, x);
                let rhs_x = interp(rhs0, rhs1, x);

                let mut prod = gate_x * (K::ONE + self.alpha * v_x + self.beta * lhs_x + self.gamma * rhs_x);
                for (e0, de) in eq0s.iter().zip(d_eqs.iter()) {
                    prod *= *e0 + *de * x;
                }
                ys[i] += prod;
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        log2_pow2(self.has_lookup.len())
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.has_lookup.fold_round_in_place(r);
        self.val.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        for (col, _w) in self.rhs_terms.iter_mut() {
            col.fold_round_in_place(r);
        }
        for col in self.time_bits.iter_mut() {
            col.fold_round_in_place(r);
        }
    }
}

fn build_twist_inc_terms_at_r_addr(lanes: &[TwistLaneSparseCols], r_addr: &[K]) -> Vec<(usize, K)> {
    let ell_addr = r_addr.len();
    let mut out: Vec<(usize, K)> = Vec::new();

    for lane in lanes.iter() {
        debug_assert_eq!(lane.wa_bits.len(), ell_addr, "wa_bits len mismatch");
        for &(t, has_w) in lane.has_write.entries() {
            let inc_t = lane.inc_at_write_addr.get(t);
            if has_w == K::ZERO || inc_t == K::ZERO {
                continue;
            }

            let mut eq_addr = K::ONE;
            for (b, col) in lane.wa_bits.iter().enumerate() {
                let bit = col.get(t);
                eq_addr *= eq_bit_affine(bit, r_addr[b]);
            }

            let inc_at_r_addr = has_w * inc_t * eq_addr;
            if inc_at_r_addr != K::ZERO {
                out.push((t, inc_at_r_addr));
            }
        }
    }

    out
}

pub struct RouteAShoutTimeOracles {
    pub lanes: Vec<RouteAShoutTimeLaneOracles>,
    pub bitness: Vec<Box<dyn RoundOracle>>,
    pub ell_addr: usize,
}

pub struct RouteAShoutTimeLaneOracles {
    pub value: Box<dyn RoundOracle>,
    pub value_claim: K,
    pub adapter: Box<dyn RoundOracle>,
    pub adapter_claim: K,
    pub event_table_hash: Option<Box<dyn RoundOracle>>,
    pub event_table_hash_claim: Option<K>,
}

pub struct RouteATwistTimeOracles {
    pub read_check: Box<dyn RoundOracle>,
    pub write_check: Box<dyn RoundOracle>,
    pub bitness: Vec<Box<dyn RoundOracle>>,
    pub ell_addr: usize,
}

pub struct RouteAMemoryOracles {
    pub shout: Vec<RouteAShoutTimeOracles>,
    pub shout_event_trace_hash: Option<RouteAShoutEventTraceHashOracle>,
    pub twist: Vec<RouteATwistTimeOracles>,
}

pub struct RouteAShoutEventTraceHashOracle {
    pub oracle: Box<dyn RoundOracle>,
    pub claim: K,
}

pub trait TimeBatchedClaims {
    fn append_time_claims<'a>(
        &'a mut self,
        ell_n: usize,
        claimed_sums: &mut Vec<K>,
        degree_bounds: &mut Vec<usize>,
        labels: &mut Vec<&'static [u8]>,
        claim_is_dynamic: &mut Vec<bool>,
        claims: &mut Vec<BatchedClaim<'a>>,
    );
}

pub(crate) struct ShoutAddrPreBatchProverData {
    pub addr_pre: ShoutAddrPreProof<K>,
    pub decoded: Vec<ShoutDecodedColsSparse>,
}

pub struct ShoutAddrPreVerifyData {
    pub is_active: bool,
    pub addr_claim_sum: K,
    pub addr_final: K,
    pub r_addr: Vec<K>,
    pub table_eval_at_r_addr: K,
}

pub(crate) struct TwistAddrPreProverData {
    pub addr_pre: BatchedAddrProof<K>,
    pub decoded: TwistDecodedColsSparse,
    /// Time-lane claimed sum for the read-check oracle (output of addr-pre).
    pub read_check_claim_sum: K,
    /// Time-lane claimed sum for the write-check oracle (output of addr-pre).
    pub write_check_claim_sum: K,
}

pub struct TwistAddrPreVerifyData {
    pub r_addr: Vec<K>,
    pub read_check_claim_sum: K,
    pub write_check_claim_sum: K,
}

#[derive(Clone, Debug)]
pub struct TwistTimeLaneOpeningsLane {
    pub wa_bits: Vec<K>,
    pub has_write: K,
    pub inc_at_write_addr: K,
}

#[derive(Clone, Debug)]
pub struct TwistTimeLaneOpenings {
    pub lanes: Vec<TwistTimeLaneOpeningsLane>,
}

#[derive(Clone, Debug)]
pub struct RouteAMemoryVerifyOutput {
    pub claim_idx_end: usize,
    pub twist_time_openings: Vec<TwistTimeLaneOpenings>,
}

#[derive(Clone, Copy)]
struct TraceCpuLinkOpenings {
    active: K,
    prog_addr: K,
    prog_value: K,
    rs1_addr: K,
    rs1_val: K,
    rs2_addr: K,
    rs2_val: K,
    rd_has_write: K,
    rd_addr: K,
    rd_val: K,
    ram_has_read: K,
    ram_has_write: K,
    ram_addr: K,
    ram_rv: K,
    ram_wv: K,
    shout_has_lookup: K,
    shout_val: K,
    shout_lhs: K,
    shout_rhs: K,
    shout_table_id: K,
}

#[inline]
fn pack_bits_lsb(bits: &[K]) -> K {
    let two = K::from(F::from_u64(2));
    let mut pow = K::ONE;
    let mut acc = K::ZERO;
    for &b in bits {
        acc += pow * b;
        pow *= two;
    }
    acc
}

#[inline]
fn unpack_interleaved_halves_lsb(addr_bits: &[K]) -> Result<(K, K), PiCcsError> {
    if !addr_bits.len().is_multiple_of(2) {
        return Err(PiCcsError::InvalidInput(format!(
            "shout linkage expects even ell_addr, got {}",
            addr_bits.len()
        )));
    }
    let half_len = addr_bits.len() / 2;
    let two = K::from(F::from_u64(2));
    let mut pow = K::ONE;
    let mut lhs = K::ZERO;
    let mut rhs = K::ZERO;
    for k in 0..half_len {
        lhs += pow * addr_bits[2 * k];
        rhs += pow * addr_bits[2 * k + 1];
        pow *= two;
    }
    Ok((lhs, rhs))
}

fn extract_trace_cpu_link_openings(
    m: usize,
    core_t: usize,
    step: &StepInstanceBundle<Cmt, F, K>,
    ccs_out0: &MeInstance<Cmt, F, K>,
) -> Result<Option<TraceCpuLinkOpenings>, PiCcsError> {
    if step.mem_insts.is_empty() && step.lut_insts.is_empty() {
        return Ok(None);
    }

    // RV32 trace linkage: the prover appends time-combined openings for selected CPU trace columns
    // to the CCS ME output at r_time. We use those to bind Twist instances (PROG/REG/RAM) to the
    // same trace, without embedding a shared CPU bus tail.
    let trace = Rv32TraceLayout::new();
    let trace_cols_to_open: Vec<usize> = vec![
        trace.active,
        trace.prog_addr,
        trace.prog_value,
        trace.rs1_addr,
        trace.rs1_val,
        trace.rs2_addr,
        trace.rs2_val,
        trace.rd_has_write,
        trace.rd_addr,
        trace.rd_val,
        trace.ram_has_read,
        trace.ram_has_write,
        trace.ram_addr,
        trace.ram_rv,
        trace.ram_wv,
        trace.shout_has_lookup,
        trace.shout_val,
        trace.shout_lhs,
        trace.shout_rhs,
        trace.shout_table_id,
    ];

    let m_in = step.mcs_inst.m_in;
    if m_in != 5 {
        return Err(PiCcsError::InvalidInput(format!(
            "no-shared-bus trace linkage expects m_in=5 (got {m_in})"
        )));
    }
    let t_len = step
        .mem_insts
        .first()
        .map(|inst| inst.steps)
        .or_else(|| {
            // Shout event-table instances may have `steps != t_len`; prefer a non-event-table
            // instance if present, otherwise fall back to inferring from the trace layout.
            step.lut_insts
                .iter()
                .find(|inst| !matches!(inst.table_spec, Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. })))
                .map(|inst| inst.steps)
        })
        .or_else(|| {
            // Trace CCS layout inference: z = [x (m_in) | trace_cols * t_len]
            let w = m.checked_sub(m_in)?;
            if trace.cols == 0 || w % trace.cols != 0 {
                return None;
            }
            Some(w / trace.cols)
        })
        .ok_or_else(|| PiCcsError::InvalidInput("missing mem/lut instances".into()))?;
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput(
            "no-shared-bus trace linkage requires steps>=1".into(),
        ));
    }
    for (i, inst) in step.mem_insts.iter().enumerate() {
        if inst.steps != t_len {
            return Err(PiCcsError::InvalidInput(format!(
                "no-shared-bus trace linkage requires stable steps across mem instances (mem_idx={i} has steps={}, expected {t_len})",
                inst.steps
            )));
        }
    }
    let trace_len = trace
        .cols
        .checked_mul(t_len)
        .ok_or_else(|| PiCcsError::InvalidInput("trace cols * t_len overflow".into()))?;
    let expected_m = m_in
        .checked_add(trace_len)
        .ok_or_else(|| PiCcsError::InvalidInput("m_in + trace_len overflow".into()))?;
    if m < expected_m {
        return Err(PiCcsError::InvalidInput(format!(
            "no-shared-bus trace linkage expects m >= m_in + trace.cols*t_len (m={}; min_m={expected_m} for t_len={t_len}, trace_cols={})",
            m, trace.cols
        )));
    }
    let expected_y_len = core_t
        .checked_add(trace_cols_to_open.len())
        .ok_or_else(|| PiCcsError::InvalidInput("core_t + trace_openings overflow".into()))?;
    if ccs_out0.y_scalars.len() != expected_y_len {
        return Err(PiCcsError::InvalidInput(format!(
            "no-shared-bus trace linkage expects CPU ME output to contain exactly core_t + trace_openings y_scalars (have {}, expected {expected_y_len})",
            ccs_out0.y_scalars.len(),
        )));
    }
    let cpu_open = |idx: usize| -> Result<K, PiCcsError> {
        ccs_out0
            .y_scalars
            .get(core_t + idx)
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError("missing CPU trace linkage opening".into()))
    };

    Ok(Some(TraceCpuLinkOpenings {
        active: cpu_open(0)?,
        prog_addr: cpu_open(1)?,
        prog_value: cpu_open(2)?,
        rs1_addr: cpu_open(3)?,
        rs1_val: cpu_open(4)?,
        rs2_addr: cpu_open(5)?,
        rs2_val: cpu_open(6)?,
        rd_has_write: cpu_open(7)?,
        rd_addr: cpu_open(8)?,
        rd_val: cpu_open(9)?,
        ram_has_read: cpu_open(10)?,
        ram_has_write: cpu_open(11)?,
        ram_addr: cpu_open(12)?,
        ram_rv: cpu_open(13)?,
        ram_wv: cpu_open(14)?,
        shout_has_lookup: cpu_open(15)?,
        shout_val: cpu_open(16)?,
        shout_lhs: cpu_open(17)?,
        shout_rhs: cpu_open(18)?,
        shout_table_id: cpu_open(19)?,
    }))
}

fn verify_no_shared_bus_twist_val_eval_phase(
    tr: &mut Poseidon2Transcript,
    m: usize,
    step: &StepInstanceBundle<Cmt, F, K>,
    prev_step: Option<&StepInstanceBundle<Cmt, F, K>>,
    proofs_mem: &[MemOrLutProof],
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    twist_pre: &[TwistAddrPreVerifyData],
    step_idx: usize,
    r_time: &[K],
) -> Result<(), PiCcsError> {
    // --------------------------------------------------------------------
    // Phase 2: Verify batched Twist val-eval sum-check, deriving shared r_val.
    // --------------------------------------------------------------------
    let has_prev = prev_step.is_some();
    let proof_offset = step.lut_insts.len();

    let mut r_val: Vec<K> = Vec::new();
    let mut val_eval_finals: Vec<K> = Vec::new();
    if !step.mem_insts.is_empty() {
        let plan = crate::memory_sidecar::claim_plan::TwistValEvalClaimPlan::build(step.mem_insts.iter(), has_prev);
        let claim_count = plan.claim_count;

        let mut per_claim_rounds: Vec<Vec<Vec<K>>> = Vec::with_capacity(claim_count);
        let mut per_claim_sums: Vec<K> = Vec::with_capacity(claim_count);
        let mut bind_claims: Vec<(u8, K)> = Vec::with_capacity(claim_count);
        let mut claim_idx = 0usize;

        for (i_mem, _inst) in step.mem_insts.iter().enumerate() {
            let twist_proof = match &proofs_mem[proof_offset + i_mem] {
                MemOrLutProof::Twist(proof) => proof,
                _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
            };
            let val = twist_proof
                .val_eval
                .as_ref()
                .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;

            per_claim_rounds.push(val.rounds_lt.clone());
            per_claim_sums.push(val.claimed_inc_sum_lt);
            bind_claims.push((plan.bind_tags[claim_idx], val.claimed_inc_sum_lt));
            claim_idx += 1;

            per_claim_rounds.push(val.rounds_total.clone());
            per_claim_sums.push(val.claimed_inc_sum_total);
            bind_claims.push((plan.bind_tags[claim_idx], val.claimed_inc_sum_total));
            claim_idx += 1;

            if has_prev {
                let prev_total = val.claimed_prev_inc_sum_total.ok_or_else(|| {
                    PiCcsError::InvalidInput("Twist(Route A): missing claimed_prev_inc_sum_total".into())
                })?;
                let prev_rounds = val
                    .rounds_prev_total
                    .clone()
                    .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing rounds_prev_total".into()))?;
                per_claim_rounds.push(prev_rounds);
                per_claim_sums.push(prev_total);
                bind_claims.push((plan.bind_tags[claim_idx], prev_total));
                claim_idx += 1;
            } else if val.claimed_prev_inc_sum_total.is_some() || val.rounds_prev_total.is_some() {
                return Err(PiCcsError::InvalidInput(
                    "Twist(Route A): rollover fields present but prev_step is None".into(),
                ));
            }
        }

        tr.append_message(
            b"twist/val_eval/batch_start",
            &(step.mem_insts.len() as u64).to_le_bytes(),
        );
        tr.append_message(b"twist/val_eval/step_idx", &(step_idx as u64).to_le_bytes());
        bind_twist_val_eval_claim_sums(tr, &bind_claims);

        let (r_val_out, finals_out, ok) = verify_batched_sumcheck_rounds_ds(
            tr,
            b"twist/val_eval_batch",
            step_idx,
            &per_claim_rounds,
            &per_claim_sums,
            &plan.labels,
            &plan.degree_bounds,
        );
        if !ok {
            return Err(PiCcsError::SumcheckError(
                "twist val-eval batched sumcheck invalid".into(),
            ));
        }
        if r_val_out.len() != r_time.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval r_val.len()={}, expected ell_n={}",
                r_val_out.len(),
                r_time.len()
            )));
        }
        if finals_out.len() != claim_count {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval finals.len()={}, expected {}",
                finals_out.len(),
                claim_count
            )));
        }
        r_val = r_val_out;
        val_eval_finals = finals_out;

        tr.append_message(b"twist/val_eval/batch_done", &[]);
    }

    // Verify val-eval terminal identity against Twist ME openings at r_val.
    let lt = if step.mem_insts.is_empty() {
        if !r_val.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "twist val-eval produced r_val but no mem instances are present".into(),
            ));
        }
        K::ZERO
    } else {
        if r_val.len() != r_time.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval r_val.len()={}, expected ell_n={}",
                r_val.len(),
                r_time.len()
            )));
        }
        lt_eval(&r_val, r_time)
    };

    let n_mem = step.mem_insts.len();
    let expected_claims = n_mem * (1 + usize::from(has_prev));
    if step.mem_insts.is_empty() {
        if !mem_proof.val_me_claims.is_empty() {
            return Err(PiCcsError::InvalidInput(
                "proof contains val-lane ME claims with no Twist instances".into(),
            ));
        }
    } else if mem_proof.val_me_claims.len() != expected_claims {
        return Err(PiCcsError::InvalidInput(format!(
            "no-shared-bus expects {} ME claim(s) at r_val (per mem instance, plus prev if any), got {}",
            expected_claims,
            mem_proof.val_me_claims.len()
        )));
    }

    for (i_mem, inst) in step.mem_insts.iter().enumerate() {
        let twist_proof = match &proofs_mem[proof_offset + i_mem] {
            MemOrLutProof::Twist(proof) => proof,
            _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
        };
        let val_eval = twist_proof
            .val_eval
            .as_ref()
            .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;
        let layout = inst.twist_layout();
        let ell_addr = layout
            .lanes
            .first()
            .ok_or_else(|| PiCcsError::InvalidInput("TwistWitnessLayout has no lanes".into()))?
            .ell_addr;

        let expected_lanes = inst.lanes.max(1);
        let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
            m,
            step.mcs_inst.m_in,
            inst.steps,
            core::iter::empty::<(usize, usize)>(),
            core::iter::once((ell_addr, expected_lanes)),
        )
        .map_err(|e| PiCcsError::InvalidInput(format!("Twist(Route A): bus layout failed: {e}")))?;

        let me_cur = mem_proof
            .val_me_claims
            .get(i_mem)
            .ok_or_else(|| PiCcsError::ProtocolError("missing Twist ME(val) claim".into()))?;
        if me_cur.r.as_slice() != r_val {
            return Err(PiCcsError::ProtocolError(
                "Twist ME(val) r mismatch (expected r_val)".into(),
            ));
        }
        if inst.comms.is_empty() || me_cur.c != inst.comms[0] {
            return Err(PiCcsError::ProtocolError("Twist ME(val) commitment mismatch".into()));
        }
        let bus_y_base_val = me_cur
            .y_scalars
            .len()
            .checked_sub(bus.bus_cols)
            .ok_or_else(|| PiCcsError::InvalidInput("Twist y_scalars too short for bus openings".into()))?;

        let r_addr = twist_pre
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput("missing Twist pre-time data".into()))?
            .r_addr
            .as_slice();

        let twist_inst_cols = bus
            .twist_cols
            .first()
            .ok_or_else(|| PiCcsError::InvalidInput("missing twist_cols[0]".into()))?;

        let mut inc_at_r_addr_val = K::ZERO;
        for twist_cols in twist_inst_cols.lanes.iter() {
            let mut wa_bits_val_open = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.wa_bits.clone() {
                wa_bits_val_open.push(
                    me_cur
                        .y_scalars
                        .get(bus.y_scalar_index(bus_y_base_val, col_id))
                        .copied()
                        .ok_or_else(|| PiCcsError::ProtocolError("missing wa_bits(val) opening".into()))?,
                );
            }
            let has_write_val_open = me_cur
                .y_scalars
                .get(bus.y_scalar_index(bus_y_base_val, twist_cols.has_write))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("missing has_write(val) opening".into()))?;
            let inc_at_write_addr_val_open = me_cur
                .y_scalars
                .get(bus.y_scalar_index(bus_y_base_val, twist_cols.inc))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("missing inc(val) opening".into()))?;

            let eq_wa_val = eq_bits_prod(&wa_bits_val_open, r_addr)?;
            inc_at_r_addr_val += has_write_val_open * inc_at_write_addr_val_open * eq_wa_val;
        }

        let expected_lt_final = inc_at_r_addr_val * lt;
        let claims_per_mem = if has_prev { 3 } else { 2 };
        let base = claims_per_mem * i_mem;
        if expected_lt_final != val_eval_finals[base] {
            return Err(PiCcsError::ProtocolError(
                "twist/val_eval_lt terminal value mismatch".into(),
            ));
        }
        let expected_total_final = inc_at_r_addr_val;
        if expected_total_final != val_eval_finals[base + 1] {
            return Err(PiCcsError::ProtocolError(
                "twist/val_eval_total terminal value mismatch".into(),
            ));
        }

        if has_prev {
            let prev = prev_step.ok_or_else(|| PiCcsError::ProtocolError("prev_step missing".into()))?;
            let prev_inst = prev
                .mem_insts
                .get(i_mem)
                .ok_or_else(|| PiCcsError::ProtocolError("missing prev mem instance".into()))?;
            let me_prev = mem_proof
                .val_me_claims
                .get(n_mem + i_mem)
                .ok_or_else(|| PiCcsError::ProtocolError("missing prev Twist ME(val)".into()))?;
            if me_prev.r.as_slice() != r_val {
                return Err(PiCcsError::ProtocolError(
                    "prev Twist ME(val) r mismatch (expected r_val)".into(),
                ));
            }
            if prev_inst.comms.is_empty() || me_prev.c != prev_inst.comms[0] {
                return Err(PiCcsError::ProtocolError(
                    "prev Twist ME(val) commitment mismatch".into(),
                ));
            }
            let bus_y_base_prev = me_prev
                .y_scalars
                .len()
                .checked_sub(bus.bus_cols)
                .ok_or_else(|| PiCcsError::InvalidInput("prev Twist y_scalars too short".into()))?;

            let mut inc_at_r_addr_prev = K::ZERO;
            for twist_cols in twist_inst_cols.lanes.iter() {
                let mut wa_bits_prev_open = Vec::with_capacity(ell_addr);
                for col_id in twist_cols.wa_bits.clone() {
                    wa_bits_prev_open.push(
                        me_prev
                            .y_scalars
                            .get(bus.y_scalar_index(bus_y_base_prev, col_id))
                            .copied()
                            .ok_or_else(|| PiCcsError::ProtocolError("missing wa_bits(prev) opening".into()))?,
                    );
                }
                let has_write_prev_open = me_prev
                    .y_scalars
                    .get(bus.y_scalar_index(bus_y_base_prev, twist_cols.has_write))
                    .copied()
                    .ok_or_else(|| PiCcsError::ProtocolError("missing has_write(prev) opening".into()))?;
                let inc_prev_open = me_prev
                    .y_scalars
                    .get(bus.y_scalar_index(bus_y_base_prev, twist_cols.inc))
                    .copied()
                    .ok_or_else(|| PiCcsError::ProtocolError("missing inc(prev) opening".into()))?;

                let eq_wa_prev = eq_bits_prod(&wa_bits_prev_open, r_addr)?;
                inc_at_r_addr_prev += has_write_prev_open * inc_prev_open * eq_wa_prev;
            }
            if inc_at_r_addr_prev != val_eval_finals[base + 2] {
                return Err(PiCcsError::ProtocolError(
                    "twist/rollover_prev_total terminal value mismatch".into(),
                ));
            }

            let claimed_prev_total = val_eval
                .claimed_prev_inc_sum_total
                .ok_or_else(|| PiCcsError::ProtocolError("twist rollover missing claimed_prev_inc_sum_total".into()))?;
            let init_prev_at_r_addr = eval_init_at_r_addr(&prev_inst.init, prev_inst.k, r_addr)?;
            let init_cur_at_r_addr = eval_init_at_r_addr(&inst.init, inst.k, r_addr)?;
            if init_cur_at_r_addr != init_prev_at_r_addr + claimed_prev_total {
                return Err(PiCcsError::ProtocolError("twist rollover init check failed".into()));
            }
        }
    }

    Ok(())
}

pub(crate) fn prove_twist_addr_pre_time(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    cpu_bus: Option<&BusLayout>,
    ell_n: usize,
    r_cycle: &[K],
) -> Result<Vec<TwistAddrPreProverData>, PiCcsError> {
    if step.mem_instances.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(step.mem_instances.len());

    let cpu_z_k = cpu_bus.map(|_| crate::memory_sidecar::cpu_bus::decode_cpu_z_to_k(params, &step.mcs.1.Z));
    if let Some(bus) = cpu_bus {
        if bus.shout_cols.len() != step.lut_instances.len() || bus.twist_cols.len() != step.mem_instances.len() {
            return Err(PiCcsError::InvalidInput(
                "shared_cpu_bus layout mismatch for step (instance counts)".into(),
            ));
        }
    }

    for (idx, (mem_inst, mem_wit)) in step.mem_instances.iter().enumerate() {
        neo_memory::addr::validate_twist_bit_addressing(mem_inst)?;
        let pow2_cycle = 1usize << ell_n;
        if mem_inst.steps > pow2_cycle {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist(Route A): steps={} exceeds 2^ell_cycle={pow2_cycle}",
                mem_inst.steps
            )));
        }

        let m = step.mcs.1.Z.cols();
        let m_in = step.mcs.0.m_in;

        let (bus, z) = match cpu_bus {
            Some(bus) => (
                bus.clone(),
                cpu_z_k
                    .as_ref()
                    .expect("cpu_z_k present when cpu_bus")
                    .clone(),
            ),
            None => {
                if mem_wit.mats.len() != 1 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Twist(Route A): non-shared-bus mode expects exactly 1 witness mat per mem instance (mem_idx={idx}, mats.len()={})",
                        mem_wit.mats.len()
                    )));
                }
                if mem_wit.mats[0].cols() != m {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Twist(Route A): mem witness width mismatch (mem_idx={idx}): mats[0].cols()={} but CPU m={m}",
                        mem_wit.mats[0].cols()
                    )));
                }
                let ell_addr = mem_inst.d * mem_inst.ell;
                let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
                    m,
                    m_in,
                    mem_inst.steps,
                    core::iter::empty::<(usize, usize)>(),
                    core::iter::once((ell_addr, mem_inst.lanes.max(1))),
                )
                .map_err(|e| PiCcsError::InvalidInput(format!("Twist(Route A): bus layout failed: {e}")))?;
                if bus.twist_cols.len() != 1 || !bus.shout_cols.is_empty() {
                    return Err(PiCcsError::ProtocolError(
                        "Twist(Route A): expected a twist-only bus layout with 1 instance".into(),
                    ));
                }
                let z = ts::decode_mat_to_k_padded(params, &mem_wit.mats[0], bus.m);
                (bus, z)
            }
        };

        let ell_addr = mem_inst.d * mem_inst.ell;
        let expected_lanes = mem_inst.lanes.max(1);
        let twist_inst_cols = if cpu_bus.is_some() {
            bus.twist_cols.get(idx).ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch: missing twist_cols for mem_idx={idx}"
                ))
            })?
        } else {
            bus.twist_cols
                .get(0)
                .ok_or_else(|| PiCcsError::ProtocolError("Twist(Route A): missing twist_cols[0]".into()))?
        };
        if twist_inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at mem_idx={idx}: expected lanes={expected_lanes}, got {}",
                twist_inst_cols.lanes.len()
            )));
        }

        let mut lanes: Vec<TwistLaneSparseCols> = Vec::with_capacity(twist_inst_cols.lanes.len());
        for (lane_idx, twist_cols) in twist_inst_cols.lanes.iter().enumerate() {
            if twist_cols.ra_bits.end - twist_cols.ra_bits.start != ell_addr
                || twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at mem_idx={idx}, lane={lane_idx}: expected ell_addr={ell_addr}"
                )));
            }

            let mut ra_bits = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.ra_bits.clone() {
                ra_bits.push(crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                    &z,
                    &bus,
                    col_id,
                    mem_inst.steps,
                    pow2_cycle,
                )?);
            }

            let mut wa_bits = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.wa_bits.clone() {
                wa_bits.push(crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                    &z,
                    &bus,
                    col_id,
                    mem_inst.steps,
                    pow2_cycle,
                )?);
            }

            let has_read = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                &z,
                &bus,
                twist_cols.has_read,
                mem_inst.steps,
                pow2_cycle,
            )?;
            let has_write = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                &z,
                &bus,
                twist_cols.has_write,
                mem_inst.steps,
                pow2_cycle,
            )?;
            let wv = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                &z,
                &bus,
                twist_cols.wv,
                mem_inst.steps,
                pow2_cycle,
            )?;
            let rv = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                &z,
                &bus,
                twist_cols.rv,
                mem_inst.steps,
                pow2_cycle,
            )?;
            let inc_at_write_addr = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                &z,
                &bus,
                twist_cols.inc,
                mem_inst.steps,
                pow2_cycle,
            )?;

            lanes.push(TwistLaneSparseCols {
                ra_bits,
                wa_bits,
                has_read,
                has_write,
                wv,
                rv,
                inc_at_write_addr,
            });
        }

        let decoded = TwistDecodedColsSparse { lanes };

        let init_sparse: Vec<(usize, K)> = match &mem_inst.init {
            MemInit::Zero => Vec::new(),
            MemInit::Sparse(pairs) => pairs
                .iter()
                .map(|(addr, val)| {
                    let addr_usize = usize::try_from(*addr).map_err(|_| {
                        PiCcsError::InvalidInput(format!("Twist: init address doesn't fit usize: addr={addr}"))
                    })?;
                    if addr_usize >= mem_inst.k {
                        return Err(PiCcsError::InvalidInput(format!(
                            "Twist: init address out of range: addr={addr} >= k={}",
                            mem_inst.k
                        )));
                    }
                    Ok((addr_usize, (*val).into()))
                })
                .collect::<Result<_, _>>()?,
        };

        let mut read_addr_oracle =
            TwistReadCheckAddrOracleSparseTimeMultiLane::new(init_sparse.clone(), r_cycle, &decoded.lanes);
        let mut write_addr_oracle =
            TwistWriteCheckAddrOracleSparseTimeMultiLane::new(init_sparse, r_cycle, &decoded.lanes);

        let labels: [&[u8]; 2] = [b"twist/read_addr_pre".as_slice(), b"twist/write_addr_pre".as_slice()];
        let claimed_sums = vec![K::ZERO, K::ZERO];
        tr.append_message(b"twist/addr_pre_time/claim_idx", &(idx as u64).to_le_bytes());
        bind_batched_claim_sums(tr, b"twist/addr_pre_time/claimed_sums", &claimed_sums, &labels);

        let mut claims = [
            BatchedClaim {
                oracle: &mut read_addr_oracle,
                claimed_sum: K::ZERO,
                label: labels[0],
            },
            BatchedClaim {
                oracle: &mut write_addr_oracle,
                claimed_sum: K::ZERO,
                label: labels[1],
            },
        ];

        let (r_addr, per_claim_results) = run_batched_sumcheck_prover_ds(tr, b"twist/addr_pre_time", idx, &mut claims)?;
        if per_claim_results.len() != 2 {
            return Err(PiCcsError::ProtocolError(format!(
                "twist addr-pre per-claim results len()={}, expected 2",
                per_claim_results.len()
            )));
        }

        out.push(TwistAddrPreProverData {
            addr_pre: BatchedAddrProof {
                claimed_sums,
                round_polys: vec![
                    per_claim_results[0].round_polys.clone(),
                    per_claim_results[1].round_polys.clone(),
                ],
                r_addr: r_addr.clone(),
            },
            decoded,
            read_check_claim_sum: per_claim_results[0].final_value,
            write_check_claim_sum: per_claim_results[1].final_value,
        });
    }

    Ok(out)
}

pub(crate) fn prove_shout_addr_pre_time(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    cpu_bus: Option<&BusLayout>,
    ell_n: usize,
    r_cycle: &[K],
    step_idx: usize,
) -> Result<ShoutAddrPreBatchProverData, PiCcsError> {
    if step.lut_instances.is_empty() {
        return Ok(ShoutAddrPreBatchProverData {
            addr_pre: ShoutAddrPreProof::default(),
            decoded: Vec::new(),
        });
    }

    let pow2_cycle = 1usize << ell_n;
    let n_lut = step.lut_instances.len();
    let total_lanes: usize = step
        .lut_instances
        .iter()
        .map(|(inst, _)| inst.lanes.max(1))
        .sum();

    let mut decoded_cols: Vec<ShoutDecodedColsSparse> = Vec::with_capacity(n_lut);
    let mut claimed_sums: Vec<K> = vec![K::ZERO; total_lanes];

    struct AddrPreGroupBuilder {
        active_lanes: Vec<u32>,
        active_claimed_sums: Vec<K>,
        addr_oracles: Vec<Box<dyn RoundOracle>>,
    }

    // Group Shout addr-pre claims by `ell_addr` so we can run one batched sumcheck per group.
    let mut groups: std::collections::BTreeMap<u32, AddrPreGroupBuilder> = std::collections::BTreeMap::new();

    let mut flat_lane_idx: usize = 0;
    if let Some(bus) = cpu_bus {
        let cpu_z_k = crate::memory_sidecar::cpu_bus::decode_cpu_z_to_k(params, &step.mcs.1.Z);
        if bus.shout_cols.len() != step.lut_instances.len() || bus.twist_cols.len() != step.mem_instances.len() {
            return Err(PiCcsError::InvalidInput(
                "shared_cpu_bus layout mismatch for step (instance counts)".into(),
            ));
        }

        for (idx, (lut_inst, _lut_wit)) in step.lut_instances.iter().enumerate() {
            neo_memory::addr::validate_shout_bit_addressing(lut_inst)?;
            if lut_inst.steps > pow2_cycle {
                return Err(PiCcsError::InvalidInput(format!(
                    "Shout(Route A): steps={} exceeds 2^ell_cycle={pow2_cycle}",
                    lut_inst.steps
                )));
            }

            let z = &cpu_z_k;
            let inst_ell_addr = lut_inst.d * lut_inst.ell;
            if matches!(
                lut_inst.table_spec,
                Some(LutTableSpec::RiscvOpcodePacked { .. } | LutTableSpec::RiscvOpcodeEventTablePacked { .. })
            ) {
                return Err(PiCcsError::InvalidInput(
                    "packed RISC-V Shout table specs are not supported on the shared CPU bus".into(),
                ));
            }
            let inst_ell_addr_u32 = u32::try_from(inst_ell_addr)
                .map_err(|_| PiCcsError::InvalidInput("Shout(Route A): ell_addr overflows u32".into()))?;
            groups
                .entry(inst_ell_addr_u32)
                .or_insert_with(|| AddrPreGroupBuilder {
                    active_lanes: Vec::new(),
                    active_claimed_sums: Vec::new(),
                    addr_oracles: Vec::new(),
                });
            let inst_cols = bus.shout_cols.get(idx).ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch: missing shout_cols for lut_idx={idx}"
                ))
            })?;
            let expected_lanes = lut_inst.lanes.max(1);
            if inst_cols.lanes.len() != expected_lanes {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at lut_idx={idx}: shout lanes={} but instance expects {}",
                    inst_cols.lanes.len(),
                    expected_lanes
                )));
            }

            let mut lanes: Vec<ShoutLaneSparseCols> = Vec::with_capacity(expected_lanes);

            for (lane_idx, shout_cols) in inst_cols.lanes.iter().enumerate() {
                if shout_cols.addr_bits.end - shout_cols.addr_bits.start != inst_ell_addr {
                    return Err(PiCcsError::InvalidInput(format!(
                        "shared_cpu_bus layout mismatch at lut_idx={idx}, lane_idx={lane_idx}: expected ell_addr={inst_ell_addr}"
                    )));
                }

                let has_lookup = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                    z,
                    bus,
                    shout_cols.has_lookup,
                    lut_inst.steps,
                    pow2_cycle,
                )?;
                let has_any_lookup = has_lookup
                    .entries()
                    .iter()
                    .any(|&(_t, gate)| gate != K::ZERO);
                let active_js: Vec<usize> = if has_any_lookup {
                    let m_in = bus.m_in;
                    let mut out: Vec<usize> = Vec::with_capacity(has_lookup.entries().len());
                    for &(t, gate) in has_lookup.entries() {
                        if gate == K::ZERO {
                            continue;
                        }
                        let j = t.checked_sub(m_in).ok_or_else(|| {
                            PiCcsError::InvalidInput(format!(
                                "Shout(Route A): has_lookup time index underflow: t={t} < m_in={m_in}"
                            ))
                        })?;
                        if j >= lut_inst.steps {
                            return Err(PiCcsError::ProtocolError(format!(
                                "Shout(Route A): has_lookup time index out of range: j={j} >= steps={}",
                                lut_inst.steps
                            )));
                        }
                        out.push(j);
                    }
                    out
                } else {
                    Vec::new()
                };

                let addr_bits: Vec<SparseIdxVec<K>> = if has_any_lookup {
                    let mut out = Vec::with_capacity(inst_ell_addr);
                    for col_id in shout_cols.addr_bits.clone() {
                        out.push(crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col_at_js(
                            z, bus, col_id, &active_js, pow2_cycle,
                        )?);
                    }
                    out
                } else {
                    vec![SparseIdxVec::new(pow2_cycle); inst_ell_addr]
                };

                let val = if has_any_lookup {
                    crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col_at_js(
                        z,
                        bus,
                        shout_cols.val,
                        &active_js,
                        pow2_cycle,
                    )?
                } else {
                    SparseIdxVec::new(pow2_cycle)
                };

                if has_any_lookup {
                    let (addr_oracle, lane_sum): (Box<dyn RoundOracle>, K) = match &lut_inst.table_spec {
                        None => {
                            let table_k: Vec<K> = lut_inst.table.iter().map(|&v| v.into()).collect();
                            let (o, sum) =
                                AddressLookupOracle::new(&addr_bits, &has_lookup, &table_k, r_cycle, inst_ell_addr);
                            (Box::new(o), sum)
                        }
                        Some(LutTableSpec::RiscvOpcode { opcode, xlen }) => {
                            let (o, sum) = RiscvAddressLookupOracleSparse::new_sparse_time(
                                *opcode,
                                *xlen,
                                &addr_bits,
                                &has_lookup,
                                r_cycle,
                            )?;
                            (Box::new(o), sum)
                        }
                        Some(LutTableSpec::RiscvOpcodePacked { .. }) => {
                            return Err(PiCcsError::InvalidInput(
                                "packed RISC-V Shout table specs are not supported on the shared CPU bus".into(),
                            ));
                        }
                        Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. }) => {
                            return Err(PiCcsError::InvalidInput(
                                "packed RISC-V Shout table specs are not supported on the shared CPU bus".into(),
                            ));
                        }
                        Some(LutTableSpec::IdentityU32) => {
                            let (o, sum) = IdentityAddressLookupOracleSparse::new_sparse_time(
                                inst_ell_addr,
                                &addr_bits,
                                &has_lookup,
                                r_cycle,
                            )?;
                            (Box::new(o), sum)
                        }
                    };

                    claimed_sums[flat_lane_idx] = lane_sum;
                    let lane_idx_u32 = u32::try_from(flat_lane_idx)
                        .map_err(|_| PiCcsError::InvalidInput("Shout(Route A): lane index overflow".into()))?;
                    let group = groups
                        .get_mut(&inst_ell_addr_u32)
                        .ok_or_else(|| PiCcsError::ProtocolError("Shout(Route A): missing ell_addr group".into()))?;
                    group.active_lanes.push(lane_idx_u32);
                    group.active_claimed_sums.push(lane_sum);
                    group.addr_oracles.push(addr_oracle);
                }

                lanes.push(ShoutLaneSparseCols {
                    addr_bits,
                    has_lookup,
                    val,
                });
                flat_lane_idx += 1;
            }

            let decoded = ShoutDecodedColsSparse { lanes };

            decoded_cols.push(decoded);
        }
    } else {
        // No-shared-bus mode: decode Shout lane columns from the committed per-instance witness mats.
        //
        // For large `ell_addr` instances (e.g. RV32 bit-addressed Shout with `ell_addr=64`), we allow
        // paging across multiple mats so each mat's bus tail fits within the CPU witness width `m`.
        let m = step.mcs.1.Z.cols();
        let m_in = step.mcs.0.m_in;

        for (lut_idx, (lut_inst, lut_wit)) in step.lut_instances.iter().enumerate() {
            neo_memory::addr::validate_shout_bit_addressing(lut_inst)?;
            if lut_inst.steps > pow2_cycle {
                return Err(PiCcsError::InvalidInput(format!(
                    "Shout(Route A): steps={} exceeds 2^ell_cycle={pow2_cycle}",
                    lut_inst.steps
                )));
            }
            if lut_wit.mats.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "Shout(Route A): missing witness mat(s) in no-shared-bus mode (lut_idx={lut_idx})"
                )));
            }
            if lut_wit.mats.len() != lut_inst.comms.len() {
                return Err(PiCcsError::InvalidInput(format!(
                    "Shout(Route A): comms/mats len mismatch (lut_idx={lut_idx}, comms.len()={}, mats.len()={})",
                    lut_inst.comms.len(),
                    lut_wit.mats.len()
                )));
            }

            let inst_ell_addr = lut_inst.d * lut_inst.ell;
            let lanes = lut_inst.lanes.max(1);
            let page_ell_addrs = plan_shout_addr_pages(m, m_in, lut_inst.steps, inst_ell_addr, lanes)?;
            if lut_wit.mats.len() != page_ell_addrs.len() {
                return Err(PiCcsError::InvalidInput(format!(
                    "Shout(Route A): paging plan mismatch (lut_idx={lut_idx}, expected {} mat(s), got {})",
                    page_ell_addrs.len(),
                    lut_wit.mats.len()
                )));
            }

            // Decode each page mat once.
            struct PageDecoded {
                bus: BusLayout,
                z: Vec<K>,
            }
            let mut pages: Vec<PageDecoded> = Vec::with_capacity(page_ell_addrs.len());
            for (page_idx, &page_ell_addr) in page_ell_addrs.iter().enumerate() {
                let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
                    m,
                    m_in,
                    lut_inst.steps,
                    core::iter::once((page_ell_addr, lanes)),
                    core::iter::empty::<(usize, usize)>(),
                )
                .map_err(|e| PiCcsError::InvalidInput(format!("Shout(Route A): bus layout failed: {e}")))?;
                if bus.shout_cols.len() != 1 || !bus.twist_cols.is_empty() {
                    return Err(PiCcsError::ProtocolError(
                        "Shout(Route A): expected a shout-only bus layout with 1 instance".into(),
                    ));
                }

                let mat = lut_wit
                    .mats
                    .get(page_idx)
                    .ok_or_else(|| PiCcsError::ProtocolError("Shout(Route A): missing page mat".into()))?;
                if mat.cols() != m {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(Route A): witness width mismatch (lut_idx={lut_idx}, page_idx={page_idx}): mat.cols()={} but CPU m={m}",
                        mat.cols()
                    )));
                }
                let z = ts::decode_mat_to_k_padded(params, mat, bus.m);
                pages.push(PageDecoded { bus, z });
            }

            // Group membership is always keyed on the *logical* instance `ell_addr`.
            let inst_ell_addr_u32 = u32::try_from(inst_ell_addr)
                .map_err(|_| PiCcsError::InvalidInput("Shout(Route A): ell_addr overflows u32".into()))?;
            groups
                .entry(inst_ell_addr_u32)
                .or_insert_with(|| AddrPreGroupBuilder {
                    active_lanes: Vec::new(),
                    active_claimed_sums: Vec::new(),
                    addr_oracles: Vec::new(),
                });

            let expected_lanes = lanes;
            let mut lanes_out: Vec<ShoutLaneSparseCols> = Vec::with_capacity(expected_lanes);

            for lane_idx in 0..expected_lanes {
                // `has_lookup`/`val` are taken from page 0 (duplicates in later pages are ignored).
                let page0 = pages.get(0).expect("pages non-empty");
                let inst_cols0 = page0
                    .bus
                    .shout_cols
                    .get(0)
                    .ok_or_else(|| PiCcsError::ProtocolError("Shout(Route A): missing shout_cols[0]".into()))?;
                let shout_cols0 = inst_cols0
                    .lanes
                    .get(lane_idx)
                    .ok_or_else(|| PiCcsError::ProtocolError("Shout(Route A): missing shout lane cols".into()))?;
                let has_lookup = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                    &page0.z,
                    &page0.bus,
                    shout_cols0.has_lookup,
                    lut_inst.steps,
                    pow2_cycle,
                )?;
                let has_any_lookup = has_lookup
                    .entries()
                    .iter()
                    .any(|&(_t, gate)| gate != K::ZERO);
                let active_js: Vec<usize> = if has_any_lookup {
                    let m_in = page0.bus.m_in;
                    let mut out: Vec<usize> = Vec::with_capacity(has_lookup.entries().len());
                    for &(t, gate) in has_lookup.entries() {
                        if gate == K::ZERO {
                            continue;
                        }
                        let j = t.checked_sub(m_in).ok_or_else(|| {
                            PiCcsError::InvalidInput(format!(
                                "Shout(Route A): has_lookup time index underflow: t={t} < m_in={m_in}"
                            ))
                        })?;
                        if j >= lut_inst.steps {
                            return Err(PiCcsError::ProtocolError(format!(
                                "Shout(Route A): has_lookup time index out of range: j={j} >= steps={}",
                                lut_inst.steps
                            )));
                        }
                        out.push(j);
                    }
                    out
                } else {
                    Vec::new()
                };

                // Concatenate addr-bit columns across pages, in-order.
                let addr_bits: Vec<SparseIdxVec<K>> = if has_any_lookup {
                    let mut out: Vec<SparseIdxVec<K>> = Vec::with_capacity(inst_ell_addr);
                    for page in pages.iter() {
                        let inst_cols =
                            page.bus.shout_cols.get(0).ok_or_else(|| {
                                PiCcsError::ProtocolError("Shout(Route A): missing shout_cols[0]".into())
                            })?;
                        let shout_cols = inst_cols.lanes.get(lane_idx).ok_or_else(|| {
                            PiCcsError::ProtocolError("Shout(Route A): missing shout lane cols".into())
                        })?;
                        for col_id in shout_cols.addr_bits.clone() {
                            out.push(crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col_at_js(
                                &page.z, &page.bus, col_id, &active_js, pow2_cycle,
                            )?);
                        }
                    }
                    if out.len() != inst_ell_addr {
                        return Err(PiCcsError::ProtocolError(format!(
                            "Shout(Route A): paging addr_bits len mismatch (lut_idx={lut_idx}, lane_idx={lane_idx}, got {}, expected {inst_ell_addr})",
                            out.len()
                        )));
                    }
                    out
                } else {
                    vec![SparseIdxVec::new(pow2_cycle); inst_ell_addr]
                };

                let val = if has_any_lookup {
                    crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col_at_js(
                        &page0.z,
                        &page0.bus,
                        shout_cols0.val,
                        &active_js,
                        pow2_cycle,
                    )?
                } else {
                    SparseIdxVec::new(pow2_cycle)
                };

                if has_any_lookup {
                    if matches!(
                        lut_inst.table_spec,
                        Some(LutTableSpec::RiscvOpcodePacked { .. } | LutTableSpec::RiscvOpcodeEventTablePacked { .. })
                    ) {
                        // Packed-key Shout lanes do not use the address-domain sumcheck (not bit-addressed).
                        // Treat them as inactive in addr-pre and enforce correctness directly in time rounds.
                    } else {
                        let (addr_oracle, lane_sum): (Box<dyn RoundOracle>, K) = match &lut_inst.table_spec {
                            None => {
                                let table_k: Vec<K> = lut_inst.table.iter().map(|&v| v.into()).collect();
                                let (o, sum) =
                                    AddressLookupOracle::new(&addr_bits, &has_lookup, &table_k, r_cycle, inst_ell_addr);
                                (Box::new(o), sum)
                            }
                            Some(LutTableSpec::RiscvOpcode { opcode, xlen }) => {
                                let (o, sum) = RiscvAddressLookupOracleSparse::new_sparse_time(
                                    *opcode,
                                    *xlen,
                                    &addr_bits,
                                    &has_lookup,
                                    r_cycle,
                                )?;
                                (Box::new(o), sum)
                            }
                            Some(LutTableSpec::IdentityU32) => {
                                let (o, sum) = IdentityAddressLookupOracleSparse::new_sparse_time(
                                    inst_ell_addr,
                                    &addr_bits,
                                    &has_lookup,
                                    r_cycle,
                                )?;
                                (Box::new(o), sum)
                            }
                            Some(LutTableSpec::RiscvOpcodePacked { .. }) => {
                                return Err(PiCcsError::ProtocolError(
                                    "unexpected RiscvOpcodePacked match drift".into(),
                                ));
                            }
                            Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. }) => {
                                return Err(PiCcsError::ProtocolError(
                                    "unexpected RiscvOpcodeEventTablePacked match drift".into(),
                                ));
                            }
                        };

                        claimed_sums[flat_lane_idx] = lane_sum;
                        let lane_idx_u32 = u32::try_from(flat_lane_idx)
                            .map_err(|_| PiCcsError::InvalidInput("Shout(Route A): lane index overflow".into()))?;
                        let group = groups.get_mut(&inst_ell_addr_u32).ok_or_else(|| {
                            PiCcsError::ProtocolError("Shout(Route A): missing ell_addr group".into())
                        })?;
                        group.active_lanes.push(lane_idx_u32);
                        group.active_claimed_sums.push(lane_sum);
                        group.addr_oracles.push(addr_oracle);
                    }
                }

                lanes_out.push(ShoutLaneSparseCols {
                    addr_bits,
                    has_lookup,
                    val,
                });
                flat_lane_idx += 1;
            }

            decoded_cols.push(ShoutDecodedColsSparse { lanes: lanes_out });
        }
    }
    if flat_lane_idx != total_lanes {
        return Err(PiCcsError::ProtocolError(format!(
            "Shout(Route A): flat lane indexing drift (got {flat_lane_idx}, expected {total_lanes})"
        )));
    }

    let labels_all: Vec<&'static [u8]> = vec![b"shout/addr_pre".as_slice(); total_lanes];
    tr.append_message(b"shout/addr_pre_time/step_idx", &(step_idx as u64).to_le_bytes());
    bind_batched_claim_sums(tr, b"shout/addr_pre_time/claimed_sums", &claimed_sums, &labels_all);

    let mut group_proofs: Vec<ShoutAddrPreGroupProof<K>> = Vec::with_capacity(groups.len());
    for (group_idx, (ell_addr, mut group)) in groups.into_iter().enumerate() {
        tr.append_message(b"shout/addr_pre_time/group_idx", &(group_idx as u64).to_le_bytes());
        tr.append_message(b"shout/addr_pre_time/group_ell_addr", &(ell_addr as u64).to_le_bytes());

        let (r_addr, round_polys) = if group.active_lanes.is_empty() {
            // No active lanes in this `ell_addr` group; sample an arbitrary `r_addr` without running sumcheck.
            tr.append_message(b"shout/addr_pre_time/no_sumcheck", &(step_idx as u64).to_le_bytes());
            tr.append_message(
                b"shout/addr_pre_time/no_sumcheck/ell_addr",
                &(ell_addr as u64).to_le_bytes(),
            );
            (
                ts::sample_ext_point(
                    tr,
                    b"shout/addr_pre_time/no_sumcheck/r_addr",
                    b"shout/addr_pre_time/no_sumcheck/r_addr/0",
                    b"shout/addr_pre_time/no_sumcheck/r_addr/1",
                    ell_addr as usize,
                ),
                Vec::new(),
            )
        } else {
            let labels_active: Vec<&'static [u8]> = vec![b"shout/addr_pre".as_slice(); group.addr_oracles.len()];
            let mut claims: Vec<BatchedClaim<'_>> = group
                .addr_oracles
                .iter_mut()
                .zip(group.active_claimed_sums.iter())
                .zip(labels_active.iter())
                .map(|((oracle, sum), label)| BatchedClaim {
                    oracle: oracle.as_mut(),
                    claimed_sum: *sum,
                    label: *label,
                })
                .collect();

            let (r_addr, per_claim_results) =
                run_batched_sumcheck_prover_ds(tr, b"shout/addr_pre_time", step_idx, claims.as_mut_slice())?;
            let round_polys = per_claim_results
                .iter()
                .map(|r| r.round_polys.clone())
                .collect::<Vec<_>>();
            (r_addr, round_polys)
        };

        group_proofs.push(ShoutAddrPreGroupProof {
            ell_addr,
            active_lanes: group.active_lanes,
            round_polys,
            r_addr,
        });
    }

    Ok(ShoutAddrPreBatchProverData {
        addr_pre: ShoutAddrPreProof {
            claimed_sums,
            groups: group_proofs,
        },
        decoded: decoded_cols,
    })
}

pub fn verify_shout_addr_pre_time(
    tr: &mut Poseidon2Transcript,
    step: &StepInstanceBundle<Cmt, F, K>,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    step_idx: usize,
) -> Result<Vec<ShoutAddrPreVerifyData>, PiCcsError> {
    let proof = &mem_proof.shout_addr_pre;

    if step.lut_insts.is_empty() {
        if !proof.claimed_sums.is_empty() || !proof.groups.is_empty() {
            return Err(PiCcsError::InvalidInput(
                "shout_addr_pre must be empty when there are no Shout instances".into(),
            ));
        }
        return Ok(Vec::new());
    }

    let total_lanes: usize = step.lut_insts.iter().map(|inst| inst.lanes.max(1)).sum();
    if proof.claimed_sums.len() != total_lanes {
        return Err(PiCcsError::InvalidInput(format!(
            "shout_addr_pre claimed_sums.len()={}, expected total_lanes={}",
            proof.claimed_sums.len(),
            total_lanes
        )));
    }

    // Flatten lane->ell_addr mapping in canonical order so we can validate group membership and
    // attach the correct `r_addr` per lane.
    let mut lane_ell_addr: Vec<u32> = Vec::with_capacity(total_lanes);
    let mut required_ell_addrs: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for lut_inst in step.lut_insts.iter() {
        neo_memory::addr::validate_shout_bit_addressing(lut_inst)?;
        let inst_ell_addr = lut_inst.d * lut_inst.ell;
        let inst_ell_addr_u32 = u32::try_from(inst_ell_addr)
            .map_err(|_| PiCcsError::InvalidInput("Shout: ell_addr overflows u32".into()))?;
        required_ell_addrs.insert(inst_ell_addr_u32);
        for _lane_idx in 0..lut_inst.lanes.max(1) {
            lane_ell_addr.push(inst_ell_addr_u32);
        }
    }
    if lane_ell_addr.len() != total_lanes {
        return Err(PiCcsError::ProtocolError(
            "shout addr-pre lane indexing drift (lane_ell_addr)".into(),
        ));
    }

    // Groups must match the step's required `ell_addr` set and be sorted/unique.
    if proof.groups.len() != required_ell_addrs.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "shout_addr_pre groups.len()={}, expected {} (distinct ell_addr values in step)",
            proof.groups.len(),
            required_ell_addrs.len()
        )));
    }
    let required_list: Vec<u32> = required_ell_addrs.into_iter().collect();
    for (idx, group) in proof.groups.iter().enumerate() {
        let expected_ell_addr = required_list[idx];
        if group.ell_addr != expected_ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "shout_addr_pre groups not sorted or mismatched: groups[{idx}].ell_addr={} but expected {expected_ell_addr}",
                group.ell_addr
            )));
        }
        if group.r_addr.len() != group.ell_addr as usize {
            return Err(PiCcsError::InvalidInput(format!(
                "shout_addr_pre group ell_addr={} has r_addr.len()={}, expected {}",
                group.ell_addr,
                group.r_addr.len(),
                group.ell_addr
            )));
        }
        if group.round_polys.len() != group.active_lanes.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "shout_addr_pre group ell_addr={} round_polys.len()={}, expected active_lanes.len()={}",
                group.ell_addr,
                group.round_polys.len(),
                group.active_lanes.len()
            )));
        }

        for (pos, &lane_idx) in group.active_lanes.iter().enumerate() {
            let lane_idx_usize = lane_idx as usize;
            if lane_idx_usize >= total_lanes {
                return Err(PiCcsError::InvalidInput(
                    "shout_addr_pre active_lanes has index out of range".into(),
                ));
            }
            if lane_ell_addr[lane_idx_usize] != group.ell_addr {
                return Err(PiCcsError::InvalidInput(format!(
                    "shout_addr_pre active_lanes contains lane_idx={} with ell_addr={}, but group ell_addr={}",
                    lane_idx, lane_ell_addr[lane_idx_usize], group.ell_addr
                )));
            }
            if pos > 0 && group.active_lanes[pos - 1] >= lane_idx {
                return Err(PiCcsError::InvalidInput(
                    "shout_addr_pre active_lanes must be strictly increasing".into(),
                ));
            }
        }
        for (pos, rounds) in group.round_polys.iter().enumerate() {
            if rounds.len() != group.ell_addr as usize {
                return Err(PiCcsError::InvalidInput(format!(
                    "shout_addr_pre group ell_addr={} round_polys[{pos}].len()={}, expected {}",
                    group.ell_addr,
                    rounds.len(),
                    group.ell_addr
                )));
            }
        }
    }

    // Bind all claimed sums (all lanes) once.
    let labels_all: Vec<&'static [u8]> = vec![b"shout/addr_pre".as_slice(); total_lanes];
    tr.append_message(b"shout/addr_pre_time/step_idx", &(step_idx as u64).to_le_bytes());
    bind_batched_claim_sums(
        tr,
        b"shout/addr_pre_time/claimed_sums",
        &proof.claimed_sums,
        &labels_all,
    );

    // Verify each `ell_addr` group independently, collecting per-lane addr-pre finals and
    // recording the shared `r_addr` for that group.
    let mut lane_is_active = vec![false; total_lanes];
    let mut lane_addr_final = vec![K::ZERO; total_lanes];
    let mut r_addr_by_ell: std::collections::BTreeMap<u32, Vec<K>> = std::collections::BTreeMap::new();
    let mut seen_active: std::collections::HashSet<u32> = std::collections::HashSet::new();

    for (group_idx, group) in proof.groups.iter().enumerate() {
        tr.append_message(b"shout/addr_pre_time/group_idx", &(group_idx as u64).to_le_bytes());
        tr.append_message(
            b"shout/addr_pre_time/group_ell_addr",
            &(group.ell_addr as u64).to_le_bytes(),
        );

        if group.active_lanes.is_empty() {
            // No active lanes in this group: match prover's deterministic fallback sampling.
            tr.append_message(b"shout/addr_pre_time/no_sumcheck", &(step_idx as u64).to_le_bytes());
            tr.append_message(
                b"shout/addr_pre_time/no_sumcheck/ell_addr",
                &(group.ell_addr as u64).to_le_bytes(),
            );
            let r_addr = ts::sample_ext_point(
                tr,
                b"shout/addr_pre_time/no_sumcheck/r_addr",
                b"shout/addr_pre_time/no_sumcheck/r_addr/0",
                b"shout/addr_pre_time/no_sumcheck/r_addr/1",
                group.ell_addr as usize,
            );
            if r_addr != group.r_addr {
                return Err(PiCcsError::ProtocolError(
                    "shout_addr_pre r_addr mismatch: transcript-derived vs proof".into(),
                ));
            }
            r_addr_by_ell.insert(group.ell_addr, r_addr);
            continue;
        }

        let active_count = group.active_lanes.len();
        let mut active_claimed_sums: Vec<K> = Vec::with_capacity(active_count);
        for &lane_idx in group.active_lanes.iter() {
            if !seen_active.insert(lane_idx) {
                return Err(PiCcsError::InvalidInput(
                    "shout_addr_pre active_lanes contains duplicates across groups".into(),
                ));
            }
            active_claimed_sums.push(
                *proof
                    .claimed_sums
                    .get(lane_idx as usize)
                    .ok_or_else(|| PiCcsError::ProtocolError("shout addr-pre active lane idx drift".into()))?,
            );
        }
        let labels_active: Vec<&'static [u8]> = vec![b"shout/addr_pre".as_slice(); active_count];
        let degree_bounds = vec![2usize; active_count];
        let (r_addr, finals, ok) = verify_batched_sumcheck_rounds_ds(
            tr,
            b"shout/addr_pre_time",
            step_idx,
            &group.round_polys,
            &active_claimed_sums,
            &labels_active,
            &degree_bounds,
        );
        if !ok {
            return Err(PiCcsError::SumcheckError(
                "shout addr-pre batched sumcheck invalid".into(),
            ));
        }
        if r_addr != group.r_addr {
            return Err(PiCcsError::ProtocolError(
                "shout_addr_pre r_addr mismatch: transcript-derived vs proof".into(),
            ));
        }
        if finals.len() != active_count {
            return Err(PiCcsError::ProtocolError(format!(
                "shout addr-pre finals.len()={}, expected active_count={active_count}",
                finals.len()
            )));
        }

        for (pos, &lane_idx) in group.active_lanes.iter().enumerate() {
            let lane_idx_usize = lane_idx as usize;
            lane_is_active[lane_idx_usize] = true;
            lane_addr_final[lane_idx_usize] = finals[pos];
        }
        r_addr_by_ell.insert(group.ell_addr, r_addr);
    }

    // Build per-lane verify data in canonical order.
    let mut out = Vec::with_capacity(total_lanes);
    for (lut_inst, inst_ell_addr) in step.lut_insts.iter().map(|inst| (inst, inst.d * inst.ell)) {
        let expected_lanes = lut_inst.lanes.max(1);
        let inst_ell_addr_u32 = u32::try_from(inst_ell_addr)
            .map_err(|_| PiCcsError::InvalidInput("Shout: ell_addr overflows u32".into()))?;
        let r_addr = r_addr_by_ell
            .get(&inst_ell_addr_u32)
            .ok_or_else(|| PiCcsError::ProtocolError("missing shout addr-pre group r_addr".into()))?;

        for _lane_idx in 0..expected_lanes {
            let flat_lane_idx = out.len();
            let addr_claim_sum = *proof
                .claimed_sums
                .get(flat_lane_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("shout addr-pre lane index drift".into()))?;
            let is_active = *lane_is_active
                .get(flat_lane_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("shout addr-pre lane idx drift".into()))?;
            let addr_final = *lane_addr_final
                .get(flat_lane_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("shout addr-pre lane idx drift".into()))?;

            let table_eval_at_r_addr = if is_active {
                match &lut_inst.table_spec {
                    None => {
                        let pow2 = 1usize
                            .checked_shl(r_addr.len() as u32)
                            .ok_or_else(|| PiCcsError::InvalidInput("Shout: 2^ell_addr overflow".into()))?;
                        let mut acc = K::ZERO;
                        for (i, &v) in lut_inst.table.iter().enumerate().take(pow2) {
                            let w = neo_memory::mle::chi_at_index(r_addr, i);
                            acc += K::from(v) * w;
                        }
                        acc
                    }
                    Some(spec) => spec.eval_table_mle(r_addr)?,
                }
            } else {
                K::ZERO
            };

            out.push(ShoutAddrPreVerifyData {
                is_active,
                addr_claim_sum,
                addr_final: if is_active { addr_final } else { K::ZERO },
                r_addr: r_addr.clone(),
                table_eval_at_r_addr,
            });
        }
    }
    if out.len() != total_lanes {
        return Err(PiCcsError::ProtocolError("shout addr-pre lane count mismatch".into()));
    }

    Ok(out)
}

pub fn verify_twist_addr_pre_time(
    tr: &mut Poseidon2Transcript,
    step: &StepInstanceBundle<Cmt, F, K>,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
) -> Result<Vec<TwistAddrPreVerifyData>, PiCcsError> {
    let mut out = Vec::with_capacity(step.mem_insts.len());
    let proof_offset = step.lut_insts.len();

    for (idx, mem_inst) in step.mem_insts.iter().enumerate() {
        let proof = match mem_proof.proofs.get(proof_offset + idx) {
            Some(MemOrLutProof::Twist(p)) => p,
            _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
        };

        if proof.addr_pre.claimed_sums.len() != 2 {
            return Err(PiCcsError::InvalidInput(format!(
                "twist addr_pre claimed_sums.len()={}, expected 2",
                proof.addr_pre.claimed_sums.len()
            )));
        }
        if proof.addr_pre.round_polys.len() != 2 {
            return Err(PiCcsError::InvalidInput(format!(
                "twist addr_pre round_polys.len()={}, expected 2",
                proof.addr_pre.round_polys.len()
            )));
        }
        if proof.addr_pre.claimed_sums[0] != K::ZERO || proof.addr_pre.claimed_sums[1] != K::ZERO {
            return Err(PiCcsError::ProtocolError(
                "twist addr_pre claimed_sums mismatch (expected both 0)".into(),
            ));
        }

        let labels: [&[u8]; 2] = [b"twist/read_addr_pre".as_slice(), b"twist/write_addr_pre".as_slice()];
        let degree_bounds = vec![2usize, 2usize];
        tr.append_message(b"twist/addr_pre_time/claim_idx", &(idx as u64).to_le_bytes());
        bind_batched_claim_sums(
            tr,
            b"twist/addr_pre_time/claimed_sums",
            &proof.addr_pre.claimed_sums,
            &labels,
        );

        let (r_addr, finals, ok) = verify_batched_sumcheck_rounds_ds(
            tr,
            b"twist/addr_pre_time",
            idx,
            &proof.addr_pre.round_polys,
            &proof.addr_pre.claimed_sums,
            &labels,
            &degree_bounds,
        );
        if !ok {
            return Err(PiCcsError::SumcheckError(
                "twist addr-pre batched sumcheck invalid".into(),
            ));
        }
        if r_addr != proof.addr_pre.r_addr {
            return Err(PiCcsError::ProtocolError(
                "twist addr_pre r_addr mismatch: transcript-derived vs proof".into(),
            ));
        }

        let ell_addr = mem_inst.d * mem_inst.ell;
        if r_addr.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "twist addr_pre r_addr.len()={}, expected ell_addr={}",
                r_addr.len(),
                ell_addr
            )));
        }
        if finals.len() != 2 {
            return Err(PiCcsError::ProtocolError(format!(
                "twist addr-pre finals.len()={}, expected 2",
                finals.len()
            )));
        }

        out.push(TwistAddrPreVerifyData {
            r_addr,
            read_check_claim_sum: finals[0],
            write_check_claim_sum: finals[1],
        });
    }

    Ok(out)
}

pub(crate) fn build_route_a_memory_oracles(
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    ell_n: usize,
    r_cycle: &[K],
    shout_pre: &ShoutAddrPreBatchProverData,
    twist_pre: &[TwistAddrPreProverData],
) -> Result<RouteAMemoryOracles, PiCcsError> {
    if ell_n != r_cycle.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "Route A: ell_n mismatch (ell_n={ell_n}, r_cycle.len()={})",
            r_cycle.len()
        )));
    }
    if shout_pre.decoded.len() != step.lut_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "shout pre-time count mismatch (expected {}, got {})",
            step.lut_instances.len(),
            shout_pre.decoded.len()
        )));
    }
    if twist_pre.len() != step.mem_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "twist pre-time decoded count mismatch (expected {}, got {})",
            step.mem_instances.len(),
            twist_pre.len()
        )));
    }

    let any_event_table_shout = step
        .lut_instances
        .iter()
        .any(|(inst, _wit)| matches!(inst.table_spec, Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. })));
    if any_event_table_shout {
        for (idx, (inst, _wit)) in step.lut_instances.iter().enumerate() {
            if !matches!(inst.table_spec, Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. })) {
                return Err(PiCcsError::InvalidInput(format!(
                    "event-table Shout mode requires all Shout instances to use RiscvOpcodeEventTablePacked (lut_idx={idx})"
                )));
            }
        }
    }

    let event_hash_coeffs = |r: &[K]| -> Result<(K, K, K), PiCcsError> {
        if r.len() < 3 {
            return Err(PiCcsError::InvalidInput("event-table Shout requires ell_n >= 3".into()));
        }
        Ok((r[0], r[1], r[2]))
    };
    let (event_alpha, event_beta, event_gamma) = if any_event_table_shout {
        event_hash_coeffs(r_cycle)?
    } else {
        (K::ZERO, K::ZERO, K::ZERO)
    };

    let shout_event_trace_hash: Option<RouteAShoutEventTraceHashOracle> = if any_event_table_shout {
        let m_in = step.mcs.0.m_in;
        if m_in != 5 {
            return Err(PiCcsError::InvalidInput(format!(
                "event-table Shout trace linkage expects m_in=5 (got {m_in})"
            )));
        }
        let trace = Rv32TraceLayout::new();
        let m = step.mcs.1.Z.cols();
        let t_len = step
            .mem_instances
            .first()
            .map(|(inst, _wit)| inst.steps)
            .or_else(|| {
                let w = m.checked_sub(m_in)?;
                if trace.cols == 0 || w % trace.cols != 0 {
                    return None;
                }
                Some(w / trace.cols)
            })
            .ok_or_else(|| PiCcsError::InvalidInput("event-table Shout trace linkage missing t_len".into()))?;
        if t_len == 0 {
            return Err(PiCcsError::InvalidInput(
                "event-table Shout trace linkage requires t_len >= 1".into(),
            ));
        }
        let pow2_cycle = 1usize
            .checked_shl(ell_n as u32)
            .ok_or_else(|| PiCcsError::InvalidInput("event-table Shout: 2^ell_n overflow".into()))?;
        if m_in
            .checked_add(t_len)
            .ok_or_else(|| PiCcsError::InvalidInput("event-table Shout: m_in + t_len overflow".into()))?
            > pow2_cycle
        {
            return Err(PiCcsError::InvalidInput(format!(
                "event-table Shout: trace time rows out of range: m_in({m_in}) + t_len({t_len}) > 2^ell_n({pow2_cycle})"
            )));
        }

        let d = neo_math::D;
        let Z = &step.mcs.1.Z;
        if Z.rows() != d {
            return Err(PiCcsError::InvalidInput(format!(
                "event-table Shout: CPU witness Z.rows()={} != D={d}",
                Z.rows()
            )));
        }
        if Z.cols() != m {
            return Err(PiCcsError::ProtocolError(
                "event-table Shout: CPU witness width drift".into(),
            ));
        }

        let bK = K::from(F::from_u64(params.b as u64));
        let mut pow_b = Vec::with_capacity(d);
        let mut cur = K::ONE;
        for _ in 0..d {
            pow_b.push(cur);
            cur *= bK;
        }
        let decode_idx = |idx: usize| -> Result<K, PiCcsError> {
            if idx >= m {
                return Err(PiCcsError::InvalidInput(format!(
                    "event-table Shout: z idx out of range (idx={idx}, m={m})"
                )));
            }
            let mut acc = K::ZERO;
            for rho in 0..d {
                acc += pow_b[rho] * K::from(Z[(rho, idx)]);
            }
            Ok(acc)
        };

        let trace_base = m_in;
        let shout_col = |col_id: usize, j: usize| -> Result<K, PiCcsError> {
            let col_offset = col_id
                .checked_mul(t_len)
                .ok_or_else(|| PiCcsError::InvalidInput("trace col_id * t_len overflow".into()))?;
            let idx = trace_base
                .checked_add(col_offset)
                .and_then(|x| x.checked_add(j))
                .ok_or_else(|| PiCcsError::InvalidInput("trace z idx overflow".into()))?;
            decode_idx(idx)
        };

        let mut gate_entries: Vec<(usize, K)> = Vec::new();
        let mut hash_entries: Vec<(usize, K)> = Vec::new();
        for j in 0..t_len {
            let t = m_in + j;
            let gate = shout_col(trace.shout_has_lookup, j)?;
            if gate == K::ZERO {
                continue;
            }
            gate_entries.push((t, gate));

            let val = shout_col(trace.shout_val, j)?;
            let lhs = shout_col(trace.shout_lhs, j)?;
            let rhs = shout_col(trace.shout_rhs, j)?;
            let hash = K::ONE + event_alpha * val + event_beta * lhs + event_gamma * rhs;
            if hash != K::ZERO {
                hash_entries.push((t, hash));
            }
        }

        let gate = SparseIdxVec::from_entries(pow2_cycle, gate_entries);
        let hash = SparseIdxVec::from_entries(pow2_cycle, hash_entries);
        let (oracle, claim) = ShoutValueOracleSparse::new(r_cycle, gate, hash);
        Some(RouteAShoutEventTraceHashOracle {
            oracle: Box::new(oracle),
            claim,
        })
    } else {
        None
    };

    let mut shout_oracles = Vec::with_capacity(step.lut_instances.len());
    let mut r_addr_by_ell: std::collections::BTreeMap<u32, &[K]> = std::collections::BTreeMap::new();
    for g in shout_pre.addr_pre.groups.iter() {
        r_addr_by_ell.insert(g.ell_addr, g.r_addr.as_slice());
    }
    for (lut_idx, ((lut_inst, _lut_wit), decoded)) in step
        .lut_instances
        .iter()
        .zip(shout_pre.decoded.iter())
        .enumerate()
    {
        let ell_addr = lut_inst.d * lut_inst.ell;
        let ell_addr_u32 = u32::try_from(ell_addr)
            .map_err(|_| PiCcsError::InvalidInput("Shout(Route A): ell_addr overflows u32".into()))?;
        let r_addr = *r_addr_by_ell
            .get(&ell_addr_u32)
            .ok_or_else(|| PiCcsError::ProtocolError("missing shout addr-pre group r_addr".into()))?;
        if r_addr.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "Shout(Route A): r_addr.len()={} != ell_addr={}",
                r_addr.len(),
                ell_addr
            )));
        }

        if decoded.lanes.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "Shout(Route A): decoded lanes empty at lut_idx={lut_idx}"
            )));
        }

        let lane_count = decoded.lanes.len();
        let mut lanes: Vec<RouteAShoutTimeLaneOracles> = Vec::with_capacity(lane_count);

        let packed_layout = rv32_packed_shout_layout(&lut_inst.table_spec)?;
        let packed_op = packed_layout.map(|(op, _time_bits)| op);
        let packed_time_bits = packed_layout.map(|(_op, time_bits)| time_bits).unwrap_or(0);
        let is_packed = packed_op.is_some();
        if packed_time_bits != 0 && packed_time_bits != ell_n {
            return Err(PiCcsError::InvalidInput(format!(
                "event-table Shout expects time_bits == ell_n (time_bits={packed_time_bits}, ell_n={ell_n})"
            )));
        }

        for lane in decoded.lanes.iter() {
            if let Some(op) = packed_op {
                let time_bits = packed_time_bits;
                let packed_cols: &[SparseIdxVec<K>] = lane.addr_bits.get(time_bits..).ok_or_else(|| {
                    PiCcsError::InvalidInput("packed RV32: addr_bits too short for time_bits prefix".into())
                })?;
                let lhs = packed_cols
                    .get(0)
                    .ok_or_else(|| PiCcsError::InvalidInput("packed RV32: missing lhs column".into()))?
                    .clone();
                let rhs = packed_cols
                    .get(1)
                    .ok_or_else(|| PiCcsError::InvalidInput("packed RV32: missing rhs column".into()))?
                    .clone();

                // Packed bitwise (AND/OR/XOR): base-4 digit decomposition.
                let (bitwise_lhs_digits, bitwise_rhs_digits) = match op {
                    Rv32PackedShoutOp::And
                    | Rv32PackedShoutOp::Andn
                    | Rv32PackedShoutOp::Or
                    | Rv32PackedShoutOp::Xor => {
                        if packed_cols.len() != 34 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 bitwise: expected ell_addr=34, got {}",
                                packed_cols.len()
                            )));
                        }
                        let lhs_digits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(2).take(16).cloned().collect();
                        let rhs_digits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(18).take(16).cloned().collect();
                        if lhs_digits.len() != 16 || rhs_digits.len() != 16 {
                            return Err(PiCcsError::ProtocolError(
                                "packed RV32 bitwise: digit slice length mismatch".into(),
                            ));
                        }
                        (lhs_digits, rhs_digits)
                    }
                    _ => (Vec::new(), Vec::new()),
                };

                let value_oracle: Box<dyn RoundOracle> = match op {
                    Rv32PackedShoutOp::And => Box::new(Rv32PackedAndOracleSparseTime::new(
                        r_cycle,
                        lane.has_lookup.clone(),
                        bitwise_lhs_digits.clone(),
                        bitwise_rhs_digits.clone(),
                        lane.val.clone(),
                    )),
                    Rv32PackedShoutOp::Andn => Box::new(Rv32PackedAndnOracleSparseTime::new(
                        r_cycle,
                        lane.has_lookup.clone(),
                        bitwise_lhs_digits.clone(),
                        bitwise_rhs_digits.clone(),
                        lane.val.clone(),
                    )),
                    Rv32PackedShoutOp::Add => Box::new(Rv32PackedAddOracleSparseTime::new(
                        r_cycle,
                        lane.has_lookup.clone(),
                        lhs.clone(),
                        rhs.clone(),
                        packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 ADD: missing carry column".into()))?
                            .clone(),
                        lane.val.clone(),
                    )),
                    Rv32PackedShoutOp::Or => Box::new(Rv32PackedOrOracleSparseTime::new(
                        r_cycle,
                        lane.has_lookup.clone(),
                        bitwise_lhs_digits.clone(),
                        bitwise_rhs_digits.clone(),
                        lane.val.clone(),
                    )),
                    Rv32PackedShoutOp::Sub => Box::new(Rv32PackedSubOracleSparseTime::new(
                        r_cycle,
                        lane.has_lookup.clone(),
                        lhs.clone(),
                        rhs.clone(),
                        packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 SUB: missing borrow column".into()))?
                            .clone(),
                        lane.val.clone(),
                    )),
                    Rv32PackedShoutOp::Xor => Box::new(Rv32PackedXorOracleSparseTime::new(
                        r_cycle,
                        lane.has_lookup.clone(),
                        bitwise_lhs_digits.clone(),
                        bitwise_rhs_digits.clone(),
                        lane.val.clone(),
                    )),
                    Rv32PackedShoutOp::Eq => Box::new(Rv32PackedEqOracleSparseTime::new(
                        r_cycle,
                        lane.has_lookup.clone(),
                        {
                            let diff_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(3).cloned().collect();
                            if diff_bits.len() != 32 {
                                return Err(PiCcsError::InvalidInput(format!(
                                    "packed RV32 EQ: expected 32 diff bits, got {}",
                                    diff_bits.len()
                                )));
                            }
                            diff_bits
                        },
                        lane.val.clone(),
                    )),
                    Rv32PackedShoutOp::Neq => Box::new(Rv32PackedNeqOracleSparseTime::new(
                        r_cycle,
                        lane.has_lookup.clone(),
                        {
                            let diff_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(3).cloned().collect();
                            if diff_bits.len() != 32 {
                                return Err(PiCcsError::InvalidInput(format!(
                                    "packed RV32 NEQ: expected 32 diff bits, got {}",
                                    diff_bits.len()
                                )));
                            }
                            diff_bits
                        },
                        lane.val.clone(),
                    )),
                    Rv32PackedShoutOp::Mul => {
                        let carry_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(2).cloned().collect();
                        if carry_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 MUL: expected 32 carry bits, got {}",
                                carry_bits.len()
                            )));
                        }
                        Box::new(Rv32PackedMulOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            lhs.clone(),
                            rhs.clone(),
                            carry_bits,
                            lane.val.clone(),
                        ))
                    }
                    Rv32PackedShoutOp::Mulhu => {
                        let lo_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(2).cloned().collect();
                        if lo_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 MULHU: expected 32 lo bits, got {}",
                                lo_bits.len()
                            )));
                        }
                        Box::new(Rv32PackedMulhuOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            lhs.clone(),
                            rhs.clone(),
                            lo_bits,
                            lane.val.clone(),
                        ))
                    }
                    Rv32PackedShoutOp::Mulh => {
                        let hi = packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULH: missing hi opening".into()))?
                            .clone();
                        let lo_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(6).cloned().collect();
                        if lo_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 MULH: expected 32 lo bits, got {}",
                                lo_bits.len()
                            )));
                        }
                        Box::new(Rv32PackedMulHiOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            lhs.clone(),
                            rhs.clone(),
                            lo_bits,
                            hi,
                        ))
                    }
                    Rv32PackedShoutOp::Mulhsu => {
                        let hi = packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULHSU: missing hi opening".into()))?
                            .clone();
                        let lo_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(5).cloned().collect();
                        if lo_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 MULHSU: expected 32 lo bits, got {}",
                                lo_bits.len()
                            )));
                        }
                        Box::new(Rv32PackedMulHiOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            lhs.clone(),
                            rhs.clone(),
                            lo_bits,
                            hi,
                        ))
                    }
                    Rv32PackedShoutOp::Slt => {
                        let lhs_sign = packed_cols
                            .get(3)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 SLT: missing lhs_sign bit".into()))?
                            .clone();
                        let rhs_sign = packed_cols
                            .get(4)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 SLT: missing rhs_sign bit".into()))?
                            .clone();
                        let diff = packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 SLT: missing diff opening".into()))?
                            .clone();
                        Box::new(Rv32PackedSltOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            lhs.clone(),
                            rhs.clone(),
                            lhs_sign,
                            rhs_sign,
                            diff,
                            lane.val.clone(),
                        ))
                    }
                    Rv32PackedShoutOp::Divu => {
                        let rem = packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIVU: missing rem opening".into()))?
                            .clone();
                        let rhs_is_zero = packed_cols
                            .get(4)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIVU: missing rhs_is_zero".into()))?
                            .clone();
                        Box::new(Rv32PackedDivuOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            lhs.clone(),
                            rhs.clone(),
                            rem,
                            rhs_is_zero,
                            lane.val.clone(),
                        ))
                    }
                    Rv32PackedShoutOp::Remu => {
                        let quot = packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REMU: missing quot opening".into()))?
                            .clone();
                        let rhs_is_zero = packed_cols
                            .get(4)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REMU: missing rhs_is_zero".into()))?
                            .clone();
                        Box::new(Rv32PackedRemuOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            lhs.clone(),
                            rhs.clone(),
                            quot,
                            rhs_is_zero,
                            lane.val.clone(),
                        ))
                    }
                    Rv32PackedShoutOp::Div => {
                        let rhs_is_zero = packed_cols
                            .get(5)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing rhs_is_zero".into()))?
                            .clone();
                        let lhs_sign = packed_cols
                            .get(6)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing lhs_sign".into()))?
                            .clone();
                        let rhs_sign = packed_cols
                            .get(7)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing rhs_sign".into()))?
                            .clone();
                        let q_abs = packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing q_abs".into()))?
                            .clone();
                        let q_is_zero = packed_cols
                            .get(9)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing q_is_zero".into()))?
                            .clone();
                        Box::new(Rv32PackedDivOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            lhs_sign,
                            rhs_sign,
                            rhs_is_zero,
                            q_abs,
                            q_is_zero,
                            lane.val.clone(),
                        ))
                    }
                    Rv32PackedShoutOp::Rem => {
                        let rhs_is_zero = packed_cols
                            .get(5)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing rhs_is_zero".into()))?
                            .clone();
                        let lhs_sign = packed_cols
                            .get(6)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing lhs_sign".into()))?
                            .clone();
                        let r_abs = packed_cols
                            .get(3)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing r_abs".into()))?
                            .clone();
                        let r_is_zero = packed_cols
                            .get(9)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing r_is_zero".into()))?
                            .clone();
                        Box::new(Rv32PackedRemOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            lhs.clone(),
                            lhs_sign,
                            rhs_is_zero,
                            r_abs,
                            r_is_zero,
                            lane.val.clone(),
                        ))
                    }
                    Rv32PackedShoutOp::Sll => {
                        let shamt_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(1).take(5).cloned().collect();
                        if shamt_bits.len() != 5 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SLL: expected 5 shamt bits, got {}",
                                shamt_bits.len()
                            )));
                        }
                        let carry_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(6).cloned().collect();
                        if carry_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SLL: expected 32 carry bits, got {}",
                                carry_bits.len()
                            )));
                        }
                        Box::new(Rv32PackedSllOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            lhs.clone(),
                            shamt_bits,
                            carry_bits,
                            lane.val.clone(),
                        ))
                    }
                    Rv32PackedShoutOp::Srl => {
                        let shamt_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(1).take(5).cloned().collect();
                        if shamt_bits.len() != 5 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SRL: expected 5 shamt bits, got {}",
                                shamt_bits.len()
                            )));
                        }
                        let rem_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(6).cloned().collect();
                        if rem_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SRL: expected 32 rem bits, got {}",
                                rem_bits.len()
                            )));
                        }
                        Box::new(Rv32PackedSrlOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            lhs.clone(),
                            shamt_bits,
                            rem_bits,
                            lane.val.clone(),
                        ))
                    }
                    Rv32PackedShoutOp::Sra => {
                        let shamt_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(1).take(5).cloned().collect();
                        if shamt_bits.len() != 5 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SRA: expected 5 shamt bits, got {}",
                                shamt_bits.len()
                            )));
                        }
                        let sign = packed_cols
                            .get(6)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 SRA: missing sign bit".into()))?
                            .clone();
                        let rem_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(7).cloned().collect();
                        if rem_bits.len() != 31 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SRA: expected 31 rem bits, got {}",
                                rem_bits.len()
                            )));
                        }
                        Box::new(Rv32PackedSraOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            lhs.clone(),
                            shamt_bits,
                            sign,
                            rem_bits,
                            lane.val.clone(),
                        ))
                    }
                    Rv32PackedShoutOp::Sltu => Box::new(Rv32PackedSltuOracleSparseTime::new(
                        r_cycle,
                        lane.has_lookup.clone(),
                        lhs.clone(),
                        rhs.clone(),
                        packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 SLTU: missing diff opening".into()))?
                            .clone(),
                        lane.val.clone(),
                    )),
                };
                let adapter_oracle: Box<dyn RoundOracle> = match op {
                    Rv32PackedShoutOp::And
                    | Rv32PackedShoutOp::Andn
                    | Rv32PackedShoutOp::Or
                    | Rv32PackedShoutOp::Xor => {
                        let weights = bitness_weights(r_cycle, 34, 0x4249_5457_4F50u64 + lut_idx as u64);
                        Box::new(Rv32PackedBitwiseAdapterOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            lhs,
                            rhs,
                            bitwise_lhs_digits,
                            bitwise_rhs_digits,
                            weights,
                        ))
                    }
                    Rv32PackedShoutOp::Add
                    | Rv32PackedShoutOp::Sub
                    | Rv32PackedShoutOp::Sll
                    | Rv32PackedShoutOp::Mul
                    | Rv32PackedShoutOp::Mulhu => Box::new(ZeroOracleSparseTime::new(r_cycle.len(), 2)),
                    Rv32PackedShoutOp::Mulh => {
                        let hi = packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULH: missing hi opening".into()))?
                            .clone();
                        let lhs_sign = packed_cols
                            .get(3)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULH: missing lhs_sign".into()))?
                            .clone();
                        let rhs_sign = packed_cols
                            .get(4)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULH: missing rhs_sign".into()))?
                            .clone();
                        let k = packed_cols
                            .get(5)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULH: missing k opening".into()))?
                            .clone();
                        let weights = bitness_weights(r_cycle, 2, 0x4D55_4C48_4144_5054u64 + lut_idx as u64);
                        let w = [weights[0], weights[1]];
                        Box::new(Rv32PackedMulhAdapterOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            lhs,
                            rhs,
                            lhs_sign,
                            rhs_sign,
                            hi,
                            k,
                            lane.val.clone(),
                            w,
                        ))
                    }
                    Rv32PackedShoutOp::Mulhsu => {
                        let hi = packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULHSU: missing hi opening".into()))?
                            .clone();
                        let lhs_sign = packed_cols
                            .get(3)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULHSU: missing lhs_sign".into()))?
                            .clone();
                        let borrow = packed_cols
                            .get(4)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULHSU: missing borrow".into()))?
                            .clone();
                        Box::new(Rv32PackedMulhsuAdapterOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            lhs,
                            rhs,
                            lhs_sign,
                            hi,
                            borrow,
                            lane.val.clone(),
                        ))
                    }
                    Rv32PackedShoutOp::Divu => {
                        let rem = packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIVU: missing rem opening".into()))?
                            .clone();
                        let rhs_is_zero = packed_cols
                            .get(4)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIVU: missing rhs_is_zero".into()))?
                            .clone();
                        let diff = packed_cols
                            .get(5)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIVU: missing diff".into()))?
                            .clone();
                        let diff_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(6).cloned().collect();
                        if diff_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 DIVU: expected 32 diff bits, got {}",
                                diff_bits.len()
                            )));
                        }
                        let weights = bitness_weights(r_cycle, 4, 0x4449_5655_4144_5054u64 + lut_idx as u64);
                        let w = [weights[0], weights[1], weights[2], weights[3]];
                        Box::new(Rv32PackedDivRemuAdapterOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            rhs,
                            rhs_is_zero,
                            rem,
                            diff,
                            diff_bits,
                            w,
                        ))
                    }
                    Rv32PackedShoutOp::Remu => {
                        let rhs_is_zero = packed_cols
                            .get(4)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REMU: missing rhs_is_zero".into()))?
                            .clone();
                        let diff = packed_cols
                            .get(5)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REMU: missing diff".into()))?
                            .clone();
                        let diff_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(6).cloned().collect();
                        if diff_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 REMU: expected 32 diff bits, got {}",
                                diff_bits.len()
                            )));
                        }
                        let weights = bitness_weights(r_cycle, 4, 0x4449_5655_4144_5054u64 + lut_idx as u64);
                        let w = [weights[0], weights[1], weights[2], weights[3]];
                        Box::new(Rv32PackedDivRemuAdapterOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            rhs,
                            rhs_is_zero,
                            lane.val.clone(),
                            diff,
                            diff_bits,
                            w,
                        ))
                    }
                    Rv32PackedShoutOp::Div => {
                        let rhs_is_zero = packed_cols
                            .get(5)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing rhs_is_zero".into()))?
                            .clone();
                        let lhs_sign = packed_cols
                            .get(6)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing lhs_sign".into()))?
                            .clone();
                        let rhs_sign = packed_cols
                            .get(7)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing rhs_sign".into()))?
                            .clone();
                        let q_abs = packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing q_abs".into()))?
                            .clone();
                        let r_abs = packed_cols
                            .get(3)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing r_abs".into()))?
                            .clone();
                        let q_is_zero = packed_cols
                            .get(9)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing q_is_zero".into()))?
                            .clone();
                        let diff = packed_cols
                            .get(10)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing diff".into()))?
                            .clone();
                        let diff_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(11).cloned().collect();
                        if diff_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 DIV: expected 32 diff bits, got {}",
                                diff_bits.len()
                            )));
                        }
                        let weights = bitness_weights(r_cycle, 7, 0x4449_565F_4144_5054u64 + lut_idx as u64);
                        let w = [
                            weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6],
                        ];
                        Box::new(Rv32PackedDivRemAdapterOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            lhs,
                            rhs,
                            rhs_is_zero,
                            lhs_sign,
                            rhs_sign,
                            q_abs.clone(),
                            r_abs,
                            q_abs,
                            q_is_zero,
                            diff,
                            diff_bits,
                            w,
                        ))
                    }
                    Rv32PackedShoutOp::Rem => {
                        let rhs_is_zero = packed_cols
                            .get(5)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing rhs_is_zero".into()))?
                            .clone();
                        let lhs_sign = packed_cols
                            .get(6)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing lhs_sign".into()))?
                            .clone();
                        let rhs_sign = packed_cols
                            .get(7)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing rhs_sign".into()))?
                            .clone();
                        let q_abs = packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing q_abs".into()))?
                            .clone();
                        let r_abs = packed_cols
                            .get(3)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing r_abs".into()))?
                            .clone();
                        let r_is_zero = packed_cols
                            .get(9)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing r_is_zero".into()))?
                            .clone();
                        let diff = packed_cols
                            .get(10)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing diff".into()))?
                            .clone();
                        let diff_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(11).cloned().collect();
                        if diff_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 REM: expected 32 diff bits, got {}",
                                diff_bits.len()
                            )));
                        }
                        let weights = bitness_weights(r_cycle, 7, 0x4449_565F_4144_5054u64 + lut_idx as u64);
                        let w = [
                            weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6],
                        ];
                        Box::new(Rv32PackedDivRemAdapterOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            lhs,
                            rhs,
                            rhs_is_zero,
                            lhs_sign,
                            rhs_sign,
                            q_abs,
                            r_abs.clone(),
                            r_abs,
                            r_is_zero,
                            diff,
                            diff_bits,
                            w,
                        ))
                    }
                    Rv32PackedShoutOp::Slt => {
                        let diff_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(5).cloned().collect();
                        if diff_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SLT: expected 32 diff bits, got {}",
                                diff_bits.len()
                            )));
                        }
                        Box::new(U32DecompOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            packed_cols
                                .get(2)
                                .ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 SLT: missing diff opening".into())
                                })?
                                .clone(),
                            diff_bits,
                        ))
                    }
                    Rv32PackedShoutOp::Srl => {
                        let shamt_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(1).take(5).cloned().collect();
                        if shamt_bits.len() != 5 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SRL: expected 5 shamt bits, got {}",
                                shamt_bits.len()
                            )));
                        }
                        let rem_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(6).cloned().collect();
                        if rem_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SRL: expected 32 rem bits, got {}",
                                rem_bits.len()
                            )));
                        }
                        Box::new(Rv32PackedSrlAdapterOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            shamt_bits,
                            rem_bits,
                        ))
                    }
                    Rv32PackedShoutOp::Sra => {
                        let shamt_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(1).take(5).cloned().collect();
                        if shamt_bits.len() != 5 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SRA: expected 5 shamt bits, got {}",
                                shamt_bits.len()
                            )));
                        }
                        let rem_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(7).cloned().collect();
                        if rem_bits.len() != 31 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SRA: expected 31 rem bits, got {}",
                                rem_bits.len()
                            )));
                        }
                        Box::new(Rv32PackedSraAdapterOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            shamt_bits,
                            rem_bits,
                        ))
                    }
                    Rv32PackedShoutOp::Eq => Box::new(Rv32PackedEqAdapterOracleSparseTime::new(
                        r_cycle,
                        lane.has_lookup.clone(),
                        lhs,
                        rhs,
                        packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 EQ: missing borrow bit".into()))?
                            .clone(),
                        {
                            let diff_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(3).cloned().collect();
                            if diff_bits.len() != 32 {
                                return Err(PiCcsError::InvalidInput(format!(
                                    "packed RV32 EQ: expected 32 diff bits, got {}",
                                    diff_bits.len()
                                )));
                            }
                            diff_bits
                        },
                    )),
                    Rv32PackedShoutOp::Neq => Box::new(Rv32PackedNeqAdapterOracleSparseTime::new(
                        r_cycle,
                        lane.has_lookup.clone(),
                        lhs,
                        rhs,
                        packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 NEQ: missing borrow bit".into()))?
                            .clone(),
                        {
                            let diff_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(3).cloned().collect();
                            if diff_bits.len() != 32 {
                                return Err(PiCcsError::InvalidInput(format!(
                                    "packed RV32 NEQ: expected 32 diff bits, got {}",
                                    diff_bits.len()
                                )));
                            }
                            diff_bits
                        },
                    )),
                    Rv32PackedShoutOp::Sltu => {
                        let diff_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(3).cloned().collect();
                        if diff_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SLTU: expected 32 diff bits, got {}",
                                diff_bits.len()
                            )));
                        }
                        Box::new(U32DecompOracleSparseTime::new(
                            r_cycle,
                            lane.has_lookup.clone(),
                            packed_cols
                                .get(2)
                                .ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 SLTU: missing diff opening".into())
                                })?
                                .clone(),
                            diff_bits,
                        ))
                    }
                };

                let (event_table_hash, event_table_hash_claim) = if time_bits > 0 {
                    let time_bits_cols: Vec<SparseIdxVec<K>> = lane.addr_bits.iter().take(time_bits).cloned().collect();

                    let lhs_col = packed_cols
                        .get(0)
                        .ok_or_else(|| PiCcsError::InvalidInput("event-table hash: missing lhs".into()))?
                        .clone();

                    let rhs_terms: Vec<(SparseIdxVec<K>, K)> = match op {
                        Rv32PackedShoutOp::Sll | Rv32PackedShoutOp::Srl | Rv32PackedShoutOp::Sra => {
                            let mut out: Vec<(SparseIdxVec<K>, K)> = Vec::with_capacity(5);
                            for i in 0..5usize {
                                let b = packed_cols
                                    .get(1 + i)
                                    .ok_or_else(|| {
                                        PiCcsError::InvalidInput("event-table hash: missing shamt bit".into())
                                    })?
                                    .clone();
                                out.push((b, K::from(F::from_u64(1u64 << i))));
                            }
                            out
                        }
                        _ => vec![(
                            packed_cols
                                .get(1)
                                .ok_or_else(|| PiCcsError::InvalidInput("event-table hash: missing rhs".into()))?
                                .clone(),
                            K::ONE,
                        )],
                    };

                    let (oracle, claim) = ShoutEventTableHashOracleSparseTime::new(
                        &r_cycle[..time_bits],
                        time_bits_cols,
                        lane.has_lookup.clone(),
                        lane.val.clone(),
                        lhs_col,
                        rhs_terms,
                        event_alpha,
                        event_beta,
                        event_gamma,
                    );
                    (Some(Box::new(oracle) as Box<dyn RoundOracle>), Some(claim))
                } else {
                    (None, None)
                };

                lanes.push(RouteAShoutTimeLaneOracles {
                    value: value_oracle,
                    // Enforce correctness: claim must be 0.
                    value_claim: K::ZERO,
                    adapter: adapter_oracle,
                    adapter_claim: K::ZERO,
                    event_table_hash,
                    event_table_hash_claim,
                });
            } else {
                let (value_oracle, value_claim) =
                    ShoutValueOracleSparse::new(r_cycle, lane.has_lookup.clone(), lane.val.clone());

                let (adapter_oracle, adapter_claim) = IndexAdapterOracleSparseTime::new_with_gate(
                    r_cycle,
                    lane.has_lookup.clone(),
                    lane.addr_bits.clone(),
                    r_addr,
                );

                lanes.push(RouteAShoutTimeLaneOracles {
                    value: Box::new(value_oracle),
                    value_claim,
                    adapter: Box::new(adapter_oracle),
                    adapter_claim,
                    event_table_hash: None,
                    event_table_hash_claim: None,
                });
            }
        }

        let bitness: Vec<Box<dyn RoundOracle>> = if is_packed {
            // Packed RV32: boolean columns depend on the packed op.
            let mut bit_cols: Vec<SparseIdxVec<K>> = Vec::new();
            for lane in decoded.lanes.iter() {
                // Event-table packed: time bits must be boolean.
                if packed_time_bits > 0 {
                    bit_cols.extend(lane.addr_bits.iter().take(packed_time_bits).cloned());
                }
                let packed_cols: &[SparseIdxVec<K>] = lane
                    .addr_bits
                    .get(packed_time_bits..)
                    .ok_or_else(|| PiCcsError::InvalidInput("packed RV32: missing packed cols".into()))?;
                match packed_op {
                    Some(
                        Rv32PackedShoutOp::And
                        | Rv32PackedShoutOp::Andn
                        | Rv32PackedShoutOp::Or
                        | Rv32PackedShoutOp::Xor,
                    ) => {
                        bit_cols.push(lane.has_lookup.clone());
                    }
                    Some(Rv32PackedShoutOp::Add | Rv32PackedShoutOp::Sub) => {
                        let aux = packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32: missing aux column".into()))?
                            .clone();
                        bit_cols.push(aux);
                        bit_cols.push(lane.has_lookup.clone());
                    }
                    Some(Rv32PackedShoutOp::Eq | Rv32PackedShoutOp::Neq) => {
                        let borrow = packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 EQ/NEQ: missing borrow bit".into()))?
                            .clone();
                        let diff_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(3).cloned().collect();
                        if diff_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 EQ/NEQ: expected 32 diff bits, got {}",
                                diff_bits.len()
                            )));
                        }
                        bit_cols.push(lane.has_lookup.clone());
                        bit_cols.push(lane.val.clone());
                        bit_cols.push(borrow);
                        bit_cols.extend(diff_bits);
                    }
                    Some(Rv32PackedShoutOp::Mul) => {
                        let carry_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(2).cloned().collect();
                        if carry_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 MUL: expected 32 carry bits, got {}",
                                carry_bits.len()
                            )));
                        }
                        bit_cols.push(lane.has_lookup.clone());
                        bit_cols.extend(carry_bits);
                    }
                    Some(Rv32PackedShoutOp::Mulhu) => {
                        let lo_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(2).cloned().collect();
                        if lo_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 MULHU: expected 32 lo bits, got {}",
                                lo_bits.len()
                            )));
                        }
                        bit_cols.push(lane.has_lookup.clone());
                        bit_cols.extend(lo_bits);
                    }
                    Some(Rv32PackedShoutOp::Mulh) => {
                        let lhs_sign = packed_cols
                            .get(3)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULH: missing lhs_sign bit".into()))?
                            .clone();
                        let rhs_sign = packed_cols
                            .get(4)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULH: missing rhs_sign bit".into()))?
                            .clone();
                        let lo_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(6).cloned().collect();
                        if lo_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 MULH: expected 32 lo bits, got {}",
                                lo_bits.len()
                            )));
                        }
                        bit_cols.push(lane.has_lookup.clone());
                        bit_cols.push(lhs_sign);
                        bit_cols.push(rhs_sign);
                        bit_cols.extend(lo_bits);
                    }
                    Some(Rv32PackedShoutOp::Mulhsu) => {
                        let lhs_sign = packed_cols
                            .get(3)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULHSU: missing lhs_sign bit".into()))?
                            .clone();
                        let borrow = packed_cols
                            .get(4)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULHSU: missing borrow bit".into()))?
                            .clone();
                        let lo_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(5).cloned().collect();
                        if lo_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 MULHSU: expected 32 lo bits, got {}",
                                lo_bits.len()
                            )));
                        }
                        bit_cols.push(lane.has_lookup.clone());
                        bit_cols.push(lhs_sign);
                        bit_cols.push(borrow);
                        bit_cols.extend(lo_bits);
                    }
                    Some(Rv32PackedShoutOp::Slt) => {
                        let lhs_sign = packed_cols
                            .get(3)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 SLT: missing lhs_sign bit".into()))?
                            .clone();
                        let rhs_sign = packed_cols
                            .get(4)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 SLT: missing rhs_sign bit".into()))?
                            .clone();
                        let diff_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(5).cloned().collect();
                        if diff_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SLT: expected 32 diff bits, got {}",
                                diff_bits.len()
                            )));
                        }
                        bit_cols.push(lane.val.clone());
                        bit_cols.push(lane.has_lookup.clone());
                        bit_cols.push(lhs_sign);
                        bit_cols.push(rhs_sign);
                        bit_cols.extend(diff_bits);
                    }
                    Some(Rv32PackedShoutOp::Sll) => {
                        let shamt_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(1).take(5).cloned().collect();
                        if shamt_bits.len() != 5 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SLL: expected 5 shamt bits, got {}",
                                shamt_bits.len()
                            )));
                        }
                        let carry_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(6).cloned().collect();
                        if carry_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SLL: expected 32 carry bits, got {}",
                                carry_bits.len()
                            )));
                        }
                        bit_cols.push(lane.has_lookup.clone());
                        bit_cols.extend(shamt_bits);
                        bit_cols.extend(carry_bits);
                    }
                    Some(Rv32PackedShoutOp::Srl) => {
                        let shamt_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(1).take(5).cloned().collect();
                        if shamt_bits.len() != 5 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SRL: expected 5 shamt bits, got {}",
                                shamt_bits.len()
                            )));
                        }
                        let rem_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(6).cloned().collect();
                        if rem_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SRL: expected 32 rem bits, got {}",
                                rem_bits.len()
                            )));
                        }
                        bit_cols.push(lane.has_lookup.clone());
                        bit_cols.extend(shamt_bits);
                        bit_cols.extend(rem_bits);
                    }
                    Some(Rv32PackedShoutOp::Sra) => {
                        let shamt_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(1).take(5).cloned().collect();
                        if shamt_bits.len() != 5 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SRA: expected 5 shamt bits, got {}",
                                shamt_bits.len()
                            )));
                        }
                        let sign = packed_cols
                            .get(6)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 SRA: missing sign bit".into()))?
                            .clone();
                        let rem_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(7).cloned().collect();
                        if rem_bits.len() != 31 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SRA: expected 31 rem bits, got {}",
                                rem_bits.len()
                            )));
                        }
                        bit_cols.push(lane.has_lookup.clone());
                        bit_cols.extend(shamt_bits);
                        bit_cols.push(sign);
                        bit_cols.extend(rem_bits);
                    }
                    Some(Rv32PackedShoutOp::Sltu) => {
                        let diff_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(3).cloned().collect();
                        if diff_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 SLTU: expected 32 diff bits, got {}",
                                diff_bits.len()
                            )));
                        }
                        bit_cols.push(lane.val.clone());
                        bit_cols.push(lane.has_lookup.clone());
                        bit_cols.extend(diff_bits);
                    }
                    Some(Rv32PackedShoutOp::Divu | Rv32PackedShoutOp::Remu) => {
                        let rhs_is_zero = packed_cols
                            .get(4)
                            .ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 DIVU/REMU: missing rhs_is_zero".into())
                            })?
                            .clone();
                        let diff_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(6).cloned().collect();
                        if diff_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 DIVU/REMU: expected 32 diff bits, got {}",
                                diff_bits.len()
                            )));
                        }
                        bit_cols.push(lane.has_lookup.clone());
                        bit_cols.push(rhs_is_zero);
                        bit_cols.extend(diff_bits);
                    }
                    Some(Rv32PackedShoutOp::Div) => {
                        let rhs_is_zero = packed_cols
                            .get(5)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing rhs_is_zero".into()))?
                            .clone();
                        let lhs_sign = packed_cols
                            .get(6)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing lhs_sign".into()))?
                            .clone();
                        let rhs_sign = packed_cols
                            .get(7)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing rhs_sign".into()))?
                            .clone();
                        let q_is_zero = packed_cols
                            .get(9)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing q_is_zero".into()))?
                            .clone();
                        let diff_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(11).cloned().collect();
                        if diff_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 DIV: expected 32 diff bits, got {}",
                                diff_bits.len()
                            )));
                        }
                        bit_cols.push(lane.has_lookup.clone());
                        bit_cols.push(rhs_is_zero);
                        bit_cols.push(lhs_sign);
                        bit_cols.push(rhs_sign);
                        bit_cols.push(q_is_zero);
                        bit_cols.extend(diff_bits);
                    }
                    Some(Rv32PackedShoutOp::Rem) => {
                        let rhs_is_zero = packed_cols
                            .get(5)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing rhs_is_zero".into()))?
                            .clone();
                        let lhs_sign = packed_cols
                            .get(6)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing lhs_sign".into()))?
                            .clone();
                        let rhs_sign = packed_cols
                            .get(7)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing rhs_sign".into()))?
                            .clone();
                        let r_is_zero = packed_cols
                            .get(9)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing r_is_zero".into()))?
                            .clone();
                        let diff_bits: Vec<SparseIdxVec<K>> = packed_cols.iter().skip(11).cloned().collect();
                        if diff_bits.len() != 32 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "packed RV32 REM: expected 32 diff bits, got {}",
                                diff_bits.len()
                            )));
                        }
                        bit_cols.push(lane.has_lookup.clone());
                        bit_cols.push(rhs_is_zero);
                        bit_cols.push(lhs_sign);
                        bit_cols.push(rhs_sign);
                        bit_cols.push(r_is_zero);
                        bit_cols.extend(diff_bits);
                    }
                    None => {
                        return Err(PiCcsError::ProtocolError(
                            "packed_op drift: is_packed=true but packed_op=None".into(),
                        ));
                    }
                }
            }
            let weights = bitness_weights(r_cycle, bit_cols.len(), 0x5348_4F55_54u64 + lut_idx as u64);
            let bitness_oracle = LazyWeightedBitnessOracleSparseTime::new_with_cycle(r_cycle, bit_cols, weights);
            vec![Box::new(bitness_oracle)]
        } else {
            let mut bit_cols: Vec<SparseIdxVec<K>> = Vec::with_capacity(lane_count * (ell_addr + 1));
            for lane in decoded.lanes.iter() {
                bit_cols.extend(lane.addr_bits.iter().cloned());
                bit_cols.push(lane.has_lookup.clone());
            }
            let weights = bitness_weights(r_cycle, bit_cols.len(), 0x5348_4F55_54u64 + lut_idx as u64);
            let bitness_oracle = LazyWeightedBitnessOracleSparseTime::new_with_cycle(r_cycle, bit_cols, weights);
            vec![Box::new(bitness_oracle)]
        };

        shout_oracles.push(RouteAShoutTimeOracles {
            lanes,
            bitness,
            ell_addr,
        });
    }

    let mut twist_oracles = Vec::with_capacity(step.mem_instances.len());
    for (mem_idx, ((mem_inst, _mem_wit), pre)) in step.mem_instances.iter().zip(twist_pre.iter()).enumerate() {
        let init_at_r_addr = eval_init_at_r_addr(&mem_inst.init, mem_inst.k, &pre.addr_pre.r_addr)?;
        let ell_addr = mem_inst.d * mem_inst.ell;
        if pre.addr_pre.r_addr.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist(Route A): r_addr.len()={} != ell_addr={}",
                pre.addr_pre.r_addr.len(),
                ell_addr
            )));
        }

        if pre.decoded.lanes.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist(Route A): decoded lanes empty at mem_idx={mem_idx}"
            )));
        }

        let inc_terms_at_r_addr = build_twist_inc_terms_at_r_addr(&pre.decoded.lanes, &pre.addr_pre.r_addr);

        let mut read_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(pre.decoded.lanes.len());
        let mut write_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(pre.decoded.lanes.len());
        for lane in pre.decoded.lanes.iter() {
            read_oracles.push(Box::new(TwistReadCheckOracleSparseTime::new_with_inc_terms(
                r_cycle,
                lane.has_read.clone(),
                lane.rv.clone(),
                lane.ra_bits.clone(),
                &pre.addr_pre.r_addr,
                init_at_r_addr,
                inc_terms_at_r_addr.clone(),
            )));
            write_oracles.push(Box::new(TwistWriteCheckOracleSparseTime::new_with_inc_terms(
                r_cycle,
                lane.has_write.clone(),
                lane.wv.clone(),
                lane.inc_at_write_addr.clone(),
                lane.wa_bits.clone(),
                &pre.addr_pre.r_addr,
                init_at_r_addr,
                inc_terms_at_r_addr.clone(),
            )));
        }
        let read_check: Box<dyn RoundOracle> = Box::new(SumRoundOracle::new(read_oracles));
        let write_check: Box<dyn RoundOracle> = Box::new(SumRoundOracle::new(write_oracles));

        let lane_count = pre.decoded.lanes.len();
        let mut bit_cols: Vec<SparseIdxVec<K>> = Vec::with_capacity(lane_count * (2 * ell_addr + 2));
        for lane in pre.decoded.lanes.iter() {
            bit_cols.extend(lane.ra_bits.iter().cloned());
            bit_cols.extend(lane.wa_bits.iter().cloned());
            bit_cols.push(lane.has_read.clone());
            bit_cols.push(lane.has_write.clone());
        }
        let weights = bitness_weights(r_cycle, bit_cols.len(), 0x5457_4953_54u64 + mem_idx as u64);
        let bitness_oracle = LazyWeightedBitnessOracleSparseTime::new_with_cycle(r_cycle, bit_cols, weights);
        let bitness: Vec<Box<dyn RoundOracle>> = vec![Box::new(bitness_oracle)];

        twist_oracles.push(RouteATwistTimeOracles {
            read_check,
            write_check,
            bitness,
            ell_addr,
        });
    }

    Ok(RouteAMemoryOracles {
        shout: shout_oracles,
        shout_event_trace_hash,
        twist: twist_oracles,
    })
}

pub struct RouteAShoutTimeClaimsGuard<'a> {
    pub lane_ranges: Vec<core::ops::Range<usize>>,
    pub lanes: Vec<RouteAShoutTimeLaneClaims<'a>>,
    pub bitness: Vec<Vec<Box<dyn RoundOracle>>>,
}

pub struct RouteAShoutTimeLaneClaims<'a> {
    pub value_prefix: RoundOraclePrefix<'a>,
    pub adapter_prefix: RoundOraclePrefix<'a>,
    pub event_table_hash_prefix: Option<RoundOraclePrefix<'a>>,
    pub value_claim: K,
    pub adapter_claim: K,
    pub event_table_hash_claim: Option<K>,
}

pub fn build_route_a_shout_time_claims_guard<'a>(
    shout_oracles: &'a mut [RouteAShoutTimeOracles],
    ell_n: usize,
) -> RouteAShoutTimeClaimsGuard<'a> {
    let mut lane_ranges: Vec<core::ops::Range<usize>> = Vec::with_capacity(shout_oracles.len());
    let mut lanes: Vec<RouteAShoutTimeLaneClaims<'a>> = Vec::new();
    let mut bitness: Vec<Vec<Box<dyn RoundOracle>>> = Vec::with_capacity(shout_oracles.len());

    for o in shout_oracles.iter_mut() {
        bitness.push(core::mem::take(&mut o.bitness));
        let start = lanes.len();
        for lane in o.lanes.iter_mut() {
            lanes.push(RouteAShoutTimeLaneClaims {
                value_prefix: RoundOraclePrefix::new(lane.value.as_mut(), ell_n),
                adapter_prefix: RoundOraclePrefix::new(lane.adapter.as_mut(), ell_n),
                event_table_hash_prefix: lane
                    .event_table_hash
                    .as_deref_mut()
                    .map(|o| RoundOraclePrefix::new(o, ell_n)),
                value_claim: lane.value_claim,
                adapter_claim: lane.adapter_claim,
                event_table_hash_claim: lane.event_table_hash_claim,
            });
        }
        let end = lanes.len();
        lane_ranges.push(start..end);
    }

    RouteAShoutTimeClaimsGuard {
        lane_ranges,
        lanes,
        bitness,
    }
}

pub struct ShoutRouteAProtocol<'a> {
    guard: RouteAShoutTimeClaimsGuard<'a>,
}

impl<'a> ShoutRouteAProtocol<'a> {
    pub fn new(shout_oracles: &'a mut [RouteAShoutTimeOracles], ell_n: usize) -> Self {
        Self {
            guard: build_route_a_shout_time_claims_guard(shout_oracles, ell_n),
        }
    }
}

impl<'o> TimeBatchedClaims for ShoutRouteAProtocol<'o> {
    fn append_time_claims<'a>(
        &'a mut self,
        _ell_n: usize,
        claimed_sums: &mut Vec<K>,
        degree_bounds: &mut Vec<usize>,
        labels: &mut Vec<&'static [u8]>,
        claim_is_dynamic: &mut Vec<bool>,
        claims: &mut Vec<BatchedClaim<'a>>,
    ) {
        append_route_a_shout_time_claims(
            &mut self.guard,
            claimed_sums,
            degree_bounds,
            labels,
            claim_is_dynamic,
            claims,
        );
    }
}

pub fn append_route_a_shout_time_claims<'a>(
    guard: &'a mut RouteAShoutTimeClaimsGuard<'_>,
    claimed_sums: &mut Vec<K>,
    degree_bounds: &mut Vec<usize>,
    labels: &mut Vec<&'static [u8]>,
    claim_is_dynamic: &mut Vec<bool>,
    claims: &mut Vec<BatchedClaim<'a>>,
) {
    if guard.lane_ranges.is_empty() {
        return;
    }
    if guard.bitness.len() != guard.lane_ranges.len() {
        panic!("shout bitness count mismatch");
    }

    let mut lane_ranges_iter = guard.lane_ranges.iter();
    let mut next_end = lane_ranges_iter.next().expect("non-empty").end;
    let mut bitness_iter = guard.bitness.iter_mut();

    for (lane_idx, lane) in guard.lanes.iter_mut().enumerate() {
        claimed_sums.push(lane.value_claim);
        degree_bounds.push(lane.value_prefix.degree_bound());
        labels.push(b"shout/value");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: &mut lane.value_prefix,
            claimed_sum: lane.value_claim,
            label: b"shout/value",
        });

        claimed_sums.push(lane.adapter_claim);
        degree_bounds.push(lane.adapter_prefix.degree_bound());
        labels.push(b"shout/adapter");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: &mut lane.adapter_prefix,
            claimed_sum: lane.adapter_claim,
            label: b"shout/adapter",
        });

        if let Some(prefix) = lane.event_table_hash_prefix.as_mut() {
            let claim = lane
                .event_table_hash_claim
                .expect("event_table_hash_claim missing");
            claimed_sums.push(claim);
            degree_bounds.push(prefix.degree_bound());
            labels.push(b"shout/event_table_hash");
            claim_is_dynamic.push(true);
            claims.push(BatchedClaim {
                oracle: prefix,
                claimed_sum: claim,
                label: b"shout/event_table_hash",
            });
        }

        if lane_idx + 1 == next_end {
            let bitness_vec = bitness_iter.next().expect("shout bitness idx drift");
            for bit_oracle in bitness_vec.iter_mut() {
                claimed_sums.push(K::ZERO);
                degree_bounds.push(bit_oracle.degree_bound());
                labels.push(b"shout/bitness");
                claim_is_dynamic.push(false);
                claims.push(BatchedClaim {
                    oracle: bit_oracle.as_mut(),
                    claimed_sum: K::ZERO,
                    label: b"shout/bitness",
                });
            }

            next_end = lane_ranges_iter.next().map(|r| r.end).unwrap_or(usize::MAX);
        }
    }

    if bitness_iter.next().is_some() {
        panic!("shout bitness not fully consumed");
    }
}

pub struct RouteATwistTimeClaimsGuard<'a> {
    pub read_check_prefixes: Vec<RoundOraclePrefix<'a>>,
    pub write_check_prefixes: Vec<RoundOraclePrefix<'a>>,
    pub read_check_claims: Vec<K>,
    pub write_check_claims: Vec<K>,
    pub bitness: Vec<Vec<Box<dyn RoundOracle>>>,
}

pub fn build_route_a_twist_time_claims_guard<'a>(
    twist_oracles: &'a mut [RouteATwistTimeOracles],
    ell_n: usize,
    read_check_claims: Vec<K>,
    write_check_claims: Vec<K>,
) -> RouteATwistTimeClaimsGuard<'a> {
    let mut read_check_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(twist_oracles.len());
    let mut write_check_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(twist_oracles.len());
    let mut bitness: Vec<Vec<Box<dyn RoundOracle>>> = Vec::with_capacity(twist_oracles.len());

    if read_check_claims.len() != twist_oracles.len() {
        panic!(
            "twist read-check claim count mismatch (claims={}, oracles={})",
            read_check_claims.len(),
            twist_oracles.len()
        );
    }
    if write_check_claims.len() != twist_oracles.len() {
        panic!(
            "twist write-check claim count mismatch (claims={}, oracles={})",
            write_check_claims.len(),
            twist_oracles.len()
        );
    }

    for o in twist_oracles.iter_mut() {
        bitness.push(core::mem::take(&mut o.bitness));
        read_check_prefixes.push(RoundOraclePrefix::new(o.read_check.as_mut(), ell_n));
        write_check_prefixes.push(RoundOraclePrefix::new(o.write_check.as_mut(), ell_n));
    }

    RouteATwistTimeClaimsGuard {
        read_check_prefixes,
        write_check_prefixes,
        read_check_claims,
        write_check_claims,
        bitness,
    }
}

pub fn append_route_a_twist_time_claims<'a>(
    guard: &'a mut RouteATwistTimeClaimsGuard<'_>,
    claimed_sums: &mut Vec<K>,
    degree_bounds: &mut Vec<usize>,
    labels: &mut Vec<&'static [u8]>,
    claim_is_dynamic: &mut Vec<bool>,
    claims: &mut Vec<BatchedClaim<'a>>,
) {
    for (((read_check_time, write_check_time), bitness_vec), (read_claim, write_claim)) in guard
        .read_check_prefixes
        .iter_mut()
        .zip(guard.write_check_prefixes.iter_mut())
        .zip(guard.bitness.iter_mut())
        .zip(
            guard
                .read_check_claims
                .iter()
                .zip(guard.write_check_claims.iter()),
        )
    {
        claimed_sums.push(*read_claim);
        degree_bounds.push(read_check_time.degree_bound());
        labels.push(b"twist/read_check");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: read_check_time,
            claimed_sum: *read_claim,
            label: b"twist/read_check",
        });

        claimed_sums.push(*write_claim);
        degree_bounds.push(write_check_time.degree_bound());
        labels.push(b"twist/write_check");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: write_check_time,
            claimed_sum: *write_claim,
            label: b"twist/write_check",
        });

        for bit_oracle in bitness_vec.iter_mut() {
            claimed_sums.push(K::ZERO);
            degree_bounds.push(bit_oracle.degree_bound());
            labels.push(b"twist/bitness");
            claim_is_dynamic.push(false);
            claims.push(BatchedClaim {
                oracle: bit_oracle.as_mut(),
                claimed_sum: K::ZERO,
                label: b"twist/bitness",
            });
        }
    }
}

pub struct TwistRouteAProtocol<'a> {
    guard: RouteATwistTimeClaimsGuard<'a>,
}

impl<'a> TwistRouteAProtocol<'a> {
    pub fn new(
        twist_oracles: &'a mut [RouteATwistTimeOracles],
        ell_n: usize,
        read_check_claims: Vec<K>,
        write_check_claims: Vec<K>,
    ) -> Self {
        Self {
            guard: build_route_a_twist_time_claims_guard(twist_oracles, ell_n, read_check_claims, write_check_claims),
        }
    }
}

impl<'o> TimeBatchedClaims for TwistRouteAProtocol<'o> {
    fn append_time_claims<'a>(
        &'a mut self,
        _ell_n: usize,
        claimed_sums: &mut Vec<K>,
        degree_bounds: &mut Vec<usize>,
        labels: &mut Vec<&'static [u8]>,
        claim_is_dynamic: &mut Vec<bool>,
        claims: &mut Vec<BatchedClaim<'a>>,
    ) {
        append_route_a_twist_time_claims(
            &mut self.guard,
            claimed_sums,
            degree_bounds,
            labels,
            claim_is_dynamic,
            claims,
        );
    }
}

pub(crate) fn finalize_route_a_memory_prover(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    cpu_bus: Option<&BusLayout>,
    s: &CcsStructure<F>,
    step: &StepWitnessBundle<Cmt, F, K>,
    prev_step: Option<&StepWitnessBundle<Cmt, F, K>>,
    prev_twist_decoded: Option<&[TwistDecodedColsSparse]>,
    oracles: &mut RouteAMemoryOracles,
    shout_addr_pre: &ShoutAddrPreProof<K>,
    twist_pre: &[TwistAddrPreProverData],
    r_time: &[K],
    m_in: usize,
    step_idx: usize,
) -> Result<MemSidecarProof<Cmt, F, K>, PiCcsError> {
    let has_prev = prev_step.is_some();
    if has_prev != prev_twist_decoded.is_some() {
        return Err(PiCcsError::InvalidInput(format!(
            "Twist rollover decoded cache mismatch: prev_step.is_some()={} but prev_twist_decoded.is_some()={}",
            has_prev,
            prev_twist_decoded.is_some()
        )));
    }
    let total_lanes: usize = step
        .lut_instances
        .iter()
        .map(|(inst, _)| inst.lanes.max(1))
        .sum();
    if shout_addr_pre.claimed_sums.len() != total_lanes {
        return Err(PiCcsError::InvalidInput(format!(
            "shout addr-pre proof count mismatch (expected claimed_sums.len()=total_lanes={}, got {})",
            total_lanes,
            shout_addr_pre.claimed_sums.len(),
        )));
    }
    {
        let mut lane_ell_addr: Vec<u32> = Vec::with_capacity(total_lanes);
        let mut required_ell_addrs: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for (lut_inst, _lut_wit) in step.lut_instances.iter().map(|(inst, wit)| (inst, wit)) {
            let inst_ell_addr = lut_inst.d * lut_inst.ell;
            let inst_ell_addr_u32 = u32::try_from(inst_ell_addr)
                .map_err(|_| PiCcsError::InvalidInput("Shout: ell_addr overflows u32".into()))?;
            required_ell_addrs.insert(inst_ell_addr_u32);
            for _lane_idx in 0..lut_inst.lanes.max(1) {
                lane_ell_addr.push(inst_ell_addr_u32);
            }
        }
        if lane_ell_addr.len() != total_lanes {
            return Err(PiCcsError::ProtocolError(
                "shout addr-pre lane indexing drift (lane_ell_addr)".into(),
            ));
        }

        if shout_addr_pre.groups.len() != required_ell_addrs.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "shout addr-pre group count mismatch (expected {}, got {})",
                required_ell_addrs.len(),
                shout_addr_pre.groups.len()
            )));
        }
        let required_list: Vec<u32> = required_ell_addrs.into_iter().collect();
        let mut seen_active: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for (idx, group) in shout_addr_pre.groups.iter().enumerate() {
            let expected_ell_addr = required_list[idx];
            if group.ell_addr != expected_ell_addr {
                return Err(PiCcsError::InvalidInput(format!(
                    "shout addr-pre groups not sorted or mismatched: groups[{idx}].ell_addr={} but expected {expected_ell_addr}",
                    group.ell_addr
                )));
            }
            if group.r_addr.len() != group.ell_addr as usize {
                return Err(PiCcsError::InvalidInput(format!(
                    "shout addr-pre group ell_addr={} has r_addr.len()={}, expected {}",
                    group.ell_addr,
                    group.r_addr.len(),
                    group.ell_addr
                )));
            }
            if group.round_polys.len() != group.active_lanes.len() {
                return Err(PiCcsError::InvalidInput(format!(
                    "shout addr-pre group ell_addr={} round_polys.len()={}, expected active_lanes.len()={}",
                    group.ell_addr,
                    group.round_polys.len(),
                    group.active_lanes.len()
                )));
            }
            for (pos, &lane_idx) in group.active_lanes.iter().enumerate() {
                let lane_idx_usize = lane_idx as usize;
                if lane_idx_usize >= total_lanes {
                    return Err(PiCcsError::InvalidInput(
                        "shout addr-pre active_lanes has index out of range".into(),
                    ));
                }
                if lane_ell_addr[lane_idx_usize] != group.ell_addr {
                    return Err(PiCcsError::InvalidInput(format!(
                        "shout addr-pre active_lanes contains lane_idx={} with ell_addr={}, but group ell_addr={}",
                        lane_idx, lane_ell_addr[lane_idx_usize], group.ell_addr
                    )));
                }
                if pos > 0 && group.active_lanes[pos - 1] >= lane_idx {
                    return Err(PiCcsError::InvalidInput(
                        "shout addr-pre active_lanes must be strictly increasing".into(),
                    ));
                }
                if !seen_active.insert(lane_idx) {
                    return Err(PiCcsError::InvalidInput(
                        "shout addr-pre active_lanes contains duplicates across groups".into(),
                    ));
                }
            }
            for (pos, rounds) in group.round_polys.iter().enumerate() {
                if rounds.len() != group.ell_addr as usize {
                    return Err(PiCcsError::InvalidInput(format!(
                        "shout addr-pre group ell_addr={} round_polys[{pos}].len()={}, expected {}",
                        group.ell_addr,
                        rounds.len(),
                        group.ell_addr
                    )));
                }
            }
        }
    }
    if twist_pre.len() != step.mem_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "twist pre-time count mismatch (expected {}, got {})",
            step.mem_instances.len(),
            twist_pre.len()
        )));
    }
    if oracles.twist.len() != step.mem_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "twist oracle count mismatch (expected {}, got {})",
            step.mem_instances.len(),
            oracles.twist.len()
        )));
    }

    match cpu_bus {
        Some(_) => {
            for (idx, (lut_inst, lut_wit)) in step.lut_instances.iter().enumerate() {
                if !lut_inst.comms.is_empty() || !lut_wit.mats.is_empty() {
                    return Err(PiCcsError::InvalidInput(format!(
                        "shared CPU bus requires metadata-only Shout instances (comms/mats must be empty, lut_idx={idx})"
                    )));
                }
            }
            for (idx, (mem_inst, mem_wit)) in step.mem_instances.iter().enumerate() {
                if !mem_inst.comms.is_empty() || !mem_wit.mats.is_empty() {
                    return Err(PiCcsError::InvalidInput(format!(
                        "shared CPU bus requires metadata-only Twist instances (comms/mats must be empty, mem_idx={idx})"
                    )));
                }
            }
            if let Some(prev) = prev_step {
                if prev.mem_instances.len() != step.mem_instances.len() {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Twist rollover requires stable mem instance count: prev has {}, current has {}",
                        prev.mem_instances.len(),
                        step.mem_instances.len()
                    )));
                }
                for (idx, (mem_inst, mem_wit)) in prev.mem_instances.iter().enumerate() {
                    if !mem_inst.comms.is_empty() || !mem_wit.mats.is_empty() {
                        return Err(PiCcsError::InvalidInput(format!(
                            "shared CPU bus requires metadata-only Twist instances (comms/mats must be empty, prev mem_idx={idx})"
                        )));
                    }
                }
            }
        }
        None => {
            for (idx, (lut_inst, lut_wit)) in step.lut_instances.iter().enumerate() {
                if lut_inst.comms.is_empty() || lut_wit.mats.is_empty() {
                    return Err(PiCcsError::InvalidInput(format!(
                        "no-shared-bus Route-A requires committed Shout instances (non-empty comms/mats, lut_idx={idx})"
                    )));
                }
                if lut_inst.comms.len() != lut_wit.mats.len() {
                    return Err(PiCcsError::InvalidInput(format!(
                        "no-shared-bus Route-A requires comms.len()==mats.len() for Shout (lut_idx={idx}, comms.len()={}, mats.len()={})",
                        lut_inst.comms.len(),
                        lut_wit.mats.len()
                    )));
                }
                let ell_addr = lut_inst.d * lut_inst.ell;
                let lanes = lut_inst.lanes.max(1);
                let page_ell_addrs = plan_shout_addr_pages(s.m, m_in, lut_inst.steps, ell_addr, lanes)?;
                if lut_wit.mats.len() != page_ell_addrs.len() {
                    return Err(PiCcsError::InvalidInput(format!(
                        "no-shared-bus Route-A requires Shout paging mat count to match the deterministic plan (lut_idx={idx}, expected {}, got {})",
                        page_ell_addrs.len(),
                        lut_wit.mats.len(),
                    )));
                }
            }
            for (idx, (mem_inst, mem_wit)) in step.mem_instances.iter().enumerate() {
                if mem_inst.comms.is_empty() || mem_wit.mats.is_empty() {
                    return Err(PiCcsError::InvalidInput(format!(
                        "no-shared-bus Route-A requires committed Twist instances (non-empty comms/mats, mem_idx={idx})"
                    )));
                }
                if mem_inst.comms.len() != 1 || mem_wit.mats.len() != 1 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "no-shared-bus Route-A requires exactly 1 comm/mat per Twist instance (mem_idx={idx}, comms.len()={}, mats.len()={})",
                        mem_inst.comms.len(),
                        mem_wit.mats.len()
                    )));
                }
            }
            if let Some(prev) = prev_step {
                if prev.mem_instances.len() != step.mem_instances.len() {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Twist rollover requires stable mem instance count: prev has {}, current has {}",
                        prev.mem_instances.len(),
                        step.mem_instances.len()
                    )));
                }
                for (idx, (lut_inst, lut_wit)) in prev.lut_instances.iter().enumerate() {
                    if lut_inst.comms.is_empty() || lut_wit.mats.is_empty() {
                        return Err(PiCcsError::InvalidInput(format!(
                            "no-shared-bus Route-A requires committed Shout instances (non-empty comms/mats, prev lut_idx={idx})"
                        )));
                    }
                    if lut_inst.comms.len() != lut_wit.mats.len() {
                        return Err(PiCcsError::InvalidInput(format!(
                            "no-shared-bus Route-A requires comms.len()==mats.len() for Shout (prev lut_idx={idx}, comms.len()={}, mats.len()={})",
                            lut_inst.comms.len(),
                            lut_wit.mats.len()
                        )));
                    }
                    let ell_addr = lut_inst.d * lut_inst.ell;
                    let lanes = lut_inst.lanes.max(1);
                    let page_ell_addrs = plan_shout_addr_pages(s.m, m_in, lut_inst.steps, ell_addr, lanes)?;
                    if lut_wit.mats.len() != page_ell_addrs.len() {
                        return Err(PiCcsError::InvalidInput(format!(
                            "no-shared-bus Route-A requires Shout paging mat count to match the deterministic plan (prev lut_idx={idx}, expected {}, got {})",
                            page_ell_addrs.len(),
                            lut_wit.mats.len(),
                        )));
                    }
                }
                for (idx, (mem_inst, mem_wit)) in prev.mem_instances.iter().enumerate() {
                    if mem_inst.comms.is_empty() || mem_wit.mats.is_empty() {
                        return Err(PiCcsError::InvalidInput(format!(
                            "no-shared-bus Route-A requires committed Twist instances (non-empty comms/mats, prev mem_idx={idx})"
                        )));
                    }
                    if mem_inst.comms.len() != 1 || mem_wit.mats.len() != 1 {
                        return Err(PiCcsError::InvalidInput(format!(
                            "no-shared-bus Route-A requires exactly 1 comm/mat per Twist instance (prev mem_idx={idx}, comms.len()={}, mats.len()={})",
                            mem_inst.comms.len(),
                            mem_wit.mats.len()
                        )));
                    }
                }
            }
        }
    }
    let mut shout_me_claims_time: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let mut twist_me_claims_time: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let mut val_me_claims: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let mut proofs: Vec<MemOrLutProof> = Vec::new();

    // --------------------------------------------------------------------
    // Phase 2: Twist val-eval sum-check (batched across mem instances).
    // --------------------------------------------------------------------
    let mut twist_val_eval_proofs: Vec<twist::TwistValEvalProof<K>> = Vec::new();
    let mut r_val: Vec<K> = Vec::new();
    if !step.mem_instances.is_empty() {
        let plan = crate::memory_sidecar::claim_plan::TwistValEvalClaimPlan::build(
            step.mem_instances.iter().map(|(inst, _)| inst),
            has_prev,
        );
        let n_mem = step.mem_instances.len();
        let claims_per_mem = plan.claims_per_mem;
        let claim_count = plan.claim_count;

        let mut val_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(claim_count);
        let mut bind_claims: Vec<(u8, K)> = Vec::with_capacity(claim_count);
        let mut claimed_sums: Vec<K> = Vec::with_capacity(claim_count);

        let mut claimed_inc_sums_lt: Vec<K> = Vec::with_capacity(n_mem);
        let mut claimed_inc_sums_total: Vec<K> = Vec::with_capacity(n_mem);
        let mut claimed_prev_inc_sums_total: Vec<Option<K>> = Vec::with_capacity(n_mem);

        let mut claim_idx = 0usize;
        for (i_mem, (mem_inst, _mem_wit)) in step.mem_instances.iter().enumerate() {
            let pre = twist_pre
                .get(i_mem)
                .ok_or_else(|| PiCcsError::ProtocolError("missing Twist pre-time data".into()))?;
            let decoded = &pre.decoded;
            let r_addr = &pre.addr_pre.r_addr;
            if decoded.lanes.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "Twist(Route A): decoded lanes empty at mem_idx={i_mem}"
                )));
            }

            let mut lt_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(decoded.lanes.len());
            let mut claimed_inc_sum_lt = K::ZERO;
            for lane in decoded.lanes.iter() {
                let (oracle, claim) = TwistValEvalOracleSparseTime::new(
                    lane.wa_bits.clone(),
                    lane.has_write.clone(),
                    lane.inc_at_write_addr.clone(),
                    r_addr,
                    r_time,
                );
                lt_oracles.push(Box::new(oracle));
                claimed_inc_sum_lt += claim;
            }
            let oracle_lt: Box<dyn RoundOracle> = Box::new(SumRoundOracle::new(lt_oracles));

            let mut total_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(decoded.lanes.len());
            let mut claimed_inc_sum_total = K::ZERO;
            for lane in decoded.lanes.iter() {
                let (oracle, claim) = TwistTotalIncOracleSparseTime::new(
                    lane.wa_bits.clone(),
                    lane.has_write.clone(),
                    lane.inc_at_write_addr.clone(),
                    r_addr,
                );
                total_oracles.push(Box::new(oracle));
                claimed_inc_sum_total += claim;
            }
            let oracle_total: Box<dyn RoundOracle> = Box::new(SumRoundOracle::new(total_oracles));

            val_oracles.push(oracle_lt);
            bind_claims.push((plan.bind_tags[claim_idx], claimed_inc_sum_lt));
            claimed_sums.push(claimed_inc_sum_lt);
            claim_idx += 1;

            val_oracles.push(oracle_total);
            bind_claims.push((plan.bind_tags[claim_idx], claimed_inc_sum_total));
            claimed_sums.push(claimed_inc_sum_total);
            claim_idx += 1;

            claimed_inc_sums_lt.push(claimed_inc_sum_lt);
            claimed_inc_sums_total.push(claimed_inc_sum_total);

            if let Some(prev) = prev_step {
                let (prev_inst, _prev_wit) = prev
                    .mem_instances
                    .get(i_mem)
                    .ok_or_else(|| PiCcsError::ProtocolError("missing prev mem instance".into()))?;
                if prev_inst.d != mem_inst.d
                    || prev_inst.ell != mem_inst.ell
                    || prev_inst.k != mem_inst.k
                    || prev_inst.lanes != mem_inst.lanes
                {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Twist rollover requires stable geometry at mem_idx={}: prev (k={}, d={}, ell={}, lanes={}) vs cur (k={}, d={}, ell={}, lanes={})",
                        i_mem,
                        prev_inst.k,
                        prev_inst.d,
                        prev_inst.ell,
                        prev_inst.lanes,
                        mem_inst.k,
                        mem_inst.d,
                        mem_inst.ell,
                        mem_inst.lanes
                    )));
                }
                let prev_decoded = prev_twist_decoded
                    .ok_or_else(|| PiCcsError::ProtocolError("missing prev Twist decoded cols".into()))?
                    .get(i_mem)
                    .ok_or_else(|| PiCcsError::ProtocolError("missing prev Twist decoded cols at mem_idx".into()))?;
                if prev_decoded.lanes.is_empty() {
                    return Err(PiCcsError::ProtocolError(
                        "missing prev Twist decoded cols lanes".into(),
                    ));
                }

                let mut prev_total_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(prev_decoded.lanes.len());
                let mut claimed_prev_total = K::ZERO;
                for lane in prev_decoded.lanes.iter() {
                    let (oracle, claim) = TwistTotalIncOracleSparseTime::new(
                        lane.wa_bits.clone(),
                        lane.has_write.clone(),
                        lane.inc_at_write_addr.clone(),
                        r_addr,
                    );
                    prev_total_oracles.push(Box::new(oracle));
                    claimed_prev_total += claim;
                }
                let oracle_prev_total: Box<dyn RoundOracle> = Box::new(SumRoundOracle::new(prev_total_oracles));

                val_oracles.push(oracle_prev_total);
                bind_claims.push((plan.bind_tags[claim_idx], claimed_prev_total));
                claimed_sums.push(claimed_prev_total);
                claim_idx += 1;

                claimed_prev_inc_sums_total.push(Some(claimed_prev_total));
            } else {
                claimed_prev_inc_sums_total.push(None);
            }
        }

        tr.append_message(
            b"twist/val_eval/batch_start",
            &(step.mem_instances.len() as u64).to_le_bytes(),
        );
        tr.append_message(b"twist/val_eval/step_idx", &(step_idx as u64).to_le_bytes());
        bind_twist_val_eval_claim_sums(tr, &bind_claims);

        let mut claims: Vec<BatchedClaim<'_>> = val_oracles
            .iter_mut()
            .zip(claimed_sums.iter())
            .zip(plan.labels.iter())
            .map(|((oracle, sum), label)| BatchedClaim {
                oracle: oracle.as_mut(),
                claimed_sum: *sum,
                label: *label,
            })
            .collect();

        let (r_val_out, per_claim_results) =
            run_batched_sumcheck_prover_ds(tr, b"twist/val_eval_batch", step_idx, claims.as_mut_slice())?;

        if per_claim_results.len() != claim_count {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval results count mismatch (expected {}, got {})",
                claim_count,
                per_claim_results.len()
            )));
        }
        if r_val_out.len() != r_time.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval r_val.len()={}, expected ell_n={}",
                r_val_out.len(),
                r_time.len()
            )));
        }
        r_val = r_val_out;

        for i in 0..n_mem {
            let base = claims_per_mem * i;
            twist_val_eval_proofs.push(twist::TwistValEvalProof {
                claimed_inc_sum_lt: claimed_inc_sums_lt[i],
                rounds_lt: per_claim_results[base].round_polys.clone(),
                claimed_inc_sum_total: claimed_inc_sums_total[i],
                rounds_total: per_claim_results[base + 1].round_polys.clone(),
                claimed_prev_inc_sum_total: claimed_prev_inc_sums_total[i],
                rounds_prev_total: has_prev.then(|| per_claim_results[base + 2].round_polys.clone()),
            });
        }

        tr.append_message(b"twist/val_eval/batch_done", &[]);
    }

    if step.lut_instances.is_empty() {
        if !shout_addr_pre.claimed_sums.is_empty() || !shout_addr_pre.groups.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "shout_addr_pre must be empty when there are no Shout instances".into(),
            ));
        }
    }

    for _ in 0..step.lut_instances.len() {
        proofs.push(MemOrLutProof::Shout(ShoutProofK::default()));
    }

    for idx in 0..step.mem_instances.len() {
        let mut proof = TwistProofK::default();
        proof.addr_pre = twist_pre
            .get(idx)
            .ok_or_else(|| PiCcsError::ProtocolError("missing Twist addr_pre".into()))?
            .addr_pre
            .clone();
        proof.val_eval = twist_val_eval_proofs.get(idx).cloned();

        proofs.push(MemOrLutProof::Twist(proof));
    }

    if !step.mem_instances.is_empty() {
        if r_val.len() != r_time.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval r_val.len()={}, expected ell_n={}",
                r_val.len(),
                r_time.len()
            )));
        }

        let core_t = s.t();

        match cpu_bus {
            Some(cpu_bus) => {
                // Shared-bus mode: val-lane checks read bus openings from CPU ME claims at r_val.
                // Emit CPU ME at r_val for current step (and previous step for rollover).
                let (mcs_inst, mcs_wit) = &step.mcs;
                let cpu_claims_cur = ts::emit_me_claims_for_mats(
                    tr,
                    b"cpu_bus/me_digest_val",
                    params,
                    s,
                    core::slice::from_ref(&mcs_inst.c),
                    core::slice::from_ref(&mcs_wit.Z),
                    &r_val,
                    m_in,
                )?;
                if cpu_claims_cur.len() != 1 {
                    return Err(PiCcsError::ProtocolError(format!(
                        "expected exactly 1 CPU ME claim at r_val, got {}",
                        cpu_claims_cur.len()
                    )));
                }
                let mut cpu_claims_cur = cpu_claims_cur;
                crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
                    params,
                    cpu_bus,
                    core_t,
                    &mcs_wit.Z,
                    &mut cpu_claims_cur[0],
                )?;
                val_me_claims.extend(cpu_claims_cur);

                if let Some(prev) = prev_step {
                    let (prev_mcs_inst, prev_mcs_wit) = &prev.mcs;
                    let cpu_claims_prev = ts::emit_me_claims_for_mats(
                        tr,
                        b"cpu_bus/me_digest_val",
                        params,
                        s,
                        core::slice::from_ref(&prev_mcs_inst.c),
                        core::slice::from_ref(&prev_mcs_wit.Z),
                        &r_val,
                        m_in,
                    )?;
                    if cpu_claims_prev.len() != 1 {
                        return Err(PiCcsError::ProtocolError(format!(
                            "expected exactly 1 prev CPU ME claim at r_val, got {}",
                            cpu_claims_prev.len()
                        )));
                    }
                    let mut cpu_claims_prev = cpu_claims_prev;
                    crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
                        params,
                        cpu_bus,
                        core_t,
                        &prev_mcs_wit.Z,
                        &mut cpu_claims_prev[0],
                    )?;
                    val_me_claims.extend(cpu_claims_prev);
                }
            }
            None => {
                // No-shared-bus mode: emit Twist ME at r_val for each Twist instance.
                for (mem_idx, (mem_inst, mem_wit)) in step.mem_instances.iter().enumerate() {
                    if mem_inst.comms.len() != mem_wit.mats.len() {
                        return Err(PiCcsError::InvalidInput(format!(
                            "Twist(Route A): comms/mats mismatch at mem_idx={mem_idx} (comms.len()={}, mats.len()={})",
                            mem_inst.comms.len(),
                            mem_wit.mats.len()
                        )));
                    }
                    if mem_wit.mats.len() != 1 {
                        return Err(PiCcsError::InvalidInput(format!(
                            "Twist(Route A): non-shared-bus mode expects exactly 1 witness mat per mem instance at mem_idx={mem_idx} (mats.len()={})",
                            mem_wit.mats.len()
                        )));
                    }

                    let ell_addr = mem_inst.d * mem_inst.ell;
                    let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
                        s.m,
                        m_in,
                        mem_inst.steps,
                        core::iter::empty::<(usize, usize)>(),
                        core::iter::once((ell_addr, mem_inst.lanes.max(1))),
                    )
                    .map_err(|e| PiCcsError::InvalidInput(format!("Twist(Route A): bus layout failed: {e}")))?;
                    if bus.twist_cols.len() != 1 || !bus.shout_cols.is_empty() {
                        return Err(PiCcsError::ProtocolError(
                            "Twist(Route A): expected a twist-only bus layout with 1 instance".into(),
                        ));
                    }

                    let mut me = ts::emit_me_claims_for_mats(
                        tr,
                        b"twist/me_digest_val",
                        params,
                        s,
                        core::slice::from_ref(&mem_inst.comms[0]),
                        core::slice::from_ref(&mem_wit.mats[0]),
                        &r_val,
                        m_in,
                    )?;
                    if me.len() != 1 {
                        return Err(PiCcsError::ProtocolError(
                            "Twist(Route A): expected exactly 1 Twist ME claim at r_val".into(),
                        ));
                    }
                    crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
                        params,
                        &bus,
                        core_t,
                        &mem_wit.mats[0],
                        &mut me[0],
                    )?;
                    val_me_claims.push(me.remove(0));
                }

                if let Some(prev) = prev_step {
                    if prev.mem_instances.len() != step.mem_instances.len() {
                        return Err(PiCcsError::InvalidInput(
                            "Twist rollover requires stable mem instance count".into(),
                        ));
                    }
                    for (mem_idx, (mem_inst, mem_wit)) in prev.mem_instances.iter().enumerate() {
                        if mem_wit.mats.len() != 1 || mem_inst.comms.len() != 1 {
                            return Err(PiCcsError::InvalidInput(format!(
                                "Twist(Route A): prev step must provide exactly 1 comm/mat per mem instance (mem_idx={mem_idx})",
                            )));
                        }
                        let ell_addr = mem_inst.d * mem_inst.ell;
                        let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
                            s.m,
                            m_in,
                            mem_inst.steps,
                            core::iter::empty::<(usize, usize)>(),
                            core::iter::once((ell_addr, mem_inst.lanes.max(1))),
                        )
                        .map_err(|e| PiCcsError::InvalidInput(format!("Twist(Route A): bus layout failed: {e}")))?;

                        let mut me = ts::emit_me_claims_for_mats(
                            tr,
                            b"twist/me_digest_val",
                            params,
                            s,
                            core::slice::from_ref(&mem_inst.comms[0]),
                            core::slice::from_ref(&mem_wit.mats[0]),
                            &r_val,
                            m_in,
                        )?;
                        if me.len() != 1 {
                            return Err(PiCcsError::ProtocolError(
                                "Twist(Route A): expected exactly 1 prev Twist ME claim at r_val".into(),
                            ));
                        }
                        crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
                            params,
                            &bus,
                            core_t,
                            &mem_wit.mats[0],
                            &mut me[0],
                        )?;
                        val_me_claims.push(me.remove(0));
                    }
                }
            }
        }
    }

    if step.mem_instances.is_empty() {
        if !twist_val_eval_proofs.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "twist val-eval proofs must be empty when no mem instances are present".into(),
            ));
        }
        if !r_val.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "twist r_val must be empty when no mem instances are present".into(),
            ));
        }
        if !val_me_claims.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "twist val-lane ME claims must be empty when no mem instances are present".into(),
            ));
        }
    } else if val_me_claims.is_empty() {
        return Err(PiCcsError::ProtocolError(
            "twist val-eval requires non-empty val-lane ME claims".into(),
        ));
    }

    // No-shared-bus mode: also emit Shout ME openings at r_time for time-lane checks and trace linkage.
    if cpu_bus.is_none() && !step.lut_instances.is_empty() {
        for (lut_idx, (lut_inst, lut_wit)) in step.lut_instances.iter().enumerate() {
            let lanes = lut_inst.lanes.max(1);
            let ell_addr = lut_inst.d * lut_inst.ell;
            let page_ell_addrs = plan_shout_addr_pages(s.m, m_in, lut_inst.steps, ell_addr, lanes)?;
            if lut_inst.comms.len() != page_ell_addrs.len() || lut_wit.mats.len() != page_ell_addrs.len() {
                return Err(PiCcsError::InvalidInput(format!(
                    "Shout(Route A): paging plan mismatch at r_time (lut_idx={lut_idx}, expected {} comms/mats, got comms.len()={}, mats.len()={})",
                    page_ell_addrs.len(),
                    lut_inst.comms.len(),
                    lut_wit.mats.len()
                )));
            }

            let mut me = ts::emit_me_claims_for_mats(
                tr,
                b"shout/me_digest_time",
                params,
                s,
                &lut_inst.comms,
                &lut_wit.mats,
                r_time,
                m_in,
            )?;
            if me.len() != page_ell_addrs.len() {
                return Err(PiCcsError::ProtocolError(format!(
                    "Shout(Route A): expected {} Shout ME claim(s) at r_time, got {}",
                    page_ell_addrs.len(),
                    me.len()
                )));
            }

            // Shout is sparse-in-time (at most one event per active row). In no-shared-bus mode we commit
            // each Shout instance separately, so avoid scanning the full chunk for every bus column when
            // appending time openings: restrict to rows where any lane's `has_lookup` is nonzero.
            let active_js: Vec<usize> = {
                let page0_ell_addr = *page_ell_addrs
                    .first()
                    .ok_or_else(|| PiCcsError::ProtocolError("Shout(Route A): empty paging plan".into()))?;
                let bus0 = build_bus_layout_for_instances_with_shout_and_twist_lanes(
                    s.m,
                    m_in,
                    lut_inst.steps,
                    core::iter::once((page0_ell_addr, lanes)),
                    core::iter::empty::<(usize, usize)>(),
                )
                .map_err(|e| PiCcsError::InvalidInput(format!("Shout(Route A): bus layout failed: {e}")))?;
                if bus0.shout_cols.len() != 1 || !bus0.twist_cols.is_empty() {
                    return Err(PiCcsError::ProtocolError(
                        "Shout(Route A): expected a shout-only bus layout with 1 instance".into(),
                    ));
                }
                let mat0 = lut_wit
                    .mats
                    .get(0)
                    .ok_or_else(|| PiCcsError::ProtocolError("missing Shout witness mat".into()))?;
                let shout0 = bus0
                    .shout_cols
                    .get(0)
                    .ok_or_else(|| PiCcsError::ProtocolError("Shout(Route A): missing shout_cols[0]".into()))?;
                let mut out: Vec<usize> = Vec::new();
                for j in 0..lut_inst.steps {
                    let mut any = false;
                    for lane in shout0.lanes.iter() {
                        let idx = bus0.bus_cell(lane.has_lookup, j);
                        for rho in 0..neo_math::D {
                            if mat0[(rho, idx)] != F::ZERO {
                                any = true;
                                break;
                            }
                        }
                        if any {
                            break;
                        }
                    }
                    if any {
                        out.push(j);
                    }
                }
                out
            };

            for (page_idx, &page_ell_addr) in page_ell_addrs.iter().enumerate() {
                let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
                    s.m,
                    m_in,
                    lut_inst.steps,
                    core::iter::once((page_ell_addr, lanes)),
                    core::iter::empty::<(usize, usize)>(),
                )
                .map_err(|e| PiCcsError::InvalidInput(format!("Shout(Route A): bus layout failed: {e}")))?;
                if bus.shout_cols.len() != 1 || !bus.twist_cols.is_empty() {
                    return Err(PiCcsError::ProtocolError(
                        "Shout(Route A): expected a shout-only bus layout with 1 instance".into(),
                    ));
                }

                let mat = lut_wit
                    .mats
                    .get(page_idx)
                    .ok_or_else(|| PiCcsError::ProtocolError("missing Shout witness mat".into()))?;
                crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance_at_js(
                    params,
                    &bus,
                    s.t(),
                    mat,
                    &mut me[page_idx],
                    &active_js,
                )?;
            }
            shout_me_claims_time.extend(me.into_iter());
        }
    }

    // No-shared-bus mode: also emit Twist ME openings at r_time for time-lane linkage and terminal checks.
    if cpu_bus.is_none() && !step.mem_instances.is_empty() {
        for (mem_idx, (mem_inst, mem_wit)) in step.mem_instances.iter().enumerate() {
            if mem_inst.comms.len() != 1 || mem_wit.mats.len() != 1 {
                return Err(PiCcsError::InvalidInput(format!(
                    "Twist(Route A): non-shared-bus mode expects exactly 1 comm/mat per mem instance (mem_idx={mem_idx})"
                )));
            }

            let ell_addr = mem_inst.d * mem_inst.ell;
            let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
                s.m,
                m_in,
                mem_inst.steps,
                core::iter::empty::<(usize, usize)>(),
                core::iter::once((ell_addr, mem_inst.lanes.max(1))),
            )
            .map_err(|e| PiCcsError::InvalidInput(format!("Twist(Route A): bus layout failed: {e}")))?;

            let mut me = ts::emit_me_claims_for_mats(
                tr,
                b"twist/me_digest_time",
                params,
                s,
                core::slice::from_ref(&mem_inst.comms[0]),
                core::slice::from_ref(&mem_wit.mats[0]),
                r_time,
                m_in,
            )?;
            if me.len() != 1 {
                return Err(PiCcsError::ProtocolError(
                    "Twist(Route A): expected exactly 1 Twist ME claim at r_time".into(),
                ));
            }
            crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
                params,
                &bus,
                s.t(),
                &mem_wit.mats[0],
                &mut me[0],
            )?;
            twist_me_claims_time.push(me.remove(0));
        }
    }

    Ok(MemSidecarProof {
        shout_me_claims_time,
        twist_me_claims_time,
        val_me_claims,
        shout_addr_pre: shout_addr_pre.clone(),
        proofs,
    })
}

// ============================================================================
// ============================================================================
pub fn verify_route_a_memory_step(
    tr: &mut Poseidon2Transcript,
    cpu_bus: Option<&BusLayout>,
    m: usize,
    core_t: usize,
    step: &StepInstanceBundle<Cmt, F, K>,
    prev_step: Option<&StepInstanceBundle<Cmt, F, K>>,
    ccs_out0: &MeInstance<Cmt, F, K>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    batched_claimed_sums: &[K],
    claim_idx_start: usize,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    shout_pre: &[ShoutAddrPreVerifyData],
    twist_pre: &[TwistAddrPreVerifyData],
    step_idx: usize,
) -> Result<RouteAMemoryVerifyOutput, PiCcsError> {
    let Some(cpu_bus) = cpu_bus else {
        return verify_route_a_memory_step_no_shared_cpu_bus(
            tr,
            m,
            core_t,
            step,
            prev_step,
            ccs_out0,
            r_time,
            r_cycle,
            batched_final_values,
            batched_claimed_sums,
            claim_idx_start,
            mem_proof,
            shout_pre,
            twist_pre,
            step_idx,
        );
    };

    let chi_cycle_at_r_time = eq_points(r_time, r_cycle);
    if ccs_out0.r.as_slice() != r_time {
        return Err(PiCcsError::ProtocolError(
            "CPU ME output r mismatch (expected shared r_time)".into(),
        ));
    }
    let has_prev = prev_step.is_some();
    if let Some(prev) = prev_step {
        if prev.mem_insts.len() != step.mem_insts.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist rollover requires stable mem instance count: prev has {}, current has {}",
                prev.mem_insts.len(),
                step.mem_insts.len()
            )));
        }
        for (idx, (prev_inst, inst)) in prev.mem_insts.iter().zip(step.mem_insts.iter()).enumerate() {
            if prev_inst.d != inst.d
                || prev_inst.ell != inst.ell
                || prev_inst.k != inst.k
                || prev_inst.lanes != inst.lanes
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "Twist rollover requires stable geometry at mem_idx={}: prev (k={}, d={}, ell={}, lanes={}) vs cur (k={}, d={}, ell={}, lanes={})",
                    idx,
                    prev_inst.k,
                    prev_inst.d,
                    prev_inst.ell,
                    prev_inst.lanes,
                    inst.k,
                    inst.d,
                    inst.ell,
                    inst.lanes
                )));
            }
        }
    }

    for (idx, inst) in step.lut_insts.iter().enumerate() {
        if !inst.comms.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "shared CPU bus requires metadata-only Shout instances (comms must be empty, lut_idx={idx})"
            )));
        }
    }
    for (idx, inst) in step.mem_insts.iter().enumerate() {
        if !inst.comms.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "shared CPU bus requires metadata-only Twist instances (comms must be empty, mem_idx={idx})"
            )));
        }
    }
    if let Some(prev) = prev_step {
        for (idx, inst) in prev.lut_insts.iter().enumerate() {
            if !inst.comms.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared CPU bus requires metadata-only Shout instances (comms must be empty, prev lut_idx={idx})"
                )));
            }
        }
        for (idx, inst) in prev.mem_insts.iter().enumerate() {
            if !inst.comms.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared CPU bus requires metadata-only Twist instances (comms must be empty, prev mem_idx={idx})"
                )));
            }
        }
    }

    let proofs_mem = &mem_proof.proofs;

    if cpu_bus.shout_cols.len() != step.lut_insts.len() || cpu_bus.twist_cols.len() != step.mem_insts.len() {
        return Err(PiCcsError::InvalidInput(
            "shared_cpu_bus layout mismatch for step (instance counts)".into(),
        ));
    }

    let bus_y_base_time = if cpu_bus.bus_cols > 0 {
        ccs_out0
            .y_scalars
            .len()
            .checked_sub(cpu_bus.bus_cols)
            .ok_or_else(|| PiCcsError::InvalidInput("CPU y_scalars too short for bus openings".into()))?
    } else {
        0usize
    };
    let claim_plan = RouteATimeClaimPlan::build(step, claim_idx_start)?;
    if claim_plan.claim_idx_end > batched_final_values.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "batched_final_values too short (need at least {}, have {})",
            claim_plan.claim_idx_end,
            batched_final_values.len()
        )));
    }
    if claim_plan.claim_idx_end > batched_claimed_sums.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "batched_claimed_sums too short (need at least {}, have {})",
            claim_plan.claim_idx_end,
            batched_claimed_sums.len()
        )));
    }

    let expected_proofs = step.lut_insts.len() + step.mem_insts.len();
    if proofs_mem.len() != expected_proofs {
        return Err(PiCcsError::InvalidInput(format!(
            "mem proof count mismatch (expected {}, got {})",
            expected_proofs,
            proofs_mem.len()
        )));
    }
    let total_shout_lanes: usize = step.lut_insts.iter().map(|inst| inst.lanes.max(1)).sum();
    if shout_pre.len() != total_shout_lanes {
        return Err(PiCcsError::InvalidInput(format!(
            "shout pre-time count mismatch (expected total_lanes={}, got {})",
            total_shout_lanes,
            shout_pre.len()
        )));
    }
    if twist_pre.len() != step.mem_insts.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "twist pre-time count mismatch (expected {}, got {})",
            step.mem_insts.len(),
            twist_pre.len()
        )));
    }

    let mut twist_time_openings: Vec<TwistTimeLaneOpenings> = Vec::with_capacity(step.mem_insts.len());

    // Shout instances first.
    let mut shout_lane_base: usize = 0;
    for (proof_idx, inst) in step.lut_insts.iter().enumerate() {
        match &proofs_mem[proof_idx] {
            MemOrLutProof::Shout(_proof) => {}
            _ => return Err(PiCcsError::InvalidInput("expected Shout proof".into())),
        }
        if matches!(
            inst.table_spec,
            Some(LutTableSpec::RiscvOpcodePacked { .. } | LutTableSpec::RiscvOpcodeEventTablePacked { .. })
        ) {
            return Err(PiCcsError::InvalidInput(
                "packed RISC-V Shout table specs are not supported on the shared CPU bus".into(),
            ));
        }

        let ell_addr = inst.d * inst.ell;
        let expected_lanes = inst.lanes.max(1);

        let inst_cols = cpu_bus
            .shout_cols
            .get(proof_idx)
            .ok_or_else(|| PiCcsError::InvalidInput("shared_cpu_bus layout mismatch (shout)".into()))?;
        if inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at lut_idx={proof_idx}: bus shout lanes={} but instance expects {expected_lanes}",
                inst_cols.lanes.len()
            )));
        }

        struct ShoutLaneOpen {
            addr_bits: Vec<K>,
            has_lookup: K,
            val: K,
        }
        let mut lane_opens: Vec<ShoutLaneOpen> = Vec::with_capacity(expected_lanes);
        for (lane_idx, shout_cols) in inst_cols.lanes.iter().enumerate() {
            if shout_cols.addr_bits.end - shout_cols.addr_bits.start != ell_addr {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at lut_idx={proof_idx}, lane_idx={lane_idx}: expected ell_addr={ell_addr}"
                )));
            }

            let mut addr_bits_open = Vec::with_capacity(ell_addr);
            for (_j, col_id) in shout_cols.addr_bits.clone().enumerate() {
                addr_bits_open.push(
                    ccs_out0
                        .y_scalars
                        .get(cpu_bus.y_scalar_index(bus_y_base_time, col_id))
                        .copied()
                        .ok_or_else(|| {
                            PiCcsError::ProtocolError("CPU y_scalars missing Shout addr_bits opening".into())
                        })?,
                );
            }
            let has_lookup_open = ccs_out0
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_time, shout_cols.has_lookup))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Shout has_lookup opening".into()))?;
            let val_open = ccs_out0
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_time, shout_cols.val))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Shout val opening".into()))?;

            lane_opens.push(ShoutLaneOpen {
                addr_bits: addr_bits_open,
                has_lookup: has_lookup_open,
                val: val_open,
            });
        }

        let shout_claims = claim_plan
            .shout
            .get(proof_idx)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("missing Shout claim schedule at index {}", proof_idx)))?;
        if shout_claims.lanes.len() != expected_lanes {
            return Err(PiCcsError::ProtocolError(format!(
                "Shout claim schedule lane count mismatch at lut_idx={proof_idx}: expected {expected_lanes}, got {}",
                shout_claims.lanes.len()
            )));
        }
        if shout_lane_base
            .checked_add(expected_lanes)
            .ok_or_else(|| PiCcsError::ProtocolError("shout lane index overflow".into()))?
            > shout_pre.len()
        {
            return Err(PiCcsError::ProtocolError("Shout pre-time lane indexing drift".into()));
        }

        // Route A Shout ordering in batched_time:
        // - value (time rounds only) per lane
        // - adapter (time rounds only) per lane
        // - aggregated bitness for (addr_bits, has_lookup)
        {
            let mut opens: Vec<K> = Vec::with_capacity(expected_lanes * (ell_addr + 1));
            for lane in lane_opens.iter() {
                opens.extend_from_slice(&lane.addr_bits);
                opens.push(lane.has_lookup);
            }
            let weights = bitness_weights(r_cycle, opens.len(), 0x5348_4F55_54u64 + proof_idx as u64);
            let mut acc = K::ZERO;
            for (w, b) in weights.iter().zip(opens.iter()) {
                acc += *w * *b * (*b - K::ONE);
            }
            let expected = chi_cycle_at_r_time * acc;
            if expected != batched_final_values[shout_claims.bitness] {
                return Err(PiCcsError::ProtocolError(
                    "shout/bitness terminal value mismatch".into(),
                ));
            }
        }

        for (lane_idx, lane) in lane_opens.iter().enumerate() {
            let pre = shout_pre.get(shout_lane_base + lane_idx).ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "missing pre-time Shout lane data at index {}",
                    shout_lane_base + lane_idx
                ))
            })?;
            let lane_claims = shout_claims
                .lanes
                .get(lane_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("shout claim schedule lane idx drift".into()))?;

            let value_claim = batched_claimed_sums[lane_claims.value];
            let value_final = batched_final_values[lane_claims.value];
            let adapter_claim = batched_claimed_sums[lane_claims.adapter];
            let adapter_final = batched_final_values[lane_claims.adapter];

            let expected_value_final = chi_cycle_at_r_time * lane.has_lookup * lane.val;
            if expected_value_final != value_final {
                return Err(PiCcsError::ProtocolError("shout value terminal value mismatch".into()));
            }

            let eq_addr = eq_bits_prod(&lane.addr_bits, &pre.r_addr)?;
            let expected_adapter_final = chi_cycle_at_r_time * lane.has_lookup * eq_addr;
            if expected_adapter_final != adapter_final {
                return Err(PiCcsError::ProtocolError(
                    "shout adapter terminal value mismatch".into(),
                ));
            }

            if value_claim != pre.addr_claim_sum {
                return Err(PiCcsError::ProtocolError(
                    "shout value claimed sum != addr claimed sum".into(),
                ));
            }

            if pre.is_active {
                let expected_addr_final = pre.table_eval_at_r_addr * adapter_claim;
                if expected_addr_final != pre.addr_final {
                    return Err(PiCcsError::ProtocolError("shout addr terminal value mismatch".into()));
                }
            } else {
                // If we skipped the addr-pre sumcheck, the only sound case is "no lookups".
                // Enforce this by requiring the addr claim + adapter claim to be zero.
                if pre.addr_claim_sum != K::ZERO {
                    return Err(PiCcsError::ProtocolError(
                        "shout addr-pre skipped but addr claim is nonzero".into(),
                    ));
                }
                if adapter_claim != K::ZERO {
                    return Err(PiCcsError::ProtocolError(
                        "shout addr-pre skipped but adapter claim is nonzero".into(),
                    ));
                }
                if pre.addr_final != K::ZERO {
                    return Err(PiCcsError::ProtocolError(
                        "shout addr-pre skipped but addr_final is nonzero".into(),
                    ));
                }
            }
        }

        shout_lane_base += expected_lanes;
    }
    if shout_lane_base != shout_pre.len() {
        return Err(PiCcsError::ProtocolError(
            "shout pre-time lanes not fully consumed".into(),
        ));
    }

    // Twist instances next.
    let proof_mem_offset = step.lut_insts.len();

    // --------------------------------------------------------------------
    // Twist time checks at addr-pre `r_addr`.
    // --------------------------------------------------------------------
    for (i_mem, inst) in step.mem_insts.iter().enumerate() {
        let twist_proof = match &proofs_mem[proof_mem_offset + i_mem] {
            MemOrLutProof::Twist(proof) => proof,
            _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
        };
        let layout = inst.twist_layout();
        let ell_addr = layout
            .lanes
            .get(0)
            .ok_or_else(|| PiCcsError::InvalidInput("TwistWitnessLayout has no lanes".into()))?
            .ell_addr;

        let twist_inst_cols = cpu_bus
            .twist_cols
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput("shared_cpu_bus layout mismatch (twist)".into()))?;
        let expected_lanes = inst.lanes.max(1);
        if twist_inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at mem_idx={i_mem}: expected lanes={expected_lanes}, got {}",
                twist_inst_cols.lanes.len()
            )));
        }

        struct TwistLaneTimeOpen {
            ra_bits: Vec<K>,
            wa_bits: Vec<K>,
            has_read: K,
            has_write: K,
            wv: K,
            rv: K,
            inc: K,
        }

        let mut lane_opens: Vec<TwistLaneTimeOpen> = Vec::with_capacity(twist_inst_cols.lanes.len());
        for (lane_idx, twist_cols) in twist_inst_cols.lanes.iter().enumerate() {
            if twist_cols.ra_bits.end - twist_cols.ra_bits.start != ell_addr
                || twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at mem_idx={i_mem}, lane={lane_idx}: expected ell_addr={ell_addr}"
                )));
            }

            let mut ra_bits_open = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.ra_bits.clone() {
                ra_bits_open.push(
                    ccs_out0
                        .y_scalars
                        .get(cpu_bus.y_scalar_index(bus_y_base_time, col_id))
                        .copied()
                        .ok_or_else(|| {
                            PiCcsError::ProtocolError("CPU y_scalars missing Twist ra_bits opening".into())
                        })?,
                );
            }
            let mut wa_bits_open = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.wa_bits.clone() {
                wa_bits_open.push(
                    ccs_out0
                        .y_scalars
                        .get(cpu_bus.y_scalar_index(bus_y_base_time, col_id))
                        .copied()
                        .ok_or_else(|| {
                            PiCcsError::ProtocolError("CPU y_scalars missing Twist wa_bits opening".into())
                        })?,
                );
            }

            let has_read_open = ccs_out0
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_time, twist_cols.has_read))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Twist has_read opening".into()))?;
            let has_write_open = ccs_out0
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_time, twist_cols.has_write))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Twist has_write opening".into()))?;
            let wv_open = ccs_out0
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_time, twist_cols.wv))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Twist wv opening".into()))?;
            let rv_open = ccs_out0
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_time, twist_cols.rv))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Twist rv opening".into()))?;
            let inc_write_open = ccs_out0
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_time, twist_cols.inc))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Twist inc opening".into()))?;

            lane_opens.push(TwistLaneTimeOpen {
                ra_bits: ra_bits_open,
                wa_bits: wa_bits_open,
                has_read: has_read_open,
                has_write: has_write_open,
                wv: wv_open,
                rv: rv_open,
                inc: inc_write_open,
            });
        }

        let pre = twist_pre
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing Twist pre-time data at index {}", i_mem)))?;
        let r_addr = &pre.r_addr;
        if r_addr.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist r_addr.len()={}, expected ell_addr={}",
                r_addr.len(),
                ell_addr
            )));
        }

        let twist_claims = claim_plan
            .twist
            .get(i_mem)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("missing Twist claim schedule at index {}", i_mem)))?;

        // Route A Twist ordering in batched_time:
        // - read_check (time rounds only)
        // - write_check (time rounds only)
        // - bitness for ra_bits then wa_bits then has_read then has_write (time-only)
        let read_check_claim = batched_claimed_sums[twist_claims.read_check];
        let read_check_final = batched_final_values[twist_claims.read_check];
        let write_check_claim = batched_claimed_sums[twist_claims.write_check];
        let write_check_final = batched_final_values[twist_claims.write_check];

        if read_check_claim != pre.read_check_claim_sum {
            return Err(PiCcsError::ProtocolError(
                "twist read_check claimed sum != addr-pre final".into(),
            ));
        }
        if write_check_claim != pre.write_check_claim_sum {
            return Err(PiCcsError::ProtocolError(
                "twist write_check claimed sum != addr-pre final".into(),
            ));
        }

        // Aggregated bitness terminal check (ra_bits, wa_bits, has_read, has_write).
        {
            let mut opens: Vec<K> = Vec::with_capacity(expected_lanes * (2 * ell_addr + 2));
            for lane in lane_opens.iter() {
                opens.extend_from_slice(&lane.ra_bits);
                opens.extend_from_slice(&lane.wa_bits);
                opens.push(lane.has_read);
                opens.push(lane.has_write);
            }
            let weights = bitness_weights(r_cycle, opens.len(), 0x5457_4953_54u64 + i_mem as u64);
            let mut acc = K::ZERO;
            for (w, b) in weights.iter().zip(opens.iter()) {
                acc += *w * *b * (*b - K::ONE);
            }
            let expected = chi_cycle_at_r_time * acc;
            if expected != batched_final_values[twist_claims.bitness] {
                return Err(PiCcsError::ProtocolError(
                    "twist/bitness terminal value mismatch".into(),
                ));
            }
        }

        let val_eval = twist_proof
            .val_eval
            .as_ref()
            .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;

        let init_at_r_addr = eval_init_at_r_addr(&inst.init, inst.k, r_addr)?;
        let claimed_val = init_at_r_addr + val_eval.claimed_inc_sum_lt;

        // Terminal checks for read_check / write_check at (r_time, r_addr).
        let mut expected_read_check_final = K::ZERO;
        let mut expected_write_check_final = K::ZERO;
        for lane in lane_opens.iter() {
            let read_eq_addr = eq_bits_prod(&lane.ra_bits, r_addr)?;
            expected_read_check_final += chi_cycle_at_r_time * lane.has_read * (claimed_val - lane.rv) * read_eq_addr;

            let write_eq_addr = eq_bits_prod(&lane.wa_bits, r_addr)?;
            expected_write_check_final +=
                chi_cycle_at_r_time * lane.has_write * (lane.wv - claimed_val - lane.inc) * write_eq_addr;
        }
        if expected_read_check_final != read_check_final {
            return Err(PiCcsError::ProtocolError(
                "twist/read_check terminal value mismatch".into(),
            ));
        }

        if expected_write_check_final != write_check_final {
            return Err(PiCcsError::ProtocolError(
                "twist/write_check terminal value mismatch".into(),
            ));
        }

        twist_time_openings.push(TwistTimeLaneOpenings {
            lanes: lane_opens
                .into_iter()
                .map(|lane| TwistTimeLaneOpeningsLane {
                    wa_bits: lane.wa_bits,
                    has_write: lane.has_write,
                    inc_at_write_addr: lane.inc,
                })
                .collect(),
        });
    }

    // --------------------------------------------------------------------
    // Phase 2: Verify batched Twist val-eval sum-check, deriving shared r_val.
    // --------------------------------------------------------------------
    let mut r_val: Vec<K> = Vec::new();
    let mut val_eval_finals: Vec<K> = Vec::new();
    if !step.mem_insts.is_empty() {
        let plan = crate::memory_sidecar::claim_plan::TwistValEvalClaimPlan::build(step.mem_insts.iter(), has_prev);
        let claim_count = plan.claim_count;

        let mut per_claim_rounds: Vec<Vec<Vec<K>>> = Vec::with_capacity(claim_count);
        let mut per_claim_sums: Vec<K> = Vec::with_capacity(claim_count);
        let mut bind_claims: Vec<(u8, K)> = Vec::with_capacity(claim_count);
        let mut claim_idx = 0usize;

        for (i_mem, _inst) in step.mem_insts.iter().enumerate() {
            let twist_proof = match &proofs_mem[proof_mem_offset + i_mem] {
                MemOrLutProof::Twist(proof) => proof,
                _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
            };
            let val = twist_proof
                .val_eval
                .as_ref()
                .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;

            per_claim_rounds.push(val.rounds_lt.clone());
            per_claim_sums.push(val.claimed_inc_sum_lt);
            bind_claims.push((plan.bind_tags[claim_idx], val.claimed_inc_sum_lt));
            claim_idx += 1;

            per_claim_rounds.push(val.rounds_total.clone());
            per_claim_sums.push(val.claimed_inc_sum_total);
            bind_claims.push((plan.bind_tags[claim_idx], val.claimed_inc_sum_total));
            claim_idx += 1;

            if has_prev {
                let prev_total = val.claimed_prev_inc_sum_total.ok_or_else(|| {
                    PiCcsError::InvalidInput("Twist(Route A): missing claimed_prev_inc_sum_total".into())
                })?;
                let prev_rounds = val
                    .rounds_prev_total
                    .clone()
                    .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing rounds_prev_total".into()))?;
                per_claim_rounds.push(prev_rounds);
                per_claim_sums.push(prev_total);
                bind_claims.push((plan.bind_tags[claim_idx], prev_total));
                claim_idx += 1;
            } else if val.claimed_prev_inc_sum_total.is_some() || val.rounds_prev_total.is_some() {
                return Err(PiCcsError::InvalidInput(
                    "Twist(Route A): rollover fields present but prev_step is None".into(),
                ));
            }
        }

        tr.append_message(
            b"twist/val_eval/batch_start",
            &(step.mem_insts.len() as u64).to_le_bytes(),
        );
        tr.append_message(b"twist/val_eval/step_idx", &(step_idx as u64).to_le_bytes());
        bind_twist_val_eval_claim_sums(tr, &bind_claims);

        let (r_val_out, finals_out, ok) = verify_batched_sumcheck_rounds_ds(
            tr,
            b"twist/val_eval_batch",
            step_idx,
            &per_claim_rounds,
            &per_claim_sums,
            &plan.labels,
            &plan.degree_bounds,
        );
        if !ok {
            return Err(PiCcsError::SumcheckError(
                "twist val-eval batched sumcheck invalid".into(),
            ));
        }
        if r_val_out.len() != r_time.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval r_val.len()={}, expected ell_n={}",
                r_val_out.len(),
                r_time.len()
            )));
        }
        if finals_out.len() != claim_count {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval finals.len()={}, expected {}",
                finals_out.len(),
                claim_count
            )));
        }
        r_val = r_val_out;
        val_eval_finals = finals_out;

        tr.append_message(b"twist/val_eval/batch_done", &[]);
    }

    // Verify val-eval terminal identity against CPU ME openings at r_val.
    let lt = if step.mem_insts.is_empty() {
        if !r_val.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "twist val-eval produced r_val but no mem instances are present".into(),
            ));
        }
        K::ZERO
    } else {
        if r_val.len() != r_time.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval r_val.len()={}, expected ell_n={}",
                r_val.len(),
                r_time.len()
            )));
        }
        lt_eval(&r_val, r_time)
    };

    let (cpu_me_val_cur, cpu_me_val_prev, bus_y_base_val) = if step.mem_insts.is_empty() {
        if !mem_proof.val_me_claims.is_empty() {
            return Err(PiCcsError::InvalidInput(
                "proof contains val-lane CPU ME claims with no Twist instances".into(),
            ));
        }
        (None, None, 0usize)
    } else {
        let expected = 1usize + usize::from(has_prev);
        if mem_proof.val_me_claims.len() != expected {
            return Err(PiCcsError::InvalidInput(format!(
                "shared bus expects {} CPU ME claim(s) at r_val, got {}",
                expected,
                mem_proof.val_me_claims.len()
            )));
        }

        let cpu_me_cur = mem_proof
            .val_me_claims
            .get(0)
            .ok_or_else(|| PiCcsError::ProtocolError("missing CPU ME claim at r_val".into()))?;
        if cpu_me_cur.r.as_slice() != r_val {
            return Err(PiCcsError::ProtocolError(
                "CPU ME(val) r mismatch (expected r_val)".into(),
            ));
        }
        if cpu_me_cur.c != step.mcs_inst.c {
            return Err(PiCcsError::ProtocolError(
                "CPU ME(val) commitment mismatch (current step)".into(),
            ));
        }
        let cpu_me_prev = if has_prev {
            let prev_inst =
                prev_step.ok_or_else(|| PiCcsError::ProtocolError("prev_step missing with has_prev=true".into()))?;
            let cpu_me_prev = mem_proof
                .val_me_claims
                .get(1)
                .ok_or_else(|| PiCcsError::ProtocolError("missing prev CPU ME claim at r_val".into()))?;
            if cpu_me_prev.r.as_slice() != r_val {
                return Err(PiCcsError::ProtocolError(
                    "CPU ME(val/prev) r mismatch (expected r_val)".into(),
                ));
            }
            if cpu_me_prev.c != prev_inst.mcs_inst.c {
                return Err(PiCcsError::ProtocolError("CPU ME(val/prev) commitment mismatch".into()));
            }
            Some(cpu_me_prev)
        } else {
            None
        };

        let bus_y_base_val = cpu_me_cur
            .y_scalars
            .len()
            .checked_sub(cpu_bus.bus_cols)
            .ok_or_else(|| PiCcsError::InvalidInput("CPU y_scalars too short for bus openings".into()))?;

        (Some(cpu_me_cur), cpu_me_prev, bus_y_base_val)
    };

    for (i_mem, inst) in step.mem_insts.iter().enumerate() {
        let twist_proof = match &proofs_mem[proof_mem_offset + i_mem] {
            MemOrLutProof::Twist(proof) => proof,
            _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
        };
        let val_eval = twist_proof
            .val_eval
            .as_ref()
            .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;
        let layout = inst.twist_layout();
        let ell_addr = layout
            .lanes
            .get(0)
            .ok_or_else(|| PiCcsError::InvalidInput("TwistWitnessLayout has no lanes".into()))?
            .ell_addr;

        let cpu_me_cur =
            cpu_me_val_cur.ok_or_else(|| PiCcsError::ProtocolError("missing CPU ME claim at r_val".into()))?;

        let twist_inst_cols = cpu_bus
            .twist_cols
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput("shared_cpu_bus layout mismatch (twist)".into()))?;
        let expected_lanes = inst.lanes.max(1);
        if twist_inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at mem_idx={i_mem}: expected lanes={expected_lanes}, got {}",
                twist_inst_cols.lanes.len()
            )));
        }

        let r_addr = twist_pre
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing Twist pre-time data at index {}", i_mem)))?
            .r_addr
            .as_slice();

        let mut inc_at_r_addr_val = K::ZERO;
        for (lane_idx, twist_cols) in twist_inst_cols.lanes.iter().enumerate() {
            if twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at mem_idx={i_mem}, lane={lane_idx}: expected ell_addr={ell_addr}"
                )));
            }

            let mut wa_bits_val_open = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.wa_bits.clone() {
                wa_bits_val_open.push(
                    cpu_me_cur
                        .y_scalars
                        .get(cpu_bus.y_scalar_index(bus_y_base_val, col_id))
                        .copied()
                        .ok_or_else(|| {
                            PiCcsError::ProtocolError("CPU y_scalars missing wa_bits(val) opening".into())
                        })?,
                );
            }
            let has_write_val_open = cpu_me_cur
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_val, twist_cols.has_write))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing has_write(val) opening".into()))?;
            let inc_at_write_addr_val_open = cpu_me_cur
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_val, twist_cols.inc))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing inc(val) opening".into()))?;

            let eq_wa_val = eq_bits_prod(&wa_bits_val_open, r_addr)?;
            inc_at_r_addr_val += has_write_val_open * inc_at_write_addr_val_open * eq_wa_val;
        }

        let expected_lt_final = inc_at_r_addr_val * lt;
        let claims_per_mem = if has_prev { 3 } else { 2 };
        let base = claims_per_mem * i_mem;
        if expected_lt_final != val_eval_finals[base] {
            return Err(PiCcsError::ProtocolError(
                "twist/val_eval_lt terminal value mismatch".into(),
            ));
        }
        let expected_total_final = inc_at_r_addr_val;
        if expected_total_final != val_eval_finals[base + 1] {
            return Err(PiCcsError::ProtocolError(
                "twist/val_eval_total terminal value mismatch".into(),
            ));
        }

        if has_prev {
            let prev =
                prev_step.ok_or_else(|| PiCcsError::ProtocolError("prev_step missing with has_prev=true".into()))?;
            let prev_inst = prev
                .mem_insts
                .get(i_mem)
                .ok_or_else(|| PiCcsError::ProtocolError("missing prev mem instance".into()))?;
            let cpu_me_prev = cpu_me_val_prev
                .ok_or_else(|| PiCcsError::ProtocolError("missing prev CPU ME claim at r_val".into()))?;

            // Terminal check for prev-total: uses previous-step openings at current r_val.
            let mut inc_at_r_addr_prev = K::ZERO;
            for (lane_idx, twist_cols) in twist_inst_cols.lanes.iter().enumerate() {
                if twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr {
                    return Err(PiCcsError::InvalidInput(format!(
                        "shared_cpu_bus layout mismatch at mem_idx={i_mem}, lane={lane_idx}: expected ell_addr={ell_addr}"
                    )));
                }

                let mut wa_bits_prev_open = Vec::with_capacity(ell_addr);
                for col_id in twist_cols.wa_bits.clone() {
                    wa_bits_prev_open.push(
                        cpu_me_prev
                            .y_scalars
                            .get(cpu_bus.y_scalar_index(bus_y_base_val, col_id))
                            .copied()
                            .ok_or_else(|| {
                                PiCcsError::ProtocolError("CPU y_scalars missing wa_bits(prev) opening".into())
                            })?,
                    );
                }
                let has_write_prev_open = cpu_me_prev
                    .y_scalars
                    .get(cpu_bus.y_scalar_index(bus_y_base_val, twist_cols.has_write))
                    .copied()
                    .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing has_write(prev) opening".into()))?;
                let inc_prev_open = cpu_me_prev
                    .y_scalars
                    .get(cpu_bus.y_scalar_index(bus_y_base_val, twist_cols.inc))
                    .copied()
                    .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing inc(prev) opening".into()))?;

                let eq_wa_prev = eq_bits_prod(&wa_bits_prev_open, r_addr)?;
                inc_at_r_addr_prev += has_write_prev_open * inc_prev_open * eq_wa_prev;
            }
            if inc_at_r_addr_prev != val_eval_finals[base + 2] {
                return Err(PiCcsError::ProtocolError(
                    "twist/rollover_prev_total terminal value mismatch".into(),
                ));
            }

            // Enforce rollover equation: Init_i(r_addr) == Init_{i-1}(r_addr) + PrevTotal(i).
            let claimed_prev_total = val_eval
                .claimed_prev_inc_sum_total
                .ok_or_else(|| PiCcsError::ProtocolError("twist rollover missing claimed_prev_inc_sum_total".into()))?;
            let init_prev_at_r_addr = eval_init_at_r_addr(&prev_inst.init, prev_inst.k, r_addr)?;
            let init_cur_at_r_addr = eval_init_at_r_addr(&inst.init, inst.k, r_addr)?;
            if init_cur_at_r_addr != init_prev_at_r_addr + claimed_prev_total {
                return Err(PiCcsError::ProtocolError("twist rollover init check failed".into()));
            }
        }
    }

    Ok(RouteAMemoryVerifyOutput {
        claim_idx_end: claim_plan.claim_idx_end,
        twist_time_openings,
    })
}

fn verify_route_a_memory_step_no_shared_cpu_bus(
    tr: &mut Poseidon2Transcript,
    m: usize,
    core_t: usize,
    step: &StepInstanceBundle<Cmt, F, K>,
    prev_step: Option<&StepInstanceBundle<Cmt, F, K>>,
    ccs_out0: &MeInstance<Cmt, F, K>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    batched_claimed_sums: &[K],
    claim_idx_start: usize,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    shout_pre: &[ShoutAddrPreVerifyData],
    twist_pre: &[TwistAddrPreVerifyData],
    step_idx: usize,
) -> Result<RouteAMemoryVerifyOutput, PiCcsError> {
    let cpu_link = extract_trace_cpu_link_openings(m, core_t, step, ccs_out0)?;

    let chi_cycle_at_r_time = eq_points(r_time, r_cycle);
    if ccs_out0.r.as_slice() != r_time {
        return Err(PiCcsError::ProtocolError(
            "CPU ME output r mismatch (expected shared r_time)".into(),
        ));
    }
    let has_prev = prev_step.is_some();
    if has_prev {
        let prev = prev_step.expect("has_prev implies prev_step");
        if prev.mem_insts.len() != step.mem_insts.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist rollover requires stable mem instance count: prev has {}, current has {}",
                prev.mem_insts.len(),
                step.mem_insts.len()
            )));
        }
    }

    let proofs_mem = &mem_proof.proofs;
    let expected_proofs = step.lut_insts.len() + step.mem_insts.len();
    if proofs_mem.len() != expected_proofs {
        return Err(PiCcsError::InvalidInput(format!(
            "mem proof count mismatch (expected {}, got {})",
            expected_proofs,
            proofs_mem.len()
        )));
    }
    let total_shout_lanes: usize = step.lut_insts.iter().map(|inst| inst.lanes.max(1)).sum();
    if shout_pre.len() != total_shout_lanes {
        return Err(PiCcsError::InvalidInput(format!(
            "shout pre-time count mismatch (expected total_lanes={}, got {})",
            total_shout_lanes,
            shout_pre.len()
        )));
    }
    if twist_pre.len() != step.mem_insts.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "twist pre-time count mismatch (expected {}, got {})",
            step.mem_insts.len(),
            twist_pre.len()
        )));
    }

    let expected_shout_me_claims_time: usize = step
        .lut_insts
        .iter()
        .map(|inst| {
            let ell_addr = inst.d * inst.ell;
            let lanes = inst.lanes.max(1);
            plan_shout_addr_pages(m, step.mcs_inst.m_in, inst.steps, ell_addr, lanes).map(|p| p.len())
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .sum();
    if mem_proof.shout_me_claims_time.len() != expected_shout_me_claims_time {
        return Err(PiCcsError::InvalidInput(format!(
            "no-shared-bus expects 1 Shout ME(time) claim per Shout paging mat (expected {}, got {})",
            expected_shout_me_claims_time,
            mem_proof.shout_me_claims_time.len()
        )));
    }
    for (i, me) in mem_proof.shout_me_claims_time.iter().enumerate() {
        if me.r.as_slice() != r_time {
            return Err(PiCcsError::ProtocolError(format!(
                "Shout ME(time) r mismatch at shout_me_idx={i} (expected r_time)"
            )));
        }
    }

    if mem_proof.twist_me_claims_time.len() != step.mem_insts.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "no-shared-bus expects 1 Twist ME(time) claim per mem instance (expected {}, got {})",
            step.mem_insts.len(),
            mem_proof.twist_me_claims_time.len()
        )));
    }
    for (i, me) in mem_proof.twist_me_claims_time.iter().enumerate() {
        if me.r.as_slice() != r_time {
            return Err(PiCcsError::ProtocolError(format!(
                "Twist ME(time) r mismatch at mem_idx={i} (expected r_time)"
            )));
        }
    }

    let claim_plan = RouteATimeClaimPlan::build(step, claim_idx_start)?;
    if claim_plan.claim_idx_end > batched_final_values.len() || claim_plan.claim_idx_end > batched_claimed_sums.len() {
        return Err(PiCcsError::InvalidInput(
            "batched final_values / claimed_sums too short for claim plan".into(),
        ));
    }

    let any_event_table_shout = step
        .lut_insts
        .iter()
        .any(|inst| matches!(inst.table_spec, Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. })));
    if any_event_table_shout {
        for (idx, inst) in step.lut_insts.iter().enumerate() {
            if !matches!(inst.table_spec, Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. })) {
                return Err(PiCcsError::InvalidInput(format!(
                    "event-table Shout mode requires all Shout instances to use RiscvOpcodeEventTablePacked (lut_idx={idx})"
                )));
            }
        }
        if claim_plan.shout_event_trace_hash.is_none() {
            return Err(PiCcsError::ProtocolError(
                "event-table Shout expects a shout/event_trace_hash claim".into(),
            ));
        }
        if r_cycle.len() < 3 {
            return Err(PiCcsError::InvalidInput("event-table Shout requires ell_n >= 3".into()));
        }
    }
    let (event_alpha, event_beta, event_gamma) = if any_event_table_shout {
        (r_cycle[0], r_cycle[1], r_cycle[2])
    } else {
        (K::ZERO, K::ZERO, K::ZERO)
    };
    let mut shout_event_table_hash_claim_sum_total: K = K::ZERO;

    // Shout instances first.
    let mut shout_lane_base: usize = 0;
    let mut shout_has_sum: K = K::ZERO;
    let mut shout_val_sum: K = K::ZERO;
    let mut shout_lhs_sum: K = K::ZERO;
    let mut shout_rhs_sum: K = K::ZERO;
    let mut shout_table_id_sum: K = K::ZERO;

    let mut shout_me_base: usize = 0;
    for (lut_idx, inst) in step.lut_insts.iter().enumerate() {
        match &proofs_mem[lut_idx] {
            MemOrLutProof::Shout(_proof) => {}
            _ => return Err(PiCcsError::InvalidInput("expected Shout proof".into())),
        }

        let packed_layout = rv32_packed_shout_layout(&inst.table_spec)?;
        let packed_op = packed_layout.map(|(op, _time_bits)| op);
        let packed_time_bits = packed_layout.map(|(_op, time_bits)| time_bits).unwrap_or(0);
        let is_packed = packed_op.is_some();
        if packed_time_bits != 0 && packed_time_bits != r_cycle.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "event-table Shout expects time_bits == ell_n (time_bits={packed_time_bits}, ell_n={})",
                r_cycle.len()
            )));
        }

        let ell_addr = inst.d * inst.ell;
        let expected_lanes = inst.lanes.max(1);

        struct ShoutLaneOpen {
            addr_bits: Vec<K>,
            has_lookup: K,
            val: K,
        }
        let page_ell_addrs = plan_shout_addr_pages(m, step.mcs_inst.m_in, inst.steps, ell_addr, expected_lanes)?;
        if inst.comms.len() != page_ell_addrs.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "no-shared-bus mode requires Shout comms.len() to match the deterministic paging plan (lut_idx={lut_idx}, expected {}, comms.len()={})",
                page_ell_addrs.len(),
                inst.comms.len()
            )));
        }
        let shout_me_start = shout_me_base;
        let shout_me_end = shout_me_base
            .checked_add(page_ell_addrs.len())
            .ok_or_else(|| PiCcsError::ProtocolError("shout_me index overflow".into()))?;
        if shout_me_end > mem_proof.shout_me_claims_time.len() {
            return Err(PiCcsError::ProtocolError("missing Shout ME(time) claim(s)".into()));
        }
        shout_me_base = shout_me_end;

        let mut lane_addr_bits: Vec<Vec<K>> = vec![Vec::with_capacity(ell_addr); expected_lanes];
        let mut lane_has_lookup: Vec<Option<K>> = vec![None; expected_lanes];
        let mut lane_val: Vec<Option<K>> = vec![None; expected_lanes];

        for (page_idx, &page_ell_addr) in page_ell_addrs.iter().enumerate() {
            // Local bus layout for this page (stored inside its own committed witness mat).
            let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
                m,
                step.mcs_inst.m_in,
                inst.steps,
                core::iter::once((page_ell_addr, expected_lanes)),
                core::iter::empty::<(usize, usize)>(),
            )
            .map_err(|e| PiCcsError::InvalidInput(format!("Shout(Route A): bus layout failed: {e}")))?;
            if bus.shout_cols.len() != 1 || !bus.twist_cols.is_empty() {
                return Err(PiCcsError::ProtocolError(
                    "Shout(Route A): expected a shout-only bus layout with 1 instance".into(),
                ));
            }

            let me_time = mem_proof
                .shout_me_claims_time
                .get(shout_me_start + page_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("missing Shout ME(time) claim".into()))?;
            if me_time.c != inst.comms[page_idx] {
                return Err(PiCcsError::ProtocolError("Shout ME(time) commitment mismatch".into()));
            }
            let bus_y_base_time = me_time
                .y_scalars
                .len()
                .checked_sub(bus.bus_cols)
                .ok_or_else(|| PiCcsError::InvalidInput("Shout y_scalars too short for bus openings".into()))?;

            let inst_cols = bus
                .shout_cols
                .get(0)
                .ok_or_else(|| PiCcsError::ProtocolError("missing shout_cols[0]".into()))?;
            if inst_cols.lanes.len() != expected_lanes {
                return Err(PiCcsError::InvalidInput("shout lane count mismatch".into()));
            }

            for (lane_idx, shout_cols) in inst_cols.lanes.iter().enumerate() {
                if shout_cols.addr_bits.end - shout_cols.addr_bits.start != page_ell_addr {
                    return Err(PiCcsError::InvalidInput(format!(
                        "shout bus layout mismatch at lut_idx={lut_idx}, page_idx={page_idx}, lane={lane_idx}: expected page_ell_addr={page_ell_addr}"
                    )));
                }

                for col_id in shout_cols.addr_bits.clone() {
                    lane_addr_bits[lane_idx].push(
                        me_time
                            .y_scalars
                            .get(bus.y_scalar_index(bus_y_base_time, col_id))
                            .copied()
                            .ok_or_else(|| PiCcsError::ProtocolError("missing Shout addr_bits(time) opening".into()))?,
                    );
                }

                // Take `has_lookup`/`val` from page 0 (duplicates in later pages are ignored).
                if page_idx == 0 {
                    let has_lookup_open = me_time
                        .y_scalars
                        .get(bus.y_scalar_index(bus_y_base_time, shout_cols.has_lookup))
                        .copied()
                        .ok_or_else(|| PiCcsError::ProtocolError("missing Shout has_lookup(time) opening".into()))?;
                    let val_open = me_time
                        .y_scalars
                        .get(bus.y_scalar_index(bus_y_base_time, shout_cols.val))
                        .copied()
                        .ok_or_else(|| PiCcsError::ProtocolError("missing Shout val(time) opening".into()))?;
                    lane_has_lookup[lane_idx] = Some(has_lookup_open);
                    lane_val[lane_idx] = Some(val_open);
                }
            }
        }

        let mut lane_opens: Vec<ShoutLaneOpen> = Vec::with_capacity(expected_lanes);
        for lane_idx in 0..expected_lanes {
            if lane_addr_bits[lane_idx].len() != ell_addr {
                return Err(PiCcsError::ProtocolError(format!(
                    "Shout paging lane addr_bits len mismatch at lut_idx={lut_idx}, lane={lane_idx} (got {}, expected {ell_addr})",
                    lane_addr_bits[lane_idx].len()
                )));
            }
            let has_lookup = lane_has_lookup[lane_idx]
                .ok_or_else(|| PiCcsError::ProtocolError("missing Shout has_lookup(time) opening".into()))?;
            let val = lane_val[lane_idx]
                .ok_or_else(|| PiCcsError::ProtocolError("missing Shout val(time) opening".into()))?;

            lane_opens.push(ShoutLaneOpen {
                addr_bits: lane_addr_bits[lane_idx].clone(),
                has_lookup,
                val,
            });
        }

        // Fixed-lane Shout view: sum lanes must match the trace (skipped in event-table mode).
        if !any_event_table_shout {
            let lane_table_id = K::from(F::from_u64(rv32_shout_table_id_from_spec(&inst.table_spec)? as u64));
            for lane in lane_opens.iter() {
                shout_has_sum += lane.has_lookup;
                shout_val_sum += lane.val;
                shout_table_id_sum += lane.has_lookup * lane_table_id;
                if is_packed {
                    let packed_cols: &[K] = lane.addr_bits.get(packed_time_bits..).ok_or_else(|| {
                        PiCcsError::InvalidInput("packed RV32: addr_bits too short for time_bits prefix".into())
                    })?;
                    let lhs = *packed_cols
                        .get(0)
                        .ok_or_else(|| PiCcsError::InvalidInput("packed RV32: missing lhs opening".into()))?;
                    shout_lhs_sum += lhs;
                    if matches!(
                        packed_op,
                        Some(Rv32PackedShoutOp::Sll | Rv32PackedShoutOp::Srl | Rv32PackedShoutOp::Sra)
                    ) {
                        let shamt_bits: &[K] = packed_cols.get(1..6).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 shift: missing shamt bit opening(s)".into())
                        })?;
                        shout_rhs_sum += pack_bits_lsb(shamt_bits);
                    } else {
                        let rhs = *packed_cols
                            .get(1)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32: missing rhs opening".into()))?;
                        shout_rhs_sum += rhs;
                    }
                } else {
                    let (lhs, rhs) = unpack_interleaved_halves_lsb(&lane.addr_bits)?;
                    shout_lhs_sum += lhs;
                    shout_rhs_sum += rhs;
                }
            }
        }

        let shout_claims = claim_plan
            .shout
            .get(lut_idx)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("missing Shout claim schedule at index {}", lut_idx)))?;
        if shout_claims.lanes.len() != expected_lanes {
            return Err(PiCcsError::ProtocolError(format!(
                "Shout claim schedule lane count mismatch at lut_idx={lut_idx}: expected {expected_lanes}, got {}",
                shout_claims.lanes.len()
            )));
        }
        if shout_lane_base
            .checked_add(expected_lanes)
            .ok_or_else(|| PiCcsError::ProtocolError("shout lane index overflow".into()))?
            > shout_pre.len()
        {
            return Err(PiCcsError::ProtocolError("Shout pre-time lane indexing drift".into()));
        }

        // Route A Shout ordering in batched_time:
        // - value (time rounds only) per lane
        // - adapter (time rounds only) per lane
        // - aggregated bitness for (addr_bits, has_lookup)
        {
            let mut opens: Vec<K> = if is_packed {
                Vec::with_capacity(expected_lanes * (ell_addr + 1))
            } else {
                Vec::with_capacity(expected_lanes * (ell_addr + 1))
            };
            for lane in lane_opens.iter() {
                if is_packed {
                    if packed_time_bits > 0 {
                        opens.extend_from_slice(&lane.addr_bits[..packed_time_bits]);
                    }
                    let packed_cols: &[K] = lane.addr_bits.get(packed_time_bits..).ok_or_else(|| {
                        PiCcsError::InvalidInput("packed RV32: addr_bits too short for time_bits prefix".into())
                    })?;
                    match packed_op {
                        Some(Rv32PackedShoutOp::Add | Rv32PackedShoutOp::Sub) => {
                            let aux = *packed_cols
                                .get(2)
                                .ok_or_else(|| PiCcsError::InvalidInput("packed RV32: missing aux opening".into()))?;
                            opens.push(aux);
                            opens.push(lane.has_lookup);
                        }
                        Some(
                            Rv32PackedShoutOp::And
                            | Rv32PackedShoutOp::Andn
                            | Rv32PackedShoutOp::Or
                            | Rv32PackedShoutOp::Xor,
                        ) => {
                            opens.push(lane.has_lookup);
                        }
                        Some(Rv32PackedShoutOp::Eq | Rv32PackedShoutOp::Neq) => {
                            opens.push(lane.has_lookup);
                            opens.push(lane.val);
                            let borrow = *packed_cols.get(2).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 EQ/NEQ: missing borrow bit opening".into())
                            })?;
                            opens.push(borrow);
                            for i in 0..32 {
                                let b = *packed_cols.get(3 + i).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 EQ/NEQ: missing diff bit opening(s)".into())
                                })?;
                                opens.push(b);
                            }
                        }
                        Some(Rv32PackedShoutOp::Mul) => {
                            opens.push(lane.has_lookup);
                            for i in 0..32 {
                                let b = *packed_cols.get(2 + i).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 MUL: missing carry bit opening(s)".into())
                                })?;
                                opens.push(b);
                            }
                        }
                        Some(Rv32PackedShoutOp::Mulhu) => {
                            opens.push(lane.has_lookup);
                            for i in 0..32 {
                                let b = *packed_cols.get(2 + i).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 MULHU: missing lo bit opening(s)".into())
                                })?;
                                opens.push(b);
                            }
                        }
                        Some(Rv32PackedShoutOp::Mulh) => {
                            opens.push(lane.has_lookup);
                            let lhs_sign = *packed_cols.get(3).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 MULH: missing lhs_sign bit opening".into())
                            })?;
                            let rhs_sign = *packed_cols.get(4).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 MULH: missing rhs_sign bit opening".into())
                            })?;
                            opens.push(lhs_sign);
                            opens.push(rhs_sign);
                            for i in 0..32 {
                                let b = *packed_cols.get(6 + i).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 MULH: missing lo bit opening(s)".into())
                                })?;
                                opens.push(b);
                            }
                        }
                        Some(Rv32PackedShoutOp::Mulhsu) => {
                            opens.push(lane.has_lookup);
                            let lhs_sign = *packed_cols.get(3).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 MULHSU: missing lhs_sign bit opening".into())
                            })?;
                            let borrow = *packed_cols.get(4).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 MULHSU: missing borrow bit opening".into())
                            })?;
                            opens.push(lhs_sign);
                            opens.push(borrow);
                            for i in 0..32 {
                                let b = *packed_cols.get(5 + i).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 MULHSU: missing lo bit opening(s)".into())
                                })?;
                                opens.push(b);
                            }
                        }
                        Some(Rv32PackedShoutOp::Sll) => {
                            opens.push(lane.has_lookup);
                            for i in 0..5 {
                                let b = *packed_cols.get(1 + i).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 SLL: missing shamt bit opening(s)".into())
                                })?;
                                opens.push(b);
                            }
                            for i in 0..32 {
                                let b = *packed_cols.get(6 + i).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 SLL: missing carry bit opening(s)".into())
                                })?;
                                opens.push(b);
                            }
                        }
                        Some(Rv32PackedShoutOp::Srl) => {
                            opens.push(lane.has_lookup);
                            for i in 0..5 {
                                let b = *packed_cols.get(1 + i).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 SRL: missing shamt bit opening(s)".into())
                                })?;
                                opens.push(b);
                            }
                            for i in 0..32 {
                                let b = *packed_cols.get(6 + i).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 SRL: missing rem bit opening(s)".into())
                                })?;
                                opens.push(b);
                            }
                        }
                        Some(Rv32PackedShoutOp::Sra) => {
                            opens.push(lane.has_lookup);
                            for i in 0..5 {
                                let b = *packed_cols.get(1 + i).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 SRA: missing shamt bit opening(s)".into())
                                })?;
                                opens.push(b);
                            }
                            let sign = *packed_cols.get(6).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 SRA: missing sign bit opening".into())
                            })?;
                            opens.push(sign);
                            for i in 0..31 {
                                let b = *packed_cols.get(7 + i).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 SRA: missing rem bit opening(s)".into())
                                })?;
                                opens.push(b);
                            }
                        }
                        Some(Rv32PackedShoutOp::Slt) => {
                            opens.push(lane.val);
                            opens.push(lane.has_lookup);
                            let lhs_sign = *packed_cols.get(3).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 SLT: missing lhs_sign bit opening".into())
                            })?;
                            let rhs_sign = *packed_cols.get(4).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 SLT: missing rhs_sign bit opening".into())
                            })?;
                            opens.push(lhs_sign);
                            opens.push(rhs_sign);
                            for i in 0..32 {
                                let b = *packed_cols.get(5 + i).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 SLT: missing diff bit opening(s)".into())
                                })?;
                                opens.push(b);
                            }
                        }
                        Some(Rv32PackedShoutOp::Sltu) => {
                            opens.push(lane.val);
                            opens.push(lane.has_lookup);
                            for i in 0..32 {
                                let b = *packed_cols.get(3 + i).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 SLTU: missing diff bit opening(s)".into())
                                })?;
                                opens.push(b);
                            }
                        }
                        Some(Rv32PackedShoutOp::Divu | Rv32PackedShoutOp::Remu) => {
                            opens.push(lane.has_lookup);
                            let rhs_is_zero = *packed_cols.get(4).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 DIVU/REMU: missing rhs_is_zero".into())
                            })?;
                            opens.push(rhs_is_zero);
                            for i in 0..32 {
                                let b = *packed_cols.get(6 + i).ok_or_else(|| {
                                    PiCcsError::InvalidInput(
                                        "packed RV32 DIVU/REMU: missing diff bit opening(s)".into(),
                                    )
                                })?;
                                opens.push(b);
                            }
                        }
                        Some(Rv32PackedShoutOp::Div) => {
                            opens.push(lane.has_lookup);
                            let rhs_is_zero = *packed_cols.get(5).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 DIV: missing rhs_is_zero".into())
                            })?;
                            let lhs_sign = *packed_cols
                                .get(6)
                                .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing lhs_sign".into()))?;
                            let rhs_sign = *packed_cols
                                .get(7)
                                .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing rhs_sign".into()))?;
                            let q_is_zero = *packed_cols
                                .get(9)
                                .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing q_is_zero".into()))?;
                            opens.push(rhs_is_zero);
                            opens.push(lhs_sign);
                            opens.push(rhs_sign);
                            opens.push(q_is_zero);
                            for i in 0..32 {
                                let b = *packed_cols.get(11 + i).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 DIV: missing diff bit opening(s)".into())
                                })?;
                                opens.push(b);
                            }
                        }
                        Some(Rv32PackedShoutOp::Rem) => {
                            opens.push(lane.has_lookup);
                            let rhs_is_zero = *packed_cols.get(5).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 REM: missing rhs_is_zero".into())
                            })?;
                            let lhs_sign = *packed_cols
                                .get(6)
                                .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing lhs_sign".into()))?;
                            let rhs_sign = *packed_cols
                                .get(7)
                                .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing rhs_sign".into()))?;
                            let r_is_zero = *packed_cols
                                .get(9)
                                .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing r_is_zero".into()))?;
                            opens.push(rhs_is_zero);
                            opens.push(lhs_sign);
                            opens.push(rhs_sign);
                            opens.push(r_is_zero);
                            for i in 0..32 {
                                let b = *packed_cols.get(11 + i).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 REM: missing diff bit opening(s)".into())
                                })?;
                                opens.push(b);
                            }
                        }
                        None => {
                            return Err(PiCcsError::ProtocolError(
                                "packed_op drift: is_packed=true but packed_op=None".into(),
                            ));
                        }
                    }
                } else {
                    opens.extend_from_slice(&lane.addr_bits);
                    opens.push(lane.has_lookup);
                }
            }
            let weights = bitness_weights(r_cycle, opens.len(), 0x5348_4F55_54u64 + lut_idx as u64);
            let mut acc = K::ZERO;
            for (w, b) in weights.iter().zip(opens.iter()) {
                acc += *w * *b * (*b - K::ONE);
            }
            let expected = chi_cycle_at_r_time * acc;
            if expected != batched_final_values[shout_claims.bitness] {
                return Err(PiCcsError::ProtocolError(
                    "shout/bitness terminal value mismatch".into(),
                ));
            }
        }

        for (lane_idx, lane) in lane_opens.iter().enumerate() {
            let pre = shout_pre.get(shout_lane_base + lane_idx).ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "missing pre-time Shout lane data at index {}",
                    shout_lane_base + lane_idx
                ))
            })?;
            let lane_claims = shout_claims
                .lanes
                .get(lane_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("shout claim schedule lane idx drift".into()))?;

            let value_claim = batched_claimed_sums[lane_claims.value];
            let value_final = batched_final_values[lane_claims.value];
            let adapter_claim = batched_claimed_sums[lane_claims.adapter];
            let adapter_final = batched_final_values[lane_claims.adapter];

            let expected_value_final = if let Some(op) = packed_op {
                let packed_cols: &[K] = lane.addr_bits.get(packed_time_bits..).ok_or_else(|| {
                    PiCcsError::InvalidInput("packed RV32: addr_bits too short for time_bits prefix".into())
                })?;
                match op {
                    Rv32PackedShoutOp::And
                    | Rv32PackedShoutOp::Andn
                    | Rv32PackedShoutOp::Or
                    | Rv32PackedShoutOp::Xor => {
                        let inv2 = K::from_u64(2).inverse();
                        let inv6 = K::from_u64(6).inverse();

                        let digit_bits = |x: K| -> (K, K) {
                            let xm1 = x - K::ONE;
                            let xm2 = x - K::from_u64(2);
                            let xm3 = x - K::from_u64(3);

                            let x_xm1 = x * xm1;
                            let l1 = (x * xm2 * xm3) * inv2;
                            let l3 = (x_xm1 * xm2) * inv6;
                            let l2 = -(x_xm1 * xm3) * inv2;

                            let bit0 = l1 + l3;
                            let bit1 = l2 + l3;
                            (bit0, bit1)
                        };

                        let digit_op = |a: K, b: K| -> K {
                            let (a0, a1) = digit_bits(a);
                            let (b0, b1) = digit_bits(b);
                            let two = K::from_u64(2);
                            match op {
                                Rv32PackedShoutOp::And => {
                                    let r0 = a0 * b0;
                                    let r1 = a1 * b1;
                                    r0 + two * r1
                                }
                                Rv32PackedShoutOp::Andn => {
                                    let r0 = a0 * (K::ONE - b0);
                                    let r1 = a1 * (K::ONE - b1);
                                    r0 + two * r1
                                }
                                Rv32PackedShoutOp::Or => {
                                    let r0 = a0 + b0 - a0 * b0;
                                    let r1 = a1 + b1 - a1 * b1;
                                    r0 + two * r1
                                }
                                Rv32PackedShoutOp::Xor => {
                                    let r0 = a0 + b0 - two * a0 * b0;
                                    let r1 = a1 + b1 - two * a1 * b1;
                                    r0 + two * r1
                                }
                                _ => unreachable!(),
                            }
                        };

                        let mut out = K::ZERO;
                        for i in 0..16usize {
                            let a = *packed_cols.get(2 + i).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 bitwise: missing lhs digit opening(s)".into())
                            })?;
                            let b = *packed_cols.get(18 + i).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 bitwise: missing rhs digit opening(s)".into())
                            })?;
                            let pow = K::from_u64(1u64 << (2 * i));
                            out += digit_op(a, b) * pow;
                        }
                        chi_cycle_at_r_time * lane.has_lookup * (out - lane.val)
                    }
                    _ => {
                        let lhs = *packed_cols
                            .get(0)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32: missing lhs opening".into()))?;
                        let rhs = *packed_cols
                            .get(1)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32: missing rhs opening".into()))?;
                        let aux = *packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32: missing aux opening".into()))?;
                        let expr = match op {
                            Rv32PackedShoutOp::Add => {
                                let two32 = K::from_u64(1u64 << 32);
                                lhs + rhs - lane.val - aux * two32
                            }
                            Rv32PackedShoutOp::Sub => {
                                let two32 = K::from_u64(1u64 << 32);
                                lhs - rhs - lane.val + aux * two32
                            }
                            Rv32PackedShoutOp::Mul => {
                                let two32 = K::from_u64(1u64 << 32);
                                let mut carry = K::ZERO;
                                for i in 0..32 {
                                    let b = *packed_cols.get(2 + i).ok_or_else(|| {
                                        PiCcsError::InvalidInput("packed RV32 MUL: missing carry bit opening(s)".into())
                                    })?;
                                    carry += b * K::from_u64(1u64 << i);
                                }
                                lhs * rhs - lane.val - carry * two32
                            }
                            Rv32PackedShoutOp::Mulhu => {
                                let two32 = K::from_u64(1u64 << 32);
                                let mut lo = K::ZERO;
                                for i in 0..32 {
                                    let b = *packed_cols.get(2 + i).ok_or_else(|| {
                                        PiCcsError::InvalidInput("packed RV32 MULHU: missing lo bit opening(s)".into())
                                    })?;
                                    lo += b * K::from_u64(1u64 << i);
                                }
                                lhs * rhs - lo - lane.val * two32
                            }
                            Rv32PackedShoutOp::Mulh => {
                                let two32 = K::from_u64(1u64 << 32);
                                let mut lo = K::ZERO;
                                for i in 0..32 {
                                    let b = *packed_cols.get(6 + i).ok_or_else(|| {
                                        PiCcsError::InvalidInput("packed RV32 MULH: missing lo bit opening(s)".into())
                                    })?;
                                    lo += b * K::from_u64(1u64 << i);
                                }
                                // Value oracle is the unsigned product decomposition: lhs*rhs = lo + hi*2^32.
                                // Here `aux` is the `hi` opening.
                                lhs * rhs - lo - aux * two32
                            }
                            Rv32PackedShoutOp::Mulhsu => {
                                let two32 = K::from_u64(1u64 << 32);
                                let mut lo = K::ZERO;
                                for i in 0..32 {
                                    let b = *packed_cols.get(5 + i).ok_or_else(|| {
                                        PiCcsError::InvalidInput("packed RV32 MULHSU: missing lo bit opening(s)".into())
                                    })?;
                                    lo += b * K::from_u64(1u64 << i);
                                }
                                lhs * rhs - lo - aux * two32
                            }
                            Rv32PackedShoutOp::Eq => {
                                let mut prod = K::ONE;
                                for i in 0..32usize {
                                    let b = *packed_cols.get(3 + i).ok_or_else(|| {
                                        PiCcsError::InvalidInput("packed RV32 EQ: missing diff bit opening(s)".into())
                                    })?;
                                    prod *= K::ONE - b;
                                }
                                lane.val - prod
                            }
                            Rv32PackedShoutOp::Neq => {
                                let mut prod = K::ONE;
                                for i in 0..32usize {
                                    let b = *packed_cols.get(3 + i).ok_or_else(|| {
                                        PiCcsError::InvalidInput("packed RV32 NEQ: missing diff bit opening(s)".into())
                                    })?;
                                    prod *= K::ONE - b;
                                }
                                lane.val + prod - K::ONE
                            }
                            Rv32PackedShoutOp::Divu => {
                                let z = *packed_cols.get(4).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 DIVU: missing rhs_is_zero opening".into())
                                })?;
                                let all_ones = K::from_u64(u32::MAX as u64);
                                z * (lane.val - all_ones) + (K::ONE - z) * (lhs - rhs * lane.val - aux)
                            }
                            Rv32PackedShoutOp::Remu => {
                                let z = *packed_cols.get(4).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 REMU: missing rhs_is_zero opening".into())
                                })?;
                                z * (lane.val - lhs) + (K::ONE - z) * (lhs - rhs * aux - lane.val)
                            }
                            Rv32PackedShoutOp::Div => {
                                let z = *packed_cols.get(5).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 DIV: missing rhs_is_zero opening".into())
                                })?;
                                let lhs_sign = *packed_cols.get(6).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 DIV: missing lhs_sign opening".into())
                                })?;
                                let rhs_sign = *packed_cols.get(7).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 DIV: missing rhs_sign opening".into())
                                })?;
                                let q_is_zero = *packed_cols.get(9).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 DIV: missing q_is_zero opening".into())
                                })?;

                                let two = K::from_u64(2);
                                let two32 = K::from_u64(1u64 << 32);
                                let all_ones = K::from_u64(u32::MAX as u64);

                                // div_sign = lhs_sign XOR rhs_sign
                                let div_sign = lhs_sign + rhs_sign - two * lhs_sign * rhs_sign;
                                // q_signed = ±q_abs (two's complement), with `q_is_zero` handling -0.
                                let neg_q = (K::ONE - q_is_zero) * (two32 - aux);
                                let q_signed = (K::ONE - div_sign) * aux + div_sign * neg_q;

                                z * (lane.val - all_ones) + (K::ONE - z) * (lane.val - q_signed)
                            }
                            Rv32PackedShoutOp::Rem => {
                                let z = *packed_cols.get(5).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 REM: missing rhs_is_zero opening".into())
                                })?;
                                let lhs_sign = *packed_cols.get(6).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 REM: missing lhs_sign opening".into())
                                })?;
                                let r_abs = *packed_cols.get(3).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 REM: missing r_abs opening".into())
                                })?;
                                let r_is_zero = *packed_cols.get(9).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 REM: missing r_is_zero opening".into())
                                })?;
                                let two32 = K::from_u64(1u64 << 32);
                                let neg_r = (K::ONE - r_is_zero) * (two32 - r_abs);
                                let r_signed = (K::ONE - lhs_sign) * r_abs + lhs_sign * neg_r;
                                z * (lane.val - lhs) + (K::ONE - z) * (lane.val - r_signed)
                            }
                            Rv32PackedShoutOp::Sll => {
                                let two32 = K::from_u64(1u64 << 32);
                                let pow2_const: [K; 5] = [
                                    K::from_u64(2),
                                    K::from_u64(4),
                                    K::from_u64(16),
                                    K::from_u64(256),
                                    K::from_u64(65536),
                                ];
                                let mut pow2 = K::ONE;
                                for i in 0..5 {
                                    let b = *packed_cols.get(1 + i).ok_or_else(|| {
                                        PiCcsError::InvalidInput("packed RV32 SLL: missing shamt bit opening(s)".into())
                                    })?;
                                    pow2 *= K::ONE + b * (pow2_const[i] - K::ONE);
                                }
                                let mut carry = K::ZERO;
                                for i in 0..32 {
                                    let b = *packed_cols.get(6 + i).ok_or_else(|| {
                                        PiCcsError::InvalidInput("packed RV32 SLL: missing carry bit opening(s)".into())
                                    })?;
                                    carry += b * K::from_u64(1u64 << i);
                                }
                                lhs * pow2 - lane.val - carry * two32
                            }
                            Rv32PackedShoutOp::Srl => {
                                let pow2_const: [K; 5] = [
                                    K::from_u64(2),
                                    K::from_u64(4),
                                    K::from_u64(16),
                                    K::from_u64(256),
                                    K::from_u64(65536),
                                ];
                                let mut pow2 = K::ONE;
                                for i in 0..5 {
                                    let b = *packed_cols.get(1 + i).ok_or_else(|| {
                                        PiCcsError::InvalidInput("packed RV32 SRL: missing shamt bit opening(s)".into())
                                    })?;
                                    pow2 *= K::ONE + b * (pow2_const[i] - K::ONE);
                                }
                                let mut rem = K::ZERO;
                                for i in 0..32 {
                                    let b = *packed_cols.get(6 + i).ok_or_else(|| {
                                        PiCcsError::InvalidInput("packed RV32 SRL: missing rem bit opening(s)".into())
                                    })?;
                                    rem += b * K::from_u64(1u64 << i);
                                }
                                lhs - lane.val * pow2 - rem
                            }
                            Rv32PackedShoutOp::Sra => {
                                let two32 = K::from_u64(1u64 << 32);
                                let pow2_const: [K; 5] = [
                                    K::from_u64(2),
                                    K::from_u64(4),
                                    K::from_u64(16),
                                    K::from_u64(256),
                                    K::from_u64(65536),
                                ];
                                let mut pow2 = K::ONE;
                                for i in 0..5 {
                                    let b = *packed_cols.get(1 + i).ok_or_else(|| {
                                        PiCcsError::InvalidInput("packed RV32 SRA: missing shamt bit opening(s)".into())
                                    })?;
                                    pow2 *= K::ONE + b * (pow2_const[i] - K::ONE);
                                }
                                let sign = *packed_cols.get(6).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 SRA: missing sign bit opening".into())
                                })?;
                                let mut rem = K::ZERO;
                                for i in 0..31 {
                                    let b = *packed_cols.get(7 + i).ok_or_else(|| {
                                        PiCcsError::InvalidInput("packed RV32 SRA: missing rem bit opening(s)".into())
                                    })?;
                                    rem += b * K::from_u64(1u64 << i);
                                }
                                let corr = sign * two32 * (K::ONE - pow2);
                                lhs - lane.val * pow2 - rem - corr
                            }
                            Rv32PackedShoutOp::Slt => {
                                let two31 = K::from_u64(1u64 << 31);
                                let two32 = K::from_u64(1u64 << 32);
                                let two = K::from_u64(2);
                                let lhs_sign = *packed_cols.get(3).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 SLT: missing lhs_sign bit opening".into())
                                })?;
                                let rhs_sign = *packed_cols.get(4).ok_or_else(|| {
                                    PiCcsError::InvalidInput("packed RV32 SLT: missing rhs_sign bit opening".into())
                                })?;
                                let lhs_b = lhs + (K::ONE - two * lhs_sign) * two31;
                                let rhs_b = rhs + (K::ONE - two * rhs_sign) * two31;
                                lhs_b - rhs_b - aux + lane.val * two32
                            }
                            Rv32PackedShoutOp::Sltu => {
                                let two32 = K::from_u64(1u64 << 32);
                                lhs - rhs - aux + lane.val * two32
                            }
                            _ => {
                                return Err(PiCcsError::ProtocolError(
                                    "packed RV32 expected_value_final match drift".into(),
                                ));
                            }
                        };
                        chi_cycle_at_r_time * lane.has_lookup * expr
                    }
                }
            } else {
                chi_cycle_at_r_time * lane.has_lookup * lane.val
            };
            if expected_value_final != value_final {
                return Err(PiCcsError::ProtocolError("shout value terminal value mismatch".into()));
            }

            let expected_adapter_final = if let Some(op) = packed_op {
                let packed_cols: &[K] = lane.addr_bits.get(packed_time_bits..).ok_or_else(|| {
                    PiCcsError::InvalidInput("packed RV32: addr_bits too short for time_bits prefix".into())
                })?;
                match op {
                    Rv32PackedShoutOp::And
                    | Rv32PackedShoutOp::Andn
                    | Rv32PackedShoutOp::Or
                    | Rv32PackedShoutOp::Xor => {
                        let weights = bitness_weights(r_cycle, 34, 0x4249_5457_4F50u64 + lut_idx as u64);
                        if weights.len() != 34 {
                            return Err(PiCcsError::ProtocolError(
                                "packed RV32 bitwise: weights len drift".into(),
                            ));
                        }
                        let w_lhs = weights[0];
                        let w_rhs = weights[1];
                        let w_digits = &weights[2..];

                        let lhs = *packed_cols.get(0).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 bitwise: missing lhs opening".into())
                        })?;
                        let rhs = *packed_cols.get(1).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 bitwise: missing rhs opening".into())
                        })?;

                        let mut lhs_recon = K::ZERO;
                        let mut rhs_recon = K::ZERO;
                        let mut range_sum = K::ZERO;
                        for i in 0..16usize {
                            let a = *packed_cols.get(2 + i).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 bitwise: missing lhs digit opening(s)".into())
                            })?;
                            let b = *packed_cols.get(18 + i).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 bitwise: missing rhs digit opening(s)".into())
                            })?;
                            let pow = K::from_u64(1u64 << (2 * i));
                            lhs_recon += a * pow;
                            rhs_recon += b * pow;

                            let ga = a * (a - K::ONE) * (a - K::from_u64(2)) * (a - K::from_u64(3));
                            let gb = b * (b - K::ONE) * (b - K::from_u64(2)) * (b - K::from_u64(3));
                            range_sum += w_digits[i] * ga;
                            range_sum += w_digits[16 + i] * gb;
                        }
                        let expr = w_lhs * (lhs - lhs_recon) + w_rhs * (rhs - rhs_recon) + range_sum;
                        chi_cycle_at_r_time * lane.has_lookup * expr
                    }
                    Rv32PackedShoutOp::Mulh => {
                        let weights = bitness_weights(r_cycle, 2, 0x4D55_4C48_4144_5054u64 + lut_idx as u64);
                        let w0 = weights[0];
                        let w1 = weights[1];

                        let lhs = *packed_cols
                            .get(0)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULH: missing lhs opening".into()))?;
                        let rhs = *packed_cols
                            .get(1)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULH: missing rhs opening".into()))?;
                        let hi = *packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULH: missing hi opening".into()))?;
                        let lhs_sign = *packed_cols.get(3).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 MULH: missing lhs_sign opening".into())
                        })?;
                        let rhs_sign = *packed_cols.get(4).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 MULH: missing rhs_sign opening".into())
                        })?;
                        let k = *packed_cols
                            .get(5)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULH: missing k opening".into()))?;

                        let two32 = K::from_u64(1u64 << 32);
                        let eq_expr = hi - lhs_sign * rhs - rhs_sign * lhs + k * two32 - lane.val;
                        let range = k * (k - K::ONE) * (k - K::from_u64(2));
                        chi_cycle_at_r_time * lane.has_lookup * (w0 * eq_expr + w1 * range)
                    }
                    Rv32PackedShoutOp::Mulhsu => {
                        let rhs = *packed_cols.get(1).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 MULHSU: missing rhs opening".into())
                        })?;
                        let hi = *packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 MULHSU: missing hi opening".into()))?;
                        let lhs_sign = *packed_cols.get(3).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 MULHSU: missing lhs_sign opening".into())
                        })?;
                        let borrow = *packed_cols.get(4).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 MULHSU: missing borrow opening".into())
                        })?;
                        let two32 = K::from_u64(1u64 << 32);
                        let expr = hi - lhs_sign * rhs - lane.val + borrow * two32;
                        chi_cycle_at_r_time * lane.has_lookup * expr
                    }
                    Rv32PackedShoutOp::Divu => {
                        let weights = bitness_weights(r_cycle, 4, 0x4449_5655_4144_5054u64 + lut_idx as u64);
                        let w = [weights[0], weights[1], weights[2], weights[3]];

                        let rhs = *packed_cols
                            .get(1)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIVU: missing rhs opening".into()))?;
                        let rem = *packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIVU: missing rem opening".into()))?;
                        let z = *packed_cols.get(4).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 DIVU: missing rhs_is_zero opening".into())
                        })?;
                        let diff = *packed_cols
                            .get(5)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIVU: missing diff opening".into()))?;

                        let mut sum = K::ZERO;
                        for i in 0..32 {
                            let b = *packed_cols.get(6 + i).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 DIVU: missing diff bit opening(s)".into())
                            })?;
                            sum += b * K::from_u64(1u64 << i);
                        }

                        let two32 = K::from_u64(1u64 << 32);
                        let c0 = z * (K::ONE - z);
                        let c1 = z * rhs;
                        let c2 = (K::ONE - z) * (rem - rhs - diff + two32);
                        let c3 = diff - sum;
                        let expr = w[0] * c0 + w[1] * c1 + w[2] * c2 + w[3] * c3;
                        chi_cycle_at_r_time * lane.has_lookup * expr
                    }
                    Rv32PackedShoutOp::Remu => {
                        let weights = bitness_weights(r_cycle, 4, 0x4449_5655_4144_5054u64 + lut_idx as u64);
                        let w = [weights[0], weights[1], weights[2], weights[3]];

                        let rhs = *packed_cols
                            .get(1)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REMU: missing rhs opening".into()))?;
                        let z = *packed_cols.get(4).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 REMU: missing rhs_is_zero opening".into())
                        })?;
                        let diff = *packed_cols
                            .get(5)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REMU: missing diff opening".into()))?;

                        let mut sum = K::ZERO;
                        for i in 0..32 {
                            let b = *packed_cols.get(6 + i).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 REMU: missing diff bit opening(s)".into())
                            })?;
                            sum += b * K::from_u64(1u64 << i);
                        }

                        let two32 = K::from_u64(1u64 << 32);
                        let c0 = z * (K::ONE - z);
                        let c1 = z * rhs;
                        let c2 = (K::ONE - z) * (lane.val - rhs - diff + two32);
                        let c3 = diff - sum;
                        let expr = w[0] * c0 + w[1] * c1 + w[2] * c2 + w[3] * c3;
                        chi_cycle_at_r_time * lane.has_lookup * expr
                    }
                    Rv32PackedShoutOp::Div => {
                        let weights = bitness_weights(r_cycle, 7, 0x4449_565F_4144_5054u64 + lut_idx as u64);
                        let w = [
                            weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6],
                        ];

                        let lhs = *packed_cols
                            .get(0)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing lhs opening".into()))?;
                        let rhs = *packed_cols
                            .get(1)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing rhs opening".into()))?;
                        let q_abs = *packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing q_abs opening".into()))?;
                        let r_abs = *packed_cols
                            .get(3)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing r_abs opening".into()))?;
                        let z = *packed_cols.get(5).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 DIV: missing rhs_is_zero opening".into())
                        })?;
                        let lhs_sign = *packed_cols.get(6).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 DIV: missing lhs_sign opening".into())
                        })?;
                        let rhs_sign = *packed_cols.get(7).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 DIV: missing rhs_sign opening".into())
                        })?;
                        let q_is_zero = *packed_cols.get(9).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 DIV: missing q_is_zero opening".into())
                        })?;
                        let diff = *packed_cols
                            .get(10)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 DIV: missing diff opening".into()))?;

                        let mut sum = K::ZERO;
                        for i in 0..32 {
                            let b = *packed_cols.get(11 + i).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 DIV: missing diff bit opening(s)".into())
                            })?;
                            sum += b * K::from_u64(1u64 << i);
                        }

                        let two = K::from_u64(2);
                        let two32 = K::from_u64(1u64 << 32);
                        let lhs_abs = lhs + lhs_sign * (two32 - two * lhs);
                        let rhs_abs = rhs + rhs_sign * (two32 - two * rhs);

                        let c0 = z * (K::ONE - z);
                        let c1 = z * rhs;
                        let c2 = q_is_zero * (K::ONE - q_is_zero);
                        let c3 = q_is_zero * q_abs;
                        let c4 = (K::ONE - z) * (lhs_abs - rhs_abs * q_abs - r_abs);
                        let c5 = (K::ONE - z) * (r_abs - rhs_abs - diff + two32);
                        let c6 = diff - sum;
                        let expr = w[0] * c0 + w[1] * c1 + w[2] * c2 + w[3] * c3 + w[4] * c4 + w[5] * c5 + w[6] * c6;
                        chi_cycle_at_r_time * lane.has_lookup * expr
                    }
                    Rv32PackedShoutOp::Rem => {
                        let weights = bitness_weights(r_cycle, 7, 0x4449_565F_4144_5054u64 + lut_idx as u64);
                        let w = [
                            weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6],
                        ];

                        let lhs = *packed_cols
                            .get(0)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing lhs opening".into()))?;
                        let rhs = *packed_cols
                            .get(1)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing rhs opening".into()))?;
                        let q_abs = *packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing q_abs opening".into()))?;
                        let r_abs = *packed_cols
                            .get(3)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing r_abs opening".into()))?;
                        let z = *packed_cols.get(5).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 REM: missing rhs_is_zero opening".into())
                        })?;
                        let lhs_sign = *packed_cols.get(6).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 REM: missing lhs_sign opening".into())
                        })?;
                        let rhs_sign = *packed_cols.get(7).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 REM: missing rhs_sign opening".into())
                        })?;
                        let r_is_zero = *packed_cols.get(9).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 REM: missing r_is_zero opening".into())
                        })?;
                        let diff = *packed_cols
                            .get(10)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 REM: missing diff opening".into()))?;

                        let mut sum = K::ZERO;
                        for i in 0..32 {
                            let b = *packed_cols.get(11 + i).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 REM: missing diff bit opening(s)".into())
                            })?;
                            sum += b * K::from_u64(1u64 << i);
                        }

                        let two = K::from_u64(2);
                        let two32 = K::from_u64(1u64 << 32);
                        let lhs_abs = lhs + lhs_sign * (two32 - two * lhs);
                        let rhs_abs = rhs + rhs_sign * (two32 - two * rhs);

                        let c0 = z * (K::ONE - z);
                        let c1 = z * rhs;
                        let c2 = r_is_zero * (K::ONE - r_is_zero);
                        let c3 = r_is_zero * r_abs;
                        let c4 = (K::ONE - z) * (lhs_abs - rhs_abs * q_abs - r_abs);
                        let c5 = (K::ONE - z) * (r_abs - rhs_abs - diff + two32);
                        let c6 = diff - sum;
                        let expr = w[0] * c0 + w[1] * c1 + w[2] * c2 + w[3] * c3 + w[4] * c4 + w[5] * c5 + w[6] * c6;
                        chi_cycle_at_r_time * lane.has_lookup * expr
                    }
                    Rv32PackedShoutOp::Add
                    | Rv32PackedShoutOp::Sub
                    | Rv32PackedShoutOp::Sll
                    | Rv32PackedShoutOp::Mul
                    | Rv32PackedShoutOp::Mulhu => K::ZERO,
                    Rv32PackedShoutOp::Srl => {
                        let mut shamt: [K; 5] = [K::ZERO; 5];
                        for i in 0..5 {
                            shamt[i] = *packed_cols.get(1 + i).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 SRL: missing shamt bit opening(s)".into())
                            })?;
                        }
                        let mut rem: [K; 32] = [K::ZERO; 32];
                        for i in 0..32 {
                            rem[i] = *packed_cols.get(6 + i).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 SRL: missing rem bit opening(s)".into())
                            })?;
                        }

                        // tail_sum[s] = Σ_{i≥s} 2^i · rem_i
                        let mut tail_sum: [K; 32] = [K::ZERO; 32];
                        let mut tail = K::ZERO;
                        for i in (0..32).rev() {
                            tail += rem[i] * K::from_u64(1u64 << i);
                            tail_sum[i] = tail;
                        }

                        let mut expr = K::ZERO;
                        for s in 0..32usize {
                            let mut prod = K::ONE;
                            for j in 0..5usize {
                                let b = shamt[j];
                                if ((s >> j) & 1) == 1 {
                                    prod *= b;
                                } else {
                                    prod *= K::ONE - b;
                                }
                            }
                            expr += prod * tail_sum[s];
                        }

                        chi_cycle_at_r_time * lane.has_lookup * expr
                    }
                    Rv32PackedShoutOp::Sra => {
                        let mut shamt: [K; 5] = [K::ZERO; 5];
                        for i in 0..5 {
                            shamt[i] = *packed_cols.get(1 + i).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 SRA: missing shamt bit opening(s)".into())
                            })?;
                        }
                        let mut rem: [K; 31] = [K::ZERO; 31];
                        for i in 0..31 {
                            rem[i] = *packed_cols.get(7 + i).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 SRA: missing rem bit opening(s)".into())
                            })?;
                        }

                        // tail_sum[s] = Σ_{i≥s} 2^i · rem_i, with tail_sum[31]=0.
                        let mut tail_sum: [K; 32] = [K::ZERO; 32];
                        let mut tail = K::ZERO;
                        for i in (0..31).rev() {
                            tail += rem[i] * K::from_u64(1u64 << i);
                            tail_sum[i] = tail;
                        }
                        tail_sum[31] = K::ZERO;

                        let mut expr = K::ZERO;
                        for s in 0..32usize {
                            let mut prod = K::ONE;
                            for j in 0..5usize {
                                let b = shamt[j];
                                if ((s >> j) & 1) == 1 {
                                    prod *= b;
                                } else {
                                    prod *= K::ONE - b;
                                }
                            }
                            expr += prod * tail_sum[s];
                        }

                        chi_cycle_at_r_time * lane.has_lookup * expr
                    }
                    Rv32PackedShoutOp::Slt => {
                        let diff = *packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 SLT: missing diff opening".into()))?;
                        let mut sum = K::ZERO;
                        for i in 0..32 {
                            let b = *packed_cols.get(5 + i).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 SLT: missing diff bit opening(s)".into())
                            })?;
                            sum += b * K::from_u64(1u64 << i);
                        }
                        chi_cycle_at_r_time * lane.has_lookup * (diff - sum)
                    }
                    Rv32PackedShoutOp::Eq | Rv32PackedShoutOp::Neq => {
                        let lhs = *packed_cols
                            .get(0)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32: missing lhs opening".into()))?;
                        let rhs = *packed_cols
                            .get(1)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32: missing rhs opening".into()))?;
                        let borrow = *packed_cols.get(2).ok_or_else(|| {
                            PiCcsError::InvalidInput("packed RV32 EQ/NEQ: missing borrow bit opening".into())
                        })?;
                        let mut diff = K::ZERO;
                        for i in 0..32usize {
                            let b = *packed_cols.get(3 + i).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 EQ/NEQ: missing diff bit opening(s)".into())
                            })?;
                            diff += b * K::from_u64(1u64 << i);
                        }
                        let two32 = K::from_u64(1u64 << 32);
                        chi_cycle_at_r_time * lane.has_lookup * (lhs - rhs - diff + borrow * two32)
                    }
                    Rv32PackedShoutOp::Sltu => {
                        let diff = *packed_cols
                            .get(2)
                            .ok_or_else(|| PiCcsError::InvalidInput("packed RV32 SLTU: missing diff opening".into()))?;
                        let mut sum = K::ZERO;
                        for i in 0..32 {
                            let b = *packed_cols.get(3 + i).ok_or_else(|| {
                                PiCcsError::InvalidInput("packed RV32 SLTU: missing diff bit opening(s)".into())
                            })?;
                            sum += b * K::from_u64(1u64 << i);
                        }
                        chi_cycle_at_r_time * lane.has_lookup * (diff - sum)
                    }
                }
            } else {
                let eq_addr = eq_bits_prod(&lane.addr_bits, &pre.r_addr)?;
                chi_cycle_at_r_time * lane.has_lookup * eq_addr
            };
            if expected_adapter_final != adapter_final {
                return Err(PiCcsError::ProtocolError(
                    "shout adapter terminal value mismatch".into(),
                ));
            }

            // Optional: event-table Shout hash linkage claim (per-lane).
            if packed_time_bits > 0 {
                let claim_idx = lane_claims.event_table_hash.ok_or_else(|| {
                    PiCcsError::ProtocolError("event-table Shout expects a shout/event_table_hash claim".into())
                })?;
                let claim_sum = batched_claimed_sums[claim_idx];
                let final_value = batched_final_values[claim_idx];

                let time_bits_open: &[K] = lane
                    .addr_bits
                    .get(..packed_time_bits)
                    .ok_or_else(|| PiCcsError::InvalidInput("event-table Shout: missing time bits openings".into()))?;
                let packed_cols: &[K] = lane.addr_bits.get(packed_time_bits..).ok_or_else(|| {
                    PiCcsError::InvalidInput("packed RV32: addr_bits too short for time_bits prefix".into())
                })?;

                let lhs = *packed_cols
                    .get(0)
                    .ok_or_else(|| PiCcsError::InvalidInput("event-table hash: missing lhs opening".into()))?;
                let rhs = if matches!(
                    packed_op,
                    Some(Rv32PackedShoutOp::Sll | Rv32PackedShoutOp::Srl | Rv32PackedShoutOp::Sra)
                ) {
                    let shamt_bits: &[K] = packed_cols.get(1..6).ok_or_else(|| {
                        PiCcsError::InvalidInput("event-table hash: missing shamt bit opening(s)".into())
                    })?;
                    pack_bits_lsb(shamt_bits)
                } else {
                    *packed_cols
                        .get(1)
                        .ok_or_else(|| PiCcsError::InvalidInput("event-table hash: missing rhs opening".into()))?
                };

                let eq_addr = eq_bits_prod(time_bits_open, &r_cycle[..packed_time_bits])?;
                let hash = K::ONE + event_alpha * lane.val + event_beta * lhs + event_gamma * rhs;
                let expected_final = lane.has_lookup * hash * eq_addr;
                if expected_final != final_value {
                    return Err(PiCcsError::ProtocolError(
                        "shout/event_table_hash terminal value mismatch".into(),
                    ));
                }
                shout_event_table_hash_claim_sum_total += claim_sum;
            }

            if is_packed {
                if value_claim != K::ZERO {
                    return Err(PiCcsError::ProtocolError("packed RV32 expects value claim == 0".into()));
                }
                if adapter_claim != K::ZERO {
                    return Err(PiCcsError::ProtocolError(
                        "packed RV32 expects adapter claim == 0".into(),
                    ));
                }
            } else {
                if value_claim != pre.addr_claim_sum {
                    return Err(PiCcsError::ProtocolError(
                        "shout value claimed sum != addr claimed sum".into(),
                    ));
                }

                if pre.is_active {
                    let expected_addr_final = pre.table_eval_at_r_addr * adapter_claim;
                    if expected_addr_final != pre.addr_final {
                        return Err(PiCcsError::ProtocolError("shout addr terminal value mismatch".into()));
                    }
                } else {
                    // If we skipped the addr-pre sumcheck, the only sound case is "no lookups".
                    // Enforce this by requiring the addr claim + adapter claim to be zero.
                    if pre.addr_claim_sum != K::ZERO {
                        return Err(PiCcsError::ProtocolError(
                            "shout addr-pre skipped but addr claim is nonzero".into(),
                        ));
                    }
                    if adapter_claim != K::ZERO {
                        return Err(PiCcsError::ProtocolError(
                            "shout addr-pre skipped but adapter claim is nonzero".into(),
                        ));
                    }
                    if pre.addr_final != K::ZERO {
                        return Err(PiCcsError::ProtocolError(
                            "shout addr-pre skipped but addr_final is nonzero".into(),
                        ));
                    }
                }
            }
        }

        shout_lane_base += expected_lanes;
    }
    if shout_lane_base != shout_pre.len() {
        return Err(PiCcsError::ProtocolError(
            "shout pre-time lanes not fully consumed".into(),
        ));
    }
    if shout_me_base != mem_proof.shout_me_claims_time.len() {
        return Err(PiCcsError::ProtocolError(
            "Shout ME(time) claims not fully consumed".into(),
        ));
    }

    // Trace linkage at r_time: bind Shout to the CPU trace.
    //
    // - Fixed-lane mode: sum lanes must match the trace's fixed-lane Shout view.
    // - Event-table mode: hash linkage (Jolt-ish): Σ_tables event_hash == trace_hash.
    if !step.lut_insts.is_empty() {
        let cpu = cpu_link.ok_or_else(|| {
            PiCcsError::ProtocolError("missing CPU trace linkage openings in no-shared-bus mode".into())
        })?;

        if any_event_table_shout {
            let trace_hash_idx = claim_plan
                .shout_event_trace_hash
                .ok_or_else(|| PiCcsError::ProtocolError("missing shout/event_trace_hash claim idx".into()))?;
            let trace_hash_claim_sum = batched_claimed_sums[trace_hash_idx];
            let trace_hash_final = batched_final_values[trace_hash_idx];

            if trace_hash_claim_sum != shout_event_table_hash_claim_sum_total {
                return Err(PiCcsError::ProtocolError(
                    "trace linkage failed: shout event-trace hash mismatch".into(),
                ));
            }

            // Terminal value check for the trace hash oracle (ShoutValueOracleSparse):
            // χ_{r_cycle}(r_time) · has_lookup(r_time) · (has_lookup + α·val + β·lhs + γ·rhs)(r_time).
            let hash_open = cpu.shout_has_lookup
                + event_alpha * cpu.shout_val
                + event_beta * cpu.shout_lhs
                + event_gamma * cpu.shout_rhs;
            let expected_final = chi_cycle_at_r_time * cpu.shout_has_lookup * hash_open;
            if expected_final != trace_hash_final {
                return Err(PiCcsError::ProtocolError(
                    "shout/event_trace_hash terminal value mismatch".into(),
                ));
            }
        } else {
            if shout_has_sum != cpu.shout_has_lookup {
                return Err(PiCcsError::ProtocolError(
                    "trace linkage failed: Shout has_lookup mismatch".into(),
                ));
            }
            if shout_val_sum != cpu.shout_val {
                return Err(PiCcsError::ProtocolError(
                    "trace linkage failed: Shout val mismatch".into(),
                ));
            }
            if shout_lhs_sum != cpu.shout_lhs {
                return Err(PiCcsError::ProtocolError(
                    "trace linkage failed: Shout lhs mismatch".into(),
                ));
            }
            if shout_rhs_sum != cpu.shout_rhs {
                return Err(PiCcsError::ProtocolError(
                    "trace linkage failed: Shout rhs mismatch".into(),
                ));
            }
            if shout_table_id_sum != cpu.shout_table_id {
                return Err(PiCcsError::ProtocolError(
                    "trace linkage failed: Shout table_id mismatch".into(),
                ));
            }
        }
    }

    let proof_offset = step.lut_insts.len();
    let mut twist_time_openings: Vec<TwistTimeLaneOpenings> = Vec::with_capacity(step.mem_insts.len());

    // Twist instances: time-lane terminal checks at r_time.
    for (i_mem, inst) in step.mem_insts.iter().enumerate() {
        let twist_proof = match &proofs_mem[proof_offset + i_mem] {
            MemOrLutProof::Twist(proof) => proof,
            _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
        };
        let layout = inst.twist_layout();
        let ell_addr = layout
            .lanes
            .get(0)
            .ok_or_else(|| PiCcsError::InvalidInput("TwistWitnessLayout has no lanes".into()))?
            .ell_addr;

        let expected_lanes = inst.lanes.max(1);

        // Local bus layout for this Twist instance (stored inside its own committed witness).
        let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
            m,
            step.mcs_inst.m_in,
            inst.steps,
            core::iter::empty::<(usize, usize)>(),
            core::iter::once((ell_addr, expected_lanes)),
        )
        .map_err(|e| PiCcsError::InvalidInput(format!("Twist(Route A): bus layout failed: {e}")))?;
        if bus.twist_cols.len() != 1 || !bus.shout_cols.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "Twist(Route A): expected a twist-only bus layout with 1 instance".into(),
            ));
        }

        let me_time = mem_proof
            .twist_me_claims_time
            .get(i_mem)
            .ok_or_else(|| PiCcsError::ProtocolError("missing Twist ME(time) claim".into()))?;
        if inst.comms.len() != 1 {
            return Err(PiCcsError::InvalidInput(format!(
                "no-shared-bus mode requires exactly 1 commitment per Twist instance (mem_idx={i_mem}, comms.len()={})",
                inst.comms.len()
            )));
        }
        if me_time.c != inst.comms[0] {
            return Err(PiCcsError::ProtocolError("Twist ME(time) commitment mismatch".into()));
        }

        let bus_y_base_time = me_time
            .y_scalars
            .len()
            .checked_sub(bus.bus_cols)
            .ok_or_else(|| PiCcsError::InvalidInput("Twist y_scalars too short for bus openings".into()))?;

        struct TwistLaneTimeOpen {
            ra_bits: Vec<K>,
            wa_bits: Vec<K>,
            has_read: K,
            has_write: K,
            wv: K,
            rv: K,
            inc: K,
        }

        let twist_inst_cols = bus
            .twist_cols
            .get(0)
            .ok_or_else(|| PiCcsError::ProtocolError("missing twist_cols[0]".into()))?;
        if twist_inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput("twist lane count mismatch".into()));
        }

        let mut lane_opens: Vec<TwistLaneTimeOpen> = Vec::with_capacity(expected_lanes);
        for (lane_idx, twist_cols) in twist_inst_cols.lanes.iter().enumerate() {
            if twist_cols.ra_bits.end - twist_cols.ra_bits.start != ell_addr
                || twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "twist bus layout mismatch at mem_idx={i_mem}, lane={lane_idx}: expected ell_addr={ell_addr}"
                )));
            }

            let mut ra_bits_open = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.ra_bits.clone() {
                ra_bits_open.push(
                    me_time
                        .y_scalars
                        .get(bus.y_scalar_index(bus_y_base_time, col_id))
                        .copied()
                        .ok_or_else(|| PiCcsError::ProtocolError("missing Twist ra_bits(time) opening".into()))?,
                );
            }
            let mut wa_bits_open = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.wa_bits.clone() {
                wa_bits_open.push(
                    me_time
                        .y_scalars
                        .get(bus.y_scalar_index(bus_y_base_time, col_id))
                        .copied()
                        .ok_or_else(|| PiCcsError::ProtocolError("missing Twist wa_bits(time) opening".into()))?,
                );
            }

            let has_read_open = me_time
                .y_scalars
                .get(bus.y_scalar_index(bus_y_base_time, twist_cols.has_read))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("missing Twist has_read(time) opening".into()))?;
            let has_write_open = me_time
                .y_scalars
                .get(bus.y_scalar_index(bus_y_base_time, twist_cols.has_write))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("missing Twist has_write(time) opening".into()))?;
            let wv_open = me_time
                .y_scalars
                .get(bus.y_scalar_index(bus_y_base_time, twist_cols.wv))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("missing Twist wv(time) opening".into()))?;
            let rv_open = me_time
                .y_scalars
                .get(bus.y_scalar_index(bus_y_base_time, twist_cols.rv))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("missing Twist rv(time) opening".into()))?;
            let inc_open = me_time
                .y_scalars
                .get(bus.y_scalar_index(bus_y_base_time, twist_cols.inc))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("missing Twist inc(time) opening".into()))?;

            lane_opens.push(TwistLaneTimeOpen {
                ra_bits: ra_bits_open,
                wa_bits: wa_bits_open,
                has_read: has_read_open,
                has_write: has_write_open,
                wv: wv_open,
                rv: rv_open,
                inc: inc_open,
            });
        }

        // Trace linkage at r_time: bind Twist(PROG/REG/RAM) to CPU trace columns.
        //
        // We key off `mem_id` (not instance ordering) so this remains robust if upstream reorders
        // instances, while still enforcing the RV32 trace path expects exactly these 3 memories.
        if step.mem_insts.len() != 3 {
            return Err(PiCcsError::InvalidInput(format!(
                "no-shared-bus trace linkage expects exactly 3 mem instances (PROG, REG, RAM), got {}",
                step.mem_insts.len()
            )));
        }
        {
            let mut ids = std::collections::BTreeSet::<u32>::new();
            for inst in step.mem_insts.iter() {
                ids.insert(inst.mem_id);
            }
            let required = std::collections::BTreeSet::from([PROG_ID.0, REG_ID.0, RAM_ID.0]);
            if ids != required {
                return Err(PiCcsError::InvalidInput(format!(
                    "no-shared-bus trace linkage expects mem_id set {{PROG_ID={}, REG_ID={}, RAM_ID={}}}, got {:?}",
                    PROG_ID.0, REG_ID.0, RAM_ID.0, ids
                )));
            }
        }
        let cpu = cpu_link.ok_or_else(|| {
            PiCcsError::ProtocolError("missing CPU trace linkage openings in no-shared-bus mode".into())
        })?;
        match inst.mem_id {
            id if id == PROG_ID.0 => {
                if expected_lanes != 1 {
                    return Err(PiCcsError::InvalidInput("PROG mem instance must have lanes=1".into()));
                }
                let lane = &lane_opens[0];
                if lane.has_read != cpu.active {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: PROG has_read != active".into(),
                    ));
                }
                if lane.has_write != K::ZERO {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: PROG has_write != 0".into(),
                    ));
                }
                if pack_bits_lsb(&lane.ra_bits) != cpu.prog_addr {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: PROG addr mismatch".into(),
                    ));
                }
                if lane.rv != cpu.prog_value {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: PROG value mismatch".into(),
                    ));
                }
                // Enforce padding discipline for write-side columns even though PROG is read-only.
                if lane.wv != K::ZERO || lane.inc != K::ZERO {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: PROG write-side cols must be 0".into(),
                    ));
                }
            }
            id if id == REG_ID.0 => {
                if expected_lanes != 2 || ell_addr != 5 {
                    return Err(PiCcsError::InvalidInput(
                        "REG mem instance must have lanes=2 and ell_addr=5".into(),
                    ));
                }
                // lane0: rs1 read + optional rd write
                let lane0 = &lane_opens[0];
                if lane0.has_read != cpu.active {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: REG lane0 has_read != active".into(),
                    ));
                }
                if pack_bits_lsb(&lane0.ra_bits) != cpu.rs1_addr {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: REG lane0 rs1 addr mismatch".into(),
                    ));
                }
                if lane0.rv != cpu.rs1_val {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: REG lane0 rs1 val mismatch".into(),
                    ));
                }
                if lane0.has_write != cpu.rd_has_write {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: REG lane0 has_write != rd_has_write".into(),
                    ));
                }
                if pack_bits_lsb(&lane0.wa_bits) != cpu.rd_addr {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: REG lane0 rd addr mismatch".into(),
                    ));
                }
                if lane0.wv != cpu.rd_val {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: REG lane0 rd val mismatch".into(),
                    ));
                }

                // lane1: rs2 read only
                let lane1 = &lane_opens[1];
                if lane1.has_read != cpu.active {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: REG lane1 has_read != active".into(),
                    ));
                }
                if pack_bits_lsb(&lane1.ra_bits) != cpu.rs2_addr {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: REG lane1 rs2 addr mismatch".into(),
                    ));
                }
                if lane1.rv != cpu.rs2_val {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: REG lane1 rs2 val mismatch".into(),
                    ));
                }
                if lane1.has_write != K::ZERO || lane1.wv != K::ZERO || lane1.inc != K::ZERO {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: REG lane1 must be read-only".into(),
                    ));
                }
            }
            id if id == RAM_ID.0 => {
                if expected_lanes != 1 {
                    return Err(PiCcsError::InvalidInput("RAM mem instance must have lanes=1".into()));
                }
                let lane = &lane_opens[0];

                if lane.has_read != cpu.ram_has_read {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: RAM has_read mismatch".into(),
                    ));
                }
                if lane.has_write != cpu.ram_has_write {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: RAM has_write mismatch".into(),
                    ));
                }
                if lane.rv != cpu.ram_rv {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: RAM rv mismatch".into(),
                    ));
                }
                if lane.wv != cpu.ram_wv {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: RAM wv mismatch".into(),
                    ));
                }

                // Address linkage is gated because the CPU trace has a single `ram_addr` column
                // that is non-zero on both read and write rows.
                let ra = pack_bits_lsb(&lane.ra_bits);
                let wa = pack_bits_lsb(&lane.wa_bits);
                if lane.has_read * (ra - cpu.ram_addr) != K::ZERO {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: RAM read addr mismatch".into(),
                    ));
                }
                if lane.has_write * (wa - cpu.ram_addr) != K::ZERO {
                    return Err(PiCcsError::ProtocolError(
                        "trace linkage failed: RAM write addr mismatch".into(),
                    ));
                }
            }
            other => {
                return Err(PiCcsError::InvalidInput(format!(
                    "unexpected mem_id={} in no-shared-bus RV32 trace linkage",
                    other
                )));
            }
        }

        let twist_claims = claim_plan
            .twist
            .get(i_mem)
            .ok_or_else(|| PiCcsError::ProtocolError("missing Twist claim schedule".into()))?;

        // Route A Twist ordering in batched_time:
        // - read_check (time rounds only)
        // - write_check (time rounds only)
        // - aggregated bitness for (ra_bits, wa_bits, has_read, has_write)
        let read_check_claim = batched_claimed_sums[twist_claims.read_check];
        let write_check_claim = batched_claimed_sums[twist_claims.write_check];
        let read_check_final = batched_final_values[twist_claims.read_check];
        let write_check_final = batched_final_values[twist_claims.write_check];

        let pre = twist_pre
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput("missing Twist pre-time data".into()))?;
        let r_addr = &pre.r_addr;

        if read_check_claim != pre.read_check_claim_sum {
            return Err(PiCcsError::ProtocolError(
                "twist read_check claimed sum != addr-pre final".into(),
            ));
        }
        if write_check_claim != pre.write_check_claim_sum {
            return Err(PiCcsError::ProtocolError(
                "twist write_check claimed sum != addr-pre final".into(),
            ));
        }

        // Aggregated bitness terminal check (ra_bits, wa_bits, has_read, has_write).
        {
            let mut opens: Vec<K> = Vec::with_capacity(expected_lanes * (2 * ell_addr + 2));
            for lane in lane_opens.iter() {
                opens.extend_from_slice(&lane.ra_bits);
                opens.extend_from_slice(&lane.wa_bits);
                opens.push(lane.has_read);
                opens.push(lane.has_write);
            }
            let weights = bitness_weights(r_cycle, opens.len(), 0x5457_4953_54u64 + i_mem as u64);
            let mut acc = K::ZERO;
            for (w, b) in weights.iter().zip(opens.iter()) {
                acc += *w * *b * (*b - K::ONE);
            }
            let expected = chi_cycle_at_r_time * acc;
            if expected != batched_final_values[twist_claims.bitness] {
                return Err(PiCcsError::ProtocolError(
                    "twist/bitness terminal value mismatch".into(),
                ));
            }
        }

        let val_eval = twist_proof
            .val_eval
            .as_ref()
            .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;

        let init_at_r_addr = eval_init_at_r_addr(&inst.init, inst.k, r_addr)?;
        let claimed_val = init_at_r_addr + val_eval.claimed_inc_sum_lt;

        // Terminal checks for read_check / write_check at (r_time, r_addr).
        let mut expected_read_check_final = K::ZERO;
        let mut expected_write_check_final = K::ZERO;
        for lane in lane_opens.iter() {
            let read_eq_addr = eq_bits_prod(&lane.ra_bits, r_addr)?;
            expected_read_check_final += chi_cycle_at_r_time * lane.has_read * (claimed_val - lane.rv) * read_eq_addr;

            let write_eq_addr = eq_bits_prod(&lane.wa_bits, r_addr)?;
            expected_write_check_final +=
                chi_cycle_at_r_time * lane.has_write * (lane.wv - claimed_val - lane.inc) * write_eq_addr;
        }
        if expected_read_check_final != read_check_final {
            return Err(PiCcsError::ProtocolError(
                "twist/read_check terminal value mismatch".into(),
            ));
        }
        if expected_write_check_final != write_check_final {
            return Err(PiCcsError::ProtocolError(
                "twist/write_check terminal value mismatch".into(),
            ));
        }

        twist_time_openings.push(TwistTimeLaneOpenings {
            lanes: lane_opens
                .into_iter()
                .map(|lane| TwistTimeLaneOpeningsLane {
                    wa_bits: lane.wa_bits,
                    has_write: lane.has_write,
                    inc_at_write_addr: lane.inc,
                })
                .collect(),
        });
    }

    verify_no_shared_bus_twist_val_eval_phase(
        tr, m, step, prev_step, proofs_mem, mem_proof, twist_pre, step_idx, r_time,
    )?;

    Ok(RouteAMemoryVerifyOutput {
        claim_idx_end: claim_plan.claim_idx_end,
        twist_time_openings,
    })
}
