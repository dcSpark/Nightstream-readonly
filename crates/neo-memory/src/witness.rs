use neo_ccs::matrix::Mat;
use neo_ccs::relations::{McsInstance, McsWitness};
use neo_math::K;
use neo_reductions::error::PiCcsError;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::ops::Range;

use crate::mem_init::MemInit;
use crate::riscv::lookups::RiscvOpcode;

fn default_one_usize() -> usize {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LutTableSpec {
    /// A "virtual" lookup table defined by the RISC-V opcode semantics over
    /// an interleaved operand address space.
    ///
    /// Addressing convention:
    /// - `n_side = 2`, `ell = 1`
    /// - `d = 2*xlen` (one bit per dimension)
    /// - Address bits are little-endian and correspond to `interleave_bits(rs1, rs2)`.
    RiscvOpcode { opcode: RiscvOpcode, xlen: usize },

    /// A packed-key (non-bit-addressed) variant of `RiscvOpcode`, intended for "no width bloat"
    /// Shout/ALU proofs that do **not** commit to `ell_addr=64` addr-bit columns.
    ///
    /// Current implementation status:
    /// - Supported: `xlen = 32` for selected RV32 ops, including:
    ///   - bitwise: `And | Andn | Or | Xor`
    ///   - arithmetic: `Add | Sub`
    ///   - compares: `Eq | Neq | Slt | Sltu`
    ///   - shifts: `Sll | Srl | Sra`
    ///   - RV32M: `Mul | Mulh | Mulhu | Mulhsu | Div | Divu | Rem | Remu`
    /// - Witness convention: the Shout lane's `addr_bits` slice is repurposed as packed columns.
    ///   The exact layout depends on `opcode`; the suffix columns are always `[has_lookup, val_u32]`.
    ///   Examples:
    ///   - `Add/Sub` (d=3): `[lhs_u32, rhs_u32, aux_bit]` (carry for `Add`, borrow for `Sub`)
    ///   - `Eq/Neq` (d=35): `[lhs_u32, rhs_u32, borrow_bit, diff_bits[0..32]]` where `val_u32` is the out bit
    ///   - `Mul` (d=34): `[lhs_u32, rhs_u32, hi_bits[0..32]]` where `val_u32` is the low 32 bits
    ///   - `Mulhu` (d=34): `[lhs_u32, rhs_u32, lo_bits[0..32]]` where `val_u32` is the high 32 bits
    ///   - `Sltu` (d=35): `[lhs_u32, rhs_u32, diff_u32, diff_bits[0..32]]` where `val_u32` is the out bit
    ///   - `Sll/Srl/Sra` (d=38): `[lhs_u32, shamt_bits[0..5], ...]`
    ///
    /// For packed-key instances, Route-A enforces correctness directly via time-domain constraints
    /// (claimed sum forced to 0); table MLE evaluation is not used.
    RiscvOpcodePacked { opcode: RiscvOpcode, xlen: usize },

    /// An "event table" packed-key variant of `RiscvOpcodePacked` for RV32.
    ///
    /// Instead of storing one Shout lane over time (one row per cycle), the witness stores only
    /// the executed lookup events (one row per lookup). Each event row carries:
    /// - a prefix of `time_bits` boolean columns encoding the original Route-A time index `t`
    ///   (little-endian), and
    /// - the same packed columns as `RiscvOpcodePacked` for `opcode`.
    ///
    /// The Route-A protocol then links the event table back to the CPU trace via a "scatter"
    /// check at a random time point `r_cycle` (Jolt-ish): roughly,
    ///   Σ_events hash(event)·χ_{r_cycle}(t_event) == Σ_t hash(trace[t])·χ_{r_cycle}(t).
    ///
    /// Notes:
    /// - This is RV32-only (`xlen = 32`), `n_side = 2`, `ell = 1`.
    /// - `time_bits` must match the Route-A `ell_n` used for the time domain.
    RiscvOpcodeEventTablePacked {
        opcode: RiscvOpcode,
        xlen: usize,
        time_bits: usize,
    },

    /// Implicit identity table over 32-bit addresses: `table[addr] = addr`.
    ///
    /// Addressing convention:
    /// - `n_side = 2`, `ell = 1`
    /// - `d = 32` (one bit per dimension)
    /// - Address bits are little-endian
    IdentityU32,
}

impl LutTableSpec {
    pub fn eval_table_mle(&self, r_addr: &[K]) -> Result<K, PiCcsError> {
        match self {
            LutTableSpec::RiscvOpcode { opcode, xlen } => {
                Ok(crate::riscv::lookups::evaluate_opcode_mle(*opcode, r_addr, *xlen))
            }
            LutTableSpec::RiscvOpcodePacked { .. } => Err(PiCcsError::InvalidInput(
                "RiscvOpcodePacked does not support eval_table_mle (not bit-addressed)".into(),
            )),
            LutTableSpec::RiscvOpcodeEventTablePacked { .. } => Err(PiCcsError::InvalidInput(
                "RiscvOpcodeEventTablePacked does not support eval_table_mle (not bit-addressed)".into(),
            )),
            LutTableSpec::IdentityU32 => {
                if r_addr.len() != 32 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "IdentityU32: expected r_addr.len()=32, got {}",
                        r_addr.len()
                    )));
                }
                Ok(crate::identity::eval_identity_mle_le(r_addr))
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemInstance<C, F> {
    /// Logical memory instance identifier (e.g. RISC-V `PROG_ID/REG_ID/RAM_ID`).
    ///
    /// This is used by higher-level protocols to link Twist instances to CPU trace columns
    /// without relying on a fixed instance ordering.
    pub mem_id: u32,
    pub comms: Vec<C>,
    pub k: usize,
    pub d: usize,
    pub n_side: usize,
    pub steps: usize,
    /// Number of access lanes per VM step for this Twist instance.
    ///
    /// Each lane carries the canonical Twist bus slice:
    /// `[ra_bits, wa_bits, has_read, has_write, wv, rv, inc]`.
    #[serde(default = "default_one_usize")]
    pub lanes: usize,
    /// Bits per address dimension: ell = ceil(log2(n_side))
    /// With index-bit addressing, we commit d*ell bit-columns instead of d*n_side one-hot columns.
    pub ell: usize,
    /// Public initial memory state for cells [0..k).
    pub init: MemInit<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemWitness<F> {
    pub mats: Vec<Mat<F>>,
}

#[derive(Clone, Debug)]
pub struct TwistWitnessLaneLayout {
    pub ell_addr: usize,
    pub ra_bits: Range<usize>,
    pub wa_bits: Range<usize>,
    pub has_read: usize,
    pub has_write: usize,
    pub wv: usize,
    pub rv: usize,
    pub inc_at_write_addr: usize,
}

#[derive(Clone, Debug)]
pub struct TwistWitnessLayout {
    pub lane_len: usize,
    pub lanes: Vec<TwistWitnessLaneLayout>,
}

impl TwistWitnessLayout {
    pub fn expected_len(&self) -> usize {
        self.lane_len
            .checked_mul(self.lanes.len())
            .expect("TwistWitnessLayout: lane_len*lanes overflow")
    }
}

impl<C, F> MemInstance<C, F> {
    pub fn twist_layout(&self) -> TwistWitnessLayout {
        let ell_addr = self.d * self.ell;
        let lane_len = 2 * ell_addr + 5;
        let lanes = self.lanes.max(1);

        let mut out = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let base = lane * lane_len;
            out.push(TwistWitnessLaneLayout {
                ell_addr,
                ra_bits: base..(base + ell_addr),
                wa_bits: (base + ell_addr)..(base + 2 * ell_addr),
                has_read: base + 2 * ell_addr,
                has_write: base + 2 * ell_addr + 1,
                wv: base + 2 * ell_addr + 2,
                rv: base + 2 * ell_addr + 3,
                inc_at_write_addr: base + 2 * ell_addr + 4,
            });
        }
        TwistWitnessLayout { lane_len, lanes: out }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LutInstance<C, F> {
    pub comms: Vec<C>,
    pub k: usize,
    pub d: usize,
    pub n_side: usize,
    pub steps: usize,
    /// Number of lookup lanes per VM step for this Shout instance.
    ///
    /// Each lane carries the canonical Shout bus slice:
    /// `[addr_bits, has_lookup, val]`.
    #[serde(default = "default_one_usize")]
    pub lanes: usize,
    /// Bits per address dimension: ell = ceil(log2(n_side))
    pub ell: usize,
    /// Optional virtual table descriptor (when the full table is not materialized).
    #[serde(default)]
    pub table_spec: Option<LutTableSpec>,
    pub table: Vec<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LutWitness<F> {
    pub mats: Vec<Mat<F>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecodeInstance<C, F> {
    /// Deterministic decode sidecar id.
    ///
    /// Track A currently uses one decode sidecar per RV32 trace step.
    pub decode_id: u32,
    /// Commitment(s) for the decode sidecar witness matrix/matrices.
    pub comms: Vec<C>,
    /// Number of rows (cycles) in the sidecar witness domain.
    pub steps: usize,
    /// Number of committed decode columns per row.
    pub cols: usize,
    #[serde(skip)]
    pub _phantom: PhantomData<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecodeWitness<F> {
    pub mats: Vec<Mat<F>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WidthInstance<C, F> {
    /// Deterministic width sidecar id.
    ///
    /// Track A W3 uses one width sidecar per RV32 trace step.
    pub width_id: u32,
    /// Commitment(s) for the width sidecar witness matrix/matrices.
    pub comms: Vec<C>,
    /// Number of rows (cycles) in the sidecar witness domain.
    pub steps: usize,
    /// Number of committed width-helper columns per row.
    pub cols: usize,
    #[serde(skip)]
    pub _phantom: PhantomData<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WidthWitness<F> {
    pub mats: Vec<Mat<F>>,
}

#[derive(Clone, Debug)]
pub struct ShoutWitnessLayout {
    pub ell_addr: usize,
    pub addr_bits: Range<usize>,
    pub has_lookup: usize,
    pub val: usize,
}

impl ShoutWitnessLayout {
    pub fn expected_len(&self) -> usize {
        self.ell_addr + 2
    }
}

impl<C, F> LutInstance<C, F> {
    pub fn shout_layout(&self) -> ShoutWitnessLayout {
        let ell_addr = self.d * self.ell;
        ShoutWitnessLayout {
            ell_addr,
            addr_bits: 0..ell_addr,
            has_lookup: ell_addr,
            val: ell_addr + 1,
        }
    }
}

/// Per-step bundle that carries CPU + memory witnesses for a single folding step.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepWitnessBundle<Cmt, F, K> {
    pub mcs: (McsInstance<Cmt, F>, McsWitness<F>),
    pub lut_instances: Vec<(LutInstance<Cmt, F>, LutWitness<F>)>,
    pub mem_instances: Vec<(MemInstance<Cmt, F>, MemWitness<F>)>,
    pub decode_instances: Vec<(DecodeInstance<Cmt, F>, DecodeWitness<F>)>,
    pub width_instances: Vec<(WidthInstance<Cmt, F>, WidthWitness<F>)>,
    #[serde(skip)]
    pub _phantom: PhantomData<K>,
}

impl<Cmt, F, K> From<(McsInstance<Cmt, F>, McsWitness<F>)> for StepWitnessBundle<Cmt, F, K> {
    fn from(mcs: (McsInstance<Cmt, F>, McsWitness<F>)) -> Self {
        Self {
            mcs,
            lut_instances: Vec::new(),
            mem_instances: Vec::new(),
            decode_instances: Vec::new(),
            width_instances: Vec::new(),
            _phantom: PhantomData,
        }
    }
}

/// Per-step bundle that carries *only public instances* (no witnesses).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepInstanceBundle<Cmt, F, K> {
    pub mcs_inst: McsInstance<Cmt, F>,
    pub lut_insts: Vec<LutInstance<Cmt, F>>,
    pub mem_insts: Vec<MemInstance<Cmt, F>>,
    pub decode_insts: Vec<DecodeInstance<Cmt, F>>,
    pub width_insts: Vec<WidthInstance<Cmt, F>>,
    #[serde(skip)]
    pub _phantom: PhantomData<K>,
}

impl<Cmt, F, K> From<McsInstance<Cmt, F>> for StepInstanceBundle<Cmt, F, K> {
    fn from(mcs_inst: McsInstance<Cmt, F>) -> Self {
        Self {
            mcs_inst,
            lut_insts: Vec::new(),
            mem_insts: Vec::new(),
            decode_insts: Vec::new(),
            width_insts: Vec::new(),
            _phantom: PhantomData,
        }
    }
}

impl<Cmt: Clone, F: Clone, K> From<&StepWitnessBundle<Cmt, F, K>> for StepInstanceBundle<Cmt, F, K> {
    fn from(step: &StepWitnessBundle<Cmt, F, K>) -> Self {
        Self {
            mcs_inst: step.mcs.0.clone(),
            lut_insts: step
                .lut_instances
                .iter()
                .map(|(inst, _)| inst.clone())
                .collect(),
            mem_insts: step
                .mem_instances
                .iter()
                .map(|(inst, _)| inst.clone())
                .collect(),
            decode_insts: step
                .decode_instances
                .iter()
                .map(|(inst, _)| inst.clone())
                .collect(),
            width_insts: step
                .width_instances
                .iter()
                .map(|(inst, _)| inst.clone())
                .collect(),
            _phantom: PhantomData,
        }
    }
}

impl<Cmt, F, K> From<StepWitnessBundle<Cmt, F, K>> for StepInstanceBundle<Cmt, F, K> {
    fn from(step: StepWitnessBundle<Cmt, F, K>) -> Self {
        let StepWitnessBundle {
            mcs: (mcs_inst, _mcs_wit),
            lut_instances,
            mem_instances,
            decode_instances,
            width_instances,
            _phantom: _,
        } = step;
        Self {
            mcs_inst,
            lut_insts: lut_instances.into_iter().map(|(inst, _)| inst).collect(),
            mem_insts: mem_instances.into_iter().map(|(inst, _)| inst).collect(),
            decode_insts: decode_instances.into_iter().map(|(inst, _)| inst).collect(),
            width_insts: width_instances.into_iter().map(|(inst, _)| inst).collect(),
            _phantom: PhantomData,
        }
    }
}
