//! Convenience wrappers for verifying RISC-V shard proofs safely.
//!
//! These helpers are intentionally small: they standardize the step-linking configuration
//! for RV32 B1 chunked execution so callers don't accidentally verify a "bag of chunks".

#![allow(non_snake_case)]

use std::collections::HashMap;
use std::collections::HashSet;
use std::time::Duration;

use crate::output_binding::{simple_output_config, OutputBindingConfig};
use crate::pi_ccs::FoldingMode;
use crate::session::FoldingSession;
use crate::shard::{CommitMixers, ShardFoldOutputs, ShardProof, StepLinkingConfig};
use crate::PiCcsError;
use neo_ajtai::{AjtaiSModule, Commitment as Cmt};
use neo_ccs::{CcsStructure, Mat, MeInstance};
use neo_math::{F, K};
use neo_memory::mem_init_from_initial_mem;
use neo_memory::output_check::ProgramIO;
use neo_memory::plain::LutTable;
use neo_memory::plain::PlainMemLayout;
use neo_memory::riscv::ccs::{
    build_rv32_b1_decode_plumbing_sidecar_ccs, build_rv32_b1_rv32m_event_sidecar_ccs,
    build_rv32_b1_semantics_sidecar_ccs, build_rv32_b1_step_ccs, estimate_rv32_b1_all_ccs_counts,
    rv32_b1_chunk_to_witness, rv32_b1_shared_cpu_bus_config, rv32_b1_step_linking_pairs, Rv32B1Layout,
};
use neo_memory::riscv::lookups::{
    decode_program, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode, RiscvShoutTables, PROG_ID, RAM_ID, REG_ID,
};
use neo_memory::riscv::shard::{extract_boundary_state, Rv32BoundaryState};
use neo_memory::witness::LutTableSpec;
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_memory::R1csCpu;
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use neo_vm_trace::Twist as _;
use p3_field::PrimeCharacteristicRing;
use p3_field::PrimeField64;

#[cfg(target_arch = "wasm32")]
use js_sys::Date;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
type TimePoint = f64;
#[cfg(not(target_arch = "wasm32"))]
type TimePoint = Instant;

#[inline]
fn time_now() -> TimePoint {
    #[cfg(target_arch = "wasm32")]
    {
        Date::now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        Instant::now()
    }
}

#[inline]
fn elapsed_duration(start: TimePoint) -> Duration {
    #[cfg(target_arch = "wasm32")]
    {
        let elapsed_ms = Date::now() - start;
        Duration::from_secs_f64(elapsed_ms / 1_000.0)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        start.elapsed()
    }
}

pub fn rv32_b1_step_linking_config(layout: &Rv32B1Layout) -> StepLinkingConfig {
    StepLinkingConfig::new(rv32_b1_step_linking_pairs(layout))
}

/// Enforce that the *public statement* initial memory matches chunk 0's `MemInstance.init`.
///
/// This lets later chunk `init` snapshots remain proof-internal rollover data (Twist needs them),
/// while keeping the user-facing statement independent of `chunk_size`.
pub fn rv32_b1_enforce_chunk0_mem_init_matches_statement<Cmt2, K2>(
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    statement_initial_mem: &HashMap<(u32, u64), F>,
    steps: &[StepInstanceBundle<Cmt2, F, K2>],
) -> Result<(), PiCcsError> {
    let chunk0 = steps
        .first()
        .ok_or_else(|| PiCcsError::InvalidInput("no steps provided".into()))?;

    let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();

    if chunk0.mem_insts.len() != mem_ids.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "mem instance count mismatch: chunk0 has {}, but mem_layouts has {}",
            chunk0.mem_insts.len(),
            mem_ids.len()
        )));
    }

    for (idx, mem_id) in mem_ids.into_iter().enumerate() {
        let layout = mem_layouts
            .get(&mem_id)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing PlainMemLayout for mem_id={mem_id}")))?;
        let expected = mem_init_from_initial_mem(mem_id, layout.k, statement_initial_mem)?;
        let got = &chunk0.mem_insts[idx].init;
        if *got != expected {
            return Err(PiCcsError::InvalidInput(format!(
                "chunk0 MemInstance.init mismatch for mem_id={mem_id}"
            )));
        }
    }

    Ok(())
}

pub fn fold_shard_verify_rv32_b1<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    layout: &Rv32B1Layout,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let _ = (mode, tr, params, s_me, steps, acc_init, proof, mixers, layout);
    Err(PiCcsError::InvalidInput(
        "fold_shard_verify_rv32_b1 is not sound for RV32 B1 in this branch: step CCS is glue-only and semantics are proven in sidecars. Use Rv32B1::prove() and Rv32B1Run::verify()/verify_proof_bundle() instead."
            .into(),
    ))
}

pub fn fold_shard_verify_rv32_b1_with_statement_mem_init<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    statement_initial_mem: &HashMap<(u32, u64), F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    layout: &Rv32B1Layout,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let _ = (
        mode,
        tr,
        params,
        s_me,
        mem_layouts,
        statement_initial_mem,
        steps,
        acc_init,
        proof,
        mixers,
        layout,
    );
    Err(PiCcsError::InvalidInput(
        "fold_shard_verify_rv32_b1_with_statement_mem_init is not sound for RV32 B1 in this branch: step CCS is glue-only and semantics are proven in sidecars. Use Rv32B1::prove() and Rv32B1Run::verify()/verify_proof_bundle() instead."
            .into(),
    ))
}

pub fn fold_shard_verify_rv32_b1_with_output_binding<MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    steps: &[StepInstanceBundle<Cmt, F, K>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    mixers: CommitMixers<MR, MB>,
    ob_cfg: &crate::output_binding::OutputBindingConfig,
    layout: &Rv32B1Layout,
) -> Result<ShardFoldOutputs<Cmt, F, K>, PiCcsError>
where
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    let _ = (mode, tr, params, s_me, steps, acc_init, proof, mixers, ob_cfg, layout);
    Err(PiCcsError::InvalidInput(
        "fold_shard_verify_rv32_b1_with_output_binding is not sound for RV32 B1 in this branch: step CCS is glue-only and semantics are proven in sidecars. Use Rv32B1::prove() and Rv32B1Run::verify()/verify_output_claim*() instead."
            .into(),
    ))
}

fn pow2_ceil_k(min_k: usize) -> (usize, usize) {
    // RV32 B1 alignment constraints require bit-addressed memories with d>=2.
    let k = min_k.next_power_of_two().max(4);
    let d = k.trailing_zeros() as usize;
    (k, d)
}

fn infer_required_shout_opcodes(program: &[RiscvInstruction]) -> HashSet<RiscvOpcode> {
    let mut ops = HashSet::new();

    // The ADD table is required because the step circuit uses it for address/PC wiring in multiple
    // instructions (LW/SW/AUIPC/JALR), even if the program has no explicit ADD/ADDI.
    ops.insert(RiscvOpcode::Add);

    for instr in program {
        match instr {
            RiscvInstruction::RAlu { op, .. } => {
                match op {
                    // RV32 B1 proves RV32M MUL* via the RV32M sidecar CCS (no Shout table required).
                    RiscvOpcode::Mul | RiscvOpcode::Mulh | RiscvOpcode::Mulhu | RiscvOpcode::Mulhsu => {}
                    // RV32 B1 proves RV32M DIV*/REM* via the RV32M sidecar CCS, but it requires a SLTU lookup to prove
                    // the remainder bound when divisor != 0 (unsigned and signed).
                    RiscvOpcode::Div | RiscvOpcode::Divu | RiscvOpcode::Rem | RiscvOpcode::Remu => {
                        ops.insert(RiscvOpcode::Sltu);
                    }
                    _ => {
                        ops.insert(*op);
                    }
                }
            }
            RiscvInstruction::IAlu { op, .. } => {
                ops.insert(*op);
            }
            RiscvInstruction::Branch { cond, .. } => {
                ops.insert(cond.to_shout_opcode());
            }
            RiscvInstruction::Load { .. } => {
                ops.insert(RiscvOpcode::Add);
            }
            RiscvInstruction::Store { .. } => {
                ops.insert(RiscvOpcode::Add);
            }
            RiscvInstruction::Jalr { .. } => {
                ops.insert(RiscvOpcode::Add);
            }
            RiscvInstruction::Auipc { .. } => {
                ops.insert(RiscvOpcode::Add);
            }
            RiscvInstruction::Amo { op, .. } => match op {
                neo_memory::riscv::lookups::RiscvMemOp::AmoaddW | neo_memory::riscv::lookups::RiscvMemOp::AmoaddD => {
                    ops.insert(RiscvOpcode::Add);
                }
                neo_memory::riscv::lookups::RiscvMemOp::AmoxorW | neo_memory::riscv::lookups::RiscvMemOp::AmoxorD => {
                    ops.insert(RiscvOpcode::Xor);
                }
                neo_memory::riscv::lookups::RiscvMemOp::AmoandW | neo_memory::riscv::lookups::RiscvMemOp::AmoandD => {
                    ops.insert(RiscvOpcode::And);
                }
                neo_memory::riscv::lookups::RiscvMemOp::AmoorW | neo_memory::riscv::lookups::RiscvMemOp::AmoorD => {
                    ops.insert(RiscvOpcode::Or);
                }
                _ => {}
            },
            _ => {}
        }
    }

    ops
}

fn all_shout_opcodes() -> HashSet<RiscvOpcode> {
    use RiscvOpcode::*;
    // RV32 B1 uses implicit Shout tables only for opcodes with a closed-form MLE implementation.
    // RV32M ops are proven via a dedicated sidecar CCS argument (not via Shout tables).
    HashSet::from([And, Xor, Or, Sub, Add, Sltu, Slt, Eq, Neq, Sll, Srl, Sra])
}

/// High-level “few lines” builder for proving/verifying an RV32 program using the B1 shared-bus step circuit.
///
/// This:
/// - chooses parameters + Ajtai committer automatically,
/// - infers the minimal Shout table set from the program (unless overridden),
/// - enforces RV32 B1 step linking, and
/// - (optionally) proves output binding against a selected Twist instance (default: RAM).
#[derive(Clone, Copy, Debug, Default)]
enum OutputTarget {
    #[default]
    Ram,
    Reg,
}

#[derive(Clone, Debug)]
pub struct Rv32B1 {
    program_base: u64,
    program_bytes: Vec<u8>,
    xlen: usize,
    ram_bytes: usize,
    chunk_size: usize,
    chunk_size_auto: bool,
    max_steps: Option<usize>,
    trace_min_len: usize,
    trace_chunk_rows: Option<usize>,
    mode: FoldingMode,
    shout_auto_minimal: bool,
    shout_ops: Option<HashSet<RiscvOpcode>>,
    output_claims: ProgramIO<F>,
    output_target: OutputTarget,
    ram_init: HashMap<u64, u64>,
    reg_init: HashMap<u64, u64>,
}

/// Default instruction cap for RV32B1 runs when `max_steps` is not specified.
///
/// The runner stops early if the guest halts (e.g. via `ecall`), so this is only a safety bound
/// against non-halting guests.
const DEFAULT_RV32B1_MAX_STEPS: usize = 1 << 20;

fn program_uses_rv32m(program: &[RiscvInstruction]) -> bool {
    program.iter().any(|instr| match instr {
        RiscvInstruction::RAlu { op, .. } => matches!(
            op,
            RiscvOpcode::Mul
                | RiscvOpcode::Mulh
                | RiscvOpcode::Mulhu
                | RiscvOpcode::Mulhsu
                | RiscvOpcode::Div
                | RiscvOpcode::Divu
                | RiscvOpcode::Rem
                | RiscvOpcode::Remu
        ),
        _ => false,
    })
}

impl Rv32B1 {
    /// Create a runner from ROM bytes (must be a valid RV32 program encoding).
    pub fn from_rom(program_base: u64, program_bytes: &[u8]) -> Self {
        Self {
            program_base,
            program_bytes: program_bytes.to_vec(),
            xlen: 32,
            ram_bytes: 0x200,
            chunk_size: 1,
            chunk_size_auto: false,
            max_steps: None,
            trace_min_len: 4,
            trace_chunk_rows: None,
            mode: FoldingMode::Optimized,
            shout_auto_minimal: true,
            shout_ops: None,
            output_claims: ProgramIO::new(),
            output_target: OutputTarget::Ram,
            ram_init: HashMap::new(),
            reg_init: HashMap::new(),
        }
    }

    pub fn xlen(mut self, xlen: usize) -> Self {
        self.xlen = xlen;
        self
    }

    pub fn ram_bytes(mut self, ram_bytes: usize) -> Self {
        self.ram_bytes = ram_bytes;
        self
    }

    /// Initialize a register `reg` (x0..x31) to a u32 value.
    ///
    /// This is applied as part of the *public statement* initial memory for the REG Twist instance.
    pub fn reg_init_u32(mut self, reg: u64, value: u32) -> Self {
        self.reg_init.insert(reg, value as u64);
        self
    }

    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self.chunk_size_auto = false;
        self
    }

    /// Automatically pick a `chunk_size` based on an estimated trace length.
    ///
    /// Note: if `max_steps` is not set, the estimate defaults to the decoded program length.
    pub fn chunk_size_auto(mut self) -> Self {
        self.chunk_size_auto = true;
        self
    }

    /// Limit the number of instructions executed from the decoded program.
    ///
    /// This is primarily for tests/benchmarks that want a tiny trace, or for non-halting guests
    /// where you want to prove only a prefix of execution.
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    /// Lower-bound for trace-wiring execution-table length.
    ///
    /// Final `t` is `max(trace_len, trace_min_len)`.
    pub fn trace_min_len(mut self, min_trace_len: usize) -> Self {
        self.trace_min_len = min_trace_len.max(1);
        self
    }

    /// Fixed rows per trace step when using `prove_trace_wiring()`.
    pub fn trace_chunk_rows(mut self, chunk_rows: usize) -> Self {
        self.trace_chunk_rows = Some(chunk_rows);
        self
    }

    pub fn mode(mut self, mode: FoldingMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn shout_auto_minimal(mut self) -> Self {
        self.shout_ops = None;
        self.shout_auto_minimal = true;
        self
    }

    pub fn shout_all(mut self) -> Self {
        self.shout_ops = None;
        self.shout_auto_minimal = false;
        self
    }

    pub fn shout_ops(mut self, ops: impl IntoIterator<Item = RiscvOpcode>) -> Self {
        self.shout_ops = Some(ops.into_iter().collect());
        self.shout_auto_minimal = false;
        self
    }

    pub fn output(mut self, output_addr: u64, expected_output: F) -> Self {
        self.output_claims = ProgramIO::new().with_output(output_addr, expected_output);
        self.output_target = OutputTarget::Ram;
        self
    }

    pub fn output_claim(mut self, addr: u64, value: F) -> Self {
        if !matches!(self.output_target, OutputTarget::Ram) {
            self.output_target = OutputTarget::Ram;
            self.output_claims = ProgramIO::new();
        }
        self.output_claims = self.output_claims.with_output(addr, value);
        self
    }

    pub fn reg_output(mut self, reg: u64, expected: F) -> Self {
        self.output_claims = ProgramIO::new().with_output(reg, expected);
        self.output_target = OutputTarget::Reg;
        self
    }

    pub fn reg_output_claim(mut self, reg: u64, expected: F) -> Self {
        if !matches!(self.output_target, OutputTarget::Reg) {
            self.output_target = OutputTarget::Reg;
            self.output_claims = ProgramIO::new();
        }
        self.output_claims = self.output_claims.with_output(reg, expected);
        self
    }

    pub fn ram_init_u32(mut self, addr: u64, value: u32) -> Self {
        self.ram_init.insert(addr, value as u64);
        self
    }

    /// Prove/verify only the Tier 2.1 trace-wiring CCS (time-in-rows).
    ///
    /// `chunk_size`, `chunk_size_auto`, `ram_bytes`, and Shout-table selection knobs are ignored
    /// by this mode; use `trace_chunk_rows` to control trace-step sizing.
    pub fn prove_trace_wiring(self) -> Result<crate::riscv_trace_shard::Rv32TraceWiringRun, PiCcsError> {
        let mut runner = crate::riscv_trace_shard::Rv32TraceWiring::from_rom(self.program_base, &self.program_bytes)
            .xlen(self.xlen)
            .mode(self.mode)
            .min_trace_len(self.trace_min_len);
        if let Some(chunk_rows) = self.trace_chunk_rows {
            runner = runner.chunk_rows(chunk_rows);
        }
        match self.output_target {
            OutputTarget::Ram => {
                for (addr, value) in self.output_claims.claims() {
                    runner = runner.output_claim(addr, value);
                }
            }
            OutputTarget::Reg => {
                for (reg, value) in self.output_claims.claims() {
                    runner = runner.reg_output_claim(reg, value);
                }
            }
        }
        if let Some(max_steps) = self.max_steps {
            runner = runner.max_steps(max_steps);
        }
        for (addr, value) in self.ram_init {
            let value_u32 = u32::try_from(value).map_err(|_| {
                PiCcsError::InvalidInput(format!(
                    "ram_init_u32: value out of u32 range at addr={addr}: value={value}"
                ))
            })?;
            runner = runner.ram_init_u32(addr, value_u32);
        }
        for (reg, value) in self.reg_init {
            let value_u32 = u32::try_from(value).map_err(|_| {
                PiCcsError::InvalidInput(format!(
                    "reg_init_u32: value out of u32 range at reg={reg}: value={value}"
                ))
            })?;
            runner = runner.reg_init_u32(reg, value_u32);
        }
        runner.prove()
    }

    pub fn prove(self) -> Result<Rv32B1Run, PiCcsError> {
        if self.xlen != 32 {
            return Err(PiCcsError::InvalidInput(format!(
                "RV32 B1 MVP requires xlen == 32 (got {})",
                self.xlen
            )));
        }
        if self.program_bytes.is_empty() {
            return Err(PiCcsError::InvalidInput("program_bytes must be non-empty".into()));
        }
        if !self.chunk_size_auto && self.chunk_size == 0 {
            return Err(PiCcsError::InvalidInput("chunk_size must be non-zero".into()));
        }
        if self.ram_bytes == 0 {
            return Err(PiCcsError::InvalidInput("ram_bytes must be non-zero".into()));
        }
        if self.program_base != 0 {
            return Err(PiCcsError::InvalidInput(
                "RV32 B1 MVP requires program_base == 0 (addresses are indices into PROG/RAM layouts)".into(),
            ));
        }
        if self.program_bytes.len() % 4 != 0 {
            return Err(PiCcsError::InvalidInput(
                "program_bytes must be 4-byte aligned (RV32 B1 runner does not support RVC)".into(),
            ));
        }
        for (i, chunk) in self.program_bytes.chunks_exact(4).enumerate() {
            let first_half = u16::from_le_bytes([chunk[0], chunk[1]]);
            if (first_half & 0b11) != 0b11 {
                return Err(PiCcsError::InvalidInput(format!(
                    "RV32 B1 runner does not support compressed instructions (RVC): found compressed encoding at word index {i}"
                )));
            }
        }

        let program = decode_program(&self.program_bytes)
            .map_err(|e| PiCcsError::InvalidInput(format!("decode_program failed: {e}")))?;
        let uses_rv32m = program_uses_rv32m(&program);
        let using_default_max_steps = self.max_steps.is_none();
        let estimated_steps = match self.max_steps {
            Some(n) => {
                if n == 0 {
                    return Err(PiCcsError::InvalidInput("max_steps must be non-zero".into()));
                }
                n
            }
            None => program.len().max(1),
        };
        let max_steps = match self.max_steps {
            Some(n) => {
                if n == 0 {
                    return Err(PiCcsError::InvalidInput("max_steps must be non-zero".into()));
                }
                n
            }
            None => DEFAULT_RV32B1_MAX_STEPS.max(program.len()),
        };
        let mut twist = neo_memory::riscv::lookups::RiscvMemory::with_program_in_twist(
            self.xlen,
            PROG_ID,
            /*base_addr=*/ 0,
            &self.program_bytes,
        );
        let shout = RiscvShoutTables::new(self.xlen);

        let (prog_layout, initial_mem) = neo_memory::riscv::rom_init::prog_rom_layout_and_init_words(
            PROG_ID,
            /*base_addr=*/ 0,
            &self.program_bytes,
        )
        .map_err(|e| PiCcsError::InvalidInput(format!("prog_rom_layout_and_init_words failed: {e}")))?;
        let mut initial_mem = initial_mem;
        for (addr, value) in self.ram_init {
            let value = value as u32 as u64;
            initial_mem.insert((neo_memory::riscv::lookups::RAM_ID.0, addr), F::from_u64(value));
            twist.store(neo_memory::riscv::lookups::RAM_ID, addr, value);
        }
        for (reg, value) in self.reg_init {
            if reg >= 32 {
                return Err(PiCcsError::InvalidInput(format!(
                    "reg_init_u32: register index out of range: reg={reg} (expected 0..32)"
                )));
            }
            if reg == 0 && value != 0 {
                return Err(PiCcsError::InvalidInput(
                    "reg_init_u32: x0 must be 0 (non-zero init is forbidden)".into(),
                ));
            }
            let value = value as u32 as u64;
            initial_mem.insert((neo_memory::riscv::lookups::REG_ID.0, reg), F::from_u64(value));
            twist.store(neo_memory::riscv::lookups::REG_ID, reg, value);
        }

        let (k_ram, d_ram) = pow2_ceil_k(self.ram_bytes.max(4));
        let mem_layouts = HashMap::from([
            (
                neo_memory::riscv::lookups::RAM_ID.0,
                PlainMemLayout {
                    k: k_ram,
                    d: d_ram,
                    n_side: 2,
                    lanes: 1,
                },
            ),
            (
                neo_memory::riscv::lookups::REG_ID.0,
                PlainMemLayout {
                    k: 32,
                    d: 5,
                    n_side: 2,
                    lanes: 2,
                },
            ),
            (PROG_ID.0, prog_layout),
        ]);

        // Shout tables (either inferred, all, or explicitly provided).
        let mut shout_ops = match &self.shout_ops {
            Some(ops) => ops.clone(),
            None if self.shout_auto_minimal => infer_required_shout_opcodes(&program),
            None => all_shout_opcodes(),
        };
        // The ADD table is required even for programs without explicit ADD/ADDI due to address/PC wiring.
        shout_ops.insert(RiscvOpcode::Add);

        let mut table_specs: HashMap<u32, LutTableSpec> = HashMap::new();
        for op in shout_ops {
            let table_id = shout.opcode_to_id(op).0;
            table_specs.insert(
                table_id,
                LutTableSpec::RiscvOpcode {
                    opcode: op,
                    xlen: self.xlen,
                },
            );
        }
        let mut shout_table_ids: Vec<u32> = table_specs.keys().copied().collect();
        shout_table_ids.sort_unstable();

        let chunk_size = if self.chunk_size_auto {
            choose_rv32_b1_chunk_size(&mem_layouts, &shout_table_ids, estimated_steps)
                .map_err(|e| PiCcsError::InvalidInput(format!("auto chunk_size failed: {e}")))?
        } else {
            self.chunk_size
        };
        if chunk_size == 0 {
            return Err(PiCcsError::InvalidInput("chunk_size must be non-zero".into()));
        }

        let (ccs_base, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, chunk_size)
            .map_err(|e| PiCcsError::InvalidInput(format!("build_rv32_b1_step_ccs failed: {e}")))?;

        let phases_start = time_now();

        // Session + Ajtai committer + params (auto-picked for this CCS).
        let mut session = FoldingSession::<AjtaiSModule>::new_ajtai(self.mode.clone(), &ccs_base)?;
        let params = session.params().clone();
        let committer = session.committer().clone();

        let mut vm = RiscvCpu::new(self.xlen);
        vm.load_program(/*base=*/ 0, program);

        let empty_tables: HashMap<u32, LutTable<F>> = HashMap::new();
        let lut_lanes: HashMap<u32, usize> = HashMap::new();

        // CPU arithmetization (builds chunk witnesses and commits them).
        let mut cpu = R1csCpu::new(
            ccs_base,
            params,
            committer.clone(),
            layout.m_in,
            &empty_tables,
            &table_specs,
            rv32_b1_chunk_to_witness(layout.clone()),
        )
        .map_err(|e| PiCcsError::InvalidInput(format!("R1csCpu::new failed: {e}")))?;
        cpu = cpu
            .with_shared_cpu_bus(
                rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts.clone(), initial_mem.clone())
                    .map_err(|e| PiCcsError::InvalidInput(format!("rv32_b1_shared_cpu_bus_config failed: {e}")))?,
                chunk_size,
            )
            .map_err(|e| PiCcsError::InvalidInput(format!("shared bus inject failed: {e}")))?;

        // Always enforce step-to-step chunk chaining for RV32 B1.
        session.set_step_linking(rv32_b1_step_linking_config(&layout));

        // Execute + collect step bundles (and aux for output binding).
        let build_start = time_now();
        session.execute_shard_shared_cpu_bus(
            vm,
            twist,
            shout,
            /*max_steps=*/ max_steps,
            chunk_size,
            &mem_layouts,
            &empty_tables,
            &table_specs,
            &lut_lanes,
            &initial_mem,
            &cpu,
        )?;
        let build_commit_duration = elapsed_duration(build_start);
        if using_default_max_steps {
            let aux = session
                .shared_bus_aux()
                .ok_or_else(|| PiCcsError::InvalidInput("missing shared-bus aux (halt status unavailable)".into()))?;
            if !aux.did_halt {
                return Err(PiCcsError::InvalidInput(format!(
                    "RV32 execution did not halt within max_steps={max_steps}; call .max_steps(...) to raise the limit or ensure the guest halts (e.g. via ecall)"
                )));
            }
        }

        // Enforce that the *statement* initial memory matches chunk 0's public MemInit.
        let steps_public = session.steps_public();
        rv32_b1_enforce_chunk0_mem_init_matches_statement(&mem_layouts, &initial_mem, &steps_public)?;
        let setup_plus_build_duration = elapsed_duration(phases_start);
        let setup_duration = setup_plus_build_duration
            .checked_sub(build_commit_duration)
            .unwrap_or(Duration::ZERO);

        let ccs = cpu.ccs.clone();

        // Prove phase (timed)
        //
        // Includes the decode+semantics sidecar proofs (always) and the optional RV32M sidecar proof,
        // so reported prove time matches total work.
        let prove_start = time_now();

        // Batch all chunks into one sidecar proof (avoid per-chunk transcript/proof overhead).
        let mut mcs_insts = Vec::with_capacity(session.steps_witness().len());
        let mut mcs_wits = Vec::with_capacity(session.steps_witness().len());
        for step in session.steps_witness() {
            let (mcs_inst, mcs_wit) = &step.mcs;
            mcs_insts.push(mcs_inst.clone());
            mcs_wits.push(mcs_wit.clone());
        }
        let num_steps = mcs_insts.len();

        // Decode plumbing sidecar: prove instruction bits/fields/immediates and one-hot flags separately
        // so other proofs can assume decoded signals are sound without paying the padding knee.
        let decode_plumbing = {
            let decode_ccs = build_rv32_b1_decode_plumbing_sidecar_ccs(&layout)
                .map_err(|e| PiCcsError::InvalidInput(format!("{e}")))?;

            let mut tr = Poseidon2Transcript::new(b"neo.fold/rv32_b1/decode_plumbing_sidecar_batch");
            tr.append_message(b"decode_plumbing_sidecar/num_steps", &(num_steps as u64).to_le_bytes());
            let (me_out, proof) =
                crate::pi_ccs_prove_simple(&mut tr, &params, &decode_ccs, &mcs_insts, &mcs_wits, &committer)
                    .map_err(|e| PiCcsError::ProtocolError(format!("decode plumbing sidecar prove failed: {e}")))?;

            PiCcsProofBundle {
                num_steps,
                me_out,
                proof,
            }
        };

        // Semantics sidecar: prove full RV32 B1 step semantics separately so the main step CCS can stay thin
        // (it mostly exists to host the injected shared-bus constraints).
        let semantics = {
            let semantics_ccs = build_rv32_b1_semantics_sidecar_ccs(&layout, &mem_layouts)
                .map_err(|e| PiCcsError::InvalidInput(format!("{e}")))?;

            let mut tr = Poseidon2Transcript::new(b"neo.fold/rv32_b1/semantics_sidecar_batch");
            tr.append_message(b"semantics_sidecar/num_steps", &(num_steps as u64).to_le_bytes());
            let (me_out, proof) =
                crate::pi_ccs_prove_simple(&mut tr, &params, &semantics_ccs, &mcs_insts, &mcs_wits, &committer)
                    .map_err(|e| PiCcsError::ProtocolError(format!("semantics sidecar prove failed: {e}")))?;

            PiCcsProofBundle {
                num_steps,
                me_out,
                proof,
            }
        };

        // Optional RV32M sidecar: prove MUL/DIV/REM helper constraints separately so the main step CCS
        // stays small on non-M workloads.
        //
        // Jolt-ish direction: charge RV32M only on lanes that actually execute an M op in a chunk.
        // We do this by proving an RV32M sidecar CCS that includes constraints only for the selected lanes.
        let rv32m = {
            if !uses_rv32m {
                None
            } else {
                fn z_at(
                    inst: &neo_ccs::relations::McsInstance<Cmt, F>,
                    wit: &neo_ccs::relations::McsWitness<F>,
                    idx: usize,
                ) -> F {
                    if idx < inst.m_in {
                        inst.x[idx]
                    } else {
                        wit.w[idx - inst.m_in]
                    }
                }

                let mut out: Vec<Rv32B1Rv32mEventSidecarChunkProof> = Vec::new();
                for (chunk_idx, step) in session.steps_witness().iter().enumerate() {
                    let (inst, wit) = &step.mcs;
                    let count = inst.x.get(layout.rv32m_count).copied().ok_or_else(|| {
                        PiCcsError::InvalidInput(format!(
                            "rv32m_count not present in public x: need idx {} but x.len()={}",
                            layout.rv32m_count,
                            inst.x.len()
                        ))
                    })?;
                    if count == F::ZERO {
                        continue;
                    }

                    let expected = count.as_canonical_u64() as usize;
                    let mut lanes: Vec<u32> = Vec::with_capacity(expected);

                    for j in 0..layout.chunk_size {
                        let mut is_m = false;
                        for &col in &[
                            layout.is_mul(j),
                            layout.is_mulh(j),
                            layout.is_mulhu(j),
                            layout.is_mulhsu(j),
                            layout.is_div(j),
                            layout.is_divu(j),
                            layout.is_rem(j),
                            layout.is_remu(j),
                        ] {
                            if z_at(inst, wit, col) != F::ZERO {
                                is_m = true;
                                break;
                            }
                        }
                        if is_m {
                            lanes.push(j as u32);
                        }
                    }

                    if lanes.len() != expected {
                        return Err(PiCcsError::InvalidInput(format!(
                            "rv32m_count mismatch in chunk {chunk_idx}: public rv32m_count={expected}, but decoded {} RV32M lanes",
                            lanes.len()
                        )));
                    }

                    let lanes_usize: Vec<usize> = lanes.iter().map(|&j| j as usize).collect();
                    let rv32m_ccs = build_rv32_b1_rv32m_event_sidecar_ccs(&layout, &lanes_usize)
                        .map_err(|e| PiCcsError::InvalidInput(format!("{e}")))?;

                    let mut tr = Poseidon2Transcript::new(b"neo.fold/rv32_b1/rv32m_event_sidecar_chunk");
                    tr.append_message(b"rv32m_event_sidecar/chunk_idx", &(chunk_idx as u64).to_le_bytes());
                    tr.append_message(b"rv32m_event_sidecar/lanes_len", &(lanes.len() as u64).to_le_bytes());
                    for &lane in &lanes {
                        tr.append_message(b"rv32m_event_sidecar/lane", &(lane as u64).to_le_bytes());
                    }

                    let (me_out, proof) = crate::pi_ccs_prove_simple(
                        &mut tr,
                        &params,
                        &rv32m_ccs,
                        core::slice::from_ref(inst),
                        core::slice::from_ref(wit),
                        &committer,
                    )
                    .map_err(|e| PiCcsError::ProtocolError(format!("rv32m event sidecar prove failed: {e}")))?;

                    out.push(Rv32B1Rv32mEventSidecarChunkProof {
                        chunk_idx,
                        lanes,
                        me_out,
                        proof,
                    });
                }

                if out.is_empty() {
                    None
                } else {
                    Some(out)
                }
            }
        };

        let (main, output_binding_cfg) = if self.output_claims.is_empty() {
            (session.fold_and_prove(&ccs)?, None)
        } else {
            let out_mem_id = match self.output_target {
                OutputTarget::Ram => neo_memory::riscv::lookups::RAM_ID.0,
                OutputTarget::Reg => neo_memory::riscv::lookups::REG_ID.0,
            };
            let out_layout = mem_layouts.get(&out_mem_id).ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "output binding: missing PlainMemLayout for mem_id={out_mem_id}"
                ))
            })?;
            let expected_k = 1usize
                .checked_shl(out_layout.d as u32)
                .ok_or_else(|| PiCcsError::InvalidInput("output binding: 2^d overflow".into()))?;
            if out_layout.k != expected_k {
                return Err(PiCcsError::InvalidInput(format!(
                    "output binding: mem_id={out_mem_id} has k={}, but expected 2^d={} (d={})",
                    out_layout.k, expected_k, out_layout.d
                )));
            }
            let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
            mem_ids.sort_unstable();
            let mem_idx = mem_ids
                .iter()
                .position(|&id| id == out_mem_id)
                .ok_or_else(|| PiCcsError::InvalidInput("output binding: mem_id not in mem_layouts".into()))?;

            let ob_cfg = OutputBindingConfig::new(out_layout.d, self.output_claims.clone()).with_mem_idx(mem_idx);
            let proof = session.fold_and_prove_with_output_binding_auto_simple(&ccs, &ob_cfg)?;
            (proof, Some(ob_cfg))
        };
        let prove_duration = elapsed_duration(prove_start);
        let prove_phase_durations = Rv32B1ProvePhaseDurations {
            setup: setup_duration,
            build_commit: build_commit_duration,
            fold_and_prove: prove_duration,
        };

        let proof_bundle = Rv32B1ProofBundle {
            main,
            decode_plumbing,
            semantics,
            rv32m,
        };

        Ok(Rv32B1Run {
            program_base: self.program_base,
            program_bytes: self.program_bytes,
            xlen: self.xlen,
            session,
            ccs,
            layout,
            mem_layouts,
            initial_mem,
            output_binding_cfg,
            proof_bundle,
            prove_duration,
            prove_phase_durations,
            verify_duration: None,
        })
    }
}

#[derive(Clone, Debug)]
pub struct PiCcsProofBundle {
    pub num_steps: usize,
    pub me_out: Vec<MeInstance<Cmt, F, K>>,
    pub proof: crate::PiCcsProof,
}

#[derive(Clone, Debug)]
pub struct Rv32B1Rv32mEventSidecarChunkProof {
    pub chunk_idx: usize,
    /// Lane indices `j` (within this chunk) that execute an RV32M instruction.
    pub lanes: Vec<u32>,
    pub me_out: Vec<MeInstance<Cmt, F, K>>,
    pub proof: crate::PiCcsProof,
}

#[derive(Clone, Debug)]
pub struct Rv32B1ProofBundle {
    pub main: ShardProof,
    pub decode_plumbing: PiCcsProofBundle,
    pub semantics: PiCcsProofBundle,
    pub rv32m: Option<Vec<Rv32B1Rv32mEventSidecarChunkProof>>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv32B1ProvePhaseDurations {
    pub setup: Duration,
    pub build_commit: Duration,
    pub fold_and_prove: Duration,
}

pub struct Rv32B1Run {
    program_base: u64,
    program_bytes: Vec<u8>,
    xlen: usize,
    session: FoldingSession<AjtaiSModule>,
    ccs: CcsStructure<F>,
    layout: Rv32B1Layout,
    mem_layouts: HashMap<u32, PlainMemLayout>,
    initial_mem: HashMap<(u32, u64), F>,
    output_binding_cfg: Option<OutputBindingConfig>,
    proof_bundle: Rv32B1ProofBundle,
    prove_duration: Duration,
    prove_phase_durations: Rv32B1ProvePhaseDurations,
    verify_duration: Option<Duration>,
}

impl Rv32B1Run {
    pub fn params(&self) -> &NeoParams {
        self.session.params()
    }

    pub fn committer(&self) -> &AjtaiSModule {
        self.session.committer()
    }

    pub fn ccs(&self) -> &CcsStructure<F> {
        &self.ccs
    }

    pub fn layout(&self) -> &Rv32B1Layout {
        &self.layout
    }

    /// Deterministically re-run the VM to recover the executed trace.
    ///
    /// This is intended for Tier 2.1 "time-in-rows" work (execution-table extraction and
    /// event-table arguments). It replays the program using the *public statement* initial memory
    /// (`initial_mem`) and the same `xlen`.
    ///
    /// Note: this is not used by proving/verification today; it's a debugging/scaffolding API.
    pub fn vm_trace(&self) -> Result<neo_vm_trace::VmTrace<u64, u64>, PiCcsError> {
        let aux = self.session.shared_bus_aux().ok_or_else(|| {
            PiCcsError::InvalidInput(
                "vm_trace requires shared-bus aux (this run was not produced by shared-bus execution)".into(),
            )
        })?;

        let program = decode_program(&self.program_bytes)
            .map_err(|e| PiCcsError::InvalidInput(format!("decode_program failed: {e}")))?;
        let mut vm = RiscvCpu::new(self.xlen);
        vm.load_program(self.program_base, program);

        let mut twist = RiscvMemory::with_program_in_twist(self.xlen, PROG_ID, self.program_base, &self.program_bytes);
        for ((mem_id, addr), value) in &self.initial_mem {
            let value_u64 = value.as_canonical_u64();
            match *mem_id {
                id if id == RAM_ID.0 => twist.store(RAM_ID, *addr, value_u64),
                id if id == REG_ID.0 => twist.store(REG_ID, *addr, value_u64),
                _ => {}
            }
        }

        let shout = RiscvShoutTables::new(self.xlen);
        let trace = neo_vm_trace::trace_program(vm, twist, shout, aux.original_len)
            .map_err(|e| PiCcsError::InvalidInput(format!("trace_program failed: {e}")))?;

        if trace.steps.len() != aux.original_len {
            return Err(PiCcsError::InvalidInput(format!(
                "vm_trace length mismatch: retrace_len={} expected_len={}",
                trace.steps.len(),
                aux.original_len
            )));
        }
        if trace.did_halt() != aux.did_halt {
            return Err(PiCcsError::InvalidInput(format!(
                "vm_trace halt mismatch: retrace_did_halt={} expected_did_halt={}",
                trace.did_halt(),
                aux.did_halt
            )));
        }

        Ok(trace)
    }

    pub fn prove_phase_durations(&self) -> Rv32B1ProvePhaseDurations {
        self.prove_phase_durations
    }

    /// Build a padded-to-power-of-two RV32 execution table from the replayed trace.
    pub fn exec_table_padded_pow2(
        &self,
        min_len: usize,
    ) -> Result<neo_memory::riscv::exec_table::Rv32ExecTable, PiCcsError> {
        let trace = self.vm_trace()?;
        neo_memory::riscv::exec_table::Rv32ExecTable::from_trace_padded_pow2(&trace, min_len)
            .map_err(|e| PiCcsError::InvalidInput(format!("Rv32ExecTable::from_trace_padded_pow2 failed: {e}")))
    }

    fn collected_mcs_instances(&self) -> Vec<neo_ccs::McsInstance<Cmt, F>> {
        let steps_public = self.session.steps_public();
        let mut mcs_insts = Vec::with_capacity(steps_public.len());
        for step in &steps_public {
            mcs_insts.push(step.mcs_inst.clone());
        }
        mcs_insts
    }

    fn verify_sidecars_inner(
        &self,
        bundle: &Rv32B1ProofBundle,
        mcs_insts: &[neo_ccs::McsInstance<Cmt, F>],
    ) -> Result<(), PiCcsError> {
        // Rebuild verifier-side expected CCSes from statement/layout.
        //
        // Security: never trust prover-supplied CCS structures from the proof bundle.
        let decode_ccs = build_rv32_b1_decode_plumbing_sidecar_ccs(&self.layout).map_err(|e| {
            PiCcsError::ProtocolError(format!("decode plumbing sidecar: failed to rebuild verifier CCS: {e}"))
        })?;
        let semantics_ccs = build_rv32_b1_semantics_sidecar_ccs(&self.layout, &self.mem_layouts).map_err(|e| {
            PiCcsError::ProtocolError(format!("semantics sidecar: failed to rebuild verifier CCS: {e}"))
        })?;

        if mcs_insts.len() != bundle.decode_plumbing.num_steps {
            return Err(PiCcsError::ProtocolError(
                "decode plumbing sidecar: step count mismatch".into(),
            ));
        }
        if mcs_insts.len() != bundle.semantics.num_steps {
            return Err(PiCcsError::ProtocolError(
                "semantics sidecar: step count mismatch".into(),
            ));
        }

        // Decode plumbing sidecar must always verify (it carries instruction decode signals).
        {
            let mut tr = Poseidon2Transcript::new(b"neo.fold/rv32_b1/decode_plumbing_sidecar_batch");
            tr.append_message(
                b"decode_plumbing_sidecar/num_steps",
                &(mcs_insts.len() as u64).to_le_bytes(),
            );
            let ok = crate::pi_ccs_verify(
                &mut tr,
                self.session.params(),
                &decode_ccs,
                mcs_insts,
                &[],
                &bundle.decode_plumbing.me_out,
                &bundle.decode_plumbing.proof,
            )?;
            if !ok {
                return Err(PiCcsError::ProtocolError(
                    "decode plumbing sidecar: verification failed".into(),
                ));
            }
        }

        // Semantics sidecar must always verify (it carries the full RV32 B1 step semantics).
        {
            let mut tr = Poseidon2Transcript::new(b"neo.fold/rv32_b1/semantics_sidecar_batch");
            tr.append_message(b"semantics_sidecar/num_steps", &(mcs_insts.len() as u64).to_le_bytes());
            let ok = crate::pi_ccs_verify(
                &mut tr,
                self.session.params(),
                &semantics_ccs,
                mcs_insts,
                &[],
                &bundle.semantics.me_out,
                &bundle.semantics.proof,
            )?;
            if !ok {
                return Err(PiCcsError::ProtocolError(
                    "semantics sidecar: verification failed".into(),
                ));
            }
        }

        match &bundle.rv32m {
            None => {
                // If the statement contains any RV32M rows, a proof must be present.
                for (chunk_idx, inst) in mcs_insts.iter().enumerate() {
                    let count = inst
                        .x
                        .get(self.layout.rv32m_count)
                        .copied()
                        .ok_or_else(|| {
                            PiCcsError::ProtocolError(format!(
                                "rv32m_count not present in public x: need idx {} but x.len()={}",
                                self.layout.rv32m_count,
                                inst.x.len()
                            ))
                        })?;
                    if count != F::ZERO {
                        return Err(PiCcsError::ProtocolError(format!(
                            "rv32m sidecar: missing proof for chunk {chunk_idx} with rv32m_count != 0"
                        )));
                    }
                }
            }
            Some(chunks) => {
                let mut by_chunk: HashMap<usize, &Rv32B1Rv32mEventSidecarChunkProof> = HashMap::new();
                for p in chunks {
                    if p.chunk_idx >= mcs_insts.len() {
                        return Err(PiCcsError::ProtocolError(format!(
                            "rv32m sidecar: proof chunk_idx {} out of range (num_chunks={})",
                            p.chunk_idx,
                            mcs_insts.len()
                        )));
                    }
                    if by_chunk.insert(p.chunk_idx, p).is_some() {
                        return Err(PiCcsError::ProtocolError(format!(
                            "rv32m sidecar: duplicate proof for chunk_idx {}",
                            p.chunk_idx
                        )));
                    }
                }

                for (chunk_idx, inst) in mcs_insts.iter().enumerate() {
                    let count = inst
                        .x
                        .get(self.layout.rv32m_count)
                        .copied()
                        .ok_or_else(|| {
                            PiCcsError::ProtocolError(format!(
                                "rv32m_count not present in public x: need idx {} but x.len()={}",
                                self.layout.rv32m_count,
                                inst.x.len()
                            ))
                        })?;
                    let expected = count.as_canonical_u64() as usize;
                    match (expected == 0, by_chunk.get(&chunk_idx)) {
                        (true, None) => {}
                        (true, Some(_)) => {
                            return Err(PiCcsError::ProtocolError(format!(
                                "rv32m sidecar: proof present for chunk {chunk_idx} but rv32m_count == 0"
                            )));
                        }
                        (false, None) => {
                            return Err(PiCcsError::ProtocolError(format!(
                                "rv32m sidecar: missing proof for chunk {chunk_idx} with rv32m_count={expected}"
                            )));
                        }
                        (false, Some(p)) => {
                            if p.lanes.len() != expected {
                                return Err(PiCcsError::ProtocolError(format!(
                                    "rv32m sidecar: lane count mismatch for chunk {chunk_idx} (expected {expected}, got {})",
                                    p.lanes.len()
                                )));
                            }
                            let lanes_usize: Vec<usize> = p.lanes.iter().map(|&j| j as usize).collect();
                            let rv32m_ccs = build_rv32_b1_rv32m_event_sidecar_ccs(&self.layout, &lanes_usize)
                                .map_err(|e| PiCcsError::ProtocolError(format!("{e}")))?;

                            let mut tr = Poseidon2Transcript::new(b"neo.fold/rv32_b1/rv32m_event_sidecar_chunk");
                            tr.append_message(b"rv32m_event_sidecar/chunk_idx", &(chunk_idx as u64).to_le_bytes());
                            tr.append_message(b"rv32m_event_sidecar/lanes_len", &(p.lanes.len() as u64).to_le_bytes());
                            for &lane in &p.lanes {
                                tr.append_message(b"rv32m_event_sidecar/lane", &(lane as u64).to_le_bytes());
                            }

                            let ok = crate::pi_ccs_verify(
                                &mut tr,
                                self.session.params(),
                                &rv32m_ccs,
                                core::slice::from_ref(inst),
                                &[],
                                &p.me_out,
                                &p.proof,
                            )?;
                            if !ok {
                                return Err(PiCcsError::ProtocolError(format!(
                                    "rv32m sidecar: verification failed for chunk {chunk_idx}"
                                )));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn verify_bundle_inner(&self, bundle: &Rv32B1ProofBundle) -> Result<(), PiCcsError> {
        let ok = match &self.output_binding_cfg {
            None => self.session.verify_collected(&self.ccs, &bundle.main)?,
            Some(cfg) => self
                .session
                .verify_with_output_binding_collected_simple(&self.ccs, &bundle.main, cfg)?,
        };
        if !ok {
            return Err(PiCcsError::ProtocolError("verification failed".into()));
        }

        let mcs_insts = self.collected_mcs_instances();
        self.verify_sidecars_inner(bundle, &mcs_insts)?;

        Ok(())
    }

    pub fn verify_proof_bundle(&self, bundle: &Rv32B1ProofBundle) -> Result<(), PiCcsError> {
        self.verify_bundle_inner(bundle)
    }

    pub fn verify(&mut self) -> Result<(), PiCcsError> {
        let verify_start = time_now();
        self.verify_proof_bundle(&self.proof_bundle)?;
        self.verify_duration = Some(elapsed_duration(verify_start));
        Ok(())
    }

    pub fn proof(&self) -> &Rv32B1ProofBundle {
        &self.proof_bundle
    }

    /// Access the collected per-step witness bundles (includes private witness).
    ///
    /// This is intended for debugging/profiling and for tests that want to inspect witness shapes.
    pub fn steps_witness(&self) -> &[StepWitnessBundle<Cmt, F, K>] {
        self.session.steps_witness()
    }

    pub fn steps_public(&self) -> Vec<StepInstanceBundle<Cmt, F, K>> {
        self.session.steps_public()
    }

    pub fn final_boundary_state(&self) -> Result<Rv32BoundaryState, PiCcsError> {
        let steps_public = self.steps_public();
        let last = steps_public
            .last()
            .ok_or_else(|| PiCcsError::InvalidInput("no steps collected".into()))?;
        extract_boundary_state(&self.layout, &last.mcs_inst.x)
            .map_err(|e| PiCcsError::InvalidInput(format!("extract_boundary_state failed: {e}")))
    }

    pub fn verify_output_claim(&self, output_addr: u64, expected_output: F) -> Result<bool, PiCcsError> {
        self.verify_output_claim_in_bundle(&self.proof_bundle, output_addr, expected_output)
    }

    /// Verify an output claim against an explicit RV32 proof bundle.
    ///
    /// This always verifies required RV32 sidecars (decode plumbing, semantics, optional RV32M)
    /// before checking the output binding against `bundle.main`.
    pub fn verify_output_claim_in_bundle(
        &self,
        bundle: &Rv32B1ProofBundle,
        output_addr: u64,
        expected_output: F,
    ) -> Result<bool, PiCcsError> {
        let cfg = self
            .output_binding_cfg
            .as_ref()
            .ok_or_else(|| PiCcsError::InvalidInput("no output binding configured".into()))?;
        let mcs_insts = self.collected_mcs_instances();
        self.verify_sidecars_inner(bundle, &mcs_insts)?;
        let ob_cfg = simple_output_config(cfg.num_bits, output_addr, expected_output).with_mem_idx(cfg.mem_idx);
        self.session
            .verify_with_output_binding_collected_simple(&self.ccs, &bundle.main, &ob_cfg)
    }

    pub fn verify_default_output_claim(&self) -> Result<bool, PiCcsError> {
        self.verify_default_output_claim_in_bundle(&self.proof_bundle)
    }

    /// Verify the configured default output binding against an explicit RV32 proof bundle.
    ///
    /// This always verifies required RV32 sidecars (decode plumbing, semantics, optional RV32M)
    /// before checking the output binding against `bundle.main`.
    pub fn verify_default_output_claim_in_bundle(&self, bundle: &Rv32B1ProofBundle) -> Result<bool, PiCcsError> {
        let ob_cfg = self
            .output_binding_cfg
            .as_ref()
            .ok_or_else(|| PiCcsError::InvalidInput("no output binding configured".into()))?;
        let mcs_insts = self.collected_mcs_instances();
        self.verify_sidecars_inner(bundle, &mcs_insts)?;
        self.session
            .verify_with_output_binding_collected_simple(&self.ccs, &bundle.main, ob_cfg)
    }

    pub fn verify_output_claims(&self, output_claims: ProgramIO<F>) -> Result<bool, PiCcsError> {
        self.verify_output_claims_in_bundle(&self.proof_bundle, output_claims)
    }

    /// Verify output claims against an explicit RV32 proof bundle.
    ///
    /// This always verifies required RV32 sidecars (decode plumbing, semantics, optional RV32M)
    /// before checking the output binding against `bundle.main`.
    pub fn verify_output_claims_in_bundle(
        &self,
        bundle: &Rv32B1ProofBundle,
        output_claims: ProgramIO<F>,
    ) -> Result<bool, PiCcsError> {
        let cfg = self
            .output_binding_cfg
            .as_ref()
            .ok_or_else(|| PiCcsError::InvalidInput("no output binding configured".into()))?;
        if output_claims.is_empty() {
            return Err(PiCcsError::InvalidInput("output_claims must be non-empty".into()));
        }
        let mcs_insts = self.collected_mcs_instances();
        self.verify_sidecars_inner(bundle, &mcs_insts)?;
        let ob_cfg = OutputBindingConfig::new(cfg.num_bits, output_claims).with_mem_idx(cfg.mem_idx);
        self.session
            .verify_with_output_binding_collected_simple(&self.ccs, &bundle.main, &ob_cfg)
    }

    /// Original unpadded RV32 trace length (instruction count), if this run was built via shared-bus execution.
    pub fn riscv_trace_len(&self) -> Result<usize, PiCcsError> {
        let aux = self
            .session
            .shared_bus_aux()
            .ok_or_else(|| PiCcsError::InvalidInput("missing shared-bus aux (trace length unavailable)".into()))?;
        Ok(aux.original_len)
    }

    /// CCS constraint count (rows). For RV32 B1 this is the size of the per-chunk step circuit.
    pub fn ccs_num_constraints(&self) -> usize {
        self.ccs.n
    }

    /// CCS variable count (cols). For RV32 B1 this is the number of witness variables per chunk.
    pub fn ccs_num_variables(&self) -> usize {
        self.ccs.m
    }

    /// Number of folding steps proven (one per collected chunk).
    pub fn fold_count(&self) -> usize {
        self.proof_bundle.main.steps.len()
    }

    /// Chunk size (steps per folding step) used for this run.
    pub fn chunk_size(&self) -> usize {
        self.layout.chunk_size
    }

    /// Count the number of Shout lookups actually used across the executed trace (active rows only).
    pub fn shout_lookup_count(&self) -> Result<usize, PiCcsError> {
        let mut count = 0usize;
        for step in self.session.steps_witness() {
            let x = &step.mcs.0.x;
            let w = &step.mcs.1.w;
            let m_in = step.mcs.0.m_in;

            let z_at = |idx: usize| -> Result<F, PiCcsError> {
                if idx < m_in {
                    x.get(idx).copied().ok_or_else(|| {
                        PiCcsError::InvalidInput(format!(
                            "witness index {idx} out of bounds for public input len={m_in}"
                        ))
                    })
                } else {
                    let w_idx = idx - m_in;
                    w.get(w_idx).copied().ok_or_else(|| {
                        PiCcsError::InvalidInput(format!(
                            "witness index {idx} (w[{w_idx}]) out of bounds for witness len={}",
                            w.len()
                        ))
                    })
                }
            };

            for j in 0..self.layout.chunk_size {
                if z_at(self.layout.is_active(j))? == F::ZERO {
                    continue;
                }
                for inst in &self.layout.bus.shout_cols {
                    for lane in &inst.lanes {
                        let col = self.layout.bus.bus_cell(lane.has_lookup, j);
                        if z_at(col)? != F::ZERO {
                            count += 1;
                        }
                    }
                }
            }
        }
        Ok(count)
    }

    pub fn mem_layouts(&self) -> &HashMap<u32, PlainMemLayout> {
        &self.mem_layouts
    }

    pub fn initial_mem(&self) -> &HashMap<(u32, u64), F> {
        &self.initial_mem
    }

    pub fn prove_duration(&self) -> Duration {
        self.prove_duration
    }

    pub fn verify_duration(&self) -> Option<Duration> {
        self.verify_duration
    }
}

fn choose_rv32_b1_chunk_size(
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    shout_table_ids: &[u32],
    estimated_steps: usize,
) -> Result<usize, String> {
    if estimated_steps == 0 {
        return Err("estimated_steps must be non-zero".into());
    }

    let mut candidates: Vec<usize> = Vec::new();
    let max_candidate = estimated_steps.min(256).max(1);
    let mut c = 1usize;
    while c <= max_candidate {
        candidates.push(c);
        c = c
            .checked_mul(2)
            .ok_or_else(|| "chunk_size overflow".to_string())?;
    }
    if estimated_steps <= 256 && !candidates.contains(&estimated_steps) {
        candidates.push(estimated_steps);
    }

    let mut best_chunk_size = 1usize;
    let mut best_bucket = usize::MAX;
    let mut best_work: u128 = u128::MAX;

    for chunk_size in candidates {
        let counts = estimate_rv32_b1_all_ccs_counts(mem_layouts, shout_table_ids, chunk_size)?;

        let chunks_est = estimated_steps.div_ceil(chunk_size);

        let m_pad = counts.step.m.next_power_of_two();
        let step_n_pad = counts.step.n.next_power_of_two();
        let decode_n_pad = counts.decode_plumbing_n.next_power_of_two();
        let semantics_n_pad = counts.semantics_n.next_power_of_two();

        let bucket = m_pad.max(step_n_pad.max(decode_n_pad).max(semantics_n_pad));
        let work = (m_pad as u128)
            .saturating_mul(chunks_est as u128)
            .saturating_mul(
                (step_n_pad as u128)
                    .saturating_add(decode_n_pad as u128)
                    .saturating_add(semantics_n_pad as u128),
            );

        if bucket < best_bucket
            || (bucket == best_bucket && (work < best_work || (work == best_work && chunk_size > best_chunk_size)))
        {
            best_bucket = bucket;
            best_work = work;
            best_chunk_size = chunk_size;
        }
    }

    Ok(best_chunk_size)
}
