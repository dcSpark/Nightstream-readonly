//! Convenience wrappers for verifying RISC-V shard proofs safely.
//!
//! These helpers are intentionally small: they standardize the step-linking configuration
//! for RV32 B1 chunked execution so callers don't accidentally verify a "bag of chunks".

#![allow(non_snake_case)]

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use crate::output_binding::{simple_output_config, OutputBindingConfig};
use crate::pi_ccs::FoldingMode;
use crate::session::FoldingSession;
use neo_reductions::engines::optimized_engine::oracle::SparseCache;
use crate::shard::{
    fold_shard_verify_with_output_binding_and_step_linking, fold_shard_verify_with_step_linking, CommitMixers,
    ShardFoldOutputs, ShardProof, StepLinkingConfig,
};
use crate::PiCcsError;
use neo_ajtai::{AjtaiSModule, Commitment as Cmt};
use neo_ccs::{CcsStructure, Mat, MeInstance};
use neo_math::{F, K};
use neo_memory::mem_init_from_initial_mem;
use neo_memory::output_check::ProgramIO;
use neo_memory::plain::LutTable;
use neo_memory::plain::PlainMemLayout;
use neo_memory::riscv::ccs::{
    build_rv32_b1_step_ccs, rv32_b1_chunk_to_witness, rv32_b1_shared_cpu_bus_config, rv32_b1_step_linking_pairs,
    Rv32B1Layout,
};
use neo_memory::riscv::lookups::{decode_program, RiscvCpu, RiscvInstruction, RiscvOpcode, RiscvShoutTables, PROG_ID};
use neo_memory::riscv::shard::{extract_boundary_state, Rv32BoundaryState};
use neo_memory::witness::LutTableSpec;
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_memory::R1csCpu;
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_vm_trace::Twist as _;
use p3_field::PrimeCharacteristicRing;

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

/// Per-phase timing breakdown for `Rv32B1::prove()`.
///
/// Captures wall-clock time for each major phase so callers can identify bottlenecks.
#[derive(Clone, Debug, Default)]
pub struct ProveTimings {
    /// Program decode, memory layout setup, and Shout table inference.
    pub decode_and_setup: Duration,
    /// CCS construction (base + shared-bus wiring) and session/committer creation.
    pub ccs_and_shared_bus: Duration,
    /// RISC-V VM execution and shard/witness collection.
    pub vm_execution: Duration,
    /// Fold-and-prove (sumcheck + Ajtai commitment).
    pub fold_and_prove: Duration,
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
    let step_linking = rv32_b1_step_linking_config(layout);
    fold_shard_verify_with_step_linking(mode, tr, params, s_me, steps, acc_init, proof, mixers, &step_linking)
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
    rv32_b1_enforce_chunk0_mem_init_matches_statement(mem_layouts, statement_initial_mem, steps)?;
    fold_shard_verify_rv32_b1(mode, tr, params, s_me, steps, acc_init, proof, mixers, layout)
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
    let step_linking = rv32_b1_step_linking_config(layout);
    fold_shard_verify_with_output_binding_and_step_linking(
        mode,
        tr,
        params,
        s_me,
        steps,
        acc_init,
        proof,
        mixers,
        ob_cfg,
        &step_linking,
    )
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
                    // RV32 B1 proves RV32M MUL* in-circuit (no Shout table required).
                    RiscvOpcode::Mul | RiscvOpcode::Mulh | RiscvOpcode::Mulhu | RiscvOpcode::Mulhsu => {}
                    // RV32 B1 proves RV32M DIV*/REM* in-circuit, but it requires a SLTU lookup to prove
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
    // RV32M ops are proven in-circuit (MUL/DIVU/REMU) or not yet supported (MULH/DIV/REM, etc.).
    HashSet::from([And, Xor, Or, Sub, Add, Sltu, Slt, Eq, Neq, Sll, Srl, Sra])
}

/// High-level “few lines” builder for proving/verifying an RV32 program using the B1 shared-bus step circuit.
///
/// Pre-computed SparseCache + matrix digest for RV32 B1 circuit preprocessing.
///
/// The `SparseCache::build()` and matrix-digest computation are the most expensive parts
/// of both proving and verification.  This struct captures those artefacts so they can be
/// computed once and reused across many `Rv32B1::prove()` / `build_verifier()` calls with
/// the same ROM, `ram_bytes`, and `chunk_size`.
///
/// Build with [`Rv32B1::build_ccs_cache`], then inject into subsequent builders via
/// [`Rv32B1::with_ccs_cache`].
///
/// **Important**: the CCS structure itself is still built from scratch each time (it is fast).
/// Only the SparseCache (CSC decomposition + matrix digest) is cached.
pub struct Rv32B1CcsCache {
    pub sparse: Arc<SparseCache<F>>,
}

impl std::fmt::Debug for Rv32B1CcsCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rv32B1CcsCache")
            .field("sparse_len", &self.sparse.len())
            .finish()
    }
}

/// This:
/// - chooses parameters + Ajtai committer automatically,
/// - infers the minimal Shout table set from the program (unless overridden),
/// - enforces RV32 B1 step linking, and
/// - (optionally) proves output binding against RAM.
#[derive(Clone, Debug)]
pub struct Rv32B1 {
    program_base: u64,
    program_bytes: Vec<u8>,
    xlen: usize,
    ram_bytes: usize,
    chunk_size: usize,
    max_steps: Option<usize>,
    mode: FoldingMode,
    shout_auto_minimal: bool,
    shout_ops: Option<HashSet<RiscvOpcode>>,
    output_claims: ProgramIO<F>,
    ram_init: HashMap<u64, u64>,
    ccs_cache: Option<Arc<Rv32B1CcsCache>>,
}

/// Default instruction cap for RV32B1 runs when `max_steps` is not specified.
///
/// The runner stops early if the guest halts (e.g. via `ecall`), so this is only a safety bound
/// against non-halting guests.
const DEFAULT_RV32B1_MAX_STEPS: usize = 1 << 20;

impl Rv32B1 {
    /// Create a runner from ROM bytes (must be a valid RV32 program encoding).
    pub fn from_rom(program_base: u64, program_bytes: &[u8]) -> Self {
        Self {
            program_base,
            program_bytes: program_bytes.to_vec(),
            xlen: 32,
            ram_bytes: 0x200,
            chunk_size: 1,
            max_steps: None,
            mode: FoldingMode::Optimized,
            shout_auto_minimal: true,
            shout_ops: None,
            output_claims: ProgramIO::new(),
            ram_init: HashMap::new(),
            ccs_cache: None,
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

    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
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
        self
    }

    pub fn output_claim(mut self, addr: u64, value: F) -> Self {
        self.output_claims = self.output_claims.with_output(addr, value);
        self
    }

    pub fn ram_init_u32(mut self, addr: u64, value: u32) -> Self {
        self.ram_init.insert(addr, value as u64);
        self
    }

    /// Attach a pre-built CCS cache to skip CCS synthesis in [`prove`] and [`build_verifier`].
    ///
    /// The cache **must** have been built from a builder with the same `program_bytes`,
    /// `ram_bytes`, `chunk_size`, and Shout configuration. No runtime validation is performed;
    /// mismatched caches produce undefined behaviour.
    pub fn with_ccs_cache(mut self, cache: Arc<Rv32B1CcsCache>) -> Self {
        self.ccs_cache = Some(cache);
        self
    }

    /// Build the CCS preprocessing cache from the current builder configuration.
    ///
    /// This performs program decoding, memory-layout setup, Shout-table inference,
    /// CCS construction, and `SparseCache` synthesis -- exactly the same work that
    /// [`prove`] and [`build_verifier`] would do on first call -- and packages the
    /// result so it can be shared across many runs.
    ///
    /// ```ignore
    /// let cache = Rv32B1::from_rom(0, &rom).ram_bytes(0x40000).chunk_size(1024)
    ///     .shout_auto_minimal().build_ccs_cache()?;
    /// let cache = Arc::new(cache);
    ///
    /// // Subsequent prove/verify calls skip CCS synthesis:
    /// let run = Rv32B1::from_rom(0, &rom).ram_bytes(0x40000).chunk_size(1024)
    ///     .shout_auto_minimal().with_ccs_cache(cache.clone())
    ///     .ram_init_u32(0x104, 42).prove()?;
    /// ```
    pub fn build_ccs_cache(&self) -> Result<Rv32B1CcsCache, PiCcsError> {
        let program = decode_program(&self.program_bytes)
            .map_err(|e| PiCcsError::InvalidInput(format!("decode_program failed: {e}")))?;

        let (prog_layout, initial_mem) = neo_memory::riscv::rom_init::prog_rom_layout_and_init_words::<F>(
            PROG_ID,
            /*base_addr=*/ 0,
            &self.program_bytes,
        )
        .map_err(|e| PiCcsError::InvalidInput(format!("prog_rom_layout_and_init_words failed: {e}")))?;

        let (k_ram, d_ram) = pow2_ceil_k(self.ram_bytes.max(4));
        let _ = d_ram;
        let mem_layouts = HashMap::from([
            (
                neo_memory::riscv::lookups::RAM_ID.0,
                PlainMemLayout {
                    k: k_ram,
                    d: pow2_ceil_k(self.ram_bytes.max(4)).1,
                    n_side: 2,
                    lanes: 1,
                },
            ),
            (PROG_ID.0, prog_layout),
        ]);

        let shout = RiscvShoutTables::new(self.xlen);
        let mut shout_ops = match &self.shout_ops {
            Some(ops) => ops.clone(),
            None if self.shout_auto_minimal => infer_required_shout_opcodes(&program),
            None => all_shout_opcodes(),
        };
        shout_ops.insert(RiscvOpcode::Add);

        let mut table_specs: HashMap<u32, LutTableSpec> = HashMap::new();
        for op in &shout_ops {
            let table_id = shout.opcode_to_id(*op).0;
            table_specs.insert(
                table_id,
                LutTableSpec::RiscvOpcode {
                    opcode: *op,
                    xlen: self.xlen,
                },
            );
        }
        let mut shout_table_ids: Vec<u32> = table_specs.keys().copied().collect();
        shout_table_ids.sort_unstable();

        let (ccs_base, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, self.chunk_size)
            .map_err(|e| PiCcsError::InvalidInput(format!("build_rv32_b1_step_ccs failed: {e}")))?;

        // Apply shared-bus wiring (adds Twist/Shout constraint matrices to the CCS).
        // This uses ROM-only initial_mem since the CCS structure doesn't depend on
        // witness-specific ram_init values.
        let session = FoldingSession::<AjtaiSModule>::new_ajtai(self.mode.clone(), &ccs_base)?;
        let params = session.params().clone();
        let committer = session.committer().clone();
        let empty_tables: HashMap<u32, LutTable<F>> = HashMap::new();

        let mut cpu = R1csCpu::new(
            ccs_base,
            params,
            committer,
            layout.m_in,
            &empty_tables,
            &table_specs,
            rv32_b1_chunk_to_witness(layout.clone()),
        );
        cpu = cpu
            .with_shared_cpu_bus(
                rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem)
                    .map_err(|e| PiCcsError::InvalidInput(format!("rv32_b1_shared_cpu_bus_config failed: {e}")))?,
                self.chunk_size,
            )
            .map_err(|e| PiCcsError::InvalidInput(format!("shared bus inject failed: {e}")))?;

        let sparse = Arc::new(SparseCache::build(&cpu.ccs));

        Ok(Rv32B1CcsCache { sparse })
    }

    /// Build only the verification context (CCS + session) without executing the program or proving.
    ///
    /// This performs the same validation, program decoding, memory layout setup, CCS construction
    /// (including shared-bus wiring), and session creation that `prove()` does -- but stops before
    /// any RISC-V execution or folding.
    ///
    /// The returned [`Rv32B1Verifier`] can verify a `ShardProof` given the public MCS instances
    /// (`mcss_public`) that were produced by the prover.
    ///
    /// **Cost:** circuit synthesis only (~ms).  No RISC-V execution, no folding.
    pub fn build_verifier(self) -> Result<Rv32B1Verifier, PiCcsError> {
        // --- Input validation (same as prove) ---
        if self.xlen != 32 {
            return Err(PiCcsError::InvalidInput(format!(
                "RV32 B1 MVP requires xlen == 32 (got {})",
                self.xlen
            )));
        }
        if self.program_bytes.is_empty() {
            return Err(PiCcsError::InvalidInput("program_bytes must be non-empty".into()));
        }
        if self.chunk_size == 0 {
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

        // --- Program decoding + memory layouts (same as prove, minus VM/Twist init) ---
        let program = decode_program(&self.program_bytes)
            .map_err(|e| PiCcsError::InvalidInput(format!("decode_program failed: {e}")))?;

        let (prog_layout, initial_mem) = neo_memory::riscv::rom_init::prog_rom_layout_and_init_words(
            PROG_ID,
            /*base_addr=*/ 0,
            &self.program_bytes,
        )
        .map_err(|e| PiCcsError::InvalidInput(format!("prog_rom_layout_and_init_words failed: {e}")))?;
        let mut initial_mem = initial_mem;
        for (&addr, &value) in &self.ram_init {
            let value = value as u32 as u64;
            initial_mem.insert((neo_memory::riscv::lookups::RAM_ID.0, addr), F::from_u64(value));
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
            (PROG_ID.0, prog_layout),
        ]);

        // --- Shout tables (same as prove) ---
        let mut shout_ops = match &self.shout_ops {
            Some(ops) => ops.clone(),
            None if self.shout_auto_minimal => infer_required_shout_opcodes(&program),
            None => all_shout_opcodes(),
        };
        shout_ops.insert(RiscvOpcode::Add);

        let shout = RiscvShoutTables::new(self.xlen);
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

        // --- CCS + Session (same as prove) ---
        let (ccs_base, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, self.chunk_size)
            .map_err(|e| PiCcsError::InvalidInput(format!("build_rv32_b1_step_ccs failed: {e}")))?;

        let mut session = FoldingSession::<AjtaiSModule>::new_ajtai(self.mode.clone(), &ccs_base)?;
        let params = session.params().clone();
        let committer = session.committer().clone();

        let empty_tables: HashMap<u32, LutTable<F>> = HashMap::new();

        // Build R1csCpu for CCS with shared-bus wiring (same as prove; no execution).
        let mut cpu = R1csCpu::new(
            ccs_base,
            params,
            committer,
            layout.m_in,
            &empty_tables,
            &table_specs,
            rv32_b1_chunk_to_witness(layout.clone()),
        );
        cpu = cpu
            .with_shared_cpu_bus(
                rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts.clone(), initial_mem.clone())
                    .map_err(|e| PiCcsError::InvalidInput(format!("rv32_b1_shared_cpu_bus_config failed: {e}")))?,
                self.chunk_size,
            )
            .map_err(|e| PiCcsError::InvalidInput(format!("shared bus inject failed: {e}")))?;

        session.set_step_linking(rv32_b1_step_linking_config(&layout));

        let ccs = cpu.ccs.clone();

        // No execution, no fold_and_prove -- just the verification context.
        // NOTE: verifier SparseCache preloading is deferred to `preload_sparse_cache()`
        // because the CCS pointer changes when moved into the Rv32B1Verifier struct.
        Ok(Rv32B1Verifier {
            session,
            ccs,
            _layout: layout,
            mem_layouts,
            statement_initial_mem: initial_mem,
            ram_num_bits: d_ram,
            output_claims: self.output_claims,
        })
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
        if self.chunk_size == 0 {
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

        // === Phase 1: Decode + setup ===
        let phase_start = time_now();

        let program = decode_program(&self.program_bytes)
            .map_err(|e| PiCcsError::InvalidInput(format!("decode_program failed: {e}")))?;
        let using_default_max_steps = self.max_steps.is_none();
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

        let t_decode_and_setup = elapsed_duration(phase_start);

        // === Phase 2: CCS construction + shared-bus wiring ===
        let phase_start = time_now();

        let prebuilt_sparse = self.ccs_cache.as_ref().map(|c| c.sparse.clone());

        let (ccs_base, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, self.chunk_size)
            .map_err(|e| PiCcsError::InvalidInput(format!("build_rv32_b1_step_ccs failed: {e}")))?;

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
            committer,
            layout.m_in,
            &empty_tables,
            &table_specs,
            rv32_b1_chunk_to_witness(layout.clone()),
        );
        cpu = cpu
            .with_shared_cpu_bus(
                rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts.clone(), initial_mem.clone())
                    .map_err(|e| PiCcsError::InvalidInput(format!("rv32_b1_shared_cpu_bus_config failed: {e}")))?,
                self.chunk_size,
            )
            .map_err(|e| PiCcsError::InvalidInput(format!("shared bus inject failed: {e}")))?;

        // Always enforce step-to-step chunk chaining for RV32 B1.
        session.set_step_linking(rv32_b1_step_linking_config(&layout));

        let t_ccs_and_shared_bus = elapsed_duration(phase_start);

        // === Phase 3: VM execution + shard collection ===
        let phase_start = time_now();

        // Execute + collect step bundles (and aux for output binding).
        session.execute_shard_shared_cpu_bus(
            vm,
            twist,
            shout,
            /*max_steps=*/ max_steps,
            self.chunk_size,
            &mem_layouts,
            &empty_tables,
            &table_specs,
            &lut_lanes,
            &initial_mem,
            &cpu,
        )?;
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

        let ccs = cpu.ccs.clone();

        // Preload the SparseCache using the *final* CCS reference that fold_and_prove will use.
        // (Must happen after cpu.ccs.clone() because the pointer-keyed cache checks identity.)
        if let Some(ref sparse) = prebuilt_sparse {
            session.preload_ccs_sparse_cache(&ccs, sparse.clone())?;
        }

        let t_vm_execution = elapsed_duration(phase_start);

        // === Phase 4: Fold-and-prove ===
        let phase_start = time_now();
        let proof = if self.output_claims.is_empty() {
            session.fold_and_prove(&ccs)?
        } else {
            let ob_cfg = OutputBindingConfig::new(d_ram, self.output_claims.clone());
            session.fold_and_prove_with_output_binding_auto_simple(&ccs, &ob_cfg)?
        };
        let t_fold_and_prove = elapsed_duration(phase_start);

        let prove_duration = t_decode_and_setup + t_ccs_and_shared_bus + t_vm_execution + t_fold_and_prove;

        Ok(Rv32B1Run {
            session,
            proof,
            ccs,
            layout,
            mem_layouts,
            initial_mem,
            ram_num_bits: d_ram,
            output_claims: self.output_claims,
            prove_duration,
            prove_timings: ProveTimings {
                decode_and_setup: t_decode_and_setup,
                ccs_and_shared_bus: t_ccs_and_shared_bus,
                vm_execution: t_vm_execution,
                fold_and_prove: t_fold_and_prove,
            },
            verify_duration: None,
        })
    }
}

/// Verification context for RV32 B1 proofs.
///
/// Created by [`Rv32B1::build_verifier`].  Contains the CCS structure and folding session
/// needed to verify a `ShardProof` without executing the RISC-V program.
pub struct Rv32B1Verifier {
    session: FoldingSession<AjtaiSModule>,
    ccs: CcsStructure<F>,
    _layout: Rv32B1Layout,
    mem_layouts: HashMap<u32, PlainMemLayout>,
    statement_initial_mem: HashMap<(u32, u64), F>,
    ram_num_bits: usize,
    output_claims: ProgramIO<F>,
}

impl Rv32B1Verifier {
    /// Preload the verifier SparseCache using `&self.ccs` (final pointer).
    ///
    /// Call this once after `build_verifier()` returns, before the first `verify()`.
    /// This ensures the pointer-keyed cache hits inside the session's verify path.
    pub fn preload_sparse_cache(&mut self, sparse: Arc<SparseCache<F>>) -> Result<(), PiCcsError> {
        self.session
            .preload_verifier_ccs_sparse_cache(&self.ccs, sparse)
    }

    /// Verify a `ShardProof` using the provided public step instance bundles.
    ///
    /// `steps_public` must be the step instance bundles produced by the prover (via
    /// `Rv32B1Run::steps_public()`).  The verifier checks the folding proof
    /// against these instances and the CCS structure -- no RISC-V execution
    /// is performed.
    pub fn verify(
        &self,
        proof: &ShardProof,
        steps_public: &[StepInstanceBundle<Cmt, F, K>],
    ) -> Result<bool, PiCcsError> {
        rv32_b1_enforce_chunk0_mem_init_matches_statement(
            &self.mem_layouts,
            &self.statement_initial_mem,
            steps_public,
        )?;

        if self.output_claims.is_empty() {
            self.session
                .verify_with_external_steps(&self.ccs, steps_public, proof)
        } else {
            let ob_cfg = OutputBindingConfig::new(self.ram_num_bits, self.output_claims.clone());
            self.session
                .verify_with_external_steps_and_output_binding(&self.ccs, steps_public, proof, &ob_cfg)
        }
    }
}

pub struct Rv32B1Run {
    session: FoldingSession<AjtaiSModule>,
    proof: ShardProof,
    ccs: CcsStructure<F>,
    layout: Rv32B1Layout,
    mem_layouts: HashMap<u32, PlainMemLayout>,
    initial_mem: HashMap<(u32, u64), F>,
    ram_num_bits: usize,
    output_claims: ProgramIO<F>,
    prove_duration: Duration,
    prove_timings: ProveTimings,
    verify_duration: Option<Duration>,
}

impl Rv32B1Run {
    pub fn params(&self) -> &NeoParams {
        self.session.params()
    }

    pub fn ccs(&self) -> &CcsStructure<F> {
        &self.ccs
    }

    pub fn verify(&mut self) -> Result<(), PiCcsError> {
        let verify_start = time_now();
        let ok = if self.output_claims.is_empty() {
            self.session.verify_collected(&self.ccs, &self.proof)?
        } else {
            let ob_cfg = OutputBindingConfig::new(self.ram_num_bits, self.output_claims.clone());
            self.session
                .verify_with_output_binding_collected_simple(&self.ccs, &self.proof, &ob_cfg)?
        };
        self.verify_duration = Some(elapsed_duration(verify_start));

        if !ok {
            return Err(PiCcsError::ProtocolError("verification failed".into()));
        }
        Ok(())
    }

    pub fn proof(&self) -> &ShardProof {
        &self.proof
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

    /// Return the public MCS instances for inclusion in a proof package.
    ///
    /// These instances must be transmitted alongside the `ShardProof` so that
    /// a standalone verifier (via [`Rv32B1Verifier::verify`]) can check the proof
    /// without re-executing the RISC-V program.
    pub fn mcss_public(&self) -> Vec<neo_ccs::McsInstance<Cmt, F>> {
        self.session.mcss_public()
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
        let ob_cfg = simple_output_config(self.ram_num_bits, output_addr, expected_output);
        self.session
            .verify_with_output_binding_collected_simple(&self.ccs, &self.proof, &ob_cfg)
    }

    pub fn verify_default_output_claim(&self) -> Result<bool, PiCcsError> {
        if self.output_claims.is_empty() {
            return Err(PiCcsError::InvalidInput("no output claim configured".into()));
        };
        let ob_cfg = OutputBindingConfig::new(self.ram_num_bits, self.output_claims.clone());
        self.session
            .verify_with_output_binding_collected_simple(&self.ccs, &self.proof, &ob_cfg)
    }

    pub fn verify_output_claims(&self, output_claims: ProgramIO<F>) -> Result<bool, PiCcsError> {
        if output_claims.is_empty() {
            return Err(PiCcsError::InvalidInput("output_claims must be non-empty".into()));
        }
        let ob_cfg = OutputBindingConfig::new(self.ram_num_bits, output_claims);
        self.session
            .verify_with_output_binding_collected_simple(&self.ccs, &self.proof, &ob_cfg)
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
        self.proof.steps.len()
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

    /// Per-phase timing breakdown for the prove call.
    pub fn prove_timings(&self) -> &ProveTimings {
        &self.prove_timings
    }

    pub fn verify_duration(&self) -> Option<Duration> {
        self.verify_duration
    }
}
