//! Convenience runner for RV32 trace-wiring CCS (time-in-rows).
//!
//! This is an ergonomic wrapper around the existing trace wiring artifacts:
//! - `neo_memory::riscv::trace` for execution-table extraction, and
//! - `neo_memory::riscv::ccs::trace` for fixed-width trace wiring CCS.
//!
//! The runner intentionally targets the current Tier 2.1 scope:
//! - fixed-width trace-wiring CCS steps with PROG/REG/RAM sidecar instances,
//! - no decode/semantics sidecar proofs in this wrapper yet.

#![allow(non_snake_case)]

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::time::Duration;

use crate::output_binding::OutputBindingConfig;
use crate::pi_ccs::FoldingMode;
use crate::session::FoldingSession;
use crate::shard::{ShardProof, StepLinkingConfig};
use crate::PiCcsError;
use neo_ajtai::AjtaiSModule;
use neo_ccs::relations::{McsInstance, McsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::CcsStructure;
use neo_math::{F, K};
use neo_memory::cpu::build_bus_layout_for_instances_with_shout_and_twist_lanes;
use neo_memory::output_check::ProgramIO;
use neo_memory::plain::{LutTable, PlainMemLayout};
use neo_memory::riscv::ccs::{
    build_rv32_trace_wiring_ccs, build_rv32_trace_wiring_ccs_with_reserved_rows,
    rv32_trace_ccs_witness_from_exec_table, rv32_trace_shared_bus_requirements, rv32_trace_shared_cpu_bus_config,
    Rv32TraceCcsLayout,
};
use neo_memory::riscv::exec_table::{Rv32ExecRow, Rv32ExecTable};
use neo_memory::riscv::lookups::{
    decode_program, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode, RiscvShoutTables, PROG_ID, RAM_ID, REG_ID,
};
use neo_memory::riscv::rom_init::prog_rom_layout_and_init_words;
use neo_memory::riscv::trace::{extract_twist_lanes_over_time, TwistLaneOverTime};
use neo_memory::witness::{LutInstance, LutWitness, MemInstance, MemWitness, StepWitnessBundle};
use neo_memory::{LutTableSpec, MemInit, R1csCpu};
use neo_params::NeoParams;
use neo_vm_trace::{StepTrace, Twist as _, TwistOpKind};
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

/// Hard instruction cap for trace-wiring mode (Option C).
const DEFAULT_RV32_TRACE_MAX_STEPS: usize = 1 << 20;

/// Default per-step trace rows for trace-mode IVC.
///
/// The full trace is split into fixed-size chunks of this row count (except when the whole
/// trace is smaller), and those chunks are folded with step-linking.
const DEFAULT_RV32_TRACE_CHUNK_ROWS: usize = 1 << 16;

fn max_ram_addr_from_exec(exec: &Rv32ExecTable) -> Option<u64> {
    exec.rows
        .iter()
        .filter(|r| r.active)
        .flat_map(|r| r.ram_events.iter().map(|e| e.addr))
        .max()
}

fn required_bits_for_max_addr(max_addr: u64) -> usize {
    if max_addr == 0 {
        1
    } else {
        (u64::BITS - max_addr.leading_zeros()) as usize
    }
}

fn build_twist_only_bus_z(
    m: usize,
    m_in: usize,
    t: usize,
    ell_addr: usize,
    lanes: usize,
    lane_data: &[TwistLaneOverTime],
    x_prefix: &[F],
) -> Result<Vec<F>, String> {
    if x_prefix.len() != m_in {
        return Err(format!(
            "build_twist_only_bus_z: x_prefix.len()={} != m_in={}",
            x_prefix.len(),
            m_in
        ));
    }
    if lane_data.len() != lanes {
        return Err(format!(
            "build_twist_only_bus_z: lane_data.len()={} != lanes={}",
            lane_data.len(),
            lanes
        ));
    }

    let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
        m,
        m_in,
        t,
        core::iter::empty::<(usize, usize)>(),
        core::iter::once((ell_addr, lanes)),
    )?;
    if bus.twist_cols.len() != 1 || !bus.shout_cols.is_empty() {
        return Err("build_twist_only_bus_z: expected 1 twist instance and 0 shout instances".into());
    }

    let mut z = vec![F::ZERO; m];
    z[..m_in].copy_from_slice(x_prefix);

    let twist = &bus.twist_cols[0];
    for (lane_idx, cols) in twist.lanes.iter().enumerate() {
        let lane = &lane_data[lane_idx];
        if lane.has_read.len() != t || lane.has_write.len() != t {
            return Err("build_twist_only_bus_z: lane length mismatch".into());
        }
        for j in 0..t {
            let has_r = lane.has_read[j];
            let has_w = lane.has_write[j];

            z[bus.bus_cell(cols.has_read, j)] = if has_r { F::ONE } else { F::ZERO };
            z[bus.bus_cell(cols.has_write, j)] = if has_w { F::ONE } else { F::ZERO };

            z[bus.bus_cell(cols.rv, j)] = if has_r { F::from_u64(lane.rv[j]) } else { F::ZERO };
            z[bus.bus_cell(cols.wv, j)] = if has_w { F::from_u64(lane.wv[j]) } else { F::ZERO };
            z[bus.bus_cell(cols.inc, j)] = if has_w { lane.inc_at_write_addr[j] } else { F::ZERO };

            for (bit_idx, col_id) in cols.ra_bits.clone().enumerate() {
                let bit_is_set = bit_idx < (u64::BITS as usize) && ((lane.ra[j] >> bit_idx) & 1) == 1;
                z[bus.bus_cell(col_id, j)] = if bit_is_set { F::ONE } else { F::ZERO };
            }
            for (bit_idx, col_id) in cols.wa_bits.clone().enumerate() {
                let bit_is_set = bit_idx < (u64::BITS as usize) && ((lane.wa[j] >> bit_idx) & 1) == 1;
                z[bus.bus_cell(col_id, j)] = if bit_is_set { F::ONE } else { F::ZERO };
            }
        }
    }

    Ok(z)
}

fn mem_init_from_u64_sparse(sparse: &HashMap<u64, u64>, k: usize, label: &str) -> Result<MemInit<F>, PiCcsError> {
    let mut pairs = Vec::<(u64, F)>::new();
    for (&addr, &value) in sparse {
        let addr_usize = usize::try_from(addr)
            .map_err(|_| PiCcsError::InvalidInput(format!("{label} init addr does not fit usize: addr={addr}")))?;
        if addr_usize >= k {
            return Err(PiCcsError::InvalidInput(format!(
                "{label} init addr out of range: addr={addr} >= k={k}"
            )));
        }
        if value != 0 {
            pairs.push((addr, F::from_u64(value)));
        }
    }
    pairs.sort_by_key(|(addr, _)| *addr);
    Ok(if pairs.is_empty() {
        MemInit::Zero
    } else {
        MemInit::Sparse(pairs)
    })
}

fn final_reg_state_dense(exec: &Rv32ExecTable, reg_init: &HashMap<u64, u64>) -> Result<Vec<F>, PiCcsError> {
    let mut regs = [0u64; 32];
    for (&reg, &value) in reg_init {
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
        regs[reg as usize] = value as u32 as u64;
    }
    regs[0] = 0;

    for r in exec.rows.iter().filter(|r| r.active) {
        if let Some(w) = &r.reg_write_lane0 {
            if w.addr >= 32 {
                return Err(PiCcsError::InvalidInput(format!(
                    "trace register write addr out of range at cycle {}: addr={}",
                    r.cycle, w.addr
                )));
            }
            if w.addr == 0 {
                return Err(PiCcsError::InvalidInput(format!(
                    "trace writes x0 at cycle {} which is invalid",
                    r.cycle
                )));
            }
            regs[w.addr as usize] = w.value as u32 as u64;
            regs[0] = 0;
        }
    }

    Ok(regs.iter().map(|&v| F::from_u64(v)).collect())
}

fn final_ram_state_dense(exec: &Rv32ExecTable, ram_init: &HashMap<u64, u64>, k: usize) -> Result<Vec<F>, PiCcsError> {
    let mut out = vec![F::ZERO; k];
    for (&addr, &value) in ram_init {
        let addr_usize = usize::try_from(addr)
            .map_err(|_| PiCcsError::InvalidInput(format!("ram_init_u32: addr does not fit usize: addr={addr}")))?;
        if addr_usize >= k {
            return Err(PiCcsError::InvalidInput(format!(
                "ram_init_u32: addr out of range for output binding domain: addr={addr} >= k={k}"
            )));
        }
        out[addr_usize] = F::from_u64(value as u32 as u64);
    }

    for r in exec.rows.iter().filter(|r| r.active) {
        for e in &r.ram_events {
            if e.kind != TwistOpKind::Write {
                continue;
            }
            let addr_usize = usize::try_from(e.addr).map_err(|_| {
                PiCcsError::InvalidInput(format!(
                    "trace RAM write addr does not fit usize at cycle {}: addr={}",
                    r.cycle, e.addr
                ))
            })?;
            if addr_usize >= k {
                return Err(PiCcsError::InvalidInput(format!(
                    "trace RAM write addr out of range for output binding domain at cycle {}: addr={} >= k={k}",
                    r.cycle, e.addr
                )));
            }
            out[addr_usize] = F::from_u64(e.value as u32 as u64);
        }
    }

    Ok(out)
}

fn init_reg_state(reg_init: &HashMap<u64, u64>) -> Result<[u64; 32], PiCcsError> {
    let mut regs = [0u64; 32];
    for (&reg, &value) in reg_init {
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
        regs[reg as usize] = value as u32 as u64;
    }
    regs[0] = 0;
    Ok(regs)
}

fn init_ram_state(ram_init: &HashMap<u64, u64>, ram_ell_addr: usize) -> Result<HashMap<u64, u64>, PiCcsError> {
    if ram_ell_addr > 64 {
        return Err(PiCcsError::InvalidInput(format!(
            "RAM ell_addr too large for u64 addressing: ell_addr={ram_ell_addr}"
        )));
    }

    let mut ram = HashMap::<u64, u64>::new();
    for (&addr, &value) in ram_init {
        if ram_ell_addr < 64 && (addr >> ram_ell_addr) != 0 {
            return Err(PiCcsError::InvalidInput(format!(
                "RAM init addr out of range for ell_addr={ram_ell_addr}: addr={addr}"
            )));
        }
        let v = value as u32 as u64;
        if v != 0 {
            ram.insert(addr, v);
        }
    }
    Ok(ram)
}

fn reg_state_to_sparse_map(regs: &[u64; 32]) -> HashMap<u64, u64> {
    let mut out = HashMap::<u64, u64>::new();
    for (idx, &value) in regs.iter().enumerate().skip(1) {
        if value != 0 {
            out.insert(idx as u64, value);
        }
    }
    out
}

fn apply_exec_chunk_writes_to_state(
    chunk: &Rv32ExecTable,
    regs: &mut [u64; 32],
    ram: &mut HashMap<u64, u64>,
) -> Result<(), PiCcsError> {
    for r in chunk.rows.iter().filter(|r| r.active) {
        if let Some(w) = &r.reg_write_lane0 {
            if w.addr == 0 {
                return Err(PiCcsError::InvalidInput(format!(
                    "trace writes x0 at cycle {} which is invalid",
                    r.cycle
                )));
            }
            if w.addr >= 32 {
                return Err(PiCcsError::InvalidInput(format!(
                    "trace register write addr out of range at cycle {}: addr={}",
                    r.cycle, w.addr
                )));
            }
            regs[w.addr as usize] = w.value as u32 as u64;
            regs[0] = 0;
        }

        for e in &r.ram_events {
            if e.kind != TwistOpKind::Write {
                continue;
            }
            let value = e.value as u32 as u64;
            if value == 0 {
                ram.remove(&e.addr);
            } else {
                ram.insert(e.addr, value);
            }
        }
    }
    Ok(())
}

fn split_exec_into_fixed_chunks(exec: &Rv32ExecTable, chunk_rows: usize) -> Result<Vec<Rv32ExecTable>, PiCcsError> {
    if chunk_rows == 0 {
        return Err(PiCcsError::InvalidInput("trace chunk_rows must be non-zero".into()));
    }
    if exec.rows.is_empty() {
        return Err(PiCcsError::InvalidInput("trace execution table is empty".into()));
    }

    if exec.rows.len() <= chunk_rows {
        return Ok(vec![exec.clone()]);
    }

    let mut out = Vec::<Rv32ExecTable>::new();
    let total = exec.rows.len();
    let mut start = 0usize;
    while start < total {
        let end = (start + chunk_rows).min(total);
        let mut rows = exec.rows[start..end].to_vec();
        if rows.len() < chunk_rows {
            let last = rows
                .last()
                .ok_or_else(|| PiCcsError::InvalidInput("trace chunk unexpectedly empty".into()))?
                .clone();
            let mut cycle = last.cycle;
            let pad_pc = last.pc_after;
            let pad_halted = last.halted;
            while rows.len() < chunk_rows {
                cycle = cycle
                    .checked_add(1)
                    .ok_or_else(|| PiCcsError::InvalidInput("cycle overflow while chunk-padding trace".into()))?;
                rows.push(neo_memory::riscv::exec_table::Rv32ExecRow::inactive(
                    cycle, pad_pc, pad_halted,
                ));
            }
        }
        out.push(Rv32ExecTable { rows });
        start = end;
    }

    Ok(out)
}

fn rv32_trace_chunk_to_witness(
    layout: Rv32TraceCcsLayout,
) -> Box<dyn Fn(&[StepTrace<u64, u64>]) -> Vec<F> + Send + Sync> {
    Box::new(move |chunk: &[StepTrace<u64, u64>]| {
        rv32_trace_chunk_to_witness_checked(&layout, chunk)
            .unwrap_or_else(|e| panic!("rv32_trace_chunk_to_witness failed for chunk_len={}: {e}", chunk.len()))
    })
}

fn rv32_trace_chunk_to_witness_checked(
    layout: &Rv32TraceCcsLayout,
    chunk: &[StepTrace<u64, u64>],
) -> Result<Vec<F>, String> {
    if chunk.is_empty() {
        return Err("trace chunk witness: chunk must contain at least one step".into());
    }
    if chunk.len() > layout.t {
        return Err(format!(
            "trace chunk witness: chunk.len()={} exceeds layout.t={}",
            chunk.len(),
            layout.t
        ));
    }

    let mut rows = Vec::with_capacity(layout.t);
    for step in chunk {
        rows.push(Rv32ExecRow::from_step(step)?);
    }

    let mut cycle = rows
        .last()
        .ok_or_else(|| "trace chunk witness: empty rows after conversion".to_string())?
        .cycle;
    let pad_pc = rows.last().expect("rows non-empty").pc_after;
    let pad_halted = rows.last().expect("rows non-empty").halted;
    while rows.len() < layout.t {
        cycle = cycle
            .checked_add(1)
            .ok_or_else(|| "trace chunk witness: cycle overflow while padding".to_string())?;
        rows.push(Rv32ExecRow::inactive(cycle, pad_pc, pad_halted));
    }

    let exec = Rv32ExecTable { rows };
    let (x, w) = rv32_trace_ccs_witness_from_exec_table(layout, &exec)?;
    Ok(x.into_iter().chain(w).collect())
}

fn infer_required_trace_shout_opcodes(program: &[RiscvInstruction]) -> HashSet<RiscvOpcode> {
    let mut ops = HashSet::new();
    // Required for shared wiring (address/PC arithmetic).
    ops.insert(RiscvOpcode::Add);
    for instr in program {
        match instr {
            RiscvInstruction::RAlu { op, .. } | RiscvInstruction::IAlu { op, .. } => {
                ops.insert(*op);
            }
            RiscvInstruction::Branch { cond, .. } => {
                ops.insert(cond.to_shout_opcode());
            }
            // Address arithmetic in these classes uses ADD shout semantics.
            RiscvInstruction::Load { .. }
            | RiscvInstruction::Store { .. }
            | RiscvInstruction::Jalr { .. }
            | RiscvInstruction::Auipc { .. } => {
                ops.insert(RiscvOpcode::Add);
            }
            _ => {}
        }
    }
    ops
}

fn rv32_trace_table_specs(program: &[RiscvInstruction]) -> HashMap<u32, LutTableSpec> {
    let shout = RiscvShoutTables::new(32);
    let mut table_specs = HashMap::new();
    for op in infer_required_trace_shout_opcodes(program) {
        let table_id = shout.opcode_to_id(op).0;
        table_specs.insert(table_id, LutTableSpec::RiscvOpcode { opcode: op, xlen: 32 });
    }
    table_specs
}

/// High-level builder for proving/verifying the RV32 trace wiring CCS.
///
/// This path is intentionally narrow:
/// - builds a padded execution table,
/// - proves one or more trace-wiring CCS steps (IVC),
/// - verifies the resulting shard proof.
#[derive(Clone, Copy, Debug, Default)]
enum OutputTarget {
    #[default]
    Ram,
    Reg,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Rv32TraceProvePhaseDurations {
    pub setup: Duration,
    pub chunk_build_commit: Duration,
    pub fold_and_prove: Duration,
}

#[derive(Clone, Debug)]
pub struct Rv32TraceWiring {
    program_base: u64,
    program_bytes: Vec<u8>,
    xlen: usize,
    max_steps: Option<usize>,
    min_trace_len: usize,
    chunk_rows: Option<usize>,
    shared_cpu_bus: bool,
    mode: FoldingMode,
    ram_init: HashMap<u64, u64>,
    reg_init: HashMap<u64, u64>,
    output_claims: ProgramIO<F>,
    output_target: OutputTarget,
}

impl Rv32TraceWiring {
    /// Create a trace runner from ROM bytes.
    pub fn from_rom(program_base: u64, program_bytes: &[u8]) -> Self {
        Self {
            program_base,
            program_bytes: program_bytes.to_vec(),
            xlen: 32,
            max_steps: None,
            min_trace_len: 4,
            chunk_rows: None,
            shared_cpu_bus: true,
            mode: FoldingMode::Optimized,
            ram_init: HashMap::new(),
            reg_init: HashMap::new(),
            output_claims: ProgramIO::new(),
            output_target: OutputTarget::Ram,
        }
    }

    pub fn xlen(mut self, xlen: usize) -> Self {
        self.xlen = xlen;
        self
    }

    /// Lower-bound for execution-table length.
    ///
    /// Final `t` is `max(trace_len, min_trace_len)`.
    pub fn min_trace_len(mut self, min_trace_len: usize) -> Self {
        self.min_trace_len = min_trace_len.max(1);
        self
    }

    /// Fixed rows per trace step for IVC folding.
    ///
    /// The trace is split into fixed-size chunks, each chunk is proven with the same step CCS,
    /// and step-linking enforces `pc_final -> pc0`.
    pub fn chunk_rows(mut self, chunk_rows: usize) -> Self {
        self.chunk_rows = Some(chunk_rows);
        self
    }

    /// Toggle shared-CPU-bus trace proving mode.
    ///
    /// `true` is the intended production default; `false` keeps the legacy no-shared-bus path.
    pub fn shared_cpu_bus(mut self, enabled: bool) -> Self {
        self.shared_cpu_bus = enabled;
        self
    }

    /// Bound executed instruction count.
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    pub fn mode(mut self, mode: FoldingMode) -> Self {
        self.mode = mode;
        self
    }

    /// Initialize RAM byte-addressed word cell to a u32 value.
    pub fn ram_init_u32(mut self, addr: u64, value: u32) -> Self {
        self.ram_init.insert(addr, value as u64);
        self
    }

    /// Initialize register `reg` (x0..x31) to a u32 value.
    pub fn reg_init_u32(mut self, reg: u64, value: u32) -> Self {
        self.reg_init.insert(reg, value as u64);
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

    pub fn prove(self) -> Result<Rv32TraceWiringRun, PiCcsError> {
        if self.xlen != 32 {
            return Err(PiCcsError::InvalidInput(format!(
                "RV32 trace wiring runner requires xlen == 32 (got {})",
                self.xlen
            )));
        }
        if self.program_base != 0 {
            return Err(PiCcsError::InvalidInput(
                "RV32 trace wiring runner requires program_base == 0".into(),
            ));
        }
        if self.program_bytes.is_empty() {
            return Err(PiCcsError::InvalidInput("program_bytes must be non-empty".into()));
        }
        if self.min_trace_len > DEFAULT_RV32_TRACE_MAX_STEPS {
            return Err(PiCcsError::InvalidInput(format!(
                "min_trace_len={} exceeds trace-mode hard cap {}. Use the chunked RV32B1 runner for longer executions.",
                self.min_trace_len, DEFAULT_RV32_TRACE_MAX_STEPS
            )));
        }
        if self.program_bytes.len() % 4 != 0 {
            return Err(PiCcsError::InvalidInput(
                "program_bytes must be 4-byte aligned (RVC is not supported)".into(),
            ));
        }
        for (i, chunk) in self.program_bytes.chunks_exact(4).enumerate() {
            let first_half = u16::from_le_bytes([chunk[0], chunk[1]]);
            if (first_half & 0b11) != 0b11 {
                return Err(PiCcsError::InvalidInput(format!(
                    "compressed instruction encoding (RVC) is not supported at word index {i}"
                )));
            }
        }

        let program = decode_program(&self.program_bytes)
            .map_err(|e| PiCcsError::InvalidInput(format!("decode_program failed: {e}")))?;
        let using_default_max_steps = self.max_steps.is_none();
        let max_steps = match self.max_steps {
            Some(n) => {
                if n == 0 {
                    return Err(PiCcsError::InvalidInput("max_steps must be non-zero".into()));
                }
                if n > DEFAULT_RV32_TRACE_MAX_STEPS {
                    return Err(PiCcsError::InvalidInput(format!(
                        "max_steps={} exceeds trace-mode hard cap {}. Use the chunked RV32B1 runner for longer executions.",
                        n, DEFAULT_RV32_TRACE_MAX_STEPS
                    )));
                }
                n
            }
            None => DEFAULT_RV32_TRACE_MAX_STEPS,
        };
        let ram_init_map = self.ram_init.clone();
        let reg_init_map = self.reg_init.clone();
        let output_claims = self.output_claims.clone();
        let output_target = self.output_target;

        let mut vm = RiscvCpu::new(self.xlen);
        vm.load_program(/*base=*/ 0, program.clone());

        let mut twist =
            RiscvMemory::with_program_in_twist(self.xlen, PROG_ID, /*base_addr=*/ 0, &self.program_bytes);
        for (&addr, &value) in &ram_init_map {
            twist.store(RAM_ID, addr, value as u32 as u64);
        }
        for (&reg, &value) in &reg_init_map {
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
            twist.store(REG_ID, reg, value as u32 as u64);
        }
        let shout = RiscvShoutTables::new(self.xlen);

        let trace = neo_vm_trace::trace_program(vm, twist, shout, max_steps)
            .map_err(|e| PiCcsError::InvalidInput(format!("trace_program failed: {e}")))?;

        if using_default_max_steps && !trace.did_halt() {
            return Err(PiCcsError::InvalidInput(format!(
                "RV32 execution did not halt within max_steps={max_steps}; call .max_steps(...) to raise the limit or ensure the guest halts"
            )));
        }

        let target_len = trace.steps.len().max(self.min_trace_len);
        if target_len > DEFAULT_RV32_TRACE_MAX_STEPS {
            return Err(PiCcsError::InvalidInput(format!(
                "trace length {} exceeds trace-mode hard cap {}. Use the chunked RV32B1 runner for longer executions.",
                target_len, DEFAULT_RV32_TRACE_MAX_STEPS
            )));
        }
        let exec = Rv32ExecTable::from_trace_padded(&trace, target_len)
            .map_err(|e| PiCcsError::InvalidInput(format!("Rv32ExecTable::from_trace_padded failed: {e}")))?;
        exec.validate_cycle_chain()
            .map_err(|e| PiCcsError::InvalidInput(format!("validate_cycle_chain failed: {e}")))?;
        exec.validate_pc_chain()
            .map_err(|e| PiCcsError::InvalidInput(format!("validate_pc_chain failed: {e}")))?;
        exec.validate_halted_tail()
            .map_err(|e| PiCcsError::InvalidInput(format!("validate_halted_tail failed: {e}")))?;
        exec.validate_inactive_rows_are_empty()
            .map_err(|e| PiCcsError::InvalidInput(format!("validate_inactive_rows_are_empty failed: {e}")))?;

        let requested_chunk_rows = self.chunk_rows.unwrap_or(DEFAULT_RV32_TRACE_CHUNK_ROWS);
        if requested_chunk_rows == 0 {
            return Err(PiCcsError::InvalidInput("trace chunk_rows must be non-zero".into()));
        }
        let step_rows = requested_chunk_rows.min(exec.rows.len().max(1));
        let exec_chunks = split_exec_into_fixed_chunks(&exec, step_rows)?;

        let mut layout = Rv32TraceCcsLayout::new(step_rows)
            .map_err(|e| PiCcsError::InvalidInput(format!("Rv32TraceCcsLayout::new failed: {e}")))?;

        let prove_start = time_now();
        let setup_start = prove_start;
        let (prog_layout, prog_init_words) =
            prog_rom_layout_and_init_words::<F>(PROG_ID, /*base_addr=*/ 0, &self.program_bytes)
                .map_err(|e| PiCcsError::InvalidInput(format!("prog_rom_layout_and_init_words failed: {e}")))?;

        let mut max_ram_addr = max_ram_addr_from_exec(&exec).unwrap_or(0);
        if let Some(max_init_addr) = ram_init_map.keys().copied().max() {
            max_ram_addr = max_ram_addr.max(max_init_addr);
        }
        if matches!(output_target, OutputTarget::Ram) {
            if let Some(max_claim_addr) = output_claims.claimed_addresses().max() {
                max_ram_addr = max_ram_addr.max(max_claim_addr);
            }
        }
        let ram_d = required_bits_for_max_addr(max_ram_addr).max(2);
        let ram_k = 1usize
            .checked_shl(ram_d as u32)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("RAM address width too large: d={ram_d}")))?;

        let mem_layouts: HashMap<u32, PlainMemLayout> = HashMap::from([
            (
                RAM_ID.0,
                PlainMemLayout {
                    k: ram_k,
                    d: ram_d,
                    n_side: 2,
                    lanes: 1,
                },
            ),
            (
                REG_ID.0,
                PlainMemLayout {
                    k: 32,
                    d: 5,
                    n_side: 2,
                    lanes: 2,
                },
            ),
            (PROG_ID.0, prog_layout.clone()),
        ]);

        let table_specs = rv32_trace_table_specs(&program);
        let mut shout_table_ids: Vec<u32> = table_specs.keys().copied().collect();
        shout_table_ids.sort_unstable();

        let mut ccs_reserved_rows = 0usize;
        if self.shared_cpu_bus {
            let (bus_region_len, reserved_rows) =
                rv32_trace_shared_bus_requirements(&layout, &shout_table_ids, &mem_layouts)
                    .map_err(|e| PiCcsError::InvalidInput(format!("rv32_trace_shared_bus_requirements failed: {e}")))?;
            layout.m = layout
                .m
                .checked_add(bus_region_len)
                .ok_or_else(|| PiCcsError::InvalidInput("trace layout m overflow after bus tail reservation".into()))?;
            ccs_reserved_rows = reserved_rows;
        }

        let mut ccs = if ccs_reserved_rows == 0 {
            build_rv32_trace_wiring_ccs(&layout)
                .map_err(|e| PiCcsError::InvalidInput(format!("build_rv32_trace_wiring_ccs failed: {e}")))?
        } else {
            build_rv32_trace_wiring_ccs_with_reserved_rows(&layout, ccs_reserved_rows).map_err(|e| {
                PiCcsError::InvalidInput(format!("build_rv32_trace_wiring_ccs_with_reserved_rows failed: {e}"))
            })?
        };

        let mut session = FoldingSession::<AjtaiSModule>::new_ajtai(self.mode.clone(), &ccs)?;
        session.set_step_linking(StepLinkingConfig::new(vec![(layout.pc_final, layout.pc0)]));

        let mut prog_init_pairs: Vec<(u64, F)> = prog_init_words
            .into_iter()
            .filter_map(|((mem_id, addr), value)| (mem_id == PROG_ID.0 && value != F::ZERO).then_some((addr, value)))
            .collect();
        prog_init_pairs.sort_by_key(|(addr, _)| *addr);
        let prog_mem_init = if prog_init_pairs.is_empty() {
            MemInit::Zero
        } else {
            MemInit::Sparse(prog_init_pairs)
        };
        let mut initial_mem: HashMap<(u32, u64), F> = HashMap::new();
        if let MemInit::Sparse(pairs) = &prog_mem_init {
            for &(addr, value) in pairs {
                if value != F::ZERO {
                    initial_mem.insert((PROG_ID.0, addr), value);
                }
            }
        }
        for (&reg, &value) in &reg_init_map {
            let v = F::from_u64(value as u32 as u64);
            if v != F::ZERO {
                initial_mem.insert((REG_ID.0, reg), v);
            }
        }
        for (&addr, &value) in &ram_init_map {
            let v = F::from_u64(value as u32 as u64);
            if v != F::ZERO {
                initial_mem.insert((RAM_ID.0, addr), v);
            }
        }

        let setup_duration = elapsed_duration(setup_start);
        let mut chunk_build_commit_duration = Duration::ZERO;
        if self.shared_cpu_bus {
            let chunk_start = time_now();

            let empty_tables: HashMap<u32, LutTable<F>> = HashMap::new();
            let lut_lanes: HashMap<u32, usize> = HashMap::new();

            let mut cpu = R1csCpu::new(
                ccs.clone(),
                session.params().clone(),
                session.committer().clone(),
                layout.m_in,
                &empty_tables,
                &table_specs,
                rv32_trace_chunk_to_witness(layout.clone()),
            )
            .map_err(|e| PiCcsError::InvalidInput(format!("R1csCpu::new failed: {e}")))?;
            cpu = cpu
                .with_shared_cpu_bus(
                    rv32_trace_shared_cpu_bus_config(
                        &layout,
                        &shout_table_ids,
                        mem_layouts.clone(),
                        initial_mem.clone(),
                    )
                    .map_err(|e| PiCcsError::InvalidInput(format!("rv32_trace_shared_cpu_bus_config failed: {e}")))?,
                    layout.t,
                )
                .map_err(|e| PiCcsError::InvalidInput(format!("shared bus inject failed: {e}")))?;

            ccs = cpu.ccs.clone();

            session.execute_shard_shared_cpu_bus_from_trace(
                &trace,
                max_steps,
                layout.t,
                &mem_layouts,
                &empty_tables,
                &table_specs,
                &lut_lanes,
                &initial_mem,
                &cpu,
            )?;
            chunk_build_commit_duration += elapsed_duration(chunk_start);
        } else {
            // Route-A legacy fallback: keep the main CPU witness as pure trace columns (no bus tail),
            // and attach PROG/REG/RAM as separately committed no-shared-bus Twist instances linked at r_time.
            let mut reg_state = init_reg_state(&reg_init_map)?;
            let mut ram_state = init_ram_state(&ram_init_map, ram_d)?;
            for exec_chunk in &exec_chunks {
                let chunk_start = time_now();
                let reg_init_chunk = reg_state_to_sparse_map(&reg_state);
                let ram_init_chunk = ram_state.clone();

                let reg_mem_init = mem_init_from_u64_sparse(&reg_init_chunk, 32, "REG")?;
                let ram_mem_init = mem_init_from_u64_sparse(&ram_init_chunk, ram_k, "RAM")?;
                let twist_lanes = extract_twist_lanes_over_time(exec_chunk, &reg_init_chunk, &ram_init_chunk, ram_d)
                    .map_err(|e| PiCcsError::InvalidInput(format!("extract_twist_lanes_over_time failed: {e}")))?;

                let (x, w) = rv32_trace_ccs_witness_from_exec_table(&layout, exec_chunk).map_err(|e| {
                    PiCcsError::InvalidInput(format!("rv32_trace_ccs_witness_from_exec_table failed: {e}"))
                })?;
                let z_cpu: Vec<F> = x.iter().copied().chain(w.iter().copied()).collect();
                let Z_cpu = neo_memory::ajtai::encode_vector_balanced_to_mat(session.params(), &z_cpu);
                let c_cpu = session.committer().commit(&Z_cpu);
                let mcs = (
                    McsInstance {
                        c: c_cpu,
                        x: x.clone(),
                        m_in: layout.m_in,
                    },
                    McsWitness { w, Z: Z_cpu },
                );

                let prog_mem_inst = MemInstance {
                    mem_id: PROG_ID.0,
                    comms: Vec::new(),
                    k: prog_layout.k,
                    d: prog_layout.d,
                    n_side: prog_layout.n_side,
                    steps: layout.t,
                    lanes: 1,
                    ell: 1,
                    init: prog_mem_init.clone(),
                };
                let reg_mem_inst = MemInstance {
                    mem_id: REG_ID.0,
                    comms: Vec::new(),
                    k: 32,
                    d: 5,
                    n_side: 2,
                    steps: layout.t,
                    lanes: 2,
                    ell: 1,
                    init: reg_mem_init,
                };
                let ram_mem_inst = MemInstance {
                    mem_id: RAM_ID.0,
                    comms: Vec::new(),
                    k: ram_k,
                    d: ram_d,
                    n_side: 2,
                    steps: layout.t,
                    lanes: 1,
                    ell: 1,
                    init: ram_mem_init,
                };

                let prog_z = build_twist_only_bus_z(
                    ccs.m,
                    layout.m_in,
                    layout.t,
                    prog_mem_inst.d * prog_mem_inst.ell,
                    prog_mem_inst.lanes,
                    std::slice::from_ref(&twist_lanes.prog),
                    &x,
                )
                .map_err(|e| PiCcsError::InvalidInput(format!("build PROG twist z failed: {e}")))?;
                let prog_Z = neo_memory::ajtai::encode_vector_balanced_to_mat(session.params(), &prog_z);
                let prog_c = session.committer().commit(&prog_Z);
                let prog_mem_inst = MemInstance {
                    comms: vec![prog_c],
                    ..prog_mem_inst
                };
                let prog_mem_wit = MemWitness { mats: vec![prog_Z] };

                let reg_z = build_twist_only_bus_z(
                    ccs.m,
                    layout.m_in,
                    layout.t,
                    reg_mem_inst.d * reg_mem_inst.ell,
                    reg_mem_inst.lanes,
                    &[twist_lanes.reg_lane0.clone(), twist_lanes.reg_lane1.clone()],
                    &x,
                )
                .map_err(|e| PiCcsError::InvalidInput(format!("build REG twist z failed: {e}")))?;
                let reg_Z = neo_memory::ajtai::encode_vector_balanced_to_mat(session.params(), &reg_z);
                let reg_c = session.committer().commit(&reg_Z);
                let reg_mem_inst = MemInstance {
                    comms: vec![reg_c],
                    ..reg_mem_inst
                };
                let reg_mem_wit = MemWitness { mats: vec![reg_Z] };

                let ram_z = build_twist_only_bus_z(
                    ccs.m,
                    layout.m_in,
                    layout.t,
                    ram_mem_inst.d * ram_mem_inst.ell,
                    ram_mem_inst.lanes,
                    std::slice::from_ref(&twist_lanes.ram),
                    &x,
                )
                .map_err(|e| PiCcsError::InvalidInput(format!("build RAM twist z failed: {e}")))?;
                let ram_Z = neo_memory::ajtai::encode_vector_balanced_to_mat(session.params(), &ram_z);
                let ram_c = session.committer().commit(&ram_Z);
                let ram_mem_inst = MemInstance {
                    comms: vec![ram_c],
                    ..ram_mem_inst
                };
                let ram_mem_wit = MemWitness { mats: vec![ram_Z] };

                session.add_step_bundle(StepWitnessBundle {
                    mcs,
                    lut_instances: Vec::<(LutInstance<_, _>, LutWitness<F>)>::new(),
                    mem_instances: vec![
                        (prog_mem_inst, prog_mem_wit),
                        (reg_mem_inst, reg_mem_wit),
                        (ram_mem_inst, ram_mem_wit),
                    ],
                    _phantom: PhantomData::<K>,
                });

                apply_exec_chunk_writes_to_state(exec_chunk, &mut reg_state, &mut ram_state)?;
                chunk_build_commit_duration += elapsed_duration(chunk_start);
            }
        }

        let mem_order = session
            .steps_public()
            .first()
            .map(|s| {
                s.mem_insts
                    .iter()
                    .map(|inst| inst.mem_id)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let ram_ob_mem_idx = mem_order
            .iter()
            .position(|&id| id == RAM_ID.0)
            .ok_or_else(|| PiCcsError::ProtocolError("missing RAM mem instance for output binding".into()))?;
        let reg_ob_mem_idx = mem_order
            .iter()
            .position(|&id| id == REG_ID.0)
            .ok_or_else(|| PiCcsError::ProtocolError("missing REG mem instance for output binding".into()))?;

        let fold_start = time_now();
        let (proof, output_binding_cfg) = if output_claims.is_empty() {
            (session.fold_and_prove(&ccs)?, None)
        } else {
            let (ob_mem_idx, ob_num_bits, final_memory_state) = match output_target {
                OutputTarget::Ram => (
                    ram_ob_mem_idx,
                    ram_d,
                    final_ram_state_dense(&exec, &ram_init_map, ram_k)?,
                ),
                OutputTarget::Reg => (reg_ob_mem_idx, 5usize, final_reg_state_dense(&exec, &reg_init_map)?),
            };
            let ob_cfg = OutputBindingConfig::new(ob_num_bits, output_claims).with_mem_idx(ob_mem_idx);
            let proof = session.fold_and_prove_with_output_binding_simple(&ccs, &ob_cfg, &final_memory_state)?;
            (proof, Some(ob_cfg))
        };
        let fold_and_prove_duration = elapsed_duration(fold_start);
        let prove_duration = elapsed_duration(prove_start);
        let prove_phase_durations = Rv32TraceProvePhaseDurations {
            setup: setup_duration,
            chunk_build_commit: chunk_build_commit_duration,
            fold_and_prove: fold_and_prove_duration,
        };

        Ok(Rv32TraceWiringRun {
            session,
            ccs,
            layout,
            exec,
            proof,
            output_binding_cfg,
            prove_duration,
            prove_phase_durations,
            verify_duration: None,
        })
    }
}

/// Completed trace-wiring proof run.
pub struct Rv32TraceWiringRun {
    session: FoldingSession<AjtaiSModule>,
    ccs: CcsStructure<F>,
    layout: Rv32TraceCcsLayout,
    exec: Rv32ExecTable,
    proof: ShardProof,
    output_binding_cfg: Option<OutputBindingConfig>,
    prove_duration: Duration,
    prove_phase_durations: Rv32TraceProvePhaseDurations,
    verify_duration: Option<Duration>,
}

impl Rv32TraceWiringRun {
    pub fn params(&self) -> &NeoParams {
        self.session.params()
    }

    pub fn committer(&self) -> &AjtaiSModule {
        self.session.committer()
    }

    pub fn ccs(&self) -> &CcsStructure<F> {
        &self.ccs
    }

    pub fn layout(&self) -> &Rv32TraceCcsLayout {
        &self.layout
    }

    pub fn exec_table(&self) -> &Rv32ExecTable {
        &self.exec
    }

    pub fn proof(&self) -> &ShardProof {
        &self.proof
    }

    pub fn verify_proof(&self, proof: &ShardProof) -> Result<(), PiCcsError> {
        let ok = match &self.output_binding_cfg {
            None => self.session.verify_collected(&self.ccs, proof)?,
            Some(cfg) => self
                .session
                .verify_with_output_binding_collected_simple(&self.ccs, proof, cfg)?,
        };
        if !ok {
            return Err(PiCcsError::ProtocolError("verification failed".into()));
        }
        Ok(())
    }

    pub fn verify(&mut self) -> Result<(), PiCcsError> {
        let verify_start = time_now();
        self.verify_proof(&self.proof)?;
        self.verify_duration = Some(elapsed_duration(verify_start));
        Ok(())
    }

    pub fn ccs_num_constraints(&self) -> usize {
        self.ccs.n
    }

    pub fn ccs_num_variables(&self) -> usize {
        self.ccs.m
    }

    /// Number of real (active) rows in the unpadded trace.
    pub fn trace_len(&self) -> usize {
        self.exec.rows.iter().filter(|r| r.active).count()
    }

    /// Number of collected folding steps.
    pub fn fold_count(&self) -> usize {
        self.proof.steps.len()
    }

    pub fn prove_duration(&self) -> Duration {
        self.prove_duration
    }

    pub fn prove_phase_durations(&self) -> Rv32TraceProvePhaseDurations {
        self.prove_phase_durations
    }

    pub fn verify_duration(&self) -> Option<Duration> {
        self.verify_duration
    }

    pub fn steps_public(&self) -> Vec<neo_memory::witness::StepInstanceBundle<neo_ajtai::Commitment, F, K>> {
        self.session.steps_public()
    }
}
