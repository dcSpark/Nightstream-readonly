//! RISC-V lookup-based execution helpers (RV32-focused proving integration).
//!
//! This module provides:
//! - Instruction decoding/encoding
//! - A traceable CPU that emits Twist (memory) and Shout (ALU/compare lookup) events
//! - Lookup helpers/tables for proving integrations
//!
//! # Proving integration scope (today)
//!
//! The shared-bus RV32 trace-wiring proving path assumes:
//! - `xlen == 32` (RV32)
//! - no compressed (RVC) instructions
//! - 4-byte aligned PC and control-flow targets
//!
//! Note: Shout operand keys are encoded via bit interleaving into a `u64` key
//! (2×32-bit → 64-bit), which is sufficient for RV32. RV64 proving requires a wider key.
//!
//! # Architecture
//!
//! ## Lookup Tables (Shout)
//!
//! ALU operations are proven using Neo's Shout (read-only memory) protocol:
//! - The **index** encodes operands via bit interleaving
//! - The **value** is the operation result
//! - MLEs enable efficient sumcheck verification
//!
//! ## Memory (Twist)
//!
//! Load/store and atomic operations use Neo's Twist (read-write memory) protocol.
//!
//! # Instruction Categories
//!
//! ## Base Integer (I Extension)
//! - **Arithmetic**: ADD, ADDI, SUB
//! - **Logical**: AND, ANDI, OR, ORI, XOR, XORI
//! - **Shifts**: SLL, SLLI, SRL, SRLI, SRA, SRAI
//! - **Compare**: SLT, SLTI, SLTU, SLTIU
//! - **Branches**: BEQ, BNE, BLT, BGE, BLTU, BGEU
//! - **Jumps**: JAL, JALR
//! - **Upper Immediate**: LUI, AUIPC
//! - **Loads**: LB, LBU, LH, LHU, LW, LWU, LD
//! - **Stores**: SB, SH, SW, SD
//!
//! ## RV64 Word Operations
//! - ADDW, SUBW, ADDIW
//! - SLLW, SLLIW, SRLW, SRLIW, SRAW, SRAIW
//!
//! ## Multiply/Divide (M Extension)
//! - MUL, MULH, MULHU, MULHSU
//! - DIV, DIVU, REM, REMU
//! - MULW, DIVW, DIVUW, REMW, REMUW (RV64)
//!
//! ## Atomics (A Extension)
//! - **Load-Reserved**: LR.W, LR.D
//! - **Store-Conditional**: SC.W, SC.D
//! - **AMO**: AMOSWAP, AMOADD, AMOXOR, AMOAND, AMOOR, AMOMIN, AMOMAX, AMOMINU, AMOMAXU
//!
//! ## Compressed (C Extension)
//! - All quadrant 0, 1, and 2 instructions
//! - Automatic detection of 16-bit vs 32-bit instructions
//!
//! ## System
//! - ECALL, FENCE
//! - EBREAK and FENCE.I are not supported
//!
//! # Example
//!
//! ```ignore
//! use neo_memory::riscv::lookups::{decode_program, RiscvCpu, RiscvMemory, RiscvShoutTables};
//! use neo_vm_trace::trace_program;
//!
//! // Load and decode a RISC-V binary
//! let program = decode_program(&binary_bytes)?;
//!
//! // Execute with full tracing
//! let mut cpu = RiscvCpu::new(32); // RV32
//! cpu.load_program(0, program);
//! let memory = RiscvMemory::new(32);
//! let shout = RiscvShoutTables::new(32);
//!
//! let trace = trace_program(cpu, memory, shout, 1000)?;
//! // trace now contains all steps for proving
//! ```

mod alu;
mod bits;
mod cpu;
mod decode;
mod encode;
mod isa;
mod memory;
mod mle;
mod tables;
mod trace;

use neo_vm_trace::TwistId;

/// Canonical Twist instance id for RISC-V data RAM.
pub const RAM_ID: TwistId = TwistId(0);

/// Canonical Twist instance id for the program ROM instruction fetch.
pub const PROG_ID: TwistId = TwistId(1);

/// Canonical Twist instance id for the architectural register file (x0..x31).
///
/// This is used by the RV32 trace-wiring circuit in "regfile-as-Twist" mode.
pub const REG_ID: TwistId = TwistId(2);

/// Poseidon2-Goldilocks hash compute ECALL identifier.
///
/// ABI: a0 = POSEIDON2_ECALL_NUM, a1 = input element count,
///      a2 = input RAM address (elements as 2×u32 LE).
/// The host reads inputs via untraced loads, computes the hash, and stores the
/// 4-element digest in CPU-internal state. Use POSEIDON2_READ_ECALL_NUM to
/// retrieve output words one at a time via register a0.
pub const POSEIDON2_ECALL_NUM: u32 = 0x504F53;

/// Poseidon2-Goldilocks digest read ECALL identifier (bit 31 set).
///
/// ABI: a0 = POSEIDON2_READ_ECALL_NUM. Returns the next u32 word of the
/// pending Poseidon2 digest in register a0. Call 8 times (4 elements × 2 words)
/// to retrieve the full digest.
pub const POSEIDON2_READ_ECALL_NUM: u32 = 0x80504F53;

/// Goldilocks field multiply ECALL identifier ("GLM").
///
/// ABI: a0 = GL_MUL_ECALL_NUM, a1 = a_lo, a2 = a_hi, a3 = b_lo, a4 = b_hi.
/// Computes (a * b) mod p and stores the 64-bit result in CPU state.
/// Retrieve via GL_READ_ECALL_NUM (2 calls for lo/hi words).
pub const GL_MUL_ECALL_NUM: u32 = 0x474C4D;

/// Goldilocks field add ECALL identifier ("GLA").
///
/// ABI: a0 = GL_ADD_ECALL_NUM, a1 = a_lo, a2 = a_hi, a3 = b_lo, a4 = b_hi.
/// Computes (a + b) mod p and stores the 64-bit result in CPU state.
pub const GL_ADD_ECALL_NUM: u32 = 0x474C41;

/// Goldilocks field subtract ECALL identifier ("GLS").
///
/// ABI: a0 = GL_SUB_ECALL_NUM, a1 = a_lo, a2 = a_hi, a3 = b_lo, a4 = b_hi.
/// Computes (a - b) mod p and stores the 64-bit result in CPU state.
pub const GL_SUB_ECALL_NUM: u32 = 0x474C53;

/// Goldilocks field operation read ECALL identifier (bit 31 set on "GLR").
///
/// ABI: a0 = GL_READ_ECALL_NUM. Returns the next u32 word of the
/// pending field operation result in register a0. Call 2 times (lo/hi).
pub const GL_READ_ECALL_NUM: u32 = 0x80474C52;

pub use alu::{compute_op, lookup_entry};
pub use bits::{interleave_bits, uninterleave_bits};
pub use cpu::RiscvCpu;
pub use decode::{decode_compressed_instruction, decode_instruction, decode_program, RiscvFormat};
pub use encode::{encode_instruction, encode_program};
pub use isa::{BranchCondition, RiscvInstruction, RiscvMemOp, RiscvOpcode};
pub use memory::{RiscvMemory, RiscvMemoryEvent};
pub use mle::{
    evaluate_add_mle, evaluate_and_mle, evaluate_eq_mle, evaluate_neq_mle, evaluate_opcode_mle, evaluate_or_mle,
    evaluate_sll_mle, evaluate_slt_mle, evaluate_sltu_mle, evaluate_sra_mle, evaluate_srl_mle, evaluate_sub_mle,
    evaluate_xor_mle,
};
pub use tables::{RangeCheckTable, RiscvLookupEvent, RiscvLookupTable, RiscvShoutTables};
pub use trace::{
    analyze_trace, build_final_memory_state, build_opcode_lut_table, extract_program_io, trace_to_plain_lut_trace,
    trace_to_plain_lut_traces_by_opcode, trace_to_plain_mem_trace, TraceConversionSummary, TraceToProofConfig,
};
