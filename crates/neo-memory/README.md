# neo-memory

Twist & Shout memory protocols with full **RV64IMAC** RISC-V support.

## RISC-V Support

This crate implements the complete RV64IMAC instruction set, matching [Jolt's](https://github.com/a16z/jolt) capabilities:

| Extension | Description | Instructions |
|-----------|-------------|--------------|
| **I** | Base Integer (64-bit) | ADD, SUB, AND, OR, XOR, SLT, shifts, branches, jumps, loads, stores |
| **M** | Multiply/Divide | MUL, MULH, MULHU, MULHSU, DIV, DIVU, REM, REMU |
| **A** | Atomics | LR, SC, AMOSWAP, AMOADD, AMOXOR, AMOAND, AMOOR, AMOMIN, AMOMAX |
| **C** | Compressed | All 16-bit instructions (C.ADDI, C.LW, C.SW, C.J, etc.) |

### RV64-specific Operations

- **Word operations**: ADDW, SUBW, SLLW, SRLW, SRAW, MULW, DIVW, REMW (32-bit ops with sign-extension)
- **64-bit loads/stores**: LD, SD, LWU

## Key Modules

### `riscv_lookups`
- Instruction decoding (32-bit and 16-bit compressed)
- Instruction encoding
- CPU execution with tracing
- Lookup tables for ALU operations

### `riscv_ccs`
- Constraint system (CCS) definitions for RISC-V instructions
- Witness generation from execution traces

### `elf_loader`
- Load ELF binaries
- Load raw RISC-V binaries

### `output_check`
- Output sumcheck protocol
- Binds program I/O to cryptographic proofs

### `twist` / `shout`
- Twist: Read-write memory protocol
- Shout: Read-only memory / lookup table protocol

## Example

```rust
use neo_memory::riscv_lookups::{RiscvCpu, RiscvMemory, RiscvShoutTables, decode_program};
use neo_memory::elf_loader::load_raw_binary;
use neo_vm_trace::trace_program;

// Load a RISC-V binary
let loaded = load_raw_binary(&binary_bytes, 0x1000)?;

// Execute with full tracing
let mut cpu = RiscvCpu::new(64);
cpu.load_program(0x1000, loaded.get_instructions());
let memory = RiscvMemory::new(64);
let shout = RiscvShoutTables::new(64);

let trace = trace_program(cpu, memory, shout, 10000)?;
assert!(trace.did_halt());

// trace.steps contains all execution steps for proving
```

## Architecture

```
┌──────────────────┐     ┌──────────────────┐
│   RiscvCpu       │────▶│   VmTrace        │
│   (execution)    │     │   (steps)        │
└──────────────────┘     └──────────────────┘
         │                        │
         ▼                        ▼
┌──────────────────┐     ┌──────────────────┐
│   Shout          │     │   Twist          │
│   (ALU lookups)  │     │   (memory)       │
└──────────────────┘     └──────────────────┘
         │                        │
         └────────────┬───────────┘
                      ▼
              ┌──────────────────┐
              │   Neo Prover     │
              │   (fold + prove) │
              └──────────────────┘
```

