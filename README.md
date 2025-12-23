# Nightstream — Lattice‑based Folding with Twist/Shout Memory

[![GitHub License](https://img.shields.io/github/license/nicarq/nightstream)](LICENSE)
[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/00000/badge)](https://bestpractices.coreinfrastructure.org/projects/00000)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/nicarq/nightstream/badge)](https://scorecard.dev/viewer/?uri=github.com/nicarq/nightstream)

Nightstream is an end‑to‑end **post‑quantum** proving system built around a lattice-based folding scheme for **CCS** plus sum-check–based memory arguments:
- **Twist** for read/write memory
- **Shout** for read‑only lookups

It targets CCS over the **Goldilocks** field (with a degree‑2 extension field for sum-check soundness) and is designed for zkVM-style workloads via **shard-level folding**.

Nightstream implements the protocol from the Neo paper "Lattice‑based folding scheme for CCS over small fields" (Nguyen & Setty, 2025), extended with Twist & Shout memory arguments.

> **🚧 Status**: Research prototype. Shard folding loop and Twist/Shout integration (including two-lane obligations) are implemented. Not production-ready.

---

## Developer Onboarding

### 1. Read the protocol + implementation overview

| Doc | Purpose |
|-----|---------|
| `docs/neo-ai-summary.md` | Developer-grade Neo protocol overview |
| `docs/system-architecture.md` | IVC architecture + emission policies |
| `docs/neo-with-twist-and-shout/integration-summary.md` | Twist/Shout integration strategy (why two lanes) |

### 2. Run tests

```bash
cargo test --workspace --release

# See full shard folding with Twist/Shout in action:
cargo test -p neo-fold full_folding_integration --release -- --nocapture

# Twist/Shout witness building:
cargo test -p neo-fold twist_shout_integration --release -- --nocapture
```

### 3. Where to start in the code

**Shard folding loop** — `crates/neo-fold/src/shard.rs`
- Look for `fold_shard_prove_impl(...)` and `fold_shard_verify(...)`
- This is where:
  - Per-step inputs are bound into the transcript
  - Π_CCS is executed
  - Twist/Shout proofs are produced/checked
  - Π_RLC → Π_DEC runs for the main lane, and (when needed) for the value lane

**Memory sidecar (Twist/Shout integration)** — `crates/neo-fold/src/memory_sidecar/memory.rs`
- Bridge layer that:
  - Runs the memory/lookup sum-checks
  - Emits ME claims/witnesses at `r_time`
  - Runs Twist's value-eval sum-check and emits value-lane ME claims at `r_val`

**Trace → per-step witnesses** — `crates/neo-memory/src/builder.rs`
- `build_shard_witness(...)` splits a VM trace into chunks and produces:
  - An MCS witness chunk
  - Matching Twist/Shout witnesses for that same chunk

**Twist/Shout encoders and invariants**
- `crates/neo-memory/src/encode.rs` — bit-address encoding, layout decisions
- `crates/neo-memory/src/twist.rs`, `shout.rs` — semantics checks, decoding
- `crates/neo-memory/src/twist_oracle.rs` — sum-check oracles

---

## Quick Start

### Prerequisites
* **Rust** (stable, or use `rust-toolchain.toml` if present)
* `git`
* C compiler (only if enabling allocators like mimalloc)

### Build & Test
```bash
cargo build --release
cargo test --workspace --release

# Focused test runs:
cargo test -p neo-fold --release
cargo test -p neo-memory --release
cargo test -p neo-reductions --release
```

### Paper-exact reference mode
```bash
cargo test -p neo-reductions --features paper-exact --release
```

---

## Core Concepts (Paper → Code)

| Concept | Meaning | Code Entry Points |
|---------|---------|-------------------|
| **Shard** | Trace segment processed chunk-by-chunk | `crates/neo-fold/src/shard.rs` |
| **Folding step** | Unit consumed per iteration of the loop | `StepWitnessBundle` in `neo_memory::witness` |
| **CCS** | Customizable Constraint System | `neo_ccs::relations::CcsStructure` |
| **MCS** | Matrix Constraint System (CCS + commitment) | `neo_ccs::relations::{McsInstance, McsWitness}` |
| **ME** | Universal foldable claim (single-point eval) | `neo_ccs::relations::MeInstance` |
| **Π_CCS** | CCS/MCS → ME claims via sum-check | `neo_reductions::engines::*` |
| **Π_RLC / Π_DEC** | Aggregate then decompose (norm control) | `crates/neo-fold/src/shard.rs` |
| **Twist** | R/W memory argument (sparse increments) | `crates/neo-memory/src/twist.rs`, `twist_oracle.rs` |
| **Shout** | Read-only lookup argument | `crates/neo-memory/src/shout.rs` |
| **IDX** | SPARK-style Index→OneHot bridge (virtual one-hot via bit-columns) | `IndexAdapterOracle` in `twist_oracle.rs` |
| **Two-lane folding** | Needed for Twist's second eval point `r_val` | `val_fold` in `shard.rs` |

### Key Structs

```rust
// One folding chunk worth of witness:
neo_memory::witness::StepWitnessBundle {
    mcs: (McsInstance, McsWitness),      // CPU chunk
    lut_instances: Vec<(LutInstance, LutWitness)>,  // Shout per chunk
    mem_instances: Vec<(MemInstance, MemWitness)>,  // Twist per chunk
}

// Final obligations after shard verification:
neo_fold::shard::ShardObligations {
    main: Vec<MeInstance>,  // ME claims at r_time
    val: Vec<MeInstance>,   // ME claims at r_val (Twist only)
}
```

---

## Architecture Overview

Nightstream implements **shard-level folding** where each step processes one CCS chunk together with its matching Twist/Shout instances, all sharing sum-check challenges.

### Per-Step Folding Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│                                  Step i                                   │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌─────────────────┐                                                     │
│   │  k running ME   │  ◄── Accumulator carried from step i-1              │
│   └────────┬────────┘                                                     │
│            │                                                              │
│            │      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│            │      │    Π_CCS     │  │   Π_Twist    │  │   Π_Shout    │    │
│            │      │  (CPU chunk) │  │ (R/W memory) │  │  (lookups)   │    │
│            │      └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│            │             │                 │                 │            │
│            │             └────────────┬────┴─────────────────┘            │
│            │                          │                                   │
│            │                          ▼                                   │
│            │             ┌────────────────────────┐                       │
│            │             │  Batched sum-check     │                       │
│            │             │  (shared r_time)       │                       │
│            │             └────────────┬───────────┘                       │
│            │                          │                                   │
│            │                          ▼                                   │
│            │             ┌────────────────────────┐                       │
│            │             │   Fresh ME claims      │                       │
│            │             │ (CCS+Twist+Shout+IDX)  │                       │
│            │             └────────────┬───────────┘                       │
│            │                          │                                   │
│            └──────────────────────────┤                                   │
│                                       ▼                                   │
│                    ┌──────────────────────────────────┐                   │
│                    │ Main lane: Π_RLC → Π_DEC         │                   │
│                    │ fold all ME@r_time → k children  │                   │
│                    └─────────────────┬────────────────┘                   │
│                                      │                                    │
│            ┌─────────────────────────┴─────────────────────────┐          │
│            │                                                   │          │
│            ▼                                                   ▼          │
│   ┌─────────────────┐                                ┌─────────────────┐  │
│   │  k ME children  │                                │   Value lane    │  │
│   │  (to step i+1)  │                                │ (Twist @ r_val) │  │
│   └────────┬────────┘                                └────────┬────────┘  │
│            │                                                  │           │
└────────────┼──────────────────────────────────────────────────┼───────────┘
             │                                                  │
             ▼                                                  ▼
    (next step i+1)                                   ┌─────────────────────┐
                                                      │  value-lane         │
                                                      │  obligations        │
                                                      │  must be enforced   │
                                                      └─────────────────────┘
```

### Unified Folding Interface

All arguments reduce to the same **ME(b, L)** relation:

```
Π_CCS   : MCS(b, L)  ⟿  ME(b, L)^t_ccs
Π_Twist : TWI(b, L)  ⟿  ME(b, L)^t_twi
Π_Shout : SHO(b, L)  ⟿  ME(b, L)^t_sho
```

At each step:
```
(k running ME + fresh CCS ME + Twist ME + Shout ME) → Π_RLC → ME^agg → Π_DEC → ME(b, L)^k
```

---

## Two-Lane Folding

Twist's val-eval subprotocol requires a separate evaluation point `r_val`, creating two parallel folding lanes:

| Lane | Evaluation Point | Contents |
|------|-----------------|----------|
| **Main** | `r_time` | CCS + Shout + Twist read/write checks |
| **Val** | `r_val` | Twist value-evaluation claims |

Both lanes produce ME obligations that must be verified by the final proof layer.

### Why Two Lanes?

- Most claims are enforced at a single shared evaluation point `r_time` (sampled once per step via Fiat–Shamir)
- Twist also needs a separate evaluation point `r_val` for its value-reconstruction subprotocol (fresh sum-check challenges)
- Because Neo's ME is a single-point evaluation relation, `ME@r_time` and `ME@r_val` cannot be mixed in the same `Π_RLC` call

**Result**: each step can emit:
- **Main obligations**: ME children at `r_time` (carried to the next step)
- **Value-lane obligations**: ME children at `r_val` (must be carried forward to the final checker)

---

## Repository Structure

```
crates/
  neo-ajtai/           # Ajtai (lattice) commitments; module-SIS binding
  neo-ccs/             # CCS/MCS/ME relations, matrices, arithmetization
  neo-fold/            # Shard folding loop, proof types, transcript plumbing
  neo-reductions/      # Π_CCS / Π_RLC / Π_DEC engines (optimized + paper-exact)
  neo-memory/          # Twist/Shout traces, encoding, MLE utilities, oracles
  neo-vm-trace/        # VM tracing traits (CPU, Twist, Shout) + trace capture
  neo-spartan-bridge/  # ME → Spartan2-style R1CS bridge (WIP)
  neo-math/            # Field/ring utilities, extension field, norms
  neo-params/          # Parameter bundles + Poseidon2 config
  neo-transcript/      # Poseidon2 transcript (Fiat–Shamir)

docs/
  neo-paper/                       # Paper text (reference)
  neo-with-twist-and-shout/        # Twist/Shout integration docs
  neo-ai-summary.md                # Implementation-facing overview
  system-architecture.md           # End-to-end architecture notes
```

---

## End-to-End: Trace → Witness → Fold

### Step 1: Build per-chunk witnesses

Use the encoding functions in `neo-memory`:

```rust
use neo_memory::encode::{encode_mem_for_twist, encode_lut_for_shout};
use neo_memory::witness::StepWitnessBundle;

// For each chunk, build:
let (mem_inst, mem_wit) = encode_mem_for_twist(&params, &layout, &init, &trace, &commit_fn, None, 0);
let (lut_inst, lut_wit) = encode_lut_for_shout(&params, &table, &trace, &commit_fn, None, 0);

let step = StepWitnessBundle {
    mcs: (mcs_inst, mcs_wit),
    lut_instances: vec![(lut_inst, lut_wit)],
    mem_instances: vec![(mem_inst, mem_wit)],
    _phantom: PhantomData,
};
```

**Reference test**: `crates/neo-fold/tests/full_folding_integration.rs::full_folding_integration_single_chunk`

### Step 2: Prove a shard

```rust
use neo_fold::shard::{fold_shard_prove, fold_shard_verify, ShardObligations};
use neo_transcript::Poseidon2Transcript;

let mut tr = Poseidon2Transcript::new(b"nightstream/shard");
let proof = fold_shard_prove(
    FoldingMode::Optimized,
    &mut tr,
    &params,
    &ccs,
    &steps,           // Vec<StepWitnessBundle>
    &acc_init,        // Initial ME accumulator
    &acc_wit_init,    // Initial witnesses
    &l,               // Commitment scheme
    mixers,
)?;
```

### Step 3: Verify and handle obligations

```rust
let mut tr_v = Poseidon2Transcript::new(b"nightstream/shard");
let outputs = fold_shard_verify(
    FoldingMode::Optimized,
    &mut tr_v,
    &params,
    &ccs,
    &step_instances,
    &acc_init,
    &proof,
    mixers,
)?;

// Handle both obligation lanes:
let main_obligations: &[MeInstance] = &outputs.obligations.main;
let val_obligations: &[MeInstance] = &outputs.obligations.val;

// Pass to final SNARK layer or ObligationFinalizer
```

---

## Memory Arguments: Twist & Shout

### Twist (Read/Write Memory)

Twist models memory as a recurrence via sparse updates:
```
Val_{t+1} = Val_t + Inc_t
```

**What's committed per chunk:**
- Address bit-columns for reads/writes
- `has_read`, `has_write` flags
- `rv`, `wv` (read/write values)
- `inc_at_write_addr` (write delta)

**What stays virtual:**
- Full memory vector `Val_t` (never committed, computed via sum-check)

**Code:**
- Encoding: `neo_memory::encode::encode_mem_for_twist`
- Oracles: `neo_memory::twist_oracle.rs`
- Semantics: `neo_memory::twist::check_twist_semantics`

### Shout (Read-Only Lookups)

Shout proves that when `has_lookup[t] = 1`, the committed `val[t]` matches `table[addr[t]]`.

**What's committed per chunk:**
- Address bit-columns (masked by `has_lookup`)
- `has_lookup` flag
- `val`

**Code:**
- Encoding: `neo_memory::encode::encode_lut_for_shout`
- Oracles: `neo_memory::shout.rs`

### Address Encoding & IDX Adapter (SPARK-style Bridge)

Addresses use compact **bit-decomposition** instead of one-hot vectors:
- Each address is `d` components in base `n_side`
- Each component commits `ell = ceil(log2(n_side))` bit-columns
- Address width: `d * ell` columns instead of `d * n_side` (~32× reduction)

The **IDX adapter** implements a SPARK-style bridge: it provides a **virtual one-hot oracle** backed by committed bit-columns. Twist/Shout protocols query conceptual one-hot MLE evaluations, and the adapter proves these are consistent with the compact index-bit representation via sum-check. This shifts work from commitments to foldable sum-check proofs.

**Code:** `neo_memory::encode::get_ell`, `IndexAdapterOracle` in `twist_oracle.rs`

---

## Development Notes

### Folding Engines

| Mode | Description |
|------|-------------|
| `FoldingMode::Optimized` | Production implementation |
| `FoldingMode::PaperExact` | Reference (feature-gated) |
| `FoldingMode::OptimizedWithCrosscheck` | Debug comparison mode |

### Debugging Tips

- Start with `chunk_size = 1` to shrink state
- If RLC alignment errors occur, check:
  - `validate_me_batch_invariants` in `shard.rs`
  - The `r` points in emitted ME claims (must match per lane)

### Key Tests

```bash
# Full shard prove/verify with Twist/Shout:
cargo test -p neo-fold full_folding_integration --release -- --nocapture

# Twist/Shout witness building:
cargo test -p neo-fold twist_shout_trace_to_witness_smoke --release -- --nocapture

# Session API (IVC-style):
cargo test -p neo-fold test_session_multifold --release -- --nocapture
```

---

## Security & Correctness

### ✅ Implemented Safeguards
* **Parameter validation**: Enforces RLC soundness inequality `(k+1)·T·(b-1) < B`
* **Transcript binding**: Poseidon2 domain separation across all phases
* **ME claim alignment**: Validates `r`-point consistency before Π_RLC
* **Two-lane obligation tracking**: Val-lane ME claims tracked in `ShardObligations`

### 🔬 Security Posture
> **Research software warning**: This demonstrates the protocol and transcript-binding structure but has not undergone independent review. Do not deploy without a full audit and complete final verification layer.

---

## Roadmap

### Near Term
- [ ] Integrate Spartan2 final SNARK layer
- [ ] Add criterion benchmarks
- [ ] Sparse weight optimizations in bridge

### Medium Term
- [ ] GPU acceleration exploration
- [ ] Security audit preparation

### Long Term
- [ ] Production deployment tools
- [ ] zkVM integration (RISC-V/WASM)

---

## References

* **Neo**: Wilson Nguyen & Srinath Setty, "[Neo: Lattice-based folding scheme for CCS over small fields](https://eprint.iacr.org/2025/294)" (ePrint 2025/294)
* **Twist/Shout integration**: `docs/neo-with-twist-and-shout/`
* **Spartan**: Srinath Setty, "Spartan: Efficient and general-purpose zkSNARKs without trusted setup" (CRYPTO 2020)
* **Nova/HyperNova**: Recursive arguments from folding schemes
* **Plonky3**: Goldilocks field, Poseidon2 used by Nightstream

---

## Contributing

* **Add tests** for behavioral changes
* **Run formatting**: `cargo fmt` and `cargo clippy`
* **Update documentation** for API changes

---

## Governance & Policies

- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Maintainers](MAINTAINERS.md)

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
