# neo-spartan-bridge

**Neo → Spartan2 bridge** using **Hash‑MLE PCS** (Merkle + Poseidon2) and a **Poseidon2 FS transcript** on Goldilocks.

## Architecture

- **Transcript**: Poseidon2 (p3) — same family as the rest of Neo (*single‑transcript policy*).
- **PCS**: Spartan2 `HashMlePcsP3` (Poseidon2 Merkle), **no FRI**.
- **Output**: standard Spartan2 R1CS SNARK (proof bytes + verifier key), plus deterministic public‑IO encoding.

## What's in this crate

- `NeoPoseidonGoldiEngine` — Spartan2 engine using Poseidon2 transcript and Hash-MLE PCS
- `compress_me_to_spartan()` — converts ME(b,L) instance to Spartan2 R1CS SNARK proof
- `verify_me_spartan()` — verifies the resulting SNARK proof
- Hash-MLE helpers for standalone polynomial commitment proofs

> We use **Hash-MLE PCS only** — no FRI, no mixed transcript families. This maintains Neo's unified Poseidon2 transcript security model.

## Status

- ✅ **Poseidon2 transcript** unified with Neo's folding phase
- ✅ **Hash‑MLE PCS** via Spartan2's `HashMlePcsP3` 
- ✅ **Production ready** — no dev features, no stub implementations
- ✅ **R1CS SNARK** — standard Spartan2 output format

## Quick start

```rust
use neo_spartan_bridge::{compress_me_to_spartan, verify_me_spartan};
use neo_ccs::{MEInstance, MEWitness};

// Convert ME instance to Spartan2 proof
let bundle = compress_me_to_spartan(&me_instance, &me_witness)?;

// Verify the SNARK proof
let is_valid = verify_me_spartan(&bundle)?;
```

## Security Properties

- **Post-quantum**: Hash-based MLE PCS, no elliptic curves or pairings
- **Transcript binding**: Fold digest included in SNARK public inputs  
- **Unified Poseidon2**: Consistent Fiat-Shamir across all Neo phases
- **Standard R1CS**: Well-audited SNARK patterns

## Running tests

```bash
cargo test -p neo-spartan-bridge -- --nocapture
```

**Security notes**:

* Hash & FS: Poseidon2 (width 16) over Goldilocks field
* Commitment: Hash‑MLE using Poseidon2 Merkle trees
* **No `unsafe`** — crate is compiled with `#![forbid(unsafe_code)]`
* **No mixed transcripts** — Poseidon2 only, matching Neo's folding phase

**This bridge provides the production-ready foundation for compressing Neo's folded claims into succinct Spartan2 proofs using unified Poseidon2 transcripts and hash-based polynomial commitments.**