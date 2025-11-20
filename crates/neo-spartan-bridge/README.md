# neo-spartan-bridge

Experimental integration layer between Neo folding (Π-CCS / `FoldRun`) and the Spartan2 SNARK.

> **Status:** WIP / non-production. The crate compiles gadgets and a circuit skeleton, but:
> - Only **sumcheck round invariants** are partially wired.
> - Terminal identity, RLC/DEC equalities, and accumulator chaining are **not enforced yet**.
> - Spartan2 is **not yet called**; no real SNARK proofs are produced.

---

## Goal

Provide a single Spartan2 proof that attests to the correctness of an entire Neo `FoldRun`, with:

- Π-CCS **sumcheck invariants and terminal identity** enforced as R1CS constraints.
- Π-RLC and Π-DEC **linear/base-b equalities** enforced as constraints.
- **Accumulator chaining** across all steps, from an initial public accumulator to a final public claim.

The Spartan proof uses whatever PCS is chosen by the Spartan **engine** (Hash-MLE today). The bridge only defines a bellpepper circuit over Neo's Goldilocks arithmetic; it does not introduce a second PCS.

---

## Architecture

The crate is split into:

1. **`circuit/`** – R1CS circuit for a `FoldRun`:
   - `FoldRunInstance` – public IO container (digests, accumulator claim, Π-CCS challenge data).
   - `FoldRunWitness` – private witness (FoldRun, Π-CCS proofs, Z, ρ, DEC digits).
   - `FoldRunCircuit` – synthesizes bellpepper constraints for all steps.

2. **`gadgets/`** – small reusable gadgets:
   - `k_field` – K-field (degree-2 extension) as 2 limbs over the base field.
   - `pi_ccs` – sumcheck-related helpers and base-b recomposition.
   - `common` – experimental eq / MLE / range-product helpers (not yet used in the main circuit).

3. **`api/`** – high-level `prove_fold_run` / `verify_fold_run` API **stubs**. At the moment, they only build instances and do not call Spartan2.

4. **`engine/`** – experimental hooks for Z-polynomial layout. This is **not wired into Spartan2** yet and can be treated as a design sketch (gated behind `experimental-engine` feature).

---

## Key Design Principles

### 1. Π-CCS Challenges as Data

The circuit **does not mint Π-CCS challenges**. Challenges (`α, β_a, β_r, γ, r', …`) come from the outer Neo transcript and are passed as inputs to the circuit (ultimately as public IO). The circuit verifies algebraic relationships given those challenges.

### 2. Spartan PCS is a Black Box

Spartan's PCS (Hash-MLE now, other PCS later) only commits to the *entire witness vector*. The bridge does **not**:

- Create separate Ajtai commitments inside Spartan.
- Use Spartan's PCS to encode Π-CCS's own commitments.

All Π-CCS/Π-RLC/Π-DEC logic is enforced as plain R1CS constraints over the base field.

### 3. One Spartan Proof per FoldRun

The intended end state is a **single** Spartan SNARK for the whole folding schedule:

- Inputs: digests and the initial/final accumulator.
- Witness: full `FoldRun`, Π-CCS proofs, Z/ρ/DEC witnesses.
- Constraints: sumcheck, terminal identity, RLC, DEC, and chaining.

At present, the circuit only contains the structure and partial sumcheck checks.

---

## Current Implementation

### ✅ Implemented

- **K-field gadgets** (`gadgets/k_field.rs`):
  - 2-limb representation `K = c0 + c1 * u` over a base `F`.
  - Addition and multiplication gadgets: `k_add`, `k_mul`, `k_scalar_mul` (no `cs.get()` calls).
  - Lifting base-field elements to `K`: `k_lift_from_f`.
  - Conversion from `neo_math::K` via canonical `u64` (`from_neo_k`).

- **Π-CCS sumcheck gadgets** (`gadgets/pi_ccs.rs`):
  - `sumcheck_round_gadget`: enforces `p(0) + p(1) = claimed_sum`.
  - `sumcheck_eval_gadget`: evaluates `p(challenge)` via Horner's method and **returns KNumVar** (no pre-allocated witness).
  - `base_b_recompose_k`: `Σ b^ℓ · y[ℓ]` in `K`.

- **Circuit skeleton** (`circuit/`):
  - `FoldRunInstance` / `FoldRunWitness` structs.
  - `FoldRunCircuit<F>` with:
    - Per-step hook `verify_fold_step`.
    - `verify_sumcheck_rounds` that uses **real proof values** (`proof.initial_sum`, `proof.sumcheck_final`) and derives next sums via gadget return values.
    - Implementations for `verify_rlc` / `verify_dec` that enforce X/y/r linear relations matching the public `rlc_public` / `verify_dec_public` helpers (commitment checks are still left to the outer Ajtai verifier).
    - A minimal `verify_accumulator_chaining` that ties the public final accumulator to the `FoldRun`'s final outputs (full cross-step chaining is still TODO).
  - Public input allocation is temporarily disabled (commented out) to avoid meaningless zero-valued inputs.

- **Digests & API shape** (`api.rs`):
  - `compute_params_digest`, `compute_ccs_digest` using BLAKE3.
  - `SpartanProof` type and `prove_fold_run` / `verify_fold_run` signatures (no real Spartan calls yet, no generic `BridgeEngine`).

### ❌ Not Implemented / TODO

- **Terminal identity gadget**:
  - `terminal_identity_rhs_gadget` is a stub returning zero.
  - Needs to re-express the paper's Step-4 RHS using `KNumVar` and the common gadgets.

- **Accumulator chaining**:
  - `verify_accumulator_chaining` currently only enforces that the public final accumulator matches the `FoldRun`'s final outputs.
  - It should be extended to connect step-0 inputs to the initial accumulator, intermediate DEC children to the next step's inputs, and the last outputs to the final accumulator.

- **eq / MLE gadgets**:
  - `eq_gadget` and `mle_eval_gadget` rely on `cs.get()` and are **gated behind `unsafe-gadgets` feature**.
  - They are currently unused; they will be rewritten or removed when terminal identity is implemented.

- **Spartan2 integration**:
  - `prove_fold_run` / `verify_fold_run` do not call Spartan2 yet.
  - No `SpartanCircuit` implementation exists for `FoldRunCircuit` yet.
  - `engine.rs` and `BridgeEngine` are experimental and gated behind `experimental-engine` feature (not tied into `spartan2::traits::Engine`).

---

## Recommended Next Steps

In order of priority:

1. **Unit test the sumcheck-only circuit (no Spartan):**
   - Write tests that:
     - Construct a tiny synthetic `PiCcsProof` with 1–2 rounds, consistent coefficients, challenges, and sums.
     - Build a `FoldRunCircuit` with one step, plug in that proof as the witness, and check that a `TestConstraintSystem` is satisfied.

2. **Integrate Spartan2 with a thin wrapper:**
   - Define a `FoldRunSpartanCircuit<E>` that wraps `FoldRunCircuit<E::Scalar>` and implements `spartan2::traits::circuit::SpartanCircuit<E>`.
   - Call `R1CSSNARK<E>::setup / prep_prove / prove / verify` in `api.rs` to produce and verify real Spartan proofs.

3. **Extend constraints gradually:**
   - Implement the terminal identity gadget and cross-check against the native Neo implementation (`rhs_terminal_identity_paper_exact`).
   - Add RLC and DEC equalities for a single step.
   - Add accumulator chaining across steps and connect the initial/final accumulator to meaningful public IO.

4. **Revisit engine / PCS modularity:**
   - Once a basic Spartan pipeline is working, reintroduce a clean `ZPolyLayout` abstraction tied to Spartan's `Engine` (not a separate `BridgeEngine`).
   - Use that to make the integration robust to future PCS changes.

---

## Safety and Caveats

- This crate is **experimental** and should not be used as a security boundary.
- Many pieces (terminal identity, RLC/DEC, chaining, Spartan integration) are still TODO.
- Any gadget that uses `cs.get()` is gated behind the `unsafe-gadgets` feature and must not be used in real proofs.

---

## License

Apache-2.0
