# InteractiveReductions

## Purpose

- **What it is**: A compact structure `InteractiveReductionAssumptions` bundling protocol-target assumptions (via `PiDECAssumptions`, now an alias of `ProtocolTargetAssumptions`) and a sum-check transition witness, plus strong and weak composition statements for the reduction pipeline Π_RLC ∘ Π_CCS and Π_DEC ∘ Π_RLC ∘ Π_CCS.
- **Key property**: Under the assumption bundle, `strongCompositionStatement` (Π_RLC ∘ Π_CCS is strong) and `weakCompositionStatement` (Π_DEC ∘ Π_RLC ∘ Π_CCS is weak) hold; composition theorems are proved from the bundle.
- **Protocol role**: ProtocolTheorem uses composition statements. This is the composition capstone for all three reduction steps (CCS → RLC → DEC).

## Target Formulas

- `strongCompositionStatement ctx ↔ piDECKnowledgeStatement ctx`
- `weakCompositionStatement ctx ↔ ceRelaxedRelation ctx ∧ SumCheckClaimTrue (sumcheckInstanceOfContext ctx)`
- `InteractiveReductionAssumptions ctx → strongCompositionStatement ctx`
- `InteractiveReductionAssumptions ctx → weakCompositionStatement ctx`
- `InteractiveReductionAssumptions ctx + (∀ n, 0 ≤ eps n) → SoundnessFailureAdvantageBound(sumcheckInstanceOfContext ctx, witnessTranscript, eps)`

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Theorem 6 (Strong-Weak Composition), Section 6, lines 438-447.
- Definition 9 (Weak Interactive Reductions), lines 404-416.
- Definition 10 (Strong Interactive Reductions), lines 418-436.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/InteractiveReductions.lean` | Theorem 6, Definitions 9–10 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Assumptions | `InteractiveReductionAssumptions` | structure | Boundary | Bundles protocol-target assumptions + SumCheck transition witness |
| Constructor | `InteractiveReductionAssumptions.ofProtocolRelations` | def | Theorem-Target | Canonical constructor from protocol-relations assumptions by extracting the target component |
| Constructor | `InteractiveReductionAssumptions.ofPaperCarrierDiff` | def | Theorem-Target | Canonical constructor from thm3 + arithmetic + active `paperCarrier`-difference route + witness |
| Constructor | `InteractiveReductionAssumptions.ofNativePaperCarrierDiff` | def | Theorem-Target | Canonical constructor from the active native-bar `paperCarrier`-difference route + witness, discharging generic Thm 3 from `thm3CoreAssumption_native` |
| Constructor | `InteractiveReductionAssumptions.ofLowNormAtLeastFive` | def | Theorem-Target | Canonical constructor from thm3 + arithmetic + stronger strict low-norm invertibility theorem with threshold at least `5` + witness |
| Constructor | `InteractiveReductionNativeAssumptions.ofPaperCarrierDiff` | def | Theorem-Target | Native canonical constructor from the same route + witness |
| Constructor | `InteractiveReductionNativeAssumptions.ofLowNormAtLeastFive` | def | Theorem-Target | Native canonical constructor from the stronger strict low-norm route + witness |
| Statements | `strongCompositionStatement` | def | Definitional | Π_RLC ∘ Π_CCS strong |
| Statements | `weakCompositionStatement` | def | Definitional | Π_DEC ∘ Π_RLC ∘ Π_CCS weak |
| Theorems | `strongComposition_of_assumptions` | theorem | Theorem-Target | Assumptions → strong |
| Theorems | `weakComposition_of_assumptions` | theorem | Theorem-Target | Assumptions → weak |
| Theorems | `sumcheckFailureAdvantageBound_of_assumptions` | theorem | Theorem-Target | Witness-level SumCheck failure-advantage bound from reduction assumptions |
| Theorems | `sumcheckFailureAdvantageBound_of_native_assumptions` | theorem | Theorem-Target | Native-path witness-level SumCheck failure-advantage bound |

## Proof Obligations and Closure Plan

All local composition obligations are closed. `InteractiveReductionAssumptions ctx` is the remaining boundary bundle: instantiate protocol-target assumptions and an accepted sum-check transition witness for concrete protocol contexts. The composition theorems themselves are already proved from that bundle, and canonical constructors now exist from an already-built protocol-relations bundle, from the narrower `paperCarrier` difference route for `ctx.invDelta` using the proved Goldilocks invertibility theorem internally, from the active native-bar `paperCarrier` difference route by rewriting the generic Thm 3 input to `thm3CoreAssumption_native`, and from the stronger strict low-norm invertibility theorem route.

## Assumption Ledger

- `InteractiveReductionAssumptions`: boundary assumption bundling protocol-target assumptions and a transition witness.
- Closure target: instantiate via `InteractiveReductionAssumptions.ofProtocolRelations` or directly from the narrowed protocol-target path.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/PiDEC.lean`: imports `PiDECAssumptions`, `piDECKnowledgeStatement`, `ceRelaxedRelation`, `SumCheckClaimTrue`, `sumcheckInstanceOfContext`, `piDEC_of_assumptions`.
- `SuperNeo/SumCheck.lean`: constructive SumCheck truth is used directly in witness-level failure-advantage bounds.

Downstream consumers:
- `SuperNeo/ProtocolTheorem.lean`: uses composition statements for the full protocol reduction.

## Implementation Plan

No further implementation work for current module scope. Closure requires instantiating the existing reduction bundle for the protocol.

## Quality Expectations

Composition statements must match Theorem 6 (Strong-Weak Composition). Strong/weak definitions must align with Definitions 9 and 10.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.

## Out of Scope

- Concrete instantiation of `InteractiveReductionAssumptions`.
- Proof of PiCCS/PiRLC/PiDEC assumptions from cryptographic primitives.
