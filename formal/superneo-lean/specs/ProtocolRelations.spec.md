# ProtocolRelations

## Purpose

- **What it is**: The CCS and CE relation predicates on protocol-target context. Defines `ccsRelation` (protocol target holds), `ceRelation` (CCS plus accepted SumCheck transcript), and `ceRelaxedRelation` (CCS only). Builds `sumcheckInstanceOfContext` from context and `SumCheckTransitionWitness` carrying round-consistency facts.
- **Key property**: `ceRelation ctx ↔ ccsRelation ctx ∧ ∃ tr, SumCheckAccepted (sumcheckInstanceOfContext ctx) tr`; and `ceRelation ctx → ceRelaxedRelation ctx`.
- **Protocol role**: PiCCS, PiRLC, PiDEC, and FoldingProtocol depend on these relation predicates for the Section 7 folding reductions (Π_CCS, Π_RLC, Π_DEC).

## Target Formulas

- `ccsRelation(ctx) ↔ protocolTargetProp(ctx)` (CCS = protocol target)
- `ceRelation(ctx) ↔ ccsRelation(ctx) ∧ ∃ tr, SumCheckAccepted inst tr` where `inst = sumcheckInstanceOfContext ctx`
- `ceRelaxedRelation(ctx) ↔ ccsRelation(ctx)`
- `sumcheckFullFieldDenominatorAlignment(ctx) ↔ ctx.cset.size = Goldilocks.q`
- `GoldilocksFullFieldLundBoundary.ofCsetCardinality(hCard)` packages the active Goldilocks/full-field Lund setup boundary from `hCard : ctx.cset.size = Goldilocks.q`
- `ceRelation_of_ccsRelation`: `ccsRelation ctx → SumCheckTransitionWitness ctx → ceRelation ctx`
- `ceRelation_of_ccsRelation_claimTrue`: `ccsRelation ctx → SumCheckClaimTrue inst → ceRelation ctx`
- `ceRelation_of_claimTrue`: `ProtocolRelationsAssumptions ctx → SumCheckClaimTrue inst → ceRelation ctx`
- `ceRelation_of_native_claimTrue`: `ProtocolRelationsNativeAssumptions ctx → SumCheckClaimTrue inst → ceRelation ctx`
- `ceClaimTrue_of_ce`: `ceRelation ctx → SumCheckClaimTrue inst`
- `ceClaimTrue_of_native_ce`: `ceRelation ctx → SumCheckClaimTrue inst`
- `ceRelaxedRelation_of_ce`: `ceRelation ctx → ceRelaxedRelation ctx`

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Definition 12 (Norm-bounded CCS), Section 7.1, lines 457-459.
- Definition 13 (Norm-bounded CCS Evaluation Relation), Section 7.1, lines 461-465.
- Section 7.1 (Relations), lines 449-465.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/ProtocolRelations.lean` | Section 7.1, Definitions 12–13 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Instance | `sumcheckInstanceOfContext` | def | Definitional | Builds SumCheckInstance from ctx |
| Instance | `sumcheckFullFieldDenominatorAlignment` | def | Theorem-Target | The protocol SumCheck instance satisfies the full-field Lund denominator requirement |
| Boundary | `GoldilocksFullFieldLundBoundary` | structure | Boundary | Named setup-side boundary for replaying the active Goldilocks/full-field Lund endpoint |
| Witness | `SumCheckTransitionWitness` | structure | Definitional | transcript, accepted, initialRound, roundSumStep |
| Relations | `ccsRelation` | def | Definitional | protocolTargetProp ctx |
| Relations | `ceRelation` | def | Definitional | ccsRelation ∧ ∃ tr, SumCheckAccepted |
| Relations | `ceRelaxedRelation` | def | Definitional | ccsRelation ctx |
| Assumptions | `ProtocolRelationsAssumptions` | structure | Boundary | Bundles target only |
| Assumptions | `ProtocolRelationsNativeAssumptions` | structure | Boundary | Bundles native target only |
| Constructors | `ProtocolRelationsAssumptions.ofPaperCarrierDiff`, `ProtocolRelationsNativeAssumptions.ofPaperCarrierDiff` | def | Theorem-Target | Canonical relations bundles from the paper-facing `paperCarrier`-difference route on the active Goldilocks path |
| Constructors | `ProtocolRelationsAssumptions.ofLowNormAtLeastFive`, `ProtocolRelationsNativeAssumptions.ofLowNormAtLeastFive` | def | Theorem-Target | Canonical relations bundles from a stronger strict low-norm invertibility theorem with threshold at least `5` |
| Theorems | `ccsRelation_of_assumptions` | theorem | Theorem-Target | Assumptions → ccsRelation |
| Theorems | `ccsRelation_of_native_assumptions` | theorem | Theorem-Target | Native assumptions → ccsRelation |
| Theorems | `ccsRelation_iff_protocolTargetProp` | theorem | Theorem-Target | `ccsRelation ↔ protocolTargetProp` |
| Theorems | `ceRelation_iff` | theorem | Theorem-Target | `ceRelation ↔ ccsRelation ∧ ∃ tr, accepted` |
| Theorems | `ceRelaxedRelation_iff` | theorem | Theorem-Target | `ceRelaxedRelation ↔ ccsRelation` |
| Theorems | `ceRelation_of_ccsRelation` | theorem | Theorem-Target | CCS + witness → ceRelation |
| Theorems | `ceRelation_of_ccsRelation_claimTrue` | theorem | Theorem-Target | CCS + claimTrue → ceRelation |
| Theorems | `ceRelation_of_assumptions` | theorem | Theorem-Target | Assumptions + witness → ceRelation |
| Theorems | `ceRelation_of_native_assumptions` | theorem | Theorem-Target | Native assumptions + witness → ceRelation |
| Theorems | `ceRelation_of_claimTrue` | theorem | Theorem-Target | Assumptions + claimTrue → ceRelation |
| Theorems | `ceRelation_of_native_claimTrue` | theorem | Theorem-Target | Native assumptions + claimTrue → ceRelation |
| Theorems | `ceClaimTrue_of_ce` | theorem | Theorem-Target | ceRelation → claimTrue |
| Theorems | `ceClaimTrue_of_native_ce` | theorem | Theorem-Target | ceRelation → claimTrue |
| Theorems | `ceRelaxedRelation_of_ce` | theorem | Theorem-Target | ceRelation → ceRelaxedRelation |
| Theorems | `sumcheckFullFieldDenominatorAlignment_iff` | theorem | Theorem-Target | `sumcheckFullFieldDenominatorAlignment ctx ↔ ctx.cset.size = Goldilocks.q` |
| Constructors | `GoldilocksFullFieldLundBoundary.ofCsetCardinality` | def | Theorem-Target | Builds the named Goldilocks/Lund setup boundary from `ctx.cset.size = Goldilocks.q` |
| Theorems | `GoldilocksFullFieldLundBoundary.csetCardinality_eq` | theorem | Theorem-Target | Recover `ctx.cset.size = Goldilocks.q` from the named setup boundary |
| Witness | `SumCheckTransitionWitness.accepted_exists` | theorem | Theorem-Target | Witness → ∃ tr, accepted |

## Proof Obligations

- The theorem-native relation surfaces are `ccsRelation`, `ceRelation`, and `ceRelaxedRelation`.
- `ProtocolRelationsAssumptions` and `ProtocolRelationsNativeAssumptions` bundle only the upstream protocol-target boundary needed to construct those relations.
- Claim-true and witness bridges factor through the relation predicates themselves rather than duplicating separate relation-specific assumption bundles.
- Canonical constructors exist from an already-built protocol-target bundle, from the paper-facing `paperCarrier` difference route for `ctx.invDelta`, and from the stronger strict low-norm invertibility route.

## Assumption Ledger

- `ProtocolRelationsAssumptions` bundles `ProtocolTargetAssumptions`.
- `ProtocolRelationsNativeAssumptions` bundles `ProtocolTargetNativeAssumptions`.
- No separate SumCheck boundary bundle is introduced in this module; SumCheck data enters through `SumCheckTransitionWitness` and the relation bridges.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/ProtocolTarget.lean`: imports `protocolTargetProp`, `ProtocolTargetAssumptions`, `ProtocolTargetNativeAssumptions`, `ProtocolTargetContext`.
- `SuperNeo/SumCheck.lean`: imports `SumCheckInstance`, `SumCheckTranscript`, `SumCheckAccepted`, `SumCheckClaimTrue`, `sumcheckSoundness_constructive`, `sumcheckCompleteness_constructive`.

Downstream consumers:
- `SuperNeo/PiCCS.lean`: uses `ceRelation`, `ceRelation_of_ccsRelation`, `ceClaimTrue_of_ce`, `SumCheckTransitionWitness`, `sumcheckInstanceOfContext`.
- `SuperNeo/PiRLC.lean`: uses `ceRelaxedRelation`, `ceRelaxedRelation_of_ce`, `piCCSStrongStatement`.
- `SuperNeo/PiDEC.lean`: uses `ceRelaxedRelation`, `piRLCWeakStatement`.
- `SuperNeo/FoldingProtocol.lean`: imports ProtocolRelations for folding relation predicates.
- `SuperNeo/ProtocolReduction.lean`: imports ProtocolRelations.

## Design Notes

Assumption bundling is used only to carry upstream protocol-target closure into the relation constructors. The theorem-facing targets remain the relation predicates and their witness/claim-true bridges.

## Quality Expectations

Relation definitions must match paper CCS/CE semantics. Soundness/completeness bridges (`ceRelation_of_ccsRelation_claimTrue`, `ceClaimTrue_of_ce`) must be proved.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- All relation theorems proved.

## Out of Scope

- Generic standalone SumCheck redesign beyond the accepted SuperNeo-path closure.
