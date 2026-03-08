# PiDEC

## Purpose

- **What it is**: The decomposition reduction step Π_DEC. Defines `piDECKnowledgeStatement` as the existence of `deltaInv` with `mulRq ctx.invDelta deltaInv = oneRq`, plus `ceRelaxedRelation ctx` and `SumCheckClaimTrue`.
- **Key property**: `piDEC_of_assumptions`: given `PiDECAssumptions ctx` and `SumCheckTransitionWitness ctx`, we have `piDECKnowledgeStatement ctx`. Invertibility is extracted directly from `protocolTargetProp` (via `ceRelaxedRelation`), not from a separate low-norm boundary input.
- **Protocol role**: ProtocolTheorem and FoldingProtocol depend on `piDECKnowledgeStatement` for the knowledge-soundness chain. Section 7.5 (Π_DEC) reduces norm from B to b via decomposition.

## Target Formulas

- `piDECKnowledgeStatement(ctx) ↔ ∃ deltaInv, mulRq ctx.invDelta deltaInv = oneRq ∧ ceRelaxedRelation(ctx) ∧ SumCheckClaimTrue(sumcheckInstanceOfContext ctx)`
- `piDEC_of_assumptions`: `PiDECAssumptions ctx → SumCheckTransitionWitness ctx → piDECKnowledgeStatement ctx`
- Theorem 7: Π_DEC : CE(B) → CE(b)^k is a reduction of knowledge.

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Section 7.5 (Π_DEC), lines 585-593.
- Theorem 7 (Π_DEC is reduction of knowledge), lines 594-596.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/PiDEC.lean` | Section 7.5, Theorem 7 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Assumptions | `PiDECAssumptions` | abbrev | Definitional | Alias of `ProtocolTargetAssumptions ctx` |
| Statement | `piDECKnowledgeStatement` | def | Definitional | ∃ deltaInv, inverse ∧ ceRelaxed ∧ claimTrue |
| Theorem | `piDEC_of_assumptions` | theorem | Theorem-Target | Assumptions + witness → knowledge statement |

## Proof Obligations and Closure Plan

All obligations closed at the module boundary. `piDEC_of_assumptions` and `piDEC_of_native_assumptions` are proved from `piRLCWeak_*` plus projection of `invertibleRq ctx.invDelta` from `protocolTargetProp`.

## Assumption Ledger

No extra invertibility boundary is threaded at `PiDEC` level; invertibility is already required upstream in `ProtocolTargetAssumptions`, and no separate SumCheck bundle remains here.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/PiRLC.lean`: uses `piRLCWeakStatement`, `piRLCWeak_of_assumptions`, `PiRLCAssumptions`.
- `SuperNeo/ProtocolTarget.lean`: `protocolTargetProp` carries `invertibleRq ctx.invDelta`.

Downstream consumers:
- `SuperNeo/ProtocolTheorem.lean`: depends on PiDEC for knowledge reduction chain.
- `SuperNeo/ProofSystem/Folding/PiDEC.lean`: imports PiDEC for decomposition step.

## Implementation Plan

Current scope complete at boundary level. Knowledge statement and derivation theorem proved. Invertibility boundary closure remains upstream.

## Quality Expectations

`piDECKnowledgeStatement` must match Theorem 7: inverse existence plus relaxed CE and sum-check claim. Derivation must use invertibility assumption correctly.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- `piDEC_of_assumptions` proved.

## Out of Scope

- Proof of Theorem 7 (deferred to appendix).
- Concrete invertibility bound instantiation.
