# PiCCS

## Purpose

- **What it is**: The strong interactive-reduction step Π_CCS. Defines `piCCSStrongStatement` as the conjunction of `ceRelation ctx` and `SumCheckClaimTrue (sumcheckInstanceOfContext ctx)`.
- **Key property**: `piCCSStrong_of_ce`: given `ceRelation ctx`, we have `piCCSStrongStatement ctx`; `piCCSStrong_of_assumptions` is the canonical wrapper from assumptions and witness.
- **Protocol role**: PiRLC depends on `piCCSStrongStatement` and `piCCSStrong_of_assumptions` for the strong→weak composition (Theorem 6). Section 7.3 (Π_CCS) reduces CCS instances to CE instances via sum-check.

## Target Formulas

- `piCCSStrongStatement(ctx) ↔ ceRelation(ctx) ∧ SumCheckClaimTrue(sumcheckInstanceOfContext ctx)`
- `piCCSStrong_of_ce`: `ceRelation ctx → piCCSStrongStatement ctx`
- `piCCSStrong_of_assumptions`: `PiCCSAssumptions ctx → SumCheckTransitionWitness ctx → piCCSStrongStatement ctx`
- Strong reduction (Lemma 3): Π_CCS : CCS^K × CE^k → CE^{K+k} is strong for φ projecting commitments.

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Section 7.3 (Π_CCS), lines 481-548.
- Lemma 3 (Π_CCS is strong), lines 545-546.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/PiCCS.lean` | Section 7.3, Lemma 3 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Assumptions | `PiCCSAssumptions` | abbrev | Definitional | Alias of `ProtocolTargetAssumptions ctx` |
| Statement | `piCCSStrongStatement` | def | Definitional | ceRelation ∧ SumCheckClaimTrue |
| Theorem | `piCCSStrong_of_ce` | theorem | Theorem-Target | CE relation → strong statement |
| Theorem | `piCCSStrong_of_assumptions` | theorem | Theorem-Target | Assumptions + witness → strong statement |

## Proof Obligations and Closure Plan

All obligations closed for the current boundary target. `piCCSStrong_of_ce` is the theorem-native entrypoint; `piCCSStrong_of_assumptions` is the canonical wrapper from `protocolTargetProp_of_assumptions` and an accepted transition witness. No separate SumCheck boundary bundle is threaded on this path.

## Assumption Ledger

No open local boundary assumptions in this module. `PiCCSAssumptions` is a direct alias to `ProtocolTargetAssumptions` from upstream; SumCheck truth is discharged from the accepted witness rather than assumed separately here.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/ProtocolTarget.lean`: uses `protocolTargetProp_of_assumptions`, `ceRelation`, `SumCheckTransitionWitness`, `sumcheckInstanceOfContext`, `ProtocolTargetAssumptions`.
- `SuperNeo/SumCheck.lean`: uses `sumcheckSoundness_constructive`.

Downstream consumers:
- `SuperNeo/PiRLC.lean`: uses `piCCSStrongStatement`, `piCCSStrong_of_assumptions`, `PiCCSAssumptions`.
- `SuperNeo/ProofSystem/Folding/PiCCS.lean`: imports PiCCS for strong reduction step.

## Implementation Plan

Current boundary scope complete. Strong statement and derivation theorem proved from protocol-target assumptions plus an accepted SumCheck witness.

## Quality Expectations

`piCCSStrongStatement` must match Lemma 3: CE relation plus sum-check claim truth. Derivation must thread assumptions correctly.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- `piCCSStrong_of_ce` proved.
- `piCCSStrong_of_assumptions` proved.

## Out of Scope

- Full protocol execution (ProofSystem layer).
- Probabilistic strong-reduction proof (Lemma 3 proof deferred to appendix).
