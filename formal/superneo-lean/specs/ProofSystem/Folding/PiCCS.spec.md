# ProofSystem/Folding/PiCCS.spec.md

## Purpose

- **What it is**: Proof-system-level wrapper that lifts the `SuperNeo.PiCCS` strong interactive reduction into the `ProofSystem.Folding` type framework.
- **Key property**: `soundness_relations` derives the strong relation statement from `PiCCSAssumptions` and a sum-check transition witness → `StrongStatement ctx`.
- **Protocol role**: Provides the typed proof-system façade for Π_CCS consumed by the folding barrel and `ProtocolTheorem`.

## Target Formulas

- `PiCCSAssumptions ctx → SumCheckTransitionWitness ctx → StrongStatement ctx` = strong soundness for Π_CCS.

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Section 7.3 (Π_CCS Strong Interactive Reduction), Lemma 3, lines 481-548.

## Module Mapping

| Paper concept | Lean symbol | Role |
|---------------|-------------|--------|
| Π_CCS assumptions | `PiCCSAssumptions` | Definitional (abbrev of `SuperNeo.PiCCSAssumptions`) |
| Strong statement | `StrongStatement` | Definitional (abbrev of `SuperNeo.piCCSStrongStatement`) |
| Strong soundness | `soundness_relations` | Theorem-Target |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Core | `PiCCSAssumptions` | Assumption bundle for Π_CCS | Definitional |
| Core | `StrongStatement` | Strong relation output | Definitional |
| Theorem | `soundness_relations` | `PiCCSAssumptions → SumCheckTransitionWitness → StrongStatement` | Theorem-Target |

## Proof Obligations and Closure Plan

- `soundness_relations`: forwards to `SuperNeo.piCCSStrong_of_assumptions`.
- No additional obligations in this module.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- **Dependencies**: `SuperNeo.PiCCS`, `SuperNeo.ProofSystem.Types`.
- **Consumers**:
  - `SuperNeo.ProofSystem.Folding`: imports PiCCS as part of the folding barrel.
  - `SuperNeo.ProofSystem.Protocol`: uses PiCCS soundness in the protocol capstone.

## Implementation Plan

- Stable thin wrapper. No implementation changes expected.

## Quality Expectations

- File stays under 25 lines.
- No logic duplication — delegates entirely to `SuperNeo.PiCCS`.

## Acceptance Criteria

- `lake build` succeeds.
- `soundness_relations` type-checks without sorry.

## Out of Scope

- Concrete sum-check round analysis (lives in `SuperNeo.PiCCS`).
