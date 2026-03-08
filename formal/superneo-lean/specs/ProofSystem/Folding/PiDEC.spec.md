# PiDEC — Π_DEC Proof-System Wrapper

## Purpose

- **What it is**: Proof-system wrapper that lifts `SuperNeo.PiDEC` into proof-system types; exposes `PiDECAssumptions`, `FinalStatement`, and the theorem `final_of_assumption` linking SumCheck transition witness to the final knowledge statement.
- **Key property**: \(\text{PiDECAssumptions}(\text{ctx}) \wedge \text{SumCheckTransitionWitness}(\text{ctx}) \to \text{FinalStatement}(\text{ctx})\) (Π_DEC is a reduction of knowledge).
- **Protocol role**: Provides the typed boundary for Theorem 7 (Π_DEC is a reduction of knowledge); used in the folding composition (Section 7).

## Target Formulas

- \(\text{FinalStatement}(\text{ctx}) \equiv \pi_{\text{DEC}}\) knowledge statement (Definition 5).
- \(\text{final\_of\_assumption}: \text{PiDECAssumptions}(\text{ctx}) \wedge \text{SumCheckTransitionWitness}(\text{ctx}) \to \text{FinalStatement}(\text{ctx})\).

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Section 7.5 (Decomposition reduction – Π_DEC), lines 585–593.
- Theorem 7 (Π_DEC is a reduction of knowledge), line 593.

## Module Mapping

| Paper concept | Lean symbol | Role |
|---------------|-------------|--------|
| Π_DEC assumptions | `PiDECAssumptions` | Definitional (abbrev) |
| Final statement | `FinalStatement` | Definitional (abbrev) |
| Reduction-of-knowledge theorem | `final_of_assumption` | Theorem-Target (forwarded) |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Assumptions | `PiDECAssumptions` | Forward to `SuperNeo.PiDECAssumptions` | Definitional |
| Statement | `FinalStatement` | Forward to `SuperNeo.piDECKnowledgeStatement` | Definitional |
| Theorem | `final_of_assumption` | Assumptions + witness → final statement | Theorem-Target |

## Proof Obligations and Closure Plan

`final_of_assumption` proved by forwarding to `SuperNeo.piDEC_of_assumptions`.

## Assumption Ledger

No open boundary assumptions in this module. Assumptions are carried in `PiDECAssumptions` (from core PiDEC).

## Dependency and Consumer Map

- **Dependencies**: imports `SuperNeo.PiDEC`, `SuperNeo.ProofSystem.Types`.
- **Consumers**:
  - `SuperNeo.ProofSystem.Folding`: imports PiDEC for barrel.
  - `SuperNeo.ProofSystem.Protocol`, `SuperNeo.FoldingProtocol`: depend on reduction-of-knowledge for composition.

## Implementation Plan

Keep as thin wrapper; no new proof work beyond forwarding.

## Quality Expectations

Wrapper stays minimal; interface docstring references spec and paper anchors.

## Acceptance Criteria

- `lake build` succeeds.
- Spec contains explicit paper anchors with line ranges.
- `final_of_assumption` proved.

## Out of Scope

- Core Π_DEC construction; that lives in `SuperNeo.PiDEC`.
