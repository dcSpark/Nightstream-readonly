# PiRLC — Π_RLC Proof-System Wrapper

## Purpose

- **What it is**: Proof-system wrapper that lifts `SuperNeo.PiRLC` into proof-system types; exposes `PiRLCAssumptions`, `WeakStatement`, and the theorem `weak_relaxed` linking SumCheck transition witness to the weak relation.
- **Key property**: \(\text{PiRLCAssumptions}(\text{ctx}) \wedge \text{SumCheckTransitionWitness}(\text{ctx}) \to \text{WeakStatement}(\text{ctx})\) (Π_RLC is weak).
- **Protocol role**: Provides the typed boundary for Lemma 4 (Π_RLC is a weak interactive reduction); used in the folding composition (Section 7).

## Target Formulas

- \(\text{WeakStatement}(\text{ctx}) \equiv \pi_{\text{RLC}}\) weak relation (Definition 9).
- \(\text{weak\_relaxed}: \text{PiRLCAssumptions}(\text{ctx}) \wedge \text{SumCheckTransitionWitness}(\text{ctx}) \to \text{WeakStatement}(\text{ctx})\).

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Section 7.4 (Random linear combination reduction – Π_RLC), lines 549–583.
- Lemma 4 (Π_RLC is weak), line 581.

## Module Mapping

| Paper concept | Lean symbol | Role |
|---------------|-------------|--------|
| Π_RLC assumptions | `PiRLCAssumptions` | Definitional (abbrev) |
| Weak statement | `WeakStatement` | Definitional (abbrev) |
| Weak reduction theorem | `weak_relaxed` | Theorem-Target (forwarded) |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Assumptions | `PiRLCAssumptions` | Forward to `SuperNeo.PiRLCAssumptions` | Definitional |
| Statement | `WeakStatement` | Forward to `SuperNeo.piRLCWeakStatement` | Definitional |
| Theorem | `weak_relaxed` | Assumptions + witness → weak statement | Theorem-Target |

## Proof Obligations and Closure Plan

`weak_relaxed` proved by forwarding to `SuperNeo.piRLCWeak_of_assumptions`.

## Assumption Ledger

No open boundary assumptions in this module. Assumptions are carried in `PiRLCAssumptions` (from core PiRLC).

## Dependency and Consumer Map

- **Dependencies**: imports `SuperNeo.PiRLC`, `SuperNeo.ProofSystem.Types`.
- **Consumers**:
  - `SuperNeo.ProofSystem.Folding`: imports PiRLC for barrel.
  - `SuperNeo.ProofSystem.Protocol`, `SuperNeo.FoldingProtocol`: depend on weak reduction for composition.

## Implementation Plan

Keep as thin wrapper; no new proof work beyond forwarding.

## Quality Expectations

Wrapper stays minimal; interface docstring references spec and paper anchors.

## Acceptance Criteria

- `lake build` succeeds.
- Spec contains explicit paper anchors with line ranges.
- `weak_relaxed` proved.

## Out of Scope

- Core Π_RLC construction; that lives in `SuperNeo.PiRLC`.
