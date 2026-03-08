# ConstraintSystem — Barrel Re-export

## Purpose

- **What it is**: Barrel module that re-exports `SuperNeo.ProofSystem.ConstraintSystem.CCS` (CCS, CE, CERelaxed, CERelaxed.ofCE).
- **Key property**: Importing `SuperNeo.ProofSystem.ConstraintSystem` provides the CCS relation types without importing the submodule directly.
- **Protocol role**: Facade for proof-system consumers that need CCS/CE relation objects (Section 7.1).

## Target Formulas

- Re-export only; no new formulas. All formulas are those of `CCS.spec.md`.

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 12 (Norm-bounded CCS), lines 457–459.
- Definition 13 (Norm-bounded CCS Evaluation Relation), lines 461–465.

## Module Mapping

| Lean module | Paper section |
|-------------|---------------|
| `SuperNeo.ProofSystem.ConstraintSystem` | Barrel; re-exports CCS (Definitions 12–13) |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Barrel | re-exports from CCS | `CCS`, `CE`, `CERelaxed`, `CERelaxed.ofCE` | Definitional |

## Proof Obligations and Closure Plan

None (barrel file).

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- **Dependencies**: imports `SuperNeo.ProofSystem.ConstraintSystem.CCS`.
- **Consumers**:
  - `SuperNeo.ProofSystem.Folding`, `SuperNeo.FoldingProtocol`: use CCS types for folding reductions.
  - `SuperNeo.ProofSystem.ConstraintSystemInterface`: imports this module for interface boundary.

## Implementation Plan

Keep barrel minimal; no new definitions.

## Quality Expectations

Barrel stays a single import line; spec documents re-export scope.

## Acceptance Criteria

- `lake build` succeeds.
- Spec contains explicit paper anchors with line ranges.

## Out of Scope

- New definitions or theorems; barrel is aggregation-only.
