# CCS — Constraint System Relations

## Purpose

- **What it is**: Paper-facing relation objects `CCS`, `CE`, and `CERelaxed` that capture the norm-bounded CCS relation, the norm-bounded CCS evaluation relation, and a relaxed CE form used in folding.
- **Key property**: `CERelaxed.ofCE` promotes a CE relation into relaxed form with `slackBound := normBound`; CCS and CE carry arity and constraint/norm bounds.
- **Protocol role**: These structures are the proof-system type carriers for Section 7.1 relations; they parameterize the folding reductions Π_CCS, Π_RLC, and Π_DEC.

## Target Formulas

- \(\text{CCS}(b, \mathcal{L})\) ↔ `CCS` with `arity`, `constraints`.
- \(\text{CE}(b, \mathcal{L})\) ↔ `CE` with `arity`, `normBound`.
- \(\text{CE}_{\text{relaxed}}\) ↔ `CERelaxed` with `arity`, `slackBound`.
- \(\text{CERelaxed.ofCE}(\text{CE}) \to \text{CERelaxed}\) with \(\text{slackBound} = \text{normBound}\).

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 12 (Norm-bounded CCS), lines 457–459.
- Definition 13 (Norm-bounded CCS Evaluation Relation), lines 461–465.

## Module Mapping

| Paper concept | Lean symbol | Role |
|---------------|-------------|--------|
| Norm-bounded CCS relation | `CCS` | Definitional |
| Norm-bounded CE relation | `CE` | Definitional |
| Relaxed CE relation | `CERelaxed` | Definitional |
| CE → CERelaxed promotion | `CERelaxed.ofCE` | Definitional |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Relations | `CCS` | `arity : Nat`, `constraints : Nat` | Definitional |
| | `CE` | `arity : Nat`, `normBound : Nat` | Definitional |
| | `CERelaxed` | `arity : Nat`, `slackBound : Nat` | Definitional |
| Promotion | `CERelaxed.ofCE` | Maps CE to CERelaxed with `slackBound := normBound` | Definitional |

## Proof Obligations and Closure Plan

All obligations closed. Structures are definitional; no theorem-level proof obligations.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- **Dependencies**: None (no imports).
- **Consumers**:
  - `SuperNeo.ProofSystem.ConstraintSystem`: imports `CCS` for barrel re-export.
  - `SuperNeo.ProofSystem.Folding`, `SuperNeo.ProtocolRelations`: depend on CCS/CE relation types for folding reductions.

## Implementation Plan

Keep structures minimal; extend with relation-specific fields as protocol theorems require.

## Quality Expectations

Structures stay lean; interface docstring references spec and paper anchors.

## Acceptance Criteria

- `lake build` succeeds.
- Spec contains explicit paper anchors with line ranges.
- Consumer map entries state what each consumer uses or imports.

## Out of Scope

- Full relation predicates (matrices, homomorphisms); those live in `ProtocolRelations` and core modules.
