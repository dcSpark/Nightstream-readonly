# Interp

## Purpose

- **What it is**: A compact interpolation module providing the proposition `interpolationProp` (pointwise interpolation/evaluation agreement) and the executable checker `interpolationCase` with sound/complete bridges, used as an obligation carrier for protocol-level arithmetic.
- **Key property**: `interpolationCase_sound`: `interpolationCase = true → interpolationProp` and `interpolationCase_complete`: `interpolationProp → interpolationCase = true`.
- **Protocol role**: Interpolation obligations arise when the folding protocol needs to verify that a polynomial evaluates correctly at a given point. Arithmetic obligation statements should rely on local `interpolationProp` hypotheses or `interpolationCase_eq_true_iff`; the legacy universal `interpolationAssumption` surface is refuted and not part of the live theorem path.

## Target Formulas

- `interpolationProp(xs, ys, coeffs, xEval, expectedEval) ↔ |xs| = |ys| ∧ |coeffs| = |xs| ∧ expectedEval = xEval`

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Definition 6 (Sum-check), Section 4, lines 352-355: polynomial evaluation checks that interpolation supports.
- Section 7.3 (Π_CCS), lines 440-470: evaluation agreement obligations in CCS folding.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/Interp.lean` | Infrastructure (supports Sections 7.3-7.4) |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Proposition | `interpolationProp` | def | Definitional | Shape + evaluation agreement |
| Boundary | `interpolationAssumption` | def | Legacy Boundary | Universal interpolation claim (refuted as stated) |
| Executable | `interpolationCase` | def | Definitional | `decide interpolationProp` |
| Sound | `interpolationCase_sound` | theorem | Theorem-Target | `Bool → Prop` |
| Complete | `interpolationCase_complete` | theorem | Theorem-Target | `Prop → Bool` |
| Bridge | `interpolationCase_eq_true_iff` | theorem | Theorem-Target | Bool↔Prop closure |
| Structure | `interpolationProp_sizes` | theorem | Theorem-Target | extracts size equalities |
| Structure | `interpolationProp_eval_eq` | theorem | Theorem-Target | extracts evaluation equality |
| Refutation | `not_interpolationAssumption` | theorem | Theorem-Target | universal boundary is false as stated |

## Proof Obligations and Closure Plan

Sound/complete bridges are closed.

Open obligations:
- `interpolationAssumption` should not be pursued as a theorem target in its current form; it is refuted in-module.
- Downstream users should rely on local `interpolationProp` hypotheses or `interpolationCase_eq_true_iff` instead.

## Assumption Ledger

- `interpolationAssumption` [Legacy Boundary / Refuted]: false as currently stated; do not thread it as a real closure target.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/Field.lean`: `F` type for field elements.

Downstream consumers:
- `SuperNeo/ArithmeticObligations.lean`: uses `interpolationProp` for arithmetic checks.

## Implementation Plan

Full interpolation correctness (Lagrange interpolation formalization) is out of scope for this scaffold module. If needed later, it must be introduced under a different, semantically correct proposition.

## Quality Expectations

Sound/complete pair must form a true biconditional. The module must not silently assume nontrivial mathematical content; the old universal boundary is explicitly refuted.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- `interpolationCase_sound` and `interpolationCase_complete` both proved.

## Out of Scope

- Lagrange interpolation algorithm.
- Polynomial evaluation at arbitrary points.
- Error-correcting code connections.
