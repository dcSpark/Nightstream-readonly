# PolyLemmas

## Purpose

- **What it is**: Standalone polynomial utility lemmas: `eqLiftFromTable` (evaluates `Σ_x eq(x,z) · q(x)` over the Boolean hypercube), `eqLiftAllBoolean` (checks that eq-lifting recovers the original table at all Boolean points), and `schwartzZippelBoundLeOne` (executable degree-vs-set-size bound check).
- **Key property**: `schwartzZippelBoundLeOne_sound` gives `setSize ≠ 0 ∧ totalDegree ≤ setSize` from a passing check; `schwartzZippelBoundLeOne_complete` gives the converse.
- **Protocol role**: The Schwartz-Zippel bound is the foundation of sum-check soundness (Definition 6) — the probability of a random evaluation vanishing is bounded by `d/|S|`. The eq-lift property validates that MLE-based evaluation recovers table entries at Boolean-cube points (Lemma 6).

## Target Formulas

- `eqLiftFromTable(q, z) = Σ_{x∈{0,1}^ℓ} eq(x, z) · q[x]`
- `eqLiftAllBoolean(q, ℓ) = ∀ j < 2^ℓ, eqLiftFromTable(q, bits(j)) = q[j]`
- `schwartzZippelBoundLeOne(d, |S|) ↔ |S| ≠ 0 ∧ d ≤ |S|`

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Lemma 5 (Schwartz-Zippel), Appendix C, lines 733-736: `Pr[p(r) = 0] ≤ d/|S|`.
- Lemma 6 (eq-lifting), Appendix C, lines 737-740: `Σ_x eq(x,z) · q(x) = q(z)` on the Boolean hypercube.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/PolyLemmas.lean` | Lemma 5 (SZ), Lemma 6 (eq-lift) |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Eq-lift | `eqLiftFromTable` | def | Definitional | `Σ eq(x,z) · q(x)` |
| Eq-lift | `eqLiftBooleanIndicator` | def | Definitional | Single-point check |
| Eq-lift | `eqLiftBooleanIndicatorProp` | def | Definitional | Single-point proposition |
| Eq-lift | `eqLiftAllBoolean` | def | Definitional | All-points check |
| SZ bound | `schwartzZippelBoundLeOne` | def | Definitional | `d ≤ |S|` executable check |
| SZ sound | `schwartzZippelBoundLeOne_sound` | theorem | Theorem-Target | `check = true → bound holds` |
| SZ complete | `schwartzZippelBoundLeOne_complete` | theorem | Theorem-Target | `bound holds → check = true` |
| Eq-lift bridge | `eqLiftBooleanIndicator_sound` | theorem | Theorem-Target | `Bool → Prop` |
| Eq-lift bridge | `eqLiftBooleanIndicator_complete` | theorem | Theorem-Target | `Prop → Bool` |
| Eq-lift bridge | `eqLiftBooleanIndicator_eq_true_iff` | theorem | Theorem-Target | Bool↔Prop closure |
| SZ bridge | `schwartzZippelBoundLeOne_eq_true_iff_prop` | theorem | Theorem-Target | Bool↔Prop closure |
| Sanity | `polyLemmaSanity` | def | Definitional | Cross-check harness |

## Proof Obligations and Closure Plan

Executable SZ and single-point eq-lift bridges are closed. The fully universal paper-style eq-lift theorem is still a separate stronger target if downstream work ever needs it.

## Assumption Ledger

No open boundary assumptions.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/EqPoly.lean`: `eqPoly`, `bitsToFArray` for eq-polynomial evaluation.

Downstream consumers:
- `SuperNeo/Checks.lean`: uses `polyLemmaSanity` for cross-check validation.
- `SuperNeo/ProofSystem.SumCheck`: could use SZ bound for soundness argument.

## Implementation Plan

No further work required; module is proof-complete.

## Quality Expectations

Sound/complete pair must be a true biconditional: `check = true ↔ bound holds`.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- Both `schwartzZippelBoundLeOne_sound` and `schwartzZippelBoundLeOne_complete` proved.

## Out of Scope

- Probabilistic Schwartz-Zippel theorem statement (this module only checks the degree-vs-set-size precondition).
- Multi-field or generic-field SZ bounds.
