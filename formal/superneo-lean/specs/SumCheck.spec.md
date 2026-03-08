# SumCheck

## Purpose

- **What it is**: The interactive sum-check protocol formalization, defining `SumCheckInstance` (claimed value, rounds, max degree), `SumCheckTranscript` (challenges, round polynomials), a stronger standalone scaffold predicate `sumcheckAcceptedCore`, and a paper-facing verifier predicate `sumcheckVerifierAccepted`. The historical `sumcheckAccepted` surface remains the executable standalone scaffold acceptance used by the honest-transcript closure path.
- **Scope note**: Accepted as paper-faithful and proof-complete for the SuperNeo protocol dependency chain. The standalone core remains table/MLE-specialized and is not claimed to be a fully generic `SumCheck(T; Q)` Definition-6 library formalization.
- **Key property**: `sumcheckAccepted inst tr` implies structural properties: `tr.roundPolys.size = inst.rounds`, `tr.challenges.size = tr.roundPolys.size`, and each round's `p(0) + p(1) = eval(p_{prev}, r_{prev})`.
- **Protocol role**: Sum-check is the interactive reduction backbone of SuperNeo. Section 7.3 (Π_CCS) and Section 7.4 (Π_RLC) both invoke sum-check to reduce multivariate polynomial claims to point-evaluation queries, which are then handled by MLE evaluation.

## Target Formulas

- `p_0(0) + p_0(1) = v` (initial round sum = claimed value)
- `p_{i+1}(0) + p_{i+1}(1) = p_i(r_i)` (round transition)
- `deg(p_i) ≤ maxDegree` (paper-facing degree bound)
- `|p_i| = maxDegree + 1` (normalized executable encoding)
- `|challenges| = |roundPolys| = rounds` (transcript shape)

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Definition 6 (Sum-check protocol), Section 4, lines 352-355.
- Section 7.3 (Π_CCS), lines 440-470: sum-check invocation in CCS folding.
- Section 7.4 (Π_RLC), lines 471-489: sum-check invocation in RLC.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/SumCheck.lean` | Definition 6 (sum-check) |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Structures | `SumCheckInstance` | structure | Definitional | `rounds`, `maxDegree`, `domainSize`, `claimedValue` |
| Structures | `SumCheckTranscript` | structure | Definitional | `challenges`, `roundPolys` |
| Structures | `SumCheckClaim` | structure | Definitional | Verifier-claim object packaging transcript witness + verifier predicates |
| Structures | `SumCheckStatement` | structure | Definitional | Paper-facing statement object: parameter/degree constraints, hypercube table, size `2^rounds`, and sum-equals-claim |
| Evaluation | `sumcheckEvalPoly` | def | Definitional | Horner-form univariate eval |
| Evaluation | `sumcheckTableSum` | def | Definitional | Hypercube-table sum surface |
| Predicates | `sumcheckRoundConsistent` | def | Definitional | Transcript shape check |
| Predicates | `sumcheckRoundPolyShape` | def | Definitional | Polynomial coefficient count |
| Predicates | `sumcheckRoundPolyDegreeLe` | def | Definitional | Paper-facing coefficient-array degree bound |
| Predicates | `sumcheckRoundDegrees` | def | Definitional | Per-round paper-facing degree bound |
| Predicates | `sumcheckRoundShapes` | def | Definitional | All round polys shaped |
| Predicates | `sumcheckFoldConsistent` | def | Definitional | Round-to-round transition |
| Predicates | `sumcheckInitialRoundConsistent` | def | Definitional | `p_0(0) + p_0(1) = v` |
| Predicates | `sumcheckFinalClaimConsistent` | def | Definitional | Legacy scaffold helper (not part of acceptance endpoint) |
| Predicates | `sumcheckFinalOracleConsistentWithTable` | def | Definitional | Shape-safe final-oracle consistency indexed by a raw table (no sum-equals-claim field required) |
| Acceptance | `sumcheckAcceptedCore` | def | Definitional | Standalone scaffold acceptance (parameter, degree-compatibility, round shape/transition, initial, fold) |
| Acceptance | `sumcheckVerifierAccepted` | def | Definitional | Paper-facing verifier predicate (round consistency, degree bound, initial, fold) |
| Acceptance | `sumcheckAccepted` | def | Definitional | Executable standalone scaffold acceptance; equal to `sumcheckAcceptedCore` |
| Acceptance | `sumcheckAcceptedForTable` | def | Definitional | `sumcheckAcceptedCore ∧ sumcheckFinalOracleConsistentWithTable inst table tr` |
| Claim | `sumcheckParameterConsistent` | def | Definitional | `maxDegree ≤ domainSize` |
| Claim | `sumcheckDegreeCompatible` | def | Definitional | Internal completeness restriction of the current table-based standalone model: `rounds = 0 ∨ 0 < maxDegree` |
| Claim | `sumcheckBaseClaimTrue` | def | Definitional | Legacy base claim: `parameterConsistent ∧ degreeCompatible` |
| Claim | `sumcheckPaperClaimTrue` | def | Definitional | Standalone statement-existence surface: `Nonempty (SumCheckStatement inst)` |
| Claim | `sumcheckClaimTrue` | def | Definitional | Alias to the standalone statement-existence surface |
| Soundness bound | `sumcheckLundSoundnessNumerator` | def | Definitional | Paper-style numerator `ℓ·d` with `ℓ = rounds`, `d = maxDegree` |
| Soundness bound | `sumcheckLundSoundnessDenominator` | def | Definitional | Paper-style denominator `|K|` modeled as `domainSize` |
| Soundness bound | `sumcheckLundSoundnessBound` | def | Definitional | Pair `(ℓ·d, |K|)` corresponding to paper bound `ℓ·d/|K|` |
| Extraction | `sumcheckAccepted_rounds_eq` | theorem | Theorem-Target | `|roundPolys| = rounds` |
| Extraction | `sumcheckAccepted_challenges_eq` | theorem | Theorem-Target | `|challenges| = |roundPolys|` |
| Extraction | `sumcheckAccepted_fold_step` | theorem | Theorem-Target | Fold-step identity |
| Extraction | `sumcheckAccepted_initial_round` | theorem | Theorem-Target | Initial round check |
| Extraction | `sumcheckAccepted_round_sum_step` | theorem | Theorem-Target | Round-sum transition |
| Rejection | `sumcheckAccepted_not_of_challenge_size_ne` | theorem | Theorem-Target | Reject bad challenge count |
| Rejection | `sumcheckAccepted_not_of_roundpoly_size_ne` | theorem | Theorem-Target | Reject bad poly count |
| Rejection | `sumcheckAccepted_not_of_bad_round_shape` | theorem | Theorem-Target | Reject bad poly shape |
| Rejection | `sumcheckAccepted_not_of_bad_initial_round` | theorem | Theorem-Target | Reject bad initial round |
| Closure | `sumcheckFinalOracleConsistent_iff_withTable` | theorem | Theorem-Target | Bridge between statement-indexed and table-indexed endpoint relations |
| Closure | `sumcheckVerifierAccepted_of_accepted` | theorem | Theorem-Target | Scaffold acceptance implies the paper-facing verifier predicate |
| Closure | `sumcheckAccepted_of_acceptedForTable` | theorem | Theorem-Target | Fixed-table acceptance + table-sum equality reconstructs statement-indexed acceptance |
| Closure | `sumcheckAccepted_parameter_consistent` | theorem | Theorem-Target | Accepted transcript enforces parameter consistency |
| Closure | `sumcheckAccepted_degree_compatible` | theorem | Theorem-Target | Accepted transcript enforces degree compatibility |
| Closure | `sumcheckSoundness_constructive` | theorem | Theorem-Target | Constructive structural closure for the standalone scaffold (`accepted -> claimTrue`) |
| Closure | `sumcheckStructuralSoundness_constructive` | abbrev | Theorem-Target | Preferred structural alias for the standalone constructive soundness theorem |
| Closure | `sumcheckCompleteness_from_baseClaim_constructive` | theorem | Theorem-Target | Honest-transcript existence from legacy base claim |
| Closure | `sumcheckHonestTranscript_accepted_of_baseClaim` | theorem | Theorem-Target | Direct accepted proof for canonical honest transcript from base claim |
| Closure | `sumcheckPaperClaimTrue_of_baseClaim_constructive` | theorem | Theorem-Target | Build paper statement witness from legacy base claim |
| Closure | `sumcheckPaperClaimTrue_of_accepted` | theorem | Theorem-Target | Build paper statement witness from accepted transcript |
| Closure | `sumcheckPaperClaimTrue_iff_baseClaim` | theorem | Theorem-Target | Explicitly exposes that the standalone claim surface collapses to scaffold well-formedness |
| Closure | `sumcheckClaimTrue_iff_baseClaim` | theorem | Theorem-Target | Same collapse for the public standalone alias |
| Closure | `sumcheckFinalOracleConsistent_of_baseClaim_constructive` | theorem | Theorem-Target | Constructive final-oracle consistency for canonical statement + honest transcript |
| Closure | `sumcheckStatementTranscriptConsistent_of_baseClaim_constructive` | theorem | Theorem-Target | Constructive existence of statement/transcript consistency witness from base claim |
| Closure | `sumcheckStatementTranscriptConsistent_of_accepted_sameTranscript` | theorem | Theorem-Target | For a chosen `stmt`, accepted + final-oracle consistency at same `tr` implies consistency |
| Closure | `sumcheckCompleteness_constructive` | theorem | Theorem-Target | Canonical standalone statement-existence completeness (`claimTrue -> ∃ accepted transcript`) |
| Closure | `sumcheckStructuralCompleteness_constructive` | abbrev | Theorem-Target | Preferred structural alias for the standalone constructive completeness theorem |
| Closure | `sumcheckAssumptions_constructive` | def | Theorem-Target | Canonical constructive `SumCheckAssumptions` package |
| Boundary | `SumcheckSoundnessAssumption` | def | Boundary | `accepted → claimTrue` |
| Boundary | `SumcheckCompletenessAssumption` | def | Boundary | `claimTrue → ∃ tr, accepted` |
| Boundary | `SumCheckAssumptions` | structure | Boundary | Bundle of both |
| Bridge | `sumcheckClaimTrue_of_soundness` | theorem | Theorem-Target | Conditional soundness |

## Proof Obligations and Closure Plan

Structural extraction/rejection theorems are closed.

Constructive closure status:
- `sumcheckSoundness_constructive` closes `SumcheckSoundnessAssumption` on the standalone structural path `sumcheckAccepted -> sumcheckClaimTrue`; the paper-facing probabilistic soundness theorem lives in the proof-system game layer.
- `sumcheckVerifierAccepted` is the paper-facing verifier contract in this module; `sumcheckAccepted` remains the stronger executable scaffold acceptance used by the standalone honest-transcript machinery.
- `sumcheckCompleteness_constructive` closes `SumcheckCompletenessAssumption` on the standalone statement-existence path `sumcheckClaimTrue -> ∃ tr, sumcheckAccepted inst tr`.
- Preferred public naming for the standalone scaffold is `sumcheckStructuralSoundness_constructive` / `sumcheckStructuralCompleteness_constructive`; the older names remain the underlying theorem names.
- `sumcheckCompleteness_from_baseClaim_constructive` remains the executable honest-transcript constructor from base claim shape.
- Statement/transcript-consistency status: closed constructively via
  `sumcheckFinalOracleConsistent_of_baseClaim_constructive`,
  `sumcheckStatementTranscriptConsistent_of_baseClaim_constructive`, and
  `sumcheckStatementTranscriptConsistent_of_accepted_sameTranscript`.
- Same-transcript closure status: the exported paper contract uses `sumcheckAccepted` plus an explicit final-oracle-consistency hypothesis; internal `...Closed` helper surfaces remain available in the implementation but are no longer part of the public contract surface.
- Game-surface status: `sumcheckFinalOracleConsistentWithTable` and `sumcheckAcceptedForTable`
  provide a non-tautological endpoint surface for probabilistic soundness games where
  false claims are modeled by `sumcheckTableSum table ≠ claimedValue`.

## Assumption Ledger

- `SumcheckSoundnessAssumption` [Boundary-surface, Constructively Closed]: closed on the standalone scaffold semantics `sumcheckAccepted -> sumcheckClaimTrue`.
- `SumcheckCompletenessAssumption` [Boundary-surface, Constructively Closed]: closed on the standalone statement-existence `claimTrue -> ∃ tr, accepted` semantics.
- Remaining paper-level gaps:
  - the standalone core statement object is still table/MLE-based rather than a fully generic `SumCheck(T; Q)` witness;
  - probabilistic/adversarial semantics are not encoded in this core scaffold and live in the proof-system security layers.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/MLE.lean`: MLE evaluators for sum-check claims.
- `SuperNeo/EqPoly.lean`: `eqPoly` for sum-check polynomial construction.

Downstream consumers:
- `SuperNeo/PiCCS.lean`: uses `SumCheckInstance`, `sumcheckAccepted` for Π_CCS.
- `SuperNeo/PiRLC.lean`: uses sum-check for Π_RLC.
- `SuperNeo/ProofSystem.SumCheck`: uses `SumCheckAssumptions` for proof-system composition.

## Implementation Plan

Current scope complete for the standalone scaffold plus the paper-facing verifier view. Explicit `...Closed` helper surfaces are retained only as internal implementation scaffolding.

## Quality Expectations

Acceptance predicate must be a conjunction of independently testable components. Extraction theorems must provide one-step access to each component without pattern-matching on the conjunction.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- Constructive closures (`sumcheckSoundness_constructive`, `sumcheckCompleteness_constructive`) are proved.

## Out of Scope

- Full probabilistic soundness proof (Schwartz-Zippel) for a non-scaffolded interactive model.
- Interactive oracle proof (IOP) formalization.
- Concrete sum-check examples (live in `Checks.lean`).
