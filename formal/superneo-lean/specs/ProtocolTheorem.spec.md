# ProtocolTheorem Spec

## Purpose

- **What it is**: The canonical final theorem shape for the SuperNeo formalization. Provides an explicit assumption registry (`FinalTheoremAssumptions`), completeness and knowledge-soundness statement shapes, and a canonical theorem constructor from assumptions.
- **Key property**: `FinalTheoremAssumptions ctx → FinalTheoremShape ctx` (completeness + knowledge-soundness); the knowledge statement keeps the minimal paper-facing surfaces (strong composition, per-component advantage bounds, total-error decomposition, total negligibility).
- **Protocol role**: CAPSTONE theorem. Combines all reductions from Sections 5–7 with lattice security from Appendix C.

## Target Formulas (Paper → Lean)

- `FinalTheoremShape ctx hA ↔ FinalCompletenessStatement ctx hA ∧ FinalKnowledgeSoundnessStatement ctx hA`
- `FinalTheoremAssumptions ctx → FinalTheoremShape ctx`
- `finalTheoremShape_of_assumptions hA` proves the above
- `FinalKnowledgeSoundnessStatement ctx hA` contains:
  - `strongCompositionStatement ctx`
  - equality between the faithful SumCheck game transcript and the reduction witness transcript
  - faithful prefix-dependent SumCheck Lund bound for the protocol-facing game package
  - faithful game-level SumCheck advantage bound aligned to `epsSumcheck`
  - advantage bounds for Schwartz-Zippel / MSIS / Ajtai-binding / Ajtai-relaxed-binding
  - `totalErrorDecompFromModel`
  - `IsNegligible epsTotal`
- `SchwartzZippelAdvantageBound eps ↔ ∀ prob n, SchwartzZippelAdvantage prob n ≤ (eps n : Rat)`
- `FinalTheoremAssumptions.sumcheckAdvantageBound`:
  `SoundnessFailureAdvantageBound (sumcheckInstanceOfContext ctx) witnessTranscript epsSumcheck`
  as a convenience theorem about the concrete reduction witness transcript
- `FinalTheoremAssumptions.sumcheckWitnessTranscriptEq`:
  the faithful prefix-game transcript instantiated on the reduction witness
  challenges is definitionally the reduction witness transcript
- `FinalTheoremAssumptions.sumcheckPrefixLundBound`:
  `game.lundBoundHolds (fullFieldUniformCoinProbModel game.inst.rounds)` for the
  protocol-facing `SumcheckPrefixLundBoundary ctx`
- `FinalTheoremAssumptions.sumcheckPrefixAdvantageBound`:
  `game.advantage (fullFieldUniformCoinProbModel game.inst.rounds) ≤ epsSumcheck sumcheckErrorIndex`
- `FinalErrorPackage.ofAlignedComponents`:
  canonical constructor from explicit SumCheck/SZ/MSIS/Ajtai boundary packages plus one alignment witness against a shared `ErrorModel`
- `FinalErrorPackage.ofComponentBoundaries`:
  canonical constructor from explicit SumCheck/SZ/MSIS/Ajtai boundary packages, deriving the shared `ErrorModel` internally from component error surfaces
- `FinalErrorPackage.ofAlignedPaperCarrierFromThreeDLe`:
  canonical constructor specialized to the proved `paperCarrier` strong-sampling path, deriving the internal `MSISToAjtaiReductions` package directly from the MSIS boundary plus `3*d ≤ params.relaxedExpansion`
- `FinalErrorPackage.ofGoldilocksPaperCarrier`:
  canonical constructor specialized further to the Goldilocks Appendix B.2 paper-parameter family, keeping only `messageLength` explicit while fixing `κ = 18`, `B = 2^14`, and `T = 216`; internally this now derives the shared `ErrorModel` and uses the direct Goldilocks `paperCarrier` MSIS-to-Ajtai constructor
- `FinalTheoremAssumptions.ofBoundaryPackages`:
  canonical constructor from the reduction bundle, faithful protocol-facing SumCheck prefix package, canonical final error package, chosen SumCheck error index, transcript-link witness, and game-level SumCheck error dominance witness
- `FinalTheoremAssumptions.ofAlignedPaperCarrierBoundaryPackages`:
  canonical constructor specialized to the proved `paperCarrier` strong-sampling path, deriving the internal `MSISToAjtaiReductions` package directly from the MSIS boundary plus `3*d ≤ params.relaxedExpansion`
- `FinalTheoremAssumptions.ofAlignedPaperCarrierDiffBoundaryPackages`:
  canonical constructor specialized further to the paper-facing challenge-difference route for `invDelta`, deriving the internal `InteractiveReductionAssumptions` bundle from `thm3`, arithmetic obligations, `samplingDiffSet paperCarrier ctx.invDelta`, `ctx.invDelta ≠ 0`, and one `SumCheckTransitionWitness`
- `FinalTheoremAssumptions.ofAlignedPaperCarrierLowNormBoundaryPackages`:
  canonical constructor specialized instead to a stronger strict low-norm invertibility theorem `lowNormInvertibilityAssumption B` with `5 ≤ B`, deriving the internal `InteractiveReductionAssumptions` bundle from `thm3`, arithmetic obligations, that low-norm theorem, `samplingDiffSet paperCarrier ctx.invDelta`, `ctx.invDelta ≠ 0`, and one `SumCheckTransitionWitness`
- `FinalTheoremAssumptions.ofGoldilocksPaperCarrierBoundaryPackages`:
  canonical constructor specialized to the Goldilocks Appendix B.2 paper-parameter family on the proved `paperCarrier` path
- `FinalTheoremAssumptions.ofGoldilocksPaperCarrierDiffBoundaryPackages`:
  canonical constructor specialized to the Goldilocks Appendix B.2 paper-parameter family on the active paper-facing challenge-difference path
- `FinalTheoremAssumptions.ofGoldilocksPaperCarrierLowNormBoundaryPackages`:
  canonical constructor specialized to the Goldilocks Appendix B.2 paper-parameter family on the stronger strict low-norm route
- `finalTheoremShape_of_alignedPaperCarrierBoundaryPackages`:
  direct final theorem specialized to the same proved `paperCarrier` path
- `finalTheoremShape_of_alignedPaperCarrierDiffBoundaryPackages`:
  direct final theorem specialized to the same paper-facing challenge-difference path
- `finalTheoremShape_of_alignedPaperCarrierLowNormBoundaryPackages`:
  direct final theorem specialized to the same proved `paperCarrier` path but entered through the stronger strict low-norm theorem
- `finalTheoremShape_of_goldilocksPaperCarrierBoundaryPackages`:
  direct final theorem specialized to the Goldilocks Appendix B.2 paper-parameter family on the proved `paperCarrier` path
- `finalTheoremShape_of_goldilocksPaperCarrierDiffBoundaryPackages`:
  direct final theorem specialized to the Goldilocks Appendix B.2 paper-parameter family on the active paper-facing challenge-difference path
- `finalTheoremShape_of_goldilocksPaperCarrierLowNormBoundaryPackages`:
  direct final theorem specialized to the Goldilocks Appendix B.2 paper-parameter family on the stronger strict low-norm route
- `msisHardnessAssumption params`, `ajtaiBindingAssumption params`, `ajtaiRelaxedBindingAssumption params C` (typed boundaries; relaxed binding is parameterized by sampling carrier `C`)
- `totalErrorDecompFromModel`: `epsTotal = epsSumcheck + epsMSIS + epsSchwartzZippel + epsBinding + epsRelaxedBinding`

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Anchors:
  - Section 7.6 (implied final theorem): Composition of Π_CCS, Π_RLC, Π_DEC with knowledge-soundness
  - Section 7, lines 447–596: Neo's folding scheme for CCS
  - Appendix B/C/D: Assumption accounting, lattice security (MSIS, Ajtai binding)

## Module Mapping

- Implementation: `SuperNeo.ProtocolTheorem`
- Interface: `SuperNeo.ProtocolTheoremInterface`

## Contract Surface

| Contract group | Lean surface | Preconditions | Guarantee | Role | Used by |
|---|---|---|---|---|---|
| Schwartz-Zippel | `schwartzZippelFailureEvent`, `SchwartzZippelAdvantage`, `SchwartzZippelAdvantageBound`, `SchwartzZippelBoundary` | None | Theorem-facing SZ surfaces | Definitional | — |
| Lattice | `LatticeParams`, `msisHardnessAssumption`, `ajtaiBindingAssumption`, `ajtaiRelaxedBindingAssumption params C` | None | Typed boundaries for MSIS/Ajtai; relaxed binding parameterized by `C : SamplingCarrier` | Definitional | — |
| SumCheck prefix game | `SumcheckPrefixLundBoundary ctx` | `ProtocolTargetContext` | Concrete protocol-facing SumCheck game/alignment package for the faithful prefix-dependent Lund endpoint | Definitional | `FinalTheoremAssumptions` |
| Canonical error package | `FinalErrorPackage params` | None | Bundles SumCheck/SZ/MSIS/MSIS→Ajtai error surfaces plus one consolidated alignment witness against a shared `ErrorModel` | Definitional | `FinalTheoremAssumptions` |
| Constructor | `FinalErrorPackage.ofComponentBoundaries` | explicit SumCheck/SZ/MSIS/Ajtai boundary packages | Canonical boundary-level final-error assembly deriving the shared `ErrorModel` internally from the component error surfaces | Theorem-Target | `FinalTheoremAssumptions` |
| Constructor | `FinalErrorPackage.ofAlignedComponents` | explicit component packages + one alignment witness | Canonical boundary-level final-error assembly | Theorem-Target | `FinalTheoremAssumptions` |
| Constructor | `FinalErrorPackage.ofAlignedPaperCarrierFromThreeDLe` | `3*d ≤ params.relaxedExpansion` + aligned SumCheck/SZ/MSIS components | Canonical final-error assembly specialized to the proved `paperCarrier` strong-sampling path, deriving Ajtai reduction data from MSIS hardness | Theorem-Target | `FinalTheoremAssumptions` |
| Constructor | `FinalErrorPackage.ofGoldilocksPaperCarrier` | `messageLength` + SumCheck/SZ/MSIS boundary packages | Canonical final-error assembly specialized to the Goldilocks Appendix B.2 paper-parameter family, fixing the concrete paper constants, deriving the shared `ErrorModel` internally, and leaving only message length explicit | Theorem-Target | `FinalTheoremAssumptions` |
| Final assumptions | `FinalTheoremAssumptions ctx` | None | Bundles reduction + protocol-facing SumCheck prefix game + lattice params + canonical error package; per-component alignments are derived accessors | Definitional | — |
| Constructor | `FinalTheoremAssumptions.ofBoundaryPackages` | reduction + SumCheck prefix package + canonical error package + transcript/error witnesses | Canonical boundary-level final theorem assembly | Theorem-Target | `ProofSystem.Protocol` |
| Constructor | `FinalTheoremAssumptions.ofAlignedPaperCarrierBoundaryPackages` | reduction + SumCheck prefix package + aligned SumCheck/SZ/MSIS components + `3*d ≤ params.relaxedExpansion` | Canonical boundary-level final theorem assembly specialized to the proved `paperCarrier` path, deriving Ajtai reduction data from MSIS hardness | Theorem-Target | `ProofSystem.Protocol` |
| Constructor | `FinalTheoremAssumptions.ofAlignedPaperCarrierDiffBoundaryPackages` | `thm3` + arithmetic + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta ≠ 0` + SumCheck witness + aligned SumCheck/SZ/MSIS components + `3*d ≤ params.relaxedExpansion` | Canonical boundary-level final theorem assembly specialized to the paper-facing challenge-difference path, deriving Ajtai reduction data from MSIS hardness and using the proved Goldilocks invertibility theorem internally | Theorem-Target | `ProofSystem.Protocol` |
| Constructor | `FinalTheoremAssumptions.ofAlignedPaperCarrierLowNormBoundaryPackages` | `thm3` + arithmetic + `lowNormInvertibilityAssumption B` with `5 ≤ B` + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta ≠ 0` + SumCheck witness + aligned SumCheck/SZ/MSIS components + `3*d ≤ params.relaxedExpansion` | Canonical boundary-level final theorem assembly specialized to the stronger strict low-norm route, deriving Ajtai reduction data from MSIS hardness | Theorem-Target | `ProofSystem.Protocol` |
| Constructor | `FinalTheoremAssumptions.ofGoldilocksPaperCarrierBoundaryPackages` | reduction + SumCheck prefix package + `messageLength` + SumCheck/SZ/MSIS boundary packages | Canonical boundary-level final theorem assembly specialized to the Goldilocks Appendix B.2 paper-parameter family on the proved `paperCarrier` path, deriving the internal `ErrorModel` and Ajtai reduction package | Theorem-Target | `ProofSystem.Protocol` |
| Constructor | `FinalTheoremAssumptions.ofGoldilocksPaperCarrierDiffBoundaryPackages` | `thm3` + arithmetic + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta ≠ 0` + SumCheck witness + `messageLength` + SumCheck/SZ/MSIS boundary packages | Canonical boundary-level final theorem assembly specialized to the Goldilocks Appendix B.2 paper-parameter family on the paper-facing challenge-difference path, using the proved Goldilocks invertibility theorem internally and deriving the internal `ErrorModel` | Theorem-Target | `ProofSystem.Protocol` |
| Constructor | `FinalTheoremAssumptions.ofGoldilocksNativePaperCarrierDiffBoundaryPackages` | `ctx.bar = nativeBarMatrix` + arithmetic + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta ≠ 0` + SumCheck witness + `messageLength` + SumCheck/SZ/MSIS boundary packages | Canonical boundary-level final theorem assembly specialized to the active native-bar Goldilocks Appendix B.2 paper-parameter family on the paper-facing challenge-difference path, discharging generic Thm 3 from `thm3CoreAssumption_native` | Theorem-Target | `ProofSystem.Protocol` |
| Constructor | `FinalTheoremAssumptions.ofGoldilocksPaperCarrierLowNormBoundaryPackages` | `thm3` + arithmetic + `lowNormInvertibilityAssumption B` with `5 ≤ B` + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta ≠ 0` + SumCheck witness + `messageLength` + SumCheck/SZ/MSIS boundary packages | Canonical boundary-level final theorem assembly specialized to the Goldilocks Appendix B.2 paper-parameter family on the stronger strict low-norm route, deriving the internal `ErrorModel` | Theorem-Target | `ProofSystem.Protocol` |
| Accessors | `FinalTheoremAssumptions.sumcheckPrefixBoundary`, `schwartzZippelBoundary`, `msisHardnessBoundary`, etc. | `FinalTheoremAssumptions` | Projected boundaries | Theorem-Target | — |
| SumCheck prefix endpoint | `FinalTheoremAssumptions.sumcheckPrefixLundBound` | `FinalTheoremAssumptions` | Faithful protocol-facing prefix-dependent Lund bound | Theorem-Target | — |
| SumCheck transcript link | `FinalTheoremAssumptions.sumcheckWitnessTranscriptEq` | `FinalTheoremAssumptions` | Identifies the faithful prefix-game transcript with the reduction witness transcript | Theorem-Target | — |
| SumCheck game advantage | `FinalTheoremAssumptions.sumcheckPrefixAdvantageBound` | `FinalTheoremAssumptions` | Faithful protocol-facing game-level SumCheck advantage bound aligned to `epsSumcheck` at the chosen final-theorem error index | Theorem-Target | — |
| SumCheck witness helper | `FinalTheoremAssumptions.sumcheckAdvantageBound` | `FinalTheoremAssumptions` | Convenience failure-event advantage bound for the witness transcript; not part of the primary final knowledge-soundness shape | Theorem-Target | — |
| Error negligibility | `sumcheckErrorNegligible`, `schwartzZippelErrorNegligible`, `msisErrorNegligible`, `bindingErrorNegligible`, `totalErrorNegligible` | `FinalTheoremAssumptions` | Aligned negligibility | Theorem-Target | — |
| Final theorem | `FinalTheoremShape`, `finalTheoremShape_of_assumptions`, `finalTheoremShape_of_alignedPaperCarrierBoundaryPackages`, `finalTheoremShape_of_alignedPaperCarrierDiffBoundaryPackages` | `FinalTheoremAssumptions` or aligned boundary packages | Completeness + knowledge-soundness | Theorem-Target | ProofSystem.Protocol |

## Proof Obligations and Closure Plan

All obligations closed. `finalTheoremShape_of_assumptions` proves completeness from `weakComposition_of_assumptions` and knowledge-soundness from `strongComposition_of_assumptions` plus the faithful protocol-facing SumCheck prefix-Lund endpoint, an explicit transcript link from the faithful game to the reduction witness, faithful game-level SumCheck advantage accounting aligned to `epsSumcheck` at one explicit final-theorem error index, Schwartz-Zippel/MSIS/Ajtai advantage-bound surfaces, and total-error accounting. `FinalErrorPackage.ofComponentBoundaries`, `FinalErrorPackage.ofAlignedComponents`, `FinalErrorPackage.ofAlignedPaperCarrierFromThreeDLe`, `FinalErrorPackage.ofGoldilocksPaperCarrier`, `FinalTheoremAssumptions.ofBoundaryPackages`, `FinalTheoremAssumptions.ofAlignedPaperCarrierBoundaryPackages`, `FinalTheoremAssumptions.ofAlignedPaperCarrierDiffBoundaryPackages`, `FinalTheoremAssumptions.ofAlignedPaperCarrierLowNormBoundaryPackages`, `FinalTheoremAssumptions.ofGoldilocksPaperCarrierBoundaryPackages`, `FinalTheoremAssumptions.ofGoldilocksPaperCarrierDiffBoundaryPackages`, `FinalTheoremAssumptions.ofGoldilocksNativePaperCarrierDiffBoundaryPackages`, and `FinalTheoremAssumptions.ofGoldilocksPaperCarrierLowNormBoundaryPackages` are the canonical boundary-level assembly points for the final theorem path. On the active narrowed route they derive the internal Ajtai reduction package directly from the MSIS hardness boundary; on the Goldilocks-specialized route they now derive both the shared `ErrorModel` and the Ajtai reduction package directly from the component boundary packages while leaving only message length explicit; on the active `paperCarrier`-difference path the concrete Goldilocks invertibility theorem is now proved in-repo and consumed directly rather than passed as an external boundary; and on the native-bar Goldilocks `paperCarrier`-difference route the generic Thm 3 input is discharged from `thm3CoreAssumption_native` rather than passed through the final theorem entrypoint. Alignment equalities are derived from the canonical `FinalErrorPackage` witness. The witness-level SumCheck bound remains available as a helper accessor, but it is no longer part of the primary final knowledge-soundness statement.

## Assumption Ledger

No open boundary assumptions in this module. All declarations are proved or definitional. `FinalTheoremAssumptions` is the boundary surface that consumers must instantiate; the theorem constructor is proved once the bundle is instantiated.

## Dependency and Consumer Map

- Upstream dependencies:
  - `SuperNeo/InteractiveReductions.lean`: imports `InteractiveReductionAssumptions`, `strongComposition_of_assumptions`, `weakComposition_of_assumptions`
  - `SuperNeo/Interp.lean`: imports `interpolationCase` for the theorem-facing Schwartz-Zippel failure witness surface
  - `SuperNeo/ProofSystem.Lattice.lean`: imports `AjtaiParams`, MSIS/Ajtai boundaries
  - `SuperNeo/ProofSystem.LatticeReductions.lean`: imports MSIS-to-Ajtai reductions
  - `SuperNeo/ProofSystem.SumCheck.lean`: imports `SoundnessErrorBoundary`, `lundSoundnessAssumptionFullFieldAlignedPosRounds_prefix`
  - `SuperNeo/ProofSystem.Security.lean`: imports `ErrorModel`, `IsNegligible`
- Downstream consumers:
  - `SuperNeo/ProofSystem.Protocol.lean`: uses `FinalTheoremShape`, `finalTheoremShape_of_assumptions` for the protocol proof

## Implementation Plan

1. Schwartz-Zippel and lattice boundaries defined as typed surfaces.
2. `FinalTheoremAssumptions` bundles reduction, protocol-facing SumCheck prefix-Lund package, lattice parameters, and one canonical `FinalErrorPackage`.
3. Accessors and error-negligibility proofs derive from the package and its single alignment witness.
4. `finalTheoremShape_of_assumptions` proves completeness and knowledge-soundness from the bundle.

## Quality Expectations

- No `sorry` in any theorem.
- 40+ declarations, all proved or definitional.

## Acceptance Criteria

1. `lake build` succeeds.
2. `lake exe check` succeeds.
3. All surfaces exported through the interface.

## Out of Scope

- Concrete instantiation of `FinalTheoremAssumptions`; that belongs to protocol setup.
- Proof of cryptographic primitives (MSIS, Ajtai); those are in ProofSystem.Lattice.
