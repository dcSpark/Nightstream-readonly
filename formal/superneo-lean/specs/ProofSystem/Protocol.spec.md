# Protocol — Proof-System Capstone

## Purpose

- **What it is**: The proof-system-level protocol capstone that lifts `SuperNeo.ProtocolTheorem` into the `SuperNeo.ProofSystem` type framework. Re-exports `FinalTheoremAssumptions`, `FinalCompletenessStatement`, `FinalKnowledgeSoundnessStatement`, `FinalTheoremShape`, and accessors for all nested boundaries (faithful SumCheck prefix-Lund game package, explicit SumCheck error boundary, Schwartz-Zippel, MSIS, Ajtai, error accounting), including the newer strict low-norm invertibility entry points on the final theorem path.
- **Key property**: `hA : FinalTheoremAssumptions ctx → FinalTheoremShape ctx hA`, i.e., the canonical theorem constructor `finalTheoremShape_of_assumptions` produces the final theorem shape from the assumption registry.
- **Protocol role**: Single entrypoint for proof-system consumers that need the composed protocol theorem (Sections 4–7 and Appendix C). Aggregates Π_CCS, Π_RLC, Π_DEC, lattice boundaries, and error decomposition into one typed surface.

## Target Formulas

- `FinalTheoremShape ctx hA ↔ FinalCompletenessStatement ctx hA ∧ FinalKnowledgeSoundnessStatement ctx hA`
- `epsTotal n = epsSumcheck n + epsMSIS n + epsSchwartzZippel n + epsBinding n + epsRelaxedBinding n` (total error decomposition)
- `IsNegligible epsTotal` (all error terms negligible)

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Section 4 (Preliminaries), Definition 1, lines 275–282; Definitions 2–6, lines 286–354.
- Section 5 (Embedding products), Definitions 7–8, Theorems 3–5, lines 358–400.
- Section 6 (Interactive reductions), Definitions 9–10, Theorem 6, lines 402–445.
- Section 7 (Neo's folding scheme), Definitions 11–14, Π_CCS (7.3), Π_RLC (7.4), Π_DEC (7.5), lines 447–593.
- Appendix C (Additional background), lines 729+.

## Module Mapping

| Lean module | Paper section |
|-------------|----------------|
| `SuperNeo.ProofSystem.Protocol` | Capstone; combines Sections 4–7 and Appendix C into a single proof-system object |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|-------|-------------|------|--------|-----------|
| Params | `LatticeParams` | abbrev | Definitional | `= SuperNeo.ProofSystem.AjtaiParams` |
| Constructor | `finalErrorPackageOfComponentBoundaries` | def | Definitional | Canonical proof-system re-export of final-error boundary assembly deriving the shared `ErrorModel` internally from component error surfaces |
| Constructor | `finalErrorPackageOfAlignedComponents` | def | Definitional | Canonical proof-system re-export of final-error boundary assembly |
| Constructor | `finalErrorPackageOfAlignedPaperCarrierFromThreeDLe` | def | Definitional | Canonical proof-system re-export of final-error assembly specialized to the proved `paperCarrier` path, deriving Ajtai reduction data from the MSIS boundary |
| Constructor | `finalErrorPackageOfGoldilocksPaperCarrier` | def | Definitional | Canonical proof-system re-export of final-error assembly specialized to the Goldilocks Appendix B.2 paper-parameter family, deriving the shared `ErrorModel` internally from component boundary packages |
| Assumptions | `FinalTheoremAssumptions` | abbrev | Definitional | Assumption registry for `ctx` |
| Constructor | `finalTheoremAssumptionsOfBoundaryPackages` | def | Definitional | Canonical proof-system re-export of final theorem boundary assembly |
| Constructor | `finalTheoremAssumptionsOfAlignedPaperCarrierBoundaryPackages` | def | Definitional | Canonical proof-system re-export of final theorem boundary assembly specialized to the proved `paperCarrier` path, deriving Ajtai reduction data from the MSIS boundary |
| Constructor | `finalTheoremAssumptionsOfAlignedPaperCarrierDiffBoundaryPackages` | def | Definitional | Canonical proof-system re-export of final theorem boundary assembly specialized to the paper-facing challenge-difference path, deriving Ajtai reduction data from the MSIS boundary |
| Constructor | `finalTheoremAssumptionsOfAlignedPaperCarrierLowNormBoundaryPackages` | def | Definitional | Canonical proof-system re-export of final theorem boundary assembly specialized to the proved `paperCarrier` path through a stronger strict low-norm invertibility theorem |
| Constructor | `finalTheoremAssumptionsOfGoldilocksPaperCarrierBoundaryPackages` | def | Definitional | Canonical proof-system re-export of final theorem boundary assembly specialized to the Goldilocks Appendix B.2 paper-parameter family |
| Constructor | `finalTheoremAssumptionsOfGoldilocksPaperCarrierDiffBoundaryPackages` | def | Definitional | Canonical proof-system re-export of final theorem boundary assembly specialized to the Goldilocks Appendix B.2 paper-parameter family on the paper-facing challenge-difference path |
| Constructor | `finalTheoremAssumptionsOfGoldilocksNativePaperCarrierDiffBoundaryPackages` | def | Definitional | Canonical proof-system re-export of final theorem boundary assembly specialized to the Goldilocks Appendix B.2 paper-parameter family on the active native-bar challenge-difference path, discharging generic Thm 3 from `thm3CoreAssumption_native` |
| Constructor | `finalTheoremAssumptionsOfGoldilocksPaperCarrierLowNormBoundaryPackages` | def | Definitional | Canonical proof-system re-export of final theorem boundary assembly specialized to the Goldilocks Appendix B.2 paper-parameter family through a stronger strict low-norm invertibility theorem |
| Statements | `FinalCompletenessStatement` | abbrev | Definitional | Completeness claim shape |
| Statements | `FinalKnowledgeSoundnessStatement` | abbrev | Definitional | Knowledge-soundness claim shape |
| Shape | `FinalTheoremShape` | abbrev | Definitional | Combined theorem shape |
| Constructor | `finalTheoremShape_of_assumptions` | theorem | Theorem-Target | `hA : FinalTheoremAssumptions ctx → FinalTheoremShape ctx hA` |
| Constructor | `finalTheoremShapeOfAlignedPaperCarrierBoundaryPackages` | def | Definitional | Canonical proof-system re-export of the direct final theorem specialized to the proved `paperCarrier` path |
| Constructor | `finalTheoremShapeOfAlignedPaperCarrierDiffBoundaryPackages` | def | Definitional | Canonical proof-system re-export of the direct final theorem specialized to the paper-facing challenge-difference path |
| Constructor | `finalTheoremShapeOfAlignedPaperCarrierLowNormBoundaryPackages` | def | Definitional | Canonical proof-system re-export of the direct final theorem specialized to the proved `paperCarrier` path through a stronger strict low-norm invertibility theorem |
| Constructor | `finalTheoremShapeOfGoldilocksPaperCarrierBoundaryPackages` | def | Definitional | Canonical proof-system re-export of the direct final theorem specialized to the Goldilocks Appendix B.2 paper-parameter family |
| Constructor | `finalTheoremShapeOfGoldilocksPaperCarrierDiffBoundaryPackages` | def | Definitional | Canonical proof-system re-export of the direct final theorem specialized to the Goldilocks Appendix B.2 paper-parameter family on the paper-facing challenge-difference path |
| Constructor | `finalTheoremShapeOfGoldilocksNativePaperCarrierDiffBoundaryPackages` | def | Definitional | Canonical proof-system re-export of the direct final theorem specialized to the Goldilocks Appendix B.2 paper-parameter family on the active native-bar challenge-difference path |
| Constructor | `finalTheoremShapeOfGoldilocksPaperCarrierLowNormBoundaryPackages` | def | Definitional | Canonical proof-system re-export of the direct final theorem specialized to the Goldilocks Appendix B.2 paper-parameter family through a stronger strict low-norm invertibility theorem |
| SumCheck | `finalSumcheckPrefixBoundary`, `finalSumcheckPrefixLundBound`, `finalSumcheckWitnessTranscriptEq`, `finalSumcheckPrefixAdvantageBound`, `finalSumcheckErrorBoundary`, `finalSumcheckAdvantageBound`, etc. | def | Definitional | Accessors into the faithful protocol-facing prefix-Lund package, transcript linkage, game-level SumCheck error accounting at the chosen final-theorem error index, explicit SumCheck error boundary, and an auxiliary witness-level failure-advantage helper |
| Schwartz-Zippel | `finalSchwartzZippelBoundaryPackage`, `finalSchwartzZippelErrorNegligible`, `finalSchwartzZippelAdvantageBound`, etc. | def | Definitional | Accessors into the theorem-facing Schwartz-Zippel boundary package and its aligned error/bound surfaces |
| MSIS | `finalMSISHardnessBoundary`, `finalMSISErrorNegligible`, etc. | def | Definitional | Accessors into `hA.msisHardnessBoundary`, etc. |
| Ajtai | `finalAjtaiBindingBoundary`, `finalBindingErrorNegligible`, etc. | def | Definitional | Accessors into `hA.ajtaiBindingBoundary`, etc. |
| Error | `finalTotalErrorAligned`, `finalTotalErrorNegligible` | def | Definitional | Accessors into derived `hA.totalErrorDecompFromModel` and `hA.totalErrorNegligible` |

## Proof Obligations and Closure Plan

All obligations closed. Every accessor is a definitional re-export of `ProtocolTheorem`; `finalErrorPackageOfComponentBoundaries`, `finalErrorPackageOfAlignedComponents`, `finalErrorPackageOfAlignedPaperCarrierFromThreeDLe`, `finalErrorPackageOfGoldilocksPaperCarrier`, `finalTheoremAssumptionsOfBoundaryPackages`, `finalTheoremAssumptionsOfAlignedPaperCarrierBoundaryPackages`, `finalTheoremAssumptionsOfAlignedPaperCarrierDiffBoundaryPackages`, `finalTheoremAssumptionsOfAlignedPaperCarrierLowNormBoundaryPackages`, `finalTheoremAssumptionsOfGoldilocksPaperCarrierBoundaryPackages`, `finalTheoremAssumptionsOfGoldilocksPaperCarrierDiffBoundaryPackages`, `finalTheoremAssumptionsOfGoldilocksNativePaperCarrierDiffBoundaryPackages`, and `finalTheoremAssumptionsOfGoldilocksPaperCarrierLowNormBoundaryPackages` are the canonical proof-system assembly points for the final theorem boundary. On the active narrowed route, those constructors derive the internal Ajtai reduction package directly from the MSIS boundary; the Goldilocks-specialized route now derives both the shared `ErrorModel` and the Ajtai reduction package directly from the component boundary packages while leaving only message length explicit; the active `paperCarrier`-difference route now consumes the proved Goldilocks invertibility theorem directly rather than an external invertibility boundary; and the active native-bar `paperCarrier`-difference route now also discharges the generic Thm 3 input from `thm3CoreAssumption_native`. `finalTheoremShape_of_assumptions`, `finalTheoremShapeOfAlignedPaperCarrierBoundaryPackages`, `finalTheoremShapeOfAlignedPaperCarrierDiffBoundaryPackages`, `finalTheoremShapeOfAlignedPaperCarrierLowNormBoundaryPackages`, `finalTheoremShapeOfGoldilocksPaperCarrierBoundaryPackages`, `finalTheoremShapeOfGoldilocksPaperCarrierDiffBoundaryPackages`, `finalTheoremShapeOfGoldilocksNativePaperCarrierDiffBoundaryPackages`, and `finalTheoremShapeOfGoldilocksPaperCarrierLowNormBoundaryPackages` are exact forwarders of the corresponding `ProtocolTheorem` surfaces.

## Assumption Ledger

No open boundary assumptions in this module. Boundary assumptions live in `ProtocolTheorem`; this module re-exports them.

## Dependency and Consumer Map

- **Dependencies**: imports `SuperNeo.ProtocolTheorem`, `SuperNeo.ProofSystem.Types`, `Security`, `Lattice`, `SumCheck`, `Folding`.
- **Consumers**:
  - `SuperNeo.ProofSystem.ProtocolInterface`: imports this module for interface boundary.
  - `SuperNeo.FoldingProtocol`, `SuperNeo.ProtocolTarget`: depend on `FinalTheoremAssumptions` and `FinalTheoremShape` for composition.

## Implementation Plan

Keep capstone minimal; no new definitions beyond re-exports. All logic lives in `ProtocolTheorem` and submodules.

## Quality Expectations

Capstone stays thin; spec documents re-export scope and paper anchors. All accessors must be definitionally equal to their `ProtocolTheorem` equivalents.

## Acceptance Criteria

- `lake build` succeeds.
- Spec contains explicit paper anchors with line ranges.
- `finalTheoremShape_of_assumptions` is proved by exact forwarding.

## Out of Scope

- New definitions or theorems; capstone is aggregation-only.
- Proof of final theorem from assumptions (handled in `ProtocolTheorem`).
