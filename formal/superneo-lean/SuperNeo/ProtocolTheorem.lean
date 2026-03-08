import SuperNeo.InteractiveReductions
import SuperNeo.Interp
import SuperNeo.ProofSystem.Lattice
import SuperNeo.ProofSystem.LatticePaper
import SuperNeo.ProofSystem.LatticeReductions
import SuperNeo.ProofSystem.LatticeReductionsDerived
import SuperNeo.ProofSystem.SumCheck
import SuperNeo.ProofSystem.Security

/-!
Canonical final theorem shape for the SuperNeo scaffold.

This file provides:
- one explicit assumption registry,
- completeness and knowledge-soundness statement shapes,
- one canonical theorem constructor from assumptions.
-/

namespace SuperNeo

/--
Concrete theorem-facing Schwartz-Zippel failure witness surface.

This avoids the refuted universal `interpolationAssumption` scaffold and instead
tracks a single interpolation-check failure instance.
-/
structure SchwartzZippelFailureWitness where
  xs : Array F
  ys : Array F
  coeffs : Array F
  xEval : F
  expectedEval : F
  failure : interpolationCase xs ys coeffs xEval expectedEval = false

/-- Canonical Schwartz-Zippel failure event surface (theorem-facing alias). -/
def schwartzZippelFailureEvent : Prop :=
  Nonempty SchwartzZippelFailureWitness

/-- Advantage of the Schwartz-Zippel failure event under a probability model. -/
def SchwartzZippelAdvantage
  (prob : SuperNeo.ProofSystem.ProbModel)
  (_n : Nat) : Rat :=
  prob.Pr schwartzZippelFailureEvent

/-- Theorem-facing Schwartz-Zippel advantage bound shape against an error function. -/
def SchwartzZippelAdvantageBound
  (eps : SuperNeo.ProofSystem.ErrorFn) : Prop :=
  ∀ prob : SuperNeo.ProofSystem.ProbModel, ∀ n : Nat,
    SchwartzZippelAdvantage prob n ≤ (eps n : Rat)

/-- Theorem-facing boundary package for Schwartz-Zippel with explicit error term. -/
structure SchwartzZippelBoundary where
  epsSchwartzZippel : SuperNeo.ProofSystem.ErrorFn
  negligibleEpsSchwartzZippel :
    SuperNeo.ProofSystem.IsNegligible epsSchwartzZippel
  advantageBound :
    SchwartzZippelAdvantageBound epsSchwartzZippel

/-- Paper-facing alias for lattice-parameter bundle. -/
abbrev LatticeParams := SuperNeo.ProofSystem.AjtaiParams

/-- Typed boundary for Module-SIS hardness. -/
def msisHardnessAssumption (params : LatticeParams) : Prop :=
  SuperNeo.ProofSystem.MSISHardnessAssumption params

/-- Typed boundary for Ajtai commitment binding. -/
def ajtaiBindingAssumption (params : LatticeParams) : Prop :=
  SuperNeo.ProofSystem.AjtaiBindingAssumption params

/-- Typed boundary for Ajtai relaxed binding. -/
def ajtaiRelaxedBindingAssumption
  (params : LatticeParams)
  (C : SuperNeo.SamplingCarrier) : Prop :=
  SuperNeo.ProofSystem.AjtaiRelaxedBindingAssumption params C

/--
Protocol-facing SumCheck soundness package built around the faithful
prefix-dependent full-field endpoint.

This does not try to reconstruct a `SoundnessGame` from `ctx`; it carries the
concrete aligned positive-round game that the protocol instantiation uses.
-/
structure SumcheckPrefixLundBoundary (ctx : ProtocolTargetContext) where
  game : SuperNeo.ProofSystem.Sumcheck.SoundnessGame
  instEq : game.inst = sumcheckInstanceOfContext ctx
  denominatorAligned :
    SuperNeo.sumcheckLundSoundnessDenominator game.inst = Goldilocks.q
  roundsPos : 0 < game.inst.rounds

/-- Access the faithful prefix-dependent SumCheck Lund bound from the package. -/
theorem SumcheckPrefixLundBoundary.lundBoundHolds
  {ctx : ProtocolTargetContext}
  (h : SumcheckPrefixLundBoundary ctx) :
  h.game.lundBoundHolds
    (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel h.game.inst.rounds) := by
  exact SuperNeo.ProofSystem.Sumcheck.lundSoundnessAssumptionFullFieldAlignedPosRounds_prefix
    h.game h.denominatorAligned h.roundsPos

/--
Canonical final error package:
groups all boundary error terms and one alignment witness against a shared
`ErrorModel`.
-/
structure FinalErrorPackage (params : LatticeParams) where
  /-- Minimal explicit SumCheck error boundary; theorem package is derived. -/
  sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary
  schwartzZippelBoundary : SchwartzZippelBoundary
  errorModel : SuperNeo.ProofSystem.ErrorModel
  msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params
  msisToAjtai : SuperNeo.ProofSystem.MSISToAjtaiReductions params
  aligned :
    errorModel.epsSumcheck = sumcheckError.epsSoundness ∧
    errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel ∧
    errorModel.epsMSIS = msisBoundary.epsMSIS ∧
    errorModel.epsBinding = msisToAjtai.epsBinding ∧
    errorModel.epsRelaxedBinding = msisToAjtai.epsRelaxedBinding

namespace FinalErrorPackage

/--
Canonical constructor from explicit component boundary packages, deriving the
shared `ErrorModel` internally from the component error surfaces.
-/
def ofComponentBoundaries
  {params : LatticeParams}
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (msisToAjtai : SuperNeo.ProofSystem.MSISToAjtaiReductions params) :
  FinalErrorPackage params where
  sumcheckError := sumcheckError
  schwartzZippelBoundary := schwartzZippelBoundary
  errorModel :=
    SuperNeo.ProofSystem.ErrorModel.ofComponents
      sumcheckError.epsSoundness
      msisBoundary.epsMSIS
      schwartzZippelBoundary.epsSchwartzZippel
      msisToAjtai.epsBinding
      msisToAjtai.epsRelaxedBinding
      sumcheckError.negligibleEpsSoundness
      msisBoundary.negligibleEpsMSIS
      schwartzZippelBoundary.negligibleEpsSchwartzZippel
      msisToAjtai.negligibleEpsBinding
      msisToAjtai.negligibleEpsRelaxedBinding
  msisBoundary := msisBoundary
  msisToAjtai := msisToAjtai
  aligned := by
    simp [SuperNeo.ProofSystem.ErrorModel.ofComponents]

/--
Canonical constructor from explicit boundary packages plus one alignment witness.
This is the intended boundary-level assembly point for the final theorem.
-/
def ofAlignedComponents
  {params : LatticeParams}
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (errorModel : SuperNeo.ProofSystem.ErrorModel)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (msisToAjtai : SuperNeo.ProofSystem.MSISToAjtaiReductions params)
  (hSumcheck : errorModel.epsSumcheck = sumcheckError.epsSoundness)
  (hSZ : errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel)
  (hMSIS : errorModel.epsMSIS = msisBoundary.epsMSIS)
  (hBinding : errorModel.epsBinding = msisToAjtai.epsBinding)
  (hRelaxed : errorModel.epsRelaxedBinding = msisToAjtai.epsRelaxedBinding) :
  FinalErrorPackage params where
  sumcheckError := sumcheckError
  schwartzZippelBoundary := schwartzZippelBoundary
  errorModel := errorModel
  msisBoundary := msisBoundary
  msisToAjtai := msisToAjtai
  aligned := ⟨hSumcheck, hSZ, hMSIS, hBinding, hRelaxed⟩

end FinalErrorPackage

namespace FinalErrorPackage

/--
Canonical final-error constructor specialized to the proved `paperCarrier`
strong-sampling path via `3*d ≤ params.relaxedExpansion`.
-/
def ofAlignedPaperCarrierFromThreeDLe
  {params : LatticeParams}
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (errorModel : SuperNeo.ProofSystem.ErrorModel)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (hSumcheck : errorModel.epsSumcheck = sumcheckError.epsSoundness)
  (hSZ : errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel)
  (hMSIS : errorModel.epsMSIS = msisBoundary.epsMSIS)
  (hBinding : errorModel.epsBinding = msisBoundary.epsMSIS)
  (hRelaxed : errorModel.epsRelaxedBinding = msisBoundary.epsMSIS) :
  FinalErrorPackage params :=
  ofAlignedComponents
    sumcheckError
    schwartzZippelBoundary
    errorModel
    msisBoundary
    (SuperNeo.ProofSystem.MSISToAjtaiReductions.ofPaperCarrierFromThreeDLeAndMSISBoundary
      (params := params)
      hTd
      hExpPos
      msisBoundary)
    hSumcheck
    hSZ
    hMSIS
    hBinding
    hRelaxed

end FinalErrorPackage

namespace FinalErrorPackage

/--
Canonical final-error constructor specialized to the Goldilocks Appendix B.2
paper-parameter family on the proved `paperCarrier` path.
-/
def ofGoldilocksPaperCarrier
  (messageLength : Nat)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength)) :
  FinalErrorPackage (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength) :=
  ofComponentBoundaries
    sumcheckError
    schwartzZippelBoundary
    msisBoundary
    (SuperNeo.ProofSystem.MSISToAjtaiReductions.ofGoldilocksPaperCarrierAndMSISBoundary
      messageLength
      msisBoundary)

end FinalErrorPackage

/-- Canonical final assumption registry. -/
structure FinalTheoremAssumptions (ctx : ProtocolTargetContext) where
  reduction : InteractiveReductionAssumptions ctx
  sumcheckPrefix : SumcheckPrefixLundBoundary ctx
  latticeParams : LatticeParams
  errorPackage : FinalErrorPackage latticeParams
  /--
  Concrete error-model index used to account for the faithful prefix-game
  SumCheck failure probability in the final theorem.
  -/
  sumcheckErrorIndex : Nat
  /--
  The faithful prefix-dependent SumCheck game is instantiated by the same
  concrete transcript carried by the reduction witness.
  -/
  sumcheckWitnessMatchesPrefix :
    sumcheckPrefix.game.transcript
        reduction.sumcheckTransitionWitness.transcript.challenges =
      reduction.sumcheckTransitionWitness.transcript
  /--
  Final SumCheck error accounting dominates the faithful prefix-game failure
  probability under the canonical full-field coin model at the chosen
  error-model index.
  -/
  sumcheckErrorCoversPrefix :
    sumcheckPrefix.game.advantage
        (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
          sumcheckPrefix.game.inst.rounds) ≤
      errorPackage.sumcheckError.epsSoundness sumcheckErrorIndex

namespace FinalTheoremAssumptions

/--
Canonical boundary-level constructor from the reduction bundle, faithful
protocol-facing SumCheck game package, and the consolidated final error package.
-/
def ofBoundaryPackages
  {ctx : ProtocolTargetContext}
  {params : LatticeParams}
  (reduction : InteractiveReductionAssumptions ctx)
  (sumcheckPrefix : SumcheckPrefixLundBoundary ctx)
  (errorPackage : FinalErrorPackage params)
  (sumcheckErrorIndex : Nat)
  (sumcheckWitnessMatchesPrefix :
    sumcheckPrefix.game.transcript
        reduction.sumcheckTransitionWitness.transcript.challenges =
      reduction.sumcheckTransitionWitness.transcript)
  (sumcheckErrorCoversPrefix :
    sumcheckPrefix.game.advantage
        (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
          sumcheckPrefix.game.inst.rounds) ≤
      errorPackage.sumcheckError.epsSoundness sumcheckErrorIndex) :
  FinalTheoremAssumptions ctx where
  reduction := reduction
  sumcheckPrefix := sumcheckPrefix
  latticeParams := params
  errorPackage := errorPackage
  sumcheckErrorIndex := sumcheckErrorIndex
  sumcheckWitnessMatchesPrefix := sumcheckWitnessMatchesPrefix
  sumcheckErrorCoversPrefix := sumcheckErrorCoversPrefix

/--
Canonical final-theorem constructor specialized to the proved `paperCarrier`
strong-sampling path via `3*d ≤ params.relaxedExpansion`.
-/
def ofAlignedPaperCarrierBoundaryPackages
  {ctx : ProtocolTargetContext}
  {params : LatticeParams}
  (reduction : InteractiveReductionAssumptions ctx)
  (sumcheckPrefix : SumcheckPrefixLundBoundary ctx)
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (errorModel : SuperNeo.ProofSystem.ErrorModel)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (hSumcheck : errorModel.epsSumcheck = sumcheckError.epsSoundness)
  (hSZ : errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel)
  (hMSIS : errorModel.epsMSIS = msisBoundary.epsMSIS)
  (hBinding : errorModel.epsBinding = msisBoundary.epsMSIS)
  (hRelaxed : errorModel.epsRelaxedBinding = msisBoundary.epsMSIS)
  (sumcheckErrorIndex : Nat)
  (sumcheckWitnessMatchesPrefix :
    sumcheckPrefix.game.transcript
        reduction.sumcheckTransitionWitness.transcript.challenges =
      reduction.sumcheckTransitionWitness.transcript)
  (sumcheckErrorCoversPrefix :
    sumcheckPrefix.game.advantage
        (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
          sumcheckPrefix.game.inst.rounds) ≤
      sumcheckError.epsSoundness sumcheckErrorIndex) :
  FinalTheoremAssumptions ctx :=
  ofBoundaryPackages
    reduction
    sumcheckPrefix
    (FinalErrorPackage.ofAlignedPaperCarrierFromThreeDLe
      (params := params)
      hTd
      hExpPos
      sumcheckError
      schwartzZippelBoundary
      errorModel
      msisBoundary
      hSumcheck
      hSZ
      hMSIS
      hBinding
      hRelaxed)
    sumcheckErrorIndex
    sumcheckWitnessMatchesPrefix
    sumcheckErrorCoversPrefix

/--
Canonical final-theorem constructor specialized to the paper-facing
challenge-difference route for `invDelta` together with the proved
`paperCarrier` strong-sampling path.

This is the narrowest in-repo constructor on the active protocol path: callers
provide arithmetic/Theorem-3 data, the active paper-carrier-difference
invertibility boundary, the fact that `invDelta` is a nonzero `paperCarrier`
difference, one SumCheck transition witness, and the aligned error/MSIS/Ajtai
packages.
-/
def ofAlignedPaperCarrierDiffBoundaryPackages
  {ctx : ProtocolTargetContext}
  {params : LatticeParams}
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (sumcheckPrefix : SumcheckPrefixLundBoundary ctx)
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (errorModel : SuperNeo.ProofSystem.ErrorModel)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (hSumcheck : errorModel.epsSumcheck = sumcheckError.epsSoundness)
  (hSZ : errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel)
  (hMSIS : errorModel.epsMSIS = msisBoundary.epsMSIS)
  (hBinding : errorModel.epsBinding = msisBoundary.epsMSIS)
  (hRelaxed : errorModel.epsRelaxedBinding = msisBoundary.epsMSIS)
  (sumcheckErrorIndex : Nat)
  (sumcheckWitnessMatchesPrefix :
    sumcheckPrefix.game.transcript
        hWitness.transcript.challenges =
      hWitness.transcript)
  (sumcheckErrorCoversPrefix :
    sumcheckPrefix.game.advantage
        (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
          sumcheckPrefix.game.inst.rounds) ≤
      sumcheckError.epsSoundness sumcheckErrorIndex) :
  FinalTheoremAssumptions ctx :=
  ofAlignedPaperCarrierBoundaryPackages
    (params := params)
    (InteractiveReductionAssumptions.ofPaperCarrierDiff
      hThm3 hArithmetic hDiff hNe hWitness)
    sumcheckPrefix
    hTd
    hExpPos
    sumcheckError
    schwartzZippelBoundary
    errorModel
    msisBoundary
    hSumcheck
    hSZ
    hMSIS
    hBinding
    hRelaxed
    sumcheckErrorIndex
    sumcheckWitnessMatchesPrefix
    sumcheckErrorCoversPrefix

/--
Canonical final-theorem constructor specialized to any strict low-norm
invertibility boundary whose threshold is at least `5`, together with the
proved `paperCarrier` strong-sampling path.
-/
def ofAlignedPaperCarrierLowNormBoundaryPackages
  {ctx : ProtocolTargetContext}
  {params : LatticeParams}
  {B : Nat}
  (hFive : 5 ≤ B)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hInv : lowNormInvertibilityAssumption B)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (sumcheckPrefix : SumcheckPrefixLundBoundary ctx)
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (errorModel : SuperNeo.ProofSystem.ErrorModel)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (hSumcheck : errorModel.epsSumcheck = sumcheckError.epsSoundness)
  (hSZ : errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel)
  (hMSIS : errorModel.epsMSIS = msisBoundary.epsMSIS)
  (hBinding : errorModel.epsBinding = msisBoundary.epsMSIS)
  (hRelaxed : errorModel.epsRelaxedBinding = msisBoundary.epsMSIS)
  (sumcheckErrorIndex : Nat)
  (sumcheckWitnessMatchesPrefix :
    sumcheckPrefix.game.transcript
        hWitness.transcript.challenges =
      hWitness.transcript)
  (sumcheckErrorCoversPrefix :
    sumcheckPrefix.game.advantage
        (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
          sumcheckPrefix.game.inst.rounds) ≤
      sumcheckError.epsSoundness sumcheckErrorIndex) :
  FinalTheoremAssumptions ctx :=
  ofAlignedPaperCarrierBoundaryPackages
    (params := params)
    (InteractiveReductionAssumptions.ofLowNormAtLeastFive
      hFive hThm3 hArithmetic hInv hDiff hNe hWitness)
    sumcheckPrefix
    hTd
    hExpPos
    sumcheckError
    schwartzZippelBoundary
    errorModel
    msisBoundary
    hSumcheck
    hSZ
    hMSIS
    hBinding
    hRelaxed
    sumcheckErrorIndex
    sumcheckWitnessMatchesPrefix
    sumcheckErrorCoversPrefix

end FinalTheoremAssumptions

namespace FinalTheoremAssumptions

/--
Canonical final-theorem constructor specialized to the Goldilocks Appendix B.2
paper-parameter family on the proved `paperCarrier` path.
-/
def ofGoldilocksPaperCarrierBoundaryPackages
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (reduction : InteractiveReductionAssumptions ctx)
  (sumcheckPrefix : SumcheckPrefixLundBoundary ctx)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength))
  (sumcheckErrorIndex : Nat)
  (sumcheckWitnessMatchesPrefix :
    sumcheckPrefix.game.transcript
        reduction.sumcheckTransitionWitness.transcript.challenges =
      reduction.sumcheckTransitionWitness.transcript)
  (sumcheckErrorCoversPrefix :
    sumcheckPrefix.game.advantage
        (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
          sumcheckPrefix.game.inst.rounds) ≤
      sumcheckError.epsSoundness sumcheckErrorIndex) :
  FinalTheoremAssumptions ctx :=
  ofBoundaryPackages
    reduction
    sumcheckPrefix
    (FinalErrorPackage.ofGoldilocksPaperCarrier
      messageLength
      sumcheckError
      schwartzZippelBoundary
      msisBoundary)
    sumcheckErrorIndex
    sumcheckWitnessMatchesPrefix
    sumcheckErrorCoversPrefix

/--
Canonical final-theorem constructor specialized further to the active
`paperCarrier`-difference invertibility route, using the Goldilocks Appendix
B.2 paper-parameter family.
-/
def ofGoldilocksPaperCarrierDiffBoundaryPackages
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (sumcheckPrefix : SumcheckPrefixLundBoundary ctx)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength))
  (sumcheckErrorIndex : Nat)
  (sumcheckWitnessMatchesPrefix :
    sumcheckPrefix.game.transcript
        hWitness.transcript.challenges =
      hWitness.transcript)
  (sumcheckErrorCoversPrefix :
    sumcheckPrefix.game.advantage
        (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
          sumcheckPrefix.game.inst.rounds) ≤
      sumcheckError.epsSoundness sumcheckErrorIndex) :
  FinalTheoremAssumptions ctx :=
  ofGoldilocksPaperCarrierBoundaryPackages
    messageLength
    (InteractiveReductionAssumptions.ofPaperCarrierDiff
      hThm3 hArithmetic hDiff hNe hWitness)
    sumcheckPrefix
    sumcheckError
    schwartzZippelBoundary
    msisBoundary
    sumcheckErrorIndex
    sumcheckWitnessMatchesPrefix
    sumcheckErrorCoversPrefix

/--
Canonical final-theorem constructor specialized further to the active native-bar
`paperCarrier`-difference invertibility route, using the Goldilocks Appendix
B.2 paper-parameter family and discharging the generic Theorem-3 boundary from
`thm3CoreAssumption_native`.
-/
def ofGoldilocksNativePaperCarrierDiffBoundaryPackages
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (hBarNative : ctx.bar = nativeBarMatrix)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (sumcheckPrefix : SumcheckPrefixLundBoundary ctx)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength))
  (sumcheckErrorIndex : Nat)
  (sumcheckWitnessMatchesPrefix :
    sumcheckPrefix.game.transcript
        hWitness.transcript.challenges =
      hWitness.transcript)
  (sumcheckErrorCoversPrefix :
    sumcheckPrefix.game.advantage
        (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
          sumcheckPrefix.game.inst.rounds) ≤
      sumcheckError.epsSoundness sumcheckErrorIndex) :
  FinalTheoremAssumptions ctx :=
  ofGoldilocksPaperCarrierBoundaryPackages
    messageLength
    (InteractiveReductionAssumptions.ofNativePaperCarrierDiff
      hBarNative hArithmetic hDiff hNe hWitness)
    sumcheckPrefix
    sumcheckError
    schwartzZippelBoundary
    msisBoundary
    sumcheckErrorIndex
    sumcheckWitnessMatchesPrefix
    sumcheckErrorCoversPrefix

/--
Canonical final-theorem constructor specialized further to any strict low-norm
invertibility boundary whose threshold is at least `5`, using the Goldilocks
Appendix B.2 paper-parameter family.
-/
def ofGoldilocksPaperCarrierLowNormBoundaryPackages
  {ctx : ProtocolTargetContext}
  {B : Nat}
  (messageLength : Nat)
  (hFive : 5 ≤ B)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hInv : lowNormInvertibilityAssumption B)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (sumcheckPrefix : SumcheckPrefixLundBoundary ctx)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength))
  (sumcheckErrorIndex : Nat)
  (sumcheckWitnessMatchesPrefix :
    sumcheckPrefix.game.transcript
        hWitness.transcript.challenges =
      hWitness.transcript)
  (sumcheckErrorCoversPrefix :
    sumcheckPrefix.game.advantage
        (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
          sumcheckPrefix.game.inst.rounds) ≤
      sumcheckError.epsSoundness sumcheckErrorIndex) :
  FinalTheoremAssumptions ctx :=
  ofGoldilocksPaperCarrierBoundaryPackages
    messageLength
    (InteractiveReductionAssumptions.ofLowNormAtLeastFive
      hFive hThm3 hArithmetic hInv hDiff hNe hWitness)
    sumcheckPrefix
    sumcheckError
    schwartzZippelBoundary
    msisBoundary
    sumcheckErrorIndex
    sumcheckWitnessMatchesPrefix
    sumcheckErrorCoversPrefix

end FinalTheoremAssumptions

namespace FinalTheoremAssumptions

end FinalTheoremAssumptions

/-- Access canonical SumCheck error boundary from final assumptions. -/
def FinalTheoremAssumptions.sumcheckError
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary :=
  hA.errorPackage.sumcheckError

/-- Access the protocol-facing faithful SumCheck prefix-Lund package. -/
def FinalTheoremAssumptions.sumcheckPrefixBoundary
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) : SumcheckPrefixLundBoundary ctx :=
  hA.sumcheckPrefix

/-- Identify the reduction witness transcript with the faithful prefix-game transcript. -/
def FinalTheoremAssumptions.sumcheckWitnessTranscriptEq
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.sumcheckPrefix.game.transcript
      hA.reduction.sumcheckTransitionWitness.transcript.challenges =
    hA.reduction.sumcheckTransitionWitness.transcript :=
  hA.sumcheckWitnessMatchesPrefix

/-- Access canonical Schwartz-Zippel boundary from final assumptions. -/
def FinalTheoremAssumptions.schwartzZippelBoundary
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) : SchwartzZippelBoundary :=
  hA.errorPackage.schwartzZippelBoundary

/-- Access canonical error model from final assumptions. -/
def FinalTheoremAssumptions.errorModel
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.ErrorModel :=
  hA.errorPackage.errorModel

/-- Access canonical MSIS boundary package from final assumptions. -/
def FinalTheoremAssumptions.msisBoundary
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.MSISHardnessBoundary hA.latticeParams :=
  hA.errorPackage.msisBoundary

/-- Access canonical MSIS-to-Ajtai reduction package from final assumptions. -/
def FinalTheoremAssumptions.msisToAjtai
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.MSISToAjtaiReductions hA.latticeParams :=
  hA.errorPackage.msisToAjtai

/-- Alignment between `epsSumcheck` and SumCheck soundness error. -/
def FinalTheoremAssumptions.sumcheckErrorAligned
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.errorModel.epsSumcheck = hA.sumcheckError.epsSoundness :=
  hA.errorPackage.aligned.1

/-- Alignment between `epsSchwartzZippel` and Schwartz-Zippel boundary error. -/
def FinalTheoremAssumptions.schwartzZippelErrorAligned
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.errorModel.epsSchwartzZippel = hA.schwartzZippelBoundary.epsSchwartzZippel :=
  hA.errorPackage.aligned.2.1

/-- Alignment between `epsMSIS` and MSIS boundary error. -/
def FinalTheoremAssumptions.msisErrorAligned
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.errorModel.epsMSIS = hA.msisBoundary.epsMSIS :=
  hA.errorPackage.aligned.2.2.1

/-- Alignment between `epsBinding` and Ajtai binding boundary error. -/
def FinalTheoremAssumptions.bindingErrorAligned
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.errorModel.epsBinding = hA.msisToAjtai.epsBinding :=
  hA.errorPackage.aligned.2.2.2.1

/-- Alignment between `epsRelaxedBinding` and Ajtai relaxed-binding boundary error. -/
def FinalTheoremAssumptions.relaxedBindingErrorAligned
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.errorModel.epsRelaxedBinding = hA.msisToAjtai.epsRelaxedBinding :=
  hA.errorPackage.aligned.2.2.2.2

/-- Canonical SumCheck error boundary carried by final assumptions. -/
def FinalTheoremAssumptions.sumcheckErrorBoundary
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary :=
  hA.sumcheckError

/-- Canonical SumCheck error negligibility aligned to final error accounting. -/
def FinalTheoremAssumptions.sumcheckErrorNegligible
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsSumcheck := by
  simpa [hA.sumcheckErrorAligned] using
    hA.sumcheckError.negligibleEpsSoundness

/--
Canonical SumCheck soundness-failure advantage bound on the witnessed transcript,
aligned to final error accounting.
-/
def FinalTheoremAssumptions.sumcheckAdvantageBound
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.Sumcheck.SoundnessFailureAdvantageBound
      (sumcheckInstanceOfContext ctx)
      hA.reduction.sumcheckTransitionWitness.transcript
      hA.errorModel.epsSumcheck := by
  apply sumcheckFailureAdvantageBound_of_assumptions
    (h := hA.reduction)
    (eps := hA.errorModel.epsSumcheck)
  intro n
  have hNonnegSoundness : 0 ≤ hA.sumcheckError.epsSoundness n :=
    hA.sumcheckError.nonnegEpsSoundness n
  simpa [hA.sumcheckErrorAligned] using hNonnegSoundness

/-- Faithful prefix-dependent SumCheck Lund bound carried by final assumptions. -/
def FinalTheoremAssumptions.sumcheckPrefixLundBound
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.sumcheckPrefix.game.lundBoundHolds
    (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
      hA.sumcheckPrefix.game.inst.rounds) :=
  hA.sumcheckPrefix.lundBoundHolds

/--
Faithful game-level SumCheck failure-probability bound aligned to the final
error model.
-/
def FinalTheoremAssumptions.sumcheckPrefixAdvantageBound
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.sumcheckPrefix.game.advantage
      (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
        hA.sumcheckPrefix.game.inst.rounds) ≤
    hA.errorModel.epsSumcheck hA.sumcheckErrorIndex := by
  simpa [hA.sumcheckErrorAligned] using hA.sumcheckErrorCoversPrefix

/-- Canonical Schwartz-Zippel error negligibility aligned to final error accounting. -/
def FinalTheoremAssumptions.schwartzZippelErrorNegligible
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsSchwartzZippel := by
  have hNeg :
      SuperNeo.ProofSystem.IsNegligible hA.schwartzZippelBoundary.epsSchwartzZippel :=
    hA.schwartzZippelBoundary.negligibleEpsSchwartzZippel
  simpa [hA.schwartzZippelErrorAligned] using hNeg

/-- Canonical Schwartz-Zippel advantage bound aligned to final error accounting. -/
def FinalTheoremAssumptions.schwartzZippelAdvantageBound
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SchwartzZippelAdvantageBound hA.errorModel.epsSchwartzZippel := by
  have hBound :
      SchwartzZippelAdvantageBound hA.schwartzZippelBoundary.epsSchwartzZippel :=
    hA.schwartzZippelBoundary.advantageBound
  intro prob n
  have hLe :
      SchwartzZippelAdvantage prob n ≤
        (hA.schwartzZippelBoundary.epsSchwartzZippel n : Rat) :=
    hBound prob n
  simpa [hA.schwartzZippelErrorAligned] using hLe

/-- Canonical MSIS hardness boundary extracted from the final lattice package. -/
def FinalTheoremAssumptions.msisHardnessBoundary
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) : msisHardnessAssumption hA.latticeParams :=
  SuperNeo.ProofSystem.MSISHardnessBoundary.hardnessFromFields hA.msisBoundary

/-- Canonical MSIS error negligibility aligned to final error accounting. -/
def FinalTheoremAssumptions.msisErrorNegligible
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsMSIS := by
  have hNeg :
      SuperNeo.ProofSystem.IsNegligible hA.msisBoundary.epsMSIS :=
    hA.msisBoundary.negligibleEpsMSIS
  simpa [hA.msisErrorAligned] using hNeg

/-- Canonical MSIS advantage bound aligned to final error accounting. -/
def FinalTheoremAssumptions.msisAdvantageBound
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.MSISAdvantageBound hA.latticeParams hA.errorModel.epsMSIS := by
  have hB :
      SuperNeo.ProofSystem.MSISAdvantageBound hA.latticeParams hA.msisBoundary.epsMSIS :=
    hA.msisBoundary.advantageBound
  intro prob n
  have hLe : SuperNeo.ProofSystem.MSISAdvantage prob
      (SuperNeo.ProofSystem.canonicalMSISGame hA.latticeParams) n ≤
      (hA.msisBoundary.epsMSIS n : Rat) :=
    hB prob n
  simpa [hA.msisErrorAligned] using hLe

/-- Canonical Ajtai binding boundary derived from MSIS hardness via reductions. -/
def FinalTheoremAssumptions.ajtaiBindingBoundaryPackage
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.AjtaiBindingBoundary hA.latticeParams :=
  SuperNeo.ProofSystem.ajtaiBindingBoundary_of_msis hA.msisToAjtai hA.msisHardnessBoundary

/-- Canonical Ajtai binding boundary derived from MSIS hardness via reductions. -/
def FinalTheoremAssumptions.ajtaiBindingBoundary
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) : ajtaiBindingAssumption hA.latticeParams :=
  SuperNeo.ProofSystem.AjtaiBindingBoundary.hardnessFromFields hA.ajtaiBindingBoundaryPackage

/-- Canonical Ajtai relaxed-binding boundary derived from MSIS hardness via reductions. -/
def FinalTheoremAssumptions.ajtaiRelaxedBindingBoundaryPackage
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.AjtaiRelaxedBindingBoundary
    hA.latticeParams hA.msisToAjtai.laws.samplingCarrier :=
  SuperNeo.ProofSystem.ajtaiRelaxedBindingBoundary_of_msis hA.msisToAjtai hA.msisHardnessBoundary

/-- Canonical Ajtai relaxed-binding boundary derived from MSIS hardness via reductions. -/
def FinalTheoremAssumptions.ajtaiRelaxedBindingBoundary
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  ajtaiRelaxedBindingAssumption hA.latticeParams hA.msisToAjtai.laws.samplingCarrier :=
  SuperNeo.ProofSystem.AjtaiRelaxedBindingBoundary.hardnessFromFields hA.ajtaiRelaxedBindingBoundaryPackage

/-- Canonical Ajtai binding-error negligibility aligned to final error accounting. -/
def FinalTheoremAssumptions.bindingErrorNegligible
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsBinding := by
  have hNeg :
      SuperNeo.ProofSystem.IsNegligible hA.ajtaiBindingBoundaryPackage.epsBinding :=
    hA.ajtaiBindingBoundaryPackage.negligibleEpsBinding
  simpa [hA.bindingErrorAligned] using hNeg

/-- Canonical Ajtai binding advantage bound aligned to final error accounting. -/
def FinalTheoremAssumptions.bindingAdvantageBound
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.AjtaiBindingAdvantageBound hA.latticeParams hA.errorModel.epsBinding := by
  have hB :
      SuperNeo.ProofSystem.AjtaiBindingAdvantageBound
          hA.latticeParams hA.ajtaiBindingBoundaryPackage.epsBinding :=
    hA.ajtaiBindingBoundaryPackage.advantageBound
  intro prob n
  have hLe :
      SuperNeo.ProofSystem.AjtaiBindingAdvantage prob
          (SuperNeo.ProofSystem.canonicalAjtaiBindingGame hA.latticeParams) n ≤
        (hA.ajtaiBindingBoundaryPackage.epsBinding n : Rat) :=
    hB prob n
  simpa [hA.bindingErrorAligned] using hLe

/-- Canonical Ajtai relaxed-binding-error negligibility aligned to final error accounting. -/
def FinalTheoremAssumptions.relaxedBindingErrorNegligible
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsRelaxedBinding := by
  have hNeg :
      SuperNeo.ProofSystem.IsNegligible hA.ajtaiRelaxedBindingBoundaryPackage.epsRelaxedBinding :=
    hA.ajtaiRelaxedBindingBoundaryPackage.negligibleEpsRelaxedBinding
  simpa [hA.relaxedBindingErrorAligned] using hNeg

/-- Canonical Ajtai relaxed-binding advantage bound aligned to final error accounting. -/
def FinalTheoremAssumptions.relaxedBindingAdvantageBound
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.AjtaiRelaxedBindingAdvantageBound
      hA.latticeParams hA.msisToAjtai.laws.samplingCarrier hA.errorModel.epsRelaxedBinding := by
  have hB :
      SuperNeo.ProofSystem.AjtaiRelaxedBindingAdvantageBound
          hA.latticeParams hA.msisToAjtai.laws.samplingCarrier
            hA.ajtaiRelaxedBindingBoundaryPackage.epsRelaxedBinding :=
    hA.ajtaiRelaxedBindingBoundaryPackage.advantageBound
  intro prob n
  have hLe :
      SuperNeo.ProofSystem.AjtaiRelaxedBindingAdvantage prob
          (SuperNeo.ProofSystem.canonicalAjtaiRelaxedBindingGame
            hA.latticeParams hA.msisToAjtai.laws.samplingCarrier) n ≤
        (hA.ajtaiRelaxedBindingBoundaryPackage.epsRelaxedBinding n : Rat) :=
    hB prob n
  simpa [hA.relaxedBindingErrorAligned] using hLe

/-- Canonical total-error decomposition derived from the error model plus alignments. -/
def FinalTheoremAssumptions.totalErrorDecompFromModel
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  ∀ n,
    hA.errorModel.epsTotal n =
      hA.sumcheckError.epsSoundness n +
        hA.msisBoundary.epsMSIS n +
          hA.schwartzZippelBoundary.epsSchwartzZippel n +
            hA.msisToAjtai.epsBinding n +
              hA.msisToAjtai.epsRelaxedBinding n := by
  intro n
  have hDecomp : hA.errorModel.epsTotal n =
      hA.errorModel.epsSumcheck n + hA.errorModel.epsMSIS n +
        hA.errorModel.epsSchwartzZippel n + hA.errorModel.epsBinding n +
          hA.errorModel.epsRelaxedBinding n :=
    hA.errorModel.epsTotal_decomp n
  simpa [hA.sumcheckErrorAligned, hA.msisErrorAligned, hA.schwartzZippelErrorAligned,
    hA.bindingErrorAligned, hA.relaxedBindingErrorAligned] using hDecomp

/-- Canonical total-error negligibility from final error accounting. -/
def FinalTheoremAssumptions.totalErrorNegligible
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsTotal :=
  hA.errorModel.negligibleTotal

/-- Final completeness statement shape. -/
def FinalCompletenessStatement
  (ctx : ProtocolTargetContext)
  (_hA : FinalTheoremAssumptions ctx) : Prop :=
  ceRelaxedRelation ctx

/-- Final knowledge-soundness statement shape. -/
def FinalKnowledgeSoundnessStatement
  (ctx : ProtocolTargetContext)
  (hA : FinalTheoremAssumptions ctx) : Prop :=
  strongCompositionStatement ctx ∧
  SchwartzZippelAdvantageBound hA.errorModel.epsSchwartzZippel ∧
  hA.sumcheckPrefix.game.transcript
      hA.reduction.sumcheckTransitionWitness.transcript.challenges =
    hA.reduction.sumcheckTransitionWitness.transcript ∧
  hA.sumcheckPrefix.game.lundBoundHolds
      (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
        hA.sumcheckPrefix.game.inst.rounds) ∧
  hA.sumcheckPrefix.game.advantage
      (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
        hA.sumcheckPrefix.game.inst.rounds) ≤
    hA.errorModel.epsSumcheck hA.sumcheckErrorIndex ∧
  SuperNeo.ProofSystem.MSISAdvantageBound hA.latticeParams hA.errorModel.epsMSIS ∧
  SuperNeo.ProofSystem.AjtaiBindingAdvantageBound hA.latticeParams hA.errorModel.epsBinding ∧
  SuperNeo.ProofSystem.AjtaiRelaxedBindingAdvantageBound
      hA.latticeParams hA.msisToAjtai.laws.samplingCarrier hA.errorModel.epsRelaxedBinding ∧
  (∀ n,
    hA.errorModel.epsTotal n =
      hA.sumcheckError.epsSoundness n +
        hA.msisBoundary.epsMSIS n +
          hA.schwartzZippelBoundary.epsSchwartzZippel n +
            hA.msisToAjtai.epsBinding n +
              hA.msisToAjtai.epsRelaxedBinding n) ∧
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsTotal

/-- Canonical final theorem container. -/
structure FinalTheoremShape
  (ctx : ProtocolTargetContext)
  (hA : FinalTheoremAssumptions ctx) : Prop where
  completeness : FinalCompletenessStatement ctx hA
  knowledgeSoundness : FinalKnowledgeSoundnessStatement ctx hA

/-- Canonical final theorem constructor. -/
theorem finalTheoremShape_of_assumptions
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  FinalTheoremShape ctx hA := by
  refine ⟨?_, ?_⟩
  · exact (weakComposition_of_assumptions hA.reduction).1
  · exact ⟨strongComposition_of_assumptions hA.reduction,
      hA.schwartzZippelAdvantageBound,
      hA.sumcheckWitnessTranscriptEq,
      hA.sumcheckPrefixLundBound,
      hA.sumcheckPrefixAdvantageBound,
      hA.msisAdvantageBound,
      hA.bindingAdvantageBound,
      hA.relaxedBindingAdvantageBound,
      hA.totalErrorDecompFromModel,
      hA.totalErrorNegligible⟩

/--
Canonical final theorem specialized to the proved `paperCarrier` strong-sampling
path via `3*d ≤ params.relaxedExpansion`.
-/
theorem finalTheoremShape_of_alignedPaperCarrierBoundaryPackages
  {ctx : ProtocolTargetContext}
  {params : LatticeParams}
  (reduction : InteractiveReductionAssumptions ctx)
  (sumcheckPrefix : SumcheckPrefixLundBoundary ctx)
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (errorModel : SuperNeo.ProofSystem.ErrorModel)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (hSumcheck : errorModel.epsSumcheck = sumcheckError.epsSoundness)
  (hSZ : errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel)
  (hMSIS : errorModel.epsMSIS = msisBoundary.epsMSIS)
  (hBinding : errorModel.epsBinding = msisBoundary.epsMSIS)
  (hRelaxed : errorModel.epsRelaxedBinding = msisBoundary.epsMSIS)
  (sumcheckErrorIndex : Nat)
  (sumcheckWitnessMatchesPrefix :
    sumcheckPrefix.game.transcript
        reduction.sumcheckTransitionWitness.transcript.challenges =
      reduction.sumcheckTransitionWitness.transcript)
  (sumcheckErrorCoversPrefix :
    sumcheckPrefix.game.advantage
        (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
          sumcheckPrefix.game.inst.rounds) ≤
      sumcheckError.epsSoundness sumcheckErrorIndex) :
  FinalTheoremShape ctx
    (FinalTheoremAssumptions.ofAlignedPaperCarrierBoundaryPackages
      (params := params)
      reduction
      sumcheckPrefix
      hTd
      hExpPos
      sumcheckError
      schwartzZippelBoundary
      errorModel
      msisBoundary
      hSumcheck
      hSZ
      hMSIS
      hBinding
      hRelaxed
      sumcheckErrorIndex
      sumcheckWitnessMatchesPrefix
      sumcheckErrorCoversPrefix) := by
  exact finalTheoremShape_of_assumptions _

/--
Canonical final theorem specialized to the paper-facing challenge-difference
route for `invDelta` together with the proved `paperCarrier` strong-sampling
path and the active paper-carrier-difference invertibility boundary.
-/
theorem finalTheoremShape_of_alignedPaperCarrierDiffBoundaryPackages
  {ctx : ProtocolTargetContext}
  {params : LatticeParams}
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (sumcheckPrefix : SumcheckPrefixLundBoundary ctx)
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (errorModel : SuperNeo.ProofSystem.ErrorModel)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (hSumcheck : errorModel.epsSumcheck = sumcheckError.epsSoundness)
  (hSZ : errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel)
  (hMSIS : errorModel.epsMSIS = msisBoundary.epsMSIS)
  (hBinding : errorModel.epsBinding = msisBoundary.epsMSIS)
  (hRelaxed : errorModel.epsRelaxedBinding = msisBoundary.epsMSIS)
  (sumcheckErrorIndex : Nat)
  (sumcheckWitnessMatchesPrefix :
    sumcheckPrefix.game.transcript
        hWitness.transcript.challenges =
      hWitness.transcript)
  (sumcheckErrorCoversPrefix :
    sumcheckPrefix.game.advantage
        (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
          sumcheckPrefix.game.inst.rounds) ≤
      sumcheckError.epsSoundness sumcheckErrorIndex) :
  FinalTheoremShape ctx
    (FinalTheoremAssumptions.ofAlignedPaperCarrierDiffBoundaryPackages
      (params := params)
      hThm3
      hArithmetic
      hDiff
      hNe
      hWitness
      sumcheckPrefix
      hTd
      hExpPos
      sumcheckError
      schwartzZippelBoundary
      errorModel
      msisBoundary
      hSumcheck
      hSZ
      hMSIS
      hBinding
      hRelaxed
      sumcheckErrorIndex
      sumcheckWitnessMatchesPrefix
      sumcheckErrorCoversPrefix) := by
  exact finalTheoremShape_of_assumptions _

/--
Canonical final theorem specialized to any strict low-norm invertibility
boundary whose threshold is at least `5`, together with the proved
`paperCarrier` strong-sampling path.
-/
theorem finalTheoremShape_of_alignedPaperCarrierLowNormBoundaryPackages
  {ctx : ProtocolTargetContext}
  {params : LatticeParams}
  {B : Nat}
  (hFive : 5 ≤ B)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hInv : lowNormInvertibilityAssumption B)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (sumcheckPrefix : SumcheckPrefixLundBoundary ctx)
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (errorModel : SuperNeo.ProofSystem.ErrorModel)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (hSumcheck : errorModel.epsSumcheck = sumcheckError.epsSoundness)
  (hSZ : errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel)
  (hMSIS : errorModel.epsMSIS = msisBoundary.epsMSIS)
  (hBinding : errorModel.epsBinding = msisBoundary.epsMSIS)
  (hRelaxed : errorModel.epsRelaxedBinding = msisBoundary.epsMSIS)
  (sumcheckErrorIndex : Nat)
  (sumcheckWitnessMatchesPrefix :
    sumcheckPrefix.game.transcript
        hWitness.transcript.challenges =
      hWitness.transcript)
  (sumcheckErrorCoversPrefix :
    sumcheckPrefix.game.advantage
        (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
          sumcheckPrefix.game.inst.rounds) ≤
      sumcheckError.epsSoundness sumcheckErrorIndex) :
  FinalTheoremShape ctx
    (FinalTheoremAssumptions.ofAlignedPaperCarrierLowNormBoundaryPackages
      (params := params)
      hFive
      hThm3
      hArithmetic
      hInv
      hDiff
      hNe
      hWitness
      sumcheckPrefix
      hTd
      hExpPos
      sumcheckError
      schwartzZippelBoundary
      errorModel
      msisBoundary
      hSumcheck
      hSZ
      hMSIS
      hBinding
      hRelaxed
      sumcheckErrorIndex
      sumcheckWitnessMatchesPrefix
      sumcheckErrorCoversPrefix) := by
  exact finalTheoremShape_of_assumptions _

/--
Canonical final theorem specialized to the Goldilocks Appendix B.2
paper-parameter family on the proved `paperCarrier` path.
-/
theorem finalTheoremShape_of_goldilocksPaperCarrierBoundaryPackages
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (reduction : InteractiveReductionAssumptions ctx)
  (sumcheckPrefix : SumcheckPrefixLundBoundary ctx)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength))
  (sumcheckErrorIndex : Nat)
  (sumcheckWitnessMatchesPrefix :
    sumcheckPrefix.game.transcript
        reduction.sumcheckTransitionWitness.transcript.challenges =
      reduction.sumcheckTransitionWitness.transcript)
  (sumcheckErrorCoversPrefix :
    sumcheckPrefix.game.advantage
        (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
          sumcheckPrefix.game.inst.rounds) ≤
      sumcheckError.epsSoundness sumcheckErrorIndex) :
  FinalTheoremShape ctx
    (FinalTheoremAssumptions.ofGoldilocksPaperCarrierBoundaryPackages
      messageLength
      reduction
      sumcheckPrefix
      sumcheckError
      schwartzZippelBoundary
      msisBoundary
      sumcheckErrorIndex
      sumcheckWitnessMatchesPrefix
      sumcheckErrorCoversPrefix) := by
  exact finalTheoremShape_of_assumptions _

/--
Canonical final theorem specialized to the active `paperCarrier`-difference
invertibility route and the Goldilocks Appendix B.2 paper-parameter family.
-/
theorem finalTheoremShape_of_goldilocksPaperCarrierDiffBoundaryPackages
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (sumcheckPrefix : SumcheckPrefixLundBoundary ctx)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength))
  (sumcheckErrorIndex : Nat)
  (sumcheckWitnessMatchesPrefix :
    sumcheckPrefix.game.transcript
        hWitness.transcript.challenges =
      hWitness.transcript)
  (sumcheckErrorCoversPrefix :
    sumcheckPrefix.game.advantage
        (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
          sumcheckPrefix.game.inst.rounds) ≤
      sumcheckError.epsSoundness sumcheckErrorIndex) :
  FinalTheoremShape ctx
    (FinalTheoremAssumptions.ofGoldilocksPaperCarrierDiffBoundaryPackages
      messageLength
      hThm3
      hArithmetic
      hDiff
      hNe
      hWitness
      sumcheckPrefix
      sumcheckError
      schwartzZippelBoundary
      msisBoundary
      sumcheckErrorIndex
      sumcheckWitnessMatchesPrefix
      sumcheckErrorCoversPrefix) := by
  exact finalTheoremShape_of_assumptions _

/--
Canonical final theorem specialized to the active native-bar
`paperCarrier`-difference invertibility route and the Goldilocks Appendix B.2
paper-parameter family.
-/
theorem finalTheoremShape_of_goldilocksNativePaperCarrierDiffBoundaryPackages
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (hBarNative : ctx.bar = nativeBarMatrix)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (sumcheckPrefix : SumcheckPrefixLundBoundary ctx)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength))
  (sumcheckErrorIndex : Nat)
  (sumcheckWitnessMatchesPrefix :
    sumcheckPrefix.game.transcript
        hWitness.transcript.challenges =
      hWitness.transcript)
  (sumcheckErrorCoversPrefix :
    sumcheckPrefix.game.advantage
        (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
          sumcheckPrefix.game.inst.rounds) ≤
      sumcheckError.epsSoundness sumcheckErrorIndex) :
  FinalTheoremShape ctx
    (FinalTheoremAssumptions.ofGoldilocksNativePaperCarrierDiffBoundaryPackages
      messageLength
      hBarNative
      hArithmetic
      hDiff
      hNe
      hWitness
      sumcheckPrefix
      sumcheckError
      schwartzZippelBoundary
      msisBoundary
      sumcheckErrorIndex
      sumcheckWitnessMatchesPrefix
      sumcheckErrorCoversPrefix) := by
  exact finalTheoremShape_of_assumptions _

/--
Canonical final theorem specialized to any strict low-norm invertibility
boundary whose threshold is at least `5`, together with the Goldilocks
Appendix B.2 paper-parameter family.
-/
theorem finalTheoremShape_of_goldilocksPaperCarrierLowNormBoundaryPackages
  {ctx : ProtocolTargetContext}
  {B : Nat}
  (messageLength : Nat)
  (hFive : 5 ≤ B)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hInv : lowNormInvertibilityAssumption B)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (sumcheckPrefix : SumcheckPrefixLundBoundary ctx)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength))
  (sumcheckErrorIndex : Nat)
  (sumcheckWitnessMatchesPrefix :
    sumcheckPrefix.game.transcript
        hWitness.transcript.challenges =
      hWitness.transcript)
  (sumcheckErrorCoversPrefix :
    sumcheckPrefix.game.advantage
        (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel
          sumcheckPrefix.game.inst.rounds) ≤
      sumcheckError.epsSoundness sumcheckErrorIndex) :
  FinalTheoremShape ctx
    (FinalTheoremAssumptions.ofGoldilocksPaperCarrierLowNormBoundaryPackages
      messageLength
      hFive
      hThm3
      hArithmetic
      hInv
      hDiff
      hNe
      hWitness
      sumcheckPrefix
      sumcheckError
      schwartzZippelBoundary
      msisBoundary
      sumcheckErrorIndex
      sumcheckWitnessMatchesPrefix
      sumcheckErrorCoversPrefix) := by
  exact finalTheoremShape_of_assumptions _

end SuperNeo
