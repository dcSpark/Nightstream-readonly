import SuperNeo.PiDEC
import SuperNeo.ProofSystem.SumCheck

/-!
Composition of reduction steps (`Π_CCS`, `Π_RLC`, `Π_DEC`).
-/

namespace SuperNeo

/-- Assumptions for composed interactive reductions. -/
structure InteractiveReductionAssumptions (ctx : ProtocolTargetContext) where
  reduction : PiDECAssumptions ctx
  sumcheckTransitionWitness : SumCheckTransitionWitness ctx

/-- Native assumptions for composed interactive reductions. -/
structure InteractiveReductionNativeAssumptions (ctx : ProtocolTargetContext) where
  reduction : PiDECNativeAssumptions ctx
  sumcheckTransitionWitness : SumCheckTransitionWitness ctx

/--
Canonical constructor from relation-level assumptions plus an explicit
sum-check transition witness.
-/
def InteractiveReductionAssumptions.ofProtocolRelations
  {ctx : ProtocolTargetContext}
  (hRelations : ProtocolRelationsAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  InteractiveReductionAssumptions ctx :=
  { reduction := hRelations.target
    sumcheckTransitionWitness := hWitness }

/--
Canonical constructor from the paper-facing challenge-difference route for
`invDelta` plus an explicit SumCheck transition witness.
-/
def InteractiveReductionAssumptions.ofPaperCarrierDiff
  {ctx : ProtocolTargetContext}
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
  (hWitness : SumCheckTransitionWitness ctx) :
  InteractiveReductionAssumptions ctx :=
  ofProtocolRelations
    (ProtocolRelationsAssumptions.ofPaperCarrierDiff
      hThm3 hArithmetic hDiff hNe)
    hWitness

/--
Canonical constructor for the active native-bar paper-carrier-difference route,
discharging the generic Theorem-3 boundary from `thm3CoreAssumption_native`.
-/
def InteractiveReductionAssumptions.ofNativePaperCarrierDiff
  {ctx : ProtocolTargetContext}
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
  (hWitness : SumCheckTransitionWitness ctx) :
  InteractiveReductionAssumptions ctx :=
  ofPaperCarrierDiff
    (by simpa [hBarNative] using thm3CoreAssumption_native)
    hArithmetic
    hDiff
    hNe
    hWitness

/--
Canonical constructor from any strict low-norm invertibility boundary whose
threshold is at least `5`, specialized to the active paper-carrier-difference
route plus an explicit SumCheck transition witness.
-/
def InteractiveReductionAssumptions.ofLowNormAtLeastFive
  {ctx : ProtocolTargetContext}
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
  (hWitness : SumCheckTransitionWitness ctx) :
  InteractiveReductionAssumptions ctx :=
  ofProtocolRelations
    (ProtocolRelationsAssumptions.ofLowNormAtLeastFive
      hFive hThm3 hArithmetic hInv hDiff hNe)
    hWitness

/--
Canonical native constructor from relation-level native assumptions plus an
explicit sum-check transition witness.
-/
def InteractiveReductionNativeAssumptions.ofProtocolRelations
  {ctx : ProtocolTargetContext}
  (hRelations : ProtocolRelationsNativeAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  InteractiveReductionNativeAssumptions ctx :=
  { reduction := hRelations.target
    sumcheckTransitionWitness := hWitness }

/--
Canonical native constructor from the paper-facing challenge-difference route
for `invDelta` plus an explicit SumCheck transition witness.
-/
def InteractiveReductionNativeAssumptions.ofPaperCarrierDiff
  {ctx : ProtocolTargetContext}
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
  (hWitness : SumCheckTransitionWitness ctx) :
  InteractiveReductionNativeAssumptions ctx :=
  ofProtocolRelations
    (ProtocolRelationsNativeAssumptions.ofPaperCarrierDiff
      hBarNative hArithmetic hDiff hNe)
    hWitness

/--
Canonical native constructor from any strict low-norm invertibility boundary
whose threshold is at least `5`, specialized to the active
paper-carrier-difference route plus an explicit SumCheck transition witness.
-/
def InteractiveReductionNativeAssumptions.ofLowNormAtLeastFive
  {ctx : ProtocolTargetContext}
  {B : Nat}
  (hFive : 5 ≤ B)
  (hBarNative : ctx.bar = nativeBarMatrix)
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
  (hWitness : SumCheckTransitionWitness ctx) :
  InteractiveReductionNativeAssumptions ctx :=
  ofProtocolRelations
    (ProtocolRelationsNativeAssumptions.ofLowNormAtLeastFive
      hFive hBarNative hArithmetic hInv hDiff hNe)
    hWitness

/-- Strong composition statement (knowledge-style). -/
def strongCompositionStatement (ctx : ProtocolTargetContext) : Prop :=
  piDECKnowledgeStatement ctx

/-- Weak composition statement (relaxed CE + claim truth). -/
def weakCompositionStatement (ctx : ProtocolTargetContext) : Prop :=
  ceRelaxedRelation ctx ∧
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx)

/-- Strong composed reduction theorem. -/
theorem strongComposition_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : InteractiveReductionAssumptions ctx) :
  strongCompositionStatement ctx := by
  exact piDEC_of_assumptions h.reduction h.sumcheckTransitionWitness

/-- Weak composed reduction theorem. -/
theorem weakComposition_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : InteractiveReductionAssumptions ctx) :
  weakCompositionStatement ctx := by
  rcases strongComposition_of_assumptions h with ⟨_deltaInv, _hMul, hWeak, hClaim⟩
  exact ⟨hWeak, hClaim⟩

/-- Strong composed reduction theorem (native assumption path). -/
theorem strongComposition_of_native_assumptions
  {ctx : ProtocolTargetContext}
  (h : InteractiveReductionNativeAssumptions ctx) :
  strongCompositionStatement ctx := by
  exact piDEC_of_native_assumptions h.reduction h.sumcheckTransitionWitness

/-- Weak composed reduction theorem (native assumption path). -/
theorem weakComposition_of_native_assumptions
  {ctx : ProtocolTargetContext}
  (h : InteractiveReductionNativeAssumptions ctx) :
  weakCompositionStatement ctx := by
  rcases strongComposition_of_native_assumptions h with ⟨_deltaInv, _hMul, hWeak, hClaim⟩
  exact ⟨hWeak, hClaim⟩

/--
Witness-level SumCheck soundness-failure advantage bound derived from reduction
assumptions plus a nonnegative target error function.
-/
theorem sumcheckFailureAdvantageBound_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : InteractiveReductionAssumptions ctx)
  (eps : SuperNeo.ProofSystem.ErrorFn)
  (hEpsNonneg : ∀ n : Nat, 0 ≤ eps n) :
  SuperNeo.ProofSystem.Sumcheck.SoundnessFailureAdvantageBound
      (sumcheckInstanceOfContext ctx)
      h.sumcheckTransitionWitness.transcript
      eps := by
  intro prob n
  have hFailFalse :
      SuperNeo.ProofSystem.Sumcheck.SoundnessFailureEvent
          (sumcheckInstanceOfContext ctx)
          h.sumcheckTransitionWitness.transcript → False := by
    intro hFail
    exact hFail.2 (sumcheckSoundness_constructive
      (sumcheckInstanceOfContext ctx)
      h.sumcheckTransitionWitness.transcript
      hFail.1)
  have hLeZero :
      SuperNeo.ProofSystem.Sumcheck.SoundnessFailureAdvantage
          prob
          (sumcheckInstanceOfContext ctx)
          h.sumcheckTransitionWitness.transcript ≤ 0 := by
    unfold SuperNeo.ProofSystem.Sumcheck.SoundnessFailureAdvantage
    calc
      prob.Pr
            (SuperNeo.ProofSystem.Sumcheck.SoundnessFailureEvent
              (sumcheckInstanceOfContext ctx)
              h.sumcheckTransitionWitness.transcript)
          ≤ prob.Pr False := prob.prMonotone hFailFalse
      _ = 0 := prob.prFalse
  exact Rat.le_trans hLeZero (hEpsNonneg n)

/--
Native-path witness-level SumCheck soundness-failure advantage bound derived
from native reduction assumptions plus a nonnegative target error function.
-/
theorem sumcheckFailureAdvantageBound_of_native_assumptions
  {ctx : ProtocolTargetContext}
  (h : InteractiveReductionNativeAssumptions ctx)
  (eps : SuperNeo.ProofSystem.ErrorFn)
  (hEpsNonneg : ∀ n : Nat, 0 ≤ eps n) :
  SuperNeo.ProofSystem.Sumcheck.SoundnessFailureAdvantageBound
      (sumcheckInstanceOfContext ctx)
      h.sumcheckTransitionWitness.transcript
      eps := by
  intro prob n
  have hFailFalse :
      SuperNeo.ProofSystem.Sumcheck.SoundnessFailureEvent
          (sumcheckInstanceOfContext ctx)
          h.sumcheckTransitionWitness.transcript → False := by
    intro hFail
    exact hFail.2 (sumcheckSoundness_constructive
      (sumcheckInstanceOfContext ctx)
      h.sumcheckTransitionWitness.transcript
      hFail.1)
  have hLeZero :
      SuperNeo.ProofSystem.Sumcheck.SoundnessFailureAdvantage
          prob
          (sumcheckInstanceOfContext ctx)
          h.sumcheckTransitionWitness.transcript ≤ 0 := by
    unfold SuperNeo.ProofSystem.Sumcheck.SoundnessFailureAdvantage
    calc
      prob.Pr
            (SuperNeo.ProofSystem.Sumcheck.SoundnessFailureEvent
              (sumcheckInstanceOfContext ctx)
              h.sumcheckTransitionWitness.transcript)
          ≤ prob.Pr False := prob.prMonotone hFailFalse
      _ = 0 := prob.prFalse
  exact Rat.le_trans hLeZero (hEpsNonneg n)

end SuperNeo
