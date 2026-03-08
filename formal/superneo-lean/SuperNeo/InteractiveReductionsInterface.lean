import SuperNeo.InteractiveReductions

/-!
Contract interface for `SuperNeo.InteractiveReductions`.

Spec: `specs/InteractiveReductions.spec.md`

Paper anchors:
- Theorem 6 (Strong-Weak Composition), Section 6, lines 438-447.
- Definition 9 (Weak Interactive Reductions), lines 404-416.
- Definition 10 (Strong Interactive Reductions), lines 418-436.
-/

namespace SuperNeo

namespace InteractiveReductionsInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `strongCompositionStatement`. -/
abbrev strongCompositionStatement := SuperNeo.strongCompositionStatement

/-- [Role: Theorem-Target] Curated re-export of `weakCompositionStatement`. -/
abbrev weakCompositionStatement := SuperNeo.weakCompositionStatement

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Boundary surface `InteractiveReductionAssumptions` requiring closure. -/
abbrev InteractiveReductionAssumptions := SuperNeo.InteractiveReductionAssumptions

/--
[Role: Theorem-Target] Canonical constructor from protocol-relations assumptions
and a SumCheck transition witness.
-/
abbrev InteractiveReductionAssumptions_ofProtocolRelations
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.InteractiveReductionAssumptions.ofProtocolRelations (ctx := ctx)

/--
[Role: Theorem-Target] Canonical constructor from the paper-facing
challenge-difference route plus a SumCheck transition witness.
-/
abbrev InteractiveReductionAssumptions_ofPaperCarrierDiff
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.InteractiveReductionAssumptions.ofPaperCarrierDiff (ctx := ctx)

/--
[Role: Theorem-Target] Canonical constructor from the active native-bar
paper-facing challenge-difference route plus a SumCheck transition witness.
-/
abbrev InteractiveReductionAssumptions_ofNativePaperCarrierDiff
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.InteractiveReductionAssumptions.ofNativePaperCarrierDiff (ctx := ctx)

/--
[Role: Theorem-Target] Canonical constructor from a stronger strict low-norm
invertibility theorem with threshold at least `5`, plus a SumCheck transition
witness.
-/
def InteractiveReductionAssumptions_ofLowNormAtLeastFive
  {ctx : SuperNeo.ProtocolTargetContext}
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
  SuperNeo.InteractiveReductionAssumptions.ofLowNormAtLeastFive
    (ctx := ctx) hFive hThm3 hArithmetic hInv hDiff hNe hWitness

/--
[Role: Theorem-Target] Canonical native constructor from native
protocol-relations assumptions and a SumCheck transition witness.
-/
abbrev InteractiveReductionNativeAssumptions_ofProtocolRelations
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.InteractiveReductionNativeAssumptions.ofProtocolRelations (ctx := ctx)

/--
[Role: Theorem-Target] Canonical native constructor from the paper-facing
challenge-difference route plus a SumCheck transition witness.
-/
abbrev InteractiveReductionNativeAssumptions_ofPaperCarrierDiff
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.InteractiveReductionNativeAssumptions.ofPaperCarrierDiff (ctx := ctx)

/--
[Role: Theorem-Target] Canonical native constructor from a stronger strict
low-norm invertibility theorem with threshold at least `5`, plus a SumCheck
transition witness.
-/
def InteractiveReductionNativeAssumptions_ofLowNormAtLeastFive
  {ctx : SuperNeo.ProtocolTargetContext}
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
  SuperNeo.InteractiveReductionNativeAssumptions.ofLowNormAtLeastFive
    (ctx := ctx) hFive hBarNative hArithmetic hInv hDiff hNe hWitness

/-- [Role: Theorem-Target] Curated theorem surface `strongComposition_of_assumptions`. -/
abbrev strongComposition_of_assumptions
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.strongComposition_of_assumptions (ctx := ctx)

/-- [Role: Theorem-Target] Curated theorem surface `weakComposition_of_assumptions`. -/
abbrev weakComposition_of_assumptions
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.weakComposition_of_assumptions (ctx := ctx)

/--
[Role: Theorem-Target] Witness-level SumCheck failure-advantage bound from
interactive-reduction assumptions.
-/
abbrev sumcheckFailureAdvantageBound_of_assumptions
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.sumcheckFailureAdvantageBound_of_assumptions (ctx := ctx)

/--
[Role: Theorem-Target] Witness-level SumCheck failure-advantage bound from
native interactive-reduction assumptions.
-/
abbrev sumcheckFailureAdvantageBound_of_native_assumptions
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.sumcheckFailureAdvantageBound_of_native_assumptions (ctx := ctx)

end InteractiveReductionsInterface

end SuperNeo
