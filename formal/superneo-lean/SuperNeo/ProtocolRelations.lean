import SuperNeo.ProtocolTarget
import SuperNeo.SumCheck

/-!
CCS/CE relation layer.

This module defines paper-facing relation predicates on top of the protocol
context and ties them to the protocol-target and SumCheck boundaries.
-/

namespace SuperNeo

/-- Build a SumCheck instance from protocol-target context fields. -/
def sumcheckInstanceOfContext (ctx : ProtocolTargetContext) : SumCheckInstance :=
  { rounds := ctx.kSplit
    maxDegree := ctx.m.size
    domainSize := ctx.cset.size
    claimedValue := ct ctx.invDelta }

/--
The protocol SumCheck instance is aligned with the full Goldilocks field
denominator required by the full-field Lund endpoint.
-/
def sumcheckFullFieldDenominatorAlignment
  (ctx : ProtocolTargetContext) : Prop :=
  SuperNeo.sumcheckLundSoundnessDenominator (sumcheckInstanceOfContext ctx) =
    Goldilocks.q

theorem sumcheckFullFieldDenominatorAlignment_iff
  {ctx : ProtocolTargetContext} :
  sumcheckFullFieldDenominatorAlignment ctx ↔
    ctx.cset.size = Goldilocks.q := by
  simp [sumcheckFullFieldDenominatorAlignment, sumcheckInstanceOfContext,
    SuperNeo.sumcheckLundSoundnessDenominator]

/--
Minimal setup-side boundary for replaying the active Goldilocks/full-field Lund
endpoint on one protocol context.
-/
structure GoldilocksFullFieldLundBoundary (ctx : ProtocolTargetContext) where
  denominatorAligned : sumcheckFullFieldDenominatorAlignment ctx

namespace GoldilocksFullFieldLundBoundary

/--
Canonical setup boundary from the concrete challenge-set cardinality equality
used by the active Goldilocks route.
-/
def ofCsetCardinality
  {ctx : ProtocolTargetContext}
  (hCard : ctx.cset.size = Goldilocks.q) :
  GoldilocksFullFieldLundBoundary ctx :=
  ⟨(sumcheckFullFieldDenominatorAlignment_iff).2 hCard⟩

/--
Recover the concrete challenge-set cardinality equality from the named setup
boundary.
-/
theorem csetCardinality_eq
  {ctx : ProtocolTargetContext}
  (h : GoldilocksFullFieldLundBoundary ctx) :
  ctx.cset.size = Goldilocks.q :=
  (sumcheckFullFieldDenominatorAlignment_iff).1 h.denominatorAligned

end GoldilocksFullFieldLundBoundary

/-- Explicit SumCheck witness carrying the transition facts used by reductions. -/
structure SumCheckTransitionWitness (ctx : ProtocolTargetContext) where
  transcript : SumCheckTranscript
  accepted : SumCheckAccepted (sumcheckInstanceOfContext ctx) transcript
  initialRound :
    sumcheckInitialRoundConsistent (sumcheckInstanceOfContext ctx) transcript
  roundSumStep :
    ∀ i : Nat,
      i + 1 < transcript.roundPolys.size →
        sumcheckEvalPoly (transcript.roundPolys[i + 1]!) 0 +
            sumcheckEvalPoly (transcript.roundPolys[i + 1]!) 1 =
          sumcheckEvalPoly (transcript.roundPolys[i]!) (transcript.challenges[i]!)

theorem SumCheckTransitionWitness.accepted_exists
  {ctx : ProtocolTargetContext}
  (h : SumCheckTransitionWitness ctx) :
  ∃ tr : SumCheckTranscript,
    SumCheckAccepted (sumcheckInstanceOfContext ctx) tr := by
  exact ⟨h.transcript, h.accepted⟩

/-- CCS relation: protocol target holds. -/
def ccsRelation (ctx : ProtocolTargetContext) : Prop :=
  protocolTargetProp ctx

/-- CE relation: CCS relation plus an accepted SumCheck transcript witness. -/
def ceRelation (ctx : ProtocolTargetContext) : Prop :=
  ccsRelation ctx ∧
  ∃ tr : SumCheckTranscript,
    SumCheckAccepted (sumcheckInstanceOfContext ctx) tr

/-- Relaxed CE relation: keep only CCS relation (claim-truth may be deferred). -/
def ceRelaxedRelation (ctx : ProtocolTargetContext) : Prop :=
  ccsRelation ctx

/-- Assumptions needed to derive relation-level statements. -/
structure ProtocolRelationsAssumptions (ctx : ProtocolTargetContext) where
  target : ProtocolTargetAssumptions ctx

/-- Native assumption bundle: protocol target closes Theorem-3 via native bar. -/
structure ProtocolRelationsNativeAssumptions (ctx : ProtocolTargetContext) where
  target : ProtocolTargetNativeAssumptions ctx

/--
Canonical protocol-relations assumptions using constructive SumCheck closure.
-/
def ProtocolRelationsAssumptions.ofTarget
  {ctx : ProtocolTargetContext}
  (hTarget : ProtocolTargetAssumptions ctx) :
  ProtocolRelationsAssumptions ctx :=
  { target := hTarget }

/--
Canonical protocol-relations assumptions using the paper-facing challenge-
difference route for `invDelta`.
-/
def ProtocolRelationsAssumptions.ofPaperCarrierDiff
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
  (hNe : ctx.invDelta ≠ zeroRq) :
  ProtocolRelationsAssumptions ctx :=
  { target := ProtocolTargetAssumptions.ofPaperCarrierDiff
      hThm3 hArithmetic hDiff hNe }

/--
Canonical protocol-relations assumptions from any strict low-norm
invertibility boundary whose threshold is at least `5`, specialized to the
active paper-carrier-difference route.
-/
def ProtocolRelationsAssumptions.ofLowNormAtLeastFive
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
  (hNe : ctx.invDelta ≠ zeroRq) :
  ProtocolRelationsAssumptions ctx :=
  { target := ProtocolTargetAssumptions.ofLowNormAtLeastFive
      hFive hThm3 hArithmetic hInv hDiff hNe }

/--
Canonical native protocol-relations assumptions using constructive SumCheck closure.
-/
def ProtocolRelationsNativeAssumptions.ofTarget
  {ctx : ProtocolTargetContext}
  (hTarget : ProtocolTargetNativeAssumptions ctx) :
  ProtocolRelationsNativeAssumptions ctx :=
  { target := hTarget }

/--
Canonical native protocol-relations assumptions using the paper-facing
challenge-difference route for `invDelta`.
-/
def ProtocolRelationsNativeAssumptions.ofPaperCarrierDiff
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
  (hNe : ctx.invDelta ≠ zeroRq) :
  ProtocolRelationsNativeAssumptions ctx :=
  { target := ProtocolTargetNativeAssumptions.ofPaperCarrierDiff
      hBarNative hArithmetic hDiff hNe }

/--
Canonical native protocol-relations assumptions from any strict low-norm
invertibility boundary whose threshold is at least `5`, specialized to the
active paper-carrier-difference route.
-/
def ProtocolRelationsNativeAssumptions.ofLowNormAtLeastFive
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
  (hNe : ctx.invDelta ≠ zeroRq) :
  ProtocolRelationsNativeAssumptions ctx :=
  { target := ProtocolTargetNativeAssumptions.ofLowNormAtLeastFive
      hFive hBarNative hArithmetic hInv hDiff hNe }

/-- Derive CCS relation from target assumptions. -/
theorem ccsRelation_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolRelationsAssumptions ctx) :
  ccsRelation ctx := by
  exact protocolTargetProp_of_assumptions h.target

/-- Derive CCS relation from native target assumptions. -/
theorem ccsRelation_of_native_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolRelationsNativeAssumptions ctx) :
  ccsRelation ctx := by
  exact protocolTargetProp_of_native_assumptions h.target

/-- Derive CE relation from explicit transcript acceptance witness. -/
theorem ccsRelation_iff_protocolTargetProp
  {ctx : ProtocolTargetContext} :
  ccsRelation ctx ↔ protocolTargetProp ctx := by
  rfl

/-- CE is exactly CCS plus an accepted SumCheck transcript witness. -/
theorem ceRelation_iff
  {ctx : ProtocolTargetContext} :
  ceRelation ctx ↔
    ccsRelation ctx ∧
      ∃ tr : SumCheckTranscript,
        SumCheckAccepted (sumcheckInstanceOfContext ctx) tr := by
  rfl

/-- Relaxed CE is definitionally CCS. -/
theorem ceRelaxedRelation_iff
  {ctx : ProtocolTargetContext} :
  ceRelaxedRelation ctx ↔ ccsRelation ctx := by
  rfl

/-- Derive CE relation from CCS relation and an explicit transcript witness. -/
theorem ceRelation_of_ccsRelation
  {ctx : ProtocolTargetContext}
  (hCCS : ccsRelation ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  ceRelation ctx := by
  exact ⟨hCCS, hWitness.accepted_exists⟩

/-- Derive CE relation from CCS relation and SumCheck claim truth. -/
theorem ceRelation_of_ccsRelation_claimTrue
  {ctx : ProtocolTargetContext}
  (hCCS : ccsRelation ctx)
  (hClaimTrue : SumCheckClaimTrue (sumcheckInstanceOfContext ctx)) :
  ceRelation ctx := by
  rcases sumcheckCompleteness_constructive (sumcheckInstanceOfContext ctx) hClaimTrue with ⟨tr, hAcc⟩
  exact ⟨hCCS, ⟨tr, hAcc⟩⟩

/-- Derive CE relation from explicit transcript acceptance witness. -/
theorem ceRelation_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolRelationsAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  ceRelation ctx := by
  exact ceRelation_of_ccsRelation (ccsRelation_of_assumptions h) hWitness

/-- Derive CE relation from claim-truth via SumCheck completeness boundary. -/
theorem ceRelation_of_claimTrue
  {ctx : ProtocolTargetContext}
  (h : ProtocolRelationsAssumptions ctx)
  (hClaimTrue : SumCheckClaimTrue (sumcheckInstanceOfContext ctx)) :
  ceRelation ctx := by
  exact ceRelation_of_ccsRelation_claimTrue (ccsRelation_of_assumptions h) hClaimTrue

/-- Derive CE relation from native assumptions and explicit transcript witness. -/
theorem ceRelation_of_native_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolRelationsNativeAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  ceRelation ctx := by
  exact ceRelation_of_ccsRelation (ccsRelation_of_native_assumptions h) hWitness

/-- Derive CE relation from claim-truth via native assumptions. -/
theorem ceRelation_of_native_claimTrue
  {ctx : ProtocolTargetContext}
  (h : ProtocolRelationsNativeAssumptions ctx)
  (hClaimTrue : SumCheckClaimTrue (sumcheckInstanceOfContext ctx)) :
  ceRelation ctx := by
  exact ceRelation_of_ccsRelation_claimTrue
    (ccsRelation_of_native_assumptions h) hClaimTrue

/-- Soundness lift: any CE witness yields SumCheck claim truth. -/
theorem ceClaimTrue_of_ce
  {ctx : ProtocolTargetContext}
  (hCE : ceRelation ctx) :
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx) := by
  rcases hCE.2 with ⟨tr, hAcc⟩
  exact sumcheckSoundness_constructive _ _ hAcc

/-- Soundness lift on the native assumption path. -/
theorem ceClaimTrue_of_native_ce
  {ctx : ProtocolTargetContext}
  (hCE : ceRelation ctx) :
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx) :=
  ceClaimTrue_of_ce hCE

/-- CE implies relaxed CE. -/
theorem ceRelaxedRelation_of_ce
  {ctx : ProtocolTargetContext}
  (hCE : ceRelation ctx) :
  ceRelaxedRelation ctx := by
  exact hCE.1

end SuperNeo
