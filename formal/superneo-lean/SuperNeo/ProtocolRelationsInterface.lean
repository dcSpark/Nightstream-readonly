import SuperNeo.ProtocolRelations

/-!
Contract interface for `SuperNeo.ProtocolRelations`.

Spec: ./formal/superneo-lean/specs/ProtocolRelations.spec.md

Paper anchors (Source: ./formal/superneo-lean/SuperNeo.pdf.md):
- Definition 12 (Norm-bounded CCS), Section 7.1, lines 457-459.
- Definition 13 (Norm-bounded CCS Evaluation Relation), Section 7.1, lines 461-465.
- Section 7.1 (Relations), lines 449-465.
-/

namespace SuperNeo

namespace ProtocolRelationsInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `sumcheckInstanceOfContext`. -/
abbrev sumcheckInstanceOfContext := SuperNeo.sumcheckInstanceOfContext

/-- [Role: Theorem-Target] Curated re-export of `ccsRelation`. -/
abbrev ccsRelation := SuperNeo.ccsRelation

/-- [Role: Theorem-Target] Curated re-export of `ceRelation`. -/
abbrev ceRelation := SuperNeo.ceRelation

/-- [Role: Theorem-Target] Curated re-export of `ceRelaxedRelation`. -/
abbrev ceRelaxedRelation := SuperNeo.ceRelaxedRelation

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `ceRelation_of_claimTrue`. -/
abbrev ceRelation_of_claimTrue
  {ctx : ProtocolTargetContext} :
  ProtocolRelationsAssumptions ctx →
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx) →
  ceRelation ctx :=
  SuperNeo.ceRelation_of_claimTrue

/-- [Role: Theorem-Target] Curated theorem surface `ceRelation_of_ccsRelation`. -/
abbrev ceRelation_of_ccsRelation
  {ctx : ProtocolTargetContext} :
  ccsRelation ctx →
  SumCheckTransitionWitness ctx →
  ceRelation ctx :=
  SuperNeo.ceRelation_of_ccsRelation

/-- [Role: Theorem-Target] Curated theorem surface `ceRelation_of_ccsRelation_claimTrue`. -/
abbrev ceRelation_of_ccsRelation_claimTrue
  {ctx : ProtocolTargetContext} :
  ccsRelation ctx →
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx) →
  ceRelation ctx :=
  SuperNeo.ceRelation_of_ccsRelation_claimTrue

/-- [Role: Theorem-Target] Curated theorem surface `ceClaimTrue_of_ce`. -/
abbrev ceClaimTrue_of_ce
  {ctx : ProtocolTargetContext} :
  ceRelation ctx →
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx) :=
  SuperNeo.ceClaimTrue_of_ce

/-- [Role: Theorem-Target] Curated theorem surface `ceClaimTrue_of_native_ce`. -/
abbrev ceClaimTrue_of_native_ce
  {ctx : ProtocolTargetContext} :
  ceRelation ctx →
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx) :=
  SuperNeo.ceClaimTrue_of_native_ce

/-- [Role: Theorem-Target] Curated theorem surface `ceRelaxedRelation_of_ce`. -/
abbrev ceRelaxedRelation_of_ce
  {ctx : ProtocolTargetContext} :
  ceRelation ctx →
  ceRelaxedRelation ctx :=
  SuperNeo.ceRelaxedRelation_of_ce

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Boundary surface `SumCheckTransitionWitness` requiring closure. -/
abbrev SumCheckTransitionWitness := SuperNeo.SumCheckTransitionWitness

/-- [Role: Boundary] Boundary surface `SumCheckTransitionWitness.accepted_exists` requiring closure. -/
abbrev SumCheckTransitionWitness_accepted_exists
  {ctx : ProtocolTargetContext} :
  SumCheckTransitionWitness ctx →
  ∃ tr : SumCheckTranscript, SumCheckAccepted (sumcheckInstanceOfContext ctx) tr :=
  SuperNeo.SumCheckTransitionWitness.accepted_exists

/-- [Role: Boundary] Boundary surface `ProtocolRelationsAssumptions` requiring closure. -/
abbrev ProtocolRelationsAssumptions := SuperNeo.ProtocolRelationsAssumptions

/-- [Role: Boundary] Native boundary bundle for protocol relations assumptions. -/
abbrev ProtocolRelationsNativeAssumptions := SuperNeo.ProtocolRelationsNativeAssumptions

/--
[Role: Theorem-Target] Canonical protocol-relations constructor from the
paper-facing challenge-difference route.
-/
abbrev ProtocolRelationsAssumptions_ofPaperCarrierDiff
  {ctx : ProtocolTargetContext} :=
  SuperNeo.ProtocolRelationsAssumptions.ofPaperCarrierDiff (ctx := ctx)

/--
[Role: Theorem-Target] Canonical protocol-relations constructor from a stronger
strict low-norm invertibility theorem with threshold at least `5`.
-/
def ProtocolRelationsAssumptions_ofLowNormAtLeastFive
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
  SuperNeo.ProtocolRelationsAssumptions.ofLowNormAtLeastFive
    (ctx := ctx) hFive hThm3 hArithmetic hInv hDiff hNe

/--
[Role: Theorem-Target] Canonical native protocol-relations constructor from the
paper-facing challenge-difference route.
-/
abbrev ProtocolRelationsNativeAssumptions_ofPaperCarrierDiff
  {ctx : ProtocolTargetContext} :=
  SuperNeo.ProtocolRelationsNativeAssumptions.ofPaperCarrierDiff (ctx := ctx)

/--
[Role: Theorem-Target] Canonical native protocol-relations constructor from a
stronger strict low-norm invertibility theorem with threshold at least `5`.
-/
def ProtocolRelationsNativeAssumptions_ofLowNormAtLeastFive
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
  SuperNeo.ProtocolRelationsNativeAssumptions.ofLowNormAtLeastFive
    (ctx := ctx) hFive hBarNative hArithmetic hInv hDiff hNe

/-- [Role: Boundary] Boundary surface `ccsRelation_of_assumptions` requiring closure. -/
abbrev ccsRelation_of_assumptions
  {ctx : ProtocolTargetContext} :
  ProtocolRelationsAssumptions ctx →
  ccsRelation ctx :=
  SuperNeo.ccsRelation_of_assumptions

/-- [Role: Boundary] Native constructor surface for `ccsRelation`. -/
abbrev ccsRelation_of_native_assumptions
  {ctx : ProtocolTargetContext} :
  ProtocolRelationsNativeAssumptions ctx →
  ccsRelation ctx :=
  SuperNeo.ccsRelation_of_native_assumptions

/-- [Role: Boundary] Boundary surface `ceRelation_of_assumptions` requiring closure. -/
abbrev ceRelation_of_assumptions
  {ctx : ProtocolTargetContext} :
  ProtocolRelationsAssumptions ctx →
  SumCheckTransitionWitness ctx →
  ceRelation ctx :=
  SuperNeo.ceRelation_of_assumptions

/-- [Role: Boundary] Native constructor surface for `ceRelation` from witness. -/
abbrev ceRelation_of_native_assumptions
  {ctx : ProtocolTargetContext} :
  ProtocolRelationsNativeAssumptions ctx →
  SumCheckTransitionWitness ctx →
  ceRelation ctx :=
  SuperNeo.ceRelation_of_native_assumptions

/-- [Role: Boundary] Native constructor surface for `ceRelation` from claim truth. -/
abbrev ceRelation_of_native_claimTrue
  {ctx : ProtocolTargetContext} :
  ProtocolRelationsNativeAssumptions ctx →
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx) →
  ceRelation ctx :=
  SuperNeo.ceRelation_of_native_claimTrue

end ProtocolRelationsInterface

end SuperNeo
