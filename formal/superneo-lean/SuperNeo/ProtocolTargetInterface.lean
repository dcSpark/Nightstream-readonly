import SuperNeo.ProtocolTarget

/-!
Contract interface for `SuperNeo.ProtocolTarget`.

Spec: `./formal/superneo-lean/specs/ProtocolTarget.spec.md`

Paper anchors (Source: `./formal/superneo-lean/SuperNeo.pdf.md`):
- Section 7 (Neo's folding scheme for CCS), lines 447–481: Relations (Definitions 11–13), Global Reduction Parameters (Definition 14)
- Section 7.3 (Π_CCS), lines 481–547: Interactive reduction for CCS
-/

namespace SuperNeo

namespace ProtocolTargetInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `ProtocolTargetContext`. -/
abbrev ProtocolTargetContext := SuperNeo.ProtocolTargetContext

/-- [Role: Theorem-Target] Curated re-export of `protocolTargetProp`. -/
abbrev protocolTargetProp := SuperNeo.protocolTargetProp

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Boundary surface `ProtocolTargetAssumptions` requiring closure. -/
abbrev ProtocolTargetAssumptions := SuperNeo.ProtocolTargetAssumptions

/-- [Role: Boundary] Native boundary bundle for protocol target assumptions. -/
abbrev ProtocolTargetNativeAssumptions := SuperNeo.ProtocolTargetNativeAssumptions

/-- [Role: Boundary] Boundary surface `matrixTransformAssumption_of_thm3CoreAssumption` requiring closure. -/
theorem matrixTransformAssumption_of_thm3CoreAssumption
  {bar m : Array (Array F)}
  (h : thm3CoreAssumption bar) :
  matrixTransformAssumption bar m :=
  SuperNeo.matrixTransformAssumption_of_thm3CoreAssumption h

/-- [Role: Boundary] Boundary surface `protocolTargetProp_of_assumptions` requiring closure. -/
theorem protocolTargetProp_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetAssumptions ctx) :
  protocolTargetProp ctx :=
  SuperNeo.protocolTargetProp_of_assumptions h

/-- [Role: Boundary] Native constructor surface for `protocolTargetProp`. -/
theorem protocolTargetProp_of_native_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetNativeAssumptions ctx) :
  protocolTargetProp ctx :=
  SuperNeo.protocolTargetProp_of_native_assumptions h

/-! ## Paper-Facing Invertibility Bridge -/

/-- [Role: Theorem-Target] Strict `< 5` window from a nonzero paper-carrier difference. -/
theorem strictInvertibilityWindowProp_five_of_paperCarrierDiff
  {δ : Coeffs}
  (hDiff : samplingDiffSet paperCarrier δ)
  (hNe : δ ≠ zeroRq) :
  strictInvertibilityWindowProp 5 δ :=
  SuperNeo.strictInvertibilityWindowProp_five_of_paperCarrierDiff hDiff hNe

/-- [Role: Theorem-Target] Derive invertibility on the active paper-carrier-difference path. -/
theorem invertibleRq_of_paperCarrierDiff
  {δ : Coeffs}
  (hDiff : samplingDiffSet paperCarrier δ)
  (hNe : δ ≠ zeroRq) :
  invertibleRq δ :=
  SuperNeo.invertibleRq_of_paperCarrierDiff hDiff hNe

/--
[Role: Theorem-Target] Canonical protocol-target constructor from the
paper-facing challenge-difference route.
-/
abbrev ProtocolTargetAssumptions_ofPaperCarrierDiff
  {ctx : ProtocolTargetContext} :=
  SuperNeo.ProtocolTargetAssumptions.ofPaperCarrierDiff (ctx := ctx)

/--
[Role: Theorem-Target] Canonical protocol-target constructor from a stronger
strict low-norm invertibility theorem with threshold at least `5`.
-/
def ProtocolTargetAssumptions_ofLowNormAtLeastFive
  {ctx : ProtocolTargetContext}
  {B : Nat}
  (hFive : 5 ≤ B)
  (thm3 : thm3CoreAssumption ctx.bar)
  (arithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hInv : lowNormInvertibilityAssumption B)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq) :
  ProtocolTargetAssumptions ctx :=
  SuperNeo.ProtocolTargetAssumptions.ofLowNormAtLeastFive
    (ctx := ctx) hFive thm3 arithmetic hInv hDiff hNe

/--
[Role: Theorem-Target] Native protocol-target constructor from the
paper-facing challenge-difference route.
-/
abbrev ProtocolTargetNativeAssumptions_ofPaperCarrierDiff
  {ctx : ProtocolTargetContext} :=
  SuperNeo.ProtocolTargetNativeAssumptions.ofPaperCarrierDiff (ctx := ctx)

/--
[Role: Theorem-Target] Native protocol-target constructor from a stronger
strict low-norm invertibility theorem with threshold at least `5`.
-/
def ProtocolTargetNativeAssumptions_ofLowNormAtLeastFive
  {ctx : ProtocolTargetContext}
  {B : Nat}
  (hFive : 5 ≤ B)
  (barNative : ctx.bar = nativeBarMatrix)
  (arithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hInv : lowNormInvertibilityAssumption B)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq) :
  ProtocolTargetNativeAssumptions ctx :=
  SuperNeo.ProtocolTargetNativeAssumptions.ofLowNormAtLeastFive
    (ctx := ctx) hFive barNative arithmetic hInv hDiff hNe

end ProtocolTargetInterface

end SuperNeo
